# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The host launch pays for the team it uses: a one-tile launch runs in place (no
OpenMP region), a team is never wider than its tiles or the process-wide host-thread
setting, a ``work_share`` stage of native nodes shares ONE region, and none of it
changes a value."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np
import pytest

import eagle
import eagle.exec as eexec
from eagle import _core
from eagle import _host_loop
from eagle._host_schedule import lower_cpu_stage  # noqa: F401  (the stage lowering under test)

from _graph_fixtures import host_fork

_ENTRY = r"""
#include <omp.h>
extern "C" void rec(void* const* a, long base, long count, long n) {
    (void)n;
    int* meta = (int*)a[0];   // [0] widest team seen, [1] calls
    double* out = (double*)a[1];
    const int team = omp_get_num_threads();
    int seen = __atomic_load_n(&meta[0], __ATOMIC_RELAXED);
    while (team > seen && !__atomic_compare_exchange_n(&meta[0], &seen, team, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
    __atomic_fetch_add(&meta[1], 1, __ATOMIC_RELAXED);
    for (long i = base; i < base + count; ++i) out[i] = out[i] * 2.0 + 1.0;
}
"""


@pytest.fixture(scope="module")
def entry(tmp_path_factory):
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if not cxx:
        pytest.skip("no C++ compiler to build the probe host entry")
    d = tmp_path_factory.mktemp("probe")
    src, so = d / "probe.cpp", d / "probe.so"
    src.write_text(_ENTRY)
    run = subprocess.run([cxx, "-O1", "-fPIC", "-shared", "-fopenmp", str(src), "-o", str(so)],
                         capture_output=True, text=True)
    if run.returncode:
        pytest.skip(f"cannot build the probe host entry: {run.stderr[-300:]}")
    lib = ctypes.CDLL(str(so))
    return ctypes.cast(lib.rec, ctypes.c_void_p).value, lib


@pytest.fixture(autouse=True)
def _restore_setting():
    yield
    eexec.set_host_threads(None)


def _run(entry_ptr, n, how="run", nodes=1):
    meta = np.zeros(2, dtype=np.int32)
    out = np.arange(n, dtype=np.float64)
    params = [meta.ctypes.data, out.ctypes.data]
    part = eexec.Partition(0, n, n)
    if how == "stage":
        _core.run_host_stage([[(entry_ptr, params, [part])] * nodes])
    else:
        for _ in range(nodes):
            _core.run_host(entry_ptr, params, part, 0)
    return meta, out


def _tiles(n):
    return _core.host_tile_count(eexec.Partition(0, n, n))


def test_one_tile_runs_in_place_with_no_team(entry):
    meta, _ = _run(entry[0], 1)
    assert (meta[0], meta[1]) == (1, 1)


def test_the_team_is_the_setting_and_never_wider_than_the_tiles(entry):
    many = 64 * _core.host_tile_size(0)
    eexec.set_host_threads(2)
    assert eexec.host_threads() == 2
    assert _run(entry[0], many)[0][0] == 2                 # threads used == the setting
    eexec.set_host_threads(3)
    assert _run(entry[0], many)[0][0] == 3
    n2 = 2 * _core.host_tile_size(0)
    assert _tiles(n2) == 2
    eexec.set_host_threads(8)
    assert _run(entry[0], n2)[0][0] == 2                   # min(tiles, setting)
    eexec.set_host_threads(1)
    assert _run(entry[0], many)[0][0] == 1


def test_the_setting_is_process_wide_across_python_threads(entry):
    import threading

    many = 64 * _core.host_tile_size(0)
    eexec.set_host_threads(2)
    seen = []
    t = threading.Thread(target=lambda: seen.append(_run(entry[0], many)[0][0]))
    t.start()
    t.join()
    assert seen == [2]


def test_none_restores_the_openmp_default(entry):
    default = _core.omp_threads()
    eexec.set_host_threads(1)
    assert eexec.host_threads() == 1
    eexec.set_host_threads(None)
    assert eexec.host_threads() == default


@pytest.mark.parametrize("bad", [0, -2, 1.5, "4", True])
def test_the_setting_refuses_a_non_positive_or_non_integer(bad):
    with pytest.raises(ValueError, match="set_host_threads"):
        eexec.set_host_threads(bad)


def test_omp_runtime_is_the_one_the_extension_links():
    get, put = _host_loop._omp_runtime()
    before = get()
    put(before)
    assert get() == before == _core.omp_threads()


@pytest.mark.parametrize("threads", [1, 4])
def test_a_stage_shares_one_region_and_equals_a_run_per_node_bitwise(entry, threads):
    n = 64 * _core.host_tile_size(0) + 5
    eexec.set_host_threads(threads)
    meta_s, stage = _run(entry[0], n, how="stage", nodes=3)
    meta_r, per_node = _run(entry[0], n, how="run", nodes=3)
    assert np.array_equal(stage, per_node)
    assert meta_s[0] == meta_r[0] == threads
    assert meta_s[1] == meta_r[1]                       # the same tiles ran


def test_a_work_share_graph_stage_runs_as_one_native_stage_and_matches(tmp_path, monkeypatch):
    n = _core.BRANCH_TASKS_MAX_SAMPLES + 1
    chains, arrays, expected = host_fork(tmp_path, branches=2, n=n, depth=2)
    calls = []
    real = _core.run_host_stage
    monkeypatch.setattr(_core, "run_host_stage", lambda stage: (calls.append(len(stage)), real(stage))[1])
    g = eagle.Graph().add_group(chains).build()
    assert g.schedule == ("work_share",)
    g.launch()
    assert calls == [2]
    for (a, _b), ref in zip(arrays, expected):
        np.testing.assert_array_equal(a, ref)
