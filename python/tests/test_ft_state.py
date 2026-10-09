# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Free-threading rows for eagle's Python-level process state.

Row classes FT-3 (global state) and FT-4 (distinct objects, host parity) of the
family's free-threading acceptance, registered against the conformance harness
that ``raptor-core`` hosts (a TEST dependency only).

The FT-3 rows exercise the pure-Python modules under real thread interleaving
and need no device: the CUDA-facing leaves (``cupy`` stream wrappers, kernel
attribute queries, device ids) are replaced by tiny stand-ins so the CACHE
LOGIC is what the threads race over. Rows that need a GIL-free interpreter
skip on a GIL build (``require_free_threaded``); the cold-registry and
``_REORDERING`` rows interleave on any build, because they race a blocking
step (a slow entry-point scan) or a mutation during iteration.

FT-4 drives the compiled host team (``eagle._core``) and is correct on any
build; importing ``eagle._core`` re-enables the GIL until the extension itself
declares free-threading support, so on a free-threaded interpreter this row
proves GIL-free behaviour only once that declaration lands (it skips with a
reason while the extension is not importable GIL-free).
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
import sysconfig
import threading
import time
import types
from pathlib import Path

import numpy as np
import pytest

from raptor.conformance import freethreading as ft
from raptor.conformance.freethreading import declare_ft_row, register_ft_row

pytestmark = pytest.mark.ft

declare_ft_row("FT-3-EAGLE-REGISTRY-COLD", "eagle",
               "cold first resolution from many threads: one winner, zero UnknownPatternError")
declare_ft_row("FT-3-EAGLE-COUNTERS-EXACT", "eagle",
               "the static-graph counters and every cache hit/miss tally are exact under threads")
declare_ft_row("FT-3-EAGLE-STREAM-IDENTITY", "eagle",
               "external_stream returns ONE wrapper per handle under threads")
declare_ft_row("FT-3-EAGLE-CAPS-HOLD", "eagle",
               "the zero-mask and kernel-attribute caches never exceed their caps under threads")
declare_ft_row("FT-3-EAGLE-REORDERING-SNAPSHOT", "eagle",
               "registering reordering sets while others scan the registry never raises")
declare_ft_row("FT-4-EAGLE-HOST-PARITY", "eagle",
               "private host-team runs on disjoint planes are bit-identical to the serial oracle")

EXPECTED_TESTS = frozenset(
    {
        "test_manifest_all_tests_collected",
        "test_rows_declared_and_collected",
        "test_cold_registry_has_one_winner",
        "test_counters_exact_under_threads",
        "test_external_stream_identity_under_threads",
        "test_caches_never_exceed_their_caps",
        "test_reordering_registry_is_snapshotted_under_lock",
        "test_host_parity_on_private_planes",
    }
)

THREADS = ft.CANARY_THREADS


def test_manifest_all_tests_collected(request):
    here = {
        i.name for i in request.session.items if i.module is sys.modules[__name__]
    }
    assert EXPECTED_TESTS <= here, sorted(EXPECTED_TESTS - here)
    assert {n for n in here if "[" not in n} <= EXPECTED_TESTS


def test_rows_declared_and_collected():
    ft.assert_ft_rows_complete("eagle")


def _run_threads(n, fn):
    """Run ``fn(i)`` on ``n`` threads released together; return the results
    and the exceptions, each in thread order."""
    barrier = threading.Barrier(n)
    results, errors = [None] * n, [None] * n

    def work(i):
        barrier.wait()
        try:
            results[i] = fn(i)
        except BaseException as exc:  # noqa: BLE001 - reported by the caller
            errors[i] = exc

    pool = [threading.Thread(target=work, args=(i,)) for i in range(n)]
    for t in pool:
        t.start()
    for t in pool:
        t.join()
    return results, errors


# --------------------------------------------------------------------------- #
# FT-3: cold registry. The entry-point scan is slow on purpose (a real scan
# imports provider packages), so a loser that returns at once sees an empty map.
# --------------------------------------------------------------------------- #
@register_ft_row("FT-3-EAGLE-REGISTRY-COLD")
def test_cold_registry_has_one_winner(monkeypatch):
    import importlib.metadata as metadata

    from eagle import registry

    loader = object()
    loads = []

    class _EP:
        name = "ft_cold_pattern"

        def load(self):
            loads.append(threading.get_ident())
            return loader

    scans = []

    def slow_entry_points(*args, **kwargs):
        scans.append(1)
        time.sleep(0.2)
        return [_EP()]

    monkeypatch.setattr(metadata, "entry_points", slow_entry_points)
    monkeypatch.setattr(registry, "_entry_points_scanned", False)
    monkeypatch.setattr(registry, "_PATTERN_LOADERS", {})
    monkeypatch.setattr(registry, "_LAZY_PROVIDERS", {})

    results, errors = _run_threads(
        16, lambda i: registry._resolve_pattern_loader("ft_cold_pattern"))
    assert errors == [None] * 16, [type(e).__name__ for e in errors if e]
    assert all(r is loader for r in results)
    assert len(scans) == 1 and len(loads) == 1, (len(scans), len(loads))


# --------------------------------------------------------------------------- #
# FT-3: counters, exact.
# --------------------------------------------------------------------------- #
@register_ft_row("FT-3-EAGLE-COUNTERS-EXACT")
def test_counters_exact_under_threads(monkeypatch):
    ft.require_free_threaded()
    import importlib

    from eagle import _counters, interop, marshal

    launch = importlib.import_module("eagle.launch")  # `eagle.launch` the name is a function

    n_iter = 25_000
    total = THREADS * n_iter

    # the static-graph counters
    before = _counters.snapshot()
    ft.hammer(lambda: _counters.bump("binds"), THREADS, n_iter)
    ft.hammer(lambda: _counters.bump("rebinds"), THREADS, n_iter)
    after = _counters.snapshot()
    assert after["binds"] - before["binds"] == total
    assert after["rebinds"] - before["rebinds"] == total
    assert after["captures"] == before["captures"]

    # launch plan cache: one key -> misses == 1, hits == calls - 1, one identity
    launch._reset_launch_plan_cache()
    spec = (("mutable", "y"), ("per_sample", "x"))
    first = launch.launch_plan(spec)
    ft.hammer(lambda: launch.launch_plan(spec), THREADS, n_iter)
    stats = launch._launch_plan_stats()
    assert stats == {"hits": total, "misses": 1}, stats
    assert launch.launch_plan(spec) is first

    # external-stream cache tallies
    monkeypatch.setattr(interop, "_wrap_external_stream", lambda ptr: object())
    interop._reset_external_stream_cache()
    ft.hammer(lambda: interop.external_stream(7), THREADS, n_iter)
    s = interop._external_stream_stats()
    assert s == {"hits": total - 1, "misses": 1}, s

    # zero-mask cache tallies
    cp = _FakeCupy()
    marshal._reset_zero_mask_cache()
    ft.hammer(lambda: marshal._cached_zero_mask(16, cp), THREADS, n_iter)
    z = marshal._zero_mask_stats()
    assert z == {"hits": total - 1, "misses": 1, "bypassed": 0}, z
    marshal._reset_zero_mask_cache()


class _FakeCupy:
    """Just enough ``cupy`` for the zero-mask cache: a device id and ``zeros``."""

    bool_ = np.bool_

    def __init__(self):
        self.cuda = types.SimpleNamespace(
            runtime=types.SimpleNamespace(getDevice=lambda: 0))

    @staticmethod
    def zeros(n, dtype=None):
        return np.zeros(n, dtype=dtype)


# --------------------------------------------------------------------------- #
# FT-3: stream identity.
# --------------------------------------------------------------------------- #
@register_ft_row("FT-3-EAGLE-STREAM-IDENTITY")
def test_external_stream_identity_under_threads(monkeypatch):
    from eagle import interop

    made = []

    def slow_wrap(ptr):
        time.sleep(0.01)  # a window for a second thread to miss as well
        w = object()
        made.append(ptr)
        return w

    monkeypatch.setattr(interop, "_wrap_external_stream", slow_wrap)
    interop._reset_external_stream_cache()
    try:
        results, errors = _run_threads(
            16, lambda i: interop.external_stream(0x1000 + i % 2))
        assert errors == [None] * 16, errors
        for handle in (0x1000, 0x1001):
            mine = [r for i, r in enumerate(results) if 0x1000 + i % 2 == handle]
            assert all(r is mine[0] for r in mine), handle
        assert sorted(made) == [0x1000, 0x1001], made
        assert interop._external_stream_stats() == {"hits": 14, "misses": 2}
    finally:
        interop._reset_external_stream_cache()


# --------------------------------------------------------------------------- #
# FT-3: caps.
# --------------------------------------------------------------------------- #
@register_ft_row("FT-3-EAGLE-CAPS-HOLD")
def test_caches_never_exceed_their_caps(monkeypatch):
    import importlib

    from eagle import marshal

    launch = importlib.import_module("eagle.launch")  # `eagle.launch` the name is a function

    cp = _FakeCupy()
    marshal._reset_zero_mask_cache()
    calls_each = 400
    distinct = 200

    def hit_masks(t):
        for k in range(calls_each):
            marshal._cached_zero_mask(1 + (k * (t + 1)) % distinct, cp)

    _, errors = _run_threads(THREADS, hit_masks)
    assert errors == [None] * THREADS, errors
    z = marshal._zero_mask_stats()
    assert len(marshal._zero_mask_cache) <= marshal.ZERO_MASK_CACHE_CAP
    assert z["hits"] + z["misses"] + z["bypassed"] == THREADS * calls_each, z
    assert z["misses"] == len(marshal._zero_mask_cache), z
    marshal._reset_zero_mask_cache()

    cap = 16
    monkeypatch.setattr(launch, "KERNEL_ATTRS_CACHE_CAP", cap)
    monkeypatch.setattr(launch, "_query_kernel_attrs", lambda fn: {"fn": fn})
    launch._reset_kernel_attrs_cache()
    kernels = [object() for _ in range(64)]

    def hit_attrs(t):
        ok = full = 0
        for k in range(calls_each):
            try:
                launch._kernel_attrs(kernels[(k * (t + 1)) % len(kernels)])
                ok += 1
            except launch.KernelAttrsCacheFull:
                full += 1
        return ok, full

    results, errors = _run_threads(THREADS, hit_attrs)
    assert errors == [None] * THREADS, errors
    assert len(launch._kernel_attrs_cache) <= cap, len(launch._kernel_attrs_cache)
    assert sum(a + b for a, b in results) == THREADS * calls_each
    launch._reset_kernel_attrs_cache()


# --------------------------------------------------------------------------- #
# FT-3: _REORDERING snapshot.
# --------------------------------------------------------------------------- #
@register_ft_row("FT-3-EAGLE-REORDERING-SNAPSHOT")
def test_reordering_registry_is_snapshotted_under_lock():
    from eagle import _active_set

    sets_per_thread = 150
    keep = [[] for _ in range(THREADS)]
    probe = np.zeros(8, dtype=np.uint8)
    stop = threading.Event()
    scan_errors = []

    def scanner():
        while not stop.is_set():
            try:
                _active_set.refuse_if_permuted(probe, "ft")
            except BaseException as exc:  # noqa: BLE001
                scan_errors.append(exc)

    scans = [threading.Thread(target=scanner) for _ in range(2)]
    for t in scans:
        t.start()

    def build(t):
        for _ in range(sets_per_thread):
            keep[t].append(_active_set.ActiveSet(np.zeros(32, dtype=bool), reorder=0.5))

    try:
        _, errors = _run_threads(THREADS, build)
    finally:
        stop.set()
        for t in scans:
            t.join()
    assert errors == [None] * THREADS, [repr(e) for e in errors if e]
    assert scan_errors == [], repr(scan_errors[:1])
    assert len(_active_set._REORDERING) >= THREADS * sets_per_thread

    # one plane is moved by one set, once: of THREADS sets racing to own the
    # same mask exactly one is admitted, the rest are refused by the claim.
    shared = np.zeros(4096, dtype=bool)
    admitted = []

    def claim(t):
        a = _active_set.ActiveSet(shared, reorder=0.5)
        admitted.append(a)  # keeps it alive; list.append is atomic
        return True

    results, errors = _run_threads(THREADS, claim)
    refused = [e for e in errors if isinstance(e, ValueError)]
    assert [e for e in errors if e is not None and not isinstance(e, ValueError)] == []
    assert len(admitted) == 1 and len(refused) == THREADS - 1, (
        len(admitted), len(refused))


# --------------------------------------------------------------------------- #
# FT-4: private host runs on disjoint planes == the serial oracle, bit for bit.
# --------------------------------------------------------------------------- #
_HOST_SRC = r"""
struct ScalarHandle {
    void* data; unsigned long long samples; unsigned long long stride;
    int deviceType; int deviceId;
};
extern "C" void ft_axpb(void* const* params, long long base, long long count,
                        long long /*nSamples*/) {
    double* y       = (double*)((const ScalarHandle*)params[0])->data;
    const double* x = (const double*)((const ScalarHandle*)params[1])->data;
    const double a  = *(const double*)params[2];
    const double b  = *(const double*)params[3];
    for (long long i = base; i < base + count; ++i) y[i] = a * x[i] + b;
}
"""


class _Handle(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("samples", ctypes.c_ulonglong),
                ("stride", ctypes.c_ulonglong), ("deviceType", ctypes.c_int),
                ("deviceId", ctypes.c_int)]


@register_ft_row("FT-4-EAGLE-HOST-PARITY")
def test_host_parity_on_private_planes(tmp_path):
    core = pytest.importorskip("eagle._core", reason="the compiled extension is not built")
    if sysconfig.get_config_var("Py_GIL_DISABLED") and sys._is_gil_enabled():
        # free-threaded interpreter, GIL back on: importing the extension re-enabled it
        pytest.skip("eagle._core re-enabled the GIL (not yet free-threading-declared); "
                    "this row proves GIL-free behaviour only once it is")
    gxx = shutil.which("g++") or "/usr/bin/g++"
    src = tmp_path / "ft_axpb.cpp"
    src.write_text(_HOST_SRC)
    so = tmp_path / "ft_axpb.so"
    proc = subprocess.run([gxx, "-O2", "-shared", "-fPIC", "-o", str(so), str(src)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    lib = ctypes.CDLL(str(so))
    entry = ctypes.cast(lib.ft_axpb, ctypes.c_void_p).value

    n, reps = 20_000, 12

    def one(t):
        rng = np.random.default_rng(100 + t)
        a, b = ctypes.c_double(1.5 + t), ctypes.c_double(-0.25 * t)
        mismatches = 0
        for _ in range(reps):
            x = rng.normal(size=n)
            y_par, y_ser = np.zeros(n), np.zeros(n)
            hx = _Handle(x.ctypes.data, n, 1, 0, 0)
            hp = _Handle(y_par.ctypes.data, n, 1, 0, 0)
            hs = _Handle(y_ser.ctypes.data, n, 1, 0, 0)
            tail = [ctypes.addressof(a), ctypes.addressof(b)]
            part = core.Partition.whole(n)
            core.run_host(entry, [ctypes.addressof(hp), ctypes.addressof(hx)] + tail, part)
            core.run_host_serial(
                entry, [ctypes.addressof(hs), ctypes.addressof(hx)] + tail, part)
            mismatches += int(not np.array_equal(y_par, y_ser))
            mismatches += int(not np.array_equal(y_ser, a.value * x + b.value))
        return mismatches

    results, errors = _run_threads(THREADS, one)
    assert errors == [None] * THREADS, [repr(e) for e in errors if e]
    assert results == [0] * THREADS, results
