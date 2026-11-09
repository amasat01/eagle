# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The host launch pays for the team it uses: a one-tile launch runs in place (no
thread woken), a team is never wider than its tiles or the process-wide host-thread
setting, a ``work_share`` stage of native nodes is ONE pool launch, the budget
follows the setting, ``OMP_NUM_THREADS`` and the physical cores, the pool survives
interpreter exit and ``os.fork``, and none of it changes a value."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

import eagle
import eagle.exec as eexec
from eagle import _core
from eagle._host_schedule import lower_cpu_stage  # noqa: F401  (the stage lowering under test)

from _graph_fixtures import host_chain, host_fork

_ENTRY = r"""
#include <unistd.h>
extern "C" void nap(void* const* a, long base, long count, long n) {
    (void)base; (void)count; (void)n;
    volatile int* meta = (volatile int*)a[0];   // [0] set: the call has started; it returns once [1] is set (5 s at most)
    meta[0] = 1;
    for (int i = 0; i < 5000 && meta[1] == 0; ++i) usleep(1000);
}
extern "C" void rec(void* const* a, long base, long count, long n) {
    (void)n;
    int* meta = (int*)a[0];   // [0] threads that ran a call, [1] calls, [2] the run's stamp
    double* out = (double*)a[1];
    static thread_local int stamp = 0;
    const int run = __atomic_load_n(&meta[2], __ATOMIC_RELAXED);
    if (stamp != run) { stamp = run; __atomic_fetch_add(&meta[0], 1, __ATOMIC_RELAXED); }
    __atomic_fetch_add(&meta[1], 1, __ATOMIC_RELAXED);
    for (long i = base; i < base + count; ++i) out[i] = out[i] * 2.0 + 1.0;
}
"""
_STAMP = [0]

@pytest.fixture(scope="module")
def entry(tmp_path_factory):
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("c++")
    if not cxx:
        pytest.skip("no C++ compiler to build the probe host entry")
    d = tmp_path_factory.mktemp("probe")
    src, so = d / "probe.cpp", d / "probe.so"
    src.write_text(_ENTRY)
    run = subprocess.run([cxx, "-O1", "-fPIC", "-shared", str(src), "-o", str(so)],
                         capture_output=True, text=True)
    if run.returncode:
        pytest.skip(f"cannot build the probe host entry: {run.stderr[-300:]}")
    lib = ctypes.CDLL(str(so))
    return ctypes.cast(lib.rec, ctypes.c_void_p).value, lib, str(so)


@pytest.fixture(autouse=True)
def _restore_setting():
    yield
    eexec.set_host_threads(None)


def _run(entry_ptr, n, how="run", nodes=1):
    _STAMP[0] += 1
    meta = np.array([0, 0, _STAMP[0]], dtype=np.int32)
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
    assert 1 <= _run(entry[0], many)[0][0] <= 2            # a late worker may find no tile left
    eexec.set_host_threads(3)
    assert 1 <= _run(entry[0], many)[0][0] <= 3
    n2 = 2 * _core.host_tile_size(0)
    assert _tiles(n2) == 2
    eexec.set_host_threads(8)
    assert _run(entry[0], n2)[0][0] <= 2                   # min(tiles, setting)
    eexec.set_host_threads(1)
    assert _run(entry[0], many)[0][0] == 1


def test_the_setting_is_process_wide_across_python_threads(entry):
    import threading

    many = 64 * _core.host_tile_size(0)
    eexec.set_host_threads(1)
    seen = []
    t = threading.Thread(target=lambda: seen.append(_run(entry[0], many)[0][0]))
    t.start()
    t.join()
    assert seen == [1]


def test_none_restores_the_default(entry):
    eexec.set_host_threads(1)
    assert eexec.host_threads() == 1
    eexec.set_host_threads(None)
    env = os.environ.get("OMP_NUM_THREADS", "").split(",")[0]
    if env.isdigit() and int(env) > 0:
        assert eexec.host_threads() == int(env)
    else:
        assert 1 <= eexec.host_threads() <= _core.physical_cores()


@pytest.mark.parametrize("bad", [0, -2, 1.5, "4", True])
def test_the_setting_refuses_a_non_positive_or_non_integer(bad):
    with pytest.raises(ValueError, match="set_host_threads"):
        eexec.set_host_threads(bad)


def test_the_openmp_bindings_are_deprecated():
    with pytest.warns(DeprecationWarning, match="omp_threads"):
        before = _core.omp_threads()
    with pytest.warns(DeprecationWarning, match="omp_set_threads"):
        _core.omp_set_threads(before)


@pytest.mark.parametrize("threads", [1, 4])
def test_a_stage_shares_one_region_and_equals_a_run_per_node_bitwise(entry, threads):
    n = 64 * _core.host_tile_size(0) + 5
    eexec.set_host_threads(threads)
    meta_s, stage = _run(entry[0], n, how="stage", nodes=3)
    meta_r, per_node = _run(entry[0], n, how="run", nodes=3)
    assert np.array_equal(stage, per_node)
    assert meta_s[0] <= threads and meta_r[0] <= threads
    assert meta_s[1] == meta_r[1]                       # the same tiles ran


def test_a_work_share_graph_stage_runs_as_one_native_stage_and_matches(tmp_path):
    n = _core.BRANCH_TASKS_MAX_SAMPLES + 1
    chains, arrays, expected = host_fork(tmp_path, branches=2, n=n, depth=2)
    g = eagle.Graph().add_group(chains).build()
    assert g.schedule == ("work_share",)
    assert g.host_lanes == (1,)                            # launch 1 runs in place and is timed
    g.launch()
    assert 1 <= g.host_lanes[0] <= eexec.host_threads()
    for (a, _b), ref in zip(arrays, expected):
        np.testing.assert_array_equal(a, ref)


def test_a_host_array_op_runs_in_place_up_to_one_tile_and_honours_the_setting():
    tile = _core.host_tile_size(0)
    eexec.set_host_threads(2)
    assert _core.host_op_team(tile) == 1
    assert _core.host_op_team(2 * tile) == 2
    eexec.set_host_threads(None)
    assert _core.host_op_team(2 * tile) == min(2, _core.host_threads())
    assert _core.host_op_team(64 * tile) == _core.host_threads()


# -- the budget and the pool's lifetime, each in a fresh interpreter under a timeout --------------

_POOL_SCRIPT = textwrap.dedent("""
    import ctypes, os, sys, threading
    import numpy as np
    import eagle.exec as eexec
    from eagle import _core
    lib = ctypes.CDLL(sys.argv[1])
    entry = ctypes.cast(lib.rec, ctypes.c_void_p).value
    n = 64 * _core.host_tile_size(0)

    def stage(tasks):
        meta, out = np.array([0, 0, 1], dtype=np.int32), np.arange(n, dtype=np.float64)
        node = (entry, [meta.ctypes.data, out.ctypes.data], [eexec.Partition(0, n, n)])
        return _core.HostStage([[node] * 2] * 9 if tasks else [[node] * 2], tasks), out, meta

    def check(tasks, launches):
        s, out, meta = stage(tasks)
        for _ in range(launches):
            s.run()
        return meta[1] >= launches * 2 * (9 if tasks else 1)   # a sized launch may cut nodes into tiles

    mode = sys.argv[2]
    if mode == "budget":
        print(eexec.host_threads(), _core.physical_cores())
    elif mode == "exit":
        eexec.set_host_threads(4)
        assert check(False, 50) and check(True, 50)
        print("exit with live workers")     # no join: the process ends, the workers with it
    elif mode == "fork":
        eexec.set_host_threads(4)
        assert check(False, 20)
        stop = threading.Event()
        hammer = threading.Thread(target=lambda: [check(True, 1) for _ in iter(stop.is_set, True)])
        hammer.start()
        pid = os.fork()
        if pid == 0:
            os._exit(0 if all(check(True, 1) for _ in range(1000)) else 3)
        _, status = os.waitpid(pid, 0)
        stop.set()
        hammer.join()
        print("child", os.waitstatus_to_exitcode(status))
""")


def _stacks(pid):
    """Where every thread of `pid` and its children waits (macOS `sample`), for an overrun's report."""
    if sys.platform != "darwin" or not shutil.which("sample"):
        return ""
    kids = subprocess.run(["pgrep", "-P", str(pid)], capture_output=True, text=True).stdout.split()
    out = []
    for p in [str(pid), *kids]:
        s = subprocess.run(["sample", p, "2"], capture_output=True, text=True, timeout=60).stdout
        out.append("== pid %s\n%s" % (p, s[s.find("Call graph:"):s.find("Total number in stack")]))
    return "\n".join(out)


def _pool_run(entry, mode, env=None, prefix=()):
    cmd = [*prefix, sys.executable, "-c", _POOL_SCRIPT, entry[2], mode]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={**os.environ, **(env or {})}, start_new_session=True)
    try:
        out, err = proc.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        stacks = _stacks(proc.pid)
        os.killpg(proc.pid, 9)       # the whole group: a forked child left alive would hold the pipes open
        proc.wait()
        pytest.fail("pool script overran 45 s (%s)\n%s" % (mode, stacks), pytrace=False)
    assert proc.returncode == 0, err[-2000:]
    return out.split()


def _mask_cores(cpus):
    cores = set()
    for cpu in cpus:
        with open(f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list") as f:
            cores.add(f.read().strip())
    return len(cores)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="affinity masks and sysfs are Linux")
def test_the_budget_is_the_setting_then_omp_num_threads_then_the_physical_cores(entry):
    env = {k: v for k, v in os.environ.items() if k != "OMP_NUM_THREADS"}
    cpus = sorted(os.sched_getaffinity(0))
    mask = cpus[-min(len(cpus), 6):]
    prefix = ("taskset", "-c", ",".join(map(str, mask))) if shutil.which("taskset") else ()
    run = subprocess.run([*prefix, sys.executable, "-c", _POOL_SCRIPT, entry[2], "budget"],
                         capture_output=True, text=True, timeout=60, env=env)
    assert run.returncode == 0, run.stderr[-2000:]
    threads, cores = map(int, run.stdout.split())
    assert cores == _mask_cores(mask if prefix else cpus)
    assert 1 <= threads <= cores                          # the cgroup quota may cap it further
    assert _pool_run(entry, "budget", {"OMP_NUM_THREADS": "3"})[0] == "3"
    assert eexec.host_threads() > 0
    eexec.set_host_threads(2)
    assert eexec.host_threads() == 2                       # the setting wins over both


def test_interpreter_exit_with_live_workers_exits_cleanly(entry):
    assert _pool_run(entry, "exit")[-1] == "workers"


@pytest.mark.skipif(not hasattr(os, "fork"), reason="no os.fork")
def test_a_forked_child_launches_on_a_fresh_pool_while_the_parent_keeps_launching(entry):
    assert _pool_run(entry, "fork") == ["child", "0"]


# -- one native call per host launch --------------------------------------------------

def _stage(entry_ptr, n, fn=None):
    meta = np.array([0, 0, 1], dtype=np.int32)
    out = np.arange(n, dtype=np.float64)
    part = eexec.Partition(0, n, n)
    params = [meta.ctypes.data, out.ctypes.data]
    stage = _core.HostStage([[(fn or entry_ptr, params, [part])]], False)
    return stage, meta, out


def test_a_native_graph_launch_is_one_native_call(tmp_path):
    b1, b2, *_ = host_chain(tmp_path)
    g = eagle.Graph().add(b1).add(b2).build()
    assert g._native is not None
    g.launch()
    frames, native = [], []

    def profile(frame, event, arg):          # nanobind calls raise no c_call: count the Python frames
        if event == "call":
            frames.append(frame.f_code.co_name)

    sys.setprofile(profile)
    try:
        g.launch()
    finally:
        sys.setprofile(None)
    assert frames == ["launch"], frames
    if not hasattr(sys, "monitoring"):
        return
    mon, tool = sys.monitoring, sys.monitoring.PROFILER_ID

    def on_call(code, offset, callable_, arg0):
        if type(callable_).__module__ == "nanobind":
            native.append(callable_)

    mon.use_tool_id(tool, "eagle-t1")
    mon.register_callback(tool, mon.events.CALL, on_call)
    mon.set_events(tool, mon.events.CALL)
    try:
        g.launch()
    finally:
        mon.set_events(tool, 0)
        mon.register_callback(tool, mon.events.CALL, None)
        mon.free_tool_id(tool)
    assert len(native) == 1, native


def test_a_host_sequence_runs_every_stage_n_times_and_matches_per_node(entry):
    n, reps = 100, 3
    guard = eagle._busy.BusyGuard("eagle.Graph")
    s1, meta, out = _stage(entry[0], n)
    s2 = _core.HostStage([[(entry[0], [meta.ctypes.data, out.ctypes.data],
                            [eexec.Partition(0, n, n)])]], False)
    _core.HostSequence(guard._g, [s1, s2]).launch(reps)
    ref = np.arange(n, dtype=np.float64)
    for _ in range(2 * reps):
        ref = ref * 2.0 + 1.0
    # launches 1-2 run the stage's plan, so a node may be cut into tiles
    assert meta[1] >= 2 * reps and np.array_equal(out, ref)


def test_a_record_or_python_node_graph_has_no_native_sequence(tmp_path):
    b1, b2, *_ = host_chain(tmp_path)
    g = eagle.Graph().add(b1).add_node(eagle.graph.opaque(lambda: None)).build()
    assert g._native is None
    g.launch()
    b1, b2, *_ = host_chain(tmp_path)
    assert eagle.Graph().add(b1).add(b2).build(record=True)._native is None


def test_the_busy_guard_holds_across_the_gil_release_of_a_native_launch(entry):
    guard = eagle._busy.BusyGuard("eagle.Graph")
    stage, meta, _out = _stage(entry[0], 8, fn=ctypes.cast(entry[1].nap, ctypes.c_void_p).value)
    seq = _core.HostSequence(guard._g, [stage])
    worker = threading.Thread(target=lambda: seq.launch(1))
    worker.start()
    t0 = time.monotonic()
    while meta[0] == 0 and time.monotonic() - t0 < 5:   # the first launch is inside its native call
        time.sleep(0.001)
    try:
        assert meta[0] == 1
        with pytest.raises(eagle.Busy):
            seq.launch(1)
    finally:
        meta[1] = 1                                       # let it return
        worker.join()
    seq.launch(1)


@pytest.mark.wall_clock
def test_the_glue_of_a_native_graph_launch_is_at_most_four_empty_native_calls(tmp_path):
    b1, *_ = host_chain(tmp_path)
    g = eagle.Graph().add(b1).build()
    stage = g._host_stages[0]
    empty = _core.HostStage([], False)

    def best(fn, reps=200_000):
        fn()
        top = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            top = min(top, (time.perf_counter() - t0) / reps)
        return top

    glue = best(g.launch) - best(stage.run)
    ratio = glue / best(empty.run)
    print(f"\nlaunch glue = {ratio:.2f} empty native calls")
    assert ratio <= 4.0


@pytest.mark.skipif(not hasattr(os, "sched_setaffinity"), reason="affinity masks are Linux")
def test_the_physical_core_count_is_measured_once_per_process():
    # it is read on launch paths (a small host run sizes its team from it): a sysfs walk per call cost a syscall per CPU
    code = ("import os; from eagle import _core; first = _core.physical_cores(); "
            "os.sched_setaffinity(0, {sorted(os.sched_getaffinity(0))[0]}); print(first, _core.physical_cores())")
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr[-2000:]
    first, again = map(int, run.stdout.split())
    assert again == first
