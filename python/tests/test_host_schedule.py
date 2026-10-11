# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The two CPU schedules of eagle.Graph: the pure rule, what a built graph
reports, bit-identity with a serial run, and the native branch runner."""

import threading

import numpy as np
import pytest

import eagle
from eagle import _core
from eagle._host_loop import _omp_runtime
from eagle.graph import opaque

from _graph_fixtures import host_fork

MAX_N = _core.BRANCH_TASKS_MAX_SAMPLES


@pytest.fixture
def team():
    """``set(threads)`` resizes the host team for one test, then restores it."""
    omp = _omp_runtime()
    if omp is None:
        pytest.skip("no OpenMP runtime to steer")
    get, put = omp
    original = get()
    yield put
    put(original)


def _need_team(minimum=2):
    if _core.host_threads() < minimum:
        pytest.skip(f"needs a host team of at least {minimum} threads")


def _stage(chains):
    return eagle.Graph().add_group(chains).build()


@pytest.mark.parametrize("n, branches, threads, native, expected", [
    (MAX_N, 2, 2, True, "branch_tasks"),
    (MAX_N + 1, 2, 2, True, "work_share"),          # one past the sample limit
    (1, 2, 2, True, "branch_tasks"),
    (1024, 1, 8, True, "work_share"),               # one branch: nothing to overlap
    (1024, 2, 1, True, "work_share"),               # one thread: nothing to overlap
    (1024, 2, 2, False, "work_share"),              # a Python host arm in the stage
    (1024, 8, 8, True, "branch_tasks"),
])
def test_resolve_pins_the_boundary_values(n, branches, threads, native, expected):
    assert _core.resolve_host_schedule(n, branches, threads, native) == expected


def test_a_small_fork_runs_as_branch_tasks_and_matches_the_oracle(tmp_path):
    _need_team()
    chains, arrays, expected = host_fork(tmp_path, branches=4, n=1024, depth=3)
    g = _stage(chains)
    assert g.schedule == ("branch_tasks",)
    g.launch()
    for (a, _b), ref in zip(arrays, expected):
        np.testing.assert_array_equal(a, ref)


def test_a_large_fork_and_a_plain_chain_work_share(tmp_path):
    chains, arrays, expected = host_fork(tmp_path, branches=2, n=MAX_N + 1, depth=2)
    g = _stage(chains)
    assert g.schedule == ("work_share",)
    g.launch()
    for (a, _b), ref in zip(arrays, expected):
        np.testing.assert_array_equal(a, ref)
    small, _arrays, _ref = host_fork(tmp_path, branches=1, n=64)
    assert eagle.Graph().add(small[0][0]).add(small[0][1]).build().schedule == ("work_share",) * 2


def test_both_schedules_equal_a_serial_run_bitwise(tmp_path):
    _need_team()
    chains, arrays, _ = host_fork(tmp_path, branches=4, n=2048, depth=4)
    for chain in chains:                          # the serial reference
        for bound in chain:
            _core.run_host_serial(bound._entry, bound._addrs, bound._parts[0])
    serial = [a.copy() for a, _b in arrays]
    g = _stage(chains)
    base = [np.arange(a.size, dtype=np.float64) + k for k, (a, _b) in enumerate(arrays)]

    def reset():
        for (a, _b), start in zip(arrays, base):
            a[:] = start

    reset(); g.launch()
    tasks = [a.copy() for a, _b in arrays]
    reset()
    for chain in chains:                          # work_share: each node over the whole team
        for bound in chain:
            bound.launch()
    shared = [a.copy() for a, _b in arrays]
    assert g.schedule == ("branch_tasks",)
    for s, t, w in zip(serial, tasks, shared):
        assert np.array_equal(s, t) and np.array_equal(s, w)


def test_one_thread_equals_the_full_team_bitwise(tmp_path, team):
    _need_team()
    results = {}
    for threads in (1, _core.host_threads()):
        team(threads)
        chains, arrays, _ = host_fork(tmp_path, branches=4, n=1024, depth=3)
        g = _stage(chains)
        g.launch()
        results[threads] = ([a.copy() for a, _b in arrays], g.schedule)
    (one, how_one), (many, how_many) = results.values()
    assert how_one == ("work_share",) and how_many == ("branch_tasks",)
    assert all(np.array_equal(x, y) for x, y in zip(one, many))


def test_a_stage_with_a_python_host_arm_work_shares(tmp_path):
    _need_team()
    chains, arrays, _ = host_fork(tmp_path, branches=2, n=256)
    ran = []
    side = np.zeros(4)
    g = (eagle.Graph()
         .add_group([chains[0], [opaque(lambda: ran.append(1), reads=[arrays[1][0]], writes=[side])]])
         .build())
    assert g.schedule == ("work_share",)
    g.launch()
    assert ran == [1]


def test_graph_schedule_is_reported_per_stage(tmp_path):
    _need_team()
    chains, _arrays, _ = host_fork(tmp_path, branches=3, n=256)
    g = (eagle.Graph().add(chains[0][0]).add_group([[c[1]] for c in chains[1:]] + [[chains[0][1]]]))
    assert g.build().schedule == ("work_share", "branch_tasks")
    assert "branch_tasks" in str(g)


def test_run_host_branches_releases_the_gil(tmp_path):
    _need_team()
    chains, _arrays, _ = host_fork(tmp_path, branches=2, n=MAX_N, depth=1)
    node = lambda bound: (bound._entry, bound._addrs, bound._parts)   # noqa: E731
    long_branch = [node(chains[0][0]), node(chains[0][1])] * 20000
    stage = [long_branch, [node(chains[1][0]), node(chains[1][1])] * 20000]
    ticks, stop = [0], threading.Event()

    def count():
        while not stop.is_set():
            ticks[0] += 1

    t = threading.Thread(target=count)
    t.start()
    try:
        while ticks[0] == 0:
            pass
        before = ticks[0]
        _core.run_host_branches(stage)
        progressed = ticks[0] - before
    finally:
        stop.set()
        t.join(60)
    assert not t.is_alive()
    assert progressed > 100


def test_a_null_entry_is_refused_not_called():
    with pytest.raises(RuntimeError, match="null host entry"):
        _core.run_host_branches([[(0, [], [_core.Partition(0, 1, 1)])]])


def test_a_graph_in_a_native_stage_is_busy_to_a_second_thread_and_distinct_graphs_overlap(tmp_path):
    _need_team()
    chains, _arrays, _ref = host_fork(tmp_path, branches=2, n=MAX_N, depth=2)
    g = _stage(chains)
    assert g.schedule == ("branch_tasks",)
    started, outcome = threading.Event(), []

    def long_launch():
        started.set()
        g.launch(3000)                    # the GIL is released for each native stage

    t = threading.Thread(target=long_launch)
    t.start()
    try:
        assert started.wait(30)
        for _ in range(100000):
            try:
                g.launch()
            except eagle.Busy:
                outcome.append("busy")
                break
            if not t.is_alive():
                break
    finally:
        t.join(60)
    assert not t.is_alive()
    assert outcome == ["busy"]

    other, _arrays, _ref = host_fork(tmp_path, branches=2, n=MAX_N, depth=2)
    h = _stage(other)
    errors = []

    def run(graph):
        try:
            graph.launch(50)
        except Exception as e:             # noqa: BLE001
            errors.append(e)

    ts = [threading.Thread(target=run, args=(x,)) for x in (g, h)]
    [x.start() for x in ts]
    [x.join(60) for x in ts]
    assert not any(x.is_alive() for x in ts)
    assert errors == []


_STRESS = """
import faulthandler, pathlib, sys, tempfile, threading
faulthandler.dump_traceback_later(110, exit=True)
sys.path.insert(0, sys.argv[1])
from eagle import _core
from _graph_fixtures import host_fork

root = pathlib.Path(tempfile.mkdtemp())
keep, jobs = [], []
for k in range(4):
    chains, arrays, _ = host_fork(root / str(k), branches=4, n=1024, depth=2)
    keep.append((chains, arrays))
    stage = [[(b._entry, b._addrs, b._parts) for b in c] for c in chains]
    if k % 2 == 0:
        jobs.append(lambda stage=stage: _core.run_host_branches(stage))
    else:
        jobs.append(lambda chains=chains: [b.launch() for c in chains for b in c])

def repeat(job):
    for _ in range(2000):
        job()

threads = [threading.Thread(target=repeat, args=(j,)) for j in jobs]
[t.start() for t in threads]
[t.join() for t in threads]
print("ok")
"""


def test_several_threads_running_stages_at_once_neither_hang_nor_abort(tmp_path):
    # the child builds its stages with hawk; without it there is nothing to stress
    pytest.importorskip("hawk")
    import os
    import subprocess
    import sys

    # Many more OpenMP threads than cores, on purpose. Idle threads sleep
    # instead of spinning, so on a busy machine the timeout still measures a
    # hang rather than how much CPU is left.
    env = dict(os.environ, OMP_NUM_THREADS="32", OMP_WAIT_POLICY="passive")
    done = subprocess.run([sys.executable, "-c", _STRESS, os.path.dirname(__file__)],
                          env=env, capture_output=True, text=True, timeout=150)
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-2000:]
