# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""One Graph, two threads: the second is told Busy (overlap FORCED by a
blocking node); distinct Graphs run concurrently; counters lose nothing."""

import threading

import numpy as np
import pytest

import eagle
from eagle import _counters
from eagle._busy import Busy, BusyGuard
from eagle.graph import opaque


def test_forced_overlap_raises_busy_naming_the_method_in_flight():
    entered, release = threading.Event(), threading.Event()

    def blocker():
        entered.set()
        assert release.wait(30)

    g = eagle.Graph().add_node(opaque(blocker, writes=[np.zeros(1)])).build()
    errs = []
    t = threading.Thread(target=g.launch)
    t.start()
    try:
        assert entered.wait(30)
        for call in (g.launch, g.enqueue):
            with pytest.raises(Busy) as exc:
                call()
            errs.append(str(exc.value))
        with pytest.raises(Busy):
            g.build()
    finally:
        release.set()
        t.join(30)
    assert errs[0] == ("eagle.Graph.launch: another call is still running on this Graph; "
                       "give each thread its own Graph, or wait for it")
    assert errs[1].startswith("eagle.Graph.enqueue: another call is still running on this Graph")
    g.launch()                                  # free again afterwards


def test_distinct_graphs_run_concurrently():
    both = threading.Barrier(2, timeout=30)

    def meet():
        both.wait()                              # deadlocks unless truly concurrent

    graphs = [eagle.Graph().add_node(opaque(meet)).build() for _ in range(2)]
    ts = [threading.Thread(target=g.launch) for g in graphs]
    [t.start() for t in ts]
    [t.join(30) for t in ts]
    assert not any(t.is_alive() for t in ts)


def test_graph_builds_counter_loses_no_increments():
    before = _counters.snapshot()["graph_builds"]
    n_threads, per = 8, 25

    def work():
        for _ in range(per):
            eagle.Graph().add_node(opaque(lambda: None)).build()

    ts = [threading.Thread(target=work) for _ in range(n_threads)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert _counters.snapshot()["graph_builds"] - before == n_threads * per


def test_busy_guard_is_exported_and_a_runtime_error():
    assert eagle.Busy is Busy and issubclass(Busy, RuntimeError)
    g = BusyGuard("x.Y")
    with g("a"):
        with pytest.raises(Busy, match="x.Y.b: another call is still running on this Y"):
            with g("b"):
                pass
