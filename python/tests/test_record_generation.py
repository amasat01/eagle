# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A Record names one launch: the graph refuses a new recording launch while
the Record is live, a stale Record is refused by a reverse, detach() keeps a
private copy, and a launch never overlaps a reverse of its Record."""

import threading

import numpy as np
import pytest

import eagle
from _reverse_block import Gate, block
from _reverse_fixtures import chain, plans

N = 32
SEED = {"sin.y": np.linspace(1, 2, N)}


@pytest.fixture
def graph(tmp_path):
    g, _z = chain(plans(tmp_path, "cpu"), np.linspace(-1, 1, N), 0.5, record=True)
    return g


def test_a_live_record_blocks_the_next_recording_launch_until_reversed(graph):
    graph.launch()
    first = graph.record
    assert first.generation == 1 and not first.held
    with pytest.raises(RuntimeError, match=(
            r"eagle.Graph.launch: the Record of the previous launch is still live; "
            r"reverse it, or Record.detach\(\) to keep a private copy")):
        graph.launch()
    eagle.diff.reverse(first, SEED)
    graph.launch()
    assert graph.record.generation == 2


def test_a_stale_record_is_refused_and_a_detached_one_is_kept(graph):
    graph.launch()
    live = graph.record
    kept = live.detach()
    before = eagle.diff.reverse(kept, SEED)
    graph._nodes[0].payload._planes["x"][...] += 1.0     # the next launch differs
    graph.launch()
    with pytest.raises(RuntimeError, match=(
            r"the forward ran again \(generation 1 -> 2\) since this Record was made")):
        eagle.diff.reverse(live, SEED)
    after = eagle.diff.reverse(kept, SEED)
    assert after["x"].tobytes() == before["x"].tobytes()
    assert after.report.generation == 1


def test_one_run_per_recording_launch(graph):
    with pytest.raises(ValueError, match=r"records one run per launch \(n=1\), got n=2"):
        graph.launch(2)
    with pytest.raises(RuntimeError, match="a record=True graph runs with launch()"):
        graph.enqueue()


def test_a_launch_during_a_reverse_is_busy():
    x, y = np.linspace(0, 1, N), np.zeros(N)
    gate = Gate()
    g = eagle.Graph().add_node(block(x, y, gate)).build(record=True).launch()
    rev = eagle.diff.Reverse(g)
    done = []
    t = threading.Thread(target=lambda: done.append(rev(g.record, {"y": np.ones(N)})))
    t.start()
    assert gate.entered.wait(30)
    assert g.record.held
    with pytest.raises(eagle.Busy, match="a reverse is reading this graph's Record"):
        g.launch()
    gate.go.set()
    t.join(30)
    np.testing.assert_array_equal(done[0]["x"], np.ones(N))
    g.launch()                          # reversed: the slot is free again


def test_forward_and_reverse_racing_never_give_a_third_value(graph):
    graph.launch()
    want = eagle.diff.reverse(graph.record, SEED)["x"].tobytes()
    seen, stop = set(), threading.Event()

    def forward():
        while not stop.is_set():
            try:
                graph.launch()
            except (RuntimeError, eagle.Busy):
                pass

    t = threading.Thread(target=forward)
    t.start()
    try:
        for _ in range(200):
            try:
                seen.add(eagle.diff.reverse(graph.record, SEED)["x"].tobytes())
            except (RuntimeError, eagle.Busy):
                pass
    finally:
        stop.set()
        t.join(30)
    assert seen <= {want}
