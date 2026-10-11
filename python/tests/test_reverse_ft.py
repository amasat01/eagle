# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Thread-safety: distinct Reverse objects reverse one Record concurrently and
agree bit for bit without writing it; one Reverse shared by two threads raises
Busy in the second (forced overlap); the reverses counter loses nothing."""

import threading

import numpy as np
import pytest

import eagle
from eagle import _counters
from _reverse_block import Gate, block
from _reverse_fixtures import chain, plans

N = 2048
THREADS = 8


def test_distinct_reverses_of_one_record_agree_and_leave_it_unwritten(tmp_path):
    g, _z = chain(plans(tmp_path, "cpu"), np.linspace(-1, 1, N), 0.5, record=True)
    g.launch()
    tape = g.record._tape
    before = {k: a.tobytes() for k, a in tape.arrays.items()}
    seed = {"sin.y": np.linspace(1, 2, N)}
    revs = [eagle.diff.Reverse(g) for _ in range(THREADS)]
    out = [None] * THREADS
    start = threading.Barrier(THREADS)
    counted = _counters.snapshot()["reverses"]

    def work(i):
        start.wait()
        for _ in range(10):
            out[i] = revs[i](g.record, seed)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert len({o["x"].tobytes() for o in out}) == 1
    assert len({o["a"].tobytes() for o in out}) == 1
    assert {k: a.tobytes() for k, a in tape.arrays.items()} == before
    assert _counters.snapshot()["reverses"] - counted == 10 * THREADS


def test_one_reverse_shared_by_two_threads_is_busy():
    x, y = np.linspace(0, 1, 8), np.zeros(8)
    gate = Gate()
    g = eagle.Graph().add_node(block(x, y, gate)).build(record=True).launch()
    rev = eagle.diff.Reverse(g)
    t = threading.Thread(target=lambda: rev(g.record, {"y": np.ones(8)}))
    t.start()
    assert gate.entered.wait(30)
    with pytest.raises(eagle.Busy, match="eagle.diff.Reverse.__call__"):
        rev(g.record, {"y": np.ones(8)})
    gate.go.set()
    t.join(30)
