# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A ``wait`` orders one branch of a fork after a node of an earlier branch:
the sibling overlap it covers is no race, the graph gains one edge, and on a
GPU the wait is an edge of the captured graph, not a node."""

import os
import re

import numpy as np
import pytest

import eagle
from eagle._graph_kinds import _wait
from eagle.graph import opaque

GPU = os.environ.get("EAGLE_TEST_GPU") == "1"
XPS = ["numpy"] + (["cupy"] if GPU else [])


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def fork(xpn, with_wait=True, declare_race=True):
    xp = np if xpn == "numpy" else __import__("cupy")
    x, y, z = (xp.zeros(4) for _ in range(3))
    x0 = xp.asarray(np.array([1.0, 2, 3, 4]))
    produce = opaque(lambda: xp.multiply(x0, 2.0, out=x), reads=[x0], writes=[x], name="produce")
    consume = opaque(lambda: xp.add(x, 1.0, out=y), reads=[x] if with_wait else [], writes=[y], name="consume")
    other = opaque(lambda: xp.add(z, 1.0, out=z), reads=[z], writes=[z], name="other")
    g = eagle.Graph()
    g.add_group([[produce], ([_wait(x)] if with_wait else []) + [consume], [other]])
    if xp is not np:
        for n in (produce, consume, other):
            n()
        x[...] = 0
        y[...] = 0
        z[...] = 0
    return g, (x, y, z)


@pytest.mark.parametrize("xpn", XPS)
def test_a_wait_orders_the_overlap_and_the_values_hold(xpn):
    g, (x, y, z) = fork(xpn)
    g.build().launch()
    np.testing.assert_array_equal(host(y), np.array([3.0, 5, 7, 9]))
    assert float(host(z)[0]) == 1.0


def test_the_overlap_without_a_wait_is_refused_and_a_wait_adds_one_edge():
    a, b = np.zeros(2), np.zeros(2)
    w = opaque(lambda: None, reads=[], writes=[a], name="w")
    r = opaque(lambda: None, reads=[a], writes=[b], name="r")
    with pytest.raises(ValueError, match="siblings of one fork/join group"):
        eagle.Graph().add_group([[w], [r]]).build()
    g, _ = fork("numpy")
    edges = g.build().edges()
    labels = g.node_labels()
    assert any(labels[e.src] == "produce" and labels[e.dst] == "consume" and e.kind == "RAW"
               for e in edges)


def test_a_wait_outside_a_region_or_on_an_unwritten_plane_is_refused():
    with pytest.raises(RuntimeError, match="only inside a parallel region"):
        eagle.Graph().add_node(_wait(np.zeros(2))).build()
    a, b = np.zeros(2), np.zeros(2)
    n1 = opaque(lambda: None, reads=[a], writes=[a], name="n1")
    n2 = opaque(lambda: None, reads=[b], writes=[b], name="n2")
    with pytest.raises(ValueError, match="not written by an earlier branch"):
        eagle.Graph().add_group([[n1], [_wait(b), n2]]).build()


@pytest.mark.skipif(not GPU, reason="EAGLE_TEST_GPU=1 under the pool lock")
def test_a_wait_is_an_edge_and_no_node_on_the_captured_graph():
    counts = {}
    for with_wait in (True, False):
        g, _ = fork("cupy", with_wait=with_wait)
        g.build().launch()
        dot = g._pipe._dot_str
        counts[with_wait] = (len(re.findall(r"^\s*\"?\w+\"?\s*\[", dot, re.M)), len(re.findall(r"->", dot)))
    assert counts[True][0] == counts[False][0] and counts[True][1] == counts[False][1] + 1
