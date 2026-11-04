# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``Graph.host_table()``: the CPU stages as an immutable table another runtime runs
(plugin/graph_handoff.h). It equals ``launch`` bitwise, holds the graph alive, goes
stale on a bind, and is refused for a node of your own code."""

import ctypes

import numpy as np
import pytest

import eagle
from eagle.graph import opaque

from _graph_fixtures import host_fork

_RUN = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)


def run_table(table):
    """``eagle_host_table_run``: the entry is the table's first member."""
    entry = ctypes.cast(ctypes.c_void_p(table.handle), ctypes.POINTER(ctypes.c_void_p))[0]
    return _RUN(entry)(ctypes.c_void_p(table.handle))


def chain_graph(tmp_path, n=40000):
    chains, arrays, _ = host_fork(tmp_path, 1, n, depth=2)
    g = eagle.Graph()
    for b in chains[0]:
        g.add(b)
    return g.build(backend="cpu"), arrays[0]


def test_table_equals_launch_bitwise(tmp_path):
    g, (a, b) = chain_graph(tmp_path)
    a0 = a.copy()
    g.launch()
    want = a.copy()
    a[:] = a0
    table = g.host_table()
    assert run_table(table) == 0
    assert np.array_equal(a, want)


def test_a_group_of_branches_is_a_table_too(tmp_path):
    chains, arrays, expected = host_fork(tmp_path, 3, 5000)
    g = eagle.Graph().add_group([list(chain) for chain in chains]).build(backend="cpu")
    assert run_table(g.host_table()) == 0
    for (a, _), ref in zip(arrays, expected):
        assert np.array_equal(a, ref)


def test_table_holds_the_graph_alive(tmp_path):
    import gc
    import weakref

    g, _ = chain_graph(tmp_path, 1000)
    ref = weakref.ref(g)
    table = g.host_table()
    del g
    gc.collect()
    assert ref() is not None
    assert run_table(table) == 0


def test_a_bind_makes_the_table_stale(tmp_path):
    g, (a, b) = chain_graph(tmp_path, 1000)
    table = g.host_table()
    assert not table.stale
    g._rebind([(a, a.copy())])
    assert table.stale
    before = b.copy()
    assert run_table(table) == 1
    assert np.array_equal(b, before)
    assert run_table(g.host_table()) == 0


def test_a_node_of_your_own_code_is_refused(tmp_path):
    chains, arrays, _ = host_fork(tmp_path, 1, 64, depth=1)
    x = arrays[0][0]
    g = eagle.Graph().add(chains[0][0]).add_node(opaque(lambda: None, reads=[x])).build(backend="cpu")
    with pytest.raises(RuntimeError, match="native host entry"):
        g.host_table()
