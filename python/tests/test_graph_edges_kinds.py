# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Edges are derived, never declared: RAW / WAR / WAW on a 3-node chain, on
the USER array, stride-aware."""

import numpy as np

import eagle
from eagle.graph import opaque
from eagle._graph_edges import RAW, WAR, WAW, arrays_overlap


def _nop():
    pass


def test_three_node_chain_has_raw_war_waw():
    a, b = np.zeros(4), np.zeros(4)
    n0 = opaque(_nop, reads=[a], writes=[b])     # b = f(a)
    n1 = opaque(_nop, reads=[b], writes=[a])     # a = g(b): RAW on b, WAR on a
    n2 = opaque(_nop, reads=[], writes=[a])      # a = h(): WAW with n1, WAR with n0
    g = eagle.Graph().add_node(n0).add_node(n1).add_node(n2)
    got = {(e.src, e.dst, e.kind) for e in g.edges()}
    assert (0, 1, RAW) in got and (0, 1, WAR) in got
    assert (1, 2, WAW) in got
    assert (0, 2, WAR) in got
    assert all(isinstance(e.plane, str) for e in g.edges())


def test_reads_only_make_no_edge_and_uniform_free():
    a = np.zeros(4)
    g = (eagle.Graph().add_node(opaque(_nop, reads=[a]))
         .add_node(opaque(_nop, reads=[a])))
    assert g.edges() == ()


def test_overlap_is_stride_aware():
    base = np.zeros(10)
    assert not arrays_overlap(base[::2], base[1::2])      # interleaved: exact
    assert arrays_overlap(base[::2], base[4:6])
    assert not arrays_overlap(base[:5], base[5:])
    assert arrays_overlap(base, base[3:4])
    m = np.zeros((4, 4))
    assert not arrays_overlap(m[:, 0], m[:, 1])
    assert arrays_overlap(m[:, 0], m[2])


def test_edges_follow_the_user_arrays_not_names():
    base = np.zeros(8)
    lo, hi = base[:4], base[4:]
    g = (eagle.Graph().add_node(opaque(_nop, writes=[lo]))
         .add_node(opaque(_nop, reads=[hi]))
         .add_node(opaque(_nop, reads=[base[2:6]])))
    assert {(e.src, e.dst) for e in g.edges()} == {(0, 2)}


def test_edges_explain_themselves_in_plain_words():
    from eagle import _graph_edges as E

    assert [E.Edge(0, 1, "x", k).meaning for k in (RAW, WAR, WAW)] == [
        "data flows", "must wait to overwrite", "same output"]
    assert E.SHARES_MEMORY_MAX_WORK == E.OVERLAP_MAX_WORK > 0
