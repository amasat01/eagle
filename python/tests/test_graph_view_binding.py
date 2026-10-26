# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The C++ graph view (eagle/graph/*.h, eagle/util/Busy.h) through its
binding: a thin layer, so these rows pin the rows in and the text out."""

import pytest

from eagle import _core
from eagle._busy import Busy


def _desc():
    return _core.GraphDescription(
        [("scale", "launch", "", -1, False), ("left", "launch", "", -1, False),
         ("right", "launch", "", -1, True), ("join", "launch", "", 0, False)],
        [(0, 1, "y", "RAW"), (0, 2, "y", "RAW"), (1, 3, "a", "RAW"), (2, 3, "b", "WAW")],
        [("if", "skip if guard", -1)])


def test_description_round_trip():
    d = _desc()
    assert (d.num_nodes(), d.num_edges()) == (4, 4)
    layer, order, xy = d.layout()
    assert layer == [0, 1, 1, 2] and sorted(order[1:3]) == [0, 1]
    assert xy[0] == (0.0, 0.0) and xy[3][1] == 2.0
    text = d.ascii()
    assert "skip if guard" in text and "stage 2" in text
    assert all(ord(c) < 128 for c in d.ascii(unicode=False))
    assert d.ascii() == d.ascii()
    m, g = d.mermaid(), d.dot()
    assert m.startswith("flowchart TD") and m.count("-->") + m.count("==>") == 4
    assert g.startswith("digraph eagle") and g.count(" -> ") == 4


def test_malformed_description_is_refused():
    with pytest.raises(ValueError, match="dangling"):
        _core.GraphDescription([("a", "k", "", -1, False)], [(0, 3, "", "RAW")])
    with pytest.raises(ValueError, match="cycle"):
        _core.GraphDescription([("a", "k", "", -1, False), ("b", "k", "", -1, False)],
                               [(0, 1, "", "RAW"), (1, 0, "", "RAW")])
    with pytest.raises(ValueError, match="RAW"):
        _core.GraphDescription([("a", "k", "", -1, False), ("b", "k", "", -1, False)],
                               [(0, 1, "", "XYZ")])


def _ref(name, node, base, shape, strides, access, item=8):
    return (name, node, base, shape, strides, item, access)


def test_derive_edges_is_stride_aware_and_orders_hazards():
    col = lambda j, n, a: _ref(f"c{j}", n, 4096 + 8 * j, [1000], [32], a)
    assert _core.derive_edges([col(0, 0, 1), col(1, 1, 0)]) == []
    assert _core.derive_edges([col(1, 0, 1), col(1, 1, 0)]) == [(0, 1, "c1", "RAW")]
    chain = [_ref("A", 0, 1000, [4], [8], 1), _ref("A", 1, 1000, [4], [8], 0),
             _ref("B", 1, 2000, [4], [8], 1), _ref("A", 2, 1000, [4], [8], 1)]
    assert [(s, d, k) for s, d, _p, k in _core.derive_edges(chain)] == [
        (0, 1, "RAW"), (0, 2, "WAW"), (1, 2, "WAR")]
    assert _core.planes_overlap(chain[0], chain[3]) and not _core.planes_overlap(
        chain[0], chain[2])


def test_sibling_conflicts_and_group_exclusion():
    refs = [_ref("w", 0, 1000, [8], [8], 1), _ref("r", 1, 1032, [8], [8], 0)]
    got = _core.sibling_conflicts(refs, [[[0], [1]]])
    assert got == [(0, 1, "w", "r", "RAW")]
    assert _core.sibling_conflicts(refs, [[[0, 1]]]) == []      # one branch: ordered
    assert _core.derive_edges(refs, [[[0], [1]]]) == []          # siblings: no edge
    assert len(_core.derive_edges(refs)) == 1


def test_busy_guard_is_the_cpp_flag_and_raises_the_python_type():
    g = _core.BusyGuard("eagle.Thing")
    g.enter("launch")
    with pytest.raises(Busy, match=r"^eagle\.Thing\.build: another call is still running on this Thing; give each thread its own Thing, or wait for it$"):
        g.enter("build")
    assert g.current() == "launch"
    g.leave()
    g.enter("build")
    g.leave()
