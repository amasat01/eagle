# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Graph on the host: launches of real plans, conflicts, conditionals,
counters, refusals."""

from types import SimpleNamespace

import numpy as np
import pytest

import eagle
from eagle import _counters
from eagle import SkipGuard, repeat_while, skippable
from eagle.graph import opaque

from _graph_fixtures import host_chain


def test_chain_of_two_plans_derives_a_raw_edge_and_runs(tmp_path):
    b1, b2, x, y, z = host_chain(tmp_path)
    before = _counters.snapshot()
    g = eagle.Graph().add(b1, name="scale").add(b2, name="addone").build()
    assert g.backend == "cpu" and g.node_labels() == ["scale", "addone"]
    assert [(e.src, e.dst, e.plane, e.kind) for e in g.edges()] == [(0, 1, "y", "RAW")]
    g.launch()
    np.testing.assert_array_equal(z, 2 * x + 1)
    assert _counters.snapshot()["graph_builds"] - before["graph_builds"] == 1
    assert _counters.snapshot()["captures"] == before["captures"]


def test_two_launches_of_one_graph(tmp_path):
    b1, b2, x, y, z = host_chain(tmp_path)
    g = eagle.Graph().add(b1).add(b2).build()
    g.launch()
    x += 1.0
    g.launch(1)
    np.testing.assert_array_equal(z, 2 * x + 1)


def test_group_conflict_refused_on_numpy(tmp_path):
    b1, b2, x, y, z = host_chain(tmp_path)
    g = eagle.Graph().add_group([b1, b2])        # b2 reads what b1 writes
    with pytest.raises(ValueError, match="siblings of one fork/join group"):
        g.build()


def test_independent_group_branches_build_and_run(tmp_path):
    b1, _b2, x, y, _z = host_chain(tmp_path)
    q = np.zeros(8)
    ran = []
    g = eagle.Graph().add_group([[b1], [opaque(lambda: ran.append(1), reads=[x], writes=[q])]],
                                names=["p", "q"])
    g.build().launch()
    assert ran == [1] and sorted(g.node_labels()) == ["p", "q"]
    np.testing.assert_array_equal(y, 2 * x)


def test_copy_bound_plane_is_refused_with_the_layout_fix():
    fake = SimpleNamespace(_arg_spec=(), _planes={}, launch=lambda: None,
                           _copies={"pos": object()})
    with pytest.raises(ValueError, match=r"COPY-bound.*contiguous"):
        eagle.Graph().add(fake)


def test_derived_plan_is_refused_at_add():
    fake = SimpleNamespace(_arg_spec=(), _planes={}, launch=lambda: None, _copies={},
                           derivative={"kind": "vjp", "primal": "p"})
    with pytest.raises(ValueError, match="vjp derivative kernel"):
        eagle.Graph().add(fake)


def test_skippable_and_repeat_while_run_their_host_arms():
    flag = np.array([1], dtype=np.uint32)
    count = np.zeros(1)
    calls = []

    def body():
        calls.append(1)
        if len(calls) == 3:
            flag[0] = 0
    node = opaque(body, writes=[count])
    g = (eagle.Graph()
         .add_node(repeat_while(node, SkipGuard.nonzero(flag), 10), name="loop")
         .add_node(skippable(opaque(lambda: calls.append("s")), SkipGuard.nonzero(flag)),
                   name="after"))
    g.build().launch()
    assert calls == [1, 1, 1]                    # loop ran 3 times, skippable skipped
    flag[0] = 1
    calls.clear()
    g2 = eagle.Graph().add_node(skippable(opaque(lambda: calls.append("s")), SkipGuard.nonzero(flag)))
    g2.build().launch()
    assert calls == ["s"]


def test_conditional_nodes_are_single_owner():
    flag = np.ones(1, dtype=np.uint32)
    node = skippable(opaque(lambda: None), SkipGuard.nonzero(flag))
    eagle.Graph().add_node(node)
    with pytest.raises(ValueError, match="already belongs to another Graph"):
        eagle.Graph().add_node(node)


def test_conditional_over_a_bound_plan_derives_its_planes(tmp_path):
    b1, b2, x, y, z = host_chain(tmp_path)
    flag = np.ones(1, dtype=np.uint32)
    g = (eagle.Graph().add(b1, name="scale")
         .add_node(skippable(b2.launch, SkipGuard.nonzero(flag)), name="maybe"))
    g.build()
    assert ("y" in {e.plane for e in g.edges()})
    g.launch()
    np.testing.assert_array_equal(z, 2 * x + 1)


def test_refusals():
    g = eagle.Graph()
    with pytest.raises(ValueError, match="empty"):
        g.build()
    with pytest.raises(ValueError, match="at least 2"):
        eagle.Graph().add_group([opaque(lambda: None)])
    with pytest.raises(TypeError):
        eagle.Graph().add(object())
    with pytest.raises(ValueError, match="record=True is not available"):
        eagle.Graph().add_node(opaque(lambda: None)).build(record=True)
    built = eagle.Graph().add_node(opaque(lambda: None)).build()
    with pytest.raises(RuntimeError, match="is built"):
        built.add_node(opaque(lambda: None))
    with pytest.raises(RuntimeError, match="build"):
        eagle.Graph().launch()
    with pytest.raises(ValueError, match="backend must be one of"):
        eagle.Graph().add_node(opaque(lambda: None)).build(backend="tpu")
    with pytest.raises(ValueError, match="nesting"):
        flag = np.ones(1, dtype=np.uint32)
        inner = skippable(opaque(lambda: None), SkipGuard.nonzero(flag))
        eagle.Graph().add_node(skippable(inner, SkipGuard.nonzero(flag)))
