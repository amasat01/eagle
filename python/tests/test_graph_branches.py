# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""IF/ELSE and SWITCH conditional nodes (driver >= 12.8): one node with an arm
per step. The device legs skip where the driver lacks them; the same
``Graph._if_else`` / ``_chain`` entry points then build negated IFs."""

import numpy as np
import pytest

import eagle
from eagle._conditional import Branches, SkipGuard, has_branch_nodes
from eagle.graph import opaque

cupy = pytest.importorskip("cupy")


def test_branches_called_directly_pick_their_arm():
    ran = []
    arms = [lambda i=i: ran.append(i) for i in range(3)]
    Branches(arms[:2], SkipGuard.nonzero(np.array([1], np.uint32)))()
    Branches(arms[:2], SkipGuard.nonzero(np.array([0], np.uint32)))()
    Branches(arms, SkipGuard.nonzero(np.array([2], np.uint32)), switch=True)()
    Branches(arms, SkipGuard.nonzero(np.array([7], np.uint32)), switch=True)()
    assert ran == [0, 1, 2]


def _arm(x, v):
    return opaque(lambda: cupy.add(x, v, out=x), reads=[x], writes=[x], name=f"add{v}")


@pytest.mark.skipif(not has_branch_nodes(), reason="driver has no IF/ELSE conditional node (needs >= 12.8)")
def test_if_else_is_one_node_with_two_arms():
    x, f = cupy.zeros(1), cupy.asarray(np.array([1], np.uint32))
    then, other = _arm(x, 1.0), _arm(x, 10.0)
    g = eagle.Graph()._if_else(SkipGuard.nonzero(f), then, other)
    assert [n.kind for n in g._nodes] == ["copy", "branches"]
    then(), other()
    for flag, want in ((1, 1.0), (0, 10.0), (1, 1.0)):
        x[...] = 0
        f[...] = flag
        g.build().launch() if not g._built else g.launch()
        assert float(x.get()[0]) == want


@pytest.mark.skipif(not has_branch_nodes(switch=True), reason="driver has no SWITCH conditional node (needs >= 12.8)")
def test_switch_runs_the_arm_its_index_names():
    x, index = cupy.zeros(1), cupy.asarray(np.array([0], np.uint32))
    arms = [_arm(x, 1.0), _arm(x, 10.0), _arm(x, 100.0)]
    g = eagle.Graph()._chain(index, arms[:2], arms[2])
    assert [n.kind for n in g._nodes] == ["copy", "branches"]
    for a in arms:
        a()
    g.build()
    for i, want in ((0, 1.0), (1, 10.0), (2, 100.0), (5, 0.0)):
        x[...] = 0
        index[...] = i
        g.launch()
        assert float(x.get()[0]) == want
