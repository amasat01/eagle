# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The ``copy`` kind (a seed copy, differentiated) and the ``cell`` kind (a
decision cell, never differentiated): gradients equal the hand-built twin and
central differences."""

import numpy as np
import pytest

import eagle
import eagle.diff
import eagle.plan
from eagle import SkipGuard, skippable
from eagle._graph_kinds import _cell, _copy

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Scalar  # noqa: E402
from hawk import math as m  # noqa: E402

N = 12
X0 = np.linspace(-1.1, 0.9, N)
SEED = np.linspace(1.0, 2.0, N)
H = 1e-6


@hawk.kernel
def cc_f1(x: Scalar, y: Mutable[Scalar]):
    y = m.sin(x) + 0.3 * x * x


@hawk.kernel
def cc_f2(x: Scalar, y: Mutable[Scalar]):
    y = x * m.cos(x)


@hawk.kernel
def cc_f3(x: Scalar, y: Mutable[Scalar]):
    y = 0.2 * x * x * x + x


@pytest.fixture(scope="module")
def ps(tmp_path_factory):
    return eagle.plan.auto([cc_f1, cc_f2, cc_f3], targets=("host",),
                           cache_dir=str(tmp_path_factory.mktemp("hawk_cache")))


def test_a_seed_copy_passes_the_cotangent_to_its_source(ps):
    f1, _f2, _f3 = ps
    x, s, y = X0.copy(), np.zeros(N), np.zeros(N)
    g = eagle.Graph().add_node(_copy(x, s), name="seed").add(f1.bind(x=s, y=y), name="f1")
    assert g.build(record=True).launch() and np.array_equal(s, x)
    got = eagle.diff.reverse(g.record, {"f1.y": SEED}, wrt=["seed.src"])["src"]
    t = eagle.Graph().add(f1.bind(x=x, y=np.zeros(N)), name="f1").build(record=True)
    t.launch()
    twin = eagle.diff.reverse(t.record, {"f1.y": SEED}, wrt=["f1.x"])["x"]
    np.testing.assert_array_equal(got, twin)
    out = lambda d: (f1.bind(x=x + d, y=(o := np.zeros(N))).launch(), o)[1]     # noqa: E731
    np.testing.assert_allclose(got, (out(H) - out(-H)) * SEED / (2 * H), rtol=1e-6, atol=1e-9)


def _if_else(ps, on, x):
    f1, f2, f3 = ps
    a, b, c = np.zeros(N), np.zeros(N), np.zeros(N)
    flag, one = np.array([on], dtype=np.uint32), np.ones(1, dtype=np.uint32)
    decide = _cell(lambda: None, reads=[a], writes=[flag], name="decide")
    g = (eagle.Graph().add(f1.bind(x=x, y=a), name="f1").add_node(decide)
         .add_node(skippable(f2.bind(x=a, y=b).launch, SkipGuard(flag, 0)), name="then")
         .add_node(skippable(f3.bind(x=a, y=c).launch, SkipGuard(flag, 0, one, 0)), name="else"))
    return g, b, c


@pytest.mark.parametrize("on", [1, 0])
def test_a_decision_cell_is_not_differentiated(ps, on):
    g, b, c = _if_else(ps, on, X0.copy())
    assert [n.kind for n in g._nodes] == ["launch", "cell", "skippable", "skippable"]
    assert [n.grad for n in g._nodes] == [True, False, True, True]
    g.build(record=True).launch()
    got = eagle.diff.reverse(g.record, {"then.y": SEED, "else.y": 2 * SEED}, wrt=["f1.x"])["x"]

    def out(d):
        _g, bb, cc = _if_else(ps, on, X0 + d)
        _g.build().launch()
        return bb * SEED + cc * 2 * SEED
    np.testing.assert_allclose(got, (out(H) - out(-H)) / (2 * H), rtol=1e-6, atol=1e-9)
