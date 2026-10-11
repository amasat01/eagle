# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Every refusal of eagle.diff names the problem and what to do instead."""

import numpy as np
import pytest

import eagle
import eagle.plan
from eagle import SkipGuard, skippable
from eagle.graph import opaque
from _reverse_fixtures import chain, plans

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Scalar  # noqa: E402

N = 16


@hawk.kernel
def rf_accumulate(x: Scalar, y: Mutable[Scalar]):
    y = y * x + y


def _chain(tmp_path):
    g, _z = chain(plans(tmp_path, "cpu"), np.linspace(-1, 1, N), 0.5, record=True)
    return g.launch()


def test_an_opaque_node_is_refused(tmp_path):
    sq = plans(tmp_path, "cpu")[0]
    x, y, q = np.linspace(0, 1, N), np.zeros(N), np.zeros(N)
    g = (eagle.Graph().add(sq.bind(x=x, a=1.0, y=y), name="square")
         .add_node(opaque(lambda: np.copyto(q, y), reads=[y], writes=[q]), name="custom")
         .build(record=True))
    with pytest.raises(ValueError, match="eagle.diff.Reverse: node 'custom' is opaque"):
        eagle.diff.Reverse(g)


def test_a_conditional_is_refused(tmp_path):
    sq = plans(tmp_path, "cpu")[0]
    x, y = np.linspace(0, 1, N), np.zeros(N)
    cell, base = np.zeros(1, dtype=np.int32), np.zeros(1, dtype=np.int32)
    maybe = skippable(sq.bind(x=x, a=1.0, y=y).launch, SkipGuard(cell, 0, base, 0))
    g = eagle.Graph().add_node(maybe, name="maybe").build(record=True)
    with pytest.raises(ValueError, match="node 'maybe' is inside a skippable or repeat_while"):
        eagle.diff.Reverse(g)


def test_a_prior_read_is_refused_with_its_fix(tmp_path):
    (p,) = eagle.plan.auto([rf_accumulate], targets=("host",), cache_dir=str(tmp_path / "c"))
    x, y = np.linspace(0, 1, N), np.ones(N)
    g = eagle.Graph().add(p.bind(x=x, y=y), name="acc").build(record=True)
    with pytest.raises(ValueError, match="eagle.diff.Reverse: 'y' is read before it is assigned"):
        eagle.diff.Reverse(g)


def test_second_order_is_refused(tmp_path):
    g = _chain(tmp_path)
    rev = eagle.diff.Reverse(g)
    rev(g.record, {"sin.y": np.ones(N)})
    (programs,) = rev._programs.values()
    (program,) = programs.values()
    inner = program.graph
    with pytest.raises(ValueError, match=r"is itself a reverse pass; reversing a reverse \(second order\)"):
        eagle.diff.Reverse(inner)


def test_the_graph_must_be_recorded(tmp_path):
    g, _z = chain(plans(tmp_path, "cpu"), np.linspace(-1, 1, N), 0.5)
    with pytest.raises(ValueError, match=r"build it with g.build\(record=True\)"):
        eagle.diff.Reverse(g.launch())
    rec = _chain(tmp_path).record
    with pytest.raises(TypeError, match=r"the Record goes to the call: rev\(sim.record, seeds\)"):
        eagle.diff.Reverse(rec)


def test_seeding_an_input_names_the_output_and_where_the_gradient_is(tmp_path):
    g = _chain(tmp_path)
    with pytest.raises(ValueError) as err:
        eagle.diff.reverse(g.record, {"x": np.ones(N)})
    assert str(err.value) == (
        "eagle.diff.reverse: seeds names 'x', which is not an output of the recorded "
        "graph (outputs: 'square.y', 'sin.y'). seeds are the gradient of your objective "
        "with respect to each OUTPUT plane (∂J/∂square.y), one value per sample; the "
        "gradient with respect to 'x' is read from the result: grads['x'].")


def test_a_seed_of_the_wrong_length_says_the_shape(tmp_path):
    g = _chain(tmp_path)
    with pytest.raises(ValueError) as err:
        eagle.diff.reverse(g.record, {"sin.y": np.ones(N - 6)})
    assert str(err.value) == (
        f"eagle.diff.reverse: seeds['sin.y'] has {N - 6} entries but the record holds "
        f"{N} samples; pass one value per sample in sample order (shape ({N},), "
        "dtype float64 like plane 'sin.y').")


def test_wrt_must_name_an_input(tmp_path):
    g = _chain(tmp_path)
    with pytest.raises(ValueError, match="wrt names 'z', which is not an input of the recorded graph"):
        eagle.diff.reverse(g.record, {"sin.y": np.ones(N)}, wrt=["z"])
    grads = eagle.diff.reverse(g.record, {"sin.y": np.ones(N)}, wrt=["a"])
    assert list(grads) == ["a"]


def test_a_max_reduce_is_refused():
    x, out = np.linspace(0, 1, N), np.zeros(1)
    g = eagle.Graph().add_node(eagle.reduce(x, out, op="max"), name="peak").build(record=True)
    g.launch()
    with pytest.raises(ValueError, match="node 'peak' is a 'max' or 'all' reduce"):
        eagle.diff.reverse(g.record, {"out": np.ones(1)})
