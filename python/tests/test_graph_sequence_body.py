# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A ``repeat_while`` body that is a tuple of parts (with a ``skippable`` part)
runs in ``eagle.Graph``; a loop whose parts are all ``grad=False`` is skipped
by a reverse; a differentiated part inside a loop and a loop inside a loop are
refused by name."""

import numpy as np
import pytest

import eagle
import eagle.diff
import eagle.plan
import hawk
from eagle.graph import opaque
from hawk import Mutable, Param, Scalar
from hawk.diff import param_planes

N = 16


@hawk.kernel
def model(x: Scalar, a: Param[Scalar], y: Mutable[Scalar]):
    y = a * x * x


def _loop(cell, go, steps, grad):
    """Halve ``cell`` each iteration, stop after ``steps``; a counter part is
    skipped when the count was even."""
    count = np.zeros(1)
    flag = np.ones(1, dtype=np.uint32)
    seen = np.zeros(1)

    def halve():
        cell[0] *= 0.5
        count[0] += 1

    def check():
        go[0] = 1 if count[0] < steps else 0
        flag[0] = int(count[0]) % 2

    def tally():
        seen[0] += 1

    parts = (opaque(halve, reads=[], writes={"cell": cell, "count": count}, grad=grad),
             eagle.skippable(opaque(tally, reads=[], writes={"seen": seen}, grad=grad),
                             eagle.SkipGuard.intent(flag, 0)),
             opaque(check, reads=[count], writes={"go": go, "flag": flag}, grad=grad))
    return eagle.repeat_while(parts, eagle.SkipGuard.intent(go, 0), 50), count, seen


def test_if_in_while_with_a_tuple_body_runs_in_a_graph():
    cell, go = np.full(1, 1.0), np.ones(1, dtype=np.uint32)
    loop, count, seen = _loop(cell, go, 4, True)
    g = eagle.Graph().add_node(loop, name="loop").build()
    g.launch()
    assert (cell[0], count[0], seen[0]) == (1.0 / 16, 4.0, 3.0)


def test_a_loop_of_grad_false_parts_leaves_the_gradient_as_without_it(tmp_path):
    (pm,) = eagle.plan.auto([param_planes(model)], targets=("host",),
                            cache_dir=str(tmp_path), scalar_type="float64")
    x = np.linspace(-1.0, 1.0, N)
    cell, go = np.full(1, 1.0), np.ones(1, dtype=np.uint32)
    loop, _count, _seen = _loop(cell, go, 3, False)
    y = np.zeros(N)
    g = eagle.Graph().add_node(loop, name="loop").add(pm.bind(x=x, a=cell, y=y), name="model")
    g.build(record=True)
    g.launch()
    grads = eagle.diff.reverse(g.record, {"model.y": np.ones(N)})
    ref_cell = np.full(1, 0.125)
    ref = eagle.Graph().add(pm.bind(x=x, a=ref_cell, y=np.zeros(N)), name="model").build(record=True)
    ref.launch()
    want = eagle.diff.reverse(ref.record, {"model.y": np.ones(N)})
    np.testing.assert_array_equal(grads["x"], want["x"])
    np.testing.assert_array_equal(grads["x"], 2.0 * 0.125 * x)


def test_a_differentiated_part_in_a_loop_is_refused_by_name(tmp_path):
    (pm,) = eagle.plan.auto([param_planes(model)], targets=("host",),
                            cache_dir=str(tmp_path), scalar_type="float64")
    x = np.linspace(-1.0, 1.0, N)
    cell, go = np.full(1, 1.0), np.ones(1, dtype=np.uint32)
    loop, _c, _s = _loop(cell, go, 3, True)
    g = eagle.Graph().add_node(loop, name="loop").add(pm.bind(x=x, a=cell, y=np.zeros(N)),
                                                      name="model")
    g.build(record=True)
    g.launch()
    with pytest.raises(ValueError, match=r"'loop'.*differentiated part.*grad=False"):
        eagle.diff.reverse(g.record, {"model.y": np.ones(N)}, vjps={"loop": lambda p, c: {}})


def test_a_loop_in_a_loop_is_still_refused():
    go = np.ones(1, dtype=np.uint32)
    inner = eagle.repeat_while(lambda: None, eagle.SkipGuard.intent(go, 0), 2)
    outer = eagle.repeat_while((inner,), eagle.SkipGuard.intent(go, 0), 2)
    with pytest.raises(ValueError, match="repeat_while inside a repeat_while"):
        eagle.Graph().add_node(outer)
