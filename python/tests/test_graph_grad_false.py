# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``Graph.add(..., grad=False)``: a node that is not differentiated (a controller
writing a parameter cell) needs no vjp, no tape, and leaves the gradient as if
it were not there; a differentiated node reading its output outside a
parameter cell is refused by name."""

import numpy as np
import pytest

import eagle
import eagle.diff
import eagle.plan
hawk = pytest.importorskip("hawk")  # the kernels are written in hawk
from hawk import Mutable, Param, Scalar, UInt32
from hawk.diff import param_planes

N = 16


@hawk.kernel
def model(x: Scalar, a: Param[Scalar], y: Mutable[Scalar]):
    y = a * x * x


@hawk.kernel
def ctrl(a: Mutable[Scalar], count: Mutable[UInt32]):
    a = a * 0.5
    count = count + 1


@hawk.kernel
def feed(x: Scalar, z: Mutable[Scalar]):
    z = x * 2.0


@hawk.kernel
def use(z: Scalar, y: Mutable[Scalar]):
    y = z * z


def _graph(tmp_path, order):
    pm, pc = eagle.plan.auto([param_planes(model), ctrl], targets=("host",),
                             cache_dir=str(tmp_path), scalar_type="float64")
    x = np.linspace(-1.0, 1.0, N)
    y = np.zeros(N)
    cell = np.full(1, 0.8)
    count = np.zeros(1, dtype=np.uint32)
    parts = [("model", pm.bind(x=x, a=cell, y=y), True),
             ("ctrl", pc.bind(a=cell, count=count), False)]
    if order == "ctrl_first":
        parts.reverse()
    g = eagle.Graph()
    for name, b, grad in parts:
        g.add(b, name=name, grad=grad)
    return g.build(record=True), x, cell, count


@pytest.mark.parametrize("order", ["model_first", "ctrl_first"])
def test_controller_node_leaves_the_gradient_untouched(tmp_path, order):
    g, x, cell, count = _graph(tmp_path, order)
    for launch in range(3):
        before = float(cell[0])
        g.launch()
        read = before if order == "model_first" else before * 0.5
        grads = eagle.diff.reverse(g.record, {"model.y": np.ones(N)})    # no vjp for ctrl
        np.testing.assert_array_equal(grads["x"], 2.0 * read * x)
        assert int(count[0]) == launch + 1


def test_default_still_derives_every_node(tmp_path):
    g, *_ = _graph(tmp_path, "model_first")
    g2 = eagle.Graph()
    pm, pc = eagle.plan.auto([param_planes(model), ctrl], targets=("host",),
                             cache_dir=str(tmp_path), scalar_type="float64")
    cell = np.full(1, 0.8)
    g2.add(pm.bind(x=np.ones(N), a=cell, y=np.zeros(N)), name="model")
    g2.add(pc.bind(a=cell, count=np.zeros(1, dtype=np.uint32)), name="ctrl")
    g2.build(record=True)
    g2.launch()
    with pytest.raises(ValueError, match="'a'"):
        eagle.diff.reverse(g2.record, {"model.y": np.ones(N)})


def test_a_plane_feeding_a_differentiated_node_is_refused(tmp_path):
    pf, pu = eagle.plan.auto([feed, use], targets=("host",), cache_dir=str(tmp_path),
                             scalar_type="float64")
    x, z, y = np.linspace(0, 1, N), np.zeros(N), np.zeros(N)
    g = eagle.Graph()
    g.add(pf.bind(x=x, z=z), name="feeder", grad=False)
    g.add(pu.bind(z=z, y=y), name="user")
    with pytest.raises(ValueError, match=r"'user'.*'feeder'.*grad=False"):
        g.build(record=True)


def test_a_one_sample_plane_is_not_a_parameter_plane(tmp_path):
    pf, pu = eagle.plan.auto([feed, use], targets=("host",), cache_dir=str(tmp_path),
                             scalar_type="float64")
    x, z, y = np.ones(1), np.zeros(1), np.zeros(1)
    g = eagle.Graph()
    g.add(pf.bind(x=x, z=z), name="feeder", grad=False)
    g.add(pu.bind(z=z, y=y), name="user")
    with pytest.raises(ValueError, match=r"'user'.*'feeder'.*grad=False"):
        g.build(record=True)
