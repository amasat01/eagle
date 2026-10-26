# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A kernel that reads ``hawk.sum/mean(x, over=...)`` back in its body runs in
stages like one that reads ``hawk.sum(x)``, through a keyed plane eagle owns and
zeroes for every run: ``plan.run``, ``bind`` + ``launch``, and a Graph replay
give the same bits twice (the plane is zeroed again, the companion of a
compensated sum with it)."""

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")
import eagle  # noqa: E402
import eagle.plan  # noqa: E402
from eagle import _split  # noqa: E402
from hawk import Grid, Scalar  # noqa: E402

R, COL = 4, 64


@hawk.kernel
def share(lane: Grid["row", "col":COL], x: Scalar) -> Scalar:
    return x / hawk.sum(x, over="col", policy="compensated")


@hawk.kernel
def centred(lane: Grid["row", "col":COL], x: Scalar) -> Scalar:
    return x - hawk.mean(x, over="col")


def _x(mode):
    return np.random.default_rng(3).uniform(0.5, 2.0, (R, COL)).astype(mode)


def _want(x):
    x = x.astype(np.float64)
    return x / x.sum(axis=1, keepdims=True)


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _auto(kernel, tmp_path, mode, targets=("host",)):
    (p,) = eagle.plan.auto([kernel], targets=targets, cache_dir=str(tmp_path), scalar_type=mode)
    assert p.plugin._keyed.keys() == {"cut0"}
    return p


@pytest.mark.parametrize("mode", ["float64", "float32"])
def test_plan_run_matches_numpy_and_hides_the_keyed_plane(tmp_path, mode):
    p = _auto(share, tmp_path, mode)
    assert [nm for _r, nm in _split.arg_spec(p.plugin)] == ["x", "out"]
    x = _x(mode)
    got = p.run(x=x)
    np.testing.assert_allclose(got, _want(x), rtol=1e-12 if mode == "float64" else 2e-5)
    np.testing.assert_array_equal(got, p.run(x=x))      # a second run starts from zero


@pytest.mark.parametrize("mode", ["float64", "float32"])
def test_a_mean_read_back_through_bind_repeats_exactly(tmp_path, mode):
    p = _auto(centred, tmp_path, mode)
    x, z = _x(mode), np.zeros((R, COL), mode)
    bound = p.bind(x=x, out=z)
    assert bound.names == ("x", "out")
    bound.launch()
    first = z.copy()
    np.testing.assert_allclose(first, x - x.astype(np.float64).mean(axis=1, keepdims=True),
                               atol=1e-12 if mode == "float64" else 2e-5)
    bound.launch()
    np.testing.assert_array_equal(z, first)


def test_a_graph_replay_repeats_exactly(tmp_path):
    p = _auto(share, tmp_path, "float64")
    x, z = _x("float64"), np.zeros((R, COL))
    g = eagle.Graph().add(p.bind(x=x, out=z), name="share").build()
    g.launch()
    first = z.copy()
    np.testing.assert_allclose(first, _want(x), rtol=1e-12)
    z[...] = 0
    g.launch()
    np.testing.assert_array_equal(z, first)


@pytest.mark.gpu
def test_a_graph_replay_on_the_device_repeats_exactly(tmp_path):
    cp = pytest.importorskip("cupy")
    p = _auto(share, tmp_path, "float64", targets=("cuda",))
    x, z = cp.asarray(_x("float64")), cp.zeros((R, COL))
    g = eagle.Graph().add(p.bind(x=x, out=z), name="share").build()
    g.launch()
    first = _host(z).copy()
    np.testing.assert_allclose(first, _want(_host(x)), rtol=1e-12)
    for _ in range(2):
        z[...] = 0
        g.launch()
        np.testing.assert_array_equal(_host(z), first)
    got = eagle.plan.plan(p.plugin, structure=eagle.exec.DeviceKernel).run(x=x)
    np.testing.assert_allclose(_host(got), _want(_host(x)), rtol=1e-12)
