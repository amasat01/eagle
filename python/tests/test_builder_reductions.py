# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Reductions inside a graph builder: a kernel that stores ``hawk.sum`` has its row
planes owned by the graph (no ``_comp`` plane to pass), the cell it fills is read by
the next kernel with ``.at(0)``, and a node added with ``g.add_node`` lands in the
order of the builder's calls."""

import numpy as np
import pytest

import eagle

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Scalar, Table  # noqa: E402

try:
    import cupy as cp
    _devices = cp.cuda.runtime.getDeviceCount()
except Exception:
    cp, _devices = None, 0


@hawk.kernel
def br_mul(d: Scalar, p: Scalar, out: Mutable[Scalar]):
    out = d * p


@hawk.kernel
def br_dot(x: Scalar, y: Scalar, t: Mutable[Scalar]):
    t = hawk.sum(x * y)


@hawk.kernel
def br_step(x: Scalar, p: Scalar, r: Scalar, ap: Scalar, rr: Table[Scalar], pap: Table[Scalar],
            x_new: Mutable[Scalar], r_new: Mutable[Scalar]):
    alpha = rr.at(0) / pap.at(0)
    x_new = x + alpha * p
    r_new = r - alpha * ap


@hawk.kernel
def br_scale(x: Scalar, s: Table[Scalar], out: Mutable[Scalar]):
    out = x * s.at(0)


@hawk.kernel
def br_max(x: Scalar, hi: Mutable[Scalar]):
    hi = hawk.max(x)


@hawk.kernel
def br_min(x: Scalar, lo: Mutable[Scalar]):
    lo = hawk.min(x)


@hawk.kernel
def br_span(x: Scalar, hi: Table[Scalar], lo: Table[Scalar], out: Mutable[Scalar]):
    out = (x - lo.at(0)) / (hi.at(0) - lo.at(0))


def _plans(tmp_path, targets):
    cache = str(tmp_path / "hawk")
    return {k.name: eagle.deploy(k, targets=targets, cache_dir=cache)
            for k in (br_mul, br_dot, br_step, br_scale, br_max, br_min, br_span)}


def _cg_step(plans):
    mul, dot, step = plans["br_mul"], plans["br_dot"], plans["br_step"]

    @eagle.Graph.builder
    def cg(d, x, r, p, ap, rr, pap, x_new, r_new):
        g = eagle.Graph()
        mul(d, p, ap)
        dot(r, r, rr)
        dot(p, ap, pap)
        g.result = step(x, p, r, ap, rr, pap, x_new, r_new)
        return g

    return cg


def _check_cg(xp, plans):
    n = 1000
    rng = np.random.default_rng(3)
    d, b = rng.uniform(1.0, 4.0, n), rng.standard_normal(n)
    x, r, p = np.zeros(n), b.copy(), b.copy()
    alpha = (r @ r) / (p @ (d * p))
    want_x, want_r = x + alpha * p, r - alpha * (d * p)
    arrays = [xp.asarray(a) for a in (d, x, r, p)]
    work = [xp.zeros(n), xp.zeros(1), xp.zeros(1), xp.zeros(n), xp.zeros(n)]
    g = _cg_step(plans)(*arrays, *work)
    g.launch(1)
    got_x, got_r = work[3], work[4]
    for got, want in ((got_x, want_x), (got_r, want_r)):
        np.testing.assert_allclose(got.get() if xp is not np else got, want, rtol=1e-12)
    assert g.schedule.count("capture") == (len(g.schedule) if xp is not np else 0)
    return g


def test_a_cg_step_in_a_builder_equals_numpy_on_the_host(tmp_path):
    g = _check_cg(np, _plans(tmp_path, ("host",)))
    assert g.backend == "cpu"
    assert g.node_labels().count("reduce") == 2          # one finish per stored sum


@pytest.mark.gpu
@pytest.mark.skipif(not _devices, reason="needs a GPU")
def test_a_cg_step_in_a_builder_equals_numpy_on_the_gpu(tmp_path):
    g = _check_cg(cp, _plans(tmp_path, None))
    assert g.backend == "gpu"


def test_the_rows_are_made_once_and_replays_reuse_them(tmp_path):
    plans = _plans(tmp_path, ("host",))
    n = 64
    x, y, t = np.arange(n, dtype=float), np.ones(n), np.zeros(1)

    @eagle.Graph.builder
    def once(x, y, t):
        g = eagle.Graph()
        plans["br_dot"](x, y, t)
        return g

    g = once(x, y, t)
    g.launch(3)
    assert t[0] == x.sum()                # a replay re-stores the sum, it does not add on


def test_a_node_added_inside_a_builder_lands_in_the_order_of_the_calls(tmp_path):
    plans = _plans(tmp_path, ("host",))
    n = 64
    x, y, s, out, total = np.arange(n, dtype=float), np.zeros(n), np.zeros(1), np.zeros(n), np.zeros(1)

    @eagle.Graph.builder
    def ordered(x, y, s, out, total):
        g = eagle.Graph()
        plans["br_mul"](x, x, y)                               # y = x * x
        g.add_node(eagle.reduce(y, total))                     # total = sum(y), between the two calls
        g.result = plans["br_scale"](x, total, out)            # out = x * total
        return g

    g = ordered(x, y, s, out, total)
    labels = g.node_labels()
    assert labels.index("reduce") == labels.index("br_mul") + 1 and labels[-1] == "br_scale"
    g.launch()
    np.testing.assert_allclose(out, x * np.sum(x * x), rtol=1e-14)


def _check_range(xp, plans):
    n = 1000
    rng = np.random.default_rng(5)
    x = rng.standard_normal(n) * 10.0
    dx, hi, lo, out = xp.asarray(x), xp.zeros(1), xp.zeros(1), xp.zeros(n)

    @eagle.Graph.builder
    def normalise(x, hi, lo, out):
        g = eagle.Graph()
        plans["br_max"](x, hi)
        plans["br_min"](x, lo)
        g.result = plans["br_span"](x, hi, lo, out)
        return g

    g = normalise(dx, hi, lo, out)
    get = (lambda a: a.get()) if xp is not np else (lambda a: a)
    for x_next in (x, x * 0.5 - 3.0):       # each replay's max/min stand alone: nothing carries over
        dx[...] = xp.asarray(x_next)
        g.launch(2)
        assert get(hi)[0] == x_next.max() and get(lo)[0] == x_next.min()
        np.testing.assert_allclose(get(out), (x_next - x_next.min()) / (x_next.max() - x_next.min()),
                                   rtol=1e-14)
    return g


def test_a_max_and_a_min_cell_in_a_builder_equal_numpy_on_the_host(tmp_path):
    g = _check_range(np, _plans(tmp_path, ("host",)))
    assert g.node_labels().count("reduce") == 2


@pytest.mark.gpu
@pytest.mark.skipif(not _devices, reason="needs a GPU")
def test_a_max_and_a_min_cell_in_a_builder_equal_numpy_on_the_gpu(tmp_path):
    _check_range(cp, _plans(tmp_path, None))
