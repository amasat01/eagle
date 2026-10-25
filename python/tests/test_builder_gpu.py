# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The graph API document's snippets on a CUDA graph: the arrays decide the backend."""

import numpy as np
import pytest

import eagle
import _builder_fixtures as bp

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")
try:
    _devices = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError:
    _devices = 0
if not _devices:                       # the kernels below build for CUDA at import
    pytest.skip("needs a GPU", allow_module_level=True)
Scalar = bp.hawk.Scalar


@pytest.fixture(scope="module")
def doc():
    errors = {}
    ns = bp.run_doc(8, gpu=True, tolerate={6}, errors=errors)
    ns["_errors"] = errors
    return ns


def ref(x):
    return x - (2 * (0.5 * x + 1)) ** 4


def test_doc_snippets_on_gpu(doc):
    x = cp.arange(8.0); y = cp.zeros(8); flag = cp.array([1], dtype=cp.uint32)
    g = doc["whole"](x, flag, y)
    assert g.backend == "gpu"
    np.testing.assert_array_equal(g().get(), ref(np.arange(8.0)))
    np.testing.assert_array_equal(x.get(), np.arange(8.0))             # the caller's x is never written
    np.testing.assert_array_equal(doc["chain"](cp.arange(8.0))().get(), (2 * np.arange(8.0) + 1) ** 2)


def test_bind_equals_fresh_build_on_gpu(doc):
    x, x2 = cp.arange(8.0), cp.arange(8.0) + 1.0
    flag = cp.array([1], dtype=cp.uint32)
    g = doc["whole"](x, flag, cp.zeros(8))
    g()
    flag[...] = 1
    s = g.bind(x=x2)().get()
    np.testing.assert_array_equal(s, ref(np.arange(8.0) + 1.0))
    fresh = doc["whole"](x2, cp.array([1], dtype=cp.uint32), cp.zeros(8))().get()
    np.testing.assert_array_equal(s, fresh)


def test_doc_reverse_on_gpu(doc):
    x = cp.arange(8.0)
    g = doc["whole"](x, cp.array([1], dtype=cp.uint32), cp.zeros(8), record=True)
    g()
    gr = eagle.diff.reverse(g.record, {"result": cp.ones(8), "y": cp.ones(8)}, wrt=("x",))
    xh = np.arange(8.0)
    np.testing.assert_allclose(gr["x"].get(), 2 * (1 - 4 * (xh + 2) ** 3) + 0.5 + 0.5 * (0.5 * xh) ** 3 * 4)


def test_wait_is_one_edge_on_the_captured_graph(doc):
    @eagle.Graph.builder
    def waiting(x):
        g = eagle.Graph()
        a = doc["left"](x)
        with g.parallel() as p:
            with p.branch() as b:
                l = doc["left"](a)
                l2 = doc["polish"](l)
            with p.branch() as w:
                w.wait(l)
                r = doc["right"](l)
        g.result = (l2, r)
        return g

    x = cp.arange(4.0)
    g = waiting(x)
    l2, r = g()
    a = np.arange(4.0) + 1; l = a + 1
    np.testing.assert_array_equal(l2.get(), 2 * l); np.testing.assert_array_equal(r.get(), l - 1)
    assert [n.kind for n in g._nodes] == ["launch", "group"] and g.backend == "gpu"


# ---- match and if/elif chains with several conditions, on a CUDA graph
Mutable, Param, UInt32 = bp.hawk.Mutable, bp.hawk.Param, bp.hawk.UInt32
from hawk import math as m
hawk = bp.hawk
GPU = ("host", "cuda")


def gk(fn): return bp.kernel(fn, GPU)


@gk
def inc(x: Scalar) -> Scalar: return x + 1.0
@gk
def dbl(x: Scalar) -> Scalar: return 2.0 * x
@gk
def half(x: Scalar) -> Scalar: return 0.5 * x
@gk
def times10(x: Scalar) -> Scalar: return 10.0 * x
@gk
def below(s: Scalar, lim: Param, flag: Mutable[UInt32]):
    flag = m.uint32(hawk.sum(s < lim))


@eagle.Graph.builder
def b_match(x, sel):
    g = eagle.Graph()
    match sel:
        case 0:
            x = times10(x)
        case 2:
            x = dbl(x)
        case _:
            x = half(x)
    g.result = x
    return g


@eagle.Graph.builder
def b_chain3(x, f1, f2, f3):
    g = eagle.Graph()
    if f1:
        x = times10(x)
        below(x, -1.0, f1)             # the arm clears its own guard
    elif f2:
        x = dbl(x)
    elif not f3:
        x = inc(x)
    else:
        x = half(x)
    g.result = x
    return g


def c(v): return cp.array([v], dtype=cp.uint32)


def test_match_on_gpu():
    x = np.arange(4.0) + 1
    for sel, want in ((0, 10 * x), (2, 2 * x), (1, 0.5 * x), (9, 0.5 * x)):
        g = b_match(cp.asarray(x), c(sel))
        assert g.backend == "gpu"
        np.testing.assert_array_equal(g().get(), want)
    g = b_match(cp.asarray(x), c(0))
    g()
    g._bld.args["sel"][...] = 2
    np.testing.assert_array_equal(g().get(), 2 * x)                      # decided per replay


def test_chain_of_three_conditions_on_gpu():
    x = np.arange(4.0) + 1
    for f, want in (((1, 1, 1), 10 * x), ((0, 1, 1), 2 * x), ((0, 0, 0), x + 1), ((0, 0, 1), 0.5 * x)):
        np.testing.assert_array_equal(b_chain3(cp.asarray(x), *map(c, f))().get(), want)
