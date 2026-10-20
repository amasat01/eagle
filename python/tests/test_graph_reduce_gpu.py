# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""eagle.reduce on the GPU: ordered is bitwise equal to the host canonical
tree, compensated meets the contract, inside a captured Graph. Needs a device
(GPU 1)."""

import numpy as np
import pytest

import eagle
from eagle.graph import opaque
from test_graph_reduce import bound, fixture, ulps

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")


def _hard(n, seed, dtype):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * 10.0 ** rng.integers(-8, 9, n)).astype(dtype)


def _host(x, op="sum", **kw):
    out = np.zeros(1 if x.ndim == 1 else x.shape[1], x.dtype)
    eagle.Graph().add_node(eagle.reduce(x, out, op=op, **kw)).build().launch()
    return out


def _gpu(x, op="sum", **kw):
    if "comp" in kw:
        kw["comp"] = cp.asarray(kw["comp"])
    d = cp.asarray(x)
    out = cp.zeros(1 if x.ndim == 1 else x.shape[1], x.dtype)
    g = eagle.Graph().add_node(eagle.reduce(d, out, op=op, **kw)).build()
    assert g.backend == "gpu"
    g.launch()
    return cp.asnumpy(out)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n", [1, 255, 256, 257, 4097, 65537, 300001])
def test_gpu_sum_equals_cpu_bitwise(dtype, n):
    x = _hard(n, n, dtype)
    assert _gpu(x, policy="ordered").tobytes() == _host(x).tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [257, 2048 * 3 + 1, 10 ** 6])
@pytest.mark.parametrize("kappa", [1, 2 ** 10, 2 ** 20])
def test_gpu_policies_meet_the_contract(kappa, n, dtype):
    x, exact, k = fixture(n, kappa, dtype)
    ordered = _gpu(x, policy="ordered")
    assert ordered.tobytes() == _host(x).tobytes()
    for got in (ordered, _gpu(x), _gpu(x, policy="auto")):
        err = ulps(got[0], exact, dtype)
        assert err <= bound(n, k, dtype)
        if n * k <= (2 ** 20 if dtype == np.float32 else 2 ** 48):
            assert err <= 1.0


@pytest.mark.parametrize("policy", ["ordered", "compensated"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("terms,want", [((np.inf,), np.inf), ((np.inf, -np.inf), np.nan),
                                        ((np.nan,), np.nan)])
def test_gpu_non_finite_terms_follow_ieee(policy, dtype, terms, want):
    x = _hard(70001, 2, dtype)
    for k, t in enumerate(terms):
        x[1000 + 30000 * k] = t
    got = _gpu(x, policy=policy)[0]
    assert (np.isnan(got) if np.isnan(want) else got == want)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rows,cols", [(1, 3), (257, 5), (9000, 80)])
def test_gpu_columns_equal_cpu_and_flat(dtype, rows, cols):
    x = _hard(rows * cols, rows + cols, dtype).reshape(rows, cols)
    v = (x.astype(np.float64) * (1 + 2.0 ** -30)).astype(dtype)
    comp = (x - v).astype(dtype)
    got = _gpu(x, policy="ordered")
    assert got.tobytes() == _host(x).tobytes()
    assert _gpu(v, policy="ordered", comp=comp).tobytes() == _host(v, comp=comp).tobytes()
    for j in range(cols):
        assert got[j].tobytes() == _gpu(np.ascontiguousarray(x[:, j]), policy="ordered").tobytes()
    tol = np.abs(x).sum(axis=0) * 4 * np.finfo(dtype).eps
    assert np.all(np.abs(_gpu(x).astype(np.float64) - got) <= tol)


@pytest.mark.parametrize("op,dtype", [("max", np.float64), ("max", np.int32),
                                      ("min", np.float64), ("min", np.int32),
                                      ("sum", np.int64), ("all", np.bool_)])
def test_gpu_exact_ops_equal_cpu(op, dtype):
    rng = np.random.default_rng(4)
    x = (rng.integers(-1000, 1000, 9001) if dtype is not np.bool_
         else rng.random(9001) > 0.0001).astype(dtype)
    d, out = cp.asarray(x), cp.zeros(1, dtype)
    eagle.Graph().add_node(eagle.reduce(d, out, op=op)).build().launch()
    assert cp.asnumpy(out).tobytes() == _host(x, op).tobytes()
    assert _gpu(x.reshape(-1, 1), op)[0] == _host(x, op)[0]


def test_gpu_relaunch_and_after_a_fork_join():
    a, b = cp.arange(5000, dtype=cp.float64), cp.arange(5000, dtype=cp.float64) * 3
    pa, pb, t, out = cp.zeros(5000), cp.zeros(5000), cp.zeros(5000), cp.zeros(1)
    g = (eagle.Graph()
         .add_group([[opaque(lambda: cp.add(a, 1, out=pa), reads=[a], writes=[pa])],
                     [opaque(lambda: cp.add(b, 1, out=pb), reads=[b], writes=[pb])]])
         .add_node(opaque(lambda: cp.add(pa, pb, out=t), reads=[pa, pb], writes=[t]), name="join")
         .add_node(eagle.reduce(t, out, policy="ordered"), name="total")).build()
    g.launch()
    first = cp.asnumpy(out).tobytes()
    assert first == _host(cp.asnumpy(t)).tobytes()
    a += 1.0
    g.launch()
    assert cp.asnumpy(out).tobytes() == _host(cp.asnumpy(t)).tobytes() != first
