# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""eagle.reduce on the GPU: bitwise equal to the host canonical tree, inside a
captured Graph. Needs a device (GPU 1)."""

import numpy as np
import pytest

import eagle
from eagle.graph import opaque

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")


def _hard(n, seed, dtype):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * 10.0 ** rng.integers(-8, 9, n)).astype(dtype)


def _host(x, op="sum"):
    out = np.zeros(1, x.dtype)
    eagle.Graph().add_node(eagle.reduce(x, out, op=op)).build().launch()
    return out


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
@pytest.mark.parametrize("n", [1, 255, 256, 257, 1000, 65537, 300001])
def test_gpu_sum_equals_cpu_bitwise(dtype, n):
    x = _hard(n, n, dtype)
    d, out = cp.asarray(x), cp.zeros(1, dtype)
    g = eagle.Graph().add_node(eagle.reduce(d, out)).build()
    assert g.backend == "gpu"
    g.launch()
    assert cp.asnumpy(out).tobytes() == _host(x).tobytes()


@pytest.mark.parametrize("op,dtype", [("max", np.float64), ("max", np.int32),
                                      ("sum", np.int64), ("all", np.bool_)])
def test_gpu_exact_ops_equal_cpu(op, dtype):
    rng = np.random.default_rng(4)
    x = (rng.integers(-1000, 1000, 9001) if dtype is not np.bool_
         else rng.random(9001) > 0.0001).astype(dtype)
    d, out = cp.asarray(x), cp.zeros(1, dtype)
    eagle.Graph().add_node(eagle.reduce(d, out, op=op)).build().launch()
    assert cp.asnumpy(out).tobytes() == _host(x, op).tobytes()


def test_gpu_relaunch_and_after_a_fork_join():
    a, b = cp.arange(5000, dtype=cp.float64), cp.arange(5000, dtype=cp.float64) * 3
    pa, pb, t, out = cp.zeros(5000), cp.zeros(5000), cp.zeros(5000), cp.zeros(1)
    g = (eagle.Graph()
         .add_group([[opaque(lambda: cp.add(a, 1, out=pa), reads=[a], writes=[pa])],
                     [opaque(lambda: cp.add(b, 1, out=pb), reads=[b], writes=[pb])]])
         .add_node(opaque(lambda: cp.add(pa, pb, out=t), reads=[pa, pb], writes=[t]), name="join")
         .add_node(eagle.reduce(t, out), name="total")).build()
    g.launch()
    first = cp.asnumpy(out).tobytes()
    assert first == _host(cp.asnumpy(t)).tobytes()
    a += 1.0
    g.launch()
    assert cp.asnumpy(out).tobytes() == _host(cp.asnumpy(t)).tobytes() != first
