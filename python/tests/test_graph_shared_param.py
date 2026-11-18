# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A shared shaped value (``Param[Matrix[r, c]]``, ``Param[Vector[n]]``: one value every row
reads) passes through a built ``eagle.Graph`` as an array of its own shape, on the CPU and on
the GPU backend. Oracle: NumPy, within the rounding bound of a k-term dot product."""

from __future__ import annotations

import numpy as np
import pytest

import eagle

hawk = pytest.importorskip("hawk")
from hawk import Matrix, Mutable, Param, Scalar, Vector  # noqa: E402

try:
    import cupy as cp
    _devices = cp.cuda.runtime.getDeviceCount()
except Exception:                      # no cupy, or no device: the GPU row skips
    cp, _devices = None, 0

ROWS = 1000
RNG = np.random.default_rng(1)
A = RNG.random((2, 2))
W = RNG.random(3)
X2 = RNG.random((2, ROWS))
X3 = RNG.random((3, ROWS))
EPS = np.finfo(np.float64).eps


@hawk.kernel
def rotate(x: Vector[2], a: Param[Matrix[2, 2]], y: Mutable[Vector[2]]):
    y = a @ x


@hawk.kernel
def dot(x: Vector[3], w: Param[Vector[3]], y: Mutable[Scalar]):
    y = x[0] * w[0] + x[1] * w[1] + x[2] * w[2]


def _bound(k, weights, x):
    """A k-term dot product in float64 is off by at most k * eps * sum |w * x| (k - 1
    additions and k products, any order, fused or not)."""
    return k * EPS * (np.abs(weights) @ np.abs(x))


def _run(xp, tmp_path):
    targets = ("host",) if xp is np else ("host", "cuda")
    rot, dt = eagle.deploy([rotate, dot], targets=targets, cache_dir=str(tmp_path))

    @eagle.Graph.builder
    def both(x2, a, y2, x3, w, y3):
        g = eagle.Graph()
        rot(x2, a, y2)
        dt(x3, w, y3)
        return g

    y2, y3 = xp.zeros((2, ROWS)), xp.zeros(ROWS)
    graph = both(xp.asarray(X2), xp.asarray(A), y2, xp.asarray(X3), xp.asarray(W), y3)
    graph.launch(1)
    to_host = cp.asnumpy if xp is not np else np.asarray
    return graph.backend, to_host(y2), to_host(y3)


def _check(got2, got3):
    assert np.all(np.abs(got2 - A @ X2) <= _bound(2, A, X2))
    assert np.all(np.abs(got3 - W @ X3) <= _bound(3, W, X3))


def test_a_shared_matrix_and_vector_pass_through_a_cpu_graph(tmp_path):
    backend, got2, got3 = _run(np, tmp_path)
    assert backend == "cpu"
    _check(got2, got3)


@pytest.mark.gpu
@pytest.mark.skipif(not _devices, reason="needs a GPU")
def test_a_shared_matrix_and_vector_pass_through_a_gpu_graph(tmp_path):
    backend, got2, got3 = _run(cp, tmp_path)
    assert backend == "gpu"
    _check(got2, got3)
