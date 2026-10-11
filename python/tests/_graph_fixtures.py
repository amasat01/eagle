# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the eagle.Graph rows (host-only unless noted)."""

from __future__ import annotations

import numpy as np
import pytest


def host_chain(tmp_path):
    """Two bound host plans: ``y = a*x`` then ``z = y + 1``. Returns
    ``(b1, b2, x, y, z)``."""
    hawk = pytest.importorskip("hawk")
    import eagle.plan as ep

    @hawk.kernel
    def gf_scale(x: hawk.Scalar, a: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
        y = a * x

    @hawk.kernel
    def gf_addone(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = x + 1.0

    p1, p2 = ep.auto([gf_scale, gf_addone], targets=("host",),
                     cache_dir=str(tmp_path / "hawk_cache"))
    x, y, z = np.arange(8.0), np.zeros(8), np.zeros(8)
    return p1.bind(x=x, a=2.0, y=y), p2.bind(x=y, y=z), x, y, z


def host_fork(tmp_path, branches: int, n: int, depth: int = 1):
    """``branches`` independent host chains of ``2 * depth`` plans each (the
    ping-pong ``b = 2*a`` then ``a = b + 1``, ``depth`` times) over ``n``
    samples. Returns ``(chains, arrays, expected)``: per-branch lists of bound
    plans, the per-branch ``(a, b)`` planes, and the exact final ``a``."""
    hawk = pytest.importorskip("hawk")
    import eagle.plan as ep

    @hawk.kernel
    def gf_double(x: hawk.Scalar, a: hawk.Param, y: hawk.Mutable[hawk.Scalar]):
        y = a * x

    @hawk.kernel
    def gf_plus_one(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = x + 1.0

    p_double, p_plus = ep.auto([gf_double, gf_plus_one], targets=("host",),
                               cache_dir=str(tmp_path / "hawk_cache"))
    chains, arrays, expected = [], [], []
    for k in range(branches):
        a, b = np.arange(n, dtype=np.float64) + k, np.zeros(n)
        ref = a.copy()
        for _ in range(depth):
            ref = 2.0 * ref + 1.0
        chains.append([bound for _ in range(depth)
                       for bound in (p_double.bind(x=a, a=2.0, y=b), p_plus.bind(x=b, y=a))])
        arrays.append((a, b))
        expected.append(ref)
    return chains, arrays, expected
