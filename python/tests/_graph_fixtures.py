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
