# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The one ULP distance the tests share (float64, elementwise)."""

from __future__ import annotations

import numpy as np


def _ordered(x):
    """Bit patterns as unsigned integers that increase with the value; -0.0 and +0.0 both map to 0."""
    i = np.ascontiguousarray(x, np.float64).view(np.int64)
    return np.where(i < 0, np.int64(-(2**63)) - i, i).view(np.uint64) ^ np.uint64(1 << 63)


def ulps(got, want):
    """Units in the last place between ``got`` and ``want``, as floats; sign-correct.

    NaN against NaN and equal infinities are 0; a NaN or infinity against anything else is ``inf``.
    """
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    a, b = _ordered(got), _ordered(want)
    d = np.where(a >= b, a - b, b - a).astype(np.float64)   # uint64 wraps to the true gap
    nan, inf = np.isnan(got) | np.isnan(want), np.isinf(got) | np.isinf(want)
    d = np.where(nan, np.where(np.isnan(got) & np.isnan(want), 0.0, np.inf), d)
    return np.where(inf & ~nan, np.where(got == want, 0.0, np.inf), d)
