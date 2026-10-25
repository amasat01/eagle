# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0


"""Sample-major and strided planes at an allocating door: accepted with one
warning per call site, the fix named."""

import warnings

import numpy as np
import pytest

import eagle
from eagle import _layout

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Scalar, Vector  # noqa: E402


@hawk.kernel
def double(x: Scalar, v: Vector[3], w: Mutable[Vector[3]]):
    w = x * v


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return eagle.deploy(double, targets=("host",),
                        cache_dir=str(tmp_path_factory.mktemp("hawk")))


def test_layout_run_names_transpose(plan):
    v = np.arange(24.0).reshape(8, 3)            # (N, 3), not a zero-copy view
    x = np.arange(8.0)
    _layout._reset_seen()
    with pytest.warns(eagle.LayoutWarning, match=r"sample-major.*\(3, N\).*transposed view"):
        w = plan(x, v)
    np.testing.assert_allclose(w, x * v.T)


def test_layout_strided_warns_once(plan):
    base = np.arange(16.0)
    v = np.arange(24.0).reshape(3, 8)
    _layout._reset_seen()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            w = plan(base[::2], v)               # one call site
    msgs = [str(m.message) for m in caught if m.category is eagle.PerformanceWarning]
    assert msgs == ["'x' is not C-contiguous; copied each call \u2014 pass "
                    "np.ascontiguousarray(x)"]
    np.testing.assert_allclose(w, base[::2] * v)
