# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``k(..., returns="numpy")`` on a direct kernel call: ``returns`` is the
call's own word, as in ``Plan.run``, never a plane to bind."""

import numpy as np
import pytest

import eagle

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar  # noqa: E402


@hawk.kernel
def affine(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x + 1.0


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return eagle.deploy(affine, targets=("host",), cache_dir=str(tmp_path_factory.mktemp("hawk")))


def test_a_direct_call_accepts_returns_numpy(plan):
    x = np.linspace(0.0, 1.0, 8)
    out = plan(x, 2.0, returns="numpy")
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, 2.0 * x + 1.0)
    by_name = plan(x=x, a=2.0, returns="numpy")
    np.testing.assert_allclose(by_name, out)


def test_without_returns_the_result_is_unchanged(plan):
    x = np.linspace(0.0, 1.0, 8)
    assert not isinstance(plan(x, 2.0), np.ndarray)
