# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``eagle.deploy`` takes any instance of hawk's ``Kernel``, a subclass defined outside hawk
included, and still refuses, by its type name, an object that is not a kernel."""

from __future__ import annotations

import numpy as np
import pytest

import eagle

hawk = pytest.importorskip("hawk")
from hawk.trace.kernel import Kernel  # noqa: E402


class OutsideKernel(Kernel):
    """A kernel type a package other than hawk defines."""

    __slots__ = ()


@hawk.kernel
def affine(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = 2.0 * x + 1.0


@hawk.kernel
def affine_outside(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
    y = 2.0 * x + 1.0


affine_outside.__class__ = OutsideKernel    # same slots: the traced kernel, retyped


def test_a_kernel_subclass_defined_outside_hawk_deploys_and_runs(tmp_path):
    subject = affine_outside
    assert type(subject).__module__ != "hawk.trace.kernel"
    x = np.linspace(-1.0, 1.0, 64)
    y = np.zeros_like(x)
    eagle.deploy(subject, targets=("host",), cache_dir=str(tmp_path))(x, y)
    np.testing.assert_array_equal(y, 2.0 * x + 1.0)   # 2*x is exact: one rounding, fused or not


def test_an_object_that_is_not_a_kernel_is_refused_by_its_type_name(tmp_path):
    with pytest.raises(ValueError, match=r"item 1 is a dict"):
        eagle.deploy([affine, {}], targets=("host",), cache_dir=str(tmp_path))
