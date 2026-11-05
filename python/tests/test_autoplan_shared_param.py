# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A direct kernel call binds a ``Param[Matrix[R,C]]`` / ``Param[Vector[W]]`` plane (one value
shared by every lane) by its natural shape and checks it by name."""

import numpy as np
import pytest

import eagle
from eagle._plan_planes import flat_shared

hawk = pytest.importorskip("hawk")
from hawk import Matrix, Mutable, Param, Vector  # noqa: E402


@hawk.kernel
def apply(x: Vector[2], A: Param[Matrix[2, 2]], y: Mutable[Vector[2]]):
    y = A @ x


@pytest.fixture(scope="module")
def plan(tmp_path_factory):
    return eagle.deploy(apply, targets=("host",), cache_dir=str(tmp_path_factory.mktemp("hawk")))


def test_the_call_binds_the_natural_shape(plan):
    x = np.random.default_rng(1).random((2, 6))
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(np.asarray(plan(x, A)), A @ x, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(plan(x=x, A=A.T.copy().T)), A @ x, rtol=1e-14)  # a view


def test_a_wrong_shape_is_refused_by_name(plan):
    x = np.ones((2, 6))
    with pytest.raises(ValueError, match=r"'A' is a shared parameter of shape \(2, 2\).*\(4,\)"):
        plan(x, np.ones(4))


def test_flat_shared_leaves_everything_else_alone():
    bound = {"x": np.ones((2, 3)), "A": np.arange(4.0).reshape(2, 2)}
    out = flat_shared({"A": (2, 2)}, bound)
    assert out["x"] is bound["x"] and out["A"].shape == (4,)
    assert flat_shared({}, bound) is bound


def test_plan_run_and_bind_take_the_natural_shape(plan):
    x = np.random.default_rng(2).random((2, 6))
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    inner = plan.host
    np.testing.assert_allclose(np.asarray(inner.run(x=x, A=A)["y"] if isinstance(
        inner.run(x=x, A=A), dict) else inner.run(x=x, A=A)), A @ x, rtol=1e-14)
    y = np.zeros((2, 6))
    inner.bind(x=x, A=A, y=y).launch()
    np.testing.assert_allclose(y, A @ x, rtol=1e-14)
    with pytest.raises(ValueError, match=r"'A' is a shared parameter of shape \(2, 2\)"):
        inner.bind(x=x, A=np.ones(4), y=y)
