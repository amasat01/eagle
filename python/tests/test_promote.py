# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0


"""Allocating doors promote a narrower float plane once-warned; no-copy doors
and a float32 plan refuse the wider one."""

import warnings

import numpy as np
import pytest

import eagle
from eagle import _layout
from eagle.frameworks._kernel_planes import ModeError, call_mode

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar  # noqa: E402


@hawk.kernel
def scale(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    cache = str(tmp_path_factory.mktemp("hawk"))
    return (eagle.deploy(scale, targets=("host",), cache_dir=cache),
            eagle.deploy(scale, targets=("host",), cache_dir=cache, scalar_type="float32"))


X32 = np.linspace(0.0, 1.0, 8).astype(np.float32)


def test_promote_widest_warns_once(plans):
    f64, _ = plans
    _layout._reset_seen()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(3):
            got = f64(X32, 2.0)             # one call site
        f64(X32, 3.0)                       # another
    msgs = [str(w.message) for w in caught if w.category is eagle.PerformanceWarning]
    assert len(msgs) == 2
    assert msgs[0] == ("plane 'x' is float32; promoted to float64 (a copy each call) "
                       "\u2014 pass float64, or build in float32: "
                       "eagle.deploy(k, scalar_type=\"float32\")")
    assert got.dtype == np.float64
    np.testing.assert_allclose(got, 2.0 * X32.astype(np.float64))
    with pytest.warns(eagle.PerformanceWarning):                   # plan.run promotes too
        np.testing.assert_allclose(f64.host.run(x=X32, a=2.0), got)


def test_promote_float32_plan_keeps_float32_and_refuses_float64(plans):
    _, f32 = plans
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert f32(X32, 2.0).dtype == np.float32
    with pytest.raises(ValueError, match=r"np.asarray\(x, np.float32\), or deploy in float64"):
        f32(X32.astype(np.float64), 2.0)


def test_promote_bind_refuses(plans):
    f64, _ = plans
    with pytest.raises(ValueError, match="float32"):
        f64.host.bind(x=X32, a=2.0, y=np.zeros(8))


def test_promote_bridges_mixed_promote():
    mixed = [("a", "torch.float32"), ("b", "torch.float64")]
    assert call_mode("who", mixed, promote=True) == "float64"
    assert call_mode("who", [("a", np.dtype("float32"))], promote=True) == "float32"
    with pytest.raises(ModeError, match="never a mix"):
        call_mode("who", mixed)
    with pytest.raises(ModeError, match="not supported"):
        call_mode("who", [("a", "int32")], promote=True)
