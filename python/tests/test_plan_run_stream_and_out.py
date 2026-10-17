# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Plan.run`` on the caller's current stream, and ``out=`` refused by name.

* a device run issues on the caller's current CuPy stream (torch's current
  stream for a torch tensor), so a producer still running on a non-blocking
  stream is waited for, not raced;
* an ``out=`` whose dtype, contiguity or device is not the plane's is refused
  before the launch (it used to be accepted and written through with the
  wrong element size);
* a Warp array is accepted as ``out=`` zero-copy.
"""

from __future__ import annotations

import numpy as np
import pytest

import eagle

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar, Vector  # noqa: E402
from hawk.math import norm, vec  # noqa: E402

from eagle.interop import BufferRefused  # noqa: E402


@hawk.kernel
def drag_step(v: Vector[3], cd: Scalar, dt: Param, g: Param, v_next: Mutable[Vector[3]]):
    a = vec(0.0, 0.0, -g) - cd * norm(v) * v
    v_next = v + dt * a


def _data(n):
    rng = np.random.default_rng(0)
    return rng.normal(size=(3, n)), rng.uniform(0, .1, n)


@pytest.fixture(scope="module")
def host_plan():
    return eagle.deploy(drag_step, targets=("host",))


@pytest.fixture(scope="module")
def plan():
    return eagle.deploy(drag_step)


# --- stream order ---------------------------------------------------------- #

@pytest.mark.gpu
def test_run_waits_for_a_producer_on_a_nonblocking_stream(plan):
    cp = pytest.importorskip("cupy")
    n = 4_000_000
    v, cd = _data(n)
    ref = plan.run(v=v, cd=cd, dt=.05, g=9.81)
    slow = cp.ElementwiseKernel(
        "float64 x", "float64 y",
        "double s=x; for(int k=0;k<3000;k++) s+=1e-300*x; y=s;", "slow_producer")
    vg, cdg = cp.asarray(v), cp.asarray(cd)
    bad = 0
    for _ in range(5):
        s = cp.cuda.Stream(non_blocking=True)
        with s:
            y = cp.zeros_like(vg)
            slow(vg, y)
            r = plan.run(v=y, cd=cdg, dt=.05, g=9.81)
            w = r * 1.0
        s.synchronize()
        bad += not np.allclose(cp.asnumpy(w), ref)
    assert bad == 0


# --- out= refusals --------------------------------------------------------- #

def test_host_out_wrong_dtype_refused_by_name(host_plan):
    v, cd = _data(64)
    o = np.zeros((3, 64), dtype=np.float32)
    with pytest.raises(BufferRefused, match=r"'v_next'.*float64.*float32"):
        host_plan.run(v=v, cd=cd, dt=.05, g=9.81, v_next=o)
    assert not o.any()


def test_host_out_non_contiguous_refused(host_plan):
    v, cd = _data(64)
    o = np.zeros((64, 3)).T  # F-ordered (3, 64) view
    with pytest.raises(BufferRefused, match="'v_next'.*contiguous"):
        host_plan.run(v=v, cd=cd, dt=.05, g=9.81, v_next=o)


@pytest.mark.gpu
def test_device_out_wrong_dtype_refused_by_name(plan):
    cp = pytest.importorskip("cupy")
    v, cd = _data(64)
    o = cp.zeros((3, 64), dtype=cp.float32)
    with pytest.raises(BufferRefused, match=r"'v_next'.*float64.*float32"):
        plan.run(v=cp.asarray(v), cd=cp.asarray(cd), dt=.05, g=9.81, v_next=o)
    o = np.zeros((3, 64), dtype=np.float32)
    with pytest.raises(BufferRefused, match=r"'v_next'.*float64.*float32"):
        plan.run(v=v, cd=cd, dt=.05, g=9.81, v_next=o)


@pytest.mark.gpu
def test_device_out_non_contiguous_refused(plan):
    cp = pytest.importorskip("cupy")
    v, cd = _data(64)
    o = cp.zeros((64, 3)).T
    with pytest.raises(BufferRefused, match="'v_next'.*contiguous"):
        plan.run(v=cp.asarray(v), cd=cp.asarray(cd), dt=.05, g=9.81, v_next=o)


# --- warp out= ------------------------------------------------------------- #

@pytest.mark.gpu
def test_warp_array_accepted_as_out(plan):
    wp = pytest.importorskip("warp")
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    n = 64
    v, cd = _data(n)
    ref = plan.run(v=v, cd=cd, dt=.05, g=9.81)
    o = wp.zeros((3, n), dtype=wp.float64, device="cuda:0")
    cp = pytest.importorskip("cupy")
    plan.run(v=cp.asarray(v), cd=cp.asarray(cd), dt=.05, g=9.81, v_next=o)
    assert np.allclose(o.numpy(), ref)


def test_warp_cpu_array_accepted_as_out(host_plan):
    wp = pytest.importorskip("warp")
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    n = 64
    v, cd = _data(n)
    ref = host_plan.run(v=v, cd=cd, dt=.05, g=9.81)
    o = wp.zeros((3, n), dtype=wp.float64, device="cpu")
    host_plan.run(v=v, cd=cd, dt=.05, g=9.81, v_next=o)
    assert np.allclose(o.numpy(), ref)


@pytest.mark.gpu
def test_warp_array_wrong_dtype_refused_by_name(plan):
    wp = pytest.importorskip("warp")
    wp.config.log_level = wp.LOG_WARNING
    wp.init()
    v, cd = _data(64)
    cp = pytest.importorskip("cupy")
    o = wp.zeros((3, 64), dtype=wp.float32, device="cuda:0")
    with pytest.raises(BufferRefused, match=r"'v_next'.*float64.*float32"):
        plan.run(v=cp.asarray(v), cd=cp.asarray(cd), dt=.05, g=9.81, v_next=o)
