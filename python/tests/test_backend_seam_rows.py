# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Behaviour that crosses the eagle._core / CUDA-backend seam, on a real device.

* A Python callback the backend calls back into (a composer step) that raises
  surfaces as ITS OWN exception, not a generic error.
* A handle consumed by another call (a Launcher registered into a composer) refuses
  later use with a RuntimeError instead of touching a moved-from object.
* Device agreement (two GPUs): with the launch framework's device set to 1, the
  stream, the replayed graph and the fence eagle creates all land on device 1, and
  an explicit ``device=1`` does the same while device 0 is current. eagle's backend
  runs on the driver API alone and follows the CUDA runtime's context rules (the
  driver's per-thread current context, the device's primary context), which is
  what this row pins.
"""

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")


def _device_count() -> int:
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


@pytest.mark.gpu
def test_a_raising_composer_step_surfaces_its_own_exception():
    from eagle import _core

    def step(stream):
        raise KeyError("raised inside a composer step")

    c = _core.GraphComposer("sequenced")
    c.register_callable(step, name="boom")
    c.build()
    with pytest.raises(KeyError, match="raised inside a composer step"):
        c.launch()


@pytest.mark.gpu
def test_a_consumed_launcher_refuses_reuse():
    from eagle import _core

    s = _core.Stream(non_blocking=True)
    a = cp.arange(8, dtype=cp.float64)
    cap = _core.StreamCapturer(s.ptr())
    with cp.cuda.ExternalStream(s.ptr()):
        cap.begin()
        a *= 2.0
        captured = cap.end()
    launcher = _core.Graph.from_captured(captured).launcher()
    assert not captured  # consumed by from_captured
    c = _core.GraphComposer("sequenced")
    c.register_launcher(launcher, name="m")
    with pytest.raises(RuntimeError, match="consumed"):
        launcher.launch()


@pytest.mark.gpu2
@pytest.mark.skipif(
    _device_count() < 2, reason="needs two CUDA devices (skipped, never passed, on one)"
)
def test_eagle_objects_follow_the_frameworks_device():
    from eagle import _core

    def roundtrip(stream_ptr):
        x = cp.arange(16, dtype=cp.float64)
        cap = _core.StreamCapturer(stream_ptr)
        with cp.cuda.ExternalStream(stream_ptr):
            cap.begin()
            x *= 3.0
            graph = cap.end()
        launcher = _core.Graph.from_captured(graph).launcher()
        launcher.stream(stream_ptr)
        launcher.launch()
        launcher.synchronize()
        return float(x[1])

    with cp.cuda.Device(1):
        s = _core.Stream(non_blocking=True)  # device -1: the framework's (1)
        assert (
            roundtrip(s.ptr()) == 3.0
        )  # a device-0 stream could not run device-1 work
        other = cp.cuda.Stream(non_blocking=True)
        _core.fence(s.ptr(), other.ptr, 1)
        other.synchronize()

    with cp.cuda.Device(0):
        explicit = _core.Stream(non_blocking=True, device=1)
    with cp.cuda.Device(1):
        assert roundtrip(explicit.ptr()) == 3.0
