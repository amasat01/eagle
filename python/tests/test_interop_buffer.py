# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The generic buffer layer: :func:`eagle.interop.import_buffer` / ``BufferView``.

Host rows (numpy, torch CPU, a legacy-only producer, the array interface) run
anywhere ``eagle._core`` is built; device rows (cupy, NVIDIA Warp, the stream
contract) are ``gpu``-marked. Every producer is checked for pointer identity
(zero-copy), the reported access / owner / producer, survival of ``del
producer`` and re-export into another library. The stream rows use an
adversarial schedule: a slow kernel on one non-blocking stream writes a
sentinel that a copy on another non-blocking stream reads; only the layer's
fence orders the two, and the RED twin (no fence) must read the stale value.
"""

from __future__ import annotations

import gc

import numpy as np
import pytest

pytest.importorskip("eagle._core")

from eagle import _core  # noqa: E402
from eagle.interop import (  # noqa: E402
    BufferRefused,
    BufferView,
    Requirements,
    import_buffer,
    refence,
)


class LegacyOnly:
    """A producer that speaks only pre-1.0 DLPack (no ``max_version``)."""

    def __init__(self, arr):
        self._arr = arr

    def __dlpack__(self, stream=None):
        return self._arr.__dlpack__()

    def __dlpack_device__(self):
        return self._arr.__dlpack_device__()


class InterfaceOnly:
    """A producer that speaks only ``__array_interface__``."""

    def __init__(self, arr):
        self._arr = arr
        self.__array_interface__ = arr.__array_interface__


# --------------------------------------------------------------------------- #
# Host producers
# --------------------------------------------------------------------------- #
def test_numpy_writable_is_zero_copy_and_reports_external_read_write():
    a = np.arange(12.0)
    v = import_buffer(a)
    assert isinstance(v, BufferView)
    assert v.ptr == a.ctypes.data
    assert (v.access, v.owner) == ("read-write", "external")
    assert v.producer == "numpy.ndarray"
    assert v.stream is None
    assert v.device == (1, 0)
    assert (v.shape, v.strides, v.dtype) == ((12,), (1,), "float64")


def test_numpy_read_only_reports_read_only_and_exports_read_only():
    a = np.arange(6.0)
    a.flags.writeable = False
    v = import_buffer(a)
    assert v.access == "read-only"
    assert not v.writable
    back = np.from_dlpack(v)
    assert back.ctypes.data == a.ctypes.data
    assert not back.flags.writeable


def test_numpy_view_survives_del_producer_and_reexports():
    a = np.arange(8.0)
    ptr = a.ctypes.data
    v = import_buffer(a)
    del a
    gc.collect()
    out = np.from_dlpack(v)
    assert out.ctypes.data == ptr
    np.testing.assert_array_equal(out, np.arange(8.0))


def test_reexport_chain_keeps_the_first_producer_alive():
    torch = pytest.importorskip("torch")
    a = np.arange(5.0)
    v = import_buffer(a)
    t = torch.from_dlpack(v)
    del a, v
    gc.collect()
    w = import_buffer(t)
    del t
    gc.collect()
    assert np.from_dlpack(w).tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_torch_cpu_is_zero_copy_and_reexports_to_numpy():
    torch = pytest.importorskip("torch")
    t = torch.arange(4, dtype=torch.float64)
    v = import_buffer(t)
    assert v.ptr == t.data_ptr()
    assert (v.access, v.owner, v.producer) == ("read-write", "external", "torch.Tensor")
    del t
    gc.collect()
    assert np.from_dlpack(v).ctypes.data == v.ptr
    assert torch.from_dlpack(v).data_ptr() == v.ptr


def test_legacy_only_producer_reports_unknown_and_records_the_assertion():
    a = np.arange(3.0)
    v = import_buffer(LegacyOnly(a))
    assert v.ptr == a.ctypes.data
    assert v.access == "unknown"
    assert not v.writable and not v.assumed_writable
    w = import_buffer(LegacyOnly(a), assume_writable=True)
    assert w.access == "unknown"
    assert w.assumed_writable and w.writable
    import_buffer(LegacyOnly(a), assume_writable=True,
                  require=Requirements(writable=True))
    with pytest.raises(BufferRefused, match="access: required read-write, got unknown"):
        import_buffer(LegacyOnly(a), require=Requirements(writable=True))


def test_array_interface_producer_through_the_raw_factory():
    a = np.arange(6.0).reshape(2, 3)
    a.flags.writeable = False
    v = import_buffer(InterfaceOnly(a))
    assert v.ptr == a.ctypes.data
    assert v.access == "read-only"
    assert v.producer.endswith("InterfaceOnly")
    assert v.shape == (2, 3)


# --------------------------------------------------------------------------- #
# Refusals
# --------------------------------------------------------------------------- #
def test_refusal_float32_where_float64_required():
    with pytest.raises(BufferRefused, match="dtype: required float64, got float32"):
        import_buffer(np.zeros(4, np.float32), require=Requirements(dtype="float64"))


def test_refusal_wrong_size():
    with pytest.raises(BufferRefused, match="count: required 10 elements, got 12"):
        import_buffer(np.zeros(12), require=Requirements(count=10))


def test_refusal_strided_view():
    x = np.zeros(10)
    msg = "stride: required C-contiguous with unit stride"
    with pytest.raises(BufferRefused, match=msg):
        import_buffer(x[::2], require=Requirements(contiguous=True))


def test_refusal_cuda_required_for_a_cpu_buffer():
    with pytest.raises(BufferRefused, match="device: required cuda, got cpu"):
        import_buffer(np.zeros(3), require=Requirements(device="cuda"))


def test_refusal_read_only_for_a_writable_requirement():
    a = np.zeros(3)
    a.flags.writeable = False
    msg = "access: required read-write, got read-only"
    with pytest.raises(BufferRefused, match=msg):
        import_buffer(a, require=Requirements(writable=True))


def test_refusals_are_all_named_at_once():
    a = np.zeros(4, np.float32)
    a.flags.writeable = False
    with pytest.raises(BufferRefused) as e:
        import_buffer(a, require=Requirements(dtype="float64", count=3, writable=True))
    msg = str(e.value)
    assert "dtype:" in msg and "; count:" in msg and "; access:" in msg


def test_host_buffer_refuses_a_stream():
    with pytest.raises(ValueError, match="stream: a CPU buffer takes no stream, got 1"):
        import_buffer(np.zeros(2), stream=1)
    v = import_buffer(np.zeros(2))
    with pytest.raises(ValueError, match="stream: a CPU buffer takes no stream"):
        v.__dlpack__(stream=1)


def _cuda_backend_usable() -> bool:
    """The plugin loads AND a driver with a device answers (a runner with no
    driver loads the plugin but cannot create a stream)."""
    import eagle
    from eagle import _core

    if not _core.cuda_backend()["loaded"]:
        return False
    try:
        _core.Stream()
    except eagle.BackendUnavailable:
        return False
    return True


# The CUDA stream codes (0 refused, 1/2 the default streams) are the CUDA
# backend's to resolve: the core passes them through untouched, so without a
# backend this is a typed BackendUnavailable, not the code refusal.
@pytest.mark.skipif(
    not _cuda_backend_usable(),
    reason="no usable CUDA backend here (plugin or driver absent)",
)
def test_stream_zero_is_refused():
    with pytest.raises(ValueError, match="stream: 0 is ambiguous"):
        refence(0, 1)


# --------------------------------------------------------------------------- #
# Device producers and the stream contract
# --------------------------------------------------------------------------- #
_SLOW_SRC = r"""
extern "C" __global__ void slow_write(double* out, long long cycles, double value) {
    long long start = clock64();
    while (clock64() - start < cycles) { }
    *out = value;
}
"""
_SPIN = 300_000_000  # ~0.2 s: far longer than a launch plus an 8-byte copy


def _slow_write(cp, arr, stream):
    k = cp.RawKernel(_SLOW_SRC, "slow_write")
    with stream:
        k((1,), (1,), (arr, np.int64(_SPIN), np.float64(42.0)))


def _read_on(cp, src, stream) -> float:
    """Copy ``src`` on ``stream`` and wait for that stream only."""
    host = cp.cuda.alloc_pinned_memory(8)
    out = np.frombuffer(host, np.float64, 1)
    out[0] = -1.0
    cp.cuda.runtime.memcpyAsync(out.ctypes.data, src.data.ptr, 8,
                                cp.cuda.runtime.memcpyDeviceToHost, stream.ptr)
    stream.synchronize()
    return float(out[0])


@pytest.mark.gpu
def test_cupy_is_zero_copy_and_survives_del_producer():
    import cupy as cp

    a = cp.arange(6, dtype=cp.float64)
    ptr = a.data.ptr
    v = import_buffer(a)
    assert v.ptr == ptr
    assert (v.access, v.owner, v.producer) == ("read-write", "external", "cupy.ndarray")
    assert v.device == (2, 0)
    assert v.stream == 1
    del a
    gc.collect()
    b = cp.from_dlpack(v)
    assert b.data.ptr == ptr
    assert b.get().tolist() == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


@pytest.mark.gpu
def test_cupy_refused_for_a_cpu_requirement():
    import cupy as cp

    with pytest.raises(BufferRefused, match="device: required cpu, got cuda"):
        import_buffer(cp.zeros(3), require=Requirements(device="cpu"))


@pytest.mark.gpu
@pytest.mark.parametrize("round_", range(3))
def test_export_fence_orders_the_consumer_after_the_producer(round_):
    import cupy as cp

    a_stream = cp.cuda.Stream(non_blocking=True)
    b_stream = cp.cuda.Stream(non_blocking=True)
    arr = cp.zeros(1, cp.float64)
    cp.cuda.runtime.deviceSynchronize()
    _slow_write(cp, arr, a_stream)
    v = import_buffer(arr, stream=a_stream.ptr)
    with b_stream:
        seen = cp.from_dlpack(v)  # __dlpack__(stream=B): fence A -> B
    assert _read_on(cp, seen, b_stream) == 42.0
    a_stream.synchronize()


@pytest.mark.gpu
def test_red_twin_without_the_fence_reads_the_stale_value():
    import cupy as cp

    a_stream = cp.cuda.Stream(non_blocking=True)
    b_stream = cp.cuda.Stream(non_blocking=True)
    arr = cp.zeros(1, cp.float64)
    cp.cuda.runtime.deviceSynchronize()
    _slow_write(cp, arr, a_stream)
    v = import_buffer(arr, stream=-1)  # -1: no ordering on either side
    with b_stream:
        seen = cp.from_dlpack(v)
    assert _read_on(cp, seen, b_stream) == 0.0, "the schedule must bite without a fence"
    a_stream.synchronize()


@pytest.mark.gpu
def test_refence_orders_a_later_read():
    import cupy as cp

    a_stream = cp.cuda.Stream(non_blocking=True)
    b_stream = cp.cuda.Stream(non_blocking=True)
    arr = cp.zeros(1, cp.float64)
    v = import_buffer(arr, stream=a_stream.ptr)
    cp.cuda.runtime.deviceSynchronize()
    _slow_write(cp, arr, a_stream)  # the producer keeps writing after import
    v.refence(b_stream.ptr)
    assert _read_on(cp, arr, b_stream) == 42.0
    a_stream.synchronize()
    created = _core.event_pool_created()
    for _ in range(50):
        v.refence(b_stream.ptr)
    assert _core.event_pool_created() == created, "events come from the pool"


@pytest.mark.gpu
@pytest.mark.torch_gpu
def test_torch_cuda_is_zero_copy_and_reexports_to_cupy():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch is a CPU-only build here")
    import cupy as cp

    t = torch.arange(4, dtype=torch.float64, device="cuda")
    v = import_buffer(t, stream=torch.cuda.current_stream().cuda_stream or 1)
    assert v.ptr == t.data_ptr()
    assert (v.access, v.owner, v.producer) == ("read-write", "external", "torch.Tensor")
    del t
    gc.collect()
    assert cp.from_dlpack(v).data.ptr == v.ptr


# --------------------------------------------------------------------------- #
# NVIDIA Warp
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def wp():
    warp = pytest.importorskip("warp")
    warp.config.quiet = True
    warp.init()
    if not warp.is_cuda_available():
        pytest.skip("warp sees no CUDA device")
    return warp


@pytest.mark.gpu
def test_warp_array_imports_zero_copy_and_exports_back(wp):
    w = wp.array(np.arange(4.0), dtype=wp.float64, device="cuda:0")
    v = import_buffer(w)
    assert v.ptr == w.ptr
    assert v.producer == "warp.array"
    assert v.access == "unknown"  # warp speaks pre-1.0 DLPack: no access flag
    back = wp.from_dlpack(import_buffer(w, assume_writable=True))
    assert back.ptr == w.ptr
    assert back.numpy().tolist() == [0.0, 1.0, 2.0, 3.0]


@pytest.mark.gpu
def test_warp_stream_as_producer_orders_the_import(wp):
    import cupy as cp

    s = wp.Stream("cuda:0")
    b_stream = cp.cuda.Stream(non_blocking=True)
    w = wp.zeros(1, dtype=wp.float64, device="cuda:0")
    wp.synchronize()
    arr = cp.from_dlpack(w)
    _slow_write(cp, arr, cp.cuda.ExternalStream(s.cuda_stream))
    with wp.ScopedStream(s):
        v = import_buffer(w, stream=b_stream.ptr)  # warp orders its stream before B
    assert v.stream == b_stream.ptr
    assert _read_on(cp, arr, b_stream) == 42.0
    wp.synchronize_stream(s)


@pytest.mark.gpu
def test_warp_stream_as_consumer_is_fenced_by_the_export(wp):
    import cupy as cp

    s = wp.Stream("cuda:0")
    a_stream = cp.cuda.Stream(non_blocking=True)
    arr = cp.zeros(1, cp.float64)
    cp.cuda.runtime.deviceSynchronize()
    _slow_write(cp, arr, a_stream)
    v = import_buffer(arr, stream=a_stream.ptr, assume_writable=True)
    with wp.ScopedStream(s):
        w = wp.from_dlpack(v)  # __dlpack__(stream=s.cuda_stream): fence A -> s
    assert w.ptr == arr.data.ptr
    assert _read_on(cp, arr, cp.cuda.ExternalStream(s.cuda_stream)) == 42.0
    a_stream.synchronize()
