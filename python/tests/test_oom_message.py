# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A launch that runs the GPU out of memory says so with the numbers that explain it:
the kernel's name, its local bytes per thread, and the GPU's free/total memory read at
the moment of failure."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest

from eagle import _core
from eagle._plan_binding import BoundPlan

_HOG = r'''
extern "C" __global__ void oom_hog(double* out, int k) {
    double a[40000];                          // 320000 bytes of local memory per thread
    for (int i = 0; i < 40000; ++i) a[i] = i * 0.5;
    out[blockIdx.x * blockDim.x + threadIdx.x] = a[(k + threadIdx.x) % 40000];
}
'''


@pytest.mark.gpu
def test_an_out_of_memory_launch_names_the_kernel_the_local_bytes_and_free_total():
    cp = pytest.importorskip("cupy")
    kernel = cp.RawKernel(_HOG, "oom_hog")
    kernel.compile()
    local = _core.function_attribute(kernel.kernel.ptr, 3)
    assert local >= 320000
    out = cp.zeros(16, "float64")
    boxes = [ctypes.c_void_p(out.data.ptr), ctypes.c_int(3)]
    plan = SimpleNamespace(_kernel_name=lambda: "oom_hog")
    # 65536 blocks x 1024 threads x 320 kB of local memory is far beyond any GPU
    with pytest.raises(RuntimeError) as err:
        BoundPlan._named(plan, _core.launch_raw, kernel.kernel.ptr, (65536, 1, 1), (1024, 1, 1),
                         [ctypes.addressof(b) for b in boxes], 0, 0, -1)
    text = str(err.value)
    assert "'oom_hog'" in text
    assert "out of memory" in text and "CUresult 2" in text
    assert f"kernel local memory {local} bytes per thread" in text
    free_total = text.split("GPU memory ")[1]
    free, total = free_total.split(" MiB free of ")[0], free_total.split(" MiB free of ")[1].split(" MiB total")[0]
    assert 0 < int(free) <= int(total)
    cp.cuda.Device().synchronize()                # the failed launch left nothing pending


_HOG_V2 = r'''
extern "C" __global__ void oom_hog_v2(double* out, long long base, long long count, long long n) {
    double a[40000];                          // 320000 bytes of local memory per thread
    for (int i = 0; i < 40000; ++i) a[i] = i * 0.5;
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < count) out[0] = a[(base + t) % 40000];
}
'''


@pytest.mark.gpu
def test_device_kernel_run_out_of_memory_carries_the_same_context():
    cp = pytest.importorskip("cupy")
    import eagle.exec as eexec

    kernel = cp.RawKernel(_HOG_V2, "oom_hog_v2")
    kernel.compile()
    local = _core.function_attribute(kernel.kernel.ptr, 3)
    out = cp.zeros(16, "float64")
    box = ctypes.c_void_p(out.data.ptr)
    count = 65536 * 1024                           # 64M threads x 320 kB of local memory
    with pytest.raises(ValueError, match="out of memory") as err:
        eexec.DeviceKernel.run(kernel.kernel.ptr, [ctypes.addressof(box)],
                               eexec.Partition(0, count, count), block=1024)
    text = str(err.value)
    assert f"kernel local memory {local} bytes per thread" in text
    assert "MiB free of" in text and "MiB total" in text
    cp.cuda.Device().synchronize()
