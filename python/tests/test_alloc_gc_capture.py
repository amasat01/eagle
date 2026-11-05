# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A cyclic-GC pass inside a global-mode capture frees an ``_alloc`` array without breaking the capture."""

import gc

import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("mode", ["legacy", "stream"])
def test_gc_free_of_an_alloc_array_mid_capture(mode):
    from eagle import _alloc, _stream
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no device")
    # the garbage earlier tests in this process left (other libraries' arrays, whose frees are not capture-safe) goes
    # first: the collection inside the capture must find only this test's cycle
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
    own = cp.cuda.Stream(non_blocking=True)
    with _stream.scope(int(own.ptr) if mode == "stream" else _stream.LEGACY):
        a = _alloc.zeros("cuda:0", 1024)
    cyc = [a]
    cyc.append(cyc)
    del a, cyc
    cp.cuda.runtime.deviceSynchronize()
    was = gc.isenabled()
    gc.disable()
    try:
        cap = cp.cuda.Stream(non_blocking=True)
        x = cp.zeros(8)
        with cap:
            cap.begin_capture(mode=cp.cuda.runtime.streamCaptureModeGlobal)
            x += 1
            gc.collect()
            x += 1
            graph = cap.end_capture()
            graph.launch()
            cap.synchronize()
        assert float(x.sum()) == 16.0
    finally:
        if was:
            gc.enable()


def test_arrays_outlive_the_stream_the_caller_dropped():
    from eagle import _alloc, _core, _stream
    own = _core.Stream(non_blocking=True)
    with _stream.scope(own.ptr()):
        a = _alloc.full("cuda:0", 256, 3.0, dtype="float64")
        b = _alloc.zeros("cuda:0", 256)
    del own                              # the caller lets go of its stream
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
    assert float(a.sum()) == 768.0 and float(b.sum()) == 0.0
    del a, b
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
