# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A cyclic-GC pass inside a global-mode capture frees an ``_alloc`` array without breaking the capture."""

import gc

import pytest

cp = pytest.importorskip("cupy")
pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("mode", ["legacy", "stream"])
def test_gc_free_of_an_alloc_array_mid_capture(mode):
    from eagle import _alloc
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("no device")
    own = cp.cuda.Stream(non_blocking=True)
    with (own if mode == "stream" else cp.cuda.Stream.null):
        a = _alloc.zeros(cp, 1024)
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
    from eagle import _alloc
    own = cp.cuda.Stream(non_blocking=True)
    with own:
        a = _alloc.full(cp, 256, 3.0, dtype="float64")
        b = _alloc.zeros(cp, 256)
    del own                              # the caller lets go of its stream
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
    assert float(a.sum()) == 768.0 and float(b.sum()) == 0.0
    del a, b
    gc.collect()
    cp.cuda.runtime.deviceSynchronize()
