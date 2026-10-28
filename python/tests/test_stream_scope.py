# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.stream`` and the stream a launch on a user's raw handle joins.

With no scope open the current stream is the legacy default (handle 1); a scope
holds the handle it was opened with, per thread; a seam launch on a user's cupy or
torch capture stream is recorded INTO that capture (it runs on every replay, not
at launch).
"""

from __future__ import annotations

import threading

import pytest

import eagle
from eagle import _stream


class _Cuda:
    def __init__(self, ptr):
        self.ptr_ = ptr

    def __cuda_stream__(self):
        return (0, self.ptr_)


def test_no_scope_is_the_legacy_default_stream():
    assert _stream.current() == 1
    assert _stream.current(0) == 1


def test_a_scope_holds_its_handle_and_unwinds():
    with eagle.stream(1234):
        assert _stream.current() == 1234
        with eagle.stream(_Cuda(99)):
            assert _stream.current() == 99
        assert _stream.current() == 1234
    assert _stream.current() == 1


def test_a_raw_zero_maps_to_the_legacy_stream():
    with eagle.stream(77):
        with eagle.stream(0):
            assert _stream.current() == 1


def test_every_spelling_of_a_stream_is_translated_once_at_the_boundary():
    class Torch:
        cuda_stream = 41

    class Cupy:
        ptr = 42

    for spelling, handle in ((43, 43), (_Cuda(44), 44), (Torch(), 41), (Cupy(), 42)):
        with eagle.stream(spelling):
            assert _stream.current() == handle
    with pytest.raises(TypeError, match="stream= takes"):
        eagle.stream(object())


def test_the_scope_is_unwound_by_an_exception():
    with pytest.raises(RuntimeError):
        with eagle.stream(5):
            raise RuntimeError("boom")
    assert _stream.current() == 1


def test_a_device_limited_scope_answers_only_for_its_device():
    with eagle.stream(8, device=1):
        assert _stream.current(1) == 8
        assert _stream.current(0) == 1
        assert _stream.current() == 8


def test_the_scope_is_per_thread():
    seen = {}
    ready, go = threading.Event(), threading.Event()

    def other():
        seen["before"] = _stream.current()
        ready.set()
        go.wait(5)
        with eagle.stream(222):
            seen["inside"] = _stream.current()
        seen["after"] = _stream.current()

    t = threading.Thread(target=other)
    with eagle.stream(111):
        t.start()
        assert ready.wait(5)
        assert _stream.current() == 111
        go.set()
        t.join(5)
        assert _stream.current() == 111
    assert seen == {"before": 1, "inside": 222, "after": 1}


# ---- a seam launch on the user's capture stream joins that capture (probe B, as rows) ---------

cp = pytest.importorskip("cupy")


def _bump(stream_handle, cell):
    from eagle import _core

    _core.cell_op(_core.CELL_BUMP, stream_handle, 0, [cell.data.ptr])


@pytest.mark.gpu
def test_a_seam_launch_on_a_cupy_capture_stream_joins_the_capture():
    cell = cp.zeros(1, dtype=cp.uint32)
    cp.cuda.Device().synchronize()      # the zeroing rides the legacy stream, which a non-blocking capture stream does not wait for
    s = cp.cuda.Stream(non_blocking=True)
    with eagle.stream(s):
        handle = _stream.current()
        assert handle == s.ptr
    with s:
        s.begin_capture()
        _bump(handle, cell)  # recorded, not run
        graph = s.end_capture()
    s.synchronize()
    assert int(cell[0]) == 0
    for _ in range(3):
        graph.launch(s)
    s.synchronize()
    assert int(cell[0]) == 3


@pytest.mark.gpu
def test_a_seam_launch_inside_with_eagle_stream_joins_the_capture():
    from eagle import _core

    cell = cp.zeros(1, dtype=cp.uint32)
    cp.cuda.Device().synchronize()      # the zeroing rides the legacy stream, which a non-blocking capture stream does not wait for
    s = cp.cuda.Stream(non_blocking=True)
    with s:
        s.begin_capture()
        with eagle.stream(s):
            assert _core.stream_is_capturing(_stream.current()) is True
            _bump(_stream.current(), cell)
        graph = s.end_capture()
    assert _core.stream_is_capturing(s.ptr) is False
    graph.launch(s)
    graph.launch(s)
    s.synchronize()
    assert int(cell[0]) == 2


@pytest.mark.gpu
def test_a_seam_launch_on_a_torch_capture_stream_joins_the_capture():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no torch CUDA device")
    cell = cp.zeros(1, dtype=cp.uint32)
    cp.cuda.Device().synchronize()      # the zeroing rides the legacy stream, which a non-blocking capture stream does not wait for
    ts = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()
    with eagle.stream(ts):
        handle = _stream.current()
    assert handle == ts.cuda_stream
    with torch.cuda.graph(g, stream=ts):
        _bump(handle, cell)
    torch.cuda.synchronize()
    assert int(cell[0]) == 0
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert int(cell[0]) == 3


@pytest.mark.gpu
def test_a_cupy_stream_is_accepted_as_a_scope_and_as_stream_equals():
    cp = pytest.importorskip("cupy")
    s = cp.cuda.Stream(non_blocking=True)
    with eagle.stream(s):
        assert _stream.current() == int(s.ptr)
    from eagle.interop import _stream_handle

    assert _stream_handle(s) == int(s.ptr)
