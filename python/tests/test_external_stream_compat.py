"""The external-stream wrapper is warning-free on CuPy 14 and works on 13.x."""
import warnings

import pytest

cp = pytest.importorskip("cupy")

from eagle.interop import _StreamProtocol, _wrap_external_stream, external_stream  # noqa: E402

pytestmark = pytest.mark.skipif(
    not cp.cuda.is_available(), reason="needs a CUDA device")


def test_protocol_tuple():
    assert _StreamProtocol(1234).__cuda_stream__() == (0, 1234)


def test_wrap_no_deprecation_and_usable():
    own = cp.cuda.Stream(non_blocking=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        s = _wrap_external_stream(own.ptr)
        with s:
            x = cp.arange(8) + 1
        s.synchronize()
    assert s.ptr == own.ptr
    assert int(x.sum()) == 36


def test_cache_identity():
    own = cp.cuda.Stream(non_blocking=True)
    assert external_stream(own.ptr) is external_stream(own.ptr)
