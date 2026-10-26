# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Where eagle's own arrays come from: ``aether`` for the device, the array module itself for the host.

``xp`` is the array module a run was bound to. On the host it is numpy and these are numpy's own
calls. On the device the memory is allocated by ``aether`` on the current stream and handed to the
kernels as a zero-copy view in the ``xp`` array type (cupy until the launch path moves, 0.7.2), so
every plane keeps the methods the kernels use. An array a caller made is never touched here."""

from __future__ import annotations

import numpy as np


def _device(xp) -> bool:
    return xp.__name__ == "cupy"


def _stream(xp):
    """The current stream as ``aether`` wants it: None for the legacy stream, else the stream OBJECT,
    so the array holds it and the stream outlives the array (a caller may drop its own reference
    first; the free is then still queued on a live stream)."""
    stream = xp.cuda.get_current_stream()
    return stream if int(stream.ptr) not in (0, 1, 2) else None


def _view(xp, array):
    return xp.asarray(array)          # zero-copy over __cuda_array_interface__


def _make(xp, make, shape, dtype, *extra):
    return _view(xp, make(shape, *extra, dtype=np.dtype(dtype), device=f"cuda:{xp.cuda.Device().id}",
                          stream=_stream(xp)))


def zeros(xp, shape, dtype=np.float64):
    if not _device(xp):
        return xp.zeros(shape, dtype)
    import aether
    return _make(xp, aether.zeros, shape, dtype)


def empty(xp, shape, dtype=np.float64):
    if not _device(xp):
        return xp.empty(shape, dtype)
    import aether
    return _make(xp, aether.empty, shape, dtype)


def full(xp, shape, value, dtype=np.float64):
    if not _device(xp):
        return xp.full(shape, value, dtype)
    import aether
    return _view(xp, aether.full(shape, value, dtype=np.dtype(dtype), device=f"cuda:{xp.cuda.Device().id}",
                                 stream=_stream(xp)))


def ones(xp, shape, dtype=np.float64):
    return full(xp, shape, 1, dtype)


def ones_like(xp, like, dtype=None):
    return ones(xp, like.shape, like.dtype if dtype is None else dtype)


def zeros_like(xp, like, dtype=None):
    return zeros(xp, like.shape, like.dtype if dtype is None else dtype)


def empty_like(xp, like, dtype=None):
    return empty(xp, like.shape, like.dtype if dtype is None else dtype)


def asarray(xp, data, dtype=None):
    """``data`` as an ``xp`` array. Host data is uploaded through aether; an array already on the
    device is cast (a different dtype) or passed through, which the device type does for itself."""
    if not _device(xp):
        return xp.asarray(data, dtype)
    import aether
    if hasattr(data, "__cuda_array_interface__"):
        return xp.asarray(data, dtype)       # device-resident input: its own library casts it
    host = np.asarray(data, dtype)
    return _view(xp, aether.asarray(host, device=f"cuda:{xp.cuda.Device().id}", stream=_stream(xp)))


def zero(xp, array, stream, nbytes=None):
    """Zero ``array`` (or its first ``nbytes``) on ``stream`` (a raw handle): a memset node under capture."""
    if not _device(xp):
        raise TypeError("zero() is for device arrays")
    import aether
    if nbytes is not None:
        array = array.reshape(-1).view(np.uint8)[:nbytes]
    aether.asarray(array).zero(stream)
