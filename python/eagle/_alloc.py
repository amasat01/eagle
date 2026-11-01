# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Where eagle's own arrays come from.

``xp`` is a device token, ``"cpu"`` or ``"cuda:N"``: the result is an :class:`eagle.Array` on that device
(device memory on the current stream, through the backend). An array a caller made is never touched here."""

from __future__ import annotations

from . import _core, arrays
from ._backend import BackendUnavailable


def place(like) -> str:
    """The device token (``"cpu"``, ``"pinned"`` or ``"cuda:N"``) where the array ``like`` lives (any array
    protocol producer, or an :class:`eagle.Array`)."""
    return (like if isinstance(like, arrays.Array) else arrays.asarray(like)).device


def zeros(xp, shape, dtype="float64"):
    return arrays.zeros(shape, dtype, xp)


def empty(xp, shape, dtype="float64"):
    return arrays.empty(shape, dtype, xp)


def full(xp, shape, value, dtype="float64"):
    return arrays.full(shape, value, dtype, xp)


def ones(xp, shape, dtype="float64"):
    return full(xp, shape, 1, dtype)


def ones_like(xp, like, dtype=None):
    return ones(xp, like.shape, like.dtype if dtype is None else dtype)


def zeros_like(xp, like, dtype=None):
    return zeros(xp, like.shape, like.dtype if dtype is None else dtype)


def empty_like(xp, like, dtype=None):
    return empty(xp, like.shape, like.dtype if dtype is None else dtype)


def asarray(xp, data, dtype=None):
    """``data`` as an array on ``xp``. Host data is uploaded; an array already on the device is cast
    (a different dtype) or passed through."""
    return arrays.asarray(data, dtype, xp)


def synchronize() -> None:
    """Block the host until the current device's work is done."""
    try:
        _core.device_synchronize(-1)
    except BackendUnavailable:      # no usable device: nothing of ours is in flight on one
        pass


def zero(xp, array, stream, nbytes=None):
    """Zero ``array`` (or its first ``nbytes``) on ``stream`` (a raw handle): a memset node under capture."""
    if array.device.startswith("cuda") and array.is_contiguous:
        _core.memset_device(array.ptr, array.nbytes if nbytes is None else nbytes, stream)
        return
    if nbytes is not None:
        array = array.ravel().view(arrays.uint8)[:nbytes]
    array.fill(0, stream)
