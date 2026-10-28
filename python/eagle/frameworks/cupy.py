# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The cupy bridge: the only place eagle's interop touches cupy.

cupy is imported inside the functions below, so ``import eagle`` stays cupy-free. Arrays enter and leave
through DLPack (:func:`eagle.interop.to_array`); what lives here is cupy's own side of that exchange: a cupy
array from an :class:`eagle.Array`, cupy streams around a raw ``cudaStream_t`` (cached, one wrapper per
handle), and the numpy-or-cupy module an array belongs to, for the code that still spells ``xp.<op>``.
"""

from __future__ import annotations

import threading

# One wrapper per raw stream handle: it owns and destroys nothing, so one per handle suffices for the
# process and a cupy DeprecationWarning fires once, not per construction. The key is the raw handle, which
# stays correct even if torch reallocates a stream at the same address.
_external_stream_cache: dict = {}
_EXTERNAL_STREAM_COUNTS = {"hits": 0, "misses": 0}
#: Serialises the cache and its tallies: one wrapper per handle, exact counts.
_EXTERNAL_STREAM_LOCK = threading.Lock()


class _StreamProtocol:
    """Minimal CUDA stream protocol object around a raw ``cudaStream_t``."""

    __slots__ = ("_ptr",)

    def __init__(self, ptr: int):
        self._ptr = int(ptr)

    def __cuda_stream__(self):
        return (0, self._ptr)


def wrap_external_stream(ptr: int):
    """A fresh cupy stream wrapping raw handle ``ptr`` (not cached).

    CuPy >= 14 deprecates ``cupy.cuda.ExternalStream`` in favour of ``Stream.from_external``; older CuPy
    (13.x, CPython 3.9) lacks the latter and keeps the ExternalStream fallback."""
    import cupy as cp

    from_external = getattr(getattr(cp.cuda, "Stream", None), "from_external", None)
    if from_external is not None:
        return from_external(_StreamProtocol(ptr))
    return cp.cuda.ExternalStream(int(ptr))


def external_stream(ptr: int):
    """The cached ``cupy.cuda.ExternalStream`` wrapping raw handle ``ptr`` (same handle -> same wrapper object)."""
    key = int(ptr)
    with _EXTERNAL_STREAM_LOCK:
        stream = _external_stream_cache.get(key)
        if stream is None:
            _EXTERNAL_STREAM_COUNTS["misses"] += 1
            stream = wrap_external_stream(key)
            _external_stream_cache[key] = stream
        else:
            _EXTERNAL_STREAM_COUNTS["hits"] += 1
        return stream


def external_stream_stats() -> dict:
    """A copy of the :func:`external_stream` cache's hit/miss counters (record-only; the gate reads them)."""
    with _EXTERNAL_STREAM_LOCK:
        return dict(_EXTERNAL_STREAM_COUNTS)


def reset_external_stream_cache() -> None:
    """Drop the cached wrappers and zero the counters (tests / teardown)."""
    with _EXTERNAL_STREAM_LOCK:
        _external_stream_cache.clear()
        _EXTERNAL_STREAM_COUNTS["hits"] = 0
        _EXTERNAL_STREAM_COUNTS["misses"] = 0


def to_cupy(x):
    """``x`` (any tensor or DLPack / CAI producer, or an :class:`eagle.Array`) as a cupy array: zero-copy for a
    device producer, a host upload otherwise."""
    import cupy as cp

    from ..interop import to_array

    a = to_array(x)
    return cp.from_dlpack(a) if a.device.startswith("cuda") else cp.asarray(a.get())


def from_array(a):
    """The :class:`eagle.Array` ``a`` as a cupy array (a host array is uploaded)."""
    return to_cupy(a)


def user_stream(ptr: int):
    """A context making raw stream ``ptr`` cupy's current stream, for the caller's own cupy code that a graph
    node or hook runs, when the caller has imported cupy; a no-op otherwise. eagle's own work never reads
    cupy's current stream: this only lets a caller's cupy kernels join the capture they are called from."""
    import contextlib
    import sys

    return external_stream(ptr) if "cupy" in sys.modules else contextlib.nullcontext()


def from_eagle(a):
    """The :class:`eagle.Array` ``a`` as a cupy array: a view of device memory (DLPack, ordered on cupy's stream),
    an upload of host memory."""
    return to_cupy(a)


def stream_handle(stream) -> int:
    """The raw handle of a cupy stream object."""
    return int(stream.ptr)
