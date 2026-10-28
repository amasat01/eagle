# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The numpy spelling over :class:`eagle.Array` (host and device); imports no array library.

``device`` is ``"cpu"``, ``"pinned"`` or ``"cuda:N"`` (default: the input's). Ops write into ``out`` when given
and only enqueue (one kernel node in a capture). ``flatnonzero``, ``unique`` and ``bincount`` read the data on
the host: bind-time bookkeeping, never inside a graph. Scalars come back as 0-d arrays (``int(x)``, ``.item()``).
"""
from __future__ import annotations

import builtins as _builtins
import math as _math

from . import _core
from ._stream import current as _current_stream, scoped as _scoped_stream

Array = _core.Array
DType = _core.DType

bool_, uint8, int32, uint32, int64, uint64, float32, float64, float16, bfloat16, int8, int16, uint16 = (
    _core.array_dtype(c) for c in range(13))

_ADD, _SUB, _MUL, _MAX, _EQ, _NE, _GT = (_core.CELL_ADD, _core.CELL_SUB, _core.CELL_MUL, _core.CELL_MAX,
                                         _core.CELL_EQ, _core.CELL_NE, _core.CELL_GT)
# the scalar forms: (array op scalar, scalar op array)
_SCALAR = {_ADD: (_core.CELL_ADDS, _core.CELL_ADDS), _SUB: (_core.CELL_SUBS, _core.CELL_RSUBS),
           _MUL: (_core.CELL_MULS, _core.CELL_MULS), _MAX: (_core.CELL_MAXS, _core.CELL_MAXS),
           _EQ: (_core.CELL_EQS, _core.CELL_EQS), _NE: (_core.CELL_NES, _core.CELL_NES),
           _GT: (_core.CELL_GTS, _core.CELL_LTS)}


def _place(device, like=None):
    return device or (like.device if isinstance(like, Array) else "cpu")


def _s(x):
    """The stream an op into ``x`` enqueues on: the open scope, else ``x``'s own stream."""
    return (_scoped_stream() if x.device.startswith("cuda") else 0) or x.stream


def _stream_for(device, stream):
    return stream if stream is not None else (_current_stream() if str(device).startswith("cuda") else 0)


def empty(shape, dtype=float64, device=None, stream=None):
    """A new array; ``stream`` is a raw handle (the caller keeps the stream alive past the array) or a cupy/torch/warp
stream object (the array keeps it alive until its memory is freed)."""
    device = _place(device)
    shape = (shape,) if isinstance(shape, int) else tuple(shape)
    if stream is not None and not isinstance(stream, int):     # a framework's stream object: the array keeps it alive
        from .frameworks._bridge import stream_handle

        handle = stream_handle(stream)
        a = _core.array_empty(shape, dtype, device, handle)
        _core.array_hold(a, stream)
        return a
    return _core.array_empty(shape, dtype, device, _stream_for(device, stream))


def full(shape, value, dtype=None, device=None, stream=None):
    a = empty(shape, float64 if dtype is None else dtype, device, stream)
    return a.fill(value) or a


def zeros(shape, dtype=float64, device=None, stream=None): return full(shape, 0, dtype, device, stream)
def ones(shape, dtype=float64, device=None, stream=None): return full(shape, 1, dtype, device, stream)


def empty_like(a, dtype=None, device=None, stream=None):
    return empty(a.shape, a.dtype if dtype is None else dtype, _place(device, a), a.stream if stream is None else stream)


def zeros_like(a, dtype=None, device=None, stream=None): return full_like(a, 0, dtype, device, stream)
def ones_like(a, dtype=None, device=None, stream=None): return full_like(a, 1, dtype, device, stream)


def full_like(a, value, dtype=None, device=None, stream=None):
    return full(a.shape, value, a.dtype if dtype is None else dtype, _place(device, a), a.stream if stream is None else stream)


def _moved(a, device):
    b = empty(a.shape, a.dtype, device)
    _core.array_copyto(b, a)
    return b


def _dlpack_stream(obj):
    """The consumer stream handed to ``obj.__dlpack__(stream=)``: eagle's current stream for a device producer, else None."""
    try:
        kind = obj.__dlpack_device__()[0]
    except Exception:       # noqa: BLE001 -- a producer with no device query is read as host memory
        return None
    return _current_stream() if kind in (2, 13) else None


def asarray(obj, dtype=None, device=None):
    """``obj`` as an :class:`Array`: zero-copy (DLPack, CAI, buffer protocol) or a copy of a list or scalar."""
    if isinstance(obj, Array):
        a = obj
    elif hasattr(obj, "__cuda_array_interface__"):     # first for a device source: DLPack's handshake refuses inside a capture
        a = _core.array_from_cai(obj)
    elif hasattr(obj, "__dlpack__"):
        a = _core.array_from_dlpack(obj, _dlpack_stream(obj))
    elif isinstance(obj, (list, tuple, int, float)):
        return _core.array_from_list(obj, dtype, _place(device))
    else:
        a = _core.array_from_buffer(obj)
    if dtype is not None and a.dtype != dtype:
        a = a.astype(dtype)
    if device is not None and a.device != device:
        a = _moved(a, device)
    return a


def view_as(base, shape, strides=None):
    """A view of ``base``'s memory with another ``shape`` and ELEMENT ``strides`` (default: C-contiguous). Rows may
    alias; every element it names must lie inside the storage."""
    shape = tuple(shape)
    if strides is None:
        strides, acc = [], 1
        for e in reversed(shape):
            strides.append(acc)
            acc *= max(e, 1)
        strides = tuple(reversed(strides))
    return _core.array_view_as(base, shape, tuple(strides))


def copyto(dst, src, stream=None):
    if not isinstance(dst, Array):          # the caller's own array (numpy, cupy...): written through a zero-copy view
        dst = asarray(dst)
    _core.array_copyto(dst, asarray(src, device=dst.device) if not isinstance(src, Array) else src,
                       _s(dst) if stream is None else stream)


def ascontiguousarray(a):
    return a if a.is_contiguous else a.copy()


def arange(start, stop=None, step=1, dtype=int64, device=None, stream=None):
    if stop is None:
        start, stop = 0, start
    if not all(isinstance(v, int) for v in (start, stop, step)) or step == 0:
        raise TypeError("arange: integer start, stop and a non-zero step (use linspace for floats)")
    out = empty((max(0, -((start - stop) // step)),), dtype, device, stream)
    _core.array_arange(out, start, step, _s(out))
    return out


def iota(out):
    """``out[i] = i`` in place (an integer array)."""
    _core.array_arange(out, 0, 1, _s(out))


def linspace(start, stop, num=50, dtype=float64, device=None, stream=None):
    out = empty((num,), dtype, device, stream)
    _core.array_linspace(out, float(start), (stop - start) / (num - 1) if num > 1 else 0.0, _s(out))
    return out


def _operands(a, b):
    if not isinstance(a, Array):
        a = full_like(b, a)
    return a, (b if isinstance(b, Array) else full_like(a, b))


def _binary(op, a, b, out, boolean=False):
    if not (isinstance(a, Array) and isinstance(b, Array)):   # a scalar operand rides in the op: nothing is allocated
        x, scalar, flipped = (b, a, True) if isinstance(b, Array) else (a, b, False)
        if out is None:
            out = empty(x.shape, bool_ if boolean else x.dtype, x.device, x.stream)
        _core.array_binary_scalar(_SCALAR[op][flipped], out, x, scalar, _s(out))
        return out
    dt = _core.array_promote(a.dtype, b.dtype)
    a, b = (x if x.dtype == dt else x.astype(dt) for x in (a, b))
    if out is None:
        out = empty(a.shape, bool_ if boolean else dt, a.device, a.stream)
    _core.array_binary(op, out, a, b, _s(out))
    return out


def add(a, b, out=None): return _binary(_ADD, a, b, out)
def subtract(a, b, out=None): return _binary(_SUB, a, b, out)
def multiply(a, b, out=None): return _binary(_MUL, a, b, out)
def maximum(a, b, out=None): return _binary(_MAX, a, b, out)
def equal(a, b, out=None): return _binary(_EQ, a, b, out, True)
def not_equal(a, b, out=None): return _binary(_NE, a, b, out, True)
def greater(a, b, out=None): return _binary(_GT, a, b, out, True)


def where(mask, a, b, out=None):
    a, b = _operands(a, b)
    dt = _core.array_promote(a.dtype, b.dtype)
    a, b = (x if x.dtype == dt else x.astype(dt) for x in (a, b))
    out = empty(a.shape, dt, a.device, a.stream) if out is None else out
    _core.array_where(out, mask, a, b, _s(out))
    return out


def clip(a, lo, hi, out=None):
    out = empty_like(a) if out is None else out
    _core.array_clip(out, a, lo, hi, _s(out))
    return out


def take(a, index, out=None):
    out = empty(index.shape, a.dtype, a.device, a.stream) if out is None else out
    _core.array_take(out, a, index, _s(out))
    return out


def scatter(out, index, src):
    """``out[index] = src`` (int32 index)."""
    _core.array_scatter(out, index, src, _s(out))


def cumsum(a, out=None, scratch=None):
    out = empty_like(a) if out is None else out
    _core.array_cumsum(out, a, scratch, _s(out))
    return out


def cumsum_scratch_bytes(n):
    return _core.array_cumsum_scratch_bytes(n)


def count_nonzero(a, out=None):
    """Nonzero count as a 0-d int64 array on ``a``'s device (``int(x)`` reads it)."""
    out = empty((), int64, a.device, a.stream) if out is None else out
    _core.array_count_nonzero(out, a, _s(out))
    return out


def compact_indices(a, index=None, count=None, scratch=None):
    """``(index, count)``: int32 indices of the nonzero elements fill the first ``count`` slots of ``index``;
    kernels only, capture-legal when the arguments are given."""
    index = empty((a.size,), int32, a.device, a.stream) if index is None else index
    count = empty((), int64, a.device, a.stream) if count is None else count
    _core.array_flatnonzero(index, count, a, scratch, _s(index))
    return index, count


def flatnonzero(a):
    """Exact-size int32 indices; reads the count and synchronises (inside a graph use :func:`compact_indices`)."""
    index, count = compact_indices(a)
    return index[:int(count)]


def sum(a, out=None, work=None, policy=0):
    """Canonical sum as a 0-d array (``policy`` 0 compensated, 1 ordered, 2 auto). With ``out`` and ``work``
    (from :func:`sum_work`) it only enqueues."""
    if out is None:
        return a.sum()
    _core.array_sum(out, a, *(work or (None, None)), policy, _s(out))
    return out


def sum_work(a):
    """Work arrays for a device :func:`sum` of ``a``."""
    n = _core.array_reduce_work_elems(a)
    return tuple(empty((n,), a.dtype, a.device, a.stream) for _ in range(2)) if n else (None, None)


def stack(arrays, axis=0):
    if axis != 0:
        raise ValueError("stack: axis 0 only")
    first = arrays[0]
    out = empty((len(arrays),) + tuple(first.shape), first.dtype, first.device, first.stream)
    for i, x in enumerate(arrays):
        _core.array_copyto(out[i], x)
    return out


def concatenate(arrays):
    first, at = arrays[0], 0
    out = empty((_builtins.sum(x.shape[0] for x in arrays),) + tuple(first.shape[1:]), first.dtype, first.device, first.stream)
    for x in arrays:
        _core.array_copyto(out[at:at + x.shape[0]], x)
        at += x.shape[0]
    return out


def diff(a):
    return subtract(a[1:], a[:-1])


def repeat(a, k):
    return take(a, asarray([i for i in range(a.shape[0]) for _ in range(k)], int32, a.device))


def bincount(a, minlength=0):
    values = a.tolist()
    counts = [0] * max(minlength, (max(values) + 1) if values else 0)
    for v in values:
        counts[v] += 1
    return asarray(counts, int64, a.device)


def unique(a):
    return asarray(sorted(set(a.tolist())), a.dtype, a.device)


def prod(x):
    return _math.prod(x.tolist() if isinstance(x, Array) else x)


def shares_memory(a, b):
    def span(x):
        reach = [(n - 1) * s for n, s in zip(x.shape, x.strides) if n > 1]
        return x.ptr + _builtins.sum(r for r in reach if r < 0), x.ptr + _builtins.sum(r for r in reach if r > 0) + x.itemsize
    (alo, ahi), (blo, bhi) = span(a), span(b)
    return a.size > 0 and b.size > 0 and alo < bhi and blo < ahi


def _to(self, framework):
    """``self`` as ``framework``'s array type (``"numpy"``, ``"cupy"``, ``"torch"``, ``"jax"``), through that
    framework's bridge (``eagle.frameworks.<framework>.from_eagle``): zero-copy where the framework can alias the
    memory. Raises ``ImportError`` with the install hint if the framework is missing."""
    from .frameworks._bridge import to

    return to(self, framework)


Array.to = _to
