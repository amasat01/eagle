# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.reduce``: the reduce node kind of :class:`eagle.Graph`.

``eagle.reduce(x, out, op="sum")`` reduces the 1-D sample plane ``x`` into the
one-element plane ``out`` with the CANONICAL fixed-order tree (leaf blocks of
256 samples, then level-by-level pairwise; see ``eagle/reduce/canonical.h``).
The result is a function of the values in sample order only: bitwise
identical for every OpenMP thread count on the CPU, and bitwise identical
between the CPU and the GPU.

The kind registers itself here; ``graph.py`` never changes.
"""

from __future__ import annotations

from .graph import Node, register_node_kind

__all__ = ["reduce"]

#: op name (numpy's) -> (core op code, element dtypes it accepts)
_OPS = {
    "sum": (0, ("float32", "float64", "int32", "int64")),
    "max": (1, ("float32", "float64", "int32", "int64")),
    "all": (2, ("bool",)),
}
#: numpy dtype name -> core dtype code (bool is one byte, reduced as u8)
_DTYPES = {"float32": 0, "float64": 1, "int32": 2, "int64": 3, "bool": 4}


def _door(msg: str) -> str:
    return f"eagle.reduce: {msg}"


def _is_cupy(arr) -> bool:
    return type(arr).__module__.split(".")[0] == "cupy"


def reduce(x, out, *, op: str = "sum") -> Node:
    """A graph node that reduces the 1-D sample plane ``x`` into ``out``.

    Args:
        x: the plane to reduce, a contiguous 1-D numpy or cupy array with at
            least one sample. Floating-point and integer planes take
            ``"sum"`` and ``"max"``; a ``bool`` plane takes ``"all"``.
        out: a one-element contiguous array of the same dtype on the same
            backend; it receives the result.
        op: ``"sum"``, ``"max"`` or ``"all"`` (numpy's names).

    Returns:
        A node for :meth:`eagle.Graph.add_node`. The edges to the nodes that
        produce ``x`` and consume ``out`` are derived from the arrays.

    Raises:
        ValueError: for an unknown ``op``, an empty ``x``, or an ``out`` that
            is not one element or overlaps ``x``.
        TypeError: for a dtype the operator does not take, a dtype mismatch,
            or ``x`` and ``out`` on different backends.
    """
    if op not in _OPS:
        raise ValueError(_door(f"op {op!r} is not one of {sorted(_OPS)}"))
    for label, arr in (("x", x), ("out", out)):
        if not hasattr(arr, "dtype") or not hasattr(arr, "shape"):
            raise TypeError(_door(f"{label} must be a numpy or cupy array, got {type(arr).__name__}"))
    if x.ndim != 1:
        raise ValueError(_door(f"x must be a 1-D sample plane, got shape {tuple(x.shape)}"))
    if x.shape[0] < 1:
        raise ValueError(_door("x holds no samples"))
    if tuple(out.shape) != (1,):
        raise ValueError(_door(f"out must hold exactly one element, got shape {tuple(out.shape)}"))
    code, allowed = _OPS[op]
    name = str(x.dtype)
    if name not in allowed:
        raise TypeError(_door(f"op {op!r} takes {list(allowed)} planes, got {name}"))
    if str(out.dtype) != name:
        raise TypeError(_door(f"out dtype {out.dtype} must equal the x dtype {name}"))
    if _is_cupy(x) != _is_cupy(out):
        raise TypeError(_door("x and out must live on the same backend (both numpy or both cupy)"))
    if not (x.flags.c_contiguous and out.flags.c_contiguous):
        raise ValueError(_door("x and out must be contiguous (numpy.ascontiguousarray)"))
    if _is_cupy(x):
        if x.data.ptr <= out.data.ptr < x.data.ptr + x.nbytes:
            raise ValueError(_door("out overlaps x"))
    else:
        import numpy as np

        if np.shares_memory(x, out):
            raise ValueError(_door("out overlaps x"))
    return Node("reduce", (x, out, code, _DTYPES[name]))


def _reads(node):
    return (("x", node.payload[0], "vec_in"),)


def _writes(node):
    return (("out", node.payload[1], "out"),)


def _lower_cpu(node, ctx):
    x, out, op, dtype = node.payload
    from . import _core

    n = int(x.shape[0])
    xp, op_ptr = x.ctypes.data, out.ctypes.data

    def step():
        _core.reduce_canonical_host(xp, n, dtype, op, op_ptr)
    return step


def _lower_gpu(node, ctx):
    x, out, op, dtype = node.payload
    import cupy

    from . import _core

    n = int(x.shape[0])
    device = int(x.device.id)
    with cupy.cuda.Device(device):
        # the work buffers belong to the lowered step, not to a launch
        elems = int(_core.reduce_canonical_device(0, n, dtype, op, 0, 0, 0, 0, device))
        work = [cupy.zeros(elems, dtype=x.dtype if x.dtype != cupy.bool_ else cupy.uint8) for _ in range(2)]
    xp, op_ptr = int(x.data.ptr), int(out.data.ptr)
    wa, wb = int(work[0].data.ptr), int(work[1].data.ptr)

    def step():
        _core.reduce_canonical_device(xp, n, dtype, op, op_ptr, wa, wb,
                                      int(cupy.cuda.get_current_stream().ptr), device)
    step.work = work
    return step


register_node_kind("reduce", reads=_reads, writes=_writes,
                   lower_gpu=_lower_gpu, lower_cpu=_lower_cpu)
