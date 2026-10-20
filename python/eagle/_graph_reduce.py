# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.reduce``: the reduce node kind of :class:`eagle.Graph`.

``eagle.reduce(x, out, op="sum", policy="compensated")`` reduces the 1-D
sample plane ``x`` into the one-element ``out``, or each column of the 2-D
``(rows, N)`` plane ``x`` into ``out`` of shape ``(N,)``. Every policy runs the
same leaf (256 slots of 16 strided samples, a Neumaier walk per slot, a fixed
in-leaf tree; see ``eagle/reduce/canonical.h``). ``ordered`` then combines the
leaves pairwise level by level: bitwise identical for every thread count and
between the CPU and the GPU. ``compensated`` commits the leaves with two-sum
atomics on the GPU (not bitwise run to run); the CPU runs ``ordered``. For n
terms with condition number k both are within (1 + n*k*u) ULP of the exact sum.
Integer sums, max and all are exact and ignore the policy.

The kind registers itself here; ``graph.py`` never changes.
"""

from __future__ import annotations

from types import MappingProxyType

from .graph import Node, register_node_kind

__all__ = ["reduce"]

#: op name (numpy's) -> (core op code, element dtypes it accepts)
_OPS = {
    "sum": (0, ("float32", "float64", "int32", "int64")),
    "max": (1, ("float32", "float64", "int32", "int64")),
    "all": (2, ("bool",)),
    "min": (3, ("float32", "float64", "int32", "int64")),
}
#: numpy dtype name -> core dtype code (bool is one byte, reduced as u8)
_DTYPES = {"float32": 0, "float64": 1, "int32": 2, "int64": 3, "bool": 4}
#: policy name -> core code (``auto`` is ``compensated`` in this release)
POLICIES = MappingProxyType({"compensated": 0, "ordered": 1, "auto": 2})


def _door(msg: str) -> str:
    return f"eagle.reduce: {msg}"


def _is_cupy(arr) -> bool:
    return type(arr).__module__.split(".")[0] == "cupy"


def _overlaps(a, b) -> bool:
    if _is_cupy(a):
        return a.data.ptr < b.data.ptr + b.nbytes and b.data.ptr < a.data.ptr + a.nbytes
    import numpy as np

    return bool(np.shares_memory(a, b))


def reduce(x, out, *, op: str = "sum", comp=None, policy: str = "compensated") -> Node:
    """A graph node that reduces the samples of ``x`` into ``out``.

    Args:
        x: the plane to reduce, a contiguous numpy or cupy array with at least
            one sample: 1-D ``(n,)``, or 2-D ``(rows, N)`` reduced column by
            column. Floating-point and integer planes take ``"sum"``,
            ``"max"`` and ``"min"``; a ``bool`` plane takes ``"all"``.
        out: a contiguous array of the same dtype on the same backend, shape
            ``(1,)`` for a 1-D ``x`` and ``(N,)`` for a 2-D one; it receives
            the result.
        op: ``"sum"``, ``"max"``, ``"min"`` or ``"all"`` (numpy's names).
        comp: the companion plane of a floating-point sum, same shape and
            dtype as ``x``: each term is the pair ``x + comp`` (rows of
            (hi, comp) partials), finished together.
        policy: ``"compensated"`` (default), ``"ordered"`` or ``"auto"``
            (``compensated`` in this release). Integer sums, max and all
            ignore it.

    Returns:
        A node for :meth:`eagle.Graph.add_node`. The edges to the nodes that
        produce ``x`` (and ``comp``) and consume ``out`` are derived from the
        arrays.

    Raises:
        ValueError: for an unknown ``op`` or ``policy``, an empty ``x``, an
            ``out`` of the wrong shape, ``comp`` of another shape, or an
            ``out`` that overlaps an input.
        TypeError: for a dtype the operator does not take, a dtype mismatch,
            arrays on different backends, or ``comp`` on anything but a
            floating-point sum.
    """
    if op not in _OPS:
        raise ValueError(_door(f"op {op!r} is not one of {sorted(_OPS)}"))
    if policy not in POLICIES:
        raise ValueError(_door(f"policy {policy!r} is not one of {sorted(POLICIES)}"))
    arrays = (("x", x), ("out", out)) + ((("comp", comp),) if comp is not None else ())
    for label, arr in arrays:
        if not hasattr(arr, "dtype") or not hasattr(arr, "shape"):
            raise TypeError(_door(f"{label} must be a numpy or cupy array, got {type(arr).__name__}"))
    if x.ndim not in (1, 2):
        raise ValueError(_door(f"x must be a 1-D sample plane or a 2-D (rows, N) plane, got shape {tuple(x.shape)}"))
    if x.shape[0] < 1 or x.size < 1:
        raise ValueError(_door("x holds no samples"))
    want = (1,) if x.ndim == 1 else (int(x.shape[1]),)
    if tuple(out.shape) != want:
        raise ValueError(_door(f"out must have shape {want} for x of shape {tuple(x.shape)}, got {tuple(out.shape)}"))
    code, allowed = _OPS[op]
    name = str(x.dtype)
    if name not in allowed:
        raise TypeError(_door(f"op {op!r} takes {list(allowed)} planes, got {name}"))
    for label, arr in arrays[1:]:
        if str(arr.dtype) != name:
            raise TypeError(_door(f"{label} dtype {arr.dtype} must equal the x dtype {name}"))
        if _is_cupy(x) != _is_cupy(arr):
            raise TypeError(_door(f"x and {label} must live on the same backend (both numpy or both cupy)"))
    if comp is not None:
        if op != "sum" or name not in ("float32", "float64"):
            raise TypeError(_door("comp= takes a float32 or float64 sum"))
        if tuple(comp.shape) != tuple(x.shape):
            raise ValueError(_door(f"comp shape {tuple(comp.shape)} must equal the x shape {tuple(x.shape)}"))
    if not all(arr.flags.c_contiguous for _, arr in arrays):
        raise ValueError(_door("x, out and comp must be contiguous (numpy.ascontiguousarray)"))
    for label, arr in arrays[::2]:
        if _overlaps(arr, out):
            raise ValueError(_door(f"out overlaps {label}"))
    return Node("reduce", (x, out, code, _DTYPES[name], comp, POLICIES[policy]))


def _shape(x):
    """(rows, columns) of a 1-D or 2-D plane."""
    return int(x.shape[0]), (int(x.shape[1]) if x.ndim == 2 else 1)


def _reads(node):
    comp = node.payload[4]
    return (("x", node.payload[0], "vec_in"),) + ((("comp", comp, "vec_in"),) if comp is not None else ())


def _writes(node):
    return (("out", node.payload[1], "out"),)


def _lower_cpu(node, ctx):
    x, out, op, dtype, comp, _policy = node.payload
    from . import _core

    rows, cols = _shape(x)
    xp, op_ptr, cp = x.ctypes.data, out.ctypes.data, (0 if comp is None else comp.ctypes.data)

    def step():
        _core.reduce_canonical_host(xp, rows, dtype, op, op_ptr, cols, cp)
    return step


def _lower_gpu(node, ctx):
    x, out, op, dtype, comp, policy = node.payload
    import cupy

    from . import _core

    rows, cols = _shape(x)
    device = int(x.device.id)
    with cupy.cuda.Device(device):
        # the work buffers belong to the lowered step, not to a launch
        elems = int(_core.reduce_canonical_device(0, rows, dtype, op, 0, 0, 0, 0, device, policy, cols))
        work = [cupy.zeros(elems, dtype=x.dtype if x.dtype != cupy.bool_ else cupy.uint8) for _ in range(2)]
    xp, op_ptr = int(x.data.ptr), int(out.data.ptr)
    cp = 0 if comp is None else int(comp.data.ptr)
    wa, wb = int(work[0].data.ptr), int(work[1].data.ptr)

    def step():
        _core.reduce_canonical_device(xp, rows, dtype, op, op_ptr, wa, wb,
                                      int(cupy.cuda.get_current_stream().ptr), device, policy, cols, cp)
    step.work = work
    return step


def _transpose(node, incoming, ctx):
    """The reverse of a sum (:mod:`eagle.diff`): every sample receives the
    adjoint of the total."""
    x, _out, op, _dtype, comp, _policy = node.payload
    if x.ndim != 1 or comp is not None:
        raise ValueError(
            f"eagle.diff.Reverse: node {ctx.label!r} is a 2-D or comp= reduce, which "
            "eagle does not pull back yet; only a 1-D op='sum' reduce reverses")
    if op != _OPS["sum"][0]:
        raise ValueError(
            f"eagle.diff.Reverse: node {ctx.label!r} is a 'max', 'min' or 'all' reduce, "
            "which has no gradient eagle can pull back (a tie has no single "
            "answer); only op='sum' reverses")
    from .diff import broadcast_adjoint

    cot, bar_x, n = incoming["out"], ctx.empty_like(x), int(x.shape[0])
    ctx.add(lambda: broadcast_adjoint(cot, n, out=bar_x), reads=[cot], writes=[bar_x])
    return {"x": bar_x}


register_node_kind("reduce", reads=_reads, writes=_writes,
                   lower_gpu=_lower_gpu, lower_cpu=_lower_cpu, transpose=_transpose)
