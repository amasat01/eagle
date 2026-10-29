# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``eagle.fft``: batched Fourier transforms as graph steps, differentiable.

Arrays are an :class:`eagle.Array` or anything :meth:`eagle.Plan.run` takes (numpy on the host; cupy, torch or any DLPack
producer on the device), wrapped without a copy: ``out`` is written in place in the caller's buffer, and a read-only
``out`` is refused by name.

Complex data is a ``float32`` / ``float64`` :class:`eagle.Array` whose LAST axis has extent 2 (real part, imaginary
part): eagle has no complex dtype, and a complex array is refused by name (pass a float64 view of (re, im) pairs).
``axes`` count the data axes only (the trailing 2 is not one). Normalisation is numpy's ``norm``: ``"backward"``
(default: forward unscaled, inverse scaled by ``1/n``), ``"forward"`` or ``"ortho"``.

Each function returns a :class:`~eagle.graph.Node` for :meth:`eagle.Graph.add_node` (or
:meth:`eagle.diff.ReverseCtx.add`). The transform runs on the stream scope the graph is built in, allocates nothing
when it runs, and is legal inside a stream capture. Its plan is built here, eagerly, so building a node inside a
capture is refused. The host runs pocketfft and the device cuFFT, both through ``aether.fft`` (``aether-dsc >= 0.7.2``;
a device transform without cuFFT raises aether's ``FftUnavailable`` naming the extra to install).

Derivatives are the opposite-direction transform, with the scaling pinned: the vjp of ``fft(norm="backward")`` is the
UNSCALED inverse (``ifft(norm="forward")``), and in general the vjp swaps the direction and ``"backward"`` with
``"forward"`` (``"ortho"`` stays). :func:`vjp` and :func:`jvp` are graph steps like the transforms, so a captured
graph can hold a forward and its backward; ``eagle.diff.Reverse`` pulls an ``fft`` node back through the same rule.
The real transforms carry the Hermitian weight (the interior of the half spectrum counts twice) in the adjoint.
"""

from __future__ import annotations

from . import _core, _stream, arrays
from .dtypes import dtype_name
from .graph import Node, register_node_kind

__all__ = ["c2c", "r2c", "c2r", "vjp", "jvp"]

_NORMS = ("backward", "forward", "ortho")
_SWAP = {"backward": "forward", "forward": "backward", "ortho": "ortho"}


def _door(msg: str) -> str:
    return f"eagle.fft: {msg}"


def _aether():
    try:
        import aether
        from aether import fft as aether_fft

        aether_fft.plan
    except (ImportError, AttributeError) as exc:
        raise ImportError(_door("needs aether-dsc >= 0.7.2 (its fft module): pip install -U aether-dsc")) from exc
    return aether, aether_fft


class _Op:
    """One transform: the aether plan and the arrays it runs over, built outside a capture."""

    __slots__ = ("mode", "forward", "norm", "axes", "x", "out", "plan", "weights", "pre", "device")

    def __init__(self, mode, forward, norm, axes, x, out, plan, weights, pre, device):
        self.mode, self.forward, self.norm, self.axes = mode, forward, norm, axes
        self.x, self.out, self.plan, self.weights, self.pre, self.device = x, out, plan, weights, pre, device


def _wrap(x, who: str, written: bool):
    """``x`` as an :class:`eagle.Array` over the caller's own memory, the import :meth:`eagle.Plan.run` uses (numpy on the
    host; cupy, torch or any DLPack / CUDA-array-interface producer on the device). A written array must be the caller's
    buffer: a read-only one, or anything that would be copied, is refused by name."""
    name = str(getattr(x, "dtype", ""))
    if "complex" in name:
        raise TypeError(_door(f"{who} is a complex array, which eagle does not hold: pass a float64 view of (re, im) pairs "
                              "(a real array whose last axis has extent 2)"))
    if isinstance(x, arrays.Array):
        return x
    if isinstance(x, (list, tuple, int, float)) or not any(
            hasattr(x, a) for a in ("__cuda_array_interface__", "__dlpack__", "__array_interface__", "__buffer__")):
        raise TypeError(_door(f"{who} must be an eagle.Array or an array from numpy, cupy, torch or any DLPack / CUDA-array-"
                              f"interface producer, got {type(x).__name__}"))
    if written and getattr(getattr(x, "flags", None), "writeable", True) is False:
        raise ValueError(_door(f"{who} is read-only, but the transform writes into it: pass a writable array"))
    if written and getattr(x, "requires_grad", False):
        raise ValueError(_door(f"{who} requires grad, but the transform writes into it: pass a detached buffer"))
    from .interop import to_array

    return to_array(x)


def _check_array(x, who: str, written: bool = False):
    x = _wrap(x, who, written)
    if dtype_name(x.dtype) not in ("float32", "float64"):
        raise TypeError(_door(f"{who} dtype {x.dtype} must be float32 or float64"))
    if not x.is_contiguous:
        raise ValueError(_door(f"{who} must be contiguous"))
    return x


def _check_pairs(x, who: str):
    if x.ndim < 2 or x.shape[-1] != 2:
        raise ValueError(_door(f"{who} is a float array with a trailing axis of 2 (re, im); this one has shape {tuple(x.shape)}"))


def _axes(axes, rank: int):
    if axes is None:
        axes = tuple(range(rank))
    elif isinstance(axes, int):
        axes = (axes,)
    out = []
    for a in axes:
        if not -rank <= a < rank:
            raise ValueError(_door(f"axis {a} is out of range for {rank} data axes"))
        out.append(a % rank)
    if len(set(out)) != len(out) or not out:
        raise ValueError(_door(f"axes {tuple(axes)} are empty or repeat an axis"))
    return tuple(out)


def _refuse_in_capture(device):
    if device != "cpu" and _core.stream_is_capturing(_stream.current()):   # a host plan has no stream to capture
        raise RuntimeError(_door("a plan is built outside a stream capture (build the node first, capture after)"))


def _hermitian(n: int, m: int, scale: float, device: str, dtype: str):
    """The 1-D weights over the ``m`` points of a half spectrum of a real axis of ``n``: 1 at DC and (n even) Nyquist,
    ``scale`` between."""
    last = m - 1 if n % 2 == 0 else m
    return arrays.asarray([1.0 if (k == 0 or k >= last) else scale for k in range(m)], dtype, device)


def _make(mode, x, out, axes, forward, norm, weights=None, pre=False, plan=None):
    if norm not in _NORMS:
        raise ValueError(_door(f"norm= wants 'backward', 'forward' or 'ortho', not {norm!r}"))
    x = _check_array(x, "x")
    out = _check_array(out, "out", written=True)
    if x.device != out.device:
        raise ValueError(_door(f"x lives on {x.device} and out on {out.device}"))
    device = x.device
    aether, aether_fft = _aether()
    if mode == "r2c":
        shape = tuple(x.shape)                       # real input
        _check_pairs(out, "out")
        data = shape
    else:
        _check_pairs(x, "x")
        data = tuple(x.shape[:-1])
    ax = _axes(axes, len(data))
    if mode == "c2c":
        _check_pairs(out, "out")
        if tuple(out.shape) != tuple(x.shape):
            raise ValueError(_door(f"out shape {tuple(out.shape)} must equal the x shape {tuple(x.shape)}"))
        real = data
    elif mode == "r2c":
        half = list(shape)
        half[ax[-1]] = shape[ax[-1]] // 2 + 1
        if tuple(out.shape) != tuple(half) + (2,):
            raise ValueError(_door(f"out shape {tuple(out.shape)} must be {tuple(half) + (2,)} (the half spectrum along axis {ax[-1]})"))
        real = shape
    else:
        real = tuple(out.shape)                      # real output
        if len(real) != len(data) or real[ax[-1]] // 2 + 1 != data[ax[-1]]:
            raise ValueError(_door(f"out shape {real} does not match a half spectrum of {data} along axis {ax[-1]}"))
    _refuse_in_capture(device)
    if plan is None:
        shape_for = real if mode != "c2c" else data
        plan = aether_fft.plan(mode, dtype_name(x.dtype), shape_for, ax, forward, norm, "cpu" if device == "cpu" else device)
    return Node("fft", _Op(mode, forward, norm, ax, x, out, plan, weights, pre, device))


def c2c(x, out, axes=None, *, forward: bool = True, norm: str = "backward") -> Node:
    """Complex to complex over ``axes`` (default: every data axis); ``x`` and ``out`` have the same shape ``(..., 2)``."""
    return _make("c2c", x, out, axes, forward, norm)


def r2c(x, out, axes=None, *, norm: str = "backward") -> Node:
    """Real to complex (forward) over ``axes``: ``x`` is real ``shape``, ``out`` is ``shape`` with the extent of the last
    of ``axes`` replaced by ``n // 2 + 1``, plus the trailing 2."""
    return _make("r2c", x, out, axes, True, norm)


def c2r(x, out, axes=None, *, norm: str = "backward") -> Node:
    """Complex to real (inverse) over ``axes``: ``x`` is the half spectrum ``(..., n // 2 + 1, ..., 2)`` and ``out`` the real
    ``shape``; ``x`` is left unchanged."""
    return _make("c2r", x, out, axes, False, norm)


def _op(node) -> _Op:
    if not isinstance(node, Node) or node.kind != "fft":
        raise TypeError(_door(f"expected a node made by eagle.fft, got {type(node).__name__}"))
    return node.payload


def jvp(node, tangent, out) -> Node:
    """The forward-mode step of an fft ``node``: the transform is linear, so the tangent of its output is the same
    transform of the tangent of its input. ``tangent`` and ``out`` have the shapes of the node's ``x`` and ``out``."""
    op = _op(node)
    return _make(op.mode, tangent, out, op.axes, op.forward, op.norm)


def vjp(node, cotangent, out) -> Node:
    """The reverse-mode step of an fft ``node``: ``cotangent`` has the shape of the node's output and ``out`` that of its
    input. The opposite-direction transform with ``"backward"`` and ``"forward"`` swapped, so the vjp of the unscaled
    forward is the unscaled inverse; a real transform also weights the interior of the half spectrum (twice for the c2r
    side, a half for the r2c side)."""
    op = _op(node)
    cotangent = _check_array(cotangent, "cotangent")
    out = _check_array(out, "out", written=True)
    norm = _SWAP[op.norm]
    if op.mode == "c2c":
        return _make("c2c", cotangent, out, op.axes, not op.forward, norm)
    # n and m along the halved axis of the half spectrum
    spec = op.out if op.mode == "r2c" else op.x
    real = op.x if op.mode == "r2c" else op.out
    last = op.axes[-1]
    n, m = int(real.shape[last]), int(spec.shape[last])
    if op.mode == "r2c":
        # cotangent is a half spectrum: halve its interior, then the unscaled c2r
        w = _hermitian(n, m, 0.5, cotangent.device, dtype_name(cotangent.dtype))
        return _make("c2r", cotangent, out, op.axes, False, norm, weights=(w, last, m), pre=True)
    w = _hermitian(n, m, 2.0, cotangent.device, dtype_name(cotangent.dtype))
    return _make("r2c", cotangent, out, op.axes, True, norm, weights=(w, last, m), pre=False)


# --- the node kind --------------------------------------------------------------------------------------------------- #

def _reads(node):
    return (("x", node.payload.x, "vec_in"),)


def _writes(node):
    return (("out", node.payload.out, "out"),)


def _weighted(spec_array, weights):
    """``weights`` (``(w, axis, m)``) as a stride-0 broadcast view shaped like the half spectrum ``spec_array``."""
    w, axis, _m = weights
    shape = tuple(spec_array.shape)
    strides = [0] * len(shape)
    strides[axis] = 1
    return arrays.view_as(w, shape, tuple(strides))


def _lower(node, ctx):
    op = node.payload
    aether, _ = _aether()
    host = op.device == "cpu"
    weights = op.weights
    if weights is not None:
        spec = op.x if op.pre else op.out
        view = arrays.empty_like(spec)              # the weights spread over the spectrum: a contiguous multiply allocates nothing
        arrays.copyto(view, _weighted(spec, weights))
        tmp = arrays.empty_like(op.x) if op.pre else None
    source = tmp if (weights is not None and op.pre) else op.x
    a_in, a_out = aether.asarray(source), aether.asarray(op.out)
    ordinal = None if host else int(op.device.split(":")[1])

    def step():
        stream = None if host else _stream.current(ordinal)
        if weights is not None and op.pre:
            arrays.multiply(op.x, view, out=tmp)
        op.plan.execute(a_in, a_out, stream)
        if weights is not None and not op.pre:
            arrays.multiply(op.out, view, out=op.out)

    step.work = [a for a in ((view, tmp) if weights is not None else ()) if a is not None]
    return step


def _transpose(node, incoming, ctx):
    """The reverse of an fft node (:mod:`eagle.diff`): its vjp step."""
    op = node.payload
    cot = incoming["out"]
    bar = ctx.empty_like(op.x)
    ctx.add(vjp(node, cot, bar), reads=[cot], writes=[bar])
    return {"x": bar}


register_node_kind("fft", reads=_reads, writes=_writes, lower_gpu=_lower, lower_cpu=_lower, transpose=_transpose)
