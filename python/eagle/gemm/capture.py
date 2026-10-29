"""A capture-legal GEMM step: cuBLAS behind eagle's backend seam.

A strides-only shape mapping plus :class:`CublasCaptureHandle`, which enqueues the product through the seam's
``gemm`` call (:func:`eagle._core.gemm`). The plugin loads cuBLAS when it is first asked for one; without it the
call is refused by name (:class:`GemmCaptureUnavailable`), never silently replaced. Operands are
:class:`eagle.Array` s or any array that exposes its device address (``.ptr``, ``.data.ptr`` or an integer).

:mod:`eagle.gemm.plan`'s ``matmul_step`` runs the same seam call outside a capture.

Two pieces: :class:`CublasCaptureHandle` (the cuBLAS handle of one captured-graph build's stream) and
:func:`sgemm_params`, the row-major to column-major mapping, a pure function over shapes and strides.

Refusals are typed and loud (:class:`GemmCaptureUnavailable` and its subclass), never fallbacks.
"""

from __future__ import annotations

from typing import NamedTuple

from .. import _core
from ..dtypes import dtype_name
from ..launch import _current_device_ordinal

__all__ = [
    "CUBLAS_OP",
    "CublasCaptureHandle",
    "GemmCaptureUnavailable",
    "GemmMappingUnavailable",
    "SgemmParams",
    "check_plan_capturable",
    "gemm_call",
    "require_float32",
    "sgemm_params",
    "sgemm_params_for_arrays",
]

CUBLAS_OP = {"N": 0, "T": 1}

#: Sgemm is single precision by definition; the zero-FP64 discipline makes
#: that a fence: a float64 operand is refused here even though
#: ``lower_dense_gemm``'s own dtype default is still ``"float64"``.
GEMM_DTYPE = "float32"
GEMM_ITEMSIZE = 4


class GemmCaptureUnavailable(Exception):
    """Base for "this GEMM cannot be run as a captured cuBLAS step"."""


class GemmMappingUnavailable(GemmCaptureUnavailable):
    """This contraction cannot be expressed as one Sgemm call: a shape,
    dtype, stride pattern or operand class the mapping does not cover.
    Permanent for the operands as bound -- leave the step uncaptured, not
    retry."""


# --------------------------------------------------------------------------- #
# The seam call
# --------------------------------------------------------------------------- #
def _device_ptr(operand) -> int:
    """``operand``'s address: an :class:`eagle.Array`'s ``ptr``, a cupy array's ``.data.ptr``, a numpy array's
    interface address, or an address already given as an integer."""
    ptr = getattr(operand, "ptr", None)
    if ptr is None:
        ptr = getattr(getattr(operand, "data", None), "ptr", None)
    if ptr is None:
        interface = getattr(operand, "__array_interface__", None)
        ptr = interface["data"][0] if interface else None
    return int(operand if ptr is None else ptr)


def gemm_call(params, operands: dict, *, itemsize: int = GEMM_ITEMSIZE, stream: int = 0, device: int = -1,
              host: bool = False, alpha: float = 1.0, beta: float = 0.0) -> None:
    """Enqueue ONE GEMM ``params`` describes through the seam (``host``: plain C++ loops on host memory, else
    cuBLAS on ``stream``). Raises :class:`GemmCaptureUnavailable` when the backend cannot run it (no cuBLAS)."""
    first, second = params.operands
    try:
        _core.gemm(0 if itemsize == 4 else 1, CUBLAS_OP[params.op_a], CUBLAS_OP[params.op_b], params.m, params.n, params.k,
                   _device_ptr(operands[first]), params.lda, _device_ptr(operands[second]), params.ldb,
                   _device_ptr(operands["out"]), params.ldc, 1, 0, 0, 0, alpha, beta, stream, device, host)
    except RuntimeError as exc:                 # the seam's refusal (no cuBLAS, a bad call), by name
        raise GemmCaptureUnavailable(str(exc)) from exc


# --------------------------------------------------------------------------- #
# The handle
# --------------------------------------------------------------------------- #
class CublasCaptureHandle:
    """The cuBLAS handle of ONE captured-graph build's stream, kept by the backend per (device, stream). The
    lifetime rules are why this is an object, not a function: the handle is created at the first call, which
    allocates, so :meth:`warm` runs on the capture stream before ``begin_capture``, once per GEMM shape; only
    :meth:`sgemm` runs inside the capture region; the handle outlives its graph (cuBLAS keeps a per-handle
    workspace, so destroying it early is a use-after-free at replay), and teardown order is graph, then
    :meth:`destroy`, then pool. ``alpha``/``beta`` are host-side scalars baked into the recorded graph node at
    enqueue time: changing them means recording again."""

    def __init__(self, stream_ptr, device=None):
        #: The CUDA device the handle (and its stream) belong to; ``None``
        #: is the current one. Operands on another device are refused by name.
        self.device = _current_device_ordinal() if device is None else int(device)
        self._stream = int(stream_ptr)
        self._alive = True
        self._warmed = set()

    @property
    def alive(self) -> bool:
        """Whether the handle is still usable (``False`` after :meth:`destroy`)."""
        return self._alive

    def warm(self, params: SgemmParams, operands: dict) -> None:
        """Run this GEMM shape once, outside the capture region.
        Idempotent per shape, so a caller may warm defensively before
        every recorded step without cost. Writes the destination exactly
        as the captured call will, so contents that matter must be
        re-filled after."""
        if params in self._warmed:
            return
        self.sgemm(params, operands)
        self._warmed.add(params)

    def sgemm(self, params: SgemmParams, operands: dict) -> None:
        """Enqueue ONE Sgemm on the handle's stream -- the only call legal
        inside a capture region. ``operands`` maps
        ``"a"``/``"b"``/``"out"`` to the caller's device buffers; which of
        ``a``/``b`` is the first cuBLAS operand is ``params.operands``,
        since the row-major to column-major mapping swaps them for every
        case except ``transpose_out``."""
        if not self._alive:
            raise GemmCaptureUnavailable(
                "this cuBLAS handle was destroyed; a handle outlives its graph "
                "and is never reused across builds"
            )
        for name, operand in operands.items():
            on = getattr(getattr(operand, "data", None), "device_id", None)
            if on is None:
                device = getattr(operand, "device", None)
                on = int(device.split(":")[1]) if isinstance(device, str) and ":" in device else self.device
            if on != self.device:
                raise ValueError(
                    f"eagle.gemm: operand {name!r} lives on device {on}, the handle on "
                    f"device {self.device}; a gemm uses one device")
        gemm_call(params, operands, stream=self._stream, device=self.device)

    def destroy(self) -> None:
        """Destroy the handle. Call after destroying the graph that
        recorded work against it, and before freeing the memory pool.
        Idempotent."""
        if self._alive:
            _core.gemm_release(self._stream, self.device)
            self._alive = False


# --------------------------------------------------------------------------- #
# The mapping
# --------------------------------------------------------------------------- #
class SgemmParams(NamedTuple):
    """Everything one ``cublasSgemm_v2`` call needs, bar the pointers.
    ``op_a``/``lda`` and ``op_b``/``ldb`` are named for the cuBLAS
    argument position, not the caller's arrays: ``operands`` says which
    caller buffer fills each position, swapped for every case except
    ``transpose_out``. ``m``/``n``/``k`` are the column-major extents
    cuBLAS is told, not necessarily the row-major result's own shape."""

    op_a: str
    op_b: str
    m: int
    n: int
    k: int
    lda: int
    ldb: int
    ldc: int
    operands: tuple


def _leading_dimension(shape, strides, itemsize, *, what: str) -> int:
    """The row stride of a 2-D row-major buffer, in elements. From the
    strides, never ``shape[1]``: production operands are slice views
    whose row stride is the parent's width. Refuses anything one Sgemm
    cannot address."""
    if len(shape) != 2:
        raise GemmMappingUnavailable(
            f"{what} must be 2-D to map onto one Sgemm; got shape {tuple(shape)}"
        )
    if len(strides) != 2:
        raise GemmMappingUnavailable(
            f"{what} has {len(strides)} strides for a 2-D shape"
        )
    if strides[1] != itemsize:
        raise GemmMappingUnavailable(
            f"{what} must be unit-stride along its last axis (row-major, "
            f"possibly row- or column-sliced); got strides {tuple(strides)} at "
            f"itemsize {itemsize}. An F-ordered or step-sliced operand has no "
            "single leading dimension that describes it"
        )
    if strides[0] % itemsize:
        raise GemmMappingUnavailable(
            f"{what}'s row stride {strides[0]} is not a whole number of "
            f"{itemsize}-byte elements"
        )
    ld = strides[0] // itemsize
    if ld < shape[1]:
        raise GemmMappingUnavailable(
            f"{what}'s row stride ({ld} elements) is smaller than its own width "
            f"({shape[1]}), which no leading dimension can express"
        )
    return int(ld)


def sgemm_params(
    a_shape,
    a_strides,
    b_shape,
    b_strides,
    out_shape,
    out_strides,
    itemsize,
    *,
    transpose_a,
    transpose_b,
    transpose_out,
    allow_double=False,
) -> SgemmParams:
    """Map ``matmul_step``'s row-major contraction onto one column-major Sgemm.
    The contraction is ``out_view = op(a) @ op(b)``; a row-major ``(r, c)``
    buffer with row stride ``ld``, read column-major with leading
    dimension ``ld``, is that buffer's transpose. So an untransposed
    destination asks cuBLAS for ``out.T = op(b).T @ op(a).T`` (operands
    swapped, flags inverted); ``transpose_out`` inverts that inversion
    back to natural order with opposite flags. Leading dimensions come
    from the strides (:func:`_leading_dimension`); refuses anything one
    Sgemm cannot address, or any itemsize that is not float32's."""
    if itemsize != GEMM_ITEMSIZE and not (allow_double and itemsize == 8):
        raise GemmMappingUnavailable(
            f"the captured GEMM step is {GEMM_DTYPE}-only (the "
            f"zero-FP64 discipline); got an itemsize of {itemsize} bytes"
        )
    lda_rows = _leading_dimension(a_shape, a_strides, itemsize, what="operand 'a'")
    ldb_rows = _leading_dimension(b_shape, b_strides, itemsize, what="operand 'b'")
    ldc_rows = _leading_dimension(out_shape, out_strides, itemsize, what="'out'")

    m, k_a = (a_shape[1], a_shape[0]) if transpose_a else (a_shape[0], a_shape[1])
    k_b, n = (b_shape[1], b_shape[0]) if transpose_b else (b_shape[0], b_shape[1])
    if k_a != k_b:
        raise GemmMappingUnavailable(
            f"contracted extents disagree: 'a' contributes {k_a} and 'b' "
            f"contributes {k_b} (shapes {tuple(a_shape)} / {tuple(b_shape)} "
            f"under transpose_a={transpose_a}, transpose_b={transpose_b})"
        )
    want_out = (n, m) if transpose_out else (m, n)
    if tuple(out_shape) != want_out:
        raise GemmMappingUnavailable(
            f"'out' must be {want_out} for a ({m}, {k_a}) @ ({k_a}, {n}) "
            f"contraction with transpose_out={transpose_out}; got "
            f"{tuple(out_shape)}"
        )

    if transpose_out:
        # The destination's column-major view is the result itself, so the
        # operands keep their order and each takes the flag the row-major form
        # would NOT have suggested.
        return SgemmParams(
            op_a="N" if transpose_a else "T",
            op_b="N" if transpose_b else "T",
            m=m,
            n=n,
            k=k_a,
            lda=lda_rows,
            ldb=ldb_rows,
            ldc=ldc_rows,
            operands=("a", "b"),
        )
    # The destination's column-major view is the result's TRANSPOSE, and
    # (X @ Y).T = Y.T @ X.T -- hence the swap.
    return SgemmParams(
        op_a="T" if transpose_b else "N",
        op_b="T" if transpose_a else "N",
        m=n,
        n=m,
        k=k_a,
        lda=ldb_rows,
        ldb=lda_rows,
        ldc=ldc_rows,
        operands=("b", "a"),
    )


def sgemm_params_for_arrays(
    a, b, out, *, transpose_a=False, transpose_b=False, transpose_out=False
) -> SgemmParams:
    """:func:`sgemm_params` read off three live arrays (numpy or cupy).

    Also enforces the dtype fence on all three, since here it can be seen.
    ``.strides`` is in BYTES for both array modules, which is what
    the pure function takes."""
    for name, arr in (("a", a), ("b", b), ("out", out)):
        if dtype_name(arr.dtype) != GEMM_DTYPE:
            raise GemmMappingUnavailable(
                f"the captured GEMM step is {GEMM_DTYPE}-only; "
                f"operand {name!r} is {dtype_name(arr.dtype)}"
            )
    itemsizes = {int(a.itemsize), int(b.itemsize), int(out.itemsize)}
    if len(itemsizes) != 1:
        raise GemmMappingUnavailable(f"operands disagree on itemsize: {itemsizes}")
    return sgemm_params(
        a.shape,
        a.strides,
        b.shape,
        b.strides,
        out.shape,
        out.strides,
        itemsizes.pop(),
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        transpose_out=transpose_out,
    )


# --------------------------------------------------------------------------- #
# Plan-level fences
# --------------------------------------------------------------------------- #
def require_float32(specs) -> None:
    """Refuse unless every :class:`~eagle.gemm.plan.WorkspaceSpec` in
    ``specs`` declares float32. Stated as a fence: ``lower_dense_gemm``'s
    own dtype default is still ``"float64"``, so a plan built without an
    explicit dtype would otherwise reach here declaring a type this step
    cannot run."""
    wrong = tuple(s.name for s in specs if s.dtype != GEMM_DTYPE)
    if wrong:
        raise GemmMappingUnavailable(
            f"the captured GEMM step is {GEMM_DTYPE}-only (the "
            f"zero-FP64 discipline); these buffers declare something else: "
            f"{sorted(wrong)}"
        )


def check_plan_capturable(plan, *, operand_class: str) -> None:
    """Refuse a lowered plan the captured raw-cuBLAS route cannot run.
    Two permanent fences: the ``"gemv"`` operand class has no mapping
    here (its lowering contracts against a ones vector), and every
    declared buffer must be float32."""
    if operand_class == "gemv":
        raise GemmMappingUnavailable(
            "the captured GEMM step does not cover the 'gemv' operand class -- "
            "its lowering contracts against a ones vector and its captured form "
            "is recorded, not funded. Leave this plan on the uncaptured path"
        )
    require_float32(plan.required)
