# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A capture-legal single-precision GEMM step: raw cuBLAS, bound by ctypes.

Pure ``ctypes`` plus a strides-only shape mapping, with zero
code-generator imports -- this is eagle runtime, not codegen.
Cross-module references point at :mod:`eagle.gemm.plan`.

:mod:`eagle.gemm.plan`'s ``matmul_step`` calls the array module's
``matmul``, which cupy refuses to record into a CUDA graph during
stream capture. This module is the direct cuBLAS call that bypasses
that restriction: it never selects a mechanism, never allocates, and is
not wired into ``matmul_step``, which remains the uncaptured
implementation.

Three pieces: :func:`load_cublas` resolves the shared library from the
process's existing mappings; :class:`CublasCaptureHandle` is one handle
per captured-graph build; :func:`sgemm_params` is the row-major to
column-major mapping, a pure function over shapes and strides.

Refusals are typed and loud (:class:`GemmCaptureUnavailable` and its
two subclasses), never fallbacks.
"""

from __future__ import annotations

import ctypes
import re
from typing import NamedTuple

__all__ = [
    "CUBLAS_OP",
    "CublasCaptureHandle",
    "CublasResolutionError",
    "GemmCaptureUnavailable",
    "GemmMappingUnavailable",
    "SgemmParams",
    "check_plan_capturable",
    "load_cublas",
    "mapped_library_paths",
    "require_float32",
    "sgemm_params",
    "sgemm_params_for_arrays",
]

#: The libraries this module binds. Version-pinned deliberately: the CUDA 12
#: and CUDA 13 cuBLAS are the ones the supported stacks map, and a soname
#: change is a fact a reader should have to see, not a silently widened match.
CUBLAS_SONAME = "libcublas.so.12"
CUBLAS_SONAMES = (CUBLAS_SONAME, "libcublas.so.13")

#: cuBLAS' own transpose enum, by the letters this module's mapping speaks.
CUBLAS_OP = {"N": 0, "T": 1}

#: Sgemm is single precision by definition; the zero-FP64 discipline makes
#: that a fence: a float64 operand is refused here even though
#: ``lower_dense_gemm``'s own dtype default is still ``"float64"``.
GEMM_DTYPE = "float32"
GEMM_ITEMSIZE = 4


class GemmCaptureUnavailable(Exception):
    """Base for "this GEMM cannot be run as a captured raw-cuBLAS step"."""


class CublasResolutionError(GemmCaptureUnavailable):
    """The cuBLAS library could not be resolved unambiguously. Zero
    matches means cupy's own cuBLAS never initialized; more than one means
    two different ``libcublas`` files are mapped, and picking either would
    risk a wrong answer or a crash far from here."""


class GemmMappingUnavailable(GemmCaptureUnavailable):
    """This contraction cannot be expressed as one Sgemm call: a shape,
    dtype, stride pattern or operand class the mapping does not cover.
    Permanent for the operands as bound -- leave the step uncaptured, not
    retry."""


# --------------------------------------------------------------------------- #
# Library resolution
# --------------------------------------------------------------------------- #
_MAPS_LINE = re.compile(r"\s(/\S+)$")

_loaded_lib = None


def mapped_library_paths(maps_text: str, soname=CUBLAS_SONAMES) -> tuple:
    """Every distinct file path in ``maps_text`` whose basename is
    ``soname`` (one name, or any of a tuple of names) itself or extended by a
    real-file suffix -- never a bare
    prefix match (conda's real file is further-versioned, e.g.
    ``libcublas.so.12.9.1.4``, and ``libcublasLt.so.12`` must not match).
    Pure function over a ``/proc/<pid>/maps`` dump: the refusals are
    provable without a GPU. Sorted and deduplicated."""
    found = []
    for line in maps_text.splitlines():
        match = _MAPS_LINE.search(line)
        if match is None:
            continue
        path = match.group(1)
        basename = path.rsplit("/", 1)[-1]
        names = (soname,) if isinstance(soname, str) else soname
        is_match = any(basename == s or basename.startswith(s + ".") for s in names)
        if is_match and path not in found:
            found.append(path)
    return tuple(sorted(found))


def _warm_cupy_cublas() -> None:
    """Force cupy to initialize its own cuBLAS, so the library is mapped.
    Touches the GPU, which is why nothing at import time calls it --
    resolution happens at graph-build time, on the machine with the
    device."""
    import cupy as cp

    a = cp.zeros((2, 2), dtype=cp.float32)
    cp.matmul(a, a)
    cp.cuda.runtime.deviceSynchronize()


def resolve_cublas_path() -> str:
    """The single ``libcublas`` file this process has mapped. Warms
    cupy's cuBLAS first, then reads its own mappings. Refuses on zero or
    more than one distinct path (see :class:`CublasResolutionError`)."""
    _warm_cupy_cublas()
    with open("/proc/self/maps") as handle:
        maps_text = handle.read()
    paths = mapped_library_paths(maps_text)
    if not paths:
        raise CublasResolutionError(
            f"no {' or '.join(CUBLAS_SONAMES)} is mapped into this process even after warming "
            "cupy's cuBLAS -- the captured GEMM step binds the library the "
            "process ALREADY uses, so there is nothing here to bind"
        )
    if len(paths) > 1:
        raise CublasResolutionError(
            f"{len(paths)} different libcublas files are mapped into this "
            f"process: {list(paths)}. Binding one of them would be a guess, and "
            "a mismatched pair fails inside a later library call rather than "
            "here -- run in an environment with exactly one"
        )
    return paths[0]


def load_cublas():
    """The ``ctypes.CDLL`` for :func:`resolve_cublas_path`'s library, cached
    per process (the handle-per-graph rule is about the cuBLAS handle, not
    this library object)."""
    global _loaded_lib
    if _loaded_lib is None:
        lib = ctypes.CDLL(resolve_cublas_path())
        _bind_signatures(lib)
        _loaded_lib = lib
    return _loaded_lib


def _bind_signatures(lib) -> None:
    """Declare the four entry points' ctypes signatures. Explicit
    ``argtypes`` rather than ctypes' default int coercion: a device
    pointer is 64-bit and would be silently truncated otherwise."""
    lib.cublasCreate_v2.restype = ctypes.c_int
    lib.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    lib.cublasDestroy_v2.restype = ctypes.c_int
    lib.cublasDestroy_v2.argtypes = [ctypes.c_void_p]
    lib.cublasSetStream_v2.restype = ctypes.c_int
    lib.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lib.cublasSgemm_v2.restype = ctypes.c_int
    lib.cublasSgemm_v2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),  # alpha (host pointer mode)
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),  # beta (host pointer mode)
        ctypes.c_void_p,
        ctypes.c_int,
    ]


def _check(status: int, what: str) -> None:
    """Raise unless a cuBLAS call returned ``CUBLAS_STATUS_SUCCESS``."""
    if status != 0:
        raise GemmCaptureUnavailable(f"{what} returned cuBLAS status {status}")


def _device_ptr(operand) -> int:
    """``operand``'s device address: a cupy array's ``.data.ptr``, or an
    address already given as an integer."""
    data = getattr(operand, "data", None)
    ptr = getattr(data, "ptr", None)
    return int(ptr if ptr is not None else operand)


# --------------------------------------------------------------------------- #
# The handle
# --------------------------------------------------------------------------- #
class CublasCaptureHandle:
    """One cuBLAS handle, owned by ONE captured-graph build. The
    lifetime rules are why this is an object, not a function:
    ``cublasSetStream`` is called once, in the constructor, outside the
    capture region (cupy itself sets the stream per call, which is why it
    blanket-refuses cuBLAS during capture); the handle outlives its graph
    (cuBLAS keeps a per-handle workspace, so destroying it early is a
    use-after-free at replay); :meth:`warm` runs on the capture stream,
    after ``setStream`` and before ``begin_capture``, once per GEMM shape
    (its workspace allocates lazily, so warming elsewhere leaves a
    ``cudaMalloc`` inside the capture region); only :meth:`sgemm` runs
    inside the capture region; teardown order is graph, then handle, then
    pool. ``alpha``/``beta`` are host-side scalars baked into the
    recorded graph node at enqueue time: changing them means recording
    again."""

    def __init__(self, stream_ptr):
        self._lib = load_cublas()
        self._handle = ctypes.c_void_p()
        _check(self._lib.cublasCreate_v2(ctypes.byref(self._handle)), "cublasCreate")
        _check(
            self._lib.cublasSetStream_v2(
                self._handle, ctypes.c_void_p(int(stream_ptr))
            ),
            "cublasSetStream",
        )
        self._alpha = ctypes.c_float(1.0)
        self._beta = ctypes.c_float(0.0)
        self._warmed = set()

    @property
    def alive(self) -> bool:
        """Whether the handle is still created (``False`` after :meth:`destroy`)."""
        return self._handle is not None

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
        if self._handle is None:
            raise GemmCaptureUnavailable(
                "this cuBLAS handle was destroyed; a handle outlives its graph "
                "and is never reused across builds"
            )
        first, second = params.operands
        _check(
            self._lib.cublasSgemm_v2(
                self._handle,
                CUBLAS_OP[params.op_a],
                CUBLAS_OP[params.op_b],
                params.m,
                params.n,
                params.k,
                ctypes.byref(self._alpha),
                ctypes.c_void_p(_device_ptr(operands[first])),
                params.lda,
                ctypes.c_void_p(_device_ptr(operands[second])),
                params.ldb,
                ctypes.byref(self._beta),
                ctypes.c_void_p(_device_ptr(operands["out"])),
                params.ldc,
            ),
            "cublasSgemm",
        )

    def destroy(self) -> None:
        """Destroy the handle. Call after destroying the graph that
        recorded work against it, and before freeing the memory pool.
        Idempotent."""
        if self._handle is not None:
            _check(self._lib.cublasDestroy_v2(self._handle), "cublasDestroy")
            self._handle = None


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
    if itemsize != GEMM_ITEMSIZE:
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
    import numpy as np

    for name, arr in (("a", a), ("b", b), ("out", out)):
        if np.dtype(arr.dtype).name != GEMM_DTYPE:
            raise GemmMappingUnavailable(
                f"the captured GEMM step is {GEMM_DTYPE}-only; "
                f"operand {name!r} is {np.dtype(arr.dtype).name}"
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
