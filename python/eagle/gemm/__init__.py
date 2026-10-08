# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.gemm`` — the raw-cuBLAS capture runtime.

GEMM/cuBLAS capture handles are eagle runtime, not the code generator's;
downstream layers compose over both. This package is that boundary,
executed: it moves the capture-facing GEMM/cuBLAS machinery OUT of the code
generator (a producer / codegen repo) and INTO eagle (the execution layer
that already owns kernel launch and CUDA-graph capture). ``import eagle``
never imports a producer (see ``eagle/__init__.py``'s own module
docstring); this package is the concrete case that law had to eventually
meet — a piece of RUNTIME machinery that used to live on the producer side
of that line because nothing else had a stream-capture story yet.

Two modules move whole:

* :mod:`eagle.gemm.capture` — :class:`~eagle.gemm.capture.CublasCaptureHandle`,
  the raw ``ctypes`` binding to ``libcublas.so.12`` (or ``.13``) that legally issues an
  ``Sgemm`` inside a CUDA-graph stream capture where cupy >=14.1.1 refuses
  (cupy's own ``cublas.setStream`` wrapper sets the stream on every call, which
  it must not do inside a capture region). Ported VERBATIM — this module
  already had zero producer imports; it is pure ``ctypes`` + a strides-only
  shape mapping, which is exactly why it was portable in the first place.
* :mod:`eagle.gemm.plan` — the BACKEND-PARAMETRIC step/plan machinery
  (:class:`~eagle.gemm.plan.LoweredPlan`, :func:`~eagle.gemm.plan.matmul_step`,
  :func:`~eagle.gemm.plan.lower_dense_gemm`,
  :func:`~eagle.gemm.plan.captured_gemm_calls`) a GEMM-lowered contraction
  runs through, PLUS the capture-legal view of it
  (:func:`~eagle.gemm.plan.captured_gemm_calls`) that reads a plan's published
  :class:`~eagle.gemm.plan.MatmulSpec` and produces the
  :class:`~eagle.gemm.capture.SgemmParams` a captured build issues. Ported
  near-verbatim: its only producer dependency was two constant strings off
  the code generator's mechanism EVIDENCE table (``DENSE_GEMM.name`` /
  ``.determinism``), copied here as literals rather than the table itself —
  see :mod:`eagle.gemm.plan`'s own docstring for why the table stays on
  the code generator's side of the line.

One module does NOT move, and the boundary is worth stating plainly rather
than glossing over: :mod:`eagle.gemm.contraction` carries only the RECORD
SHAPE (:class:`~eagle.gemm.contraction.ContractionOperand` /
:class:`~eagle.gemm.contraction.RecognizedContraction`) — pure, dependency-
free dataclasses. The MATCHER, the code generator's ``recognize`` function,
stays there: it walks its own pre-codegen trace IR (several thousand
lines of tracer/DSL machinery), which is CODEGEN-time analysis over a
kernel's trace, not EAGLE runtime. Porting ``recognize()``
here would mean eagle importing the code generator's frontend IR wholesale,
breaking the
"eagle never imports a producer" law this package exists to uphold in the
first place. Downstream `recognize` call sites are therefore NOT switched by
this move; everything downstream of a
recognized record (``lower_dense_gemm``, ``captured_gemm_calls``) IS switched,
because a caller holding a record — from the code generator's recognizer
today, from any other producer tomorrow — never needed it for anything past
that point.
"""

from __future__ import annotations

from .capture import (
    CUBLAS_OP,
    CublasCaptureHandle,
    CublasResolutionError,
    GemmCaptureUnavailable,
    GemmMappingUnavailable,
    SgemmParams,
    check_plan_capturable,
    load_cublas,
    mapped_library_paths,
    require_float32,
    sgemm_params,
    sgemm_params_for_arrays,
)
from .contraction import ContractionOperand, RecognizedContraction
from .plan import (
    CapturedGemmCall,
    LoweredPlan,
    LoweringUnavailable,
    MatmulSpec,
    MatmulUnavailable,
    PlanStep,
    PlanUnavailable,
    WorkspaceSpec,
    assembly_step,
    captured_gemm_calls,
    generated_step,
    lower_dense_gemm,
    matmul_fallback_note,
    matmul_step,
)

__all__ = [
    "CUBLAS_OP",
    "CapturedGemmCall",
    "ContractionOperand",
    "CublasCaptureHandle",
    "CublasResolutionError",
    "GemmCaptureUnavailable",
    "GemmMappingUnavailable",
    "LoweredPlan",
    "LoweringUnavailable",
    "MatmulSpec",
    "MatmulUnavailable",
    "PlanStep",
    "PlanUnavailable",
    "RecognizedContraction",
    "SgemmParams",
    "WorkspaceSpec",
    "assembly_step",
    "captured_gemm_calls",
    "check_plan_capturable",
    "generated_step",
    "load_cublas",
    "lower_dense_gemm",
    "mapped_library_paths",
    "matmul_fallback_note",
    "matmul_step",
    "require_float32",
    "sgemm_params",
    "sgemm_params_for_arrays",
]
