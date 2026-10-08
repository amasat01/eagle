# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The record shape a recognized contraction hands to :mod:`eagle.gemm.plan`
— not the recognizer itself.

``recognize`` lives in the code generator (a structural walk over its trace
IR); eagle never imports a producer, so this module owns only the shape of
the result: :class:`ContractionOperand` and :class:`RecognizedContraction`,
pure frozen dataclasses with no imports beyond ``dataclasses``. A caller
that already has a recognized-contraction result can pass it straight into
:func:`eagle.gemm.plan.lower_dense_gemm`; a future producer with its own
recognizer can construct a :class:`RecognizedContraction` here directly —
neither path depends on the code generator.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ContractionOperand",
    "RecognizedContraction",
]


@dataclass(frozen=True)
class ContractionOperand:
    """One buffer a recognized contraction reads, described from its declaration.

    ``dims`` is the declared axis layout ``((label, size, stride), …)`` in
    declared order, ``()`` for a positional table with no declared axes.
    ``reduce_stride`` is how far one step of the reduce axis moves this
    operand's flat index: ``0`` if invariant along it, ``None`` if the index
    is not affine in the loop variable at all. ``reduce_dim`` names the
    declared axis that stride belongs to, when it is exactly one
    declaration's named constant.

    ``dense`` is this operand's half of the contraction's density verdict;
    ``refusals`` is why not, one sentence per reason."""

    name: str
    role: str  # "staged" | "named" | "flat"
    dims: tuple[tuple[str, int, int], ...]
    bound: int
    reads: int
    reduce_stride: int | None
    reduce_dim: str | None
    stride_labels: tuple[str, ...]
    dense: bool
    refusals: tuple[str, ...]


@dataclass(frozen=True)
class RecognizedContraction:
    """A reduce loop described as a contraction — the whole seam, frozen.

    ``operand_class`` is ``"gemv"`` for a single-operand row reduction or
    ``"gemm"`` for two operands multiplied along the reduce axis. ``dense``
    is the AND of every operand's verdict; ``refusals`` collects their
    reasons, each prefixed with the operand it came from.

    ``term_form`` is the value property beside ``dense``'s index property:
    ``"bare_read"``, ``"product_of_reads"`` or ``"transformed"``. Both are
    needed to lower: density says the operands can be walked as matrices,
    term form says a matmul over those matrices computes what the loop
    computes.

    ``staged_fanin`` is how many lanes read one staged slot — the number of
    cotangent contributions the reverse pass accumulates into it — carried
    as declared, with ``None`` where the trace derives none. ``fanin_source``
    records where it was read from. ``adjoint_accumulates`` is
    ``((name, atomic?), …)`` for the shared accumulate outputs the derived
    kernel contributes to."""

    carry: str
    reduce_var: str
    reduce_extent: int | None
    reduce_start: int | None
    reduce_step: int
    operands: tuple[ContractionOperand, ...]
    operand_class: str
    term_form: str
    outputs: tuple[tuple[str, str], ...]
    dense: bool
    refusals: tuple[str, ...]
    staged_fanin: tuple[tuple[str, int | None], ...]
    fanin_source: str
    adjoint_accumulates: tuple[tuple[str, bool], ...]

    @property
    def operand_count(self) -> int:
        """How many distinct buffers the addend reads (1 or 2 — see
        ``operand_class``)."""
        return len(self.operands)

    @property
    def operand_names(self) -> tuple[str, ...]:
        """The read operands' declared names, in first-read order."""
        return tuple(op.name for op in self.operands)
