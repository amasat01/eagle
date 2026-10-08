# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The lowered plan — an ordered, backend-parametric sequence of GEMM steps.

:mod:`eagle.gemm.contraction` says what a reduce loop's record looks like
once recognized; this module is what a lowering actually is, given one. A
:class:`LoweredPlan` is a frozen tuple of named steps plus the metadata a
caller needs to run them: the mechanism row it came from, its determinism
typing, and the specs of every buffer it touches.

Three properties, each a deliberate exclusion:

* **backend-parametric.** Every step carries both a device form and a
  host twin, so a plan is checked against a numpy reference without a
  GPU;
* **it allocates nothing.** Every buffer is the caller's, bound by
  name; the plan publishes :class:`WorkspaceSpec` s and validates what
  it is handed against them. In v1 the allocator is the runner;
* **it does not schedule.** Steps run in list order, enqueued on the
  current stream and left there; the caller keeps control of ordering.

What a plan is NOT: a place where a contraction is decided. The steps
arrive already chosen. :func:`lower_dense_gemm` derives one from the
record alone, consuming it entirely by duck-typed attribute access, so
a record built by the code generator's ``recognize()``, or hand-built
to the same shape, drives this module identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "CapturedGemmCall",
    "LoweredPlan",
    "LoweringUnavailable",
    "MatmulSpec",
    "MatmulUnavailable",
    "PlanStep",
    "PlanUnavailable",
    "WorkspaceSpec",
    "assembly_step",
    "captured_gemm_calls",
    "generated_step",
    "lower_dense_gemm",
    "matmul_fallback_note",
    "matmul_step",
    "verify_staged_fanin",
]

#: The step kinds a plan may hold: generated (one of a producer's own
#: compiled kernels), matmul (the array module's ``matmul``), or
#: assembly (a caller-supplied elementwise rearrangement).
STEP_KINDS = ("generated", "matmul", "assembly")

#: The mechanism ``lower_dense_gemm`` always builds against, copied
#: verbatim from the code generator's mechanism evidence table.
_DENSE_GEMM_MECHANISM_NAME = "dense_gemm"
_DENSE_GEMM_DETERMINISM = (
    "gate-established on the pinned device (y/x̄/θ̄ bitwise "
    "identical across 20 repeated calls) — ESTABLISHED at the device "
    "gate (×20 single digest, schema wa4-1 era); a red on any future "
    "determinism gate demotes this row to non-deterministic typing. "
    "RE-ESTABLISHED PER REGIME at the capture close: a "
    "single ×20 digest in BOTH the uncaptured_purekernel and the "
    "captured_graph regime — the captured arm runs raw cuBLAS INSIDE the "
    "graph, so its determinism is its own claim and was measured as one"
)


# --------------------------------------------------------------------------- #
# The dtype-name cache behind ``WorkspaceSpec.check``: ``np.dtype(x).name``
# costs real microseconds, and ``check`` runs on every bound buffer of every
# plan call. Keyed by value (numpy dtypes hash/compare structurally), so the
# cache is bounded by the handful of distinct dtypes a process sees. An
# unhashable descriptor simply bypasses the cache.
# --------------------------------------------------------------------------- #
_DTYPE_NAME_CACHE: dict = {}


def _dtype_name(descriptor) -> str:
    """``np.dtype(descriptor).name``, memoized per distinct descriptor
    value. Byte-identical to the expression it replaces, including for
    non-dtype descriptors (``None``, a type object, a string)."""
    try:
        hit = _DTYPE_NAME_CACHE.get(descriptor)
    except TypeError:  # unhashable descriptor: no cache, original path
        return np.dtype(descriptor).name
    if hit is not None:
        return hit
    name = np.dtype(descriptor).name
    _DTYPE_NAME_CACHE[descriptor] = name
    return name


class LoweringUnavailable(Exception):
    """Base for "this lowering cannot be used" — raised at BUILD time
    (:class:`PlanUnavailable`) or at CALL time (:class:`MatmulUnavailable`)."""


class PlanUnavailable(LoweringUnavailable):
    """A plan cannot be built for this contraction: a permanent property
    of the record, not a retry target."""


class MatmulUnavailable(LoweringUnavailable):
    """A plan's matmul FAILED on the device at call time: the failure a
    mechanism-selection layer's ``cublas_available()`` check cannot
    predict (a cupy that imports cleanly but whose cuBLAS library is
    missing, mismatched, or cannot create a handle). The contract is
    that the caller maps it to an uncaptured fallback plus a loud note
    (:func:`matmul_fallback_note`), never a silent retry or a swallowed
    genuine shape/dtype bug."""


def matmul_fallback_note(error: BaseException, *, mechanism: str = "dense_gemm") -> str:
    """The note emitted when a plan's matmul fails and falls back,
    naming the abandoned mechanism and the underlying error."""
    return (
        f"eagle.gemm: the {mechanism!r} mechanism's matmul failed on this device "
        f"({type(error).__name__}: {error}); falling back to an uncaptured "
        "mechanism for this call. A cupy that imports cleanly and counts "
        "devices CANNOT be taken as proof cuBLAS itself is intact — if this "
        "repeats, the selection is wrong for this machine, not for this call."
    )


@dataclass(frozen=True)
class WorkspaceSpec:
    """One buffer a plan touches: its name, shape and dtype. The plan
    never allocates it -- this is the declaration the caller allocates
    against. ``dtype`` is a normalized name string, not a numpy dtype
    object, so the spec stays hashable and comparable."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self):
        object.__setattr__(self, "shape", tuple(int(d) for d in self.shape))
        object.__setattr__(self, "dtype", _dtype_name(self.dtype))
        if any(d < 0 for d in self.shape):
            raise PlanUnavailable(f"workspace {self.name!r}: negative extent in shape")

    def check(self, array) -> None:
        """Raise unless ``array`` matches this spec (shape and dtype): a
        plan that wrote through ``out=`` into a wrongly-typed buffer would
        either raise deep inside the array module or silently downcast.
        The dtype-name lookup goes through :func:`_dtype_name`'s
        per-descriptor memo, since this runs once per bound buffer per
        plan call."""
        shape = tuple(int(d) for d in getattr(array, "shape", ()))
        dtype = _dtype_name(getattr(array, "dtype", None))
        if shape != self.shape or dtype != self.dtype:
            raise PlanUnavailable(
                f"buffer {self.name!r} was bound as {dtype}{list(shape)}, but the "
                f"plan declares {self.dtype}{list(self.shape)}"
            )


@dataclass(frozen=True)
class MatmulSpec:
    """What a matmul step multiplies, by name, and which way round. A
    step's two closures already know this, which was enough while the
    only consumer was :meth:`LoweredPlan.__call__`. Recording a plan into
    a CUDA graph needs more: cupy refuses every cuBLAS call during stream
    capture, so the captured form is a raw ``Sgemm`` issued through
    :mod:`eagle.gemm.capture`, which needs the operand names, the three
    transpose flags and the destination -- the same description
    :func:`matmul_step` was built from. Published rather than re-derived
    from the record, since a plan is what actually runs: a second
    derivation would be a second thing to keep in step with the first."""

    a: str
    b: str
    out: str
    transpose_a: bool = False
    transpose_b: bool = False
    transpose_out: bool = False


@dataclass(frozen=True)
class PlanStep:
    """One step: what it is called, what kind it is, which buffers it
    reads and writes, and its two forms. ``device_fn``/``host_fn`` take
    the bound buffer mapping and return nothing. Two callables rather
    than one taking an array module, because a generated kernel's two
    forms are not the same function with a different ``xp`` (a
    producer's own dispatch picks the provider), while a matmul's are."""

    name: str
    kind: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    device_fn: object = field(compare=False)
    host_fn: object = field(compare=False)
    #: The step's own description when it is a matmul, ``None`` otherwise.
    #: Excluded from comparison/hashing, like the two callables beside it:
    #: plan equality is used to check a hand-built plan against a derived
    #: one.
    matmul: object = field(default=None, compare=False)

    def __post_init__(self):
        if self.kind not in STEP_KINDS:
            raise PlanUnavailable(
                f"step {self.name!r}: unknown kind {self.kind!r} (expected one of "
                f"{list(STEP_KINDS)})"
            )
        if not self.writes:
            raise PlanUnavailable(f"step {self.name!r} writes nothing")

    def run(self, buffers, backend: str) -> None:
        (self.device_fn if backend == "device" else self.host_fn)(buffers)


@dataclass(frozen=True)
class LoweredPlan:
    """An ordered sequence of steps plus everything needed to run them.
    ``inputs``/``outputs``/``workspaces`` are all :class:`WorkspaceSpec` s,
    uniform since the caller allocates and the plan validates all three
    the same way; the split is about role: inputs arrive filled, outputs
    leave filled, workspaces are intermediates a later step reads.
    Construction validates the read/write graph: a step may only read a
    name that is an input or was written by an earlier step, may only
    write a declared name, every output must be written, and every
    workspace must be both written and later read (one nothing reads
    would be over-specified)."""

    steps: tuple[PlanStep, ...]
    mechanism: str
    determinism: str
    inputs: tuple[WorkspaceSpec, ...] = ()
    outputs: tuple[WorkspaceSpec, ...] = ()
    workspaces: tuple[WorkspaceSpec, ...] = ()

    def __post_init__(self):
        if not self.steps:
            raise PlanUnavailable("a plan needs at least one step")
        specs = {}
        for spec in (*self.inputs, *self.outputs, *self.workspaces):
            if spec.name in specs:
                raise PlanUnavailable(f"buffer {spec.name!r} is declared twice")
            specs[spec.name] = spec

        available = {s.name for s in self.inputs}
        written: set[str] = set()
        read: set[str] = set()
        for step in self.steps:
            for name in step.reads:
                if name not in available:
                    raise PlanUnavailable(
                        f"step {step.name!r} reads {name!r}, which is neither a "
                        "declared input nor written by an earlier step"
                    )
                read.add(name)
            for name in step.writes:
                if name not in specs:
                    raise PlanUnavailable(
                        f"step {step.name!r} writes {name!r}, which the plan does "
                        "not declare (an undeclared buffer would have to be "
                        "allocated somewhere, and a plan allocates nothing)"
                    )
                available.add(name)
                written.add(name)

        for spec in self.outputs:
            if spec.name not in written:
                raise PlanUnavailable(f"output {spec.name!r} is never written")
        for spec in self.workspaces:
            if spec.name not in written:
                raise PlanUnavailable(
                    f"workspace {spec.name!r} is never written — it would be read "
                    "uninitialized"
                )
            if spec.name not in read:
                raise PlanUnavailable(
                    f"workspace {spec.name!r} is never read — the caller would "
                    "allocate memory the plan never uses (over-specified)"
                )

    @property
    def required(self) -> tuple[WorkspaceSpec, ...]:
        """Every buffer the caller must allocate and bind, in declaration
        order."""
        return (*self.inputs, *self.outputs, *self.workspaces)

    @property
    def kinds(self) -> tuple[str, ...]:
        return tuple(s.kind for s in self.steps)

    def __call__(self, backend: str, buffers: dict) -> dict:
        """Run every step in order and return ``{output name: buffer}``.
        ``backend`` is ``"host"`` (numpy, sequential) or ``"device"``
        (cupy, enqueued on the current stream and left there -- the
        caller's stream ordering is the only ordering). ``buffers`` must
        already hold every name in :attr:`required`, matching its spec:
        nothing is allocated here, and a missing or mis-shaped binding is
        refused before any step runs."""
        if backend not in ("host", "device"):
            raise PlanUnavailable(
                f"unknown backend {backend!r} (expected 'host' or 'device')"
            )
        for spec in self.required:
            if spec.name not in buffers:
                raise PlanUnavailable(
                    f"buffer {spec.name!r} was not bound; this plan needs "
                    f"{[s.name for s in self.required]} — the caller allocates, "
                    "the plan never does"
                )
            spec.check(buffers[spec.name])
        for step in self.steps:
            step.run(buffers, backend)
        return {spec.name: buffers[spec.name] for spec in self.outputs}


# --------------------------------------------------------------------------- #
# Step factories — each builds BOTH forms from one description
# --------------------------------------------------------------------------- #
def _cupy():
    try:
        import cupy
    except Exception as exc:  # pragma: no cover - exercised by the poisoned test
        raise MatmulUnavailable(f"cupy is not importable: {exc}") from exc
    return cupy


def matmul_step(
    name, *, a, b, out, transpose_a=False, transpose_b=False, transpose_out=False
) -> PlanStep:
    """``out = op(a) @ op(b)``, where ``op`` is a transpose when asked.

    A transpose is a view in both array modules, so an operand
    orientation costs nothing; the product is written through ``out=``
    the same way. ``transpose_out`` writes through a transposed view of
    the destination, which is how a gradient lands in its operand's own
    declared layout when that layout puts the contracted axis first. The
    device form re-raises :class:`MatmulUnavailable` on every call, since
    a broken cuBLAS shows up on the first one."""
    spec = MatmulSpec(
        a=a,
        b=b,
        out=out,
        transpose_a=bool(transpose_a),
        transpose_b=bool(transpose_b),
        transpose_out=bool(transpose_out),
    )

    def run(buffers, xp):
        left = buffers[a]
        right = buffers[b]
        target = buffers[out]
        xp.matmul(
            left.T if transpose_a else left,
            right.T if transpose_b else right,
            out=target.T if transpose_out else target,
        )

    def device_fn(buffers):
        xp = _cupy()
        try:
            run(buffers, xp)
        except MatmulUnavailable:
            raise
        except Exception as exc:
            raise MatmulUnavailable(
                f"matmul step {name!r} failed on the device: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    return PlanStep(
        name=name,
        kind="matmul",
        reads=(a, b),
        writes=(out,),
        device_fn=device_fn,
        host_fn=lambda buffers: run(buffers, np),
        matmul=spec,
    )


def assembly_step(name, *, reads, writes, fn) -> PlanStep:
    """A rearrangement written ONCE against an array module:
    ``fn(buffers, xp)``. Where a plan puts the reshape/scale/scatter-free
    rearrangement a matmul cannot express; writing it twice is how the
    two backends drift apart."""
    cupy_fn = None

    def device_fn(buffers):
        nonlocal cupy_fn
        if cupy_fn is None:
            cupy_fn = _cupy()
        fn(buffers, cupy_fn)

    return PlanStep(
        name=name,
        kind="assembly",
        reads=tuple(reads),
        writes=tuple(writes),
        device_fn=device_fn,
        host_fn=lambda buffers: fn(buffers, np),
    )


def generated_step(name, *, kernel, reads, writes, bind) -> PlanStep:
    """A launch of one of a producer's own compiled kernels.

    ``bind(buffers) -> kwargs`` maps the plan's names onto the kernel's
    declared parameters. One form here, not two: a producer's kernels
    already dispatch host-or-device off their inputs, so the twin a plan
    needs is the one the kernel already has."""

    def run(buffers):
        result = kernel(**bind(buffers))
        for out in writes:
            if isinstance(result, dict) and out in result:
                buffers[out] = result[out]

    return PlanStep(
        name=name,
        kind="generated",
        reads=tuple(reads),
        writes=tuple(writes),
        device_fn=run,
        host_fn=run,
    )


# --------------------------------------------------------------------------- #
# Record -> plan derivation
# --------------------------------------------------------------------------- #
def verify_staged_fanin(record, *, contracted_extent: int, operand=None) -> None:
    """Check a record's staged fan-in against the extent the adjoint
    GEMM contracts -- a check, never a scale factor. The banded adjoint
    accumulates ``fanin`` contributions into each staged slot, one per
    lane that read it. The GEMM adjoint ``Ā = Ȳ @ B`` performs that same
    sum in one contraction: the axis it sums over is Ȳ's free axis, whose
    extent is the other operand's free extent -- exactly the number of
    lanes sharing a slot. So both routes sum the same contributions and
    no multiplicative correction applies anywhere.

    If the declared fan-in disagrees with the contracted extent, the GEMM
    would sum a different set than the banded route accumulates,
    surfacing as an unlocalizable twin mismatch. ``None`` (no fan-in
    derived) is not a disagreement and passes. ``operand`` narrows the
    check to one staged input, needed when two operands are staged: each
    one's adjoint contracts the OTHER's free extent, so a single shared
    number would be wrong for one of them."""
    for name, fanin in record.staged_fanin:
        if fanin is None or (operand is not None and name != operand):
            continue
        if fanin != contracted_extent:
            raise PlanUnavailable(
                f"staged input {name!r} declares a fan-in of {fanin}, but the "
                f"adjoint GEMM contracts an extent of {contracted_extent}: the "
                "lowering would sum a different set of contributions than the "
                "generated route accumulates"
            )


#: The term forms a matmul over the declared buffers reproduces, per
#: class. ``"transformed"`` appears in neither: its value needs an
#: elementwise pre-step whose content lives only in the trace.
_TERM_FORM_FOR_CLASS = {"gemm": "product_of_reads", "gemv": "bare_read"}


@dataclass(frozen=True)
class _Operand:
    """One operand resolved to a MATRIX: the buffer's declared shape, which axis
    is contracted, and the extent of the one that is not."""

    name: str
    shape: tuple[int, ...]
    reduce_axis: int
    reduce_extent: int
    free_extent: int
    free_dim: str


def _resolve_operand(op, reduce_extent: int) -> _Operand:
    """Turn a record's operand description into a matrix, or refuse.
    Declaration-derived: the declared dims are the buffer's row-major
    shape, and ``reduce_dim`` says which axis the loop walks. Exactly two
    dims -- a third free axis is an einsum, excluded from v1 and refused
    by name rather than silently flattened."""
    if len(op.dims) != 2:
        raise PlanUnavailable(
            f"operand {op.name!r} declares {len(op.dims)} axes; v1 lowers a "
            "two-axis operand (one contracted, one free) — a third axis is an "
            "einsum, which this lowering does not do"
        )
    if op.reduce_dim is None:
        raise PlanUnavailable(
            f"operand {op.name!r}: the record names no declared axis for the "
            "reduce stride, so which axis to contract is not derivable"
        )
    labels = [d for d, _size, _stride in op.dims]
    axis = labels.index(op.reduce_dim)
    sizes = tuple(size for _d, size, _stride in op.dims)
    if sizes[axis] != reduce_extent:
        raise PlanUnavailable(
            f"operand {op.name!r}: its {op.reduce_dim!r} axis has extent "
            f"{sizes[axis]}, but the loop runs {reduce_extent} iterations — the "
            "matmul would contract a different range than the loop does"
        )
    free = 1 - axis
    return _Operand(
        name=op.name,
        shape=sizes,
        reduce_axis=axis,
        reduce_extent=sizes[axis],
        free_extent=sizes[free],
        free_dim=labels[free],
    )


def _check_lowerable(record, direction: str) -> None:
    """The gates every dense lowering passes, in the order a reader would ask
    them: is there a record, is it dense, is its term form one a matmul
    reproduces, and is the reduce a full contiguous sweep."""
    if record is None:
        raise PlanUnavailable("nothing was recognized, so there is nothing to lower")
    if direction not in ("forward", "adjoint"):
        raise PlanUnavailable(
            f"unknown direction {direction!r} (expected 'forward' or 'adjoint')"
        )
    if not record.dense:
        raise PlanUnavailable(
            "the contraction is not dense, so its operands cannot be walked as "
            "matrices: " + "; ".join(record.refusals or ("no reason recorded",))
        )
    wanted = _TERM_FORM_FOR_CLASS.get(record.operand_class)
    if record.term_form != wanted:
        raise PlanUnavailable(
            f"the contraction's term form is {record.term_form!r}; a "
            f"{record.operand_class} lowering reproduces {wanted!r} only. A "
            "transformed term needs an elementwise pre-step whose content lives "
            "in the trace, and this seam carries no trace handle — the caller "
            "should route it to an uncaptured mechanism instead"
        )
    if record.reduce_start != 0 or record.reduce_step != 1:
        raise PlanUnavailable(
            f"the reduce runs range({record.reduce_start}, {record.reduce_extent}, "
            f"{record.reduce_step}); a matmul contracts a full unit-step axis"
        )
    if record.reduce_extent is None:
        raise PlanUnavailable("the reduce extent is not a compile-time constant")


def _plan_metadata(record):
    return _DENSE_GEMM_MECHANISM_NAME, _DENSE_GEMM_DETERMINISM


def _gemm_plan(record, direction, dtype):
    """``C = A' @ B'ᵀ`` and its two adjoints, with every orientation
    read off the operands' declared axes. ``A'`` is the operand with its
    contracted axis last; when the declaration put it first, the primed
    form is a transposed view (free). The adjoint results are written
    back in each operand's own declared layout, via a transposed ``out``
    view when needed, so the caller binds gradient buffers shaped like
    the operands they belong to."""
    a, b = (_resolve_operand(op, record.reduce_extent) for op in record.operands)
    out_name = record.outputs[0][0]
    mechanism, determinism = _plan_metadata(record)

    # The adjoint of each operand contracts the OTHER's free extent, which is
    # exactly the number of lanes sharing one staged slot.
    verify_staged_fanin(record, contracted_extent=b.free_extent, operand=a.name)
    verify_staged_fanin(record, contracted_extent=a.free_extent, operand=b.name)

    c_shape = (a.free_extent, b.free_extent)
    if direction == "forward":
        return LoweredPlan(
            steps=(
                matmul_step(
                    out_name,
                    a=a.name,
                    b=b.name,
                    out=out_name,
                    transpose_a=(a.reduce_axis == 0),
                    transpose_b=(b.reduce_axis == 1),
                ),
            ),
            mechanism=mechanism,
            determinism=determinism,
            inputs=(
                WorkspaceSpec(a.name, a.shape, dtype),
                WorkspaceSpec(b.name, b.shape, dtype),
            ),
            outputs=(WorkspaceSpec(out_name, c_shape, dtype),),
        )

    bar = f"{out_name}_bar"
    # A_bar' = C_bar @ B' (free_a, free_b) @ (free_b, K)
    # B_bar' = C_barᵀ @ A' (free_b, free_a) @ (free_a, K)
    return LoweredPlan(
        steps=(
            matmul_step(
                f"{a.name}_bar",
                a=bar,
                b=b.name,
                out=f"{a.name}_bar",
                transpose_b=(b.reduce_axis == 0),
                transpose_out=(a.reduce_axis == 0),
            ),
            matmul_step(
                f"{b.name}_bar",
                a=bar,
                b=a.name,
                out=f"{b.name}_bar",
                transpose_a=True,
                transpose_b=(a.reduce_axis == 0),
                transpose_out=(b.reduce_axis == 0),
            ),
        ),
        mechanism=mechanism,
        determinism=determinism,
        inputs=(
            WorkspaceSpec(bar, c_shape, dtype),
            WorkspaceSpec(a.name, a.shape, dtype),
            WorkspaceSpec(b.name, b.shape, dtype),
        ),
        outputs=(
            WorkspaceSpec(f"{a.name}_bar", a.shape, dtype),
            WorkspaceSpec(f"{b.name}_bar", b.shape, dtype),
        ),
    )


def _gemv_plan(record, direction, dtype):
    """The row sum and its broadcast adjoint, both as matmuls against a
    ones vector -- the shipped ``_dx_bar_reduce`` shape. The ones vector
    is a declared input rather than built here, since the plan allocates
    nothing; the caller fills it once and reuses it."""
    (x,) = (_resolve_operand(op, record.reduce_extent) for op in record.operands)
    out_name = record.outputs[0][0]
    mechanism, determinism = _plan_metadata(record)
    ones = f"{x.name}_ones"
    y_shape = (x.free_extent, 1)

    if direction == "forward":
        return LoweredPlan(
            steps=(
                matmul_step(
                    out_name,
                    a=x.name,
                    b=ones,
                    out=out_name,
                    transpose_a=(x.reduce_axis == 0),
                ),
            ),
            mechanism=mechanism,
            determinism=determinism,
            inputs=(
                WorkspaceSpec(x.name, x.shape, dtype),
                WorkspaceSpec(ones, (x.reduce_extent, 1), dtype),
            ),
            outputs=(WorkspaceSpec(out_name, y_shape, dtype),),
        )

    return LoweredPlan(
        steps=(
            matmul_step(
                f"{x.name}_bar",
                a=f"{out_name}_bar",
                b=ones,
                out=f"{x.name}_bar",
                transpose_b=True,
                transpose_out=(x.reduce_axis == 0),
            ),
        ),
        mechanism=mechanism,
        determinism=determinism,
        inputs=(
            WorkspaceSpec(f"{out_name}_bar", y_shape, dtype),
            WorkspaceSpec(ones, (x.reduce_extent, 1), dtype),
        ),
        outputs=(WorkspaceSpec(f"{x.name}_bar", x.shape, dtype),),
    )


def lower_dense_gemm(record, *, direction="forward", dtype="float64") -> LoweredPlan:
    """Build the dense-gemm plan for a recognized contraction, from the
    record alone. ``direction`` picks the half: ``"forward"`` is the
    contraction itself, ``"adjoint"`` its gradients -- two plans rather
    than one, since the adjoint's input (the output's cotangent) does not
    exist when the forward runs.

    Everything is declaration-derived: each operand's declared dims give
    its matrix, ``reduce_dim`` gives the contracted axis, the free axis
    gives the free extent, and the GEMM adjoints follow from those
    shapes. ``record`` is consumed by duck-typed attribute access -- see
    :mod:`eagle.gemm.contraction`'s docstring for why this module never
    needs to know how the record was produced. NOT synthesized: an output
    assembly step. The plan's output is the contracted result with its
    axes in operand order; how that maps onto the flat per-sample buffer
    the generated kernel writes depends on the launch domain's lane
    decomposition, which the record does not carry, so v1 hands back the
    labelled result and leaves the flattening to the caller."""
    _check_lowerable(record, direction)
    build = _gemm_plan if record.operand_class == "gemm" else _gemv_plan
    return build(record, direction, dtype)


# --------------------------------------------------------------------------- #
# The captured (raw-cuBLAS) view of a plan.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CapturedGemmCall:
    """One plan step, described as the raw ``Sgemm`` a captured graph
    records. ``operands`` maps ``"a"``/``"b"``/``"out"`` to the caller's
    own buffers -- never copies, since a captured graph records device
    pointers. Which of ``"a"``/``"b"`` is the first cuBLAS operand is
    ``params.operands``, not this mapping."""

    name: str
    params: object
    operands: dict


def captured_gemm_calls(plan, buffers, *, operand_class: str) -> tuple:
    """Describe every step of ``plan`` as a capture-legal raw ``Sgemm``.
    The seam a captured build reads a lowered plan through. Runs nothing
    and allocates nothing: it maps each step's published
    :class:`MatmulSpec` and the caller's bound buffers onto
    :class:`~eagle.gemm.capture.SgemmParams`, a pure function over shapes
    and strides, and hands the results back for the caller to issue
    inside its own capture region.

    Three refusals (:class:`~eagle.gemm.capture.GemmMappingUnavailable`
    throughout): ``operand_class == "gemv"`` or any non-float32 buffer; a
    step that is not a matmul (skipping it would replay a different
    computation than the plan describes); a name the caller did not
    bind. Leading dimensions come from each buffer's strides, so a plan
    bound to slice views of larger allocations maps correctly."""
    from .capture import (
        GemmMappingUnavailable,
        check_plan_capturable,
        sgemm_params_for_arrays,
    )

    check_plan_capturable(plan, operand_class=operand_class)
    calls = []
    for step in plan.steps:
        spec = step.matmul
        if spec is None:
            raise GemmMappingUnavailable(
                f"step {step.name!r} is a {step.kind!r} step, not a matmul: a "
                "captured raw-cuBLAS build can issue an Sgemm and nothing else, "
                "and a step it silently skipped would make the replayed graph a "
                "different computation than this plan"
            )
        operands = {}
        for role, bound in (("a", spec.a), ("b", spec.b), ("out", spec.out)):
            if bound not in buffers:
                raise GemmMappingUnavailable(
                    f"step {step.name!r} reads or writes {bound!r}, which the "
                    f"caller did not bind; this plan needs "
                    f"{[s.name for s in plan.required]}"
                )
            operands[role] = buffers[bound]
        params = sgemm_params_for_arrays(
            operands["a"],
            operands["b"],
            operands["out"],
            transpose_a=spec.transpose_a,
            transpose_b=spec.transpose_b,
            transpose_out=spec.transpose_out,
        )
        calls.append(CapturedGemmCall(step.name, params, operands))
    return tuple(calls)
