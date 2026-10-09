# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Launch-layer primitives shared by every kernel launcher.

The single capturable launch verb (:func:`launch`, kind-dispatched over the
kernel ABIs), the kernel-argument assembler, and the pure allocate-and-launch
body, imported by both the framework-polymorphic ``__call__`` launchers and
the PTX-loaded ``Loaded*`` ones so the launcher families can never drift.
Nothing here allocates or synchronizes, so every launch is safe to record
inside a CUDA-graph stream capture.

Also home to :class:`LaunchMixin`, the framework-polymorphic launch
skeleton (``origin -> launch_context -> sync-if-blocking``) every launcher
inherits.
"""

from __future__ import annotations

import threading

import numpy as np
from raptor.schema.manifest import KERNEL_NAME

from ._launch_policy import current_siblings, resolve_block
from .abi import DEVICE_CUDA, GREF_DTYPE, HANDLE_DTYPE, make_gref, make_handle
from .marshal import (
    _is_array_value,
    _reconcile_n,
    coerce_mat_inputs,
    coerce_mutable,
    coerce_per_sample,
    coerce_tables,
    coerce_terminated,
    coerce_uniforms,
    coerce_vec_inputs,
    coerce_wide_inputs,
    coerce_wide_outputs,
    fill_mutable,
    require_n,
)
from .roles import classify_arg

# KERNEL_NAME: the entry-point symbol of a generated vector / pure kernel
# (the schema's own, ``raptor.schema.manifest.KERNEL_NAME``).
#: default launch block size
DEFAULT_BLOCK = 256


# -- block-size resolution ---------------------------------------------------
#: process-wide device-properties cache (queried once per process).
_device_props_cache = None
#: per-kernel-object ``.attributes`` cache (captured at first launch, never
#: re-queried -- replay-critical paths must not re-touch the driver).
_kernel_attrs_cache: dict = {}
#: Serialises the check-cap-insert of :data:`_kernel_attrs_cache` (its lookup
#: fast path stays lock-free: a dict read is atomic and entries never change).
_KERNEL_ATTRS_LOCK = threading.Lock()

#: The hard cap on :data:`_kernel_attrs_cache`. Never evicts (an eviction
#: would re-query the driver on a replay-critical path), so growth past the
#: cap is refused instead -- a bug report about kernel churn, not a silent
#: latency spike. Deliberately generous: a real deployment binds tens to
#: hundreds of distinct kernel objects.
KERNEL_ATTRS_CACHE_CAP = 4096


class KernelAttrsCacheFull(RuntimeError):
    """Raised when a NEW kernel would push :data:`_kernel_attrs_cache` past
    :data:`KERNEL_ATTRS_CACHE_CAP`; already-cached kernels keep launching."""


def _reset_kernel_attrs_cache() -> None:
    """Empty the per-kernel ``.attributes`` cache (tests / teardown only)."""
    with _KERNEL_ATTRS_LOCK:
        _kernel_attrs_cache.clear()


def _device_props():
    """The current device's properties, queried once per process and
    cached thereafter."""
    global _device_props_cache
    if _device_props_cache is None:
        import cupy as cp

        try:
            _device_props_cache = cp.cuda.runtime.getDeviceProperties(
                cp.cuda.runtime.getDevice()
            )
        except Exception:
            _device_props_cache = {}
    return _device_props_cache


def _kernel_attrs(fn):
    """``fn``'s ``.attributes`` dict, cached per kernel object at its first
    launch so a replayed graph never re-queries it. Capped at
    :data:`KERNEL_ATTRS_CACHE_CAP`, refusing growth past it
    (:class:`KernelAttrsCacheFull`) rather than evicting."""
    cached = _kernel_attrs_cache.get(fn)
    if cached is not None:
        return cached
    with _KERNEL_ATTRS_LOCK:
        cached = _kernel_attrs_cache.get(fn)
        if cached is not None:
            return cached
        if len(_kernel_attrs_cache) >= KERNEL_ATTRS_CACHE_CAP:
            raise KernelAttrsCacheFull(
                f"eagle.launch._kernel_attrs_cache is full at "
                f"KERNEL_ATTRS_CACHE_CAP={KERNEL_ATTRS_CACHE_CAP} entries and "
                f"refuses to grow (it never evicts: an evicted entry would be "
                f"re-queried from the driver on a replay-critical path, which "
                f"is exactly what this cache exists to prevent). A process "
                f"holding this many distinct kernel objects is manufacturing "
                f"them — reuse the compiled kernels, or call "
                f"eagle.launch._reset_kernel_attrs_cache() at a deliberate "
                f"teardown point. Refused kernel: {fn!r}"
            )
        cached = _query_kernel_attrs(fn)
        _kernel_attrs_cache[fn] = cached
    return cached


#: ``CUfunction_attribute`` enumerators (cuda.h): the three the block-size
#: policy reads, under the names cupy's ``RawKernel.attributes`` gives them.
_FUNC_ATTRIBUTES = (("max_threads_per_block", 0), ("shared_size_bytes", 1),
                    ("num_regs", 4))


def _query_kernel_attrs(fn) -> dict:
    """The kernel properties this module caches: a cupy ``RawKernel``
    answers through ``.attributes``; a v2 door's raw ``CUfunction`` handle
    through the driver itself. ``{}`` on anything that goes wrong."""
    if isinstance(fn, int):
        try:
            from cupy_backends.cuda.api import driver
            return {name: int(driver.funcGetAttribute(attr, fn))
                    for name, attr in _FUNC_ATTRIBUTES}
        except Exception:
            return {}
    try:
        return fn.attributes
    except Exception:
        return {}


def _resolved_block(fn, n, block):
    """The block size to launch ``fn`` with: ``block`` verbatim if the
    caller passed one, else the policy resolved over the ambient sibling
    count and cached device/kernel properties."""
    if block is not None:
        return block
    return resolve_block(n, current_siblings(), _device_props(), _kernel_attrs(fn))


def require_cupy(value, name, what="array"):
    """The capturable-path contract: ``value`` must be a pre-allocated
    cupy array (``launch`` never imports, casts, or allocates -- that
    would break a CUDA-graph stream capture)."""
    import cupy as cp

    if not isinstance(value, cp.ndarray):
        raise TypeError(
            f"launch requires a pre-allocated cupy {what}; '{name}' is "
            f"{type(value).__name__} (no allocation is allowed during graph capture)"
        )
    return value


def _vec_mutables(mdecl) -> set:
    """The ``Mutable`` names bound as VECTOR-shaped ``GRef`` views, so
    :func:`~eagle.roles.classify_arg` can tell a vector Mutable from a
    matrix one. A ``frozenset`` feeding :func:`launch_plan`'s cache key."""
    return frozenset(name for name, m in mdecl.items() if m.dtype == "vector")


def _mat_mutables(mdecl) -> set:
    """The ``Mutable`` names bound as MATRIX-shaped ``GRef`` views, the
    twin of :func:`_vec_mutables`."""
    return frozenset(name for name, m in mdecl.items() if m.dtype == "matrix")


# -- The LaunchPlan (kernel-STATIC launch metadata) -------------------------
#: Cached :class:`LaunchPlan`\\ s, keyed on the static signature triple.
#: Two kernels with the same signature share a plan by design.
_launch_plan_cache: dict = {}
_LAUNCH_PLAN_COUNTS = {"hits": 0, "misses": 0}
#: Serialises the plan cache and its tallies: one plan per key, exact counts.
_LAUNCH_PLAN_LOCK = threading.Lock()


def _launch_plan_stats() -> dict:
    """A copy of the :class:`LaunchPlan` cache's hit/miss counters
    (record-only; the gate reads them)."""
    with _LAUNCH_PLAN_LOCK:
        return dict(_LAUNCH_PLAN_COUNTS)


def _reset_launch_plan_cache() -> None:
    """Empty the plan cache and zero its counters (tests / teardown)."""
    with _LAUNCH_PLAN_LOCK:
        _launch_plan_cache.clear()
        _LAUNCH_PLAN_COUNTS["hits"] = 0
        _LAUNCH_PLAN_COUNTS["misses"] = 0


def _new_box(tag):
    """The pre-allocated by-value POD a plan reuses for one argument
    position, or ``None`` for a non-POD tag (``NSAMPLES``/``UNIFORM``, built
    fresh per call)."""
    if tag in ("GREF_VEC", "GREF_MAT"):
        return np.zeros((), dtype=GREF_DTYPE)
    if tag in ("HANDLE", "WIDE_IN", "WIDE_OUT"):
        return np.zeros((), dtype=HANDLE_DTYPE)
    return None


class LaunchPlan:
    """The kernel-STATIC half of a launch, resolved ONCE per signature:
    the ABI shape each ``(role, name)`` binds as, which ``Mutable``\\ s are
    vector- vs matrix-shaped, and a reusable by-value POD box per argument
    -- all things :func:`assemble_args` used to re-derive and
    re-allocate on every call. Block-size derivation is deliberately not
    here: it is a per-call function of the batch size, not the kernel.

    **The boxes are reused, and that is the point.** ``_boxes[i]`` is
    refilled in place on every planned launch, which is safe since a
    launch packs its by-value arguments synchronously -- no launch reads a
    box after ``fn(...)`` returns. Do NOT hold an ``assemble_args`` result
    across a second planned launch of the same signature: that list
    aliases the plan's boxes. Omit ``plan`` to get fresh boxes per call.

    **Single-threaded launch.** Reused per-signature boxes mean two
    threads launching the same signature concurrently would interleave
    their refills -- not a regression, since the launch path is host-side
    sequencing behind the GIL and no launch pool in this codebase is
    threaded.
    """

    __slots__ = ("arg_spec", "tags", "vec_mutables", "mat_mutables", "boxes")

    def __init__(self, arg_spec, vec_mutables=(), mat_mutables=()):
        self.arg_spec = tuple(tuple(a) for a in arg_spec)
        self.vec_mutables = frozenset(vec_mutables)
        self.mat_mutables = frozenset(mat_mutables)
        self.tags = tuple(
            classify_arg(
                role,
                name,
                vec_mutables=self.vec_mutables,
                mat_mutables=self.mat_mutables,
            )
            for role, name in self.arg_spec
        )
        self.boxes = tuple(_new_box(tag) for tag in self.tags)

    def __len__(self) -> int:
        return len(self.arg_spec)


def _bind_gref(box, arr, n):
    """Fill one by-value ``GRef`` -- the plan's reusable ``box`` refilled
    in place, or a fresh :func:`~eagle.abi.make_gref`; byte-identical
    either way."""
    if box is None:
        return make_gref(arr, n)
    box["data_"] = arr.data.ptr
    box["samples_"] = n
    box["compStride_"] = n
    box["sampleStride_"] = 1
    box["deviceType_"] = DEVICE_CUDA
    return box


def _bind_handle(box, arr, n):
    """Fill one by-value ``HandleT`` -- the plan's reusable ``box``
    refilled in place, or a fresh :func:`~eagle.abi.make_handle`."""
    if box is None:
        return make_handle(arr, n)
    box["data"] = arr.data.ptr
    box["samples"] = n
    box["stride"] = 1
    box["deviceType"] = DEVICE_CUDA
    return box


def launch_plan(arg_spec, vec_mutables=(), mat_mutables=()) -> LaunchPlan:
    """The cached :class:`LaunchPlan` for one kernel signature (identity,
    not equality); a changed signature gets its own plan. Nothing per-call
    enters the key."""
    key = (
        tuple(tuple(a) for a in arg_spec),
        frozenset(vec_mutables),
        frozenset(mat_mutables),
    )
    with _LAUNCH_PLAN_LOCK:
        plan = _launch_plan_cache.get(key)
        if plan is None:
            _LAUNCH_PLAN_COUNTS["misses"] += 1
            plan = LaunchPlan(key[0], key[1], key[2])
            _launch_plan_cache[key] = plan
        else:
            _LAUNCH_PLAN_COUNTS["hits"] += 1
        return plan


def assemble_args(
    arg_spec,
    *,
    out,
    vec,
    per_sample,
    terminated,
    uniforms,
    n,
    tables=None,
    mutables=None,
    mutable_vec=None,
    mutable_mat=None,
    mats=None,
    wide_in=None,
    wide_out=None,
    plan=None,
):
    """Build the kernel argument tuple in the order the signature expects.

    The mutable output, input vectors, and matrix inputs are by-value
    ``GRef`` structs (a matrix GRef composes the flat vector one);
    per-sample scalars, the termination mask, lookup tables, and a pure
    kernel's scalar/int ``Mutable`` slots are by-value ``HandleT`` views;
    broadcast constants are ``float64``. A vector kernel's ``GRef`` carries
    its own ``samples_``; a pure kernel takes an explicit ``nsamples``
    instead. ``wide_in``/``wide_out`` are also ``HandleT`` views, each
    riding its own caller-populated dict so they never collide with an
    unrelated per-sample-scalar or lookup-table name.

    The ABI-shape decision per role is :func:`eagle.roles.classify_arg`
    (single-sourced with the ctypes host path); this function keeps only
    the construction -- which dict a role's value comes from, and which
    ``make_*`` helper builds it. An unrecognised role raises before any
    role-specific branch runs.

    ``plan`` (optional) is the :class:`LaunchPlan` for THIS signature.
    When given, the per-argument classification and POD boxes come from it
    instead of being re-derived here; the resulting argument list is
    byte-identical either way (read :class:`LaunchPlan`'s box-reuse note
    first). Omit ``plan`` for fresh boxes per call, as always."""
    import numpy as np

    tables = tables or {}
    mutables = mutables or {}
    mutable_vec = mutable_vec or set()
    mutable_mat = mutable_mat or set()
    mats = mats or {}
    wide_in = wide_in or {}
    wide_out = wide_out or {}
    if plan is not None and len(plan) != len(arg_spec):
        raise ValueError(
            f"assemble_args: the passed LaunchPlan describes {len(plan)} "
            f"argument(s) but arg_spec has {len(arg_spec)} — the plan does not "
            "belong to this signature (get it from eagle.launch.launch_plan)"
        )
    args = []
    for i, (role, name) in enumerate(arg_spec):
        if plan is None:
            tag = classify_arg(
                role, name, vec_mutables=mutable_vec, mat_mutables=mutable_mat
            )
            box = None
        else:
            tag = plan.tags[i]
            box = plan.boxes[i]
        if tag == "GREF_VEC":
            source = (
                out
                if role == "out"
                else mutables[name] if role == "mutable" else vec[name]
            )
            args.append(_bind_gref(box, source, n))
        elif tag == "GREF_MAT":
            source = mutables[name] if role == "mutable" else mats[name]
            args.append(_bind_gref(box, source, n))
        elif tag == "HANDLE":
            if role == "per_sample":
                source = per_sample[name]
            elif role == "lookup":
                source = tables[name]
            elif role == "terminated":
                source = terminated
            else:  # mutable, scalar/int
                source = mutables[name]
            args.append(_bind_handle(box, source, n))
        elif tag == "WIDE_IN":
            args.append(_bind_handle(box, wide_in[name], n))
        elif tag == "WIDE_OUT":
            args.append(_bind_handle(box, wide_out[name], n))
        elif tag == "NSAMPLES":
            args.append(np.uint32(n))
        elif tag == "UNIFORM":
            args.append(uniforms[name])
        else:
            # Defensive only -- classify_arg's return is exactly ARG_TAGS, so
            # this can never fire unless a NEW tag is added there without a
            # matching branch here (the per-path tag-exhaustiveness property).
            # Never silently skip an arg.
            raise ValueError(
                f"assemble_args: unhandled ABI tag {tag!r} for role {role!r}"
            )
    return args


def _launch_vector(
    fn,
    arg_spec,
    vector_inputs,
    params,
    per_sample,
    *,
    out,
    kw,
    grid,
    block,
    matrix_inputs=(),
    dt=np.float64,
):
    """Issue ONE vector-kernel launch on the current stream -- capturable.
    Every array argument must be pre-allocated cupy (nothing is allocated
    here). ``matrix_inputs`` mirrors ``_launch_pure``'s matrix leg: a
    matrix batch is contiguous ``(R, C, N)`` or flat ``(R*C, N)``, both
    binding directly with no reshape."""
    n = int(out.shape[1])
    vec = {k: require_cupy(kw.get(k), k) for k in vector_inputs}
    mats = {k: require_cupy(kw.get(k), k, "matrix batch") for k in matrix_inputs}
    ps = {s: require_cupy(kw.get(s), s, "per-sample array") for s in per_sample}
    term = require_cupy(kw.get("terminated"), "terminated", "`terminated` mask")
    tables = {
        name: require_cupy(kw.get(name), name, "flat lookup table")
        for role, name in arg_spec
        if role == "lookup"
    }
    uniforms = coerce_uniforms(kw, params, dt=dt)
    args = assemble_args(
        arg_spec,
        out=out,
        vec=vec,
        per_sample=ps,
        terminated=term,
        uniforms=uniforms,
        n=n,
        tables=tables,
        mats=mats,
        plan=launch_plan(arg_spec),
    )
    block = _resolved_block(fn, n, block)
    g = grid or ((n + block - 1) // block,)
    fn(g, (block,), tuple(args))
    return out


def _launch_pure(
    fn,
    arg_spec,
    vector_inputs,
    params,
    per_sample,
    mutables_decl,
    *,
    kw,
    grid,
    block,
    matrix_inputs=(),
    wide_inputs=(),
    wide_outputs=(),
    wide_out_exempt=(),
    dt=np.float64,
):
    """Issue ONE pure launch on the current stream -- capturable. Every
    per-sample array (writable ``Mutable`` buffers, vector/matrix inputs,
    per-sample scalars, lookup tables, ``terminated``) must be
    pre-allocated cupy; ``N`` comes from the ``Mutable``/vector/matrix
    arrays, and a ``Mutable`` is read-modify-written in place.

    ``wide_inputs``/``wide_outputs`` mirror ``per_sample``, reaching this
    capturable path as :func:`pure_prepare`'s same-named keywords do the
    allocating one -- but bound as-is here (never coerced, since that is
    illegal mid-capture). ``wide_out_exempt`` names the ``wide_outputs``
    that are row-indexed destinations with no batch axis (e.g. an atomic
    ``Accum`` output), binding at their own width and skipping
    reconciliation.

    Returns the ``Mutable`` handoff merged with the wide gradient outputs,
    mirroring :func:`pure_prepare`'s one-dict convention."""
    mdecl = {m.name: m for m in mutables_decl}
    n = None

    # the writable Mutable buffers -- also the primary source of N (the
    # LAST axis for every non-scalar slot).
    mut = {}
    for name in mdecl:
        a = require_cupy(kw.get(name), name, "Mutable array")
        mut[name] = a
        scalar_slot = mdecl[name].dtype in ("float", "int")
        n = _reconcile_n(n, int(a.size) if scalar_slot else int(a.shape[-1]))
    vec = {}
    for name in vector_inputs:
        a = require_cupy(kw.get(name), name)
        vec[name] = a
        n = _reconcile_n(n, int(a.shape[1]))
    mats = {}
    for name in matrix_inputs:
        a = require_cupy(kw.get(name), name, "matrix batch")
        mats[name] = a
        n = _reconcile_n(n, int(a.shape[-1]))
    ps = {}
    for name in per_sample:
        a = require_cupy(kw.get(name), name, "per-sample array")
        ps[name] = a
        n = _reconcile_n(n, int(a.size))
    wide_in = {}
    for name in wide_inputs:
        a = require_cupy(kw.get(name), name, "(rows, N) wide input")
        wide_in[name] = a
        n = _reconcile_n(n, int(a.shape[-1]))
    wide_out = {}
    for name in wide_outputs:
        a = require_cupy(kw.get(name), name, "(rows, N) wide gradient output")
        wide_out[name] = a
        # an exempt name is a row-indexed destination, not a per-sample plane.
        if name not in wide_out_exempt:
            n = _reconcile_n(n, int(a.shape[-1]))
    n = require_n(
        n, hint="pre-allocated per-sample array (a Mutable or a vector input)"
    )
    term = require_cupy(kw.get("terminated"), "terminated", "`terminated` mask")
    tables = {
        name: require_cupy(kw.get(name), name, "flat lookup table")
        for role, name in arg_spec
        if role == "lookup"
    }
    uniforms = coerce_uniforms(kw, params, dt=dt)
    plan = launch_plan(arg_spec, _vec_mutables(mdecl), _mat_mutables(mdecl))
    args = assemble_args(
        arg_spec,
        out=None,
        vec=vec,
        per_sample=ps,
        terminated=term,
        uniforms=uniforms,
        n=n,
        tables=tables,
        mutables=mut,
        mutable_vec=plan.vec_mutables,
        mutable_mat=plan.mat_mutables,
        mats=mats,
        wide_in=wide_in,
        wide_out=wide_out,
        plan=plan,
    )
    block = _resolved_block(fn, n, block)
    g = grid or ((n + block - 1) // block,)
    fn(g, (block,), tuple(args))
    return {**wide_out, **mut}  # merged exactly as pure_prepare merges them


def launch(
    kind, fn, arg_spec, vector_inputs, params, per_sample, *, kw, grid, block,
    dt=np.float64, **extra,
):
    """Issue ONE capturable launch of any kernel kind on the current
    stream. ``kind`` (``"vector"`` | ``"pure"``) selects the ABI binding
    shape; kind-specific inputs ride ``extra`` (a vector kernel's ``out=``,
    or a pure kernel's ``mutables_decl=``, each with optional
    ``matrix_inputs=``/``wide_inputs=``/``wide_outputs=``/
    ``wide_out_exempt=``, named identically to :func:`pure_prepare`'s).
    Every array argument must be pre-allocated cupy. Returns the vector
    ``out``, or the pure ``Mutable`` dict merged with wide outputs."""
    if kind == "vector":
        return _launch_vector(
            fn, arg_spec, vector_inputs, params, per_sample,
            out=extra["out"], kw=kw, grid=grid, block=block,
            matrix_inputs=extra.get("matrix_inputs", ()), dt=dt,
        )
    if kind == "pure":
        return _launch_pure(
            fn, arg_spec, vector_inputs, params, per_sample, extra["mutables_decl"],
            kw=kw, grid=grid, block=block,
            matrix_inputs=extra.get("matrix_inputs", ()),
            wide_inputs=extra.get("wide_inputs", ()),
            wide_outputs=extra.get("wide_outputs", ()),
            wide_out_exempt=extra.get("wide_out_exempt", ()),
            dt=dt,
        )
    raise ValueError(
        f"unknown launch kind {kind!r}; expected 'vector' | 'pure'"
    )


def pure_origin(kw, *, vector_inputs, mutable_names, per_sample):
    """The caller's framework, chosen from every per-sample source in
    signature order (vector inputs, provided ``Mutable`` arrays, then
    per-sample scalars) else numpy. Detected up front so the coercion
    rides that framework's stream too."""
    from . import interop

    cands = [kw[k] for k in vector_inputs if k in kw]
    cands += [kw[k] for k in mutable_names if k in kw and _is_array_value(kw[k])]
    cands += [kw[k] for k in per_sample if k in kw]
    return interop.origin_adapter(cands)


def pure_prepare(
    fn,
    arg_spec,
    *,
    vector_inputs,
    per_sample,
    params,
    mutables_decl,
    mutable_defaults,
    lookup_counts,
    kw,
    mat_shapes=None,
    vec_widths=None,
    wide_inputs=None,
    wide_outputs=None,
    wide_out_exempt=(),
    dt=np.float64,
    readonly_mask=False,
):
    """Coerce inputs, bind the writable ``Mutable``\\ s, assemble args,
    launch (no sync). The pure allocate-and-launch body, shared by the
    in-process compiled pure launcher and the deployed ``LoadedPure`` so
    the two can never drift.

    State vectors -> ``(3, N)`` f64; matrix inputs -> flat ``(R*C, N)``
    f64; provided ``Mutable`` arrays -> device buffers (also sizing N);
    per-sample scalars -> ``(N,)``; lookup tables -> flat handles; wide
    inputs -> 2-D ``(rows, N)``; wide gradient outputs -> 2-D ``(rows, N)``
    written in place, never auto-allocated. ``wide_out_exempt`` names the
    ``wide_outputs`` that are row-indexed destinations (e.g. an atomic
    ``Accum`` output), skipping N-reconciliation. A ``Mutable`` omitted
    from ``kw`` is a broadcast scalar fill or its declared default. Returns
    the ``Mutable`` buffers merged with any wide gradient outputs (RMW in
    place). Meant to run inside the caller's launch context (see
    :meth:`LaunchMixin._dispatch`).

    ``readonly_mask`` (default ``False``) is the sidecar-declared opt-in
    letting an omitted ``terminated`` mask be served from the cached
    all-false mask (see :func:`~eagle.marshal.coerce_terminated`); never
    inferred, since marshal cannot read a kernel's source."""
    mdecl = {m.name: m for m in mutables_decl}
    mut_names = tuple(m.name for m in mutables_decl)

    # ``vec_widths`` keeps the pure path width-honest (a derivative seed's
    # width is its output's, not necessarily 3).
    vec, n = coerce_vec_inputs(kw, vector_inputs, widths=vec_widths, dt=dt)
    mats, n = coerce_mat_inputs(kw, mat_shapes or {}, n, dt=dt)

    provided = {}
    adapted = {}     # sample-major Mutables copied in: written back after the launch
    for name in mut_names:
        if name in kw and _is_array_value(kw[name]):
            arr, count = coerce_mutable(name, kw[name], mdecl[name], dt=dt,
                                        adapted=adapted)
            n = _reconcile_n(n, count)
            provided[name] = arr

    per_sample_arr, n = coerce_per_sample(kw, per_sample, n, dt=dt)
    wide_in, n = coerce_wide_inputs(kw, wide_inputs or (), n, dt=dt)
    wide_out, n = coerce_wide_outputs(
        kw, wide_outputs or (), n, dt=dt, exempt=wide_out_exempt
    )
    n = require_n(
        n,
        hint="per-sample array (a Mutable, a vocabulary input, or a per-sample scalar)",
    )

    mut = {}
    for name in mut_names:
        if name in provided:
            mut[name] = provided[name]
        elif name in kw:  # a scalar -> broadcast-fill
            mut[name] = fill_mutable(name, kw[name], mdecl[name], n, dt=dt)
        elif mutable_defaults.get(name) is not None:
            default = mutable_defaults[name]
            mut[name] = fill_mutable(name, default, mdecl[name], n, dt=dt)
        else:
            raise TypeError(
                f"missing required Mutable '{name}' (no default to fill from)"
            )

    tables = coerce_tables(kw, lookup_counts, dt=dt)
    term = coerce_terminated(kw, n, readonly_mask=readonly_mask)
    uniforms = coerce_uniforms(kw, params, dt=dt)

    plan = launch_plan(arg_spec, _vec_mutables(mdecl), _mat_mutables(mdecl))
    args = assemble_args(
        arg_spec,
        out=None,
        vec=vec,
        per_sample=per_sample_arr,
        terminated=term,
        uniforms=uniforms,
        n=n,
        tables=tables,
        mutables=mut,
        mutable_vec=plan.vec_mutables,
        mutable_mat=plan.mat_mutables,
        mats=mats,
        wide_in=wide_in,
        wide_out=wide_out,
        plan=plan,
    )
    # Eager path, never mid-capture: routes through the same resolver with
    # siblings=1 fixed, single-sourced rather than a second DEFAULT_BLOCK.
    block = resolve_block(n, 1, _device_props(), _kernel_attrs(fn))
    grid = ((n + block - 1) // block,)
    fn(grid, (block,), tuple(args))
    for name, a in adapted.items():
        if a.scratch is not None:
            a.write_back_from(mut[name])
    return {**wide_out, **mut}


class LaunchMixin:
    """The framework-polymorphic launch skeleton, shared by every
    launcher: owns the ``origin -> with origin.launch_context(): ... -> if
    origin.blocking: deviceSynchronize()`` wrapper and the unknown-keyword
    guard -- the single source of "one call mirrors the input framework"."""

    def _reject_unknown(self, kw, allowed) -> None:
        unknown = set(kw) - allowed
        if unknown:
            raise TypeError(
                f"unexpected argument(s): {sorted(unknown)}; expected {sorted(allowed)}"
            )

    def _origin_from(self, kw, names):
        """The caller's tensor framework, chosen from the passed values of
        ``names``; sets both the output type and the stream/sync policy."""
        from . import interop

        return interop.origin_adapter([kw[k] for k in names if k in kw])

    def _dispatch(self, *, origin, prepare, export):
        """Run ``prepare`` inside the origin framework's launch context,
        synchronize only for a blocking framework, then hand the result
        back via ``export``. ``prepare``/``export`` are the per-kind hooks."""
        import cupy as cp

        with origin.launch_context():
            artifacts = prepare()
        if origin.blocking:
            cp.cuda.runtime.deviceSynchronize()
        return export(artifacts)
