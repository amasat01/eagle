# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Private, invisible block-size resolver for eagle's capturable launch entries.

This module implements a locked launch-policy formula verbatim; the one
rule of its own is the persistent launch's threads per SM (:func:`persistent_launch`).
The sibling-count plumbing below (:class:`GraphPipeline`'s ``add_concurrent``
groups are the only source of ``siblings >= 2``; every other call site is 1).

Nothing here is public API: no module in this package re-exports
:func:`resolve_block` / :func:`sibling_context` from ``eagle/__init__.py``, and
nothing outside :mod:`eagle.launch` and :mod:`eagle.pipeline` should import
this module. The design constraint ("users should not need to think
about tuning kernel launch parameters") is met structurally: there is no env
var, no kwarg, and no knob anywhere in this file.

**Deliberately cupy-free — at import time AND at call time.** :func:`resolve_block`
takes ``dev``/``kern`` as plain values (a dict of device/kernel properties, or
any object exposing the same fields as attributes); it never imports cupy to
go get them itself. That is what makes it: (a) pure and directly unit-testable
with synthetic inputs, and (b) importable and
fully exercisable with cupy entirely absent from the environment, matching the
discipline established for eagle-less tests. The actual cupy-backed device
query and per-kernel ``.attributes`` read (+ their process-lifetime caching)
live in :mod:`eagle.launch`, the one module that already imports cupy
lazily on every other capturable path.
"""

from __future__ import annotations

import contextlib
import math
import contextvars

#: Block size used whenever the ``siblings <= 1`` small-N formula cannot
#: resolve one (n/dev/kern missing or malformed) AND whenever the
#: ``siblings >= 2`` branch cannot resolve a feasible co-resident block size
#: (the "saturation fallback", the same formula's last clause). Equal to
#: :data:`eagle.launch.DEFAULT_BLOCK`, deliberately not imported from there:
#: this module must stay import-cycle-free and standalone-testable; the LOCK
#: is the shared *value* (256), not a shared symbol. It is also the small-N
#: formula's own upper clamp, so large N still converges to exactly this
#: value (today's identity, preserved, not special-cased).
_FALLBACK_BLOCK = 256

#: CUDA warp size. The block-size formula mirrors ``eagle::cuda::computeBlocks``'s
#: shape (`eagle/eagle/cuda/ComputeBlocks.h`), which rounds to a hard-coded
#: WARP=32 rather than a device-queried ``warpSize`` — kept identical here.
_WARP = 32

#: Device rule's warps/SM saturation knee (:func:`capacity`'s sibling
#: constant): the warps/SM level above which the SM issue pipe stays busy
#: enough that a compaction's rebuild cost pays for itself. A per-device
#: constant, measured once rather than fitted to any one perf card — this is
#: the P2000 (Pascal, sm_61) measurement; a newer architecture (Volta+) is
#: expected to carry a higher knee. Exposed as a module constant (not a
#: card-fitted literal buried in a formula) so a later device profile can
#: override it without touching the decision logic that reads it.
W_KNEE_DEFAULT = 8


def _attr(obj, name):
    """Best-effort read of ``name`` off ``obj`` — mapping OR attribute style —
    coerced to a non-negative int, or ``0`` on ANYTHING that goes wrong
    (missing key/attribute, wrong/unconvertible type, negative value, ``obj``
    being ``None``, an exception from a pathological ``__getattr__``/``__getitem__``).
    The single point where a missing or zero attribute degrades to the
    fallback instead of raising, keeping this formula's "never raises" clause."""
    if obj is None:
        return 0
    try:
        if isinstance(obj, dict):
            value = obj.get(name, 0)
        else:
            value = getattr(obj, name, 0)
        value = int(value)
    except Exception:
        return 0
    return value if value > 0 else 0


def _warp_round_up(x: int) -> int:
    return ((x + _WARP - 1) // _WARP) * _WARP


def _warp_floor(x: int) -> int:
    return (x // _WARP) * _WARP


def _register_feasible(bs, siblings, num_regs, regs_per_sm):
    """The largest warp multiple <= ``bs`` that keeps ``siblings`` co-resident
    blocks within the SM's register file, or ``None`` if no warp multiple
    >= 32 fits (the caller's cue to return the saturation fallback)."""
    candidate = bs
    while candidate >= _WARP:
        if siblings * candidate * num_regs <= regs_per_sm:
            return candidate
        candidate -= _WARP
    return None


def resolve_block(n, siblings, dev, kern) -> int:
    """The locked launch-block-size formula: the launch block size for an ``n``-sample
    kernel co-resident with ``siblings - 1`` others on the same SM.

    ``dev`` carries ``multiProcessorCount`` / ``regsPerMultiprocessor`` /
    ``sharedMemPerMultiprocessor`` (a ``cp.cuda.runtime.getDeviceProperties``
    dict, or anything exposing the same fields); ``kern`` carries ``num_regs``
    / ``shared_size_bytes`` / ``max_threads_per_block`` (a ``RawKernel``'s
    ``.attributes`` dict, or the same shape). Both are read defensively via
    :func:`_attr` — a missing/zero/malformed field degrades to the fallback,
    it never raises.

    - ``siblings <= 1`` (the permanent, non-concurrent identity): the small-N
      launch-geometry rule, ``block = clamp(32 * ceil(n / (2 * SMs * 32)), 32,
      256)``, narrowed to the largest register-feasible warp multiple (one
      block alone on the SM). This CONVERGES to exactly 256 once ``n >= 2 *
      SMs * 256`` (the clamp's own upper arm), so today's identity for a
      large batch is preserved, not special-cased; only a small batch moves.
      Missing/malformed dev/kern properties, or no register-feasible warp
      multiple, degrade to the 256 fallback, same as every other branch.
    - ``siblings = S >= 2``: mirrors ``eagle::cuda::computeBlocks``'s
      occupancy-target shape with the per-launch machine share divided by
      ``S``, then narrows to the largest register-feasible warp multiple; if
      shared memory alone rules out ``S``-way co-residency, or no warp
      multiple >= 32 fits the register file, co-residency is impossible and
      the saturation fallback (256, today's value) is returned.

    Never raises.
    """
    try:
        siblings = int(siblings)
    except Exception:
        return _FALLBACK_BLOCK

    try:
        n = int(n)
    except Exception:
        return _FALLBACK_BLOCK
    if n <= 0:
        return _FALLBACK_BLOCK

    n_sms = _attr(dev, "multiProcessorCount")
    regs_per_sm = _attr(dev, "regsPerMultiprocessor")
    smem_per_sm = _attr(dev, "sharedMemPerMultiprocessor")
    max_threads = _attr(kern, "max_threads_per_block")
    num_regs = _attr(kern, "num_regs")
    smem_bytes = _attr(kern, "shared_size_bytes")  # 0 is a legal "no smem" value

    if (
        n_sms <= 0
        or regs_per_sm <= 0
        or smem_per_sm <= 0
        or max_threads <= 0
        or num_regs <= 0
    ):
        return _FALLBACK_BLOCK

    upper = min(_FALLBACK_BLOCK, _warp_floor(max_threads))
    if upper < _WARP:
        return _FALLBACK_BLOCK

    if siblings <= 1:
        # Small batches: block = clamp(32*ceil(n / (2*SMs*32)), 32, 256). No
        # shared-memory check here -- a solo launch never competes with a
        # sibling for the SM's shared-memory budget, so that constraint
        # (below, S-way co-residency only) does not apply.
        denom = 2 * n_sms * _WARP
        ideal = _WARP * -(-n // denom)  # 32 * ceil(n / denom)
        bs = min(max(ideal, _WARP), upper)
        found = _register_feasible(bs, 1, num_regs, regs_per_sm)
        return found if found is not None else _FALLBACK_BLOCK

    # siblings = S >= 2: the co-residency occupancy-target shape (unchanged).
    # Shared memory alone can rule out S-way co-residency, independent of
    # everything below.
    if siblings * smem_bytes > smem_per_sm:
        return _FALLBACK_BLOCK

    share = max(1, (n_sms * 4) // siblings)
    ideal = _warp_round_up(-(-n // share))  # ceil(n / share), then to a warp multiple
    bs = min(max(ideal, _WARP), upper)

    found = _register_feasible(bs, siblings, num_regs, regs_per_sm)
    return found if found is not None else _FALLBACK_BLOCK


#: Persistent (work-stealing) launch geometry's own defaults. Relocated
#: here (not re-derived) from where :mod:`eagle._until_done` first measured
#: them for the one-launch persist entry: the grid targets this many
#: resident blocks per SM -- measured the better of the 1-2 blocks/SM range
#: on the P2000, not a card-fitted literal: it falls out of the SM count
#: alone (see :func:`persistent_geometry`) -- and narrows the block down
#: from this ideal only when the entry's own registers demand it.
PERSISTENT_BLOCKS_PER_SM = 1
PERSISTENT_IDEAL_BLOCK = 256


def persistent_geometry(dev, kern, *, blocks_per_sm: int = PERSISTENT_BLOCKS_PER_SM,
                        ideal_block: int = PERSISTENT_IDEAL_BLOCK):
    """Block size and grid (block count) for a persistent, work-stealing
    launch: ``(block, grid)``.

    ``grid = SMs * blocks_per_sm`` is FIXED, independent of the sample
    count -- a persistent launch's lanes steal work off a shared counter
    until it runs out, so (unlike :func:`resolve_block`'s one-shot-kernel
    grid) there is no ``n`` to spread over. ``block`` starts at
    ``ideal_block`` and narrows to the largest warp multiple that keeps
    ``blocks_per_sm`` CO-RESIDENT blocks within the SM's register file --
    the same :func:`_register_feasible` search :func:`resolve_block` runs
    for one resident block, run here for ``blocks_per_sm`` of them, since
    that many are meant to live on one SM at once.

    ``dev``/``kern`` are read the SAME defensive way as :func:`resolve_block`
    (:func:`_attr`: a missing/zero/malformed field never raises); a missing
    SM count returns ``(ideal_block, 0)`` (no persistent launch is possible
    without it -- the caller's cue this geometry is unavailable), a missing
    register/thread-limit property returns ``(ideal_block, grid)`` (block
    narrowing skipped, the grid is still known)."""
    n_sms = _attr(dev, "multiProcessorCount")
    if n_sms <= 0:
        return ideal_block, 0
    grid = n_sms * blocks_per_sm
    regs_per_sm = _attr(dev, "regsPerMultiprocessor")
    max_threads = _attr(kern, "max_threads_per_block")
    num_regs = _attr(kern, "num_regs")
    if regs_per_sm <= 0 or max_threads <= 0 or num_regs <= 0:
        return ideal_block, grid
    upper = min(ideal_block, _warp_floor(max_threads))
    if upper < _WARP:
        return ideal_block, grid
    found = _register_feasible(upper, blocks_per_sm, num_regs, regs_per_sm)
    return (found if found is not None else ideal_block), grid


#: The persistent launch's threads-per-SM rule (:func:`persistent_launch`).
#: Each SM needs enough warps per scheduler partition to cover three costs:
#: the step's dependent arithmetic chain, the stall a lane takes each time it
#: fetches its next sample, and -- for a small batch -- the drain at the end,
#: when every warp runs its slowest remaining sample with few lanes live. The
#: chain and the fetch set the saturation point from the device's issue rate
#: (the FP32:FP64 ratio for a float64 step), the drain caps the threads a small
#: batch can use. The constants were measured on a Quadro P2000 (cc 6.1) in
#: float64 and float32 and checked on cells the fit never saw; re-derive them
#: on a new GPU class from a uniform and a spread sweep of threads per SM.
PERSIST_CHAIN_CYCLES = 4          # latency of the step's dependent op chain
PERSIST_FETCH_CYCLES = 2000       # one sample fetch: the counter atomic + dependent loads
PERSIST_FETCH_SPACING = 147       # steps between a warp's fetch stalls = max_steps / this
#: Dominant-pipe issue slots per step. The kernel's own count is not known
#: here yet, so one value per precision (an RK4 oscillator step).
PERSIST_STEP_OPS = {"float64": 33, "float32": 66}
#: Samples per lane below which more threads cost more in the drain than they
#: gain in the fill.
PERSIST_SAMPLES_PER_LANE = {"float64": 30, "float32": 4}
#: (scheduler partitions, FP32 lanes) per SM by compute capability, where it
#: differs from four partitions of 32 lanes.
_SM_LAYOUT = {(6, 0): (2, 64), (7, 0): (4, 64), (7, 2): (4, 64), (7, 5): (4, 64),
              (8, 0): (4, 64)}


def persistent_threads_per_sm(dev, kern, n: int, max_steps: int,
                              scalar_type: str = "float64") -> int:
    """Threads per SM for a persistent launch of ``n`` samples with a step
    budget of ``max_steps``: ``n / (SMs * q)`` clamped between the threads
    that cover the step's chain latency and those that also hide the
    sample-fetch stalls (plus one spare warp per partition), never above the
    register-limited resident threads. ``0`` when the SM count is unknown."""
    n_sms = _attr(dev, "multiProcessorCount")
    if n_sms <= 0:
        return 0
    parts, lanes = _SM_LAYOUT.get((_attr(dev, "major"), _attr(dev, "minor")), (4, 128))
    issue = max(1, _WARP * parts // lanes)            # cycles per FP32 warp instruction
    fp64 = scalar_type != "float32"
    if fp64:
        issue *= max(1, _attr(dev, "singleToDoublePrecisionPerfRatio"))
    ops = PERSIST_STEP_OPS["float64" if fp64 else "float32"]
    w_lat = 1 + PERSIST_CHAIN_CYCLES / issue
    w_fetch = PERSIST_FETCH_CYCLES / (ops * issue * max(max_steps / PERSIST_FETCH_SPACING, 1.0))
    per_warp = _WARP * parts
    t_lat = per_warp * math.ceil(w_lat)
    t_sat = per_warp * math.ceil(w_lat + w_fetch + 1)
    cap = resident_threads_per_sm(dev, kern)
    if cap > 0:
        t_lat, t_sat = min(t_lat, cap), min(t_sat, cap)
    q = PERSIST_SAMPLES_PER_LANE["float64" if fp64 else "float32"]
    t = max(t_lat, min(t_sat, n / (n_sms * q)))
    return max(_WARP, _WARP * round(t / _WARP))


def persistent_launch(dev, kern, n: int, max_steps: int, scalar_type: str = "float64"):
    """``(block, grid)`` of a persistent launch: :func:`persistent_threads_per_sm`
    split into the fewest blocks per SM of at most 512 threads (the shape
    does not matter, only the threads per SM), through
    :func:`persistent_geometry`'s register narrowing. Unknown device
    properties fall back to :func:`persistent_geometry`'s defaults."""
    threads = persistent_threads_per_sm(dev, kern, n, max_steps, scalar_type)
    if threads <= 0:
        return persistent_geometry(dev, kern)
    blocks = 1
    while threads // blocks > 512:
        blocks *= 2
    return persistent_geometry(dev, kern, blocks_per_sm=blocks,
                               ideal_block=_warp_floor(threads // blocks))


def resident_threads_per_sm(dev, kern) -> int:
    """The resident-thread ceiling one SM can hold at ``kern``'s register
    count: ``regsPerMultiprocessor // num_regs``, clamped to
    ``maxThreadsPerMultiProcessor`` when the device reports one, floored to
    a warp multiple. ``0`` when ``dev``/``kern`` properties are unavailable
    (read defensively via :func:`_attr`, same degradation as
    :func:`resolve_block`) — the caller treats ``0`` as "unknown"."""
    regs_per_sm = _attr(dev, "regsPerMultiprocessor")
    num_regs = _attr(kern, "num_regs")
    if regs_per_sm <= 0 or num_regs <= 0:
        return 0
    threads = regs_per_sm // num_regs
    max_resident = _attr(dev, "maxThreadsPerMultiProcessor")
    if max_resident > 0:
        threads = min(threads, max_resident)
    return _warp_floor(threads)


def capacity(dev, kern) -> int:
    """Capacity ``C = SMs * resident threads at the kernel's registers`` —
    the device rule's "fits inside one resident wave" bound: a batch of at
    most ``C`` samples never needs more warps than the device can hold
    resident at once for this kernel. ``0`` when device/kernel properties
    are unavailable, which the caller must read as "unknown" (no real ``n``
    is ``<= 0``, so ``n <= capacity(...)`` is never satisfied by an unknown
    capacity).

    This is a hardware RESIDENCY ceiling, not by itself the "compaction
    buys nothing below here" bound — see :func:`latency_regime_capacity`,
    which is what the no-map-rebuild decision actually gates on."""
    n_sms = _attr(dev, "multiProcessorCount")
    threads = resident_threads_per_sm(dev, kern)
    if n_sms <= 0 or threads <= 0:
        return 0
    return n_sms * threads


def latency_regime_capacity(dev, kern, w_knee: int = W_KNEE_DEFAULT) -> int:
    """The safe "no active-set rebuild" bound for the small-N decision: the
    SMALLER of :func:`capacity` (hardware residency)
    and the issue-pipe saturation point ``SMs * 32 * w_knee`` (the thread
    count that keeps every SM's issue pipe busy at ``w_knee`` warps/SM, the
    same constant the compaction knee is stated against).

    Residency alone is NOT the right bound: measured on the P2000
    (oscillator kernel, spread distribution, 1000 steps), a batch of 10000
    samples is comfortably hardware-resident (``capacity`` is 16384 for
    this kernel) but already sits well above the issue-pipe knee
    (``8 SMs * 32 * 8 = 2048``) -- skipping the active-set rebuild there
    cost ~2x wall time (3.56 ms compacting vs 6.95-7.53 ms not), because a
    spread/thinning batch keeps nearly every warp holding at least one
    still-live lane until the active-set map is rebuilt and dead warps can
    exit via the live-count check; that saving does not care whether the
    batch is hardware-resident, only whether the issue pipe has slack to
    begin with. ``0`` when either bound is unavailable."""
    reg_cap = capacity(dev, kern)
    n_sms = _attr(dev, "multiProcessorCount")
    if reg_cap <= 0 or n_sms <= 0:
        return 0
    return min(reg_cap, n_sms * _WARP * w_knee)


# --------------------------------------------------------------------------- #
# Sibling-count plumbing: a module-level ContextVar, set only by
# GraphPipeline.build's internal wrapper around each `_ConcurrentGroup`
# member invocation. Nothing else in the codebase should set it; every
# call site not inside a concurrent group sees the default, 1.
# --------------------------------------------------------------------------- #

_siblings: contextvars.ContextVar[int] = contextvars.ContextVar(
    "eagle_launch_siblings", default=1
)


def current_siblings() -> int:
    """The ambient sibling count for the launch about to happen: ``1``
    everywhere except inside a :meth:`~eagle.pipeline.GraphPipeline.build`
    concurrent-group member, where it is that group's member count."""
    return _siblings.get()


@contextlib.contextmanager
def sibling_context(count: int):
    """Set the ambient sibling count to ``count`` for the duration of the
    ``with`` block (reentrant-safe via :mod:`contextvars` token reset — a
    nested/concurrent use restores the exact prior value, not a hard-coded 1).
    Internal: the only caller is :meth:`~eagle.pipeline.GraphPipeline.build`,
    wrapping one ``_ConcurrentGroup`` member's invocation at a time."""
    token = _siblings.set(int(count))
    try:
        yield
    finally:
        _siblings.reset(token)
