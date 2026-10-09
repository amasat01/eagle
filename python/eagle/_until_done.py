# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Run a batch until every sample is done, in one call.

A step kernel that finishes its own samples (``terminated = cond``,
counting into :data:`FINISHED_PLANE`) carries a ``finish`` declaration in
its sidecar. Given such a plan, :func:`until_done` allocates what the loop
needs, binds it once, and builds the loop; :func:`run_until_done` is the
one-call form::

    report = eagle.run_until_done(plan, max_steps=10_000, x=x, v=v, dt=1e-3)
    assert report.done

The loop is a :func:`eagle.repeat_while` over the bound launch, guarded by
the finished count, plus — for ``Guard(active_set=True)`` — a
:func:`eagle.compaction_body` cadence over an :class:`eagle.ActiveSet`
(read off the artifact, never a flag). Device runs capture the loop into
one CUDA graph on the first :meth:`Runner.run` and replay it; host runs
drive the same loop directly.

An automatic kernel (``finish.steps == "auto"``) reads its steps per
launch from a device word, :data:`FUSED_STEPS_PLANE`, written by a policy
that doubles ``k`` when few samples finished, halves it when many, clamps
it to ``[AUTO_K_MIN, steps_max]``, and cuts the last launch exact so no
sample exceeds ``max_steps`` (``every=`` is refused). A host run paces
these launches its own way (:attr:`Runner.host_policy`,
:mod:`eagle._host_loop`): ``steps_max`` steps per launch when the entry's
fused side is tiled, else one step per launch until a timed fused launch
beats it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass, field

from ._active_set import (
    COUNT_PLANE,
    DEFAULT_EVERY,
    MAP_PLANE,
    ActiveSet,
    compaction_body,
)
from . import _layout
from ._compat import zip_strict
from ._conditional import SkipGuard, Skippable, repeat_while
from .roles import PER_SAMPLE_ROLES
from .sidecar import FINISH_AUTO, FINISH_COUNTER, read_finish

#: The reserved ``lookup`` plane a finishing kernel counts newly finished
#: samples into: one ``uint32`` cell, viewed as ``int32`` at bind.
FINISHED_PLANE = FINISH_COUNTER

#: The reserved ``lookup`` plane an automatic kernel reads its steps per
#: launch from: one ``int64`` cell, written by the runner's policy.
FUSED_STEPS_PLANE = "fused_steps"

#: The policy's constants: the fewest steps one launch takes (but the
#: last, cut to the remainder of ``max_steps``), and the first launch's
#: steps. The most is the artifact's ``finish.steps_max``.
AUTO_K_MIN = 8
AUTO_K0 = 16

#: The names the runner allocates and binds itself.
_RESERVED = (FINISHED_PLANE, MAP_PLANE, COUNT_PLANE, FUSED_STEPS_PLANE)

# The policy cells (int64): the band's current k, the counter at the last
# launch, the steps run so far, the step cap.
_K, _FIN_AT, _DONE, _CAP = range(4)

_STRUCTURES = {"device_kernel": True, "host_team": False}

__all__ = ["FINISHED_PLANE", "FUSED_STEPS_PLANE", "RunReport", "Runner",
           "run_until_done", "until_done"]


@dataclass(frozen=True)
class RunReport:
    """What one :meth:`Runner.run` did.

    ``finished`` counts samples whose mask is set when the run ends,
    ``done`` is ``finished == n``, ``launches`` the loop's iteration count,
    and ``steps`` the steps the loop ran (``launches * K``, or the summed
    ``k`` for an automatic artifact). ``exact_steps`` says ``steps`` is
    exact (true for ``K=1``, or an automatic artifact that launched
    nothing, hit the cap early, or ended on a one-step launch).
    ``compactions``/``reorders`` count what ran, ``build_s`` the one-time
    capture wall (0 on host), ``launches_by_k`` each ``k``'s launch count.
    ``lane_utilisation`` (active lanes / warp-iterations, in ``(0, 1]``) is
    set only for a persist-entry run that counted it; ``None`` otherwise
    (every other mode, or a persist run that skipped the counter).
    ``mode`` is the launch shape this run actually took: ``"fused_one"``/
    ``"persist"`` for a no-graph fast-path run, ``"band"`` for the
    WHILE-graph/policy path (the only mode before the fast path existed).
    ``probe`` is true for a fast-path run taken to time the entry the batch
    is not currently using (see :meth:`Runner._pick_entry`)."""

    n: int
    finished: int
    done: bool
    launches: int
    steps: int
    compactions: int
    reorders: int
    build_s: float
    launches_by_k: dict = field(default_factory=dict)
    exact_steps: bool = False
    lane_utilisation: float | None = None
    mode: str = "band"
    probe: bool = False


def _array_module(device: bool):
    if device:
        import cupy

        return cupy
    import numpy

    return numpy


#: A settled runner sums its steps (the step fill f) on one run in this many;
#: the runs between pass no sum cell and pay no device atomics.
STEP_FILL_EVERY = 8


def _packed_read(xp, device: bool, cells) -> tuple:
    """Read several 1-element uint32 counter cells as ONE combined transfer
    instead of a separate full round-trip sync per cell (a run's
    fixed overhead is dominated by sync COUNT, not by what each one moves).
    ``cells`` is a sequence of ``(array, index)`` pairs, each a uint32 cell
    (device: cupy; host: numpy, where there is no sync to fold, so each is
    just read directly). Device: every cell is copied -- async, no sync --
    into one small staging buffer on the current stream, then ONE ``.get()``
    reads them all together. Returns the values as plain Python ints, in
    ``cells`` order."""
    if not device:
        return tuple(int(arr[idx]) for arr, idx in cells)
    first, lo = cells[0]
    if all(arr is first and idx == lo + j for j, (arr, idx) in enumerate(cells)):
        # consecutive words of one array: no staging copies, the ONE read
        return tuple(int(v) for v in first[lo : lo + len(cells)].get())
    staging = xp.empty(len(cells), dtype=xp.uint32)
    for i, (arr, idx) in enumerate(cells):
        staging[i : i + 1] = arr[idx : idx + 1]
    host = staging.get()
    return tuple(int(v) for v in host)


# The step fill f = (sum of the lanes' steps) / (n * longest lane's steps) below
# which the fused entry is not timed against persist. Structural bound: the
# persist loop is the fused loop plus at most ~9 bookkeeping instructions per
# step on a body of at least ~25 instructions, so the fused entry cannot win
# below this fill on any device; above it the measured wall arbitrates.
FUSED_MIN_STEP_RATIO = 0.6


def _fast_mode_for(step, n: int, *, compacting: bool = False) -> str | None:
    """The no-graph launch mode a FIRST run picks for one single-kernel,
    contiguous-partition auto run of ``n`` samples, reorder not requested --
    ``"fused_one"`` (one fused launch, word = the whole budget), ``"persist"``
    (one persist-entry launch), or ``None`` (fall back to the band/WHILE-graph
    path: capacity unknown, or the entry the size picked is unavailable).

    The threshold is :meth:`~eagle._plan_binding.BoundPlan.device_capacity`
    alone (the same latency-regime bound :meth:`Runner._build_auto` already
    reads for the compacting case) -- measured on the P2000 (oscillator and
    rk7(8), spread S=1000): at n=2048 (=capacity for this
    kernel/device) fused-one and persist are within 1.2% of each other;
    from n=2200 on, persist already wins by >=1.4x, and the gap only grows
    (4096: 1.7x, 8192: 2.5x, 100k rk78: 1.78x). A device/kernel pair whose
    own SMs*256*4 sits well above capacity was NOT found on this hardware,
    so no separate persist-threshold band was needed -- capacity is the one
    crossover, read fresh per kernel/device rather than card-fitted.

    ``compacting`` (an active-set/map-kind artifact) launches the SAME
    contiguous entries a plain kernel does for ``"persist"`` (unchanged,
    by name), but needs the sibling ``<name>_range`` entry for
    ``"fused_one"`` -- the map entry's own default :meth:`~BoundPlan.launch`
    is active-set-indexed, not contiguous, so it is never used here. A unit
    built without hawk's ``HAWK_FAST_ENTRIES=1`` carries neither sibling for
    a compacting kernel: this falls through to the band path, same as an
    unknown capacity.

    This size rule is the first-run rule. Above capacity, when both the
    persist and the range entries exist, later runs follow the measured wall
    of each entry on this device (:meth:`Runner._pick_entry`); at or below
    capacity the size pick stands for every run."""
    cap = step.device_capacity()
    if cap <= 0:
        return None
    if n <= cap and not (compacting and step.range_entry() is None):
        return "fused_one"
    if step.persist_entry() is not None:
        return "persist"
    return None


def _kernel_name(plugin) -> str:
    for attr in ("name", "kernel_name"):
        value = getattr(plugin, attr, None)
        if isinstance(value, str) and value:
            return value
    meta = getattr(plugin, "sidecar", None)
    if isinstance(meta, Mapping) and isinstance(meta.get("name"), str):
        return meta["name"]
    return "the plan's kernel"


def _finish_of(plugin, name: str):
    """The plugin's ``finish`` declaration: its ``sidecar``'s key when it
    carries one, else its ``finish`` attribute, else ``None``."""
    meta = getattr(plugin, "sidecar", None)
    if isinstance(meta, Mapping):
        return read_finish(dict(meta), name=name)
    finish = getattr(plugin, "finish", None)
    if finish is None:
        return None
    return read_finish({"finish": finish}, name=name)


def _require_counter_slot(arg_spec, name: str) -> None:
    """A finishing kernel binds the counter it counts into."""
    if ("lookup", FINISHED_PLANE) not in arg_spec:
        raise ValueError(
            f"eagle.until_done: {name} declares 'finish' but its arg_spec binds no "
            f"('lookup', {FINISHED_PLANE!r}) plane to count finished samples into")


def _check_count(label: str, value) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"eagle.until_done: {label} is an int, got {type(value).__name__}")
    if value < 1:
        raise ValueError(f"eagle.until_done: {label} must be >= 1, got {value}")
    return value


def _bump_kernel():
    """``cell[0] += 1`` as one single-thread device kernel (cupy), compiled
    before any capture."""
    import cupy

    return cupy.ElementwiseKernel("", "raw uint32 cell", "if (i == 0) cell[0] += 1u;",
                                  "eagle_until_done_bump")


_MASK_COUNT = None


def _mask_count_kernel():
    """``cell[0] = count_nonzero(mask)`` as one device kernel (cupy), built
    once per process: a plain per-element ``atomicAdd`` into a cell the
    caller has already zeroed, dispatched over the mask's own N elements --
    NOT cupy's own :func:`cupy.count_nonzero` (a cub reduction whose kernel
    JIT-compiles slowly on a cold ``CUPY_CACHE_DIR``, ~2.9 s measured) nor a
    single-thread trick like :func:`_bump_kernel` (this one scans N
    elements, so it needs N threads, not one)."""
    global _MASK_COUNT
    if _MASK_COUNT is None:
        import cupy

        _MASK_COUNT = cupy.ElementwiseKernel(
            "bool mask", "raw uint32 cell",
            "if (mask) atomicAdd(&cell[0], 1u);",
            "eagle_until_done_mask_count")
    return _MASK_COUNT


def _mask_count(xp, device: bool, finished, terminated) -> None:
    """``finished[0] = count_nonzero(terminated)``, re-synced to the mask
    EVERY call (never cached): on the device, :func:`_mask_count_kernel`
    (not cupy's own ``count_nonzero``, see there); on the host, numpy's
    ``count_nonzero`` is a plain loop, no JIT, so it stays."""
    if not device:
        finished[0] = xp.count_nonzero(terminated)
        return
    rt = xp.cuda.runtime
    stream = xp.cuda.get_current_stream().ptr
    rt.memsetAsync(finished.data.ptr, 0, finished.nbytes, stream)
    if terminated.shape[0] > 0:
        _mask_count_kernel()(terminated, finished)


def _warm_device_kernels() -> None:
    """Compile the cupy kernels a device run of the loop launches first --
    the active set's identity map and count, the step word's fill, the
    bump and the mask count -- on a one-sample batch. None depends on the
    kernel being run, so :func:`eagle.deploy` overlaps them with its own
    device build; the run then finds them compiled."""
    import cupy as xp

    from ._active_set import ActiveSet

    mask = xp.zeros(1, dtype=xp.bool_)
    ActiveSet(mask, _lazy_scratch=True)
    word = xp.zeros(1, dtype=xp.int64)
    word[0] = 1
    _bump_kernel()(xp.zeros(1, dtype=xp.uint32), size=1)
    _mask_count(xp, True, xp.zeros(1, dtype=xp.uint32), mask)
    xp.cuda.get_current_stream().synchronize()


#: The policy band, device face: ``k_next`` from ``k`` (current steps per
#: launch), ``newly`` (finished in the launch) and ``live`` (not finished).
_BAND_SRC = (
    "if (newly * 16 < live) k_next = (2 * k < KMAX) ? 2 * k : KMAX;\n"
    "else if (newly * 4 > live) k_next = (k / 2 > KMIN) ? k / 2 : KMIN;\n")

_POLICY_SRC = r"""
extern "C" __global__ void eagle_until_done_policy(
        const unsigned int* finished, const unsigned int* total, long long* cells,
        long long* word, unsigned int* go, long long* hist) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long KMIN = %(kmin)d, KMAX = %(kmax)d;
    const long long fin = finished[0], all = total[0];
    const long long live = all - fin, newly = fin - cells[%(fin_at)d];
    const long long done = cells[%(done)d] + word[0];
    cells[%(fin_at)d] = fin;
    cells[%(done)d] = done;
    const long long k = cells[%(k)d];
    long long k_next = k;
    %(band)s
    cells[%(k)d] = k_next;
    const long long left = cells[%(cap)d] - done;
    const unsigned int g = (fin != all) && (left > 0);
    go[0] = g;
    if (g) {
        const long long w = k_next < left ? k_next : left;
        word[0] = w;
        hist[w] += 1;
    }
}
"""


def _next_k(k: int, newly: int, live: int, k_max: int, k_min: int = AUTO_K_MIN) -> int:
    """The band of the policy, host face (the twin of :data:`_BAND_SRC`).
    ``k_min`` defaults to :data:`AUTO_K_MIN`; a caller that pins
    ``k_min == k_max`` (a batch skipping compaction
    because it fits inside one resident wave) makes both arms of the band a
    no-op, so ``k`` never leaves ``k_max`` once seeded there."""
    if newly * 16 < live:
        return min(2 * k, k_max)
    if newly * 4 > live:
        return max(k // 2, k_min)
    return k


def _host_policy(cells, finished, total, word, go, hist, k_max: int,
                 k_min: int = AUTO_K_MIN) -> None:
    """One policy step on numpy cells: the twin of the device kernel."""
    fin, all_ = int(finished[0]), int(total[0])
    live, newly = all_ - fin, fin - int(cells[_FIN_AT])
    done = int(cells[_DONE]) + int(word[0])
    cells[_FIN_AT] = fin
    cells[_DONE] = done
    k_next = _next_k(int(cells[_K]), newly, live, k_max, k_min)
    cells[_K] = k_next
    left = int(cells[_CAP]) - done
    g = fin != all_ and left > 0
    go[0] = 1 if g else 0
    if g:
        w = min(k_next, left)
        word[0] = w
        hist[w] += 1


def _policy_kernel(k_max: int, k_min: int = AUTO_K_MIN):
    """The device policy as one single-thread cupy kernel."""
    import cupy

    src = _POLICY_SRC % {"kmin": k_min, "kmax": k_max, "k": _K,
                         "fin_at": _FIN_AT, "done": _DONE, "cap": _CAP,
                         "band": _BAND_SRC}
    return cupy.RawKernel(src, "eagle_until_done_policy")


def _run_host(runner) -> None:
    """The host branch: the host team's loop (``eagle._host_loop``)."""
    from ._host_loop import run_host

    run_host(runner)


def _select_step(plans, planes) -> list:
    """Each plan of a several-plan step, resolved to the plan it runs as:
    an :class:`eagle.plan.AutoPlan` picks its target from where the step's
    data lives."""
    from .plan import residency

    if not any(hasattr(p, "select") for p in plans):
        return plans
    where = residency({nm: v for nm, v in planes.items()
                       if hasattr(v, "shape") or hasattr(v, "__cuda_array_interface__")},
                      door="eagle.until_done", what="step")
    return [getattr(p, where) if hasattr(p, "select") else p for p in plans]


def _step_planes(plans, specs, planes) -> tuple:
    """``(planes, n)``: one sample's head-shaped arrays as batch-of-one
    views (same memory, written in place), and the batch size. A
    several-plan step shares one namespace: a name is one plane in every
    plan that declares it; an undeclared name is refused."""
    from ._plan_pack import _n_from_kw
    from ._plan_planes import _adapt_planes

    if len(plans) == 1:
        adapted, _ = _adapt_planes(plans[0].plugin, specs[0], planes, door="bind",
                                   single_only=True)
        return adapted, _n_from_kw(specs[0], adapted)
    declared = {nm for spec in specs for _role, nm in spec}
    unknown = sorted(set(planes) - declared)
    if unknown:
        raise ValueError(
            f"eagle.until_done: {unknown} name no plane of the step's plans "
            f"(their planes: {sorted(declared)})")
    views = {}
    for plan, spec in zip_strict(plans, specs):
        sub = {nm: planes[nm] for _role, nm in spec if nm in planes}
        adapted, _ = _adapt_planes(plan.plugin, spec, sub, door="bind",
                                   single_only=True)
        for nm, value in adapted.items():
            views.setdefault(nm, value)
    union = tuple(dict.fromkeys(pair for spec in specs for pair in spec))
    return views, _n_from_kw(union, views)


class Runner:
    """A bound, built run-until-done loop over one plan, or a step of
    several plans launched in order (see :func:`until_done`).

    Attributes: :attr:`step` (the :class:`eagle.plan.BoundPlan`, or a
    tuple), :attr:`loop`, :attr:`pipeline` (the device
    :class:`eagle.GraphPipeline`, built on first :meth:`run`),
    :attr:`active` (the :class:`eagle.ActiveSet`, or ``None``),
    :attr:`finished`/:attr:`terminated`, :attr:`steps_per_launch` (``K``,
    or ``"auto"``), :attr:`every`/:attr:`launches_per_compaction`. For an
    automatic artifact :attr:`fused_steps` is the device steps-per-launch
    word and :attr:`policy` reads the policy's cells (a sync).
    :attr:`host_policy` picks how a host run paces automatic launches:
    ``"auto"`` (default: ``"tiled"`` when the entry's fused side is tiled,
    else ``"measured"``), ``"tiled"`` (``steps_max`` steps per launch),
    ``"measured"`` (sweeps one step per launch until a timed fused probe
    wins) or ``"band"`` (the device's band). :attr:`host_threads` is the
    host team size: ``None`` (default) lets eagle pick it (one thread per
    physical core for a short run, every logical CPU for a long one — see
    :func:`eagle._host_loop.host_team_size`), an int forces it. Planes are held
    by reference: a run writes the caller's arrays in place; a new binding
    is a new runner."""

    __slots__ = ("step", "loop", "pipeline", "active", "finished", "terminated",
                 "n", "steps_per_launch", "every", "launches_per_compaction",
                 "fused_steps", "auto", "host_policy", "host_threads",
                 "_device", "_xp", "_total", "_max_steps", "_k_max", "_k_min", "_k0",
                 "_cells", "_go", "_hist",
                 "_compactions", "_bump", "_build_s",
                 "_fast_mode", "_fast_counter", "_fast_grid", "_fast_block", "_fast_util",
                 "_fast_util_on", "_fast_exact", "_fast_views",
                 "_fast_counts", "_fast_pick", "_fast_wall", "_fast_runs", "_fast_last", "_fast_probe",
                 "_fast_prior", "_fast_ev", "_fast_prep",
                 "_fast_ratio")

    @_layout.door
    def __init__(self, plan, *, max_steps, every=None, reorder=None, **planes):
        # Internal only (never a bound plane name, popped before the
        # plane-binding machinery below ever sees it): forces
        # :func:`_fast_mode_for`'s decision for the card's explicit-arm row
        # -- "fused_one" / "persist" / False (off,
        # keep today's band/WHILE-graph path even when capacity would pick
        # a fast mode). Omitted, the decision is :func:`_fast_mode_for`'s
        # own, as for every other caller; no public signature changes.
        fast_mode_override = planes.pop("_fast_mode", None)
        # Internal only, same convention as ``_fast_mode`` above: turns on the
        # persist entry's lane-utilisation counter (off by default -- see
        # :meth:`_build_fast`/:meth:`_run_fast`), for a card arm that still
        # wants :attr:`RunReport.lane_utilisation` filled.
        lane_utilisation_override = planes.pop("_lane_utilisation", None)
        # Internal only: ``False`` keeps the size pick for every run instead of
        # timing the other entry (see :meth:`_pick_entry`).
        self._fast_probe = planes.pop("_fast_probe", True)
        plans = list(plan) if isinstance(plan, (list, tuple)) else [plan]
        if not plans:
            raise ValueError("eagle.until_done: the step is empty; pass a plan or a "
                             "list of plans")
        if len(plans) == 1:
            select = getattr(plans[0], "select", None)
            if select is not None:      # an eagle.plan.AutoPlan: the data decides
                plans[0] = select(planes)
        else:
            plans = _select_step(plans, planes)
        names = [_kernel_name(p.plugin) for p in plans]
        specs = [tuple(tuple(pair) for pair in p.plugin.arg_spec) for p in plans]
        max_steps = _check_count("max_steps", max_steps)
        finishes = [_finish_of(p.plugin, nm) for p, nm in zip_strict(plans, names)]
        finishers = [i for i, f in enumerate(finishes) if f is not None]
        if not finishers:
            raise ValueError(
                f"eagle.until_done: {names[-1]} does not finish its own samples (its "
                "sidecar carries no 'finish'); drive it through the explicit path: "
                "eagle.repeat_while(step, eagle.SkipGuard(done, 0, total, 0), "
                "max_iters) with your own stop rule")
        structures = {getattr(p.structure, "name", None) for p in plans}
        for structure in structures:
            if structure not in _STRUCTURES:
                raise ValueError(
                    f"eagle.until_done: runs a DeviceKernel or HostTeam plan, not "
                    f"{structure!r}")
        if len(structures) > 1:
            raise ValueError(
                "eagle.until_done: the plans of one step run in one place; got "
                f"{sorted(structures)}")
        device = _STRUCTURES[structures.pop()]
        xp = _array_module(device)
        for i in finishers:
            _require_counter_slot(specs[i], names[i])
        for reserved in _RESERVED:
            if reserved in planes:
                raise ValueError(
                    f"eagle.until_done: {reserved!r} is allocated and bound by the "
                    "runner; drop it from the call")
        finish, name = finishes[finishers[0]], names[finishers[0]]
        mask_name = finish["mask"]
        for i in finishers:
            if ("terminated", mask_name) not in specs[i]:
                raise ValueError(
                    f"eagle.until_done: {names[i]}'s 'finish' names mask "
                    f"{mask_name!r}, which its arg_spec does not declare as a "
                    "'terminated' plane")
        compacting = all({MAP_PLANE, COUNT_PLANE}
                         <= {nm for role, nm in spec if role == "lookup"}
                         for spec in specs)
        if not compacting and (every is not None or reorder is not None):
            raise ValueError(
                f"eagle.until_done: {name} is a plain artifact (it reads no active "
                "set), so every=/reorder= do not apply; build it with "
                "Guard(active_set=True) to compact")
        k = finish["steps"]
        if len(plans) > 1:
            for i in finishers:
                if finishes[i]["steps"] not in (FINISH_AUTO, 1):
                    raise ValueError(
                        f"eagle.until_done: {names[i]} advances "
                        f"{finishes[i]['steps']} steps per launch, but each plan of "
                        "a several-plan step takes one step per launch")
            k = 1
        auto = k == FINISH_AUTO
        if auto:
            if every is not None:
                raise ValueError(
                    f"eagle.until_done: {name} picks its steps per launch at run "
                    "time (steps='auto') and compacts after every launch in which "
                    "a sample finished; every= does not apply, drop it (or build "
                    "the kernel with steps=1 for a fixed compaction cadence)")
            if ("lookup", FUSED_STEPS_PLANE) not in specs[0]:
                raise ValueError(
                    f"eagle.until_done: {name} declares steps 'auto' but its "
                    f"arg_spec binds no ('lookup', {FUSED_STEPS_PLANE!r}) word to "
                    "read its steps per launch from")
        else:
            every = _cadence(name, every, k) if compacting else None

        planes, n = _step_planes(plans, specs, planes)
        if mask_name not in planes:
            planes[mask_name] = xp.zeros(n, dtype=xp.bool_)
        mask = planes[mask_name]
        # the finished count is word [2] of the fast entries' counter block
        # (see _build_fast): one memset zeroes it with the other counters and
        # one read returns it with them
        finished = xp.zeros(6, dtype=xp.uint32)[2:3]
        word = xp.zeros(1, dtype=xp.int64) if auto else None
        active = None
        if compacting:
            active = ActiveSet(mask, reorder=reorder, _lazy_scratch=True,
                               _lazy_map=True)
            if active.theta is not None:
                roles = {}
                for spec in specs:
                    for role, nm in spec:
                        roles.setdefault(nm, role)
                owned = [planes[nm] for nm, role in roles.items()
                         if role in PER_SAMPLE_ROLES and nm != mask_name
                         and hasattr(planes.get(nm), "shape")]
                active.own(*owned)

        bound = []
        for plan_i, spec in zip_strict(plans, specs):
            if len(plans) == 1:
                bind = dict(planes)
            else:
                bind = {nm: planes[nm] for _role, nm in spec if nm in planes}
            if ("lookup", FINISHED_PLANE) in spec:
                bind[FINISHED_PLANE] = finished.view(xp.int32)
            if ("lookup", FUSED_STEPS_PLANE) in spec:
                bind[FUSED_STEPS_PLANE] = (word if auto
                                           else xp.ones(1, dtype=xp.int64))
            if active is not None:
                bind.update(active.planes())
            bound.append(plan_i.bind(**bind))

        self.step = bound[0] if len(bound) == 1 else tuple(bound)
        self.n = bound[0].n
        self.terminated = mask
        self.finished = finished
        self.active = active
        self.pipeline = None
        self.steps_per_launch = k
        self.auto = auto
        self.fused_steps = word
        self.host_policy = "auto"
        self.host_threads = None
        # `every` is the compaction cadence in steps; one iteration runs
        # `launches_per_compaction` launches of K steps, then compacts
        if auto:
            self.every, self.launches_per_compaction = None, 1
        else:
            self.every = every if compacting else k
            self.launches_per_compaction = every // k if compacting else 1
        self._max_steps = max_steps
        self._k_max = finish.get("steps_max", k if not auto else None)
        self._k_min = AUTO_K_MIN
        self._k0 = AUTO_K0
        self._cells = self._go = self._hist = None
        self._device = device
        self._xp = xp
        self._total = xp.asarray([self.n], dtype=xp.uint32)
        self._compactions = xp.zeros(1, dtype=xp.uint32)
        self._build_s = 0.0
        self._bump = None
        self._fast_mode = None
        self._fast_pick = self._fast_last = self._fast_prior = None
        self._fast_ev = None
        self._fast_prep = {}
        self._fast_ratio = FUSED_MIN_STEP_RATIO
        self._fast_wall = {}
        self._fast_runs = 0
        self._fast_counter = self._fast_grid = self._fast_block = self._fast_util = None
        self._fast_exact = False
        self._fast_counts = False
        self._fast_views = None
        self._fast_util_on = False
        if device:
            self._bump = _bump_kernel()
            self._bump(xp.zeros(1, dtype=xp.uint32), size=1)  # compile before capture

        launches = tuple(b.launch for b in bound)

        def launch():
            for one in launches:
                one()

        if auto:
            self._build_auto(launch, max_steps, fast_mode_override,
                             lane_utilisation_override)
            return
        guard = SkipGuard(finished, 0, self._total, 0)
        if active is not None:
            self._materialize_map()
            active.prepare_compaction()
        if active is None:
            self.loop = repeat_while(launch, guard, -(-max_steps // k))
        else:
            parts = list(compaction_body(launch, active,
                                         every=self.launches_per_compaction,
                                         finished=finished, steps_per_call=k))
            parts[1] = self._counted(parts[1])
            self.loop = repeat_while(tuple(parts), guard, -(-max_steps // every))

    def _build_auto(self, launch, max_steps: int, fast_mode_override=None,
                    lane_utilisation_override=None) -> None:
        """The automatic loop: ``(launch, compact?, reorder?, policy)`` under
        the ``go`` guard, capped at ``ceil(max_steps / AUTO_K_MIN) + 1``
        iterations (the policy's ``go`` is the real stop).

        A compacting (active-set) batch that fits inside
        one resident wave (``n <= capacity``, :meth:`BoundPlan.device_capacity`
        — SMs x resident threads at this kernel's registers) never needs the
        active-set map rebuilt: the map exists to refill warps the SM would
        otherwise idle, which only matters once the batch needs more warps
        than the device can hold resident at once. Below that capacity,
        compaction's rebuild/copy/policy nodes are pure cost, so
        this skips the compaction body entirely and pins the band's floor at
        its own ceiling (``k_min = k_max``), which makes both arms of
        :data:`_BAND_SRC`/:func:`_next_k` a no-op once K reaches ``k_max`` --
        every launch then takes as many steps as the kernel's own clamp
        allows, seeded there from the first launch via ``_k0``. A batch
        ABOVE capacity keeps today's band and compaction-every-launch cadence
        unchanged (the device rule's further "stop rebuilding once
        live/(32*SMs) <= W_knee" refinement needs a live-count THRESHOLD
        guard that eagle._conditional.SkipGuard does not express today --
        deferred, not wired here).

        A single-kernel device, contiguous-partition run with no reorder
        REQUESTED (``self.active is None``, or an active set the caller
        never asked to reorder, ``self.active.theta is None``) never
        reaches any of the above: see :meth:`_build_fast` -- one launch, no
        WHILE graph, no policy kernel, no map (rebuild or otherwise, even
        for a compacting artifact -- it launches the map entry's
        contiguous siblings instead, :meth:`~eagle._plan_binding.BoundPlan.launch_range`/
        ``launch_persist``, which bind ``active_map``/``active_count`` the
        SAME way but never read them), chosen by :func:`_fast_mode_for` (or
        ``fast_mode_override``, internal-only: "fused_one" / "persist"
        forces that mode past what :func:`_fast_mode_for` would pick,
        ``False`` keeps today's band path even where it would pick one --
        the card's explicit persist-arm row, no public API change). An
        artifact whose caller passed ``reorder=theta`` keeps today's band
        path unconditionally -- the fast path has nowhere to run a reorder."""
        eligible = (self._device and len(self.step.partitions) == 1
                   and (self.active is None or self.active.theta is None))
        if fast_mode_override not in (None, False, "persist", "fused_one"):
            raise ValueError(
                "eagle.until_done: _fast_mode must be 'persist', 'fused_one', False "
                f"or None; got {fast_mode_override!r}")
        probe = False
        if fast_mode_override is not None:
            if fast_mode_override is False:
                mode = None
            elif not eligible:
                raise ValueError(
                    "eagle.until_done: _fast_mode is internal-only and needs a "
                    "single-kernel, contiguous, device run with no reorder "
                    f"requested; got device={self._device}, "
                    f"reorder={self.active is not None and self.active.theta is not None}, "
                    f"partitions={len(self.step.partitions)}"
                )
            elif fast_mode_override == "persist" and self.step.persist_entry() is None:
                raise ValueError(
                    "eagle.until_done: _fast_mode='persist' was forced but this "
                    "artifact carries no persist entry"
                )
            elif (fast_mode_override == "fused_one" and self.active is not None
                  and self.step.range_entry() is None):
                raise ValueError(
                    "eagle.until_done: _fast_mode='fused_one' was forced but this "
                    "compacting artifact carries no range entry"
                )
            else:
                mode = fast_mode_override
        elif eligible:
            mode = _fast_mode_for(self.step, self.n, compacting=self.active is not None)
            # above capacity with both entries present, the other entry can be
            # timed against the size pick
            probe = (self._fast_probe is not False and mode is not None
                     and self.n > self.step.device_capacity()
                     and self.step.persist_entry() is not None
                     and self.step.range_entry() is not None)
        else:
            mode = None
        if mode is not None:
            self._build_fast(mode, lane_utilisation_override, probe)
            return
        if self.active is not None:
            self._materialize_map()
            self.active.prepare_compaction()
        xp = self._xp
        self._cells = xp.zeros(4, dtype=xp.int64)
        self._go = xp.zeros(1, dtype=xp.uint32)
        self._hist = xp.zeros(self._k_max + 1, dtype=xp.int64)
        cells, go, hist, word = self._cells, self._go, self._hist, self.fused_steps
        finished, total, k_max = self.finished, self._total, self._k_max

        skip_compaction = False
        # a reorder lives in the compaction body, so a reordering set keeps it
        if self._device and self.active is not None and self.active.theta is None:
            cap = self.step.device_capacity()
            if cap > 0 and self.n <= cap:
                skip_compaction = True
        if skip_compaction:
            self._k_min = k_max
            self._k0 = k_max
        k_min = self._k_min

        if self._device:
            kernel = _policy_kernel(k_max, k_min)
            args = (finished, total, cells, word, go, hist)
            # compile (and check) before any capture, on scratch cells
            kernel((1,), (1,), (xp.zeros(1, dtype=xp.uint32), total,
                                xp.zeros(4, dtype=xp.int64),
                                xp.zeros(1, dtype=xp.int64),
                                xp.zeros(1, dtype=xp.uint32),
                                xp.zeros(k_max + 1, dtype=xp.int64)))

            def policy():
                kernel((1,), (1,), args)
        else:
            def policy():
                _host_policy(cells, finished, total, word, go, hist, k_max, k_min)
        parts = [launch]
        if self.active is not None and not skip_compaction:
            body = list(compaction_body(launch, self.active, every=1,
                                        finished=finished,
                                        steps_per_call=AUTO_K_MIN))
            body[1] = self._counted(body[1])
            parts = body
        parts.append(policy)
        cap = -(-max_steps // AUTO_K_MIN) + 1
        self.loop = repeat_while(tuple(parts), SkipGuard(go, 0), cap)

    def _materialize_map(self) -> None:
        """The band path reads the active-set map: allocate the deferred one
        and re-point the bound plans at it."""
        if self.active.materialize_map():
            for bound in (self.step if isinstance(self.step, tuple) else (self.step,)):
                if MAP_PLANE in bound.names:
                    bound.rebind(**{MAP_PLANE: self.active.map})

    def _build_fast(self, mode: str, lane_utilisation_override=None,
                    probe: bool = False) -> None:
        """Set up the no-graph shape :func:`_fast_mode_for` picked: no
        ``self.loop``/``self.pipeline`` at all, since one call of
        :meth:`run` IS the whole thing. ``fused_steps`` is written here,
        ONCE (it never changes again -- ``self._max_steps`` is fixed for
        the life of the runner), so :meth:`_run_fast` never writes it.
        ``"persist"`` allocates the lane-stealing counter (memset at each
        :meth:`run`, not here -- the one-time allocation only); the
        lane-utilisation counter is allocated, and later filled, ONLY when
        ``lane_utilisation_override`` asks for it (off by default -- it
        costs 2 same-address 64-bit atomics per warp per step, see
        :meth:`_run_fast`'s docstring). ``probe`` builds the persist
        block/grid whatever ``mode`` is, so :meth:`_pick_entry` can switch
        between the two entries; it also (re)starts the measured walls."""
        self._fast_mode = mode
        self._fast_pick = mode if probe else None
        self._fast_wall = {}
        self._fast_runs = 0
        self._fast_last = self._fast_prior = None
        # the step-fill bound below which the fused entry is never timed: the
        # persist loop's extra bookkeeping (~12 instructions per step) over a
        # step of step_ops instructions, so a heavy step has little to lose
        step_ops = getattr(self.step.plan.plugin, "step_ops", None)
        # an artifact whose fast entries count the samples finished on entry
        self._fast_counts = bool(getattr(self.step.plan.plugin, "entry_counts_finished", False))
        self._fast_ratio = (max(FUSED_MIN_STEP_RATIO, 1.0 - 12.0 / step_ops)
                            if step_ops else FUSED_MIN_STEP_RATIO)
        # the timing events, allocated once: only a probing runner times runs
        self._fast_ev = (self._xp.cuda.Event(), self._xp.cuda.Event()) if probe else None
        self.loop = None  # no WHILE-graph body exists in this mode
        self.fused_steps[0] = self._max_steps
        self._fast_util_on = bool(lane_utilisation_override) and (mode == "persist" or probe)
        xp = self._xp
        # [0] the persist entry's lane-stealing counter, [1] the run's exact
        # step count (both entries max each sample's steps into it): ONE
        # memset zeroes both per run. A plain fused_one with no range entry
        # (a unit built without the fast entries) reports no count.
        # uint32 words: [0] next, [1] steps, [2] finished, [3] pad, [4:6] the
        # uint64 sum of every lane's executed steps (a probing runner's
        # step-fill report): one memset zeroes all, one read returns them
        self._fast_counter = self.finished.base
        self._fast_views = (self._fast_counter[0:1], self._fast_counter[1:2],
                            self._fast_counter[4:6].view(xp.uint64) if probe else None)
        self._fast_exact = (mode == "persist"
                            or self.step.range_entry() is not None)
        # the launch argument blocks, packed once for the entries this runner
        # can take (the run's own cells are fixed for its life)
        steps_cell = self._fast_views[1] if self._fast_exact else None
        stepsum = self._fast_views[2] if self._fast_exact else None
        prep = self._fast_prep = {}
        if (mode == "fused_one" or probe) and steps_cell is not None:
            prep["range"] = self.step.prepare_range(steps=steps_cell, stepsum=stepsum)
            if stepsum is not None:
                prep["range_bare"] = self.step.prepare_range(steps=steps_cell)
        if mode != "persist" and not probe:
            return
        self._fast_util = (xp.zeros(2, dtype=xp.uint64)
                           if self._fast_util_on else None)
        # block/grid: the persistent launch's threads-per-SM rule
        # (eagle._launch_policy.persistent_launch), not a literal here --
        # from the device's issue rate for the step's precision, the batch
        # size, the step budget and the persist entry's own register-limited
        # occupancy.
        import importlib

        from ._launch_policy import persistent_launch

        # by module name, not attribute: eagle exports a `launch` function
        # that shadows the submodule (same reason _plan_binding.py's own
        # .launch() resolves it this way).
        _launch = importlib.import_module("eagle.launch")
        fn = self.step.persist_entry()
        scalar_type = getattr(self.step.plan.plugin, "scalar_type", None) or "float64"
        self._fast_block, self._fast_grid = persistent_launch(
            _launch._device_props(), _launch._kernel_attrs(fn), self.n, self._max_steps,
            scalar_type)
        for key, util in (("persist", None), ("persist_util", self._fast_util)):
            if key == "persist" or util is not None:
                prep[key] = self.step.prepare_persist(
                    grid=self._fast_grid, counter=self._fast_views[0], util=util,
                    block=self._fast_block, steps=steps_cell, stepsum=stepsum)
                if stepsum is not None:
                    prep[key + "_bare"] = self.step.prepare_persist(
                        grid=self._fast_grid, counter=self._fast_views[0], util=util,
                        block=self._fast_block, steps=steps_cell)

    def _pick_entry(self) -> tuple:
        """The entry this run takes and whether it is a probe, as
        ``(mode, probe)``. A pinned or below-capacity runner always takes its
        bind-time mode. Otherwise the first run takes the size pick, the
        second the other entry, and from then on the one with the smaller
        measured launch wall; the loser is timed again when a run's readback
        (``steps``/``finished``) differs from the last probe's, and every
        64 runs. The step fill f is read back after every run, either entry; an f
        under the bound (:data:`FUSED_MIN_STEP_RATIO`, raised by a heavy step's
        ``step_ops``) keeps the run on persist with no probe."""
        pick = self._fast_pick
        if pick is None:
            return self._fast_mode, False
        if self._fast_prior is None:
            return pick, False
        if self._fast_prior < self._fast_ratio:
            return "persist", False
        wall = self._fast_wall
        other = "persist" if pick == "fused_one" else "fused_one"
        if not wall:
            # an untimed run invalidated the walls: the pick has no wall,
            # so the other entry goes first, as a probe
            return other, True
        if pick not in wall:
            return pick, False
        if other not in wall:
            return other, True
        return pick, False

    def _note_run(self, mode: str, wall_ms, steps: int, finished: int, f) -> None:
        """Fold one probing-eligible run into the pick: its step fill ``f``
        (from either entry; ``None`` on a settled run that did not sum its
        steps), its readback and, when it was timed, its wall (``None`` for a
        settled run, which only checks for invalidation and keeps the 64-run
        re-probe cadence)."""
        wall = self._fast_wall
        last, now = self._fast_last, (steps, finished)
        prior = self._fast_prior
        self._fast_runs += 1
        if f is None:
            f = prior
        # a new readback regime, or f crossing the bound either way: the old
        # walls no longer compare
        if last != now or (prior is not None
                           and (f < self._fast_ratio) != (prior < self._fast_ratio)):
            wall.clear()
            self._fast_pick = mode
            self._fast_last = now
            if wall_ms is not None:
                wall[mode] = wall_ms
        elif wall_ms is not None:
            wall[mode] = wall_ms
            if len(wall) == 2:
                self._fast_pick = min(wall, key=wall.get)
        self._fast_prior = f
        if f < self._fast_ratio:
            self._fast_pick = "persist"
            wall.pop("fused_one", None)
        elif self._fast_runs % 64 == 0 and len(wall) == 2:
            wall.pop("persist" if self._fast_pick == "fused_one" else "fused_one")

    def _run_fast(self) -> RunReport:
        """:meth:`run`'s body for a no-graph mode: one launch (fused with
        the whole budget as its word, or the persist entry), then one
        packed readback -- no iterations to count, no compaction, no
        reorder, and (the one host<->device sync this mode ever takes) no
        OTHER sync either.

        ``finished`` is re-synced to the mask EVERY call, fresh
        (:func:`_mask_count`, eagle's own kernel -- never cupy's
        ``count_nonzero``, see there): the public contract (:meth:`run`'s
        own docstring) is that a run continues from the CURRENT state, so a
        caller who edits ``terminated`` between calls without
        :meth:`reset` must see that edit reflected in this call's report
        and launch -- a value cached from a past call (what this method
        used to do) cannot tell an edited mask from an untouched one. There
        is no "already done, skip the launch entirely" shortcut any more
        either, for the same reason: it would need to trust the SAME kind
        of stale cache. A batch that is genuinely still fully finished
        launches again anyway -- harmless, the kernel skips every
        terminated lane -- at the cost of one tiny extra kernel (the mask
        count) beyond the step launch, never an extra SYNC: both are async
        on the current stream, ahead of the one packed read this method
        has always made.

        ``steps`` is exact (``exact_steps=True``): the range and persist
        entries max each sample's steps taken this run into a run-level
        cell (zeroed by the same per-run memset as the persist counter,
        read back in the same packed transfer as ``finished``) -- the
        longest sample's steps, as the band path counts them. Only a plain
        fused_one on a unit without the range entry (``launch()``) falls
        back to the budget OFFERED: exact when the budget was the
        bottleneck, an upper bound otherwise.

        The lane-utilisation counter (:attr:`_fast_util_on`) is read back
        with its own ``.get()`` only when it was asked for at bind
        (:meth:`_build_fast`) -- an extra sync, opted into, never the
        default."""
        xp = self._xp
        lane_utilisation = None
        # ONE device word block per run: [0] the persist entry's
        # lane-stealing counter, [1] the run's exact step count, [2] the
        # finished count -- zeroed by one async memset, read back by ONE
        # transfer.
        counter = self._fast_counter
        rt = xp.cuda.runtime
        stream = xp.cuda.get_current_stream().ptr
        mode, probing = self._pick_entry()
        # the range/persist entry counts the samples finished on entry
        # itself, as the step counts those finishing in the run: the memset
        # zeroes the count too; else the mask is counted after it
        counts = self._fast_counts and (mode == "persist" or self._fast_exact)
        probed = self._fast_pick is not None
        # only a run that decides is timed: the first, a probe, and the
        # run(s) after an invalidation cleared the walls; a settled run is
        # not (an event pair costs more than the launch policy is worth)
        prior = self._fast_prior
        timed = probed and (probing or prior is None
                            or (prior >= self._fast_ratio and len(self._fast_wall) < 2)
                            or (prior < self._fast_ratio and "persist" not in self._fast_wall))
        if timed:
            t0, t1 = self._fast_ev
            t0.record()
        # the step sum: on every run that decides, and on every
        # STEP_FILL_EVERY-th settled one (a batch whose spread changed under
        # the same readback is seen within that many runs); the other runs
        # pass no sum cell, so the device adds nothing
        summed = probed and (timed or self._fast_runs % STEP_FILL_EVERY == 0)
        bare = "" if summed or not probed else "_bare"
        if self._fast_exact:
            rt.memsetAsync(counter.data.ptr, 0, 24 if summed else 12, stream)
        elif counts:
            rt.memsetAsync(self.finished.data.ptr, 0, self.finished.nbytes, stream)
        if not counts:
            _mask_count(xp, self._device, self.finished, self.terminated)
        next_cell, steps_cell, stepsum = self._fast_views
        if not summed:
            stepsum = None
        if not self._fast_exact:
            steps_cell = None
        if mode == "fused_one":
            if steps_cell is not None:
                # the range entry: the kernel's own two paths over the
                # contiguous range (for a compacting artifact the map
                # entry's own .launch() is active-map-indexed), plus the
                # exact step count.
                self.step.launch_range(steps=steps_cell,
                                       prepared=self._fast_prep["range" + bare])
            else:
                self.step.launch()
        else:
            util = None
            key = "persist"
            if self._fast_util_on:
                self._fast_util[...] = 0
                util = self._fast_util
                key = "persist_util"
            self.step.launch_persist(grid=self._fast_grid,
                                     counter=next_cell, util=util,
                                     block=self._fast_block, steps=steps_cell,
                                     stepsum=stepsum, prepared=self._fast_prep[key + bare])
        if timed:
            t1.record()
        cells = [(counter, i) for i in (range(1, 6) if summed else (1, 2))]
        words = _packed_read(xp, self._device, cells)
        steps_read, finished = words[0], words[1]
        if probed:
            # the step fill f = (sum of the lanes' steps) / (n * longest lane's steps)
            f = None
            if summed:
                total = words[3] | (words[4] << 32)
                f = total / (self.n * steps_read) if steps_read > 0 else 1.0
            self._note_run(mode, xp.cuda.get_elapsed_time(t0, t1) if timed else None,
                           steps_read, finished, f)
        if self._fast_util_on and mode == "persist":
            active_lanes, warp_iters = (
                self._fast_util.get() if self._device else self._fast_util
            )
            if warp_iters > 0:
                lane_utilisation = float(active_lanes) / float(warp_iters)
        done = finished == self.n
        steps, exact = ((steps_read, True) if steps_cell is not None
                        else (self._max_steps, not done))
        return RunReport(
            n=self.n, finished=finished, done=done, launches=1,
            steps=steps, compactions=0, reorders=0, build_s=0.0,
            launches_by_k={self._max_steps: 1}, exact_steps=exact,
            lane_utilisation=lane_utilisation, mode=mode, probe=probing)

    @property
    def policy(self) -> dict | None:
        """The automatic loop's policy cells as host ints (a device sync):
        ``k``, ``finished_at_launch``, ``steps_done``, ``max_steps``,
        ``go``; ``None`` for a fixed artifact."""
        if not self.auto:
            return None
        cells = self._cells.get() if self._device else self._cells
        return {"k": int(cells[_K]), "finished_at_launch": int(cells[_FIN_AT]),
                "steps_done": int(cells[_DONE]), "max_steps": int(cells[_CAP]),
                "go": self._word(self._go)}

    def _seed(self) -> None:
        """Seed the automatic loop's cells from the host before a run (the
        counter is already re-synchronised to the mask)."""
        xp = self._xp
        fin = self._word(self.finished)
        first = min(self._k0, self._max_steps)
        go = fin != self.n and self._max_steps > 0
        self._cells[...] = xp.asarray([self._k0, fin, 0, self._max_steps],
                                      dtype=xp.int64)
        self.fused_steps[0] = first
        self._go[0] = 1 if go else 0
        self._hist.fill(0)
        if go:
            self._hist[first] = 1

    def _counted(self, tail: Skippable) -> Skippable:
        """The compaction tail, also counting each compaction it runs."""
        cell = self._compactions
        if self._device:
            bump = self._bump

            def compact():
                tail.step()
                bump(cell, size=1)
        else:
            def compact():
                tail.step()
                cell[0] += 1
        return Skippable(compact, tail.guard)

    def __repr__(self) -> str:
        where = "device" if self._device else "host"
        if self.auto:
            return (f"Runner({where}, n={self.n}, steps_per_launch='auto', "
                    f"steps_max={self._k_max}, active={self.active is not None})")
        return (f"Runner({where}, n={self.n}, "
                f"steps_per_launch={self.steps_per_launch}, every={self.every}, "
                f"launches_per_compaction={self.launches_per_compaction}, "
                f"active={self.active is not None})")

    def _word(self, arr) -> int:
        value = arr[0]
        get = getattr(value, "get", None)
        return int(get()) if get is not None else int(value)

    def run(self) -> RunReport:
        """Run until every sample is done (or the cap is reached) and
        report; the counter is re-synced to the mask first, so a run
        continues from the current state (a reordering set's planes land
        back in sample order).

        The counters this needs before/after the launch (not
        counting the automatic policy's own ``steps_done``/histogram/
        ``fused_steps`` reads, still separate -- see :meth:`_build_auto`) are
        read as ONE packed transfer per snapshot (:func:`_packed_read`)
        instead of a separate full round-trip sync each: a plain,
        non-compacting run (``self.active is None``) needs no PRE snapshot
        at all (``_compactions`` can only ever be 0 without an active set,
        so it is never even read), and the POST snapshot folds
        ``iterations``/``finished``/``compactions``/``reorders`` into one.

        A fast mode (:meth:`_build_fast`) skips all of the above: one
        launch IS the run, so :meth:`_run_fast` handles it entirely."""
        if self._fast_mode is not None:
            return self._run_fast()
        xp = self._xp
        _mask_count(xp, self._device, self.finished, self.terminated)
        pre_cells = []
        if self.active is not None:
            pre_cells.append((self._compactions, 0))
            if self._reorders():
                pre_cells.append((self.active.reorders, 0))
        pre = _packed_read(xp, self._device, pre_cells) if pre_cells else ()
        compactions0 = pre[0] if pre_cells else 0
        reorders0 = pre[1] if len(pre_cells) > 1 else 0
        if self.auto:
            self._seed()
        if self._device:
            if self.pipeline is None:
                from .pipeline import GraphPipeline

                t0 = time.perf_counter()
                self.pipeline = GraphPipeline().add(self.loop).build()
                xp.cuda.Device().synchronize()
                self._build_s = time.perf_counter() - t0
            self.pipeline.launch()
        else:
            _run_host(self)
        post_cells = [(self.loop.iteration_index, 0), (self.finished, 0)]
        if self.active is not None:
            post_cells.append((self._compactions, 0))
            if self._reorders():
                post_cells.append((self.active.reorders, 0))
        launches, finished, *rest = _packed_read(xp, self._device, post_cells)
        reorders = 0
        compactions_post = 0
        if self.active is not None:
            compactions_post = rest[0]
            if self._reorders():
                reorders = rest[1] - reorders0
                self.active.restore()
        if self.auto:
            hist = self._hist.get() if self._device else self._hist
            nz = hist.nonzero()[0]
            by_k = dict(zip(nz.tolist(), hist[nz].tolist()))
            steps = self.policy["steps_done"]
            # exact when nothing launched, the cap stopped an unfinished
            # sample, or the last launch took one step
            exact = (launches == 0 or finished != self.n
                     or self._word(self.fused_steps) == 1)
        else:
            ran = launches * self.launches_per_compaction
            by_k = {self.steps_per_launch: ran} if ran else {}
            steps = launches * self.every
            exact = self.steps_per_launch == 1 and self.launches_per_compaction == 1
        return RunReport(
            n=self.n, finished=finished, done=finished == self.n, launches=launches,
            steps=steps,
            compactions=compactions_post - compactions0,
            reorders=reorders, build_s=self._build_s, launches_by_k=by_k,
            exact_steps=exact, mode="band")

    def _reorders(self) -> bool:
        return self.active is not None and self.active.theta is not None

    def reset(self) -> None:
        """Clear the mask/counter/active set: the next :meth:`run` starts the batch over."""
        if self._reorders():
            self.active.restore()
        if self._device and self.terminated.flags.c_contiguous:
            # two async memsets on the current stream, not two fill kernels
            rt = self._xp.cuda.runtime
            stream = self._xp.cuda.get_current_stream().ptr
            rt.memsetAsync(self.terminated.data.ptr, 0, self.terminated.nbytes, stream)
            rt.memsetAsync(self.finished.data.ptr, 0, self.finished.nbytes, stream)
        else:
            self.terminated.fill(False)
            self.finished.fill(0)
        if self.active is not None:
            self.active.reset()


def _cadence(name: str, every, k: int) -> int:
    """The compaction cadence in steps: ``every`` (default: the smallest
    multiple of ``k`` at least :data:`DEFAULT_EVERY`)."""
    if every is None:
        return k * -(-DEFAULT_EVERY // k)
    if isinstance(every, bool) or not isinstance(every, int) or every < 1:
        raise ValueError(
            f"eagle.until_done: every is a positive int (steps), got {every!r}")
    if every % k:
        lo, hi = k * (every // k), k * -(-every // k)
        nearest = f"every={hi}" if lo == 0 else f"every={lo} or every={hi}"
        raise ValueError(
            f"eagle.until_done: {name} advances {k} steps per launch, so the "
            f"compaction cadence every={every} (in steps) is not a whole number of "
            f"launches; use {nearest}")
    return every


def until_done(plan, *, max_steps: int, every: int | None = None, reorder=None,
               **planes) -> Runner:
    """Bind ``plan`` and build the loop that runs it until every sample is
    done; returns the :class:`Runner` (``.run()`` runs it).

    ``**planes`` binds every name of the plan's ``arg_spec`` by
    :meth:`eagle.plan.Plan.bind`'s rules, except the reserved
    :data:`FINISHED_PLANE`, ``active_map`` and ``active_count`` (the
    runner allocates these); the ``terminated`` mask is all-false when
    omitted. ``plan`` may be an :class:`eagle.plan.AutoPlan` (residency
    picks host/device), or a list of plans run as one step, in order,
    over one namespace of planes; each takes one step per launch, at
    least one finishes the samples, and they share one mask and guard.

    ``max_steps`` caps the steps a sample may take (a compacting loop may
    overrun by up to ``every - 1`` steps; :attr:`RunReport.steps` counts
    them). ``Guard(active_set=True)`` compacts every ``every`` steps (a
    multiple of ``K``, at least 4, default the smallest at least 16), and
    ``reorder=theta`` also reorders the bound planes; both are refused on
    a plain artifact.

    An automatic artifact (``steps="auto"``) runs the policy loop instead
    (see the module docstring): the cap is exact, ``every=`` is refused,
    reorder only with ``reorder=theta``, and the runner owns the
    :data:`FUSED_STEPS_PLANE` word."""
    return Runner(plan, max_steps=max_steps, every=every, reorder=reorder, **planes)


def run_until_done(plan, *, max_steps: int, every: int | None = None, reorder=None,
                   **planes) -> RunReport:
    """:func:`until_done` then :meth:`Runner.run`: the one-call form."""
    return until_done(plan, max_steps=max_steps, every=every, reorder=reorder,
                      **planes).run()
