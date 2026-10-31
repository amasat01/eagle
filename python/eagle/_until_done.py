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

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import NamedTuple

from . import _alloc
from ._active_set import (
    COUNT_PLANE,
    DEFAULT_EVERY,
    MAP_PLANE,
    ActiveSet,
    _view,
    compaction_body,
)
from . import _core, _lane_bind, _layout, _split, _stream, arrays
from ._busy import BusyGuard
from ._compat import zip_strict
from ._conditional import SkipGuard, Skippable, _host_scalar, repeat_while
from .graph import LowerCtx, Node, get_node_kind
from .roles import OUTPUT_ROLES, PER_SAMPLE_ROLES
from .sidecar import FINISH_AUTO, FINISH_COUNTER, read_finish, refuse_derived

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

#: The planes a recording / event-refining / reverse run allocates and binds
#: itself. Reserved ONLY while that mode is on (see :func:`_reserved_names`);
#: :data:`_RESERVED` never grows, so a model that names a plane ``tape`` and
#: runs without them keeps working.
_CONDITIONAL_RESERVED = ("applied_steps", "t_stop", "tape", "event_status")


def _reserved_names(*, record: bool = False, events=None,
                    reverse: bool = False) -> tuple:
    """The plane names the runner owns for this run: :data:`_RESERVED`, plus
    :data:`_CONDITIONAL_RESERVED` when ``record=``, ``events="refine"`` or a
    reverse is on."""
    if record or reverse or events == "refine":
        return _RESERVED + _CONDITIONAL_RESERVED
    return _RESERVED

# The policy cells (int64): the band's current k, the counter at the last
# launch, the steps run so far, the step cap, and R in thousandths (a
# compaction's cost in steps of a full-live launch).
_K, _FIN_AT, _DONE, _CAP, _R, _MAPPED = range(6)
RATE_SCALE = 1000

#: R until the cost model can price the device: the middle of the measured range.
AUTO_RATE0 = 8.0
#: Issue slots per step the cost model assumes when the kernel does not say.
AUTO_STEP_OPS = 33


def ops_per_step(plugin) -> float:
    """The step's operation count the launch rules read, from the artifact's
    sidecar (``fp64_ops_ir``, else ``ops_per_step``, else ``step_ops``);
    ``AUTO_STEP_OPS`` when it carries none. The compaction cadence and the
    persistent chunk both price a step with it."""
    for key in ("fp64_ops_ir", "ops_per_step", "step_ops"):
        value = getattr(plugin, key, None)
        if value:
            return float(value)
    return float(AUTO_STEP_OPS)

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
    ``slot_fill`` (the steps run over the persist entry's chunk slots, 32
    lanes times the chunk length per warp per chunk head, in ``(0, 1]``; a
    lane waiting masked for the next head is an empty slot) is set only
    for a persist-entry run that counted it; ``None`` otherwise
    (every other mode, or a persist run that skipped the counter).
    ``mode`` is the launch shape this run actually took: ``"fused_one"``/
    ``"persist"`` for a no-graph fast-path run, ``"band"`` for the
    WHILE-graph/policy path (the only mode before the fast path existed).
    ``probe`` is true for a fast-path run taken to time the entry the batch
    is not currently using (see :meth:`Runner._pick_entry`); ``timed`` is
    true for a fast-path run that carried the timing event pair (a probe, the
    first run, or a run after the walls were invalidated), so a caller can
    warm until a run is neither ``probe`` nor ``timed``.
    ``dispatch`` is ``None`` unless the run decided a dispatch group
    (``dispatch=``): ``{group: (policy, mix, gain)}`` with the policy in force
    at the end, the samples that took each branch at the last look and the
    predicted gain of splitting (above 1 the split wins); ``class_sorts``
    counts the sorts of samples by branch a split run made. ``mapped`` is the
    samples the launches ran over, summed over the launches: ``n`` for a
    fast-path run (one launch over the whole batch), the live samples of each
    launch for a compacting run, ``launches * n`` for one that does not
    compact; ``0`` for a report that does not count them."""

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
    slot_fill: float | None = None
    mode: str = "band"
    probe: bool = False
    timed: bool = False
    dispatch: dict | None = None
    class_sorts: int = 0
    mapped: int = 0


def _device_token(device: bool, planes) -> str:
    """Where the run's own arrays live: ``"cpu"``, or ``"cuda:N"`` for the GPU the caller's device planes are
    on (the current one when it passed none)."""
    if not device:
        return "cpu"
    for value in planes.values():
        if hasattr(value, "shape") or hasattr(value, "__cuda_array_interface__"):
            where = _view(value).device
            if where.startswith("cuda"):
                return where
    from .launch import _current_device_ordinal

    return f"cuda:{_current_device_ordinal()}"


#: A settled runner sums its steps (the step fill f) on one run in this many;
#: the runs between pass no sum cell and pay no device atomics.
STEP_FILL_EVERY = 8
#: the other entry replaces the pick only when its wall is below this fraction of the pick's
PICK_HYSTERESIS = 0.98
#: A batch of at most ``capacity / PICK_FUSED_DIVISOR`` samples is a fused-entry
#: batch and one of at least ``PICK_PERSIST_MULTIPLE * capacity`` a persist-entry
#: batch, whatever the readback says: the entry that wins there wins by a wide
#: margin on every device measured, so the other is never timed. Between the
#: two the entries trade places with the step fill and the number of steps.
PICK_FUSED_DIVISOR = 4
PICK_PERSIST_MULTIPLE = 8


def _entry_regime(persist_capacity: int, wave_capacity: int, n: int) -> str:
    """Which entry a batch of ``n`` samples is sure to want: ``"persist"`` from
    ``PICK_PERSIST_MULTIPLE`` times the threads the persist entry keeps resident
    across the device (``persist_capacity``, :meth:`Runner._persist_capacity`),
    ``"fused"`` up to ``1 / PICK_FUSED_DIVISOR`` of what one fused wave holds
    (``wave_capacity``, :meth:`~eagle._plan_binding.BoundPlan.device_capacity`),
    else ``"band"``: the readback cannot say and both entries are timed. An
    unknown capacity (``<= 0``) never makes a batch sure."""
    if persist_capacity > 0 and n >= PICK_PERSIST_MULTIPLE * persist_capacity:
        return "persist"
    if wave_capacity > 0 and n * PICK_FUSED_DIVISOR <= wave_capacity:
        return "fused"
    return "band"


class _Pick(NamedTuple):
    """What a probing runner knows about its entry pick, as ONE immutable value.
    A runner replaces it as a whole (an atomic reference swap), so a reader on
    another thread sees the old pick or the new one, never a mixture, and the
    run path takes no lock. ``wall`` maps an entry to its timed wall in ms and is
    never modified once published; ``last`` is the readback regime the walls were
    timed in (:func:`_readback_regime`); ``prior`` the last step fill; ``runs``
    the runs folded in; ``ratio`` the model's predicted persist-over-fused wall
    for the last readback (a record only: it never picks); ``settled`` says the
    pick is final for this regime (it came from the process cache or from both
    entries' walls), so neither
    entry is timed again until the regime moves."""

    pick: str | None
    wall: dict
    last: tuple | None
    prior: float | None
    runs: int
    settled: bool
    ratio: float | None = None


_NO_PICK = _Pick(None, {}, None, None, 0, False)


def _readback_regime(steps: int, fill) -> tuple:
    """The readback bucket two runs share when their timings compare: the
    longest sample's steps to the nearest power of two below, and the step fill
    to one decimal."""
    return (max(int(steps), 1).bit_length(), None if fill is None else round(fill, 1))


def _artifact_digest(step) -> str:
    """A digest naming the kernel a bound plan launches: its entry's name and
    source, or the registers it was built with when no source is kept."""
    import importlib

    entry = getattr(step, "_entry", None)
    digest = hashlib.md5(usedforsecurity=False)
    digest.update(str(getattr(entry, "name", "")).encode())
    code = getattr(entry, "code", None)
    if isinstance(code, str):
        code = code.encode()
    digest.update(bytes(code) if code else repr(importlib.import_module("eagle.launch")._kernel_attrs(entry)).encode())
    return digest.hexdigest()


def _packed_read(dev: str, cells) -> tuple:
    """Read several 1-element uint32 counter cells as ONE combined transfer
    instead of a separate full round-trip sync per cell (a run's
    fixed overhead is dominated by sync COUNT, not by what each one moves).
    ``cells`` is a sequence of ``(array, index)`` pairs, each a uint32 cell
    of an :class:`eagle.Array` on ``dev`` (``"cpu"``, where there is no sync
    to fold, so each is just read directly). Device: every cell is copied
    -- async, no sync -- into one small staging buffer on the current
    stream, then ONE read returns them all together. Returns the values as
    plain Python ints, in ``cells`` order."""
    if dev == "cpu":
        return tuple(int(arr[idx]) for arr, idx in cells)
    first, lo = cells[0]
    if all(arr is first and idx == lo + j for j, (arr, idx) in enumerate(cells)):
        # consecutive words of one array: no staging copies, the ONE read
        return tuple(int(v) for v in first[lo : lo + len(cells)].tolist())
    staging = _alloc.empty(dev, len(cells), dtype=arrays.uint32)
    for i, (arr, idx) in enumerate(cells):
        staging[i : i + 1] = arr[idx : idx + 1]
    return tuple(int(v) for v in staging.tolist())


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

    This size rule is the first-run rule. Inside the band between the sure
    fused and sure persist batches (:func:`_entry_regime`), when both the
    persist and the range entries exist, later runs follow the measured wall of
    each entry on this device (:meth:`Runner._pick_entry`); outside the band
    the regime's entry stands for every run."""
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


def _bump(dev: str, cell) -> None:
    """``cell[0] += 1`` as one single-thread built-in kernel node on the current stream."""
    _core.cell_op(_core.CELL_BUMP, _stream.current(), 1, [cell.ptr], [], int(dev.split(":")[1]))


def _mask_count(dev: str, finished, terminated) -> None:
    """``finished[0] = count_nonzero(terminated)``, re-synced to the mask EVERY call (never cached): on the
    device the built-in ``MASK_COUNT`` kernel (one per-element ``atomicAdd`` into a cell zeroed first, a memset
    node under capture); on the host a plain count."""
    mask = _view(terminated)
    if dev == "cpu":
        finished[0] = int(arrays.count_nonzero(mask))
        return
    stream = _stream.current()
    _alloc.zero(dev, finished, stream)
    if mask.shape[0] > 0:
        if not mask.is_contiguous:
            mask = mask.copy()
        _core.cell_op(_core.CELL_MASK_COUNT, stream, mask.shape[0], [mask.ptr, finished.ptr], [], int(dev.split(":")[1]))


def _next_k(k: int, newly: int, live: int, k_max: int, k_min: int = AUTO_K_MIN,
            rate: float = AUTO_RATE0) -> int:
    """Steps to the next compaction, host face (the twin of the device ``POLICY`` op, the same
    expression as ``eagle::policy::bandK``):
    the power of two nearest ``sqrt(2R / h)`` held to ``[k_min, k_max]``, with
    ``R = rate`` and ``h = newly / (live * k)`` the rate samples finish at
    (:func:`eagle._core.band_k`). A caller that pins ``k_min == k_max`` (a
    batch skipping compaction because it fits inside one resident wave) makes
    the band a no-op, so ``k`` never leaves ``k_max`` once seeded there."""
    return _core.band_k(rate, newly, live, k, k_max, k_min)


def _host_policy(cells, finished, total, word, go, hist, k_max: int,
                 k_min: int = AUTO_K_MIN) -> None:
    """One policy step on host cells: the twin of the device ``POLICY`` op."""
    fin, all_ = int(finished[0]), int(total[0])
    fin_at = int(cells[_FIN_AT])
    live, newly = all_ - fin, fin - fin_at
    done = int(cells[_DONE]) + int(word[0])
    cells[_MAPPED] = int(cells[_MAPPED]) + all_ - fin_at
    cells[_FIN_AT] = fin
    cells[_DONE] = done
    k_next = _next_k(int(cells[_K]), newly, live, k_max, k_min,
                     int(cells[_R]) / RATE_SCALE)
    cells[_K] = k_next
    left = int(cells[_CAP]) - done
    g = fin != all_ and left > 0
    go[0] = 1 if g else 0
    if g:
        w = min(k_next, left)
        word[0] = w
        hist[w] = int(hist[w]) + 1


def _policy_op(device: int, cells_args, k_max: int, k_min: int = AUTO_K_MIN) -> None:
    """One policy step as the built-in ``POLICY`` kernel node on the current stream.
    ``cells_args`` is ``(finished, total, cells, word, go, hist)``, all device arrays."""
    _core.cell_op(_core.CELL_POLICY, _stream.current(), 1, [a.ptr for a in cells_args],
                  [k_min, k_max, _K, _FIN_AT, _DONE, _CAP, _R, _MAPPED, RATE_SCALE], device)


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


def _is_foreign(value) -> bool:
    """A DLPack producer that is not eagle's own array, numpy or cupy (jax, torch, ...): eagle's own arrays
    are the planes themselves, never imported or copied."""
    from .interop import _root_module

    return hasattr(value, "__dlpack__") and _root_module(value) not in ("eagle", "numpy", "cupy")


def _import_foreign(planes) -> tuple:
    """``(planes, foreign)``: each foreign plane as a zero-copy
    :class:`eagle.interop.BufferView` (the plan door's own intake), and
    ``{name: the caller's array}``."""
    foreign = {nm: v for nm, v in planes.items() if _is_foreign(v)}
    if not foreign:
        return planes, {}
    from .interop import import_buffer

    return {**planes, **{nm: import_buffer(v) for nm, v in foreign.items()}}, foreign


def _native_planes(planes, foreign, specs, device) -> tuple:
    """``planes`` with each foreign view as the array the loop runs on: read
    in place, and, where the kernel writes it, a copy (the caller's array is
    never written; :meth:`Runner.result` hands the answer back). A view on the
    other side of the plan's device is refused by name. Returns ``(planes,
    the names the kernel writes)``."""
    written = {nm for spec in specs for role, nm in spec
               if role in OUTPUT_ROLES or role == "terminated"}
    if not foreign:
        return planes, written
    out = dict(planes)
    for nm in foreign:
        view = planes[nm]
        if (view.device[0] == 2) != device:
            raise ValueError(
                f"eagle.until_done: {nm!r} is a {'device' if view.device[0] == 2 else 'host'} "
                f"array but this plan runs on the {'device' if device else 'host'}; move it "
                "(all planes of one run live in one place)")
        arr = arrays.asarray(view)
        out[nm] = arr.copy() if nm in written else arr
    return out, written


def _step_planes(plans, specs, planes) -> tuple:
    """``(planes, n)``: one sample's head-shaped arrays as batch-of-one
    views (same memory, written in place), and the batch size. A
    several-plan step shares one namespace: a name is one plane in every
    plan that declares it; an undeclared name is refused."""
    from ._plan_pack import _n_from_kw
    from ._plan_planes import _adapt_planes, _declared_widths

    def flat(plan, spec, sub):
        return _lane_bind.flatten(plan.plugin, spec, sub, _declared_widths(plan.plugin),
                                  door="bind")

    if len(plans) == 1:
        adapted, _ = _adapt_planes(plans[0].plugin, specs[0], flat(plans[0], specs[0], planes),
                                   door="bind", single_only=True)
        return adapted, _n_from_kw(specs[0], adapted)
    declared = {nm for spec in specs for _role, nm in spec}
    unknown = sorted(set(planes) - declared)
    if unknown:
        raise ValueError(
            f"eagle.until_done: {unknown} name no plane of the step's plans "
            f"(their planes: {sorted(declared)})")
    views = {}
    for plan, spec in zip_strict(plans, specs):
        sub = flat(plan, spec, {nm: planes[nm] for _role, nm in spec if nm in planes})
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

    __slots__ = ("_busy", "step", "loop", "pipeline", "active", "finished", "terminated",
                 "n", "steps_per_launch", "every", "launches_per_compaction",
                 "fused_steps", "auto", "host_policy", "host_threads",
                 "_device", "_dev", "_counter", "_total", "_max_steps", "_k_max", "_k_min", "_k0",
                 "_cells", "_go", "_hist",
                 "_compactions", "_build_s",
                 "_fast_mode", "_fast_counter", "_fast_grid", "_fast_block", "_fast_util",
                 "_fast_util_on", "_fast_exact", "_fast_views",
                 "_fast_counts", "_fast_state", "_fast_probe", "_fast_model", "_fast_key",
                 "_mapped_live", "_fast_ev", "_fast_prep",
                 "_fast_fill", "_fast_chunk", "_fast_chunk_ok", "_fast_chunk_pin", "_fast_fit", "_rate", "_recorder",
                 "_hooks", "_foreign", "_work", "_written")

    @_layout.door
    def __init__(self, plan, *, max_steps, every=None, reorder=None, record=False,
                 window=None, steps_reverse="per_step", memory_budget=None, events="step", on_no_event="raise",
                 **planes):
        self._busy = BusyGuard("eagle.Runner")
        # Internal only (never a bound plane name, popped before the
        # plane-binding machinery below ever sees it): forces
        # :func:`_fast_mode_for`'s decision for the card's explicit-arm row
        # -- "fused_one" / "persist" / False (off,
        # keep today's band/WHILE-graph path even when capacity would pick
        # a fast mode). Omitted, the decision is :func:`_fast_mode_for`'s
        # own, as for every other caller; no public signature changes.
        fast_mode_override = planes.pop("_fast_mode", None)
        # Internal only, same convention as ``_fast_mode`` above: turns on the
        # persist entry's slot-fill counter (off by default -- see
        # :meth:`_build_fast`/:meth:`_run_fast`), for a card arm that still
        # wants :attr:`RunReport.slot_fill` filled.
        slot_fill_override = planes.pop("_slot_fill", None)
        # Internal only, same convention: pins the persist entry's chunk length
        # (steps per work fetch) for the life of the runner and turns off the
        # refit of the chunk from the readback; the launch geometry still
        # follows the step fill. For a card arm that sweeps the chunk. Omitted
        # (``None``), the chunk follows the launch rule.
        chunk_pin = planes.pop("_chunk", None)
        if chunk_pin is not None and int(chunk_pin) < 1:
            raise ValueError(f"eagle.until_done: _chunk must be a positive number of steps, got {chunk_pin!r}")
        # Internal only: ``False`` keeps the size pick for every run instead of
        # timing the other entry (see :meth:`_pick_entry`).
        self._fast_probe = planes.pop("_fast_probe", True)
        planes, self._foreign = _import_foreign(planes)
        steps = list(plan) if isinstance(plan, (list, tuple)) else [plan]
        hooks = _hook_nodes(steps, record)
        plans = [p for p in steps if not isinstance(p, Node)]
        if not plans:
            raise ValueError("eagle.until_done: the step is empty; pass a plan or a "
                             "list of plans")
        if len(plans) == 1:
            select = getattr(plans[0], "select", None)
            if select is not None:      # an eagle.plan.AutoPlan: the data decides
                plans[0] = select(planes)
        else:
            plans = _select_step(plans, planes)
        for i, p in enumerate(plans):
            refuse_derived(p, door="eagle.until_done", what=f"step {i}")
            _split.refuse(p.plugin, "eagle.until_done")
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
        dev = _device_token(device, planes)
        for i in finishers:
            _require_counter_slot(specs[i], names[i])
        from .diff.events import check_events

        check_events(events, on_no_event, "eagle.until_done")
        refine = events == "refine"
        for reserved in _reserved_names(record=record, events=events):
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
        if hooks and any(finishes[i]["steps"] != 1 for i in finishers):
            raise ValueError(
                f"eagle.until_done: a per-step hook needs one step per launch; build the "
                f"plan with steps=1 and without steps='auto' ({name} takes {k!r})")
        if len(plans) > 1:
            for i in finishers:
                if finishes[i]["steps"] not in (FINISH_AUTO, 1):
                    raise ValueError(
                        f"eagle.until_done: {names[i]} advances "
                        f"{finishes[i]['steps']} steps per launch, but each plan of "
                        "a several-plan step takes one step per launch")
            k = 1
        auto = k == FINISH_AUTO
        if record or refine:
            _refuse_record(len(plans))
        if steps_reverse not in ("per_step", "fused", "auto"):
            raise ValueError(f"eagle.until_done: steps_reverse is 'per_step', 'fused' or 'auto', "
                             f"got {steps_reverse!r}")
        if steps_reverse == "fused" and (not record or refine):
            raise ValueError("eagle.until_done: steps_reverse='fused' reverses a plain "
                             "record=True run (no refined events)")
        if record:
            pass
        elif window is not None or memory_budget is not None:
            raise ValueError("eagle.until_done: window= and memory_budget= set how a "
                             "record=True run is reversed; pass record=True, or drop them")
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

        planes, self._written = _native_planes(planes, self._foreign, specs, device)
        self._work = dict(planes)
        planes, n = _step_planes(plans, specs, planes)
        if reorder is not None:
            _refuse_moving_node_planes(hooks, specs, planes)
        if mask_name not in planes:
            planes[mask_name] = _alloc.zeros(dev, n, dtype=arrays.bool_)
        mask = planes[mask_name]
        self._work.setdefault(mask_name, mask)
        nodes = _record_nodes(steps, specs[0], planes) if hooks and record else ()
        # the finished count is word [2] of the fast entries' counter block
        # (see _build_fast): one memset zeroes it with the other counters and
        # one read returns it with them
        counter = _alloc.zeros(dev, 6, dtype=arrays.uint32)
        finished = counter[2:3]
        word = _alloc.zeros(dev, 1, dtype=arrays.int64) if auto else None
        active = None
        if compacting:
            active = ActiveSet(mask, reorder=reorder, group=_lane_bind.space_of(plans[0].plugin).group,
                               _lazy_scratch=True, _lazy_map=True)
            if active.theta is not None:
                roles = {}
                for spec in specs:
                    for role, nm in spec:
                        roles.setdefault(nm, role)
                owned = [planes[nm] for nm, role in roles.items()
                         if role in PER_SAMPLE_ROLES and nm != mask_name
                         and hasattr(planes.get(nm), "shape")]
                active.own(*owned)

        if record and steps_reverse == "auto":      # the table's answer for this batch, K and device
            from hawk.ir.steps_class import _choose

            fits = not auto and k > 1 and window in (None, k) and not refine
            width = sum(1 for role, _nm in specs[0] if role == "mutable")
            steps_reverse = _choose(device, width, k, n) if fits else "per_step"
        elif steps_reverse == "auto":
            steps_reverse = "per_step"
        if steps_reverse == "fused":
            if auto or k <= 1 or window not in (None, k):
                raise ValueError("eagle.until_done: steps_reverse='fused' needs a steps=K kernel "
                                 "(K > 1) and window=None or window=K: each window of K steps "
                                 "is one reverse launch")
            window = k
        extras = {}
        if record or refine:    # the run's own per-sample planes move with a reorder
            from .diff._run import array_module, record_planes

            extras = record_planes(plans[0], max_steps, window, device, array_module(device), n, planes,
                                   events=events)
            if active is not None and active.theta is not None:
                active.own(*(v for nm, v in extras.items() if nm != "tape" and v.shape))
                if refine:      # a real row per sample: it moves with its sample
                    active.own(extras["tape"][0])
                else:
                    active.indirect(extras["tape"][0])
        bound = []
        for plan_i, spec in zip_strict(plans, specs):
            if len(plans) == 1:
                bind = dict(planes)
            else:
                bind = {nm: planes[nm] for _role, nm in spec if nm in planes}
            if ("lookup", FINISHED_PLANE) in spec:
                bind[FINISHED_PLANE] = finished
            if ("lookup", FUSED_STEPS_PLANE) in spec:
                bind[FUSED_STEPS_PLANE] = (word if auto
                                           else _alloc.ones(dev, 1, dtype=arrays.int64))
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
        self._rate = AUTO_RATE0
        self._cells = self._go = self._hist = None
        self._device = device
        self._dev = dev
        self._counter = counter
        self._total = _alloc.asarray(dev, [self.n], arrays.uint32)
        self._compactions = _alloc.zeros(dev, 1, dtype=arrays.uint32)
        self._build_s = 0.0
        self._fast_mode = None
        self._fast_state = _NO_PICK
        self._fast_model = self._fast_key = None
        self._mapped_live = False
        self._fast_ev = None
        self._fast_prep = {}
        self._fast_fill = _core.PERSIST_FIRST_FILL
        self._fast_chunk = None
        self._fast_chunk_ok = False
        self._fast_chunk_pin = None if chunk_pin is None else int(chunk_pin)
        self._fast_fit = None
        self._fast_counter = self._fast_grid = self._fast_block = self._fast_util = None
        self._fast_exact = False
        self._fast_counts = False
        self._fast_views = None
        self._fast_util_on = False
        self._recorder = None
        if record or refine:
            from .diff._run import RunRecorder

            self._recorder = RunRecorder(self, plans[0], "eagle.until_done", window,
                                         memory_budget, extras, self, record=record,
                                         events=events, on_no_event=on_no_event, nodes=nodes,
                                         steps_reverse=steps_reverse)
        self._hooks = tuple(hooks)
        ctx = LowerCtx("gpu" if device else "cpu", body=True)
        by_plan = iter(b.launch for b in bound)
        launches = tuple(ctx.lower(p) if isinstance(p, Node) else next(by_plan) for p in steps)

        def launch():
            for one in launches:
                one()

        if auto:
            self._build_auto(launch, max_steps, fast_mode_override,
                             slot_fill_override)
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
                    slot_fill_override=None) -> None:
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
        :func:`_next_k` a no-op once K reaches ``k_max`` --
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
            regime = _entry_regime(self._persist_capacity(), self.step.device_capacity(), self.n)
            if regime == "fused" and (self.active is None or self.step.range_entry() is not None):
                mode = "fused_one"
            elif regime == "persist" and self.step.persist_entry() is not None:
                mode = "persist"
            # inside the band between the two sure regimes, with both entries
            # present, the other entry can be timed against the size pick
            probe = (self._fast_probe is not False and mode is not None
                     and regime == "band"
                     and self.step.persist_entry() is not None
                     and self.step.range_entry() is not None)
        else:
            mode = None
        if mode is not None:
            self._build_fast(mode, slot_fill_override, probe)
            return
        if self.active is not None:
            self._materialize_map()
            self.active.prepare_compaction()
        dev = self._dev
        self._rate = self._band_rate()
        self._cells = _alloc.zeros(dev, 6, dtype=arrays.int64)
        self._go = _alloc.zeros(dev, 1, dtype=arrays.uint32)
        self._hist = _alloc.zeros(dev, self._k_max + 1, dtype=arrays.int64)
        cells, go, hist, word = self._cells, self._go, self._hist, self.fused_steps
        finished, total, k_max = self.finished, self._total, self._k_max

        skip_compaction = False
        # a reorder lives in the compaction body, so a reordering set keeps it
        if self._device and self.active is not None and self.active.theta is None:
            cap = self.step.device_capacity()
            if cap > 0 and self.n <= cap:
                skip_compaction = True
        self._mapped_live = self.active is not None and not skip_compaction
        if skip_compaction:
            self._k_min = k_max
            self._k0 = k_max
        k_min = self._k_min

        if self._device:
            args, ordinal = (finished, total, cells, word, go, hist), int(dev.split(":")[1])

            def policy():
                _policy_op(ordinal, args, k_max, k_min)
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

    def _band_rate(self) -> float:
        """R, a compaction's cost in steps of a full-live launch, from the cost
        model (:func:`eagle._core.band_rate`) on a device run; the middle of the
        measured range where the device or the model is unknown."""
        if not self._device:
            return AUTO_RATE0
        import importlib

        from ._launch_policy import _attr, _fp64_args

        dev = importlib.import_module("eagle.launch")._device_props(self._run_device())
        major, minor, sms, ratio, fp64 = _fp64_args(dev, self._scalar_type())
        rate = _core.band_rate(major, minor, sms, ratio, _attr(dev, "clockRate"), fp64,
                               self.n, ops_per_step(self.step.plan.plugin),
                               _attr(dev, "memoryClockRate"), _attr(dev, "memoryBusWidth"),
                               float(self.active.bytes_per_sample() if self.active is not None else 0))
        return rate if rate > 0 else AUTO_RATE0

    def _scalar_type(self) -> str:
        return getattr(self.step.plan.plugin, "scalar_type", None) or "float64"

    def _materialize_map(self) -> None:
        """The band path reads the active-set map: allocate the deferred one
        and re-point the bound plans at it."""
        if self.active.materialize_map():
            for bound in (self.step if isinstance(self.step, tuple) else (self.step,)):
                if MAP_PLANE in bound.names:
                    bound.rebind(**{MAP_PLANE: self.active.map})

    def _build_fast(self, mode: str, slot_fill_override=None,
                    probe: bool = False) -> None:
        """Set up the no-graph shape :func:`_fast_mode_for` picked: no
        ``self.loop``/``self.pipeline`` at all, since one call of
        :meth:`run` IS the whole thing. ``fused_steps`` is written here,
        ONCE (it never changes again -- ``self._max_steps`` is fixed for
        the life of the runner), so :meth:`_run_fast` never writes it.
        ``"persist"`` allocates the lane-stealing counter (memset at each
        :meth:`run`, not here -- the one-time allocation only); the
        slot-fill counter is allocated, and later filled, ONLY when
        ``slot_fill_override`` asks for it (off by default -- it
        costs a same-address 64-bit atomic per warp per chunk, see
        :meth:`_run_fast`'s docstring). ``probe`` builds the persist
        block/grid whatever ``mode`` is, so :meth:`_pick_entry` can switch
        between the two entries; it also (re)starts the measured walls."""
        self._fast_mode = mode
        self._fast_state = _NO_PICK._replace(pick=mode if probe else None)
        self._fast_model, self._fast_key = self._entry_model() if probe else (None, None)
        # an artifact whose fast entries count the samples finished on entry
        self._fast_counts = bool(getattr(self.step.plan.plugin, "entry_counts_finished", False))
        # every persist entry takes the chunk length as a launch argument (one
        # that predates it ignores the trailing argument)
        self._fast_chunk_ok = True
        self._fast_fill = _core.PERSIST_FIRST_FILL
        self._fast_fit = None
        self._fast_chunk = self._chunk_for(1e9)
        # the timing events, allocated once: only a probing runner times runs
        self._fast_ev = (tuple(_stream.Event(int(self._dev.split(":")[1]), timed=True) for _ in range(2))
                         if probe else None)
        self.loop = None  # no WHILE-graph body exists in this mode
        self.fused_steps[0] = self._max_steps
        self._fast_util_on = bool(slot_fill_override) and (mode == "persist" or probe)
        dev = self._dev
        # [0] the persist entry's lane-stealing counter, [1] the run's exact
        # step count (both entries max each sample's steps into it): ONE
        # memset zeroes both per run. A plain fused_one with no range entry
        # (a unit built without the fast entries) reports no count.
        # uint32 words: [0] next, [1] steps, [2] finished, [3] pad, [4:6] the
        # uint64 sum of every lane's executed steps (a probing runner's
        # step-fill report): one memset zeroes all, one read returns them
        self._fast_counter = self._counter
        self._fast_views = (self._fast_counter[0:1], self._fast_counter[1:2],
                            self._fast_counter[4:6].view(arrays.uint64) if probe else None)
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
        self._fast_util = (_alloc.zeros(dev, 2, dtype=arrays.uint64)
                           if self._fast_util_on else None)
        self._size_persist()

    def _run_device(self) -> int:
        """The CUDA ordinal the step's planes live on (``-1``: the current one)."""
        on = getattr(self.step, "_device", None)
        return -1 if on is None else int(on)

    def _device_and_ops(self) -> tuple:
        """The device properties and the step's operation count
        (:func:`ops_per_step`) the launch rules read."""
        import importlib

        return (importlib.import_module("eagle.launch")._device_props(self._run_device()),
                ops_per_step(self.step.plan.plugin))

    def _chunk_for(self, mean_steps: float) -> int:
        """The chunk length for samples of ``mean_steps`` mean steps: the pinned
        one, else the launch rule (:func:`eagle._launch_policy.persistent_chunk`)."""
        if self._fast_chunk_pin is not None:
            return self._fast_chunk_pin
        from ._launch_policy import persistent_chunk

        dev, ops = self._device_and_ops()
        return persistent_chunk(dev, ops, mean_steps, self._scalar_type())

    def _persist_geometry(self, fill: float, chunk: int) -> tuple:
        """``(block, grid)`` of the persist launch at step fill ``fill`` and
        chunk length ``chunk``."""
        import importlib

        from ._launch_policy import persistent_launch

        # by module name, not attribute: eagle exports a `launch` function
        # that shadows the submodule (same reason _plan_binding.py's own
        # .launch() resolves it this way).
        _launch = importlib.import_module("eagle.launch")
        return persistent_launch(_launch._device_props(self._run_device()),
                                 _launch._kernel_attrs(self.step.persist_entry()),
                                 self.n, fill, self._scalar_type(), chunk, self._device_and_ops()[1])

    def _size_persist(self) -> None:
        """Size the persist launch and pack its argument blocks: threads per SM
        from the device, the batch size, the step fill read back so far (first
        run: 0.5) and the entry's own register-limited residency
        (:func:`eagle._launch_policy.persistent_launch`), the chunk length from
        the mean steps seen. Called at bind and again when a run's readback moves
        the fill or the chunk."""
        self._fast_block, self._fast_grid = self._persist_geometry(self._fast_fill, self._fast_chunk)
        counter, steps, stepsum = self._fast_views
        steps_cell = steps if self._fast_exact else None
        stepsum = stepsum if self._fast_exact else None
        chunk = self._fast_chunk if self._fast_chunk_ok else None
        prep = self._fast_prep
        for key, util in (("persist", None), ("persist_util", self._fast_util)):
            if key == "persist" or util is not None:
                prep[key] = self.step.prepare_persist(
                    grid=self._fast_grid, counter=counter, util=util, block=self._fast_block,
                    steps=steps_cell, stepsum=stepsum, chunk=chunk)
                if stepsum is not None:
                    prep[key + "_bare"] = self.step.prepare_persist(
                        grid=self._fast_grid, counter=counter, util=util, block=self._fast_block,
                        steps=steps_cell, chunk=chunk)

    # Read-only views of single fields of the one pick-state value; an assignment
    # swaps in a copy of the state with that field replaced.
    def _view(self, name):
        return getattr(getattr(self, "_fast_state", _NO_PICK), name)

    def _put(self, name, value) -> None:
        self._fast_state = getattr(self, "_fast_state", _NO_PICK)._replace(**{name: value})

    _fast_pick = property(lambda self: self._view("pick"), lambda self, v: self._put("pick", v))
    _fast_wall = property(lambda self: self._view("wall"), lambda self, v: self._put("wall", v))
    _fast_last = property(lambda self: self._view("last"), lambda self, v: self._put("last", v))
    _fast_prior = property(lambda self: self._view("prior"), lambda self, v: self._put("prior", v))
    _fast_runs = property(lambda self: self._view("runs"), lambda self, v: self._put("runs", v))

    def _persist_capacity(self) -> int:
        """SMs times the threads per SM the persist entry keeps resident at its own
        registers (``0`` when unknown); the size the regime rule weighs a batch
        against. Not ``device_capacity()``: a lighter kernel, such as the same
        step in single precision, keeps more threads resident per SM, so one
        wave holds more samples."""
        import importlib

        from ._launch_policy import capacity

        persist = self.step.persist_entry()
        if persist is None:
            return 0
        launch = importlib.import_module("eagle.launch")
        return capacity(launch._device_props(self._run_device()), launch._kernel_attrs(persist))

    def _entry_model(self) -> tuple:
        """``(model, key)`` of a runner that can time both entries. ``model`` holds
        what :func:`eagle._core.entry_ratio` needs besides the readback (``None``
        when the device or the step cost is unknown); ``key`` names this device,
        kernel, precision and batch size for the process's settled picks
        (:func:`eagle.launch._pick_cache_get`)."""
        import importlib

        from ._launch_policy import _attr, _fp64_args, capacity

        launch = importlib.import_module("eagle.launch")
        dev = launch._device_props(self._run_device())
        scalar = self._scalar_type()
        major, minor, sms, ratio, fp64 = _fp64_args(dev, scalar)
        resident = capacity(dev, launch._kernel_attrs(self.step._entry))
        ops = ops_per_step(self.step.plan.plugin)
        clock = _attr(dev, "clockRate")
        model = None
        if sms > 0 and clock > 0 and resident > 0:
            model = (major, minor, sms, ratio, clock, fp64, resident, ops)
        name = dev.get("name", b"") if isinstance(dev, dict) else getattr(dev, "name", b"")
        key = (self._run_device() if self._run_device() >= 0 else launch._current_device_ordinal(), str(name), major, minor, sms, clock // 50000,
               _artifact_digest(self.step), scalar, self.n.bit_length())
        return model, key

    def _pick_entry(self) -> tuple:
        """The entry this run takes and whether it is a probe, as
        ``(mode, probe)``. A pinned or below-capacity runner always takes its
        bind-time mode. Otherwise the first run takes the size pick (real
        work, often the only run: never a probe), the second the other entry,
        and from then on the one with the smaller
        measured launch wall. The loser is timed again only when a run's readback
        moves to another regime (:func:`_readback_regime`), and then not if
        a pick the process already settled for the regime says which entry wins.
        However low the step fill f, a batch inside the band is timed: a spread
        batch is exactly where a persistent launch's refills can cost more than the fused entry's
        idle lanes, and only the wall says which."""
        state = self._fast_state
        pick = state.pick
        if pick is None:
            return self._fast_mode, False
        if state.prior is None or state.settled:
            return pick, False
        wall = state.wall
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

    def _carries_timing(self, probing: bool) -> bool:
        """Whether the run just picked carries the timing event pair: a
        probe, the first run, and a run while either wall is missing."""
        state = self._fast_state
        return not state.settled and (probing or state.prior is None or len(state.wall) < 2)

    def _note_run(self, mode: str, wall_ms, steps: int, finished: int, f) -> None:
        """Fold one probing-eligible run into the pick: its step fill ``f``
        (from either entry; ``None`` on a settled run that did not sum its
        steps), its readback and, when it was timed, its wall (``None`` for a
        settled run, which only checks whether the readback regime moved).

        When the readback regime changes, walls timed under the previous regime
        are not comparable. The pick is then taken,
        without timing, from the process's settled picks; failing that, the entry
        just run is the pick and the other is timed next. The new
        state is built whole and swapped in by one assignment."""
        state = self._fast_state
        prior = state.prior if f is None else f
        regime = _readback_regime(steps, prior)
        pick, wall, settled = state.pick, state.wall, state.settled
        if regime != state.last:
            pick, wall, settled = mode, {}, False
            decided = self._decide(regime)
            if decided is not None:
                pick, wall, settled = decided
            elif wall_ms is not None:
                wall = {mode: wall_ms}
        elif wall_ms is not None:
            wall = {**wall, mode: wall_ms}
            if len(wall) == 2:
                # a tie keeps the pick: the other entry must win by 2 %
                other = "persist" if pick == "fused_one" else "fused_one"
                if pick in wall and wall[other] < PICK_HYSTERESIS * wall[pick]:
                    pick = other
                elif pick not in wall:
                    pick = min(wall, key=wall.get)
                settled = True
                self._remember(regime, pick, wall)
        self._fast_state = _Pick(pick, wall, regime, prior, state.runs + 1, settled,
                                 self._model_ratio(steps, prior))
        if prior is not None:
            self._refit_persist(prior, steps)

    def _decide(self, regime: tuple):
        """``(pick, walls, True)`` when a pick the process already settled for this
        device, kernel and regime exists, else ``None``."""
        key = getattr(self, "_fast_key", None)
        if key is None:
            return None
        import importlib

        cached = importlib.import_module("eagle.launch")._pick_cache_get(key + (regime,))
        return None if cached is None else (cached[0], dict(cached[1]), True)

    def _model_ratio(self, steps: int, fill):
        """The model's predicted persist-over-fused wall for this readback
        (:func:`eagle._core.entry_ratio`), or ``None`` when the device, the step
        cost or the fill is unknown. A record of what the model expects; the
        timed walls pick."""
        model = getattr(self, "_fast_model", None)
        if model is None or fill is None or fill <= 0.0:
            return None
        ratio = _core.entry_ratio(*model[:7], self.n, int(steps), float(fill), model[7])
        return ratio if ratio > 0.0 else None

    def _remember(self, regime: tuple, pick: str, wall: dict) -> None:
        """Keep a pick settled on measured walls for the process's later runners."""
        key = getattr(self, "_fast_key", None)
        if key is not None:
            import importlib

            importlib.import_module("eagle.launch")._pick_cache_put(key + (regime,), pick, wall)

    def _refit_persist(self, fill: float, steps: int) -> None:
        """Resize the persist launch when the readback moved the step fill or
        the steps: the threads per SM follow the fill (a drain limit) and the
        chunk follows the mean steps ``fill * steps``. A readback whose
        ``(round(fill, 2), steps)`` is the one already fitted costs nothing.
        A refit that changes the launch drops the persist entry's stored wall,
        which was timed on the old geometry; it is timed again. A settled pick
        keeps its walls: its regime has not moved, and a settled key never times
        an entry again."""
        if "persist" not in self._fast_prep:
            return
        key = (round(fill, 2), steps)
        if key == self._fast_fit:
            return
        self._fast_fit = key
        chunk = self._chunk_for(fill * steps)
        geometry = self._persist_geometry(fill, chunk)
        if geometry == (self._fast_block, self._fast_grid) and chunk == self._fast_chunk:
            self._fast_fill = fill
            return
        self._fast_fill, self._fast_chunk = fill, chunk
        state = self._fast_state
        if not state.settled and "persist" in state.wall:
            self._fast_state = state._replace(wall={k: v for k, v in state.wall.items() if k != "persist"})
        self._size_persist()

    def _run_fast(self) -> RunReport:
        """:meth:`run`'s body for a no-graph mode: one launch (fused with
        the whole budget as its word, or the persist entry), then one
        packed readback -- no iterations to count, no compaction, no
        reorder, and (the one host<->device sync this mode ever takes) no
        OTHER sync either.

        ``finished`` is re-synced to the mask EVERY call, fresh
        (:func:`_mask_count`, eagle's own built-in kernel): the public contract (:meth:`run`'s
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

        The slot-fill counter (:attr:`_fast_util_on`) is read back
        with its own ``.get()`` only when it was asked for at bind
        (:meth:`_build_fast`) -- an extra sync, opted into, never the
        default."""
        dev = self._dev
        slot_fill = None
        # ONE device word block per run: [0] the persist entry's
        # lane-stealing counter, [1] the run's exact step count, [2] the
        # finished count -- zeroed by one async memset, read back by ONE
        # transfer.
        counter = self._fast_counter
        stream = _stream.current()
        mode, probing = self._pick_entry()
        # the range/persist entry counts the samples finished on entry
        # itself, as the step counts those finishing in the run: the memset
        # zeroes the count too; else the mask is counted after it
        counts = self._fast_counts and (mode == "persist" or self._fast_exact)
        state = self._fast_state
        probed = state.pick is not None
        # only a run that decides is timed: the first, a probe, and the
        # run(s) after an invalidation cleared the walls; a settled run is
        # not (an event pair costs more than the launch policy is worth)
        timed = probed and self._carries_timing(probing)
        if timed:
            t0, t1 = self._fast_ev
            t0.record(stream)
        # the step sum: on every run that decides, and on every
        # STEP_FILL_EVERY-th settled one (a batch whose spread changed under
        # the same readback is seen within that many runs); the other runs
        # pass no sum cell, so the device adds nothing
        summed = probed and (timed or state.runs % STEP_FILL_EVERY == 0)
        bare = "" if summed or not probed else "_bare"
        if self._fast_exact:
            _alloc.zero(dev, counter, stream, 24 if summed else 12)
        elif counts:
            _alloc.zero(dev, self.finished, stream)
        if not counts:
            _mask_count(dev, self.finished, self.terminated)
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
                self._fast_util.fill(0)
                util = self._fast_util
                key = "persist_util"
            self.step.launch_persist(grid=self._fast_grid,
                                     counter=next_cell, util=util,
                                     block=self._fast_block, steps=steps_cell,
                                     stepsum=stepsum, chunk=self._fast_chunk,
                                     prepared=self._fast_prep[key + bare])
        if timed:
            t1.record(stream)
        cells = [(counter, i) for i in (range(1, 6) if summed else (1, 2))]
        words = _packed_read(dev, cells)
        steps_read, finished = words[0], words[1]
        if probed:
            # the step fill f = (sum of the lanes' steps) / (n * longest lane's steps)
            f = None
            if summed:
                total = words[3] | (words[4] << 32)
                f = total / (self.n * steps_read) if steps_read > 0 else 1.0
            self._note_run(mode, t1.elapsed_ms(t0) if timed else None,
                           steps_read, finished, f)
        if self._fast_util_on and mode == "persist":
            steps_run, slots = self._fast_util.tolist()
            if slots > 0:
                slot_fill = float(steps_run) / float(slots)
        done = finished == self.n
        steps, exact = ((steps_read, True) if steps_cell is not None
                        else (self._max_steps, not done))
        return RunReport(
            n=self.n, finished=finished, done=done, launches=1,
            steps=steps, compactions=0, reorders=0, build_s=0.0,
            launches_by_k={self._max_steps: 1}, exact_steps=exact,
            slot_fill=slot_fill, mode=mode, probe=probing, timed=timed, mapped=self.n)

    @property
    def policy(self) -> dict | None:
        """The automatic loop's policy cells as host ints (a device sync):
        ``k``, ``finished_at_launch``, ``steps_done``, ``max_steps``,
        ``go``; ``None`` for a fixed artifact."""
        if not self.auto:
            return None
        cells = self._cells.tolist()
        return {"k": int(cells[_K]), "finished_at_launch": int(cells[_FIN_AT]),
                "steps_done": int(cells[_DONE]), "max_steps": int(cells[_CAP]),
                "go": _host_scalar(self._go, 0)}

    @property
    def launch_policy(self) -> dict:
        """What the launch rules chose for this runner, for a probe report:
        ``block`` and ``grid`` of the persist launch (``None`` when the runner has
        none), ``chunk`` (steps per fetch; ``None`` when the entry takes none),
        ``fill`` (the step fill the sizing last read, first run 0.5), ``rate`` (R)
        and ``k`` (the band's current steps per launch; ``None`` off the band).
        Pair it with :meth:`eagle.plan.BoundPlan.entry_reports` (registers, local
        memory, code size, bound, demanded threads)."""
        persist = self._fast_block is not None
        k = None
        if self.auto and self._cells is not None:
            k = int(self._cells[_K])
        return {"block": self._fast_block if persist else None,
                "grid": self._fast_grid if persist else None,
                "chunk": self._fast_chunk, "fill": self._fast_fill, "rate": self._rate, "k": k}

    def _seed(self) -> None:
        """Seed the automatic loop's cells from the host before a run (the
        counter is already re-synchronised to the mask)."""
        fin = _host_scalar(self.finished, 0)
        first = min(self._k0, self._max_steps)
        go = fin != self.n and self._max_steps > 0
        self._cells[...] = [self._k0, fin, 0, self._max_steps, round(self._rate * RATE_SCALE), 0]
        self.fused_steps[0] = first
        self._go[0] = 1 if go else 0
        self._hist.fill(0)
        if go:
            self._hist[first] = 1

    def _counted(self, tail: Skippable) -> Skippable:
        """The compaction tail, also counting each compaction it runs."""
        cell = self._compactions
        if self._device:
            dev = self._dev

            def compact():
                tail.step()
                _bump(dev, cell)
        else:
            def compact():
                tail.step()
                cell[0] = int(cell[0]) + 1
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
        with self._busy("run"):
            return self._run()

    def _run(self) -> RunReport:
        for node in self._hooks:
            check = get_node_kind(node.kind).check
            if check is not None:
                check(node)
        if self._recorder is not None:
            return self._recorder.run()
        if self._fast_mode is not None:
            return self._run_fast()
        dev = self._dev
        _mask_count(dev, self.finished, self.terminated)
        pre_cells = []
        if self.active is not None:
            pre_cells.append((self._compactions, 0))
            if self._reorders():
                pre_cells.append((self.active.reorders, 0))
        pre = _packed_read(dev, pre_cells) if pre_cells else ()
        compactions0 = pre[0] if pre_cells else 0
        reorders0 = pre[1] if len(pre_cells) > 1 else 0
        if self.auto:
            self._seed()
        if self._device:
            if self.pipeline is None:
                from .pipeline import GraphPipeline

                t0 = time.perf_counter()
                self.pipeline = GraphPipeline().add(self.loop).build()
                _core.device_synchronize()
                self._build_s = time.perf_counter() - t0
            self.pipeline.launch()
        else:
            _run_host(self)
        post_cells = [(self.loop.iteration_index, 0), (self.finished, 0)]
        if self.active is not None:
            post_cells.append((self._compactions, 0))
            if self._reorders():
                post_cells.append((self.active.reorders, 0))
        launches, finished, *rest = _packed_read(dev, post_cells)
        reorders = 0
        compactions_post = 0
        if self.active is not None:
            compactions_post = rest[0]
            if self._reorders():
                reorders = rest[1] - reorders0
                self.active.restore()
        mapped = launches * self.n
        if self.auto:
            if self._mapped_live and launches:
                mapped = _host_scalar(self._cells, _MAPPED) or mapped
            by_k = {k: v for k, v in enumerate(self._hist.tolist()) if v}
            steps = self.policy["steps_done"]
            # exact when nothing launched, the cap stopped an unfinished
            # sample, or the last launch took one step
            exact = (launches == 0 or finished != self.n
                     or _host_scalar(self.fused_steps, 0) == 1)
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
            exact_steps=exact, mode="band", mapped=mapped)

    @property
    def record(self):
        """The :class:`eagle.diff.Record` of the last run of a ``record=True``
        runner (``None`` before the first run, or without ``record=True``):
        pass it to :func:`eagle.diff.reverse`.

        Example: ``runner.run(); grads = eagle.diff.reverse(runner.record, {"x": dJ_dx})``."""
        return None if self._recorder is None else self._recorder.live

    @property
    def t_stop(self):
        """The event time of each sample of an ``events="refine"`` run (the
        step's end time for a sample with no located event); ``None``
        otherwise. A per-sample plane, written by every run."""
        refine = getattr(self._recorder, "refine", None)
        return None if refine is None else refine.t_stop

    @property
    def event_status(self):
        """Per sample, how an ``events="refine"`` run located its event
        (int8): 1 inside the stop step, 2 at the step's end, 3 no event (with
        ``on_no_event="zero"``); ``None`` without ``events="refine"``."""
        refine = getattr(self._recorder, "refine", None)
        return None if refine is None else refine.status

    def _reorders(self) -> bool:
        return self.active is not None and self.active.theta is not None

    def result(self, name: str):
        """The plane ``name`` after a run, as the caller's own array type. A
        numpy or cupy plane is the caller's array, written in place. A plane
        from another framework (jax, torch, ...) is read in place and never
        written: a plane the kernel writes comes back as a NEW array of that
        framework holding the answer (a copy per call), an unwritten one as
        the caller's array itself."""
        if name not in self._work:
            raise KeyError(f"eagle.until_done: no plane {name!r} (planes: {sorted(self._work)})")
        original = self._foreign.get(name)
        native = self._work[name]
        if original is None:
            return native
        if name not in self._written:
            return original
        from .interop import export_like

        return export_like(original, native)

    def reset(self) -> None:
        """Clear the mask/counter/active set: the next :meth:`run` starts the batch over."""
        if self._reorders():
            self.active.restore()
        mask = _view(self.terminated)
        if self._device and (not mask.size or mask.is_contiguous):
            # two async memsets on the current stream, not two fill kernels
            stream = _stream.current()
            _alloc.zero(self._dev, mask, stream)
            _alloc.zero(self._dev, self.finished, stream)
        else:
            self.terminated.fill(False)
            self.finished.fill(0)
        if self.active is not None:
            self.active.reset()


def _hook_nodes(steps, record) -> list:
    """The nodes of a step list (per-step hooks). Unrecorded, each must be
    captured work (a stream or host node cannot run inside one device graph);
    a recorded run calls every node from the host each step, so any tier runs."""
    from ._graph_kinds import OWN_CODE_KINDS

    hooks = [p for p in steps if isinstance(p, Node)]
    for node in hooks:
        if node.kind not in OWN_CODE_KINDS:
            raise ValueError(f"eagle.until_done: a step list takes plans and opaque or child "
                             f"nodes, not a {node.kind!r} node")
        if not record and get_node_kind(node.kind).tier != "capture":
            raise ValueError(
                f"eagle.until_done: node {node.label!r} runs on the host each launch, but a "
                "stepping loop is one device graph: run host work between runs "
                "(runner.run() with max_steps=k, then your code), or make the node "
                "captured (drop stream= / eager=), or pass record=True (a recorded run "
                "calls nodes from the host each step)")
    return hooks


def _refuse_moving_node_planes(hooks, specs, planes) -> None:
    """``reorder=`` moves the plans' per-sample planes; a plane only a node
    declares would stay put and belong to other samples after a reorder."""
    declared = {nm for spec in specs for _role, nm in spec}
    bound = [v for v in planes.values()]
    for node in hooks:
        kind = get_node_kind(node.kind)
        for nm, arr, _role in (*kind.reads(node), *kind.writes(node)):
            if nm in declared or any(arr is v for v in bound) or not getattr(arr, "ndim", 0):
                continue
            raise ValueError(
                f"eagle.until_done: reorder= cannot move plane {nm!r}, which only node "
                f"{node.label!r} declares; drop reorder=, or bind the plane to a plan")


def _record_nodes(steps, spec, planes) -> list:
    """``[(node, reads, writes)]`` of a recorded step ``[node, ..., plan]``,
    ``reads``/``writes`` as ``[(declared name, plan plane, array)]``; the shapes a
    reverse cannot re-run are refused by name."""
    roles = {nm: role for role, nm in spec}
    plane_of = lambda nm, a: next((p for p, v in planes.items() if v is a),  # noqa: E731
                                  nm if nm in planes else None)
    first = next(j for j, p in enumerate(steps) if not isinstance(p, Node))
    out = []
    for i, node in enumerate(steps):
        if not isinstance(node, Node):
            continue
        if i > first:
            raise ValueError(
                f"eagle.until_done: node {node.label!r} comes after the plan in a recorded "
                "step: its output would be a carried state with an initial value; put the "
                "node before the plan: it prepares the step from the current state")
        kind = get_node_kind(node.kind)
        out.append((node, *([(nm, plane_of(nm, a), a) for nm, a, _r in got]
                            for got in (kind.reads(node), kind.writes(node)))))
    labels = [node.label for node, _r, _w in out]
    for label in labels:
        if labels.count(label) > 1:
            raise ValueError(f"eagle.until_done: two nodes of the recorded step are labelled "
                             f"{label!r}; give each node a name= (vjps= keys on it)")
    writer = {pn: node.label for node, _r, writes in out for _nm, pn, _a in writes}
    order = {label: i for i, label in enumerate(labels)}
    for node, reads, writes in out:
        for (nm, pn, _a), w in [(x, False) for x in reads] + [(x, True) for x in writes]:
            role = roles.get(pn)
            if role == "uniform":
                if w:
                    raise ValueError(f"eagle.until_done: node {node.label!r} writes parameter "
                                     f"{pn!r}: a node only reads a parameter")
                continue
            if role is None or (w and role in ("mutable", "terminated", "lookup", "out")):
                raise ValueError(
                    f"eagle.until_done: node {node.label!r} declares {nm!r}, which the plan "
                    + ("reads as state, not as a per-sample input" if role else "does not bind")
                    + "; in a recorded step a node reads the plan's planes and writes planes "
                    "the plan reads")
            if not w and pn in writer and order[writer[pn]] >= order[node.label]:
                raise ValueError(
                    f"eagle.until_done: node {node.label!r} reads {pn!r}, which node "
                    f"{writer[pn]!r} writes after it; put {writer[pn]!r} first")
    return out


def _refuse_record(nplans: int) -> None:
    """The run shapes a ``record=True`` run does not take yet, refused by name."""
    from .diff._run import _not_yet

    door = "eagle.until_done"
    if nplans > 1:
        raise _not_yet(door, "for a step of several plans")


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
               record: bool = False, window=None, steps_reverse="per_step", memory_budget: int | None = None,
               events: str = "step", on_no_event: str = "raise", **planes) -> Runner:
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
    :data:`FUSED_STEPS_PLANE` word.

    ``record=True`` records each run for a reverse pass: :attr:`Runner.record`
    after :meth:`Runner.run` goes to :func:`eagle.diff.reverse`. The run
    takes one step per launch and keeps the state every ``window`` steps
    (``None``: clamp(ceil(sqrt(max_steps)), 16, 1024); ``"full"``: every
    step, one window). ``memory_budget`` (bytes; ``None``: half the free
    device memory, or of the available host RAM) bounds what a reverse holds:
    over it, the reverse runs the batch in passes, with the same bits
    (``grads.report.chunks`` counts the passes). Example::

        runner = eagle.until_done(plan, max_steps=1000, record=True, x=x, v=v)
        runner.run()
        grads = eagle.diff.reverse(runner.record, {"x": dJ_dx})

    ``events="refine"`` locates each sample's stop INSIDE its stop step (the
    default ``"step"`` stops at the step the rule fires, with a zero
    derivative through the stop time). The kernel declares its time plane
    and step, ``@hawk.kernel(time="t", time_step="dt")``, and its stop rule
    is a smooth crossing (``terminated = x < a * t``). Each run then leaves
    the state AT the crossing in the planes, :attr:`Runner.t_stop` holds the
    event time and :attr:`Runner.event_status` how it was located; under
    ``record=True`` the reverse includes the stop time's sensitivity. A
    sample with no crossing in its stop step (it hit ``max_steps``) is
    refused unless ``on_no_event="zero"`` (its stop stays the stepped one,
    with zero event sensitivity); a grazing crossing and two stop rules
    crossing in one step are always refused. Example::

        runner = eagle.until_done(plan, max_steps=1000, events="refine",
                                  record=True, x=x, v=v, t=t)
        runner.run()
        runner.t_stop                       # the event time of each sample
        grads = eagle.diff.reverse(runner.record, {"t": numpy.ones(n)})"""
    return Runner(plan, max_steps=max_steps, every=every, reorder=reorder,
                  record=record, window=window, steps_reverse=steps_reverse, memory_budget=memory_budget,
                  events=events, on_no_event=on_no_event, **planes)


def run_until_done(plan, *, max_steps: int, every: int | None = None, reorder=None,
                   record: bool = False, window=None, steps_reverse="per_step", memory_budget: int | None = None,
                   events: str = "step", on_no_event: str = "raise",
                   **planes) -> RunReport:
    """:func:`until_done` then :meth:`Runner.run`: the one-call form."""
    return until_done(plan, max_steps=max_steps, every=every, reorder=reorder,
                      record=record, window=window, steps_reverse=steps_reverse, memory_budget=memory_budget,
                      events=events, on_no_event=on_no_event, **planes).run()
