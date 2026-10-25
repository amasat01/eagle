# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ``dispatch=`` door: which ``"auto"`` dispatch groups run fused, and
which run split into one kernel per branch.

``dispatch=None`` leaves every group the author declared alone and resolves
the ``"auto"`` ones by measurement: the model starts fused, counts how many
samples take each branch, and every :data:`~eagle._dispatch_policy.DISPATCH_EVERY`
steps asks the rule (:mod:`eagle._dispatch_policy`) whether splitting would
pay. A string or a dict forces the ``"auto"`` groups to a policy instead;
a group the author gave a policy is never overridden. A model with no
``"auto"`` group never reaches this module's runner: forced ``"switch"`` and
``"predicated"`` are plain kernels, so the launch path is the one it always was.

A split step runs the branches as separate kernels over runs of samples sorted
by branch (:class:`eagle.ActiveSet` with classes). A sample whose branch
changed since the sort is left alone by its kernel and finished by a fused
sweep over just those samples; the batch is sorted again when more than
:data:`~eagle._active_set.STALE_MAX` of it went stale.
"""

from __future__ import annotations

import importlib
import threading
import warnings

import numpy as np

from . import _dispatch_policy as _policy
from . import _split
from . import exec as eexec
from ._active_set import STALE_MAX, ActiveSet
from ._compat import zip_strict
from ._active_set import COUNT_PLANE, MAP_PLANE
from ._busy import BusyGuard
from ._until_done import FINISHED_PLANE, FUSED_STEPS_PLANE, RunReport
from .roles import PER_SAMPLE_ROLES

__all__ = ["PerformanceWarning"]

#: The policies a caller may force an ``"auto"`` group to.
POLICIES = ("switch", "segmented", "predicated")


class PerformanceWarning(UserWarning):
    """A request that is correct but known to run slower than the alternative.

    Raised once per call, for example by ``dispatch="segmented"`` on a group
    whose selector reads a plane the step writes: the split kernels must then
    re-check the selector and sweep the samples that changed branch. Filter it
    on its own::

        warnings.simplefilter("ignore", eagle.PerformanceWarning)
    """


# --- the request --- #
def auto_groups(kernel) -> list:
    """Names of the ``"auto"`` dispatch groups of ``kernel``'s single step."""
    step = kernel.step if kernel.step is not None else kernel
    return [d.name for d in step.walk.dispatches if d.policy == "auto"]


def _step(kernel):
    return kernel.step if kernel.step is not None else kernel


def request(kernels, dispatch) -> dict:
    """``{group: policy or None}`` for the ``"auto"`` groups ``dispatch`` names.

    ``None`` stands for "decide by measurement". Refuses a group that does not
    exist or that the author gave a policy, and a policy that is not one of
    :data:`POLICIES` (or ``"auto"`` for measuring)."""
    groups = {}
    declared = {}
    for k in kernels:
        step = _step(k)
        for d in step.walk.dispatches:
            declared.setdefault(d.name, set()).add(d.policy)
            if d.policy == "auto":
                groups[d.name] = None
    if dispatch is None:
        return groups
    if isinstance(dispatch, str):
        wanted = dict.fromkeys(groups, dispatch)
        if not wanted:
            raise ValueError(
                "eagle: dispatch= picks the policy of an 'auto' dispatch group, and this "
                "model has none; declare one with policy='auto' in the kernel, or drop "
                "dispatch=")
    elif isinstance(dispatch, dict):
        wanted = dict(dispatch)
    else:
        raise ValueError(
            f"eagle: dispatch= is None, a policy name or a {{group: policy}} dict, "
            f"not {type(dispatch).__name__}")
    for group, policy in wanted.items():
        if policy not in (*POLICIES, "auto", None):
            raise ValueError(
                f"eagle: dispatch policy {policy!r} is unknown; the policies are "
                f"{', '.join(repr(p) for p in POLICIES)}, or 'auto' to measure")
        if group not in declared:
            raise ValueError(
                f"eagle: dispatch= names {group!r}, which is not a dispatch group of this "
                f"model; its groups are {sorted(declared)}")
        if group not in groups:
            raise ValueError(
                f"eagle: dispatch= names {group!r}, which the kernel declared "
                f"policy={sorted(declared[group])[0]!r}; only 'auto' groups can be "
                "chosen here, the author's policy stays")
        groups[group] = None if policy in ("auto", None) else policy
    return groups


def resolve_kernels(kernels, groups, *, door) -> tuple:
    """``(deploy, spec)``: ``kernels`` with every group lowered to a plain
    policy, ready to deploy, and ``spec``, the one group a
    :class:`DispatchRunner` takes over (or ``None``).

    The runner takes a group that is forced to ``"segmented"`` and a group
    decided by measurement, which needs a static selector over the WHOLE step
    (:func:`~eagle._dispatch_policy.static_selector`) in a one-kernel model;
    every other ``"auto"`` group runs ``"switch"``. Forcing ``"segmented"`` on
    a group whose selector is not static is allowed and answers with one
    :class:`PerformanceWarning`. ``spec`` is ``(kernel, group, forced)`` with
    the model's other groups already lowered in ``kernel``."""
    import hawk
    from hawk.ir import dispatch_costs
    from hawk.ir.walk import written_names

    written = set()
    for k in kernels:
        written |= written_names(_step(k).walk.order)
    static = {(k.name, name): _policy.static_selector(cost.selector_reads, written)
              for k in kernels for name, cost in dispatch_costs(_step(k).walk).items()}
    spec, slow, deploy = None, False, []
    for k in kernels:
        plain, mine = {}, None
        for group in auto_groups(k):
            policy, is_static = groups.get(group), static[(k.name, group)]
            if policy == "segmented" and not is_static:
                slow = True
            takes = policy == "segmented" or (policy is None and is_static)
            if takes and mine is None and spec is None and len(kernels) == 1:
                mine = (group, policy)
            else:
                if policy == "segmented":
                    raise ValueError(
                        f"{door}: dispatch='segmented' splits one group of a one-kernel model "
                        "at a time; force the others (or the other kernels' groups) to "
                        "'switch', or fuse the step into one kernel")
                plain[group] = policy or "switch"
        rest = hawk.resolve_dispatch(k, plain) if plain else k
        if mine is not None:
            spec = (rest, mine[0], mine[1])
            rest = hawk.resolve_dispatch(rest, {mine[0]: "switch"})
        deploy.append(rest)
    if slow:
        warnings.warn(
            f"{door}: dispatch='segmented' on a group whose selector reads a plane the step "
            "writes is correct but slower than the fused kernel: the split kernels re-check "
            "the selector and a sweep finishes every sample that changed branch. Leave "
            "dispatch=None to keep such a group fused.", PerformanceWarning, stacklevel=4)
    return deploy, spec


# --- a graph's member plans --- #
def relower_graph_nodes(nodes, dispatch, *, door):
    """``(nodes, decided)``: the launch nodes of a graph with every ``"auto"``
    dispatch group of their hawk kernels lowered to a plain policy, and
    ``{node label: {group: policy}}`` of what was chosen.

    A built graph is one fixed structure, so a group is never measured and
    never split here: ``dispatch=None`` lowers ``"auto"`` groups to
    ``"switch"``; a string or dict may also say ``"predicated"``. Splitting
    needs a decision every few steps, which a replayed graph cannot take:
    ``eagle.simulate(..., dispatch="segmented")`` is the door for that.
    Nodes without an ``"auto"`` group are returned untouched."""
    import hawk

    from ._plan_deploy import auto

    out, decided = [], {}
    for node in nodes:
        plan = getattr(node.payload, "plan", None) if node.kind == "launch" else None
        source = getattr(getattr(plan, "plugin", None), "_hawk_source", None)
        kernel = source[0] if source else None
        groups = auto_groups(kernel) if kernel is not None else []
        if not groups:
            out.append(node)
            continue
        wanted = request([kernel], dispatch)
        for group, policy in wanted.items():
            if policy == "segmented":
                raise ValueError(
                    f"{door}: dispatch='segmented' needs a sort and a sweep decided as the "
                    "run goes, which a built graph cannot do; use "
                    "eagle.simulate(..., dispatch='segmented'), or 'switch' here")
        chosen = {g: (wanted.get(g) or "switch") for g in groups}
        lowered = hawk.resolve_dispatch(kernel, chosen)
        keep = {nm: v for nm, v in node.payload._planes.items() if not nm.endswith("_mix")}
        rebound = auto(lowered).select(keep).bind(**keep)
        from .graph import Node

        out.append(Node("launch", rebound, node.name))
        decided[node.label] = chosen
    return out, decided


# --- the runner --- #
def _names(plugin) -> list:
    return [nm for _role, nm in plugin.arg_spec]


def _fused_steps_of(runner, steps: int) -> int:
    """Steps one launch of the plain fused ``runner`` covers over a run of
    ``steps``: the whole run for a single-launch device mode, the entry's tile
    for an automatic host loop, else its fixed steps per launch."""
    if runner._fast_mode is not None:
        return max(int(steps), 1)
    if runner.auto:
        return max(int(runner._k_max or 1), 1)
    return max(int(runner.steps_per_launch), 1)


def time_launch(bound, reps: int, fused_steps: int = 1) -> list:
    """Seconds of each of ``reps`` launches of the bound plan ``bound`` over
    its first partition, every repetition kept (calibration; private).

    The host team run is timed with ``steady_clock``, a device launch with a
    CUDA event pair. A plan that reads a steps-per-launch word runs
    ``fused_steps`` steps per launch."""
    from . import _core
    from ._plan_binding import DEVICE_CUDA

    _split.single(bound, "eagle dispatch timing")
    word = bound._planes.get(FUSED_STEPS_PLANE) if isinstance(bound._planes, dict) else None
    if word is not None:
        word[...] = fused_steps
    part = bound._parts[0]
    if bound._device_type != DEVICE_CUDA:
        return list(_core.time_launch(bound._entry, bound._addrs, part, reps))
    import cupy

    block = int(importlib.import_module("eagle.launch")._resolved_block(
        bound._entry, part.count, None, declared=bound._declared_block))
    return list(_core.time_launch(bound._entry, bound._addrs, part, reps, device=True,
                                  stream=int(cupy.cuda.get_current_stream().ptr), block=block))


_PARTS: dict = {}
_PARTS_LOCK = threading.Lock()
_PARTS_MAX = 32


def _split_parts(kernel, group: str, device: bool) -> tuple:
    """``(cost, unit plugins, classify plugin or None, sweep plan or None)``
    of ``kernel``'s ``group``, built once per kernel, group and target (and,
    on the host, host profile: it decides whether library math vectorises, so
    the cost depends on it; the IR walks and the bundle key are the expensive
    half of a short run)."""
    import dataclasses

    import hawk
    from hawk.compile.toolchain import host_profile
    from hawk.ir import dispatch_costs
    from hawk.trace.kernel import Kernel

    from ._plan_deploy import _build_by_name, auto

    profile = None if device else host_profile()
    key = (id(kernel), group, device, profile)
    with _PARTS_LOCK:
        hit = _PARTS.get(key)
    if hit is not None and hit[0] is kernel:
        return hit[1]
    step = _step(kernel)
    cost = dispatch_costs(step.walk, profile)[group]
    seg = hawk.resolve_dispatch(kernel, {group: "segmented"})
    by_name, _ = _build_by_name([seg], None if device else ("host",), None)
    units = [by_name[f"{seg.name}__k{j}"] for j in range(len(cost.branches))]
    sweep = None
    if not cost.static_selector:
        twin = _step(hawk.resolve_dispatch(kernel, {group: "switch"}))
        guard = dataclasses.replace(twin.kind.guard, active_set=True)
        sweep = auto(Kernel(twin.name, twin.sinks, twin.planes,
                            kind=dataclasses.replace(twin.kind, guard=guard)))
    parts = (cost, units, by_name.get(f"{seg.name}__cls"), sweep)
    with _PARTS_LOCK:
        if len(_PARTS) >= _PARTS_MAX:
            _PARTS.pop(next(iter(_PARTS)))
        _PARTS[key] = (kernel, parts)
    return parts


class DispatchRunner:
    """Runs one kernel whose ``"auto"`` group is decided at run time, on the
    host or the device (wherever the planes live). Interface of
    :class:`eagle.Runner` that :class:`eagle.Simulation` uses: ``run``,
    ``reset``, ``n``, ``loop``, ``pipeline``, ``active``, ``finished``,
    ``every``, ``record``.

    ``forced`` is ``"segmented"`` for a group forced to that policy (never
    fused, never re-decided), or ``None`` to decide by measurement.

    A static selector is classified once per run by a selector-only kernel:
    the classes give the exact mix, the rule decides, and

    * on ``"switch"`` the rest of the run is the plain fused run (``plain``,
      the same runner a forced ``"switch"`` builds, with its own steps per
      launch); the next look is at the next :meth:`run`;
    * on ``"segmented"`` the samples are sorted by class once and every step
      launches one unit per branch over its run, nothing else; the mix is
      looked at again every :data:`~eagle._dispatch_policy.DISPATCH_EVERY`
      steps, from the sorted classes of the live samples.

    A forced split of a state-dependent selector runs guarded units that mark
    the samples whose branch moved; a fused sweep finishes those, and the batch
    is sorted again when more than :data:`~eagle._active_set.STALE_MAX` of it
    went stale."""

    loop = None
    active = None
    pipeline = None
    every = None
    record = None  # a split run is never recorded: simulate refuses record=True with dispatch=
    _fast_mode = None

    def __init__(self, kernel, group, planes, mask, *, max_steps, plain, forced=None,
                 theta=True):
        self._busy = BusyGuard("eagle.Runner")
        self._step = _step(kernel)
        self._group, self._forced = group, forced
        self._planes, self._mask, self._max_steps = planes, mask, max_steps
        self._plain_of, self._plains = plain, {}
        self.terminated = mask
        self.n = int(mask.shape[-1])
        self._device = type(mask).__module__.split(".")[0] == "cupy"
        if self._device:
            import cupy

            self._xp = cupy
            self._target = "device"
            from ._device_props import device_props

            self._props = device_props(int(mask.device.id))
        else:
            self._xp, self._target, self._props = np, "host", None
        xp = self._xp
        self._cost, self._unit_plugins, self._classify_plugin, sweep = _split_parts(
            kernel, group, self._device)
        self._static = bool(self._cost.static_selector)
        self._k = len(self._cost.branches)
        self._state = _policy.DispatchState("segmented" if forced == "segmented" else "switch")
        self._gain, self._mix, self._theta = 1.0, (0,) * self._k, theta
        self._cells = xp.zeros(6, dtype=xp.uint32)
        self.finished = self._cells[2:3]
        self._aset = self._cls = None
        self._sorted = False
        if not self._static:
            self._sweep_plan = sweep
            self._smap = xp.zeros(self.n, dtype=xp.int32)
            self._scount = xp.zeros(1, dtype=xp.uint32)
        self._seg_ready = False

    # -- binding -- #
    def _bind(self, plan, extra=None):
        xp = self._xp
        plan = plan.select(self._planes) if hasattr(plan, "select") else plan
        spec = plan.plugin.arg_spec
        bind = {nm: self._planes[nm] for _r, nm in spec if nm in self._planes}
        bind.update({nm: v for nm, v in (extra or {}).items() if ("lookup", nm) in spec
                     or any(n == nm for _r, n in spec)})
        if ("lookup", FINISHED_PLANE) in spec:
            bind[FINISHED_PLANE] = self._cells[2:3]
        if ("lookup", FUSED_STEPS_PLANE) in spec:
            bind[FUSED_STEPS_PLANE] = xp.ones(1, dtype=xp.int64)
        if ("lookup", MAP_PLANE) in spec:
            bind[MAP_PLANE], bind[COUNT_PLANE] = self._smap, self._scount
        return plan.bind(**bind)

    def _classify_setup(self) -> None:
        from ._plan_deploy import AutoPlan

        xp, n, k, g = self._xp, self.n, self._k, self._group
        self._cls = xp.zeros(n, dtype=xp.int64)
        self._offsets = xp.zeros(k + 1, dtype=xp.int64)
        self._extra = {f"{g}_class": self._cls, f"{g}_offsets": self._offsets}
        if not self._static:
            self._stale = xp.zeros(n, dtype=xp.bool_)
            self._extra[f"{g}_stale"] = self._stale
        self._classify = (self._bind(AutoPlan(self._classify_plugin, {}), self._extra)
                          if self._classify_plugin is not None else None)

    def _segmented_setup(self) -> None:
        from ._plan_deploy import AutoPlan

        if self._cls is None:
            self._classify_setup()
        xp, n, k, extra = self._xp, self.n, self._k, self._extra
        self._cls8 = xp.zeros(n, dtype=xp.uint8)
        self._aset = ActiveSet(self._mask, reorder=self._theta, classes=k, class_plane=self._cls8)
        readers = list(self._unit_plugins)
        if self._classify_plugin is not None:
            readers.append(self._classify_plugin)
        owned = {id(a): a for p in readers for role, nm in p.arg_spec
                 if role in PER_SAMPLE_ROLES and role != "terminated"
                 for a in [self._planes.get(nm, extra.get(nm))]
                 if hasattr(a, "shape") and a.shape[-1:] == (n,) and a is not self._mask
                 and a is not self._cls}
        self._aset.own(*owned.values())
        self._units = [_split.single(self._bind(AutoPlan(p, {}), extra), "eagle dispatch")
                       for p in self._unit_plugins]
        self._sweep = None if self._static else self._bind(self._sweep_plan)
        self._run_len = np.zeros(k, dtype=np.int64)
        self._blocks = [256] * k
        self._sorted = False
        self._stale_now = 0
        self._seg_ready = True

    # -- launching one bound plan over [0, count) -- #
    def _launch(self, bound, count: int, block: int = 256) -> None:
        part = eexec.Partition(0, int(count), self.n)
        if self._device:
            eexec.DeviceKernel.run(bound._entry, bound._addrs, part,
                                   stream=self._xp.cuda.get_current_stream().ptr, block=block)
        else:
            eexec.HostTeam.run(bound._entry, bound._addrs, part)

    def _classes(self) -> None:
        """The class of every sample, from the selector-only kernel."""
        if self._cls is None:
            self._classify_setup()
        self._launch(self._classify, self.n)

    def _sort(self) -> None:
        if not self._seg_ready:
            self._segmented_setup()
        aset = self._aset
        if self._static:
            self._cls8[...] = self._cls
        aset.class_sort()
        self._offsets[...] = aset.offsets
        self._run_len = np.diff(self._host(self._offsets).astype(np.int64))
        if self._device:
            _launch = importlib.import_module("eagle.launch")
            self._blocks = [int(_launch._resolved_block(u._entry, max(int(c), 1), None,
                                                        declared=u._declared_block))
                            for u, c in zip_strict(self._units, self._run_len)]
        self._sorted = True

    # -- one split step -- #
    def _step_segmented(self) -> None:
        if not self._static:
            self._step_drifting()
            return
        for unit, count, block in zip_strict(self._units, self._run_len, self._blocks):
            if count > 0:
                self._launch(unit, count, block)

    def _step_drifting(self) -> None:
        xp, aset = self._xp, self._aset
        live_count = int(xp.count_nonzero(self._mask == 0))
        aset.stale_count[0] = self._stale_now
        if not self._sorted or self._stale_now > STALE_MAX * max(live_count, 1):
            self._cls8[...] = xp.clip(self._cls, 0, self._k - 1).astype(xp.uint8)
            self._sort()
        for unit, count, block in zip_strict(self._units, self._run_len, self._blocks):
            if count > 0:
                self._launch(unit, count, block)
        stale = xp.flatnonzero(self._stale & (self._mask == 0))
        self._stale_now = int(stale.size)
        if self._stale_now:
            self._smap[: self._stale_now] = stale
            self._scount[0] = self._stale_now
            self._sweep.launch()

    def _host(self, array):
        return array.get() if self._device else array

    # -- the decision -- #
    def _live_mix(self) -> np.ndarray:
        """Live samples per class: from the sorted class plane while sorted,
        else from the classes in sample order."""
        xp = self._xp
        cls = self._cls8 if self._sorted else self._cls
        if not self._device:
            if self._sorted or np.count_nonzero(self._mask):
                cls = cls[self._mask == 0]
            return np.bincount(cls, minlength=self._k)
        # reductions and one read, never a device bincount: it piles every
        # sample of a dominant class onto one atomic
        if not self._sorted:            # the whole batch live: no mask in the counts
            others = [xp.count_nonzero(cls if self._k == 2 else cls == j)
                      for j in range(1, self._k)]
            done, *rest = (int(v) for v in xp.stack([xp.count_nonzero(self._mask), *others]).get())
            if done == 0:
                return np.array([self.n - sum(rest), *rest], dtype=np.int64)
        live = self._mask == 0
        counts = xp.stack([xp.count_nonzero((cls == j) & live) for j in range(self._k)])
        return counts.get().astype(np.int64)

    def _look(self, steps_left: int) -> None:
        counts = self._live_mix()
        if counts.sum() == 0:
            return
        self._mix = tuple(int(c) for c in counts)
        if self._forced is None:
            fused = _fused_steps_of(self._plain(steps_left), steps_left)
            self._gain = _policy.predicted_gain(self._cost, counts, target=self._target,
                                                props=self._props, fused_steps=fused,
                                                remaining_steps=steps_left)
            self._state.apply(self._gain)

    def _plain(self, steps: int):
        """The plain fused runner over ``steps`` steps (built once per length)."""
        runner = self._plains.get(steps)
        if runner is None:
            runner = self._plains[steps] = self._plain_of(steps)
        return runner

    # -- the loop -- #
    def _live(self) -> bool:
        return bool((self._mask == 0).any())

    def _run_plain(self, steps: int):
        if self._sorted:
            self._aset.restore()
            self._sorted = False
        return self._plain(steps).run()

    def run(self) -> RunReport:
        with self._busy("run"):
            return self._run()

    def _run(self) -> RunReport:
        steps, split_steps, plain = 0, 0, None
        if self._static:
            self._classes()
            self._look(self._max_steps)
        elif not self._seg_ready:
            self._segmented_setup()
        while steps < self._max_steps:
            if self._state.policy == "switch":
                plain = self._run_plain(self._max_steps - steps)
                steps += plain.steps
                break
            if not self._sorted and self._static:
                self._sort()
            self._step_segmented()
            steps += 1
            split_steps += 1
            if self._state.due():
                if not self._live():
                    break
                self._look(self._max_steps - steps)
        if self._sorted:
            self._aset.restore()
        self._sorted = False
        sorts = int(self._aset.class_sorts[0]) if self._aset is not None else 0
        if self._aset is not None:
            self._aset.class_sorts[0] = 0
        finished = int(self._xp.count_nonzero(self._mask))
        if self._device:
            self._xp.cuda.Device().synchronize()
        launches = split_steps * self._k + (plain.launches if plain else 0)
        by_k = dict(plain.launches_by_k) if plain else {}
        if split_steps:
            by_k[1] = by_k.get(1, 0) + split_steps
        return RunReport(
            n=self.n, finished=finished, done=finished == self.n, launches=launches,
            steps=steps, compactions=0, reorders=0,
            build_s=plain.build_s if plain else 0.0, launches_by_k=by_k,
            exact_steps=plain.exact_steps if plain else True,
            mapped=(plain.mapped if plain else 0) + split_steps * self.n,
            mode=plain.mode if plain else "band",
            dispatch={self._group: (self._state.policy, self._mix, self._gain)},
            class_sorts=sorts)

    def reset(self) -> None:
        self._mask[...] = 0
        self._sorted = False
        if self._aset is not None:
            self._aset.reset()
        for runner in self._plains.values():
            runner.reset()
