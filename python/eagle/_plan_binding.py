# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The capture-legal launch door: a plan bound to the caller's planes."""

from __future__ import annotations

import ctypes
import importlib

import numpy as np

from . import _counters, _layout
from . import exec as eexec
from .abi import DEVICE_CPU, DEVICE_CUDA
from .roles import OUTPUT_ROLES, PER_SAMPLE_ROLES
from ._plan_pack import (
    _integer_uniforms,
    _mutable_shapes,
    _n_from_kw,
    _pack_args,
    _pack_one,
)
from ._plan_planes import (
    FUSED_STEPS_SLOT,
    _BIND_TARGETS,
    _adapt_planes,
    _check_bound_names,
    _check_plane,
    _check_uniform,
    _declared_widths,
    _plane_dtype,
)



def one_step_word(arg_spec, planes: dict, framework: str = "numpy") -> dict:
    """``planes`` plus a one-cell ``int64`` word of ``1`` under
    ``fused_steps`` when the artifact reads that word and the caller binds
    none, so a launch outside :func:`eagle.until_done` takes exactly one
    step. ``framework`` is where the word lives (``"cupy"`` for a device
    bind, ``"numpy"`` otherwise). Any other artifact gets ``planes`` back
    unchanged."""
    name = FUSED_STEPS_SLOT[1]
    if FUSED_STEPS_SLOT not in tuple(arg_spec) or name in planes:
        return planes
    if framework == "cupy":
        import cupy as xp
    else:
        xp = np
    return {**planes, name: xp.ones(1, dtype=xp.int64)}


def _bind_plan(p: Plan, planes: dict) -> BoundPlan:
    """:meth:`Plan.bind`'s body — see that method for the contract."""
    plugin = p.plugin
    arg_spec = getattr(plugin, "arg_spec", None)
    if arg_spec is None:
        raise ValueError(
            "eagle.plan: plugin has no .arg_spec -- .bind() cannot pack args"
        )
    target = _BIND_TARGETS.get(p.structure.name)
    if target is None:
        raise ValueError(
            f"eagle.plan.bind: the capture-legal door serves the single-process "
            f"structures {sorted(_BIND_TARGETS)}, not {p.structure.name!r}: a "
            "rank/group run is a collective, and a collective has no single "
            "launch to record; drive it through Plan.run()"
        )
    attr, device_type, framework = target
    entry = getattr(plugin, attr, None)
    if entry is None:
        raise ValueError(
            f"eagle.plan.bind: this plan runs under eagle.exec.{p.structure.name} "
            f"but the plugin declares no .{attr} to launch"
        )

    planes = one_step_word(arg_spec, planes, framework)
    _check_bound_names(arg_spec, planes)
    planes, adapted = _adapt_planes(plugin, arg_spec, planes, door="bind")
    n = _n_from_kw(arg_spec, planes)
    parts = p.partitions_for(n)
    active = _active_count_plane(arg_spec, planes, parts)
    dtype = _plane_dtype(plugin)
    widths = _declared_widths(plugin)
    vec_mut, mat_mut = _mutable_shapes(plugin)
    for role, name in arg_spec:
        if role == "nsamples":
            continue
        if role == "uniform":
            _check_uniform(name, planes[name])
        else:
            _check_plane(role, name, planes[name], n=n, dtype=dtype,
                         width=widths.get(name), framework=framework)

    boxes, addrs = _pack_args(
        arg_spec, planes, planes, n, device_type,
        vec_mutables=vec_mut, mat_mutables=mat_mut, dtype=dtype,
        int_uniforms=_integer_uniforms(plugin),
    )
    return BoundPlan(
        p, arg_spec, dict(planes), n, parts, boxes, addrs, int(entry),
        device_type=device_type, framework=framework, dtype=dtype, widths=widths,
        vec_mutables=vec_mut, mat_mutables=mat_mut,
        int_uniforms=_integer_uniforms(plugin), active_count=active,
        copies=[a for a in adapted.values() if a.scratch is not None],
    )


def _active_count_plane(arg_spec, planes, parts):
    """The bound active-set count plane, when the artifact reads an index
    map (:mod:`eagle._active_set`'s ``active_map``/``active_count``), else
    ``None``. Such a kernel addresses samples through one map over the whole
    batch, so a plan split into several partitions is refused here: each
    partition would index the map from its own base and no longer follow the
    live samples."""
    from ._active_set import COUNT_PLANE, MAP_PLANE

    declared = {name for role, name in arg_spec if role == "lookup"}
    if not {MAP_PLANE, COUNT_PLANE} <= declared:
        return None
    if len(parts) != 1:
        raise ValueError(
            f"eagle.plan.bind: this artifact reads an active-set index map "
            f"({MAP_PLANE!r}/{COUNT_PLANE!r}), which addresses the whole batch; "
            f"bind it under a single partition, not {len(parts)}"
        )
    from ._active_set import check_bound_planes

    check_bound_planes(planes[MAP_PLANE], [
        (role, name, planes[name]) for role, name in arg_spec
        if name in planes and name not in (MAP_PLANE, COUNT_PLANE)])
    return planes[COUNT_PLANE]


class BoundPlan:
    """A :class:`Plan` with its argument block already packed over the
    caller's own planes — the thing a captured consumer holds.

    It owns the packed host-side mirror block and a reference to every
    plane those mirrors point at (a captured graph holds raw device
    pointers, so a plane freed too early reads as a correct answer once and
    somebody else's memory afterwards), but no plane itself: nothing here
    is allocated, uploaded, cast or copied home, at bind or at launch.

    The bound block is a snapshot, frozen at bind; :meth:`rebind` changes
    only the slots it is given."""

    __slots__ = (
        "plan", "_arg_spec", "_planes", "_n", "_parts", "_boxes", "_addrs",
        "_entry", "_device_type", "_framework", "_dtype", "_widths",
        "_vec_mutables", "_mat_mutables", "_int_uniforms", "_slot_of", "_cupy",
        "_active_count", "_copies", "_persist_fn", "_range_fn",
        "_gen",
    )

    #: Sentinel for "not yet looked up" (the persist/range entry's own cupy
    #: Function object is never falsy, but comparing by identity against a
    #: dedicated sentinel is still cheaper and clearer than relying on that).
    _PERSIST_UNRESOLVED = object()

    def __init__(self, plan, arg_spec, planes, n, parts, boxes, addrs, entry, *,
                 device_type, framework, dtype, widths, vec_mutables,
                 mat_mutables, int_uniforms=frozenset(), active_count=None,
                 copies=()):
        self.plan = plan
        self._arg_spec = tuple(arg_spec)
        self._planes = planes
        self._n = int(n)
        self._parts = tuple(parts)
        self._boxes = list(boxes)
        self._addrs = list(addrs)
        self._entry = entry
        self._device_type = device_type
        self._framework = framework
        self._dtype = dtype
        self._widths = widths
        self._vec_mutables = vec_mutables
        self._mat_mutables = mat_mutables
        self._int_uniforms = int_uniforms
        self._slot_of = {name: i for i, (_role, name) in enumerate(self._arg_spec)}
        # the bound active-set count plane (see _active_count_plane), or None
        self._active_count = active_count
        # imported once, at bind: resolving cupy per launch would cost a
        # sys.modules lookup on a path that promises launch-and-nothing-else
        self._cupy = None
        if device_type == DEVICE_CUDA:
            import cupy

            self._cupy = cupy
        # sample-major planes copied at bind (eagle._layout.Adapted):
        # refreshed before each launch, written ones copied back after
        self._copies = {a.name: a for a in copies}
        for a in self._copies.values():
            self._warm(a)
        # the persist/range entries' cupy Functions, looked up lazily (not
        # every bound plan's launch ever uses them) and cached once resolved
        self._persist_fn = BoundPlan._PERSIST_UNRESOLVED
        self._range_fn = BoundPlan._PERSIST_UNRESOLVED
        # bumped by rebind: a prepared launch block packed before it is stale
        self._gen = 0

    def _warm(self, a) -> None:
        """Run both copies once at bind, so a launch recorded into a graph
        never compiles a copy kernel mid-capture (value-preserving: the
        scratch was just made from the caller's array)."""
        a.copy_in()
        if a.writes:
            a.copy_back()

    @property
    def n(self) -> int:
        """The sample count this block was packed for (frozen at bind)."""
        return self._n

    def device_capacity(self) -> int:
        """The safe "no active-set rebuild" bound at this bound plan's
        kernel/device (see :func:`eagle._launch_policy.latency_regime_capacity`
        — the smaller of hardware residency and the issue-pipe saturation
        point, NOT residency alone). ``0`` for a host-team plan, or when
        device/kernel properties are unavailable; the caller
        (:mod:`eagle._until_done`'s auto policy) treats ``0`` as "unknown",
        which never satisfies ``n <= capacity``. By module name (not a
        top-level import), matching :meth:`launch`'s own lazy
        ``eagle.launch`` lookup -- this stays import-cycle-free."""
        if self._device_type != DEVICE_CUDA:
            return 0
        _launch = importlib.import_module("eagle.launch")
        from ._launch_policy import latency_regime_capacity

        return latency_regime_capacity(
            _launch._device_props(), _launch._kernel_attrs(self._entry)
        )

    def persist_entry(self):
        """This bound plan's ``<kernel>_persist`` cupy ``Function`` --
        same compiled unit as the fused entry (hawk's
        ``hawk.emit.cuda.persist_entry``). ``None`` when unavailable: a
        host-team plan, a non-automatic kernel, or an artifact built
        before the persist entry existed. Resolved once, from the loaded
        device module the plugin's ``_keepalive`` already holds -- no new
        compile, no new module load."""
        if self._persist_fn is not BoundPlan._PERSIST_UNRESOLVED:
            return self._persist_fn
        fn = None
        if self._device_type == DEVICE_CUDA:
            plugin = self.plan.plugin
            keep = getattr(plugin, "_keepalive", ())
            loaded = keep[-1] if keep else None
            module = getattr(loaded, "module", None)
            name = getattr(plugin, "name", None)
            if module is not None and name:
                try:
                    fn = module.get_function(f"{name}_persist")
                except Exception:
                    fn = None
        self._persist_fn = fn
        return fn

    def launch_persist(self, stream=None, *, grid, counter, util=None,
                      block: int = 256, steps=None, stepsum=None,
                       prepared=None) -> None:
        """Launch ``<kernel>_persist`` once -- the persistent launch's whole
        shape: lanes fetch ``base + atomicAdd(counter, 1)`` until the
        partition's own ``count``, an already-Terminated sample skipped,
        each running up to the bound ``fused_steps`` word's budget before
        writing back and fetching the next; no WHILE graph, no active-set
        map, no policy kernel -- this one call is the whole run.

        ``grid`` is the block count and ``block`` the block size -- the
        caller's own choice, both geometry knobs this method does not
        resolve itself (:func:`eagle._launch_policy.persistent_geometry` is
        the documented rule a caller derives them from, from the entry's own
        register-limited occupancy and the SM count; ``block=256`` here is
        only a bare fallback for a caller that does not). ``counter`` is a
        one-element uint32 cupy array the caller has already memset to 0
        (not done here, so a caller replaying this inside a captured graph
        can fold the reset into a graph node instead). ``util`` is an
        optional 2-element uint64 cupy array (active lanes,
        warp-iterations*32); omitted, the entry gets a null pointer and
        skips counting. ``steps`` is an optional one-element uint32 cupy
        array the caller has zeroed: the entry maxes each sample's steps
        taken in this launch into it (the run's exact step count); omitted,
        a null pointer and no report. The entry's trailing ABI is
        ``base, count, nSamples, hawk_next, hawk_util, hawk_steps, hawk_stepsum``;
        ``stepsum`` is an optional one-element uint64 cupy array the caller
        has zeroed, into which the entry adds every step it executes (the
        run's total; omitted, a null pointer and no report).

        Refused for anything but a single, contiguous partition -- the
        only shape the persist entry's ``base``/``count`` pair means.
        ``prepared`` (from :meth:`prepare_persist`) skips the checks and the
        packing and only launches: the other arguments are then unread."""
        if prepared is not None:
            return prepared(stream)
        return self.prepare_persist(
            grid=grid, counter=counter, util=util, block=block, steps=steps,
            stepsum=stepsum)(stream)

    def prepare_persist(self, *, grid, counter, util=None, block: int = 256,
                        steps=None, stepsum=None):
        """:meth:`launch_persist`'s checks and argument packing done ONCE:
        returns ``call(stream=None)`` that only issues the driver launch (a
        resident runner's per-run cost). The packed block is re-made if
        :meth:`rebind` ran since."""
        fn = self.persist_entry()
        if fn is None:
            raise ValueError(
                "eagle.plan: this bound plan has no persist entry to launch "
                "(BoundPlan.persist_entry() returned None -- not an automatic "
                "kernel, not built for CUDA, or the artifact predates it)"
            )
        if len(self._parts) != 1:
            raise ValueError(
                "eagle.plan: BoundPlan.launch_persist needs exactly one "
                f"contiguous partition, got {len(self._parts)}"
            )
        part = self._parts[0]
        # hawk_next is a uint32 the lanes keep fetching from until they pass
        # count: count plus one overshooting fetch per lane must not wrap it
        if part.count + int(grid) * int(block) > 2**32:
            raise ValueError(
                "eagle.plan: BoundPlan.launch_persist runs fewer than 2**32 "
                f"samples per launch (count {part.count:,} plus "
                f"{int(grid) * int(block):,} lanes); split the batch"
            )
        util_ptr = 0 if util is None else int(util.data.ptr)
        steps_ptr = 0 if steps is None else int(steps.data.ptr)
        stepsum_ptr = 0 if stepsum is None else int(stepsum.data.ptr)
        extra = (
            ctypes.c_longlong(part.base), ctypes.c_longlong(part.count),
            ctypes.c_longlong(part.n_samples),
            ctypes.c_void_p(int(counter.data.ptr)), ctypes.c_void_p(util_ptr),
            ctypes.c_void_p(steps_ptr), ctypes.c_void_p(stepsum_ptr),
        )
        grid, block = int(grid), int(block)
        cupy, ptr = self._cupy, fn.kernel.ptr
        addrs = self._addrs + [ctypes.addressof(box) for box in extra]
        packed = [self._gen, (ctypes.c_void_p * len(addrs))(*addrs)]

        def call(stream=None) -> None:
            if packed[0] != self._gen:
                addrs = self._addrs + [ctypes.addressof(box) for box in extra]
                packed[:] = [self._gen, (ctypes.c_void_p * len(addrs))(*addrs)]
            params = packed[1]
            cupy.cuda.driver.launchKernel(
                ptr, grid, 1, 1, block, 1, 1, 0, _stream_handle(stream, cupy),
                ctypes.addressof(params), 0,
            )

        return call

    def range_entry(self):
        """This bound plan's ``<kernel>_range`` cupy ``Function`` -- an
        automatic kernel's contiguous-range sibling entry (same cubin as
        the kernel's own entry; for an active-set kernel
        ``active_map``/``active_count`` ride along in the SAME packed
        argument block, bound but never read), which also reports the
        launch's exact step count. ``None`` when unavailable: a host-team
        plan, a non-automatic kernel, or a unit built without hawk's
        ``HAWK_FAST_ENTRIES=1`` define. Resolved once,
        same lazy pattern as :meth:`persist_entry`."""
        if self._range_fn is not BoundPlan._PERSIST_UNRESOLVED:
            return self._range_fn
        fn = None
        if self._device_type == DEVICE_CUDA:
            plugin = self.plan.plugin
            keep = getattr(plugin, "_keepalive", ())
            loaded = keep[-1] if keep else None
            module = getattr(loaded, "module", None)
            name = getattr(plugin, "name", None)
            if module is not None and name:
                try:
                    fn = module.get_function(f"{name}_range")
                except Exception:
                    fn = None
        self._range_fn = fn
        return fn

    def launch_range(self, stream=None, *, block=None, steps=None, stepsum=None,
                     prepared=None) -> None:
        """Launch ``<kernel>_range`` once over this bound plan's single
        contiguous partition -- an active-set automatic kernel's fast-path
        equivalent of the plain kernel's own :meth:`launch`: the SAME packed
        argument block the map entry uses (``active_map``/``active_count``
        included, unread), plus ``base, count, nSamples, hawk_steps,
        hawk_stepsum``; no
        counter, no util (that is the persist entry's own shape,
        :meth:`launch_persist`). ``steps`` is an optional one-element uint32
        cupy array the caller has zeroed, into which the entry maxes each
        sample's steps taken (the launch's exact step count); omitted, a
        null pointer and no report. ``stepsum`` is the persist entry's
        one-element uint64 total of executed steps (:meth:`launch_persist`).
        The entry itself picks single-step vs fused from the bound
        ``fused_steps`` word (1 vs anything else); this call does not choose.

        ``block`` (``None``: resolved) takes the SAME policy :meth:`launch`
        already resolves its own block with (:func:`eagle.launch._resolved_block`
        -> ``eagle._launch_policy.resolve_block``, keyed off this plan's own
        entry -- the range entry shares its cubin/registers, so the same
        block size applies); the grid follows
        :class:`eagle.exec.DeviceKernel`'s own grid-from-count rule, over
        ``count`` samples densely (one lane per sample, no work-stealing).
        A count of 0 launches nothing.

        Refused for anything but a single, contiguous partition -- the only
        shape the range entry's ``base``/``count`` pair means.
        ``prepared`` (from :meth:`prepare_range`) skips the checks, the
        block/grid resolution and the packing and only launches."""
        if prepared is not None:
            return prepared(stream)
        return self.prepare_range(block=block, steps=steps, stepsum=stepsum)(stream)

    def prepare_range(self, *, block=None, steps=None, stepsum=None):
        """:meth:`launch_range`'s checks, block/grid resolution and argument
        packing done ONCE: returns ``call(stream=None)`` that only issues the
        driver launch (a count of 0 launches nothing). The packed block is
        re-made if :meth:`rebind` ran since."""
        fn = self.range_entry()
        if fn is None:
            raise ValueError(
                "eagle.plan: this bound plan has no range entry to launch "
                "(BoundPlan.range_entry() returned None -- not an automatic "
                "active-set kernel, not built for CUDA, or the artifact "
                "predates it)"
            )
        if len(self._parts) != 1:
            raise ValueError(
                "eagle.plan: BoundPlan.launch_range needs exactly one "
                f"contiguous partition, got {len(self._parts)}"
            )
        part = self._parts[0]
        if part.count <= 0:
            return lambda stream=None: None
        _launch = importlib.import_module("eagle.launch")
        resolved = int(_launch._resolved_block(self._entry, part.count, block))
        grid = int(eexec.DeviceKernel.grid(part.count, resolved))
        extra = (
            ctypes.c_longlong(part.base), ctypes.c_longlong(part.count),
            ctypes.c_longlong(part.n_samples),
            ctypes.c_void_p(0 if steps is None else int(steps.data.ptr)),
            ctypes.c_void_p(0 if stepsum is None else int(stepsum.data.ptr)),
        )
        cupy, ptr = self._cupy, fn.kernel.ptr
        addrs = self._addrs + [ctypes.addressof(box) for box in extra]
        packed = [self._gen, (ctypes.c_void_p * len(addrs))(*addrs)]

        def call(stream=None) -> None:
            if packed[0] != self._gen:
                addrs = self._addrs + [ctypes.addressof(box) for box in extra]
                packed[:] = [self._gen, (ctypes.c_void_p * len(addrs))(*addrs)]
            params = packed[1]
            cupy.cuda.driver.launchKernel(
                ptr, grid, 1, 1, resolved, 1, 1, 0, _stream_handle(stream, cupy),
                ctypes.addressof(params), 0,
            )

        return call

    @property
    def partitions(self) -> tuple:
        """The :class:`eagle.exec.Partition` triples :meth:`launch` issues,
        in order (resolved at bind)."""
        return self._parts

    @property
    def names(self) -> tuple:
        """The bound names, in ``arg_spec`` order."""
        return tuple(name for _role, name in self._arg_spec)

    @property
    def planes(self):
        """The bound values by name (read-only): the arrays the packed block
        points at (or, for a sample-major plane, the component-major view
        that was packed)."""
        from types import MappingProxyType

        return MappingProxyType(self._planes)

    def __repr__(self) -> str:
        return (
            f"BoundPlan({self.plan.structure.name}, n={self._n}, "
            f"partitions={len(self._parts)}, names={list(self.names)})"
        )

    def launch(self, stream=None, *, grid=None, block=None) -> None:
        """Issue this plan's launches — one per partition — and nothing
        else. Returns ``None``: the answers are already in the caller's own
        planes.

        ``stream`` is the CUDA stream to issue on (an int, or anything
        carrying ``.ptr``); ``None`` resolves to the framework's current
        stream at launch time, not at bind, since the stream a graph
        captures on is not the one the plan was bound on. A host_team run
        has no stream and is refused one.

        ``grid`` is refused (eagle derives it from each partition's
        ``count``); ``block=None`` defers to eagle's launch-policy resolver
        (``eagle.launch._resolved_block``), the same one the v1 door
        consults."""
        if grid is not None:
            raise ValueError(
                "eagle.plan: BoundPlan.launch takes no grid — eagle derives it "
                f"from each partition's own count; got grid={grid!r}"
            )
        addrs = self._addrs
        if self._device_type == DEVICE_CPU:
            if stream is not None:
                raise ValueError(
                    "eagle.plan: a host_team run has no stream to issue on; "
                    f"got stream={stream!r}"
                )
            parts = self._parts
            if self._active_count is not None:
                # host live count is a free read: split only [0, count)
                live = min(int(self._active_count[0]), self._n)
                if live <= 0:
                    return None
                parts = (eexec.Partition(0, live, self._n),)
            copies = self._copies.values()
            for a in copies:
                a.copy_in()
            for part in parts:
                eexec.HostTeam.run(self._entry, addrs, part)
            for a in copies:
                if a.writes:
                    a.copy_back()
            return None
        # by module name: eagle exports a `launch` function that shadows the
        # submodule, and a per-call lookup picks up a consumer's monkeypatch
        _launch = importlib.import_module("eagle.launch")
        handle = _stream_handle(stream, self._cupy)
        if self._copies:
            # the copies ride the launch's own stream: ordered, no sync
            with self._cupy.cuda.ExternalStream(handle):
                for a in self._copies.values():
                    a.copy_in()
        for part in self._parts:
            eexec.DeviceKernel.run(
                self._entry, addrs, part, stream=handle,
                block=int(_launch._resolved_block(self._entry, part.count, block)),
            )
        if self._copies:
            with self._cupy.cuda.ExternalStream(handle):
                for a in self._copies.values():
                    if a.writes:
                        a.copy_back()
        return None

    def rebind(self, /, **changed):
        """Re-pack only the named slots — one box per name, in one call: for
        a resident pipeline whose planes move (a double-buffered state
        swaps) while the plan, partitions and sample count stay the same.
        ``self`` is positional-only, as ``Plan.run``'s and ``Plan.bind``'s
        are.

        A rebound plane is checked exactly as the original was, at the same
        frozen ``n`` — a different length is a different run and must be
        bound afresh. Returns this same :class:`BoundPlan`, updated in
        place; it cannot change a launch already recorded into a graph,
        only what the NEXT capture or launch sees."""
        for name, value in changed.items():
            slot = self._slot_of.get(name)
            if slot is None:
                raise ValueError(
                    f"eagle.plan.rebind: {name!r} is not a bound name "
                    f"(bound: {list(self.names)})"
                )
            role = self._arg_spec[slot][0]
            if role == "nsamples":
                raise ValueError(
                    f"eagle.plan.rebind: {name!r} is the 'nsamples' role, which "
                    "is derived from the bound planes and never bound"
                )
            if role == "uniform":
                _check_uniform(name, value)
            else:
                a = None
                width = self._widths.get(name, 1)
                if role in PER_SAMPLE_ROLES and width > 1:
                    a = _layout.adapt(name, value, (width,),
                                      writes=role in OUTPUT_ROLES)
                    if a is not None:
                        value = a.native
                _check_plane(role, name, value, n=self._n, dtype=self._dtype,
                             width=self._widths.get(name),
                             framework=self._framework)
                self._copies.pop(name, None)
                if a is not None and a.scratch is not None:
                    self._copies[name] = a
                    self._warm(a)
            self._planes[name] = value
            if self._active_count is not None:
                from ._active_set import COUNT_PLANE

                if name == COUNT_PLANE:
                    self._active_count = value
            box = _pack_one(
                role, name, self._planes, self._planes, self._n,
                self._device_type, vec_mutables=self._vec_mutables,
                mat_mutables=self._mat_mutables, dtype=self._dtype,
                int_uniforms=self._int_uniforms,
            )
            self._boxes[slot] = box
            self._addrs[slot] = ctypes.addressof(box)
        self._gen += 1
        _counters.bump("rebinds")  # only on a completed rebind
        return self


def _stream_handle(stream, cupy) -> int:
    """The ``CUstream`` a launch issues on, as an int. ``None`` resolves to
    the framework's current stream, read late (at launch, not at bind):
    cupy's capture idiom makes the capturing stream current only inside the
    capture region, so reading it earlier would record onto the wrong
    stream — and the default stream may not even be capturable."""
    if stream is None:
        return int(cupy.cuda.get_current_stream().ptr)
    ptr = getattr(stream, "ptr", None)
    return int(stream if ptr is None else ptr)
