# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A kernel split in stages: one stage, the finish of its cells, the next stage.

A kernel that reads a reduction over samples later is built as several
kernels of one bundle (``<name>``, ``<name>_2``, ...), and its sidecar lists
them under ``stages``: ``[{"kernel", "cells_in", "cells_out"}]``. Stage ``k``
reads the cells of the earlier stages as one-cell ``lookup`` planes named
like the cell. Every eagle door that runs such a kernel goes through this
module: :func:`run` for :meth:`Plan.run`, :class:`SplitBound` for
:meth:`Plan.bind` (and through it every door that binds: a Graph, the
reverse, the device bridges). A door that cannot sequence the stages calls
:func:`refuse`.

The caller sees the kernel's own planes (:func:`arg_spec`): the rows of every
cell and the cells a later stage reads are eagle's. A cell the sidecar marks
``internal`` (only a later stage reads it) is never handed back.

A kernel that reads ``hawk.sum/mean(x, over=...)`` in its body has a keyed
plane (sidecar ``keyed``): the first stage accumulates into it by key, a later
one reads it at its own lane's key. Eagle owns the plane and zeroes it, with
the companion of a compensated sum, for every run.

A stage the sidecar marks ``team`` (``stages[k].team``: a block of ``threads`` per
work item, see :func:`team_grid`) is launched with exactly that grid, inside the same
launch order and so inside a captured graph. The scratch planes it works on
(``team_planes`` of the first sidecar) are made once at bind, on the current stream,
and held by the :class:`SplitBound`; a plane the sidecar gives a ``fill`` is made
filled with it (the key log of an ordered scatter is -1 in every slot).
"""

from __future__ import annotations

import threading
import warnings
import weakref
from dataclasses import dataclass, replace
from math import prod

from . import _alloc
from . import _core
from . import _cell_rows
from . import _lane_bind
from . import _stream
from . import arrays
from .roles import OUTPUT_ROLES


@dataclass(frozen=True)
class Stage:
    """One stage: its plugin and the cell planes it reads and writes."""

    plugin: object
    cells_in: tuple
    cells_out: tuple
    team: dict | None = None


def link(by_name: dict, bundle) -> None:
    """Attach the stages of every split kernel of ``bundle`` to its first
    plugin (``by_name`` is ``{name: plugin}``). Idempotent; a kernel
    without ``stages`` is left as it is."""
    arts = {a.name: a.sidecar for a in bundle.artifacts}
    cells = {}
    for side in arts.values():
        cells.update(side.get("cells", {}))

    def planes(names):
        return _cell_rows._planes({c: cells[c] for c in names})

    for name, side in arts.items():
        stages = side.get("stages")
        if not stages or getattr(by_name.get(name), "_stages", None) is not None:
            continue
        built = []
        for entry in stages:
            plugin = by_name[entry["kernel"]]
            plugin._stage_cells = planes(entry["cells_out"])
            built.append(Stage(plugin, tuple(planes(entry["cells_in"])), tuple(plugin._stage_cells),
                               entry.get("team")))
        first = by_name[name]
        # hawk marks a cell only a later stage reads ("internal"): eagle's own
        mine = [c for c, spec in side.get("cells", cells).items() if spec.get("internal")]
        first._internal = frozenset(planes(mine))
        first._keyed = side.get("keyed") or {}
        first._team_planes = side.get("team_planes") or {}
        # the declared dtype of every plane a caller binds, whichever stage reads it
        if isinstance(getattr(first, "arg_dtypes", None), dict):
            for st in built[1:]:
                for nm, dt in (getattr(st.plugin, "arg_dtypes", None) or {}).items():
                    first.arg_dtypes.setdefault(nm, dt)
        first._stages = tuple(built)


def stages_of(plugin) -> tuple:
    """The :class:`Stage` tuple of a split kernel's first plugin; empty for
    any other plugin."""
    return getattr(plugin, "_stages", None) or ()


def _keyed(plugin) -> dict:
    """``{plane: spec}`` of the keyed planes eagle owns between the stages."""
    return getattr(plugin, "_keyed", None) or {}


def _keyed_names(plugin, stages) -> frozenset:
    """The keyed planes eagle owns between the stages, and the companions of
    their compensated sums."""
    keyed = _keyed(plugin)
    return frozenset(keyed) | {comp for st in stages
                               for comp, target in _cell_rows.companions_of(st.plugin).items()
                               if target in keyed}


def staged(plugin, planes):
    """``plugin`` itself, or, for a single kernel that stores a reduction over samples whose rows the caller did
    not bind (its cell planes are one element, or absent), a view of it as one stage, so a bare bind makes the
    rows and finishes the cell as for a split kernel. A caller who binds the rows themselves keeps the old path."""
    if stages_of(plugin):
        return plugin
    cells = _cell_rows.cells_of(plugin)
    if not cells:
        return plugin
    if any(_cell_rows.comp_name(nm) in planes for nm in cells):      # the caller binds the rows itself
        return plugin
    held = {nm: planes[nm] for nm in cells if nm in planes}
    if any(prod(getattr(a, "shape", (1,))) != 1 for a in held.values()):
        return plugin
    view = getattr(plugin, "_staged", None)
    if view is None:
        import copy

        view = copy.copy(plugin)
        plugin._stage_cells = dict(cells)
        view._stages = (Stage(plugin, (), tuple(cells)),)
        view._internal, view._keyed, view._team_planes = frozenset(), {}, {}
        plugin._staged = view
    return view


def _run_team(p, kw):
    """:meth:`Plan.run` of a kernel with team stages: the outputs not supplied are made here, the kernel is bound
    (:class:`SplitBound`, which owns the scratch and the stage grids) and launched once, and the outputs come
    back, the caller's own where supplied."""
    from ._plan_binding import _bind_plan
    from ._plan_planes import _declared_widths, _plane_dtype
    from .plan import _output_dtype

    plugin, spec = p.plugin, arg_spec(p.plugin)
    xp = _place(p)
    _kw, n, _dims = _lane_bind.bind_n(plugin, spec, kw, _declared_widths(plugin), door="run")
    if n is None:
        n = _lane_bind.n_of(spec, kw)
    dtype = _plane_dtype(plugin)
    skip, made = _plumbing(plugin, stages_of(plugin)), dict(kw)
    for st in stages_of(plugin):
        sp = st.plugin
        extents = _lane_bind.extents_of(sp, kw)
        for role, nm in sp.arg_spec:
            if role in OUTPUT_ROLES and nm not in skip and nm not in made and nm not in getattr(plugin, "_internal", ()):
                if st.team and st.team.get("keyed"):    # a block sum's output: its entries over the keys
                    spec_k = st.team["keyed"]
                    shape = (*(spec_k.get("payload") or (spec_k["width"],)), _lane_bind.key_count(plugin, spec_k, n))
                else:
                    shape = _lane_bind.output_shape(sp, nm, _declared_widths(sp).get(nm, 1), n, extents, kw)
                made[nm] = _alloc.zeros(xp, shape, dtype=_output_dtype(sp, nm, dtype))
    bound = _bind_plan(p, made)
    bound.launch()
    if xp != "cpu":
        _alloc.synchronize()
        bound.check_status()
    out = {nm: (kw[nm] if nm in kw else made.get(nm, bound.planes.get(nm)))
           for r, nm in spec if r in OUTPUT_ROLES}
    out = {nm: (v if nm in kw or v is None or v.device in ("cpu", "pinned") else v.get()) for nm, v in out.items()}
    return next(iter(out.values())) if len(out) == 1 else out


def _team_planes(plugin) -> dict:
    """``{plane: spec}`` of the team scratch planes eagle owns between the stages."""
    return getattr(plugin, "_team_planes", None) or {}


def _plumbing(plugin, stages) -> frozenset:
    """Every plane eagle owns between the stages: the rows of each cell, the
    cells read back as one-cell planes, the keyed planes and the team scratch."""
    out = set(_keyed_names(plugin, stages)) | set(_team_planes(plugin))
    for st in stages:
        out |= _cell_rows.row_planes(st.plugin) | set(st.cells_in)
    return frozenset(out)


def _place(p) -> str:
    """The device token where the planes of ``p`` live: the current CUDA device, or the host."""
    if p.structure.name == "device_kernel":
        from .launch import _current_device_ordinal

        return f"cuda:{_current_device_ordinal()}"
    return "cpu"


def _keyed_planes(plugin, spec, planes, xp, dtype) -> dict:
    """The zeroed keyed planes of a split kernel for a launch binding ``planes``
    (``spec`` the caller's planes), sized by the kept lane axes."""
    keyed = _keyed(plugin)
    if not keyed:
        return {}
    from ._plan_planes import _declared_widths

    kw, n, _dims = _lane_bind.bind_n(plugin, spec, planes, _declared_widths(plugin), door="bind")
    if n is None:
        n = _lane_bind.n_of(spec, kw)
    return {cell: _alloc.zeros(xp, _lane_bind.keyed_size(plugin, s, n), dtype=dtype)
            for cell, s in keyed.items()}


#: The binds whose last launch has a scatter status nobody has read yet (a device launch is asynchronous).
_PENDING: "weakref.WeakSet" = weakref.WeakSet()
_CHECKING = threading.local()


def run_pending(stream: int = -1) -> None:
    """Read the status of the pending launches on ``stream`` (every stream: -1) and raise the first error, by
    name. Eagle runs this where it already waits (the C++ hook of ``_core.set_sync_hook``: a stream or
    device synchronise, a device read-back; a pipeline's own synchronise), at the next launch of the same bind
    and when a bind is freed. It waits only on the event each bind recorded after its last stage, so a
    synchronise on one stream never waits for a scatter on another."""
    if not _PENDING or getattr(_CHECKING, "on", False):
        return
    stream = _stream.LEGACY if stream == 0 else stream
    _CHECKING.on = True
    try:
        for bound in list(_PENDING):
            if stream == -1 or bound._event._stream == stream:
                _PENDING.discard(bound)
                bound.check_status()
    finally:
        _CHECKING.on = False


_core.set_sync_hook(run_pending)


class _Collect(threading.local):
    """The list a capture in progress gathers the binds it launches into (a pipeline's, so a replay knows them)."""

    found = None
    names = None        # the kernel names a capture in progress launches (a pipeline's, for its error text)


COLLECT = _Collect()


def mark_pending(bound, stream: int) -> None:
    """Record that ``bound`` launched work on the raw ``stream`` whose scatter status is still to be read:
    an event recorded now, after its last stage (a no-op without a scatter status)."""
    if not bound._status:
        return
    if bound._event is None:
        bound._event = _stream.Event(-1 if bound._device is None else bound._device)
    bound._event.record(stream)
    _PENDING.add(bound)


def _free_bytes(device) -> int:
    """Free bytes of the CUDA device ``device`` (``None``: the current one)."""
    from . import _core

    return int(_core.device_mem_info(-1 if device is None else device)[1])


def _team_scratch(plugin, stages, n, planes, xp, dtype, device) -> dict:
    """The scratch planes the team stages work on (sidecar ``team_planes``), made once for ``n``
    lanes: ``per_lane`` elements for each lane, key or scatter cell (:func:`_lane_bind.team_units`),
    filled with the spec's ``fill`` (0 unless given). A device plane set larger than the free
    device memory is refused here, by name, with the byte sum and the stage that needs most."""
    specs = _team_planes(plugin)
    if not specs:
        return {}
    size = {nm: max(1, spec.get("per_lane", 1) * _lane_bind.team_units(plugin, spec, n, planes))
            for nm, spec in specs.items()}
    item = {nm: 4 if spec["dtype"] == "int32" else dtype.itemsize for nm, spec in specs.items()}
    nbytes = {nm: size[nm] * item[nm] for nm in specs}
    if xp != "cpu":
        total, free = sum(nbytes.values()), _free_bytes(device)
        if total > free:
            need = {st.plugin.name: sum(nbytes[nm] for _r, nm in st.plugin.arg_spec if nm in nbytes)
                    for st in stages if st.team}
            stage, most = max(need.items(), key=lambda kv: kv[1])
            raise ValueError(
                f"eagle.plan.bind: kernel {plugin.name!r} needs {total:,} bytes of team scratch "
                f"({', '.join(f'{nm} {b:,}' for nm, b in nbytes.items())}) at {n:,} lanes, but the device "
                f"has {free:,} bytes free; stage {stage!r} alone needs {most:,}")
    made = {nm: _alloc.full(xp, (size[nm],), spec.get("fill", 0),
                            arrays.int32 if spec["dtype"] == "int32" else dtype)
            for nm, spec in specs.items()}
    if xp != "cpu":
        _alloc.synchronize()            # the fills are ordered before any stream a launch uses
    return made


def _first_extent(plugin, n: int, names) -> dict:
    """``{label: extent}`` of the runtime first lane axis of ``plugin``, for a stage that binds no
    lane-sized plane to read it from (empty when it has none or the caller binds the label)."""
    axes = _lane_bind.declared_axes(plugin)
    if axes and axes[0][0] is not None and axes[0][1] is None and axes[0][0] not in names:
        return {axes[0][0]: n // prod(e for _a, e in axes[1:])}
    return {}


def arg_spec(plugin) -> tuple:
    """The planes a caller binds: ``plugin.arg_spec`` for one kernel; for a
    split one, every stage's own planes once (an output role wins over an
    input one), then each cell handed back, as ``accum_out`` of one element
    per column."""
    stages = stages_of(plugin)
    if not stages:
        return tuple(plugin.arg_spec)
    skip, role = _plumbing(plugin, stages), {}
    for st in stages:
        for r, nm in st.plugin.arg_spec:
            if nm not in skip and (nm not in role or r in OUTPUT_ROLES):
                role[nm] = r
    return tuple((r, nm) for nm, r in role.items()) + tuple(("accum_out", c) for c in cells(plugin))


def cells(plugin) -> tuple:
    """The cell planes a split kernel hands back (each one element), in
    stage order; empty for one kernel."""
    return tuple(c for st in stages_of(plugin) for c in st.cells_out
                 if c not in plugin._internal)


def kernels_of(kernel) -> tuple:
    """The stages of a traced hawk ``kernel`` as kernels of their own, in run
    order (``()`` when it is one kernel): for a door that chains one ordinary
    call per stage, the cells of a stage passed to the next as one-cell
    planes."""
    from hawk import Kernel
    from hawk.ir.split import stages

    found = stages(kernel.name, kernel.sinks)
    return () if found is None else tuple(
        Kernel(st.name, st.sinks, kernel.planes, kind=getattr(kernel, "kind", None))
        for st in found)


def single(bound, door: str):
    """``bound``, refused by name when it is a :class:`SplitBound`: for a door
    that drives one kernel entry itself (a step loop, a recorded run, a
    timing)."""
    if isinstance(bound, SplitBound):
        raise ValueError(
            f"{door}: {bound.plan.plugin.name!r} runs in {len(bound.stages)} stages; "
            f"{door} runs single-stage kernels in this release")
    return bound


def refuse(plugin, door: str) -> None:
    """Refuse a split kernel at a door that launches one stage at a time."""
    stages = stages_of(plugin)
    if stages:
        raise ValueError(
            f"{door}: kernel {plugin.name!r} is split in {len(stages)} stages (it reads a "
            "reduction over samples after computing it), and this door launches one kernel "
            "per step, so it cannot run the finish between the stages; run it with "
            "eagle.plan.auto(kernel).run(...) or .bind(...), which sequence them")


def run(p, kw: dict, run_plan):
    """:meth:`Plan.run` of a split kernel: ``run_plan(plan, **kw)`` per stage,
    the finished cells of a stage bound as the one-cell inputs of the next.
    Returns what a single kernel returns: the plane for one output, else a
    ``dict`` by name, without the internal cells."""
    if _team_planes(p.plugin):
        return _run_team(p, kw)
    stages = stages_of(p.plugin)
    from ._plan_planes import _plane_dtype

    carried = _keyed_planes(p.plugin, arg_spec(p.plugin), kw, _place(p),
                            _plane_dtype(p.plugin))
    out = {}
    later = {c for st in stages for c in st.cells_in}
    for st in stages:
        sp = st.plugin
        planes = {nm for _role, nm in sp.arg_spec}
        got = run_plan(replace(p, plugin=sp),
                       **{k: v for k, v in kw.items() if k in planes},
                       **{k: v for k, v in carried.items() if k in planes})
        names = tuple(nm for role, nm in sp.arg_spec
                      if role in OUTPUT_ROLES and nm not in _cell_rows.hidden(sp))
        got = {names[0]: got} if not isinstance(got, dict) else got
        out.update(got)
        carried.update({c: got[c] for c in st.cells_out if c in later})
    hide = p.plugin._internal | _keyed_names(p.plugin, stages)
    out = {nm: v for nm, v in out.items() if nm not in hide}
    return next(iter(out.values())) if len(out) == 1 else out


class SplitBound:
    """:meth:`Plan.bind` of a split kernel: one bound plan per stage, and
    between two stages the finish of the earlier one's cells, which the next
    reads in place. The rows and cells are allocated here, once; a launch
    allocates nothing.

    It stands wherever a :class:`~eagle.plan.BoundPlan` does (a Graph's
    launch node, the reverse): :meth:`launch` issues every stage and finish
    in order on one stream, each step legal inside a stream capture.
    The caller's planes are those of :func:`arg_spec`; a cell handed back may
    be bound to the caller's own one-element array, else eagle makes it."""

    __slots__ = ("plan", "_arg_spec", "_planes", "_bound", "_rows", "_dev", "_dtype", "_device",
                 "_copies", "_active_count", "_origins", "_stages", "_keyed", "_team", "_chunks", "_fills", "_status", "_event", "_capacity", "_count", "__weakref__")

    def __init__(self, p, planes: dict, bind_one, *, policy: str = "compensated", device=None):
        from ._plan_planes import device_scope, planes_device

        self._device = None
        if p.structure.name == "device_kernel":
            self._device = planes_device(arg_spec(p.plugin), planes, door="plan.bind", device=device)
        with device_scope(self._device):        # the rows and cells are allocated on the planes' GPU
            self._build(p, planes, bind_one, policy)

    def _build(self, p, planes, bind_one, policy):
        from ._plan_pack import _n_from_kw
        from ._plan_planes import _declared_widths, _plane_dtype

        stages = stages_of(p.plugin)
        self.plan, self._stages = p, stages
        self._arg_spec = spec = arg_spec(p.plugin)
        named = {nm for _r, nm in spec}
        axes = _lane_bind.declared_axes(p.plugin)
        labels = ({axes[0][0]} if axes and axes[0][0] is not None else set()) | _lane_bind.free_labels(p.plugin)
        stray = sorted(set(planes) - named - labels)
        if stray:
            raise ValueError(
                f"eagle.plan.bind: {stray} are not planes of kernel {p.plugin.name!r} "
                f"(bound: {sorted(named)}); its rows and the cells between its stages "
                "are eagle's")
        xp = self._dev = _place(p)
        self._dtype = _plane_dtype(p.plugin)
        skip = _plumbing(p.plugin, stages)
        team = _team_planes(p.plugin)
        n = None
        if team:                        # a Grid's first extent may come from its label or a lane-shaped plane
            _kw, n, _dims = _lane_bind.bind_n(p.plugin, spec, planes, _declared_widths(p.plugin), door="bind")
        if n is None:
            n = _n_from_kw([(r, nm) for r, nm in spec if nm not in skip], planes)
        cell, self._bound, self._rows = {}, [], []
        keyed = _keyed_planes(p.plugin, spec, planes, xp, self._dtype)
        self._team = _team_scratch(p.plugin, stages, n, planes, xp, self._dtype, self._device)
        self._chunks = {}
        # planes the sidecar fills (the key log) are refilled at the start of every launch
        self._fills = tuple((self._team[nm], spec["fill"]) for nm, spec in team.items() if "fill" in spec)
        self._capacity = self._count = None     # a prefix kernel's capacity= and count plane (see set_count)
        self._event = None          # recorded after the last stage of a launch with a pending status
        self._status = []           # (kernel, team record, status plane, target cells) of each ordered scatter
        for st in stages:
            ops = getattr(st.plugin, "_stage_cells", {})
            rows = (_cell_rows.CellRows(ops, n, xp, self._dtype, policy,
                                        outs={c: planes[c] for c in ops if c in planes})
                    if ops else None)
            if rows is not None:
                cell.update(rows.cell)
            own = {nm: planes[nm] for _r, nm in st.plugin.arg_spec
                   if nm in planes and nm not in skip}
            own.update(rows.rows if rows is not None else {})
            own.update({c: cell[c] for c in st.cells_in})
            names = {nm for _r, nm in st.plugin.arg_spec}
            own.update({c: a for c, a in keyed.items() if c in names})
            own.update({c: a for c, a in self._team.items() if c in names})
            if team and st.team is None:
                own.update(_first_extent(st.plugin, n, names))
            bound = bind_one(replace(p, plugin=st.plugin), own)
            if st.team is not None:
                grid = _lane_bind.team_grid(p.plugin, st.team, n, planes)
                bound.team_stage(st.team["threads"], grid, n)
                if st.team.get("grid") == "chunks":
                    self._chunks[st.team["of"]] = grid
                if st.team.get("status"):
                    self._status.append((p.plugin.name, st.team, self._team[st.team["status"]],
                                         _lane_bind._cells_of(planes[st.team["of"]])))
            self._bound.append(bound)
            self._rows.append(rows)
        # what a launch zeroes: each keyed plane and the companion its stage owns
        self._keyed = tuple(keyed.values()) + tuple(
            b._planes[comp] for b in self._bound
            for target, comp in b._owned.items() if target in keyed)
        self._planes = {**planes, **{c: cell[c] for _r, c in spec if c in cell}}
        self._copies, self._active_count, self._origins = {}, None, ()

    @property
    def n(self) -> int:
        return self._bound[0].n

    @property
    def chunks(self) -> dict:
        """``{target plane: chunks}`` the ordered scatters of this bind add in (``B`` chosen at bind from
        the lane and target counts: the same counts give the same bits on every launch)."""
        return dict(self._chunks)

    @property
    def _entry(self):
        """A door that launches one kernel entry itself reaches here only
        without its own :func:`single` check: refused by name all the same."""
        single(self, "a door that launches one kernel entry")

    @property
    def names(self) -> tuple:
        return tuple(nm for _r, nm in self._arg_spec)

    @property
    def planes(self):
        from types import MappingProxyType

        return MappingProxyType(self._planes)

    @property
    def stages(self) -> tuple:
        """The bound plan of each stage, in run order."""
        return tuple(self._bound)

    def __repr__(self) -> str:
        return (f"SplitBound({self.plan.plugin.name!r}, stages={len(self._bound)}, "
                f"n={self.n}, names={list(self.names)})")

    def set_count(self, m: int, stream=None) -> None:
        """:meth:`BoundPlan.set_count` for a split prefix kernel: the one count plane every stage reads."""
        from ._plan_binding import _set_count

        _set_count(self, m, stream)

    def check_status(self) -> None:
        """Read the status plane of each ordered scatter and raise, by name, when a lane's key was outside
        its target cells (hawk's ``check_scatter_status``). The check is explicit here; eagle also runs it,
        pending since the last launch or replay, wherever it next waits (see :func:`run_pending`). It waits
        for the device first."""
        if self._event is not None and self in _PENDING:
            self._event.synchronize()                       # that stream only
        _PENDING.discard(self)
        for kernel, team, plane, cells in self._status:
            from hawk._lane_bind import check_scatter_status      # the plugin is hawk's: so is its check

            check_scatter_status(kernel, team, plane.get().tolist(), cells)

    def launch(self, stream=None, *, grid=None, block=None) -> None:
        """Every stage, each followed by the finish of its cells, on
        ``stream`` (see :meth:`BoundPlan.launch`). A host launch checks a scatter's status at once; a device
        launch records it as pending (no wait here) and eagle reads it where it next waits."""
        if self._dev == "cpu":
            for plane in self._keyed:
                plane.fill(0)
            for plane, value in self._fills:
                plane.fill(value)
            for bound, rows in zip(self._bound, self._rows):
                bound.launch(stream, grid=grid, block=block)
                if rows is not None:
                    rows.finish()
            self.check_status()
            return None
        from ._plan_binding import _stream_handle
        from ._plan_planes import device_scope

        handle = _stream_handle(stream, self._device)
        capturing = _core.stream_is_capturing(handle)
        if self in _PENDING and not capturing:      # the previous launch's status, before this one
            _PENDING.discard(self)
            self.check_status()
        for plane in self._keyed:       # a memset node when captured
            _alloc.zero(self._dev, plane, handle)
        for plane, value in self._fills:
            plane.fill(value, handle)
        for bound, rows in zip(self._bound, self._rows):
            bound.launch(handle, grid=grid, block=block)
            if rows is not None:
                with device_scope(self._device), _stream.scope(handle):
                    rows.finish()
        if capturing:
            if COLLECT.found is not None and self._status:
                COLLECT.found.append(self)
        else:
            mark_pending(self, handle)
        return None

    def __del__(self):
        if getattr(self, "_status", None) and self in _PENDING:
            _PENDING.discard(self)
            try:
                self.check_status()
            except Exception as exc:        # freed with an unread error: it can only be said, not raised
                warnings.warn(str(exc), RuntimeWarning, stacklevel=2)

    def clear(self, *names) -> None:
        """:meth:`BoundPlan.clear` across every stage that binds a name."""
        valid = tuple(nm for nm in self.names if nm in self._planes
                      and any(nm in b.names for b in self._bound)
                      and nm not in cells(self.plan.plugin))
        stray = [nm for nm in names if nm not in valid]
        if stray:
            raise ValueError(f"eagle.plan.clear: {stray} are not bound planes "
                             f"(valid: {list(valid)})")
        for bound in self._bound:
            mine = [nm for nm in names if nm in bound.names]
            if mine:
                bound.clear(*mine)

    def rebind(self, /, **changed):
        """Re-pack the named planes in every stage that binds them; a cell
        handed back is fixed at bind."""
        stray = sorted(set(changed) - ({nm for _r, nm in self._arg_spec}
                                     - set(cells(self.plan.plugin))))
        if stray:
            raise ValueError(f"eagle.plan.rebind: {stray} are not bound names "
                             f"(bound: {list(self.names)})")
        for bound in self._bound:
            mine = {k: v for k, v in changed.items() if k in bound.names}
            if mine:
                bound.rebind(**mine)
        self._planes.update(changed)
        return self
