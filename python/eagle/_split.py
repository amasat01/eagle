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
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from . import _alloc
from . import _cell_rows
from .roles import OUTPUT_ROLES


@dataclass(frozen=True)
class Stage:
    """One stage: its plugin and the cell planes it reads and writes."""

    plugin: object
    cells_in: tuple
    cells_out: tuple


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
            built.append(Stage(plugin, tuple(planes(entry["cells_in"])), tuple(plugin._stage_cells)))
        first = by_name[name]
        # hawk marks a cell only a later stage reads ("internal"): eagle's own
        mine = [c for c, spec in side.get("cells", cells).items() if spec.get("internal")]
        first._internal = frozenset(planes(mine))
        first._keyed = side.get("keyed") or {}
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


def _plumbing(plugin, stages) -> frozenset:
    """Every plane eagle owns between the stages: the rows of each cell, the
    cells read back as one-cell planes, and the keyed planes."""
    out = set(_keyed_names(plugin, stages))
    for st in stages:
        out |= _cell_rows.row_planes(st.plugin) | set(st.cells_in)
    return frozenset(out)


def _array_module(p):
    """The array module the planes of ``p`` live in."""
    if p.structure.name == "device_kernel":
        import cupy

        return cupy
    return __import__("numpy")


def _keyed_planes(plugin, spec, planes, xp, dtype) -> dict:
    """The zeroed keyed planes of a split kernel for a launch binding ``planes``
    (``spec`` the caller's planes), sized by the kept lane axes."""
    keyed = _keyed(plugin)
    if not keyed:
        return {}
    from . import _lane_bind
    from ._plan_planes import _declared_widths

    kw, n, _dims = _lane_bind.bind_n(plugin, spec, planes, _declared_widths(plugin), door="bind")
    if n is None:
        n = _lane_bind.n_of(spec, kw)
    return {cell: _alloc.zeros(xp, _lane_bind.keyed_size(plugin, s, n), dtype=dtype)
            for cell, s in keyed.items()}


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
    stages = stages_of(p.plugin)
    from ._plan_planes import _plane_dtype

    carried = _keyed_planes(p.plugin, arg_spec(p.plugin), kw, _array_module(p),
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

    __slots__ = ("plan", "_arg_spec", "_planes", "_bound", "_rows", "_cupy", "_dtype", "_device",
                 "_copies", "_active_count", "_origins", "_stages", "_keyed")

    def __init__(self, p, planes: dict, bind_one, *, policy: str = "compensated", device=None):
        from ._plan_planes import device_scope, planes_device

        self._device = None
        if p.structure.name == "device_kernel":
            self._device = planes_device(arg_spec(p.plugin), planes, door="plan.bind", device=device)
        with device_scope(self._device):        # the rows and cells are allocated on the planes' GPU
            self._build(p, planes, bind_one, policy)

    def _build(self, p, planes, bind_one, policy):
        from ._plan_pack import _n_from_kw
        from ._plan_planes import _plane_dtype

        stages = stages_of(p.plugin)
        self.plan, self._stages = p, stages
        self._arg_spec = spec = arg_spec(p.plugin)
        named = {nm for _r, nm in spec}
        stray = sorted(set(planes) - named)
        if stray:
            raise ValueError(
                f"eagle.plan.bind: {stray} are not planes of kernel {p.plugin.name!r} "
                f"(bound: {sorted(named)}); its rows and the cells between its stages "
                "are eagle's")
        self._cupy = None
        xp = __import__("numpy")
        if p.structure.name == "device_kernel":
            import cupy

            self._cupy = xp = cupy
        self._dtype = _plane_dtype(p.plugin)
        skip = _plumbing(p.plugin, stages)
        n = _n_from_kw([(r, nm) for r, nm in spec if nm not in skip], planes)
        cell, self._bound, self._rows = {}, [], []
        keyed = _keyed_planes(p.plugin, spec, planes, xp, self._dtype)
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
            self._bound.append(bind_one(replace(p, plugin=st.plugin), own))
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

    def launch(self, stream=None, *, grid=None, block=None) -> None:
        """Every stage, each followed by the finish of its cells, on
        ``stream`` (see :meth:`BoundPlan.launch`)."""
        if self._cupy is None:
            for plane in self._keyed:
                plane.fill(0)
            for bound, rows in zip(self._bound, self._rows):
                bound.launch(stream, grid=grid, block=block)
                if rows is not None:
                    rows.finish()
            return None
        from ._plan_binding import _stream_handle
        from .interop import _wrap_external_stream

        from ._plan_planes import device_scope

        handle = _stream_handle(stream, self._cupy, self._device)
        for plane in self._keyed:       # a memset node when captured
            _alloc.zero(self._cupy, plane, handle)
        for bound, rows in zip(self._bound, self._rows):
            bound.launch(handle, grid=grid, block=block)
            if rows is not None:
                with device_scope(self._device), _wrap_external_stream(handle):
                    rows.finish()
        return None

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
