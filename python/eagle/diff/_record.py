# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The forward half of a reverse pass: what a ``record=True`` graph keeps per
launch, and the :class:`Record` naming one launch.

A recorded graph holds every parameter (a kernel ``uniform``) in a one-cell
plane of its own instead of a by-value immediate, so :meth:`eagle.Graph.update`
changes it with a copy into the cell: neither the forward nor its reverse is
rebuilt or recaptured.

A reverse needs the values every node READ. Most are still in the forward's
own planes when the reverse runs; a plane that a later node (or the node
itself, in place) overwrites is copied once, just before that overwrite. The
graph owns those copies and one live record slot: a :class:`Record` points at
them, stamped with the launch's GENERATION, so a reverse can always tell
whether the forward ran again underneath it and refuse rather than return a
wrong gradient.

Thread-safety: the slot (generation, live record, readers) changes only
under the recorder's lock. Many reverses may read one Record at once; a
launch while one is reading raises :class:`eagle.Busy`.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from .. import _core, _counters
from .._active_set import _address, _overlaps, _reordering_snapshot
from .._busy import Busy
from ..roles import STATE_ROLES

__all__ = ["Record"]


def xp_of(arr):
    """numpy or cupy, whichever owns ``arr``."""
    if type(arr).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    import numpy

    return numpy


class Planes:
    """Graph-unique keys for the arrays a graph's nodes touch: one key per
    array, however many nodes or names use it. ``reads[i]`` / ``writes[i]``
    list ``(local name, key)`` for leaf ``i``; a plane the node only writes
    (an output role) is not among its reads, since its derivative recomputes
    from inputs. ``names[key]`` lists every ``(leaf, local name)`` it goes by."""

    __slots__ = ("array", "names", "reads", "writes")

    def __init__(self, accesses):
        self.array, self.names, self.reads, self.writes = {}, {}, [], []
        where = {}
        for i, a in enumerate(accesses):
            key = lambda name, arr: self._key(where, i, name, arr)
            self.reads.append([(n, key(n, arr)) for n, arr, role in a.reads
                               if role not in STATE_ROLES and role != "guard"])
            self.writes.append([(n, key(n, arr)) for n, arr, _r in a.writes])
            if a.node.kind in ("skippable", "repeat_while"):
                # skipped, a conditional leaves the plane as the earlier writer made it
                earlier = {k for w in self.writes[:i] for _n, k in w}
                have = {k for _n, k in self.reads[i]}
                self.reads[i] += [(n, k) for n, k in self.writes[i] if k in earlier and k not in have]
        spans = [(k, where_k) for where_k, k in where.items()]
        for x in range(len(spans)):
            for y in range(x + 1, len(spans)):
                if _overlaps(spans[x][1], spans[y][1]):
                    a, b = self.names[spans[x][0]][0][1], self.names[spans[y][0]][0][1]
                    raise ValueError(
                        f"eagle.Graph.build: record=True: planes {a!r} and {b!r} are "
                        "different views of overlapping memory; a recorded graph "
                        "needs each plane to be one whole array (bind the same "
                        "array, or separate arrays)")

    def _key(self, where, leaf, name, arr):
        addr = _address(arr)
        key = where.get(addr)
        if key is None:
            key = where[addr] = f"p{len(where)}"
            self.array[key] = arr
            self.names[key] = []
        if (leaf, name) not in self.names[key]:
            self.names[key].append((leaf, name))
        return key

    def rows(self):
        """``(leaf, read keys, written keys)`` per leaf."""
        return [(i, [k for _, k in r], [k for _, k in w])
                for i, (r, w) in enumerate(zip(self.reads, self.writes))]


def _pairing_set(planes):
    """The reordering ActiveSet owning the graph's per-sample planes, or
    ``None``. Its ``perm`` pairs each slot with its sample in the reverse."""
    found = set()
    for key, arr in planes.array.items():
        where = _address(arr)
        for aset in _reordering_snapshot():
            if any(r and _overlaps(where, r) for r in aset._owned_ranges()):
                found.add(aset)
            elif any(_address(p) == where for p in aset._indirect):
                raise ValueError(
                    f"eagle.Graph.build: record=True: plane {planes.names[key][0][1]!r} "
                    "is declared indirect on an ActiveSet; a recorded graph pairs "
                    "samples through owned planes only -- own() it instead")
    if len(found) > 1:
        raise ValueError("eagle.Graph.build: record=True: the planes belong to two "
                         "reordering ActiveSets; a recorded graph pairs its samples "
                         "through one")
    if not found:
        return None
    (aset,) = found
    for key, arr in planes.array.items():
        if arr.shape and arr.shape[-1] == aset.n and not any(
                r and _overlaps(_address(arr), r) for r in aset._owned_ranges()):
            raise ValueError(
                f"eagle.Graph.build: record=True: per-sample plane "
                f"{planes.names[key][0][1]!r} is not owned by the ActiveSet that "
                "reorders the other planes, so its samples would not stay paired; "
                "own() it")
    return aset


class Tape:
    """What a reverse reads: the arrays by key (parameter cells included), the
    copies, and the slot->sample map (``None`` when nothing reorders)."""

    __slots__ = ("source", "arrays", "copies", "perm")
    steps = window = None

    def __init__(self, source, arrays, copies, perm):
        self.source = source
        self.arrays = arrays
        self.copies = copies
        self.perm = perm

    @property
    def nbytes(self) -> int:
        return sum(int(c.nbytes) for c in self.copies)

    def copy(self):
        dup = lambda a: xp_of(a).array(a, copy=True)
        return Tape(None, {k: dup(a) for k, a in self.arrays.items()},
                    [dup(c) for c in self.copies],
                    None if self.perm is None else dup(self.perm))


def _parameter_variant(bound, label, uniforms):
    """The plugin of ``bound``'s kernel with its uniforms read from one-cell
    planes (``hawk.diff.param_planes``), built once beside it and kept."""
    plugin = bound.plan.plugin
    found = getattr(plugin, "_params", None)
    source = getattr(plugin, "_hawk_source", None)
    if found is not None:
        return found
    if source is None:
        raise ValueError(
            f"eagle.Graph.build: record=True: node {label!r} takes {uniforms} by value, "
            "and a recorded graph holds its parameters in one-cell planes (so a "
            "parameter can change without a rebuild); make the plan with "
            "eagle.plan.auto(kernel), or bind a kernel made with "
            "hawk.diff.param_planes(kernel)")
    kernel, _unit, cache_dir = source
    from hawk.diff import param_planes

    from .._plan_deploy import _hawk_plugins

    targets = ("cuda",) if bound._cupy is not None else ("host",)
    (plugin._params,) = _hawk_plugins([param_planes(kernel)], targets, cache_dir,
                                      plugin.scalar_type)
    return plugin._params


def _with_parameter_planes(node, cells):
    """``node`` with each launch re-bound through its parameter-plane variant,
    every uniform seeded into a new one-cell plane recorded in ``cells``
    (``{(node label, name): cell}``)."""
    from ..graph import Node
    from ..plan import plan

    if node.kind == "group":
        return Node("group", tuple(tuple(_with_parameter_planes(n, cells) for n in branch)
                                   for branch in node.payload), node.name)
    if node.kind in ("skippable", "repeat_while"):
        cond = node.payload     # owned by this graph: its lowering reads the new child
        cond.child = cond.late.child = _with_parameter_planes(cond.child, cells)
        return node
    if node.kind != "launch":
        return node
    bound, label = node.payload, node.label
    uniforms = [name for role, name in bound._arg_spec if role == "uniform"]
    if not uniforms:
        return node
    variant = _parameter_variant(bound, label, uniforms)
    xp = _xp_of_bound(bound)
    planes = dict(bound._planes)
    for name in uniforms:
        planes[name] = cells[(label, name)] = xp.full(
            1, bound._planes[name], dtype=variant.arg_dtypes[name])
    rebound = plan(variant, structure=bound.plan.structure,
                   npartitions=bound.plan.npartitions).bind(**planes)
    return Node("launch", rebound, label)


def _xp_of_bound(bound):
    import numpy

    return bound._cupy or numpy


def _copier(dst, src):
    xp = xp_of(dst)

    def copy():
        xp.copyto(dst, src)
    return copy


class Recorder:
    """The graph's single live record slot (see the module docstring). Made
    before the graph derives its edges: it re-binds the launches through
    parameter planes; :meth:`attach` then plans the copies."""

    def __init__(self, graph):
        self._graph = graph
        self.cells = {}
        graph._nodes = [_with_parameter_planes(n, self.cells) for n in graph._nodes]
        self._lock = threading.Lock()
        self._generation = 0
        self._live = None
        self._readers = 0

    def attach(self, accesses):
        from .._graph_view import _stage_of_leaves
        from ..graph import get_node_kind

        graph = self._graph
        self.planes = planes = Planes(accesses)
        stage_of = _stage_of_leaves(graph)
        self.copy_of = {}
        copies, self._before = [], {}
        for leaf, key in _core.diff_snapshots(planes.rows()):
            src = planes.array[key]
            buf = xp_of(src).empty_like(src)
            self.copy_of[(leaf, key)] = len(copies)
            copies.append(buf)
            self._before.setdefault(stage_of[leaf], []).append(_copier(buf, src))
        self.recorded = {}
        for leaf, a in enumerate(accesses):
            saved = get_node_kind(a.node.kind).record_planes
            for name, src in (saved(a.node) if saved else ()):
                buf = xp_of(src).empty_like(src)
                self.recorded[(leaf, name)] = len(copies)
                copies.append(buf)
                self._before.setdefault(stage_of[leaf], []).append(_copier(buf, src))
        aset = _pairing_set(planes)
        self._aset = aset
        perm = None if aset is None else xp_of(aset.perm).empty_like(aset.perm)
        self._live_tape = (dict(planes.array), copies, perm)

    def before(self, stage):
        """The copies to run before top-level stage ``stage``."""
        return self._before.get(stage, ())

    def update(self, values):
        """Write ``values`` (``{name or "node.name": value}``) into the
        parameter cells; refused while the last Record still needs them."""
        targets = []
        for key, value in values.items():
            hit = [c for (label, name), c in self.cells.items()
                   if key in (name, f"{label}.{name}")]
            if not hit:
                known = sorted({n for _l, n in self.cells} | {f"{l}.{n}" for l, n in self.cells})
                raise ValueError(f"eagle.Graph.update: {key!r} is not a parameter of this "
                                 f"graph (parameters: {known})")
            targets += [(c, value) for c in hit]
        with self._lock:
            if self._readers or (self._live is not None and not self._live._released):
                raise RuntimeError(
                    "eagle.Graph.update: the Record of the last launch is still live and "
                    "its reverse reads these parameters; reverse it, or "
                    "record.keep() first")
            for cell, value in targets:
                cell[...] = value

    def begin(self, n, method):
        if n != 1:
            raise ValueError(
                f"eagle.Graph.{method}: a record=True graph records one run per "
                f"launch (n=1), got n={n}; call launch() n times, reversing (or "
                "keeping) each Record")
        with self._lock:
            if self._readers:
                raise Busy(f"eagle.Graph.{method}: a reverse is reading this graph's "
                           "Record; wait for it, or record.keep() a copy first")
            if self._live is not None and not self._live._released:
                raise RuntimeError(
                    f"eagle.Graph.{method}: the Record of the previous launch is "
                    "still live; reverse it, or record.keep() to keep a private copy")
            self._generation += 1      # older Records are stale from here on

    def finish(self):
        arrays, copies, perm = self._live_tape
        if perm is not None:
            perm[...] = self._aset.perm
        with self._lock:
            tape = Tape(self, arrays, copies, perm)
            self._live = Record(self._graph, self._generation, tape, self)
        _counters.bump("records")
        return self._live

    def hold(self, record, door):
        with self._lock:
            if record._generation != self._generation:
                raise RuntimeError(
                    f"{door}: the forward ran again (generation {record._generation} "
                    f"-> {self._generation}) since this Record was made; reverse the "
                    "newest graph.record, or keep() a Record to keep it")
            self._readers += 1

    def release(self, record, door):
        with self._lock:
            self._readers -= 1
            if record._generation != self._generation:
                raise RuntimeError(
                    f"{door}: the forward ran again (generation {record._generation} "
                    f"-> {self._generation}) while this reverse was running")
            record._released = True


class Record:
    """One recorded forward, as its reverse needs it: get it from ``.record``
    after the forward ran (``graph.launch()`` of a ``record=True``
    :class:`eagle.Graph`, or a ``record=True`` :func:`eagle.simulate`,
    :class:`eagle.Simulation` or :func:`eagle.until_done` run) and pass it to
    :func:`eagle.diff.reverse`.

    A Record points at buffers the next recorded forward overwrites. So the
    forward refuses to record again while the Record is live (neither
    reversed nor kept), and a reverse refuses a Record whose forward is no
    longer the newest. :meth:`keep` makes a private copy that stays valid
    however often the forward runs again. Immutable: a reverse never writes
    it, so several reverses may read one Record at once.

    Example::

        sim = eagle.simulation(model, x=x0, v=v0, max_steps=1000, record=True)
        sim.run()
        rec = sim.record
        rec.steps          # how many steps each sample took
        rec.nbytes         # the memory the record holds
        grads = eagle.diff.reverse(rec, {"x": dJ_dx})
    """

    __slots__ = ("_graph", "_generation", "_tape", "_recorder", "_released",
                 "_readers", "_lock", "__weakref__")

    def __init__(self, graph, generation, tape, recorder):
        self._graph = graph
        self._generation = generation
        self._tape = tape
        self._recorder = recorder
        self._released = recorder is None
        self._readers = 0
        self._lock = threading.Lock()

    @property
    def forward(self):
        """What ran the recorded forward: the :class:`eagle.Graph`,
        :class:`eagle.Simulation` or :class:`eagle.Runner`."""
        return self._graph

    @property
    def graph(self):
        """The same object as :attr:`forward`."""
        return self._graph

    @property
    def generation(self) -> int:
        """Which recorded run of the forward this is (1 for the first)."""
        return self._generation

    @property
    def held(self) -> bool:
        """Whether a reverse is reading this Record right now."""
        return self._readers > 0

    @property
    def steps(self):
        """The steps each sample took, in sample order (a host int64 array),
        for a recorded run; ``None`` for a graph."""
        steps = self._tape.steps
        return None if steps is None else steps.copy()

    @property
    def window(self):
        """The reverse window W of a recorded run (the state is kept every W
        steps); ``None`` for a graph."""
        return self._tape.window

    @property
    def nbytes(self) -> int:
        """The bytes this Record holds of its own: the saved checkpoints of a
        run, or the saved copies of a graph."""
        return self._tape.nbytes

    def keep(self) -> Record:
        """A private copy of everything a reverse reads, which stays valid
        however often the forward runs again; it costs that memory and one
        copy. The forward may record again right away.

        Example: ``kept = sim.record.keep(); sim.run(); eagle.diff.reverse(kept, seeds)``."""
        if self._recorder is not None:
            self._recorder.hold(self, "eagle.diff.Record.keep")
        try:
            tape = self._tape.copy()
            copy = Record(self._graph, self._generation, tape, None)
            if hasattr(tape, "source"):
                tape.source = copy
        finally:
            if self._recorder is not None:
                self._recorder.release(self, "eagle.diff.Record.keep")
        return copy

    @contextmanager
    def _hold(self, door):
        """Held for one reverse: generation checked at the start AND the end."""
        if self._recorder is not None:
            self._recorder.hold(self, door)
        with self._lock:
            self._readers += 1
        try:
            yield self._tape
        finally:
            with self._lock:
                self._readers -= 1
            if self._recorder is not None:
                self._recorder.release(self, door)

    def __repr__(self) -> str:
        kind = "live" if self._recorder is not None else "kept"
        return f"<eagle.diff.Record generation {self._generation}, {kind}>"
