# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The reverse pass of a recorded :class:`eagle.Graph`.

The plan (which derivative step runs when, which adjoints are summed, which
copies are read) is the C++ transposition of the graph (``eagle/diff/
Transpose.h``). This module chooses each node's derivative kernel, binds the
planes, and runs the result as an ordinary graph of its own: captured once on
the GPU, a sequence of host launches on the CPU.

Thread-safety: a :class:`Reverse` owns its adjoint planes, so distinct
Reverse objects run concurrently, also on one Record; one Reverse shared by
two threads raises :class:`eagle.Busy` in the second.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Mapping
from contextlib import contextmanager
from typing import NamedTuple


from .. import _alloc
from .. import _core, _counters
from .._busy import BusyGuard
from .._conditional import skippable
from .._graph_kinds import OWN_CODE_KINDS
from ..graph import _DERIVED_DOOR, Graph, Node, get_node_kind, opaque
from ..roles import OUTPUT_ROLES
from ..sidecar import derived_tag, refuse_prior_read
from ._record import Record, xp_of

__all__ = ["Gradients", "Report", "Reverse", "broadcast_adjoint", "reverse"]

_FLOATS = ("float32", "float64")
_CONDITIONALS = ("skippable", "repeat_while")


def _is_float(arr) -> bool:
    return str(getattr(arr, "dtype", "")) in _FLOATS


def broadcast_adjoint(cotangent, n: int, *, out=None):
    """The adjoint of a sum over ``n`` samples: every sample receives the
    cotangent of the sum. ``cotangent`` is one value (a one-element numpy or
    cupy array); the result is ``n`` values on the same backend, written into
    ``out`` when given. Safe inside a captured graph."""
    xp = xp_of(cotangent)
    cot = cotangent.reshape(-1)
    if cot.size != 1:
        raise ValueError("eagle.diff.broadcast_adjoint: the cotangent of a sum is one "
                         f"value, got shape {tuple(cotangent.shape)}")
    if out is None:
        out = _alloc.empty(xp, n, dtype=cot.dtype)
    out[...] = cot
    return out


class Report(NamedTuple):
    """How a :class:`Gradients` was made: the ``generation`` of the recorded
    launch, per returned plane whether it is ``bitwise`` reproducible
    (``False`` only for a plane fed by a cross-sample scatter-add, whose
    summing order varies run to run), and in how many passes (``chunks``)
    the reverse ran the batch to stay within ``memory_budget``."""

    generation: int
    bitwise: dict
    chunks: int = 1


class Gradients(Mapping):
    """The gradients of one reverse pass, read-only, keyed by plane name:
    ``grads["x"]`` is one row per sample, in sample order, for a per-sample
    plane, and a 0-d array (the sum over samples, bit-reproducible) for a
    shared parameter. :attr:`report` says how it was made."""

    __slots__ = ("_values", "report", "cotangents")

    def __init__(self, values: dict, report: Report, cotangents=None):
        self._values = dict(values)
        self.report = report
        self.cotangents = cotangents or {}

    def __getitem__(self, name):
        return self._values[name]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def __repr__(self) -> str:
        return f"<eagle.diff.Gradients {sorted(self._values)}>"


class ReverseCtx:
    """What a node kind's ``transpose(node, incoming, ctx)`` hook receives.

    ``incoming`` maps each plane the node writes to its adjoint; the hook
    returns the adjoint of each plane it reads, adding the steps that compute
    them with :meth:`add`. ``backend`` is ``"cpu"`` or ``"gpu"``."""

    __slots__ = ("backend", "label", "vjp", "_primal", "_steps", "inexact", "_gate",
                 "_recorded", "_written", "reads", "comp")

    def __init__(self, backend, label, vjp, primal, recorded=None, written=None, reads=()):
        self.backend = backend
        self.label = label
        self.vjp = vjp
        self._primal = primal
        self._steps = []
        self.inexact = set()
        self._gate = None
        self._recorded = recorded
        self._written = written
        self.reads = frozenset(reads)
        #: ``{id(rows): companion rows}`` of the adjoints that arrive as rows
        #: of per-32-sample partial sums (finished by the reverse)
        self.comp = {}

    def primal(self, name):
        """The value the node read for plane ``name`` in the recorded launch."""
        return self._primal(name)

    def recorded(self, name):
        """The value the forward saved for ``name`` of this node's record
        planes (a conditional's guard cell), as the launch saw it."""
        return self._recorded(name)

    def written(self, name):
        """The final value of plane ``name`` this node wrote in the recorded
        launch (a ``repeat_while``'s iteration cell)."""
        return self._written(name)

    @staticmethod
    def empty_like(arr):
        return xp_of(arr).empty_like(arr)

    @contextmanager
    def gated(self, guard):
        """Steps added inside run only while ``guard`` (an
        :class:`eagle.SkipGuard` over arrays the reverse owns) holds."""
        before, self._gate = self._gate, guard
        try:
            yield
        finally:
            self._gate = before

    def add(self, step, *, reads=(), writes=()):
        """Run ``step`` in the reverse: a bound derivative plan, a graph
        :class:`~eagle.graph.Node` of a registered kind, or a zero-arg
        callable touching the declared arrays."""
        if self._gate is not None:
            if hasattr(step, "_arg_spec"):
                node, kind = Node("launch", step), get_node_kind("launch")
                reads = [a for _n, a, _r in kind.reads(node)]
                writes = [a for _n, a, _r in kind.writes(node)]
                step = step.launch
            if not isinstance(step, Node):
                step = opaque(step, reads=reads, writes=writes)
            self._steps.append(skippable(step, self._gate))
            return
        keep = hasattr(step, "_arg_spec") or isinstance(step, Node)
        self._steps.append(step if keep else opaque(step, reads=reads, writes=writes))

    def not_bitwise(self, name):
        """Mark the adjoint of ``name`` as a scatter-add (not bit-reproducible)."""
        self.inexact.add(name)


# --- The derivative kernel of a plan launch --- #
def _vjp_from_auto(bound):
    """The vjp of a plan :func:`eagle.plan.auto` built from a hawk kernel: built
    once beside it (its own bundle, naming the primal's unit) and kept on the
    plugin. ``None`` for a plan built any other way."""
    plugin = bound.plan.plugin
    found = getattr(plugin, "_vjp", None)
    source = getattr(plugin, "_hawk_source", None)
    if found is not None or source is None:
        return found
    kernel, unit, cache_dir = source
    refuse_prior_read(getattr(kernel.walk, "prior_reads", ()) or (), door="eagle.diff.Reverse")
    from hawk import Kernel
    from hawk.diff import vjp

    from .._plan_deploy import _hawk_plugins

    derived = Kernel(f"{kernel.name}_vjp", vjp(kernel, primal_unit=unit))
    targets = ("cuda",) if bound._cupy is not None else ("host",)
    (plugin._vjp,) = _hawk_plugins([derived], targets, cache_dir, plugin.scalar_type)
    return plugin._vjp


def _vjp_plan(label, bound, supplied):
    """The vjp plan of launch node ``label``: ``supplied`` (checked against
    the kernel it differentiates) or the one derived beside an auto plan."""
    from ..plan import plan

    kernel = bound.plan.plugin.name
    if supplied is None:
        found = _vjp_from_auto(bound)
        if found is None:
            raise ValueError(
                f"eagle.diff.Reverse: node {label!r} has no vjp kernel and none was "
                "passed: eagle derives one only for a plan made by "
                f"eagle.plan.auto(kernel). Build hawk.diff.vjp(<kernel {kernel!r}>) "
                f"and pass vjps={{{label!r}: plan}}.")
        return plan(found, structure=bound.plan.structure)
    tag = derived_tag(supplied)
    if tag is None or tag[0] != "vjp":
        raise ValueError(
            f"eagle.diff.Reverse: vjps[{label!r}] is not a vjp kernel (it carries no "
            f"vjp derivative tag); pass the plan of hawk.diff.vjp(<kernel {kernel!r}>)")
    if tag[1] != kernel:
        raise ValueError(
            f"eagle.diff.Reverse: vjps[{label!r}] differentiates kernel {tag[1]!r}, "
            f"but node {label!r} launches {kernel!r}; pass the vjp of {kernel!r}.")
    return supplied if hasattr(supplied, "bind") else plan(supplied, structure=bound.plan.structure)


def _check_vjp_covers(label, bound, vplan):
    """Refuse a vjp that leaves out the gradient of a differentiable input
    (one built with a narrower ``wrt``): its share of that gradient would be
    silently missing."""
    from .. import _split

    returned = {name[4:] for _role, name in _split.arg_spec(vplan.plugin)
                if name.startswith("bar_")}
    for role, name in bound._arg_spec:
        value = bound._planes.get(name)
        wanted = role not in OUTPUT_ROLES and _is_float(value)
        if wanted and name not in returned:
            raise ValueError(
                f"eagle.diff.Reverse: the vjp of node {label!r} returns no gradient "
                f"for {name!r}; build it with hawk.diff.vjp(kernel) and the default "
                "wrt (every input)")


# --- The reverse --- #
def _names_by(per_leaf):
    """``{key: [(leaf, local name), ...]}`` from per-leaf ``(local, key)`` lists."""
    out = {}
    for leaf, pairs in enumerate(per_leaf):
        for local, key in pairs:
            if (leaf, local) not in out.setdefault(key, []):
                out[key].append((leaf, local))
    return out


def _name_tables(names, labels, keys):
    """``(display, lookup)`` for ``keys``: the name each key is shown under (its
    plane name, or ``"node.plane"`` when two keys share that name) and every
    name a caller may use for it."""
    owners = {}
    for key in keys:
        for _leaf, local in names[key]:
            owners.setdefault(local, set()).add(key)
    display, lookup = {}, {}
    for key in keys:
        for leaf, local in names[key]:
            lookup[f"{labels[leaf]}.{local}"] = key
            if len(owners[local]) == 1:
                lookup.setdefault(local, key)
                display.setdefault(key, local)
        leaf, local = names[key][0]
        display.setdefault(key, f"{labels[leaf]}.{local}")
    return display, lookup


class Reverse:
    """The reverse pass of a recorded forward, built once and called per
    Record: ``rev = eagle.diff.Reverse(forward)``, then ``grads =
    rev(forward.record, seeds)`` after each recorded forward. On the GPU a
    graph's reverse is captured once and replayed.

    ``forward`` is what ran the forward with ``record=True``: a built
    :class:`eagle.Graph`, an :class:`eagle.Simulation` or an
    :class:`eagle.Runner`. ``vjps`` maps a node name to the
    plan of its vjp (``hawk.diff.vjp`` of its kernel): needed for a plan not
    made by :func:`eagle.plan.auto` (eagle derives the vjp of an auto plan by
    itself), and overriding the derived one otherwise. For a node of your own
    code (:func:`eagle.graph.opaque`, :func:`eagle.graph.child`) it is
    required: a callable ``vjp(primals, cotangents) -> {read name:
    gradient}`` (``primals`` the values the node read in the recorded launch,
    ``cotangents`` the adjoints of what it wrote, both by declared name),
    run host-synchronized in the reverse on the GPU and eagerly on the CPU,
    or a Node run at its own tier (graphs only). For a recorded run whose step
    list is ``[node, plan]`` the callable runs once per step; it must be pure
    (same inputs, same outputs; row-wise; writes only its declared planes; the
    weights untouched until the reverse returns), and the node's planes are
    eagle's scratch until the reverse returns them restored. Your model's own parameter gradients are
    yours to accumulate (``p.grad += ...``) across reverses. ``policy`` is the
    :func:`eagle.reduce` policy of the sums over samples eagle inserts
    (``"compensated"`` by default; ``report.bitwise`` is True for such a
    gradient only under ``"ordered"``). ``strict=True`` sums with
    ``"ordered"`` and refuses a gradient that is not bit-reproducible (one fed
    by a cross-sample scatter-add) instead of reporting it.

    Example::

        sim = eagle.simulation(model, x=x0, v=v0, max_steps=1000, record=True)
        rev = eagle.diff.Reverse(sim)
        for _ in range(iterations):
            sim.reset(); sim.run()
            grads = rev(sim.record, {"x": dJ_dx})
    """

    def __init__(self, forward, *, vjps=None, strict=False, policy="compensated"):
        door = "eagle.diff.Reverse"
        from .._graph_reduce import POLICIES, check_policy

        check_policy(policy, door)
        self._policy = "ordered" if strict else policy
        if isinstance(forward, Record):
            raise TypeError(f"{door}: takes what ran the forward (Reverse(sim) or "
                            "Reverse(g)); the Record goes to the call: rev(sim.record, seeds)")
        from ._run import RunReverse, reverse_of

        self._run = None
        if not isinstance(forward, Graph):
            recorder = reverse_of(forward)
            if recorder is None:
                if hasattr(getattr(forward, "runner", forward), "_max_steps"):
                    raise ValueError(f"{door}: the run was not built with record=True; "
                                     "pass record=True to the forward, then run it")
                raise TypeError(f"{door}: expected an eagle.Graph, Simulation or Runner, "
                                f"got {type(forward).__name__}")
            for label, fn in (vjps or {}).items():
                if isinstance(fn, Node):
                    raise ValueError(
                        f"{door}: vjps[{label!r}] is a Node; a recorded run takes a callable "
                        "vjp(primals, cotangents) for now (a captured backward as a run's "
                        "node vjp is not supported yet)")
            self._graph = recorder.owner
            self._run = RunReverse(recorder, door, dict(vjps or {}), POLICIES[self._policy])
            self._strict = bool(strict)
            return
        graph = forward
        if graph._is_reverse:
            raise ValueError(f"{door}: this graph is itself a reverse pass; reversing "
                             "a reverse (second order) is not supported")
        if not graph._built or graph._recorder is None:
            raise ValueError(f"{door}: the graph was not built with record=True; "
                             "build it with g.build(record=True), then g.launch()")
        from .._graph_view import description

        self._graph = graph
        self._strict = bool(strict)
        self._planes = planes = graph._recorder.planes
        self._accesses = accesses = graph._leaves()[0]
        self._labels = labels = [a.label for a in accesses]
        self._desc = description(graph, drawing=False)
        vjps = dict(vjps or {})
        unknown = sorted(set(vjps) - set(labels))
        if unknown:
            raise ValueError(f"{door}: vjps names {unknown[0]!r}, which is not a node "
                             f"of the graph (nodes: {labels})")
        for a in accesses:
            node = a.node.payload.child if a.node.kind in _CONDITIONALS else a.node
            if node.kind in OWN_CODE_KINDS and a.label not in vjps:
                raise ValueError(
                    f"{door}: node {a.label!r} is opaque: pass vjps={{{a.label!r}: fn}}, a "
                    "vjp(primals, cotangents) -> {read name: gradient} callable (or a Node "
                    "computing it)")
        rows = planes.rows()
        self._reads = planes.reads
        for i in range(len(accesses)):
            seen = {}
            for local, key in self._reads[i]:
                if seen.setdefault(key, local) != local:
                    raise ValueError(
                        f"{door}: node {labels[i]!r} reads one array as both "
                        f"{seen[key]!r} and {local!r}; a reverse needs a separate "
                        "array per input (pass a copy)")
        self._rows = rows
        try:
            gradients = _core.diff_transpose(self._desc, rows, [])["gradients"]
        except ValueError as exc:
            raise ValueError(f"{door}: {exc}") from None
        self._vjp = {}
        for i, a in enumerate(accesses):
            kind = get_node_kind(a.node.kind)
            if kind.transpose is None:
                raise ValueError(f"{door}: node {labels[i]!r} (kind {a.node.kind!r}) "
                                 "has no transpose; it cannot be reversed")
            node = a.node
            if node.kind in _CONDITIONALS:
                node = node.payload.child
            if not a.grad:
                continue
            if node.kind == "launch":
                b = node.payload
                self._vjp[i] = _vjp_plan(labels[i], b, vjps.get(labels[i]))
                _check_vjp_covers(labels[i], b, self._vjp[i])
            elif labels[i] in vjps:
                given = vjps[labels[i]]
                if not callable(given) and not isinstance(given, Node):
                    raise TypeError(f"{door}: vjps[{labels[i]!r}] is the vjp of a node of your "
                                    "own code: a callable vjp(primals, cotangents) or a Node, "
                                    f"got {type(given).__name__}")
                self._vjp[i] = given
        written = sorted({k for w in planes.writes for _, k in w})
        inputs = [k for k, _src in gradients if _is_float(planes.array[k])]
        # an output goes by the names its writers give it, an input by its readers'
        self._outputs, self._output_names = _name_tables(
            _names_by(planes.writes), labels, written)
        self._inputs, self._input_names = _name_tables(_names_by(self._reads), labels, inputs)
        self._busy = BusyGuard("eagle.diff.Reverse")
        self._programs = weakref.WeakKeyDictionary()
        self._lock = threading.Lock()

    def _seed_keys(self, seeds, tape):
        door = "eagle.diff.reverse"
        out = {}
        for name, value in seeds.items():
            key = self._output_names.get(name)
            if key is None:
                outputs = list(self._outputs.values())
                hint = (f"; the gradient with respect to {name!r} is read from the "
                        f"result: grads[{name!r}]") if name in self._input_names else ""
                raise ValueError(
                    f"{door}: seeds names {name!r}, which is not an output of the "
                    f"recorded graph (outputs: {', '.join(map(repr, outputs))}). seeds "
                    "are the gradient of your objective with respect to each OUTPUT "
                    f"plane (∂J/∂{outputs[0]}), one value per sample{hint}.")
            plane = tape.arrays[key]
            shape = tuple(getattr(value, "shape", ()))
            if shape != tuple(plane.shape):
                n = plane.shape[-1]
                got = shape[-1] if shape else 1
                raise ValueError(
                    f"{door}: seeds[{name!r}] has {got} entries but the record holds "
                    f"{n} samples; pass one value per sample in sample order (shape "
                    f"{tuple(plane.shape)}, dtype {plane.dtype} like plane "
                    f"{self._outputs[key]!r}).")
            out[key] = value
        return out

    def __call__(self, record, seeds, *, wrt=None, _at=()) -> Gradients:
        """The gradients for one recorded launch. ``seeds`` maps output planes
        to ∂J/∂(output), one value per sample in sample order (unnamed outputs
        are seeded 0; JAX calls these cotangents, PyTorch grad_outputs, CasADi
        adj_seed, optimal control the terminal costate). ``wrt`` names the
        input planes to return (default: all)."""
        door = "eagle.diff.reverse"
        if not isinstance(record, Record) or record.forward is not self._graph:
            raise ValueError(f"{door}: expected the Record of this Reverse's forward "
                             "(its .record after it ran)")
        if self._run is not None:
            return self._run(record, seeds, wrt, door)
        keys = (list(self._inputs) if wrt is None else
                [self._input_key(name) for name in ([wrt] if isinstance(wrt, str) else wrt)])
        with self._busy("__call__"), record._hold(door) as tape:
            seeded = self._seed_keys(seeds, tape)
            program = self._program(tape, frozenset(seeded))
            values, bitwise = program.run(tape, seeded, keys)
            cot = {at: program.cot[at].copy() for at in _at}
        _counters.bump("reverses")
        names = {self._inputs[k]: v for k, v in values.items()}
        return Gradients(names, Report(record.generation,
                                       {self._inputs[k]: bitwise[k] for k in values}), cot)

    def _input_key(self, name):
        key = self._input_names.get(name)
        if key is None:
            raise ValueError(
                f"eagle.diff.reverse: wrt names {name!r}, which is not an input of the "
                f"recorded graph (inputs: {', '.join(map(repr, self._inputs.values()))})")
        return key

    def _program(self, tape, seeded):
        """The reverse graph for this tape's arrays and seeded set, built on
        first use and replayed after (a parameter lives in a plane, so a
        rebind changes nothing here)."""
        with self._lock:
            held = self._programs.setdefault(tape.source, {}).get(seeded)
        if held is None:
            held = _Program(self, tape, seeded)
            with self._lock:
                self._programs[tape.source][seeded] = held
        return held


class _Program:
    """One built reverse graph and the planes it owns."""

    def __init__(self, rev, tape, seeded):
        plan = _core.diff_transpose(rev._desc, rev._rows, sorted(seeded))
        planes, accesses = rev._planes, rev._accesses
        backend = rev._graph.backend
        self.xp = xp = xp_of(next(iter(tape.arrays.values())))
        self.perm = tape.perm
        n = None if tape.perm is None else int(tape.perm.shape[0])
        self.inv = None if n is None else _alloc.empty(xp, n, dtype=xp.int32)
        self.seed = {k: _alloc.empty_like(xp, tape.arrays[k]) for k in seeded}
        zeros = {}

        def zero(key):
            arr = tape.arrays[key]
            k = (tuple(arr.shape), str(arr.dtype))
            if k not in zeros:
                zeros[k] = _alloc.zeros(xp, arr.shape, dtype=arr.dtype)
            return zeros[k]

        out, taint = {}, {}
        self.comp = {}
        self.cot = {}     # {(forward leaf, plane): the adjoint arriving at it}

        def source(s, key):
            if s == _core.DIFF_SEED:
                return self.seed[key], None
            if s == _core.DIFF_ZERO or (s, key) not in out:
                return zero(key), None
            return out[(s, key)], taint.get((s, key))

        g = Graph()
        joins = {t: (plane, srcs) for t, plane, srcs in plan["joins"]}
        for r, fwd in enumerate(plan["forward"]):
            if fwd < 0:
                plane, srcs = joins[r]
                if not _is_float(tape.arrays[plane]):
                    continue
                parts = [source(s, plane) for s in srcs]
                arrays = [a for a, _ in parts]
                dst = _alloc.empty_like(xp, arrays[0])
                g.add_node(opaque(_ordered_sum(dst, arrays), reads=arrays, writes=[dst]),
                           name=f"sum({plane})")
                if any(id(a) in self.comp for a in arrays):     # rows: their companions add too
                    los = [self.comp.get(id(a), zero(plane)) for a in arrays]
                    self.comp[id(dst)] = lo = _alloc.empty_like(xp, dst)
                    g.add_node(opaque(_ordered_sum(lo, los), reads=los, writes=[lo]),
                               name=f"sum({plane}, comp)")
                out[(r, plane)] = dst
                taint[(r, plane)] = next((t for _, t in parts if t), None)
                continue
            leaf = accesses[fwd]
            if not leaf.grad:       # not differentiated: its outputs pass no gradient back
                continue
            reads, writes = rev._reads[fwd], planes.writes[fwd]
            incoming, came = {}, None
            for (local, key), s in zip(writes, plan["seed_of"][r]):
                incoming[local], t = source(s, key)
                self.cot[(fwd, local)] = incoming[local]
                came = came or t
            read_of = dict(zip((k for _, k in reads), plan["read_of"][r]))
            local_key = dict(reads)

            def primal(name, read_of=read_of, local_key=local_key):
                key = local_key[name]
                idx = read_of.get(key, -1)
                return tape.copies[idx] if idx >= 0 else tape.arrays[key]

            snaps, wkey = rev._graph._recorder.recorded, dict(writes)

            def recorded(name, fwd=fwd, snaps=snaps):
                return tape.copies[snaps[(fwd, name)]]

            def written(name, wkey=wkey):
                return tape.arrays[wkey[name]]

            ctx = ReverseCtx(backend, leaf.label, rev._vjp.get(fwd), primal, recorded, written,
                             [n for n, _k in reads])
            result = get_node_kind(leaf.node.kind).transpose(leaf.node, incoming, ctx)
            self.comp.update(ctx.comp)
            for step in ctx._steps:
                if hasattr(step, "_arg_spec"):
                    g._add_derived(step, _DERIVED_DOOR, name=f"vjp({leaf.label})")
                else:
                    g.add_node(step)
            for local, key in reads:
                if local in result:
                    out[(r, key)] = result[local]
                    taint[(r, key)] = leaf.label if local in ctx.inexact else came
        self.grads, self.bitwise, self.reduced = {}, {}, {}
        for key, s in plan["gradients"]:
            arr, origin = source(s, key)
            if not _is_float(arr):
                continue
            if origin and rev._strict:
                raise ValueError(
                    f"eagle.diff.reverse: strict=True, but the gradient of "
                    f"{rev._inputs.get(key, key)!r} comes through a cross-sample "
                    f"scatter-add in node {origin!r}, whose summing order varies run "
                    "to run; drop strict=True to accept it (grads.report.bitwise "
                    "marks it)")
            self.bitwise[key] = origin is None
            lo = self.comp.get(id(arr))
            if lo is not None or arr.shape != tape.arrays[key].shape:
                # one parameter's partial sums (rows of 32 samples) or per-sample terms
                self.bitwise[key] = origin is None and rev._policy == "ordered"
                if lo is None:
                    arr = self._sample_order_step(g, arr)
                total = _alloc.empty(xp, 1, dtype=arr.dtype)
                from .._graph_reduce import reduce

                g.add_node(reduce(arr, total, comp=lo, policy=rev._policy),
                           name=f"sum({rev._inputs.get(key, key)})")
                self.reduced[key] = total
            else:
                self.grads[key] = arr
        g.build(backend=backend)
        self.graph = g

    def _sample_order_step(self, g, arr):
        """``arr`` gathered into sample order inside the graph (a reordered
        batch), so the parameter sum runs in sample order, never slot order."""
        if self.inv is None:
            return arr
        xp, inv, dst = self.xp, self.inv, _alloc.empty_like(self.xp, arr)
        g.add_node(opaque(lambda: xp.take(arr, inv, axis=-1, out=dst),
                          reads=[arr, inv], writes=[dst]), name="sample order")
        return dst

    def run(self, tape, seeded, keys):
        xp = self.xp
        if self.perm is not None:
            self.inv[self.perm] = xp.arange(self.perm.shape[0], dtype=xp.int32)
        for key, buf in self.seed.items():
            value = xp.asarray(seeded[key], dtype=buf.dtype)
            if self.perm is None:
                buf[...] = value
            else:
                xp.take(value, self.perm, axis=-1, out=buf)
        if xp.__name__ == "cupy":
            xp.cuda.get_current_stream().synchronize()
        self.graph.launch()
        values = {}
        for key in keys:
            if key in self.reduced:
                values[key] = self.reduced[key].reshape(()).copy()
            elif key in self.grads:
                arr = self.grads[key]
                values[key] = arr.copy() if self.inv is None else xp.take(arr, self.inv, axis=-1)
        return values, self.bitwise


def _ordered_sum(dst, arrays):
    """``dst = ((a0 + a1) + a2) + ...``: one fixed order, no atomics."""
    if xp_of(dst).__name__ == "cupy":
        import cupy

        def step():
            cupy.add(arrays[0], arrays[1], out=dst)
            for a in arrays[2:]:
                cupy.add(dst, a, out=dst)
        return step
    code = 1 if str(dst.dtype) == "float64" else 0
    ptrs = [a.ctypes.data for a in arrays]

    def step():
        _core.ordered_sum(dst.ctypes.data, ptrs, dst.size, code)
    return step


def reverse(record, seeds, *, wrt=None, vjps=None, strict=False, policy="compensated") -> Gradients:
    """The gradients of one recorded forward, in one call:
    ``grads = eagle.diff.reverse(sim.record, seeds={"x": dJ_dx})``.

    ``record`` is the ``.record`` of a ``record=True`` forward (a Graph
    launch, a simulate / Simulation / until_done run); ``seeds`` maps output
    planes to ∂J/∂(output), one value per sample in sample order. The
    result maps each input plane to ∂J/∂(input): per sample for a per-sample
    plane, summed over samples for a shared parameter.

    Builds a :class:`Reverse` per call; to reverse the same forward many
    times, build ``eagle.diff.Reverse(forward)`` once and call it. Arguments
    as :class:`Reverse` and its call.

    Example::

        res = eagle.simulate(model, x=x0, v=v0, max_steps=1000, record=True)
        grads = eagle.diff.reverse(res.record, {"x": numpy.ones_like(x0)})
        grads["x"]     # dJ/dx0 per sample"""
    if not isinstance(record, Record):
        raise TypeError("eagle.diff.reverse: expected a Record (the .record of a "
                        f"record=True forward), got {type(record).__name__}")
    keys = None
    if getattr(getattr(record._graph, "_bld", None), "builder", False):
        from ._keys import translate

        seeds, wrt, keys = translate(record._graph, seeds, wrt)
    at = [] if keys is None else [v[1] for v in keys.values() if not isinstance(v, str)]
    grads = Reverse(record.forward, vjps=vjps, strict=strict, policy=policy)(record, seeds, wrt=wrt, _at=at)
    if keys is None:
        return grads
    from ._keys import own_shape

    return Gradients({name: own_shape(record._graph, name, grads.cotangents[k[1]] if not isinstance(k, str)
                                      else grads[k if k in grads else k.split(".", 1)[1]])
                      for name, k in keys.items()}, grads.report)
