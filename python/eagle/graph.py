# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.Graph``: one backend-neutral graph of launches, built once and
lowered to the GPU (stream capture, conditional weave) or to the CPU (add
order, eager host arms).

Edges are DERIVED from the arrays the nodes touch (``eagle._graph_edges``),
never declared; two siblings of a fork/join group that overlap are refused at
:meth:`Graph.build` on both backends. Node kinds live in a registry
(:func:`register_node_kind`): a later feature adds a kind from its own
module and this one never changes.

Thread-safety: module state is the registry (a dict under a named lock,
written at import time). One :class:`Graph` shared across threads is
caller-ordered, and ``build``/``launch``/``enqueue`` raise
:class:`eagle.Busy` when another of them is in flight on that object;
distinct Graphs run concurrently.
"""

from __future__ import annotations

import contextlib
import threading
import warnings
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

from . import _alloc, _split
from . import _counters, _graph_view
from ._busy import BusyGuard
from ._conditional import (Branches, RepeatWhile, SkipGuard, Skippable,
                           has_branch_nodes)
from ._device_kind import GPU_KINDS, device_kind, gpu_kind
from ._graph_edges import (SHARES_MEMORY_MAX_WORK, Edge, check_conflicts,
                           arrays_overlap, derive_edges)
from ._active_set import _address
from .interop import scoped_stream
from .sidecar import refuse_derived

__all__ = ["Edge", "Graph", "Node", "NodeKind", "SHARES_MEMORY_MAX_WORK",
           "child", "get_node_kind", "opaque", "register_node_kind"]

_BACKENDS = ("gpu", "cpu", *GPU_KINDS)
#: How a node runs on the GPU, and what ``g.schedule`` names its stage there.
_TIERS = MappingProxyType({"capture": "capture", "stream": "eager_stream", "host": "eager_host"})


# --- The node-kind registry --- #
@dataclass(frozen=True)
class NodeKind:
    """What the Graph needs to know about one kind of node.

    ``reads(node)`` / ``writes(node)``: ``(plane_name, user_array, role)``
    tuples; an in-place plane (a STATE role) is in both. ``lower_gpu(node,
    ctx)`` returns what the capture loop runs (a zero-arg callable, a
    ``Skippable``/``RepeatWhile``, or a :class:`ForkStage`);
    ``lower_cpu(node, ctx)`` the eager host arm (a zero-arg callable).
    ``ctx`` is a :class:`LowerCtx`. ``transpose`` (reverse), ``record_planes
    (node) -> ((name, array), ...)`` (what a recorder snapshots per launch)
    and ``native_entry(node, ctx) -> (entry, params, partition) | None``
    (what a native host runner executes; ``None`` = a Python host arm) are
    optional. ``check(node)`` (optional) runs before every launch and
    raises when the node can no longer run (a framework graph reset).

    ``tier`` is how the node runs on the GPU: ``"capture"`` (captured into
    the graph), ``"stream"`` (a host call each launch whose device work is
    on its own stream, fenced on the device; ``lower_gpu`` returns a callable
    whose ``.stream`` is the stream handle) or ``"host"`` (a host call between two device syncs,
    host readbacks allowed). A graph with a non-captured node is captured
    in segments around it; a graph without one is one capture."""

    kind: str
    reads: Callable
    writes: Callable
    lower_gpu: Callable
    lower_cpu: Callable
    transpose: Callable | None = None
    record_planes: Callable | None = None
    native_entry: Callable | None = None
    tier: str = "capture"
    check: Callable | None = None


_REGISTRY_LOCK = threading.Lock()
_DERIVED_DOOR = object()    # the token of Graph._add_derived, used by eagle.diff only
_REGISTRY: dict = {}
#: ``factory(graph) -> recorder``, installed once by eagle.diff (under
#: _REGISTRY_LOCK). Made before the edges are derived (it may re-bind the
#: launches); ``attach(accesses)`` then plans the copies to run before each stage
#: (``before(stage)``); it checks a launch may record (``begin(n, method)``),
#: makes the ``Record`` after it (``finish()``) and writes parameters
#: (``update(values)``).
_RECORD_FACTORY = None


def register_node_kind(kind: str, *, reads, writes, lower_gpu, lower_cpu,
                       transpose=None, record_planes=None,
                       native_entry=None, tier="capture", check=None) -> NodeKind:
    """Register a node kind; idempotent for an identical registration.
    Registering a DIFFERENT kind under an existing name is refused (a silent
    swap would re-type every Graph built so far). Import-time only."""
    if not isinstance(kind, str) or not kind:
        raise TypeError("eagle.graph.register_node_kind: kind must be a non-empty str")
    if tier not in _TIERS:
        raise ValueError(f"eagle.graph.register_node_kind: tier must be one of "
                         f"{sorted(_TIERS)}, got {tier!r}")
    new = NodeKind(kind, reads, writes, lower_gpu, lower_cpu, transpose,
                   record_planes, native_entry, tier, check)
    with _REGISTRY_LOCK:
        old = _REGISTRY.get(kind)
        if old is None:
            _REGISTRY[kind] = new
            return new
        if old == new:
            return old
    raise ValueError(
        f"eagle.graph.register_node_kind: node kind {kind!r} is already "
        "registered with a different definition; a registered kind is never "
        "replaced -- pick another name")


def get_node_kind(kind: str) -> NodeKind:
    """The registered :class:`NodeKind` named ``kind``; ``KeyError``-free:
    raises ``ValueError`` naming the registry when absent."""
    with _REGISTRY_LOCK:
        found = _REGISTRY.get(kind)
        known = sorted(_REGISTRY)
    if found is None:
        raise ValueError(
            f"eagle.Graph: no node kind {kind!r} is registered "
            f"(eagle.graph.register_node_kind); registered: {known}")
    return found


def _install_record_factory(factory) -> None:
    """Private: ``eagle.diff`` installs the recorder factory here."""
    global _RECORD_FACTORY
    with _REGISTRY_LOCK:
        _RECORD_FACTORY = factory


class Node:
    """One node: a registered ``kind``, its ``payload`` and an optional
    ``name`` (the label). Immutable; build them with :meth:`Graph.add`,
    :func:`opaque`, or a kind's own
    constructor."""

    __slots__ = ("kind", "payload", "name", "grad")

    def __init__(self, kind: str, payload, name: str | None = None, grad: bool = True):
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "grad", grad)

    def __setattr__(self, key, value):
        raise AttributeError("eagle.graph.Node is immutable")

    def _named(self, name):
        return self if name is None or name == self.name else Node(
            self.kind, self.payload, name, self.grad)

    @property
    def label(self) -> str:
        if self.name:
            return self.name
        if self.kind == "launch":      # an unnamed plan launch is its kernel's name
            kernel = getattr(getattr(getattr(self.payload, "plan", None), "plugin", None),
                             "name", None)
            if isinstance(kernel, str) and kernel:
                return kernel
        return self.kind

    def __call__(self):
        """Run this node eagerly on the host (its CPU lowering): what lets a
        Node be the step of ``eagle.skippable`` / ``eagle.repeat_while``."""
        LowerCtx("cpu").lower(self)()

    def __repr__(self) -> str:
        return f"Node({self.kind!r}, name={self.name!r})"


class ForkStage:
    """What a group's ``lower_gpu`` returns: the members of one fork/join
    stage, run concurrently under capture."""

    __slots__ = ("members", "syncs")

    def __init__(self, members, syncs=()):
        self.members = list(members)
        self.syncs = list(syncs)       # the cross-branch waits the weave wires


@dataclass(frozen=True)
class LowerCtx:
    """What a lowering hook receives: the ``backend`` (``"gpu"``/``"cpu"``),
    whether the node is lowered into a conditional or loop ``body``, and
    :meth:`lower`, which lowers a child node for that backend. ``host`` holds
    the ids of the nodes of a GPU graph whose planes are all host arrays: they
    lower to their CPU arm, run from the capture as a host function."""

    backend: str
    body: bool = False
    host: frozenset = frozenset()

    def lower(self, node: Node):
        k = get_node_kind(node.kind)
        if self.backend != "gpu":
            return k.lower_cpu(node, self)
        if self.host and id(node) in self.host:
            return _transfer.HostNode(k.lower_cpu(node, self))
        return k.lower_gpu(node, self)


class _Eager:
    """A segment of a GPU graph that runs on the host each launch: the
    recorder's ``copies`` then ``call``, on ``stream`` (a stream node) or
    between two device syncs (``stream is None``, a host node)."""

    __slots__ = ("copies", "call", "stream", "event")

    def __init__(self, copies, call, stream):
        self.copies, self.call = copies, call
        self.stream = self.event = None
        if stream is not None:
            self.stream = int(stream)
            self.event = _stream.Event()

    def run(self, waits):
        """Run once after ``waits`` (events); returns the event that ends it,
        or ``None`` for a host node (everything before and in it is done)."""
        if self.stream is None:
            _device_sync()
            for copy in self.copies:
                copy()
            self.call()
            _device_sync()
            return None
        _wait_stream(self.stream, waits)
        with scoped_stream(self.stream):
            for copy in self.copies:
                copy()
            self.call()
            self.event.record(self.stream)
        return self.event


def _device_sync():
    """A host node's fence: every stream of the device, whichever framework
    issued on it, is done (this also waits other threads' device work)."""
    from . import _core

    _core.device_synchronize(-1)


def _wait_stream(stream, events):
    for event in events:
        event.wait(stream)


def _run_seeds(cache, tail):
    """What the first stage of a run waits on: the stream the caller works on,
    the legacy stream (``cache`` holds the two re-recorded events) and the
    previous run's ``tail`` event (run-to-run WAR/WAW)."""
    events = _stream.seed_events(cache)
    if tail is not None:
        events.append(tail)
    return events


_CONDITIONALS = ("skippable", "repeat_while", "branches")


def _refuse_mixed(access):
    names = {k: next((n, a) for n, a, _r in (*access.reads, *access.writes) if (device_kind(a) == "host") == (k == "host"))
             for k in ("host", "device")}
    host, device = names["host"][0], names["device"][0]
    raise ValueError(
        f"eagle.Graph.build: node {access.label!r} has planes on the host ({host!r}) and on the "
        f"device ({device!r}); a node runs where its planes live. Add "
        f"eagle.copy(<{host}>, into=<a device array>) before it, or "
        f"eagle.copy(<a device array>, into=<{host}>) after it")


@contextlib.contextmanager
def _pinned_hint(nodes):
    """CUDA instantiates a host copy inside a device body only from page-locked
    memory; its refusal names nothing, so say what to change."""
    try:
        yield
    except RuntimeError as exc:
        copy = next((n for c in _walk(nodes) if c.kind in _CONDITIONALS
                     for n in _walk([c.payload.child])
                     if n.kind == "copy" and any(device_kind(a) == "host" for a in n.payload[:2])), None)
        if copy is None:
            raise
        raise ValueError(
            "eagle.Graph.build: a copy between host and device inside a device if/for body needs "
            "page-locked host memory; make the host array with eagle.pinned(shape, dtype) "
            f"(or move the copy out of the body) [{str(exc).strip().splitlines()[-1]}]") from None


def _refuse_host_in_body(access):
    """A node of host arrays inside a device ``if`` / ``for`` body cannot be
    captured (CUDA has no host function node in a conditional body)."""
    for node in _walk([access.node.payload.child]):
        if node.kind in ("group", "sequence", *_CONDITIONALS, "copy"):
            continue
        k = get_node_kind(node.kind)
        if {device_kind(a) for _n, a, _r in (*k.reads(node), *k.writes(node))} == {"host"}:
            raise ValueError(
                f"eagle.Graph.build: node {node.label!r} runs on the host (its planes are host "
                f"arrays) inside the device if/for body of {access.label!r}, and CUDA cannot run "
                "a host function there. Move it out of the body, or make its planes device "
                "arrays so it runs on the device (eagle.copy is allowed in a body)")


# --- The Graph --- #
class _Access:
    """One executable node for the edge derivation."""

    __slots__ = ("label", "reads", "writes", "node", "grad")

    def __init__(self, node):
        k = get_node_kind(node.kind)
        self.node = node
        self.grad = node.grad
        self.label = node.label
        self.reads = tuple(k.reads(node))
        self.writes = tuple(k.writes(node))


@contextlib.contextmanager
def _on_device(device):
    """Make ``device`` (a CUDA ordinal, ``None`` for a host graph) the calling
    thread's current device and context, whatever a framework left current
    there (thread-local, restored on exit)."""
    if device is None:
        yield
        return
    with _stream.on_device(device):
        yield


def _plane_device(accesses, device=None):
    """The CUDA ordinal the graph's planes live on, which is the one device it runs on. A plane on another
    device is refused by name (plane, its device, the graph's device) before any kernel is reached: planes
    on two devices, and an explicit ``device`` a plane is not on."""
    from ._plan_planes import array_device

    first = {}          # device ordinal -> the first plane found on it
    for a in accesses:
        for name, arr, _r in (*a.reads, *a.writes):
            if device_kind(arr) != "host":
                first.setdefault(array_device(arr), name)
    found = sorted(first)
    if len(found) > 1:
        raise ValueError(
            f"eagle.graph: plane {first[found[1]]!r} lives on device {found[1]}, the graph runs on "
            f"device {found[0]} (where plane {first[found[0]]!r} is); the planes span devices {found}, "
            "and a graph runs on one device")
    if device is not None and found and found[0] != device:
        raise ValueError(
            f"eagle.graph: plane {first[found[0]]!r} lives on device {found[0]}, the graph runs on "
            f"device {device} (device={device} was asked); device= is checked, never a move")
    return found[0] if found else device


def _place(array) -> str:
    """The device token (``"cpu"`` or ``"cuda:N"``) ``array`` lives on, for allocating next to it."""
    if device_kind(array) == "host":
        return "cpu"
    from ._plan_planes import array_device

    return f"cuda:{array_device(array)}"


def _walk(nodes):
    """Every node of the tree, bodies and branches included."""
    for node in nodes:
        yield node
        if node.kind in ("skippable", "repeat_while"):
            yield from _walk([node.payload.child])
        elif node.kind == "branches":
            yield from _walk(node.payload.child.payload)
        elif node.kind == "sequence":
            yield from _walk(node.payload)
        elif node.kind == "group":
            yield from _walk(n for branch in node.payload for n in branch)


def _inner_groups(nodes, inside=False):
    """The fork/join groups held in the bodies of conditionals and loops: the
    leaves of the Graph see such a body as one node, so each is checked on its
    own (a group nested in a group is covered by its outer group's check)."""
    for node in nodes:
        if node.kind in ("skippable", "repeat_while", "branches"):
            yield from _inner_groups([node.payload.child], True)
        elif node.kind == "sequence":
            yield from _inner_groups(node.payload, inside)
        elif node.kind == "group":
            if inside:
                yield node
            yield from _inner_groups((n for branch in node.payload for n in branch))


def _writes_only_cells(bound) -> bool:
    """Does the launch write only ``uint32`` planes (a loop or branch guard's
    cells)? Such a launch carries no gradient."""
    writes = [arr for _n, arr, _r in get_node_kind("launch").writes(Node("launch", bound))]
    return bool(writes) and all(str(arr.dtype) == "uint32" for arr in writes)


def _array_backend(arr):
    return "cpu" if device_kind(arr) == "host" else "gpu"


class Graph:
    """A graph of launches: build it once, launch it many times.

    ``add``/``add_group``/``add_node`` append (chaining); ``build`` derives
    the edges, refuses conflicts, and lowers to the GPU or CPU; ``launch`` /
    ``enqueue`` / ``synchronize`` are :class:`eagle.GraphPipeline`'s verbs
    (on the CPU ``enqueue`` is ``launch``). ``show()`` / ``str()`` /
    ``plot()`` / ``to_mermaid()`` / ``to_dot()`` draw it. The keywords
    ``record``, ``backend``, ``dispatch`` and ``device`` are the defaults of
    :meth:`build`'s."""

    def __init__(self, *, record=False, backend=None, dispatch=None, device=None):
        from . import _builder

        self._defaults = dict(record=record, backend=backend, dispatch=dispatch, device=device)
        self._bld = _builder.State()
        self._nodes: list = []
        self._busy = BusyGuard("eagle.Graph")
        self._built = False
        self._waited = (set(), [])
        self._steps: list = []     # a GPU build's captured steps, each with its graph nodes
        self._host_nodes: list = []   # its host function nodes, which keep the error they hit
        self._rebuilt = False
        self._patch = None         # where each array is used, for _rebind
        self._backend = None
        self._edges: tuple = ()
        self._labels: list = []
        self._schedule: tuple = ()
        self._device = None     # the CUDA ordinal of the planes (GPU graph)
        self._pipe = None
        self._segments = None   # a GPU graph with an eager node: [(pipeline, event) | _Eager]
        self._seeds = None
        self._tail_ev = None
        self._live = ()         # what must hold before each launch (child handles, bound weights)
        self._cpu_steps: list = []
        self._record_on = False
        self._recorder = None
        self._record = None
        self._is_reverse = False
        self._dispatch = {}
        _builder.attach(self)

    @property
    def dispatch(self) -> dict:
        """``{node label: {group: policy}}`` the last :meth:`build` lowered
        (empty when no node had an ``"auto"`` dispatch group)."""
        return dict(self._dispatch)

    # -- construction -- #
    def _check_open(self, method):
        if self._built:
            raise RuntimeError(f"eagle.Graph.{method}: this Graph is built; "
                               "make a new Graph to add nodes")

    def _child_of(self, step, door, loop=False):
        """The Node a conditional's step stands for: a Node itself (a group
        too), a ``BoundPlan.launch`` (its planes are then derived), a
        ``skippable`` / ``repeat_while`` (adopted: conditionals nest), a tuple
        of such parts run in order (a ``"sequence"`` node), or any other
        zero-arg callable (opaque: no declared planes). A loop's single
        conditional step is wrapped in a sequence."""
        if isinstance(step, list):
            step = tuple(step)
        if loop and isinstance(step, (Skippable, RepeatWhile)):
            step = (step,)
        if isinstance(step, tuple):
            parts = tuple(self._child_of(p, door) if not isinstance(p, (Skippable, RepeatWhile))
                          else self._node_of(p, None, door=door) for p in step)
            return Node("sequence", parts, None, any(p.grad for p in parts))
        if isinstance(step, (Skippable, RepeatWhile)):
            return self._node_of(step, None, door=door)
        if isinstance(step, Node):
            return step
        owner = getattr(step, "__self__", None)
        if owner is not None and _is_bound(owner) and step.__name__ == "launch":
            return self._node_of(owner, None, door=door)
        if not callable(step):
            raise TypeError(f"{door}: a conditional's step must be callable")
        return Node("opaque", (step, (), ()))

    def _adopt_branches(self, cond, door):
        if isinstance(cond.arms[0], _Late):
            if cond.arms[0].owner is not self:
                raise ValueError(f"{door}: this Branches already belongs to another Graph")
            lates = cond.arms
        else:
            lates = cond.arms = tuple(_Late(self._child_of(a, door), self) for a in cond.arms)
        seq = Node("sequence", tuple(late.child for late in lates), None,
                   any(late.child.grad for late in lates))
        return Node("branches", _Cond(cond, seq, lates), None, seq.grad)

    def _adopt(self, cond, door):
        if isinstance(cond, Branches):
            return self._adopt_branches(cond, door)
        late = cond.step
        if isinstance(late, _Late):
            if late.owner is not self:
                raise ValueError(f"{door}: this {type(cond).__name__} already "
                                 "belongs to another Graph")
            child = late.child
        else:
            child = self._child_of(cond.step, door, loop=isinstance(cond, RepeatWhile))
            late = cond.step = _Late(child, self)
        kind = "skippable" if isinstance(cond, Skippable) else "repeat_while"
        return Node(kind, _Cond(cond, child, late), None, child.grad)

    def _node_of(self, item, name=None, *, door):
        if isinstance(item, (Skippable, RepeatWhile, Branches)):
            return self._adopt(item, door)._named(name)
        if isinstance(item, Node):
            return item._named(name)
        if _is_bound(item):
            refuse_derived(item, door=door, what="the plan")
            copies = getattr(item, "_copies", None)
            if copies:
                raise ValueError(
                    f"{door}: plane(s) {sorted(copies)} would be COPY-bound "
                    "(non-contiguous or sample-major); a Graph needs zero-copy "
                    "planes -- pass contiguous component-major arrays "
                    "(numpy.ascontiguousarray, or eagle.samples_last(x))")
            return Node("launch", item, name, not _writes_only_cells(item))
        raise TypeError(f"{door}: expected a BoundPlan or an eagle.graph.Node, "
                        f"got {type(item).__name__}")

    def _validate(self, node):
        k = get_node_kind(node.kind)
        for _n, arr, _r in (*k.reads(node), *k.writes(node)):
            if not hasattr(arr, "shape"):
                raise TypeError(f"eagle.Graph: node {node.label!r} declares a "
                                f"non-array plane {type(arr).__name__}")

    def add(self, bound, *, name=None, grad=True) -> Graph:
        """Append one plan launch (a :class:`eagle.plan.BoundPlan`).

        ``grad=False`` marks the launch as not differentiated (a controller
        that writes a step size, say): a recorded run and
        :class:`eagle.diff.Reverse` skip it, so it needs no vjp and no tape,
        and its outputs carry no gradient back. A differentiated node may
        read what it writes only through a one-cell parameter plane;
        :meth:`build` with ``record=True`` refuses any other plane by name."""
        self._check_open("add")
        node = self._node_of(bound, name, door="eagle.Graph.add")
        if node.kind != "launch":
            raise TypeError("eagle.Graph.add: takes a BoundPlan; use add_node "
                            "for a Node")
        if not grad:
            node = Node(node.kind, node.payload, node.name, False)
        self._validate(node)
        self._nodes.append(node)
        return self

    def _add_derived(self, bound, token, *, name=None) -> Graph:
        """Private door of :mod:`eagle.diff`: a derivative plan enters only the
        reverse graph eagle.diff builds, where it is paired with its primal.
        Every public door refuses one."""
        if token is not _DERIVED_DOOR:
            raise ValueError("eagle.Graph._add_derived is eagle.diff's own door")
        self._check_open("add")
        node = Node("launch", bound, name)
        self._validate(node)
        self._nodes.append(node)
        self._is_reverse = True
        return self

    def add_node(self, node, *, name=None) -> Graph:
        """Append any registered-kind :class:`Node`, or an existing
        ``eagle.skippable`` / ``eagle.repeat_while`` object."""
        self._check_open("add_node")
        if not isinstance(node, (Node, Skippable, RepeatWhile, Branches)):
            raise TypeError("eagle.Graph.add_node: expected an eagle.graph.Node, "
                            "or an eagle.skippable / eagle.repeat_while")
        if isinstance(node, Node) and _transfer.record(node._named(name), self):
            return self       # inside a builder: kept in the order of its calls
        node = self._node_of(node, name, door="eagle.Graph.add_node")
        get_node_kind(node.kind)          # unregistered -> refusal naming the registry
        self._validate(node)
        self._nodes.append(node)
        return self

    def _group_node(self, branches, names=None, door="eagle.Graph.add_group") -> Node:
        """A fork/join group :class:`Node`: each branch a BoundPlan, a Node
        (a nested group, a conditional) or a list of them."""
        branches = list(branches)
        if len(branches) < 2:
            raise ValueError(f"{door}: a group needs at least 2 "
                             f"branches, got {len(branches)}")
        if names is not None and len(names) != len(branches):
            raise ValueError(f"{door}: names has {len(names)} "
                             f"entries but there are {len(branches)} branches")
        built = []
        for i, br in enumerate(branches):
            items = list(br) if isinstance(br, (list, tuple)) else [br]
            if not items:
                raise ValueError(f"{door}: branch {i} is empty")
            nm = None if names is None else names[i]
            nodes = [self._node_of(it, nm if len(items) == 1 else None, door=door)
                     for it in items]
            for n in nodes:
                self._validate(n)
            built.append(tuple(nodes))
        return Node("group", tuple(built))

    def add_group(self, branches, *, names=None) -> Graph:
        """Append a fork/join stage: at least two independent branches, each
        a BoundPlan, a Node, or a list of them (run in order inside the
        branch; a branch may hold a nested group or a conditional), joined
        before the next node. Siblings that overlap are refused at
        :meth:`build`."""
        self._check_open("add_group")
        self._nodes.append(self._group_node(branches, names))
        return self

    def _decision(self, cell):
        """A fresh ``uint32`` cell holding a copy of ``cell``, taken by a node
        added here: what the arms of a conditional read."""
        decision = _alloc.zeros(_place(cell), 1, dtype="uint32")
        self.add_node(_copy(cell, decision), name="decision")
        return decision

    def _if_else(self, guard, then, other=None) -> Graph:
        """Append ``then`` guarded by ``guard`` and, when given, ``other``
        guarded by its negation. The guard's count cell is copied once into a
        decision cell first, so an arm that writes its own guard cannot make
        both arms run."""
        self._check_open("if_else")
        i = guard.count_idx
        decision = self._decision(guard.count_arr[i:i + 1])
        held = SkipGuard(decision, 0, guard.baseline_arr, guard.baseline_idx, guard.negate)
        if other is not None and _array_backend(decision) == "gpu" and has_branch_nodes():
            return self.add_node(Branches((then, other), held))       # one IF/ELSE node
        self.add_node(Skippable(then, held))
        if other is not None:
            self.add_node(Skippable(other, held.negated()))
        return self

    def _chain(self, index, arms, otherwise=None) -> Graph:
        """An ``if``/``elif``/``else`` chain over ONE decision: ``index`` is a
        ``uint32`` cell holding the number of the arm to run, evaluated once
        by the caller; ``len(arms)`` runs ``otherwise``. The cell is copied
        once at the head, then each arm is an IF over ``decision == k``."""
        self._check_open("chain")
        decision = self._decision(index)
        where = _place(index)
        if _array_backend(decision) == "gpu" and has_branch_nodes(switch=True):
            return self.add_node(Branches((*arms, *([otherwise] if otherwise is not None else [])),
                                          SkipGuard(decision, 0), switch=True))     # one SWITCH node
        for k, arm in enumerate((*arms, otherwise)):
            if arm is None:
                continue
            base = _alloc.full(where, 1, k, dtype="uint32") if k else None
            self.add_node(Skippable(arm, SkipGuard(decision, 0, base, 0, True)))
        return self

    # -- build -- #
    def _leaves(self):
        """``(leaves, groups)``: the executable nodes in add order, and per
        fork/join group (nested ones too) its branches as lists of leaf
        indices; an outer branch holds the leaves of its nested groups. The
        ``wait`` points of the groups leave ``self._waited = (ordered leaf
        pairs, intra-group edges)``."""
        out, groups, ordered, edges = [], [], set(), []

        def walk(node, in_branch=False):
            if node.kind == "wait" and not in_branch:
                raise RuntimeError("eagle.Graph.wait: only inside a parallel region, on a branch handle")
            if node.kind != "group":
                out.append(_Access(node))
                return [len(out) - 1]
            rows = [[walk(n, True) for n in branch] for branch in node.payload]
            for q, k, j, m, arrays in _wait_links(node):
                src, name = rows[j][m][-1], next(n for n, w, _r in out[rows[j][m][-1]].writes
                                                 if any(arrays_overlap(w, a) for a in arrays))
                for later in rows[q][k:]:
                    if later is rows[q][k] or any(arrays_overlap(arr, a) for i in later for _n, arr, _r
                                                  in (*out[i].reads, *out[i].writes) for a in arrays):
                        edges.extend(Edge(src, i, name, "RAW") for i in later)
                ordered.update((i, o) for row in rows[j][:m + 1] for i in row
                               for later in rows[q][k:] for o in later)
            branches = [[i for ids in row for i in ids] for row in rows]
            groups.append(branches)
            return [i for ids in branches for i in ids]

        for node in self._nodes:
            walk(node)
        self._waited = (ordered, edges)
        return out, groups

    def _resolve_backend(self, accesses, backend):
        """``(backend, host)``: where the graph runs, and the ids of the
        nodes of a GPU graph that run on the host (``None`` when no plane of a
        GPU graph is on the host). A node runs where its planes
        live; a node with both is refused (a copy is the one node that spans)."""
        kinds = {device_kind(arr) for a in accesses for _n, arr, _r in (*a.reads, *a.writes)}
        gpus = kinds - {"host"}
        if len(gpus) > 1:
            raise ValueError(f"eagle.Graph.build: the planes live on {sorted(gpus)}; "
                             "a Graph runs on one backend")
        if backend is not None and backend not in _BACKENDS:
            raise ValueError(f"eagle.Graph.build: backend must be one of "
                             f"{_BACKENDS} or None, got {backend!r}")
        kind = next(iter(gpus)) if gpus else "host" if kinds else None
        if backend is None:
            be = "cpu" if kind in (None, "host") else "gpu"
        else:
            want = "host" if backend == "cpu" else gpu_kind() if backend == "gpu" else backend
            if kind is not None and kind != want:
                raise ValueError(f"eagle.Graph.build: backend={backend!r} but the "
                                 f"planes live on the {'cpu' if kind == 'host' else 'gpu'}")
            be = "cpu" if want == "host" else "gpu"
        if be != "gpu" or "host" not in kinds:
            return be, None
        host = set()
        for a in accesses:
            ks = {device_kind(arr) for _n, arr, _r in (*a.reads, *a.writes)}
            if a.node.kind in _CONDITIONALS:
                _refuse_host_in_body(a)
            elif a.node.kind != "copy" and "host" in ks:
                if len(ks) > 1:
                    _refuse_mixed(a)
                if get_node_kind(a.node.kind).tier == "capture":
                    host.add(id(a.node))
        return be, frozenset(host)

    def build(self, *, backend=None, record=False, dispatch=None, device=None) -> Graph:
        """Derive the edges, refuse conflicts (always), and lower. ``backend``
        is inferred from where the planes live; ``"cpu"``, ``"gpu"`` (the one
        installed GPU kind; two installed kinds are refused) or a kind such as
        ``"cuda"`` states it. ``record=True`` keeps what a
        reverse pass needs (:mod:`eagle.diff`): after each :meth:`launch`,
        :attr:`record` is that launch's ``Record``.

        The graph runs on the GPU its planes live on (``device=n`` is checked
        against them); planes on two GPUs are refused.

        The keywords of :class:`Graph` are the defaults of the same-named ones here.

        ``dispatch`` lowers the ``policy="auto"`` dispatch groups of the
        graph's hawk kernels (``None`` or ``"switch"``: the fused kernel;
        ``"predicated"``; or ``{group: policy}``), rebuilding those plans over
        the planes they were bound with; a plan without an ``"auto"`` group is
        not touched. A built graph is one fixed structure, so a split
        (``"segmented"``) is refused here: it is decided as a run goes, in
        :func:`eagle.simulate`. :attr:`dispatch` says what was chosen.

        Example::

            graph = eagle.Graph().add(bound_step).build(dispatch="switch")
            graph.dispatch   # {"walker": {"dispatch0": "switch"}}"""
        with self._busy("build"), contextlib.ExitStack() as pin:
            self._check_open("build")
            if not self._nodes:
                raise ValueError("eagle.Graph.build: the graph is empty")
            backend = self._defaults["backend"] if backend is None else backend
            record = record or self._defaults["record"]
            dispatch = self._defaults["dispatch"] if dispatch is None else dispatch
            device = self._defaults["device"] if device is None else device
            self._dispatch = {}
            if not self._is_reverse and (dispatch is not None or any(
                    n.kind == "launch" and getattr(getattr(n.payload.plan, "plugin", None),
                                                   "_hawk_source", None) for n in self._nodes)):
                from ._dispatch_door import relower_graph_nodes

                self._nodes, self._dispatch = relower_graph_nodes(
                    self._nodes, dispatch, door="eagle.Graph.build")
                if dispatch is not None and not self._dispatch:
                    raise ValueError(
                        "eagle.Graph.build: dispatch= picks the policy of an 'auto' dispatch "
                        "group, and no plan of this graph has one; declare one with "
                        "policy='auto' in the kernel, or drop dispatch=")
            if record and _RECORD_FACTORY is None:
                from . import diff  # noqa: F401  (installs the recorder)
            recorder = _RECORD_FACTORY(self) if record else None
            accesses, groups = self._leaves()
            check_conflicts(accesses, groups, self._waited[0])
            for inner in _inner_groups(self._nodes):    # groups inside conditional and loop bodies
                sub = Graph()
                sub._nodes = [inner]
                check_conflicts(*sub._leaves(), sub._waited[0])
            be, host = self._resolve_backend(accesses, backend)
            edges = tuple(derive_edges(accesses, groups, self._waited[1]))
            if recorder is not None:
                recorder.attach(accesses)
            before = recorder.before if recorder else (lambda stage: ())
            ctx = LowerCtx(be, host=host) if host else LowerCtx(be)
            if be == "gpu":
                self._device = _plane_device(accesses, device)
                pin.enter_context(_on_device(self._device))
                self._steps, self._host_nodes = [], []
                pin.callback(_STEP_SINK.reset, _STEP_SINK.set(self._steps))
                if host:
                    pin.callback(_transfer.HOST_SINK.reset, _transfer.HOST_SINK.set(self._host_nodes))

                from .pipeline import GraphPipeline

                top = {id(n) for n in self._nodes}
                for a in accesses:
                    if get_node_kind(a.node.kind).tier != "capture" and id(a.node) not in top:
                        raise ValueError(
                            f"eagle.Graph.build: node {a.label!r} runs on the host between "
                            "captured stages on the GPU, so it cannot run concurrently "
                            "inside add_group; place it at the top level")
                segments, stages = [], []
                for i, node in enumerate(self._nodes):
                    tier = get_node_kind(node.kind).tier
                    if tier != "capture":
                        if stages:
                            segments.append(stages)
                            stages = []
                        call = ctx.lower(node)
                        segments.append(_Eager(list(before(i)), call,
                                               call.stream if tier == "stream" else None))
                        continue
                    stages += [("step", copy) for copy in before(i)]
                    res = ctx.lower(node)
                    if isinstance(res, ForkStage):
                        stages.append(("group", res.members, res.syncs))
                    else:
                        stages.append(("step", res))
                if not segments:
                    pipe = GraphPipeline()
                    with _pinned_hint(self._nodes) if host is not None else contextlib.nullcontext():
                        pipe._capture_stages(stages)      # bumps `captures` once
                    self._pipe = pipe
                else:
                    if stages:
                        segments.append(stages)
                    self._segments = [
                        (GraphPipeline()._capture_stages(seg), _stream.Event())
                        if isinstance(seg, list) else seg for seg in segments]
                    self._seeds = []
                self._schedule = tuple(_TIERS[get_node_kind(n.kind).tier] for n in self._nodes)
            else:
                from ._host_schedule import lower_cpu_stage

                lowered = [lower_cpu_stage(n, ctx) for n in self._nodes]
                self._cpu_steps = [s for i, (step, _) in enumerate(lowered)
                                   for s in (*before(i), step)]
                self._schedule = tuple(how for _, how in lowered)
            self._live = _live_checks(accesses)
            self._backend = be
            self._edges = edges
            self._labels = [a.label for a in accesses]
            self._record_on = bool(record)
            self._recorder = recorder
            self._built = True
            self._built_with = dict(backend=backend, record=record, dispatch=dispatch, device=device)
            _counters.bump("graph_builds")
        return self

    def _rebind(self, mapping) -> Graph:
        """Swap arrays under a built graph: ``mapping`` is ``(old array, new
        array)`` pairs (arrays do not hash, so not a dict; same shape, dtype
        and device). Every plan launch, copy source and guard cell holding an
        old array is rebound in place; on the GPU the instantiated graph's
        kernel, memcpy and conditional-setter nodes are patched in one call
        each, with no re-capture. What cannot be patched (a node of your own
        code, a copy's destination, a plan of several kernels) rebuilds the
        graph, with one ``PerformanceWarning`` per graph."""
        self._need_built("_rebind")
        if self._recorder is not None:
            raise ValueError("eagle.Graph.bind: a record=True graph is bound by the builder again")
        patch = self._patch or self._patch_index()
        todo, rebuild = [], False
        for old, new in mapping:
            if (old.shape, old.dtype, _array_backend(old)) != (new.shape, new.dtype, _array_backend(new)) \
                    or (_array_backend(new) == "gpu" and old.device.id != new.device.id):
                raise ValueError(
                    f"eagle.Graph.bind: the new array is {new.shape} {new.dtype} where the graph was "
                    f"built over {old.shape} {old.dtype}; call the builder again")
            if id(old) in patch.own:
                raise ValueError("eagle.Graph.bind: a node runs your own code over an array to swap; "
                                 "call the builder again")
            todo.append((old, new, patch.index.pop(id(old), ())))
        bounds, wrappers = {}, {}
        for old, new, uses in todo:
            patch.index[id(new)] = uses
            for kind, obj, name in uses:
                if kind == "plan":
                    swap = getattr(obj, "_swap", None)
                    swap(name, new) if swap else obj.rebind(**{name: new})
                    bounds[id(obj)] = obj
                elif kind == "src":
                    obj.payload[0] = new
                elif kind == "dst":
                    obj.payload[1] = new
                    rebuild = True
                else:
                    setattr(obj[0], obj[1], new)
                    wrappers[id(name)] = name
        if self._backend == "gpu":
            self._patch_or_rebuild(bounds.values(), todo, wrappers.values(), rebuild)
        return self

    def _patch_index(self):
        """Where each array of the graph is used (built once per build): plan
        planes, copy ends, guard cells, and what only your own code holds."""
        from ._graph_kinds import _Patch

        patch = self._patch = _Patch()
        for node in _walk(self._nodes):
            if node.kind == "launch":
                for name, arr in node.payload._planes.items():
                    if hasattr(arr, "shape"):
                        patch.index.setdefault(id(arr), []).append(("plan", node.payload, name))
            elif node.kind == "copy":
                patch.index.setdefault(id(node.payload[0]), []).append(("src", node, None))
                patch.index.setdefault(id(node.payload[1]), []).append(("dst", node, None))
            elif node.kind in ("skippable", "repeat_while", "branches"):
                wrapper = node.payload.wrapper
                guard = wrapper.guard
                for obj, attr in (((guard, "count_arr"), (guard, "baseline_arr")) if guard is not None else ()) + (
                        (wrapper, "limit"), (wrapper, "brk")):
                    arr = getattr(obj, attr, None)
                    if arr is not None:
                        patch.index.setdefault(id(arr), []).append(("attr", (obj, attr), wrapper))
            elif node.kind not in ("group", "sequence"):
                for _n, arr, _r in (*get_node_kind(node.kind).reads(node), *get_node_kind(node.kind).writes(node)):
                    patch.own.add(id(arr))
        return patch

    def _patch_or_rebuild(self, bounds, todo, wrappers, rebuild):
        """Patch the instantiated graph's nodes after a rebind (one native
        call per node kind); re-capture it when a node cannot be patched."""
        steps = self._patch.steps
        pipe, launcher = self._pipe, getattr(self._pipe, "_launcher", None)
        try:
            if rebuild or pipe is None or launcher is None:
                raise LookupError
            if steps is None:
                steps = self._patch.steps = {id(s.key): s for s in self._steps}
            nodes, params = [], []
            for b in bounds:
                step_nodes = steps[id(b)].nodes
                if len(step_nodes) != 1:
                    raise LookupError
                nodes.append(step_nodes[0])
                params.append(b._addrs)
            copies = [steps[id(u[1].payload)] for _o, _n, uses in todo for u in uses if u[0] == "src"]
            if any(len(s.nodes) != 1 for s in copies):
                raise LookupError
            with _on_device(self._device):
                launcher.set_kernel_params_many(nodes, params)
                launcher.set_memcpy_source_many([s.nodes[0] for s in copies],
                                                [_address(s.key[0])[0] for s in copies])
                for wrapper in wrappers:
                    pipe._patch_setters(launcher, wrapper)
        except (LookupError, ValueError, RuntimeError, KeyError, AttributeError):
            self._rebuild()

    def _rebuild(self):
        if not self._rebuilt:
            from ._dispatch_door import PerformanceWarning

            self._rebuilt = True
            warnings.warn("eagle.Graph.bind: the swap could not be patched into the instantiated "
                          "graph, so it was captured again", PerformanceWarning, stacklevel=4)
        keep = self._built_with
        self._patch = None
        self._built, self._pipe, self._segments, self._tail_ev = False, None, None, None
        self.build(**keep)

    # -- run -- #
    def _need_built(self, method):
        if not self._built:
            raise RuntimeError(f"eagle.Graph.{method}: call build() first")

    def _check_live(self):
        """Refuse a launch whose framework graph was reset or whose bound
        weights moved (one handle or pointer read each)."""
        for check in self._live:
            check()

    def launch(self, n: int = 1) -> Graph:
        """Run the graph ``n`` times and wait (``enqueue(n)`` then
        :meth:`synchronize`). A ``record=True`` graph runs once per launch and
        keeps that run's ``Record`` in :attr:`record`."""
        with self._busy("launch"), _on_device(self._device):
            self._need_built("launch")
            if self._device is not None:
                refuse_in_capture("eagle.Graph.launch")
            self._check_live()
            if self._recorder is not None:
                self._recorder.begin(n, "launch")
            if self._segments is not None:
                self._run_segments(n)
                self._sync_tail()
            elif self._backend == "gpu":
                self._pipe.launch(n)
            else:
                for _ in range(n):
                    for step in self._cpu_steps:
                        step()
            if self._host_nodes:
                _transfer.raise_host_errors(self._host_nodes)
            if self._recorder is not None:
                self._record = self._recorder.finish()
            self._scatter_status(wait=True)
        return self

    def enqueue(self, n: int = 1) -> Graph:
        """Enqueue ``n`` runs without waiting (CPU: same as :meth:`launch`).
        A graph with a stream node stays non-blocking (each stage is fenced on
        the device); one with a host node waits as :meth:`launch` does. As for
        :class:`eagle.GraphPipeline`, write no plane of the graph between
        ``enqueue`` and :meth:`synchronize`."""
        with self._busy("enqueue"), _on_device(self._device):
            self._need_built("enqueue")
            if self._device is not None:
                refuse_in_capture("eagle.Graph.enqueue")
            self._check_live()
            if self._recorder is not None:
                raise RuntimeError(
                    "eagle.Graph.enqueue: a record=True graph runs with launch(), "
                    "which waits for the run its Record describes")
            if self._segments is not None:
                self._run_segments(n)
            elif self._backend == "gpu":
                self._pipe.enqueue(n)
            else:
                for _ in range(n):
                    for step in self._cpu_steps:
                        step()
            self._scatter_status(wait=False)
        return self

    def _scatter_status(self, *, wait: bool) -> None:
        """A segmented GPU graph replays through captures of its own: its split binds are marked pending on
        the stream its tail event closes. (A plain GPU graph is one pipeline, which marks its own.)"""
        if self._segments is None or self._tail_ev is None:
            return
        for node in _walk(self._nodes):
            if node.kind == "launch" and hasattr(node.payload, "_status"):
                _split.mark_pending(node.payload, self._tail_ev._stream)
        if wait:
            _split.run_pending(self._tail_ev._stream)

    def update(self, values=None, /, **kw) -> Graph:
        """Change parameters of a ``record=True`` graph between launches:
        ``g.update(k=0.3)`` sets every node's ``k``, ``g.update({"join.k":
        0.3})`` one node's. A recorded graph holds its parameters in one-cell
        planes, so this is a copy into them: nothing is rebuilt or recaptured,
        forward or reverse. Refused while the last launch's Record is live
        (reverse it, or ``keep()`` it, first)."""
        with self._busy("update"):
            self._need_built("update")
            if self._recorder is None:
                raise ValueError(
                    "eagle.Graph.update: only a record=True graph holds its parameters "
                    "in planes; for this graph rebind the plan (bound.rebind) and build "
                    "the graph again")
            self._recorder.update({**(values or {}), **kw})
            _counters.bump("updates")
        return self

    def synchronize(self) -> Graph:
        """Block until enqueued runs complete (CPU: a no-op). An exception a host
        node raised in those runs is raised here."""
        self._need_built("synchronize")
        if self._backend == "gpu":
            with _on_device(self._device):
                if self._segments is None:
                    self._pipe.synchronize()
                else:
                    self._sync_tail()
            if self._host_nodes:
                _transfer.raise_host_errors(self._host_nodes)
            if self._segments is not None and self._tail_ev is not None:
                _split.run_pending(self._tail_ev._stream)
        return self

    def _sync_tail(self):
        tail = self._tail_ev
        if tail is not None:
            tail.synchronize()

    def _run_segments(self, n):
        """A segmented GPU graph, fenced on the device: the first stage of a
        run waits the caller's streams and the previous run's tail; each
        captured segment and stream node waits the stage before it; a host
        node syncs the device before and after."""
        tail = self._tail_ev
        for _ in range(n):
            prev, first = None, True
            for seg in self._segments:
                waits = _run_seeds(self._seeds, tail) if first else (
                    [] if prev is None else [prev])
                if isinstance(seg, _Eager):
                    prev = seg.run(waits)
                else:
                    pipe, event = seg
                    pipe.enqueue(1, seed_events=waits if first or prev is not None else None)
                    event.record(pipe._sptr)
                    prev = event
                first = False
            tail = prev
        self._tail_ev = tail

    # -- introspection -- #
    def node_labels(self) -> list:
        """Labels (the node's name, else its kind) of every executable node,
        in add order, the members of a group included."""
        return list(self._labels) if self._built else [a.label for a in self._leaves()[0]]

    def num_nodes(self) -> int:
        return len(self.node_labels())

    def edges(self) -> tuple:
        """``tuple[Edge]``: ``Edge(src, dst, plane, kind)`` with ``kind`` in
        ``"RAW" | "WAR" | "WAW"``, indices into :meth:`node_labels`."""
        if self._built:
            return self._edges
        leaves, groups = self._leaves()
        return tuple(derive_edges(leaves, groups, self._waited[1]))

    @property
    def schedule(self) -> tuple:
        """Per stage: on the CPU ``"work_share"`` | ``"branch_tasks"``; on the
        GPU ``"capture"`` | ``"eager_stream"`` | ``"eager_host"``; empty before
        :meth:`build`."""
        return self._schedule

    @property
    def backend(self):
        """``"gpu"``, ``"cpu"``, or ``None`` before :meth:`build`."""
        return self._backend

    @property
    def record(self):
        """The ``Record`` of the last launch of a ``record=True`` graph (what
        :func:`eagle.diff.reverse` takes), else ``None``."""
        return self._record

    # -- seeing the graph (never builds, launches or takes the Busy guard) -- #
    def show(self, *, file=None) -> None:
        """Print the graph as a box block diagram (:func:`str` returns the
        same text): nodes by stage, the planes between them, conditionals as
        frames, opaque nodes dashed, the schedule per stage after
        :meth:`build`. Works on the graph so far; never builds or launches."""
        print(str(self), file=file)

    def __str__(self) -> str:
        return _graph_view.text(self)

    def to_mermaid(self) -> str:
        """The graph as Mermaid ``flowchart TD`` text."""
        return _graph_view.description(self).mermaid()

    def to_dot(self) -> str:
        """The graph as Graphviz DOT text."""
        return _graph_view.description(self).dot()

    def plot(self, ax=None, *, edge_labels: bool = True, show_schedule: bool = True):
        """Draw the graph with matplotlib (optional extra: ``pip install
        raptor-eagle[plot]``) at the coordinates of the same layout
        :meth:`show` uses; returns the ``Figure``. ``ax`` draws into an
        existing axes. A figure made here is a plain ``matplotlib.figure.Figure``
        (no pyplot state), so it is safe on a headless machine and from any
        thread."""
        return _graph_view.plot(self, ax, edge_labels, show_schedule)

    def __repr__(self) -> str:
        return (f"<eagle.Graph {len(self._leaves()[0])} nodes, {len(self._nodes)} stages, "
                f"{self._backend if self._built else 'unbuilt'}>")

    # -- the builder surface (eagle._builder) -- #
    @classmethod
    def builder(cls, fn):
        """Decorate ``fn``: a parameter annotated ``eagle.Graph`` makes a
        block that appends to the graph it is given; none makes a builder
        that makes, records into and returns its Graph (calling it on arrays
        builds that Graph; nothing runs). Control flow on 1-cell uint32 arrays
        becomes device control flow."""
        from ._builder import make_builder

        return make_builder(fn)

    def parallel(self):
        """``with g.parallel() as p:`` opens a fork in the builder; leaving
        the ``with`` joins. ``p.branch()`` yields a branch handle."""
        from ._builder import parallel

        return parallel(self)

    def wait(self, *planes):
        raise RuntimeError("eagle.Graph.wait: only inside a parallel region, on a branch handle")

    @property
    def result(self):
        """The arrays this graph returns: set while building (an array, or a
        tuple / list / dict of arrays), read-only after."""
        return self._bld.result

    @result.setter
    def result(self, tree):
        from ._builder import tree_leaves

        if self._built:
            raise RuntimeError("eagle.Graph.result: declared while building only")
        tree_leaves(tree)
        self._bld.result = tree

    def __call__(self, n: int = 1, /):
        """Replay ``n`` times and wait; returns :attr:`result` (``None``
        without one)."""
        self.launch(n)
        return self._bld.result

    def bind(self, **planes) -> Graph:
        """New arrays for the builder's parameters, by name, in place: same
        shape, dtype and device, no copy, no rebuild."""
        from ._builder import bind

        return bind(self, planes)


def _live_checks(accesses):
    """The per-launch checks of a built graph: each node kind's ``check``
    (a conditional's step included) and each bound plan's weight pointers."""
    out = []
    for a in accesses:
        node = a.node
        if node.kind in ("skippable", "repeat_while"):
            node = node.payload.child
        check = get_node_kind(node.kind).check
        if check is not None:
            out.append(lambda c=check, nd=node: c(nd))
        if node.kind == "launch" and node.payload._origins:
            out.append(lambda b=node.payload: b._check_origins())
    return tuple(out)


# the built-in kinds register themselves on import; Graph needs a few of their helpers
from ._graph_kinds import (_STEP_SINK, _Cond, _copy, _Late, _is_bound, _wait_links,  # noqa: E402
                           child, opaque)  # noqa: E402,F401
from . import _transfer  # noqa: E402
from . import _stream  # noqa: E402
from .pipeline import refuse_in_capture  # noqa: E402
