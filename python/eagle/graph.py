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
from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable

from . import _counters, _graph_view
from ._busy import BusyGuard
from ._conditional import RepeatWhile, Skippable
from ._graph_edges import (SHARES_MEMORY_MAX_WORK, Edge, check_conflicts,
                           derive_edges)
from .sidecar import refuse_derived

__all__ = ["Edge", "Graph", "Node", "NodeKind", "SHARES_MEMORY_MAX_WORK",
           "child", "get_node_kind", "opaque", "register_node_kind"]

_BACKENDS = ("gpu", "cpu")
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

    __slots__ = ("members",)

    def __init__(self, members):
        self.members = list(members)


@dataclass(frozen=True)
class LowerCtx:
    """What a lowering hook receives: the ``backend`` (``"gpu"``/``"cpu"``),
    whether the node is lowered into a conditional or loop ``body``, and
    :meth:`lower`, which lowers a child node for that backend."""

    backend: str
    body: bool = False

    def lower(self, node: Node):
        k = get_node_kind(node.kind)
        return (k.lower_gpu if self.backend == "gpu" else k.lower_cpu)(node, self)


class _Eager:
    """A segment of a GPU graph that runs on the host each launch: the
    recorder's ``copies`` then ``call``, on ``stream`` (a stream node) or
    between two device syncs (``stream is None``, a host node)."""

    __slots__ = ("copies", "call", "stream", "event")

    def __init__(self, copies, call, stream):
        import cupy

        self.copies, self.call = copies, call
        self.stream = self.event = None
        if stream is not None:
            self.stream = _wrap_external_stream(stream)
            self.event = cupy.cuda.Event(disable_timing=True)

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
        with self.stream:
            for copy in self.copies:
                copy()
            self.call()
            self.event.record(self.stream)
        return self.event


def _device_sync():
    """A host node's fence: every stream of the device, whichever framework
    issued on it, is done (this also waits other threads' device work)."""
    import cupy

    cupy.cuda.Device().synchronize()


def _wait_stream(stream, events):
    for event in events:
        stream.wait_event(event)


def _run_seeds(cache, tail):
    """What the first stage of a run waits on: the caller's current stream,
    the legacy stream (``cache`` holds the two re-recorded events) and the
    previous run's ``tail`` event (run-to-run WAR/WAW)."""
    import cupy

    current = cupy.cuda.get_current_stream()
    cache[0].record(current)
    events = [cache[0]]
    if current.ptr != cupy.cuda.Stream.null.ptr:
        cache[1].record(cupy.cuda.Stream.null)
        events.append(cache[1])
    if tail is not None:
        events.append(tail)
    return events


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
    import cupy

    with cupy.cuda.Device(device):
        cupy.cuda.runtime.setDevice(device)   # Device() skips this when the id already matches
        yield


def _plane_device(accesses):
    """The CUDA ordinal the graph's planes live on (the first one's)."""
    for a in accesses:
        for _n, arr, _r in (*a.reads, *a.writes):
            return arr.device.id
    return None


def _array_backend(arr):
    return "gpu" if type(arr).__module__.split(".")[0] == "cupy" else "cpu"


class Graph:
    """A graph of launches: build it once, launch it many times.

    ``add``/``add_group``/``add_node`` append (chaining); ``build`` derives
    the edges, refuses conflicts, and lowers to the GPU or CPU; ``launch`` /
    ``enqueue`` / ``synchronize`` are :class:`eagle.GraphPipeline`'s verbs
    (on the CPU ``enqueue`` is ``launch``). ``show()`` / ``str()`` /
    ``plot()`` / ``to_mermaid()`` / ``to_dot()`` draw it."""

    def __init__(self):
        self._nodes: list = []
        self._busy = BusyGuard("eagle.Graph")
        self._built = False
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
        """The Node a conditional's step stands for: a Node itself, a
        ``BoundPlan.launch`` (its planes are then derived), or any other
        zero-arg callable (opaque: no declared planes). A repeat_while
        (``loop``) also takes a tuple of such parts, run in order, and a
        ``skippable`` part: a ``"sequence"`` node."""
        if isinstance(step, RepeatWhile) and loop:
            raise ValueError(f"{door}: nesting a repeat_while inside a "
                             "repeat_while is not supported; flatten it")
        if isinstance(step, Skippable) and loop:
            step = (step,)
        if isinstance(step, (Skippable, RepeatWhile)):
            raise ValueError(f"{door}: nesting a conditional inside a "
                             "conditional is not supported; flatten it")
        if isinstance(step, tuple):
            if not loop:
                raise ValueError(f"{door}: only a repeat_while takes a tuple of "
                                 "body parts")
            parts = tuple(self._node_of(p, None, door=door) if isinstance(p, Skippable)
                          else self._child_of(p, door, loop=True) for p in step)
            if any(p.kind == "repeat_while" for p in parts):
                raise ValueError(f"{door}: nesting a repeat_while inside a "
                                 "repeat_while is not supported; flatten it")
            return Node("sequence", parts, None, all(p.grad for p in parts))
        if isinstance(step, Node):
            if step.kind in ("group",):
                raise ValueError(f"{door}: a group cannot be a conditional's step")
            return step
        owner = getattr(step, "__self__", None)
        if owner is not None and _is_bound(owner) and step.__name__ == "launch":
            return self._node_of(owner, None, door=door)
        if not callable(step):
            raise TypeError(f"{door}: a conditional's step must be callable")
        return Node("opaque", (step, (), ()))

    def _adopt(self, cond, door):
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
        if isinstance(item, (Skippable, RepeatWhile)):
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
            return Node("launch", item, name)
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
        if not isinstance(node, (Node, Skippable, RepeatWhile)):
            raise TypeError("eagle.Graph.add_node: expected an eagle.graph.Node, "
                            "or an eagle.skippable / eagle.repeat_while")
        node = self._node_of(node, name, door="eagle.Graph.add_node")
        get_node_kind(node.kind)          # unregistered -> refusal naming the registry
        self._validate(node)
        self._nodes.append(node)
        return self

    def add_group(self, branches, *, names=None) -> Graph:
        """Append a fork/join stage: at least two independent branches, each
        a BoundPlan, a Node, or a list of them (run in order inside the
        branch), joined before the next node. Siblings that overlap are
        refused at :meth:`build`."""
        self._check_open("add_group")
        branches = list(branches)
        if len(branches) < 2:
            raise ValueError("eagle.Graph.add_group: a group needs at least 2 "
                             f"branches, got {len(branches)}")
        if names is not None and len(names) != len(branches):
            raise ValueError(f"eagle.Graph.add_group: names has {len(names)} "
                             f"entries but there are {len(branches)} branches")
        built = []
        for i, br in enumerate(branches):
            items = list(br) if isinstance(br, (list, tuple)) else [br]
            if not items:
                raise ValueError(f"eagle.Graph.add_group: branch {i} is empty")
            nm = None if names is None else names[i]
            nodes = []
            for it in items:
                n = self._node_of(it, nm if len(items) == 1 else None,
                                  door="eagle.Graph.add_group")
                if n.kind == "group":
                    raise ValueError("eagle.Graph.add_group: groups do not nest "
                                     "(series-parallel stages only)")
                self._validate(n)
                nodes.append(n)
            built.append(tuple(nodes))
        self._nodes.append(Node("group", tuple(built)))
        return self

    # -- build -- #
    def _leaves(self):
        """``(leaves, groups)``: the executable nodes in add order, and per
        fork/join group its branches as lists of leaf indices."""
        out, groups = [], []
        for node in self._nodes:
            if node.kind == "group":
                branches = []
                for branch in node.payload:
                    ids = []
                    for n in branch:
                        ids.append(len(out))
                        out.append(_Access(n))
                    branches.append(ids)
                groups.append(branches)
            else:
                out.append(_Access(node))
        return out, groups

    def _resolve_backend(self, accesses, backend):
        seen = set()
        for a in accesses:
            for _n, arr, _r in (*a.reads, *a.writes):
                seen.add(_array_backend(arr))
        if len(seen) > 1:
            raise ValueError("eagle.Graph.build: the graph mixes host and device "
                             "planes; a Graph runs on one backend")
        inferred = seen.pop() if seen else "cpu"
        if backend is None:
            return inferred
        if backend not in _BACKENDS:
            raise ValueError(f"eagle.Graph.build: backend must be one of "
                             f"{_BACKENDS} or None, got {backend!r}")
        if backend != inferred and any(a.reads or a.writes for a in accesses):
            raise ValueError(f"eagle.Graph.build: backend={backend!r} but the "
                             f"planes live on the {inferred}")
        return backend

    def build(self, *, backend=None, record=False, dispatch=None) -> Graph:
        """Derive the edges, refuse conflicts (always), and lower. ``backend``
        is inferred from where the planes live. ``record=True`` keeps what a
        reverse pass needs (:mod:`eagle.diff`): after each :meth:`launch`,
        :attr:`record` is that launch's ``Record``.

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
            check_conflicts(accesses, groups)
            be = self._resolve_backend(accesses, backend)
            edges = tuple(derive_edges(accesses, groups))
            if recorder is not None:
                recorder.attach(accesses)
            before = recorder.before if recorder else (lambda stage: ())
            ctx = LowerCtx(be)
            if be == "gpu":
                import cupy

                self._device = _plane_device(accesses)
                pin.enter_context(_on_device(self._device))

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
                        stages.append(("group", res.members))
                    else:
                        stages.append(("step", res))
                if not segments:
                    pipe = GraphPipeline()
                    pipe._capture_stages(stages)      # bumps `captures` once
                    self._pipe = pipe
                else:
                    if stages:
                        segments.append(stages)
                    self._segments = [
                        (GraphPipeline()._capture_stages(seg), cupy.cuda.Event(disable_timing=True))
                        if isinstance(seg, list) else seg for seg in segments]
                    self._seeds = [cupy.cuda.Event(disable_timing=True) for _ in range(2)]
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
            _counters.bump("graph_builds")
        return self

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
            if self._recorder is not None:
                self._record = self._recorder.finish()
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
        return self

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
        """Block until enqueued runs complete (CPU: a no-op)."""
        self._need_built("synchronize")
        if self._backend == "gpu":
            with _on_device(self._device):
                if self._segments is None:
                    self._pipe.synchronize()
                else:
                    self._sync_tail()
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
                    event.record(pipe._ext)
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
        return tuple(derive_edges(leaves, groups))

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
from ._graph_kinds import _Child, _Cond, _Late, _is_bound, child, opaque  # noqa: E402,F401
from .interop import _wrap_external_stream  # noqa: E402
from .pipeline import refuse_in_capture  # noqa: E402
