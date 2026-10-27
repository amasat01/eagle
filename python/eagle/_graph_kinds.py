# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The built-in node kinds of :class:`eagle.graph.Graph`: a plan launch, an
opaque callable with declared planes, a copy, a cell setter, a fork/join
group, a sequence, and the existing
``eagle.skippable`` / ``eagle.repeat_while``. Each registers itself with the
node-kind registry; a new kind lives in a module like this one and never
touches ``graph.py``."""

from __future__ import annotations

import contextvars

from . import _alloc
from ._conditional import RepeatWhile
from .graph import ForkStage, LowerCtx, Node, get_node_kind, register_node_kind
from ._graph_edges import arrays_overlap
from . import _core, _transfer
from .interop import _stream_handle
from .roles import OUTPUT_ROLES, STATE_ROLES

# --- Built-in kind: launch (a BoundPlan) --- #
def _is_bound(obj) -> bool:
    return hasattr(obj, "_arg_spec") and hasattr(obj, "launch") and hasattr(obj, "_planes")


def _launch_access(node, *, writing: bool):
    bound = node.payload
    out = []
    for role, name in bound._arg_spec:
        if role in ("uniform", "nsamples") or name not in bound._planes:
            continue
        arr = bound._planes[name]
        if writing:
            if role in OUTPUT_ROLES:
                out.append((name, arr, role))
        elif role not in OUTPUT_ROLES or role in STATE_ROLES:
            out.append((name, arr, role))
    return tuple(out)


def _launch_step(node, ctx):
    bound = node.payload

    def step():
        bound.launch()
    return step


#: Where a capturing build collects the steps it lowers (see :class:`_Captured`).
_STEP_SINK = contextvars.ContextVar("eagle_graph_step_sink", default=None)


class _Captured:
    """A captured step that remembers which graph nodes (handles of the
    capture graph, root or body) it contributed: what an in-place rebind
    patches. ``key`` is the plan or copy payload it stands for. Only the GPU
    lowering makes one, so the host path pays nothing."""

    __slots__ = ("key", "run", "nodes")

    def __init__(self, key, run):
        self.key, self.run, self.nodes = key, run, ()

    def __call__(self):
        import cupy

        stream = cupy.cuda.get_current_stream().ptr
        before = _core.capture_snapshot_nodes(stream)
        self.run()
        self.nodes = sorted(set(_core.capture_snapshot_nodes(stream)) - set(before))


class _Patch:
    """What ``Graph._rebind`` needs: array id -> its uses, the arrays only
    your own code holds, and the captured step of each plan or copy."""

    __slots__ = ("index", "own", "steps")

    def __init__(self):
        self.index, self.own, self.steps = {}, set(), None


def _capturing(step):
    sink = _STEP_SINK.get()
    if sink is not None:
        sink.append(step)
    return step


def _launch_gpu(node, ctx):
    return _capturing(_Captured(node.payload, node.payload.launch))


def _launch_native(node, ctx):
    b = node.payload
    if b._cupy is not None or getattr(b, "_stages", None):
        return None
    if b._copies or b._active_count is not None or b._fold is not None:
        return None
    return b._entry, b._addrs, b._parts


def _launch_transpose(node, incoming, ctx):
    """A plan launch's reverse (:mod:`eagle.diff`): its vjp kernel bound to the
    recorded values of the planes it read, the incoming adjoints of the planes
    it wrote, and fresh adjoint planes for everything it read."""
    import numpy

    from . import _cell_rows, _split

    bound = node.payload
    role_of = {name: role for role, name in bound._arg_spec}
    xp = bound._cupy or numpy
    vjp = ctx.vjp.plugin
    cells, rowset = _cell_rows.cells_of(vjp), _cell_rows.row_planes(vjp)
    if _split.stages_of(vjp):
        # a split vjp owns its rows and finishes each gradient cell itself
        cells, rowset, finished = {}, frozenset(), frozenset(_split.cells(vjp))
    else:
        finished = frozenset()
    planes, out = {}, {}
    owned = _cell_rows.companions_of(vjp)
    for role, name in _split.arg_spec(vjp):
        src = name[4:] if name.startswith("bar_") else None
        if role == "nsamples" or name in owned:     # a keyed sum's companion is the bind's own
            continue
        if name in finished:
            planes[name] = out[src] = _alloc.zeros(xp, 1, dtype=bound._dtype)
        elif name in role_of:
            planes[name] = ctx.primal(name)
        elif src in incoming:
            planes[name] = incoming[src]
        elif name in rowset:
            # a parameter's gradient arrives as rows of 32-sample partial sums
            # (and their companions); the reverse finishes them into the cell
            planes[name] = buf = _alloc.zeros(xp, -(-bound.n // _cell_rows.ROW), dtype=bound._dtype)
            if name in cells:
                out[src] = buf
            else:
                ctx.comp[id(planes[name[:-len("_comp")]])] = buf
        elif src in role_of and role_of[src] not in OUTPUT_ROLES:
            # a parameter cell's gradient comes back per sample; the reverse sums it
            per_sample = role_of[src] == "lookup" and role == "mutable"
            buf = (_alloc.empty(xp, bound.n, dtype=bound._dtype) if per_sample
                   else _alloc.empty_like(xp, bound._planes[src]))
            if role == "accum_out":     # a scatter-add accumulates: start from zero
                ctx.add(lambda b=buf: b.fill(0), writes=[buf])
                ctx.not_bitwise(src)
            planes[name] = out[src] = buf
        else:
            raise ValueError(
                f"eagle.diff.Reverse: the vjp of node {ctx.label!r} takes plane "
                f"{name!r}, which node {ctx.label!r} does not bind")
    ctx.add(ctx.vjp.bind(**planes))
    return out


register_node_kind(
    "launch",
    reads=lambda n: _launch_access(n, writing=False),
    writes=lambda n: _launch_access(n, writing=True),
    lower_gpu=_launch_gpu, lower_cpu=_launch_step, native_entry=_launch_native,
    transpose=_launch_transpose)


# --- Built-in kinds: opaque (a callable with declared planes), child --- #
#: The kinds that run the reader's own code; a reverse through one needs its vjp.
OWN_CODE_KINDS = ("opaque", "opaque_stream", "opaque_host", "child")


class _OnStream:
    """A stream node's GPU lowering: the call, and the stream its work is on."""

    __slots__ = ("call", "stream")

    def __init__(self, call, stream):
        self.call, self.stream = call, stream

    def __call__(self):
        self.call()


def _named_arrays(arrays, prefix):
    if isinstance(arrays, dict):
        return tuple((str(k), v, "declared") for k, v in arrays.items())
    return tuple((f"{prefix}{i}", a, "declared") for i, a in enumerate(arrays))


def opaque(fn, *, reads=(), writes=(), eager=False, stream=None, name=None,
           grad=True) -> Node:
    """Your own code as a graph node: a zero-arg callable (a cupy custom
    kernel, a torch op, a host closure) with the arrays it touches DECLARED
    -- a sequence of arrays, or a ``{name: array}`` mapping. Edges are derived
    from those arrays (the declaration says WHICH planes, never which edges)
    and the conflict check covers them; name every plane the node touches,
    since an array it touches but you did not declare is invisible to the
    conflict check and the reverse.

    How it runs on the GPU (the CPU always calls it eagerly, in order):

    * default -- **captured**: ``fn`` issues its device work on the current
      stream at build and is replayed with the graph (also inside groups,
      conditionals and ``until_done`` steps). It must not allocate.
    * ``stream=S`` -- **stream-ordered**: ``fn`` runs on the host each
      launch with ``S`` (an int, a torch/Warp/CuPy stream or a
      ``__cuda_stream__`` object) as cupy's current stream; issue its device
      work on ``S`` (``torch.cuda.stream(S)`` for torch). ``S`` waits for the
      stage before and the stage after waits for ``S``; the host never waits.
      Keep what that work reads alive until it ran (torch:
      ``tensor.record_stream(S)``).
    * ``eager=True`` -- **host-synchronized**: ``fn`` runs on the host each
      launch between two device syncs, on any stream, host readbacks
      allowed. The sync waits for every stream of the device, other
      threads' work included.

    ``g.schedule`` names the tier per stage (``capture`` / ``eager_stream`` /
    ``eager_host``) and ``str(g)`` tags the node. A reverse pass through it
    needs ``vjps={label: fn}`` (:func:`eagle.diff.reverse`): the label is
    ``name`` when given (``Graph.add(name=)`` wins over it), the kind otherwise.
    ``grad=False`` marks the node as not differentiated (see
    :meth:`eagle.Graph.add`): a reverse skips it and needs no vjp for it."""
    if not callable(fn):
        raise TypeError("eagle.graph.opaque: fn must be a zero-arg callable")
    planes = (fn, _named_arrays(reads, "in"), _named_arrays(writes, "out"))
    node = (Node("opaque_stream", (*planes, stream, _stream_handle(stream)), name, grad)
            if stream is not None else Node("opaque_host" if eager else "opaque", planes, name, grad))
    _transfer.record(node)
    return node


def _opaque_transpose(node, incoming, ctx):
    """The reverse of a node eagle cannot see into: the reader's vjp. A
    callable ``vjp(primals, cotangents) -> {read name: gradient}`` runs
    host-synchronized; a Node runs at its own tier, its declared reads named
    like this node's writes taking the cotangents and its declared writes
    named like this node's reads giving the gradients."""
    vjp = ctx.vjp                   # eagle.diff.Reverse refused a node without one
    _fn, reads, writes = node.payload[:3]
    floats = lambda planes: [(nm, a) for nm, a, _r in planes  # noqa: E731
                             if str(getattr(a, "dtype", "")) in ("float32", "float64")]
    cots = {nm: incoming[nm] for nm, _a in floats(writes) if nm in incoming}
    if isinstance(vjp, Node):
        kind = get_node_kind(vjp.kind)
        takes = {nm: a for nm, a, _r in kind.reads(vjp)}
        gives = {nm: a for nm, a, _r in kind.writes(vjp)}
        for nm, cot in cots.items():
            if nm in takes:
                ctx.add(lambda d=takes[nm], c=cot: d.__setitem__(Ellipsis, c),
                        reads=[cot], writes=[takes[nm]])
        ctx.add(vjp)
        return {nm: gives[nm] for nm, _a in floats(reads) if nm in gives}
    primals = {nm: ctx.primal(nm) for nm, _a in floats(reads)}
    out = {nm: ctx.empty_like(a) for nm, a in primals.items()}

    def step():
        got = vjp(dict(primals), dict(cots))
        for nm, buf in out.items():
            if nm not in got:
                raise ValueError(f"eagle.diff.reverse: vjps[{ctx.label!r}] returned no "
                                 f"gradient for {nm!r} (it returned {sorted(got)})")
            buf[...] = got[nm]
    ctx.add(Node("opaque_host", (step, _named_arrays([*primals.values(), *cots.values()], "in"),
                                 _named_arrays(list(out.values()), "out"))))
    return out


def _declared(i):
    return lambda n: n.payload[i]


register_node_kind("opaque", reads=_declared(1), writes=_declared(2),
                   lower_gpu=lambda n, ctx: n.payload[0], lower_cpu=lambda n, ctx: n.payload[0],
                   transpose=_opaque_transpose)
register_node_kind("opaque_stream", reads=_declared(1), writes=_declared(2), tier="stream",
                   lower_gpu=lambda n, ctx: _OnStream(n.payload[0], n.payload[4]),
                   lower_cpu=lambda n, ctx: n.payload[0], transpose=_opaque_transpose)
register_node_kind("opaque_host", reads=_declared(1), writes=_declared(2), tier="host",
                   lower_gpu=lambda n, ctx: n.payload[0], lower_cpu=lambda n, ctx: n.payload[0],
                   transpose=_opaque_transpose)


# --- Built-in kinds: copy (a seed or tape copy), cell (a guard or decision cell) --- #
def _copy(src, dst, name=None) -> Node:
    """``dst[...] = src`` as a node: it reads ``src``, writes ``dst``; the
    reverse hands the cotangent of ``dst`` to ``src``."""
    return Node("copy", [src, dst], name)      # a list: a rebind swaps the arrays in place


def _copy_transpose(node, incoming, ctx):
    src = node.payload[0]
    if str(src.dtype) not in ("float32", "float64") or "dst" not in incoming:
        return {}
    cot = incoming["dst"]
    buf = ctx.empty_like(src)           # lives where src does: a transfer's reverse is the copy back
    ctx.add(_copy(cot, buf))
    return {"src": buf}


register_node_kind("copy", reads=lambda n: (("src", n.payload[0], "declared"),),
                   writes=lambda n: (("dst", n.payload[1], "declared"),),
                   lower_gpu=lambda n, ctx: _capturing(_Captured(n.payload, _transfer.copy_run(n.payload))),
                   lower_cpu=lambda n, ctx: _transfer.copy_run_host(n.payload),
                   transpose=_copy_transpose)


def _no_transpose(node, incoming, ctx):
    return {}


def _cell(fn, reads=(), writes=(), name=None) -> Node:
    """A zero-arg callable that sets a decision or guard cell: declared like
    :func:`opaque`, but never differentiated and needing no vjp."""
    return Node("cell", (fn, _named_arrays(reads, "in"), _named_arrays(writes, "out")), name, False)


register_node_kind("cell", reads=_declared(1), writes=_declared(2),
                   lower_gpu=lambda n, ctx: n.payload[0], lower_cpu=lambda n, ctx: n.payload[0],
                   transpose=_no_transpose)


class _Child:
    """The framework object owning a captured graph, held for the eagle
    Graph's life, and the graph handle it had at :func:`child`."""

    __slots__ = ("owner", "handle", "_read")

    def __init__(self, owner):
        if isinstance(owner, int):
            raise TypeError("eagle.graph.child: pass the graph object; eagle must keep it "
                            "alive (a bare handle dies with the object that owns it)")
        raw = getattr(owner, "raw_cuda_graph", None)            # torch.cuda.CUDAGraph
        if callable(raw):
            read = raw
        elif hasattr(owner, "graph"):                           # warp.Graph
            read = lambda: getattr(owner.graph, "value", owner.graph)  # noqa: E731
        else:
            raise TypeError("eagle.graph.child: takes a torch.cuda.CUDAGraph (captured with "
                            f"keep_graph=True) or a warp.Graph, got {type(owner).__name__}")
        self.owner, self._read = owner, read
        try:
            self.handle = int(read())
        except Exception as exc:
            raise TypeError("eagle.graph.child: the framework graph holds no captured graph; "
                            "capture it first (torch: CUDAGraph(keep_graph=True))") from exc

    def check(self):
        try:
            now = int(self._read())
        except Exception:
            now = None
        if now != self.handle:
            raise RuntimeError("eagle.Graph.launch: the framework graph was reset or "
                               "re-captured; rebuild the eagle Graph")

    def insert(self, body):
        import cupy

        from . import _core

        _core.capture_add_child(cupy.cuda.get_current_stream().ptr, self.handle, body)


def child(owner, *, reads=(), writes=(), name=None) -> Node:
    """A graph your framework captured itself, inserted as ONE child node of
    eagle's graph: ``owner`` is the object that owns it, a
    ``torch.cuda.CUDAGraph`` captured with ``keep_graph=True`` or a
    ``warp.Graph`` (a bare handle is refused: eagle keeps the owner alive,
    since the graph's kernels point into memory the owner holds).
    ``reads``/``writes`` and ``name`` are :func:`opaque`'s. The
    graph is captured (GPU only; on the host call the model in an
    :func:`opaque`), also inside conditionals and ``until_done`` steps. Each
    launch re-reads the owner's handle and refuses a reset or re-captured
    graph by name. Do not replay the framework graph yourself while an eagle
    launch of it is in flight: both share the framework's memory pool."""
    return Node("child", (_Child(owner), _named_arrays(reads, "in"),
                          _named_arrays(writes, "out")), name)


def _child_cpu(node, ctx):
    raise ValueError(f"eagle.Graph.build: child {node.label!r} is device work; on the host "
                     "call the model in an opaque")


register_node_kind("child", reads=_declared(1), writes=_declared(2), transpose=_opaque_transpose,
                   lower_gpu=lambda n, ctx: (lambda: n.payload[0].insert(ctx.body)),
                   lower_cpu=_child_cpu, check=lambda n: n.payload[0].check())


# --- Built-in kind: group (fork/join) --- #
class _Seq:
    __slots__ = ("parts",)

    def __init__(self, parts):
        self.parts = list(parts)

    def __call__(self):
        for p in self.parts:
            p()


def _group_nodes(node):
    return [n for branch in node.payload for n in branch]


def _group_union(which):
    def get(node):
        out = []
        for n in _group_nodes(node):
            out.extend(getattr(get_node_kind(n.kind), which)(n))
        return tuple(out)
    return get


class _Sync:
    """One cross-branch wait of a fork: the producer branch marks, the waiting
    branch waits on the mark (:class:`_MarkStep`, :class:`_WaitStep`); the fork
    is set where the weave builds it."""

    __slots__ = ("branch", "fork", "index")

    def __init__(self, branch):
        self.branch, self.fork, self.index = branch, None, None


class _MarkStep:
    __slots__ = ("sync",)

    def __init__(self, sync):
        self.sync = sync

    def __call__(self):
        self.sync.index = self.sync.fork.mark(self.sync.branch)


class _WaitStep:
    __slots__ = ("sync", "branch")

    def __init__(self, sync, branch):
        self.sync, self.branch = sync, branch

    def __call__(self):
        self.sync.fork.wait(self.branch, self.sync.index)


def _wait(*planes, name=None) -> Node:
    """A sync point inside a fork/join branch: continue after the node of an
    earlier sibling branch that wrote ``planes``. One edge, no node."""
    return Node("wait", (tuple(planes),), name, False)


def _wait_links(group):
    """``[(branch, position, producer branch, producer position, arrays)]``:
    for each ``wait`` of ``group``'s branches, the latest node of the nearest
    earlier branch that writes what it waits for."""
    links = []
    for q, branch in enumerate(group.payload):
        for k, n in enumerate(branch):
            if n.kind != "wait":
                continue
            arrays = n.payload[0]
            found = next(((j, m) for j in reversed(range(q))
                          for m in reversed(range(len(group.payload[j])))
                          if any(arrays_overlap(w, a) for _n, w, _r
                                 in get_node_kind(group.payload[j][m].kind).writes(group.payload[j][m])
                                 for a in arrays)), None)
            if found is None:
                raise ValueError(
                    "eagle.Graph.wait: that plane is not written by an earlier branch of this "
                    "parallel region: before the fork everything is already ordered, and a later "
                    "branch cannot be waited for (no cycles)")
            links.append((q, k, *found, arrays))
    return links


register_node_kind(
    "wait", reads=lambda n: tuple((f"wait{i}", a, "guard") for i, a in enumerate(n.payload[0])),
    writes=lambda n: (), lower_gpu=lambda n, ctx: (lambda: None), lower_cpu=lambda n, ctx: (lambda: None),
    transpose=_no_transpose)


def _group_members(node, ctx):
    """``(members, syncs)``: each branch a lowered part or a :class:`_Seq` of
    them; on the GPU a ``wait`` becomes a wait on the producer branch's mark,
    taken right after the producing part."""
    parts = [[ctx.lower(n) for n in branch] for branch in node.payload]
    syncs = {}
    if ctx.backend == "gpu":
        links = _wait_links(node)
        for q, k, j, m, _arrays in links:
            parts[q][k] = _WaitStep(syncs.setdefault((j, m), _Sync(j)), q)
        for (j, m), sync in syncs.items():
            parts[j][m] = _Seq([parts[j][m], _MarkStep(sync)])
    return [p[0] if len(p) == 1 else _Seq(p) for p in parts], list(syncs.values())


def _group_gpu(node, ctx):
    members, syncs = _group_members(node, ctx)
    return ForkStage(members, syncs)


def _group_cpu(node, ctx):
    return _Seq(_group_members(node, ctx)[0])


register_node_kind(
    "group", reads=_group_union("reads"), writes=_group_union("writes"),
    lower_gpu=_group_gpu, lower_cpu=_group_cpu)


# --- Built-in kind: sequence (the parts of a body or an arm, in order) --- #
def _sequence_planes(which):
    """The parts' planes, each qualified ``"<part>.<plane>"`` (two parts may
    name a plane alike); what a part that is not differentiated reads is a
    ``guard`` (never taped)."""
    def get(node):
        return tuple((f"{i}.{n}", a, r if (p.grad or which == "writes") else "guard")
                     for i, p in enumerate(node.payload)
                     for n, a, r in getattr(get_node_kind(p.kind), which)(p))
    return get


register_node_kind(
    "sequence", reads=_sequence_planes("reads"), writes=_sequence_planes("writes"),
    lower_gpu=lambda n, ctx: _Seq([ctx.lower(p) for p in n.payload]),
    lower_cpu=lambda n, ctx: _Seq([ctx.lower(p) for p in n.payload]))


# --- Built-in kinds: skippable / repeat_while (the EXISTING eagle objects) --- #
class _Late:
    """The step slot of an adopted conditional. Before a build it runs the
    child's host arm; a build lowers the child for its backend into
    :attr:`fn`. It also marks the conditional as owned by one Graph (the
    lowering writes into it)."""

    __slots__ = ("child", "owner", "fn")

    def __init__(self, child, owner):
        self.child = child
        self.owner = owner
        self.fn = None

    def __call__(self):
        (self.fn or self.child)()


class _Cond:
    __slots__ = ("wrapper", "child", "late")

    def __init__(self, wrapper, child, late):
        self.wrapper = wrapper
        self.child = child
        self.late = late


def _guard_arrays(wrapper):
    out = []
    guard = wrapper.guard
    if guard is not None:
        out.append(("guard", guard.count_arr, "guard"))
        if guard.baseline_arr is not None:
            out.append(("guard_baseline", guard.baseline_arr, "guard"))
    for name in ("limit", "brk"):
        cell = getattr(wrapper, name, None)
        if cell is not None:
            out.append((name, cell, "guard"))
    return tuple(out)


def _cond_reads(node):
    c = node.payload
    return (*get_node_kind(c.child.kind).reads(c.child), *_guard_arrays(c.wrapper))


def _cond_writes(node):
    c = node.payload
    out = get_node_kind(c.child.kind).writes(c.child)
    if isinstance(c.wrapper, RepeatWhile):
        out = (*out, ("iterations", c.wrapper._cell, "guard"))
    return out


def _cond_lower(node, ctx):
    c = node.payload
    if c.child.kind == "sequence":      # a loop's parts are woven one by one; a skippable's as one
        body = LowerCtx(ctx.backend, body=True)
        parts = [body.lower(p) for p in c.child.payload]
        if isinstance(c.wrapper, RepeatWhile):
            c.wrapper.step = tuple(parts)
        else:
            c.late.fn = _Seq(parts)
        return c.wrapper
    c.late.fn = LowerCtx(ctx.backend, body=True).lower(c.child)
    return c.wrapper


def _branches_lower(node, ctx):
    c = node.payload
    body = LowerCtx(ctx.backend, body=True)
    for late, child in zip(c.late, c.child.payload):
        late.fn = body.lower(child)
    return c.wrapper


def _cond_record(node):
    """What the forward saves per launch: a skippable's guard cells, read as
    the launch saw them. A repeat_while's iteration cell is already a plane
    its node writes, so it needs no copy."""
    c = node.payload
    if isinstance(c.wrapper, RepeatWhile):
        return ()
    g = c.wrapper.guard
    out = [("guard", g.count_arr[g.count_idx:g.count_idx + 1])]
    if g.baseline_arr is not None:
        out.append(("guard_baseline", g.baseline_arr[g.baseline_idx:g.baseline_idx + 1]))
    return tuple(out)


def _cond_transpose(node, incoming, ctx):
    from ._conditional_reverse import transpose

    return transpose(node, incoming, ctx)


for _kind in ("skippable", "repeat_while"):
    register_node_kind(_kind, reads=_cond_reads, writes=_cond_writes,
                       lower_gpu=_cond_lower, lower_cpu=_cond_lower,
                       transpose=_cond_transpose, record_planes=_cond_record)


# a branches node (IF/ELSE, SWITCH) has no reverse yet: Reverse refuses it by name
register_node_kind("branches", reads=_cond_reads, writes=lambda n: get_node_kind("sequence").writes(n.payload.child),
                   lower_gpu=_branches_lower, lower_cpu=_branches_lower)
