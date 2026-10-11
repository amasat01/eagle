# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The built-in node kinds of :class:`eagle.graph.Graph`: a plan launch, an
opaque callable with declared planes, a fork/join group, and the existing
``eagle.skippable`` / ``eagle.repeat_while``. Each registers itself with the
node-kind registry; a new kind lives in a module like this one and never
touches ``graph.py``."""

from __future__ import annotations

from ._conditional import RepeatWhile, Skippable
from .graph import ForkStage, Node, get_node_kind, register_node_kind
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


def _launch_native(node, ctx):
    b = node.payload
    if b._cupy is not None:
        return None
    if b._copies or b._active_count is not None:
        return None
    return b._entry, b._addrs, b._parts


def _launch_transpose(node, incoming, ctx):
    """A plan launch's reverse (:mod:`eagle.diff`): its vjp kernel bound to the
    recorded values of the planes it read, the incoming adjoints of the planes
    it wrote, and fresh adjoint planes for everything it read."""
    import numpy

    bound = node.payload
    role_of = {name: role for role, name in bound._arg_spec}
    xp = bound._cupy or numpy
    planes, out = {}, {}
    for role, name in ctx.vjp.plugin.arg_spec:
        src = name[4:] if name.startswith("bar_") else None
        if role == "nsamples":
            continue
        if name in role_of:
            planes[name] = ctx.primal(name)
        elif src in incoming:
            planes[name] = incoming[src]
        elif src in role_of and role_of[src] not in OUTPUT_ROLES:
            # a parameter cell's gradient comes back per sample; the reverse sums it
            per_sample = role_of[src] == "lookup" and role == "mutable"
            buf = (xp.empty(bound.n, dtype=bound._dtype) if per_sample
                   else xp.empty_like(bound._planes[src]))
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
    lower_gpu=_launch_step, lower_cpu=_launch_step, native_entry=_launch_native,
    transpose=_launch_transpose)


# --- Built-in kind: opaque (a callable with declared planes) --- #
def _named_arrays(arrays, prefix):
    if isinstance(arrays, dict):
        return tuple((str(k), v, "declared") for k, v in arrays.items())
    return tuple((f"{prefix}{i}", a, "declared") for i, a in enumerate(arrays))


def opaque(fn, *, reads=(), writes=()) -> Node:
    """A zero-arg callable (a cupy custom kernel, a torch op, a host closure)
    with the arrays it touches DECLARED -- a sequence of arrays, or a
    ``{name: array}`` mapping. Edges are still derived from those arrays (the
    declaration says WHICH planes, never which edges) and the conflict check
    covers them. An undeclared write is not detected. GPU lowering is a
    capture step, CPU an eager call; a reverse that meets one refuses by
    name."""
    if not callable(fn):
        raise TypeError("eagle.graph.opaque: fn must be a zero-arg callable")
    return Node("opaque", (fn, _named_arrays(reads, "in"), _named_arrays(writes, "out")))


register_node_kind(
    "opaque",
    reads=lambda n: n.payload[1], writes=lambda n: n.payload[2],
    lower_gpu=lambda n, ctx: n.payload[0], lower_cpu=lambda n, ctx: n.payload[0])


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


def _group_members(node, ctx):
    members = []
    for branch in node.payload:
        parts = [ctx.lower(n) for n in branch]
        if len(parts) > 1 and any(isinstance(p, (Skippable, RepeatWhile)) for p in parts):
            raise ValueError(
                "eagle.Graph.build: a fork/join branch holding a skippable or "
                "repeat_while must hold only that node (a conditional is woven "
                "as one region); split the branch")
        members.append(parts[0] if len(parts) == 1 else _Seq(parts))
    return members


def _group_cpu(node, ctx):
    return _Seq(_group_members(node, ctx))


register_node_kind(
    "group", reads=_group_union("reads"), writes=_group_union("writes"),
    lower_gpu=lambda n, ctx: ForkStage(_group_members(n, ctx)), lower_cpu=_group_cpu)


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


def _guard_arrays(guard):
    out = [("guard", guard.count_arr, "guard")]
    if guard.baseline_arr is not None:
        out.append(("guard_baseline", guard.baseline_arr, "guard"))
    return tuple(out)


def _cond_reads(node):
    c = node.payload
    return (*get_node_kind(c.child.kind).reads(c.child), *_guard_arrays(c.wrapper.guard))


def _cond_writes(node):
    c = node.payload
    out = get_node_kind(c.child.kind).writes(c.child)
    if isinstance(c.wrapper, RepeatWhile):
        out = (*out, ("iterations", c.wrapper._cell, "guard"))
    return out


def _cond_lower(node, ctx):
    c = node.payload
    c.late.fn = ctx.lower(c.child)
    return c.wrapper


def _cond_record(node):
    c = node.payload
    out = [(n, a) for n, a, _r in _guard_arrays(c.wrapper.guard)]
    if isinstance(c.wrapper, RepeatWhile):
        out.append(("iterations", c.wrapper._cell))
    return tuple(out)


for _kind in ("skippable", "repeat_while"):
    register_node_kind(_kind, reads=_cond_reads, writes=_cond_writes,
                       lower_gpu=_cond_lower, lower_cpu=_cond_lower,
                       record_planes=_cond_record)
