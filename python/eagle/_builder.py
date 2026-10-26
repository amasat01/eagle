# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The graph builder: ``@eagle.Graph.builder`` runs a Python function once,
and what it means becomes a Graph.

An AST pass rewrites ``if`` / ``while`` / the guarded ``for`` of the body into
calls on a recorder. The recorder (:mod:`eagle._recording`) takes each kernel
call instead of launching it and appends a node; a condition that is a 1-cell
uint32 array becomes device control flow, any other condition is decided once,
in Python. The recorded tree is lowered through ``Graph.add_node`` /
``skippable`` / ``repeat_while`` / ``_group_node`` and nothing else."""

from __future__ import annotations

import ast
import contextlib
import copy
import functools
import inspect
import itertools
import linecache
import textwrap
import time
import warnings

import numpy

from . import _alloc
from . import _cell_rows, _lane_bind, _recording
from ._graph_reduce import reduce
from ._conditional import RepeatWhile, SkipGuard
from ._graph_kinds import _cell, _copy, _wait
from ._conditional import skippable
from .graph import Graph, Node, get_node_kind
from ._plan_planes import _declared_widths
from .interop import _xp_of

_MATCH = getattr(ast, "Match", ())  # a `match` statement exists from Python 3.10
_COUNT = itertools.count()
_MISSING = object()
_DTYPES = {"f64": "float64", "f32": "float32", "uint32": "uint32", "int32": "int32"}


# ----------------------------------------------------------------------------- values
def is_cell(v) -> bool:
    return hasattr(v, "shape") and hasattr(v, "dtype") and tuple(v.shape) == (1,) and str(v.dtype) == "uint32"


class Neg:
    """``not cell``: a device condition, negated."""

    def __init__(self, cell): self.cell = cell


def classify(v, where):
    """``("cell", array)``, ``("neg", array)`` or ``("py", bool)``."""
    if isinstance(v, Neg): return "neg", v.cell
    if is_cell(v): return "cell", v
    if hasattr(v, "shape") and getattr(v, "ndim", 0) > 0:
        raise ValueError(f"{where}: a device condition is a 1-cell uint32 array (or a Python value); "
                         f"got shape {tuple(v.shape)} {v.dtype}")
    return "py", bool(v)


def tree_leaves(tree):
    """``((path, array), ...)`` of a hawk return tree; refuses anything else."""
    if tree is None: return ()
    if hasattr(tree, "shape"): return (("", tree),)
    if isinstance(tree, (tuple, list)):
        return tuple((f"[{i}]{k}", a) for i, t in enumerate(tree) for k, a in tree_leaves(t))
    if isinstance(tree, dict):
        if not all(isinstance(key, str) for key in tree):
            raise TypeError("eagle.Graph.result: dict keys are strings (hawk's return tree)")
        return tuple((f".{key}{k}", a) for key, t in tree.items() for k, a in tree_leaves(t))
    raise TypeError("eagle.Graph.result: an array, a tuple/list/dict of arrays, nested (hawk's "
                    f"return tree), got {type(tree).__name__}")


def in_place_ok(kernel) -> bool:
    """A kernel may read and write one plane only when it is ``sample_local``
    and reduces nothing across samples."""
    walk = kernel.walk
    return walk.access.cls == "sample_local" and not any(
        type(n).__name__ == "ReduceSamples" for n in walk.order)


# ----------------------------------------------------------------------------- the recorded tree
class K:
    """A kernel launch."""

    def __init__(self, bound, label): self.bound, self.label = bound, label


class Cnd:
    """``body`` guarded by ``guard`` (the rest of a loop body after a ``continue`` or ``break``)."""

    def __init__(self, guard, body): self.guard, self.body = guard, body


class Chain:
    """An ``if``/``elif``/``else`` chain on device cells: ``kinds`` is ``(kind, cell, value)`` per condition (``value`` for a ``match`` case),
    ``arms`` one body per condition and one more for the ``else``, if any."""

    def __init__(self, kinds, arms): self.kinds, self.arms = kinds, arms


class Whl:
    """A device loop: the ``RepeatWhile`` (made when the loop opened, so its iteration index could be read) and its body."""

    def __init__(self, rw, body): self.rw, self.body = rw, body


class Wt:
    def __init__(self, planes): self.planes = planes


class Frk:
    def __init__(self, branches): self.branches = branches


def _uniq(arrs):
    seen, out = set(), []
    for a in arrs:
        if id(a) not in seen:
            seen.add(id(a)); out.append(a)
    return out


def _node_arrays(node):
    k = get_node_kind(node.kind)
    return [a for _n, a, _r in k.reads(node)], [a for _n, a, _r in k.writes(node)]


def arrays(ir):
    """``(reads, writes)`` of a recorded item (what a ``wait`` looks for)."""
    if isinstance(ir, K): return _node_arrays(Node("launch", ir.bound))
    if isinstance(ir, Node): return _node_arrays(ir)
    if isinstance(ir, Cnd): return seq_arrays(ir.body)
    if isinstance(ir, Chain): return seq_arrays([i for a in ir.arms for i in a])
    if isinstance(ir, Whl): return seq_arrays(ir.body)
    if isinstance(ir, Frk):
        r, w = [], []
        for b in ir.branches:
            rr, ww = seq_arrays(b.ops); r += rr; w += ww
        return _uniq(r), _uniq(w)
    return [], []


def seq_arrays(seq):
    r, w = [], []
    for it in seq:
        rr, ww = arrays(it); r += rr; w += ww
    return _uniq(r), _uniq(w)


def _holds_cell(seq) -> bool:
    """A derived cell or an if/match chain (its decision copy cannot be re-run by a reverse) in ``seq``."""
    return any(isinstance(i, Node) and i.kind == "cell" or isinstance(i, Chain)
               or isinstance(i, (Cnd, Whl)) and _holds_cell(i.body)
               or isinstance(i, Frk) and any(_holds_cell(b.ops) for b in i.branches) for i in seq)


# ----------------------------------------------------------------------------- lowering
class _Scratch(Graph):
    """Collects the nodes ``_if_else`` / ``_chain`` append, so an arm can sit inside another body."""

    def __init__(self):
        with _recording.recording(None):
            super().__init__()


def _step(g, body):
    nodes = [n for it in body for n in nodes_of(g, it)]
    return nodes[0] if len(nodes) == 1 else Node("sequence", tuple(nodes), None, any(n.grad for n in nodes))


def nodes_of(g, it) -> list:
    if isinstance(it, K): return [g._node_of(it.bound, it.label, door="eagle.Graph.builder")]
    if isinstance(it, Node): return [it]
    if isinstance(it, Cnd):
        return [g._node_of(skippable(_step(g, it.body), it.guard), door="eagle.Graph.builder")]
    if isinstance(it, Chain): return _chain_nodes(g, it)
    if isinstance(it, Whl):
        it.rw.step = _step(g, it.body)
        return [g._node_of(it.rw, door="eagle.Graph.builder")]
    if isinstance(it, Wt): return [_wait(*it.planes)]
    branches = [[n for op in b.ops for n in nodes_of(g, op)] for b in it.branches]
    branches = [b for b in branches if any(n.kind != "wait" for n in b)]
    if len(branches) < 2: return [n for b in branches for n in b if n.kind != "wait"]
    return [g._group_node(branches, door="eagle.Graph.builder")]


def _chain_nodes(g, it):
    s = _Scratch()
    steps = [_step(g, a) if a else None for a in it.arms]
    if not any(steps): return []
    n = len(it.kinds)
    if n == 1 and it.kinds[0][0] != "eq":
        kind, cell, _ = it.kinds[0]
        guard = SkipGuard(cell, 0)
        if kind == "neg": guard = guard.negated()
        if steps[0] is None: s._if_else(guard.negated(), steps[1])
        else: s._if_else(guard, steps[0], steps[1] if len(steps) > 1 else None)
    else:
        node, idx = _index_cell(it.kinds)
        s.add_node(node)
        none = _cell(lambda: None, name="empty")
        s._chain(idx, [st or none for st in steps[:n]], steps[n] if len(steps) > n else None)
    return s._nodes


def _index_cell(kinds):
    """The arm to run: the first condition that holds, else ``len(kinds)``. Plain ufuncs with ``out=``: capturable."""
    xp = _xp_of(kinds[0][1])
    n = len(kinds)
    idx, t = _alloc.zeros(xp, 1, dtype=numpy.uint32), _alloc.zeros(xp, 1, dtype=numpy.uint32)
    gap = _alloc.zeros(xp, 1, dtype=numpy.uint32)
    picks = [_alloc.full(xp, 1, i, dtype=numpy.uint32) for i in range(n)]

    def run():
        idx.fill(n)
        for i in reversed(range(n)):
            k, c, v = kinds[i]
            (xp.not_equal if k == "cell" else xp.equal)(c, 0 if v is None else v, out=t)
            xp.subtract(picks[i], idx, out=gap); xp.multiply(gap, t, out=gap); xp.add(idx, gap, out=idx)
    return _cell(run, [c for _, c, _v in kinds], [idx], "index"), idx


def lower_into(g, items):
    for it in items:
        for n in nodes_of(g, it): g.add_node(n)


# ----------------------------------------------------------------------------- the recorder
class Scope:
    def __init__(self, nodes, handle, kind): self.nodes, self.handle, self.kind = nodes, handle, kind


class Rec:
    """One builder (or block-on-a-hand-made-graph) run: the recorder
    :meth:`record_call` hands kernel calls to."""

    def __init__(self, graph, kind, args, fname):
        self.graph, self.kind, self.fname = graph, kind, fname
        self.args = {id(a): n for n, a in args.items() if hasattr(a, "shape")}
        self.top = []
        self.scopes = [Scope(self.top, graph, "top")]
        self.frames, self.block_depth, self.target = [], 0, None
        self.labels, self.buffers, self.deep = {}, {}, set()
        self.views = {}
        self.stats = {"allocs": 0, "seeds": 0}
        self.rewrite = 0.0

    @property
    def cur(self): return self.scopes[-1].nodes

    def handle(self):
        for s in reversed(self.scopes):
            if s.handle is not None: return s.handle
        return self.graph

    def where(self, line): return f"{self.fname}:{line}"

    def label(self, base):
        n = self.labels[base] = self.labels.get(base, 0) + 1
        return base if n == 1 else f"{base}#{n}"

    def record_call(self, plan, args, kw):
        kw = dict(kw)
        buffer = kw.pop("buffer", None)
        target, self.target = self.target, None
        kernel = plan.kernel
        kw, lifted = _lane_bind.take_extent(plan.plugin, kw)    # the direct call's own helper
        axes0 = _lane_bind.declared_axes(plan.plugin)
        extent = lifted.get(axes0[0][0]) if axes0 else None
        if kernel is None:
            planes = plan.bind_args(args, kw)
        else:   # a kernel call: positional in the kernel's own order, a reduction's cell by position or name
            comps = _cell_rows.companions_of(plan.plugin)  # eagle's: no position, by name only
            order = [nm for nm, decl in kernel.planes.items()      # the Grid declaration is no plane
                     if not getattr(decl, "grid", None) and nm not in comps]
            if len(args) > len(order):
                raise ValueError(f"{kernel.name}() too many positional arguments")
            given = dict(zip(order, args))
            if given.keys() & kw.keys():
                raise ValueError(f"{kernel.name}() got multiple values for {sorted(given.keys() & kw.keys())[0]!r}")
            given.update(kw)
            call = plan._call = plan._call or plan._signature()
            cells = {nm: given.pop(nm) for nm in list(given)
                     if nm in order and nm not in call[4] or nm in comps}
            planes = plan.bind_args((), given)
            planes.update(cells)
        omitted = [] if kernel is None else [nm for nm in kernel.planes if nm in plan._call[4] and nm not in planes]
        label = self.label(target[0] if target else (kernel.name if kernel is not None else "kernel"))
        arrs = [v for v in planes.values() if hasattr(v, "shape")]
        samples = next((a for a in arrs if not is_cell(a)), arrs[0] if arrs else None)
        lanes, dims = self._lanes(plan, planes, extent)
        extents = {} if lanes is None else _lane_bind.extents_of(plan.plugin, {**kw, **lifted, **planes})
        out = {}
        for i, nm in enumerate(omitted):
            spec = buffer.get(nm) if isinstance(buffer, dict) else buffer[i] if isinstance(buffer, (tuple, list)) else buffer
            slot = kernel.planes[nm]
            if hasattr(spec, "shape"):
                out[nm] = spec
            else:
                out[nm] = self._inplace(target, nm, kernel, arrs) if spec is None else None
                if out[nm] is None:
                    if slot.dims and lanes is None:
                        raise ValueError(f"{self.fname}: {kernel.name}() writes {nm!r} per vector: pass the array yourself")
                    floats = [a.dtype for a in arrs if a.dtype.kind == "f"]
                    dt = _DTYPES[slot.ttype.dtype] if slot.ttype.dtype != "f64" or not floats else numpy.result_type(*floats)
                    out[nm] = _alloc.zeros(_xp_of(samples), 
                        samples.shape[-1] if lanes is None else _lane_bind.output_shape(plan.plugin, nm, 1, lanes, extents), dtype=dt)
                    self.stats["allocs"] += 1
            if spec is not None:
                key = label if len(omitted) == 1 else f"{label}.{nm}"
                self.buffers[key] = out[nm]
                if self.frames: self.deep.add(key)
        planes.update(out)
        rows = self._own_rows(plan, planes, samples.shape[-1] if lanes is None else lanes) \
            if _cell_rows.cells_of(plan.plugin) else ()
        if extent is not None:
            planes[axes0[0][0]] = extent       # a free label sized the sinks above; bind reads their shapes
        bound = plan.bind(**planes)
        for nm, held in planes.items():       # the caller's lane-shaped array is a view of the flat plane the graph holds
            flat = bound._planes.get(nm)
            if flat is not None and flat is not held and hasattr(held, "shape"):
                self.views[id(held)] = (held, flat)
        k = K(bound, label)
        self.cur.append(k)
        self.cur.extend(rows)
        # the planes stay flat in the graph; the caller sees the lane shape a direct call returns
        shaped = _lane_bind.shape_result(plan.plugin, next(iter(out.values())) if len(out) == 1 else out,
                                         list(out), dims, lanes, _declared_widths(plan.plugin)) if out and lanes is not None else out
        for nm, view in (shaped if isinstance(shaped, dict) else {next(iter(out), None): shaped}).items():
            if view is not out.get(nm) and hasattr(view, "shape"):
                self.views[id(view)] = (view, out[nm])
        if getattr(kernel, "returns", None) is not None:
            return plan.returned(shaped if isinstance(shaped, dict) else {next(iter(out)): shaped})
        return shaped if out else None

    @staticmethod
    def _own_rows(plan, planes, n):
        """A kernel that stores a reduction over samples writes one row per ``ROW`` samples (and a
        ``_comp`` row for a sum) into its plane, not the cell. The graph owns those rows, made here
        once and reused by every replay, and finishes them into the caller's cell with a reduce
        node right after the launch."""
        nodes = []
        for name, op in _cell_rows.cells_of(plan.plugin).items():
            if name not in planes:          # a stage-internal cell, not a plane of the call
                continue
            cell = planes[name]
            xp, rows = _xp_of(cell), -(-int(n) // _cell_rows.ROW)
            planes[name] = hi = _alloc.zeros(xp, rows, cell.dtype)
            if op in ("sum", "mean"):
                planes[_cell_rows.comp_name(name)] = lo = _alloc.zeros(xp, rows, cell.dtype)
                nodes.append(reduce(hi, cell, comp=lo))
            else:
                nodes.append(reduce(hi, cell, op=op))
        return nodes

    @staticmethod
    def _lanes(plan, planes, extent):
        """``(lane count, lane shape)`` of a launch of a kernel that declares a lane space
        (``(None, None)`` for any other): what sizes an output the call omits, since a Table or a
        lane-shaped plane is not ``samples.shape[-1]``."""
        axes = _lane_bind.declared_axes(plan.plugin)
        if not axes:
            return None, None
        if extent is not None:
            planes = {**planes, axes[0][0]: extent}
        _kw, n, dims = _lane_bind.bind_n(plan.plugin, plan.plugin.arg_spec, planes,
                                         _declared_widths(plan.plugin), door="call")
        return n, dims

    def _inplace(self, target, slot, kernel, arrs):
        if target is None: return None
        for fr in reversed(self.frames):
            if target[0] in fr:
                out = fr[target[0]]
                if any(a is out for a in arrs) and not in_place_ok(kernel):
                    raise ValueError(f"{self.where(target[1])}: `{target[0]} = {kernel.name}(...)` inside device control flow "
                                     f"writes '{target[0]}' in place, and '{kernel.name}' is not sample_local: declare the "
                                     "write (Mutable) or move it out of the loop")
                return out
        return None

    def push_frame(self, assigned, after, env, where, loop, readfirst=()):
        """The liveness rule for one device construct: a name read after it merges in place into the buffer it
        held on entry (a builder argument gets a graph-owned copy, seeded per launch); a name local to it is a
        fresh buffer. Returns ``{name: array}`` to rebind."""
        inplace, rebind = {}, {}
        for n in sorted(assigned):
            v = env.get(n, _MISSING)
            if n not in after:
                if loop and n in readfirst and v is not _MISSING and hasattr(v, "shape"):
                    raise ValueError(f"{where}: '{n}' is read before it is written in the loop body: it is carried across "
                                     f"iterations; bind {n} before the loop or write it before reading it")
                continue
            if v is _MISSING:
                raise ValueError(f"{where}: '{n}' is assigned inside device control flow and read after it, but is not bound "
                                 f"before it: bind it first ({n} = ...)")
            if not hasattr(v, "shape"): continue
            if id(v) in self.args:
                G = _alloc.zeros_like(_xp_of(v), v); self.stats["seeds"] += 1
                self.top.insert(0, _copy(v, G, f"seed({n})"))
                inplace[n] = rebind[n] = G
            else:
                inplace[n] = v
        self.frames.append(inplace)
        return rebind


class _ScopeCM:
    def __init__(self, rec, kind, done): self.rec, self.kind, self.done = rec, kind, done

    def __enter__(self):
        self.scope = Scope([], None, self.kind); self.rec.scopes.append(self.scope); return self

    def __exit__(self, et, ev, tb):
        self.rec.scopes.pop()
        if et is None: self.done(self.scope.nodes)


class IfCtl:
    def __init__(self, rec, conds, has_else, assigned, after, env, line):
        self.rec = rec
        where = rec.where(line)
        self.kinds = [(k, v, None) for k, v in (classify(c(), where) for c in conds)]
        self.device = any(k != "py" for k, _, _ in self.kinds)
        self.rebind = {}
        if not self.device:
            self.chosen = next((i for i, (_, v, _x) in enumerate(self.kinds) if v), len(conds) if has_else else None)
            return
        if any(k == "py" for k, _, _v in self.kinds):
            raise ValueError(f"{where}: an if/elif chain mixes a device cell and a Python value: make every condition a cell, or none")
        self.rebind = rec.push_frame(assigned, after, env, where, False)
        self.arms = [[] for _ in range(len(conds) + has_else)]

    def runs(self, k): return self.chosen == k if not self.device else True

    def arm(self, k):
        if not self.device: return contextlib.nullcontext()
        return _ScopeCM(self.rec, "arm", self.arms[k].extend)

    def end(self):
        if self.device:
            self.rec.frames.pop()
            if any(self.arms): self.rec.cur.append(Chain(self.kinds, self.arms))


class MatchCtl(IfCtl):
    """``match cell:`` with integer cases: a chain whose conditions are ``cell == value``."""

    def __init__(self, rec, subject, values, has_default, assigned, after, env, line):
        v = subject()
        if not is_cell(v):
            self.rec, self.device, self.rebind = rec, False, {}
            self.chosen = next((i for i, c in enumerate(values) if v == c), len(values) if has_default else None)
            return
        self.rec, self.device = rec, True
        self.kinds = [("eq", v, c) for c in values]
        self.rebind = rec.push_frame(assigned, after, env, rec.where(line), False)
        self.arms = [[] for _ in range(len(values) + has_default)]


class LoopCtl:
    def __init__(self, rec, kind, condf, cap, assigned, after, env, readfirst, line):
        self.rec, self.where = rec, rec.where(line)
        v = condf()
        if kind == "count":
            k, self.device = "limit", is_cell(v)
        else:
            k, v = classify(v, self.where)
            self.device = k != "py"
        if not self.device: return
        self.record = rec.graph is not None and rec.graph._defaults["record"]
        if kind == "while":
            if self.record:
                raise ValueError(f"{self.where}: an uncapped device loop cannot be recorded (record=True): cap it: "
                                 "for _ in range(N): if not cell: break")
            warnings.warn(f"uncapped device loop at {self.where}: a body that never clears its guard never returns; "
                          "cap it with for _ in range(N): if not cell: break", stacklevel=2)
        if k == "limit":
            self.rw = RepeatWhile(_idle, None, None, limit=v)
        else:
            self.rw = RepeatWhile(_idle, SkipGuard(v, 0) if k == "cell" else SkipGuard(v, 0).negated(), cap)
        self.index, self.brk, self.nodes = self.rw.iteration_index, None, []
        self.rebind = rec.push_frame(assigned, after, env, self.where, True, readfirst)

    def body(self): return _ScopeCM(self.rec, "body", lambda nodes: setattr(self, "nodes", nodes))

    def exit_if(self, condf, what):
        """The guard of the rest of the body after ``if cell: break`` (or ``continue``)."""
        k, v = classify(condf(), self.where)
        if k == "py":
            raise ValueError(f"{self.where}: `{what}` on a Python value inside a device loop: use a 1-cell uint32 array")
        if what == "continue":
            return SkipGuard(v, 0).negated() if k == "cell" else SkipGuard(v, 0)
        xp = _xp_of(v)
        if self.brk is None: self.brk = _alloc.zeros(xp, 1, dtype=numpy.uint32)
        brk = self.brk
        self.rec.cur.append(_copy(v, brk, "break") if k == "cell"
                            else _cell(lambda: xp.equal(v, 0, out=brk), [v], [brk], "break"))
        return SkipGuard(brk, 0).negated()

    def rest(self, guard):
        return _ScopeCM(self.rec, "arm", lambda nodes: nodes and self.rec.cur.append(Cnd(guard, nodes)))

    def end(self):
        self.rec.frames.pop()
        body = list(self.nodes)
        if self.record and (self.brk is not None or _holds_cell(body)):
            raise ValueError(f"{self.where}: a device loop holding a break, an if/elif/else or a match cannot be recorded "
                             "(record=True): keep the loop body to kernels")
        self.rw.brk = self.brk
        self.rec.cur.append(Whl(self.rw, body))


def _idle(): pass


class Runtime:
    """The object the rewritten code calls (``_eagle_rt``)."""

    @staticmethod
    def env(loc, names): return {n: loc[n] for n in names if n in loc}

    @staticmethod
    def neg(v): return Neg(v) if is_cell(v) else (not v)

    @staticmethod
    def into(name, line, f):
        from ._plan_deploy import AutoPlan
        if not (isinstance(f, AutoPlan) or hasattr(f, "planes") and hasattr(f, "returns")):   # a plan, or a hawk kernel
            return f

        def call(*a, **k):
            rec = _recording.current()
            rec.target = (name, line)
            try:
                return f(*a, **k)
            finally:
                rec.target = None
        return call

    @staticmethod
    def begin_if(conds, has_else, assigned, after, env, line):
        return IfCtl(_recording.current(), conds, has_else, assigned, after, env, line)

    @staticmethod
    def begin_match(subject, values, has_default, assigned, after, env, line):
        return MatchCtl(_recording.current(), subject, values, has_default, assigned, after, env, line)

    @staticmethod
    def begin_loop(kind, condf, cap, assigned, after, env, readfirst, line):
        rec = _recording.current()
        if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int)):
            raise TypeError(f"{rec.where(line)}: the cap of a device loop is a Python int, got {type(cap).__name__}")
        return LoopCtl(rec, kind, condf, cap, assigned, after, env, readfirst, line)

    @staticmethod
    def bad_exit(line):
        raise ValueError(f"{_recording.current().where(line)}: `break`/`continue` in a device loop must be the whole body of an "
                         "`if <cell>:` directly in the loop body")


RT = Runtime()


# ----------------------------------------------------------------------------- the AST pass
def _loads(stmts): return {n.id for s in stmts for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
def _stores(stmts): return {n.id for s in stmts for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}


def _readfirst(stmts):
    seen, out = set(), set()
    for s in stmts:
        out |= _loads([s]) - seen
        seen |= _stores([s])
    return out


def _parse(src): return ast.parse(textwrap.dedent(src)).body


class _Fill(ast.NodeTransformer):
    """Replace placeholder names (statement lists or expressions) in a parsed template."""

    def __init__(self, m): self.m = m

    def visit_Name(self, n):
        v = self.m.get(n.id)
        return v if v is not None and not isinstance(v, list) else n

    def _block(self, stmts):
        out = []
        for s in stmts:
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Name) and isinstance(self.m.get(s.value.id), list):
                out += self.m[s.value.id]
            else:
                out.append(self.visit(s))
        return out

    def generic_visit(self, node):
        for f, v in ast.iter_fields(node):
            if isinstance(v, list) and v and all(isinstance(x, ast.stmt) for x in v): setattr(node, f, self._block(v))
            elif isinstance(v, list): setattr(node, f, [self.visit(x) if isinstance(x, ast.AST) else x for x in v])
            elif isinstance(v, ast.AST): setattr(node, f, self.visit(v))
        return node


def _fill(src, **m): return [_Fill(m).visit(s) for s in _parse(src)]


class _NoBreak(ast.NodeTransformer):
    """Device copy of a loop body: a break/continue the pass does not lower is refused at record."""

    def visit_For(self, n): return n
    visit_While = visit_FunctionDef = visit_For

    def visit_Break(self, n):
        return ast.copy_location(ast.Expr(ast.parse(f"_eagle_rt.bad_exit({n.lineno})", mode="eval").body), n)
    visit_Continue = visit_Break


def _lam(e): return ast.Lambda(ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]), e)


def _names(s): return "(" + "".join(repr(n) + "," for n in sorted(s)) + ")"


class Rewriter:
    def stmts(self, body, outer_after):
        out = []
        for i, st in enumerate(body):
            out += self.stmt(st, _loads(body[i + 1:]) | outer_after)
        return out

    def cond(self, t):
        if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
            return ast.Call(ast.Attribute(ast.Name("_eagle_rt", ast.Load()), "neg", ast.Load()), [t.operand], [])
        return t

    def stmt(self, st, after):
        if isinstance(st, ast.If): return self.if_(st, after)
        if isinstance(st, ast.While): return self.while_(st, after)
        if isinstance(st, _MATCH): return self.match_(st, after)
        if isinstance(st, ast.For):
            g = self.guarded_for(st, after) or self.counted_for(st, after)
            if g is not None: return g
            st.body = self.stmts(st.body, after | _loads(st.body)); st.orelse = self.stmts(st.orelse, after)
        elif isinstance(st, (ast.With, ast.Try)):
            for f in ("body", "orelse", "finalbody"):
                if hasattr(st, f): setattr(st, f, self.stmts(getattr(st, f), after))
            for h in getattr(st, "handlers", []): h.body = self.stmts(h.body, after)
        elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                and isinstance(st.value, ast.Call):
            c = st.value
            c.func = ast.Call(ast.Attribute(ast.Name("_eagle_rt", ast.Load()), "into", ast.Load()),
                              [ast.Constant(st.targets[0].id), ast.Constant(st.lineno), c.func], [])
        return [st]

    def rebinds(self, ctl, names):
        out = []
        for n in sorted(names):
            out += _parse(f"try:\n    {n} = {ctl}.rebind[{n!r}]\nexcept KeyError:\n    pass\n")
        return out

    def if_(self, st, after):
        arms, conds, node = [], [], st
        while True:
            conds.append(self.cond(node.test)); arms.append(node.body)
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If): node = node.orelse[0]
            else: els = node.orelse; break
        if els: arms.append(els)
        assigned = _stores([s for a in arms for s in a])
        ctl = f"__eg{next(_COUNT)}"
        head = _parse(f"{ctl} = _eagle_rt.begin_if(CONDS, {bool(els)}, frozenset({_names(assigned)}), "
                      f"frozenset({_names(after)}), _eagle_rt.env(locals(), {_names(assigned)}), {st.lineno})")
        head[0].value.args[0] = ast.Tuple([_lam(c) for c in conds], ast.Load())
        out = head + self.rebinds(ctl, assigned & after)
        for k, a in enumerate(arms):
            out += _fill(f"if {ctl}.runs({k}):\n    with {ctl}.arm({k}):\n        BODY\n", BODY=self.stmts(a, after))
        return out + _parse(f"{ctl}.end()")

    def match_(self, st, after):
        values, arms, default = [], [], None
        for c in st.cases:
            p = c.pattern
            if c.guard is not None:
                raise SyntaxError(f"line {c.pattern.lineno}: a match case with a guard is not supported in a builder")
            if isinstance(p, ast.MatchValue) and isinstance(p.value, ast.Constant) and type(p.value.value) is int:
                if default is not None:
                    raise SyntaxError(f"line {p.lineno}: `case _:` must be the last case")
                values.append(p.value.value); arms.append(c.body)
            elif isinstance(p, ast.MatchAs) and p.pattern is None and p.name is None:
                default = c.body
            else:
                raise SyntaxError(f"line {p.lineno}: a builder's match takes integer literals and `case _:`; "
                                  f"{type(p).__name__} is not supported")
        if default is not None: arms.append(default)
        assigned = _stores([s for a in arms for s in a])
        ctl = f"__eg{next(_COUNT)}"
        head = _parse(f"{ctl} = _eagle_rt.begin_match(SUBJ, {tuple(values)!r}, {default is not None}, frozenset({_names(assigned)}), "
                      f"frozenset({_names(after)}), _eagle_rt.env(locals(), {_names(assigned)}), {st.lineno})")
        head[0].value.args[0] = _lam(st.subject)
        out = head + self.rebinds(ctl, assigned & after)
        for k, a in enumerate(arms):
            out += _fill(f"if {ctl}.runs({k}):\n    with {ctl}.arm({k}):\n        BODY\n", BODY=self.stmts(a, after))
        return out + _parse(f"{ctl}.end()")

    def dev_body(self, body, after, ctl):
        out = []
        for i, st in enumerate(body):
            if isinstance(st, ast.If) and not st.orelse and len(st.body) == 1 and isinstance(st.body[0], (ast.Break, ast.Continue)):
                what = "break" if isinstance(st.body[0], ast.Break) else "continue"
                ok = f"__eg{next(_COUNT)}"
                h = _parse(f"{ok} = {ctl}.exit_if(COND, {what!r})"); h[0].value.args[0] = _lam(self.cond(st.test))
                rest = self.dev_body(body[i + 1:], after, ctl)
                return out + h + (_fill(f"with {ctl}.rest({ok}):\n    BODY\n", BODY=rest) if rest else [])
            out += [_NoBreak().visit(s) for s in self.stmt(copy.deepcopy(st), _loads(body[i + 1:]) | after)]
        return out

    def loop(self, kind, cond, body, after, line, py_stmt, cap="None", index=None):
        ctl = f"__eg{next(_COUNT)}"
        assigned = _stores(body)
        head = _parse(f"{ctl} = _eagle_rt.begin_loop({kind!r}, COND, {cap}, frozenset({_names(assigned)}), "
                      f"frozenset({_names(after)}), _eagle_rt.env(locals(), {_names(assigned)}), "
                      f"frozenset({_names(_readfirst(body))}), {line})")
        head[0].value.args[1] = _lam(cond)
        dev = self.rebinds(ctl, assigned & after) + (_parse(f"{index} = {ctl}.index") if index else []) + _fill(f"with {ctl}.body():\n    BODY\n",
                                                          BODY=self.dev_body(body, after | _loads(body), ctl)) + _parse(f"{ctl}.end()")
        return head, [ast.If(ast.Attribute(ast.Name(ctl, ast.Load()), "device", ast.Load()), dev, [py_stmt])]

    def while_(self, st, after):
        if st.orelse: raise SyntaxError(f"line {st.lineno}: while/else is not supported in a builder")
        py = ast.While(st.test, self.stmts(copy.deepcopy(st.body), after | _loads(st.body)), [])
        head, rest = self.loop("while", self.cond(st.test), st.body, after, st.lineno, py)
        return head + rest

    def counted_for(self, st, after):
        it = st.iter
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and len(it.args) == 1
                and isinstance(st.target, ast.Name) and not st.orelse):
            return None
        py = ast.For(st.target, it, self.stmts(copy.deepcopy(st.body), after | _loads(st.body)), [])
        head, rest = self.loop("count", it.args[0], st.body, after, st.lineno, py, index=st.target.id)
        return head + rest

    def guarded_for(self, st, after):
        it, f = st.iter, (st.body[0] if st.body else None)
        if not (isinstance(it, ast.Call) and isinstance(it.func, ast.Name) and it.func.id == "range" and len(it.args) == 1
                and isinstance(f, ast.If) and not f.orelse and len(f.body) == 1 and isinstance(f.body[0], ast.Break)
                and isinstance(f.test, ast.UnaryOp) and isinstance(f.test.op, ast.Not)):
            return None
        py = ast.For(st.target, it, self.stmts(copy.deepcopy(st.body), after | _loads(st.body)), [])
        head, rest = self.loop("for", f.test.operand, st.body[1:], after, st.lineno, py, cap="CAP")
        head[0].value.args[2] = it.args[0]
        return head + rest


_MEMO = {}


def _definition(fn):
    """The ``FunctionDef`` of ``fn`` from its source. A source that is not the loaded function (a file edited under a
    running process, a stale line cache) is refused by name, never rewritten."""
    for attempt in range(2):
        fd = ast.parse(textwrap.dedent(inspect.getsource(fn))).body[0]
        want = [p for p in inspect.signature(fn).parameters]
        if isinstance(fd, ast.FunctionDef) and fd.name == fn.__name__ and [a.arg for a in fd.args.args] == want[:len(fd.args.args)]:
            return fd
        linecache.checkcache(fn.__code__.co_filename)
    raise RuntimeError(f"{fn.__name__}: the source of {fn.__code__.co_filename}:{fn.__code__.co_firstlineno} is not this "
                       "function (the file changed since it was loaded): reload the module")


def rewrite_function(fn):
    """``fn`` with its control flow lowered to recorder calls (code memoised per code object, the closure
    rebuilt per call; the runtime is one more free variable)."""
    code = fn.__code__
    key = (code.co_filename, code.co_firstlineno, code)
    if key not in _MEMO:
        fd = _definition(fn)
        fd.decorator_list = []
        fd.body = Rewriter().stmts(fd.body, set())
        mk = ast.FunctionDef("__eagle_make", ast.arguments(posonlyargs=[], args=[ast.arg(n) for n in (*code.co_freevars, "_eagle_rt")],
                             kwonlyargs=[], kw_defaults=[], defaults=[]), [fd, ast.Return(ast.Name(fd.name, ast.Load()))], [], None)
        mod = ast.Module([mk], [])
        for nd in (mk, mk.body[1]):
            nd.lineno, nd.col_offset, nd.end_lineno, nd.end_col_offset = fd.lineno, 0, fd.end_lineno, 0
        ast.fix_missing_locations(mod)
        ast.increment_lineno(mod, code.co_firstlineno - 1)
        _MEMO[key] = compile(mod, code.co_filename, "exec")
    local = {}
    exec(_MEMO[key], fn.__globals__, local)
    new = local["__eagle_make"](*(c.cell_contents for c in (fn.__closure__ or ())), RT)
    new.__defaults__, new.__kwdefaults__ = fn.__defaults__, fn.__kwdefaults__
    return new


# ----------------------------------------------------------------------------- forks
class Fork:
    """``with g.parallel() as p``: opens a fork; leaving the ``with`` joins."""

    def __init__(self, rec): self.rec, self.branches, self.parent, self.closed = rec, [], rec.cur, False

    def __enter__(self): return self

    def __exit__(self, et, ev, tb):
        self.closed = True
        if et is None and self.branches: self.parent.append(Frk(self.branches))

    def branch(self):
        if self.closed:
            raise RuntimeError("eagle.Graph.branch: the parallel region is closed")
        return _BranchCM(self)


class _BranchCM:
    def __init__(self, fork): self.fork = fork

    def __enter__(self):
        rec = self.fork.rec
        self.b = Branch(self.fork, rec)
        self.fork.branches.append(self.b)
        rec.scopes.append(Scope(self.b.ops, self.b, "branch"))
        return self.b

    def __exit__(self, et, ev, tb): self.fork.rec.scopes.pop()


class Branch(Graph):
    """A branch of a fork: the same vocabulary as a Graph at every depth (``parallel``, blocks, ``wait``)."""

    def __init__(self, fork, rec):
        with _recording.recording(None):
            Graph.__init__(self)
        self.fork, self.rec, self.ops = fork, rec, []

    def wait(self, *planes):
        mine = _uniq([a for o in self.ops if not isinstance(o, Wt) for a in arrays(o)[1]])
        earlier = _uniq([a for b in self.fork.branches if b is not self for o in b.ops if not isinstance(o, Wt) for a in arrays(o)[1]])
        for p in planes:
            if any(p is a for a in mine):
                raise ValueError("eagle.Graph.wait: that plane is written by this branch, which is already in order")
            if not any(p is a for a in earlier):
                raise ValueError("eagle.Graph.wait: that plane is not written by an earlier branch of this parallel region: before "
                                 "the fork everything is already ordered, and a later branch cannot be waited for (no cycles)")
        self.ops.append(Wt(list(planes)))
        return self


def parallel(g):
    rec = _recording.current()
    if not isinstance(rec, Rec): raise RuntimeError("eagle.Graph.parallel: only inside a builder or a block")
    if rec.handle() is not g:
        raise RuntimeError("eagle.Graph.parallel: inside a branch, open the fork from the branch handle: `with <branch>.parallel()`")
    return Fork(rec)


# ----------------------------------------------------------------------------- builders and blocks
class State:
    """What a Graph holds for the builder surface."""

    def __init__(self):
        self.result, self.args, self.names = None, {}, ()
        self.stats = {"allocs": 0, "seeds": 0}
        self.buffers, self.timing, self.builder, self.deep = {}, {}, False, set()
        #: ``{id(view): (view, array the graph holds)}``: a lane-shaped view the caller holds of a flat plane.
        self.views = {}


def _is_graph_ann(a):
    if a is None or a is inspect.Parameter.empty: return False
    if isinstance(a, str): return a.replace(" ", "").endswith("Graph")
    return isinstance(a, type) and issubclass(a, Graph)


def make_builder(fn):
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    block = bool(params) and _is_graph_ann(params[0].annotation)
    ret = sig.return_annotation
    if not block and ret is not inspect.Signature.empty and not _is_graph_ann(ret):
        raise TypeError(f"{fn.__name__}: a builder returns its Graph (-> eagle.Graph), not {ret}")
    if any(p.name == "buffer" for p in params):
        raise TypeError(f"{fn.__name__}: `buffer` is reserved")
    holder = {}

    @functools.wraps(fn)
    def wrapper(*args, **kw):
        if "f" not in holder:
            t = time.perf_counter(); holder["f"] = rewrite_function(fn); wrapper.rewrite_ms = (time.perf_counter() - t) * 1e3
        return (run_block if block else run_builder)(fn, holder["f"], sig, args, kw)

    wrapper.rewrite_ms = None
    return wrapper


def run_block(fn, newfn, sig, args, kw):
    rec = _recording.current()
    if not args: raise TypeError(f"{fn.__name__}: a block takes the graph first")
    h = args[0]
    if getattr(h, "_built", False):
        raise RuntimeError(f"{fn.__name__}: 'g' is built; a build is one instantiation: compose blocks inside the builder, before its `return g`")
    if not isinstance(rec, Rec):
        if not isinstance(h, Graph):
            raise RuntimeError(f"{fn.__name__}: a block takes the open eagle.Graph it appends to")
        rec = Rec(h, "block", {}, fn.__name__)
        rec.block_depth = 1
        with _recording.recording(rec):
            out = newfn(*args, **kw)
        lower_into(h, rec.top)
        return out
    if h is not rec.handle():
        raise RuntimeError(f"{fn.__name__}: pass the handle of the scope you are in (the branch handle inside a branch)")
    rec.block_depth += 1
    try:
        return newfn(*args, **kw)
    finally:
        rec.block_depth -= 1


def run_builder(fn, newfn, sig, args, kw):
    if isinstance(_recording.current(), Rec):
        raise RuntimeError(f"{fn.__name__}: a builder inside a builder; make it a block: give it the graph (`def {fn.__name__}(g: eagle.Graph, ...)`)")
    bound = sig.bind(*args, **kw); bound.apply_defaults()
    rec = Rec(None, "builder", bound.arguments, fn.__name__)
    t0 = time.perf_counter()
    with _recording.recording(rec):
        out = newfn(*args, **kw)
    t1 = time.perf_counter()
    g = rec.graph
    if g is None: raise TypeError(f"{fn.__name__}: a builder makes its graph: g = eagle.Graph()")
    if out is not g: raise TypeError(f"{fn.__name__}: a builder returns its graph: `return g`; declare the arrays with `g.result = ...`")
    st = g._bld
    st.args, st.names, st.stats, st.buffers = dict(bound.arguments), tuple(bound.arguments), rec.stats, rec.buffers
    st.builder, st.deep, st.views = True, rec.deep, rec.views
    lower_into(g, rec.top)
    try:
        Graph.build(g)
    except ValueError as e:
        if "siblings of one fork/join group" in str(e):
            raise ValueError("eagle.Graph.build: two branches of one parallel region touch one plane and one writes it: put them "
                             f"in one branch, or order them with <branch>.wait(plane) [{e}]") from None
        raise
    st.timing = dict(record_ms=(t1 - t0) * 1e3, lower_build_ms=(time.perf_counter() - t1) * 1e3)
    return g


def attach(g):
    """Register a Graph made inside a builder (refusing a second, or one inside a block)."""
    rec = _recording.current()
    if not isinstance(rec, Rec): return
    if rec.block_depth:
        raise RuntimeError("eagle.Graph: a block appends to the graph it is given; a new graph is made in a builder")
    if rec.graph is not None:
        raise RuntimeError("eagle.Graph: a builder makes ONE graph")
    rec.graph = g
    rec.scopes[0].handle = g


def bind(g, planes):
    """``Graph.bind``: new arrays for the builder's parameters, in place."""
    st = g._bld
    for name, new in planes.items():
        if name not in st.names:
            raise ValueError(f"eagle.Graph.bind: {name!r} is not a builder parameter")
        old = st.args[name]
        if not hasattr(old, "shape"):
            raise ValueError(f"eagle.Graph.bind: {name!r} is a Python value, decided at build: call the builder again")
        if not hasattr(new, "shape") or new.shape != old.shape or new.dtype != old.dtype:
            raise ValueError(f"eagle.Graph.bind: {name!r} changed shape/dtype: call the builder again")
    g._rebind([(st.args[n], new) for n, new in planes.items()])
    st.args.update(planes)
    return g
