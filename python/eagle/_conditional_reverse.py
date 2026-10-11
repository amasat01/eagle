# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Reversing ``eagle.skippable`` and ``eagle.repeat_while`` nodes.

A guard is never evaluated in the reverse. The forward saved the cells a
skippable's guard reads (a plane of the record), and a repeat_while's
iteration count is a plane its node writes; the reverse turns those into
``uint32`` flags it owns, and every step that must run only when the member
ran is a skippable over a flag.

A skipped member leaves its planes as they were: the adjoint of a plane it
updates in place passes through, every other adjoint is zero.

A repeat_while repeats its body ``cap`` times at most, so its reverse has
``cap`` flagged copies of the body's vjp, newest first, each seeing the state
its iteration started with. Those states are rebuilt from checkpoints kept
every ``window`` iterations (one forward sweep, then one window at a time),
on private copies, so the recorded planes are never written. The adjoint of
a plane the body only reads sums the iterations' terms oldest first, the
order of the same chain written out as separate nodes.
"""

from __future__ import annotations

import numpy

from . import _core
from ._active_set import _address
from ._conditional import SkipGuard
from .graph import Node, get_node_kind
from .roles import STATE_ROLES


def _same(a, b) -> bool:
    return a is b or _address(a) == _address(b)


def _body(child, ctx):
    if child.kind != "launch":
        raise ValueError(
            f"eagle.diff.Reverse: node {ctx.label!r} repeats a {child.kind!r} step; a "
            "repeat_while reverses only a plan launch as its body")
    return child.payload


def _in_place(child):
    """``{read name: written name}`` for the planes the body updates in place
    (one array under two names), and the names it only reads or only writes."""
    kind = get_node_kind(child.kind)
    reads = {n: a for n, a, role in kind.reads(child) if role not in STATE_ROLES}
    writes = {n: a for n, a, _r in kind.writes(child)}
    pairs = {r: w for r, ra in reads.items() for w, wa in writes.items() if _same(ra, wa)}
    return pairs, [r for r in reads if r not in pairs], [w for w in writes if w not in pairs.values()]


def _flag_cell(ctx, xp, count, baseline):
    """A one-element flag the reverse owns, set from the saved guard cells."""
    flag = xp.zeros(1, dtype=xp.uint32)
    if baseline is None:
        def step():
            xp.not_equal(count, 0, out=flag)
        ctx.add(step, reads=[count], writes=[flag])
    else:
        def step():
            xp.not_equal(count, baseline, out=flag)
        ctx.add(step, reads=[count, baseline], writes=[flag])
    return SkipGuard(flag, 0)


def _insert_front(ctx, start, add):
    """Run ``add()`` (which appends steps) and move what it added to ``start``."""
    mark = len(ctx._steps)
    add()
    added = ctx._steps[mark:]
    del ctx._steps[mark:]
    ctx._steps[start:start] = added


def _skippable(node, incoming, ctx):
    child = node.payload.child
    kind = get_node_kind(child.kind)
    count = ctx.recorded("guard")
    baseline = ctx.recorded("guard_baseline") if node.payload.wrapper.guard.baseline_arr is not None else None
    xp = _numpy_or_cupy(count)
    start = len(ctx._steps)
    guard = _flag_cell(ctx, xp, count, baseline)
    with ctx.gated(guard):
        result = kind.transpose(child, incoming, ctx)
    pairs, _reads, outs = _in_place(child)

    def skipped():
        for name, buf in result.items():
            if name in pairs:
                src = incoming[pairs[name]]
                ctx.add(lambda d=buf, s=src: xp.copyto(d, s), reads=[src], writes=[buf])
            else:
                ctx.add(lambda d=buf: d.fill(0), writes=[buf])
    _insert_front(ctx, start + 1, skipped)
    for w in outs:
        if w in ctx.reads:      # skipped, the adjoint goes on to the plane's earlier writer
            carry = result[w] = xp.empty_like(incoming[w])
            _insert_front(ctx, start + 1, lambda: _copy(ctx, xp, carry, incoming[w]))
            with ctx.gated(guard):
                ctx.add(lambda b=carry: b.fill(0), writes=[carry])
    return result


def _numpy_or_cupy(arr):
    if type(arr).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    return numpy


def _copy(ctx, xp, dst, src):
    ctx.add(lambda: xp.copyto(dst, src), reads=[src], writes=[dst])


def _repeat_while(node, incoming, ctx):
    cond = node.payload
    cap = cond.wrapper.max_iters
    child = cond.child
    bound = _body(child, ctx)
    kind = get_node_kind("launch")
    pairs, consts, outs = _in_place(child)
    names = {n: a for n, a, _r in kind.reads(child)}
    xp = bound._cupy or numpy
    count = ctx.written("iterations")

    window = min(cap, int(_core.diff_default_window(cap)))
    groups = -(-cap // window)
    like = lambda r: xp.empty_like(names[r])
    cur = {r: like(r) for r in pairs}
    ckpt = [{r: like(r) for r in pairs} for _ in range(groups)]
    win = [{r: like(r) for r in pairs} for _ in range(window)]
    # an integer state plane (a counter) is re-run state, never differentiated
    adjoint = {r: w for r, w in pairs.items() if names[r].dtype.kind == "f"}
    abar = {r: xp.empty_like(incoming[w]) for r, w in adjoint.items()}
    pbar = {w: xp.empty_like(incoming[w]) for w in outs}

    # the body on private copies of the state it updates
    private = dict(bound._planes)
    for r, w in pairs.items():
        private[r] = private[w] = cur[r]
    for w in outs:
        private[w] = xp.empty_like(incoming[w])
    for r in consts:
        private[r] = ctx.primal(r)
    advance = bound.plan.bind(**private).launch
    scratch = [*cur.values(), *(private[w] for w in outs)]

    live = xp.zeros(cap, dtype=xp.uint32)
    index = xp.arange(cap, dtype=xp.uint32)
    ran = count[1:2]

    def flag():
        xp.greater(ran, index, out=live)
    ctx.add(flag, reads=[count], writes=[live])
    on = [SkipGuard(live, j) for j in range(cap)]

    def save(dst, src, j):
        with ctx.gated(on[j]):
            for r in pairs:
                _copy(ctx, xp, dst[r], src[r])

    def step(j):
        with ctx.gated(on[j]):
            ctx.add(advance, reads=scratch, writes=scratch)

    for r in pairs:
        _copy(ctx, xp, cur[r], ctx.primal(r))
    for r in adjoint:
        _copy(ctx, xp, abar[r], incoming[adjoint[r]])
    for w in outs:
        _copy(ctx, xp, pbar[w], incoming[w])
    last = (groups - 1) * window
    for j in range(last + 1):                  # checkpoints, oldest first
        if j % window == 0:
            save(ckpt[j // window], cur, j)
        if j < last:
            step(j)

    zeroed, terms = [], {r: [] for r in consts}
    result = {}
    for g in reversed(range(groups)):
        first, end = g * window, min((g + 1) * window, cap)
        save(cur, ckpt[g], first)
        for j in range(first, end):             # the window's start states
            save(win[j - first], cur, j)
            if j < end - 1:
                step(j)
        for j in reversed(range(first, end)):
            states = win[j - first]
            sub = type(ctx)(ctx.backend, ctx.label, ctx.vjp,
                            lambda n, states=states: states[n] if n in states else ctx.primal(n))
            inc = {w: abar[r] for r, w in adjoint.items()} | pbar
            with sub.gated(on[j]):
                out = get_node_kind("launch").transpose(child, inc, sub)
            ctx._steps.extend(sub._steps)
            ctx.inexact |= sub.inexact
            for r in consts:
                if r in out:
                    terms[r].append((j, out[r]))
                    zeroed.append(out[r])
            with ctx.gated(on[j]):
                for r in adjoint:
                    _copy(ctx, xp, abar[r], out[r])
                for w in outs:
                    ctx.add(lambda b=pbar[w]: b.fill(0), writes=[pbar[w]])
    for r in adjoint:
        result[r] = abar[r]
    for w in outs:
        if w in ctx.reads:      # no iteration ran: the adjoint goes on to the plane's earlier writer
            carry = result[w] = xp.empty_like(incoming[w])
            _copy(ctx, xp, carry, incoming[w])
            with ctx.gated(on[0]):
                ctx.add(lambda b=carry: b.fill(0), writes=[carry])
    _insert_front(ctx, 0, lambda: [ctx.add(lambda b=b: b.fill(0), writes=[b]) for b in zeroed])
    for r, parts in terms.items():
        arrays = [a for _j, a in sorted(parts, key=lambda p: p[0])]
        if len(arrays) == 1:
            result[r] = arrays[0]
            continue
        from .diff._reverse import _ordered_sum

        total = xp.empty_like(arrays[0])
        ctx.add(_ordered_sum(total, arrays), reads=arrays, writes=[total])
        result[r] = total
    return result


def transpose(node, incoming, ctx):
    """The reverse hook of the ``skippable`` and ``repeat_while`` node kinds."""
    return (_skippable if node.kind == "skippable" else _repeat_while)(node, incoming, ctx)
