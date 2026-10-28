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
order of the same chain written out as separate nodes. A body of several
launches (a ``sequence`` node) is reversed the same way, each iteration's
parts newest first, each seeing the state it started with; a part that is
not a plan launch cannot be re-run on those private copies and is refused.
"""

from __future__ import annotations

from . import _alloc, _core, _stream, arrays
from ._active_set import _address
from ._conditional import SkipGuard
from .graph import get_node_kind
from .roles import STATE_ROLES


def _zero(buf):
    """Zero ``buf`` on the stream the step runs on (a capture's, in a captured graph)."""
    buf.fill(0, _stream.current())


def _same(a, b) -> bool:
    return a is b or _address(a) == _address(b)


def _in_place(child):
    """``{read name: written name}`` for the planes the body updates in place
    (one array under two names), and the names it only reads or only writes."""
    kind = get_node_kind(child.kind)
    reads = {n: a for n, a, role in kind.reads(child) if role not in STATE_ROLES}
    writes = {n: a for n, a, _r in kind.writes(child)}
    pairs = {r: w for r, ra in reads.items() for w, wa in writes.items() if _same(ra, wa)}
    return pairs, [r for r in reads if r not in pairs], [w for w in writes if w not in pairs.values()]


def _flag_cell(ctx, xp, count, baseline, negate=False):
    """A one-element flag the reverse owns, set from the saved guard cells."""
    differ = arrays.equal if negate else arrays.not_equal
    flag = _alloc.zeros(xp, 1, dtype=arrays.uint32)
    raw = _alloc.zeros(xp, 1, dtype=arrays.uint8)       # a comparison writes bytes; the flag is a uint32
    if baseline is None:
        def step():
            differ(count, 0, out=raw)
            arrays.copyto(flag, raw)
        ctx.add(step, reads=[count], writes=[flag])
    else:
        def step():
            differ(count, baseline, out=raw)
            arrays.copyto(flag, raw)
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
    if kind.transpose is None:
        raise ValueError(f"eagle.diff.Reverse: node {ctx.label!r} guards a {child.kind!r} "
                         "step, which has no transpose; it cannot be reversed")
    count = ctx.recorded("guard")
    baseline = ctx.recorded("guard_baseline") if node.payload.wrapper.guard.baseline_arr is not None else None
    xp = count.device
    start = len(ctx._steps)
    guard = _flag_cell(ctx, xp, count, baseline, node.payload.wrapper.guard.negate)
    with ctx.gated(guard):
        result = kind.transpose(child, incoming, ctx)
    pairs, _reads, outs = _in_place(child)

    def skipped():
        for name, buf in result.items():
            if name in pairs:
                src = incoming[pairs[name]]
                ctx.add(lambda d=buf, s=src: arrays.copyto(d, s), reads=[src], writes=[buf])
            else:
                ctx.add(lambda d=buf: _zero(d), writes=[buf])
            lo = ctx.comp.get(id(buf))      # a row sum's companion rows
            if lo is not None:
                ctx.add(lambda d=lo: _zero(d), writes=[lo])
    _insert_front(ctx, start + 1, skipped)
    for w in outs:
        if w in ctx.reads:      # skipped, the adjoint goes on to the plane's earlier writer
            carry = result[w] = _alloc.empty_like(xp, incoming[w])
            _insert_front(ctx, start + 1, lambda: _copy(ctx, xp, carry, incoming[w]))
            with ctx.gated(guard):
                ctx.add(lambda b=carry: _zero(b), writes=[carry])
    return result


def _copy(ctx, xp, dst, src):
    ctx.add(lambda: arrays.copyto(dst, src), reads=[src], writes=[dst])


def _sum_terms(ctx, xp, terms):
    """``{name: total}`` of each loop constant's per-iteration terms, summed
    oldest first (one fixed order, as the same chain written out)."""
    from .diff._reverse import _ordered_sum

    result = {}
    for r, parts in terms.items():
        terms_in = [a for _k, a in sorted(parts, key=lambda p: p[0])]
        if not terms_in:
            continue
        if len(terms_in) == 1:
            result[r] = terms_in[0]
            continue
        total = _alloc.empty_like(xp, terms_in[0])
        ctx.add(_ordered_sum(total, terms_in), reads=terms_in, writes=[total])
        if any(id(a) in ctx.comp for a in terms_in):
            los = [ctx.comp.get(id(a)) for a in terms_in]
            los = [lo if lo is not None else _alloc.zeros_like(xp, terms_in[0]) for lo in los]
            ctx.comp[id(total)] = lo_total = _alloc.empty_like(xp, total)
            ctx.add(_ordered_sum(lo_total, los), reads=los, writes=[lo_total])
        result[r] = total
    return result


def _repeat_while(node, incoming, ctx):
    """The reverse of a ``repeat_while`` whose body is a plan launch, or a
    ``sequence`` of them (a part that is not differentiated, a guard writer,
    is re-run and not transposed). A lone launch is a one-part sequence whose
    planes keep their own names."""
    from .diff._reverse import _vjp_plan

    cond = node.payload
    cap = cond.wrapper.max_iters
    if cap is None:
        raise ValueError(f"eagle.diff.Reverse: node {ctx.label!r} is a repeat_while with no "
                         "max_iters; a reverse pass needs the cap")
    single = cond.child.kind != "sequence"
    parts = (cond.child,) if single else cond.child.payload
    bad = [p for p in parts if p.kind != "launch"]
    if bad:
        raise ValueError(
            f"eagle.diff.Reverse: node {ctx.label!r} repeats a {bad[0].kind!r} " +
            ("step; a repeat_while reverses a plan launch, or several as a sequence" if single else
             "part; a reversed loop body holds plan launches only (a decision cell cannot be re-run)"))
    key = (lambda i, n: n) if single else (lambda i, n: f"{i}.{n}")
    launch = get_node_kind("launch")
    access = [(launch.reads(p), launch.writes(p)) for p in parts]
    device = parts[0].payload._device
    xp = "cpu" if device is None else f"cuda:{device}"
    written = {}
    for _reads, writes in access:
        for _n, arr, _r in writes:
            written.setdefault(_address(arr), arr)
    floats = {ad for ad, arr in written.items() if arr.dtype.kind == "f"}
    fresh = lambda: {ad: _alloc.empty_like(xp, arr) for ad, arr in written.items()}  # noqa: E731
    window = min(cap, int(_core.diff_default_window(cap)))
    groups = -(-cap // window)
    cur, before = fresh(), [fresh() for _ in parts]
    ckpt, win = [fresh() for _ in range(groups)], [fresh() for _ in range(window)]

    def state_of(i, name, states):
        arr = next(a for n, a, _r in (*access[i][0], *access[i][1]) if n == name)
        return states[_address(arr)] if _address(arr) in states else ctx.primal(key(i, name))

    def private(i, states):
        planes = dict(parts[i].payload._planes)
        for n, _a, _r in (*access[i][0], *access[i][1]):
            planes[n] = state_of(i, n, states)
        return parts[i].payload.plan.bind(**planes).launch
    advance = [private(i, cur) for i in range(len(parts))]
    scratch = list(cur.values())

    live = _alloc.zeros(xp, cap, dtype=arrays.uint32)
    ran = ctx.written("iterations")[1:2]
    index = arrays.arange(cap, dtype=ran.dtype, device=xp)
    ran_all, hit = _alloc.empty(xp, cap, dtype=ran.dtype), _alloc.empty(xp, cap, dtype=arrays.uint8)

    def count_live():
        arrays.copyto(ran_all, arrays.view_as(ran, (cap,), (0,)))      # one value, repeated: no temporary
        arrays.greater(ran_all, index, out=hit)
        arrays.copyto(live, hit)
    ctx.add(count_live, reads=[ran], writes=[live])
    on = [SkipGuard(live, j) for j in range(cap)]

    def save(dst, src, j):
        with ctx.gated(on[j]):
            for ad in written:
                _copy(ctx, xp, dst[ad], src[ad])

    def step(j):
        with ctx.gated(on[j]):
            for run in advance:
                ctx.add(run, reads=scratch, writes=scratch)

    taped = {}      # array address -> the loop node's name for it (taped at entry)
    for i, (reads, _w) in enumerate(access):
        for n, arr, _r in reads:
            if key(i, n) in ctx.reads:
                taped.setdefault(_address(arr), key(i, n))
    for ad, name in taped.items():
        if ad in written:
            _copy(ctx, xp, cur[ad], ctx.primal(name))
    adjoint = {ad: _alloc.zeros_like(xp, written[ad]) for ad in floats}
    for i, (_r, writes) in enumerate(access):
        for n, arr, _ in writes:
            if key(i, n) in incoming and _address(arr) in adjoint:
                _copy(ctx, xp, adjoint[_address(arr)], incoming[key(i, n)])

    last = (groups - 1) * window
    for j in range(last + 1):                  # checkpoints, oldest first
        if j % window == 0:
            save(ckpt[j // window], cur, j)
        if j < last:
            step(j)
    terms, zeroed = {}, []
    for g in reversed(range(groups)):
        first, end = g * window, min((g + 1) * window, cap)
        save(cur, ckpt[g], first)
        for j in range(first, end):             # the window's start states
            save(win[j - first], cur, j)
            if j < end - 1:
                step(j)
        for j in reversed(range(first, end)):
            save(cur, win[j - first], j)
            for i, run in enumerate(advance):   # the state each part starts with
                save(before[i], cur, j)
                with ctx.gated(on[j]):
                    ctx.add(run, reads=scratch, writes=scratch)
            for i in reversed(range(len(parts))):
                reads, writes = access[i]
                if parts[i].grad:
                    sub = type(ctx)(ctx.backend, ctx.label,
                                    ctx.vjp if single else _vjp_plan(f"{ctx.label}[{i}]", parts[i].payload, None),
                                    lambda n, i=i: state_of(i, n, before[i]))
                    inc = {n: adjoint[_address(a)] for n, a, _r in writes if _address(a) in adjoint}
                    with sub.gated(on[j]):
                        out = launch.transpose(parts[i], inc, sub)
                    ctx._steps.extend(sub._steps)
                    ctx.inexact |= sub.inexact
                    ctx.comp.update(sub.comp)
                else:
                    out = {}
                with ctx.gated(on[j]):
                    for _n, a, _r in writes:    # overwritten: no adjoint before this part
                        if _address(a) in adjoint:
                            ctx.add(lambda b=adjoint[_address(a)]: _zero(b), writes=[adjoint[_address(a)]])
                    for n, a, _r in reads:
                        if n not in out:
                            continue
                        if _address(a) in adjoint:
                            t = adjoint[_address(a)]
                            ctx.add(lambda t=t, o=out[n]: arrays.add(t, o, out=t), reads=[t, out[n]], writes=[t])
                        else:                   # a loop constant: one term per (iteration, part)
                            terms.setdefault(_address(a), []).append(((j, i), out[n]))
                            zeroed += [out[n], ctx.comp[id(out[n])]] if id(out[n]) in ctx.comp else [out[n]]
    _insert_front(ctx, 0, lambda: [ctx.add(lambda b=b: _zero(b), writes=[b]) for b in zeroed])
    totals = _sum_terms(ctx, xp, terms)
    result = {name: adjoint[ad] if ad in adjoint else totals[ad]
              for ad, name in taped.items() if ad in adjoint or ad in totals}
    for i, (_r, writes) in enumerate(access):
        for n, arr, _ in writes:
            name = key(i, n)
            if name in ctx.reads and name in incoming and name not in result:
                # no iteration ran: the adjoint goes on to the plane's earlier writer
                carry = result[name] = _alloc.empty_like(xp, incoming[name])
                _copy(ctx, xp, carry, incoming[name])
                with ctx.gated(on[0]):
                    ctx.add(lambda b=carry: _zero(b), writes=[carry])
    return result


def transpose(node, incoming, ctx):
    """The reverse hook of the ``skippable`` and ``repeat_while`` node kinds."""
    return (_skippable if node.kind == "skippable" else _repeat_while)(node, incoming, ctx)
