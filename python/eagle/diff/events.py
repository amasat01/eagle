# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Event refinement: ``events="refine"`` on a run-until-done forward.

A stepped run stops a sample at the first step whose stop rule fires, so its
stop time is a multiple of the step and its derivative is zero. Refinement
locates the crossing INSIDE that step. The kernel's stop rule is read as a
smooth guard ``g`` (``hawk.diff.event``: the sample stops where ``g <= 0``),
the kernel declares its time plane and step (``@hawk.kernel(time="t",
time_step="dt")``), and the run:

* steps with the closed, tolerant stop test of hawk's taping twin, its one
  tape row switched on: the row a sample wrote last is the state it held
  before its stop step, so nothing is recomputed;
* evaluates hawk's refine entry ``<name>_refine`` (one partial step of
  per-sample size ``tau``, with the guard and its ``tau``-derivative) and
  locates ``tau*`` in ``[0, h]`` by Newton with a bisection fallback, at most
  8 rounds (``eagle/events/Refine.h``); a crossing within tolerance of the
  step's end keeps the stepped state (``tau* = h``);
* writes the refined state over the stepped one, ``t_stop`` and the
  ``event_status`` plane.

The reverse (``eagle.diff.reverse`` of a ``record=True`` run, no extra flag)
adds the implicit-function correction for the moved stop time: one vjp of the
refine entry seeded with the end-state adjoint and a rank-1 guard seed
(``eagle/events/Sensitivity.h``), then the step-by-step reverse from the
state before the stop step.

Refused by name (``RefineError`` messages are pinned by the tests): a sample
with no crossing in its stop step (pass ``on_no_event="zero"`` to take a zero
event sensitivity for those), a grazing crossing, two stop rules crossing in
one step. Thread-safety: a :class:`RefinePass` belongs to one runner and runs
under that runner's record slot; distinct runners refine concurrently.
"""

from __future__ import annotations

import numpy

from .. import _core
from ._record import xp_of

__all__ = ["phase"]

#: ``event_status`` values (``eagle::events::Status``).
PENDING, REFINED, AT_STEP, NO_EVENT, GRAZE, TWO_RULES = range(6)
#: ``eagle::events::Op``.
_CLASSIFY, _NEWTON, _SETTLE, _GRAZE, _COMMIT, _SEED, _SELECT = range(7)

EVENTS = ("step", "refine")
ON_NO_EVENT = ("raise", "zero")
_REFINED = "_refined"
_DOT = "dot_"
_BAR = "bar_"
_GUARD = "event_g"


def check_events(events, on_no_event, door):
    """Refuse an unknown ``events=`` / ``on_no_event=`` value by name."""
    if events not in EVENTS:
        raise ValueError(f"{door}: events is 'step' (the default: stop at the step "
                         f"the rule fires) or 'refine' (locate the crossing inside "
                         f"it), got {events!r}")
    if on_no_event not in ON_NO_EVENT:
        raise ValueError(f"{door}: on_no_event is 'raise' (the default) or 'zero', "
                         f"got {on_no_event!r}")
    if on_no_event != "raise" and events != "refine":
        raise ValueError(f'{door}: on_no_event= applies to events="refine" only')


def _addr(a):
    return int(a.data.ptr) if hasattr(a, "data") and hasattr(a.data, "ptr") else int(a.ctypes.data)


def _rows(a):
    """The row addresses of a sample-last plane (one per component)."""
    n = a.shape[-1]
    base, step = _addr(a), n * a.itemsize
    return [base + r * step for r in range(max(a.size // max(n, 1), 1))]


class EventKernels:
    """The refine entry's builds for one plugin and device: ``newton`` (the
    refine entry plus its ``tau``-derivative), ``vjp`` (the reverse of the
    refine entry) and ``guard_vjp`` (``g_x`` at the refined state)."""

    __slots__ = ("newton", "vjp", "guard_vjp", "guards", "state", "tau", "time_step",
                 "time", "t0")


def event_kernels(plugin, device, door):
    """The :class:`EventKernels` of ``plugin`` (cached on it)."""
    cache = getattr(plugin, "_events", None)
    if cache is None:
        cache = plugin._events = {}
    if device in cache:
        return cache[device]
    source = getattr(plugin, "_hawk_source", None)
    if source is None:
        raise ValueError(f'{door}: events="refine" needs the hawk kernel behind the plan; '
                         "pass the kernel (eagle.simulate) or a plan made by "
                         "eagle.plan.auto(kernel)")
    kernel, _unit, cache_dir = source
    from hawk import Kernel
    from hawk.diff import jvp, refine_entry, vjp
    from hawk.diff.event import event_guard, state_guard
    from hawk.diff.tape import T0_SUFFIX, TAU_SUFFIX

    from .._plan_deploy import _hawk_plugins

    step = getattr(kernel, "step", None) or getattr(kernel, "one_step", None) or kernel
    if getattr(step, "time_step", None) is None:
        raise ValueError(f'{door}: events="refine" steps by a partial step; declare the '
                         f"time plane and step of {step.name!r} with "
                         '@hawk.kernel(time="t", time_step="dt")')
    guards = len(event_guard(kernel).guards)   # refuses a stop rule with no smooth crossing
    # the refine kernels run over every sample: plain, whatever the step's kind
    plain = lambda k: Kernel(k.name, tuple(k.sinks), {})
    entry = plain(refine_entry(kernel))
    tau = step.time_step + TAU_SUFFIX
    newton = Kernel(f"{step.name}_newton", tuple(entry.sinks) + tuple(jvp(entry, wrt=(tau,))), {})
    back = Kernel(f"{entry.name}_vjp", vjp(entry))
    sg = plain(state_guard(kernel))
    sg_back = Kernel(f"{sg.name}_vjp", vjp(sg))
    targets = ("cuda",) if device else ("host",)
    built = _hawk_plugins([newton, entry, back, sg, sg_back], targets, cache_dir,
                          plugin.scalar_type)
    out = EventKernels()
    out.newton, out.vjp, out.guard_vjp = built[0], built[2], built[4]
    out.guards = guards
    out.state = tuple(nm[:-len(_REFINED)] for _r, nm in entry.walk.arg_spec
                      if nm.endswith(_REFINED))
    out.tau, out.time_step = tau, step.time_step
    out.time = step.time or "t"
    out.t0 = out.time + T0_SUFFIX
    cache[device] = out
    return out


class RefinePass:
    """The refinement of one runner's forward: its planes and bindings, made
    once, run after every forward (see the module docstring)."""

    def __init__(self, rec, on_no_event):
        from ..plan import plan as make_plan

        self.rec = rec
        self.on_no_event = on_no_event
        xp, n, door = rec.xp, rec.n, rec.door
        self.xp, self.n, self.device = xp, n, rec.device
        self.k = k = event_kernels(rec.plan.plugin, rec.device, door)
        if k.time not in rec.planes:
            raise ValueError(f'{door}: events="refine" reports the event time from the time '
                             f"plane {k.time!r}; bind it (its start values, e.g. zeros)")
        scalar = rec.scalar
        self.bytes = scalar.itemsize
        offsets, off = {}, 0
        for nm, rows in rec.fwd_kernels.taped:
            offsets[nm] = (off, rows)
            off += rows
        tape = rec.planes["tape"]
        # the state before each sample's stop step: the one tape row it wrote last
        self.prestop = {nm: tape[lo:lo + rows].reshape(rec.state[nm].shape)
                        for nm, (lo, rows) in offsets.items()}
        self.steps = xp.zeros(n, dtype=rec.planes["applied_steps"].dtype)
        self.tau = xp.zeros(n, dtype=scalar)
        self.lo, self.hi = xp.zeros(n, dtype=scalar), xp.zeros(n, dtype=scalar)
        self.status = xp.zeros(n, dtype=xp.int8)
        self.which = xp.zeros(n, dtype=xp.int8)
        self.t_stop = xp.zeros(n, dtype=scalar)
        self.h = rec.planes[k.time_step]
        self.out = {}
        bind = {}
        for role, nm in k.newton.arg_spec:
            if role == "mutable":
                base = nm[len(_DOT):] if nm.startswith(_DOT) else nm
                base = base[:-len(_REFINED)] if base.endswith(_REFINED) else base
                like = rec.state.get(base)
                self.out[nm] = (xp.zeros_like(like) if like is not None
                                else xp.zeros(n, dtype=scalar))
                bind[nm] = self.out[nm]
            elif nm == "applied_steps":
                bind[nm] = self.steps
            elif nm == k.tau:
                bind[nm] = self.tau
            elif nm == _DOT + k.tau:
                bind[nm] = xp.ones(n, dtype=scalar)
            elif nm in self.prestop:
                bind[nm] = self.prestop[nm]
            else:
                bind[nm] = rec.planes[nm]
        self.newton = make_plan(k.newton, structure=rec.structure).bind(**bind)
        self.g = [self.out[f"{_GUARD}{j}"] for j in range(k.guards)]
        self.dg = [self.out[f"{_DOT}{_GUARD}{j}"] for j in range(k.guards)]
        self.refined = {nm: self.out[nm + _REFINED] for nm in k.state}
        self.phi_tau = {nm: self.out[_DOT + nm + _REFINED] for nm in k.state}
        # g_x at the refined state, for the graze test
        gb, self.gx = {}, {}
        self.seeds = [xp.zeros(n, dtype=scalar) for _ in range(k.guards)]
        for role, nm in k.guard_vjp.arg_spec:
            if nm.startswith(_BAR + _GUARD):
                gb[nm] = self.seeds[int(nm[len(_BAR + _GUARD):])]
            elif nm.startswith(_BAR):
                base = nm[len(_BAR):]
                like = rec.state.get(base)
                gb[nm] = xp.zeros_like(like) if like is not None else xp.zeros(n, dtype=scalar)
                if base in self.refined:
                    self.gx[base] = gb[nm]
            elif nm in self.refined:
                gb[nm] = self.refined[nm]
            else:
                gb[nm] = rec.planes[nm]
        self.guard_vjp = make_plan(k.guard_vjp, structure=rec.structure).bind(**gb)
        self.zero = xp.zeros(n, dtype=scalar)

    def _scalars(self):
        rec = self.rec
        return [_addr(rec.runner.terminated), _addr(rec.planes["applied_steps"]),
                _addr(rec.planes[self.k.t0]), _addr(self.h), _addr(self.status),
                _addr(self.which), _addr(self.tau), _addr(self.lo), _addr(self.hi),
                _addr(self.t_stop)]

    def _args(self, *, g=(), dg=(), gbar=(), a=(), b=(), c=()):
        return (self.n, [_addr(x) for x in g], [_addr(x) for x in dg], [_addr(x) for x in gbar],
                list(a), list(b), list(c), self._scalars())

    def _apply(self, op, **planes):
        args = self._args(**planes)
        if self.device:
            _core.events_apply_device(op, self.bytes, args,
                                      int(self.xp.cuda.get_current_stream().ptr), -1)
        else:
            _core.events_apply_host(op, self.bytes, args)

    def run(self):
        """Refine the forward that just ran; raise the pinned refusals."""
        xp, rec, b = self.xp, self.rec, self.newton
        xp.subtract(rec.planes["applied_steps"], 1, out=self.steps)
        xp.maximum(self.steps, 0, out=self.steps)
        xp.copyto(self.tau, self.h)          # the first evaluation is the step's end
        args = self._args(g=self.g, dg=self.dg)
        if self.device:
            import importlib

            launch = importlib.import_module("eagle.launch")
            block = int(launch._resolved_block(b._entry, self.n, None))
            _core.events_refine_device(b._entry, b._addrs, block, self.bytes, args,
                                       int(xp.cuda.get_current_stream().ptr), -1)
        else:
            _core.events_refine_host(b._entry, b._addrs, self.bytes, args)
        for j, s in enumerate(self.seeds):
            xp.copyto(s, (self.which == j).astype(s.dtype))
        self.guard_vjp.launch()
        rows = lambda planes: [r for nm in self.k.state for r in _rows(planes[nm])]
        gx = {nm: self.gx.get(nm, None) for nm in self.k.state}
        b_rows = [r for nm in self.k.state
                  for r in (_rows(gx[nm]) if gx[nm] is not None
                            else [_addr(self.zero)] * len(_rows(self.phi_tau[nm])))]
        self._apply(_GRAZE, g=self.g, dg=self.dg, b=b_rows, c=rows(self.phi_tau))
        self._refuse()
        self._apply(_COMMIT, a=rows(rec.state), b=rows(self.refined))

    def _refuse(self):
        status = self.status.get() if self.device else self.status
        for code, text in ((TWO_RULES, None), (GRAZE, "graze"), (NO_EVENT, "none")):
            hit = numpy.flatnonzero(status == code)
            if not hit.size or (code == NO_EVENT and self.on_no_event == "zero"):
                continue
            first = int(hit[0])
            if code == TWO_RULES:
                raise ValueError(f'events="refine": two stop rules cross in the same step '
                                 f"for sample {first}")
            if code == GRAZE:
                raise ValueError(f'events="refine": {hit.size} samples (first: {first}) graze '
                                 "the surface (|dg/dτ| below GRAZE_TOL); the event time is not "
                                 "differentiable there")
            raise ValueError(f'events="refine": {hit.size} samples (first: {first}) have no '
                             "crossing in their stop step; they hit max_steps or jumped the "
                             'surface — pass on_no_event="zero" to take a zero event '
                             "sensitivity for them")

    def held(self):
        """What a reverse reads of this refinement (references, as the
        checkpoints are: the next forward overwrites them)."""
        planes = dict(self.prestop)
        planes.update(steps=self.steps, tau=self.tau, status=self.status, which=self.which)
        planes.update({f"dg{j}": d for j, d in enumerate(self.dg)})
        planes.update({_DOT + nm: p for nm, p in self.phi_tau.items()})
        return planes


class RefineReverse:
    """The implicit-function correction of a refined run's reverse: the vjp
    of the refine entry, bound per call over the Record's refinement."""

    def __init__(self, rec):
        self.rec = rec
        self.k = rec.refine.k

    def seeds(self, tape, seeds, door):
        """``(pre_seeds, steps, extra)``: the adjoint before each located
        sample's stop step, the steps the step-by-step reverse runs, and the
        refine entry's own share of every input gradient."""
        from ..plan import plan as make_plan

        rec, k = self.rec, self.k
        xp, n = rec.xp, rec.n
        ev = tape.events
        xbar = {nm: (xp.asarray(seeds[nm], dtype=rec.state[nm].dtype) if nm in seeds
                     else xp.zeros_like(rec.state[nm])) for nm in k.state}
        rbar = {nm: xp.zeros_like(xbar[nm]) for nm in k.state}
        gbar = [xp.zeros(n, dtype=rec.scalar) for _ in range(k.guards)]
        dg = [ev[f"dg{j}"] for j in range(k.guards)]
        scal = [0, 0, 0, 0, _addr(ev["status"]), _addr(ev["which"]), 0, 0, 0, 0]
        rows = lambda planes: [r for nm in k.state for r in _rows(planes[nm])]
        op = (lambda code, args: _core.events_apply_device(
            code, rec.scalar.itemsize, args, int(xp.cuda.get_current_stream().ptr), -1)
            ) if rec.device else (lambda code, args: _core.events_apply_host(
                code, rec.scalar.itemsize, args))
        op(_SEED, (n, [_addr(x) for x in dg], [_addr(x) for x in dg], [_addr(x) for x in gbar],
                   rows(rbar), rows(xbar), rows({nm: ev[_DOT + nm] for nm in k.state}), scal))
        bind, grads = {}, {}
        for role, nm in k.vjp.arg_spec:
            if nm.startswith(_BAR) and nm.endswith(_REFINED):
                bind[nm] = rbar[nm[len(_BAR):-len(_REFINED)]]
            elif nm.startswith(_BAR + _GUARD):
                bind[nm] = gbar[int(nm[len(_BAR + _GUARD):])]
            elif nm.startswith(_BAR):
                base = nm[len(_BAR):]
                like = ev.get(base, rec.planes.get(base))
                if like is None or not hasattr(like, "shape") or base in rec.uniforms:
                    grads[base] = bind[nm] = xp.zeros(n, dtype=rec.scalar)
                else:
                    grads[base] = bind[nm] = xp.zeros_like(like)
            elif nm == "applied_steps":
                bind[nm] = ev["steps"]
            elif nm == k.tau:
                bind[nm] = ev["tau"]
            elif nm in k.state:
                bind[nm] = ev[nm]
            elif nm in tape.reads:
                bind[nm] = tape.reads[nm]
            else:
                bind[nm] = rec.planes[nm]
        make_plan(k.vjp, structure=rec.structure).bind(**bind).launch()
        pre = {nm: xp.empty_like(xbar[nm]) for nm in k.state}
        op(_SELECT, (n, [], [], [], rows(pre), rows(xbar),
                     rows({nm: grads[nm] if nm in grads else xp.zeros_like(rbar[nm])
                           for nm in k.state}), scal))
        steps = tape.steps.copy()
        status = ev["status"].get() if rec.device else ev["status"]
        status = numpy.ascontiguousarray(status)
        _core.events_reverse_steps(steps.ctypes.data, status.ctypes.data, n)
        extra = {nm: g for nm, g in grads.items() if nm not in k.state and nm != k.tau}
        return pre, steps, extra


def copy_events(events):
    """A private copy of a refinement's held planes (``Record.keep``)."""
    return None if events is None else {k: xp_of(a).array(a, copy=True)
                                        for k, a in events.items()}


# --- phases: a hybrid trajectory as a static Graph of refined runs --- #
_PHASE_COUNT = "a phase count must be static: unroll the phases into the Dag"


def _phase_runner(obj):
    """The refined Runner behind ``obj`` (a phase Node, a Runner or
    Simulation, or one of their bound methods), else ``None``."""
    from ..graph import Node

    if isinstance(obj, Node):
        return obj.payload if obj.kind == "phase" else None
    obj = getattr(obj, "__self__", obj)
    runner = getattr(obj, "runner", obj)
    refine = getattr(getattr(runner, "_recorder", None), "refine", None)
    return runner if refine is not None else None


def refuse_repeated_phase(step):
    """Refuse a refined run phase (or a tuple holding one) as the body of a
    data-dependent loop."""
    parts = step if isinstance(step, tuple) else (step,)
    if any(_phase_runner(p) is not None for p in parts):
        raise ValueError(f"eagle.repeat_while: {_PHASE_COUNT}")


def phase(run):
    """A refined run as one node of an :class:`eagle.Graph`: a hybrid
    trajectory is a static Graph of phases, ``phase(A) -> reset kernel ->
    phase(B) -> ...``, each ``A``/``B`` an :func:`eagle.until_done` runner (or
    :class:`eagle.Simulation`) built with ``events="refine", record=True``
    over the planes the reset kernel reads and writes. Each launch of the
    graph restarts every phase (its finished mask cleared) and runs it to its
    event; :func:`eagle.diff.reverse` of the recorded graph chains the
    phases' event sensitivities through the reset kernels' vjps (the
    saltation of each jump, by the chain rule). The number of phases is the
    graph's: a phase inside :func:`eagle.repeat_while` is refused. On the
    GPU a phase is an eager node (it reads its own progress back between
    launches): the graph is captured in segments around it.

    Example::

        fall = eagle.until_done(plan, max_steps=500, events="refine", record=True,
                                y=y, v=v, t=t)
        g = eagle.Graph()
        g.add_node(eagle.diff.phase(fall)).add(bounce).add_node(eagle.diff.phase(fall2))
        g.build(record=True).launch()
        grads = eagle.diff.reverse(g.record, {"y": dJ_dy})
    """
    from ..graph import Node

    runner = getattr(run, "runner", run)
    rec = getattr(runner, "_recorder", None)
    if rec is None or rec.refine is None or not rec.recording:
        raise ValueError('eagle.diff.phase: a phase is a run built with events="refine", '
                         "record=True (eagle.until_done or eagle.simulation)")
    return Node("phase", runner)


def _phase_planes(node, writing):
    """A phase reads its state (as an input: the graph snapshots it, the
    phase advances it in place) and its per-sample inputs; it writes its
    state."""
    rec = node.payload._recorder
    role = "mutable" if writing else "per_sample"
    out = [(nm, a, role) for nm, a in rec.state.items()
           if rec.roles.get(nm) == "mutable" and nm != "applied_steps"
           and a.dtype.kind == "f"]
    if not writing:
        out += [(nm, rec.planes[nm], "per_sample") for nm in rec.read_names
                if hasattr(rec.planes.get(nm), "shape") and nm not in rec.promoted
                and nm != rec.t0]
    return tuple(out)


def _phase_run(node, ctx):
    """A phase's host call on either backend (on the GPU an eager node: the
    run reads its progress back between launches)."""
    runner = node.payload

    def run():
        runner.reset()
        runner.run()
    return run


def _phase_transpose(node, incoming, ctx):
    """A phase's reverse: the reverse of its own Record, seeded with the
    adjoints of the planes it wrote, copied into fresh adjoint planes of
    the planes it read."""
    from ._run import RunReverse

    runner = node.payload
    rec = runner._recorder
    reverse = RunReverse(rec, "eagle.diff.reverse")
    reads = [nm for nm, _a, _r in _phase_planes(node, False)]
    out = {nm: rec.xp.zeros_like(rec.planes[nm]) for nm in reads}
    seeds = {nm: a for nm, a in incoming.items() if nm in reverse.taped}

    def step():
        grads = reverse(runner.record, seeds, None, "eagle.diff.reverse")
        for nm, buf in out.items():
            if nm in grads:
                rec.xp.copyto(buf, grads[nm])
            else:
                buf.fill(0)

    from ..graph import Node

    named = lambda arrays, p: tuple((f"{p}{i}", a, "declared") for i, a in enumerate(arrays))
    ctx.add(Node("phase_reverse", (step, named(seeds.values(), "in"),
                                   named(out.values(), "out"))))
    return out


def _register():
    from ..graph import register_node_kind

    register_node_kind("phase", reads=lambda n: _phase_planes(n, False),
                       writes=lambda n: _phase_planes(n, True), lower_gpu=_phase_run,
                       lower_cpu=_phase_run, transpose=_phase_transpose, tier="host")
    # a phase's reverse: a host call in the reverse graph, eager on the GPU too
    register_node_kind("phase_reverse", reads=lambda n: n.payload[1],
                       writes=lambda n: n.payload[2], lower_gpu=lambda n, ctx: n.payload[0],
                       lower_cpu=lambda n, ctx: n.payload[0], tier="host")


_register()
