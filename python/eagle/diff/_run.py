# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Recording a run-until-done forward and reversing it, window by window.

A ``record=True`` run steps the batch with the kernel's TAPING TWIN
(``hawk.diff.tape``) with its tape switched off, saving the state at every
window start (steps 0, W, 2W, ...) and counting each sample's steps in
``applied_steps``. Launches never cross a window boundary: a single-step
kernel takes one step per launch, an automatic kernel runs its fused twin
up to the next boundary. The reverse then works window by window, last
first: it loads the window's start state, recomputes the window with the
SAME twin entry (tape on, one row per step), and steps the adjoint back
through the rows with the vjp of one step. The C++ driver
(``eagle/diff/Reverse.h``) sequences it on the host and on the device;
samples run in the reverse's own order, sorted by stop step, so the samples
still running at any step are a prefix.

The reverse holds ``(windows + W)`` states per sample (checkpoints and
tape). Under ``memory_budget`` it runs the batch in passes of ``leaf *
2^q`` samples, each pass a whole subtree of the canonical parameter sum, so
the combined gradients are bitwise those of a single pass.

Thread-safety: the record slot is the C++ ``RecordSlot`` (one mutex). A
:class:`RunReverse` owns its planes and tape and is guarded by a busy flag,
so distinct reverses of one run proceed concurrently.
"""

from __future__ import annotations

import os
import threading

from .. import _alloc, _stream, arrays
from .. import _core, _counters, _split
from .._busy import Busy, BusyGuard
from ..graph import get_node_kind

__all__ = []

APPLIED_STEPS = "applied_steps"
TAPE = "tape"
TAPE_ON = "tape_on"
FUSED_STEPS = "fused_steps"
_STATE = ("mutable", "terminated")
_EPOCH = "slot_samples"   # a checkpoint's slot -> sample map, when compaction reorders
_NEXT = "_next"      # hawk.diff.step_vjp's naming contract (bar_<plane>_next)
_LEAF = 4096         # the canonical reduction's leaf (eagle/reduce/canonical.h)

# Module-level so a row can plant a wrong chunking or combine and see it fail.
_chunk_samples = _core.diff_chunk_samples
_combine = _core.diff_combine_chunks


def _not_yet(door, what):
    return ValueError(f"{door}: record=True {what} is not yet supported; record "
                      "one kernel per step")


def resolve_window(window, max_steps, door):
    """The window W: ``None`` -> clamp(ceil(sqrt(max_steps)), 16, 1024),
    ``"full"`` -> ``max_steps`` (one window), or a positive int."""
    if window is None:
        return int(_core.diff_default_window(max_steps))
    if window == "full":
        return max(int(max_steps), 1)
    if isinstance(window, bool) or not isinstance(window, int) or window < 1:
        raise ValueError(f"{door}: window is a positive int of steps, 'full' or None "
                         f"(the default), got {window!r}")
    return window


def resolve_budget(budget, device, door):
    """``memory_budget`` in bytes: ``None`` -> half the free device memory
    (device runs) or half the available host RAM."""
    if budget is None:
        if device:
            return int(_core.device_mem_info(-1)[1]) // 2
        return int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) // 2
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 1:
        raise ValueError(f"{door}: memory_budget is a positive int of bytes or None "
                         f"(half the free memory), got {budget!r}")
    return budget


def _uniform_plane(value, like):
    """A parameter's recorded value as a plane shaped and placed like ``like``: a scalar fills one."""
    if hasattr(value, "reshape"):
        return value.reshape(like.shape)
    like = arrays.asarray(like)         # the caller's plane (cupy, torch, numpy) as eagle's own view: its dtype and device
    return arrays.full(like.shape, value, like.dtype, like.device)


class _Kernels:
    """The kernels a recorded run needs, built once per window and device:
    the single-step taping twin, the fused twin of an automatic kernel, the
    vjp of one step as a pure function, the taped ``(name, rows)``, and the
    time plane name of a ``time_step=`` kernel."""

    __slots__ = ("single", "fused", "vjp", "taped", "time")


def _step_kernels(plugin, window, door, device, events="step"):
    cache = getattr(plugin, "_tapes", None)
    if cache is None:
        cache = plugin._tapes = {}
    key = (window, device, events)
    if key in cache:
        return cache[key]
    source = getattr(plugin, "_hawk_source", None)
    if source is None:
        raise ValueError(f"{door}: record=True needs the hawk kernel behind the plan to "
                         "build its taping twin; pass the kernel (eagle.simulate) or a "
                         "plan made by eagle.plan.auto(kernel)")
    kernel, _unit, cache_dir = source
    from hawk import Kernel
    from hawk.diff import step_vjp, tape
    from hawk.diff.tape import taped_planes
    from hawk.ir.nodes import element_count

    from .._plan_deploy import _hawk_plugins

    step = getattr(kernel, "step", None) or getattr(kernel, "one_step", None) or kernel
    finish = getattr(step.walk, "finish", None)
    if finish is not None and finish[1] != 1:
        raise _not_yet(door, f"for a kernel taking {finish[1]} steps per launch")
    twin = tape(kernel, window, events=events)
    single = getattr(twin, "one_step", None) or twin
    fused = twin if single is not twin else None
    derived = Kernel(f"{step.name}_step_vjp", step_vjp(kernel))
    targets = ("cuda",) if device else ("host",)
    # the fused twin shares the single twin's name (its word-1 branch IS that
    # step), so a bundle holds one or the other; the step itself is there as
    # the primal the vjp names
    built = _hawk_plugins([fused or single, step, derived], targets, cache_dir,
                          plugin.scalar_type)
    out = _Kernels()
    out.single = None if fused is not None else built[0]
    out.fused = built[0] if fused is not None else None
    out.vjp = built[2]
    out.taped = tuple((name, element_count(t)) for name, t in taped_planes(single))
    out.time = getattr(step, "time", None) if getattr(step, "time_step", None) else None
    cache[key] = out
    return out


def array_module(device):
    """The device token (``"cuda:N"`` or ``"cpu"``) where a recording or event-refining run keeps its planes."""
    if device:
        from ..launch import _current_device_ordinal

        return f"cuda:{_current_device_ordinal()}"
    return "cpu"


def _plane(value):
    """A plane the runner holds (a scalar stays a number) as an :class:`eagle.Array` over the same memory."""
    from .._layout import _view, is_number

    return value if is_number(value) else _view(value)


def record_planes(plan, max_steps, window, device, xp, n, planes, events="step"):
    """The per-sample planes a recorded run adds, made before the runner
    binds so a reordering ActiveSet can own them: ``applied_steps``, the
    time plane's start ``<t>_0``, a time step the twin reads per sample, and
    the forward's tape (never touched: declared indirect, never moved). An
    ``events="refine"`` run's tape is one real row per sample: the state it
    held before its last step."""
    refine = events == "refine"
    k = _step_kernels(plan.plugin, 1 if refine else resolve_window(window, max_steps,
                                                                    "eagle.until_done"),
                      "eagle.until_done", device, events)
    twin = k.fused or k.single
    scalar = planes[k.taped[0][0]].dtype
    out = {APPLIED_STEPS: _alloc.zeros(xp, n, dtype=twin.arg_dtypes[APPLIED_STEPS])}
    if k.time is not None:
        out[f"{k.time}_0"] = _alloc.empty_like(xp, planes[k.time])
    for role, nm in twin.arg_spec:
        if role == "per_sample" and nm in planes and not hasattr(planes[nm], "shape"):
            out[nm] = _alloc.full(xp, n, planes[nm], dtype=scalar)
    width = sum(rows for _nm, rows in k.taped)
    if refine:
        row = _alloc.zeros(xp, (width, n), dtype=scalar)
        out[TAPE] = (row, row)
        return out
    rows = resolve_window(window, max_steps, "eagle.until_done") * width
    out[TAPE] = _dummy_tape(xp, rows, width, n, scalar)
    return out


def _dummy_tape(xp, rows, width, n, dtype):
    """``(view, backing)``: a ``(rows, n)`` C-contiguous view over ONE row of
    ``width * n`` elements. The forward's twin runs with its tape stores
    switched off (``tape_on = 0``), so it never touches the tape, and binding
    needs a contiguous plane of the full shape; the reverse owns the real
    tape."""
    backing = _alloc.empty(xp, (width, n), dtype=dtype)
    return arrays.view_as(backing, (rows, n)), backing


def _eager(node, device=True):
    """A step-list node run eagerly, once per step: an opaque's ``fn`` by its
    tier (captured: on eagle's current stream; ``stream=``: on its own stream,
    fenced both sides by events; ``eager=True``: between two device syncs); a
    torch ``child`` replays its owner on eagle's current stream."""
    if node.kind != "child":
        fn = node.payload[0]
        tier = get_node_kind(node.kind).tier
        if tier == "capture" or not device:
            return fn
        from ..graph import _Eager

        eager = _Eager([], fn, None if tier == "host" else node.payload[4])
        if tier == "host":
            return lambda: eager.run([])

        fence = _stream.Event()

        def on_stream():
            current = _stream.current()
            fence.record(current)
            eager.run([fence]).wait(current)
        return on_stream
    owned = node.payload[0]
    if not callable(getattr(owned.owner, "replay", None)):
        raise ValueError(f"eagle.until_done: child {node.label!r} of a recorded run replays "
                         "a torch.cuda.CUDAGraph; run another framework's graph in an opaque")

    def replay():
        import torch

        owned.check()
        ptr = _stream.current()
        # torch.cuda.ExternalStream(0) is a new stream object, not the null stream
        stream = torch.cuda.default_stream() if ptr == _stream.LEGACY else torch.cuda.ExternalStream(ptr)
        with torch.cuda.stream(stream):
            owned.owner.replay()
    return replay


def _columns(a, m):
    """A new plane shaped like ``a`` with ``m`` samples."""
    return _alloc.empty(a.device, () if a.ndim == 0 else a.shape[:-1] + (m,), dtype=a.dtype)


class RunTape:
    """What a reverse reads from one recorded run: the checkpoints (one dict
    of state planes per window start), the per-sample steps (sample order,
    int64 host array), the window, the budget, and the non-state planes the
    step read."""

    __slots__ = ("checkpoints", "steps", "window", "reads", "budget", "events")

    def __init__(self, checkpoints, steps, window, reads, budget, events=None):
        self.checkpoints = checkpoints
        self.steps = steps
        self.window = window
        self.reads = reads
        self.budget = budget
        self.events = events     # an events="refine" run's refinement (diff/events.py)

    @property
    def nbytes(self) -> int:
        return sum(int(a.nbytes) for c in self.checkpoints for a in c.values())

    def copy(self):
        dup = lambda a: a.copy()
        from .events import copy_events

        return RunTape([{k: dup(a) for k, a in c.items()} for c in self.checkpoints],
                       self.steps.copy(), self.window,
                       {k: dup(a) for k, a in self.reads.items()}, self.budget,
                       copy_events(self.events))


class RunRecorder:
    """The record slot of one runner: the twin bound over the runner's own
    planes (tape off) and the checkpoint pool."""

    def __init__(self, runner, plan, door, window, budget, extras, owner, *, record=True,
                 events="step", on_no_event="raise", nodes=()):
        from ..plan import plan as make_plan

        self.runner = runner
        self.owner = owner            # what Record.forward names: the Runner or Simulation
        self.door = door
        self.recording = record
        refine = events == "refine"
        # without record=, a refined run keeps no checkpoints but the first
        self.window = (resolve_window(window, runner._max_steps, door) if record
                       else max(int(runner._max_steps), 1))
        self.budget = resolve_budget(budget, runner._device, door) if record else 0
        xp, n = array_module(runner._device), runner.n
        self.xp, self.n, self.device = xp, n, runner._device
        # the forward of a refined run tapes one row (the state before the
        # stop step); the reverse recomputes with the window's own twin
        self.fwd_kernels = (_step_kernels(plan.plugin, 1, door, runner._device, events)
                            if refine else None)
        self.kernels = k = (_step_kernels(plan.plugin, self.window, door, runner._device, events)
                            if record else self.fwd_kernels)
        self.fwd_kernels = self.fwd_kernels or k
        # an automatic or steps=K kernel replays its fused twin: at most its
        # own steps per launch, never across a window boundary
        self.fused = k.fused is not None
        self.max_k = int(runner._k_max or 1) if self.fused else 1
        fk = self.fwd_kernels
        twin = fk.fused if self.fused else fk.single
        self.roles = roles = {nm: role for role, nm in twin.arg_spec}
        planes = {nm: _plane(v) for nm, v in runner.step._planes.items() if nm in roles}
        if runner.active is not None:   # the real map, not a deferred placeholder
            runner._materialize_map()
            planes.update({nm: _plane(v) for nm, v in runner.active.planes().items() if nm in roles})
        scalar = planes[k.taped[0][0]].dtype
        planes.update(extras)
        planes[TAPE], self._backing = planes[TAPE]
        self.t0 = f"{k.time}_0" if k.time is not None else None
        self.promoted = set(extras) - {APPLIED_STEPS, self.t0, TAPE}
        self.active = runner.active
        self.every = 1 if runner.auto else runner.every
        self.reorders = runner._reorders()
        width = sum(rows for _nm, rows in k.taped)
        planes[TAPE_ON] = 1 if refine else 0
        self.word = None
        if self.fused:
            self.word = planes[FUSED_STEPS] = _alloc.ones(xp, 1, dtype=arrays.int64)
        self.structure = plan.structure
        self.planes = planes
        self.width, self.scalar = width, scalar
        self.forward = _split.single(make_plan(twin, structure=plan.structure).bind(**planes),
                                     "eagle.diff (record=True)")
        self.state = {nm: planes[nm] for nm, role in roles.items() if role in _STATE}
        self.uniforms = {nm: planes[nm] for nm, role in roles.items()
                         if role == "uniform" and nm != TAPE_ON}
        self.read_names = tuple(nm for nm, role in roles.items()
                                if role not in _STATE and role not in ("uniform", "lookup", "out"))
        # the step's nodes ([node, ..., plan]): re-run per step in the reverse,
        # their writes kept in a W-row ring; exclusive, as they run over your planes
        self.nodes, self.calls = nodes, [_eager(node, runner._device) for node, _r, _w in nodes]
        self.ring = tuple(dict.fromkeys(pn for _n, _r, w in nodes for _nm, pn, _a in w))
        self.node_lock = threading.Lock()
        for node, reads, _w in nodes:
            for _nm, pn, _a in reads:
                if roles.get(pn) == "mutable" and pn not in dict(k.taped):
                    raise ValueError(
                        f"{door}: node {node.label!r} reads {pn!r}, a state plane the step "
                        "writes without reading, so it has no adjoint; read it in the step")
        self.pool = []
        self.slot = _core.RecordSlot()
        self.live = None
        self._taken = 0
        self.plan = plan
        self.refine = None
        if refine:
            from .events import RefinePass

            self.refine = RefinePass(self, on_no_event)

    def per_sample_bytes(self):
        """What one sample costs a reverse: its checkpoints over the longest
        possible run plus one window of tape."""
        state = sum(int(a.nbytes) // self.n for a in self.state.values())
        windows = -(-self.runner._max_steps // self.window)
        ring = sum(int(self.planes[nm].nbytes) // self.n for nm in self.ring)
        return windows * state + self.window * (self.width * self.scalar.itemsize + ring)

    # --- the forward --- #
    def _held(self):
        """The planes a checkpoint keeps: the state, and the slot -> sample
        map when compaction reorders the planes."""
        held = dict(self.state)
        if self.reorders:
            held[_EPOCH] = self.active.perm
        return held

    def _checkpoint(self, k, copy=True):
        held = self._held()
        if k == len(self.pool):
            self.pool.append({nm: _alloc.empty_like(self.xp, a) for nm, a in held.items()})
        if copy:
            for nm, a in held.items():
                arrays.copyto(self.pool[k][nm], a)
        self._taken = k + 1

    def _compact(self, done):
        """After a launch ending at step ``done``: compact at the run's
        cadence, reordering when the set's trigger fired; then the step's
        nodes prepare the next step."""
        if self.active is not None and done % self.every == 0:
            self.active.compact()
            if self.reorders and int(self.active.fire[0]):
                self.active.reorder()
        for call in self.calls:
            call()

    def run(self):
        from .._until_done import RunReport, _mask_count
        from ._record import Record

        runner, n = self.runner, self.n
        status = self.slot.begin()
        if status == 1:
            raise Busy(f"{self.door}: a reverse is reading the previous run's Record; "
                       "wait for it, or record.keep() a copy first")
        if status == 2:
            raise RuntimeError(f"{self.door}: the previous run's Record is still live; "
                               "reverse it, or record.keep() to keep it, before "
                               "recording again")
        try:
            _mask_count(runner._dev, runner.finished, runner.terminated)
            if self.active is not None:     # the map follows the mask from the first launch
                self.active.compact()
            self.planes[APPLIED_STEPS].fill(0)
            if self.t0 is not None:
                arrays.copyto(self.planes[self.t0], self.planes[self.kernels.time])
            self._taken = 0
            max_steps, window = runner._max_steps, self.window
            for call in self.calls:     # the nodes before step 0
                call()
            b = self.forward
            if self.device:
                ran, launches = self._run_device(max_steps, window)
            else:
                if b._copies:
                    raise ValueError(f"{self.door}: record=True binds every plane "
                                     "directly; pass contiguous arrays")
                word = 0 if self.word is None else int(self.word.ptr)
                after = self._compact if self.active is not None or self.calls else None
                ran = int(_core.diff_record_host(b._entry, b._addrs, n, max_steps, window,
                                                 self.max_k, word, runner.finished.ptr,
                                                 self._checkpoint, after))
                launches = self._launch_count(ran)
            if self.reorders:
                self.active.restore()
            if self.refine is not None:
                self.refine.run()
            if not self.recording:
                self.slot.abandon()
                finished = int(runner.terminated.sum())
                return RunReport(n=n, finished=finished, done=finished == n,
                                 launches=launches, steps=ran, compactions=0, reorders=0,
                                 build_s=0.0, exact_steps=True, mode="refine")
            steps = self.planes[APPLIED_STEPS]
            steps = steps.astype(arrays.int64)
            steps = steps.get() if self.device else steps
            reads = {nm: self.planes[nm] for nm in self.read_names}
            tape = RunTape(self.pool[:self._taken], steps, window, reads, self.budget,
                           None if self.refine is None else self.refine.held())
            record = Record(self.owner, self.slot.generation, tape, self)
        except BaseException:
            self.slot.abandon()
            raise
        self.live = record
        _counters.bump("records")
        finished = int(runner.terminated.sum())
        return RunReport(n=n, finished=finished, done=finished == n, launches=launches,
                         steps=ran, compactions=0, reorders=0, build_s=0.0,
                         exact_steps=True, mode="record")

    def _launch_count(self, ran):
        """Launches of ``ran`` steps under the boundary rule (``recordForward``)."""
        if not self.fused:
            return ran
        per_window = -(-self.window // self.max_k)
        full, rest = divmod(ran, self.window)
        return full * per_window + -(-rest // self.max_k)

    def _run_device(self, max_steps, window):
        """The host's loop on the device, in C++ (``diff_record_device``):
        Python only allocates a window's checkpoint the first time it is
        taken, and compacts at the run's cadence."""
        import importlib

        launch = importlib.import_module("eagle.launch")
        b = self.forward
        state = [(int(a.ptr), int(a.nbytes)) for a in self._held().values()]

        def checkpoint(k):
            self._checkpoint(k, copy=False)
            return [int(a.ptr) for a in self.pool[k].values()]

        after = self._compact if self.active is not None or self.calls else None
        word = 0 if self.word is None else int(self.word.ptr)
        ran = int(_core.diff_record_device(
            b._entry, b._addrs, self.n, max_steps, window, self.max_k, word,
            int(self.runner.finished.ptr), int(launch._resolved_block(b._entry, self.n, None, declared=b._declared_block)),
            state, checkpoint, after, _stream.current(), -1))
        return ran, self._launch_count(ran)

    # --- the Record's slot protocol --- #
    def hold(self, record, door):
        if not self.slot.hold(record._generation):
            raise RuntimeError(
                f"{door}: this Record is stale — the forward ran again (generation "
                f"{record._generation} → {self.slot.generation}); reverse the new "
                "record, or keep() a Record before relaunching")

    def release(self, record, door, consumed=True):
        if not self.slot.release(record._generation, consumed):
            raise RuntimeError(
                f"{door}: this Record is stale — the forward ran again (generation "
                f"{record._generation} → {self.slot.generation}) while it was read")
        if consumed:
            record._released = True


def _node_gradient(got, nm, primal, node, door):
    """The gradient a node's vjp returned for the read ``nm``, checked against it."""
    if nm not in got:
        raise ValueError(f"{door}: vjps[{node.label!r}] returned no gradient for "
                         f"{nm!r} (it returned {sorted(got)})")
    g = arrays.asarray(got[nm], device=primal.device)
    if g.shape != primal.shape or g.dtype != primal.dtype:
        raise ValueError(
            f"{door}: vjps[{node.label!r}] returned {nm!r} of shape {g.shape} "
            f"and dtype {g.dtype}; the read is {primal.shape} {primal.dtype}")
    return g


def _take_columns(dst, src, cols, count):
    """``dst[:, :count] = src[:, cols]`` for 2-D ``(rows, samples)`` planes; ``cols`` is int32."""
    for r in range(dst.shape[0]):
        arrays.take(src[r], cols, out=dst[r, :count])


def _accumulate(view, g):
    """``view += g`` in place (same size and dtype)."""
    arrays.add(view, g, out=view)


def _put_columns(dst, cols, src, count):
    """``dst[:, cols] = src[:, :count]`` for 2-D ``(rows, samples)`` planes; ``cols`` is int32."""
    for r in range(dst.shape[0]):
        arrays.scatter(dst[r], cols, src[r, :count])


def _mb(b):
    return f"{b / 2**20:.3g} MB"


class RunReverse:
    """The reverse of a recorded run, built once per forward: planes for one
    pass of ``m`` samples, the recompute twin bound over them, the vjp bound
    once per (tape row, adjoint buffer parity), the gradient buffers."""

    def __init__(self, recorder, door, vjps=None, policy=1):
        from ..plan import plan as make_plan

        self.policy = policy    # core code of the parameter sums (eagle.reduce policies)
        self.rec = rec = recorder
        labels = [node.label for node, _r, _w in rec.nodes]
        vjps = vjps or {}
        if vjps and not labels:
            raise ValueError(f"{door}: vjps= names nodes of the step, and this run's step has "
                             "none; eagle derives the vjp of the step from its kernel")
        for label in sorted(set(vjps) - set(labels)) or sorted(set(labels) - set(vjps)):
            raise ValueError(
                f"{door}: " + (f"vjps names {label!r}, which is not a node of the step (its "
                               f"nodes: {labels})" if label in vjps else
                               f"node {label!r} is opaque: pass vjps={{{label!r}: fn}}, a "
                               "vjp(primals, cotangents) -> {read name: gradient} callable"))
        for label in labels:
            if not callable(vjps[label]):
                raise TypeError(f"{door}: vjps[{label!r}] must be a vjp(primals, cotangents) "
                                "callable")
        self.fns = [vjps[label] for label in labels]
        xp, n, W = rec.xp, rec.n, rec.window
        k = rec.kernels
        per = rec.per_sample_bytes()
        leaf = _LEAF if self.policy == 1 else 1     # only ordered keeps gradients bitwise per leaf
        m = int(_chunk_samples(rec.budget, per, n, leaf))
        if m == 0:
            windows = -(-rec.runner._max_steps // W)
            if per > rec.budget:
                raise ValueError(
                    f"eagle.diff.reverse: one sample alone needs {_mb(per)} ({windows} "
                    f"checkpoints and a {W}-step tape of {rec.width} values), more than "
                    f"memory_budget={_mb(rec.budget)}; raise memory_budget=, lower "
                    "max_steps, or set window=")
            raise ValueError(
                f"eagle.diff.reverse: memory_budget={_mb(rec.budget)} holds "
                f"{rec.budget // per} samples per pass, fewer than the {_LEAF} a pass "
                "needs to keep parameter gradients bitwise; raise memory_budget= to at "
                f"least {_mb(_LEAF * per)}")
        self.xp, self.n, self.m, self.window = xp, n, m, W
        self.passes = -(-n // m)
        self.state = {nm: _columns(a, m) for nm, a in rec.state.items()}
        self.reads = {nm: _columns(rec.planes[nm], m) for nm in rec.read_names}
        self.tape = _alloc.empty(xp, (W * rec.width, m), dtype=rec.scalar)
        twin = k.fused if rec.fused else k.single
        twin_planes = dict(rec.planes)
        twin_planes.update(self.state)
        twin_planes.update(self.reads)
        twin_planes[TAPE], twin_planes[TAPE_ON] = self.tape, 1
        from .._active_set import COUNT_PLANE, MAP_PLANE

        for nm, role in rec.roles.items():      # every slot live, in place: identity map
            if nm == MAP_PLANE:
                twin_planes[nm] = arrays.arange(m, dtype=rec.planes[nm].dtype, device=xp)
            elif nm == COUNT_PLANE:
                twin_planes[nm] = _alloc.full(xp, rec.planes[nm].shape, m, dtype=rec.planes[nm].dtype)
            elif role == "lookup":
                twin_planes[nm] = _alloc.zeros_like(xp, rec.planes[nm])
        self.word = twin_planes.get(FUSED_STEPS) if rec.fused else None
        self.recompute = _split.single(make_plan(twin, structure=rec.structure).bind(**twin_planes),
                                       "eagle.diff.reverse")

        vspec = k.vjp.arg_spec
        taped = dict(k.taped)
        offsets, off = {}, 0
        for nm, rows in k.taped:
            offsets[nm] = off
            off += rows
        self.taped = list(taped)
        self.adj = [{nm: _columns(rec.state[nm], m) for nm in taped} for _ in range(2)]
        self.step_plane = None
        if any(nm == APPLIED_STEPS for _r, nm in vspec):
            self.step_plane = _alloc.zeros(xp, m, dtype=k.vjp.arg_dtypes[APPLIED_STEPS])
        self.grad = {}        # input name -> its share of one step, per slot
        for _role, nm in vspec:
            if nm.startswith("bar_") and not nm.endswith(_NEXT) and nm[4:] not in taped:
                base = nm[4:]
                if base in rec.uniforms:
                    self.grad[base] = _alloc.zeros(xp, m, dtype=rec.scalar)
                elif base in self.reads:
                    self.grad[base] = _alloc.zeros_like(xp, self.reads[base])
        # a node-written plane is an intermediate, not an input
        self.acc = {nm: _alloc.zeros_like(xp, g) for nm, g in self.grad.items() if nm not in rec.ring}
        for _n, reads, _w in rec.nodes:     # a parameter only a node reads
            for _nm, pn, _a in reads:
                if pn in rec.uniforms and pn not in self.acc:
                    self.acc[pn] = _alloc.zeros(xp, m, dtype=rec.scalar)
        self.ring = {nm: _alloc.empty(xp, (W,) + self.reads[nm].shape, dtype=self.reads[nm].dtype)
                     for nm in rec.ring}
        self.offsets = offsets
        self.vjps = []
        for row in range(W):
            for parity in range(2):
                bind = {}
                for _role, nm in vspec:
                    if nm.startswith("bar_") and nm.endswith(_NEXT):
                        bind[nm] = self.adj[parity][nm[4:-len(_NEXT)]]
                    elif nm.startswith("bar_") and nm[4:] in taped:
                        bind[nm] = self.adj[1 - parity][nm[4:]]
                    elif nm.startswith("bar_"):
                        bind[nm] = self.grad[nm[4:]]
                    elif nm in taped:
                        lo = row * rec.width + offsets[nm]
                        bind[nm] = self.tape[lo:lo + taped[nm]].reshape(self.state[nm].shape)
                    elif nm == APPLIED_STEPS:
                        bind[nm] = self.step_plane
                    elif nm in rec.uniforms:
                        bind[nm] = rec.uniforms[nm]
                    elif nm in self.reads:
                        bind[nm] = self.reads[nm]
                    else:
                        raise ValueError(f"{door}: the step's vjp reads {nm!r}, which "
                                         "the recorded run does not hold")
                self.vjps.append(_split.single(make_plan(k.vjp, structure=rec.structure)
                                              .bind(**bind), "eagle.diff.reverse"))
        # parameters sum over samples; a time-indexed run's t_0 share belongs to t
        self.summed = {nm for nm in self.acc if nm in rec.uniforms or nm in rec.promoted}
        self.folded = {rec.t0: k.time} if rec.t0 in self.acc else {}
        self.inputs = tuple(self.taped) + tuple(nm for nm in self.acc if nm not in self.folded)
        if rec.refine is not None:      # inputs only the stop rule reads (a guard's parameter)
            ek = rec.refine.k
            more = [nm[4:] for _r, nm in ek.vjp.arg_spec if nm.startswith("bar_")
                    and not nm.endswith("_refined") and not nm.startswith("bar_event_g")]
            self.inputs += tuple(nm for nm in more if nm not in self.inputs
                                 and nm not in (ek.tau, rec.t0) and nm in rec.planes)
        self._busy = BusyGuard("eagle.diff.Reverse")
        self._tape = None

    def __call__(self, record, seeds, wrt, door):
        from ._reverse import Gradients, Report

        rec = self.rec
        unknown = sorted(set(seeds) - set(self.taped))
        if unknown:
            hint = ", ".join(map(repr, self.taped))
            raise ValueError(
                f"{door}: seeds names {unknown[0]!r}, which is not a state plane of the "
                f"recorded run (state: {hint}). seeds are the gradient of your "
                "objective with respect to the END state, one value per sample.")
        names = self.inputs if wrt is None else ([wrt] if isinstance(wrt, str) else list(wrt))
        for nm in names:
            if nm not in self.inputs:
                raise ValueError(f"{door}: wrt names {nm!r}, which is not an input of the "
                                 f"recorded run (inputs: {', '.join(map(repr, self.inputs))})")
        for nm, value in seeds.items():
            shape = tuple(getattr(value, "shape", ()))
            if shape != tuple(rec.state[nm].shape):
                raise ValueError(
                    f"{door}: seeds[{nm!r}] has shape {shape}, but the record holds "
                    f"{self.n} samples of {nm!r} (shape {tuple(rec.state[nm].shape)}); pass "
                    "one value per sample in sample order")
        if rec.nodes and not rec.node_lock.acquire(blocking=False):
            raise Busy(f"{door}: a reverse of this run is re-running node "
                       f"{rec.nodes[0][0].label!r} over your planes; wait for it")
        try:
            with self._busy("__call__"), record._hold(door) as tape:
                extra = None
                if tape.events is not None:     # an events="refine" run: the moved stop time
                    from .events import RefineReverse

                    seeds, steps, extra = RefineReverse(rec).seeds(tape, seeds, door)
                    if rec.nodes:
                        self._refine_nodes(tape, seeds, extra)
                    tape = RunTape(tape.checkpoints, steps, tape.window, tape.reads,
                                   tape.budget, tape.events)
                # the nodes' planes are scratch until restored bitwise
                self._saved = [(a, a.copy()) for a in {
                    id(a): a for _n, r, w in rec.nodes for _nm, _pn, a in (*r, *w)}.values()]
                try:
                    values = self._run(tape, seeds, names)
                finally:
                    self._restore()
        finally:
            if rec.nodes:
                rec.node_lock.release()
        if extra:
            self._add_refine(values, extra, names)
        _counters.bump("reverses")
        return Gradients(values, Report(record.generation, {nm: True for nm in values},
                                        self.passes))

    def _run(self, tape, seeds, names):
        rec, xp, n, m = self.rec, self.xp, self.n, self.m
        flat = lambda a, width: a.reshape((a.size // width, width))
        out = {nm: _alloc.zeros_like(xp, rec.state[nm]) for nm in self.taped}
        per = {nm: _alloc.zeros_like(xp, tape.reads[nm]) for nm in self.acc if nm not in self.summed}
        partials = {nm: [] for nm in self.summed}
        seeded = {nm: arrays.asarray(v, rec.state[nm].dtype, xp) for nm, v in seeds.items()}
        self._tape = tape
        for s0 in range(0, n, m):
            cnt = min(m, n - s0)
            steps = arrays.ascontiguousarray(tape.steps[s0:s0 + cnt])
            order = arrays.asarray(_core.diff_stop_order(steps.ptr, cnt), arrays.int64)
            live = _core.diff_live_counts(steps.ptr, cnt)
            cols = arrays.asarray(arrays.add(order, s0), device=xp)
            cols32 = cols.astype(arrays.int32)
            orders = self._orders(tape, cols, cols32)
            self._restore()     # a node's planes may be the record's reads
            for nm, buf in self.reads.items():
                _take_columns(flat(buf, m), flat(tape.reads[nm], n), cols32, cnt)
            self._live = live
            for _n, r, _w in rec.nodes:     # step-invariant reads, in reverse slot order
                for _nm, pn, a in r:
                    if pn in self.reads and pn not in rec.ring:
                        arrays.asarray(a)[..., :cnt] = self.reads[pn][..., :cnt]
            for nm in self.taped:
                for a in self.adj:
                    a[nm].fill(0)
                    if nm in seeded:
                        _take_columns(flat(a[nm], m), flat(seeded[nm], n), cols32, cnt)
            for a in self.acc.values():
                a.fill(0)
            if live:
                (self._device if rec.device else self._host)(orders, live)
            for nm in self.taped:
                _put_columns(flat(out[nm], n), cols32, flat(self.adj[1][nm], m), cnt)
            for nm, acc in self.acc.items():
                if nm in self.summed:
                    sample = _alloc.empty(xp, cnt, dtype=acc.dtype)
                    arrays.scatter(sample, arrays.asarray(order, arrays.int32, xp), acc[:cnt])
                    partials[nm].append(self._sum(sample, pair=True))
                else:
                    _put_columns(flat(per[nm], n), cols32, flat(acc, m), cnt)
        values = {}
        for nm in names:
            if nm in out:
                values[nm] = out[nm]
            elif nm in partials:
                code = 1 if self.acc[nm].dtype == arrays.float64 else 0
                values[nm] = arrays.asarray(_combine(partials[nm], code), self.acc[nm].dtype, xp)
            elif nm in per:
                values[nm] = per[nm]
        for t0, t in self.folded.items():
            if t in values:
                values[t] = arrays.add(values[t], per[t0])
        if rec.device:
            _core.stream_sync(_stream.current(), -1)
        return values

    def _restore(self):
        for a, b in self._saved:
            arrays.copyto(a, b)

    def _sync(self):
        if self.rec.device:
            _core.device_synchronize(-1)
            _counters.bump("host_syncs")

    def _hook(self, j):
        """Before recomputing step ``j``: the nodes re-run over your planes (live
        prefix, the rest masked finished); their writes go to the twin and the ring."""
        count, row = self._live[j], j % self.window
        for _n, reads, _w in self.rec.nodes:
            for _nm, pn, a in reads:
                if pn in self.state and pn in self.adj[0]:
                    arrays.asarray(a)[..., :count] = self.state[pn][..., :count]
                elif self.rec.roles.get(pn) == "terminated":
                    a[:count], a[count:] = False, True
        for call in self.rec.calls:
            call()
        for _n, _r, writes in self.rec.nodes:
            for _nm, pn, a in writes:
                written = arrays.asarray(a)[..., :count]      # the caller's plane, viewed in place
                self.reads[pn][..., :count] = written
                self.ring[pn][row][..., :count] = written

    def _ring_row(self, j):
        """Step ``j``'s node writes back in the twin's read buffer, for its vjp."""
        count = self._live[j]
        for pn, ring in self.ring.items():
            self.reads[pn][..., :count] = ring[j % self.window][..., :count]

    def _node_vjp(self, j, count):
        """After step ``j``'s vjp launch: each node's vjp, last node first, host-synchronized,
        on the live prefix, added to the state adjoint (taped read), the cotangent of a plane
        an earlier node writes, or the input's sum (invariant)."""
        row, door = j % self.window, "eagle.diff.reverse"
        self._sync()
        for (node, reads, writes), fn in reversed(list(zip(self.rec.nodes, self.fns))):
            primals, into = {}, {}
            reads_pn = {nm: pn for nm, pn, _a in reads}
            for nm, pn, a in reads:
                if a.dtype.kind != "f":
                    continue
                if pn in self.rec.uniforms:     # a parameter: its value, one gradient
                    primals[nm] = _uniform_plane(self.rec.uniforms[pn], a)
                    into[nm] = self.acc[pn]
                elif pn in self.adj[0]:
                    lo = row * self.rec.width + self.offsets[pn]
                    rows = self.tape[lo:lo + max(int(self.state[pn].size) // self.m, 1)]
                    primals[nm] = rows.reshape(self.state[pn].shape)[..., :count]
                    into[nm] = self.adj[1 - (j & 1)][pn]
                else:       # a plane another node writes passes its cotangent back to it
                    primals[nm] = self.reads[pn][..., :count]
                    into[nm] = self.grad[pn] if pn in self.rec.ring else self.acc.get(pn)
            got = fn(dict(primals), {nm: self.grad[pn][..., :count] for nm, pn, _a in writes})
            self._sync()
            for nm, primal in primals.items():
                g = _node_gradient(got, nm, primal, node, door)
                if reads_pn[nm] in self.rec.uniforms:   # summed over samples at the end
                    _accumulate(into[nm][:1], g.sum())
                elif into[nm] is not None:
                    _accumulate(into[nm][..., :count], g)
        if row:
            self._ring_row(j - 1)

    def _refine_nodes(self, tape, pre, extra):
        """The nodes' vjps at the refined step: the cotangent the refine entry left on a
        plane a node writes goes through that node, last first, from the state before
        the stop step; it adds to that state's adjoint ``pre``, to the cotangent of a
        plane an earlier node writes, to an input's share, or to a parameter's."""
        xp, ev, door = self.xp, tape.events, "eagle.diff.reverse"
        self._sync()
        for (node, reads, writes), fn in reversed(list(zip(self.rec.nodes, self.fns))):
            primals, into = {}, {}
            for nm, pn, a in reads:
                if a.dtype.kind != "f":
                    continue
                if pn in self.rec.uniforms:
                    primals[nm] = _uniform_plane(self.rec.uniforms[pn], a)
                elif pn in pre:
                    primals[nm] = ev[pn]
                else:
                    primals[nm] = tape.reads[pn]
                into[nm] = pn
            cots = {nm: extra.get(pn, _alloc.zeros_like(xp, tape.reads[pn])) for nm, pn, _a in writes}
            got = fn(dict(primals), cots)
            self._sync()
            for nm, primal in primals.items():
                g = _node_gradient(got, nm, primal, node, door)
                pn = into[nm]
                if pn in self.rec.uniforms:
                    _accumulate(extra.setdefault(pn, _alloc.zeros(xp, self.n, dtype=self.rec.scalar))[:1], g.sum())
                elif pn in pre:
                    _accumulate(pre[pn], g)
                else:
                    extra[pn] = arrays.add(extra[pn], g) if pn in extra else g.copy()

    def _add_refine(self, values, extra, names):
        """The refine entry's share of each input gradient, added to the
        step-by-step reverse's: summed in sample order for a parameter, per
        sample otherwise, a start time's share to its time plane."""
        rec = self.rec
        for nm, g in extra.items():
            target = self.folded.get(nm, rec.kernels.time if nm == rec.t0 else nm)
            if target not in names:
                continue
            summed = target in self.summed or target in rec.uniforms
            share = arrays.asarray(self._sum(g), g.dtype, self.xp) if summed else g
            values[target] = arrays.add(values[target], share) if target in values else share

    def _orders(self, tape, cols, cols32):
        """Per window, the checkpoint column of each reverse slot: the
        sample's own column, or its slot at that checkpoint when compaction
        reordered the planes."""
        xp = self.xp
        out = []
        for c in tape.checkpoints:
            perm = c.get(_EPOCH)
            if perm is None:
                out.append(cols)
                continue
            inv = _alloc.empty(xp, perm.shape[0], dtype=arrays.int64)
            arrays.scatter(inv, arrays.asarray(perm, arrays.int32), arrays.arange(perm.shape[0], dtype=arrays.int64, device=xp))
            out.append(arrays.take(inv, cols32))
        return out

    def _sum(self, x, pair=False):
        """The canonical sum of ``x`` (one pass, sample order) as a float, or
        its unrounded (hi, comp) partial with ``pair``."""
        code = 1 if x.dtype == arrays.float64 else 0
        n2 = 2 if pair else 1
        if self.rec.device:
            work = int(_core.reduce_canonical_device(0, x.size, code, 0, 0, 0, 0, 0, -1, self.policy))
            wa, wb = (_alloc.empty(self.xp, max(work, 1), dtype=x.dtype) for _ in range(2))
            res = _alloc.empty(self.xp, n2, dtype=x.dtype)
            _core.reduce_canonical_device(int(x.ptr), x.size, code, 0, int(res.ptr),
                                          int(wa.ptr), int(wb.ptr), _stream.current(), -1,
                                          self.policy, 1, 0, pair)
            res = res.get()
        else:
            res = arrays.empty(n2, x.dtype, "cpu")
            from .._graph_reduce import fixed_f64

            _core.reduce_canonical_host(x.ptr, x.size, code, 0, res.ptr, 1, 0, pair,
                                        code == 1 and not pair and fixed_f64())
        got = res.tolist()
        return (float(got[0]), float(got[1])) if pair else float(got[0])

    def _plans(self):
        addr = lambda a: int(a.ptr)
        rows = lambda a: max(int(a.size) // self.m, 1)
        gathers = [(addr(buf), [addr(c[nm]) for c in self._tape.checkpoints],
                    rows(buf), int(buf.itemsize)) for nm, buf in self.state.items()]
        code = lambda a: 1 if a.dtype == arrays.float64 else 0
        accums = [(addr(acc), addr(self.grad[nm]), rows(acc), code(acc))
                  for nm, acc in self.acc.items() if nm in self.grad]
        step = (0, 0) if self.step_plane is None else (addr(self.step_plane),
                                                      int(self.step_plane.itemsize))
        word = 0 if self.word is None else addr(self.word)
        return gathers, accums, step, word

    def _callbacks(self):
        return (self._hook, self._node_vjp) if self.rec.nodes else (None, None)

    def _host(self, orders, live):
        gathers, accums, step, word = self._plans()
        _core.diff_reverse_host(self.m, self.n, self.window, live,
                                [int(o.ptr) for o in orders], gathers,
                                self.recompute._entry, self.recompute._addrs, word,
                                self.vjps[0]._entry, [v._addrs for v in self.vjps], accums,
                                step[0], step[1], *self._callbacks())

    def _device(self, orders, live):
        import importlib

        launch = importlib.import_module("eagle.launch")
        gathers, accums, step, word = self._plans()
        block = lambda b: int(launch._resolved_block(b._entry, self.m, None, declared=b._declared_block))
        stream = _stream.current()
        _core.diff_reverse_device(self.m, self.n, self.window, live,
                                  [int(o.ptr) for o in orders], gathers,
                                  self.recompute._entry, self.recompute._addrs, word,
                                  block(self.recompute), self.vjps[0]._entry,
                                  [v._addrs for v in self.vjps], block(self.vjps[0]), accums,
                                  step[0], step[1], stream, -1, *self._callbacks())


def reverse_of(forward):
    """The :class:`RunRecorder` behind a Runner or Simulation, or ``None``."""
    runner = getattr(forward, "runner", forward)
    return getattr(runner, "_recorder", None)
