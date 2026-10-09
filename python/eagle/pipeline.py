# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Capture a sequence of GPU launches into a replayable CUDA graph.

This is the Python face of EAGLE's graph engine: it drives the same
``eagle::cuda`` C++ classes other native C++ consumers use --
:class:`Stream`, :class:`StreamCapturer`, :class:`Graph`,
:class:`Launcher` -- through the :mod:`eagle._core` nanobind binding.
There is one graph implementation for the whole ecosystem; the Python
side only orchestrates it. The C++ core owns capture/instantiate/replay;
kernels are still issued by the caller onto the capture stream (a
launcher's ``launch`` defaults to the current stream, so the launch
becomes a graph node with no special plugin machinery). EAGLE creates
the stream and hands its raw handle to cupy via
:class:`cupy.cuda.ExternalStream`, so cupy ``RawKernel`` launches land
on the C++-owned stream and are captured.

Usage::

    pipe = GraphPipeline()
    pipe.add(lambda: backend_fn(grid, block, (pos, scale, scaled, n)))
    pipe.add(lambda: force.launch(out=out, position=scaled, mu=MU))
    pipe.build()  # capture -> instantiate
    pipe.launch(n=100)  # replay 100x

Steps must not allocate device memory or synchronize (forbidden
mid-capture); pre-allocate all arrays and pre-compile JIT kernels
(``force.precompile()``) before ``build()``.
"""

from __future__ import annotations

import os
import re
import tempfile
from typing import NamedTuple

from . import _counters
from .interop import _wrap_external_stream
from ._conditional import RepeatWhile, Skippable
from ._launch_policy import sibling_context
from ._member_enable import MemberHandle, NonToggleableMemberError, _MemberRegistry


class _LaunchHandles(NamedTuple):
    """(stream, launcher) pair for enqueue-without-sync replay of one
    built :class:`GraphPipeline`. Module-private: :mod:`eagle.compose`'s
    cross-pipeline overlap group uses it to enqueue several pipelines'
    replays on their own streams first, then synchronize them in a
    second pass, so independent pipelines' device work overlaps instead
    of serializing host-side. The same capability is reachable through
    the public :meth:`GraphPipeline.enqueue` / :meth:`GraphPipeline.
    synchronize` pair, which the overlap group actually uses. This
    accessor stays module-private as the "does this member own its own
    stream?" structural probe, and the C++-composer parity tests bind
    through it directly."""

    ext: object  #: cupy.cuda.ExternalStream wrapping this pipeline's own stream
    launcher: object  #: the bound eagle._core Launcher (async.launch() /.synchronize())


class _ConcurrentGroup:
    """Tagged record for a fork/join group registered via
    :meth:`GraphPipeline.add_concurrent`, distinguishable from the bare
    callables :meth:`GraphPipeline.add` appends. Module-private.
    ``_fork`` is set by :meth:`GraphPipeline.build`, before stream
    capture begins: the bound ``CaptureFork`` creates streams/events,
    which is illegal once capture is in flight.
    """

    __slots__ = ("members", "_fork")

    def __init__(self, members):
        self.members = members
        self._fork = None


class _GuardedRun:
    """The pre-built weave(s) of ONE conditional step -- a
    :class:`~eagle._conditional.Skippable` (an IF weave) or a
    :class:`~eagle._conditional.RepeatWhile` (a WHILE weave, plus one
    inner IF weave per ``Skippable`` part of its body). Module-private.
    Built by :meth:`GraphPipeline.build`'s pre-capture pass, since every
    ``_core.CaptureConditional`` creates its body stream at construction
    (illegal mid-capture), and the inner weave's origin is the outer
    weave's body stream. :meth:`run` is the in-capture half: nested
    begin/end pairs, innermost body launched on the innermost body
    stream; for a loop the outer ``end()`` also launches eagle's tail
    setter as the body's last node, after the inner IF node, so the
    continue/stop decision follows the whole iteration.
    """

    __slots__ = ("step", "outer", "inners")

    def __init__(self, step, outer, inners=None):
        self.step = step
        self.outer = outer
        # One entry per body part of a RepeatWhile (the part's inner IF weave,
        # or None for a plain part); None for a top-level Skippable.
        self.inners = inners

    def run(self, cp):
        with _wrap_external_stream(self.outer.begin()):
            if self.inners is None:
                self.step.step()  # a Skippable's body
            else:
                for part, inner in zip(self.step.parts, self.inners):
                    if inner is None:
                        part()
                        continue
                    with _wrap_external_stream(inner.begin()):
                        part.step()  # the Skippable part's body, inside the loop body
                    inner.end()
        self.outer.end()


def _concurrent_write_conflict_check(write_targets) -> None:
    """Assert no two different :meth:`GraphPipeline.add_concurrent`
    members declare an overlapping device write range -- the fork/join
    group's mutual-independence precondition. Debug/test-path only:
    O(k^2) in the total number of declared targets. ``write_targets``
    has one entry per member, each an iterable of device arrays that
    member writes. Overlap within one member is legal (a member's own
    launches serialize); only a byte-range overlap across two different
    members is a conflict. Raises ``AssertionError`` naming the first
    conflicting pair.
    """
    spans = []  # (member_index, start, end, array), one per declared target
    for member_index, targets in enumerate(write_targets):
        for arr in targets:
            start = int(arr.data.ptr)
            spans.append((member_index, start, start + int(arr.nbytes), arr))

    for a in range(len(spans)):
        mi, start_i, end_i, arr_i = spans[a]
        for b in range(a + 1, len(spans)):
            mj, start_j, end_j, arr_j = spans[b]
            if mi == mj:
                continue  # same member: serializes with itself, not a conflict
            if start_i < end_j and start_j < end_i:
                raise AssertionError(
                    "add_concurrent write conflict: members "
                    f"{mi} and {mj} both declare an overlapping device write "
                    f"range ({arr_i!r} vs {arr_j!r})"
                )


class GraphPipeline:
    """A CUDA-graph wrapper: capture a sequence of launches once,
    replay cheaply. Backed by the :mod:`eagle._core` binding of the
    ``eagle::cuda`` C++ core, so capture/instantiate/replay is the same
    code a native C++ consumer runs.
    """

    def __init__(self):
        import cupy as cp

        from . import _core

        self._core = _core
        # non-blocking so capture never entangles the legacy default stream that
        # cupy/torch launch on (matches the pre-uniformization cupy pipeline).
        self._stream = _core.Stream(non_blocking=True)  # owns the cudaStream_t
        self._sptr = self._stream.ptr()  # raw handle shared with cupy + the graph
        self._ext = _wrap_external_stream(self._sptr)
        self._steps = []
        self._graph = None
        self._launcher = None
        self._dot_str = None
        # Member-enable surface (mode="enabled"): every add()/add_concurrent()
        # step gets a stable index (and optionally a name) at registration
        # time. Node attribution/toggleability is filled in by build();
        # _member_enabled tracks the live on/off state.
        self._registry = _MemberRegistry()
        self._member_nodes: list = []       # list[list[int]], index-aligned
        self._member_toggleable: list = []  # list[bool], index-aligned
        self._member_enabled: list = []     # list[bool], index-aligned
        # The ordering events, instance-cached: allocated lazily on the first
        # enqueue() and re-recorded thereafter (cudaStreamWaitEvent snapshots
        # the event's state at the wait). _legacy_event stays None when the
        # caller's stream IS the legacy stream. See enqueue()'s ONE-IN-FLIGHT rule.
        self._seed_event = None
        self._legacy_event = None

    def add(self, step, *, name=None):
        """Register a zero-arg callable that issues launches on the
        current stream. The callable may issue one launch or several;
        what matters is not the count but the stream -- :meth:`build`
        captures whatever it launches on the stream that is current at
        the time it runs. ``name`` (optional, keyword-only) aliases
        this step as an individually addressable member: after
        :meth:`build`, ``pipe.set_member_enabled(name, False)`` (or its
        registration-order index, always valid -- see :meth:`member`)
        disables its launches on replay with no recapture. Every step
        is a member, named or not.
        """
        self._registry.register(name)
        self._steps.append(step)
        return self

    def add_concurrent(self, members, *, names=None):
        """Register a fork/join stage: every member is a sibling of
        every other, and both join before whatever is registered next.
        ``members`` is a sequence of zero-arg callables,
        ``len(members) >= 2`` -- each has the same contract as
        :meth:`add`'s step. Co-execution is permitted, never required: replaying members
        serialized, in registration order, is always a conforming
        schedule. Within one member its own launches still serialize;
        fork/join only forks between members. Actual concurrency, if
        any, is bounded by the device's own occupancy, never a fixed
        constant. v1 is series-parallel only: a plain chain of
        :meth:`add`/:meth:`add_concurrent` registrations, each stage
        joining fully before the next begins. This does not check that
        members are actually independent -- call
        :func:`_concurrent_write_conflict_check` yourself before
        :meth:`build` if you want that asserted. ``names`` (optional)
        aliases each member the same way ``add``'s ``name`` does -- a
        sequence the same length as ``members``, entries may be
        ``None``. Every member is individually addressable by its
        registration-order index regardless of whether it is named."""
        members = list(members)
        if len(members) < 2:
            raise ValueError(
                f"add_concurrent needs at least 2 members, got {len(members)}"
            )
        if names is not None and len(names) != len(members):
            raise ValueError(
                f"names has {len(names)} entries but members has {len(members)}"
            )
        for i in range(len(members)):
            self._registry.register(names[i] if names is not None else None)
        self._steps.append(_ConcurrentGroup(members))
        return self

    def build(self):
        """Stream-capture the registered launches into an instantiated
        graph. Immediately before and after each registered member's
        launches, this snapshots the in-construction graph's node set
        (``_core.capture_snapshot_nodes``) and records the set
        difference as that member's contributed nodes -- the only place
        member boundaries are known at capture time. Toggleability
        classification (``_core.is_node_toggleable``) runs separately,
        right after capture ends: a member is toggleable later only if
        every one of its nodes is a kernel, memcpy or memset node. A
        :class:`~eagle._conditional.Skippable` or
        :class:`~eagle._conditional.RepeatWhile` member is marked
        non-toggleable structurally, from this layer's own knowledge
        that it wove a conditional region, without ever driver-querying
        its nodes (``cudaGraphNodeGetType`` fails on this driver for a
        conditional/IF node). Either way the member is recorded
        non-toggleable, never silently accepted.
        """
        import cupy as cp

        # CaptureFork and CaptureConditional both create streams/events,
        # which must happen before capture begins (illegal mid-capture), so
        # every group's fork and every Skippable's/RepeatWhile's weave are
        # built in this up-front pass (_prebuild_guarded); the capture loop
        # below only calls fork()/branch()/join() and weave.begin()/end().
        runs = [None] * len(self._steps)  # top-level Skippable/RepeatWhile steps
        group_runs = {}  # id(_ConcurrentGroup) -> [_GuardedRun-or-None per member]
        for i, step in enumerate(self._steps):
            if isinstance(step, _ConcurrentGroup):
                step._fork = self._core.CaptureFork(self._sptr, len(step.members))
                group_runs[id(step)] = [
                    self._prebuild_guarded(step._fork.branch(j), member)
                    for j, member in enumerate(step.members)
                ]
            else:
                runs[i] = self._prebuild_guarded(self._sptr, step)

        capturer = self._core.StreamCapturer(self._sptr)
        self._member_nodes = []
        self._member_structurally_toggleable = []
        with self._ext:  # make the C++-owned stream cupy's current stream
            capturer.begin()
            try:
                for i, step in enumerate(self._steps):
                    if isinstance(step, _ConcurrentGroup):
                        self._run_concurrent(cp, step, group_runs[id(step)])
                    elif runs[i] is not None:
                        before = self._core.capture_snapshot_nodes(self._sptr)
                        runs[i].run(cp)
                        self._record_member_(before, self._sptr, skippable=True)
                    else:
                        before = self._core.capture_snapshot_nodes(self._sptr)
                        step()
                        self._record_member_(before, self._sptr, skippable=False)
            except Exception:
                try:
                    capturer.end()  # leave the stream out of capture mode
                except Exception:
                    pass
                raise
            captured = capturer.end()  # CapturedGraph owning the cudaGraph_t
            _counters.bump("captures")  # only on a completed capture

        # POST-INCIDENT FIX: on this driver, cudaGraphNodeGetType
        # (_core.is_node_toggleable) returns cudaErrorUnknown for a
        # cudaGraphNodeTypeConditional node (not a capture-timing issue --
        # isolated to specifically the conditional/IF node, never the
        # setCond kernel beside it). So a Skippable member's nodes are
        # never driver-queried for type; they are known non-toggleable
        # structurally and short-circuited via `and`. Only plain members'
        # nodes are driver-queried. This matches intent anyway: a
        # Skippable-wrapped member is always non-toggleable by design.
        self._member_toggleable = [
            ok and all(self._core.is_node_toggleable(n) for n in nodes)
            for ok, nodes in zip(
                self._member_structurally_toggleable, self._member_nodes
            )
        ]

        # Introspection off the captured graph (matches cupy's debug_dot_str),
        # taken BEFORE add_node consumes the CapturedGraph.
        self._dot_str = _debug_dot(captured)

        # Adopt and instantiate directly (no parent-graph clone, no kernel
        # harvest): cupy's Graph+add_node harvest calls
        # cudaGraphKernelNodeGetParams on captured kernels, corrupting the
        # shared CUDA context for later torch/cupy launches; adopting avoids it.
        g = self._core.Graph.from_captured(captured)
        g.stream(self._sptr)
        self._graph = g
        self._launcher = g.launcher()
        _counters.bump("instantiations")  # only on a completed instantiate
        self._launcher.stream(self._sptr)
        # Every registered member starts enabled: a freshly-instantiated
        # exec graph's nodes are all enabled by construction, so this is
        # bookkeeping only.
        self._member_enabled = [True] * len(self._registry)
        assert len(self._member_nodes) == len(self._registry), (
            "internal error: attributed member count "
            f"({len(self._member_nodes)}) != registered member count "
            f"({len(self._registry)})"
        )
        return self

    def _record_member_(self, before, stream, *, skippable: bool) -> None:
        """Finish one member's node-handle attribution: snapshot
        ``stream``'s current node set, diff against ``before``, and
        record the delta. Appends to ``self._member_nodes`` /
        ``self._member_structurally_toggleable`` in the same order
        ``self._registry`` assigned indices -- call this exactly once
        per member, in registration order. ``skippable`` (``True`` for
        a top-level :class:`~eagle._conditional.Skippable`/
        :class:`~eagle._conditional.RepeatWhile` step or an
        ``add_concurrent`` member wrapped in one) records whether this
        layer already knows the member is non-toggleable, without a
        driver query -- see :meth:`build`'s post-capture pass for why.
        Deliberately does not classify toggleability here -- only the
        node-handle delta; ``_core.is_node_toggleable`` runs in a
        separate pass in :meth:`build`, strictly after
        ``capturer.end()``."""
        after = self._core.capture_snapshot_nodes(stream)
        delta = sorted(set(after) - set(before))
        self._member_nodes.append(delta)
        self._member_structurally_toggleable.append(not skippable)

    def member(self, name_or_index) -> MemberHandle:
        """Resolve a name or registration-order index to this
        pipeline's stable :class:`~eagle._member_enable.MemberHandle`.
        Available any time after registration; does not require
        :meth:`build` (only :meth:`set_member_enabled` does, since
        toggleability is a build-time fact)."""
        idx = self._registry.resolve(name_or_index)
        return MemberHandle(idx, self._registry.name_of(idx))

    def set_member_enabled(self, handle_or_index, enabled: bool) -> GraphPipeline:
        """Enable or disable one registered member's launches on this
        instantiated graph, effective on the next :meth:`launch` -- no
        recapture, no change to graph structure (composition mode
        "enabled": capture once at full sibling concurrency, then flip
        driver-level enabled bits between replays). The host-paced
        sibling of the device-paced conditional (IF) node: where a
        conditional's predicate is re-read by an in-graph kernel on
        every replay, this toggles a driver bit from Python between
        replays, for zero per-replay gate cost -- the right tool when
        routing decisions are made in Python, not on-device. A
        member's captured node set never changes after :meth:`build`;
        this only flips whether that fixed node set runs. A disabled
        member's nodes become driver-level no-ops (any buffer it would
        have written is left untouched); an enabled member's nodes run
        exactly as captured. Requires CUDA >= 12.3 (the same floor the
        conditional facility already imposes on the compiled
        ``eagle._core`` extension). Raises
        :class:`~eagle._member_enable.NonToggleableMemberError`
        if :meth:`build` recorded this member as non-toggleable (its
        node set contains something other than kernel/memcpy/memset --
        most commonly a :class:`~eagle._conditional.Skippable`'s
        conditional node). :class:`~eagle._member_enable.UnknownMemberError`
        for a bad name or index, and ``RuntimeError`` if called before
        :meth:`build`.
        """
        if self._launcher is None:
            raise RuntimeError("call build() before set_member_enabled()")
        idx = self._registry.resolve(handle_or_index)
        if not self._member_toggleable[idx]:
            raise NonToggleableMemberError(
                f"member {self._registry.label(idx)} is not toggleable: its "
                "captured node set contains a node type other than "
                "kernel/memcpy/memset (e.g. a Skippable's conditional/IF "
                "node or a RepeatWhile's WHILE node) -- eagle can only "
                "enable/disable the node types CUDA "
                "itself supports toggling; compose this member without an "
                "inner skippable region, or leave it permanently enabled"
            )
        enabled = bool(enabled)
        for node in self._member_nodes[idx]:
            self._launcher.set_node_enabled(node, enabled)
        self._member_enabled[idx] = enabled
        return self

    def set_node_enabled(self, node: int, enabled: bool) -> GraphPipeline:
        """Enable/disable ONE raw captured node directly, bypassing
        this pipeline's own member registry. For a caller that tracks
        its own node attribution outside
        :meth:`add`/:meth:`add_concurrent`'s member bookkeeping. The
        same underlying call :meth:`set_member_enabled` makes per node
        (``Launcher.set_node_enabled``); this is only a narrower, public
        door to it for a caller already holding valid node handles
        (from :func:`~eagle._core.capture_snapshot_nodes`).

        ``node`` must be a real handle from this pipeline's own capture
        -- passing a foreign or stale one is undefined at the driver
        level. Requires :meth:`build` to have run."""
        if self._launcher is None:
            raise RuntimeError("call build() before set_node_enabled()")
        self._launcher.set_node_enabled(int(node), bool(enabled))
        return self

    def is_member_toggleable(self, handle_or_index) -> bool:
        """Whether :meth:`set_member_enabled` will accept this member (its
        captured node set is entirely kernel/memcpy/memset). Requires
        :meth:`build` to have run."""
        if self._launcher is None:
            raise RuntimeError("call build() before is_member_toggleable()")
        idx = self._registry.resolve(handle_or_index)
        return self._member_toggleable[idx]

    def member_nodes(self, handle_or_index) -> list:
        """Raw node handles (as ints) attributed to one registered
        member by :meth:`build` -- the per-member analogue of
        :meth:`node_labels`. A copy; mutating it has no effect. Requires
        :meth:`build` to have run."""
        if self._launcher is None:
            raise RuntimeError("call build() before member_nodes()")
        idx = self._registry.resolve(handle_or_index)
        return list(self._member_nodes[idx])

    def _run_concurrent(self, cp, group, member_runs):
        """Fork ``group.members`` onto separate branch streams inside
        the current capture, then join -- the ``_ConcurrentGroup``
        dispatch for :meth:`add_concurrent`. ``group._fork`` was already
        constructed in :meth:`build`, before capture began; this only
        calls ``fork()``/``branch()``/``join()``, all legal mid-capture.
        Each member runs with the ambient sibling count set to
        ``len(group.members)`` -- the only place in the codebase that
        sets it to anything but the default 1 -- so an eagle kernel
        launched from inside a member can resolve a co-residency-aware
        block size. ``member_runs[i]`` is the pre-built
        :class:`_GuardedRun` for
        member ``i`` if it is a :class:`~eagle._conditional.Skippable`
        or :class:`~eagle._conditional.RepeatWhile` (``None`` otherwise,
        built via :meth:`_prebuild_guarded`). Each member's node-set
        attribution (:meth:`_record_member_`) is taken on its own
        branch stream, immediately before and after it runs -- capture
        is single-threaded host-side, so member ``i``'s launches fully
        enter the graph before member ``i+1``'s capture begins.
        """
        fork = group._fork
        fork.fork()
        siblings = len(group.members)
        for i, member in enumerate(group.members):
            branch = fork.branch(i)
            before = self._core.capture_snapshot_nodes(branch)
            with _wrap_external_stream(branch):
                with sibling_context(siblings):
                    run = member_runs[i]
                    if run is not None:
                        run.run(cp)
                    else:
                        member()
            self._record_member_(before, branch, skippable=(run is not None))
        fork.join()

    def _prebuild_guarded(self, origin, step):
        """Pre-capture half of a conditional step: construct the
        ``_core.CaptureConditional`` weave(s) a
        :class:`~eagle._conditional.Skippable` (one IF weave) or a
        :class:`~eagle._conditional.RepeatWhile` (one WHILE weave, plus
        an inner IF weave when the body step is a ``Skippable``) needs
        on ``origin``. Returns a :class:`_GuardedRun`, or ``None`` for a
        plain step. Dispatches on the exact type: ``RepeatWhile`` is not
        a ``Skippable`` subclass (0..N runs vs 0..1 runs). The guard
        predicate of every weave is evaluated device-side, by eagle's
        own setter kernels, on every replay -- never baked in at
        build/capture time. Refused here, at build: a ``RepeatWhile``
        directly inside a
        ``RepeatWhile`` (no nested-loop contract in v1) and a
        ``RepeatWhile`` inside a ``Skippable`` (v1 supports only the
        other way round)."""
        if isinstance(step, RepeatWhile):
            parts = step.parts
            if any(isinstance(part, RepeatWhile) for part in parts):
                raise ValueError(
                    "repeat_while inside repeat_while is not supported: nest "
                    "the inner iteration inside the body callable, or flatten "
                    "the two loops into one guard"
                )
            if any(isinstance(part, Skippable) and isinstance(part.step, RepeatWhile)
                   for part in parts):
                raise ValueError(
                    "repeat_while inside skippable is not supported: put the "
                    "skippable region inside the loop body instead "
                    "(repeat_while(skippable(step, flag), guard, max_iters))"
                )
            count_ptr, baseline_ptr = step.guard._device_ptrs()
            outer = self._core.CaptureConditional(
                origin, count_ptr, baseline_ptr,
                loop_cap=step.max_iters, counter_ptr=step._counter_ptr(),
            )
            inners = []
            for part in parts:
                inner = None
                if isinstance(part, Skippable):
                    inner_count, inner_baseline = part.guard._device_ptrs()
                    inner = self._core.CaptureConditional(
                        outer.body_stream(), inner_count, inner_baseline
                    )
                inners.append(inner)
            return _GuardedRun(step, outer, tuple(inners))
        if isinstance(step, Skippable):
            if isinstance(step.step, RepeatWhile):
                raise ValueError(
                    "repeat_while inside skippable is not supported: put the "
                    "skippable region inside the loop body instead "
                    "(repeat_while(skippable(step, flag), guard, max_iters))"
                )
            count_ptr, baseline_ptr = step.guard._device_ptrs()
            return _GuardedRun(
                step, self._core.CaptureConditional(origin, count_ptr, baseline_ptr)
            )
        return None

    def launch(self, n=1):
        """Replay the captured graph ``n`` times. ``launch(n)``
        guarantees all work previously enqueued on the caller's current
        cupy stream and the legacy default stream is visible to replay
        1.

        **A stream-ordering fix.** The pipeline's own stream is
        ``non_blocking=True`` precisely so capture never entangles the
        legacy default stream ordinary cupy/torch launches land on --
        but that means replay is not otherwise sequenced after
        host-side seeding (``buffer[...] = ...``) done on the caller's
        current or legacy stream. Under ambient GPU load the seed can
        lose the race and replay 1 reads pre-seed buffer contents. The
        fix: record a cupy Event on the caller's current stream and on
        the legacy ``Stream.null`` (when different) and wait on both,
        on the pipeline's own stream, before the replay loop. This
        orders every consumer of :meth:`launch` for free, at the
        facility seam, with no host block (~1-2us/call). Exactly
        ``enqueue(n)`` followed by :meth:`synchronize` -- same events,
        same order, same host block, same return value. The two halves
        are also available separately (see :meth:`enqueue`) for a
        caller that wants several pipelines' replays in flight at once."""
        return self.enqueue(n).synchronize()

    def _record_seed_events(self, cp) -> list:
        """Record the ordering events and return the list to wait on.
        The event objects are this pipeline's cached pair (see
        :meth:`__init__`); this re-records them on the caller's current
        stream and, only when that is not already the legacy default
        stream, on ``Stream.null``."""
        current = cp.cuda.get_current_stream()
        if self._seed_event is None:
            self._seed_event = cp.cuda.Event()
        self._seed_event.record(current)
        events = [self._seed_event]
        if current.ptr != cp.cuda.Stream.null.ptr:
            if self._legacy_event is None:
                self._legacy_event = cp.cuda.Event()
            self._legacy_event.record(cp.cuda.Stream.null)
            events.append(self._legacy_event)
        return events

    def enqueue(self, n=1, *, pre_launch=None, seed_events=None):
        """Enqueue ``n`` replays on this pipeline's own stream and
        return immediately -- the non-blocking half of :meth:`launch`.
        ``launch(n)`` is ``enqueue(n)`` + :meth:`synchronize`, verbatim.
        Use this pair directly only when several independent pipelines
        should have their device work in flight simultaneously: enqueue
        them all first, then synchronize them in a second pass, so the
        total wall time approximates ``max`` over the pipelines rather
        than their ``sum``.

        The ordering (see :meth:`launch`) is applied here, not in
        :meth:`synchronize`: an event is recorded on the caller's
        current stream (and the legacy ``Stream.null`` when different)
        and waited on this pipeline's own stream before the replays are
        issued. ``pre_launch`` (optional) is a zero-arg callable run on
        this pipeline's own stream after the seed waits and before the
        replays -- the seam for copying per-launch varying data into
        this pipeline's buffers. ``seed_events`` (optional) are
        pre-recorded events to wait on instead of recording this
        pipeline's own -- the shared-event mode
        :meth:`eagle.compose.GraphComposer._async_launch_group` needs,
        one event pair recorded once and waited on every member's
        stream. An empty sequence means "I have already ordered this
        replay myself, issue no seed waits". **HAZARD 1 -- one
        in-flight enqueue per pipeline.** A second
        ``enqueue`` before the first has been synchronized re-records
        the same cached events; overlapping enqueues simply serialize
        on this pipeline's single stream, interleaving in a way the
        caller cannot control. Enqueue once, synchronize, then enqueue
        again.

        **HAZARD 2 -- the reverse write fence is gone.** ``enqueue``
        returns with the replay still in flight; the ordering events
        only say replay-after-seeds, nothing about writes issued
        after::

            pipe.enqueue()
            buf[...] = new_values  # RACE: the in-flight replay may read these
            pipe.synchronize()

        is a data race, silently. The discipline is: ``enqueue`` ->
        (only work that does not touch this pipeline's buffers) ->
        ``synchronize`` -> write; stage new inputs via ``pre_launch``
        instead. Raises ``RuntimeError`` if called before
        :meth:`build`."""
        if self._launcher is None:
            raise RuntimeError("call build() before launch()")
        import cupy as cp

        if seed_events is None:
            seed_events = self._record_seed_events(cp)
        for event in seed_events:
            self._ext.wait_event(event)

        if pre_launch is not None:
            with self._ext:
                pre_launch()

        for _ in range(int(n)):
            self._launcher.launch()
        return self

    def synchronize(self):
        """Block the host until this pipeline's enqueued replays have
        completed -- the blocking half of :meth:`launch`. Safe to call
        with nothing in flight (a cheap no-op query on an already-idle
        stream). Raises ``RuntimeError`` if called before
        :meth:`build`."""
        if self._launcher is None:
            raise RuntimeError("call build() before launch()")
        self._launcher.synchronize()
        return self

    def _launch_handles(self) -> _LaunchHandles:
        """Module-private: hand back this pipeline's own (stream,
        launcher) -- see :class:`_LaunchHandles` for what it is for.
        Never call this instead of :meth:`launch` for ordinary replay,
        or instead of :meth:`enqueue`/:meth:`synchronize` for the
        in-flight case -- those are the public, hazard-documented door.
        Raises ``RuntimeError`` if called before :meth:`build`."""
        if self._launcher is None:
            raise RuntimeError("call build() before _launch_handles()")
        return _LaunchHandles(self._ext, self._launcher)

    @property
    def graph(self):
        return self._graph

    @property
    def stream(self) -> int:
        """This pipeline's own ``cudaStream_t``, as a raw integer handle
        (the family's shared ``stream`` name: the same value a DLPack
        consumer passes as ``__dlpack__(stream=...)``). For capture
        composition: a caller that needs to bind a cuBLAS/cuSOLVER-style
        handle to this pipeline's stream, or run warm-up passes before
        :meth:`build` captures, needs this exact stream -- launching on
        any other stream would race the graph. Read-only, valid for
        this pipeline's entire lifetime."""
        return self._sptr

    # -- introspection (proves the JIT launch really is a graph node) --------
    def introspection_available(self) -> bool:
        """Whether node introspection works here: the bound core dots
        the captured graph through ``cudaGraphDebugDotPrint`` (always
        available), so this is ready as soon as :meth:`build` has run."""
        return self._dot_str is not None

    def _dot(self) -> str:
        if self._dot_str is None:
            raise RuntimeError("call build() first")
        return self._dot_str

    def num_nodes(self) -> int:
        """Number of kernel nodes in the captured graph."""
        return self._dot().count('shape="octagon"')

    def node_labels(self) -> list:
        """Kernel function names of the captured nodes, in declaration order."""
        return re.findall(r'label="\d+\n([^\n"]+)', self._dot())


def _debug_dot(captured) -> str:
    """Graphviz dot of a captured graph, via a short-lived temp file."""
    fd, path = tempfile.mkstemp(suffix=".dot")
    os.close(fd)
    try:
        return captured.debug_dot(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
