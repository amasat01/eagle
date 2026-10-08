# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.compose`` — compose a set of independently-built launchable
entities into ONE launch list, four ways.

Where :class:`~eagle.pipeline.GraphPipeline` captures one sequence of
launches into one replayable graph, :class:`GraphComposer` is the layer
above it: it takes a collection of already-built members -- each a
self-contained launchable (most commonly its own ``GraphPipeline``, but a
built ``mode="sequenced"`` :class:`GraphComposer`, or a plain zero-arg
step callable, both qualify too) -- and drives them as one unit, with
per-member enable/disable, four routing mechanisms to choose from, and an
audit trail of which members actually fired on each launch.

The motivating shape is generic, not AI-specific: an ensemble of
independently-captured launchables a caller wants to fire together,
selectively, and cheaply re-route between replays without re-capturing
everything -- e.g. a batch of independent spacecraft-trajectory
propagation pipelines, where "which trajectories are still active"
changes over time. Nothing here knows or cares what a member computes,
and this module never imports anything outside eagle.

**The four modes** (``mode=`` at construction):

* ``mode="sequenced"`` (the default) -- every member keeps its own
  launchable; this composer decides, per launch, which fire. Members
  may nest (the recursion lock, below).
* ``mode="enabled"`` -- host-paced toggles: every member is captured
  once into one flat super-graph, and switching a member on/off is a
  microsecond host-side flag flip, no recapture ever.
* ``mode="rebuild"`` -- rarely-changing routing: every
  :meth:`~GraphComposer.set_routing` call recaptures a fresh
  super-graph containing only the active members.
* ``mode="conditional"`` -- device-paced: one captured super-graph with
  an in-graph conditional (IF) node per member, re-read from a device
  array every replay -- the only mode that can route a decision made
  *by a kernel*.

**The recursion lock.** A built ``mode="sequenced"`` :class:`GraphComposer`
is itself a valid member of another ``mode="sequenced"`` composer, to any
depth. Nesting under, or of, any other mode is rejected loudly at
registration time: the three flat modes fuse everyone into one captured
artifact with no per-member launch list to select a nested composer into.

**Fired-member recording**, in every mode: every :meth:`~GraphComposer.launch`
call appends the active-member index set to a history, retrievable via
:meth:`~GraphComposer.fired_history` and clearable via
:meth:`~GraphComposer.reset_fired_history`.

**Scope.** This module never reasons about what a member computes. A
caller that needs input marshaling supplies that as part of the member
itself, or, for ``mode="sequenced"`` members, via the optional
``pre_launch`` hook :meth:`~GraphComposer.register` accepts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._conditional import SkipGuard, skippable
from .pipeline import GraphPipeline

#: The four composition mechanisms :class:`GraphComposer` dispatches on.
#: ``"sequenced"`` is the default; the other three fuse members into one
#: flat super-graph.
_MODES = ("sequenced", "conditional", "enabled", "rebuild")


@dataclass(frozen=True)
class _Member:
    """One registered member.

    ``kind`` is ``"launchable"`` (a built object replayed via its own
    ``.launch(n)`` -- ``mode="sequenced"`` only), ``"callable"`` (a plain
    zero-arg step -- every mode), or ``"nested"`` (a built
    ``mode="sequenced"`` :class:`GraphComposer` -- the recursion lock).
    ``pre_launch`` (``mode="sequenced"`` only) is an optional zero-arg
    hook run immediately before this member's replay(s). ``name`` is
    optional caller bookkeeping (duplicate checks, and the eagle member
    name for the three flat modes)."""

    obj: Any
    name: Any
    kind: str  # "launchable" | "callable" | "nested"
    pre_launch: Any = None


def _module_name(arr) -> str:
    return type(arr).__module__.split(".")[0]


def _validate_member_flags(flags) -> None:
    """``mode="conditional"``'s intent-guard contract: a uint32 device
    array, since ``SkipGuard._device_ptrs()`` reads ``arr.data.ptr`` and
    capture itself is cupy-only. Both rejects are named here rather than
    left to surface as a bare ``AttributeError`` deep inside the capture
    weave. The other three modes are host-paced and never call this."""
    module = _module_name(flags)
    if module != "cupy":
        raise TypeError(
            "GraphComposer: member_flags must be a cupy (device) uint32 "
            f"array — got a {module!r}-module array ({type(flags).__name__}); "
            "this composer's single shared capture needs a device pointer "
            "per lane (eagle.SkipGuard._device_ptrs() has no host-array "
            "path) — upload with cupy.asarray(...) first"
        )
    if flags.dtype != np.uint32:
        raise TypeError(
            f"GraphComposer: member_flags dtype must be uint32; got "
            f"{flags.dtype} (the intent-guard contract: "
            "eagle.SkipGuard.intent(...) reads one uint32 word per lane)"
        )


def _normalize_pattern(pattern, n: int) -> list:
    """Host-side routing-pattern normalization for ``"enabled"``/
    ``"rebuild"``/``"sequenced"`` (device-paced ``"conditional"`` never
    calls this). Accepts any sequence of truthy/falsy values -- a list,
    tuple, numpy array, or cupy array (downloaded via ``.get()`` first).
    Returns a plain ``list[bool]``, length-checked against the registered
    member count."""
    if hasattr(pattern, "get"):  # a cupy device array -- host it first
        pattern = pattern.get()
    seq = [bool(int(v)) for v in pattern]
    if len(seq) != n:
        raise ValueError(
            f"GraphComposer: routing pattern length {len(seq)} != "
            f"registered member count {n}"
        )
    return seq


class GraphComposer:
    """Compose N members into one launch list, four ways (``mode=``) --
    see the module docstring for the pitch and each mode's guidance.

    ``member_flags`` (device routing flags / host routing pattern,
    retained by reference, never rebound):

    * ``mode="conditional"`` -- required, a ``uint32`` cupy array, one
      lane per member.
    * the other three modes -- any host-side sequence of truthy/falsy
      values, or ``None`` for "every member starts active".

    A caller may mutate ``member_flags``'s contents (via
    :meth:`set_routing`, or -- ``mode="conditional"`` only -- direct
    index assignment) between replays, without recapture, but must never
    rebind the array itself.
    """

    def __init__(self, member_flags, *, mode: str = "sequenced") -> None:
        if mode not in _MODES:
            raise ValueError(
                f"GraphComposer: mode must be one of {list(_MODES)}; got "
                f"{mode!r}"
            )
        if mode == "conditional":
            _validate_member_flags(member_flags)
        self._mode = mode
        self._member_flags = member_flags
        self._members: list[_Member] = []
        self._registered_ids: set[int] = set()
        self._names: set[Any] = set()
        self._pipe = None
        self._built = False
        #: Host-known active-member mask, index-aligned with
        #: ``self._members`` -- authoritative for every mode except
        #: "conditional" (see _record_fired_flat()).
        self._active_mask: list = []
        self._fired_history: list = []

    # ------------------------------------------------------------ registration

    def register(self, obj, *, name=None, pre_launch=None) -> int:
        """Register one member; returns its assigned index (registration
        order).

        ``obj`` may be:

        * a built launchable -- any object with a working ``.launch(n)``
          (duck-typed). ``mode="sequenced"`` only.
        * a plain zero-arg step callable -- valid in every mode: one
          member of this composer's own super-``GraphPipeline`` under
          the three flat modes, replayed directly under
          ``mode="sequenced"``.
        * a built ``mode="sequenced"`` :class:`GraphComposer` -- the
          recursion lock. ``mode="sequenced"`` outer composers only.

        ``pre_launch`` (``mode="sequenced"`` only) is an optional
        zero-arg callable run immediately before this member's
        replay(s) -- on that member's own stream if it joins the
        cross-member overlap group, otherwise host-side. Common
        rejections, loud: registering after :meth:`build`; the same
        object registered twice; a colliding ``name``; nesting a
        composer under a non-``"sequenced"`` outer composer, or nesting
        an unbuilt/wrong-mode composer under a ``"sequenced"`` one; a
        ``pre_launch`` hook outside ``mode="sequenced"``; an ``obj``
        that is none of the three accepted shapes.
        """
        if self._built:
            raise RuntimeError(
                "GraphComposer.register: cannot register after build() — "
                "this composer's capture is already committed"
            )
        if isinstance(obj, GraphComposer) and self._mode != "sequenced":
            raise ValueError(
                f"GraphComposer.register: this composer is mode="
                f"{self._mode!r} — nesting a composer as a member is only "
                "supported under a mode='sequenced' OUTER composer (the "
                "RECURSION LOCK covers sequenced-mode nesting only this "
                "phase) — build the outer composer with mode='sequenced' "
                "instead"
            )
        if pre_launch is not None and self._mode != "sequenced":
            raise ValueError(
                "GraphComposer.register: pre_launch is only accepted for "
                "mode='sequenced' members — the other three modes fuse "
                "every member into ONE captured/flat step already"
            )
        if name is not None and name in self._names:
            raise ValueError(
                f"GraphComposer.register: duplicate member name {name!r}"
            )
        key = id(obj)
        if key in self._registered_ids:
            raise ValueError(
                "GraphComposer.register: this object is already "
                "registered — each member may be registered at most once"
            )

        if isinstance(obj, GraphComposer):
            member = self._make_nested_member(obj, name=name, pre_launch=pre_launch)
        elif self._mode == "sequenced":
            member = self._make_sequenced_member(
                obj, name=name, pre_launch=pre_launch
            )
        else:
            member = self._make_flat_member(obj, name=name)

        index = len(self._members)
        self._members.append(member)
        self._registered_ids.add(key)
        if name is not None:
            self._names.add(name)
        return index

    def _make_nested_member(self, obj, *, name, pre_launch) -> _Member:
        """The recursion lock's own validation: ``obj`` must be a built,
        ``mode="sequenced"`` composer."""
        if obj._mode != "sequenced":
            raise ValueError(
                "GraphComposer.register: nesting a composer requires BOTH "
                f"composers be mode='sequenced' (this composer is "
                f"mode={self._mode!r}, the registered one is "
                f"mode={obj._mode!r}) — sequenced-mode nesting is "
                "supported only this phase; other-mode nesting is rejected "
                "loudly rather than silently misbehaving"
            )
        if not obj._built:
            raise ValueError(
                "GraphComposer.register: a nested composer must be "
                "build()-ed before registration — its own launch() "
                "contract requires it"
            )
        return _Member(obj, name, kind="nested", pre_launch=pre_launch)

    def _make_sequenced_member(self, obj, *, name, pre_launch) -> _Member:
        """``mode="sequenced"``'s non-composer path: a built launchable
        (anything with a working ``.launch(n)``) or a plain zero-arg
        step callable."""
        if hasattr(obj, "launch") and callable(obj.launch):
            return _Member(obj, name, kind="launchable", pre_launch=pre_launch)
        if callable(obj):
            return _Member(obj, name, kind="callable", pre_launch=pre_launch)
        raise TypeError(
            "GraphComposer.register: mode='sequenced' accepts a built "
            "launchable (an object with .launch(n)), a plain zero-arg step "
            f"callable, or a built mode='sequenced' GraphComposer — got "
            f"{type(obj).__name__}"
        )

    def _make_flat_member(self, obj, *, name) -> _Member:
        """The three flat modes' registration path: one plain zero-arg
        step callable per member, fused into a single
        super-``GraphPipeline`` at :meth:`build` time."""
        if not callable(obj):
            raise TypeError(
                f"GraphComposer.register: mode={self._mode!r} accepts a "
                f"plain zero-arg step callable — got {type(obj).__name__}"
            )
        return _Member(obj, name, kind="callable", pre_launch=None)

    # ------------------------------------------------------------------ build

    def _normalize(self, pattern, n: int) -> list:
        """Instance-level indirection to :func:`_normalize_pattern`, as a
        subclass extension point (e.g. monkeypatch-based fault injection
        in a test) without overriding :meth:`build`/:meth:`set_routing`."""
        return _normalize_pattern(pattern, n)

    def build(self):
        """Perform whatever one-time setup ``mode`` needs. Returns
        ``self`` (rejected, every mode, if no members are registered).

        ``mode="conditional"`` wraps every step in a ``skippable`` guard
        and captures one ``GraphPipeline`` once; ``mode="enabled"``
        registers every member as a plain member of one
        ``GraphPipeline``, captures once, then applies the initial
        pattern via ``set_member_enabled`` (no recapture from here on);
        ``mode="rebuild"`` captures a super-graph of only the active
        members, re-capturing from scratch on every later
        :meth:`set_routing`; ``mode="sequenced"`` is bookkeeping only,
        since every member already owns its own launchable.
        """
        if self._built:
            raise RuntimeError("GraphComposer.build: already built (one capture)")
        if not self._members:
            raise ValueError(
                "GraphComposer.build: no members registered — call "
                "register(...) at least once before build()"
            )

        if self._mode == "sequenced":
            self._active_mask = (
                [True] * len(self._members)
                if self._member_flags is None
                else self._normalize(self._member_flags, len(self._members))
            )
            self._built = True
            return self

        if self._mode == "conditional":
            if len(self._member_flags) != len(self._members):
                raise ValueError(
                    f"GraphComposer.build: flags array length "
                    f"{len(self._member_flags)} != registered member count "
                    f"{len(self._members)}"
                )
            gated = []
            for i, member in enumerate(self._members):
                guard = SkipGuard.intent(self._member_flags, i)
                gated.append(skippable(member.obj, guard))
            self._pipe = _build_flat_pipe(gated, names=None)
            self._built = True
            return self

        pattern = (
            [True] * len(self._members)
            if self._member_flags is None
            else self._normalize(self._member_flags, len(self._members))
        )

        if self._mode == "enabled":
            members = [m.obj for m in self._members]
            names = [m.name for m in self._members]
            self._pipe = _build_flat_pipe(members, names=names)
            for i, on in enumerate(pattern):
                self._pipe.set_member_enabled(i, on)
            self._active_mask = pattern
            self._built = True
            return self

        # mode == "rebuild"
        self._rebuild_pipe(pattern)
        self._built = True
        return self

    def _rebuild_pipe(self, pattern) -> None:
        """``mode="rebuild"``'s per-routing-change recapture: a fresh
        ``GraphPipeline`` containing only the members active under
        ``pattern`` (no ``SkipGuard`` wrapping -- an inactive member is
        simply absent). ``pattern == all False`` captures nothing
        (:meth:`launch` becomes a no-op)."""
        active_indices = [i for i, on in enumerate(pattern) if on]
        if not active_indices:
            self._pipe = None
            self._active_mask = list(pattern)
            return
        members = [self._members[i].obj for i in active_indices]
        names = [self._members[i].name for i in active_indices]
        self._pipe = _build_flat_pipe(members, names=names)
        self._active_mask = list(pattern)

    # ---------------------------------------------------------------- routing

    def set_routing(self, pattern) -> GraphComposer:
        """Update which members are active.

        Mechanism + cost depend on ``mode``: ``"conditional"`` writes the
        device flags array in place (no recapture); ``"enabled"`` calls
        ``GraphPipeline.set_member_enabled`` per member (no recapture,
        µs-scale toggle); ``"rebuild"`` re-captures a fresh super-graph
        (paid here, not per replay); ``"sequenced"`` updates the
        host-known launch list.

        ``pattern`` is any sequence of truthy/falsy values, length ==
        registered member count."""
        if not self._built:
            raise RuntimeError(
                "GraphComposer.set_routing: call build() before set_routing()"
            )
        normalized = self._normalize(pattern, len(self._members))
        if self._mode == "conditional":
            for i, on in enumerate(normalized):
                self._member_flags[i] = 1 if on else 0
            return self
        if self._mode == "enabled":
            for i, on in enumerate(normalized):
                self._pipe.set_member_enabled(i, on)
            self._active_mask = normalized
        elif self._mode == "rebuild":
            self._rebuild_pipe(normalized)
        else:  # "sequenced"
            self._active_mask = normalized
        return self

    # ------------------------------------------------------------------ replay

    def launch(self, n=1):
        """Replay ``n`` times. Returns ``self``.

        The three flat modes delegate to the one super-pipeline's
        ``GraphPipeline.launch()`` (a ``mode="rebuild"`` composer with an
        all-inactive pattern owns no pipe, so ``n`` replays of nothing is
        a no-op). ``mode="sequenced"`` fires every currently-active
        registered member (:meth:`_launch_sequenced`), with the
        cross-member overlap treatment for stream-owning launchables."""
        if not self._built:
            raise RuntimeError("GraphComposer.launch: call build() before launch()")
        if self._mode == "sequenced":
            return self._launch_sequenced(n)
        self._record_fired_flat()
        if self._pipe is None:
            return self
        self._pipe.launch(n)
        return self

    def _record_fired_flat(self) -> None:
        """Fired-set bookkeeping for the three flat modes. For
        ``mode="conditional"`` this downloads the live device flags array
        (once, host round-trip) rather than trusting
        ``self._active_mask``, since a caller may mutate ``member_flags``
        by direct index assignment, bypassing :meth:`set_routing`. The
        other two modes only ever change routing through
        :meth:`set_routing`, so the tracked mask is authoritative."""
        if self._mode == "conditional":
            host = self._member_flags.get()
            fired = frozenset(i for i, v in enumerate(host) if int(v) != 0)
        else:
            fired = frozenset(i for i, on in enumerate(self._active_mask) if on)
        self._fired_history.append(fired)

    # -- mode="sequenced": the two-level launch-list + cross-member overlap --

    def _launch_sequenced(self, n=1):
        active = [i for i, on in enumerate(self._active_mask) if on]
        self._fired_history.append(frozenset(active))
        if not active:
            return self

        overlap_group, blocking = [], []
        for i in active:
            member = self._members[i]
            if (
                member.kind == "launchable"
                and hasattr(member.obj, "_launch_handles")
                and hasattr(member.obj, "enqueue")
            ):
                overlap_group.append(member)
            else:
                blocking.append(member)

        # Blocking members (a plain callable, a stream-less launchable,
        # or a nested composer, whose own launch() already owns its
        # join) run in registration order; only the stream-owning
        # subset gets the overlap treatment.
        for member in blocking:
            self._replay_member_blocking(member, n)

        if overlap_group:
            self._async_launch_group(overlap_group, n)
        return self

    def _replay_member_blocking(self, member: _Member, n) -> None:
        if member.pre_launch is not None:
            member.pre_launch()
        if member.kind == "callable":
            for _ in range(int(n)):
                member.obj()
        else:
            member.obj.launch(n)

    def _async_launch_group(self, members, n) -> None:
        """The sequenced-mode overlap seam: enqueue every stream-owning
        active member's replay(s) first, on its own stream, then
        synchronize every stream in a second pass, so independent
        members' device work overlaps instead of serializing host-side.

        A member owns an independent, non-blocking CUDA stream if it is
        a built :class:`~eagle.pipeline.GraphPipeline` (every
        ``GraphPipeline.__init__`` creates its own
        ``_core.Stream(non_blocking=True)``), so replays are
        structurally able to overlap on device -- but the only public
        entry point, :meth:`~eagle.pipeline.GraphPipeline.launch`, ends
        in a host-blocking ``synchronize()`` every call, so calling it
        once per member in a loop serializes every replay host-side.
        This method does the same split across N pipelines instead of
        within one: :meth:`~eagle.pipeline.GraphPipeline.enqueue` every
        member's replay(s) on its own stream first (fast --
        ``cudaGraphLaunch`` does not block for device completion), then
        :meth:`~eagle.pipeline.GraphPipeline.synchronize` every member in
        a second pass, so elapsed time approximates ``max`` over active
        members rather than their ``sum``. Threading was rejected: this
        binding never releases the GIL, so a thread blocked inside
        ``synchronize()`` would stall the thread enqueuing the next
        member, fully re-serializing with extra overhead.

        **Correctness.** Each pipeline's stream is ``non_blocking=True``
        and has no implicit ordering against the caller's current/legacy
        stream, so a host-side buffer seed done immediately before
        replay can otherwise race it. One event pair is recorded here --
        on the caller's current stream, and on the legacy default stream
        when different -- and handed to every member's
        ``enqueue(seed_events=...)``, which waits on both before issuing
        its ``pre_launch`` hook (if any, see :meth:`register`; it runs on
        that member's own stream) or its graph launch(es): exactly
        ``GraphPipeline.launch``'s own two-event ordering, recorded once
        across every member's stream instead of once per pipeline.

        **In-flight window.** Between the two passes every member's
        replay is in flight, so ``GraphPipeline.enqueue``'s hazard about
        buffer writes applies here too: nothing may write a member's
        device buffers until the second pass returns, and a
        ``pre_launch`` hook must not touch a different member's buffers.
        """
        import cupy as cp

        current = cp.cuda.get_current_stream()
        seed_event = cp.cuda.Event()
        seed_event.record(current)
        seed_events = [seed_event]
        if current.ptr != cp.cuda.Stream.null.ptr:
            legacy_event = cp.cuda.Event()
            legacy_event.record(cp.cuda.Stream.null)
            seed_events.append(legacy_event)

        for member in members:
            member.obj.enqueue(
                n, pre_launch=member.pre_launch, seed_events=seed_events
            )

        for member in members:
            member.obj.synchronize()

    # ------------------------------------------------------------ introspection

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def member_flags(self):
        return self._member_flags

    @property
    def num_members(self) -> int:
        return len(self._members)

    def fired_history(self) -> list:
        """Per-:meth:`launch`-call fired-member index sets
        (``frozenset``), oldest first, since the last
        :meth:`reset_fired_history` (or construction)."""
        return list(self._fired_history)

    def reset_fired_history(self) -> None:
        """Clear the fired-set history (does not affect current routing)."""
        self._fired_history.clear()

    def num_nodes(self) -> int:
        """Kernel-node count of the one captured super-graph (after
        :meth:`build`). Not available for ``mode="sequenced"`` (no single
        super-graph -- introspect each member on its own), nor for a
        ``mode="rebuild"`` composer whose current pattern is
        all-inactive."""
        self._require_flat_pipe("num_nodes")
        return self._pipe.num_nodes()

    def node_labels(self) -> list:
        """Kernel names of the captured nodes, in declaration order.
        Same availability caveats as :meth:`num_nodes`."""
        self._require_flat_pipe("node_labels")
        return self._pipe.node_labels()

    def _require_flat_pipe(self, who: str) -> None:
        if not self._built:
            raise RuntimeError(
                f"GraphComposer.{who}: call build() before introspection"
            )
        if self._mode == "sequenced":
            raise RuntimeError(
                f"GraphComposer.{who}: mode='sequenced' has no single "
                "super-graph to introspect — it is a LIST of independent "
                f"per-member launchables (the two-level launch-list "
                f"shape); call {who}() on each registered member's own "
                "launchable instead"
            )
        if self._pipe is None:
            raise RuntimeError(
                f"GraphComposer.{who}: no members are currently active "
                "(mode='rebuild' with an all-inactive routing pattern "
                "captures nothing)"
            )


def _build_flat_pipe(members, *, names) -> GraphPipeline:
    """Shared helper for the three flat (single-super-graph) modes'
    construction: ``add_concurrent`` at N>=2 members, plain ``add`` at
    N==1."""
    pipe = GraphPipeline()
    if len(members) >= 2:
        if names is not None and any(nm is not None for nm in names):
            pipe.add_concurrent(members, names=names)
        else:
            pipe.add_concurrent(members)
    else:
        name = names[0] if names else None
        if name is not None:
            pipe.add(members[0], name=name)
        else:
            pipe.add(members[0])
    pipe.build()
    return pipe
