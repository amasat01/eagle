# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Declarative skip-guard surface for CUDA-graph conditional nodes.

``eagle::CountGuard`` (C++) is the predicate this facility is built on: a
region runs iff ``count_arr[count_idx] != (baseline_arr[baseline_idx] or
0)``. :class:`SkipGuard` is the Python mirror, retaining the arrays and
exposing both :meth:`evaluate` (host read, eager/numpy/debug paths) and
the raw device pointers :meth:`GraphPipeline.build` weaves into a
``_core.CaptureConditional`` (CUDA capture path).

:class:`Skippable` (:func:`skippable`) wraps one zero-arg step with a
guard. :class:`RepeatWhile` (:func:`repeat_while`) is the loop sibling:
the step repeats on the device while the guard holds, up to a build-time
cap -- one CUDA WHILE conditional node under :meth:`GraphPipeline.build`,
a host loop when called directly.
"""

from __future__ import annotations


from . import _alloc, arrays
from . import _core

#: The device cap of a loop with no ``max_iters``: "until the guard fails".
UNCAPPED = _core.LOOP_UNCAPPED


def _host_scalar(arr, idx: int) -> int:
    """Read ``arr[idx]`` as a plain Python ``int``. Indexing a cupy array
    returns a 0-d cupy array; reading it (``.get()``) is a device sync,
    which is why this is eager/debug-path only. A numpy array's element
    has no ``.get()``, so it is coerced directly."""
    value = arr[idx]
    get = getattr(value, "get", None)
    return int(get()) if get is not None else int(value)


def _device_ptr(arr, idx: int) -> int:
    """Raw device pointer to ``arr[idx]`` as a Python int (cupy arrays
    only). ``idx`` is scaled by the array's own element size, so this
    works for any uint32-sized dtype view."""
    ptr = arr.ptr if hasattr(arr, "ptr") else arr.data.ptr
    return int(ptr) + int(idx) * arr.dtype.itemsize


class SkipGuard:
    """Region runs iff ``count_arr[count_idx] != (baseline_arr[baseline_idx]
    or 0)``. Arrays are uint32 device (cupy) or host (numpy) arrays;
    references are retained so pointer lifetime stays pinned to the
    pipeline."""

    __slots__ = ("count_arr", "count_idx", "baseline_arr", "baseline_idx", "negate")

    def __init__(self, count_arr, count_idx=0, baseline_arr=None, baseline_idx=0, negate=False):
        self.negate = bool(negate)
        self.count_arr = count_arr
        self.count_idx = int(count_idx)
        self.baseline_arr = baseline_arr
        self.baseline_idx = int(baseline_idx)

    @classmethod
    def nonempty_bucket(cls, type_offsets, t):
        """Fencepost liveness: ``type_offsets[t + 1] != type_offsets[t]``."""
        return cls(type_offsets, t + 1, type_offsets, t)

    @classmethod
    def nonzero(cls, n_active):
        """Plain nonzero count: ``n_active[0] != 0``."""
        return cls(n_active, 0)

    @classmethod
    def intent(cls, flags, index):
        """Router/user-owned intent flag: ``flags[index] != 0``.

        ``flags`` is a uint32 array the caller owns and writes between
        graph replays, one lane per guarded region (e.g.
        ``SkipGuard.intent(expert_flags, e)`` for expert ``e``). No
        baseline: a plain on/off decision.

        Under :meth:`GraphPipeline.build` the predicate is read fresh
        from the device every replay, so flipping ``flags[index]`` takes
        effect with no recapture; in eager/numpy mode the same flag is
        read host-side by :meth:`evaluate`.

        ``index`` has no default (unlike :meth:`nonzero`'s ``0``): it is
        load-bearing at every call site.
        """
        return cls(flags, index)

    def negated(self) -> SkipGuard:
        """The guard that holds exactly when this one does not (the ``else``
        arm); one device-side flag, no derived cell."""
        return SkipGuard(self.count_arr, self.count_idx, self.baseline_arr,
                         self.baseline_idx, not self.negate)

    def evaluate(self) -> bool:
        """Host read; eager/debug paths ONLY (syncs on cupy)."""
        count = _host_scalar(self.count_arr, self.count_idx)
        baseline = (
            _host_scalar(self.baseline_arr, self.baseline_idx)
            if self.baseline_arr is not None
            else 0
        )
        return (count != baseline) != self.negate

    def _device_ptrs(self):
        """(count_ptr, baseline_ptr) raw ints for the CUDA capture weave.
        ``baseline_ptr`` is ``0`` (reads as nullptr) when there is no
        baseline array. Internal, used by :meth:`GraphPipeline.build`."""
        count_ptr = _device_ptr(self.count_arr, self.count_idx)
        baseline_ptr = (
            _device_ptr(self.baseline_arr, self.baseline_idx)
            if self.baseline_arr is not None
            else 0
        )
        return count_ptr, baseline_ptr


class Skippable:
    """A zero-arg step wrapped with a :class:`SkipGuard`. Works as a
    :meth:`GraphPipeline.add` step or an ``add_concurrent`` member --
    :meth:`GraphPipeline.build` recognizes it by type and weaves an IF
    node around it; called directly, it evaluates its guard eagerly: ``if
    guard.evaluate(): step()``."""

    __slots__ = ("step", "guard")

    def __init__(self, step, guard: SkipGuard):
        self.step = step
        self.guard = guard

    def __call__(self):
        if self.guard.evaluate():
            self.step()


def skippable(step, guard: SkipGuard) -> Skippable:
    """Wrap ``step`` (a zero-arg callable) so it only runs while ``guard``
    holds: one IF node under :meth:`GraphPipeline.build`, an eager guard
    check everywhere else.
    """
    return Skippable(step, guard)


class Branches:
    """An IF/ELSE (``switch=False``: arm 0 when ``guard`` holds, arm 1
    otherwise) or a SWITCH (``guard.count_arr[count_idx]`` is the arm index;
    none runs out of range) over zero-arg steps. One conditional node with an
    arm per step where the driver has it (:func:`has_branch_nodes`); the Graph
    builds the same two arms from IFs otherwise. Called directly it runs the
    arm eagerly."""

    __slots__ = ("arms", "guard", "switch")

    def __init__(self, arms, guard: SkipGuard, switch=False):
        self.arms, self.guard, self.switch = tuple(arms), guard, bool(switch)

    def __call__(self):
        i = (_host_scalar(self.guard.count_arr, self.guard.count_idx) if self.switch
             else 0 if self.guard.evaluate() else 1)
        if i < len(self.arms):
            self.arms[i]()


def has_branch_nodes(switch=False) -> bool:
    """Does the driver have IF/ELSE (``switch=False``) or SWITCH conditional
    nodes?"""
    try:
        bit = _core.CAP_CONDITIONAL_SWITCH if switch else _core.CAP_CONDITIONAL_IF_ELSE
        return bool(_core.cuda_backend()["capabilities"] & bit)
    except Exception:
        return False


class RepeatWhile:
    """A zero-arg step repeated on the device while a :class:`SkipGuard`
    holds, at most ``max_iters`` times per replay (``None``: until the guard
    fails) -- the loop sibling of :class:`Skippable` (not a subclass: 0..N
    runs and 0..1 runs are different contracts).

    Under :meth:`GraphPipeline.build` the step becomes the body of one
    CUDA WHILE conditional node: a head kernel resets the
    ``[remaining, ran]`` cell and decides entry, the body runs, and a tail
    kernel decides continuation. ``launch(1)`` is one ``cudaGraphLaunch``
    regardless of iteration count; ``launch(n)`` replays the whole loop
    ``n`` times.

    ``max_iters`` is ``None`` or a plain Python ``int >= 1`` frozen at build
    as a kernel argument -- a new cap needs a new ``build()``. A reverse pass
    needs the cap. ``limit`` (a ``uint32`` cell) is a runtime trip count
    (``ran < limit[0]``) and ``brk`` (a ``uint32`` cell the body writes) ends the
    loop after the iteration that leaves it nonzero; ``guard`` may be ``None``
    when one of them stops the loop. Called directly
    it is the host arm: ``while ran < max_iters and guard.evaluate():
    step()``, writing the same cell so :meth:`iterations` reads the same
    in every mode. The cell lives in the guard's array module (cupy or
    numpy), allocated at construction; :attr:`iteration_index` exposes
    its ``ran`` word to body kernels as the device-visible iteration
    index (0 on the first iteration).

    Composition: the body ``step`` may be a :class:`Skippable`, or a
    tuple of such parts run in order (e.g.
    ``(k_steps, skippable(compact, guard))``); a ``RepeatWhile``, a
    ``Skippable`` or a fork/join group nests in any of them, and a
    ``RepeatWhile`` may be an ``add_concurrent`` member. One hidden inside an
    opaque body callable is not detected structurally: called mid-capture it
    runs its host arm, whose guard read syncs and fails the capture loudly.
    """

    __slots__ = ("step", "guard", "max_iters", "limit", "brk", "_cell")

    def __init__(self, step, guard: SkipGuard | None, max_iters: int | None = None,
                 limit=None, brk=None):
        if max_iters is not None and (isinstance(max_iters, bool) or not isinstance(max_iters, int)):
            raise TypeError(
                f"repeat_while: max_iters must be a Python int or None, got "
                f"{type(max_iters).__name__}"
            )
        if max_iters is not None and max_iters < 1:
            raise ValueError(f"repeat_while: max_iters must be >= 1, got {max_iters}")
        if isinstance(step, list):
            step = tuple(step)
        if isinstance(step, tuple):
            if not step:
                raise ValueError(
                    "repeat_while: a sequence body needs at least one part")
            bad = [p for p in step if not callable(p)]
            if bad:
                raise TypeError(
                    "repeat_while: every part of a sequence body is a zero-arg "
                    f"callable or a Skippable, got {type(bad[0]).__name__}"
                )
        self.step = step
        anchor = guard.count_arr if guard is not None else limit if limit is not None else brk
        if anchor is None:
            raise ValueError("repeat_while: a loop needs a guard, a limit or a brk cell to stop")
        self.guard = guard
        self.max_iters = max_iters
        self.limit, self.brk = limit, brk
        self._cell = _alloc.zeros(_alloc.place(anchor), 2, dtype=arrays.uint32)  # [remaining, ran]

    @property
    def parts(self) -> tuple:
        """The body as a tuple of parts, run in order (a single-step body is
        a one-part tuple)."""
        return self.step if isinstance(self.step, tuple) else (self.step,)

    @property
    def iteration_index(self):
        """One-element ``uint32`` view of the ``ran`` word -- the 0-based
        index of the iteration in flight. Pass it to a body kernel that
        needs it; same array module as the guard."""
        return self._cell[1:2]

    @property
    def cap(self) -> int:
        """The device cap: ``max_iters``, or :data:`UNCAPPED`."""
        return UNCAPPED if self.max_iters is None else self.max_iters

    def iterations(self) -> int:
        """Iterations the last run performed (host read of ``ran``; a
        device sync under a cupy guard, eager/debug paths only). 0 before
        any run."""
        return _host_scalar(self._cell, 1)

    def _counter_ptr(self) -> int:
        """Raw device pointer to the ``[remaining, ran]`` cell for the CUDA
        capture weave. Internal -- used by :meth:`GraphPipeline.build`."""
        return _device_ptr(self._cell, 0)

    def __call__(self):
        ran = 0
        parts = self.parts
        if self.brk is not None:
            self.brk[0] = 0
        while (ran < self.cap and (self.limit is None or ran < _host_scalar(self.limit, 0))
               and (self.guard is None or self.guard.evaluate())):
            self._cell[1] = ran          # the index a body kernel reads
            for part in parts:
                part()
            ran += 1
            if self.brk is not None and _host_scalar(self.brk, 0):
                break
        self._cell[0] = self.cap - ran
        self._cell[1] = ran


def repeat_while(step, guard: SkipGuard | None = None, max_iters: int | None = None, *,
                 limit=None, brk=None) -> RepeatWhile:
    """Repeat ``step`` (a zero-arg callable, a :class:`Skippable`, or a
    tuple of such parts) while ``guard`` holds, at most ``max_iters``
    times (``None``: until ``guard`` fails): one device-side WHILE node under :meth:`GraphPipeline.build`,
    a host loop everywhere else. A refined run phase is refused: a phase
    count must be static.
    """
    from .diff.events import refuse_repeated_phase

    refuse_repeated_phase(step)
    return RepeatWhile(step, guard, max_iters, limit, brk)
