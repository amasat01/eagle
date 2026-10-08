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
    return int(arr.data.ptr) + int(idx) * arr.dtype.itemsize


class SkipGuard:
    """Region runs iff ``count_arr[count_idx] != (baseline_arr[baseline_idx]
    or 0)``. Arrays are uint32 device (cupy) or host (numpy) arrays;
    references are retained so pointer lifetime stays pinned to the
    pipeline."""

    __slots__ = ("count_arr", "count_idx", "baseline_arr", "baseline_idx")

    def __init__(self, count_arr, count_idx=0, baseline_arr=None, baseline_idx=0):
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

    def evaluate(self) -> bool:
        """Host read; eager/debug paths ONLY (syncs on cupy)."""
        count = _host_scalar(self.count_arr, self.count_idx)
        baseline = (
            _host_scalar(self.baseline_arr, self.baseline_idx)
            if self.baseline_arr is not None
            else 0
        )
        return count != baseline

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


def _array_module_of(arr):
    """``cupy`` for a cupy array, ``numpy`` otherwise, decided from the
    array's own type."""
    if type(arr).__module__.split(".")[0] == "cupy":
        import cupy

        return cupy
    import numpy

    return numpy


class RepeatWhile:
    """A zero-arg step repeated on the device while a :class:`SkipGuard`
    holds, at most ``max_iters`` times per replay -- the loop sibling of
    :class:`Skippable` (not a subclass: 0..N runs and 0..1 runs are
    different contracts).

    Under :meth:`GraphPipeline.build` the step becomes the body of one
    CUDA WHILE conditional node: a head kernel resets the
    ``[remaining, ran]`` cell and decides entry, the body runs, and a tail
    kernel decides continuation. ``launch(1)`` is one ``cudaGraphLaunch``
    regardless of iteration count; ``launch(n)`` replays the whole loop
    ``n`` times.

    ``max_iters`` is a plain Python ``int >= 1`` frozen at build as a
    kernel argument -- a new cap needs a new ``build()``. Called directly
    it is the host arm: ``while ran < max_iters and guard.evaluate():
    step()``, writing the same cell so :meth:`iterations` reads the same
    in every mode. The cell lives in the guard's array module (cupy or
    numpy), allocated at construction; :attr:`iteration_index` exposes
    its ``ran`` word to body kernels as the device-visible iteration
    index (0 on the first iteration).

    Composition: the body ``step`` may be a :class:`Skippable`, or a
    tuple of such parts run in order (e.g.
    ``(k_steps, skippable(compact, guard))``); a ``RepeatWhile`` may be an
    ``add_concurrent`` member, but nesting one directly inside another
    ``RepeatWhile`` or a ``Skippable`` is refused at
    :meth:`GraphPipeline.build`. One hidden inside an opaque body callable
    is not detected structurally: called mid-capture it runs its host
    arm, whose guard read syncs and fails the capture loudly.
    """

    __slots__ = ("step", "guard", "max_iters", "_cell")

    def __init__(self, step, guard: SkipGuard, max_iters: int):
        if isinstance(max_iters, bool) or not isinstance(max_iters, int):
            raise TypeError(
                f"repeat_while: max_iters must be a Python int, got "
                f"{type(max_iters).__name__}"
            )
        if max_iters < 1:
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
        self.guard = guard
        self.max_iters = max_iters
        xp = _array_module_of(guard.count_arr)
        self._cell = xp.zeros(2, dtype=xp.uint32)  # [remaining, ran]

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
        while ran < self.max_iters and self.guard.evaluate():
            for part in parts:
                part()
            ran += 1
        self._cell[0] = self.max_iters - ran
        self._cell[1] = ran


def repeat_while(step, guard: SkipGuard, max_iters: int) -> RepeatWhile:
    """Repeat ``step`` (a zero-arg callable, a :class:`Skippable`, or a
    tuple of such parts) while ``guard`` holds, at most ``max_iters``
    times: one device-side WHILE node under :meth:`GraphPipeline.build`,
    a host loop everywhere else.
    """
    return RepeatWhile(step, guard, max_iters)
