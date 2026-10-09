# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Active-set compaction: the live samples of a thinning batch, as an index map.

When samples finish at different times, a per-sample kernel that maps
thread ``t`` to sample ``t`` keeps every warp with one live lane running.
An :class:`ActiveSet` gathers the indices of samples a ``terminated`` mask
has NOT finished into a dense, ascending map, so a kernel built to read it
maps thread ``t`` to sample ``map[t]`` and exits past ``count``.
Nothing moves: the map holds SLOT indices into the caller's own planes,
which stay where they are, so the caller always sees sample ``i`` at slot
``i``. Two reserved planes (:data:`MAP_PLANE`/:data:`COUNT_PLANE`) are
what a kernel binds; :meth:`ActiveSet.planes` returns them for
:meth:`eagle.plan.Plan.bind`. Compaction scans every sample, so it runs
on a cadence (:func:`compaction_body`); termination must be monotone, and
a cleared mask needs :meth:`ActiveSet.reset` first.

The occasional physical reorder (opt-in, ``ActiveSet(mask, reorder=theta)``)
pays on large, irregularly thinning batches (from about 1e5 samples;
below that the map alone is as fast, since gathers through it lose
coalescing). When the live samples sit spread thin over their occupied
32-sample groups, a reorder moves every :meth:`ActiveSet.own`-ed plane to
the front, at most once per compaction; ``perm``/``inv`` restore the
caller's order at exit (:meth:`ActiveSet.restore`), a plane the caller
cannot move is declared :meth:`ActiveSet.indirect` instead, and export
doors refuse an owned plane while it is permuted. Device and host run
the same compaction and reorder bit-identically.
"""

from __future__ import annotations

import threading
import weakref

from .roles import PER_SAMPLE_ROLES

#: The reserved ``lookup`` plane holding the ascending live slot indices.
MAP_PLANE = "active_map"
#: The reserved ``lookup`` plane holding the live count (one element).
COUNT_PLANE = "active_count"
#: The largest batch the ``int32`` map addresses.
MAX_SAMPLES = 2**31 - 1
#: The smallest cadence :func:`compaction_body` accepts (it scans every
#: sample, so compacting every step or two costs more than it saves).
MIN_EVERY = 4
#: The default cadence.
DEFAULT_EVERY = 16

#: The default locality threshold of the reorder trigger.
DEFAULT_THETA = 0.5
#: The accepted range of the threshold: above it a reorder fires at almost
#: every compaction (buys nothing); below it the batch stays scattered too long.
THETA_RANGE = (0.25, 0.75)
#: The smallest span a reorder fires for (the trigger's floor): below it the
#: fixed launch cost of the moves exceeds what the restored coalescing saves.
MIN_REORDER_SPAN = 4096
#: Element sizes a plane may carry, in bytes.
ELEM_BYTES = (1, 2, 4, 8, 16)

_MASK_IS_DROP = 1  # EAGLE_BACKEND_COMPACT_MASK_IS_DROP

#: Every live ActiveSet that reorders (the owned-plane registry).
_REORDERING = weakref.WeakSet()
#: Guards :data:`_REORDERING` and the claim-then-register sequence that admits a
#: set into it: every scan takes its snapshot under this lock, and a set's claim
#: on its planes and its registration are one critical section, so two threads
#: cannot both be admitted for the same plane. Re-entrant (``_claim`` snapshots).
_REORDER_LOCK = threading.RLock()


def _reordering_snapshot() -> list:
    """The reordering sets alive right now, copied under :data:`_REORDER_LOCK`."""
    with _REORDER_LOCK:
        return list(_REORDERING)


__all__ = ["ActiveSet", "compaction_body", "MAP_PLANE", "COUNT_PLANE",
           "DEFAULT_EVERY", "MIN_EVERY", "DEFAULT_THETA", "THETA_RANGE"]


def _module_of(arr):
    mod = type(arr).__module__.split(".")[0]
    if mod == "cupy":
        import cupy

        return cupy
    if mod == "numpy":
        import numpy

        return numpy
    raise TypeError(
        f"eagle.ActiveSet: the mask must be a cupy (device) or numpy (host) "
        f"array, got {type(arr).__module__}.{type(arr).__name__}"
    )


def _address(arr):
    """``(first byte, byte count)`` of an array's storage, or ``None`` when
    it exposes none of the array interfaces."""
    data = getattr(arr, "data", None)
    ptr = getattr(data, "ptr", None)  # cupy
    if ptr is not None:
        return int(ptr), int(arr.nbytes)
    for attr in ("__cuda_array_interface__", "__array_interface__"):
        iface = getattr(arr, attr, None)
        if iface is not None:
            nbytes = int(getattr(arr, "nbytes", 0) or 0)
            if not nbytes:
                import math

                nbytes = math.prod(iface["shape"]) * int(iface["typestr"][2:])
            return int(iface["data"][0]), nbytes
    return None


def _overlaps(a, b) -> bool:
    return a[0] < b[0] + b[1] and b[0] < a[0] + a[1]


def _check_theta(theta) -> float:
    if theta is True:
        return DEFAULT_THETA
    if isinstance(theta, bool) or not isinstance(theta, (int, float)):
        raise TypeError(
            f"eagle.ActiveSet: reorder is the locality threshold theta (a float), "
            f"got {type(theta).__name__}")
    lo, hi = THETA_RANGE
    if not lo <= float(theta) <= hi:
        raise ValueError(
            f"eagle.ActiveSet: reorder={theta} is outside [{lo}, {hi}]: above "
            f"{hi} a reorder fires at almost every compaction on a short run, "
            f"below {lo} the batch stays scattered long after a reorder pays "
            f"(default {DEFAULT_THETA})")
    return float(theta)


def _capturing() -> bool:
    """Whether cupy's current stream is being captured into a graph."""
    import cupy

    stream = cupy.cuda.get_current_stream()
    probe = getattr(stream, "is_capturing", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:
        return True  # a stream that cannot say is treated as capturing


def refuse_if_permuted(obj, door: str) -> None:
    """Raise when ``obj`` is (part of) a plane an :class:`ActiveSet` owns
    while that set holds its samples permuted (``door`` names the export
    asked for); a no-op when no set reorders."""
    if not _REORDERING:
        return
    where = _address(obj)
    if where is None or where[1] == 0:
        return
    for aset in _reordering_snapshot():
        if any(_overlaps(where, r) for r in aset._owned_ranges()) and aset.permuted:
            raise ValueError(
                f"{door}: this array is a plane an eagle.ActiveSet owns, and its "
                "samples are physically reordered (sample i is at plane[inv[i]], "
                "not at i); call ActiveSet.restore() first, or export "
                "ActiveSet.in_sample_order(plane), a copy in sample order")


def check_bound_planes(index_map, bound) -> None:
    """The bind-time cross-check of a reordering set: every bound per-sample
    plane (``bound`` = ``(role, name, value)`` triples) is owned or
    declared indirect, and the set's planes are fixed from here on. A set
    that does not reorder (or a map no set owns) has nothing to check."""
    where = _address(index_map)
    aset = next((a for a in _reordering_snapshot()
                 if where is not None and _address(a.map)[0] == where[0]), None)
    if aset is None:
        return
    owned = aset._owned_ranges()
    indirect = [_address(p) for p in aset._indirect]
    for role, name, value in bound:
        if role not in PER_SAMPLE_ROLES:
            continue
        r = _address(value)
        if r is None:
            continue
        inside = lambda spans: any(s[0] <= r[0] and r[0] + r[1] <= s[0] + s[1] for s in spans)
        if not (inside(owned) or inside(indirect)):
            raise ValueError(
                f"eagle.plan.bind: {role} {name!r} is a per-sample plane of a run "
                f"whose ActiveSet reorders its samples, but the set neither owns it "
                f"nor declares it indirect: a reorder would move every other plane "
                f"and leave this one behind. Call active.own({name}) (moved with "
                f"the rest) or active.indirect({name}) (kept in sample order, read "
                f"through active.perm) before binding")
    aset._freeze()


class ActiveSet:
    """The live samples of ``mask`` (a ``terminated`` plane: set = finished),
    as an ascending index map plus a count -- and, with ``reorder=theta``,
    the occasional physical reorder of the planes it owns.
    ``mask`` is a one-dimensional, C-contiguous ``bool``/``uint8`` array
    (cupy or numpy), held by reference and read where it is every time.
    The map, count and device scratch are allocated once, before any
    capture, and live as long as this object.
    ``reorder`` is an explicit opt-in: the locality threshold ``theta``
    (in :data:`THETA_RANGE`, or ``True`` for :data:`DEFAULT_THETA`) that
    pays on large, irregularly thinning batches (from about 1e5 samples;
    below that the map alone is as fast). The set then owns the mask and
    every plane given to :meth:`own`. The set starts as the identity
    (every sample live, ``count == n``)."""

    __slots__ = ("mask", "n", "map", "count", "_xp", "_device", "_scratch",
                 "_done_at_last", "theta", "perm", "inv", "span", "fire",
                 "live32", "reorders", "_owned", "_indirect", "_frozen",
                 "_reorder_scratch", "_zero", "_reorders_at_restore", "_map_lazy",
                 "__weakref__")

    def __init__(self, mask, *, reorder=None, _lazy_scratch=False, _lazy_map=False):
        xp = _module_of(mask)
        if getattr(mask, "ndim", None) != 1:
            raise ValueError(
                f"eagle.ActiveSet: the mask is one-dimensional (one byte per "
                f"sample), got shape {tuple(getattr(mask, 'shape', ()))}"
            )
        if mask.dtype.itemsize != 1 or mask.dtype.kind not in "bu":
            raise ValueError(
                f"eagle.ActiveSet: the mask is bool or uint8, got {mask.dtype}"
            )
        if not mask.flags.c_contiguous:
            raise ValueError("eagle.ActiveSet: the mask must be C-contiguous")
        n = int(mask.shape[0])
        if n > MAX_SAMPLES:
            raise ValueError(
                f"eagle.ActiveSet: the int32 map addresses at most {MAX_SAMPLES} "
                f"samples, got {n}"
            )
        self.theta = None if reorder is None or reorder is False else _check_theta(reorder)
        self.mask = mask
        self.n = n
        self._xp = xp
        self._device = xp.__name__ == "cupy"
        # a device, non-reordering set may defer the (n * 4 byte) map: a
        # one-word stand-in is bound until a compacting loop needs the real
        # one (:meth:`materialize_map`); a run that never reads it pays nothing
        self._map_lazy = bool(_lazy_map) and self._device and self.theta is None
        self.map = xp.empty(1 if self._map_lazy else n, dtype=xp.int32)
        self.count = xp.zeros(1, dtype=xp.uint32)
        self._done_at_last = xp.zeros(1, dtype=xp.uint32)
        self._scratch = None
        self._owned = []
        self._indirect = []
        self._frozen = False
        self._reorder_scratch = None
        self._reorders_at_restore = 0
        self.perm = self.inv = self.span = self.fire = self.live32 = None
        self.reorders = self._zero = None
        if self.theta is not None:
            with _REORDER_LOCK:
                self._claim(mask)
                self.perm = xp.empty(n, dtype=xp.int32)
                self.inv = xp.empty(n, dtype=xp.int32)
                self.span = xp.zeros(1, dtype=xp.uint32)
                self.fire = xp.zeros(1, dtype=xp.uint32)
                self.live32 = xp.zeros(1, dtype=xp.uint32)
                self.reorders = xp.zeros(1, dtype=xp.uint32)
                self._zero = xp.zeros(1, dtype=xp.uint32)
                self._owned.append(mask)
                _REORDERING.add(self)
        if not _lazy_scratch:  # eagle.until_done defers it to its compacting loops
            self.prepare_compaction()
        self.reset()

    def __repr__(self) -> str:
        where = "device" if self._device else "host"
        extra = "" if self.theta is None else f", reorder={self.theta}"
        return f"ActiveSet(n={self.n}, {where}{extra})"

    def reset(self) -> None:
        """Back to the identity: every sample live, ``count == n`` (and, for
        a reordering set, ``perm``/``inv`` identity, ``span == n``, reorder
        counter zero). Not capture-legal: a plain array write, run before a
        launch, never recorded."""
        identity = _identity if self._device else _host_identity
        if not self._map_lazy:
            identity(self.map)
        self.count[0] = self.n
        self._done_at_last[0] = 0
        if self.theta is not None:
            identity(self.perm)
            identity(self.inv)
            self.span[0] = self.n
            self.fire[0] = 0
            self.live32[0] = 0
            self.reorders[0] = 0
            self._reorders_at_restore = 0

    def materialize_map(self) -> bool:
        """Allocate the deferred map (see ``_lazy_map``) and set it to the
        identity. ``True`` when it was deferred, so the caller must re-point
        every bound plan at :attr:`map`; ``False`` when it already is real."""
        if not self._map_lazy:
            return False
        self.map = self._xp.empty(self.n, dtype=self._xp.int32)
        self._map_lazy = False
        _identity(self.map)
        return True

    def compact(self) -> None:
        """Recompute the map and the count from the mask (and, for a
        reordering set, ``live32``/``fire``). Device: enqueued on the
        current cupy stream (capturable). Host: computed now."""
        if self._device:
            import cupy

            from . import _core

            trig = {}
            if self.theta is not None:
                trig = dict(live32=int(self.live32.data.ptr),
                            span=int(self.span.data.ptr),
                            fire=int(self.fire.data.ptr), theta=self.theta)
            if self._scratch is None:
                if cupy.cuda.get_current_stream().is_capturing():
                    raise RuntimeError(
                        "eagle: an ActiveSet's first device compaction cannot run "
                        "under graph capture; call prepare_compaction() before capturing")
                self.prepare_compaction()
            _core.compact_device(
                int(self.mask.data.ptr), int(self.map.data.ptr),
                int(self.count.data.ptr), int(self._scratch.data.ptr), self.n,
                int(cupy.cuda.get_current_stream().ptr), _MASK_IS_DROP,
                int(self.mask.device.id), **trig,
            )
            return
        import numpy as np

        live = np.flatnonzero(self.mask == 0).astype(np.int32, copy=False)
        self.map[: live.size] = live
        self.count[0] = live.size
        if self.theta is not None:
            groups = np.zeros(-(-self.n // 32), dtype=bool)
            groups[live // 32] = True
            live32 = int(np.count_nonzero(groups))
            self.live32[0] = live32
            # the device rule, to the bit: double(theta) * 32 * live32
            limit = float(np.float32(self.theta)) * 32.0 * float(live32)
            span = int(self.span[0])
            self.fire[0] = int(live.size < limit and live.size < span
                               and span >= MIN_REORDER_SPAN)

    __call__ = compact

    @property
    def live(self) -> int:
        """The live count as a Python int (a device sync for a device set)."""
        return _read_word(self.count)

    def prepare_compaction(self) -> None:
        """Allocate the device compaction scratch (once). The first
        :meth:`compact` does it when needed; anything that captures a
        compaction into a graph calls it first. A runner on a no-compaction
        fast path never pays for it."""
        if not self._device or self._scratch is not None:
            return
        from . import _core

        nbytes = _core.compact_device(0, 0, 0, 0, self.n, 0, _MASK_IS_DROP,
                                      int(self.mask.device.id))
        self._scratch = self._xp.empty(max(int(nbytes), 1), dtype=self._xp.uint8)

    def planes(self) -> dict:
        """The two reserved planes as :meth:`eagle.plan.Plan.bind` keywords."""
        return {MAP_PLANE: self.map, COUNT_PLANE: self.count.view(self._xp.int32)}

    def when_finished(self, finished):
        """A :class:`eagle.Skippable` compaction that runs only when
        ``finished`` moved since it last ran (the guard is evaluated on the
        device under a graph)."""
        from ._conditional import SkipGuard, Skippable

        self.prepare_compaction()  # before any capture of the compaction
        xp = self._xp
        if getattr(finished, "dtype", None) != xp.uint32 or finished.shape != (1,):
            raise ValueError(
                "eagle.ActiveSet.when_finished: `finished` is a one-element "
                f"uint32 array of the mask's array module, got {finished!r}"
            )
        snapshot = self._done_at_last

        def compact_and_remember():
            self.compact()
            xp.copyto(snapshot, finished)

        return Skippable(compact_and_remember, SkipGuard(finished, 0, snapshot, 0))

    # --- The physical reorder --- #
    def _need_reorder(self, what: str) -> None:
        if self.theta is None:
            raise ValueError(
                f"eagle.ActiveSet.{what}: this set does not reorder; construct it "
                f"as ActiveSet(mask, reorder={DEFAULT_THETA})")

    def _owned_ranges(self) -> list:
        return [_address(p) for p in self._owned]

    def _claim(self, arr) -> None:
        """Refuse ``arr`` if it overlaps a plane any reordering set owns."""
        where = _address(arr)
        for other in _reordering_snapshot():
            for r in other._owned_ranges():
                if where[1] and r[1] and _overlaps(where, r):
                    who = "this set" if other is self else "another ActiveSet"
                    raise ValueError(
                        "eagle.ActiveSet.own: the plane overlaps a plane "
                        f"{who} already owns; a plane is moved by one set, once")

    def _check_plane(self, arr, what: str) -> None:
        xp = self._xp
        if _module_of(arr) is not xp:
            raise TypeError(
                f"eagle.ActiveSet.{what}: a plane lives where the mask lives "
                f"({xp.__name__}), got {type(arr).__module__}.{type(arr).__name__}")
        shape = tuple(arr.shape)
        if not (shape == (self.n,) or (len(shape) == 2 and shape[1] == self.n)):
            raise ValueError(
                f"eagle.ActiveSet.{what}: a plane is ({self.n},) or (D, {self.n}) "
                f"(one element per sample, D planes for a 2-D array), got {shape}")
        if not arr.flags.c_contiguous:
            raise ValueError(f"eagle.ActiveSet.{what}: a plane must be C-contiguous")
        if arr.dtype.itemsize not in ELEM_BYTES:
            raise ValueError(
                f"eagle.ActiveSet.{what}: {arr.dtype.itemsize}-byte elements are "
                f"not movable; supported sizes are {ELEM_BYTES}")

    def own(self, *planes) -> "ActiveSet":
        """Hand the set per-sample planes to move with the mask at every
        reorder (a ``(D, n)`` array is D planes). Refused once fixed, for
        a plane another set owns, or one overlapping an owned plane."""
        self._need_reorder("own")
        if self._frozen:
            raise ValueError(
                "eagle.ActiveSet.own: the owned planes are fixed once the set is "
                "bound (or its reorder step built); own every plane before bind")
        for arr in planes:
            self._check_plane(arr, "own")
            with _REORDER_LOCK:
                self._claim(arr)
                self._owned.append(arr)
        return self

    def indirect(self, *planes) -> "ActiveSet":
        """Declare bound per-sample planes the caller keeps in sample order
        (its kernel reads sample ``perm[slot]`` from them). Returns the set."""
        self._need_reorder("indirect")
        if self._frozen:
            raise ValueError(
                "eagle.ActiveSet.indirect: the planes are fixed once the set is bound")
        for arr in planes:
            self._check_plane(arr, "indirect")
            self._indirect.append(arr)
        return self

    def _table(self) -> list:
        """The owned planes as ``(address, element bytes)`` pairs, one per row
        of a 2-D plane."""
        out = []
        for arr in self._owned:
            ptr, _ = _address(arr)
            size = arr.dtype.itemsize
            rows = 1 if arr.ndim == 1 else int(arr.shape[0])
            out.extend((ptr + r * self.n * size, size) for r in range(rows))
        return out

    def _freeze(self) -> None:
        """Fix the owned planes and allocate the reorder scratch, before
        any capture."""
        if self._frozen:
            return
        self._frozen = True
        if self._device:
            from . import _core

            nbytes = _core.reorder_device(self._table(), 0, 0, 0, 0, 0, 0, 0, 0,
                                          self.n, 0, _MASK_IS_DROP,
                                          int(self.mask.device.id))
            self._reorder_scratch = self._xp.empty(max(int(nbytes), 1),
                                                   dtype=self._xp.uint8)
            _count_fired()(self._zero, self._xp.zeros(1, dtype=self._xp.uint32))  # compile now, not under capture

    def reorder(self) -> None:
        """Run the reorder now if the last compaction fired it: owned
        planes and ``perm`` move so live samples come first, ``inv``/the
        map reset, ``span = count``, ``reorders += 1``. Device: enqueued
        on the current stream; host: now."""
        self._need_reorder("reorder")
        self._freeze()
        xp = self._xp
        if self._device:
            import cupy

            from . import _core

            # counted before the moves clear `fire`, and only when it is set
            _count_fired()(self.fire, self.reorders)
            _core.reorder_device(
                self._table(), int(self.mask.data.ptr), int(self.perm.data.ptr),
                int(self.inv.data.ptr), int(self.map.data.ptr),
                int(self.count.data.ptr), int(self.span.data.ptr),
                int(self.fire.data.ptr), int(self._reorder_scratch.data.ptr),
                self.n, int(cupy.cuda.get_current_stream().ptr), _MASK_IS_DROP,
                int(self.mask.device.id))
            return
        if int(self.fire[0]) == 0:
            return
        sp = int(self.span[0])
        keep = self.mask[:sp] == 0
        incl = xp.cumsum(keep, dtype=xp.int64)
        cnt = int(incl[-1]) if sp else 0
        slots = xp.arange(sp, dtype=xp.int64)
        dst = xp.where(keep, incl - 1, cnt + slots - incl)
        for arr in self._owned:
            arr[..., dst] = arr[..., :sp].copy()
        self.perm[dst] = self.perm[:sp].copy()
        self.inv[self.perm[:sp]] = slots.astype(xp.int32)
        self.map[:cnt] = xp.arange(cnt, dtype=xp.int32)
        self.count[0] = cnt
        self.span[0] = cnt
        self.fire[0] = 0
        self.reorders[0] += 1

    def reorder_if_degraded(self):
        """The reorder as a :class:`eagle.Skippable` guarded by ``fire``
        (one IF node under a graph, an eager check elsewhere)."""
        from ._conditional import SkipGuard, Skippable

        self._need_reorder("reorder_if_degraded")
        self._freeze()
        return Skippable(self.reorder, SkipGuard(self.fire, 0, self._zero, 0))

    @property
    def permuted(self) -> bool:
        """Whether a reorder ran since the last :meth:`restore` / :meth:`reset`
        (a device sync for a device set)."""
        if self.theta is None:
            return False
        return _read_word(self.reorders) != self._reorders_at_restore

    def restore(self) -> None:
        """Put every owned plane back in sample order; ``perm``/``inv``
        reset to identity, ``span = n``, the map recomputed. Eager only
        (refused under stream capture); a no-op for a set that does not
        reorder."""
        if self.theta is None:
            return
        self._freeze()
        xp = self._xp
        if self._device and _capturing():
            raise RuntimeError(
                "eagle.ActiveSet.restore: refused under stream capture; it is "
                "the end-of-run exit, run it eagerly after the graph launch")
        if not self.permuted:
            self.fire[0] = 0
            return  # nothing moved since the last restore / reset
        if self._device:
            import cupy

            from . import _core

            if _capturing():
                raise RuntimeError(
                    "eagle.ActiveSet.restore: refused under stream capture; it is "
                    "the end-of-run exit, run it eagerly after the graph launch")
            stream = cupy.cuda.get_current_stream()
            _core.restore_device(self._table(), int(self.perm.data.ptr),
                                 int(self.inv.data.ptr),
                                 int(self._reorder_scratch.data.ptr), self.n,
                                 int(stream.ptr), int(self.mask.device.id))
        else:
            for arr in self._owned:
                arr[..., self.perm] = arr.copy()
            self.perm[...] = xp.arange(self.n, dtype=xp.int32)
            self.inv[...] = xp.arange(self.n, dtype=xp.int32)
        self.span[0] = self.n
        self.fire[0] = 0
        self.compact()
        self.fire[0] = 0
        self._reorders_at_restore = _read_word(self.reorders)

    def in_sample_order(self, plane):
        """A copy of ``plane`` in sample order (``plane[..., inv]``);
        legal while permuted, capture-legal on the device."""
        self._need_reorder("in_sample_order")
        where = _address(plane)
        if not any(r == where for r in self._owned_ranges()):
            raise ValueError(
                "eagle.ActiveSet.in_sample_order: the plane is not one this set "
                "owns (an indirect plane is already in sample order)")
        return self._xp.take(plane, self.inv, axis=-1)


def _host_identity(arr) -> None:
    import numpy

    arr[...] = numpy.arange(arr.shape[0], dtype=arr.dtype)


_IDENTITY = None


def _identity(arr) -> None:
    """``arr[i] = i`` in place on the device: no ``arange`` temporary, which
    cupy's pool would keep as large as the batch after it is freed."""
    global _IDENTITY
    if _IDENTITY is None:
        import cupy

        _IDENTITY = cupy.ElementwiseKernel("", "raw int32 a", "a[i] = (int)i;",
                                           "eagle_active_set_identity")
    if arr.shape[0] > 0:
        _IDENTITY(arr, size=arr.shape[0])


_COUNT_FIRED = None


def _count_fired():
    """``reorders += fire != 0`` as one device kernel (cupy)."""
    global _COUNT_FIRED
    if _COUNT_FIRED is None:
        import cupy

        _COUNT_FIRED = cupy.ElementwiseKernel(
            "raw uint32 fire", "uint32 reorders",
            "reorders += fire[0] != 0u ? 1u : 0u;", "eagle_active_set_count_fired")
    return _COUNT_FIRED


def _read_word(arr) -> int:
    value = arr[0]
    get = getattr(value, "get", None)
    return int(get()) if get is not None else int(value)


def compaction_body(step, active: ActiveSet, *, every: int = DEFAULT_EVERY,
                    finished=None, reorder=None, steps_per_call: int = 1) -> tuple:
    """The :func:`eagle.repeat_while` body that runs ``step`` ``every`` times,
    then compacts ``active`` -- only when ``finished`` moved, if given,
    else unconditionally -- and, for a set built with ``reorder=theta``
    over at least :data:`MIN_REORDER_SPAN` samples, reorders when the
    compaction fired (``reorder=False``/``True`` overrides this).
    The loop guard is checked once per ``every`` steps, so it may run up
    to ``every - 1`` steps past the last sample finishing; size
    ``max_iters`` as ``ceil(max_steps / every)``. ``steps_per_call`` is
    the steps one ``step()`` call advances (a fused kernel's ``K``): the
    body then runs ``every * K`` steps between compactions, and a cadence
    below :data:`MIN_EVERY` steps is refused."""
    if isinstance(every, bool) or not isinstance(every, int):
        raise TypeError(f"compaction_body: every is an int, got {type(every).__name__}")
    if (isinstance(steps_per_call, bool) or not isinstance(steps_per_call, int)
            or steps_per_call < 1):
        raise ValueError(
            f"compaction_body: steps_per_call is an int >= 1, got {steps_per_call!r}")
    if every < 1 or every * steps_per_call < MIN_EVERY:
        raise ValueError(
            f"compaction_body: every={every} call(s) of {steps_per_call} step(s) is "
            f"below {MIN_EVERY} steps; the compaction scans every sample, so a "
            "shorter cadence costs more than it saves"
        )
    if not callable(step):
        raise TypeError("compaction_body: step is a zero-arg callable")
    active.prepare_compaction()  # before any capture of the compaction
    if reorder is None:
        # a batch smaller than the trigger's span floor can never fire: leave
        # the conditional node out rather than evaluate it every iteration
        reorder = active.theta is not None and active.n >= MIN_REORDER_SPAN
    elif reorder and active.theta is None:
        raise ValueError(
            "compaction_body: reorder=True needs a set built with "
            f"ActiveSet(mask, reorder={DEFAULT_THETA})")

    def steps():
        for _ in range(every):
            step()

    tail = active.compact if finished is None else active.when_finished(finished)
    if reorder:
        return (steps, tail, active.reorder_if_degraded())
    return (steps, tail)
