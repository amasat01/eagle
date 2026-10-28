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

``ActiveSet(mask, reorder=theta, classes=K, class_plane=cls)`` generalises the
two-way live/dead split to ``K`` classes: :meth:`ActiveSet.class_sort` is a
stable counting sort of the slots by ``(live, class)`` (``perm``/``inv`` stay
the only permutation planes, so the sample pairing is unchanged), the
``offsets`` plane (``K + 1`` words) holds the class prefix of the live samples,
and the trigger generalises ``live32`` to ``mixed32`` (32-sample groups holding
two or more classes). A sort inside the live prefix only (``within_live=True``,
the reverse's form) leaves ``count``/``span`` alone. Both are counted as
``class_sorts``, never as ``reorders``.
"""

from __future__ import annotations

import ctypes
import threading
import weakref

from . import _alloc, _core, _stream, arrays
from ._conditional import _host_scalar
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
#: The mixed-group threshold of the K-class trigger: it fires when
#: ``mixed32 >= CLASS_THETA * live32`` (32-slot groups holding two or more live
#: classes against groups holding a live sample), over at least
#: :data:`MIN_CLASS_SORT_SPAN` samples. A divergence/coalescing proxy: a mixed
#: warp serialises both branches. Its own name; accepted within :data:`THETA_RANGE`.
CLASS_THETA = DEFAULT_THETA
#: The stale trigger of a state-dependent selector: the K-class sort also fires
#: when more than this fraction of the live samples changed class since the last
#: sort (``stale_count > STALE_MAX * count``). Exact integer counts, no hysteresis.
STALE_MAX = 0.05
#: The most classes a set distinguishes (the class plane is one byte per sample).
MAX_CLASSES = 255
#: The smallest span a class sort is triggered for (as :data:`MIN_REORDER_SPAN`).
MIN_CLASS_SORT_SPAN = MIN_REORDER_SPAN
#: Element sizes a plane may carry, in bytes.
ELEM_BYTES = (1, 2, 4, 8, 16)

_MASK_IS_DROP = 1  # EAGLE_BACKEND_COMPACT_MASK_IS_DROP
_CLASSES_WITHIN_LIVE = 2  # EAGLE_BACKEND_CLASSES_WITHIN_LIVE

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
           "DEFAULT_EVERY", "MIN_EVERY", "DEFAULT_THETA", "THETA_RANGE",
           "MAX_CLASSES", "MIN_CLASS_SORT_SPAN", "CLASS_THETA", "STALE_MAX"]


def _view(arr):
    """``arr`` (an :class:`eagle.Array` or any array-protocol producer: DLPack, CUDA array interface, buffer
    protocol) as an :class:`eagle.Array` over the same memory."""
    if isinstance(arr, arrays.Array):
        return arr
    if any(hasattr(arr, f) for f in ("__dlpack__", "__cuda_array_interface__", "__array_interface__")):
        return arrays.asarray(arr)
    raise TypeError(
        f"eagle.ActiveSet: expected an array (device or host), got {type(arr).__module__}.{type(arr).__name__}")


def _host_bytes(a) -> bytes:
    """The bytes of a host one-byte-per-element array."""
    return bytes(memoryview(a))


def _float32(x: float) -> float:
    """``x`` rounded to float32, as the device rule reads the threshold."""
    return ctypes.c_float(x).value


def _address(arr):
    """``(first byte, byte count)`` of an array's storage, or ``None`` when
    it exposes none of the array interfaces."""
    if hasattr(arr, "ptr") and hasattr(arr, "nbytes"):     # an eagle.Array
        return int(arr.ptr), int(arr.nbytes)
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
    """Whether eagle's current stream is being captured into a graph."""
    try:
        return bool(_core.stream_is_capturing(_stream.current()))
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
            for other in _reordering_snapshot():
                if other is not aset and any(
                        s and s[0] <= r[0] and r[0] + r[1] <= s[0] + s[1]
                        for s in other._owned_ranges()):
                    raise ValueError(
                        f"eagle.plan.bind: {role} {name!r} is owned by a different "
                        "ActiveSet than the one this run compacts through; a "
                        "reverse registers its adjoint planes on its OWN set, never "
                        "on the forward's (or another reverse's)")
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
    (any device or host array), held by reference and read where it is every time.
    The map, count and device scratch are allocated once, before any
    capture, and live as long as this object.
    ``reorder`` is an explicit opt-in: the locality threshold ``theta``
    (in :data:`THETA_RANGE`, or ``True`` for :data:`DEFAULT_THETA`) that
    pays on large, irregularly thinning batches (from about 1e5 samples;
    below that the map alone is as fast). The set then owns the mask and
    every plane given to :meth:`own`. The set starts as the identity
    (every sample live, ``count == n``).

    ``group=G`` compacts whole groups of ``G`` lanes, for a kernel emitted for
    that group (its ``lane_space.group``): a group is live while any of its
    lanes is, the map holds group indices (``map[t / G] * G + t % G`` is the
    lane thread ``t`` runs) and ``count`` the lanes launched, live groups times
    ``G``, so a last partial group launches past ``n``. It does not combine
    with ``reorder=``. :func:`check_group` refuses, at bind, a kernel
    emitted for another group.

    ``classes=K`` (with ``class_plane``) makes it a K-class set, for a kernel
    whose samples fall into ``K`` branches that diverge a warp when
    interleaved. ``class_plane`` is a C-contiguous ``uint8`` array of ``n``
    entries, each sample's class in ``[0, K)``; the set owns it (it moves with
    every other plane) and ``K`` is at most :data:`MAX_CLASSES` (255). The set
    must reorder (``reorder=theta``). :meth:`class_sort` groups the live
    samples into one run per class, in one stable permutation carried by
    ``perm``/``inv``, so the sample pairing is the one of any reorder. The
    planes it adds:

    * ``offsets`` -- ``K + 1`` words, the exclusive prefix over the live
      samples: class ``c`` occupies slots ``[offsets[c], offsets[c + 1])`` and
      ``offsets[K] == count``. Valid after a sort, until the next reorder or
      :meth:`restore`.
    * ``class_sorts`` -- how many class sorts ran (a class sort is never counted
      as a ``reorders``).
    * ``mixed32`` -- 32-slot groups holding two or more live classes (written
      by every compaction).
    * ``class_fire`` -- 1 when the last compaction found a class sort worth
      running (see below); :meth:`reorder` and :meth:`reorder_if_degraded` run
      the sort exactly then.
    * ``stale_count`` -- an input: the samples whose class changed since the
      last sort, written by a kernel whose branch is chosen from evolving state
      (0 when classes never change).

    A compaction sets ``class_fire`` to ``(mixed32 > 0 and mixed32 >=
    class_theta * live32 and span >= MIN_CLASS_SORT_SPAN) or stale_count >
    STALE_MAX * count or fire``: the live samples are mixed across at least
    ``class_theta`` of the warps they occupy, or more than :data:`STALE_MAX` of
    them changed class, or the liveness reorder already fired. ``class_theta``
    defaults to :data:`CLASS_THETA` and is accepted in :data:`THETA_RANGE`.
    With ``epochs=True`` the set keeps a snapshot of ``perm`` after every sort
    (:attr:`epochs`). On the device the K-class set needs a backend providing the
    seam symbols ``compact_classes_device`` and ``reorder_classes_device``
    (:func:`eagle._core.backend_has`); without them construction is refused,
    naming the backend and the symbol."""

    __slots__ = ("mask", "n", "map", "count", "_xp", "_device", "_scratch",
                 "_done_at_last", "theta", "_m", "_cls", "_views", "_dev", "_fired", "_one", "_tmp", "_group_flag", "_group_zero", "_group_scale", "perm", "inv", "span", "fire",
                 "live32", "reorders", "_owned", "_indirect", "_frozen",
                 "_reorder_scratch", "_zero", "_reorders_at_restore", "_map_lazy",
                 "classes", "class_theta", "stale_count", "class_plane", "offsets", "class_sorts", "mixed32",
                 "class_fire", "_epochs", "_lock", "group", "n_groups", "_group_drop",
                 "_group_count", "__weakref__")

    def __init__(self, mask, *, reorder=None, classes=None, class_plane=None,
                 class_theta=None, epochs=False, group=1, _lazy_scratch=False, _lazy_map=False):
        m = _view(mask)
        if m.ndim != 1:
            raise ValueError(
                f"eagle.ActiveSet: the mask is one-dimensional (one byte per "
                f"sample), got shape {tuple(m.shape)}"
            )
        if m.dtype.itemsize != 1 or m.dtype.kind not in "bu":
            raise ValueError(
                f"eagle.ActiveSet: the mask is bool or uint8, got {m.dtype}"
            )
        if m.size and not m.is_contiguous:
            raise ValueError("eagle.ActiveSet: the mask must be C-contiguous")
        on_device = m.device.startswith("cuda")
        xp = m.device if on_device else "cpu"
        n = int(m.shape[0])
        if n > MAX_SAMPLES:
            raise ValueError(
                f"eagle.ActiveSet: the int32 map addresses at most {MAX_SAMPLES} "
                f"samples, got {n}"
            )
        self.theta = None if reorder is None or reorder is False else _check_theta(reorder)
        if isinstance(group, bool) or not isinstance(group, int) or group < 1:
            raise ValueError(f"eagle.ActiveSet: group is an int >= 1 (lanes per group), got {group!r}")
        if group > 1 and self.theta is not None:
            raise ValueError(
                f"eagle.ActiveSet: group={group} compacts whole groups of lanes, which a "
                "physical reorder (reorder=) would split; use one or the other")
        self.group = group
        self.n_groups = -(-n // group)
        self._group_drop = self._group_count = None
        self.classes = self.class_plane = self.offsets = None
        self.class_theta = self.stale_count = None
        self.class_sorts = self.mixed32 = self.class_fire = None
        self._epochs = None
        self._lock = threading.RLock()
        if classes is not None:
            if isinstance(classes, bool) or not isinstance(classes, int) \
                    or not 2 <= classes <= MAX_CLASSES:
                raise ValueError(
                    f"eagle.ActiveSet: classes is an int in [2, {MAX_CLASSES}], "
                    f"got {classes!r}")
            if self.theta is None:
                raise ValueError(
                    "eagle.ActiveSet: classes=K sorts the samples physically, so it "
                    f"needs a reordering set; construct it as ActiveSet(mask, "
                    f"reorder={DEFAULT_THETA}, classes={classes}, class_plane=...)")
            if class_plane is None:
                raise ValueError(
                    "eagle.ActiveSet: classes=K needs class_plane, the per-sample "
                    "uint8 plane holding each sample's class in [0, K)")
            cm = _view(class_plane)
            if cm.device.startswith("cuda") != on_device:
                raise TypeError(
                    f"eagle.ActiveSet: class_plane lives where the mask lives "
                    f"({'device' if on_device else 'host'})")
            if cm.dtype != arrays.uint8 or tuple(cm.shape) != (n,) or (cm.size and not cm.is_contiguous):
                raise ValueError(
                    f"eagle.ActiveSet: class_plane is a C-contiguous uint8 ({n},) "
                    f"array, got {cm.dtype} {tuple(cm.shape)}")
            if on_device:
                _require_class_symbols()
            self.classes = classes
            self.class_theta = CLASS_THETA if class_theta is None else _check_theta(class_theta)
        elif class_plane is not None:
            raise ValueError("eagle.ActiveSet: class_plane needs classes=K")
        self.mask = mask
        self._views = {id(mask): m}      # the views taken at bind: nothing imports a producer under a capture
        self._m = m if m.dtype == arrays.uint8 else (m.view(arrays.uint8) if n else arrays.empty((0,), arrays.uint8, xp))
        self._cls = cm if classes is not None else None
        self._dev = int(xp.split(":")[1]) if on_device else -1
        self._fired = self._one = self._tmp = None
        self._group_flag = self._group_zero = self._group_scale = None
        self.n = n
        self._xp = xp
        self._device = on_device
        # a device, non-reordering set may defer the (n * 4 byte) map: a
        # one-word stand-in is bound until a compacting loop needs the real
        # one (:meth:`materialize_map`); a run that never reads it pays nothing
        self._map_lazy = bool(_lazy_map) and self._device and self.theta is None
        self.map = _alloc.empty(xp, 1 if self._map_lazy else n, dtype=arrays.int32)
        self.count = _alloc.zeros(xp, 1, dtype=arrays.uint32)
        if group > 1:   # check_group finds the set by its count plane
            _GROUP_BY_COUNT[_address(self.count)[0]] = group
            weakref.finalize(self, _GROUP_BY_COUNT.pop, _address(self.count)[0], None)
        self._done_at_last = _alloc.zeros(xp, 1, dtype=arrays.uint32)
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
                self.perm = _alloc.empty(xp, n, dtype=arrays.int32)
                self.inv = _alloc.empty(xp, n, dtype=arrays.int32)
                self.span = _alloc.zeros(xp, 1, dtype=arrays.uint32)
                self.fire = _alloc.zeros(xp, 1, dtype=arrays.uint32)
                self.live32 = _alloc.zeros(xp, 1, dtype=arrays.uint32)
                self.reorders = _alloc.zeros(xp, 1, dtype=arrays.uint32)
                self._zero = _alloc.zeros(xp, 1, dtype=arrays.uint32)
                self._owned.append(mask)
                if self.classes is not None:
                    self._claim(class_plane)
                    self.class_plane = class_plane
                    self._owned.append(class_plane)
                    self._views[id(class_plane)] = cm
                    self.offsets = _alloc.zeros(xp, self.classes + 1, dtype=arrays.uint32)
                    self.class_sorts = _alloc.zeros(xp, 1, dtype=arrays.uint32)
                    self.mixed32 = _alloc.zeros(xp, 1, dtype=arrays.uint32)
                    self.class_fire = _alloc.zeros(xp, 1, dtype=arrays.uint32)
                    #: IN: samples whose class changed since the last sort (written
                    #: by the units of a forced segmented policy; 0 = none)
                    self.stale_count = _alloc.zeros(xp, 1, dtype=arrays.uint32)
                    if epochs:
                        self._epochs = []
                _REORDERING.add(self)
        if not _lazy_scratch:  # eagle.until_done defers it to its compacting loops
            self.prepare_compaction()
        self.reset()

    def __repr__(self) -> str:
        where = "device" if self._device else "host"
        extra = "" if self.theta is None else f", reorder={self.theta}"
        if self.group > 1:
            extra += f", group={self.group}"
        return f"ActiveSet(n={self.n}, {where}{extra})"

    def reset(self) -> None:
        """Back to the identity: every sample live, ``count == n`` (and, for
        a reordering set, ``perm``/``inv`` identity, ``span == n``, reorder
        counter zero). Run before a launch, never recorded; a reordering device
        set resets with one kernel on the current stream."""
        identity = _identity
        self._done_at_last[0] = 0
        if self.theta is not None and self._device:
            _core.reset_device(
                self.perm.ptr, self.inv.ptr, 0 if self._map_lazy else self.map.ptr,
                self.count.ptr, self.span.ptr, self.n, _stream.current(), self._dev)
        else:
            if not self._map_lazy:
                identity(self.map)
            self.count[0] = self.n_groups * self.group
            if self.theta is not None:
                identity(self.perm)
                identity(self.inv)
                self.span[0] = self.n
        if self.theta is not None:
            self.fire[0] = 0
            self.live32[0] = 0
            self.reorders[0] = 0
            self._reorders_at_restore = 0
            if self.classes is not None:
                self.offsets[...] = 0
                self.class_sorts[0] = 0
                self.mixed32[0] = 0
                self.class_fire[0] = 0
                if self._epochs is not None:
                    self._epochs.clear()

    def materialize_map(self) -> bool:
        """Allocate the deferred map (see ``_lazy_map``) and set it to the
        identity. ``True`` when it was deferred, so the caller must re-point
        every bound plan at :attr:`map`; ``False`` when it already is real."""
        if not self._map_lazy:
            return False
        self.map = _alloc.empty(self._xp, self.n, dtype=arrays.int32)
        self._map_lazy = False
        _identity(self.map)
        return True

    def compact(self) -> None:
        """Recompute the map and the count from the mask (and, for a
        reordering set, ``live32``/``fire``). Device: enqueued on the
        eagle's current stream (capturable). Host: computed now. With
        ``group > 1`` a group of lanes is live while any of its lanes is, the
        map holds group indices and ``count`` the lanes launched (live groups
        times ``group``)."""
        if self.group > 1:
            return self._compact_groups()
        if self._device:
            trig = {}
            if self.theta is not None:
                trig = dict(live32=self.live32.ptr, span=self.span.ptr,
                            fire=self.fire.ptr, theta=self.theta)
            if self.classes is not None:
                if self._scratch is None:
                    if _capturing():
                        raise RuntimeError(
                            "eagle: an ActiveSet's first device compaction cannot run "
                            "under graph capture; call prepare_compaction() before capturing")
                    self.prepare_compaction()
                _core.compact_classes_device(
                    self._m.ptr, self.map.ptr, self.count.ptr, self._scratch.ptr, self.n,
                    _stream.current(), _MASK_IS_DROP,
                    self._dev, class_plane=self._cls.ptr,
                    classes=self.classes, class_theta=self.class_theta,
                    mixed32=self.mixed32.ptr, stale_count=self.stale_count.ptr,
                    class_fire=self.class_fire.ptr, **trig)
                return
            if self._scratch is None:
                if _capturing():
                    raise RuntimeError(
                        "eagle: an ActiveSet's first device compaction cannot run "
                        "under graph capture; call prepare_compaction() before capturing")
                self.prepare_compaction()
            _core.compact_device(
                self._m.ptr, self.map.ptr, self.count.ptr, self._scratch.ptr, self.n,
                _stream.current(), _MASK_IS_DROP, self._dev, **trig,
            )
            return
        live = arrays.flatnonzero(arrays.equal(self._m, 0))
        k = int(live.shape[0])
        self.map[:k] = live
        self.count[0] = k
        if self.theta is not None:
            mask = _host_bytes(self._m)
            live32 = sum(1 for g in range(0, self.n, 32) if mask.find(0, g, g + 32) >= 0)
            self.live32[0] = live32
            # the device rule, to the bit: double(theta) * 32 * live32
            limit = _float32(self.theta) * 32.0 * float(live32)
            span = int(self.span[0])
            self.fire[0] = int(k < limit and k < span and span >= MIN_REORDER_SPAN)
            if self.classes is not None:
                self._host_mixed32(live.tolist(), live32, span)

    def _compact_groups(self) -> None:
        """:meth:`compact` for ``group > 1``: the group's drop byte is the AND of
        its lanes' mask bytes (lanes past ``n`` count as finished), then the
        same predicate, scan and ascending map run over ``n / group`` bytes."""
        g, ng, n = self.group, self.n_groups, self.n
        if not self._device:
            mask = _host_bytes(self._m)
            live = [i for i in range(ng) if mask.find(0, i * g, min(n, (i + 1) * g)) >= 0]     # any lane unfinished
            self.map[:len(live)] = arrays.asarray(live, arrays.int32, "cpu")
            self.count[0] = len(live) * g
            return
        if self._scratch is None:
            if _capturing():
                raise RuntimeError(
                    "eagle: an ActiveSet's first device compaction cannot run "
                    "under graph capture; call prepare_compaction() before capturing")
            self.prepare_compaction()
        # drop[i] = AND of the group's lanes' finished flags (a lane past n counts as finished): one multiply per
        # lane position over the groups that have it
        arrays.not_equal(self._m, self._group_zero, out=self._group_flag)
        arrays.copyto(self._group_drop, self._group_flag[0::g])
        for j in range(1, g):
            have = -(-(n - j) // g) if n > j else 0
            if have:
                head = self._group_drop[:have]
                arrays.multiply(head, self._group_flag[j::g], out=head)
        _core.compact_device(
            self._group_drop.ptr, self.map.ptr, self._group_count.ptr, self._scratch.ptr, ng,
            _stream.current(), _MASK_IS_DROP, self._dev)
        arrays.multiply(self._group_count, self._group_scale, out=self.count)

    __call__ = compact

    @property
    def live(self) -> int:
        """The live count as a Python int (a device sync for a device set)."""
        return _host_scalar(self.count, 0)

    def prepare_compaction(self) -> None:
        """Allocate the device compaction scratch (once). The first
        :meth:`compact` does it when needed; anything that captures a
        compaction into a graph calls it first. A runner on a no-compaction
        fast path never pays for it."""
        if not self._device or self._scratch is not None:
            return
        if self.group > 1:
            self._group_drop = _alloc.zeros(self._xp, self.n_groups, dtype=arrays.uint8)
            self._group_count = _alloc.zeros(self._xp, 1, dtype=arrays.uint32)
            self._group_flag = _alloc.empty(self._xp, self.n, dtype=arrays.uint8)
            self._group_zero = _alloc.zeros(self._xp, self.n, dtype=arrays.uint8)
            self._group_scale = _alloc.full(self._xp, 1, self.group, dtype=arrays.uint32)
        nbytes = _core.compact_device(0, 0, 0, 0, self.n_groups, 0, _MASK_IS_DROP, self._dev)
        self._scratch = _alloc.empty(self._xp, max(int(nbytes), 1), dtype=arrays.uint8)

    def planes(self) -> dict:
        """The two reserved planes as :meth:`eagle.plan.Plan.bind` keywords."""
        return {MAP_PLANE: self.map, COUNT_PLANE: self.count}

    def when_finished(self, finished):
        """A :class:`eagle.Skippable` compaction that runs only when
        ``finished`` moved since it last ran (the guard is evaluated on the
        device under a graph)."""
        from ._conditional import SkipGuard, Skippable

        self.prepare_compaction()  # before any capture of the compaction
        fin = _view(finished)
        if fin.dtype != arrays.uint32 or tuple(fin.shape) != (1,):
            raise ValueError(
                "eagle.ActiveSet.when_finished: `finished` is a one-element "
                f"uint32 array where the mask lives, got {finished!r}"
            )
        snapshot = self._done_at_last

        def compact_and_remember():
            self.compact()
            arrays.copyto(snapshot, fin)

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

    def _check_mapped_host(self, arr, what: str) -> None:
        """A host plane on a device set must be mapped pinned host memory
        (checked once, here: a pageable buffer would fault the kernel)."""
        from . import _core

        where = _address(arr)
        cls = _core.pointer_residency(where[0], self._dev)
        if cls != 3:
            raise TypeError(
                f"eagle.ActiveSet.{what}: a host plane on a device set must live in "
                "pinned (mapped) host memory, but this one is pageable or not "
                "device-addressable; allocate it with pinned memory (for example "
                "eagle.arrays.empty(shape, dtype, 'pinned')) or register the buffer (cudaHostRegister with "
                "the mapped flag) before owning it")

    def _check_plane(self, arr, what: str, mapped_host: bool = False) -> None:
        try:
            v = _view(arr)
        except (BufferError, ValueError) as err:      # a dtype eagle has no name for
            raise ValueError(f"eagle.ActiveSet.{what}: this plane's dtype is not movable ({err})") from err
        here = v.device.startswith("cuda")
        if mapped_host and self._device and not here:
            self._check_mapped_host(arr, what)
        elif here != self._device:
            raise TypeError(
                f"eagle.ActiveSet.{what}: a plane lives where the mask lives "
                f"({'device' if self._device else 'host'}), got {type(arr).__module__}.{type(arr).__name__}")
        shape = tuple(v.shape)
        if not (shape == (self.n,) or (len(shape) == 2 and shape[1] == self.n)):
            raise ValueError(
                f"eagle.ActiveSet.{what}: a plane is ({self.n},) or (D, {self.n}) "
                f"(one element per sample, D planes for a 2-D array), got {shape}")
        if v.size and not v.is_contiguous:
            raise ValueError(f"eagle.ActiveSet.{what}: a plane must be C-contiguous")
        if v.dtype.itemsize not in ELEM_BYTES:
            raise ValueError(
                f"eagle.ActiveSet.{what}: {v.dtype.itemsize}-byte elements are "
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
            self._check_plane(arr, "own", mapped_host=True)
            with _REORDER_LOCK:
                self._claim(arr)
                self._owned.append(arr)
                self._views[id(arr)] = _view(arr)
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
            self._views[id(arr)] = _view(arr)
        return self

    def _table(self) -> list:
        """The owned planes as ``(address, element bytes)`` pairs, one per row
        of a 2-D plane."""
        out = []
        for arr in self._owned:
            ptr, _ = _address(arr)
            v = self._views[id(arr)]
            size = v.dtype.itemsize
            rows = 1 if v.ndim == 1 else int(v.shape[0])
            out.extend((ptr + r * self.n * size, size) for r in range(rows))
        return out

    def bytes_per_sample(self) -> int:
        """Bytes one sample's owned planes occupy: what a step moves per live
        sample, read by the compaction cadence's memory term."""
        return sum(v.dtype.itemsize * (1 if v.ndim == 1 else int(v.shape[0]))
                   for v in (self._views[id(a)] for a in self._owned))

    def _freeze(self) -> None:
        """Fix the owned planes and allocate the reorder scratch, before
        any capture."""
        if self._frozen:
            return
        self._frozen = True
        if self._device:
            if self.classes is not None:
                nbytes = _core.reorder_classes_device(
                    self._table(), 0, 0, 0, 0, 0, 0, 0, 0, self.n, 0, _MASK_IS_DROP,
                    self._dev, 0, self.classes, 0, 0)
            else:
                nbytes = _core.reorder_device(self._table(), 0, 0, 0, 0, 0, 0, 0, 0,
                                              self.n, 0, _MASK_IS_DROP, self._dev)
            self._reorder_scratch = _alloc.empty(self._xp, max(int(nbytes), 1),
                                                   dtype=arrays.uint8)
            # the scratch of ``reorders += fire != 0`` (three kernels; nothing may allocate under a capture)
            self._fired = _alloc.empty(self._xp, 1, dtype=arrays.bool_)
            self._one = _alloc.full(self._xp, 1, 1, dtype=arrays.uint32)
            self._tmp = _alloc.empty(self._xp, 1, dtype=arrays.uint32)

    def reorder(self) -> None:
        """Run the reorder now if the last compaction fired it: owned
        planes and ``perm`` move so live samples come first, ``inv``/the
        map reset, ``span = count``, ``reorders += 1``. Device: enqueued
        on the current stream; host: now."""
        self._need_reorder("reorder")
        self._freeze()
        if self.classes is not None:  # one permutation: the liveness reorder IS the class sort
            self.class_sort(if_fired=True)
            return
        if self._device:
            # counted before the moves clear `fire`, and only when it is set
            arrays.not_equal(self.fire, self._zero, out=self._fired)
            arrays.where(self._fired, self._one, self._zero, out=self._tmp)
            arrays.add(self.reorders, self._tmp, out=self.reorders)
            _core.reorder_device(
                self._table(), self._m.ptr, self.perm.ptr, self.inv.ptr, self.map.ptr,
                self.count.ptr, self.span.ptr, self.fire.ptr, self._reorder_scratch.ptr,
                self.n, _stream.current(), _MASK_IS_DROP, self._dev)
            return
        if int(self.fire[0]) == 0:
            return
        sp = int(self.span[0])
        keep = arrays.equal(self._m[:sp], 0)
        incl = arrays.cumsum(keep.astype(arrays.int64))
        cnt = int(incl[sp - 1]) if sp else 0
        slots = arrays.arange(sp, dtype=arrays.int64)
        dst = arrays.where(keep, arrays.subtract(incl, 1), arrays.subtract(arrays.add(slots, cnt), incl)).astype(arrays.int32)
        for arr in (self._views[id(a)] for a in self._owned):
            for row in ([arr] if arr.ndim == 1 else [arr[r] for r in range(arr.shape[0])]):
                arrays.scatter(row[:sp], dst, row[:sp].copy())
        arrays.scatter(self.perm[:sp], dst, self.perm[:sp].copy())
        arrays.scatter(self.inv, self.perm[:sp], slots.astype(arrays.int32))
        arrays.copyto(self.map[:cnt], arrays.arange(cnt, dtype=arrays.int32))
        self.count[0] = cnt
        self.span[0] = cnt
        self.fire[0] = 0
        self.reorders[0] = int(self.reorders[0]) + 1

    def reorder_if_degraded(self):
        """The reorder as a :class:`eagle.Skippable` guarded by ``fire``
        (one IF node under a graph, an eager check elsewhere)."""
        from ._conditional import SkipGuard, Skippable

        self._need_reorder("reorder_if_degraded")
        self._freeze()
        gate = self.fire if self.classes is None else self.class_fire
        return Skippable(self.reorder, SkipGuard(gate, 0, self._zero, 0))

    # --- The K-class sort --- #
    def _need_classes(self, what: str) -> None:
        if self.classes is None:
            raise ValueError(
                f"eagle.ActiveSet.{what}: this set has no classes; construct it as "
                f"ActiveSet(mask, reorder={DEFAULT_THETA}, classes=K, class_plane=...)")

    def _host_mixed32(self, live: list, live32: int, span: int) -> None:
        """The host form of the class trigger: ``mixed32`` = 32-slot groups
        holding two or more classes among the live samples, and
        ``class_fire = (mixed32 > 0 and mixed32 >= class_theta * live32 and
        span >= MIN_CLASS_SORT_SPAN) or stale_count > STALE_MAX * count or
        fire`` (the last term: the liveness trigger of the same compaction, so a
        thinning batch is never lost because the classes happen to be grouped).
        The device rule, to the bit."""
        cls = _host_bytes(self._cls)
        seen: dict = {}
        for i in live:
            seen.setdefault(i >> 5, set()).add(cls[i])
        mixed = sum(1 for classes in seen.values() if len(classes) >= 2)
        self.mixed32[0] = mixed
        limit = _float32(self.class_theta) * float(live32)
        warp_mix = mixed > 0 and float(mixed) >= limit and span >= MIN_CLASS_SORT_SPAN
        drift = float(int(self.stale_count[0])) > STALE_MAX * float(len(live))
        self.class_fire[0] = int(warp_mix or drift or int(self.fire[0]) != 0)

    def _record_epoch(self) -> None:
        if self._epochs is not None:
            self._epochs.append(self.perm.copy())

    @property
    def epochs(self) -> tuple:
        """The epoch table: the ``perm`` snapshot written after every
        reorder / class sort of a set built with ``epochs=True`` (empty
        otherwise), oldest first. A snapshot is a private copy."""
        return () if self._epochs is None else tuple(self._epochs)

    def class_sort(self, *, within_live: bool = False, if_fired: bool = False) -> bool:
        """Stable counting sort of the slots by ``(live, class)``. Every owned
        plane, ``perm`` and ``inv`` move together; ``offsets[c]`` becomes the
        first live slot of class ``c`` (``offsets[K] == count``), the map the
        identity over ``[0, count)``, ``class_sorts += 1``. ``within_live=True``
        permutes the slots ``[0, count)`` among themselves by class only (the
        reverse's form: liveness is the prefix, ``count``/``span`` unchanged);
        otherwise ``[0, span)`` is sorted by ``(live, class)`` and
        ``span = count``. ``if_fired=True`` sorts only when the last compaction
        raised ``class_fire`` (``mixed32`` >= ``theta * live32`` over at least
        :data:`MIN_CLASS_SORT_SPAN` samples). Returns whether a sort ran.
        Host: now. The device face needs the backend's class symbols."""
        self._need_classes("class_sort")
        self._freeze()
        if self._device:
            _core.reorder_classes_device(
                self._table(), self._m.ptr, self.perm.ptr, self.inv.ptr, self.map.ptr,
                self.count.ptr, self.span.ptr, self.class_fire.ptr if if_fired else 0,
                self._reorder_scratch.ptr, self.n, _stream.current(),
                _MASK_IS_DROP | (_CLASSES_WITHIN_LIVE if within_live else 0),
                self._dev, self._cls.ptr, self.classes, self.offsets.ptr, self.class_sorts.ptr)
            if self._epochs is not None and not _capturing():
                self._record_epoch()
            return True  # enqueued (an if_fired sort is a no-op on the device when it did not fire)
        with self._lock:
            if if_fired and int(self.class_fire[0]) == 0:
                return False
            k = self.classes
            sp = int(self.count[0]) if within_live else int(self.span[0])
            top = max(_host_bytes(self._cls)[:sp], default=0)
            if top >= k:
                raise ValueError(
                    f"eagle.ActiveSet.class_sort: the class plane holds "
                    f"{top}, outside [0, {k})")
            _core.reorder_classes_host(
                self._table(), self._m.ptr, self.perm.ptr, self.inv.ptr, self.map.ptr,
                self.count.ptr, self.span.ptr, 0, self.n,
                _MASK_IS_DROP | (_CLASSES_WITHIN_LIVE if within_live else 0),
                self._cls.ptr, k, self.offsets.ptr, self.class_sorts.ptr)
            self.fire[0] = 0
            self.class_fire[0] = 0
            self._record_epoch()
        return True

    @property
    def permuted(self) -> bool:
        """Whether a reorder ran since the last :meth:`restore` / :meth:`reset`
        (a device sync for a device set)."""
        if self.theta is None:
            return False
        return self._moves() != self._reorders_at_restore

    def restore(self) -> None:
        """Put every owned plane back in sample order; ``perm``/``inv``
        reset to identity, ``span = n``, the map recomputed. Eager only
        (refused under stream capture); a no-op for a set that does not
        reorder."""
        if self.theta is None:
            return
        self._freeze()
        if self._device and _capturing():
            raise RuntimeError(
                "eagle.ActiveSet.restore: refused under stream capture; it is "
                "the end-of-run exit, run it eagerly after the graph launch")
        if not self.permuted:
            self.fire[0] = 0
            return  # nothing moved since the last restore / reset
        if self._device:
            _core.restore_device(self._table(), self.perm.ptr, self.inv.ptr,
                                 self._reorder_scratch.ptr, self.n, _stream.current(), self._dev)
        else:
            for arr in (self._views[id(a)] for a in self._owned):
                for row in ([arr] if arr.ndim == 1 else [arr[r] for r in range(arr.shape[0])]):
                    arrays.scatter(row, self.perm, row.copy())
            _identity(self.perm)
            _identity(self.inv)
        self.span[0] = self.n
        self.fire[0] = 0
        self.compact()
        self.fire[0] = 0
        if self.classes is not None:
            self.offsets[...] = 0
            self.class_fire[0] = 0
        self._reorders_at_restore = self._moves()

    def _moves(self) -> int:
        """Reorders plus class sorts so far: the count :attr:`permuted` compares."""
        n = _host_scalar(self.reorders, 0)
        return n + (0 if self.class_sorts is None else _host_scalar(self.class_sorts, 0))

    def in_sample_order(self, plane):
        """A copy of ``plane`` in sample order (``plane[..., inv]``);
        legal while permuted, capture-legal on the device."""
        self._need_reorder("in_sample_order")
        where = _address(plane)
        if not any(r == where for r in self._owned_ranges()):
            raise ValueError(
                "eagle.ActiveSet.in_sample_order: the plane is not one this set "
                "owns (an indirect plane is already in sample order)")
        src = self._views.get(id(plane))
        src = _view(plane) if src is None else src
        if src.ndim == 1:
            return arrays.take(src, self.inv)
        out = arrays.empty(src.shape, src.dtype, src.device)
        for r in range(src.shape[0]):
            arrays.take(src[r], self.inv, out=out[r])
        return out


def _require_class_symbols() -> None:
    """Refuse a device K-class set when the backend lacks a class symbol,
    naming the backend and the symbol."""
    from . import _core

    info = _core.cuda_backend()
    for sym in ("eagle_backend_compact_classes_device", "eagle_backend_reorder_classes_device"):
        if not _core.backend_has(sym):
            where = info.get("path") or "not loaded"
            raise RuntimeError(
                f"eagle.ActiveSet: classes=K on the device needs the seam symbol {sym}; "
                f"the {info.get('kind', 'cuda')} backend ({where}) does not provide it")


def _identity(arr) -> None:
    """``arr[i] = i`` in place (one kernel on the device)."""
    if arr.shape[0] > 0:
        arrays.iota(arr)


#: Compaction group of each live set built with ``group > 1``, by the address of
#: its count plane (all :func:`check_group` sees of the set).
_GROUP_BY_COUNT: dict = {}


def check_group(plugin, count_plane) -> None:
    """The bind-time group check. A kernel's ``lane_space.group`` is the group it
    was emitted for: it reads its neighbours inside a group, and decodes the map
    as ``map[t / group] * group + t % group``. Refused, naming both groups: a set
    compacting at a group that is no multiple of it (the neighbours are no longer
    adjacent), and any group other than it (the map would be read in the wrong
    form, silently wrong). A kernel with a thread box (``box``: ``T`` lanes
    ``stride`` apart per thread) is refused first unless the group holds whole
    boxes, ``T * stride`` lanes each."""
    from ._lane_bind import space_of

    where = _address(count_plane)
    group = _GROUP_BY_COUNT.get(where[0], 1) if where is not None else 1
    required = space_of(plugin).group
    name = getattr(plugin, "name", "?")
    box = getattr(plugin, "box", None)
    if box and group % (box["T"] * box["stride"]):
        span = box["T"] * box["stride"]
        raise ValueError(
            f"eagle.plan.bind: kernel {name!r} computes T={box['T']} lanes per thread "
            f"(a box of {span} lanes), but the ActiveSet compacts at group {group}, which "
            f"splits a box; build it with ActiveSet(mask, group={span}) (or a multiple), "
            "or build the kernel with HAWK_BOX=0")
    if group % required:
        raise ValueError(
            f"eagle.plan.bind: kernel {name!r} needs compaction groups of a multiple of "
            f"{required} lanes (its lane_space.group), but the ActiveSet compacts at "
            f"group {group}; build it with ActiveSet(mask, group={required})")
    if group != required:
        raise ValueError(
            f"eagle.plan.bind: kernel {name!r} reads its active map at group {required}, "
            f"but the ActiveSet compacts at group {group}; build it with "
            f"ActiveSet(mask, group={required})")


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
