# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The bind side of a caller-built lane set (a kernel declared with ``lanes=IndexSet()``).

Launch lane ``t < count`` runs lane ``map[t]`` of the kernel's own space. The caller binds two planes beside the
kernel's own: ``active_map`` (``int32``, or ``int64`` for a 64-bit unit) and ``active_count`` (``uint32``, one
cell). :class:`IndexSet` builds them on the host; :func:`check_index_set` is the check a bind runs on them.

The map's word is the unit's, and the dense extent decides which unit that is: ``auto`` is 32 bits and becomes 64
only past ``2**31 - 1`` lanes (a pair grid of 65537 items). A 64-bit map over a 32-bit extent is refused, never
narrowed; a host ``int32`` map handed to a 64-bit unit is widened once, at bind. A device ``int32`` map cannot be.

Checks run once, at bind (a device map is read back to the host for them). A captured graph replays without them:
a caller who rewrites the map or the count in place between replays owns their contents, and may run
:func:`check_index_set` after the rewrite."""

from __future__ import annotations

from . import arrays

__all__ = ["INT32_MAX", "IndexSet", "check_index_set", "index_width"]

#: The largest dense extent a 32-bit map word can address.
INT32_MAX = 2**31 - 1


def index_width(extent: int, forced="auto") -> int:
    """The map word, 32 or 64 bits, for ``extent`` dense lanes: ``forced`` (32 or 64) as given, else 32, else 64
    past ``2**31 - 1`` lanes. Refuses a forced 32 that cannot address ``extent``."""
    if forced not in ("auto", 32, 64):
        raise ValueError(f"eagle.IndexSet: width is 'auto', 32 or 64, got {forced!r}")
    wide = int(extent) > INT32_MAX
    if wide and forced == 32:
        raise ValueError(f"eagle.IndexSet: width=32 cannot address {extent} lanes (more than {INT32_MAX}); use width=64 or 'auto'")
    return 64 if wide or forced == 64 else 32


def width_of(plugin):
    """The map word in bits of an index-set kernel (read from the sidecar's ``active_map`` dtype), or ``None`` when
    ``plugin`` does not declare a lane set. A thinning kernel (``Guard(active_set=True)``) is not one: its map is
    :class:`eagle.ActiveSet`'s, trusted."""
    declared = getattr(plugin, "index_set", None)
    if declared is None:
        source = getattr(plugin, "_hawk_source", None)
        lanes = getattr(getattr(source[0], "kind", None), "lanes", None) if source else None
        if lanes is None:
            return None
    dtype = (getattr(plugin, "arg_dtypes", None) or {}).get("active_map")
    return 64 if str(dtype) in ("int64", "i64") else 32


def _host(x):
    """``x`` as a host :class:`eagle.Array` (a device array is read back)."""
    a = x if isinstance(x, arrays.Array) else arrays.asarray(x)
    return a if a.device in ("cpu", "pinned") else a.get()


def check_index_set(map, count, extent, *, width=None, what="the index set") -> None:
    """Check the plane pair of an index set: ``count[0]`` entries of ``map``, each in ``[0, extent)``, none twice.

    ``map`` and ``count`` are arrays (an :class:`eagle.Array`, numpy, cupy, torch...; a device array is read
    back). ``width`` (32 or 64), when given, is the unit's map word: a map of the other word is refused. Raises
    ``ValueError`` naming the first offending position. Run it after rewriting a bound map in place: a replay
    does not."""
    m, c = _host(map), _host(count)
    bits = int(m.dtype.itemsize) * 8
    if width is not None and bits != width:
        raise ValueError(
            f"{what} is a {width}-bit unit, but active_map holds {bits}-bit words: "
            + ("cast the map to int32, or build the unit with IndexSet(width=64)" if bits == 64
               else "pass an int64 map (the unit reads 64-bit words)"))
    held = int(c.tolist()[0] if c.size else 0)
    if held < 0 or held > m.size:
        raise ValueError(f"{what}: active_count is {held}, but active_map holds {m.size} entries")
    if int(extent) > INT32_MAX and bits == 32:
        raise ValueError(f"{what}: {extent} lanes need a 64-bit map (int64); a 32-bit map cannot address them")
    seen = {}
    for position, lane in enumerate(m.tolist()[:held]):
        if lane < 0 or lane >= extent:
            raise ValueError(f"{what}: active_map[{position}] = {lane} is outside the dense lanes [0, {extent})")
        if lane in seen:
            raise ValueError(f"{what}: active_map[{position}] = {lane} repeats active_map[{seen[lane]}]; each lane may be listed once")
        seen[lane] = position


def check_bound(plugin, arg_spec, kw, n) -> None:
    """The bind-time checks of an index-set kernel, run by the plan doors on the ``active_map``/``active_count`` in
    ``kw`` over ``n`` dense lanes (``None``: read off the per-sample planes). Other kernels are left alone."""
    bits = width_of(plugin)
    if bits is None or "active_map" not in kw or "active_count" not in kw:
        return
    # the lane extent is the extent of the per-sample inputs: a later stage that binds scratch and sinks only (a team
    # stage over keys or chunks) has an extent of its own, and the map is not checked against it
    from ._plan_pack import _SAMPLE_SHAPED_ROLES, _n_from_kw

    reading = [(r, nm) for r, nm in arg_spec if r in _SAMPLE_SHAPED_ROLES and nm in kw]
    if not reading:
        return
    if n is None:
        n = _n_from_kw(reading, kw)
    name = getattr(plugin, "kernel", None) or getattr(plugin, "name", "?")
    check_index_set(kw["active_map"], kw["active_count"], n, width=_effective(kw["active_map"], bits),
                    what=f"kernel {name!r}: the index set")


def _effective(map, bits):
    """The unit's word for the comparison: a host ``int32`` map handed to a 64-bit unit is widened at bind (see
    :func:`widen`), so it passes as the unit's word; a device one is not widened and is refused."""
    given = int(_dtype_bytes(map)) * 8
    if bits == 64 and given == 32 and _is_host(map):
        return 32
    return bits


def _dtype_bytes(x):
    d = getattr(x, "dtype", None)
    return getattr(d, "itemsize", None) or arrays.asarray(x).dtype.itemsize


def _is_host(x):
    return (x if isinstance(x, arrays.Array) else arrays.asarray(x)).device in ("cpu", "pinned")


class IndexSet:
    """The host builder of an index set's planes: ``IndexSet.planes(selection, extent)``."""

    @staticmethod
    def planes(selection, extent, *, width="auto", capacity=None, device=None) -> dict:
        """``{"active_map": ..., "active_count": ...}`` for the lanes ``selection`` names out of ``extent``.

        ``selection`` is a boolean mask over the ``extent`` lanes, or the lane ids (any integer sequence or array,
        each lane once). The map holds the ids in the order given (a mask: ascending) and has room for
        ``capacity`` entries (default: as many as selected; the unused tail is zero); ``active_count`` is one
        ``uint32`` cell. ``width`` is ``"auto"`` (32, or 64 past ``2**31 - 1`` lanes), 32 or 64. The planes are
        checked, and live on ``device`` (default: the host)."""
        bits = index_width(extent, width)
        sel = _host(selection)
        ids = arrays.flatnonzero(sel).tolist() if sel.dtype == arrays.bool_ else sel.tolist()
        room = len(ids) if capacity is None else int(capacity)
        if len(ids) > room:
            raise ValueError(f"eagle.IndexSet: {len(ids)} lanes do not fit a capacity of {room}")
        word = arrays.int64 if bits == 64 else arrays.int32
        lane_map = arrays.zeros((room,), word)
        if ids:
            lane_map[: len(ids)] = arrays.asarray(ids, word)
        count = arrays.asarray([len(ids)], arrays.uint32)
        check_index_set(lane_map, count, extent, width=bits)
        if device not in (None, "cpu"):
            lane_map, count = arrays.asarray(lane_map, device=device), arrays.asarray(count, device=device)
        return {"active_map": lane_map, "active_count": count}


# ---- the prefix form (run-time sample count) -------------------------------------------------------------------------

def is_prefix(plugin) -> bool:
    """Whether ``plugin`` is a prefix-form kernel (``IndexSet(prefix=True)``), as its sidecar declares
    (``launch.index_set``, carried on the plugin as ``index_set``): it reads the one-element ``uint32`` plane
    ``active_count`` and no map, and launch lane ``t < count`` runs lane ``t``."""
    declared = getattr(plugin, "index_set", None)
    return bool(declared and declared.get("prefix"))


def check_prefix_shape(plugin) -> None:
    """Refuse, by name, a declared prefix kernel whose planes disagree: it must bind ``active_count`` and no
    ``active_map``."""
    names = {nm for _r, nm in (getattr(plugin, "arg_spec", None) or ())}
    if is_prefix(plugin) and ("active_count" not in names or "active_map" in names):
        name = getattr(plugin, "kernel", None) or getattr(plugin, "name", "?")
        raise ValueError(f"eagle.plan.bind: kernel {name!r} declares a prefix lane set, but its planes are "
                         f"{sorted(n for n in names if n.startswith('active_'))}: a prefix kernel binds "
                         "active_count and no active_map")


def prefix_planes(plugin, planes: dict, capacity) -> tuple:
    """``(planes, count plane, capacity)`` for ``bind(..., capacity=n_max)`` of a prefix kernel: ``planes`` plus an
    ``active_count`` eagle makes (``uint32``, one cell, set to ``capacity``) when the caller bound none. Without
    ``capacity`` nothing changes and the count stays the caller's. Refuses a ``capacity`` on any other kernel."""
    check_prefix_shape(plugin)
    if capacity is None:
        return planes, None, None
    name = getattr(plugin, "kernel", None) or getattr(plugin, "name", "?")
    if not is_prefix(plugin):
        raise ValueError(f"eagle.plan.bind: capacity= is for a kernel built with IndexSet(prefix=True) (a run-time "
                         f"sample count); kernel {name!r} is not one")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 0:
        raise ValueError(f"eagle.plan.bind: capacity= is the largest sample count, a non-negative int; got {capacity!r}")
    if "active_count" in planes:
        return planes, planes["active_count"], capacity
    from . import _alloc

    like = next((v for v in planes.values() if getattr(v, "shape", None) is not None), None)
    place = _alloc.place(like) if like is not None else "cpu"
    count = _alloc.full(place, (1,), capacity, arrays.uint32)
    return {**planes, "active_count": count}, count, capacity


def write_count(plane, m, capacity, stream=None) -> None:
    """Set the count plane to ``m`` (``0 <= m <= capacity``): one fill on ``stream`` (a raw handle), ordered after
    the work already issued there, with no host wait, so it is legal between replays of a captured graph. Writing the
    plane directly, without this, is the caller's contract: nothing then checks ``m``."""
    if isinstance(m, bool) or not isinstance(m, int) or m < 0:
        raise ValueError(f"eagle.set_count: the count is a non-negative int; got {m!r}")
    if m > capacity:
        raise ValueError(f"eagle.set_count: count {m} is above this bind's capacity {capacity}; a larger batch is a "
                         "rebind (bind again with a larger capacity=)")
    a = plane if isinstance(plane, arrays.Array) else arrays.asarray(plane)
    if a.device in ("cpu", "pinned"):
        a.fill(m)
    else:
        a.fill(m, stream)
