# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The lane space a kernel declares, read at bind.

A plugin's ``lane_space`` (sidecar ``launch.lane_space``) names the logical
shape of the flat sample axis: row-major, the LAST axis fastest, only the
FIRST extent known at run time. A declared kernel takes that first extent by
its label (``channel=64``) or reads it off a lane-shaped plane, ``head +
lane shape`` (``(64, 8192)``, ``(W, 64, 8192)``). C-contiguous, such a plane
IS the flat lane-fastest plane the launch addresses, so it is flattened with a
view. A shape that contradicts the declared static extents is refused by name;
nothing broadcasts here (``eagle.deploy(k)`` of a hawk kernel routes a broadcast
shape to the variant of its pattern before it reaches this door).

The lane form has ``len(head) + len(axes)`` dimensions and the flat form
``len(head) + 1``, so with the declared head (a Scalar's ``()``, a Vector's
``(w,)``) the two never collide.
"""

from __future__ import annotations

from math import prod

from ._plan_pack import _BUFFER_SHAPED_ROLES, _SAMPLE_SHAPED_ROLES, _n_from_kw
from ._plan_planes import _plane_shape
from .roles import OUTPUT_ROLES, PER_SAMPLE_ROLES
from .sidecar import UNDECLARED_LANE_SPACE, LaneSpace, lane_space_from


def threads(plugin, n: int) -> int:
    """How many threads a launch over ``n`` lanes runs: ``n``, or, for a
    kernel emitted with a thread box (``box``: ``T`` lanes ``stride`` apart per
    thread), ``ceil(n / (T * stride)) * stride``; its ``base``/``count`` then
    count threads."""
    box = getattr(plugin, "box", None)
    if not box:
        return n
    span = box["T"] * box["stride"]
    return -(-n // span) * box["stride"]


def space_of(plugin) -> LaneSpace:
    """The plugin's lane space (its ``lane_space`` attribute, a
    :class:`~eagle.sidecar.LaneSpace` or the sidecar block); the undeclared
    one when it carries none."""
    raw = getattr(plugin, "lane_space", None)
    if raw is None:
        return UNDECLARED_LANE_SPACE
    return raw if isinstance(raw, LaneSpace) else lane_space_from(raw)


def declared_axes(plugin) -> tuple:
    """The plugin's ``((label, extent), ...)``, or ``()`` when it declares
    no lane space (one unlabeled run-time axis)."""
    axes = space_of(plugin).axes
    return () if axes == UNDECLARED_LANE_SPACE.axes else axes


def spelling(axes) -> str:
    """``Grid["channel", "sample":8192]``: the declaration as written."""
    def one(label, extent):
        if label is None:
            return "None" if extent is None else str(extent)
        return f'"{label}"' if extent is None else f'"{label}":{extent}'
    return "Grid[" + ", ".join(one(a, e) for a, e in axes) + "]"


def free_labels(plugin) -> set:
    """The labels of the sinks' runtime axes that no lane axis shares: each is its
    own extent, given as ``<label>=N`` or by the plane passed."""
    lane = {a for a, _e in declared_axes(plugin)}
    return {ax["label"] for spec in (getattr(plugin, "plane_axes", None) or {}).values()
            for ax in spec["axes"] if ax["extent"] is None and ax["label"] not in lane}


def _total(axes, runtime) -> int:
    """The lane count for first-axis extent ``runtime`` (unused when static)."""
    return prod(e if e is not None else runtime for _a, e in axes)


def _contiguous(value) -> bool:
    flags = getattr(value, "flags", None)
    return flags.c_contiguous if flags is not None else bool(value.is_contiguous())


def _first_label(axes) -> str:
    return axes[0][0] if axes[0][0] is not None else "the first axis"


def _refuse_n(kw, axes, names, door):
    """``n=`` is a flat kernel's; a declared lane space takes its extent by label."""
    if "n" in kw and "n" not in names:
        first = axes[0]
        how = ("drop n=" if first[1] is not None else
               f"pass {first[0]}=<extent>" if first[0] is not None else
               "bind a lane-shaped plane")
        raise ValueError(
            f"eagle.plan.{door}: n= is not an argument of a kernel that declares "
            f"{spelling(axes)}; {how}")


def _refuse_static(kw, axes, names, door):
    for label, extent in axes[1:]:
        if label is not None and label in kw and label not in names:
            raise ValueError(
                f"eagle.plan.{door}: axis {label!r} of {spelling(axes)} is static "
                f"({extent}); only the first axis' extent is the caller's, drop {label}=")


def _take_extent(kw: dict, axes, names, door):
    """Pop the first-axis extent from ``kw``: given by its label. ``n=`` and a
    static axis's label are refused. ``None`` when not given."""
    where = f"eagle.plan.{door}"
    _refuse_n(kw, axes, names, door)
    _refuse_static(kw, axes, names, door)
    label, extent = axes[0]
    if label is None or label in names or label not in kw:
        return None
    value = kw.pop(label)
    if extent is not None:
        raise ValueError(
            f"{where}: every axis of {spelling(axes)} is static ({label!r} is "
            f"{extent}); drop {label}=")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{where}: {label}= is the extent of the first lane axis, a positive int; "
            f"got {value!r}")
    return value


def _flatten(axes, arg_spec, kw: dict, widths, door):
    """Flatten every lane-shaped per-sample plane in ``kw`` (in place); the
    lane dims seen, or ``None`` when no plane was lane-shaped."""
    if len(axes) < 2:
        return None
    static = tuple(e for _a, e in axes)
    seen = None
    for role, name in arg_spec:
        if role not in PER_SAMPLE_ROLES or role == "mat_in" or name not in kw:
            continue
        value = kw[name]
        shape = getattr(value, "shape", None)
        head = (widths[name],) if widths.get(name, 1) > 1 else ()
        if shape is None or len(shape) != len(head) + len(axes):
            continue                                # a flat plane: the flat rules apply
        got = tuple(int(e) for e in shape)
        if got[:len(head)] != head:
            raise ValueError(
                f"eagle.plan.{door}: argument {name!r} has shape {got}, but its "
                f"per-sample head is {head}; a lane-shaped plane is head + the "
                f"kernel's {spelling(axes)} lane shape")
        dims = got[len(head):]
        first = axes[0][1] is None
        if (dims[1:] != static[1:]) if first else (dims != static):
            want = "(" + ", ".join(["n" if first else str(static[0])] + [str(e) for e in static[1:]]) + ")"
            hint = ("; a read-only input that varies along fewer axes binds through the hawk "
                    "kernel (eagle.deploy(k), or a built variant), not a prebuilt plugin"
                    if all(d in (1, e) for d, e in zip(dims[1:], static[1:])) else "")
            raise ValueError(
                f"eagle.plan.{door}: argument {name!r} has lane shape {dims}, which "
                f"contradicts the kernel's declared {spelling(axes)} = {want}: the "
                "static extents are part of the kernel, only the runtime first axis "
                f"is the caller's{hint}")
        if not _contiguous(value):
            raise ValueError(
                f"eagle.plan.{door}: lane-shaped argument {name!r} (shape {got}) must be "
                "C-contiguous to flatten without a copy; pass a contiguous array")
        if seen is not None and dims != seen:
            raise ValueError(
                f"eagle.plan.{door}: lane-shaped arguments disagree on the lane shape: "
                f"{seen} vs {dims} ({name!r})")
        seen = dims
        kw[name] = value.reshape(head + (prod(dims),))
    return seen


def bind_n(plugin, arg_spec, kw, widths, *, door="run"):
    """``(kw, n, dims)`` for a launch: ``kw`` with lane-shaped planes
    flattened and the first-axis extent consumed, ``n`` the lane count (``None``
    when the flat planes carry it: the caller reads it off them once their
    layouts are adapted), and ``dims`` the lane shape the caller used
    (``None`` for a flat launch).

    A declared kernel's extent comes from its label or a lane-shaped plane
    (all-static: from the declaration). Any kernel whose only bound arrays
    are Tables is refused: a Table's or buffer's length is never the lane
    count."""
    axes = declared_axes(plugin)
    out = dict(kw)
    extent = dims = None
    if axes:
        extent = _take_extent(out, axes, {nm for _r, nm in arg_spec}, door)
        dims = _flatten(axes, arg_spec, out, widths, door)
    n = None
    if dims is not None:
        n = prod(dims)
    if extent is not None:
        want = _total(axes, extent)
        if n is not None and n != want:
            raise ValueError(
                f"eagle.plan.{door}: {_first_label(axes)}={extent} gives {want} lanes with "
                f"{spelling(axes)}, but the bound lane-shaped planes hold {n}")
        n = want
    elif n is None and axes and axes[0][1] is not None:
        n = _total(axes, 1)                         # every axis static
    for label in free_labels(plugin) & out.keys() - {nm for _r, nm in arg_spec}:
        if isinstance(out[label], bool) or not isinstance(out[label], int) or out[label] < 1:
            raise ValueError(
                f"eagle.plan.{door}: {label}= is the extent of the runtime axis {label!r}, "
                f"a positive int; got {out[label]!r}")
    if n is None:
        bound = {role for role, name in arg_spec if name in out}
        if bound & set(_BUFFER_SHAPED_ROLES) and not bound & (
                set(_SAMPLE_SHAPED_ROLES) | OUTPUT_ROLES):
            how = (f"pass {_first_label(axes)}=<extent>" if axes and axes[0][0] is not None
                   else "bind a per-sample plane, or the output plane,")
            raise ValueError(
                f"eagle.plan.{door}: this launch binds no per-sample plane to read the "
                f"lane count from{' of ' + spelling(axes) if axes else ''}; {how} "
                "(a Table's or buffer's length is never the lane count)")
    return out, n, dims


def flatten(plugin, arg_spec, kw, widths, *, door):
    """``kw`` with each lane-shaped per-sample plane viewed flat, for the doors
    that take planes whose extent is already fixed (``rebind``, ``until_done``)."""
    axes = declared_axes(plugin)
    out = dict(kw)
    if axes:
        _flatten(axes, arg_spec, out, widths, door)
    return out


def n_of(arg_spec, kw) -> int:
    """The lane count of a flat launch: :func:`~eagle._plan_pack._n_from_kw`
    over the per-sample planes alone (a Table's length is never the count)."""
    return _n_from_kw([(r, nm) for r, nm in arg_spec if r not in _BUFFER_SHAPED_ROLES], kw)


def output_shape(plugin, name, width, n, extents=None, planes=None):
    """The shape of output plane ``name`` for ``n`` lanes and per-lane
    ``width``. A sink declared with named axes (sidecar ``plane_axes``; a
    reduction's output, a ``Mutable["bin":256]``) is ``(*payload, *axes)``,
    a runtime axis as long as the launch's first lane axis, or, when no lane
    axis shares its label, the ``extents[label]`` the caller gave. Any other
    output is the flat plane, which :func:`shape_result` views lane-shaped.

    This is the rule of ``hawk._lane_bind.output_shape`` (eagle does not
    import hawk): the kept axes are not ``n``, so a sink sized ``(n,)`` is
    the wrong shape, and overruns when ``payload * kept > n``. The adjoint of a
    lookup plane the caller bound (``planes``) is that plane's shape."""
    shape = _like_shape(plugin, name, planes)
    if shape is not None:
        return shape
    spec = (getattr(plugin, "plane_axes", None) or {}).get(name)
    if spec is None:
        return _plane_shape(width, n)
    axes = declared_axes(plugin)
    runtime = n // prod(e for _a, e in axes[1:]) if axes else n
    free = free_labels(plugin)

    def extent(a):
        if a["extent"] is not None:
            return a["extent"]
        if a["label"] not in free:
            return runtime
        if (extents or {}).get(a["label"]) is None:
            raise ValueError(f"{name!r} has a runtime axis {a['label']!r}; pass "
                             f"{a['label']}=<extent of {a['label']!r}>")
        return extents[a["label"]]

    return tuple(spec["payload"]) + tuple(extent(a) for a in spec["axes"])


def keyed_size(plugin, spec: dict, n: int) -> int:
    """The elements of a keyed plane eagle owns between the stages of a split
    kernel (sidecar ``keyed``): ``spec["width"]`` payload entries over the kept
    lane axes, a runtime one as long as the first lane axis (the rule of
    ``hawk._lane_bind.keyed_size``)."""
    axes = declared_axes(plugin)
    runtime = n // prod(e for _a, e in axes[1:]) if axes else n
    return spec["width"] * prod(runtime if e is None else e for _a, e in spec["axes"])


def gather_columns(plugin, name, n):
    """The column count the compact Matrix plane ``name`` must have at ``n`` lanes:
    the product of the extents of the axes it varies along (sidecar ``gather_mats``;
    a runtime axis is ``n`` over the static ones). ``None`` when no pattern is recorded."""
    declared = (getattr(plugin, "gather_mats", None)
                or (getattr(plugin, "_meta", None) or {}).get("gather_mats"))
    kept = declared.get(name) if isinstance(declared, dict) else None
    axes = declared_axes(plugin)
    if kept is None or not axes:
        return None
    static = prod(e for _a, e in axes if e is not None)
    return prod((n // static if e is None else e) for (_a, e), k in zip(axes, kept) if k)


def adjoint_like(plugin) -> dict:
    """``{adjoint plane: the lookup plane it is the gradient of}`` (sidecar
    ``adjoint_like``): such a plane is the lookup plane's size, not the lane
    count; a Table's length is the caller's."""
    declared = (getattr(plugin, "adjoint_like", None)
                or (getattr(plugin, "_meta", None) or {}).get("adjoint_like"))
    if declared:
        return declared
    source = getattr(plugin, "_hawk_source", None)
    return {} if source is None else source[0].walk.adjoint_like


def _like_shape(plugin, name, planes):
    """The shape of the lookup plane ``name`` is the adjoint of, when the caller
    bound it; else ``None``."""
    like = (planes or {}).get(adjoint_like(plugin).get(name))
    shape = None if like is None else getattr(like, "shape", None)
    return None if shape is None else tuple(int(e) for e in shape)


def extents_of(plugin, kw) -> dict:
    """``{label: N}`` for each runtime axis no lane axis shares: the caller's
    ``<label>=N``, else the length of the first sink plane passed with that axis."""
    free, out = free_labels(plugin), {}
    for name, spec in (getattr(plugin, "plane_axes", None) or {}).items():
        shape = getattr(kw.get(name), "shape", None)
        for i, a in enumerate(spec["axes"]):
            at = len(spec["payload"]) + i
            if a["label"] in free and a["label"] not in out:
                if a["label"] in kw:
                    out[a["label"]] = kw[a["label"]]
                elif shape is not None and at < len(shape):
                    out[a["label"]] = int(shape[at])
    return out


def check_sink(plugin, name, plane, n, extents=None, planes=None, *, door="run"):
    """``plane`` (a sink the caller passed) when it has exactly the shape
    :func:`output_shape` gives, else refused by name; the adjoint of a lookup
    plane only has to hold as many elements as that plane. Other planes pass."""
    like = _like_shape(plugin, name, planes)
    if like is not None:
        if prod(int(e) for e in plane.shape) < prod(like):
            raise ValueError(
                f"eagle.plan.{door}: {name!r} is the gradient of the lookup plane "
                f"{adjoint_like(plugin)[name]!r} (shape {like}), so it needs that many "
                f"elements; got shape {tuple(int(e) for e in plane.shape)}")
        return plane
    if name not in (getattr(plugin, "plane_axes", None) or {}):
        return plane
    got = tuple(int(e) for e in plane.shape)
    spec = plugin.plane_axes[name]
    want = output_shape(plugin, name, 1, n, extents)
    if got != want:
        axes = ", ".join(a["label"] if a["extent"] is None else f'{a["label"]}={a["extent"]}'
                         for a in spec["axes"])
        raise ValueError(
            f"eagle.plan.{door}: {name!r} is declared with the axes ({axes}), so at "
            f"n={n} it is bound {want} (payload first); got {got}")
    return plane


def shape_result(plugin, result, out_names, dims, n, widths):
    """Per-sample outputs in the plugin's lane shape (views): ``dims`` when a
    lane-shaped plane set it, else the declaration at ``n`` lanes. A flat
    launch of an undeclared kernel is returned as it is."""
    axes = declared_axes(plugin)
    if dims is None and axes:
        tail = prod(e for _a, e in axes[1:])
        if axes[0][1] is not None:
            dims = tuple(e for _a, e in axes)
        elif n % tail == 0:
            dims = (n // tail, *(e for _a, e in axes[1:]))
    if dims is None or len(dims) < 2:
        return result
    role_of = {name: role for role, name in plugin.arg_spec}

    def one(name, plane):
        if role_of.get(name) not in PER_SAMPLE_ROLES or plane is None:
            return plane
        head = (widths[name],) if widths.get(name, 1) > 1 else ()
        return plane.reshape(head + tuple(dims))

    if len(out_names) == 1:
        return one(out_names[0], result)
    return {k: one(k, v) for k, v in result.items()}


def take_extent(plugin, kw: dict):
    """``(kw without the extents, {label: extent})`` for a kernel CALL: the first
    axis' label and each free runtime label (a sink's own extent, see
    :func:`free_labels`) are lifted out before the call signature binds the
    planes (a plane of that name keeps it)."""
    axes = declared_axes(plugin)
    if not axes:
        return kw, {}
    names = {nm for _r, nm in plugin.arg_spec} | {
        nm for st in getattr(plugin, "_stages", None) or () for _r, nm in st.plugin.arg_spec}
    _refuse_n(kw, axes, names, "call")
    _refuse_static(kw, axes, names, "call")
    label = axes[0][0]
    wanted = ({label} if label is not None else set()) | free_labels(plugin)
    stray = [k for k in kw if k not in names and k not in wanted]
    if stray:
        raise ValueError(
            f"eagle.plan.call: unexpected keyword {stray[0]!r}; this kernel declares "
            f"{spelling(axes)}, so its run-time extent is passed as "
            + (f"{label}=<extent>" if label is not None else "a lane-shaped plane"))
    lifted = {k: kw[k] for k in wanted if k in kw and k not in names}
    return {k: v for k, v in kw.items() if k not in lifted}, lifted
