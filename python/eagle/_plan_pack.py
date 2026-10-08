# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The argument packer: one by-value block per partition, in ``arg_spec`` order."""

from __future__ import annotations

import ctypes

import numpy as np

from .roles import OUTPUT_ROLES, classify_arg

#: The input roles in the order :func:`_n_from_kw` reads them: the sample
#: count comes off the first bound one, never a ``lookup``/``wide_in`` length.
_SAMPLE_SHAPED_ROLES = ("per_sample", "vec_in", "mat_in", "terminated")
_BUFFER_SHAPED_ROLES = ("lookup", "wide_in")


def _trailing_extent(value) -> int:
    """``value``'s last axis, without dragging it through host memory:
    ``np.asarray`` on a cupy array raises rather than silently copying
    device to host, so the extent is read as metadata (``.shape``) first,
    falling back to ``np.asarray`` only for values with no shape of their
    own (a list, a scalar sequence)."""
    shape = getattr(value, "shape", None)
    if shape is None:
        shape = np.asarray(value).shape
    return int(shape[-1])


def _n_from_kw(arg_spec, kw) -> int:
    """The sample count implied by ``kw``: the trailing extent of the first
    bound sample-shaped input, then of a bound ``lookup``/``wide_in``
    buffer, then of an explicitly-bound output plane."""
    for group in (_SAMPLE_SHAPED_ROLES, _BUFFER_SHAPED_ROLES, tuple(OUTPUT_ROLES)):
        for role, name in arg_spec:
            if role in group and name in kw:
                return _trailing_extent(kw[name])
    raise ValueError(
        "eagle.plan: could not determine the sample count from the given "
        "arguments (no bound per_sample/vec_in/mat_in/terminated/lookup/"
        "wide_in/output array)"
    )


def _data_ptr(arr) -> int:
    """The raw address of ``arr``'s buffer (cupy device pointer, or numpy host one)."""
    ptr = getattr(getattr(arr, "data", None), "ptr", None)
    return int(ptr) if ptr is not None else int(arr.ctypes.data)


def _scalar_handle_box(addr: int, n: int, device_type: int):
    from .host_launch import ScalarHandle

    return ScalarHandle(int(addr), int(n), 1, int(device_type), 0)


def _gref_mirror_box(arr, n: int, device_type: int):
    """The 40-byte ``GRefMirror`` (``plugin/gref_abi.h``) for ``arr``, shaped
    ``(width, n)`` (or ``(n,)``). Extents/strides are read off the array, so
    a non-contiguous plane binds correctly; ``samples_`` is the run's
    sample count."""
    from .host_launch import GRefMirror

    strides = [s // arr.itemsize for s in arr.strides]
    comp = strides[0] if arr.ndim >= 2 else int(n)
    sample = strides[-1] if strides else 1
    return GRefMirror(
        _data_ptr(arr), int(n), int(comp), int(sample), int(device_type), 0
    )


def _uniform_box(value, dtype, *, integer: bool = False):
    """A ``uniform``'s by-value scalar at the artifact's declared width
    (``c_float``/``c_double``). An integer uniform (``dtype: "int"``,
    :func:`_integer_uniforms`) crosses as ``c_longlong`` instead — a double
    box would put its bit pattern in the integer argument."""
    if integer:
        return ctypes.c_longlong(int(value))
    if dtype == np.float32:
        return ctypes.c_float(float(value))
    return ctypes.c_double(float(value))


def _integer_uniforms(plugin) -> frozenset:
    """The ``uniform`` slots declared integer (v2 ``dtype: "int"``; v1 bare
    names are float, :mod:`eagle.sidecar`'s ``PARAM_DTYPES``). No
    ``params`` means float uniforms only."""
    out = set()
    for entry in getattr(plugin, "params", None) or ():
        if isinstance(entry, dict) and entry.get("dtype") == "int":
            out.add(str(entry["name"]))
    return frozenset(out)


def _mutable_shapes(plugin):
    """The vector-/matrix-shaped ``mutable`` names (:func:`eagle.roles.
    classify_arg`'s context: a vector packs as the 40-byte mirror,
    scalar/int as the 32-byte handle). From the plugin's own
    ``vec_mutables``/``mat_mutables``, else a ``Loaded*`` artifact's
    ``mutables`` block; neither means handle-shaped only."""
    vec = getattr(plugin, "vec_mutables", None)
    mat = getattr(plugin, "mat_mutables", None)
    if vec is None or mat is None:
        decl = getattr(plugin, "_mutables_decl", ()) or ()
        if vec is None:
            vec = [m.name for m in decl if getattr(m, "dtype", None) == "vector"]
        if mat is None:
            mat = [m.name for m in decl if getattr(m, "dtype", None) == "matrix"]
    return frozenset(vec), frozenset(mat)


def _pack_args(
    arg_spec,
    values,
    kw,
    n: int,
    device_type: int,
    *,
    vec_mutables=frozenset(),
    mat_mutables=frozenset(),
    dtype=None,
    int_uniforms=frozenset(),
):
    """Build the ctypes boxes + int addresses every partition's launch needs.

    Every parameter is a by-value ``ScalarHandle`` struct or scalar (never
    a raw pointer), so ``ctypes.addressof(box)`` is what both
    ``cuLaunchKernel``'s ``kernelParams`` and the host ``void* const*``
    convention need. ``values`` maps an array-role name to its bound plane
    (cupy or numpy); ``vec_mutables``/``mat_mutables``/``dtype`` are the
    artifact's own declarations.

    Which shape each role takes is :func:`eagle.roles.classify_arg`'s
    decision, shared with the v1 launch paths. Returns ``(boxes, addrs)``;
    ``boxes`` must outlive the launch call."""
    boxes = []
    addrs = []
    for role, name in arg_spec:
        box = _pack_one(
            role, name, values, kw, n, device_type,
            vec_mutables=vec_mutables, mat_mutables=mat_mutables, dtype=dtype,
            int_uniforms=int_uniforms,
        )
        boxes.append(box)
        addrs.append(ctypes.addressof(box))
    return boxes, addrs


def _pack_one(
    role,
    name,
    values,
    kw,
    n: int,
    device_type: int,
    *,
    vec_mutables=frozenset(),
    mat_mutables=frozenset(),
    dtype=None,
    int_uniforms=frozenset(),
):
    """The ctypes box for one ``arg_spec`` slot — the part of
    :func:`_pack_args` that depends on the slot, not the block.

    Factored out because :meth:`BoundPlan.rebind` re-packs a named subset
    of an already-packed block, so rebinding one plane costs one box, not a
    second pass over every role."""
    tag = classify_arg(
        role, name, vec_mutables=vec_mutables, mat_mutables=mat_mutables
    )
    if tag in ("GREF_VEC", "GREF_MAT"):
        # a matrix binds through the same 40-byte mirror as a vector: its
        # (R, C) shape is in-kernel flat indexing only
        return _gref_mirror_box(values[name], n, device_type)
    if tag in ("HANDLE", "WIDE_IN", "WIDE_OUT", "ACCUM_OUT"):
        arr = values[name]
        # a lookup/wide_in buffer's own length is not a sample count, and no
        # sidecar declares one (it's a runtime quantity), so the handle
        # carries the plane's own extent instead of packing n
        extent = _trailing_extent(arr) if role in _BUFFER_SHAPED_ROLES else n
        return _scalar_handle_box(_data_ptr(arr), extent, device_type)
    if tag == "UNIFORM":
        return _uniform_box(kw[name], dtype, integer=name in int_uniforms)
    if tag == "NSAMPLES":
        return ctypes.c_uint32(int(n))  # the ROLE is aether::idx_t-wide
    # defensive only: fires only if a new tag is added to classify_arg
    # without a matching branch here; never silently skip an arg
    raise ValueError(f"eagle.plan: unhandled ABI tag {tag!r} for role {role!r}")
