# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Planes: residency, single-sample and layout adaptation, checks and write-back."""

from __future__ import annotations


import numpy as np

from . import _layout
from .abi import DEVICE_CPU, DEVICE_CUDA
from .dtypes import np_dtype
from .roles import INPUT_ROLES, OUTPUT_ROLES, PER_SAMPLE_ROLES
from ._plan_pack import _data_ptr, _trailing_extent


#: The reserved ``lookup`` word an automatic kernel reads its steps per
#: launch from; every launch but :func:`eagle.until_done`'s binds it to 1.
FUSED_STEPS_SLOT = ("lookup", "fused_steps")


def _output_names(arg_spec) -> tuple:
    """Every declared output plane's name, in ``arg_spec`` order."""
    names = tuple(name for role, name in arg_spec if role in OUTPUT_ROLES)
    if not names:
        raise ValueError(
            "eagle.plan: this plugin's arg_spec declares no out/mutable/"
            "wide_out/accum_out output role -- .run() has nothing to return"
        )
    return names


def _check_writable(role, name, value) -> None:
    """Refuse a caller-supplied output plane :meth:`Plan.run` cannot write
    into, naming the argument: a numpy view with ``flags.writeable`` false,
    or a torch tensor with ``requires_grad=True``."""
    writeable = getattr(getattr(value, "flags", None), "writeable", None)
    if writeable is False:
        raise ValueError(
            f"eagle.plan.run: {role} {name!r} was supplied read-only "
            "(array.flags.writeable is False); Plan.run writes the launch's "
            "result into every supplied output, so pass a writable array, "
            "or omit it to get a freshly allocated one back"
        )
    if getattr(value, "requires_grad", False):
        raise ValueError(
            f"eagle.plan.run: {role} {name!r} is a torch tensor with "
            "requires_grad=True; Plan.run writes the launch's result into it "
            "in place, which autograd refuses for a tracked tensor -- pass "
            "requires_grad=False, a .detach() tensor, or omit it"
        )


def _check_run_writable(arg_spec, out_names, kw) -> None:
    """:meth:`Plan.run`'s read-only pre-flight, over every output the
    caller supplied."""
    role_of = {name: role for role, name in arg_spec}
    for name in out_names:
        if name in kw:
            if _layout.is_number(kw[name]):
                _check_number("run", role_of[name], name, kw[name])
            _check_writable(role_of[name], name, kw[name])


def _write_into(dst, src) -> None:
    """Copy ``src`` into ``dst`` (the caller's output object) in dst's own
    framework: torch's in-place ``copy_``, ``cupy.copyto``, or plain numpy.
    Safe when ``dst``/``src`` already alias the same buffer (a no-op)."""
    src_on_device = not isinstance(src, np.ndarray)
    copy_ = getattr(dst, "copy_", None)
    if copy_ is not None:  # torch
        import torch

        copy_(torch.as_tensor(src, device=dst.device) if src_on_device
              else torch.from_numpy(src))
        return
    if type(dst).__module__.partition(".")[0] == "cupy":
        import cupy as cp

        cp.copyto(dst, cp.asarray(src))
    else:
        np.copyto(dst, src.get() if src_on_device else src)


def _write_back_outputs(out_names, supplied, adapted, result):
    """Write ``result`` into every output plane the caller supplied, in
    its own layout and framework, and still return it. An adapted name
    (:func:`_adapt_planes`) is skipped — :func:`_restore_layouts` already
    handled it; one not supplied is untouched, returned only."""
    if not supplied:
        return result
    names = out_names
    planes = dict(result) if len(names) > 1 else {names[0]: result}
    for name in names:
        dst = supplied.get(name)
        if dst is None or name in adapted:
            continue
        _write_into(dst, planes[name])
    return result


def _returned(arrays, names, to_host=None):
    """The result :meth:`Plan.run` hands back: the plane itself for a
    single-output plugin (never a 1-tuple), else a ``dict`` of every output
    plane keyed by its NAME -- never a positional tuple: ``arg_spec``'s own
    order is ROLE-grouped then NAME-sorted (:mod:`hawk.ir.walk`), not the
    kernel's authored parameter order, so a positional result would
    silently swap two outputs declared in a different order. Matches what
    :class:`~eagle._simulate.SimResult`/:func:`eagle.loaded.LoadedPure`
    already return."""
    planes = {
        name: (arrays[name] if to_host is None else to_host(arrays[name]))
        for name in names
    }
    return next(iter(planes.values())) if len(planes) == 1 else planes


def _plane_dtype(plugin):
    """The numpy dtype of every Real-typed plane this artifact reads and
    writes; ``float64`` when it declares no ``scalar_type``."""
    return np.dtype(np_dtype(getattr(plugin, "scalar_type", None) or "float64"))


def _declared_widths(plugin) -> dict:
    """``{name: width}`` for the output planes this packer allocates (the
    plugin's optional ``arg_widths``); undeclared means rank-1."""
    declared = getattr(plugin, "arg_widths", None) or {}
    return {str(k): int(v) for k, v in declared.items()}


def _plane_shape(width: int, n: int):
    return (n,) if int(width) <= 1 else (int(width), n)


def _bound(kw, name: str, role: str):
    """The caller's value for an INPUT role, named in the error when absent."""
    try:
        return kw[name]
    except KeyError:
        raise ValueError(
            f"eagle.plan: this plugin's arg_spec declares a {role!r} input "
            f"{name!r}, but .run() was not given it"
        ) from None


def _host_plane(value, dtype):
    """A caller-supplied input plane as a contiguous host array of the
    artifact's Real dtype; an integer/bool array (an index table, a
    ``terminated`` mask) passes through at its own dtype instead."""
    arr = np.asarray(value)
    if arr.dtype.kind not in "iub":
        arr = arr.astype(dtype, copy=False)
    return np.ascontiguousarray(arr)


def residency(planes, *, door: str = "eagle.plan.auto", what: str = "kernel") -> str:
    """Where ``planes`` says the work runs: ``"device"`` when any value is
    device-resident, else ``"host"``; both sides refused, naming one of
    each (``door``/``what`` name the caller)."""
    on_device = on_host = None
    for name, value in planes.items():
        if device_resident(value):
            on_device = on_device or name
        else:
            on_host = on_host or name
    if on_device is not None and on_host is not None:
        raise ValueError(
            f"{door}: {on_device!r} is on the device but {on_host!r} is on the "
            f"host; the data decides where the {what} runs, and it runs in one "
            "place: move one side with cp.asarray / .get()")
    return "device" if on_device is not None else "host"


def device_resident(value) -> bool:
    """Whether ``value`` is already in device memory (a cupy array, or
    anything exporting ``__cuda_array_interface__``); a CPU or
    grad-tracked tensor raises there, so both answer ``False``."""
    try:
        return getattr(value, "__cuda_array_interface__", None) is not None
    except Exception:  # noqa: BLE001 -- torch raises RuntimeError for a grad-tracked tensor
        return False


def _device_inputs(arg_spec, kw) -> bool:
    """Whether :meth:`Plan.run` was handed device-resident input planes
    (the signal that keeps a device plan's outputs on the device)."""
    return any(
        role in INPUT_ROLES and name in kw and device_resident(kw[name])
        for role, name in arg_spec
        if (role, name) != FUSED_STEPS_SLOT  # eagle's own step word, not the caller's
    )


def _device_plane(value, dtype):
    """:func:`_host_plane`'s device twin: a device-resident plane is
    wrapped where it lies and cast only on the device when needed; anything
    else goes through :func:`_host_plane` and is uploaded once."""
    import cupy as cp

    if not device_resident(value):
        return cp.asarray(_host_plane(value, dtype))
    arr = cp.asarray(value)
    if arr.dtype.kind not in "iub":
        arr = arr.astype(dtype, copy=False)
    return cp.ascontiguousarray(arr)


def _adapt_planes(plugin, arg_spec, kw, *, door: str = "run",
                  single_only: bool = False):
    """Adapt every sample-major or single-sample per-sample plane in ``kw``
    (:mod:`eagle._layout`). Returns ``(kw, adapted)``: ``kw`` with each
    adapted plane replaced by its component-major form, and
    ``{name: Adapted}``. Only a plane whose width the artifact declares is
    adapted as sample-major; one sample adapts to its ``(w, 1)``/``(1,)``
    view, and mixing one sample with a batch is refused. ``door`` is
    ``"run"`` (materialises a number) or ``"bind"`` (refuses one).
    ``single_only`` adapts one-sample values alone, leaving batches as
    given."""
    widths = _declared_widths(plugin)
    adapted = {}
    out = kw
    single = None
    for role, name in arg_spec:
        if role not in PER_SAMPLE_ROLES or name not in kw:
            continue
        value = kw[name]
        width = widths.get(name, 1)
        shape = getattr(value, "shape", None)
        if shape is None or isinstance(value, np.generic):
            if not _layout.is_number(value):
                continue                    # a list or sequence: bound as given
            _check_number(door, role, name, value)
            a = _layout.adapt(name, value, (), writes=False, single=True,
                              dtype=_number_dtype(plugin, name, value))
        elif width <= 1 and shape != ():
            continue
        elif single_only and not _layout.is_single(shape, (width,)):
            continue
        else:
            a = _layout.adapt(name, value, (width,), writes=role in OUTPUT_ROLES,
                              single=True)
        if a is None:
            continue
        if a.layout == "single" and single is None:
            single = (name, tuple(getattr(value, "shape", ())))
        if out is kw:
            out = dict(kw)
        out[name] = a.native
        adapted[name] = a
    if single is not None:
        _check_one_call(arg_spec, out, adapted, single)
    return out, adapted


def _check_number(door, role, name, value) -> None:
    """A number is one sample's value: legal only as an input to
    :meth:`Plan.run`, never as an output or a bound value."""
    kind = type(value).__name__
    if role in OUTPUT_ROLES:
        raise ValueError(
            f"eagle.plan.{door}: {role} {name!r} was given a Python {kind}, which "
            "cannot receive a write; pass a writable 0-d array "
            + ("(np.zeros(())) or leave it out to have it returned" if door == "run"
               else f"(np.array({name}))"))
    if door != "run":
        raise ValueError(
            f"eagle.plan.{door}: {role} {name!r} was given a Python {kind}; this "
            f"door allocates nothing, so bind one sample as a 0-d array "
            f"(np.array({name}))")


def _number_dtype(plugin, name, value):
    """The dtype a number on an input plane is materialised in: the plane's
    declared wire dtype (an ``Index`` plane is ``int64``, a mask ``bool``),
    else the artifact's Real dtype."""
    declared = (getattr(plugin, "arg_dtypes", None) or {}).get(name)
    if declared:
        return np.dtype(declared)
    if isinstance(value, (bool, np.bool_)):
        return np.dtype(bool)
    return _plane_dtype(plugin)


def _check_one_call(arg_spec, planes, adapted, single) -> None:
    """A call is one sample or one batch, never both (no broadcast): refuse a
    per-sample plane that is a batch beside one that is one sample."""
    for role, name in arg_spec:
        if role not in PER_SAMPLE_ROLES or name not in planes:
            continue
        a = adapted.get(name)
        if a is not None and a.layout == "single":
            continue
        caller = planes[name] if a is None else a.caller
        shape = tuple(int(e) for e in getattr(caller, "shape", np.shape(caller)))
        n = _trailing_extent(planes[name]) if shape else 0
        s_name, s_shape = single
        tile = (f"np.full({n}, {s_name})" if s_shape == ()
                else f"np.repeat({s_name}.reshape(-1, 1), {n}, axis=1)")
        raise ValueError(
            f"eagle.plan: {s_name!r} is one sample (shape {s_shape}) but "
            f"{name!r} is a batch of {n} (shape {shape}); a call is one sample "
            "or one batch, never both. A value shared by every sample is a Param "
            f"(scalar) or a Table (vector); or tile it once: {tile}")


def _is_single_call(adapted) -> bool:
    return any(a.layout == "single" for a in adapted.values())


def _result_head(plugin, name, width: int) -> tuple:
    """The per-sample head a one-sample result takes: ``()``, ``(w,)``, or a
    matrix output's declared ``(R, C)``."""
    if width <= 1:
        return ()
    shape = (getattr(plugin, "mat_shapes", None) or {}).get(name)
    if shape is None:
        for m in getattr(plugin, "_mutables_decl", ()) or ():
            if getattr(m, "name", None) == name and getattr(m, "shape", None):
                shape = m.shape
    return tuple(int(e) for e in shape) if shape else (int(width),)


def _single_returned(plugin, arg_spec, out_names, supplied, result):
    """A one-sample call's result: every per-sample output in its head
    shape, the view of the n = 1 plane the run wrote (a supplied output is
    written in place and returned itself); a runtime-length output comes
    back as the plane it is."""
    role_of = {name: role for role, name in arg_spec}
    widths = _declared_widths(plugin)
    planes = dict(result) if len(out_names) > 1 else {out_names[0]: result}
    for name in out_names:
        plane = planes[name]
        if role_of[name] not in PER_SAMPLE_ROLES:
            continue
        dst = supplied.get(name)
        if dst is not None:
            same = (type(plane).__module__ == type(dst).__module__
                    and hasattr(dst, "data") and _data_ptr(plane) == _data_ptr(dst))
            if not same:
                _write_into(dst, plane.reshape(tuple(dst.shape)))
            planes[name] = dst
            continue
        head = _result_head(plugin, name, widths.get(name, 1))
        view = plane.reshape(head)
        planes[name] = view[()] if head == () and isinstance(view, np.ndarray) else view
    return planes if len(out_names) > 1 else planes[out_names[0]]


def _restore_layouts(arg_spec, adapted, result):
    """Hand every adapted output plane back in the caller's own layout: a
    copied plane's values are written into the caller's array. When the
    result and the caller's array are on the same side, the caller's array
    is returned itself; otherwise the result is transposed."""
    if not adapted:
        return result
    names = _output_names(arg_spec)
    planes = dict(result) if len(names) > 1 else {names[0]: result}
    for name in names:
        a = adapted.get(name)
        if a is None:
            continue
        r = planes[name]
        host_caller = isinstance(a.caller, np.ndarray)
        host_result = isinstance(r, np.ndarray)
        if a.scratch is not None and not isinstance(a.scratch, np.ndarray):
            a.copy_back()                       # device scratch -> device caller
        elif host_caller and not host_result:
            np.copyto(a.caller_view(), r.get())  # device result -> host caller
        elif host_caller and not np.shares_memory(r, a.caller):
            np.copyto(a.caller_view(), r)        # host result -> host caller
        planes[name] = a.caller if host_caller == host_result else r.T
    return planes if len(names) > 1 else planes[names[0]]


def _check_run_shapes(plugin, arg_spec, kw, n: int) -> None:
    """:meth:`Plan.run`'s shape check: a per-sample plane must be ``(n,)``,
    or ``(w, n)`` when the artifact declares its width ``w`` (without this
    a sample-major ``(N, 3)`` input silently read as a 3-sample run).
    Runtime-length buffers are not shape-checked, as in :meth:`Plan.bind`."""
    widths = _declared_widths(plugin)
    for role, name in arg_spec:
        if role not in PER_SAMPLE_ROLES or name not in kw:
            continue
        value = kw[name]
        shape = getattr(value, "shape", None)
        if shape is None:
            shape = np.shape(value)
        shape = tuple(int(s) for s in shape)
        width = widths.get(name)
        if width is not None:
            want = _plane_shape(width, n)
            if shape != want:
                raise ValueError(
                    f"eagle.plan.run: {role} {name!r} is declared {width} "
                    f"component(s) wide, so at n={n} its plane is {want}; got "
                    f"{shape}"
                )
        elif not shape or shape[-1] != n:
            raise ValueError(
                f"eagle.plan.run: {role} {name!r} is a per-sample plane, so its "
                f"last axis is the run's sample count n={n}; got shape {shape}"
            )


# --------------------------------------------------------------------------- #
# The capture-legal, by-name, caller-supplied-plane launch door.
# --------------------------------------------------------------------------- #

# PER_SAMPLE_ROLES planes: trailing extent must equal the run's n. Every
# other role is width-declared or a buffer whose length nothing declares.

#: What each single-process structure needs off the plugin, and the device
#: tag its mirrors carry: ``(entry attribute, device type, framework name)``.
_BIND_TARGETS = {
    "device_kernel": ("device_function", DEVICE_CUDA, "cupy"),
    "host_team": ("host_entry", DEVICE_CPU, "numpy"),
}


def _is_device_array(value) -> bool:
    """Does ``value`` expose device memory (the CUDA array interface every
    cupy array carries)? Asked by protocol, not ``isinstance``, so this
    module never imports cupy to answer."""
    return hasattr(value, "__cuda_array_interface__")


def _check_uniform(name, value) -> None:
    """A bound ``uniform`` is a scalar, frozen into its by-value box at
    bind: refused loudly rather than coerced, since ``float(some_array)``
    silently succeeds for a size-1 array (a caller meaning to vary it per
    sample would get the first element forever)."""
    if getattr(value, "shape", ()) != () and getattr(value, "ndim", 0) != 0:
        raise ValueError(
            f"eagle.plan.bind: uniform {name!r} is a by-value scalar frozen at "
            f"bind time, but an array of shape {tuple(value.shape)} was bound "
            "to it; pass the number itself (and rebind it when it changes)"
        )


def _check_plane(role, name, value, *, n, dtype, width, framework) -> None:
    """Refuse a bound plane this door cannot address, naming the field:
    :meth:`BoundPlan.launch` uploads, allocates or casts nothing, so a
    plane wrong at bind time is wrong in every replay it is recorded into."""
    if getattr(value, "shape", None) is None:
        raise ValueError(
            f"eagle.plan.bind: {role} {name!r} must be bound to a pre-allocated "
            f"{framework} array, got {type(value).__name__}; this door allocates "
            "nothing and coerces nothing (that is Plan.run's job)"
        )
    device = _is_device_array(value)
    if framework == "cupy" and not device:
        raise ValueError(
            f"eagle.plan.bind: {role} {name!r} is a HOST array, but this plan "
            "runs under eagle.exec.device_kernel and the packed mirror carries "
            "a device pointer; bind a cupy array (an upload here would be an "
            "allocation and a copy, neither of which may happen under capture)"
        )
    if framework == "numpy":
        if device:
            raise ValueError(
                f"eagle.plan.bind: {role} {name!r} is a DEVICE array, but this "
                "plan runs under eagle.exec.host_team, whose entry dereferences "
                "the pointer on the host; bind a numpy array"
            )
        if not isinstance(value, np.ndarray):
            raise ValueError(
                f"eagle.plan.bind: {role} {name!r} must be a numpy array for a "
                f"host_team plan, got {type(value).__name__}"
            )
    if not value.flags.c_contiguous:
        raise ValueError(
            f"eagle.plan.bind: {role} {name!r} is not C-contiguous "
            f"(shape {tuple(value.shape)}, strides {tuple(value.strides)}). A "
            "scalar handle carries a UNIT stride by construction and the launch "
            "may not repack, so a strided plane would be decoded at the wrong "
            "addresses; bind a contiguous plane"
        )
    bound_dtype = np.dtype(value.dtype)
    if bound_dtype.kind not in "iub" and bound_dtype != dtype:
        raise ValueError(
            f"eagle.plan.bind: {role} {name!r} has dtype {bound_dtype}, but this "
            f"artifact declares scalar_type {dtype} and its body reads and writes "
            f"{dtype.itemsize}-byte elements; bind a {dtype} plane (an integer or "
            "bool plane — an index vector, a terminated mask — is passed at its "
            "own dtype and is not this case)"
        )
    if width is not None:
        want = _plane_shape(width, n)
        if tuple(value.shape) != want:
            raise ValueError(
                f"eagle.plan.bind: {role} {name!r} is declared {width} component(s) "
                f"wide, so at n={n} its plane is {want}; got {tuple(value.shape)}"
            )
    elif role in PER_SAMPLE_ROLES and _trailing_extent(value) != n:
        raise ValueError(
            f"eagle.plan.bind: {role} {name!r} is a per-sample plane, so its last "
            f"axis is the run's sample count n={n}; got shape "
            f"{tuple(value.shape)}"
        )


def _check_bound_names(arg_spec, planes) -> None:
    """Every declared name bound, and no stray one, before anything is
    packed: a stray one is more likely a typo than a courtesy, since
    silently ignoring ``postition=`` would launch over whatever the
    correctly spelled plane held last."""
    bindable = [name for role, name in arg_spec if role != "nsamples"]
    for role, name in arg_spec:
        if role == "nsamples":
            if name in planes:
                raise ValueError(
                    f"eagle.plan.bind: {name!r} is this plugin's 'nsamples' role, "
                    "which is DERIVED from the bound planes and never bound; drop "
                    "it from the bind call"
                )
            continue
        if name not in planes:
            raise ValueError(
                f"eagle.plan.bind: this plugin's arg_spec declares a {role!r} "
                f"named {name!r}, but .bind() was not given it; every declared "
                f"name is bound by the caller (bindable: {bindable})"
            )
    for name in planes:
        if name not in bindable:
            raise ValueError(
                f"eagle.plan.bind: {name!r} is not a name this plugin's arg_spec "
                f"declares (bindable: {bindable})"
            )
