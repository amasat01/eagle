# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.plan`` — PLAN in Python, EXECUTION in C++.

:func:`plan` resolves a v2 plugin, an execution ``structure`` (always named
explicitly, never residency-derived) and a partitioning into a :class:`Plan`.
Placement legality (:func:`eagle.exec.check_placement`) is checked here, at
plan time, against every partition count the plan will launch.

:meth:`Plan.run` drives the plugin over every partition and returns the
per-sample result, marshalling each role to the ABI shape
:func:`eagle.roles.classify_arg` resolves it to — the same classifier the v1
launch paths dispatch on. Output planes are allocated in the artifact's
declared ``scalar_type`` (:mod:`eagle.dtypes`); a single-output plugin
returns the plane itself, never a 1-tuple, and several return a ``dict``
keyed by plane NAME — never a positional tuple, since ``arg_spec``'s own
order is role-grouped then name-sorted, not the kernel's authored order.

:meth:`Plan.bind` is the capture-legal door: it packs the caller's own named
planes into the argument block ONCE and returns a :class:`BoundPlan` whose
:meth:`~BoundPlan.launch` issues the structure's launches and nothing
else — no upload, allocation, sync or copy home — so a consumer can bind
once and replay the launches as ordinary nodes inside a captured graph.

Under :data:`eagle.exec.RankPartition` the plan's partitions stay WHOLE and
the rank cut happens inside the structure, so the same plan describes the
run at any world size; every rank returns the whole output plane(s).

Both doors share one packer (:func:`_pack_args` over :func:`_pack_one`), so
they cannot disagree about how a role packs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _counters
from . import exec as eexec
from .abi import DEVICE_CPU, DEVICE_CUDA
from .roles import INPUT_ROLES, OUTPUT_ROLES
from ._plan_binding import BoundPlan, FUSED_STEPS_SLOT, _bind_plan, one_step_word
from ._plan_pack import _integer_uniforms, _mutable_shapes, _n_from_kw, _pack_args
from ._plan_planes import (
    _adapt_planes,
    _bound,
    _check_run_shapes,
    _check_run_writable,
    _declared_widths,
    _device_inputs,
    _device_plane,
    _host_plane,
    _is_single_call,
    _output_names,
    _plane_dtype,
    _plane_shape,
    _restore_layouts,
    _returned,
    _single_returned,
    _write_back_outputs,
    device_resident,
    residency,
)

__all__ = ["FUSED_STEPS_SLOT", "AutoPlan", "BoundPlan", "Plan", "auto", "device_resident",
           "one_step_word", "plan", "residency"]


@dataclass(frozen=True)
class Plan:
    """WHERE and HOW ``plugin`` runs — the result of :func:`plan`.

    ``partitions`` is the resolved, explicit tuple of
    :class:`eagle.exec.Partition` triples the plan will launch (contiguous,
    covering ``[0, n_samples)``), or ``None`` when deferred: no
    ``partitions=``/``n_samples=`` was given to :func:`plan`, so the shape is
    resolved from the sample count :meth:`run` observes. Placement legality
    is already checked by :func:`plan`; :meth:`run` never re-checks it."""

    plugin: object
    structure: eexec.Structure
    access: str
    op: str | None
    partitions: tuple | None
    npartitions: int
    n_samples: int | None
    #: The structure each rank's share runs through under
    #: :data:`eagle.exec.RankPartition` (default :data:`~eagle.exec.HostTeam`);
    #: ``None`` for every other structure, which has no inner.
    inner: eexec.Structure | None = None
    #: Whether a :data:`eagle.exec.RankPartition` run gathers its output
    #: planes before returning (default ``True``, every rank holding the
    #: whole plane); ``False`` is the pre-gather arm, for asserting a rank's
    #: own sub-partition before the gather overwrites it.
    gather: bool = True

    def partitions_for(self, n: int) -> tuple:
        """The concrete :class:`eagle.exec.Partition` tuple this plan runs
        over a sample count of ``n``: the resolved :attr:`partitions` when
        explicit, else a fresh even split into :attr:`npartitions`
        contiguous partitions."""
        if self.partitions is not None:
            if self.n_samples is not None and self.n_samples != n:
                raise ValueError(
                    f"eagle.plan: this plan was built for n_samples="
                    f"{self.n_samples}, but .run() observed n={n}"
                )
            return self.partitions
        if self.npartitions == 1:
            return (eexec.Partition.whole(n),)
        base, parts, remaining, left = 0, [], n, self.npartitions
        for _ in range(self.npartitions):
            count = -(-remaining // left)  # ceil division: spread the remainder early
            parts.append(eexec.Partition(base, count, n))
            base += count
            remaining -= count
            left -= 1
        return tuple(parts)

    def run(self, /, **kw):
        """Drive :attr:`plugin` over every partition through :attr:`structure`
        and return the assembled per-sample result: the declared output
        plane as a numpy array for a single-output plugin, or a ``dict``
        keyed by plane NAME when it declares several (never a positional
        tuple — the same shape :class:`~eagle._simulate.SimResult`/
        :func:`eagle.loaded.LoadedPure` already return).

        Device inputs stay on the device: on a
        :data:`~eagle.exec.DeviceKernel` plan, an input already in device
        memory (cupy, or a CUDA torch tensor) is bound where it lies, and the
        outputs then come back as cupy arrays, never brought home — a
        caller-supplied contiguous output of the declared dtype is the
        caller's own buffer, written in place. With host inputs only, the
        outputs come home as numpy arrays. A rank-partitioned run always
        gathers on the host.

        Every output/mutable plane the caller supplies is written into in
        place, in the caller's own layout (:mod:`eagle._layout`), and still
        returned. A read-only supplied array (``flags.writeable is False``,
        or a torch tensor with ``requires_grad=True``) raises, naming the
        argument. An output not supplied is allocated fresh and returned.

        A value shaped as a plane's per-sample head (a number or 0-d array
        for a scalar, ``(w,)`` for a vector) is one sample
        (:mod:`eagle._layout`): the call runs as a batch of one and every
        output comes back in its head shape. Mixing one sample with a batch
        is refused.

        An automatic kernel (it reads the reserved ``fused_steps`` word)
        launched here without that word takes exactly one step
        (:func:`one_step_word`); only :func:`eagle.until_done` takes several
        steps per launch.

        ``self`` is positional-only, so a plugin may declare a plane
        literally named ``self`` without colliding with this method's own
        bound-method parameter."""
        return _run_plan(self, **kw)

    def bind(self, /, **planes) -> BoundPlan:
        """Bind the caller's own planes by name and pack the argument block
        once — the capture-legal door (see the module docstring).

        Every name this plugin's ``arg_spec`` declares must be given, as the
        thing the plan's structure can address: a cupy array for a
        :data:`eagle.exec.DeviceKernel` plan, numpy for
        :data:`~eagle.exec.HostTeam`, a plain scalar for a ``uniform``
        (frozen into its by-value box here). ``nsamples`` is derived, never
        bound. The reserved ``fused_steps`` word of an automatic kernel may
        be left out: a one-step word is then bound for you
        (:func:`one_step_word`); :func:`eagle.until_done` binds its own.

        Nothing is allocated, coerced or copied, with one exception: a
        sample-major ``(N, w)`` plane (:mod:`eagle._layout`) binds zero-copy
        when its transpose is C-contiguous, else is copied here into a
        component-major buffer this :class:`BoundPlan` owns (with an
        :class:`eagle.LayoutWarning`) and refreshes before each launch,
        copying back after for a written plane — capture-legal, no
        device-wide sync. The output planes are always the caller's own:
        this door cannot allocate one, since it would have to outlive the
        graph that writes it with no handle for the caller to hold.

        What's checked here (every refusal names the field): every declared
        name is bound and no stray one is; each plane is an array of the
        structure's own framework (a host array to a device plan is refused,
        not uploaded); it is C-contiguous; its dtype is the artifact's
        declared ``scalar_type`` (an integer/bool plane passes at its own
        dtype, as :meth:`run` does); and its shape agrees with what the
        artifact declared. An input plane's component width is NOT checked:
        a v2 sidecar declares widths for its output planes only.

        ``self`` is positional-only for the same reason :meth:`run`'s is."""
        bound = _bind_plan(self, planes)
        _counters.bump("binds")  # only on a completed bind
        return bound


def _resolve_explicit_partitions(partitions):
    """Validate and build the :class:`eexec.Partition` tuple for an explicit
    ``partitions=`` argument: contiguous, covering ``[0, n)``, every triple
    sharing the same ``nSamples``. Returns ``(partitions, n_samples)``."""
    parts = [eexec.Partition(int(b), int(c), int(n)) for b, c, n in partitions]
    if not parts:
        raise ValueError("eagle.plan: partitions=[] carries no partitions")
    ordered = sorted(parts, key=lambda p: p.base)
    cursor = 0
    n_samples = None
    for p in ordered:
        if p.base != cursor:
            raise ValueError(
                "eagle.plan: partitions must be contiguous and cover [0, n) "
                f"with no gap/overlap -- expected base {cursor}, got {p.base}"
            )
        if n_samples is None:
            n_samples = p.n_samples
        elif p.n_samples != n_samples:
            raise ValueError(
                "eagle.plan: every partition triple must share the same "
                f"nSamples -- got {n_samples} and {p.n_samples}"
            )
        cursor = p.base + p.count
    if n_samples is not None and cursor != n_samples:
        raise ValueError(
            f"eagle.plan: partitions cover [0, {cursor}) but nSamples={n_samples}"
        )
    return tuple(parts), n_samples


def plan(
    plugin,
    *,
    structure,
    inner=None,
    npartitions: int = 1,
    partitions=None,
    n_samples: int | None = None,
    _exec_access: str | None = None,
    _exec_op: str | None = None,
    _gather: bool = True,
) -> Plan:
    """Build a :class:`Plan` for running ``plugin`` under ``structure``
    (always named explicitly, never residency-derived).

    ``structure`` is one of :data:`eagle.exec.DeviceKernel` /
    :data:`~eagle.exec.HostTeam` / :data:`~eagle.exec.RankPartition` /
    :data:`~eagle.exec.DeviceGroup`. ``inner`` names the structure each
    rank's share runs through under :data:`~eagle.exec.RankPartition`
    (:data:`~eagle.exec.HostTeam` by default) and is refused for any other
    structure. The plan's partitions stay WHOLE either way — the rank cut
    happens inside the structure, so the same plan describes the run at any
    world size. ``partitions`` (a list of ``(base, count, n_samples)``
    triples) takes precedence over ``npartitions`` and is validated for
    contiguity/coverage; otherwise the plan is deferred to a whole view
    (``npartitions=1``, the default) or an even split, resolved once
    :meth:`Plan.run` observes the sample count. The execution axis is the
    plugin's own ``.exec_access``/``.exec_op`` declaration.

    Placement legality is checked here, at plan time, against the partition
    count this plan will launch. A refused plan names the rule.

    A plugin declaring ``.abi_tag == eagle.exec.ABI_TAG_V1`` (a legacy v1
    ``LoadedVector``/``LoadedPure``) carries no partition triple, so a
    partition count above 1 is refused naming "legacy"."""
    # test seams: a placement check without a v2 plugin body, and the rank
    # structure's pre-gather arm
    access = (
        _exec_access if _exec_access is not None
        else getattr(plugin, "exec_access", None)
    )
    if not access:
        raise ValueError(
            "eagle.plan: exec_access is required -- declare .exec_access on "
            "the plugin"
        )
    op = _exec_op if _exec_op is not None else getattr(plugin, "exec_op", None)

    if partitions is not None:
        resolved, resolved_n = _resolve_explicit_partitions(partitions)
        if n_samples is not None and resolved_n is not None and n_samples != resolved_n:
            raise ValueError(
                f"eagle.plan: n_samples={n_samples} disagrees with the "
                f"partitions' own nSamples={resolved_n}"
            )
        n_samples = n_samples if n_samples is not None else resolved_n
        npart = len(resolved)
    else:
        resolved = None
        if npartitions < 1:
            raise ValueError(f"eagle.plan: npartitions must be >= 1, got {npartitions}")
        npart = npartitions

    if (
        _exec_access is None
        and npart > 1
        and getattr(plugin, "abi_tag", None) == eexec.ABI_TAG_V1
    ):
        raise ValueError(
            "eagle.plan: illegal placement: this plugin is a legacy "
            f"{eexec.ABI_TAG_V1!r} artifact and carries no partition triple "
            "-- it may only be driven whole-view, single-structure; "
            f"got {npart} partitions"
        )

    if structure is not eexec.RankPartition:
        if inner is not None:
            raise ValueError(
                "eagle.plan: inner= names the structure each RANK's share is "
                "driven through, so it is meaningful only for rank_partition; "
                f"got structure={structure!r}"
            )
        if not _gather:
            raise ValueError(
                "eagle.plan: _gather=False is the rank structure's PRE-GATHER "
                f"arm; a {structure.name!r} run has no gather to suppress"
            )

    eexec.check_placement(access, structure, npart)  # placement legality, at PLAN time

    return Plan(
        plugin=plugin,
        structure=structure,
        access=access,
        op=op,
        partitions=resolved,
        npartitions=npart,
        n_samples=n_samples,
        inner=inner,
        gather=_gather,
    )


def _run_device_plan(plugin, arg_spec, kw, n: int, parts, *, launch=None, gather=None,
                     keep_device=False):
    """Run a device plan. Outputs come back as host numpy arrays, except
    when ``keep_device`` (device-resident inputs, no gather follows): then
    they stay as cupy arrays — a caller-supplied contiguous output of the
    declared dtype is the caller's own buffer, written in place."""
    import cupy as cp

    launch = launch if launch is not None else _launch_device
    out_names = _output_names(arg_spec)
    dtype = _plane_dtype(plugin)
    widths = _declared_widths(plugin)
    vec_mut, mat_mut = _mutable_shapes(plugin)
    device_arrays = {}
    for role, name in arg_spec:
        if role in OUTPUT_ROLES:
            device_arrays[name] = (
                cp.ascontiguousarray(cp.asarray(kw[name]))
                if name in kw
                else cp.zeros(_plane_shape(widths.get(name, 1), n), dtype=dtype)
            )
        elif role in INPUT_ROLES:
            device_arrays[name] = _device_plane(_bound(kw, name, role), dtype)

    if gather is not None:
        for name in out_names:
            _check_gatherable(device_arrays[name])

    fn = plugin.device_function
    for part in parts:
        boxes, addrs = _pack_args(
            arg_spec, device_arrays, kw, n, DEVICE_CUDA,
            vec_mutables=vec_mut, mat_mutables=mat_mut, dtype=dtype,
            int_uniforms=_integer_uniforms(plugin),
        )
        launch(fn, addrs, part)
        del boxes  # kept alive until run() returns, below this loop's scope
    cp.cuda.runtime.deviceSynchronize()
    if keep_device and gather is None:
        return _returned(device_arrays, out_names)
    # planes come home before anything else looks at them: this makes the
    # rank gather a HOST gather (CUDA-aware MPI was ruled out)
    planes = {name: cp.asnumpy(device_arrays[name]) for name in out_names}
    if gather is not None:
        for name in out_names:
            gather(planes[name], n, parts)
    return _returned(planes, out_names)


def _run_host_plan(plugin, arg_spec, kw, n: int, parts, *, launch=None, gather=None):
    out_names = _output_names(arg_spec)
    dtype = _plane_dtype(plugin)
    widths = _declared_widths(plugin)
    vec_mut, mat_mut = _mutable_shapes(plugin)
    host_arrays = {}
    for role, name in arg_spec:
        if role in OUTPUT_ROLES:
            # a caller-supplied output plane is written IN PLACE and returned
            # as given (its own dtype and width), never re-cast.
            host_arrays[name] = (
                np.ascontiguousarray(kw[name])
                if name in kw
                else np.zeros(_plane_shape(widths.get(name, 1), n), dtype=dtype)
            )
        elif role in INPUT_ROLES:
            host_arrays[name] = _host_plane(_bound(kw, name, role), dtype)

    if gather is not None:
        for name in out_names:
            _check_gatherable(host_arrays[name])

    entry = plugin.host_entry
    launch = launch if launch is not None else _launch_host
    for part in parts:
        boxes, addrs = _pack_args(
            arg_spec, host_arrays, kw, n, DEVICE_CPU,
            vec_mutables=vec_mut, mat_mutables=mat_mut, dtype=dtype,
            int_uniforms=_integer_uniforms(plugin),
        )
        launch(entry, addrs, part)
        del boxes
    if gather is not None:
        for name in out_names:
            gather(host_arrays[name], n, parts)
    return _returned(host_arrays, out_names)


def _launch_host(entry, addrs, part):
    """The single-process host launch (:data:`eagle.exec.HostTeam`, one
    partition). Named so the rank path can substitute its own launch without
    duplicating the packer."""
    return eexec.HostTeam.run(entry, addrs, part)


def _launch_device(fn, addrs, part):
    """The single-process device launch — the twin of :func:`_launch_host`."""
    return eexec.DeviceKernel.run(fn, addrs, part, stream=0)


def _check_gatherable(plane) -> None:
    """Refuse an output plane the rank gather cannot carry, before anything
    launches: the wire is ``MPI_DOUBLE`` at computed sample offsets, so a
    non-float64 or non-contiguous plane cannot cross it correctly — and a
    refusal issued after the launch would already be reporting a buffer the
    body had overrun."""
    if plane.dtype != np.float64:
        raise ValueError(
            "eagle.plan: eagle.exec.RankPartition gathers output planes over "
            f"MPI_DOUBLE, so a {plane.dtype} plane cannot cross the wire "
            "correctly; declare scalar_type 'float64' for a rank-partitioned run"
        )
    if not plane.flags["C_CONTIGUOUS"]:
        raise ValueError(
            "eagle.plan: eagle.exec.RankPartition gathers an output plane at "
            "computed sample offsets, so the plane must be C-contiguous"
        )


def _gather_plane(plane, n: int, parts) -> None:
    """Gather one output ``plane`` across the world, in place
    (:meth:`eagle.exec.RankPartition.allgather_plane`), so every rank ends
    holding the whole of it.

    A width>1 plane is ``(width, n)`` C-contiguous, so each component row is
    itself a contiguous per-sample plane and is gathered separately, row by
    row.

    ``parts`` are the WHOLE partitions the plan launched — the world's cut,
    before the per-rank one."""
    addr = plane.ctypes.data
    rows = 1 if plane.ndim < 2 else int(plane.size // n)
    stride = 0 if rows == 1 else int(n) * plane.itemsize
    for part in parts:
        for r in range(rows):
            eexec.RankPartition.allgather_plane(addr + r * stride, part, 1)


def _run_rank_plan(p: Plan, plugin, arg_spec, kw, n: int, parts):
    """Drive ``plugin`` under :data:`eagle.exec.RankPartition` — the same
    packing paths as the single-process runs, with the launch structure
    swapped and a gather appended.

    The plan's ``parts`` stay WHOLE partitions: the rank cut happens inside
    ``RankPartition``, never here. Every rank runs the same loop and returns
    the whole output plane(s) (the collectives are blocking)."""
    inner = p.inner if p.inner is not None else eexec.HostTeam
    if inner not in (eexec.HostTeam, eexec.DeviceKernel):
        raise ValueError(
            f"eagle.plan: rank_partition's inner structure must be host_team or "
            f"device_kernel, got {inner!r}"
        )

    def launch(entry, addrs, part):
        return eexec.RankPartition.run(entry, addrs, part, inner=inner)

    runner = _run_device_plan if inner is eexec.DeviceKernel else _run_host_plan
    return runner(plugin, arg_spec, kw, n, parts, launch=launch,
                  gather=_gather_plane if p.gather else None)


def _run_plan(p: Plan, /, **kw):
    """``Plan.run``'s body. ``p`` is positional-only: a plugin declaring a
    plane literally named ``p`` would otherwise collide with this
    function's own first parameter."""
    plugin = p.plugin
    arg_spec = getattr(plugin, "arg_spec", None)
    if arg_spec is None:
        raise ValueError(
            "eagle.plan: plugin has no .arg_spec -- .run() cannot pack args"
        )
    out_names = _output_names(arg_spec)
    _check_run_writable(arg_spec, out_names, kw)
    kw = one_step_word(arg_spec, kw,
                       "cupy" if p.structure is eexec.DeviceKernel else "numpy")
    # the caller's original output objects, before _adapt_planes can replace
    # a sample-major one with its native-layout view/scratch
    supplied = {name: kw[name] for name in out_names if name in kw}
    kw, adapted = _adapt_planes(plugin, arg_spec, kw)
    n = _n_from_kw(arg_spec, kw)
    _check_run_shapes(plugin, arg_spec, kw, n)
    parts = p.partitions_for(n)
    if adapted and _is_single_call(adapted):
        result = _run_structure(p, plugin, arg_spec, kw, n, parts)
        return _single_returned(plugin, arg_spec, out_names, supplied, result)

    result = _run_structure(p, plugin, arg_spec, kw, n, parts)
    result = _restore_layouts(arg_spec, adapted, result)
    return _write_back_outputs(out_names, supplied, adapted, result)


def _run_structure(p: Plan, plugin, arg_spec, kw, n: int, parts):
    """Run the adapted ``kw`` under the plan's structure; the raw result."""
    if p.structure is eexec.DeviceKernel:
        return _run_device_plan(plugin, arg_spec, kw, n, parts,
                                keep_device=_device_inputs(arg_spec, kw))
    if p.structure is eexec.HostTeam:
        return _run_host_plan(plugin, arg_spec, kw, n, parts)
    if p.structure is eexec.RankPartition:
        return _run_rank_plan(p, plugin, arg_spec, kw, n, parts)
    # DeviceGroup: check_placement already refused it at plan() time for
    # every access class, so .run() can never reach here -- defensive only.
    raise ValueError(
        f"eagle.plan: execution structure {p.structure!r} is not runnable in "
        "this build (NCCL not implemented)"
    )

# AutoPlan builds plans with plan() above, so its module loads last.
from ._plan_deploy import AutoPlan, auto  # noqa: E402
