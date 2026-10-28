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


from . import _alloc, _core, _stream, arrays
from .dtypes import dtype_name
from .interop import BufferRefused, scoped_stream
from .launch import _current_device_ordinal

from . import _cell_rows, _counters, _layout, _split
from ._graph_reduce import check_policy
from .frameworks._bridge import route
from . import _index_set
from . import exec as eexec
from .abi import DEVICE_CPU, DEVICE_CUDA
from .roles import INPUT_ROLES, OUTPUT_ROLES
from ._plan_binding import BoundPlan, FUSED_STEPS_SLOT, _bind_plan, one_step_word
from . import _lane_bind
from ._plan_pack import _data_ptr, _integer_uniforms, _mutable_shapes, _pack_args
from ._plan_planes import (
    _adapt_planes,
    _bound,
    _plugin_name,
    sample_spec,
    _check_run_shapes,
    _check_run_writable,
    _declared_widths,
    _device_inputs,
    _declared_dtype,
    _device_plane,
    _host_plane,
    _is_single_call,
    _output_names,
    _plane_dtype,
    _restore_layouts,
    _returned,
    _single_returned,
    _write_back_outputs,
    device_resident,
    function_on,
    planes_device,
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
    #: How the rows of a reduction over samples are finished into the cell:
    #: ``"compensated"`` (the default), ``"ordered"`` (the same bits on every
    #: run, host and device, and across ranks) or ``"auto"``, as
    #: :func:`eagle.reduce` takes it.
    policy: str = "compensated"

    def partitions_for(self, n: int) -> tuple:
        """The concrete :class:`eagle.exec.Partition` tuple this plan runs
        over a sample count of ``n``: the resolved :attr:`partitions` when
        explicit, else a fresh even split into :attr:`npartitions`
        contiguous partitions. A thread-box kernel splits its threads
        (:func:`eagle._lane_bind.threads`), each partition still over ``n``."""
        if self.partitions is not None:
            if self.n_samples is not None and self.n_samples != n and not (
                    _index_set.is_prefix(self.plugin) and n < self.n_samples):   # a prefix kernel runs any count below its bound
                raise ValueError(
                    f"eagle.plan: this plan was built for n_samples="
                    f"{self.n_samples}, but .run() observed n={n}"
                )
            return self.partitions
        m = _lane_bind.threads(self.plugin, n)
        if self.npartitions == 1:
            return (eexec.Partition(0, m, n),)
        base, parts, remaining, left = 0, [], m, self.npartitions
        for _ in range(self.npartitions):
            count = -(-remaining // left)  # ceil division: spread the remainder early
            parts.append(eexec.Partition(base, count, n))
            base += count
            remaining -= count
            left -= 1
        return tuple(parts)

    @_layout.door
    def run(self, /, *, device=None, **kw):
        """Drive :attr:`plugin` over every partition through :attr:`structure`
        and return the assembled per-sample result: the declared output
        plane as an :class:`eagle.Array` for a single-output plugin, or a ``dict``
        keyed by plane NAME when it declares several (never a positional
        tuple — the same shape :class:`~eagle._simulate.SimResult`/
        :func:`eagle.loaded.LoadedPure` already return).

        Device inputs stay on the device: on a
        :data:`~eagle.exec.DeviceKernel` plan, an input already in device
        memory (cupy, or a CUDA torch tensor) is bound where it lies, and the
        outputs then come back as cupy arrays, never brought home — a
        caller-supplied contiguous output of the declared dtype is the
        caller's own buffer, written in place. With host inputs only, the
        outputs come home as :class:`eagle.Array` objects. A rank-partitioned run always
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

        A device run uses the GPU its arrays live on, whatever is current;
        arrays on two GPUs are refused by name. ``device=n`` is checked against
        the arrays (it never moves them); with host arrays only, it picks the
        GPU they are uploaded to.

        ``returns="numpy"|"cupy"|"torch"|"jax"`` hands the results back as that
        framework's arrays through its bridge (a missing framework warns once and
        returns :class:`eagle.Array` s); the default is :class:`eagle.Array`.

        ``self`` is positional-only, so a plugin may declare a plane
        literally named ``self`` without colliding with this method's own
        bound-method parameter."""
        returns = kw.pop("returns", None)
        if device is None:
            return route(_run_plan(self, **kw), returns)
        arg_spec = getattr(self.plugin, "arg_spec", None) or ()
        on = _checked_device(self, arg_spec, kw, "plan.run", device)
        with _stream.on_device(on):
            return route(_run_plan(self, **kw), returns)

    @_layout.door
    def bind(self, /, *, device=None, capacity=None, **planes) -> BoundPlan:
        """Bind the caller's own planes by name and pack the argument block
        once — the capture-legal door (see the module docstring).

        Every name this plugin's ``arg_spec`` declares must be given, as the
        thing the plan's structure can address: a cupy array for a
        :data:`eagle.exec.DeviceKernel` plan, numpy for
        :data:`~eagle.exec.HostTeam`, a plain scalar for a ``uniform``
        (frozen into its by-value box here). ``nsamples`` is derived, never
        bound. A kernel split in stages binds the planes of
        :func:`eagle._split.arg_spec` and returns a
        :class:`eagle._split.SplitBound`, which owns the rows and cells between
        its stages and stands wherever a :class:`BoundPlan` does. The reserved ``fused_steps`` word of an automatic kernel may
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
        graph that writes it with no handle for the caller to hold. An
        ``int32`` input plane for an ``int64`` ``Index`` is widened by one
        copy here, once (the bound plan holds the copy; :meth:`BoundPlan.rebind`
        widens again), so a launch copies nothing.

        What's checked here (every refusal names the field): every declared
        name is bound and no stray one is; each plane is an array of the
        structure's own framework (a host array to a device plan is refused,
        not uploaded); it is C-contiguous; its dtype is the artifact's
        declared ``scalar_type`` (an integer/bool plane passes at its own
        dtype, as :meth:`run` does); and its shape agrees with what the
        artifact declared. An input plane's component width is NOT checked:
        a v2 sidecar declares widths for its output planes only.

        The plan is bound for the GPU its planes live on (``device=n`` is
        checked against them, as in :meth:`run`); :meth:`BoundPlan.launch`
        issues there whatever is current.

        ``capacity=n_max`` is for a kernel built with ``IndexSet(prefix=True)``: the planes hold ``n_max`` lanes,
        eagle makes the ``active_count`` plane (set to ``n_max``), and ``bound.set_count(m)`` then runs the first
        ``m`` lanes of the same bind and captured graph. A larger ``m`` is refused by name.

        ``self`` is positional-only for the same reason :meth:`run`'s is."""
        bound = _bind_plan(self, planes, device, capacity)
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
    policy: str = "compensated",
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
        if getattr(plugin, "box", None):
            raise ValueError(
                "eagle.plan: this kernel computes several lanes per thread (a thread "
                "box), so its partitions count threads, not lanes; use npartitions=, "
                "or build the kernel with HAWK_BOX=0")
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

    check_policy(policy, "eagle.plan")
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

    if structure is eexec.RankPartition and _cell_rows.cells_of(plugin):
        for part in resolved or ():
            _cell_rows.whole_rows(part)     # a base off a row is refused here
    eexec.check_placement(access, structure, npart)  # placement legality, at PLAN time
    if any(name == "device" for _role, name in getattr(plugin, "arg_spec", None) or ()):
        raise ValueError("eagle.plan: a parameter named 'device' clashes with the "
                         "device= keyword; rename it")

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
        policy=policy,
    )


def _output_dtype(plugin, name, real):
    """The dtype an output plane ``run`` allocates: the sidecar's declared wire
    dtype (an ``Int32`` plane is ``int32``, an ``Index`` one ``int64``), else
    the artifact's Real."""
    declared = (getattr(plugin, "arg_dtypes", None) or {}).get(name)
    return dtype_name(declared) if declared else real


def _host_view(name, role, value):
    """``value`` as a host :class:`eagle.Array` view, zero-copy; a device-resident output
    is refused by name (a host plan cannot write device memory)."""
    from ._plan_planes import device_resident

    if device_resident(value):
        raise BufferRefused(
            f"eagle.plan.run: {role} {name!r} is device-resident but this plan "
            "runs on the host; pass a host array or omit it to get one back"
        )
    return arrays.asarray(value)


def _checked_device(p, arg_spec, kw, door, device):
    """The device a ``device=`` call runs on: the arrays', checked against the
    one asked for. A host plan has no device to name."""
    if p.structure is not eexec.DeviceKernel and p.inner is not eexec.DeviceKernel:
        raise BufferRefused(f"eagle.{door}: device={device} names a CUDA device, "
                            "and this plan runs on the host")
    return planes_device(arg_spec, kw, door=door, device=device)


def _checked_out(name, role, arr, want, *, device):
    """Refuse, before any launch, a caller-supplied output ``arr`` that is
    not the plane the kernel writes: the dtype, contiguity or device must
    match. Returns ``arr``."""
    if dtype_name(arr.dtype) != dtype_name(want):
        raise BufferRefused(
            f"eagle.plan.run: {role} {name!r} must be {dtype_name(want)} "
            f"(the plane's dtype); got {dtype_name(arr.dtype)}"
        )
    if not (arr.is_contiguous if hasattr(arr, "is_contiguous") else arr.flags.c_contiguous):
        raise BufferRefused(
            f"eagle.plan.run: {role} {name!r} must be C-contiguous (the "
            f"kernel writes the plane in place); got strides {arr.strides}"
        )
    if device:
        on, here = _core.pointer_device(_data_ptr(arr)), _current_device_ordinal()
        if on != here:
            raise BufferRefused(
                f"eagle.plan.run: {role} {name!r} lives on device {on}, the run is on device {here}"
            )
    return arr


def _on_run_device(value, device: str):
    """``value`` (a caller's output) as an :class:`eagle.Array`: where it lies on a device, else uploaded to the
    run's ``device`` (the result is written back to the caller's own buffer afterwards)."""
    a = arrays.asarray(value)
    return a if a.device.startswith("cuda") else arrays.asarray(a, device=device)


def _torch_stream_context(kw):
    """The context the device run executes within: torch's current stream
    when a CUDA torch tensor was passed (so every copy and the launch ride
    it), else eagle's own stream scope as it stands."""
    import contextlib

    for value in kw.values():
        if type(value).__module__.partition(".")[0] == "torch" and \
                getattr(value, "is_cuda", False):
            import torch

            # eagle's launch rides it, and so does a caller's cupy (scoped_stream sets it when imported)
            return scoped_stream(torch.cuda.current_stream(value.device).cuda_stream)
    return contextlib.nullcontext()


def _run_device_plan(plugin, arg_spec, kw, n: int, parts, *, launch=None, gather=None,
                     keep_device=False, policy="compensated"):
    """Run a device plan. Outputs come back as host :class:`eagle.Array` objects, except
    when ``keep_device`` (device-resident inputs, no gather follows): then
    they stay as cupy arrays — a caller-supplied contiguous output of the
    declared dtype is the caller's own buffer, written in place."""
    xp = f"cuda:{_current_device_ordinal()}"

    if launch is None:
        declared = getattr(plugin, "block", None)     # the artifact's exact threads per block

        def launch(fn, addrs, part):
            return _launch_device(fn, addrs, part, declared)

    out_names = _output_names(arg_spec)
    dtype = _plane_dtype(plugin)
    widths = _declared_widths(plugin)
    vec_mut, mat_mut = _mutable_shapes(plugin)
    extents = _lane_bind.extents_of(plugin, kw)
    cells = _cell_rows.scratch(plugin, n, xp, dtype, policy)
    device_arrays = {}
    for role, name in arg_spec:
        if cells is not None and name in cells.rows:
            _refuse_supplied_cell(name, kw)
            device_arrays[name] = cells.rows[name]
        elif role in OUTPUT_ROLES:
            device_arrays[name] = (
                _lane_bind.check_sink(
                    plugin, name,
                    _checked_out(name, role, _on_run_device(kw[name], xp),
                                 _output_dtype(plugin, name, dtype), device=True), n, extents, kw)
                if name in kw
                else _alloc.zeros(xp, _lane_bind.output_shape(plugin, name, widths.get(name, 1), n, extents, kw),
                               dtype=_output_dtype(plugin, name, dtype))
            )
        elif role in INPUT_ROLES:
            device_arrays[name] = _device_plane(_bound(kw, name, role), dtype,
                                                   declared=_declared_dtype(plugin, name),
                                                   name=name, role=role, kernel=_plugin_name(plugin))
    owned = _cell_rows.owned_pairs(plugin, kw)
    for target, comp in owned:      # the run owns it: the target's shape, zeroed
        device_arrays[comp] = _alloc.zeros_like(xp, device_arrays[target])

    if gather is not None:
        for name in out_names:
            _check_gatherable(device_arrays[name])

    device = _current_device_ordinal()
    fn = function_on(plugin, device)
    for part in parts:
        boxes, addrs = _pack_args(
            arg_spec, device_arrays, kw, n, DEVICE_CUDA,
            vec_mutables=vec_mut, mat_mutables=mat_mut, dtype=dtype,
            int_uniforms=_integer_uniforms(plugin),
        )
        launch(fn, addrs, part)
        for entry, whole in _cell_rows.fold_steps(plugin, device_arrays, owned, device):
            _launch_device(entry, addrs, whole)
        del boxes  # kept alive until run() returns, below this loop's scope
    name = str(getattr(plugin, "name", None) or getattr(plugin, "kernel", "?"))
    _stream.check_launch(name)
    try:
        _core.device_synchronize()
    except RuntimeError as exc:
        raise RuntimeError(f"eagle: kernel {name!r}: the launch failed ({exc})") from None
    if keep_device and gather is None:
        return _returned(*_finished(plugin, cells, device_arrays, out_names))
    # planes come home before anything else looks at them: this makes the
    # rank gather a HOST gather (CUDA-aware MPI was ruled out)
    planes = {name: arrays.asarray(device_arrays[name], device="cpu") for name in out_names
              if cells is None or name not in cells.rows}
    if gather is not None:
        for name in planes:
            gather(planes[name], n, parts)
        for name, rows in (cells.rows if cells is not None else {}).items():
            host = arrays.asarray(rows, device="cpu")
            gather(host, n, parts, rows=True)
            rows.set(host)
    if cells is not None:
        cells.finish()
        planes.update({name: arrays.asarray(c, device="cpu") for name, c in cells.cell.items()})
    out_names = tuple(nm for nm in out_names if nm not in _cell_rows.hidden(plugin))
    return _returned(planes, out_names)


def _run_host_plan(plugin, arg_spec, kw, n: int, parts, *, launch=None, gather=None,
                   policy="compensated"):
    out_names = _output_names(arg_spec)
    dtype = _plane_dtype(plugin)
    widths = _declared_widths(plugin)
    vec_mut, mat_mut = _mutable_shapes(plugin)
    extents = _lane_bind.extents_of(plugin, kw)
    cells = _cell_rows.scratch(plugin, n, "cpu", dtype, policy)
    host_arrays = {}
    for role, name in arg_spec:
        if cells is not None and name in cells.rows:
            _refuse_supplied_cell(name, kw)
            host_arrays[name] = cells.rows[name]
        elif role in OUTPUT_ROLES:
            # a caller-supplied output plane is written IN PLACE and returned
            # as given; one that is not the plane's own dtype/layout/device
            # is refused before the launch (the kernel would write through
            # the wrong element size).
            host_arrays[name] = (
                _lane_bind.check_sink(
                    plugin, name,
                    _checked_out(name, role, _host_view(name, role, kw[name]),
                                 _output_dtype(plugin, name, dtype), device=False), n, extents, kw)
                if name in kw
                else arrays.zeros(_lane_bind.output_shape(plugin, name, widths.get(name, 1), n, extents, kw),
                               dtype=_output_dtype(plugin, name, dtype))
            )
        elif role in INPUT_ROLES:
            host_arrays[name] = _host_plane(_bound(kw, name, role), dtype,
                                               declared=_declared_dtype(plugin, name),
                                               name=name, role=role, kernel=_plugin_name(plugin))
    owned = _cell_rows.owned_pairs(plugin, kw)
    for target, comp in owned:      # the run owns it: the target's shape, zeroed
        host_arrays[comp] = arrays.zeros_like(host_arrays[target])

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
        for fold, whole in _cell_rows.fold_steps(plugin, host_arrays, owned):
            eexec.HostTeam.run_serial(fold, addrs, whole)
        del boxes
    if gather is not None:
        for name in out_names:
            if cells is not None and name in cells.rows:
                gather(host_arrays[name], n, parts, rows=True)
            else:
                gather(host_arrays[name], n, parts)
    supplied = {name: kw[name] for name in out_names if name in kw}     # handed back as the caller's own objects
    return _returned(*_finished(plugin, cells, {**host_arrays, **supplied}, out_names))


def _refuse_supplied_cell(name, kw) -> None:
    if name in kw:
        raise BufferRefused(
            f"eagle.plan.run: output {name!r} is a reduction over samples; run() "
            "finishes its rows into the one-element cell and returns it, so do "
            "not pass a plane for it")


def _finished(plugin, cells, arrays, out_names):
    """``(arrays, names)`` of a run's result: the finished cell in place of its
    rows (the ``_comp`` companions are not handed back)."""
    if cells is None:
        return arrays, tuple(nm for nm in out_names if nm not in _cell_rows.hidden(plugin))
    cells.finish()
    arrays = {**arrays, **cells.cell}
    return arrays, tuple(nm for nm in out_names if nm not in _cell_rows.hidden(plugin))


def _launch_host(entry, addrs, part):
    """The single-process host launch (:data:`eagle.exec.HostTeam`, one
    partition). Named so the rank path can substitute its own launch without
    duplicating the packer."""
    return eexec.HostTeam.run(entry, addrs, part)


def _launch_device(fn, addrs, part, block=None):
    """The single-process device launch — the twin of :func:`_launch_host`.
    ``block`` is the artifact's declared ``launch.block`` (``None``: the default)."""
    # eagle's current stream (the ``eagle.stream`` scope, which a torch call opens for the run): the legacy
    # stream does not wait for a non-blocking one
    return eexec.DeviceKernel.run(fn, addrs, part, stream=_stream.current(),
                                  **({} if block is None else {"block": block}))


def _check_gatherable(plane) -> None:
    """Refuse an output plane the rank gather cannot carry, before anything
    launches: the wire is ``MPI_DOUBLE`` at computed sample offsets, so a
    non-float64 or non-contiguous plane cannot cross it correctly — and a
    refusal issued after the launch would already be reporting a buffer the
    body had overrun."""
    if dtype_name(plane.dtype) != "float64":
        raise ValueError(
            "eagle.plan: eagle.exec.RankPartition gathers output planes over "
            f"MPI_DOUBLE, so a {plane.dtype} plane cannot cross the wire "
            "correctly; declare scalar_type 'float64' for a rank-partitioned run"
        )
    if not plane.is_contiguous:
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
    addr = plane.ptr
    rows = 1 if plane.ndim < 2 else int(plane.size // n)
    stride = 0 if rows == 1 else int(n) * plane.itemsize
    for part in parts:
        for r in range(rows):
            eexec.RankPartition.allgather_plane(addr + r * stride, part, 1)


def _gather_aligned(plane, n: int, parts, rows: bool = False) -> None:
    """The gather of a run whose ranks were cut at whole rows
    (see :func:`eagle._cell_rows.rank_samples`): ``rows`` is a cell's row plane, gathered
    one element per row; any other plane goes through a copy padded to whole
    rows, because the world's cut is over rows, not samples."""
    for part in parts:
        whole = eexec.Partition(*_cell_rows.whole_rows(part))
        if rows:
            eexec.RankPartition.allgather_plane(plane.ptr, whole, 1)
            continue
        for comp in ([plane[i] for i in range(plane.shape[0])] if plane.ndim > 1 else [plane]):
            padded = arrays.zeros(whole.n_samples * _cell_rows.ROW, dtype="float64")
            arrays.copyto(padded[:n], comp)
            eexec.RankPartition.allgather_plane(padded.ptr, whole, _cell_rows.ROW)
            arrays.copyto(comp, padded[:n])


def _run_rank_plan(p: Plan, plugin, arg_spec, kw, n: int, parts, policy="compensated"):
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

    aligned = bool(_cell_rows.cells_of(plugin))
    inner_launch = _launch_device if inner is eexec.DeviceKernel else _launch_host

    def launch(entry, addrs, part):
        if not aligned:
            return eexec.RankPartition.run(entry, addrs, part, inner=inner)
        # a cells kernel writes one row per 32 samples: cut at whole rows, so
        # no row has two writers
        rank, size = eexec.RankPartition.world()
        mine = eexec.Partition(*_cell_rows.rank_samples(part, rank, size))
        return inner_launch(entry, addrs, mine) if mine.count else 0

    runner = _run_device_plan if inner is eexec.DeviceKernel else _run_host_plan
    gather = _gather_aligned if aligned else _gather_plane
    return runner(plugin, arg_spec, kw, n, parts, launch=launch,
                  gather=gather if p.gather else None, policy=policy)


def _run_plan(p: Plan, /, **kw):
    """``Plan.run``'s body. ``p`` is positional-only: a plugin declaring a
    plane literally named ``p`` would otherwise collide with this
    function's own first parameter."""
    if _split.stages_of(p.plugin):
        return _split.run(p, kw, _run_one)
    return _run_one(p, **kw)


def _run_one(p: Plan, /, **kw):
    """One kernel's run: the body of :func:`_run_plan` (a split kernel's stages
    each run through here)."""
    arg_spec = getattr(p.plugin, "arg_spec", None)
    if arg_spec is None:
        raise ValueError(
            "eagle.plan: plugin has no .arg_spec -- .run() cannot pack args"
        )
    on = None
    if p.structure is eexec.DeviceKernel or p.inner is eexec.DeviceKernel:
        on = planes_device(arg_spec, kw, door="plan.run")
    with _stream.on_device(on):      # the planes' GPU, whatever is current
        plugin = p.plugin
        owned = _cell_rows.companions_of(plugin)
        out_names = tuple(nm for nm in _output_names(arg_spec) if nm not in owned)
        _check_run_writable(arg_spec, out_names, kw)
        kw = one_step_word(arg_spec, kw,
                           "cupy" if p.structure is eexec.DeviceKernel else "numpy")
        # the caller's original output objects, before _adapt_planes can replace
        # a sample-major one with its native-layout view/scratch
        kw, n, lane_dims = _lane_bind.bind_n(plugin, arg_spec, kw, _declared_widths(plugin))
        _index_set.check_bound(plugin, arg_spec, kw, n)
        supplied = {name: kw[name] for name in out_names if name in kw}
        kw, adapted = _adapt_planes(plugin, arg_spec, kw)
        if n is None:
            n = _lane_bind.n_of(sample_spec(plugin, arg_spec), kw)
        _check_run_shapes(plugin, arg_spec, kw, n)
        parts = p.partitions_for(n)
        result = _run_structure(p, plugin, arg_spec, kw, n, parts, p.policy)
        if adapted and _is_single_call(adapted):
            return _single_returned(plugin, arg_spec, out_names, supplied, result)
        result = _restore_layouts(arg_spec, adapted, result)
        result = _write_back_outputs(out_names, supplied, adapted, result)
        return _lane_bind.shape_result(plugin, result, out_names, lane_dims, n,
                                       _declared_widths(plugin))


def _run_structure(p: Plan, plugin, arg_spec, kw, n: int, parts, policy):
    """Run the adapted ``kw`` under the plan's structure; the raw result."""
    if p.structure is eexec.DeviceKernel:
        with _torch_stream_context(kw):
            return _run_device_plan(plugin, arg_spec, kw, n, parts,
                                    keep_device=_device_inputs(arg_spec, kw), policy=policy)
    if p.structure is eexec.HostTeam:
        return _run_host_plan(plugin, arg_spec, kw, n, parts, policy=policy)
    if p.structure is eexec.RankPartition:
        return _run_rank_plan(p, plugin, arg_spec, kw, n, parts, policy)
    # DeviceGroup: check_placement already refused it at plan() time for
    # every access class, so .run() can never reach here -- defensive only.
    raise ValueError(
        f"eagle.plan: execution structure {p.structure!r} is not runnable in "
        "this build (NCCL not implemented)"
    )

# AutoPlan builds plans with plan() above, so its module loads last.
from ._plan_deploy import AutoPlan, auto  # noqa: E402
