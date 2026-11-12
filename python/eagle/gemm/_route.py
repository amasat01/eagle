"""The opt-in matmul route of a contraction kernel: built under
``contraction="gemm"``, a kernel hawk recognises as
ONE dense two-operand contraction runs as the lowered GEMM
(:func:`eagle.gemm.plan.lower_dense_gemm`'s plans), forward and adjoint. Its
values are within a few ulp of the kernel's own loop (another summing
order), not the same bits, so the route is never taken unless asked.

:func:`attach` marks the built plugin: ``plugin._gemm`` (the route, read by
the plan doors of a DEVICE plan; its ``vjp`` is the adjoint :func:`eagle.diff.reverse`
takes on the device). The host keeps the kernel's own loop, bit for bit: the
host product is plain loops, several times slower than the kernel. Everything else about the
plugin is unchanged, so a door the route does not cover runs the kernel."""

from __future__ import annotations

import threading

from .. import _alloc, _stream, arrays
from ..roles import OUTPUT_ROLES
from .plan import PlanUnavailable, _check_lowerable, _gemm_plan


def attach(plugin, kernel) -> None:
    """Route ``plugin`` (built from ``kernel``) through the GEMM when hawk
    recognises one dense two-operand contraction with one output whose lanes
    are its result in row-major order; anything else stays the kernel."""
    from hawk.passes.contraction import recognize

    record = recognize(kernel)
    if record is None or record.operand_class != "gemm" or len(record.outputs) != 1:
        return
    try:
        _check_lowerable(record, "forward")
    except PlanUnavailable:
        return
    major = _major_operand(kernel, record)
    if major is None:
        return
    plugin._gemm = route = _Route(record, major, plugin.scalar_type or "float64")
    route.vjp = _VjpPlugin(plugin, route)


def _major_operand(kernel, record):
    """The operand read by the MAJOR half of the lane split (only through
    ``lane // m``, ``m`` the other operand's free extent), the other one by
    its minor half (``lane - (lane // m) * m``), or ``None``: the lanes are
    then the product in row-major order. Read off hawk's own walk of the
    fold (:mod:`hawk.ir.contraction`, whose fan-in reading gives ``m``)."""
    from hawk.passes import contraction as hc
    from raptor.ir.nodes import Op, SampleIndex

    per = {}
    for sink in hc._sinks_of(kernel):
        _loop, _slot, terms = hc._fold(sink)
        for term in terms:
            for read in hc._reads_of(term):
                per.setdefault(hc._read_name(read, {}), []).append(hc._read_index(read))

    def split(name):        # (m, the ops applied to the lane index)
        ops = {node.kind for ix in per.get(name, ()) for node in hc._walk(ix)
               if isinstance(node, Op) and any(isinstance(o, SampleIndex) for o in node.operands)}
        return hc._fanin(per.get(name, ())), ops

    a, b = record.operands
    for major, minor in ((a, b), (b, a)):
        (m, ops), (m_minor, ops_minor) = split(major.name), split(minor.name)
        if ops == {"div"} and ops_minor > {"div"} and m == m_minor == _free_size(minor):
            return major.name
    return None


def _free_size(op):
    """The declared extent of ``op``'s free axis (``None`` when it is the run's)."""
    sizes = [size for label, size, _stride in op.dims if label != op.reduce_dim]
    return sizes[0] if len(sizes) == 1 and isinstance(sizes[0], int) else None


class _Route:
    """The record, which operand is the lane-major one, and the two plans per
    sample count (a plane counted by the launch, ``Staged[..., n_samples]``,
    has a run-time free extent: ``n // fan-in``)."""

    def __init__(self, record, major, dtype):
        self.record, self.dtype = record, dtype
        self.out = record.outputs[0][0]
        self.inputs = tuple(op.name for op in record.operands)
        self.lane_flat = major == record.operands[1].name
        self._plans, self._lock = {}, threading.Lock()

    def _resolved(self, n):
        """The record with every run-time extent made the one ``n`` gives."""
        from dataclasses import replace

        ops = []
        for op in self.record.operands:
            if all(isinstance(size, int) for _l, size, _s in op.dims):
                ops.append(op)
                continue
            other = next(o for o in self.record.operands if o is not op)
            free = n // _free_size(other)
            ops.append(replace(op, dims=tuple(
                (label, free if label != op.reduce_dim else size, stride)
                for label, size, stride in op.dims)))
        return replace(self.record, operands=tuple(ops))

    def plans(self, n):
        """``(forward, adjoint)`` for ``n`` lanes, built once per count."""
        with self._lock:
            held = self._plans.get(n)
            if held is None:
                rec = self._resolved(n)
                held = self._plans[n] = tuple(
                    _gemm_plan(rec, d, self.dtype, lane_flat=self.lane_flat, accumulate=(d == "adjoint"))
                    for d in ("forward", "adjoint"))
        return held

    def matrices(self, planes, n, adjoint):
        """The plan's buffers as views of the bound planes (no copy): an
        operand (or its gradient) in its declared matrix shape, a slice of
        a launch-counted staged plane, the per-sample planes as the lane-flat
        product."""
        plan = self.plans(n)[adjoint]
        out = {}
        for spec in plan.required:
            plane = planes["bar_" + spec.name[:-4] if spec.name.endswith("_bar") else spec.name]
            rows, cols = spec.shape
            view = plane.reshape((rows, _size(plane) // rows))
            out[spec.name] = view if view.shape[1] == cols else view[:, :cols]
        return plan, out


def _size(a) -> int:
    n = 1
    for d in a.shape:
        n *= int(d)
    return n


class _VjpPlugin:
    """The adjoint as eagle.diff reads a vjp plugin: the hawk vjp's planes
    (``bar_<output>`` per sample, the operands, ``bar_<operand>`` summed into),
    and the route its bind runs."""

    exec_op = None
    exec_access = "cross_sample_write"

    def __init__(self, primal, route):
        self._gemm, self._adjoint = route, True
        self.name, self.scalar_type = f"{primal.name}_vjp", primal.scalar_type
        self.arg_spec = (*(("accum_out", f"bar_{nm}") for nm in route.inputs),
                         ("per_sample", f"bar_{route.out}"),
                         *(("lookup", nm) for nm in route.inputs))


def _check_planes(p, planes):
    names = {nm for role, nm in p.plugin.arg_spec if role not in ("uniform", "nsamples")}
    missing, stray = sorted(names - set(planes)), sorted(set(planes) - names)
    if missing or stray:
        raise ValueError(f"eagle.plan.bind: {p.plugin.name!r} binds {sorted(names)}; "
                         + (f"missing {missing}" if missing else f"got stray {stray}"))


class _Bound:
    """A bound GEMM route: stands where a :class:`~eagle.plan.BoundPlan`
    does (a graph launch node, a reverse step) and launches the lowered plan
    on the current stream."""

    _origins = ()

    def __init__(self, p, planes, device=None):
        from .._plan_planes import planes_device

        _check_planes(p, planes)
        self.plan, self._gemm = p, p.plugin._gemm
        self._adjoint = int(getattr(p.plugin, "_adjoint", False))
        self._arg_spec = tuple(p.plugin.arg_spec)
        self._device = planes_device(self._arg_spec, planes, door="plan.bind", device=device)
        self._dtype = self._gemm.dtype
        lanes = planes[f"bar_{self._gemm.out}" if self._adjoint else self._gemm.out]
        self.n = _size(lanes)
        self.rebind(**planes)

    @property
    def names(self):
        return tuple(nm for _r, nm in self._arg_spec)

    def rebind(self, /, **changed):
        """Re-point the named planes (checked against the plan's shapes)."""
        self._planes = {**getattr(self, "_planes", {}), **changed}
        self._steps, self._buffers = self._gemm.matrices(self._planes, self.n, self._adjoint)
        for spec in self._steps.required:
            spec.check(self._buffers[spec.name])
        return self

    def _warm_stream(self, stream) -> None:
        """Make this device's cuBLAS handle for ``stream`` (an empty product):
        creating it allocates, which a capture region refuses."""
        from .. import _core
        from .capture import _device_ptr

        ptr = _device_ptr(next(iter(self._buffers.values())))
        double = int(self._dtype != "float32")
        _core.gemm(double, 0, 0, 0, 0, 0, ptr, 1, ptr, 1, ptr, 1, 1, 0, 0, 0, 1.0, 0.0, int(stream), self._device, False)

    def launch(self, stream=None, *, grid=None, block=None) -> None:
        handle = _stream.current(self._device) if stream is None else int(getattr(stream, "ptr", stream))
        with _stream.on_device(self._device), _stream.scope(handle, self._device):
            self._steps("device", self._buffers)
        return None


def routes(p) -> bool:
    """Whether plan ``p`` runs the GEMM route: a routed plugin on the device."""
    from .. import exec as eexec

    return getattr(p.plugin, "_gemm", None) is not None and p.structure is eexec.DeviceKernel


def bind(p, planes, device=None):
    """:meth:`eagle.plan.Plan.bind` of a routed plugin."""
    return _Bound(p, planes, device)


def _placed(value, dtype, xp):
    """``value`` on the device: a CUDA-interface array of the plan's dtype is
    kept as it is (a view may be sliced), anything else becomes an
    :class:`eagle.Array` there."""
    if hasattr(value, "__cuda_array_interface__") and str(getattr(value, "dtype", "")) == dtype:
        return value
    return arrays.asarray(value, dtype, xp)


def run(p, kw):
    """:meth:`eagle.plan.Plan.run` of a routed plugin: the inputs where the
    plan's structure runs, the output allocated when not supplied."""
    from .._plan_planes import planes_device

    route = p.plugin._gemm
    on = planes_device(p.plugin.arg_spec, kw, door="plan.run")
    if on is None:
        from ..launch import _current_device_ordinal

        on = _current_device_ordinal()
    xp = f"cuda:{on}"
    planes = {nm: _placed(kw[nm], route.dtype, xp) for nm in route.inputs}
    out = kw.get(route.out)
    if out is None:
        sizes = [_free_size(op) for op in route.record.operands]
        if None in sizes:
            raise ValueError(f"eagle.plan.run: {p.plugin.name!r} reads a plane counted by the launch, so the "
                             f"count comes from the output: pass {route.out}= (one value per lane)")
        out = _alloc.zeros(xp, sizes[0] * sizes[1], dtype=route.dtype)
    planes[route.out] = _placed(out, route.dtype, xp)
    for role, nm in p.plugin.arg_spec:
        if role in OUTPUT_ROLES and nm != route.out:
            planes[nm] = kw[nm]
    with _stream.on_device(on):
        _Bound(p, planes, on).launch()
    return out
