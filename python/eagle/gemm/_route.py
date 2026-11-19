"""The opt-in matmul route of a contraction kernel: built under
``contraction="gemm"``, a kernel whose unit hawk's description (``contractions``) states as
ONE matrix product the library chose to run beside the kernel runs as that GEMM, forward and adjoint. Its
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
from .plan import (
    _DENSE_GEMM_DETERMINISM,
    _DENSE_GEMM_MECHANISM_NAME,
    LoweredPlan,
    WorkspaceSpec,
    matmul_step,
)

#: ``contractions`` product flags (hawk states them): the operand or the product is read or written transposed, or accumulated
_TRANS_A, _TRANS_B, _TRANS_C, _ACCUMULATE = 1, 2, 4, 8
#: an extent that depends on the launch's lane count
_PER_LANE = -1


def attach(plugin, kernel) -> None:
    """Route ``plugin`` (built from ``kernel``) through the GEMM when hawk's description of ``kernel`` for the device states
    its unit as one matrix product on the ``gemm`` route that writes the kernel's one output; anything else stays the kernel."""
    from hawk import Kernel
    from hawk.artifact import describe
    from hawk.diff import vjp

    adjoint = Kernel(f"{kernel.name}_vjp", vjp(kernel))
    docs = describe([kernel, adjoint], targets=("cuda",))
    doc, adjoint_doc = docs[kernel.name], docs[adjoint.name]
    found = [c for c in doc.get("contractions") or () if c["unit"] == kernel.name]
    if len(found) != 1 or found[0]["route"] != "gemm" or len(found[0]["products"]["calls"]) != 1:
        return
    route = _Route(doc["arg_spec"], found[0]["products"], plugin.scalar_type or "float64")
    if set(doc["planes_written"]) != {route.out}:
        return
    # the vjp unit states which plane is the gradient of which lookup plane (``adjoint_like``) and which cotangent seeds which
    # output (``cotangent_of``); a product it does not link is refused by name, never run half-routed
    gradient = {primal: adjoint_plane for adjoint_plane, primal in (adjoint_doc.get("adjoint_like") or {}).items()}
    seeded = {output: cotangent for cotangent, output in (adjoint_doc.get("cotangent_of") or {}).items()}
    missing = [nm for nm in route.inputs if nm not in gradient] + ([route.out] if route.out not in seeded else [])
    if missing:
        raise ValueError(f"eagle.gemm: {kernel.name!r} is a matrix product on the gemm route, but its vjp states no gradient "
                         f"plane (adjoint_like) or cotangent (cotangent_of) for {missing}")
    route.planes = {f"{route.out}_bar": seeded[route.out], **{f"{nm}_bar": gradient[nm] for nm in route.inputs}}
    plugin._gemm = route
    route.vjp = _VjpPlugin(plugin, route, adjoint_doc["arg_spec"])


class _Route:
    """The product of a unit (its operands ``a`` and ``b`` and its output, named from the call's slots) and the two plans per
    sample count (a plane counted by the launch has an extent that is the lane count over the call's fan-in)."""

    def __init__(self, arg_spec, products, dtype):
        names = [nm for _role, nm in arg_spec]
        (self.call,) = products["calls"]
        self.fanin, self.dtype = products["lane_fanin"], dtype
        self.a, self.b, self.out = (names[self.call[k]] for k in ("a_slot", "b_slot", "c_slot"))
        self.inputs = (self.a, self.b)
        self.planes = {}    # the plan's buffer of each adjoint operand and the cotangent -> its plane of the vjp unit
        self._plans, self._lock = {}, threading.Lock()

    def _extent(self, value, n):
        """``value`` as an extent for ``n`` lanes."""
        if value != _PER_LANE:
            return value
        if not self.fanin:
            raise ValueError(f"eagle.gemm: {self.out!r}'s product has an extent counted by the launch but states no lane fan-in")
        return n // self.fanin

    def lanes(self):
        """The lanes of the product (its output's elements), ``None`` when the launch counts them."""
        rows, cols = self.call["c_rows"], self.call["c_cols"]
        return None if _PER_LANE in (rows, cols) else rows * cols

    def plans(self, n):
        """``(forward, adjoint)`` for ``n`` lanes, built once per count.

        Forward is the call as stated: ``C = op(A) @ op(B)``, ``C`` stored transposed when flagged. The adjoint follows from it
        (``dA = dC @ opB^T``, ``dB = opA^T @ dC``, each stored in its operand's own layout), accumulating into the gradients."""
        with self._lock:
            held = self._plans.get(n)
            if held is None:
                held = self._plans[n] = (self._forward(n), self._adjoint(n))
        return held

    def _shapes(self, n):
        c = self.call
        return tuple((self._extent(c[f"{x}_rows"], n), self._extent(c[f"{x}_cols"], n)) for x in "abc")

    def _forward(self, n):
        a_shape, b_shape, c_shape = self._shapes(n)
        flags = self.call["flags"]
        return LoweredPlan(
            steps=(matmul_step(self.out, a=self.a, b=self.b, out=self.out, transpose_a=bool(flags & _TRANS_A),
                               transpose_b=bool(flags & _TRANS_B), transpose_out=bool(flags & _TRANS_C),
                               accumulate=bool(flags & _ACCUMULATE)),),
            mechanism=_DENSE_GEMM_MECHANISM_NAME, determinism=_DENSE_GEMM_DETERMINISM,
            inputs=(WorkspaceSpec(self.a, a_shape, self.dtype), WorkspaceSpec(self.b, b_shape, self.dtype)),
            outputs=(WorkspaceSpec(self.out, c_shape, self.dtype),))

    def _adjoint(self, n):
        a_shape, b_shape, c_shape = self._shapes(n)
        flags = self.call["flags"]
        ta, tb, tc = bool(flags & _TRANS_A), bool(flags & _TRANS_B), bool(flags & _TRANS_C)
        bar = f"{self.out}_bar"
        return LoweredPlan(
            steps=(
                matmul_step(f"{self.a}_bar", a=bar, b=self.b, out=f"{self.a}_bar", transpose_a=tc, transpose_b=not tb,
                            transpose_out=ta, accumulate=True),
                matmul_step(f"{self.b}_bar", a=self.a, b=bar, out=f"{self.b}_bar", transpose_a=not ta, transpose_b=tc,
                            transpose_out=tb, accumulate=True)),
            mechanism=_DENSE_GEMM_MECHANISM_NAME, determinism=_DENSE_GEMM_DETERMINISM,
            inputs=(WorkspaceSpec(bar, c_shape, self.dtype), WorkspaceSpec(self.a, a_shape, self.dtype),
                    WorkspaceSpec(self.b, b_shape, self.dtype)),
            outputs=(WorkspaceSpec(f"{self.a}_bar", a_shape, self.dtype), WorkspaceSpec(f"{self.b}_bar", b_shape, self.dtype)))

    def matrices(self, planes, n, adjoint):
        """The plan's buffers as views of the bound planes (no copy): an
        operand (or its gradient) in its declared matrix shape, a slice of
        a launch-counted staged plane, the per-sample planes as the lane-flat
        product."""
        plan = self.plans(n)[adjoint]
        out = {}
        for spec in plan.required:
            plane = planes[self.planes.get(spec.name, spec.name)]
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
    """The adjoint as eagle.diff reads a vjp plugin: the planes of the vjp unit hawk states (the output's cotangent per sample,
    the operands, the gradients summed into), and the route its bind runs."""

    exec_op = None
    exec_access = "cross_sample_write"

    def __init__(self, primal, route, arg_spec):
        self._gemm, self._adjoint = route, True
        self.name, self.scalar_type = f"{primal.name}_vjp", primal.scalar_type
        self.arg_spec = tuple(tuple(pair) for pair in arg_spec)


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
        lanes = planes[self._gemm.planes[f"{self._gemm.out}_bar"] if self._adjoint else self._gemm.out]
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
        lanes = route.lanes()
        if lanes is None:
            raise ValueError(f"eagle.plan.run: {p.plugin.name!r} reads a plane counted by the launch, so the "
                             f"count comes from the output: pass {route.out}= (one value per lane)")
        out = _alloc.zeros(xp, lanes, dtype=route.dtype)
    planes[route.out] = _placed(out, route.dtype, xp)
    for role, nm in p.plugin.arg_spec:
        if role in OUTPUT_ROLES and nm != route.out:
            planes[nm] = kw[nm]
    with _stream.on_device(on):
        _Bound(p, planes, on).launch()
    return out
