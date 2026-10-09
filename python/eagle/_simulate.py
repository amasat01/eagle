# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Step a model until its samples are done, in the words of the model.

:func:`simulate` is the problem-level door: hand it the kernel (or the
kernels of one step, in order), then its own arguments -- bound exactly
like a call to the kernel function, by position and/or by name -- and a
step cap, and it runs every sample until that sample finishes::

    result = eagle.simulate(oscillator,
                            omega=cp.linspace(1.0, 3.0, n), t_end=1.0, dt=1e-3,
                            x=cp.ones(n), v=cp.zeros(n), t=cp.zeros(n),
                            max_steps=10_000)
    result.x, result.done, result.steps

The kernel is the step and the stop rule (``terminated = cond``); the
door brings no solver of its own. One kernel runs through
:func:`eagle.until_done` verbatim; several run as one step in list order
over one namespace of planes (a name is one plane in every kernel that
declares it). A ``Mutable`` parameter is the model's own state, bound to
its initial value (one never read before it is written may be left out --
zero-filled instead, named in the result's ``allocated``); a ``Param``
shares one value across samples, a ``Scalar``/``Vector``/``Table`` plane
takes one per sample -- a number where the kernel allows it, an array
otherwise. ``Terminated`` is never passed: every sample starts running.
eagle's own options (``until``, ``max_steps``, ``every``, ``reorder``) are
keyword-only, after the kernel's own arguments; a kernel parameter sharing
one of those names is refused. The data decides where the model runs
(numpy on the CPU, cupy on the GPU); a sample of plain numbers comes back
head-shaped. Reaching ``max_steps`` is a result (``status ==
"max_steps"``), not an error.
"""

from __future__ import annotations

import inspect
import time

from ._compat import zip_strict
from ._until_done import (
    _RESERVED,
    _array_module,
    _check_count,
    _finish_of,
    _kernel_name,
    until_done,
)
from .roles import STATE_ROLES

__all__ = ["SimResult", "Simulation", "simulate", "simulation"]

#: The result's own fields: a state plane must not shadow one.
_RESULT_FIELDS = ("state", "finished", "done", "steps", "status", "report", "n",
                  "wall_s", "allocated")

#: The roles that hold one value (or one column) per sample.
_SAMPLE_ROLES = frozenset({"per_sample", "vec_in", "mat_in", "mutable"})

#: The declaration's family: two kernels must agree on it for one name.
_FAMILY = {"uniform": "Param", "per_sample": "per-sample", "mutable": "per-sample",
           "vec_in": "per-sample", "mat_in": "per-sample", "lookup": "table",
           "wide_in": "table", "accum_out": "output", "wide_out": "output",
           "terminated": "mask"}

#: eagle's own call-time options: keyword-only, after the kernel's own
#: arguments; a kernel parameter sharing one of these names is refused.
_DOOR_OPTIONS = frozenset({"until", "max_steps", "every", "reorder"})


def _refuse(text: str):
    raise ValueError(f"eagle.simulate: {text}")


# --- The model: kernels and plans, in order --- #
def _is_plan(obj) -> bool:
    return hasattr(obj, "plugin") and hasattr(obj, "bind")


def _check_until(until) -> None:
    from ._plan_deploy import _is_hawk_kernel

    if until is None or _is_hawk_kernel(until) or _is_plan(until):
        return
    what = f"a {type(until).__name__}"
    if isinstance(until, str):
        what = f"the string {until!r}"
    elif callable(until):
        what = "a Python function"
    _refuse(
        f"until= is the kernel that finishes your samples, not {what}. A stop "
        "rule runs on the device, once per sample, so it is a line of a kernel: "
        "terminated = t >= t_end; pass that kernel as until= (or as the last "
        "kernel of the model)")


def _model_items(model, until) -> list:
    _check_until(until)
    items = list(model) if isinstance(model, (list, tuple)) else [model]
    if until is not None:
        items.append(until)
    if not items:
        _refuse("the model is empty; pass a hawk kernel or a list of them")
    return items


def _deploy(items, scalar_type=None):
    """``(plans, kernels)``: an :class:`eagle.plan.AutoPlan` per item, hawk
    kernels built into one bundle, and each item's own kernel (``None``
    for a prebuilt plan)."""
    from .plan import auto
    from ._plan_deploy import _is_hawk_kernel

    kernels = [it for it in items if _is_hawk_kernel(it)]
    for i, it in enumerate(items):
        if not _is_hawk_kernel(it) and not _is_plan(it):
            _refuse(
                f"item {i} of the model is a {type(it).__name__}; the model is a "
                "hawk kernel, a list of them, or plans from eagle.deploy")
    if scalar_type is not None and len(kernels) != len(items):
        _refuse("scalar_type= picks the precision eagle builds hawk kernels in; a "
                "prebuilt plan already has one: pass scalar_type to eagle.deploy instead")
    built = iter(())
    if kernels:
        built = auto(kernels[0] if len(kernels) == 1 else kernels, scalar_type=scalar_type)
        built = iter((built,) if len(kernels) == 1 else built)
    plans = [next(built) if _is_hawk_kernel(it) else it for it in items]
    return plans, [it if _is_hawk_kernel(it) else None for it in items]


def _name(plan, kernel) -> str:
    return kernel.name if kernel is not None else _kernel_name(plan.plugin)


# --- Declarations across the pipeline --- #
def _declarations(plans, names) -> dict:
    """``{plane: (role, width, dtype, kernel)}`` over the whole pipeline,
    refusing one name declared two ways (family, width or dtype)."""
    seen = {}
    for plan, kname in zip_strict(plans, names):
        plugin = plan.plugin
        widths = getattr(plugin, "arg_widths", None) or {}
        dtypes = getattr(plugin, "arg_dtypes", None) or {}
        for role, nm in plugin.arg_spec:
            if role == "nsamples" or nm in _RESERVED:
                continue
            if role == "out":
                _refuse(
                    f"{kname} returns its result ({nm!r} is an 'out' sink): it is "
                    "a function of the state, not a step. Run it on the result "
                    "with eagle.deploy(...).run(...), or make the plane a "
                    "Mutable the step writes")
            width = int(widths.get(nm, 1) or 1)
            dtype = dtypes.get(nm)
            prev = seen.get(nm)
            if prev is None:
                seen[nm] = (role, width, dtype, kname)
                continue
            p_role, p_width, p_dtype, p_kname = prev
            if _FAMILY.get(p_role) != _FAMILY.get(role):
                _refuse(
                    f"{nm!r} is declared {_FAMILY.get(p_role, p_role)} in {p_kname} "
                    f"but {_FAMILY.get(role, role)} in {kname}; one name is one "
                    "plane across the kernels of a step: declare it the same way "
                    "in both")
            if p_width != width or (p_dtype and dtype and p_dtype != dtype):
                _refuse(
                    f"{nm!r} is {p_width} wide ({p_dtype}) in {p_kname} but "
                    f"{width} wide ({dtype}) in {kname}; one name is one plane "
                    "across the kernels of a step: declare it the same way in both")
            if role in STATE_ROLES and p_role not in STATE_ROLES:
                seen[nm] = (role, width, dtype, kname)
    return seen


def _guards(plans, kernels, names) -> tuple:
    """``(mask name, compacting)``, the same for every kernel of the step."""
    from ._active_set import COUNT_PLANE, MAP_PLANE

    first = None
    for plan, kernel, kname in zip_strict(plans, kernels, names):
        spec = tuple(tuple(p) for p in plan.plugin.arg_spec)
        masks = [nm for role, nm in spec if role == "terminated"]
        lookups = {nm for role, nm in spec if role == "lookup"}
        compacting = {MAP_PLANE, COUNT_PLANE} <= lookups
        guard = getattr(getattr(kernel, "kind", None), "guard", None)
        mask = getattr(guard, "mask", None) or (masks[0] if masks else None)
        if guard is not None:
            # the kernel's own declaration: a bundle may carry its artifact
            # under another kernel's guard, so the Kind decides
            compacting = bool(getattr(guard, "active_set", False))
        this = (mask, compacting, kname)
        if first is None:
            first = this
            continue
        if mask != first[0] and mask is not None and first[0] is not None:
            _refuse(
                f"{first[2]} is guarded by the mask {first[0]!r} but {kname} by "
                f"{mask!r}; the kernels of one step share one guard: build them "
                "under one Kind, as eagle's tutorial 4 does")
        if compacting != first[1]:
            a, b = (first[2], kname) if first[1] else (kname, first[2])
            _refuse(
                f"{a} reads the active set (Guard(active_set=True)) but {b} does "
                "not; the kernels of one step share one guard: build them under "
                "one Kind, as eagle's tutorial 4 does")
        if first[0] is None:
            first = (mask, first[1], first[2])
    return first[0] or "terminated", first[1]


def _finishers(plans, names) -> list:
    """The indices of the kernels that finish their own samples."""
    return [i for i, (p, nm) in enumerate(zip_strict(plans, names))
            if _finish_of(p.plugin, nm) is not None]


# --- the call shape: bind like a call to the kernel --- #
def _declared_order(plans, kernels, decl) -> list:
    """Every bindable plane name (everything but the Terminated mask and the
    door's own reserved planes) in FIRST-DECLARED order across the model:
    each raw hawk kernel's own parameter order (``kernel.planes``, which
    tracing keeps in signature order even through the automatic-kernel
    wrap), or -- an item with no raw kernel, a prebuilt plan -- its
    plugin's ``arg_spec`` order."""
    order, seen = [], set()
    for plan, kernel in zip_strict(plans, kernels):
        source = (kernel.planes if kernel is not None
                 else dict.fromkeys(nm for _role, nm in plan.plugin.arg_spec))
        for nm in source:
            if nm in seen or nm not in decl or decl[nm][0] == "terminated":
                continue
            seen.add(nm)
            order.append(nm)
    return order


def _prior_reads(kernels) -> frozenset:
    """Every plane a kernel of the model reads before it is first written --
    the Mutable planes that need an initial value rather than a zero fill."""
    prior = set()
    for k in kernels:
        if k is not None:
            prior |= set(getattr(k.walk, "prior_reads", ()) or ())
    return frozenset(prior)


def _check_option_collision(decl) -> None:
    for nm, (_role, _width, _dtype, kname) in decl.items():
        if nm in _DOOR_OPTIONS:
            _refuse(
                f"{kname} declares a parameter named {nm!r}, which is "
                f"eagle.simulate's own {nm}= option; rename the kernel's "
                f"parameter {nm!r}")


def _check_result_shadow(decl) -> None:
    for nm, (role, *_rest) in decl.items():
        if role in STATE_ROLES and nm in _RESULT_FIELDS:
            _refuse(
                f"the state plane {nm!r} would shadow the result's own field of "
                f"that name (the result's fields: {', '.join(_RESULT_FIELDS)}); "
                "rename the plane in the kernel")


def _check_old_keywords(kname, order, kwargs) -> None:
    if "state" in kwargs or "params" in kwargs:
        example = ", ".join(f"{nm}=..." for nm in order)
        _refuse(
            "state=/params= were removed: call it the way you would call the "
            f"kernel itself, e.g. eagle.simulate({kname}, {example}, max_steps=...)")


def _check_positional_needs_a_raw_kernel(kernels, order, args) -> None:
    """Refuse positional kernel arguments when any item of the model (or
    ``until=``) is a PREBUILT plan with no raw hawk kernel to read a call
    signature from: :func:`_declared_order` then falls back to that plan's
    own ``arg_spec`` order, which is role-grouped then name-sorted
    (:mod:`hawk.ir.walk`), not the kernel's authored parameter order, so
    binding positionally against it would silently misbind. Keyword
    binding is unchanged; a model built entirely of raw kernels still binds
    positionally."""
    if not args or all(k is not None for k in kernels):
        return
    example = ", ".join(f"{nm}=..." for nm in order[:len(args)])
    _refuse(
        "this model includes a prebuilt plan (from eagle.deploy/eagle.simulation) "
        "with no raw hawk kernel to read a call signature from, so positional "
        "arguments are refused -- their bind order is not necessarily the "
        "kernel's own; pass them by name"
        + (f": {example}" if example else ""))


def _check_special_kwargs(decl, kwargs) -> None:
    for nm in kwargs:
        if nm in _RESERVED:
            _refuse(
                f"{nm!r} is allocated and bound by eagle.simulate (the door owns "
                "it); drop it from the call")
        if decl.get(nm, (None,))[0] == "terminated":
            _refuse(
                f"{nm!r} is the Terminated mask; every sample starts running, so "
                "eagle.simulate drives it itself -- it is never passed by the "
                "caller")


def _signature(decl, order, prior) -> tuple:
    """``(Signature, required names)``: every name of ``order``, positional
    in that order and/or by name -- except an un-prior-read Mutable, the one
    role the caller may omit (zero-filled, named in the result's
    ``allocated``), which binds keyword-only so it may fall anywhere in the
    model's own order without upsetting the others' positions."""
    required, optional = [], []
    for nm in order:
        if decl[nm][0] == "mutable" and nm not in prior:
            optional.append(nm)
        else:
            required.append(nm)
    params = [inspect.Parameter(nm, inspect.Parameter.POSITIONAL_OR_KEYWORD)
              for nm in required]
    params += [inspect.Parameter(nm, inspect.Parameter.KEYWORD_ONLY, default=None)
               for nm in optional]
    return inspect.Signature(params), required


def _bind(kname, sig, required, prior, args, kwargs) -> dict:
    """``{name: value}``, bound exactly like a call to the kernel: positional
    in its own order and/or by name, naming the kernel and the parameter on a
    missing / unknown / duplicated argument -- the same kind of message
    Python itself gives."""
    try:
        bound = sig.bind_partial(*args, **kwargs)
    except TypeError as exc:
        _refuse(f"{kname}() {exc}")
    missing = [nm for nm in required if nm not in bound.arguments]
    if missing:
        first = missing[0]
        if first in prior:
            _refuse(
                f"{first!r} is updated from its previous value, so it needs an "
                f"initial value: pass {first}=...")
        _refuse(f"{kname}() missing a required argument: {first!r}")
    return dict(bound.arguments)


# --- state / params against the declarations --- #
def _is_number(value) -> bool:
    from ._layout import is_number

    return getattr(value, "shape", None) is None and is_number(value)


def _shape(value) -> tuple:
    return tuple(int(e) for e in getattr(value, "shape", ()))


def _head(width: int) -> tuple:
    return () if width <= 1 else (width,)


def _sample_count(decl, planes):
    """``(n, single)``: the batch size implied by the sample-shaped values,
    and whether the call is ONE sample (every such value head-shaped)."""
    from ._layout import is_single

    for nm, value in planes.items():
        role, width = decl[nm][0], decl[nm][1]
        if role not in _SAMPLE_ROLES or _is_number(value):
            continue
        shape = _shape(value)
        if not is_single(shape, (width,)):
            return (int(shape[-1]) if shape else 1), False
    return 1, True


def _check_params(decl, planes, n: int, single: bool) -> None:
    for nm, value in planes.items():
        role = decl[nm][0]
        if role == "uniform":
            if not _is_number(value) and _shape(value) != ():
                count = _shape(value)[-1] if _shape(value) else 1
                _refuse(
                    f"{nm!r} is declared Param (one value shared by all samples) "
                    f"but you passed {count} values; declare it `{nm}: Scalar` to "
                    "give each sample its own")
        elif role in ("per_sample", "vec_in", "mat_in") and not single:
            if _is_number(value) or _shape(value) == ():
                _refuse(
                    f"{nm!r} needs one value per sample (n = {n}): "
                    f"np.full(n, {value!r}), or declare it `{nm}: Param` to share "
                    "one value")


def _dtype(xp, plan, name, decl):
    from ._plan_planes import _plane_dtype

    declared = decl[name][2]
    return xp.dtype(declared) if declared else xp.dtype(_plane_dtype(plan.plugin))


def _select(plans, planes):
    """The plan each item runs as, picked by where the data lives
    (:func:`eagle.plan.residency`, over the whole step)."""
    from .plan import device_resident, residency

    where = residency({nm: v for nm, v in planes.items()
                       if not _is_number(v)
                       and (hasattr(v, "shape") or device_resident(v))},
                      door="eagle.simulate", what="model")
    out = [getattr(p, where) if hasattr(p, "select") else p for p in plans]
    return out, where == "device"


# --- Result --- #
class SimResult:
    """What one :meth:`Simulation.run` produced.
    :attr:`state` maps each state plane to its array (written in place,
    plus allocated ones, listed in :attr:`allocated`); each is also an
    attribute (``result.x``) and an item (``result["x"]``).
    :attr:`finished` is the per-sample mask, :attr:`done` whether every
    sample did, :attr:`status` ``"finished"``/``"max_steps"``,
    :attr:`steps`/:attr:`report`/:attr:`wall_s` the loop's steps, full
    :class:`eagle.RunReport` and wall time."""

    __slots__ = ("state", "finished", "done", "steps", "status", "report", "n",
                 "wall_s", "allocated")

    def __init__(self, *, state, finished, report, wall_s, allocated):
        put = object.__setattr__
        put(self, "state", dict(state))
        put(self, "finished", finished)
        put(self, "report", report)
        put(self, "done", bool(report.done))
        put(self, "steps", int(report.steps))
        put(self, "status", "finished" if report.done else "max_steps")
        put(self, "n", int(report.n))
        put(self, "wall_s", float(wall_s))
        put(self, "allocated", tuple(allocated))

    def __setattr__(self, name, value):
        raise AttributeError("a SimResult is read-only")

    def __getattr__(self, name):
        state = object.__getattribute__(self, "state")
        if name in state:
            return state[name]
        raise AttributeError(
            f"the result has no field or state plane {name!r} (state: "
            f"{sorted(state)})")

    def __getitem__(self, name):
        return self.state[name]

    def __repr__(self) -> str:
        return (f"SimResult(status={self.status!r}, n={self.n}, steps={self.steps}, "
                f"state={list(self.state)})")


# --- The prepared simulation --- #
class Simulation:
    """A bound, built simulation (see :func:`simulation`): :meth:`run` runs
    it, :meth:`reset` starts the batch over.
    :attr:`runner` is the :class:`eagle.Runner` driving the model,
    :attr:`loop`/:attr:`pipeline` its loop and device
    :class:`eagle.GraphPipeline` (``None`` on host), :attr:`plans` the
    plans per kernel, :attr:`n` the batch size, :attr:`names` the
    ``state``/``params``/``allocated`` plane names."""

    __slots__ = ("runner", "loop", "plans", "n", "names", "active", "finished",
                 "terminated", "every", "_state", "_device", "_xp")

    def __init__(self, model, args, kwargs, *, until=None, max_steps,
                 every=None, reorder=None, scalar_type=None):
        items = _model_items(model, until)
        plans, kernels = _deploy(items, scalar_type)
        names = [_name(p, k) for p, k in zip_strict(plans, kernels)]
        decl = _declarations(plans, names)
        mask, compacting = _guards(plans, kernels, names)
        kname = names[0]
        order = _declared_order(plans, kernels, decl)
        _check_positional_needs_a_raw_kernel(kernels, order, args)
        _check_old_keywords(kname, order, kwargs)
        _check_option_collision(decl)
        _check_result_shadow(decl)
        _check_special_kwargs(decl, kwargs)
        prior = _prior_reads(kernels)
        sig, required = _signature(decl, order, prior)
        planes = _bind(kname, sig, required, prior, args, kwargs)
        finishers = _finishers(plans, names)
        if not finishers:
            last = names[-1]
            _refuse(
                f"no kernel of the model finishes its samples, so nothing would "
                f"stop them; add the stop rule to a kernel (terminated = "
                f"t >= t_end in {last}, say) or pass the kernel that carries it: "
                "eagle.simulate(model, until=event, ...)")
        max_steps = _check_count("max_steps", max_steps)
        n, single = _sample_count(decl, planes)
        _check_params(decl, planes, n, single)
        selected, device = _select(plans, planes)
        xp = _array_module(device)
        if len(plans) > 1:
            _check_step(selected, names, finishers, compacting, every, reorder)

        final = {}
        for nm, value in planes.items():
            role = decl[nm][0]
            if _is_number(value) and role != "uniform":
                # one sample: a fresh 0-d array of the artifact's dtype
                value = xp.asarray(value, dtype=_dtype(xp, selected[0], nm, decl))
            final[nm] = value
        allocated = []
        for nm, (role, width, _dt, _k) in decl.items():
            if role != "mutable" or nm in final:
                continue
            if nm in prior:
                _refuse(
                    f"{nm!r} is updated from its previous value, so it needs an "
                    f"initial value: pass {nm}=...")
            shape = _head(width) if single else (_head(width) + (n,))
            final[nm] = xp.zeros(shape, dtype=_dtype(xp, selected[0], nm, decl))
            allocated.append(nm)
        final[mask] = xp.zeros(() if single else (n,), dtype=xp.bool_)
        self._state = {nm: v for nm, v in final.items() if decl[nm][0] in STATE_ROLES}
        self.terminated = final[mask]
        self.names = {"state": tuple(nm for nm in self._state if nm not in allocated),
                      "params": tuple(nm for nm in final
                                      if decl[nm][0] not in STATE_ROLES),
                      "allocated": tuple(allocated)}
        self.plans = tuple(plans)
        self._device, self._xp = device, xp

        step = plans[0] if len(plans) == 1 else selected
        self.runner = until_done(step, max_steps=max_steps, every=every,
                                 reorder=reorder, **final)
        self.loop = self.runner.loop
        self.n = self.runner.n
        self.active, self.finished = self.runner.active, self.runner.finished
        self.every = self.runner.every

    @property
    def pipeline(self):
        """The device :class:`eagle.GraphPipeline` (built on the first
        :meth:`run`); ``None`` on the host and before."""
        return self.runner.pipeline

    def __repr__(self) -> str:
        where = "device" if self._device else "host"
        return (f"Simulation({where}, n={self.n}, kernels={len(self.plans)}, "
                f"state={list(self._state)})")

    def run(self) -> SimResult:
        """Run until every sample finishes or the step cap is reached; the
        state is written in place and returned in the :class:`SimResult`.
        A second call continues from the current state."""
        t0 = time.perf_counter()
        report = self.runner.run()
        if self._device and self.runner._fast_mode is None:
            # a fast run ends on its own blocking readback; the band path can
            # still queue work after its read (a reordering set's restore)
            self._xp.cuda.Device().synchronize()
        wall = time.perf_counter() - t0
        return SimResult(state=self._state, finished=self.terminated, report=report,
                         wall_s=wall, allocated=self.names["allocated"])

    def reset(self) -> None:
        """Clear the finished mask (and the active set): the next :meth:`run`
        starts the batch over; the caller re-seeds its state."""
        self.runner.reset()


def _check_step(selected, names, finishers, compacting, every, reorder) -> None:
    """The several-kernel step's own rules, in the model's words: one step per
    launch of each kernel, and ``every``/``reorder`` only when they compact."""
    if not compacting and (every is not None or reorder is not None):
        _refuse(
            "the kernels read no active set, so every=/reorder= do not apply; "
            "build them under Kind(..., guard=Guard(active_set=True)) to compact")
    for i in finishers:
        k = _finish_of(selected[i].plugin, names[i])["steps"]
        if k != "auto" and k != 1:
            _refuse(
                f"{names[i]} advances {k} steps per launch, but each kernel of "
                "a several-kernel step takes one step per launch; build it "
                "with steps=1 (or fuse the step into one kernel)")


def simulation(model, *args, until=None, max_steps: int,
               every: int | None = None, reorder=None, scalar_type=None,
               **kwargs) -> Simulation:
    """Bind and build ``model`` once; returns the :class:`Simulation`, whose
    :meth:`~Simulation.run` runs it (again, after :meth:`~Simulation.reset`,
    without rebuilding). ``*args``/``**kwargs`` are the kernel's own, bound
    exactly like a call to it; the rest are :func:`simulate`'s own options."""
    return Simulation(model, args, kwargs, until=until, max_steps=max_steps,
                      every=every, reorder=reorder, scalar_type=scalar_type)


def simulate(model, *args, until=None, max_steps: int,
             every: int | None = None, reorder=None, scalar_type=None,
             **kwargs) -> SimResult:
    """Run ``model`` on every sample until each one finishes (or ``max_steps``
    is reached) and return the :class:`SimResult`.

    ``model`` is a hawk kernel, a list of them (one step, in list order),
    or plans from :func:`eagle.deploy`; ``until`` is the kernel that
    finishes the samples, placed last (at least one kernel must finish).
    The rest of the call is the kernel's own: ``*args``/``**kwargs`` bind
    to its parameters exactly as a call to the kernel function would
    (positionally, by name, or both), by its own annotations -- except that
    a model built from a PREBUILT plan (no raw kernel to read a call
    signature from) refuses ``*args`` outright, naming the keyword call to
    use instead; keyword binding always works. A ``Mutable`` plane is the
    state it updates (its initial value; one
    never read before it is written may be left out, zero-filled
    instead), a ``Param`` a number shared by every sample, a
    ``Scalar``/``Vector``/``Table`` plane one value per sample (a number
    where the kernel allows it, an array otherwise); ``Terminated`` is
    never passed. numpy/numbers run on the CPU, cupy on the GPU.
    ``max_steps`` is required; ``every``/``reorder`` are
    :func:`eagle.until_done`'s compaction cadence and reorder threshold,
    ``scalar_type`` the precision the kernels are built in (``"float64"``,
    the default, or ``"float32"``; see :func:`eagle.deploy`),
    keyword-only like every one of eagle's own options -- after the
    kernel's own arguments."""
    return simulation(model, *args, until=until, max_steps=max_steps,
                      every=every, reorder=reorder, scalar_type=scalar_type,
                      **kwargs).run()
