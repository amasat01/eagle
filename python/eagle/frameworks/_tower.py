# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The derivative tower every framework bridge stands on.

A :class:`Level` is one hawk kernel: the primal, or a derivative of the level
below it (``jvp`` or ``vjp``). Its own derivatives are derived on demand,
once, and cached; nothing is refused for depth (``jvp`` of ``jvp``, ``vjp`` of
``vjp``, any mix). A level's build carries its whole primal chain in one
bundle (hawk refuses a lone derivative; its compile cache makes the repeat
builds of the chain free).

Derived plane names are read off the derived kernel itself: hawk spells a
derived plane ``<prefix><level>_<primal plane>`` with one level for the whole
derivative, the first at which none of its names is already held. The tower
looks for exactly those spellings among the derived kernel's OWN planes that
the level below does not hold, and refuses a plane it cannot find; it never
assumes the first level.

Thread safety: :data:`LOCK` is ONE process-wide re-entrant lock. Every
derivation, name map, build, load, and every framework registration (JAX rule
tables, FFI targets, recipes) runs under it, and every per-level cache is read
inside it, so concurrent first calls of a fresh kernel derive, build and
register each level exactly once. Launches never take it.
"""

from __future__ import annotations

import threading
from pathlib import Path

from ._kernel_planes import Signature, promote

__all__ = ["LOCK", "Level", "finished_identity", "per_cell_source", "plane_names"]

#: The process-wide lock around derive / name / build / load / register.
LOCK = threading.RLock()


class Level:
    """One kernel of the tower (see the module doc).

    :ivar kernel: the hawk kernel (``hawk.Kernel``).
    :ivar label: the user's kernel name (``<kernel>_jvp``...), what messages show.
    :ivar name: its name (``<primal>``, ``<primal>_jvp``, ``<primal>_jvp_vjp``...).
    :ivar kind: ``None`` for the primal, else ``"jvp"`` / ``"vjp"``.
    :ivar base: the level this one differentiates (``None`` for the primal).
    :ivar sig: its :class:`~eagle.frameworks._kernel_planes.Signature`.
    """

    def __init__(self, kernel, *, kind=None, base=None, wrt=None, who="eagle.frameworks",
                 label=None):
        self.kernel = kernel
        self.name = kernel.name
        #: the name messages show: the user's kernel, not a promoted copy
        self.label = label or kernel.name
        self.kind = kind
        self.base = base
        self.who = who
        self.chain = (base.chain if base is not None else ()) + (kernel,)
        self.sig = Signature(kernel, primals=self.chain[:-1], who=who, label=self.label)
        self._wrt = None if wrt is None else tuple(wrt)
        self._derived = {}
        self._names = {}
        self._units = {}
        #: Framework-owned per-level state (a JAX primitive, a torch Function...).
        self.slots = {}
        #: how a level with no base makes its per-cell twin (see :meth:`cell_twin`)
        self.cell_source = None
        #: the hawk derivative this level was built from (``None`` for the primal): its gradient, cotangent and next-value relations
        self.derivative = None

    def __repr__(self) -> str:
        return f"Level({self.name!r}, kind={self.kind!r})"

    @property
    def root(self) -> Level:
        return self if self.base is None else self.base.root

    # -- derivation ---------------------------------------------------------
    @property
    def wrt(self) -> tuple:
        """The inputs this level is differentiated with respect to (hawk's
        resolution: the float planes and cells its body reads)."""
        with LOCK:
            if self._wrt is None:
                which = "step_vjp" if self.sig.state else "jvp"
                self._wrt = tuple(self._derive(which).wrt) if self._readable() else ()
            return self._wrt

    def _readable(self) -> bool:
        return any(self.sig.differentiable(n) for n in self.sig.inputs)

    def _derive(self, which):
        from hawk.diff import jvp, step_vjp, vjp

        return {"jvp": jvp, "vjp": vjp, "step_vjp": step_vjp}[which](self.kernel, wrt=self._wrt)

    def derived(self, which) -> Level:
        """The ``"jvp"`` / ``"vjp"`` level of this one, derived once."""
        with LOCK:
            level = self._derived.get(which)
            if level is None:
                from hawk import Kernel

                wrt = self.wrt
                if not wrt:
                    raise ValueError(f"{self.who}: {self.label!r} has no differentiable input")
                from hawk.diff import jvp, vjp

                fn = jvp if which == "jvp" else vjp
                derivative = fn(self.kernel, wrt=wrt)
                # the scattered gradient of a per-cell kernel cannot share a bundle with its
                # primal (hawk builds one execution axis per bundle): it builds alone, untagged
                alone = self.sig.per_cell and which == "vjp"
                # a cell's seed is a scalar in the derivative: a one-cell plane here
                level = Level(promote(Kernel(f"{self.name}_{which}",
                                             tuple(derivative) if alone else derivative)),
                              kind=which, base=self, who=self.who, label=f"{self.label}_{which}")
                if alone:
                    level.chain = (level.kernel,)
                level.derivative = derivative       # the derivative's own relations: promoting a Param rebuilds the kernel without them
                self._derived[which] = level
            return level

    def stepped(self) -> Level:
        """The vjp of ONE step of this carried-state level (``hawk.diff.step_vjp``),
        derived once. It reads each plane's value before the step and
        ``bar_<plane>_next`` (the adjoint after it), and writes ``bar_<input>``.
        It reads no mask: a bridge applies :func:`finished_identity` itself."""
        with LOCK:
            level = self._derived.get("step_vjp")
            if level is None:
                from hawk.diff import step_vjp
                from hawk import Kernel

                derived = step_vjp(self.kernel, wrt=self.wrt)
                level = Level(Kernel(f"{self.name}_step_vjp", derived),
                              kind="vjp", base=self, who=self.who,
                              label=f"{self.label}_step_vjp")
                level.derivative = derived
                self._derived["step_vjp"] = level
            return level

    def cell_twin(self):
        """This level with each ``Param`` read from a table of one entry per parameter cell
        (``hawk.diff.param_planes(per_cell=True)`` and its derivatives), so one launch runs
        a batch of parameter values; ``None`` when hawk refuses the kernel (it reduces over
        samples itself, or has a ``jump=`` plan) or this level has no source for one."""
        with LOCK:
            if "twin" not in self._derived:
                from hawk.errors import HawkError

                try:
                    if self.base is not None:
                        below = self.base.cell_twin()
                        twin = None if below is None else below.derived(self.kind)
                    else:
                        twin = None if self.cell_source is None else self.cell_source(self)
                except HawkError:
                    twin = None
                self._derived["twin"] = twin
            return self._derived["twin"]

    def step_function(self) -> Level:
        """ONE step of this carried-state level as a pure kernel, derived once:
        it reads each plane's value before the step and writes
        ``<output>_next``. It is an ordinary level, so its jvp is hawk's and
        its transpose is the vjp of one step."""
        with LOCK:
            level = self._derived.get("step_function")
            if level is None:
                from hawk.diff import step_function

                function = step_function(self.kernel)
                level = Level(promote(function), who=self.who, label=self.label)
                level._wrt = tuple(n for n in self.wrt if n in level.wrt)
                level.cell_source = lambda _me, root=self: (lambda t: t and t.step_function())(
                    root.cell_twin())
                self._derived["step_function"] = level
            return level

    def step_names(self) -> tuple:
        """``(nexts, grads)`` of :meth:`stepped`: ``{output: bar_<output>_next}``
        for every float output and ``{input: bar_<input>}`` for every :attr:`wrt`
        input, read off the derived kernel (a ``wrt`` input with none is refused)."""
        with LOCK:
            got = self._names.get("step_vjp")
            if got is None:
                derived = self.stepped().derivative
                outputs = [o for o in self.sig.outputs
                           if o != self.sig.terminated and self.sig.is_float(o)]
                nexts = {o: c for o, c in step_seeds(derived).items() if o in outputs}
                got = self._names["step_vjp"] = (nexts, gradient_names(self.who, self.label, self.wrt, derived))
            return got

    def names(self, which) -> dict:
        """``{plane of this level: its plane in the derived level}`` for every
        :attr:`wrt` input and every float output (``which`` = ``"jvp"``:
        tangents; ``"vjp"``: adjoints). A plane with no counterpart (an
        integer output) is absent."""
        with LOCK:
            got = self._names.get(which)
            if got is None:
                got = self._match(which)
                self._names[which] = got
            return got

    def _match(self, which) -> dict:
        derived = self.derived(which)
        outputs = [o for o in self.sig.outputs if o != self.sig.terminated and self.sig.is_float(o)]
        if which == "vjp":
            return reverse_names(self.who, self.label, self.wrt, outputs, derived.derivative)
        return plane_names(self.who, self.label, self.wrt, outputs, derived.derivative)

    # -- build and load ------------------------------------------------------
    def unit(self, directory, target, mode, device_arch=""):
        """This level's hawk bundle for ``target`` (``"host"``/``"cuda"``) in
        ``mode``, built once into ``directory`` (the whole primal chain)."""
        key = (str(directory), target, mode, device_arch)
        with LOCK:
            unit = self._units.get(key)
            if unit is None:
                from hawk.artifact import build_bundle

                sub = Path(directory) / self.name / (
                    f"{target}_{mode}" + (f"_{device_arch}" if device_arch else ""))
                unit = build_bundle(list(self.chain), sub, targets=(target,), mode=mode,
                                    device_arch=device_arch)
                self._units[key] = unit
            return unit

    def artifact(self, unit):
        """This level's artifact in ``unit``."""
        return next(a for a in unit.artifacts if a.name == self.name)


class StageLevel:
    """One stage of a kernel hawk builds in stages: a unit of its parent's bundle, bound from hawk's document of it
    (``describe(...)["stages"][i]["sidecar"]``) and never differentiated. It offers what a framework asks of a level
    to run it: :attr:`sig`, :attr:`name`, :attr:`label`, :attr:`slots`, :meth:`unit` and :meth:`artifact`."""

    def __init__(self, parent, entry):
        self.parent, self.who, self.name = parent, parent.who, entry["kernel"]
        self.label = f"{parent.label} ({self.name})"
        self.slots = {}
        self.sig = Signature(description=entry["sidecar"], who=parent.who, label=self.label)

    def unit(self, directory, target, mode, device_arch=""):
        """The parent's bundle: every stage is built into it."""
        return self.parent.unit(directory, target, mode, device_arch)

    def artifact(self, unit):
        """This stage's artifact in ``unit``."""
        return next(a for a in unit.artifacts if a.name == self.name)


def stage_levels(level) -> tuple:
    """The :class:`StageLevel` of each stage of ``level``'s kernel, in run order; ``()`` for a kernel that is one unit."""
    from hawk.artifact import describe

    stages = describe(list(level.chain), targets=("host",))[level.name].get("stages")
    return tuple(StageLevel(level, entry) for entry in stages or ())


def per_cell_source(declared):
    """The :attr:`Level.cell_source` of the primal level of the kernel ``declared``."""
    def make(level):
        from hawk.diff import param_planes

        twin = Level(param_planes(declared, per_cell=True), who=level.who, label=level.label)
        twin._wrt = tuple(n for n in level.wrt if n in twin.sig.inputs)
        return twin

    return make


def finished_identity(where, mask, bar_next, jt):
    """The adjoint of one step for each sample: a finished sample's step is the
    identity, so its adjoint passes through (``bar_next``); a live sample's is
    the step's ``Jᵀ bar_next`` (``jt``). ``where`` is the framework's select."""
    return where(mask, bar_next, jt)


def gradient_names(who, name, wrt, derived) -> dict:
    """``{input: its gradient plane}`` for every ``wrt`` input, as the reverse derivative ``derived`` states it
    (``gradients``); an input with none is refused."""
    gradient = {inp: plane for plane, inp in derived.gradients.items()}
    missing = [p for p in wrt if p not in gradient]
    if missing:
        raise ValueError(f"{who}: the vjp of {name!r} declares no gradient plane for {missing[0]!r} "
                         f"(its gradients: {sorted(gradient)})")
    return {p: gradient[p] for p in wrt}


def reverse_names(who, name, wrt, outputs, derived) -> dict:
    """``{plane: its plane in the reverse derivative}``: every ``wrt`` input's gradient and each of ``outputs``' cotangent, as
    the derivative states them (``gradients``, ``cotangents``); an output with no cotangent (an integer one) is absent."""
    seeds = {out: plane for plane, out in derived.cotangents.items()}
    return {**gradient_names(who, name, wrt, derived), **{o: seeds[o] for o in outputs if o in seeds}}


def step_seeds(derived) -> dict:
    """``{carried plane: the cotangent seeding its next value}`` of a step derivative: a cotangent names the next value it
    seeds (``cotangents``), and that names the plane it carries (``next_values``)."""
    return {derived.next_values[out]: plane for plane, out in derived.cotangents.items() if out in derived.next_values}


def plane_names(who, name, wrt, outputs, derived) -> dict:
    """``{plane: its tangent plane}`` for every ``wrt`` input and every ``outputs`` plane, as the forward derivative
    ``derived`` states them (``tangents``: each tangent plane with the plane it is the tangent of, at that derivative's own
    level). A plane with no tangent (an integer output) is absent; a ``wrt`` input with none is refused."""
    tangent = {primal: plane for plane, primal in derived.tangents.items()}
    missing = [p for p in wrt if p not in tangent]
    if missing:
        raise ValueError(f"{who}: the jvp of {name!r} declares no tangent plane for {missing[0]!r} "
                         f"(its tangents: {sorted(tangent)})")
    return {p: tangent[p] for p in list(wrt) + list(outputs) if p in tangent}
