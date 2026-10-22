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

    :ivar kernel: the hawk kernel (``hawk.trace.Kernel``).
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
        self.sig = Signature(kernel, who=who, label=self.label)
        self.chain = (base.chain if base is not None else ()) + (kernel,)
        self._wrt = None if wrt is None else tuple(wrt)
        self._derived = {}
        self._names = {}
        self._units = {}
        #: Framework-owned per-level state (a JAX primitive, a torch Function...).
        self.slots = {}
        #: how a level with no base makes its per-cell twin (see :meth:`cell_twin`)
        self.cell_source = None

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
                from hawk.trace import Kernel

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
                from hawk.trace import Kernel

                derived = step_vjp(self.kernel, wrt=self.wrt)
                level = Level(Kernel(f"{self.name}_step_vjp", derived),
                              kind="vjp", base=self, who=self.who,
                              label=f"{self.label}_step_vjp")
                self._derived["step_vjp"] = level
            return level

    def cell_twin(self):
        """This level with each ``Param`` read from a table of one entry per parameter cell
        (``hawk.diff.param_planes(per_cell=True)`` and its derivatives), so one launch runs
        a batch of parameter values; ``None`` when hawk refuses the kernel (it reduces over
        samples itself, or has a ``jump=`` plan) or this level has no source for one."""
        with LOCK:
            if "twin" not in self._derived:
                from hawk.ir.nodes import HawkError

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
                from hawk.diff import tape
                from hawk.diff.event import as_plain_reads
                from hawk.diff.tape import APPLIED_STEPS, NEXT_SUFFIX
                from hawk.ir.nodes import Assign
                from hawk.trace import Kernel

                kernel = self.kernel
                step = getattr(kernel, "step", None) or getattr(kernel, "one_step", None) or kernel
                twin = tape(kernel, 1)
                single = getattr(twin, "one_step", None) or twin
                commits = [s for s in single.sinks
                           if isinstance(s, Assign) and s.name != APPLIED_STEPS]
                values = as_plain_reads(tuple(s.value for s in commits))
                function = Kernel(f"{step.name}_function",
                                  tuple(Assign(s.name + NEXT_SUFFIX, v, s.ttype)
                                        for s, v in zip(commits, values)), {})
                function.jump = getattr(step, "jump", None)
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
                derived = {n for _, n in self.stepped().sig.arg_spec}
                outputs = [o for o in self.sig.outputs
                           if o != self.sig.terminated and self.sig.is_float(o)]
                nexts = {o: f"bar_{o}_next" for o in outputs if f"bar_{o}_next" in derived}
                grads = plane_names(self.who, self.label, "vjp", self.wrt, [],
                                    {n for _, n in self.sig.arg_spec} | set(nexts.values()),
                                    derived)
                got = self._names["step_vjp"] = (nexts, grads)
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
        return plane_names(self.who, self.label, which, self.wrt, outputs,
                           {n for _, n in self.sig.arg_spec}, {n for _, n in derived.sig.arg_spec})

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


def plane_names(who, name, which, wrt, outputs, own, derived) -> dict:
    """``{plane: its plane in the derived kernel}`` for every ``wrt`` input and
    every ``outputs`` plane (``which`` = ``"jvp"``: tangents; ``"vjp"``:
    adjoints), read off the derived kernel's planes ``derived`` that the kernel
    ``name`` (planes ``own``) does not hold. A plane with no counterpart (an
    integer output) is absent; a ``wrt`` input with none is refused."""
    from hawk.diff import ADJOINT_PREFIX, TANGENT_PREFIX
    from hawk.ir.nodes import NAMESPACE_SEP

    prefix = TANGENT_PREFIX if which == "jvp" else ADJOINT_PREFIX
    fresh = set(derived) - set(own)
    wanted = list(wrt) + list(outputs)
    flat = {p: p.replace(NAMESPACE_SEP, "_") for p in wanted}

    def spell(level, p):
        return f"{prefix}{level if level > 1 else ''}_{flat[p]}"

    top = len(own) + len(fresh) + 2
    # one level for the whole derivative (hawk's rule), else per plane
    common = next((lv for lv in range(1, top) if all(spell(lv, p) in fresh for p in wrt)), None)
    out = {}
    for p in wanted:
        levels = [lv for lv in range(1, top) if spell(lv, p) in fresh]
        if common is not None and spell(common, p) in fresh:
            out[p] = spell(common, p)
        elif levels:
            out[p] = spell(levels[0], p)
        elif p in wrt:
            raise ValueError(
                f"{who}: the {which} of {name!r} declares no {prefix}-plane for "
                f"{p!r} (its new planes: {sorted(fresh)})")
    return out
