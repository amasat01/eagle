# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What every framework bridge reads off a hawk kernel: its planes, their
shapes and dtypes, the call's float mode, and the ``Param`` promotion.

Framework-neutral and import-light (numpy and hawk only, hawk imported by the
functions that need it): the JAX bridge and the torch bridge read a kernel the
same way through this module.

Conventions, off the kernel's own declaration:

* inputs are its per-sample planes, its ``Param`` cells and its ``Terminated``
  mask, in ``arg_spec`` order; outputs are its ``Mutable`` planes, plus the mask
  itself (last) when the kernel's body finishes it;
* a plane of width ``w`` is ``(n,)`` when ``w == 1``, else ``(w, n)``
  (component-major, row-major); a ``Param`` cell is ``(1,)``;
* the float MODE (``"float32"``/``"float64"``) is the dtype of the call's first
  floating plane, selects the build, and every other floating plane must agree.
"""

from __future__ import annotations

import math
from types import MappingProxyType

import numpy as np

from .. import FINISHED_PLANE, _cell_rows

__all__ = ["FLOAT_MODES", "INPUT_ROLES", "OUTPUT_ROLES", "ModeError", "Signature", "call_mode", "promote"]

#: The build modes, by the numpy dtype that selects them.
FLOAT_MODES = MappingProxyType({"float64": np.dtype(np.float64), "float32": np.dtype(np.float32)})

#: Input roles the bridges bind (lookup = a promoted ``Param`` cell).
INPUT_ROLES = frozenset({"per_sample", "vec_in", "mat_in", "terminated", "lookup"})
#: Output roles the bridges allocate and return.
OUTPUT_ROLES = frozenset({"mutable", "out"})
#: Per-sample input roles (the sample count comes off these).
SAMPLE_ROLES = frozenset({"per_sample", "vec_in", "mat_in", "terminated"})

#: hawk IR dtype -> numpy dtype, for the non-float slots.
_FIXED_DTYPE = MappingProxyType({"bool": np.dtype(np.bool_), "i64": np.dtype(np.int64), "i32": np.dtype(np.int64),
                                 "int32": np.dtype(np.int32), "uint32": np.dtype(np.uint32)})
_FLOAT_IR = frozenset({"f64", "f32"})

#: The reserved step-count word of a kernel declared without ``steps=1``.
FUSED_STEPS = "fused_steps"

#: The by-value roles no bridge binds: a ``Param`` is promoted to a cell first
#: (:func:`promote`), and nothing else may stay a by-value scalar.
_BY_VALUE = frozenset({"uniform", "nsamples"})


def promote(kernel):
    """``kernel`` with every ``Param`` read as a one-cell plane
    (:func:`hawk.diff.param_planes`, loop bodies included), so a framework can
    trace a ``Param`` like any other array; ``kernel`` itself when it reads
    none."""
    from hawk.diff import param_planes

    if not any(role == "uniform" for role, _ in kernel.arg_spec):
        return kernel
    return param_planes(kernel)


class Signature:
    """One kernel's slots, as a bridge binds them.

    :ivar name: the kernel name.
    :ivar arg_spec: ``((role, name), ...)``, the kernel's own slot order.
    :ivar inputs: the bound input names (``arg_spec`` order).
    :ivar outputs: the returned names: every output plane, then the
        ``Terminated`` mask when :attr:`finishes` (``arg_spec`` order).
    :ivar terminated: the mask's name, or ``None``.
    :ivar finishes: whether the kernel's body finishes the mask.
    :ivar state: the ``Mutable`` planes the kernel reads before it writes them
        (carried state); they are inputs too.
    :ivar fused: whether the kernel reads the step-count word (bound to 1 by
        every bridge, not an input).
    :ivar counter: whether the kernel carries the ``finished_count`` cell (a
        result every launch zero-fills and the bridge discards).
    """

    def __init__(self, kernel, *, who="eagle.frameworks", label=None):
        self.name = label or kernel.name
        self.arg_spec = tuple(tuple(pair) for pair in kernel.arg_spec)
        types = kernel.walk.slot_types
        self.role = {name: role for role, name in self.arg_spec}
        self._ir = {name: types[(role, name)] for role, name in self.arg_spec}
        #: reductions over samples ``{cell: op}``: the kernel writes rows (and a
        #: ``<cell>_comp`` companion) that the bridge finishes into the cell
        self.cells = _cell_rows.kernel_cells(kernel)
        self.rows = _cell_rows.row_names(self.cells)
        bad = sorted(f"{n} ({r})" for r, n in self.arg_spec
                     if r in _BY_VALUE or (r not in INPUT_ROLES | OUTPUT_ROLES
                                           and n not in self.rows))
        if bad:
            raise ValueError(
                f"{who}: kernel {self.name!r} declares {', '.join(bad)}; a framework "
                "bridge serves per-sample kernels (Scalar/Vector/Matrix planes, Param, "
                "Terminated, Mutable outputs; a scattered accum_out and a reduction over "
                "samples the bridge cannot finish are refused)")
        self.counter = ("lookup", FINISHED_PLANE) in self.arg_spec
        #: a kernel declared without ``steps=1`` reads the one-cell step-count word
        #: every launch but ``until_done``'s binds to 1: never a user input
        self.fused = ("lookup", FUSED_STEPS) in self.arg_spec
        #: ``Mutable`` planes the kernel reads before it writes them (carried
        #: state): inputs as well as outputs
        self.state = tuple(n for r, n in self.arg_spec
                           if r == "mutable" and n in _prior_reads(kernel))
        self.inputs = tuple(n for r, n in self.arg_spec
                            if (r in INPUT_ROLES or n in self.state)
                            and n not in (FINISHED_PLANE, FUSED_STEPS))
        self.terminated = next((n for n in self.inputs
                                if self.role[n] == "terminated"), None)
        finish = getattr(kernel.walk, "finish", None)
        self.finishes = (self.terminated is not None and finish is not None
                         and finish[0] == self.terminated)
        outs = [n for r, n in self.arg_spec
                if (r in OUTPUT_ROLES or n in self.cells) and n not in self.rows - set(self.cells)]
        if not outs:
            raise ValueError(f"{who}: kernel {self.name!r} writes no output")
        self.outputs = tuple(outs + ([self.terminated] if self.finishes else []))

    def width(self, name) -> int:
        """Components per sample of plane ``name`` (1 for a scalar plane)."""
        return math.prod(self._ir[name].shape)

    def is_cell(self, name) -> bool:
        """Whether ``name`` is a lookup cell (a promoted ``Param``)."""
        return self.role[name] == "lookup"

    def is_float(self, name) -> bool:
        return self._ir[name].dtype in _FLOAT_IR

    def shape(self, name, n) -> tuple:
        """The array shape of slot ``name`` for ``n`` samples."""
        if self.is_cell(name) or name in self.cells:
            return (1,)
        w = self.width(name)
        return (n,) if w == 1 else (w, n)

    def dtype(self, name, mode) -> np.dtype:
        """The array dtype of slot ``name`` in build ``mode``."""
        ir = self._ir[name].dtype
        if ir in _FLOAT_IR:
            return FLOAT_MODES[mode]
        try:
            return _FIXED_DTYPE[ir]
        except KeyError:
            raise ValueError(f"kernel {self.name!r}: slot {name!r} has the "
                             f"unsupported dtype {ir!r}") from None

    def differentiable(self, name) -> bool:
        """Whether ``name`` carries a derivative (a float plane or cell)."""
        return self.is_float(name) and self.role[name] != "terminated"


def _prior_reads(kernel):
    """The ``Mutable`` names ``kernel`` reads before writing them, per step (an
    automatic kernel's fused loop carries its outputs; its single step does not)."""
    from hawk.trace.steps import default_step

    return default_step(kernel).walk.prior_reads


class ModeError(TypeError, ValueError):
    """A call's floating planes name no single float32/float64 build."""


def call_mode(who, dtypes) -> str:
    """The float mode of one call, read off ``dtypes`` (``[(name, dtype)]`` for
    its floating planes, in input order; a numpy or torch dtype): the first
    plane's. A plane that is not float32/float64, or disagrees with the first,
    is refused by name. Plain string work, so a framework can trace it."""
    mode = None
    for name, dt in dtypes:
        found = str(dt).rsplit(".", 1)[-1]
        if found not in FLOAT_MODES:
            raise ModeError(
                f"{who}: plane {name!r} is {found}, which is not supported; a kernel runs "
                "float32 or float64 planes")
        if mode is None:
            mode = found
        elif found != mode:
            raise ModeError(
                f"{who}: mixed floating dtypes in one call (plane {name!r} is {found}, the "
                f"first plane is {mode}); pass every plane in one float dtype (a call runs "
                "the float32 build or the float64 one, never a mix)")
    if mode is None:
        raise ModeError(f"{who}: no floating plane bound")
    return mode
