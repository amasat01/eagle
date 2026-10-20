# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The rows of a reduction over samples, and the finish that turns them into the cell.

A kernel that stores ``all_samples(x).sum()`` (or ``.mean()``, ``.max()``,
``.min()``) writes one row per 32 samples into its plane, with a ``<name>_comp``
companion for a sum. Every eagle door that launches such a kernel owns that
scratch (made once per kernel and sample count, never per launch), launches,
and finishes the rows into the one-element cell with :func:`eagle.reduce`.
"""

from __future__ import annotations

import threading

import numpy as np

from ._graph_reduce import POLICIES, _lower_cpu, _lower_gpu, reduce

#: samples per row, fixed by the kernel
ROW = 32
#: scratch sets made so far (one per kernel, sample count, policy, backend)
made = 0
_LOCK = threading.Lock()


def comp_name(name: str) -> str:
    return f"{name}_comp"


def cells_of(plugin) -> dict:
    """``{cell plane: op}`` the kernel reduces over samples (empty for a kernel
    without any). Reads the sidecar's ``cells``."""
    found = getattr(plugin, "_stage_cells", None) or getattr(plugin, "_cells", None)
    if found is not None:
        return found
    source = getattr(plugin, "_hawk_source", None)
    meta = getattr(plugin, "_meta", None) or {}
    if source is not None:
        cells = kernel_cells(source[0])
    elif "cells" in meta:
        cells = _planes(meta["cells"])
    else:       # a raw plugin: a row plane and its companion
        names = {n for r, n in plugin.arg_spec if r == "accum_out"}
        cells = {n: "sum" for n in names if comp_name(n) in names}
    try:
        plugin._cells = cells
    except AttributeError:
        pass
    return cells


def row_planes(plugin) -> frozenset:
    """Every plane the rows occupy: each cell and its ``_comp`` companion."""
    return row_names(cells_of(plugin))


def row_names(cells) -> frozenset:
    """Every plane the rows of ``cells`` (``{name: op}``) occupy."""
    return frozenset(cells) | {comp_name(n) for n, op in cells.items() if op in ("sum", "mean")}


def hidden(plugin) -> frozenset:
    """The companion planes a door does not hand back."""
    return row_planes(plugin) - frozenset(cells_of(plugin))


class CellRows:
    """The row scratch of one kernel at one sample count, and its finish.
    ``outs`` (``{plane: (1,) array}``) names cells finished into the caller's
    own arrays instead of new ones."""

    def __init__(self, cells, n, xp, dtype, policy, outs=None):
        rows = -(-int(n) // ROW)
        self.rows = {}
        self.cell = {}
        lower = _lower_gpu if xp.__name__ == "cupy" else _lower_cpu
        self._steps = []        # plain callables: legal inside a stream capture
        for name, op in cells.items():
            dt = np.dtype(dtype)
            hi = self.rows[name] = xp.zeros(rows, dtype=dt)
            out = self.cell[name] = (outs or {}).get(name)
            if out is None:
                out = self.cell[name] = xp.zeros(1, dtype=dt)
            if op in ("sum", "mean"):
                lo = self.rows[comp_name(name)] = xp.zeros(rows, dtype=dt)
                self._steps.append(lower(reduce(hi, out, comp=lo, policy=policy), None))
            else:
                self._steps.append(lower(reduce(hi, out, op=op), None))
        #: held from the launch to the read of the cell (the rows are shared)
        self.lock = threading.RLock()

    def __enter__(self):
        self.lock.acquire()
        return self

    def __exit__(self, *exc):
        self.lock.release()

    def finish(self):
        """Fold the rows into the cells, in place; returns ``{name: (1,) cell}``."""
        for step in self._steps:
            step()
        return self.cell


def kernel_cells(kernel) -> dict:
    """``{cell plane: op}`` of a traced hawk kernel."""
    return _planes(getattr(kernel.walk, "cells", {}))


def _planes(cells) -> dict:
    """``{plane: op}`` from hawk's ``{name: {op, columns}}``: a cell ``columns``
    wide is the planes ``name_0 .. name_{columns-1}``."""
    out = {}
    for name, cell in cells.items():
        width = cell.get("columns", 1)
        out.update({name: cell["op"]} if width == 1
                   else {f"{name}_{i}": cell["op"] for i in range(width)})
    return out


def _key(n, policy, xp, dtype):
    return (int(n), policy, xp.__name__, str(np.dtype(dtype)))


def scratch(owner, n, xp, dtype, policy="compensated", cells=None):
    """The :class:`CellRows` of ``owner`` (a plugin, or any object that keeps
    attributes) at ``n`` samples, made on first use and kept; ``None`` for a
    kernel with no reduction over samples. ``cells`` defaults to the plugin's."""
    global made
    cells = cells_of(owner) if cells is None else cells
    if not cells:
        return None
    if policy not in POLICIES:
        raise ValueError(f"policy {policy!r} is not one of {sorted(POLICIES)}")
    held = owner.__dict__.setdefault("_cell_scratch", {})
    key = _key(n, policy, xp, dtype)
    with _LOCK:
        if key not in held:
            held[key] = CellRows(cells, n, xp, dtype, policy)
            made += 1
        return held[key]


def has(owner, n, xp, dtype, policy="compensated") -> bool:
    """Whether the scratch of ``owner`` at ``n`` samples already exists."""
    return _key(n, policy, xp, dtype) in owner.__dict__.get("_cell_scratch", {})
