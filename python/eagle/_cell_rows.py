# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The rows of a reduction over samples, and the finish that turns them into the cell.

A kernel that stores ``hawk.sum(x)`` (or ``hawk.mean``, ``hawk.max``,
``hawk.min``) writes one row per 32 samples into its plane, with a ``<name>_comp``
companion for a sum. Every eagle door that launches such a kernel owns that
scratch (made once per kernel and sample count, never per launch), launches,
and finishes the rows into the one-element cell with :func:`eagle.reduce`.
"""

from __future__ import annotations

import threading


from . import _alloc
from .dtypes import dtype_name
from ._graph_reduce import _lower_cpu, _lower_gpu, check_policy, reduce

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


def companions_of(plugin) -> dict:
    """``{companion plane: its target}`` for the compensated keyed sums the
    kernel holds (sidecar ``companions``). Every eagle door makes these planes
    itself, zeroed, with the bind; the caller never passes them and they are
    not handed back."""
    declared = (getattr(plugin, "companions", None)
                or (getattr(plugin, "_meta", None) or {}).get("companions"))
    if declared:                # a stage of a split kernel is not its source kernel
        return declared
    source = getattr(plugin, "_hawk_source", None)
    return {} if source is None else source[0].walk.companions


def fold_steps(plugin, planes: dict, pairs, device=None) -> tuple:
    """``(entry, partition)`` for each owned companion of ``pairs``
    (``(target, companion)`` names, :func:`owned_pairs`): the kernel's own
    fold entry, which takes the kernel's argument block and folds one slot per
    lane, over the slots of the target in ``planes``. Launched after the
    kernel on the same structure and stream, it leaves the target alone
    holding the compensated sum and the residual in the companion.
    ``device`` is the CUDA device (``None``: the host entry)."""
    from .exec import Partition

    out = []
    for target, _comp in pairs:
        size = int(planes[target].size)
        out.append((_fold_entry(plugin, target, device), Partition(0, size, size)))
    return tuple(out)


def _fold_entry(plugin, target: str, device):
    """The entry folding ``target``'s companion: the host address the plugin
    carries, or the device function of the plugin's loaded module."""
    name = f"{plugin.kernel}__fold_{target}"
    if device is None:
        found = (getattr(plugin, "fold_entries", None) or {}).get(target)
    else:
        keep = getattr(plugin, "_keepalive", ())
        loaded = keep[-1] if isinstance(keep, tuple) and keep else None
        found = (int(loaded.function(device, name))
                 if hasattr(loaded, "function") else None)
    if found is None:
        raise ValueError(
            f"eagle: kernel {plugin.kernel!r} owns the companion of {target!r} but "
            f"its artifact carries no fold entry {name!r}; rebuild it with the "
            "hawk that wrote the companion")
    return found


def owned_pairs(plugin, bound: dict) -> tuple:
    """``(target, companion)`` of each companion a door makes itself: the
    kernel's compensated keyed sums whose companion ``bound`` (what the caller
    passed) lacks. A companion the caller passes is the caller's: never folded."""
    return tuple((target, comp) for comp, target in companions_of(plugin).items()
                 if comp not in bound)


def row_planes(plugin) -> frozenset:
    """Every plane the rows occupy: each cell and its ``_comp`` companion."""
    return row_names(cells_of(plugin))


def row_names(cells) -> frozenset:
    """Every plane the rows of ``cells`` (``{name: op}``) occupy."""
    return frozenset(cells) | {comp_name(n) for n, op in cells.items() if op in ("sum", "mean")}


def hidden(plugin) -> frozenset:
    """The companion planes a door does not hand back."""
    return row_planes(plugin) - frozenset(cells_of(plugin)) | frozenset(companions_of(plugin))


def whole_rows(part) -> tuple:
    """``(first row, rows, total rows)`` a whole partition covers. Its base must
    sit on a row (a multiple of :data:`ROW`), or a row would start before it."""
    if part.base % ROW:
        raise ValueError(
            f"eagle.plan: a kernel that reduces over samples is cut across ranks "
            f"in rows of {ROW} samples, so a partition must start on a multiple "
            f"of {ROW}; got base {part.base}")
    first = part.base // ROW
    return first, -(-(part.base + part.count) // ROW) - first, -(-part.n_samples // ROW)


def rank_samples(part, rank: int, size: int) -> tuple:
    """``(base, count, n_samples)`` rank ``rank`` of ``size`` runs of a whole
    partition: the rows are split as eagle splits samples (the first
    ``rows % size`` ranks take one more), so no row has two writers and the
    rows are the single-rank run's, bit for bit."""
    first, rows, _ = whole_rows(part)
    q, rem = divmod(rows, size)
    lo = (first + rank * q + min(rank, rem)) * ROW
    hi = min(lo + (q + (rank < rem)) * ROW, part.base + part.count)
    return lo, max(hi - lo, 0), part.n_samples


def _on_device(xp) -> bool:
    """Is ``xp`` a CUDA device token (``"cuda:N"``)?"""
    return xp.startswith("cuda")


class CellRows:
    """The row scratch of one kernel at one sample count, and its finish.
    ``outs`` (``{plane: (1,) array}``) names cells finished into the caller's
    own arrays instead of new ones."""

    def __init__(self, cells, n, xp, dtype, policy, outs=None):
        rows = -(-int(n) // ROW)
        self.rows = {}
        self.cell = {}
        lower = _lower_gpu if _on_device(xp) else _lower_cpu
        self._steps = []        # plain callables: legal inside a stream capture
        for name, op in cells.items():
            dt = dtype_name(dtype)
            hi = self.rows[name] = _alloc.zeros(xp, rows, dtype=dt)
            out = self.cell[name] = (outs or {}).get(name)
            if out is None:
                out = self.cell[name] = _alloc.zeros(xp, 1, dtype=dt)
            if op in ("sum", "mean"):
                lo = self.rows[comp_name(name)] = _alloc.zeros(xp, rows, dtype=dt)
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
    return (int(n), policy, xp, dtype_name(dtype))


def scratch(owner, n, xp, dtype, policy="compensated", cells=None):
    """The :class:`CellRows` of ``owner`` (a plugin, or any object that keeps
    attributes) at ``n`` samples, made on first use and kept; ``None`` for a
    kernel with no reduction over samples. ``cells`` defaults to the plugin's."""
    global made
    cells = cells_of(owner) if cells is None else cells
    if not cells:
        return None
    check_policy(policy, "eagle.reduce")
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
