# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Edge derivation for :class:`eagle.graph.Graph`: the Python side extracts
what each node touches (base address, shape, byte strides, item size) from
the USER's arrays, numpy or cupy alike; the derivation itself is C++
(``eagle/graph/Edges.h``, exposed as ``eagle._core.derive_edges``).

Edges are never declared. Two planes overlap when they share a byte: exact
for strided views under a work cap (so column views of one matrix do not
overlap), conservative past it -- a doubt reports an overlap, which can only
add an edge or refuse a Graph, never miss a conflict.

A node here is anything exposing ``label`` (str), ``reads`` and ``writes``
(sequences of ``(plane_name, array, role)``)."""

from __future__ import annotations

from typing import NamedTuple

from . import _core

__all__ = ["Edge", "RAW", "WAR", "WAW", "OVERLAP_MAX_WORK", "SHARES_MEMORY_MAX_WORK", "arrays_overlap",
           "check_conflicts", "derive_edges", "plane_ref"]

RAW = "RAW"
WAR = "WAR"
WAW = "WAW"

#: The work cap of the exact overlap search (search steps); past it the answer
#: is the conservative "overlap".
OVERLAP_MAX_WORK = _core.OVERLAP_MAX_WORK
SHARES_MEMORY_MAX_WORK = OVERLAP_MAX_WORK      # the earlier name, kept as an alias


class Edge(NamedTuple):
    """``src`` must finish before ``dst`` starts. ``kind`` is the hazard:
    ``RAW`` (src writes what dst reads -- the only kind that carries
    adjoint flow), ``WAR`` (src reads what dst overwrites) or ``WAW``."""

    src: int
    dst: int
    plane: str
    kind: str

    @property
    def meaning(self) -> str:
        """The hazard in plain words: "data flows" (RAW), "must wait to
        overwrite" (WAR) or "same output" (WAW)."""
        return _core.kind_meaning(self.kind)


def _pointer(arr) -> int:
    data = getattr(arr, "data", None)
    ptr = getattr(data, "ptr", None)               # cupy
    if ptr is not None:
        return int(ptr)
    for attr in ("__cuda_array_interface__", "__array_interface__"):
        iface = getattr(arr, attr, None)
        if iface is not None:
            return int(iface["data"][0])
    raise TypeError(f"eagle.Graph: {type(arr).__name__} exposes no address "
                    "(expected a numpy or cupy array)")


def plane_ref(name: str, node: int, arr, access: int):
    """The row ``_core.derive_edges`` takes: ``(name, node, base, shape,
    byte_strides, itemsize, access)`` with ``access`` 0 read, 1 write,
    2 in place."""
    return (name, int(node), _pointer(arr), [int(s) for s in arr.shape],
            [int(s) for s in arr.strides], int(arr.itemsize), int(access))


def arrays_overlap(a, b) -> bool:
    """Do arrays ``a`` and ``b`` share a byte?"""
    if a is b:
        return True
    return _core.planes_overlap(plane_ref("a", 0, a, 0), plane_ref("b", 1, b, 0))


def _refs(nodes) -> list:
    refs = []
    for i, n in enumerate(nodes):
        written = {(name, id(arr)) for name, arr, _r in n.writes}
        read = {(name, id(arr)) for name, arr, _r in n.reads}
        for name, arr, _r in n.reads:
            if (name, id(arr)) not in written:
                refs.append(plane_ref(name, i, arr, 0))
        for name, arr, _r in n.writes:
            refs.append(plane_ref(name, i, arr, 2 if (name, id(arr)) in read else 1))
    return refs


def derive_edges(nodes, groups=()) -> list:
    """Every hazard edge between nodes, in add order. ``groups`` lists, per
    fork/join group, its branches as lists of node indices; siblings get no
    edge (their overlap is a conflict, see :func:`check_conflicts`)."""
    return [Edge(*e) for e in _core.derive_edges(_refs(nodes), [list(g) for g in groups])]


def check_conflicts(nodes, groups=()) -> None:
    """Raise ``ValueError`` when two sibling nodes of one fork/join group
    overlap and at least one writes: they may run concurrently, so one
    writing what the other touches is a race, not an ordering."""
    nodes = list(nodes)
    for a, b, plane_a, plane_b, kind in _core.sibling_conflicts(
            _refs(nodes), [list(g) for g in groups]):
        la, lb = nodes[a].label, nodes[b].label
        raise ValueError(
            f"eagle.Graph.build: {la!r} (node {a}) and {lb!r} (node {b}) "
            f"are siblings of one fork/join group but overlap ({kind}): "
            f"{la!r} plane {plane_a!r} and {lb!r} plane {plane_b!r} share "
            "memory; give each branch its own plane, or order them in one branch")
