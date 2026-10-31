# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

# The hawk kernels of feature_card.py. No `from __future__ import annotations` here: hawk reads
# the plane annotations when a kernel is defined.

import hawk
import hawk.math as m
from hawk import Index, Int32, Matrix, Mutable, Scalar, Table, Vector
from hawk.ext import IndexSet, Kind
from hawk.geometry import closest_hit


def solve_eigh(n):
    """Per lane: ``A x = b`` and the eigenvalues of ``A`` (one thread per lane for n <= 32)."""
    @hawk.kernel
    def kernel(A: Matrix[n, n], b: Vector[n], x: Mutable[Vector[n]], w: Mutable[Vector[n]], info: Mutable[Int32]):
        x, info = m.solve(A, b)
        w, V, info2 = m.eigh(A)
    return kernel


def solve_only(n):
    """Per lane ``A x = b``; past n = 32 hawk runs a team of threads on each matrix."""
    @hawk.kernel
    def kernel(A: Matrix[n, n], b: Vector[n], x: Mutable[Vector[n]], info: Mutable[Int32]):
        x, info = m.solve(A, b)
    return kernel


def hits(bvh):
    """Closest hit of one ray per lane against ``bvh``: ``[hit, t, u, v, leaf_position]``."""
    @hawk.kernel
    def kernel(o: Vector[3], d: Vector[3], tmax: Scalar, y: Mutable[Vector[5]],
               lo: Table[Scalar], hi: Table[Scalar], left: Table[Index], right: Table[Index],
               first: Table[Index], size: Table[Index], tri: Table[Scalar]):
        y = closest_hit(bvh.tables(lo, hi, left, right, first, size, tri), o, d, tmax)
    return kernel


BATCH = Kind("feature_batch", lanes=IndexSet(prefix=True))


def _body(x):
    v = x
    for _ in range(8):
        v = m.sin(v) * 0.5 + m.cos(v * 1.3) * 0.25 + v * 0.1
    return v


@hawk.kernel
def dense(x: Scalar, y: Mutable[Scalar]):
    y = _body(x)


@BATCH
def prefix(x: Scalar, y: Mutable[Scalar]):
    y = _body(x)


@hawk.kernel
def keyed_sum(key: Index, v: Scalar, rs: Mutable["node"]):
    hawk.scatter(rs)[key] += v
