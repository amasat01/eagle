# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F821  (hawk axis labels such as Table["row":ROWS] are labels, not names)

# The hawk kernels of feature_card.py. No `from __future__ import annotations` here: hawk reads
# the plane annotations when a kernel is defined.

import hawk
import hawk.math as m
from hawk import Grid, Index, Int32, Matrix, Mutable, Scalar, Table, UInt32, Vector
from hawk.ext import IndexSet, Kind


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


PREFIX = Kind("feature_prefix", lanes=IndexSet(prefix=True))
MAPPED = Kind("feature_map", lanes=IndexSet(width=32))


def _body(x):
    v = x
    for _ in range(8):
        v = m.sin(v) * 0.5 + m.cos(v * 1.3) * 0.25 + v * 0.1
    return v


@hawk.kernel
def dense(x: Scalar, y: Mutable[Scalar]):
    y = _body(x)


@PREFIX
def prefix(x: Scalar, y: Mutable[Scalar]):
    y = _body(x)


@MAPPED
def mapped(x: Scalar, y: Mutable[Scalar]):
    y = _body(x)


@hawk.kernel
def keyed_sum(key: Index, v: Scalar, rs: Mutable["node"]):
    hawk.scatter(rs)[key] += v


@hawk.kernel
def or_mask(word: Index, bit: UInt32, mask: Mutable[UInt32]):
    hawk.scatter(mask, policy="or")[word] |= bit


@hawk.kernel
def ordered_sum(key: Index, v: Scalar, rs: Mutable["node"]):
    hawk.scatter(rs, policy="ordered")[key] += v


def total(policy, samples):
    """The sum of ``samples`` lanes under ``policy`` (auto, ordered or compensated), one key."""
    @hawk.kernel
    def kernel(lane: Grid["key", "sample":samples], x: Scalar) -> Scalar:
        return hawk.sum(x, over="sample", policy=policy)
    return kernel


# The contraction of the thread box: y[sample, unit] = sum over row of w[row, unit] * f[row, sample], 64 x 64 tables.
ROWS, UNITS, SAMPLES = 64, 64, 1024        # 65536 lanes


@hawk.kernel
def contract(w: Table["row":ROWS, "neuron":UNITS], f: Table["row":ROWS, "sample":SAMPLES], y: Mutable[Scalar]):
    sample, unit = m.split_index(UNITS)
    acc = 0.0
    for r in range(ROWS):
        acc = acc + w.at(row=r, neuron=unit) * f.at(row=r, sample=sample)
    y = acc


def contract_planes(dtype, rng=None):
    """Integer-valued planes, so every sum is exact in both precisions."""
    import numpy as np

    rng = rng or np.random.default_rng(3)
    return {"w": rng.integers(-3, 4, (ROWS, UNITS)).astype(dtype), "f": rng.integers(-3, 4, (ROWS, SAMPLES)).astype(dtype)}


def contract_want(planes):
    import numpy as np

    return (planes["f"].astype(np.float64).T @ planes["w"].astype(np.float64)).reshape(-1)


# Control flow: one step function, four ways to run a runtime count of it (at most CAP).
CAP = 4000


def _step(v, k):
    return m.sin(v) * 0.5 + v * 0.5 + 0.01 + m.real(k) * 1e-3


def _step_dispatched(v, k):
    """The same step with the bias chosen by a dispatch on ``k``: hawk splits it into one unit per branch."""
    return m.sin(v) * 0.5 + v * 0.5 + 0.01 + m.dispatch(k, [0.0, 1.0, 2.0], policy="auto") * 1e-3


@hawk.kernel
def loop_guard(k: Index, n: Index, v: Scalar, y: Mutable[Scalar]):
    a = v
    for _ in range(n, max=CAP):
        a = _step(a, k)
    y = a


@hawk.kernel
def loop_noguard(k: Index, n: Index, v: Scalar, y: Mutable[Scalar]):
    a = v
    for i in range(CAP):
        a = m.select(i < n, _step(a, k), a)
    y = a


@hawk.kernel(jump=0.04)
def loop_jump(k: Index, n: Index, v: Scalar, y: Mutable[Scalar]):
    a = v
    for _ in range(n, max=CAP):
        a = _step(a, k)
        a = a - 0.001 if a > 0.9 else a + 0.001
    y = a




@hawk.kernel
def loop_segmented(k: Mutable[Index], count: Scalar, v: Mutable[Scalar], terminated: hawk.Terminated, c: Mutable[Scalar]):
    v = _step_dispatched(v, k)
    k = m.select(k > 99, k - 1, k)           # a state-dependent selector: each unit guards its commit
    cn = c + 1.0
    c = cn
    terminated = cn >= count


def loop_want(v, n, k, jump=False):
    """The reference after ``n`` steps of each lane (NumPy)."""
    import numpy as np

    a, bias = v.astype(np.float64).copy(), np.asarray([0.0, 1.0, 2.0])[k] * 1e-3
    for i in range(int(n.max())):
        live = i < n
        nxt = np.sin(a) * 0.5 + a * 0.5 + 0.01 + bias
        if jump:
            nxt = np.where(nxt > 0.9, nxt - 0.001, nxt + 0.001)
        a = np.where(live, nxt, a)
    return a
