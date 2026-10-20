# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""eagle.reduce on the host: the canonical tree through eagle.Graph, the
contract on exact fixtures, 2-D columns, bitwise thread-count invariance,
derived edges after a fork/join, refusals."""

import os
import subprocess
import sys

import numpy as np
import pytest

import eagle
from eagle.graph import opaque

SLOTS, TERMS = 256, 16
LEAF = SLOTS * TERMS


def _hard(n, seed=1, dtype=np.float64):
    """Adversarial values: normals scaled by 10^[-8, 8]."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * 10.0 ** rng.integers(-8, 9, n)).astype(dtype)


def _two_sum(a, b):
    s = a + b
    bb = s - a
    return s, (a - (s - bb)) + (b - bb)


def _join(p, q):
    s, e = _two_sum(p[0], q[0])
    return (s, (p[1] + q[1]) + e if np.isfinite(s) else type(s)(0))


def _canonical(x, comp=None):
    """Independent oracle of the ordered tree: per leaf, slot t folds samples
    t, t+256, ... with Neumaier steps, then the in-leaf tree, then pairwise
    levels; the value is hi + comp (hi alone when hi is not finite)."""
    t0 = x.dtype.type(0)
    c_in = comp if comp is not None else np.zeros_like(x)
    parts = []
    with np.errstate(all="ignore"):
        for b in range(0, len(x), LEAF):
            cnt = min(LEAF, len(x) - b)
            s = []
            for t in range(min(cnt, SLOTS)):
                h = x[b + t]
                c = c_in[b + t] if np.isfinite(h) else t0
                for k in range(t + SLOTS, cnt, SLOTS):
                    h, e = _two_sum(h, x[b + k])
                    c = (c + e) + c_in[b + k] if np.isfinite(h) else t0
                s.append((h, c))
            off = SLOTS // 2
            while off:
                for t in range(min(off, max(len(s) - off, 0))):
                    s[t] = _join(s[t], s[t + off])
                off //= 2
            parts.append(s[0])
        while len(parts) > 1:
            parts = [_join(parts[i], parts[i + 1]) if i + 1 < len(parts) else parts[i]
                     for i in range(0, len(parts), 2)]
        h, c = parts[0]
        return h + c if np.isfinite(h) else h


def _plain_tree(x, leaf=256):
    """The pre-compensation canonical tree (plain adds, leaf 256)."""
    parts = []
    for b in range(0, len(x), leaf):
        s = x[b:b + leaf].copy()
        cnt, off = len(s), leaf // 2
        while off:
            lim = min(off, max(cnt - off, 0))
            s[:lim] = s[:lim] + s[off:off + lim]
            off //= 2
        parts.append(s[0])
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i]
                 for i in range(0, len(parts), 2)]
    return parts[0]


def fixture(n, kappa, dtype, seed=0):
    """Mirrored pairs +-x cancelling exactly plus a 2^-10-grid residual: the
    exact sum S is known. Returns (x, S, kappa actually reached)."""
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    j = np.minimum(i, n - 1 - i)
    small = (j % 4 == 0) | (i == n - 1 - i)
    units = np.where(small, rng.integers(1, 9, n), 0)
    exact = float(units.sum()) * 2.0 ** -10
    nbig = int((~small).sum())
    scale = kappa * exact / (1.5 * max(nbig, 1))
    big = (rng.uniform(1.0, 2.0, n)[j] * scale).astype(dtype)
    x = np.where(small, (units * 2.0 ** -10).astype(dtype), np.where(i < n - 1 - i, big, -big))
    x = x.astype(dtype)
    return x, exact, float(np.abs(x.astype(np.float64)).sum()) / exact


def ulps(got, exact, dtype):
    return abs(float(got) - exact) / float(np.spacing(dtype(exact)))


def bound(n, kappa, dtype):
    """L3: within (1 + n*kappa*u) ULP of the exact sum."""
    u = 2.0 ** -24 if dtype == np.float32 else 2.0 ** -53
    return 1.0 + n * kappa * u


def _run(x, op="sum", **kw):
    out = np.zeros(1 if x.ndim == 1 else x.shape[1], x.dtype)
    eagle.Graph().add_node(eagle.reduce(x, out, op=op, **kw)).build().launch()
    return out[0] if x.ndim == 1 else out


@pytest.mark.parametrize("policy", ["ordered", "compensated"])
@pytest.mark.parametrize("n", [1, 2, 255, 257, 4095, 4096, 4097, 8193, 70001])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sum_is_the_canonical_tree_bitwise(n, dtype, policy):
    # the host runs the ordered tree under every policy
    x = _hard(n, n, dtype)
    assert _run(x, policy=policy).tobytes() == dtype(_canonical(x)).tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n", [257, 2048 * 3 + 1, 10 ** 6])
@pytest.mark.parametrize("kappa", [1, 2 ** 10, 2 ** 20])
def test_sum_meets_the_contract_on_exact_fixtures(kappa, n, dtype):
    x, exact, k = fixture(n, kappa, dtype)
    err = ulps(_run(x), exact, dtype)
    assert err <= bound(n, k, dtype)
    if n * k <= (2 ** 20 if dtype == np.float32 else 2 ** 48):
        assert err <= 1.0


def test_plain_tree_exceeds_the_contract():
    # non-vacuity: the uncompensated tree misses 1 ULP where eagle meets it
    x, exact, _ = fixture(10 ** 6, 2 ** 10, np.float32)
    assert ulps(_plain_tree(x), exact, np.float32) > 1.0
    assert ulps(_run(x), exact, np.float32) <= 1.0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("terms,want", [((np.inf,), np.inf), ((-np.inf,), -np.inf),
                                        ((np.inf, -np.inf), np.nan), ((np.nan,), np.nan)])
def test_non_finite_terms_follow_ieee(dtype, terms, want):
    x = _hard(9001, 2, dtype)
    for k, t in enumerate(terms):
        x[1000 + 3000 * k] = t
    got = _run(x)
    assert (np.isnan(got) if np.isnan(want) else got == want)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("rows,cols", [(1, 3), (257, 5), (9000, 7)])
def test_columns_equal_flat_reduces(dtype, rows, cols):
    x = _hard(rows * cols, rows + cols, dtype).reshape(rows, cols)
    got = _run(x)
    for j in range(cols):
        assert got[j].tobytes() == _run(np.ascontiguousarray(x[:, j])).tobytes()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_comp_rows_finish_with_their_companion(dtype):
    # rows of (hi, comp) pairs: the residual each pair carries is not lost
    rows, cols = 5000, 3
    v = _hard(rows * cols, 8, np.float64).reshape(rows, cols)
    hi = v.astype(dtype)
    comp = (v - hi.astype(np.float64)).astype(dtype)
    got = _run(hi, comp=comp)
    for j in range(cols):
        assert got[j].tobytes() == dtype(_canonical(hi[:, j].copy(), comp[:, j].copy())).tobytes()


def test_sum_differs_from_a_left_fold():
    # non-vacuity: the tree is NOT numpy's/a left fold's order
    x = _hard(20011, 5)
    fold = 0.0
    for v in x:
        fold += v
    assert _run(x) != fold


@pytest.mark.parametrize("op,dtype", [("max", np.float64), ("max", np.int32),
                                      ("min", np.float64), ("min", np.int32),
                                      ("sum", np.int64), ("all", np.bool_)])
def test_exact_ops_match_numpy(op, dtype):
    rng = np.random.default_rng(3)
    x = (rng.integers(-1000, 1000, 5003) if dtype is not np.bool_
         else rng.random(5003) > 0.001).astype(dtype)
    want = {"max": x.max, "min": x.min, "sum": x.sum, "all": x.all}[op]()
    assert _run(x, op) == want
    assert _run(x.reshape(-1, 1), op)[0] == want


_CHILD = """
import numpy as np, eagle, sys
rng = np.random.default_rng(9)
x = rng.standard_normal(300001) * 10.0 ** rng.integers(-8, 9, 300001)
out = np.zeros(1)
eagle.Graph().add_node(eagle.reduce(x, out)).build().launch()
sys.stdout.write(out.tobytes().hex())
"""


def test_result_is_bitwise_identical_for_every_thread_count():
    got = set()
    for threads in (1, 2, 4, 8):
        env = dict(os.environ, OMP_NUM_THREADS=str(threads))
        r = subprocess.run([sys.executable, "-I", "-c", _CHILD], env=env,
                           capture_output=True, text=True, check=True)
        got.add(r.stdout.strip())
    assert len(got) == 1


def test_edges_are_derived_after_a_fork_join():
    a, b, t = np.arange(4096.0), np.arange(4096.0) * 3, np.zeros(4096)
    pa, pb = np.zeros(4096), np.zeros(4096)
    out = np.zeros(1)
    g = (eagle.Graph()
         .add_group([[opaque(lambda: np.add(a, 1, out=pa), reads=[a], writes=[pa])],
                     [opaque(lambda: np.add(b, 1, out=pb), reads=[b], writes=[pb])]],
                    names=["pa", "pb"])
         .add_node(opaque(lambda: np.add(pa, pb, out=t), reads=[pa, pb], writes=[t]),
                   name="join")
         .add_node(eagle.reduce(t, out), name="total"))
    g.build()
    labels = g.node_labels()
    edges = {(labels[e.src], labels[e.dst], e.kind) for e in g.edges()}
    assert ("join", "total", "RAW") in edges
    g.launch()
    assert out[0].tobytes() == np.float64(_canonical(t)).tobytes()


def test_two_reduces_of_one_plane_share_no_edge():
    x = _hard(3000, 11)
    s, s2 = np.zeros(1), np.zeros(1)
    g = (eagle.Graph().add_node(eagle.reduce(x, s), name="r1")
         .add_node(eagle.reduce(x, s2, op="max"), name="r2"))
    assert g.edges() == ()                      # two readers of x, distinct outputs


@pytest.mark.parametrize("make,exc,match", [
    (lambda: eagle.reduce(np.zeros(4), np.zeros(1), op="prod"), ValueError, r"'prod'.*\['all', 'max', 'min', 'sum'\]"),
    (lambda: eagle.reduce(np.zeros(4), np.zeros(1), op="all"), TypeError, "bool"),
    (lambda: eagle.reduce(np.zeros(4, np.bool_), np.zeros(1, np.bool_)), TypeError, "float32"),
    (lambda: eagle.reduce(np.zeros((2, 2, 2)), np.zeros(1)), ValueError, "1-D sample plane or a 2-D"),
    (lambda: eagle.reduce(np.zeros((2, 2)), np.zeros(1)), ValueError, r"shape \(2,\)"),
    (lambda: eagle.reduce(np.zeros(0), np.zeros(1)), ValueError, "no samples"),
    (lambda: eagle.reduce(np.zeros(4), np.zeros(2)), ValueError, r"shape \(1,\)"),
    (lambda: eagle.reduce(np.zeros(4), np.zeros(1), policy="fast"), ValueError, "'fast'.*compensated"),
    (lambda: eagle.reduce(np.zeros(4, np.int64), np.zeros(1, np.int64), comp=np.zeros(4, np.int64)),
     TypeError, "comp= takes"),
    (lambda: eagle.reduce(np.zeros(4), np.zeros(1), comp=np.zeros(3)), ValueError, "comp shape"),
    (lambda: eagle.reduce(np.zeros(4), np.zeros(1, np.float32)), TypeError, "dtype"),
    (lambda: eagle.reduce(np.zeros(8)[::2], np.zeros(1)), ValueError, "contiguous"),
])
def test_refusals_name_the_problem(make, exc, match):
    with pytest.raises(exc, match=match):
        make()


def test_out_overlapping_x_is_refused():
    x = np.zeros(8)
    with pytest.raises(ValueError, match="overlaps"):
        eagle.reduce(x, x[3:4])
