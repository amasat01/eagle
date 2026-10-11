# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""eagle.reduce on the host: the canonical tree through eagle.Graph, bitwise
thread-count invariance, derived edges after a fork/join, refusals."""

import os
import subprocess
import sys

import numpy as np
import pytest

import eagle
from eagle.graph import opaque


def _hard(n, seed=1, dtype=np.float64):
    """Adversarial values: normals scaled by 10^[-8, 8]."""
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * 10.0 ** rng.integers(-8, 9, n)).astype(dtype)


def _canonical(x, leaf=256):
    """Independent oracle: stride-halving leaf, then pairwise levels."""
    parts = []
    for b in range(0, len(x), leaf):
        s = list(x[b:b + leaf])
        cnt, off = len(s), leaf // 2
        while off:
            for t in range(min(off, max(cnt - off, 0))):
                s[t] = s[t] + s[t + off]
            off //= 2
        parts.append(s[0])
    while len(parts) > 1:
        parts = [parts[i] + parts[i + 1] if i + 1 < len(parts) else parts[i]
                 for i in range(0, len(parts), 2)]
    return parts[0]


def _run(x, op="sum"):
    out = np.zeros(1, x.dtype)
    eagle.Graph().add_node(eagle.reduce(x, out, op=op)).build().launch()
    return out[0]


@pytest.mark.parametrize("n", [1, 2, 255, 256, 257, 1000, 4097, 70001])
@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_sum_is_the_canonical_tree_bitwise(n, dtype):
    x = _hard(n, n, dtype)
    assert _run(x).tobytes() == dtype(_canonical(x)).tobytes()


def test_sum_differs_from_a_left_fold():
    # non-vacuity: the tree is NOT numpy's/a left fold's order
    x = _hard(20011, 5)
    fold = 0.0
    for v in x:
        fold += v
    assert _run(x) != fold


@pytest.mark.parametrize("op,dtype", [("max", np.float64), ("max", np.int32),
                                      ("sum", np.int64), ("all", np.bool_)])
def test_exact_ops_match_numpy(op, dtype):
    rng = np.random.default_rng(3)
    x = (rng.integers(-1000, 1000, 5003) if dtype is not np.bool_
         else rng.random(5003) > 0.001).astype(dtype)
    want = {"max": x.max, "sum": x.sum, "all": x.all}[op]()
    assert _run(x, op) == want


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
    (lambda: eagle.reduce(np.zeros(4), np.zeros(1), op="prod"), ValueError, r"'prod'.*\['all', 'max', 'sum'\]"),
    (lambda: eagle.reduce(np.zeros(4), np.zeros(1), op="all"), TypeError, "bool"),
    (lambda: eagle.reduce(np.zeros(4, np.bool_), np.zeros(1, np.bool_)), TypeError, "float32"),
    (lambda: eagle.reduce(np.zeros((2, 2)), np.zeros(1)), ValueError, "1-D"),
    (lambda: eagle.reduce(np.zeros(0), np.zeros(1)), ValueError, "no samples"),
    (lambda: eagle.reduce(np.zeros(4), np.zeros(2)), ValueError, "exactly one element"),
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
