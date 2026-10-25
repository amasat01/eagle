# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Fork/join groups, ``skippable`` and ``repeat_while`` nest in each other:
each shape equals its plain-Python reference on the host, and on a GPU
(``EAGLE_TEST_GPU=1``, under the pool lock) is bit-equal to the host run."""

import os

import numpy as np
import pytest

import eagle
from eagle._conditional import SkipGuard, repeat_while, skippable
from eagle.graph import opaque

F, U = np.float64, np.uint32
GPU = os.environ.get("EAGLE_TEST_GPU") == "1"


def arr(xp, vals, dt=F):
    return xp.asarray(np.array(vals, dtype=dt))


class Ops:
    def __init__(self, xp):
        self.xp, self.one = xp, arr(xp, [1], U)

    def node(self, fn, reads, writes, name):
        return opaque(fn, reads=list(reads), writes=list(writes), name=name)

    def inc(self, a, c):
        return self.node(lambda: self.xp.add(a, c, out=a), [a], [a], "inc")

    def mul(self, a, c):
        return self.node(lambda: self.xp.multiply(a, c, out=a), [a], [a], "mul")

    def add(self, o, a, b):
        return self.node(lambda: self.xp.add(a, b, out=o), [a, b], [o], "add")

    def dec(self, c):
        return self.node(lambda: self.xp.subtract(c, self.one, out=c), [c, self.one], [c], "dec")

    def band(self, o, c):
        return self.node(lambda: self.xp.bitwise_and(c, self.one, out=o), [c, self.one], [o], "band")

    def copy(self, d, s):
        return self.node(lambda: self.xp.copyto(d, s), [s], [d], "copy")

    def reset(self, pairs):
        def run():
            for d, s in pairs:
                self.xp.copyto(d, s)
        return self.node(run, [s for _, s in pairs], [d for d, _ in pairs], "reset")


def shape_a(xp):      # fork -> branch -> fork
    o, g = Ops(xp), eagle.Graph()
    p, q, r, s, out = (arr(xp, [0] * 4) for _ in range(5))
    p0, q0, s0 = arr(xp, [1, 2, 3, 4]), arr(xp, [5, 6, 7, 8]), arr(xp, [.5, 1.5, 2.5, 3.5])
    g.add_node(o.reset([(p, p0), (q, q0), (s, s0)]))
    inner = g._group_node([[o.inc(p, 1.0)], [o.inc(q, 2.0)]])
    g.add_group([[inner, o.add(r, p, q)], [o.mul(s, 3.0)]])
    g.add_node(o.add(out, r, s))
    return g, [p, q, r, s, out], None


def ref_a(_):
    p, q = np.array([1, 2, 3, 4.]) + 1, np.array([5, 6, 7, 8.]) + 2
    s = np.array([.5, 1.5, 2.5, 3.5]) * 3
    return [p, q, p + q, s, p + q + s]


def shape_b(xp):      # an if over three launches
    o, g = Ops(xp), eagle.Graph()
    x, x0, flag = arr(xp, [0] * 4), arr(xp, [1, 2, 3, 4]), arr(xp, [1], U)
    g.add_node(o.reset([(x, x0)]))
    g.add_node(skippable((o.inc(x, 1.0), o.mul(x, 2.0), o.inc(x, 10.0)), SkipGuard.nonzero(flag)))
    return g, [x], flag


def ref_b(f):
    x = np.array([1, 2, 3, 4.])
    return [(x + 1) * 2 + 10 if f else x]


def shape_c(xp):      # a loop in a loop
    o, g = Ops(xp), eagle.Graph()
    x, x0, n, i = arr(xp, [0] * 4), arr(xp, [1, 2, 3, 4]), arr(xp, [3], U), arr(xp, [0], U)
    n0, i0 = arr(xp, [3], U), arr(xp, [2], U)
    inner = repeat_while((o.inc(x, 1.0), o.dec(i)), SkipGuard.nonzero(i), 4)
    outer = repeat_while((o.copy(i, i0), inner, o.mul(x, 2.0), o.dec(n)), SkipGuard.nonzero(n), 5)
    g.add_node(o.reset([(x, x0), (n, n0)]))
    g.add_node(outer)
    return g, [x, n, i], None


def ref_c(_):
    x = np.array([1, 2, 3, 4.])
    for _n in range(3):
        x = (x + 2) * 2
    return [x, np.array([0], U), np.array([0], U)]


def shape_d(xp):      # a loop in an if
    o, g = Ops(xp), eagle.Graph()
    x, x0, n, n0, flag = arr(xp, [0] * 4), arr(xp, [1, 2, 3, 4]), arr(xp, [0], U), arr(xp, [3], U), arr(xp, [1], U)
    loop = repeat_while((o.mul(x, 2.0), o.dec(n)), SkipGuard.nonzero(n), 8)
    g.add_node(o.reset([(x, x0)]))
    g.add_node(skippable((o.inc(x, 5.0), o.copy(n, n0), loop), SkipGuard.nonzero(flag)))
    return g, [x], flag


def ref_d(f):
    x = np.array([1, 2, 3, 4.])
    return [(x + 5) * 8 if f else x]


def shape_e(xp):      # an if in a loop
    o, g = Ops(xp), eagle.Graph()
    x, x0, n, n0, par = arr(xp, [0] * 4), arr(xp, [1, 2, 3, 4]), arr(xp, [0], U), arr(xp, [5], U), arr(xp, [0], U)
    loop = repeat_while((o.band(par, n), skippable(o.inc(x, 10.0), SkipGuard.nonzero(par)),
                         o.inc(x, 1.0), o.dec(n)), SkipGuard.nonzero(n), 8)
    g.add_node(o.reset([(x, x0), (n, n0)]))
    g.add_node(loop)
    return g, [x, n], None


def ref_e(_):
    x, n = np.array([1, 2, 3, 4.]), 5
    while n:
        x = x + 10 * (n & 1) + 1
        n -= 1
    return [x, np.array([0], U)]


def shape_f(xp):      # a fork in a loop body
    o, g = Ops(xp), eagle.Graph()
    p, q, r = (arr(xp, [0] * 4) for _ in range(3))
    p0, q0, n, n0 = arr(xp, [1, 2, 3, 4]), arr(xp, [5, 6, 7, 8]), arr(xp, [0], U), arr(xp, [3], U)
    fork = g._group_node([[o.inc(p, 1.0)], [o.inc(q, 2.0)]])
    loop = repeat_while((fork, o.add(r, p, q), o.dec(n)), SkipGuard.nonzero(n), 6)
    g.add_node(o.reset([(p, p0), (q, q0), (n, n0)]))
    g.add_node(loop)
    return g, [p, q, r, n], None


def ref_f(_):
    p, q = np.array([1, 2, 3, 4.]) + 3, np.array([5, 6, 7, 8.]) + 6
    return [p, q, p + q, np.array([0], U)]


def shape_g(xp):      # a fork whose second branch waits for the first (one edge on a GPU)
    o, g = Ops(xp), eagle.Graph()
    x, y, z, c, s = (arr(xp, [0] * 4) for _ in range(5))
    x0, z0, c0 = arr(xp, [1, 2, 3, 4]), arr(xp, [10, 20, 30, 40]), arr(xp, [100, 200, 300, 400])
    cupy = None if xp is np else __import__("cupy")
    ev = cupy.cuda.Event(disable_timing=True) if xp is not np else None
    sig = (lambda: ev.record()) if ev else (lambda: None)
    wait = (lambda: cupy.cuda.get_current_stream().wait_event(ev)) if ev else (lambda: None)
    # the waiter's read of x is ordered by the wait, so it is not declared as a sibling's plane
    ay = o.node(lambda: xp.add(x, 1.0, out=y), [], [y], "y=x+1")
    bx = o.node(lambda: xp.multiply(x0, 2.0, out=x), [x0], [x], "x=2*x0")
    g.add_node(o.reset([(z, z0), (c, c0)]))
    g.add_group([[bx, o.node(sig, [], [], "signal"), o.inc(z, 1.0)],
                 [o.node(wait, [], [], "wait"), ay], [o.inc(c, 1.0)]])
    g.add_node(o.node(lambda: (xp.add(y, z, out=s), xp.add(s, c, out=s)), [y, z, c], [s], "sum"))
    return g, [x, y, z, c, s], None


def ref_g(_):
    x = np.array([1, 2, 3, 4.]) * 2
    y, z, c = x + 1, np.array([10, 20, 30, 40.]) + 1, np.array([100, 200, 300, 400.]) + 1
    return [x, y, z, c, y + z + c]


SHAPES = dict(a=(shape_a, ref_a, (None,)), b=(shape_b, ref_b, (1, 0)), c=(shape_c, ref_c, (None,)),
              d=(shape_d, ref_d, (1, 0)), e=(shape_e, ref_e, (None,)), f=(shape_f, ref_f, (None,)),
              g=(shape_g, ref_g, (None,)))


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def run(shape, xp, flag, n=1):
    make = SHAPES[shape][0]
    g, out, cell = make(xp)
    if flag is not None:
        cell[...] = arr(xp, [flag], U)
    if xp is not np:
        for nd in g._nodes:     # compile cupy's elementwise kernels before capture
            nd()
        g._nodes[0]()
    g.build()
    g.launch(n)
    return g, out


@pytest.mark.parametrize("shape", list(SHAPES))
def test_host_equals_reference(shape):
    _make, ref, flags = SHAPES[shape]
    for f in flags:
        g, out = run(shape, np, f)
        assert g.backend == "cpu"
        for got, want in zip(out, ref(f)):
            np.testing.assert_array_equal(got, want)


@pytest.mark.skipif(not GPU, reason="EAGLE_TEST_GPU=1 under the pool lock")
@pytest.mark.parametrize("shape", list(SHAPES))
def test_gpu_bit_equal_to_host(shape):
    import cupy
    _make, _ref, flags = SHAPES[shape]
    for f in flags:
        _, want = run(shape, np, f)
        g, out = run(shape, cupy, f)
        assert g.backend == "gpu"
        for got, w in zip(out, want):
            np.testing.assert_array_equal(host(got), w)
        g.launch(100)
        for got, w in zip(out, want):
            np.testing.assert_array_equal(host(got), w)


@pytest.mark.skipif(not GPU, reason="EAGLE_TEST_GPU=1 under the pool lock")
@pytest.mark.parametrize("shape", ["b", "d"])
def test_gpu_guard_flips_on_one_built_graph(shape):
    import cupy
    make, ref, _ = SHAPES[shape]
    g, out, cell = make(cupy)
    for nd in g._nodes:
        nd()
    g.build()
    for f in (1, 0, 1, 0, 0, 1):
        cell[...] = arr(cupy, [f], U)
        g.launch(1)
        for got, w in zip(out, ref(f)):
            np.testing.assert_array_equal(host(got), w)


def test_sibling_overlap_inside_a_nested_group_is_refused():
    o, x = Ops(np), arr(np, [0.0])
    inner = eagle.Graph()._group_node([[o.inc(x, 1.0)], [o.inc(x, 2.0)]])
    g = eagle.Graph().add_group([[inner], [o.inc(arr(np, [0.0]), 1.0)]])
    with pytest.raises(ValueError, match="siblings of one fork/join group"):
        g.build()


def test_uncapped_loop_runs_until_the_guard_fails():
    o, n, x = Ops(np), arr(np, [7], U), arr(np, [0.0])
    loop = repeat_while((o.inc(x, 1.0), o.dec(n)), SkipGuard.nonzero(n))
    eagle.Graph().add_node(loop).build().launch()
    assert x[0] == 7 and loop.iterations() == 7 and loop.max_iters is None


def test_sibling_overlap_inside_a_loop_body_is_refused():
    o, x, n = Ops(np), arr(np, [0.0]), arr(np, [2], U)
    fork = eagle.Graph()._group_node([[o.inc(x, 1.0)], [o.inc(x, 2.0)]])
    loop = repeat_while((fork, o.dec(n)), SkipGuard.nonzero(n), 3)
    with pytest.raises(ValueError, match="siblings of one fork/join group"):
        eagle.Graph().add_node(loop).build()
