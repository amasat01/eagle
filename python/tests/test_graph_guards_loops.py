# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Negated guards (else / elif chains with a decision cell) and the loop
terms (no cap, a counted trip, a mid-body break): host and, under
``EAGLE_TEST_GPU=1``, the device, with values."""

import os

import numpy as np
import pytest

import eagle
from eagle._conditional import SkipGuard, repeat_while
from eagle.graph import opaque

GPU = os.environ.get("EAGLE_TEST_GPU") == "1"
XPS = ["numpy"] + (["cupy"] if GPU else [])
F, U = np.float64, np.uint32


def xpmod(name):
    if name == "numpy":
        return np
    import cupy
    return cupy


def cell(xp, v, dt=U):
    return xp.asarray(np.array([v], dtype=dt))


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def node(fn, reads, writes, name):
    return opaque(fn, reads=list(reads), writes=list(writes), name=name)


def finish(g, xp):
    if xp is not np:        # compile cupy's elementwise kernels before capture
        for nd in g._nodes:
            nd()
    return g.build()


@pytest.mark.parametrize("xpn", XPS)
@pytest.mark.parametrize("start,want", [(0, 1.0), (1, 10.0), (2, 100.0)])
def test_elif_chain_with_an_arm_that_writes_its_guard(xpn, start, want):
    xp = xpmod(xpn)
    x, index, one = cell(xp, 0, F), cell(xp, start), cell(xp, 1)
    arm0 = node(lambda: (xp.add(x, 1.0, out=x), xp.copyto(index, one)), [x, one], [x, index], "arm0")
    arm1 = node(lambda: xp.add(x, 10.0, out=x), [x], [x], "arm1")
    other = node(lambda: xp.add(x, 100.0, out=x), [x], [x], "other")
    g = eagle.Graph()._chain(index, [arm0, arm1], other)
    if xp is not np:
        for nd in g._nodes:
            nd()
        x[...] = 0
        index[...] = start
    g.build()
    g.launch()
    assert float(host(x)[0]) == want        # without the decision copy, arm0 would also run arm1


@pytest.mark.parametrize("xpn", XPS)
@pytest.mark.parametrize("flag", [1, 0])
def test_if_else_on_a_guard_the_then_arm_rewrites(xpn, flag):
    xp = xpmod(xpn)
    x, f, zero = cell(xp, 0, F), cell(xp, flag), cell(xp, 0)
    then = node(lambda: (xp.add(x, 1.0, out=x), xp.copyto(f, zero)), [x, zero], [x, f], "then")
    other = node(lambda: xp.add(x, 10.0, out=x), [x], [x], "other")
    g = eagle.Graph()._if_else(SkipGuard.nonzero(f), then, other)
    if xp is not np:
        for nd in g._nodes:
            nd()
        x[...] = 0
        f[...] = flag
    g.build().launch()
    assert float(host(x)[0]) == (1.0 if flag else 10.0)


@pytest.mark.parametrize("xpn", XPS)
def test_uncapped_loop(xpn):
    xp = xpmod(xpn)
    n, x, one = cell(xp, 7), cell(xp, 0, F), cell(xp, 1)
    loop = repeat_while((node(lambda: xp.add(x, 1.0, out=x), [x], [x], "inc"),
                         node(lambda: xp.subtract(n, one, out=n), [n, one], [n], "dec")),
                        SkipGuard.nonzero(n))
    g = eagle.Graph().add_node(loop)
    if xp is not np:
        loop()
        x[...] = 0
        n[...] = 7
    g.build().launch()
    assert float(host(x)[0]) == 7 and loop.iterations() == 7


@pytest.mark.parametrize("xpn", XPS)
@pytest.mark.parametrize("trips", [5, 0])
def test_counted_loop_reads_its_index(xpn, trips):
    xp = xpmod(xpn)
    acc, limit = cell(xp, 0, F), cell(xp, trips)
    holder = []
    add = node(lambda: xp.add(acc, holder[0].iteration_index, out=acc), [acc], [acc], "add_index")
    loop = repeat_while(add, None, 20, limit=limit)
    holder.append(loop)
    g = eagle.Graph().add_node(loop)
    if xp is not np:
        add()
        acc[...] = 0
    g.build().launch()
    assert float(host(acc)[0]) == sum(range(trips)) and loop.iterations() == trips


@pytest.mark.parametrize("xpn", XPS)
def test_loop_with_a_mid_body_break(xpn):
    xp = xpmod(xpn)
    n, two, one, brk = cell(xp, 6), cell(xp, 2), cell(xp, 1), cell(xp, 0)
    dec = node(lambda: xp.subtract(n, one, out=n), [n, one], [n], "dec")
    stop = node(lambda: xp.equal(n, two, out=brk), [n, two], [brk], "stop")
    loop = repeat_while((dec, stop), SkipGuard.nonzero(n), 20, brk=brk)
    g = eagle.Graph().add_node(loop)
    if xp is not np:
        loop()
        n[...] = 6
    g.build().launch()
    assert int(host(n)[0]) == 2 and loop.iterations() == 4
