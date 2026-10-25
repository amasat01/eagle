# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``Graph._rebind``: arrays swapped under a built graph give the result of a
fresh build over the new arrays, the old arrays untouched; on a GPU the
instantiated graph's nodes (root, loop body, fork branch, seed copy) are
patched with no re-capture."""

import os
import time
import warnings

import numpy as np
import pytest

import eagle
import eagle.plan
from eagle import SkipGuard, repeat_while
from eagle._graph_kinds import _copy

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Scalar  # noqa: E402
from hawk import math as m  # noqa: E402

GPU = os.environ.get("EAGLE_TEST_GPU") == "1"
XPS = ["numpy"] + (["cupy"] if GPU else [])
N = 64


@hawk.kernel
def rb_step(x: Scalar, y: Mutable[Scalar]):
    y = m.sin(x) + 0.5 * x


@hawk.kernel
def rb_add(a: Scalar, b: Scalar, y: Mutable[Scalar]):
    y = a + 2.0 * b


def xpmod(name):
    if name == "numpy":
        return np
    import cupy
    return cupy


def host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


@pytest.fixture(scope="module")
def plans(tmp_path_factory):
    return {name: eagle.plan.auto([rb_step, rb_add], targets=(("cuda",) if name == "cupy" else ("host",)),
                                  cache_dir=str(tmp_path_factory.mktemp(f"hawk_{name}")))
            for name in XPS}


def build(xpn, plans, x, t):
    """seed copy x -> s; loop (3x): s = add(s, t); fork: u = step(s) | v = step(t); w = add(u, v)."""
    xp = xpmod(xpn)
    step, add = plans[xpn]
    s, u, v, w = (xp.zeros(N) for _ in range(4))
    n, one = xp.asarray(np.array([3], np.uint32)), xp.asarray(np.array([1], np.uint32))
    dec = eagle.graph.opaque(lambda: xp.subtract(n, one, out=n), reads=[n, one], writes=[n], name="dec")
    loop = repeat_while((add.bind(a=s, b=t, y=s).launch, dec), SkipGuard.nonzero(n), 5)
    g = eagle.Graph()
    g.add_node(_copy(x, s), name="seed").add_node(loop, name="loop")
    g.add_group([[step.bind(x=s, y=u)], [step.bind(x=t, y=v)]]).add(add.bind(a=u, b=v, y=w), name="join")
    if xp is not np:
        loop()
        n[...] = 3
    g.build()
    return g, (s, u, v, w, n)


def run(g, n):
    n[...] = 3
    g.launch()


@pytest.mark.parametrize("xpn", XPS)
def test_a_rebind_equals_a_fresh_build_and_leaves_the_old_arrays(xpn, plans):
    xp = xpmod(xpn)
    rng = np.random.default_rng(3)
    x1, t1 = (xp.asarray(rng.normal(size=N)) for _ in range(2))
    x2, t2 = (xp.asarray(rng.normal(size=N)) for _ in range(2))
    keep = host(x1).copy(), host(t1).copy()
    g, (s, u, v, w, n) = build(xpn, plans, x1, t1)
    run(g, n)
    first = host(w).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # a patch must not rebuild
        g._rebind([(x1, x2), (t1, t2)])
    captures = eagle._counters.snapshot().get("graph_builds") if hasattr(eagle, "_counters") else None
    run(g, n)
    ref, (_s, _u, _v, wr, nr) = build(xpn, plans, x2, t2)
    run(ref, nr)
    np.testing.assert_array_equal(host(w), host(wr))
    assert not np.array_equal(host(w), first)
    np.testing.assert_array_equal(host(x1), keep[0])
    np.testing.assert_array_equal(host(t1), keep[1])
    del captures


def test_a_changed_shape_is_refused_by_name(plans):
    x, t = np.zeros(N), np.zeros(N)
    g, _ = build("numpy", plans, x, t)
    with pytest.raises(ValueError, match="call the builder again"):
        g._rebind([(x, np.zeros(N + 1))])


@pytest.mark.parametrize("xpn", XPS)
def test_a_swapped_guard_cell_is_patched_with_no_rebuild(xpn):
    xp = xpmod(xpn)
    x, f_on, f_off = xp.zeros(1), xp.asarray(np.array([1], np.uint32)), xp.asarray(np.array([0], np.uint32))
    bump = eagle.graph.opaque(lambda: xp.add(x, 1.0, out=x), reads=[x], writes=[x], name="bump")
    g = eagle.Graph().add_node(eagle.skippable(bump, SkipGuard.nonzero(f_on)))
    if xp is not np:
        bump()
        x[...] = 0
    g.build().launch()
    assert float(host(x)[0]) == 1.0
    builds = eagle._counters.snapshot()["graph_builds"] if hasattr(eagle._counters, "snapshot") else None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        g._rebind([(f_on, f_off)])
        g.launch()
        assert float(host(x)[0]) == 1.0                # the new guard is off
        g._rebind([(f_off, f_on)])
        g.launch()
        assert float(host(x)[0]) == 2.0
    assert builds is None or eagle._counters.snapshot()["graph_builds"] == builds


@pytest.mark.parametrize("xpn", XPS)
def test_swapped_limit_and_break_cells_of_a_loop_are_patched(xpn):
    xp = xpmod(xpn)
    cell = lambda v: xp.asarray(np.array([v], np.uint32))      # noqa: E731
    acc, lim3, lim5 = xp.zeros(1), cell(3), cell(5)
    loop = repeat_while(eagle.graph.opaque(lambda: xp.add(acc, 1.0, out=acc), reads=[acc], writes=[acc],
                                           name="inc"), None, 20, limit=lim3)
    g = eagle.Graph().add_node(loop)
    if xp is not np:
        loop()
        acc[...] = 0
    g.build()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for lim, want in ((lim3, 3), (lim5, 5)):
            acc[...] = 0
            if lim is lim5:
                g._rebind([(lim3, lim5)])
            g.launch()
            assert float(host(acc)[0]) == want


def test_the_rebind_cost_by_graph_size(plans):
    if not GPU:
        pytest.skip("EAGLE_TEST_GPU=1 under the pool lock")
    import cupy
    step, _add = plans["cupy"]
    for count in (5, 16, 256):
        a, b = cupy.ones(N), cupy.ones(N)
        outs = [cupy.zeros(N) for _ in range(count)]
        g = eagle.Graph()
        for o in outs:
            g.add(step.bind(x=a, y=o))
        g.build()
        reps = 300
        t0 = time.perf_counter()
        for i in range(reps):
            g._rebind([(a, b)] if i % 2 == 0 else [(b, a)])
        per = (time.perf_counter() - t0) / reps
        print(f"\nrebind {count} nodes: {per * 1e6:.0f} us ({per / count * 1e6:.2f} us/node)")
