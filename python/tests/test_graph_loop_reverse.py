# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A ``repeat_while`` whose body is several launches (one a guard writer)
reverses like the same chain written out; a loop with no cap runs until its
guard fails and is refused by a reverse; a loop nests in a loop."""

import numpy as np
import pytest

import eagle
import eagle.diff
import eagle.plan
from eagle import SkipGuard, repeat_while

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar, UInt32  # noqa: E402
from hawk import math as m  # noqa: E402

N = 12
X0 = np.linspace(-1.1, 0.9, N)
SEED = np.linspace(1.0, 2.0, N)
L0 = np.linspace(0.2, 0.9, N)


@hawk.kernel
def lr_f1(x: Scalar, y: Mutable[Scalar]):
    y = m.sin(x) + 0.3 * x * x


@hawk.kernel
def lr_tweak(x: Scalar, y: Mutable[Scalar]):
    y = 0.9 * x + 0.1 * m.cos(x)


@hawk.kernel
def lr_merge(a: Scalar, b: Scalar, y: Mutable[Scalar]):
    y = a * (1.0 + 0.1 * m.sin(b)) + 0.05 * b


@hawk.kernel
def lr_below(s: Scalar, lim: Param, flag: Mutable[UInt32]):    # 1 while all 12 samples are below lim
    flag = m.uint32(hawk.sum(s < lim) - 11.0)


@pytest.fixture(scope="module")
def ps(tmp_path_factory):
    return eagle.plan.auto([lr_f1, lr_tweak, lr_merge, lr_below], targets=("host",),
                           cache_dir=str(tmp_path_factory.mktemp("hawk_cache")))


def looped(ps, x, l, cap, lim):
    f1, tweak, merge, below = ps
    s, tmp, flag = np.zeros(N), np.zeros(N), np.ones(1, dtype=np.uint32)
    parts = (tweak.bind(x=l, y=tmp).launch, merge.bind(a=s, b=tmp, y=s).launch,
             below.bind(s=s, lim=lim, flag=flag).launch)
    loop = repeat_while(parts, SkipGuard.nonzero(flag), cap)
    g = eagle.Graph().add(f1.bind(x=x, y=s), name="f1").add_node(loop, name="loop")
    return g, loop, s


def unrolled(ps, x, l, k):
    f1, tweak, merge, _below = ps
    s, tmp = np.zeros(N), np.zeros(N)
    g = eagle.Graph().add(f1.bind(x=x, y=s), name="f1")
    for i in range(k):
        g.add(tweak.bind(x=l, y=tmp), name=f"tw{i}").add(merge.bind(a=s, b=tmp, y=s), name=f"mg{i}")
    return g, s


def central(ps, which, cap, lim):
    def out(dx, dl):
        g, _loop, s = looped(ps, X0 + dx, L0 + dl, cap, lim)
        g.build().launch()
        return s.copy()
    h = 1e-6
    d = {"x": (lambda e: out(e, 0.0)), "l": (lambda e: out(0.0, e))}[which]
    return (d(h) - d(-h)) * SEED / (2 * h)


@pytest.mark.parametrize("cap,lim", [(6, 50.0), (40, 6.0)])
def test_multi_launch_body_equals_the_unrolled_chain(ps, cap, lim):
    g, loop, s = looped(ps, X0.copy(), L0.copy(), cap, lim)
    node = g._nodes[-1]
    assert [p.grad for p in node.payload.child.payload] == [True, True, False]   # the guard writer
    assert node.grad
    g.build(record=True).launch()
    k = loop.iterations()
    assert k == cap if cap == 6 else 16 < k < cap       # cap 40 exits early, past a 16-iteration window
    got = eagle.diff.reverse(g.record, {"loop.1.y": SEED}, wrt=["f1.x", "loop.0.x"])
    t, want_s = unrolled(ps, X0.copy(), L0.copy(), k)
    t.build(record=True).launch()
    np.testing.assert_array_equal(s, want_s)
    twin = eagle.diff.reverse(t.record, {f"mg{k - 1}.y": SEED},
                              wrt=["f1.x"] + [f"tw{i}.x" for i in range(k)])
    np.testing.assert_array_equal(got["x"], twin["f1.x"])
    np.testing.assert_array_equal(got["0.x"], twin["tw0.x"])   # l is one plane: already the sum
    np.testing.assert_allclose(got["x"], central(ps, "x", cap, lim), rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(got["0.x"], central(ps, "l", cap, lim), rtol=1e-6, atol=1e-9)


def test_a_launch_writing_only_uint32_cells_is_not_differentiated(ps):
    _f1, _tweak, _merge, below = ps
    flag = np.ones(1, dtype=np.uint32)
    g = eagle.Graph().add(below.bind(s=np.zeros(N), lim=1.0, flag=flag))
    assert g._nodes[0].grad is False


def test_a_sequence_names_its_planes_by_part(ps):
    g, _loop, _s = looped(ps, X0.copy(), L0.copy(), 3, 50.0)
    names = [n for n, _a, _r in eagle.graph.get_node_kind("sequence").writes(g._nodes[-1].payload.child)]
    assert names[:2] == ["0.y", "1.y"] and "2.flag" in names


def test_an_uncapped_loop_is_refused_by_a_reverse(ps):
    f1, _tweak, merge, below = ps
    s, flag = np.zeros(N), np.ones(1, dtype=np.uint32)
    loop = repeat_while((merge.bind(a=s, b=L0, y=s).launch, below.bind(s=s, lim=6.0, flag=flag).launch),
                        SkipGuard.nonzero(flag))
    g = eagle.Graph().add(f1.bind(x=X0.copy(), y=s), name="f1").add_node(loop, name="loop")
    g.build(record=True).launch()
    with pytest.raises(ValueError, match=r"'loop'.*no max_iters.*needs the cap"):
        eagle.diff.reverse(g.record, {"loop.0.y": SEED}, wrt=["f1.x"])
