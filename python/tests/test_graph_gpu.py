# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""eagle.Graph on the GPU: capture-once, CPU/GPU parity, conflicts on cupy,
conditional iteration counts. Needs a device (GPU 1)."""

import numpy as np
import pytest

import eagle
from eagle import _counters
from eagle.graph import opaque

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")


def test_build_captures_exactly_once_and_runs_a_chain():
    a = cp.arange(8, dtype=cp.float64)
    b = cp.zeros(8)
    c = cp.zeros(8)

    def k1():
        cp.multiply(a, 2.0, out=b)

    def k2():
        cp.add(b, 1.0, out=c)

    before = _counters.snapshot()
    g = (eagle.Graph().add_node(opaque(k1, reads=[a], writes=[b]))
         .add_node(opaque(k2, reads=[b], writes=[c]))).build()
    after = _counters.snapshot()
    assert g.backend == "gpu" and g.schedule == ("capture", "capture")
    assert after["captures"] - before["captures"] == 1
    assert after["graph_builds"] - before["graph_builds"] == 1
    g.launch()
    np.testing.assert_array_equal(cp.asnumpy(c), 2 * np.arange(8.0) + 1)
    assert [(e.src, e.dst, e.kind) for e in g.edges()] == [(0, 1, "RAW")]


def test_fork_join_parity_with_the_cpu_lowering():
    def build(xp, backend):
        x = xp.arange(8, dtype=xp.float64)
        l, r, out = xp.zeros(8), xp.zeros(8), xp.zeros(8)
        g = (eagle.Graph()
             .add_group([opaque(lambda: xp.multiply(x, 2.0, out=l), reads=[x], writes=[l]),
                         opaque(lambda: xp.add(x, 3.0, out=r), reads=[x], writes=[r])])
             .add_node(opaque(lambda: xp.add(l, r, out=out), reads=[l, r], writes=[out])))
        g.build(backend=backend).launch()
        return g, out

    gg, og = build(cp, "gpu")
    gc, oc = build(np, "cpu")
    assert sorted(gg.node_labels()) == sorted(gc.node_labels())
    np.testing.assert_allclose(cp.asnumpy(og), oc, rtol=0, atol=0)


def test_sibling_conflict_refused_on_cupy_at_build():
    s = cp.zeros(8)
    g = eagle.Graph().add_group([opaque(lambda: None, writes={"w": s}),
                                 opaque(lambda: None, writes={"v": s[2:]})])
    with pytest.raises(ValueError, match="siblings of one fork/join group"):
        g.build()


def _loop_graph(xp, backend, start, max_iters):
    from eagle import SkipGuard, repeat_while

    flag = xp.full(1, start, dtype=xp.uint32)

    def body():
        with cp.cuda.ExternalStream(eagle._stream.current()):    # cupy's work joins the capture
            xp.subtract(flag, 1, out=flag)

    rw = repeat_while(opaque(body, writes=[flag]), SkipGuard.nonzero(flag), max_iters)
    g = eagle.Graph().add_node(rw).build(backend=backend)
    return g, rw, flag


@pytest.mark.parametrize("start,cap", [(3, 10), (5, 2), (0, 4)])
def test_conditional_counts_match_the_cpu_host_arm(start, cap):
    gg, rg, fg = _loop_graph(cp, "gpu", start, cap)
    gg.launch()
    gc, rc, fc = _loop_graph(np, "cpu", start, cap)
    gc.launch()
    assert rg.iterations() == rc.iterations() == min(start, cap)
    assert int(fg.get()[0]) == int(fc[0])


def test_empty_body_repeat_while_is_refused_on_the_gpu():
    from eagle import SkipGuard, repeat_while

    flag = cp.ones(1, dtype=cp.uint32)
    rw = repeat_while(opaque(lambda: None, writes=[cp.zeros(1)]),
                      SkipGuard.nonzero(flag), 3)
    with pytest.raises(Exception, match="launched no work"):
        eagle.Graph().add_node(rw).build()


def test_a_built_gpu_graph_shows_its_capture_schedule():
    a = cp.zeros(4)
    b = cp.zeros(4)
    g = (eagle.Graph().add_node(opaque(lambda: cp.add(a, 1.0, out=b),
                                       reads={"a": a}, writes={"b": b}), name="step")
         .build())
    text = str(g)
    assert "backend: gpu" in text and "stage 0 capture" in text
    assert g.to_dot().count("capture") >= 1
