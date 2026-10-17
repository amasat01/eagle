# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A node kind registered ``tier="host"`` runs as a host call on the GPU:
the graph is captured in segments around it and each launch runs segments
and eager calls in order. Rows: a graph without one is still exactly one
capture (counters pinned, same launch path, same bits); a segmented graph
equals its CPU twin and captures one pipeline per segment; the graph still
describes itself as one graph; two threads with two segmented graphs are
correct, and one graph used twice at once is Busy."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import eagle
from eagle import _counters
from eagle._busy import Busy
from eagle.graph import Node, register_node_kind
from _reverse_fixtures import plans

N = 64


def _double_after_a_read(node, ctx):
    """y2 = 2 y, after a host read of y: a capture would fail on the read."""
    src, dst, gate = node.payload

    def run():
        float(src.sum())                 # a host read: never capturable
        if gate is not None:
            gate()
        dst[...] = 2 * src
    return run


register_node_kind("test_gpu_eager", reads=lambda n: (("y", n.payload[0], "per_sample"),),
                   writes=lambda n: (("y2", n.payload[1], "mutable"),),
                   lower_gpu=_double_after_a_read, lower_cpu=_double_after_a_read,
                   tier="host")


def _graph(ps, xp, x, gate=None, *, eager=True):
    sq, sn, _join = ps
    y, y2, z = xp.zeros_like(x), xp.zeros_like(x), xp.zeros_like(x)
    g = eagle.Graph().add(sq.bind(x=x, a=0.75, y=y), name="square")
    if eager:
        g.add_node(Node("test_gpu_eager", (y, y2, gate)), name="double")
        g.add(sn.bind(x=y2, y=z), name="sin")
    else:
        g.add(sn.bind(x=y, y=z), name="sin")
    return g, z


def _x(xp):
    return xp.asarray(np.linspace(-1.5, 1.5, N))


def _delta(before):
    after = _counters.snapshot()
    return {k: after[k] - before[k] for k in ("captures", "instantiations", "graph_builds")}


@pytest.mark.gpu
def test_without_an_eager_node_the_graph_is_one_capture(tmp_path):
    cp = pytest.importorskip("cupy")
    ps = plans(tmp_path, "gpu")
    before = _counters.snapshot()
    g, z = _graph(ps, cp, _x(cp), eager=False)
    g.build()
    assert _delta(before) == {"captures": 1, "instantiations": 1, "graph_builds": 1}
    assert g._segments is None and g._pipe is not None
    g.launch()
    again, z2 = _graph(ps, cp, _x(cp), eager=False)
    again.build().launch()
    np.testing.assert_array_equal(z.get(), z2.get())
    np.testing.assert_allclose(z.get(), np.sin(0.75 * _x(np) ** 2), rtol=1e-12, atol=1e-15)


@pytest.mark.gpu
def test_a_segmented_graph_equals_its_cpu_twin_and_describes_one_graph(tmp_path):
    cp = pytest.importorskip("cupy")
    gpu, cpu = plans(tmp_path, "gpu"), plans(tmp_path, "cpu")
    before = _counters.snapshot()
    g, z = _graph(gpu, cp, _x(cp))
    g.build()
    assert _delta(before) == {"captures": 2, "instantiations": 2, "graph_builds": 1}
    g.launch(2)
    h, zc = _graph(cpu, np, _x(np))
    h.build().launch()
    np.testing.assert_allclose(z.get(), zc, rtol=1e-12, atol=1e-15)
    assert g.node_labels() == h.node_labels() == ["square", "double", "sin"]
    assert g.edges() == h.edges()
    assert g.schedule == ("capture", "eager_host", "capture")
    assert "double" in str(g)


@pytest.mark.gpu
def test_two_threads_two_segmented_graphs(tmp_path):
    cp = pytest.importorskip("cupy")
    ps = plans(tmp_path, "gpu")
    want = np.sin(2 * 0.75 * _x(np) ** 2)
    out, errors = [None, None], []

    def work(i):
        try:
            with cp.cuda.Device(0):
                g, z = _graph(ps, cp, _x(cp))
                g.build().launch()
                out[i] = z.get()
        except BaseException as exc:        # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors, errors
    for got in out:
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-15)


@pytest.mark.gpu
def test_one_segmented_graph_used_twice_at_once_is_busy(tmp_path):
    cp = pytest.importorskip("cupy")
    entered, release = threading.Event(), threading.Event()

    def gate():
        entered.set()
        assert release.wait(30)

    g, _z = _graph(plans(tmp_path, "gpu"), cp, _x(cp), gate)
    g.build()
    t = threading.Thread(target=g.launch)
    t.start()
    try:
        assert entered.wait(30)
        with pytest.raises(Busy):
            g.launch()
    finally:
        release.set()
        t.join(30)
    assert not t.is_alive()
    g.launch()


def _grouped(ps, xp, x):
    sq, _sn, _join = ps
    y, y2, w = xp.zeros_like(x), xp.zeros_like(x), xp.zeros_like(x)
    g = eagle.Graph().add(sq.bind(x=x, a=0.75, y=y), name="square")
    g.add_group([Node("test_gpu_eager", (y, y2, None)), sq.bind(x=y, a=1.0, y=w)],
                names=["double", "again"])
    return g, y2


@pytest.mark.gpu
def test_an_eager_node_inside_a_group_is_refused_on_the_gpu(tmp_path):
    cp = pytest.importorskip("cupy")
    g, _y2 = _grouped(plans(tmp_path, "gpu"), cp, _x(cp))
    with pytest.raises(ValueError, match="eagle.Graph.build: node 'double' runs on the host "
                                         "between captured stages on the GPU, so it cannot run "
                                         "concurrently inside add_group; place it at the top "
                                         "level"):
        g.build()


def test_the_same_grouped_graph_runs_on_the_host(tmp_path):
    g, y2 = _grouped(plans(tmp_path, "cpu"), np, _x(np))
    g.build().launch()
    np.testing.assert_allclose(y2, 2 * 0.75 * _x(np) ** 2, rtol=1e-15)
