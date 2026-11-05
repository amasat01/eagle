# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The three tiers of ``eagle.graph.opaque`` and ``eagle.graph.child`` beyond
the interop rows: one stream-handle reader for every spelling; on the CPU
every tier is the eager call, in order, and ``child`` is refused by name;
the fence chain keeps one graph exact under eight threads on both legs; a
recorded stream node's copies hold what it read; a per-step hook with
``record=True`` is refused by name; a rebound host parameter is refused; one
JAX node shared by eight graphs in eight threads stays exact."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import eagle
from eagle import graph as G
from eagle import interop
from eagle._busy import Busy
from eagle.graph import opaque


class _Proto:
    def __cuda_stream__(self):
        return (0, 77)


class _Torchish:
    cuda_stream = 78


class _Cupyish:
    ptr = 79


def test_one_stream_handle_reader_for_every_spelling():
    assert [interop._stream_handle(s) for s in (76, _Proto(), _Torchish(), _Cupyish())] == [76, 77, 78, 79]
    with pytest.raises(TypeError, match="stream= takes an int"):
        interop._stream_handle("s")


def test_cpu_runs_every_tier_eagerly_in_order():
    a, b, c, d = np.ones(4), np.zeros(4), np.zeros(4), np.zeros(4)
    order = []

    def step(src, dst, tag):
        def run():
            order.append(tag)
            np.add(src, 1, out=dst)
        return run
    g = (eagle.Graph()
         .add_node(opaque(step(a, b, "s"), reads=[a], writes=[b], stream=5), name="s")
         .add_node(opaque(step(b, c, "h"), reads=[b], writes=[c], eager=True), name="h")
         .add_node(opaque(step(c, d, "c"), reads=[c], writes=[d]), name="c").build())
    g.launch(2)
    assert order == ["s", "h", "c"] * 2 and float(d[0]) == 4.0
    assert "s [stream: cpu runs eagerly]" in str(g) and "h [host]" in str(g)


def test_child_takes_the_owner_and_is_refused_on_the_cpu():
    with pytest.raises(TypeError, match="pass the graph object; eagle must keep it alive"):
        G.child(1234, reads={}, writes={})

    class _Owner:                       # a torch.cuda.CUDAGraph's shape
        def raw_cuda_graph(self):
            return 1234

    a = np.zeros(4)
    g = eagle.Graph().add_node(G.child(_Owner(), reads=[a], writes=[a]), name="m")
    with pytest.raises(ValueError, match="child 'm' is device work; on the host call the "
                                         "model in an opaque"):
        g.build()


def _hammer(g, launches=8):
    """``launches`` threads each launch ``g`` once (retrying while Busy)."""
    errors = []

    def work():
        try:
            while True:
                try:
                    g.launch()
                    return
                except Busy:
                    time.sleep(0.001)
        except BaseException as exc:        # surfaced below
            errors.append(exc)
    threads = [threading.Thread(target=work) for _ in range(launches)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors


def _chain(xp, stream):
    w, y, acc = xp.zeros(64), xp.zeros(64), xp.zeros(64)
    g = (eagle.Graph()
         .add_node(opaque(lambda: xp.add(w, 1.0, out=w), reads=[w], writes=[w]), name="seg0")
         .add_node(opaque(lambda: xp.add(w, 1.0, out=y), reads=[w], writes=[y], stream=stream),
                   name="model")
         .add_node(opaque(lambda: xp.add(acc, y, out=acc), reads=[acc, y], writes=[acc]),
                   name="seg1").build())
    return g, acc


def test_eight_threads_one_graph_cpu():
    g, acc = _chain(np, 0)
    _hammer(g)
    assert float(acc[0]) == sum(r + 1 for r in range(1, 9))


@pytest.mark.gpu
def test_eight_threads_one_graph_gpu():
    cp = pytest.importorskip("cupy")
    g, acc = _chain(cp, cp.cuda.Stream(non_blocking=True))
    _hammer(g)
    assert float(acc[0]) == sum(r + 1 for r in range(1, 9))


@pytest.mark.gpu
def test_host_tier_two_threads_two_graphs_gpu():
    cp = pytest.importorskip("cupy")
    out = []

    def work():
        a, b = cp.full(64, 2.0), cp.zeros(64)
        g = eagle.Graph().add_node(opaque(lambda: cp.multiply(a, float(a.sum().get()), out=b),
                                          reads=[a], writes=[b], eager=True)).build()
        g.launch(3)
        out.append(float(b[0]))
    threads = [threading.Thread(target=work) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert out == [256.0, 256.0]


@pytest.mark.gpu
def test_a_recorded_stream_node_keeps_what_it_read():
    cp = pytest.importorskip("cupy")
    S = cp.cuda.Stream(non_blocking=True)
    w, y = cp.zeros(8), cp.zeros(8)
    g = (eagle.Graph()
         .add_node(opaque(lambda: cp.add(w, 1.0, out=w), reads={"w": w}, writes={"w": w}),
                   name="seg0")
         .add_node(opaque(lambda: cp.multiply(w, 3.0, out=y), reads={"w": w}, writes={"y": y},
                          stream=S), name="model")
         .add_node(opaque(lambda: cp.add(w, 10.0, out=w), reads={"w": w}, writes={"w": w}),
                   name="seg1").build(record=True))
    g.launch()
    with g.record._hold("test") as tape:
        kept = sorted(float(c[0]) for c in tape.copies)
    assert 1.0 in kept and float(y[0]) == 3.0 and float(w[0]) == 11.0


def test_a_rebound_host_parameter_is_refused(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("hawk")
    from test_interop_inject import inj_lin

    from eagle.interop import import_buffer

    (p,) = eagle.plan.auto([inj_lin], targets=("host",), cache_dir=str(tmp_path / "c"))
    w = torch.nn.Parameter(torch.tensor([2.0, 0.5], dtype=torch.float64))
    x, y = np.linspace(-1, 1, 16), np.zeros(16)
    g = eagle.Graph().add(p.bind(w=import_buffer(w), x=x, y=y), name="lin").build()
    with torch.no_grad():
        w.mul_(2.0)
    g.launch()
    np.testing.assert_array_equal(y, 4.0 * x + 1.0)
    w.data = torch.zeros(2, dtype=torch.float64)
    with pytest.raises(ValueError, match="plane 'w' was rebound"):
        g.launch()


@pytest.mark.gpu
def test_one_jax_node_in_eight_graphs_on_eight_threads():
    jax = pytest.importorskip("jax")
    cp = pytest.importorskip("cupy")
    jax.config.update("jax_enable_x64", True)
    from eagle.frameworks import jax as ej

    w = cp.arange(4096, dtype=cp.float64)
    ys = [cp.zeros(4096) for _ in range(8)]
    bump = jax.jit(lambda a: a + 1.0)
    graphs = [eagle.Graph().add_node(ej.node(bump, reads={"w": w}, writes={"y": y})).build()
              for y in ys]
    errors = []

    def work(g):
        try:
            for _ in range(5):
                g.launch()
        except BaseException as exc:        # surfaced below
            errors.append(exc)
    threads = [threading.Thread(target=work, args=(g,)) for g in graphs]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors
    assert all(bool((y == w + 1).all()) for y in ys)


@pytest.mark.torch_gpu
def test_a_child_with_a_node_type_a_loop_body_forbids_is_refused_by_name():
    torch = pytest.importorskip("torch")
    cp = pytest.importorskip("cupy")
    x, y, flag = cp.ones(8), cp.zeros(8), cp.full(1, 2, dtype=cp.uint32)
    xt, yt = torch.as_tensor(x, device="cuda"), torch.as_tensor(y, device="cuda")
    tg = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(tg, stream=torch.cuda.Stream()):
        yt.add_(xt)
        torch.cuda.Event(external=True).record()      # an event record node
    loop = eagle.repeat_while(G.child(tg, reads={"x": x}, writes={"y": y}),
                              eagle.SkipGuard.nonzero(flag), 3)
    with pytest.raises(RuntimeError, match="node of type 'event record', which CUDA does "
                                           "not allow inside a conditional or loop body"):
        eagle.Graph().add_node(loop, name="loop").build()


def test_a_node_vjp_runs_in_the_reverse(tmp_path):
    pytest.importorskip("hawk")
    from test_interop_inject import inj_addone

    import eagle.diff

    (p,) = eagle.plan.auto([inj_addone], targets=("host",), cache_dir=str(tmp_path / "c"))
    y, z, o = np.linspace(-1, 1, 16), np.zeros(16), np.zeros(16)
    g = (eagle.Graph()
         .add_node(opaque(lambda: np.multiply(y, 3.0, out=z), reads={"y": y}, writes={"z": z}),
                   name="model")
         .add(p.bind(x=z, y=o), name="addone").build(record=True))
    g.launch()
    cz, gy = np.zeros(16), np.zeros(16)
    back = opaque(lambda: np.multiply(cz, 3.0, out=gy), reads={"z": cz}, writes={"y": gy})
    grads = eagle.diff.reverse(g.record, {"addone.y": np.full(16, 2.0)}, vjps={"model": back})
    np.testing.assert_array_equal(grads["y"], np.full(16, 6.0))


@pytest.mark.gpu
@pytest.mark.interop_matrix  # certifies torch: deselected with the other framework rows, run by the Kaggle leg
# the refused launch leaves torch's capture empty on purpose, and torch says so
@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty:UserWarning")
def test_a_launch_inside_a_stream_capture_is_refused(tmp_path, monkeypatch):
    """Inside torch.cuda.graph an eagle graph's launch and enqueue refuse by name
    (they used to record an empty graph); outside a capture they run."""
    pytest.importorskip("hawk")
    from test_interop_inject import _torch_cuda, inj_lin

    from eagle import pipeline
    from eagle.interop import import_buffer

    torch = _torch_cuda()
    import cupy as cp

    (p,) = eagle.plan.auto([inj_lin], targets=("cuda",), cache_dir=str(tmp_path / "c"))
    w = torch.tensor([2.0, 0.5], dtype=torch.float64, device="cuda")
    x, y = cp.linspace(-1, 1, 64), cp.zeros(64)
    g = eagle.Graph().add(p.bind(w=import_buffer(w), x=x, y=y), name="lin").build()
    g.launch()
    want = y.copy()
    assert bool((y == 2.0 * x + 0.5).all())          # outside a capture: unaffected
    for call in ("launch", "enqueue"):
        tg = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="inside a stream capture.*eagle.graph.child"):
            with torch.cuda.graph(tg, stream=torch.cuda.Stream()):
                getattr(g, call)()
    g.launch()
    assert bool((y == want).all())
    # delete the check: the capture records nothing and the replay is silently wrong
    monkeypatch.setattr(G, "refuse_in_capture", lambda what: None)
    monkeypatch.setattr(pipeline, "refuse_in_capture", lambda what: None)
    tg = torch.cuda.CUDAGraph()
    y[...] = 0
    torch.cuda.synchronize()
    with pytest.warns(UserWarning, match="empty"), torch.cuda.graph(tg, stream=torch.cuda.Stream()):
        g.enqueue()
    y[...] = 0                      # the launch ran live; the graph itself holds nothing
    torch.cuda.synchronize()
    tg.replay()
    torch.cuda.synchronize()
    assert not bool((y == want).all())


@pytest.mark.gpu
def test_a_launch_inside_a_warp_capture_is_refused(tmp_path):
    """Under wp.ScopedCapture the refusal is the named one, not a raw CUDA error."""
    pytest.importorskip("hawk")
    wp = pytest.importorskip("warp")
    from test_interop_inject import inj_lin

    from eagle.interop import import_buffer

    wp.init()
    cp = pytest.importorskip("cupy")
    (p,) = eagle.plan.auto([inj_lin], targets=("cuda",), cache_dir=str(tmp_path / "c"))
    w = cp.asarray([2.0, 0.5])
    x, y = cp.linspace(-1, 1, 64), cp.zeros(64)
    g = eagle.Graph().add(p.bind(w=import_buffer(w), x=x, y=y), name="lin").build()
    g.launch()
    for call in ("launch", "enqueue"):
        with pytest.raises(RuntimeError, match="inside a stream capture.*eagle.graph.child"):
            with wp.ScopedCapture(device="cuda:0"):
                getattr(g, call)()
    g.launch()
    assert bool((y == 2.0 * x + 0.5).all())
