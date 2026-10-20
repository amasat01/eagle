# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The reader's own framework code inside an eagle graph: the interop rows.

One seam, three tiers: ``eagle.graph.opaque(fn)`` is captured,
``opaque(fn, stream=S)`` runs on the host each launch with its device work on
``S``, fenced on the device, and ``opaque(fn, eager=True)`` runs between two
device syncs. ``eagle.graph.child(owner)`` inserts a graph the framework
captured itself. Rows (one test per declared leg, each with its falsifier in
the row, fixture only):

* ``INJ-TIER-NAMED``: every node's tier is named in ``g.schedule`` and
  ``str(g)``; a kind patched to report ``capture`` turns the check red.
* ``STREAM-INJECT-STREAM-ORDER``: a stream node on a torch stream fenced
  across ``launch(3)``, two back-to-back enqueues, a launch from another
  thread, and first in the graph; the host returns early. Falsifiers: the
  stream's wait removed reads stale; the previous run's tail removed races.
* ``STREAM-INJECT-HOST-ORDER``: a host node's work on a non-default torch
  stream is seen by the next stage; a sync of the current stream alone
  (the old rule) reads stale.
* ``T-CHILD-OWNED``: ``G.child(CUDAGraph)`` keeps its owner alive; a reset and a
  bare handle are refused by name; without the handle check the reset passes.
* ``INJ-LOOP-HOOK``: a per-step hook inside a compacting ``until_done``
  agrees host vs device and with a reference, runs once per step and never
  writes a finished row; the shapes it cannot keep are refused by name.
* ``INJ-VJP-CHAIN``: hawk -> torch model -> hawk reversed through
  ``vjps={"model": fn}``, the model's parameter gradients accumulating.
* ``INJ-TABLE-LIVE``: weights bound zero-copy are read live; a rebound
  parameter is refused by name.
* ``STREAM-JAX-INJECT-ORDER`` / ``JX-INJECT-HOST``: a JAX step between eagle
  stages, stream-ordered (its result kept alive) or host-synchronized.
* ``INJ-RUN-VJP``: ``until_done([node, plan], record=True)`` reversed with
  ``vjps=`` matches pure-torch autograd; the node's planes come back bitwise.
* ``JX-LOOP-INVARIANCE``: a carried-state kernel in ``jit(while_loop)``
  equals ``until_done``; ``grad`` of a ``scan`` over it, with no ``where``,
  matches a plain JAX reference.
"""

from __future__ import annotations

import contextlib
import dataclasses
import gc
import os
import threading
import time
import weakref

import numpy as np
import pytest

hawk = pytest.importorskip("hawk", reason="the inject rows trace their kernels with hawk")
from hawk import Mutable, Param, Scalar, Table, Terminated  # noqa: E402
from hawk.ext import Guard, Kind
from raptor.conformance.interop import certified_framework, register_row

import eagle
import eagle.diff
import eagle.plan
from eagle import _counters
from eagle import graph as G
from eagle.graph import opaque

pytestmark = pytest.mark.interop_matrix

N = 1 << 14
CYCLES = 2 * 10**8          # ~0.1-0.2 s of one GPU thread spinning

_SPIN = r"""extern "C" __global__ void inj_spin(const double* x, double* y, long long c, int n) {
  long long t0 = clock64(); while (clock64() - t0 < c) {}
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) y[i] = x[i] + 1.0; }"""


def _cp():
    import cupy

    return cupy


def _spin(x, y, cycles=CYCLES):
    """``y = x + 1`` after ``cycles`` of spinning, on cupy's current stream."""
    cp = _cp()
    k = cp.RawKernel(_SPIN, "inj_spin")
    n = x.size
    k(((n + 255) // 256,), (256,), (x, y, np.int64(cycles), np.int32(n)))


def _torch_cuda():
    certified_framework("torch")
    import torch

    assert torch.cuda.is_available(), "the cuda leg needs a CUDA build of torch"
    return torch


def _dev(torch):
    return torch.device("cuda", _cp().cuda.Device().id)


# --------------------------------------------------------------------------- #
# INJ-TIER-NAMED
# --------------------------------------------------------------------------- #
_TAGS = {"stream": "[stream]", "host": "[host]", "child": "[child]"}


def _tiers_named(g, want_schedule, want_tags):
    text = str(g)
    return tuple(g.schedule) == tuple(want_schedule) and all(t in text for t in want_tags)


def _capture_tier_patched(monkeypatch, kind="opaque_stream"):
    reg = G._REGISTRY
    monkeypatch.setitem(reg, kind, dataclasses.replace(reg[kind], tier="capture"))


@register_row("INJ-TIER-NAMED", leg="cpu")
def test_tier_named_cpu(monkeypatch):
    def graph():
        a, b, c = np.ones(8), np.zeros(8), np.zeros(8)
        return (eagle.Graph()
                .add_node(opaque(lambda: np.add(a, 1, out=b), reads={"a": a}, writes={"b": b}),
                          name="captured")
                .add_node(opaque(lambda: np.add(b, 1, out=c), reads={"b": b}, writes={"c": c},
                                 stream=0), name="model")
                .add_node(opaque(lambda: np.add(c, 1, out=a), reads={"c": c}, writes={"a": a},
                                 eager=True), name="readback")
                .build()), a
    g, a = graph()
    g.launch()
    assert float(a[0]) == 4.0
    schedule = ("work_share",) * 3
    tags = ("[stream: cpu runs eagerly]", "[host]")
    assert _tiers_named(g, schedule, tags), str(g)
    _capture_tier_patched(monkeypatch)
    assert not _tiers_named(graph()[0], schedule, tags)


@register_row("INJ-TIER-NAMED", leg="cuda")
def test_tier_named_cuda(monkeypatch):
    torch = _torch_cuda()
    cp = _cp()
    x = cp.ones(8)
    y = cp.zeros(8)
    yt, xt = torch.as_tensor(y, device=_dev(torch)), torch.as_tensor(x, device=_dev(torch))
    tg = torch.cuda.CUDAGraph(keep_graph=True)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        yt.copy_(xt * 2)
    torch.cuda.synchronize()
    with torch.cuda.graph(tg, stream=s):
        yt.copy_(xt * 2)
    S = torch.cuda.Stream()

    def graph():
        b, c, d = cp.zeros(8), cp.zeros(8), cp.zeros(8)
        return (eagle.Graph()
                .add_node(G.child(tg, reads={"x": x}, writes={"y": y}), name="torch_graph")
                .add_node(opaque(lambda: cp.add(y, 1, out=b), reads={"y": y}, writes={"b": b}),
                          name="captured")
                .add_node(opaque(lambda: cp.add(b, 1, out=c), reads={"b": b}, writes={"c": c},
                                 stream=S), name="model")
                .add_node(opaque(lambda: cp.add(c, 1, out=d), reads={"c": c}, writes={"d": d},
                                 eager=True), name="readback")
                .build()), d
    g, d = graph()
    g.launch()
    assert float(d[0]) == 5.0
    schedule = ("capture", "capture", "eager_stream", "eager_host")
    assert _tiers_named(g, schedule, ("[child]", "[stream]", "[host]")), str(g)
    _capture_tier_patched(monkeypatch)
    assert not _tiers_named(graph()[0], schedule, ("[child]", "[stream]", "[host]"))


# --------------------------------------------------------------------------- #
# STREAM-INJECT-STREAM-ORDER
# --------------------------------------------------------------------------- #
def _chain(cp, S, *, slow_node=True, slow_first=False):
    """seg0 ``w += 1`` -> stream node ``y = w + 1`` on ``S`` -> seg1 ``acc += y``."""
    w, y, acc = cp.zeros(N), cp.zeros(N), cp.zeros(N)
    seg0 = (lambda: _spin(w, w)) if slow_first else (lambda: cp.add(w, 1.0, out=w))
    node = (lambda: _spin(w, y)) if slow_node else (lambda: cp.add(w, 1.0, out=y))
    g = (eagle.Graph()
         .add_node(opaque(seg0, reads={"w": w}, writes={"w": w}), name="seg0")
         .add_node(opaque(node, reads={"w": w}, writes={"y": y}, stream=S), name="model")
         .add_node(opaque(lambda: cp.add(acc, y, out=acc), reads={"acc": acc, "y": y},
                          writes={"acc": acc}), name="seg1")
         .build())
    return g, w, y, acc


def _acc_after(runs):
    """acc after ``runs`` in-order runs: sum over r of (r + 1)."""
    return float(sum(r + 1 for r in range(1, runs + 1)))


def _in_threads(targets, *, setup=None):
    """Run each target in its own thread (after ``setup()`` in that thread);
    a thread's exception is raised here, never lost."""
    errors = []

    def run(target):
        try:
            if setup is None:
                target()
            else:
                with setup():
                    target()
        except BaseException as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=run, args=(t,)) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not any(t.is_alive() for t in threads)
    if errors:
        raise errors[0]


@contextlib.contextmanager
def _foreign_context():
    """A CUDA context nobody built a graph under, current on this thread (what
    a framework that owns its own context leaves behind)."""
    cp = _cp()
    drv = cp.cuda.driver
    ctx = drv.ctxCreate(cp.cuda.Device().id)
    try:
        yield
    finally:
        cp.cuda.runtime.setDevice(cp.cuda.Device().id)
        drv.ctxDestroy(ctx)


def test_launch_from_a_thread_with_a_foreign_context():
    torch = _torch_cuda()
    cp = _cp()
    ts = torch.cuda.Stream()
    chains = [_chain(cp, ts, slow_node=False) for _ in range(4)]
    _in_threads([c[0].launch for c in chains], setup=_foreign_context)
    for _g, w, _y, acc in chains:
        assert float(w[0]) == 1.0 and bool((acc == _acc_after(1)).all())


def test_graph_enters_its_planes_device(monkeypatch):
    cp = _cp()
    entered = []
    real = G._on_device

    def spy(device):
        entered.append(device)
        return real(device)

    monkeypatch.setattr(G, "_on_device", spy)
    w = cp.zeros(8)
    g = (eagle.Graph().add_node(opaque(lambda: cp.add(w, 1.0, out=w), reads={"w": w},
                                       writes={"w": w}), name="inc").build())
    here = cp.cuda.Device().id
    assert entered == [here]
    g.launch()
    g.enqueue()
    g.synchronize()
    assert entered == [here] * 4 and float(w[0]) == 2.0


@register_row("STREAM-INJECT-STREAM-ORDER", leg="cuda")
def test_stream_node_is_fenced_on_the_device(monkeypatch):
    torch = _torch_cuda()
    cp = _cp()
    ts = torch.cuda.Stream()
    # E1: launch(3), and the host returns early from enqueue
    g, w, y, acc = _chain(cp, ts)
    cp.cuda.Device().synchronize()
    t0 = time.perf_counter()
    g.enqueue(3)
    host = time.perf_counter() - t0
    g.synchronize()
    wall = time.perf_counter() - t0
    assert float(w[0]) == 3.0 and bool((acc == _acc_after(3)).all())
    assert host / wall < 0.1, (host, wall)
    g.launch(3)
    assert float(w[0]) == 6.0 and bool((acc == _acc_after(6)).all())
    # E2: two back-to-back enqueues, then a launch from another thread in flight
    w.fill(0)
    acc.fill(0)
    g.enqueue(1)
    g.enqueue(1)
    _in_threads([g.launch])
    g.synchronize()
    assert bool((acc == _acc_after(3)).all())
    # E3: the stream node first, after a slow producer on the caller's stream
    w.fill(0)
    z = cp.zeros(N)
    first = (eagle.Graph()
             .add_node(opaque(lambda: cp.multiply(w, 1.0, out=y), reads={"w": w},
                              writes={"y": y}, stream=ts), name="model")
             .add_node(opaque(lambda: cp.multiply(y, 2.0, out=z), reads={"y": y},
                              writes={"z": z}), name="seg1").build())
    x = cp.arange(N, dtype=cp.float64)
    _spin(x, w)
    first.launch()
    assert bool((z == 2 * (x + 1)).all())
    # falsifier (a): the stream's wait removed reads stale
    ga, wa, ya, acca = _chain(cp, ts, slow_node=False, slow_first=True)
    with monkeypatch.context() as m:
        m.setattr(G, "_wait_stream", lambda stream, events: None)
        ga.launch()
    assert not bool((acca == 2.0).all())
    # falsifier (b): the previous run's tail removed races run to run
    gb, wb, yb, accb = _chain(cp, ts)
    seeds = G._run_seeds
    with monkeypatch.context() as m:
        m.setattr(G, "_run_seeds", lambda cache, tail: seeds(cache, None))
        gb.enqueue(1)
        gb.enqueue(1)
        gb.synchronize()
    assert not bool((accb == _acc_after(2)).all())


@register_row("STREAM-INJECT-HOST-ORDER", leg="cuda")
def test_host_node_sees_a_non_default_torch_stream(monkeypatch):
    torch = _torch_cuda()
    cp = _cp()
    ts = torch.cuda.Stream()
    on_ts = cp.cuda.ExternalStream(ts.cuda_stream)
    x = cp.arange(N, dtype=cp.float64)
    y, z = cp.zeros(N), cp.zeros(N)
    yt = torch.as_tensor(y, device=_dev(torch))

    def model():
        with on_ts:
            _spin(x, y)                 # y = x + 1, slowly, on the torch stream
        with torch.cuda.stream(ts):
            yt.mul_(1.0)

    def graph():
        y.fill(0)
        z.fill(0)
        return (eagle.Graph()
                .add_node(opaque(model, reads={"x": x}, writes={"y": y}, eager=True),
                          name="model")
                .add_node(opaque(lambda: cp.multiply(y, 2.0, out=z), reads={"y": y},
                                 writes={"z": z}), name="seg1").build())
    graph().launch()
    assert bool((z == 2 * (x + 1)).all())
    with monkeypatch.context() as m:
        m.setattr(G, "_device_sync", lambda: cp.cuda.get_current_stream().synchronize())
        graph().launch()
    assert not bool((z == 2 * (x + 1)).all())


# --------------------------------------------------------------------------- #
# T-CHILD-OWNED
# --------------------------------------------------------------------------- #
def _torch_graph(torch, xt, yt):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 32), torch.nn.Tanh(),
                                torch.nn.Linear(32, 1)).to(xt.device)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            model(xt)
    torch.cuda.synchronize()
    tg = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(tg, stream=s), torch.no_grad():
        yt.copy_(model(xt).squeeze(1))
    return tg, model


@register_row("T-CHILD-OWNED", leg="cuda")
def test_child_owns_the_framework_graph(monkeypatch):
    torch = _torch_cuda()
    cp = _cp()
    n = 1 << 12
    x = cp.random.rand(n, 8, dtype=cp.float32)
    y, z = cp.zeros(n, cp.float32), cp.zeros(n, cp.float32)
    xt, yt = torch.as_tensor(x, device=_dev(torch)), torch.as_tensor(y, device=_dev(torch))
    tg, model = _torch_graph(torch, xt, yt)
    g = (eagle.Graph().add_node(G.child(tg, reads={"x": x}, writes={"y": y}), name="model")
         .add_node(opaque(lambda: cp.multiply(y, 3.0, out=z), reads={"y": y},
                          writes={"z": z}), name="scale").build())
    with torch.no_grad():
        ref = model(xt).squeeze(1).clone()
    del tg
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    junk = [torch.full((n, 32), 999.0, device=_dev(torch)) for _ in range(4)]
    torch.cuda.synchronize()
    g.launch()
    assert float((torch.as_tensor(z, device=_dev(torch)) - 3 * ref).abs().max()) == 0.0
    assert not any(bool((j != 999.0).any()) for j in junk)
    # a reset owner is refused by name at the next launch
    tg2, _ = _torch_graph(torch, xt, yt)
    g2 = eagle.Graph().add_node(G.child(tg2, reads={"x": x}, writes={"y": y})).build()
    g2.launch()
    tg2.reset()
    with pytest.raises(RuntimeError, match="reset or re-captured; rebuild the eagle Graph"):
        g2.launch()
    # a bare handle is refused by name
    tg3, _ = _torch_graph(torch, xt, yt)
    with pytest.raises(TypeError, match="pass the graph object; eagle must keep it alive"):
        G.child(tg3.raw_cuda_graph(), reads={"x": x}, writes={"y": y})
    # falsifier: without the handle check, nothing stands between a reset and a launch
    monkeypatch.setattr(G._Child, "check", lambda self: None)
    g2._check_live()


# --------------------------------------------------------------------------- #
# INJ-LOOP-HOOK
# --------------------------------------------------------------------------- #
DT, C = 0.01, 0.3


def _hook_step(dt: Param, t_final: Scalar, acc: Scalar, terminated: Terminated,
               x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    vn = v + dt * acc
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


inj_hook_step = hawk.kernel(_hook_step, kind=Kind("inj_hook_step", guard=Guard(active_set=True)),
                            steps=1)
inj_hook_step_x4 = hawk.kernel(_hook_step, kind=Kind("inj_hook_step4",
                                                    guard=Guard(active_set=True)), steps=4)


def _hook_inputs(n=256):
    rng = np.random.default_rng(3)
    return dict(x=rng.uniform(-1, 1, n), v=rng.uniform(-0.5, 0.5, n), t=np.zeros(n),
                t_final=rng.uniform(0.1, 1.0, n), acc=np.zeros(n))


def _hook_reference(inp, steps):
    x, v, t, acc = (inp[k].copy() for k in ("x", "v", "t", "acc"))
    term = np.zeros(x.size, bool)
    for _ in range(steps):
        live = ~term
        acc[live] = (-np.sin(x) - C * v)[live]
        vn = v + DT * acc
        x = np.where(live, x + DT * v, x)
        v = np.where(live, vn, v)
        t = np.where(live, t + DT, t)
        term = term | (live & (t >= inp["t_final"]))
    return dict(x=x, v=v, t=t, acc=acc)


def _run_hooked(torch, xp, plan, model_node, inp, mask, count):
    planes = {k: xp.asarray(v).copy() for k, v in inp.items()}
    node = model_node(planes, mask, count)
    runner = eagle.until_done([node, plan], max_steps=400, dt=DT, terminated=mask, **planes)
    rep = runner.run()
    return rep, planes


def _hook_cpu(torch, write_finished=False):
    def make(planes, mask, count):
        tx, tv, ta = (torch.from_numpy(planes[k]) for k in ("x", "v", "acc"))
        tm, tc = torch.from_numpy(mask), torch.from_numpy(count)

        def model():
            new = -torch.sin(tx) - C * tv
            ta.copy_(new if write_finished else torch.where(tm, ta, new))
            tc.add_(1)
        return opaque(model, reads={"x": planes["x"], "v": planes["v"], "terminated": mask},
                      writes={"acc": planes["acc"], "count": count})
    return make


def _check_hook(rep, planes, count, inp, get):
    want = _hook_reference(inp, rep.steps)
    assert rep.done
    assert int(get(count)[0]) == rep.steps          # once per step
    for k in ("x", "v", "t", "acc"):
        np.testing.assert_allclose(get(planes[k]), want[k], rtol=1e-12, atol=1e-14, err_msg=k)


@register_row("INJ-LOOP-HOOK", leg="cpu")
def test_loop_hook_host(tmp_path):
    certified_framework("torch")
    import torch

    (plan,) = eagle.plan.auto([inj_hook_step], targets=("host",), cache_dir=str(tmp_path / "c"))
    inp = _hook_inputs()
    mask, count = np.zeros(256, bool), np.zeros(1, np.int64)
    rep, planes = _run_hooked(torch, np, plan, _hook_cpu(torch), inp, mask, count)
    assert rep.compactions >= 1
    _check_hook(rep, planes, count, inp, np.asarray)
    # the refusals, by name
    (k4,) = eagle.plan.auto([inj_hook_step_x4], targets=("host",), cache_dir=str(tmp_path / "c4"))
    node = _hook_cpu(torch)({k: np.asarray(v).copy() for k, v in inp.items()}, mask, count)
    with pytest.raises(ValueError, match="a per-step hook needs one step per launch"):
        eagle.until_done([node, k4], max_steps=40, dt=DT, terminated=mask,
                         **{k: np.asarray(v).copy() for k, v in inp.items()})
    hidden = np.zeros(256)
    extra = opaque(lambda: None, reads={"x": planes["x"]}, writes={"hidden": hidden})
    with pytest.raises(ValueError, match="reorder= cannot move plane 'hidden', which only node"):
        eagle.until_done([extra, plan], max_steps=40, reorder=0.5, dt=DT, terminated=mask,
                         **{k: np.asarray(v).copy() for k, v in inp.items()})
    for host_node in (opaque(lambda: None, stream=0), opaque(lambda: None, eager=True)):
        with pytest.raises(ValueError, match="a stepping loop is one device graph"):
            eagle.until_done([host_node, plan], max_steps=40, dt=DT, terminated=mask,
                             **{k: np.asarray(v).copy() for k, v in inp.items()})
    # falsifier: a hook that writes finished rows breaks the contract the row checks
    mask2, count2 = np.zeros(256, bool), np.zeros(1, np.int64)
    rep2, planes2 = _run_hooked(torch, np, plan, _hook_cpu(torch, write_finished=True), inp,
                                mask2, count2)
    with pytest.raises(AssertionError):
        _check_hook(rep2, planes2, count2, inp, np.asarray)


@register_row("INJ-LOOP-HOOK", leg="cuda")
def test_loop_hook_device(tmp_path):
    torch = _torch_cuda()
    cp = _cp()
    dev = _dev(torch)
    (gplan,) = eagle.plan.auto([inj_hook_step], targets=("cuda",), cache_dir=str(tmp_path / "g"))
    (hplan,) = eagle.plan.auto([inj_hook_step], targets=("host",), cache_dir=str(tmp_path / "h"))
    inp = _hook_inputs()
    graphs = []

    def make(planes, mask, count):
        tx, tv, ta, tm = (torch.as_tensor(planes[k] if k != "m" else mask, device=dev)
                          for k in ("x", "v", "acc", "m"))
        tc = torch.as_tensor(count, device=dev)
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            ta.copy_(torch.where(tm, ta, -torch.sin(tx) - C * tv))
        torch.cuda.synchronize()
        ta.zero_()
        tg = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(tg, stream=s):
            ta.copy_(torch.where(tm, ta, -torch.sin(tx) - C * tv))
            tc.add_(1)
        graphs.append(tg)
        return G.child(tg, reads={"x": planes["x"], "v": planes["v"], "terminated": mask},
                     writes={"acc": planes["acc"], "count": count})

    mask, count = cp.zeros(256, cp.bool_), cp.zeros(1, cp.int64)
    rep, planes = _run_hooked(torch, cp, gplan, make, inp, mask, count)
    assert rep.compactions >= 1
    _check_hook(rep, planes, count, inp, cp.asnumpy)
    hmask, hcount = np.zeros(256, bool), np.zeros(1, np.int64)
    hrep, hplanes = _run_hooked(torch, np, hplan, _hook_cpu(torch), inp, hmask, hcount)
    assert hrep.launches == rep.launches and hrep.steps == rep.steps
    for k in ("x", "v", "t", "acc"):
        np.testing.assert_allclose(cp.asnumpy(planes[k]), hplanes[k], rtol=1e-12, atol=1e-14)


# --------------------------------------------------------------------------- #
# INJ-VJP-CHAIN
# --------------------------------------------------------------------------- #
@hawk.kernel
def inj_scale(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x


@hawk.kernel
def inj_addone(x: Scalar, y: Mutable[Scalar]):
    y = x + 1.0


def _vjp_chain(torch, xp, targets, cache, eager, drop_c=False):
    p1, p2 = eagle.plan.auto([inj_scale, inj_addone], targets=targets, cache_dir=cache)
    m = 1024
    dev = "cuda" if xp is not np else "cpu"
    x = xp.linspace(-1.0, 1.0, m)
    y, z, o = xp.zeros(m), xp.zeros(m), xp.zeros(m)
    c = torch.tensor(0.7, dtype=torch.float64, device=dev, requires_grad=True)
    d = torch.tensor(-0.2, dtype=torch.float64, device=dev, requires_grad=True)
    yt, zt = torch.as_tensor(y, device=dev), torch.as_tensor(z, device=dev)

    def forward():
        with torch.no_grad():
            zt.copy_(torch.tanh(c * yt + d))

    def vjp(primals, cotangents):
        y_ = torch.as_tensor(primals["y"], device=dev).requires_grad_(True)
        out = torch.tanh(c * y_ + d)
        gy, gc, gd = torch.autograd.grad(out, [y_, c, d], torch.as_tensor(cotangents["z"],
                                                                          device=dev))
        for p, gp in ((c, gc), (d, gd)):
            p.grad = gp if p.grad is None else p.grad + gp
        if drop_c:
            gy = gy / c.detach()
        return {"y": xp.asarray(gy)}

    g = (eagle.Graph().add(p1.bind(x=x, a=2.0, y=y), name="scale")
         .add_node(opaque(forward, reads={"y": y}, writes={"z": z}, eager=eager), name="model")
         .add(p2.bind(x=z, y=o), name="addone").build(record=True))
    return g, x, o, c, vjp


def _check_vjp_chain(torch, g, x, o, c, vjp, xp):
    get = (lambda a: a.get()) if xp is not np else np.asarray
    xs = get(x)
    sech2 = 1 - np.tanh(0.7 * 2 * xs - 0.2) ** 2
    for launch in (1, 2):
        g.launch()
        np.testing.assert_allclose(get(o), np.tanh(1.4 * xs - 0.2) + 1.0, rtol=0, atol=1e-14)
        grads = eagle.diff.reverse(g.record, {"addone.y": xp.ones(x.size)}, vjps={"model": vjp})
        np.testing.assert_allclose(get(grads["x"]), 2 * 0.7 * sech2, rtol=0, atol=1e-12)
        np.testing.assert_allclose(float(get(grads["a"])), (0.7 * xs * sech2).sum(), rtol=1e-12)
        assert abs(c.grad.item() - launch * (2 * xs * sech2).sum()) < 1e-9


@register_row("INJ-VJP-CHAIN", leg="cpu")
def test_vjp_chain_host(tmp_path):
    certified_framework("torch")
    import torch

    g, x, o, c, vjp = _vjp_chain(torch, np, ("host",), str(tmp_path / "c"), False)
    _check_vjp_chain(torch, g, x, o, c, vjp, np)
    with pytest.raises(ValueError, match=r"node 'model' is opaque: pass vjps=\{'model'"):
        eagle.diff.reverse(g.record, {"addone.y": np.ones(x.size)})
    g2, x2, o2, c2, bad = _vjp_chain(torch, np, ("host",), str(tmp_path / "c"), False, True)
    with pytest.raises(AssertionError):
        _check_vjp_chain(torch, g2, x2, o2, c2, bad, np)


@register_row("INJ-VJP-CHAIN", leg="cuda")
def test_vjp_chain_device(tmp_path):
    torch = _torch_cuda()
    cp = _cp()
    g, x, o, c, vjp = _vjp_chain(torch, cp, ("cuda",), str(tmp_path / "g"), True)
    assert g.schedule == ("capture", "eager_host", "capture")
    _check_vjp_chain(torch, g, x, o, c, vjp, cp)
    with pytest.raises(ValueError, match=r"node 'model' is opaque: pass vjps=\{'model'"):
        eagle.diff.reverse(g.record, {"addone.y": cp.ones(x.size)})
    g2, x2, o2, c2, bad = _vjp_chain(torch, cp, ("cuda",), str(tmp_path / "g"), True, True)
    with pytest.raises(AssertionError):
        _check_vjp_chain(torch, g2, x2, o2, c2, bad, cp)


# --------------------------------------------------------------------------- #
# INJ-TABLE-LIVE
# --------------------------------------------------------------------------- #
@hawk.kernel
def inj_lin(w: Table[Scalar], x: Scalar, y: Mutable[Scalar]):
    y = w.at(0) * x + w.at(1)


@register_row("INJ-TABLE-LIVE", leg="cuda")
def test_table_weights_are_read_live(tmp_path, monkeypatch):
    torch = _torch_cuda()
    cp = _cp()
    from eagle.interop import import_buffer

    (p,) = eagle.plan.auto([inj_lin], targets=("cuda",), cache_dir=str(tmp_path / "c"))
    w = torch.nn.Parameter(torch.tensor([2.0, 0.5], dtype=torch.float64, device=_dev(torch)))
    x, y = cp.linspace(-1, 1, 64), cp.zeros(64)
    g = eagle.Graph().add(p.bind(w=import_buffer(w), x=x, y=y), name="lin").build()
    g.launch()
    assert bool((y == 2.0 * x + 0.5).all())
    before = _counters.snapshot()
    with torch.no_grad():
        w.mul_(3.0)                      # the optimizer's in-place step IS the refresh
    g.launch()
    assert bool(cp.allclose(y, 6.0 * x + 1.5, rtol=0, atol=1e-15))
    after = _counters.snapshot()
    assert all(after[k] == before[k] for k in ("captures", "graph_builds", "instantiations"))
    w.data = torch.tensor([1.0, 0.0], dtype=torch.float64, device=_dev(torch))
    with pytest.raises(ValueError, match="plane 'w' was rebound \\(its data pointer moved\\); "
                                         "bind the new tensor"):
        g.launch()
    # falsifier: without the pointer check the stale weights pass silently
    monkeypatch.setattr(eagle.plan.BoundPlan, "_check_origins", lambda self: None)
    g.launch()
    assert bool(cp.allclose(y, 6.0 * x + 1.5, rtol=0, atol=1e-15))
    assert not bool(cp.allclose(y, 1.0 * x, rtol=0, atol=1e-15))


# --------------------------------------------------------------------------- #
# STREAM-JAX-INJECT-ORDER / JX-INJECT-HOST
# --------------------------------------------------------------------------- #
def _jax_spin(tmp_path_factory):
    certified_framework("jax")
    from test_interop_jax import _jax, _spin_cycles, _spin_op

    jax = _jax()
    spin = _spin_op(tmp_path_factory)
    return jax, spin, _spin_cycles


def test_jax_node_made_in_a_foreign_context_runs_on_the_planes_device():
    certified_framework("jax")
    from test_interop_jax import _jax

    jax = _jax()
    from eagle.frameworks import jax as ej

    cp = _cp()
    x, y = cp.arange(8, dtype=cp.float64), cp.zeros(8)
    with _foreign_context():
        step = ej.node(jax.jit(lambda a: a * 2.0), reads={"x": x}, writes={"y": y})
    holder = []
    _in_threads([lambda: holder.append(eagle.Graph().add_node(step).build().launch())])
    assert bool((y == 2 * x).all())


def test_jax_node_on_the_second_device_survives_jax_leaving_device_0_current():
    certified_framework("jax")
    from test_interop_jax import _jax

    jax = _jax()
    from eagle.frameworks import jax as ej

    cp = _cp()
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two GPUs")
    with cp.cuda.Device(1):
        x, y = cp.arange(8, dtype=cp.float64), cp.zeros(8)
    step = ej.node(jax.jit(lambda a: a * 2.0), reads={"x": x}, writes={"y": y})
    graph = eagle.Graph().add_node(step).build()
    for _ in range(2):                      # the second launch starts where JAX left device 0
        cp.cuda.Device(0).use()
        graph.launch()
    with cp.cuda.Device(1):
        cp.cuda.Device(1).synchronize()
        assert bool((y == 2 * x).all())


def test_jax_node_refuses_planes_on_two_devices():
    from eagle.frameworks import jax as ej

    def fake(ordinal):
        cls = type("ndarray", (), {"__module__": "cupy.fake"})
        obj = cls()
        obj.device = type("D", (), {"id": ordinal})()
        return obj

    with pytest.raises(ValueError, match="span devices"):
        ej.node(lambda a: a, reads={"x": fake(0)}, writes={"y": fake(1)})


@register_row("STREAM-JAX-INJECT-ORDER", leg="cuda")
def test_jax_node_is_stream_ordered_and_retained(tmp_path_factory, monkeypatch):
    jax, spin, cycles_for = _jax_spin(tmp_path_factory)
    import jax.numpy as jnp

    from eagle.frameworks import jax as ej

    cp = _cp()
    n = 1 << 20
    x = cp.arange(n, dtype=cp.float64)
    w, y, z = cp.zeros(n), cp.zeros(n), cp.zeros(n)
    cyc = cycles_for(spin, jnp.zeros(n), 0.15)
    sentinel = jax.jit(lambda a: jnp.full_like(a, 123.0) + a * 0.0)
    bump = jax.jit(lambda a: spin(a, 1000) + 0.0)
    sentinel(jnp.zeros(n)).block_until_ready()
    busy = cp.zeros(1)

    def model(wj):              # the node's stream stays busy after JAX's result is ready
        _spin(busy, busy, cyc)
        return bump(wj)

    def graph(node):
        return (eagle.Graph()
                .add_node(opaque(lambda: _spin(x, w, cyc), reads={"x": x}, writes={"w": w}),
                          name="seg0")
                .add_node(node(model, reads={"w": w}, writes={"y": y}),
                          name="model")
                .add_node(opaque(lambda: cp.multiply(y, 2.0, out=z), reads={"y": y},
                                 writes={"z": z}), name="seg1").build())

    def trials(g, k):
        bad = 0
        for _ in range(k):
            y.fill(-7.0)
            z.fill(0.0)
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            g.enqueue(1)
            host = time.perf_counter() - t0
            junk = [sentinel(jnp.zeros(n)) for _ in range(4)]   # JAX allocates behind it
            g.synchronize()
            wall = time.perf_counter() - t0
            del junk
            bad += not bool((z == 2 * (x + 2)).all())
        return bad, host / wall

    g = graph(ej.node)
    assert g.schedule == ("capture", "eager_stream", "capture")
    bad, ratio = trials(g, 10)
    assert bad == 0 and ratio < 0.1, (bad, ratio)

    # every result a queued copy still reads is alive (the node's stream is busy, so the copy is queued)
    seen = []
    keep = ej._Retain.keep

    def dropped_early(g):
        g.enqueue(1)
        gc.collect()
        n = sum(ref() is None and not event.done for event, ref in seen)
        g.synchronize()
        return n

    monkeypatch.setattr(ej._Retain, "keep", lambda self, event, view: (
        seen.append((event, weakref.ref(view))), keep(self, event, view)))
    assert dropped_early(graph(ej.node)) == 0 and seen
    # falsifier: the retain list off -> a result is dropped while its copy is still queued
    seen.clear()
    monkeypatch.setattr(ej._Retain, "keep", lambda self, event, view: seen.append((event, weakref.ref(view))))
    assert dropped_early(graph(ej.node)) > 0


@register_row("JX-INJECT-HOST", leg="cuda")
def test_jax_call_in_a_host_node(tmp_path_factory, monkeypatch):
    jax, _spin_jax, _cycles = _jax_spin(tmp_path_factory)

    cp = _cp()
    n = 1 << 16
    x = cp.arange(n, dtype=cp.float64)
    w, y, z = cp.zeros(n), cp.zeros(n), cp.zeros(n)
    bump = jax.jit(lambda a: a + 1.0)

    def model():                                # reads w back through JAX on the host
        out = bump(jax.dlpack.from_dlpack(w))
        cp.copyto(y, cp.from_dlpack(out))

    def graph():
        w.fill(0)
        y.fill(0)
        z.fill(0)
        return (eagle.Graph()
                .add_node(opaque(lambda: _spin(x, w), reads={"x": x}, writes={"w": w}),
                          name="seg0")
                .add_node(opaque(model, reads={"w": w}, writes={"y": y}, eager=True),
                          name="model")
                .add_node(opaque(lambda: cp.multiply(y, 2.0, out=z), reads={"y": y},
                                 writes={"z": z}), name="seg1").build())
    graph().launch()
    assert bool((z == 2 * (x + 2)).all())
    with monkeypatch.context() as m:
        m.setattr(G, "_device_sync", lambda: None)
        graph().launch()
        stale = not bool((z == 2 * (x + 2)).all())
        cp.cuda.Device().synchronize()
    assert stale


# --------------------------------------------------------------------------- #
# JX-LOOP-INVARIANCE
# --------------------------------------------------------------------------- #
NSTOP = 50.0


@hawk.kernel(steps=1)
def inj_grow(nstop: Param, terminated: Terminated, k: Mutable[Scalar]):
    k = k + 0.1 * k * k
    terminated = k >= nstop


@hawk.kernel(steps=1)
def inj_grow_b(nstop: Param, terminated: Terminated, k: Mutable[Scalar]):
    k = k + 0.1 * k * k
    terminated = k >= nstop


def _grow_runs(jax, K, k0, mask0, length=120):
    import jax.numpy as jnp
    from jax import lax

    def step(k, m):
        return K(NSTOP, m, k)

    def while_run(k, m):
        return lax.while_loop(lambda c: ~jnp.all(c[1]), lambda c: step(*c), (k, m))

    def scan_run(k, m):
        return lax.scan(lambda c, _: (step(*c), None), (k, m), None, length=length)[0]

    def ref_run(k, m):
        def body(c, _):
            k, m = c
            kn = jnp.where(m, k, k + 0.1 * k * k)
            return (kn, m | (kn >= NSTOP)), None
        return lax.scan(body, (k, m), None, length=length)[0]

    kw, _ = jax.jit(while_run)(k0, mask0)
    g = jax.grad(lambda k: scan_run(k, mask0)[0].sum())(k0)
    gr = jax.grad(lambda k: ref_run(k, mask0)[0].sum())(k0)
    return kw, g, gr


def _loop_invariance(jax, platform, plan, xp, tmp, monkeypatch):
    import jax.numpy as jnp

    from eagle.frameworks import jax as ej

    device = jax.devices(platform)[0]
    n = 2048
    k_host = np.random.default_rng(1).uniform(0.5, 2.0, n)
    with jax.default_device(device):
        k0, mask0 = jnp.asarray(k_host), jnp.zeros(n, bool)
        K = ej.function(inj_grow, cache_dir=tmp / "f")
        kw, g, gr = _grow_runs(jax, K, k0, mask0)
    k = xp.asarray(k_host).copy()
    rep = eagle.run_until_done(plan, max_steps=1000, nstop=NSTOP, k=k)
    assert rep.done
    np.testing.assert_array_equal(np.asarray(kw), k.get() if xp is not np else k)
    np.testing.assert_allclose(np.asarray(g), np.asarray(gr), rtol=1e-12)
    # falsifier: without the finished-sample identity the gradient explodes
    with monkeypatch.context() as m:
        m.setattr(ej, "finished_identity", lambda where, mask, bar_next, jt: jt)
        with jax.default_device(device):
            K2 = ej.function(inj_grow, cache_dir=tmp / "f2")
            _, g2, _ = _grow_runs(jax, K2, k0, mask0)
    assert not np.allclose(np.asarray(g2), np.asarray(gr), rtol=1e-6)
    # falsifier: without the alias the carried state collapses to zero
    with monkeypatch.context() as m:
        m.setattr(ej, "_carried", lambda sig: ())
        with jax.default_device(device):
            K3 = ej.function(inj_grow_b, cache_dir=tmp / "f3", wrt=())
            k3 = jax.jit(lambda k, m_: jax.lax.fori_loop(
                0, 200, lambda i, c: K3(NSTOP, c[1], c[0]), (k, m_)))(k0, mask0)[0]
    assert np.any(np.asarray(k3) == 0.0)


@register_row("JX-LOOP-INVARIANCE", leg="cpu")
def test_jax_loop_invariance_host(tmp_path, monkeypatch):
    certified_framework("jax")
    from test_interop_jax import _jax

    jax = _jax()
    monkeypatch.setenv("HAWK_HOST_PROFILE", "native")
    (plan,) = eagle.plan.auto([inj_grow], targets=("host",), cache_dir=str(tmp_path / "p"))
    _loop_invariance(jax, "cpu", plan, np, tmp_path, monkeypatch)


@register_row("JX-LOOP-INVARIANCE", leg="cuda")
def test_jax_loop_invariance_device(tmp_path, monkeypatch):
    certified_framework("jax")
    from test_interop_jax import _jax

    jax = _jax()
    (plan,) = eagle.plan.auto([inj_grow], targets=("cuda",), cache_dir=str(tmp_path / "p"))
    _loop_invariance(jax, "cuda", plan, _cp(), tmp_path, monkeypatch)


# --------------------------------------------------------------------------- #
# INJ-RUN-VJP
# --------------------------------------------------------------------------- #
def _osc(dt: Param, c: Param, t_final: Scalar, corr: Scalar, terminated: Terminated,
         x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    vn = v + dt * (-x - c * v + corr * v)     # corr's value enters the step's vjp
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


inj_osc = hawk.kernel(_osc, kind=Kind("inj_osc"), steps=1)
inj_osc_c = hawk.kernel(_osc, kind=Kind("inj_osc_c", guard=Guard(active_set=True)), steps=1)


class _RunVjp:
    """One recorded ``[node, plan]`` run: a 2-8-1 torch MLP writes ``corr``
    from ``(x, v)`` per step; 50..200 steps per sample, W=16."""

    def __init__(self, torch, xp, dev, plan, n=512, counter=False, node_kw=None, around=None, **kw):
        torch.manual_seed(0)
        self.torch, self.xp, self.dev, self.calls, self.fwd_calls = torch, xp, dev, 0, None
        lin1 = torch.nn.Linear(2, 8, dtype=torch.float64, device=dev)
        lin2 = torch.nn.Linear(8, 1, dtype=torch.float64, device=dev)
        self.params = [*lin1.parameters(), *lin2.parameters()]
        self.model = lambda x, v: lin2(torch.tanh(lin1(torch.stack([x, v], 1)))).squeeze(1)
        self.x0, self.v0 = (xp.asarray(np.linspace(a, b, n)) for a, b in ((0.5, 1.5), (-0.2, 0.3)))
        self.p = dict(x=self.x0.copy(), v=self.v0.copy(), t=xp.zeros(n), corr=xp.zeros(n),
                      t_final=xp.asarray(np.linspace(0.5, 2.0, n)))
        self.mask = xp.zeros(n, bool)
        tt = self.tt = lambda a: torch.as_tensor(a, device=dev)

        def fwd():
            self.calls += 1
            with torch.no_grad():
                drift = self.calls - (self.fwd_calls or self.calls)    # 0 in the forward
                tt(self.p["corr"])[...] = self.model(tt(self.p["x"]), tt(self.p["v"])) + (
                    0.1 * drift if counter else 0.0)
        self.node = opaque(around(fwd) if around else fwd,
                           reads={"x": self.p["x"], "v": self.p["v"]},
                           writes={"corr": self.p["corr"]}, name="model", **(node_kw or {}))
        if plan is None:        # through eagle.simulation: the step list as given
            self.runner = eagle.simulation([self.node, inj_osc], max_steps=400, record=True,
                                           window=16, dt=DT, c=C, **self.p)
        else:
            self.runner = eagle.until_done([self.node, plan], max_steps=400, record=True,
                                           window=16, dt=DT, c=C, terminated=self.mask, **kw,
                                           **self.p)
        self.rep = self.runner.run()
        self.fwd_calls = self.calls

    def vjp(self, primals, cots):
        torch, tt = self.torch, self.tt
        x, v = (tt(primals[k]).detach().requires_grad_(True) for k in ("x", "v"))
        g = torch.autograd.grad(self.model(x, v), [x, v, *self.params], tt(cots["corr"]))
        for p, gp in zip(self.params, g[2:]):
            p.grad = gp if p.grad is None else p.grad + gp
        return {"x": g[0], "v": g[1]}

    def reverse(self, vjp=None, **kw):
        for p in self.params:
            p.grad = None
        g = eagle.diff.reverse(self.runner.record, {"x": self.xp.ones_like(self.x0)},
                               vjps={"model": vjp or self.vjp}, **kw)
        return {k: _host(g[k]) for k in ("x", "v", "c")}, [p.grad for p in self.params]

    def reference(self):
        """The same rollout in pure torch autograd, each sample stopped at its step."""
        torch = self.torch
        xs, vs = (torch.as_tensor(_host(a), device=self.dev).requires_grad_(True)
                  for a in (self.x0, self.v0))
        c = torch.tensor(C, dtype=torch.float64, device=self.dev, requires_grad=True)
        x, v, st = xs, vs, torch.as_tensor(self.runner.record.steps, device=self.dev)
        for p in self.params:
            p.grad = None
        for j in range(int(st.max())):
            on = st > j
            x, v = (torch.where(on, x + DT * v, x),
                    torch.where(on, v + DT * (-x - c * v + self.model(x, v) * v), v))
        x.sum().backward()
        return ({"x": xs.grad.cpu().numpy(), "v": vs.grad.cpu().numpy(), "c": c.grad.cpu().numpy()},
                [p.grad.clone() for p in self.params])


def _host(a):
    if hasattr(a, "detach"):
        return a.detach().cpu().numpy()
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _rel(a, b):
    return max(float(np.abs(np.asarray(_host(x)) - np.asarray(_host(y))).max()
                     / max(np.abs(np.asarray(_host(y))).max(), 1e-300)) for x, y in zip(a, b))


def _run_vjp_row(torch, xp, dev, targets, tmp, monkeypatch):
    plan, cplan = (eagle.plan.auto([k], targets=targets, cache_dir=str(tmp / k.name))[0]
                   for k in (inj_osc, inj_osc_c))
    run = _RunVjp(torch, xp, dev, plan)
    assert run.rep.steps >= 100 and len(run.runner.record._tape.checkpoints) >= 5
    assert run.calls == run.rep.launches + 1     # once before each launch, once after the last
    end = {k: _host(a).copy() for k, a in (*run.p.items(), ("mask", run.mask))}
    want, want_p = run.reference()
    got, got_p = run.reverse()
    assert _rel([got[k] for k in want], list(want.values())) < 1e-12
    assert _rel(got_p, want_p) < 1e-12
    for k, a in (*run.p.items(), ("mask", run.mask)):       # the node's planes, restored
        np.testing.assert_array_equal(_host(a), end[k], err_msg=k)
    syncs, t0 = _counters.snapshot()["host_syncs"], time.perf_counter()
    eagle.diff.reverse(run.runner.record, {"x": xp.ones_like(run.x0)}, vjps={"model": run.vjp})
    wall = time.perf_counter() - t0
    assert _rel([p.grad for p in run.params], [2 * g for g in got_p]) < 1e-12   # accumulate
    steps = int(run.runner.record.steps.max())      # the reverse's steps (the device
    if xp is not np:                                # forward pads to the window)
        assert _counters.snapshot()["host_syncs"] - syncs == 2 * steps
        if os.environ.get("EAGLE_WALL_CLOCK_ROWS"):   # a wall-time bound: opt-in, like every wall_clock row
            t0 = time.perf_counter()
            run.vjp({"x": run.x0, "v": run.v0}, {"corr": run.x0})
            torch.cuda.synchronize()
            assert wall <= steps * (time.perf_counter() - t0 + 1e-3)
    bare = eagle.until_done(plan, max_steps=400, record=True, window=16, dt=DT, c=C,
                            terminated=xp.zeros(512, bool), x=run.x0.copy(), v=run.v0.copy(),
                            t=xp.zeros(512), corr=xp.zeros(512), t_final=run.p["t_final"])
    bare.run()
    assert bare.record.nbytes == run.runner.record.nbytes
    # compaction arm: same gradients (ULP contract), the hook still once per launch
    comp = _RunVjp(torch, xp, dev, cplan, every=16, reorder=0.5)
    assert comp.calls == comp.rep.launches + 1
    assert _rel([comp.reverse()[0][k] for k in want], list(want.values())) < 1e-12
    sim = _RunVjp(torch, xp, dev, None)
    assert _rel([sim.reverse()[0][k] for k in want], list(want.values())) < 1e-12
    with pytest.raises(ValueError, match="comes after the plan in a recorded step"):
        eagle.simulation([inj_osc, run.node], max_steps=4, record=True, dt=DT, c=C, **run.p)
    # budget-forced passes: eagle's gradients bitwise those of one pass
    big = _RunVjp(torch, xp, dev, plan, n=1024)
    one = big.reverse()[0]
    rec = big.runner._recorder
    small = _RunVjp(torch, xp, dev, plan, n=1024, memory_budget=300 * rec.per_sample_bytes())
    many = small.reverse()[0]
    for k in one:
        np.testing.assert_array_equal(many[k], one[k], err_msg=k)
    # refusals, by name
    bad = dict(dt=DT, c=C, terminated=run.mask, **run.p)
    with pytest.raises(ValueError, match="comes after the plan in a recorded step"):
        eagle.until_done([plan, run.node], max_steps=4, record=True, **bad)
    other = opaque(lambda: None, reads=[run.p["corr"]], writes={"t_final": run.p["t_final"]})
    with pytest.raises(ValueError, match="node-to-node planes inside a recorded step"):
        eagle.until_done([run.node, other, plan], max_steps=4, record=True, **bad)
    twin = opaque(lambda: None, reads={"x": run.p["x"]}, writes={"t_final": run.p["t_final"]},
                  name="model")
    with pytest.raises(ValueError, match="labelled 'model'; give each node a name="):
        eagle.until_done([run.node, twin, plan], max_steps=4, record=True, **bad)
    par = opaque(lambda: None, reads={"dt": xp.zeros(1)}, writes={"corr": run.p["corr"]})
    with pytest.raises(ValueError, match="reads parameter 'dt'"):
        eagle.until_done([par, plan], max_steps=4, record=True, **bad)
    with pytest.raises(ValueError, match="the refine entry re-steps the last step"):
        eagle.until_done([run.node, plan], max_steps=4, events="refine", **bad)
    rec = run.runner.record
    with pytest.raises(ValueError, match="a captured backward as a run's node vjp"):
        eagle.diff.reverse(rec, {"x": xp.ones_like(run.x0)}, vjps={"model": run.node})
    with pytest.raises(ValueError, match=r"node 'model' is opaque: pass vjps="):
        eagle.diff.reverse(rec, {"x": xp.ones_like(run.x0)})
    with pytest.raises(ValueError, match="returned no gradient for 'v'"):
        run.reverse(lambda p, c: {"x": run.vjp(p, c)["x"]})
    with pytest.raises(ValueError, match="returned 'x' of shape"):
        run.reverse(lambda p, c: {**run.vjp(p, c), "x": torch.zeros(3, dtype=torch.float64)})
    # a second concurrent reverse -> Busy; the first's gradients unchanged
    gate, errors, out = threading.Event(), [], {}

    def slow(p, c):
        gate.wait(10)
        return run.vjp(p, c)

    def unrelated():        # the reader's other torch work, on its own stream
        side = torch.cuda.Stream() if xp is not np else None
        with torch.cuda.stream(side) if side else contextlib.nullcontext():
            a = torch.ones(256, 256, dtype=torch.float64, device=dev)
            for _ in range(50):
                a = (a @ a) / 256
    first = threading.Thread(target=lambda: out.update(g=run.reverse(slow)[0]))
    other_work = threading.Thread(target=unrelated)
    first.start()
    other_work.start()
    time.sleep(0.2)
    try:
        run.reverse()
    except eagle.Busy as exc:
        errors.append(str(exc))
    gate.set()
    first.join()
    other_work.join()
    assert errors and "re-running node 'model' over your planes" in errors[0]
    assert _rel([out["g"][k] for k in want], list(want.values())) < 1e-12
    # falsifiers: (a) the node's share dropped; (b) an impure node; (c) no ring row copy
    zeros = lambda p, c: {k: run.torch.zeros_like(run.tt(p[k])) for k in p}  # noqa: E731
    assert _rel([run.reverse(zeros)[0]["x"]], [want["x"]]) > 1e-2
    impure = _RunVjp(torch, xp, dev, plan, counter=True)
    assert _rel([impure.reverse()[0]["x"]], [got["x"]]) > 1e-3
    from eagle.diff import _run

    monkeypatch.setattr(_run.RunReverse, "_ring_row", lambda self, j: None)
    assert _rel([run.reverse()[0]["x"]], [want["x"]]) > 1e-9


@register_row("INJ-RUN-VJP", leg="cpu")
def test_run_vjps_host(tmp_path, monkeypatch):
    certified_framework("torch")
    import torch

    _run_vjp_row(torch, np, "cpu", ("host",), tmp_path, monkeypatch)


@register_row("INJ-RUN-VJP", leg="cuda")
def test_run_vjps_device(tmp_path, monkeypatch):
    torch = _torch_cuda()
    _run_vjp_row(torch, _cp(), _dev(torch), ("cuda",), tmp_path, monkeypatch)


def _tier_arm(torch, xp, dev, plan, **kw):
    """One recorded run + reverse: ``(forward state, planes' gradients, params' gradients)``."""
    run = _RunVjp(torch, xp, dev, plan, **kw)
    got, got_p = run.reverse()
    return {k: _host(run.p[k]).copy() for k in ("x", "v", "t")}, got, got_p


def _same_arm(a, b):
    for k in a[0]:
        np.testing.assert_array_equal(a[0][k], b[0][k], err_msg=k)
    assert _rel([b[1][k] for k in a[1]], list(a[1].values())) < 1e-12
    assert _rel(b[2], a[2]) < 1e-12


def test_run_node_tiers_device(tmp_path, monkeypatch):
    """A recorded run takes a stream-ordered or host node: forward state bitwise and
    gradients within the row's 1e-12 ULP contract those of the captured node; a busy
    separate stream is still ordered by the fences."""
    torch = _torch_cuda()
    xp, dev = _cp(), _dev(torch)
    plan = eagle.plan.auto([inj_osc], targets=("cuda",), cache_dir=str(tmp_path / "p"))[0]
    base = _tier_arm(torch, xp, dev, plan)
    _same_arm(base, _tier_arm(torch, xp, dev, plan,
                              node_kw=dict(stream=torch.cuda.current_stream())))
    _same_arm(base, _tier_arm(torch, xp, dev, plan, node_kw=dict(eager=True)))
    side = torch.cuda.Stream()

    def busy(fn):
        def run():
            with torch.cuda.stream(side):
                torch.cuda._sleep(3_000_000)
                fn()
        return run
    on_side = dict(node_kw=dict(stream=side), around=busy)
    _same_arm(base, _tier_arm(torch, xp, dev, plan, **on_side))
    # the fence after the node: without it the plan reads corr before the busy stream wrote it
    real = G._Eager.run

    def no_wait(self, waits):
        real(self, waits)
        return _cp().cuda.Event()       # never recorded: waiting on it waits for nothing
    monkeypatch.setattr(G._Eager, "run", no_wait)
    cut = _tier_arm(torch, xp, dev, plan, **on_side)
    torch.cuda.synchronize()
    assert _rel([cut[0]["x"]], [base[0]["x"]]) > 1e-6
    monkeypatch.undo()
    node = opaque(lambda: None, stream=torch.cuda.current_stream(), name="model")
    with pytest.raises(ValueError, match="pass record=True"):
        eagle.until_done([node, plan], max_steps=4, dt=DT, c=C, terminated=xp.zeros(4, bool),
                         x=xp.zeros(4), v=xp.zeros(4), t=xp.zeros(4), corr=xp.zeros(4),
                         t_final=xp.ones(4))


def test_run_node_host_tier_cpu(tmp_path):
    """On the CPU an ``eager=True`` node in a recorded run gives the captured node's results."""
    certified_framework("torch")
    import torch

    plan = eagle.plan.auto([inj_osc], targets=("host",), cache_dir=str(tmp_path / "p"))[0]
    _same_arm(_tier_arm(torch, np, "cpu", plan), _tier_arm(torch, np, "cpu", plan,
                                                           node_kw=dict(eager=True)))
