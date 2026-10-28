# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Every launch shape a recorded run takes gives the gradients of the
single-step, full-tape oracle: an automatic kernel (fused launches clamped to
window boundaries, replayed by the same fused entry), a host run whatever its
host pacing, and a time_step= kernel (the time plane in index form); each is
bitwise stable across windows and FD-checked on a nonlinear model."""

from __future__ import annotations

import numpy as np
import pytest

import eagle
import eagle.diff
from _run_reverse_fixtures import (C, DT, MAX_STEPS, N, host, inputs, rr_pend, rr_pend_auto,
                                   simulation, xp_for)

hawk = pytest.importorskip("hawk")
from hawk import Param, Scalar, Terminated  # noqa: E402
from hawk import math as hm  # noqa: E402

SEEDS = {"x": 1.0, "v": 0.5}


def _seeds(xp, n=N):
    return {k: xp.full(n, v) for k, v in SEEDS.items()}


def _grads(model, window, backend="cpu", policy=None):
    xp = xp_for(backend)
    sim = simulation(model, *inputs(xp), window=window)
    if policy is not None:
        sim.runner.host_policy = policy
    sim.run()
    return sim, {k: host(v) for k, v in eagle.diff.reverse(sim.record, _seeds(xp)).items()}


def _close(a, b, rtol=1e-12):
    if not isinstance(b, dict):
        a, b = {"value": a}, {"value": b}
    for k in b:
        scale = max(float(np.max(np.abs(b[k]))), 1e-300)
        np.testing.assert_allclose(a[k], b[k], rtol=rtol, atol=rtol * scale, err_msg=k)


def _bitwise(a, b):
    for k in b:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("window", [16, 37, "full"])
def test_automatic_kernel_fused_launches(window):
    sim, got = _grads(rr_pend_auto, window)
    assert sim.runner._recorder.fused and sim.runner._recorder.max_k > 1
    report_launches = sim.runner._recorder._launch_count(int(host(sim.record.steps).max()))
    assert report_launches < int(host(sim.record.steps).max())      # fewer launches than steps
    _close(got, _grads(rr_pend, "full")[1])
    _bitwise(got, _grads(rr_pend_auto, 16)[1])


@pytest.mark.parametrize("policy", ["auto", "tiled", "measured", "band"])
def test_host_pacing_does_not_change_a_recorded_run(policy):
    _bitwise(_grads(rr_pend_auto, 37, policy=policy)[1], _grads(rr_pend_auto, 37)[1])


def test_automatic_kernel_matches_central_differences():
    _sim, got = _grads(rr_pend_auto, None)
    h = 1e-6
    x0, v0, tf = inputs(np)

    def end(x):
        res = simulation(rr_pend_auto, x, v0, tf, record=False).run()
        return host(res.x) + 0.5 * host(res.v)

    fd = (end(x0 + h) - end(x0 - h)) / (2 * h)
    _close(got["x"], fd, rtol=1e-6)


def _timed(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
           x: hawk.Mutable[Scalar], v: hawk.Mutable[Scalar], t: hawk.Mutable[Scalar]):
    a = -hm.sin(x) - c * v + 0.3 * hm.cos(2.0 * t)
    vn = v + dt * a
    x = x + dt * v
    v = vn
    t = t + dt
    terminated = t >= t_final


rr_timed = hawk.kernel(_timed, steps=1, time="t", time_step="dt")


def _timed_inputs(xp):
    """Stops off the step grid: the record runs the time plane in index form
    (t0 + k dt), the plain kernel accumulates it, and a stop exactly on a grid
    point may round to different steps in the two."""
    x0, v0, tf = inputs(xp)
    return x0, v0, tf + 0.0047


def _timed_grads(window, backend="cpu"):
    xp = xp_for(backend)
    sim = simulation(rr_timed, *_timed_inputs(xp), window=window)
    sim.run()
    return sim, {k: host(v) for k, v in eagle.diff.reverse(sim.record, _seeds(xp)).items()}


def test_time_step_kernel_matches_central_differences():
    sim, got = _timed_grads(37)
    assert sim.runner._recorder.t0 == "t_0"
    assert sorted(got) == ["c", "dt", "t", "v", "x"]
    h = 1e-6
    x0, v0, tf = _timed_inputs(np)

    def end(x=x0, v=v0, t=None, c=C):
        t = np.zeros(N) if t is None else t
        res = eagle.simulation(rr_timed, dt=DT, c=c, t_final=tf, x=x.copy(), v=v.copy(),
                               t=t.copy(), max_steps=MAX_STEPS).run()
        return host(res.x) + 0.5 * host(res.v)

    _close(got["x"], (end(x=x0 + h) - end(x=x0 - h)) / (2 * h), rtol=1e-6)
    _close(got["v"], (end(v=v0 + h) - end(v=v0 - h)) / (2 * h), rtol=1e-6)
    assert float(got["c"]) == pytest.approx(float(((end(c=C + h) - end(c=C - h)) / (2 * h)).sum()),
                                            rel=1e-6)
    # t shifts the forcing phase; the stop rule reads t too, so shift less than one
    # step's slack: FD of the forcing only (every sample keeps its stop step)
    ht = 1e-7
    fd_t = (end(t=np.full(N, ht)) - end(t=np.full(N, -ht))) / (2 * ht)
    _close(got["t"], fd_t, rtol=1e-5)


def test_time_step_kernel_windows_are_bitwise():
    _bitwise(_timed_grads(16)[1], _timed_grads("full")[1])


@pytest.mark.gpu
@pytest.mark.parametrize("window", [16, 37])
def test_automatic_kernel_gpu_matches_host(window):
    _sim, dev = _grads(rr_pend_auto, window, backend="gpu")
    _close(dev, _grads(rr_pend_auto, window)[1])
    _bitwise(dev, _grads(rr_pend_auto, "full", backend="gpu")[1])


@pytest.mark.gpu
def test_time_step_kernel_gpu_matches_host():
    _sim, dev = _timed_grads(37, backend="gpu")
    _close(dev, _timed_grads(37)[1])
    _bitwise(dev, _timed_grads("full", backend="gpu")[1])


from hawk.ext import Guard, Kind  # noqa: E402

_MAP = Kind("rr_map", guard=Guard(active_set=True))
def _pend_body(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
               x: hawk.Mutable[Scalar], v: hawk.Mutable[Scalar], t: hawk.Mutable[Scalar]):
    a = -hm.sin(x) - c * v
    vn = v + dt * a
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


rr_pend_map = hawk.kernel(_pend_body, steps=1, kind=_MAP)
rr_pend_k4 = hawk.kernel(_pend_body, steps=4)
rr_pend_map_auto = hawk.kernel(_pend_body, kind=Kind("rr_map_auto", guard=Guard(active_set=True)))


def _scattered(xp, n):
    """Inputs whose stops are scattered over the batch, so finished samples
    leave holes in every warp and a reordering set does reorder."""
    x0, v0, tf = inputs(xp, n=n)
    perm = xp.asarray(np.random.default_rng(7).permutation(n))
    return x0, v0, tf[perm]


def _run(model, window, *, n=N, every=None, reorder=None, backend="cpu", record=True):
    xp = xp_for(backend)
    x0, v0, tf = _scattered(xp, n)
    kw = dict(every=every, reorder=reorder) if (every or reorder) else {}
    sim = eagle.simulation(model, dt=DT, c=C, t_final=tf, x=x0.copy(), v=v0.copy(),
                           t=xp.zeros_like(x0), max_steps=MAX_STEPS, record=True,
                           window=window, **kw)
    sim.run()
    grads = eagle.diff.reverse(sim.record, _seeds(xp, n))
    return sim, {k: host(v) for k, v in grads.items()}


@pytest.mark.parametrize("every", [None, 4])          # 4: the smallest cadence eagle takes
def test_compacting_run_matches_the_oracle(every):
    oracle = _run(rr_pend, "full")[1]
    got = _run(rr_pend_map, 37, every=every)[1]
    _close(got, oracle)
    _bitwise(got, _run(rr_pend_map, "full", every=every)[1])


def test_reordering_run_matches_the_oracle():
    n = 4096                                           # the smallest span a reorder fires on
    sim, got = _run(rr_pend_map, 37, n=n, every=4, reorder=0.5)
    epochs = [c["slot_samples"] for c in sim.record._tape.checkpoints]
    assert any((host(e) != np.arange(n)).any() for e in epochs)   # it did reorder
    _close(got, _run(rr_pend, "full", n=n)[1])
    _bitwise(got, _run(rr_pend_map, "full", n=n, every=4, reorder=0.5)[1])


@pytest.mark.parametrize("window", [16, 37])
def test_four_steps_per_launch_matches_the_oracle(window):
    sim, got = _run(rr_pend_k4, window)
    rec = sim.runner._recorder
    assert rec.fused and rec.max_k == 4
    assert rec._launch_count(int(host(sim.record.steps).max())) < int(host(sim.record.steps).max())
    _close(got, _run(rr_pend, "full")[1])
    _bitwise(got, _run(rr_pend_k4, "full")[1])


def test_four_steps_per_launch_matches_central_differences():
    _sim, got = _run(rr_pend_k4, 37)
    h = 1e-6
    x0, v0, tf = _scattered(np, N)

    def end(x):
        res = eagle.simulation(rr_pend, dt=DT, c=C, t_final=tf, x=x.copy(), v=v0.copy(),
                               t=np.zeros(N), max_steps=MAX_STEPS).run()
        return host(res.x) + 0.5 * host(res.v)

    _close(got["x"], (end(x0 + h) - end(x0 - h)) / (2 * h), rtol=1e-6)


def test_compacting_automatic_run_matches_the_oracle():
    _close(_run(rr_pend_map_auto, 16)[1], _run(rr_pend, "full")[1])


@pytest.mark.gpu
def test_four_steps_per_launch_gpu_matches_host():
    dev = _run(rr_pend_k4, 37, backend="gpu")[1]
    _close(dev, _run(rr_pend_k4, 37)[1])
    _bitwise(dev, _run(rr_pend_k4, "full", backend="gpu")[1])


@pytest.mark.gpu
@pytest.mark.parametrize("reorder", [None, 0.5])
def test_compacting_run_gpu_matches_host(reorder):
    n = 4096
    dev = _run(rr_pend_map, 37, n=n, every=4, reorder=reorder, backend="gpu")[1]
    _close(dev, _run(rr_pend_map, 37, n=n, every=4, reorder=reorder)[1])
    _bitwise(dev, _run(rr_pend_map, "full", n=n, every=4, reorder=reorder, backend="gpu")[1])
