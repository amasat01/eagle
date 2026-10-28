# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Shared models for the ``events="refine"`` rows: an RK4 oscillator on a
moving surface ``x = a t`` (and on ``x = 0``), a fixed horizon ``t = T``,
exact free fall and rise (the bouncing ball's phases), a bounce reset, and
an RK4 damped pendulum hitting a moving wall ``theta = a t``."""

from __future__ import annotations

import numpy as np
import pytest

import eagle
import eagle.diff

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar, Terminated  # noqa: E402
from hawk import math as hm  # noqa: E402

GPU = pytest.param("gpu", id="gpu", marks=pytest.mark.gpu)
BACKENDS = [pytest.param("cpu", id="cpu"), GPU]
GRAVITY = 9.81


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_osc(dt: Param, w: Param, a: Param, terminated: Terminated,
           x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    w2 = w * w
    k1x = v
    k1v = -w2 * x
    k2x = v + 0.5 * dt * k1v
    k2v = -w2 * (x + 0.5 * dt * k1x)
    k3x = v + 0.5 * dt * k2v
    k3v = -w2 * (x + 0.5 * dt * k2x)
    k4x = v + dt * k3v
    k4v = -w2 * (x + dt * k3x)
    xn = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    vn = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    x = xn
    v = vn
    tn = t + dt
    t = tn
    terminated = xn <= a * tn


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_horizon(dt: Param, w: Param, big_t: Param, terminated: Terminated,
               x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    vn = v - dt * w * w * x
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= big_t


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_fall(dt: Param, g: Param, terminated: Terminated,
            y: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    yn = y + dt * v - 0.5 * g * dt * dt
    v = v - g * dt
    y = yn
    t = t + dt
    terminated = yn < 0.0


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_rise(dt: Param, g: Param, terminated: Terminated,
            y: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    vn = v - g * dt
    y = y + dt * v - 0.5 * g * dt * dt
    v = vn
    t = t + dt
    terminated = vn <= 0.0


@hawk.kernel
def ev_bounce(e: Scalar, v_in: Scalar, v: Mutable[Scalar]):
    v = -e * v_in


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_pend(dt: Param, c: Param, a: Param, terminated: Terminated,
            x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    k1x = v
    k1v = -hm.sin(x) - c * v
    k2x = v + 0.5 * dt * k1v
    k2v = -hm.sin(x + 0.5 * dt * k1x) - c * k2x
    k3x = v + 0.5 * dt * k2v
    k3v = -hm.sin(x + 0.5 * dt * k2x) - c * k3x
    k4x = v + dt * k3v
    k4v = -hm.sin(x + dt * k3x) - c * k4x
    xn = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    vn = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    x = xn
    v = vn
    tn = t + dt
    t = tn
    terminated = xn <= a * tn


def xp_for(backend):
    return pytest.importorskip("cupy") if backend == "gpu" else np


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def osc(xp, n, *, w=2.0, a=0.3, dt=0.1, x0=1.0, record=True, model=ev_osc, **kw):
    """The moving-surface oscillator from ``x0`` at rest: ``(result, x0 plane)``."""
    x = xp.full(n, x0)
    res = eagle.simulate(model, dt=dt, w=w, a=a, x=x, v=xp.zeros(n), t=xp.zeros(n),
                         max_steps=200, events="refine", record=record, **kw)
    return res


def fall(xp, y0, *, dt=0.01, record=True, model=None, **kw):
    """Free fall from ``y0`` at rest, refined at ``y = 0``."""
    return eagle.simulate(model or ev_fall, dt=dt, g=GRAVITY, y=xp.asarray(y0, dtype="float64").copy(),
                          v=xp.zeros(len(y0)), t=xp.zeros(len(y0)), max_steps=2000,
                          events="refine", record=record, **kw)


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_grid(dt: Scalar, big_t: Scalar, terminated: Terminated,
            x: Mutable[Scalar], t: Mutable[Scalar]):
    x = x + dt
    tn = t + dt
    t = tn
    terminated = tn >= big_t


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_pend_rise(dt: Param, c: Param, a: Param, terminated: Terminated,
                 x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    k1x = v
    k1v = -hm.sin(x) - c * v
    k2x = v + 0.5 * dt * k1v
    k2v = -hm.sin(x + 0.5 * dt * k1x) - c * k2x
    k3x = v + 0.5 * dt * k2v
    k3v = -hm.sin(x + 0.5 * dt * k2x) - c * k3x
    k4x = v + dt * k3v
    k4v = -hm.sin(x + dt * k3x) - c * k4x
    xn = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    vn = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    x = xn
    v = vn
    t = t + dt
    terminated = vn <= a


@hawk.kernel(time="t", time_step="dt")
def ev_fall_auto(dt: Param, g: Param, terminated: Terminated,
                 y: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    """``ev_fall`` as an automatic kernel: several steps per launch."""
    yn = y + dt * v - 0.5 * g * dt * dt
    v = v - g * dt
    y = yn
    t = t + dt
    terminated = yn < 0.0


from hawk.ext import Guard, Kind  # noqa: E402


def _pend_body(dt: Param, c: Param, a: Param, terminated: Terminated,
               x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    k1x = v
    k1v = -hm.sin(x) - c * v
    k2x = v + 0.5 * dt * k1v
    k2v = -hm.sin(x + 0.5 * dt * k1x) - c * k2x
    k3x = v + 0.5 * dt * k2v
    k3v = -hm.sin(x + 0.5 * dt * k2x) - c * k3x
    k4x = v + dt * k3v
    k4v = -hm.sin(x + dt * k3x) - c * k4x
    xn = x + dt / 6.0 * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    vn = v + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    x = xn
    v = vn
    tn = t + dt
    t = tn
    terminated = xn <= a * tn


#: ``ev_pend`` reading an active set: a compacting (and reordering) run.
ev_pend_map = hawk.kernel(_pend_body, steps=1, time="t", time_step="dt",
                          kind=Kind("ev_map", guard=Guard(active_set=True)))


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_throw(dt: Param, g: Param, terminated: Terminated, x: Mutable[Scalar],
             u: Mutable[Scalar], y: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    """A ball thrown sideways at speed ``u``, falling to ``y = 0``."""
    yn = y + dt * v - 0.5 * g * dt * dt
    x = x + dt * u
    u = u + 0.0 * dt
    v = v - g * dt
    y = yn
    t = t + dt
    terminated = yn < 0.0


def chain_bounces(xp, y0, e, bounces, *, dt=0.05, fall=None, rise=None, seed="y",
                  state=None, **params):
    """Fall, bounce (``v -> -e v``) and rise ``bounces`` times by running
    each refined phase and chaining the reverses by hand (the bounce's own
    adjoint is ``bar v_in = -e bar v``, ``bar e = -v_in bar v``): ``(end
    state, {input: gradient of sum(end[seed])})``; ``seed`` may instead be a
    callable of the end state returning the seeds. Works on either backend."""
    fall, rise = fall or ev_fall, rise or ev_rise
    params = params or {"g": GRAVITY}
    st = dict(state or {"y": y0.copy(), "v": xp.zeros_like(y0), "t": xp.zeros_like(y0)})
    runs = []
    for _ in range(bounces):
        down = eagle.simulate(fall, dt=dt, max_steps=4000, events="refine", record=True,
                              **params, **{k: a.copy() for k, a in st.items()})
        st = {k: down[k].copy() for k in st}
        v_in = st["v"].copy()
        st["v"] = -e * v_in
        up = eagle.simulate(rise, dt=dt, max_steps=4000, events="refine", record=True,
                            **params, **{k: a.copy() for k, a in st.items()})
        runs.append((down.record, v_in, up.record))
        st = {k: up[k].copy() for k in st}
    bar = seed(st) if callable(seed) else {seed: xp.ones_like(y0)}
    bar_e = xp.zeros_like(y0)
    for down, v_in, up in reversed(runs):
        g = eagle.diff.reverse(up, bar)
        bar = {k: g[k] for k in st}
        bar_e = bar_e - v_in * bar["v"]
        bar["v"] = -e * bar["v"]
        g = eagle.diff.reverse(down, bar)
        bar = {k: g[k] for k in st}
    bar["e"] = bar_e
    return st, bar
