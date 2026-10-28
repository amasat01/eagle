# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Shared models for the recorded-run reverse rows: a damped oscillator
(linear, with an analytic adjoint) and a damped pendulum (nonlinear), both
explicit Euler, each sample stopping at its own ``t_final``."""

from __future__ import annotations

import math

import numpy as np
import pytest

import eagle
import eagle.diff

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar, Terminated  # noqa: E402
from hawk import math as hm  # noqa: E402

N = 64
DT = 0.01
C = 0.3
MAX_STEPS = 400


@hawk.kernel(steps=1)
def rr_osc(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
           x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    a = -x - c * v
    vn = v + dt * a
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@hawk.kernel(steps=1)
def rr_pend(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
            x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    a = -hm.sin(x) - c * v
    vn = v + dt * a
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


@hawk.kernel
def rr_pend_auto(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
                 x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    a = -hm.sin(x) - c * v
    vn = v + dt * a
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


def xp_for(backend):
    return pytest.importorskip("cupy") if backend == "gpu" else np


def host(a):
    return np.asarray(a.get() if hasattr(a, "get") else a)


def inputs(xp, n=N, dtype="float64"):
    """``(x0, v0, t_final)``: every sample stops at a different step."""
    x0 = xp.asarray(np.linspace(0.5, 1.5, n), dtype=dtype)
    v0 = xp.asarray(np.linspace(-0.2, 0.3, n), dtype=dtype)
    tf = xp.asarray(np.linspace(0.5, 3.0, n), dtype=dtype)
    return x0, v0, tf


def simulation(model, x0, v0, tf, *, c=C, window=None, record=True, dtype="float64"):
    xp = type(x0).__module__.split(".")[0] == "cupy" and xp_for("gpu") or np
    return eagle.simulation(model, dt=DT, c=c, t_final=tf, x=x0.copy(), v=v0.copy(),
                            t=xp.zeros_like(x0), max_steps=MAX_STEPS, record=record,
                            window=window, scalar_type=dtype)


def end_x(model, x0, v0, tf, c=C):
    sim = simulation(model, x0, v0, tf, c=c, record=False)
    return host(sim.run().x)


def central_difference(model, x0, v0, tf, h=1e-6):
    """``(dJ/dx0, dJ/dv0, dJ/dc)`` of ``J = sum(x_end)``."""
    dx = (end_x(model, x0 + h, v0, tf) - end_x(model, x0 - h, v0, tf)) / (2 * h)
    dv = (end_x(model, x0, v0 + h, tf) - end_x(model, x0, v0 - h, tf)) / (2 * h)
    dc = float((end_x(model, x0, v0, tf, C + h) - end_x(model, x0, v0, tf, C - h)).sum()
               / (2 * h))
    return dx, dv, dc


def analytic_osc(steps):
    """The adjoint of ``J = x_end`` through ``s`` explicit Euler steps of the
    damped oscillator: ``(M^T)^s e_x``, per sample."""
    m = np.array([[1.0, DT], [-DT, 1.0 - C * DT]])
    out = np.empty((2, len(steps)))
    for i, s in enumerate(steps):
        out[:, i] = np.linalg.matrix_power(m.T, int(s)) @ np.array([1.0, 0.0])
    return out


def default_window(steps):
    return min(max(math.isqrt(steps - 1) + 1, 16), 1024)
