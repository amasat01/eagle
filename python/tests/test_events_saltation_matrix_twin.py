# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The saltation twin of a refined step: the RK4 oscillator stopping at the
moving surface x = a t, one refined step from an extended state (x, v, t).
The Jacobian of the refined end state by reverse mode equals the discrete
projector (I - Phi_tau g^T / (g . Phi_tau)) Phi_x over the extended state
(the same algebra, so a round-off bound), and its distance to the
continuous saltation form (Phi_tau replaced by the vector field at x*)
shrinks by at least 2^4 from h to h/2, the RK4 order (the step's crossing at
the same fraction of h in both)."""

from __future__ import annotations

import numpy as np
import pytest

import eagle
import eagle.diff
from _events_fixtures import BACKENDS, ev_osc, host, xp_for

W, A = 2.0, 0.3
MAT = np.array([[0.0, 1.0], [-W * W, 0.0]])
G_X = np.array([1.0, 0.0, -A])
N = 4


def _poly(z, last):
    out, term = np.eye(2), np.eye(2)
    for k in range(1, last + 1):
        term = term @ z / k
        out = out + term
    return out


def _crossing():
    """The continuous crossing time of x = cos 2t with x = a t (bisection)."""
    lo, hi = 0.5, 0.8
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if np.cos(2 * mid) - A * mid > 0 else (lo, mid)
    return 0.5 * (lo + hi)


def _refined_jacobian(xp, h, s0):
    """``(J, tau*, x*)`` from reverse mode: one refined step from the exact
    trajectory's state at ``s0``."""
    x0 = np.array([np.cos(2 * s0), -2 * np.sin(2 * s0)])
    res = eagle.simulate(ev_osc, dt=h, w=W, a=A, x=xp.full(N, x0[0]), v=xp.full(N, x0[1]),
                         t=xp.full(N, s0), max_steps=1, events="refine", record=True)
    assert res.record.steps.tolist() == [1] * N
    rows = []
    for out in ("x", "v", "t"):
        g = eagle.diff.reverse(res.record, {out: xp.ones(N)})
        rows.append([host(g[k])[0] for k in ("x", "v", "t")])
    return np.array(rows), float(host(res.t_stop)[0]) - s0, x0


def _forms(tau, x0):
    phi_x = np.eye(3)
    phi_x[:2, :2] = _poly(tau * MAT, 4)
    x_star = phi_x[:2, :2] @ x0
    phi_tau = np.r_[MAT @ _poly(tau * MAT, 3) @ x0, 1.0]
    field = np.r_[MAT @ x_star, 1.0]
    project = lambda d: (np.eye(3) - np.outer(d, G_X) / (G_X @ d)) @ phi_x
    return project(phi_tau), project(field)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reverse_mode_is_the_discrete_projector(backend):
    xp = xp_for(backend)
    h = 0.1
    jac, tau, x0 = _refined_jacobian(xp, h, _crossing() - 0.6 * h)
    discrete, _ = _forms(tau, x0)
    assert 0 < tau < h
    np.testing.assert_allclose(jac, discrete, rtol=0, atol=64 * np.finfo(float).eps * 4)


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_continuous_form_is_approached_at_the_rk4_order(backend):
    xp = xp_for(backend)
    gaps = []
    for h in (0.2, 0.1):
        jac, tau, x0 = _refined_jacobian(xp, h, _crossing() - 0.6 * h)
        _, continuous = _forms(tau, x0)
        gaps.append(np.abs(jac - continuous).max())
    assert gaps[0] / gaps[1] >= 2 ** 4, gaps
