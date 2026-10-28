# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The event-time sensitivity of a refined run, by reverse mode:

* moving surface: the RK4 oscillator (w = 2, h = 0.1, from x = 1 at rest)
  stopping at x = a t, a = 0.3: dt*/dw = -0.295819832, dt*/da = -0.302237878
  (a formula without the guard's own a- and t-derivatives misses them by
  15 % and 100 %);
* the same oscillator stopping at x = 0, and an RK4 damped pendulum hitting a
  moving wall, against central differences of the same refined forward
  (bound: the 2nd-order central difference's truncation + round-off);
* fixed horizon t = T: dt*/dT = 1, no graze, and on the step grid T = m h the
  stop is step m with |t* - T| <= 1 ulp(T), for every (h, m) of a sweep.
"""

from __future__ import annotations

import numpy as np
import pytest

import eagle
import eagle.diff
from _events_fixtures import (BACKENDS, ev_grid, ev_horizon, ev_osc, ev_pend, host, osc,
                              xp_for)

N = 8
FD_H = 1e-6
FD_RTOL = 1e-7


@pytest.mark.parametrize("backend", BACKENDS)
def test_moving_surface_reproduces_the_references(backend):
    xp = xp_for(backend)
    res = osc(xp, N)
    assert set(host(res.event_status).tolist()) == {1}
    grads = eagle.diff.reverse(res.record, {"t": xp.ones(N)})
    assert float(grads["w"]) / N == pytest.approx(-0.295819832, abs=5e-10)
    assert float(grads["a"]) / N == pytest.approx(-0.302237878, abs=5e-10)


def _fresh(kw):
    """The run writes its state planes in place: every run gets its own."""
    return {k: (v.copy() if hasattr(v, "copy") else v) for k, v in kw.items()}


def _t_stop(model, **kw):
    res = eagle.simulate(model, max_steps=400, events="refine", **_fresh(kw))
    return host(res.t_stop), host(res.v)


def _fd(model, name, base, **kw):
    plus = _t_stop(model, **{**kw, name: base + FD_H})
    minus = _t_stop(model, **{**kw, name: base - FD_H})
    return [(p - m) / (2 * FD_H) for p, m in zip(plus, minus)]


@pytest.mark.parametrize("model, a", [(ev_osc, 0.0), (ev_pend, 0.05)], ids=["oscillator", "pendulum"])
def test_reverse_matches_central_differences(model, a):
    n = 6
    x0 = np.linspace(0.6, 1.1, n)
    second = "w" if model is ev_osc else "c"
    value = 2.0 if model is ev_osc else 0.2
    kw = dict(dt=0.05, a=a, v=np.zeros(n), t=np.zeros(n), **{second: value})
    res = eagle.simulate(model, max_steps=400, events="refine", record=True,
                         **_fresh(dict(kw, x=x0)))
    for seed in ("t", "v"):
        grads = eagle.diff.reverse(res.record, {seed: np.ones(n)})
        col = 0 if seed == "t" else 1
        want_p = _fd(model, second, value, x=x0, **{k: v for k, v in kw.items() if k != second})
        want_a = _fd(model, "a", a, x=x0, **{k: v for k, v in kw.items() if k != "a"})
        assert float(grads[second]) == pytest.approx(want_p[col].sum(), rel=FD_RTOL, abs=1e-9)
        assert float(grads["a"]) == pytest.approx(want_a[col].sum(), rel=FD_RTOL, abs=1e-9)
        dx = [(p - m) / (2 * FD_H) for p, m in zip(
            _t_stop(model, x=x0 + FD_H, **kw), _t_stop(model, x=x0 - FD_H, **kw))]
        np.testing.assert_allclose(grads["x"], dx[col], rtol=FD_RTOL, atol=1e-9)


@pytest.mark.parametrize("backend", BACKENDS)
def test_fixed_horizon_moves_one_for_one(backend):
    xp = xp_for(backend)
    res = eagle.simulate(ev_horizon, dt=0.1, w=2.0, big_t=1.25, x=xp.ones(N), v=xp.zeros(N),
                         t=xp.zeros(N), max_steps=100, events="refine", record=True)
    np.testing.assert_allclose(host(res.t_stop), 1.25, rtol=0, atol=4 * np.spacing(1.25))
    grads = eagle.diff.reverse(res.record, {"t": xp.ones(N)})
    assert float(grads["big_t"]) / N == pytest.approx(1.0, abs=1e-12)


def _grid_pairs():
    hs = [0.1, 0.01, 0.05, 0.2, 1.0 / 3.0, 0.3, 0.7, 1e-3]
    return [(h, m) for h in hs for m in range(1, 61)]


@pytest.mark.parametrize("backend", BACKENDS)
def test_a_horizon_on_the_step_grid_stops_on_its_step(backend):
    xp = xp_for(backend)
    pairs = _grid_pairs()
    h = np.array([p[0] for p in pairs])
    m = np.array([p[1] for p in pairs])
    big_t = m * h
    n = len(pairs)
    res = eagle.simulate(ev_grid, dt=xp.asarray(h), big_t=xp.asarray(big_t), x=xp.zeros(n),
                         t=xp.zeros(n), max_steps=80, events="refine", record=True)
    assert (host(res.record.steps) == m).all()
    assert (np.abs(host(res.t_stop) - big_t) <= np.spacing(big_t)).all()
    assert set(host(res.event_status).tolist()) <= {1, 2}
    grads = eagle.diff.reverse(res.record, {"t": xp.ones(n)})
    np.testing.assert_allclose(host(grads["big_t"]), 1.0, rtol=0, atol=1e-12)
