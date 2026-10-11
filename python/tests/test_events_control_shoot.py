# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The control example: a ball thrown sideways at speed u from height y0
falls, bounces (v -> -e v) and lands again; gradient descent on u, with the
gradient of J = (x_land - X)^2 / 2 from the refined phases' reverses chained
across the bounce, converges to the closed form u* = X / (t1 (1 + 2 e)),
t1 = sqrt(2 y0 / g), on the host and on the device."""

from __future__ import annotations

import numpy as np
import pytest

from _events_fixtures import BACKENDS, GRAVITY, chain_bounces, ev_throw, host, xp_for

N = 8


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradient_descent_lands_on_the_target(backend):
    xp = xp_for(backend)
    y0 = np.linspace(1.0, 3.0, N)
    e = np.linspace(0.6, 0.8, N)
    target = np.linspace(2.0, 5.0, N)
    t1 = np.sqrt(2 * y0 / GRAVITY)
    slope = t1 * (1 + 2 * e)
    u = np.ones(N)
    for _ in range(30):
        state = {"x": xp.zeros(N), "u": xp.asarray(u), "y": xp.asarray(y0),
                 "v": xp.zeros(N), "t": xp.zeros(N)}
        st, bar = chain_bounces(xp, xp.asarray(y0), xp.asarray(e), 1, fall=ev_throw,
                                rise=ev_throw, state=state,
                                seed=lambda end: {"x": end["x"] - xp.asarray(target)})
        u = u - 0.5 / slope ** 2 * host(bar["u"])
    np.testing.assert_allclose(u, target / slope, rtol=1e-8)


@pytest.mark.parametrize("backend", BACKENDS)
def test_gradient_descent_through_a_phase_graph(tmp_path, backend):
    """The same descent with the throw, the bounce and the second flight as
    one recorded :class:`eagle.Graph`, built once and launched per step."""
    import eagle
    import eagle.diff
    import eagle.plan
    from _events_fixtures import ev_bounce

    xp = xp_for(backend)
    y0 = np.linspace(1.0, 3.0, N)
    e = xp.asarray(np.linspace(0.6, 0.8, N))
    target = np.linspace(2.0, 5.0, N)
    slope = np.sqrt(2 * y0 / GRAVITY) * (1 + 2 * host(e))
    x, u, y, v, w, t = (xp.zeros(N) for _ in range(6))
    common = dict(dt=0.05, g=GRAVITY, x=x, u=u, y=y, t=t, max_steps=4000, events="refine",
                  record=True)
    targets = ("cuda",) if backend == "gpu" else ("host",)
    (bounce,) = eagle.plan.auto([ev_bounce], targets=targets, cache_dir=str(tmp_path / "hc"))
    g = (eagle.Graph()
         .add_node(eagle.diff.phase(eagle.simulation(ev_throw, v=v, **common)), name="throw")
         .add(bounce.bind(e=e, v_in=v, v=w), name="bounce")
         .add_node(eagle.diff.phase(eagle.simulation(ev_throw, v=w, **common)), name="land")
         .build(record=True))
    speed = np.ones(N)
    for _ in range(30):
        for a in (x, v, w, t):
            a.fill(0)
        y[...] = xp.asarray(y0)
        u[...] = xp.asarray(speed)
        g.launch()
        grads = eagle.diff.reverse(g.record, {"x": x - xp.asarray(target)})
        bar_u = host([a for k, a in grads.items() if k.endswith("u")][0])
        speed = speed - 0.5 / slope ** 2 * bar_u
    np.testing.assert_allclose(speed, target / slope, rtol=1e-8)
