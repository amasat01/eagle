# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Hybrid trajectories as a static Graph of refined phases: fall to the
ground (refined), bounce (v -> -e v, a reset kernel), rise to the apex
(refined at v = 0). The chain rule across the phases is the saltation of the
jump: the ball's apex after one bounce is h1 = e^2 y0 exactly (exact
dynamics, round-off bound), so dh1/dy0 = e^2 and dh1/de = 2 e y0; after two
bounces h2 = e^4 y0, dh2/dy0 = e^4, dh2/de = 4 e^3 y0. The pendulum-on-a-wall
twin (non-polynomial, RK4) is checked against central differences of the
same graph. Every graph runs on the host and on the device (where each phase
is an eager node between captured segments), and the device equals the
host. The same phases chained by hand give the closed forms too."""

from __future__ import annotations

import numpy as np
import pytest

import eagle
import eagle.diff
import eagle.plan
from _events_fixtures import (BACKENDS, GRAVITY, chain_bounces, ev_bounce, ev_fall, ev_pend,
                              ev_pend_rise, ev_rise, host, xp_for)

N = 16
U = np.finfo(float).eps


def _bounce_plan(tmp_path, backend):
    targets = ("cuda",) if backend == "gpu" else ("host",)
    (bounce,) = eagle.plan.auto([ev_bounce], targets=targets,
                                cache_dir=str(tmp_path / f"hawk_cache_{backend}"))
    return bounce


def _ball(tmp_path, backend, y0, e, bounces=1):
    """fall -> bounce -> rise, ``bounces`` times, as one recorded Graph."""
    xp = xp_for(backend)
    y, t = xp.asarray(y0).copy(), xp.zeros(N)
    vs = [xp.zeros(N) for _ in range(bounces + 1)]
    common = dict(dt=0.05, g=GRAVITY, y=y, t=t, max_steps=4000, events="refine", record=True)
    phase = lambda model, v: eagle.diff.phase(eagle.simulation(model, v=v, **common))
    bounce = _bounce_plan(tmp_path, backend)
    e = xp.asarray(e)
    g = eagle.Graph()
    for b in range(bounces):
        down, up = vs[b], vs[b + 1]       # a later fall starts from the last apex
        (g.add_node(phase(ev_fall, down), name=f"fall{b}")
         .add(bounce.bind(e=e, v_in=down, v=up), name=f"bounce{b}")
         .add_node(phase(ev_rise, up), name=f"rise{b}"))
    return g.build(record=True), y


def _named(grads, end):
    return host([v for k, v in grads.items() if k.endswith(end)][0])


@pytest.mark.parametrize("bounces", [1, 2])
@pytest.mark.parametrize("backend", BACKENDS)
def test_the_apex_after_the_bounces_and_its_sensitivities(tmp_path, backend, bounces):
    xp = xp_for(backend)
    y0 = np.linspace(1.0, 5.0, N)
    e = np.linspace(0.5, 0.9, N)
    g, y = _ball(tmp_path, backend, y0, e, bounces)
    g.launch()
    k = 2 * bounces
    np.testing.assert_allclose(host(y), e ** k * y0, rtol=4e3 * U)
    grads = eagle.diff.reverse(g.record, {"y": xp.ones(N)})
    np.testing.assert_allclose(host(grads["e"]), k * e ** (k - 1) * y0, rtol=1e5 * U)
    np.testing.assert_allclose(_named(grads, "y"), e ** k, rtol=1e5 * U)
    if backend == "gpu":      # the device graph equals the host graph
        gc, yc = _ball(tmp_path, "cpu", y0, e, bounces)
        gc.launch()
        np.testing.assert_allclose(host(y), yc, rtol=4e3 * U)
        cpu = eagle.diff.reverse(gc.record, {"y": np.ones(N)})
        np.testing.assert_allclose(host(grads["e"]), cpu["e"], rtol=1e5 * U)


def _pendulum(tmp_path, backend, x0, e):
    xp = xp_for(backend)
    x, v, w, t = xp.asarray(x0).copy(), xp.zeros(N), xp.zeros(N), xp.zeros(N)
    common = dict(dt=0.05, c=0.1, a=0.0, x=x, t=t, max_steps=4000,
                  events="refine", record=True)
    down = eagle.simulation(ev_pend, v=v, **common)
    up = eagle.simulation(ev_pend_rise, v=w, **common)
    bounce = _bounce_plan(tmp_path, backend)
    g = (eagle.Graph().add_node(eagle.diff.phase(down), name="swing")
         .add(bounce.bind(e=xp.asarray(e), v_in=v, v=w), name="wall")
         .add_node(eagle.diff.phase(up), name="apex").build(record=True))
    return g, x


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_pendulum_on_a_wall_matches_central_differences(tmp_path, backend):
    xp = xp_for(backend)
    x0 = np.linspace(0.4, 0.9, N)
    e = np.full(N, 0.8)
    g, x = _pendulum(tmp_path, backend, x0, e)
    g.launch()
    grads = eagle.diff.reverse(g.record, {"x": xp.ones(N)})
    h = 1e-6

    def apex(xs, es):
        gg, xx = _pendulum(tmp_path, "cpu", xs, es)
        gg.launch()
        return xx.copy()

    dx = (apex(x0 + h, e) - apex(x0 - h, e)) / (2 * h)
    de = (apex(x0, e + h) - apex(x0, e - h)) / (2 * h)
    np.testing.assert_allclose(host(grads["e"]), de, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(_named(grads, "x"), dx, rtol=1e-6, atol=1e-9)
    if backend == "gpu":
        np.testing.assert_allclose(host(x), apex(x0, e), rtol=1e-12)


def test_a_phase_inside_a_loop_is_refused(tmp_path):
    g, _y = _ball(tmp_path, "cpu", np.ones(N), np.full(N, 0.5))
    node = eagle.diff.phase(g._nodes[0].payload)
    guard = eagle.SkipGuard(np.zeros(1, dtype=np.int32), 0, np.ones(1, dtype=np.uint32), 0)
    with pytest.raises(ValueError, match="a phase count must be static: unroll the phases "
                                         "into the Dag"):
        eagle.repeat_while(node, guard, 3)


def test_a_phase_needs_a_recorded_refined_run():
    sim = eagle.simulation(ev_fall, dt=0.05, g=GRAVITY, y=np.ones(N), v=np.zeros(N),
                           t=np.zeros(N), max_steps=100)
    with pytest.raises(ValueError, match='a phase is a run built with events="refine", '
                                         "record=True"):
        eagle.diff.phase(sim)


@pytest.mark.parametrize("bounces", [1, 2])
@pytest.mark.parametrize("backend", BACKENDS)
def test_chained_phases_give_the_closed_form(backend, bounces):
    xp = xp_for(backend)
    y0 = xp.asarray(np.linspace(1.0, 5.0, N))
    e = xp.asarray(np.linspace(0.5, 0.9, N))
    st, bar = chain_bounces(xp, y0, e, bounces)
    y0, e = host(y0), host(e)
    k = 2 * bounces
    np.testing.assert_allclose(host(st["y"]), e ** k * y0, rtol=4e3 * U)
    np.testing.assert_allclose(host(bar["y"]), e ** k, rtol=1e5 * U)
    np.testing.assert_allclose(host(bar["e"]), k * e ** (k - 1) * y0, rtol=1e5 * U)
