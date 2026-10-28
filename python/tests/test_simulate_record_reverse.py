# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A recorded simulate run reversed with the full tape (``window="full"``):
gradients match central differences for a linear and a nonlinear model, and
the analytic adjoint for the linear one, with every sample stopping at its
own step. The device rows also match the host."""

from __future__ import annotations

import numpy as np
import pytest

import eagle.diff
from _run_reverse_fixtures import (N, analytic_osc, central_difference, host, inputs,
                                   rr_osc, rr_pend, simulation, xp_for)

MODELS = {"oscillator": rr_osc, "pendulum": rr_pend}


def _reverse(model, backend, window="full"):
    xp = xp_for(backend)
    x0, v0, tf = inputs(xp)
    sim = simulation(model, x0, v0, tf, window=window)
    res = sim.run()
    grads = eagle.diff.reverse(res.record, {"x": xp.ones(N)})
    return res, {k: host(v) for k, v in grads.items()}


def _close(got, want, rtol=1e-6):
    scale = np.max(np.abs(want))
    np.testing.assert_allclose(got, want, rtol=rtol, atol=rtol * scale)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_full_tape_matches_central_differences(name):
    model = MODELS[name]
    res, grads = _reverse(model, "cpu")
    steps = res.record.steps
    assert len(set(steps.tolist())) > N // 2      # samples stop at many different steps
    assert sorted(grads) == ["c", "dt", "t", "v", "x"]
    dx, dv, dc = central_difference(model, *inputs(np))
    _close(grads["x"], dx)
    _close(grads["v"], dv)
    assert float(grads["c"]) == pytest.approx(dc, rel=1e-6)


def test_linear_matches_the_analytic_adjoint():
    res, grads = _reverse(rr_osc, "cpu")
    want = analytic_osc(res.record.steps)
    np.testing.assert_allclose(grads["x"], want[0], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(grads["v"], want[1], rtol=1e-12, atol=1e-14)


def test_a_sample_that_never_steps_keeps_its_seed():
    x0, v0, tf = inputs(np)
    tf[:3] = -1.0                                 # stops at its first step
    sim = simulation(rr_osc, x0, v0, tf)
    sim.runner.terminated[3:5] = True             # never steps at all
    sim.run()
    assert sim.record.steps[3] == 0 and sim.record.steps[0] == 1
    grads = eagle.diff.reverse(sim.record, {"x": np.full(N, 2.0)})
    assert grads["x"][3] == 2.0 and grads["v"][3] == 0.0
    assert grads["x"][0] == 2.0 and grads["v"][0] == pytest.approx(2.0 * 0.01)


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(MODELS))
def test_full_tape_matches_central_differences_gpu(name):
    model = MODELS[name]
    _res, grads = _reverse(model, "gpu")
    dx, dv, dc = central_difference(model, *inputs(np))
    _close(grads["x"], dx)
    _close(grads["v"], dv)
    assert float(grads["c"]) == pytest.approx(dc, rel=1e-6)


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(MODELS))
def test_device_matches_host(name):
    model = MODELS[name]
    res_d, dev = _reverse(model, "gpu")
    res_h, hst = _reverse(model, "cpu")
    np.testing.assert_array_equal(res_d.record.steps, res_h.record.steps)
    for k in hst:
        scale = max(float(np.max(np.abs(hst[k]))), 1.0)
        np.testing.assert_allclose(dev[k], hst[k], rtol=1e-12, atol=1e-12 * scale)
