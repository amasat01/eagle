# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The reverse window does not change a gradient, bit for bit: W = 16,
ceil(sqrt(S)) and 37 (not a divisor of the run) give the gradients of the
full tape (W = S) exactly, in float64 and float32, through until_done and
through one Reverse built once and called per run."""

from __future__ import annotations

import math

import numpy as np
import pytest

import eagle
import eagle.diff
import eagle.plan
from _run_reverse_fixtures import (C, DT, MAX_STEPS, N, host, inputs, rr_pend,
                                   rr_pend_auto, simulation, xp_for)

WINDOWS = (16, math.isqrt(MAX_STEPS - 1) + 1, 37)


def _grads(model, backend, window, dtype):
    xp = xp_for(backend)
    x0, v0, tf = inputs(xp, dtype=dtype)
    sim = simulation(model, x0, v0, tf, window=window, dtype=dtype)
    sim.run()
    seeds = {"x": xp.ones(N, dtype=dtype), "v": xp.full(N, 0.5, dtype=dtype)}
    return {k: host(v) for k, v in eagle.diff.reverse(sim.record, seeds).items()}


def _assert_bitwise(a, b):
    assert sorted(a) == sorted(b)
    for k in a:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("window", WINDOWS)
def test_window_is_bitwise_the_full_tape(window, dtype):
    _assert_bitwise(_grads(rr_pend, "cpu", window, dtype),
                    _grads(rr_pend, "cpu", "full", dtype))


def test_automatic_kernel_matches_the_single_step_kernel():
    _assert_bitwise(_grads(rr_pend_auto, "cpu", 16, "float64"),
                    _grads(rr_pend, "cpu", "full", "float64"))


def test_until_done_door_and_a_reverse_built_once(tmp_path):
    plan = eagle.plan.auto(rr_pend, targets=("host",), cache_dir=str(tmp_path / "hc"))
    x0, v0, tf = inputs(np)
    x, v, t = x0.copy(), v0.copy(), np.zeros(N)
    runner = eagle.until_done(plan, max_steps=MAX_STEPS, record=True, window=37,
                              dt=DT, c=C, t_final=tf, x=x, v=v, t=t)
    rev = eagle.diff.Reverse(forward=runner)
    want = _grads(rr_pend, "cpu", "full", "float64")
    for _ in range(2):
        x[...], v[...], t[...] = x0, v0, 0.0
        runner.reset()
        report = runner.run()
        assert report.mode == "record"
        got = rev(runner.record, {"x": np.ones(N), "v": np.full(N, 0.5)})
        _assert_bitwise({k: got[k] for k in got}, want)


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("window", WINDOWS)
def test_window_is_bitwise_the_full_tape_gpu(window, dtype):
    _assert_bitwise(_grads(rr_pend, "gpu", window, dtype),
                    _grads(rr_pend, "gpu", "full", dtype))
