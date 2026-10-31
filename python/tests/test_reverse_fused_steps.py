# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``steps_reverse="fused"``: a window of K steps is ONE reverse launch.

The gradients of ``eagle.diff.reverse`` over ``hawk.steps(step, K)`` equal the
per-step reverse of the same run: the state seeds' adjoints bit for bit, the
parameter gradients (summed over steps in another order) within a few ULP of
their magnitude. K = 3 and 7 are odd, 4 and 8 even; the run is ~300 steps, so
K = 7 and 8 end on a short window that falls back to step-by-step launches."""

from __future__ import annotations

import numpy as np
import pytest

import eagle
import eagle.diff
from _run_reverse_fixtures import C, DT, MAX_STEPS, N, host, inputs, rr_pend, xp_for

hawk = pytest.importorskip("hawk")

KS = (3, 4, 7, 8)


def _grads(k, backend, steps_reverse, dtype="float64"):
    xp = xp_for(backend)
    x0, v0, tf = inputs(xp, dtype=dtype)
    sim = eagle.simulation(hawk.steps(rr_pend, k), dt=DT, c=C, t_final=tf, x=x0.copy(), v=v0.copy(),
                           t=xp.zeros_like(x0), max_steps=MAX_STEPS, record=True,
                           steps_reverse=steps_reverse, scalar_type=dtype)
    sim.run()
    seeds = {"x": xp.ones(N, dtype=dtype), "v": xp.full(N, 0.5, dtype=dtype)}
    return {n: host(v) for n, v in eagle.diff.reverse(sim.record, seeds).items()}


def _check(fused, plain, rtol):
    assert sorted(fused) == sorted(plain)
    for name in plain:
        if name in ("x", "v", "t"):
            np.testing.assert_array_equal(fused[name], plain[name], err_msg=name)
        else:
            scale = np.abs(plain[name]).max() + 1e-300
            np.testing.assert_allclose(fused[name], plain[name], rtol=rtol, atol=rtol * scale,
                                       err_msg=name)


@pytest.mark.parametrize("k", KS)
def test_a_fused_window_equals_the_per_step_reverse(k):
    _check(_grads(k, "cpu", "fused"), _grads(k, "cpu", "per_step"), 1e-12)


@pytest.mark.gpu
@pytest.mark.parametrize("k", KS)
def test_a_fused_window_equals_the_per_step_reverse_on_the_device(k):
    fused, plain = _grads(k, "gpu", "fused"), _grads(k, "gpu", "per_step")
    for name in plain:      # nvcc keeps FMA contraction: ULP-level, not bitwise
        np.testing.assert_allclose(fused[name], plain[name], rtol=1e-10,
                                   atol=1e-10 * (np.abs(plain[name]).max() + 1e-300), err_msg=name)


def test_fused_needs_a_steps_k_kernel_and_a_plain_run():
    xp = np
    x0, v0, tf = inputs(xp)

    def sim(model, **kw):
        return eagle.simulation(model, dt=DT, c=C, t_final=tf, x=x0.copy(), v=v0.copy(),
                                t=xp.zeros_like(x0), max_steps=MAX_STEPS, **kw)

    with pytest.raises(ValueError, match="steps=K kernel"):
        sim(rr_pend, record=True, steps_reverse="fused")
    with pytest.raises(ValueError, match="steps=K kernel"):
        sim(hawk.steps(rr_pend, 4), record=True, steps_reverse="fused", window=8)
    with pytest.raises(ValueError, match="plain record=True"):
        sim(hawk.steps(rr_pend, 4), steps_reverse="fused")
    with pytest.raises(ValueError, match="'per_step', 'fused' or 'auto'"):
        sim(hawk.steps(rr_pend, 4), record=True, steps_reverse="quick")


def _chosen(k, n, backend):
    xp = xp_for(backend)
    x0, v0, tf = inputs(xp, n=n)
    sim = eagle.simulation(hawk.steps(rr_pend, k), dt=DT, c=C, t_final=tf, x=x0.copy(), v=v0.copy(),
                           t=xp.zeros_like(x0), max_steps=MAX_STEPS, record=True, steps_reverse="auto")
    return sim.runner._recorder.steps_reverse


def test_auto_follows_the_measured_table_on_the_host():
    from hawk.ir import steps_class

    assert steps_class._ROWS["host"], "the host class has measured rows"
    assert _chosen(4, 8, "cpu") == "fused"           # inside the measured batch range
    assert _chosen(4, 4096, "cpu") == "per_step"     # beyond it
    assert _chosen(1, 8, "cpu") == "per_step"        # a single-step kernel has nothing to fuse


@pytest.mark.gpu
def test_auto_is_per_step_on_a_device_class_with_no_measured_row():
    assert _chosen(4, 8, "gpu") == "per_step"
