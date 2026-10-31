# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A per-step state history: ``eagle.simulation(..., max_steps=1)`` then ``run()`` in a loop
advances the state in place by one step per call, so the history is what the loop copies, and
it equals the same run taken in one go."""

import numpy as np
import pytest

import eagle

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar, Terminated  # noqa: E402

N, STEPS = 4, 5


@hawk.kernel(steps=1)
def osc(dt: Param, t_final: Scalar, terminated: Terminated,
        x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    v0 = v
    v = v0 - dt * x
    x = x + dt * v0
    tn = t + dt
    t = tn
    terminated = tn >= t_final


def _start():
    return dict(dt=0.1, t_final=np.full(N, 99.0), x=np.linspace(1.0, 2.0, N),
                v=np.zeros(N), t=np.zeros(N))


def test_one_step_per_run_gives_the_history():
    sim = eagle.simulation(osc, **_start(), max_steps=1)
    history = []
    for _ in range(STEPS):
        result = sim.run()
        assert result.report.steps == 1
        history.append(np.array(np.asarray(result.state["x"])))
    for k in range(1, STEPS + 1):
        whole = eagle.simulate(osc, **_start(), max_steps=k)
        np.testing.assert_array_equal(history[k - 1], np.asarray(whole.state["x"]))
