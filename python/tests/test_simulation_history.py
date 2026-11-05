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


# --- history= : the same loop, inside the engine --- #
BACKENDS = ["host", pytest.param("device", marks=pytest.mark.gpu)]


def _planes(where, t_final):
    start = dict(_start(), t_final=np.array(t_final))
    if where == "device":
        cp = pytest.importorskip("cupy")
        start = {k: cp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in start.items()}
    return start


@pytest.mark.parametrize("where", BACKENDS)
@pytest.mark.parametrize("k", [1, 3])
@pytest.mark.parametrize("t_final", [[99.0] * N, [0.25, 0.35, 99.0, 99.0]], ids=["running", "early-finish"])
def test_history_rows_equal_the_loops_rows_bitwise(where, k, t_final):
    res = eagle.simulate(osc, **_planes(where, t_final), max_steps=STEPS, history=k, returns="numpy")
    rows = STEPS // k + 1
    for nm in ("x", "v", "t"):
        assert res.history[nm].shape == (rows, N)
    # row 0 is the initial state, row i the state after i*k steps taken in one go
    for i in range(rows):
        whole = eagle.simulate(osc, **_planes(where, t_final), max_steps=i * k, returns="numpy") if i else None
        for nm in ("x", "v", "t"):
            want = _start()[nm] if whole is None else np.asarray(whole.state[nm])
            np.testing.assert_array_equal(np.asarray(res.history[nm][i]), want)
    final = eagle.simulate(osc, **_planes(where, t_final), max_steps=STEPS, returns="numpy")
    for nm in ("x", "v", "t"):
        np.testing.assert_array_equal(np.asarray(res.state[nm]), np.asarray(final.state[nm]))
    assert res.steps == final.steps


def test_a_sample_that_finished_early_repeats_its_final_state():
    res = eagle.simulate(osc, **_planes("host", [0.25, 0.35, 99.0, 99.0]), max_steps=STEPS, history=True,
                         returns="numpy")
    x = np.asarray(res.history["x"])
    np.testing.assert_array_equal(x[3:, 0], np.full(STEPS - 2, x[3, 0]))    # sample 0 finished at step 3
    assert not np.array_equal(x[3:, 2], np.full(STEPS - 2, x[3, 2]))        # a running sample moves on


def test_history_is_off_by_default_and_refuses_what_a_step_loop_cannot_keep():
    assert eagle.simulate(osc, **_start(), max_steps=2).history is None
    for bad in (0, -1, 1.5, "2"):
        with pytest.raises(ValueError, match="history= is True or a step interval"):
            eagle.simulate(osc, **_start(), max_steps=2, history=bad)
    with pytest.raises(ValueError, match="record= is True or False"):
        eagle.simulate(osc, **_start(), max_steps=2, record=3)
    with pytest.raises(ValueError, match="cannot be combined"):
        eagle.simulate(osc, **_start(), max_steps=2, history=True, record=True)
