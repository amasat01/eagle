# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A recorded run's Record: its defaults, the live and stale refusals,
keep(), the run shapes not yet recorded, and concurrent use."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import eagle
import eagle.diff
from _run_reverse_fixtures import (MAX_STEPS, N, default_window, inputs, rr_osc, rr_pend,
                                   simulation)

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar, Terminated  # noqa: E402

SEEDS = {"x": np.ones(N)}


def _sim(model=rr_osc, **kw):
    return simulation(model, *inputs(np), **kw)


def _fresh_grads(model=rr_osc):
    sim = _sim(model)
    sim.run()
    return eagle.diff.reverse(sim.record, SEEDS)


def test_record_defaults():
    sim = _sim()
    assert sim.record is None
    res = sim.run()
    rec = res.record
    assert rec is sim.record and rec.forward is sim and rec.graph is sim
    assert rec.window == default_window(MAX_STEPS) == 20
    assert rec.generation == 1 and not rec.held
    assert rec.steps.dtype == np.int64 and rec.steps.shape == (N,)
    assert rec.steps.max() == res.steps
    windows = -(-int(rec.steps.max()) // rec.window)
    assert rec.nbytes == windows * N * (4 * 8 + 1)    # x, v, t, applied_steps, mask
    assert _sim(window="full").run().record.window == MAX_STEPS


def test_window_full_is_one_checkpoint():
    rec = _sim(window="full").run().record
    assert rec.nbytes == N * (4 * 8 + 1)


def test_a_live_record_refuses_the_next_recording():
    sim = _sim()
    sim.run()
    with pytest.raises(RuntimeError, match=r"eagle\.simulate: the previous run's Record is "
                       r"still live; reverse it, or record\.keep\(\) to keep it, before "
                       r"recording again"):
        sim.run()


def test_a_stale_record_is_refused():
    sim = _sim()
    sim.run()
    first = sim.record
    eagle.diff.reverse(first, SEEDS)
    sim.reset()
    sim.run()
    with pytest.raises(RuntimeError, match=r"eagle\.diff\.reverse: this Record is stale — the "
                       r"forward ran again \(generation 1 → 2\); reverse the new record, or "
                       r"keep\(\) a Record before relaunching"):
        eagle.diff.reverse(first, SEEDS)


def test_kept_record_outlives_the_next_run():
    want = _fresh_grads()
    x0, v0, tf = inputs(np)
    sim = _sim()
    sim.run()
    kept = sim.record.keep()
    assert kept.nbytes == sim.record.nbytes and kept.window == sim.record.window
    np.testing.assert_array_equal(kept.steps, sim.record.steps)
    sim.reset()
    sim.runner.step._planes["x"][...] = x0 + 1.0       # a different second run
    sim.run()
    got = eagle.diff.reverse(kept, SEEDS)
    for k in want:
        np.testing.assert_array_equal(got[k], want[k])
    assert "kept" in repr(kept)


def test_record_off_window_is_refused():
    with pytest.raises(ValueError, match="window= and memory_budget= set how a record=True run is reversed"):
        _sim(record=False, window=16)


@hawk.kernel(steps=4)
def rr_four(dt: Param, t_final: Scalar, terminated: Terminated, x: Mutable[Scalar],
            t: Mutable[Scalar]):
    x = x - dt * x
    tn = t + dt
    t = tn
    terminated = tn >= t_final


def test_two_kernels_in_one_step_are_not_yet_supported():
    x0, v0, tf = inputs(np)
    with pytest.raises(ValueError, match="for a step of several plans is not yet supported"):
        eagle.simulation([rr_osc, rr_pend], dt=0.01, c=0.3, t_final=tf, x=x0, v=v0,
                         t=np.zeros(N), max_steps=100, record=True)


def test_vjps_for_a_run_is_not_yet_supported():
    sim = _sim()
    with pytest.raises(ValueError, match="vjps= for a recorded run is not yet supported"):
        eagle.diff.Reverse(sim, vjps={"rr_osc": None})


def test_reverse_of_an_unrecorded_run_names_the_fix():
    with pytest.raises(ValueError, match="pass record=True to the forward"):
        eagle.diff.Reverse(_sim(record=False))


def test_seed_on_a_non_state_plane_names_the_state():
    sim = _sim()
    sim.run()
    with pytest.raises(ValueError, match=r"seeds names 't_final', which is not a state plane"):
        eagle.diff.reverse(sim.record, {"t_final": np.ones(N)})


def test_two_threads_two_simulations():
    want = {m.name: _fresh_grads(m) for m in (rr_osc, rr_pend)}
    got, errors = {}, []

    def work(model):
        try:
            for _ in range(3):
                got[model.name] = _fresh_grads(model)
        except BaseException as exc:      # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(m,)) for m in (rr_osc, rr_pend)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    for name, grads in want.items():
        for k in grads:
            np.testing.assert_array_equal(got[name][k], grads[k])


def test_one_simulation_reversed_from_two_threads_is_refused_or_correct():
    want = _fresh_grads()
    sim = _sim()
    sim.run()
    rec = sim.record
    results, refused, errors = [], [], []
    start = threading.Barrier(4)

    def work():
        start.wait()
        try:
            results.append(eagle.diff.reverse(rec, SEEDS))
        except eagle.Busy:
            refused.append(1)
        except BaseException as exc:      # pragma: no cover - reported below
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(results) + len(refused) == 4 and results
    for grads in results:
        for k in want:
            np.testing.assert_array_equal(grads[k], want[k])


@pytest.mark.parametrize("backend", ["cpu", pytest.param("gpu", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("window", [None, 16, "full"])
def test_n_threads_n_simulations_agree_with_serial_and_count_records_exactly(backend, window):
    from eagle import _counters
    from _run_reverse_fixtures import host, xp_for

    xp = xp_for(backend)
    threads_n, rounds = 6, 3
    seeds = {"x": xp.ones(N)}
    models = [rr_osc, rr_pend] * (threads_n // 2)

    def grads(model):
        sim = simulation(model, *inputs(xp), window=window)
        sim.run()
        return {k: host(v).tobytes() for k, v in eagle.diff.reverse(sim.record, seeds).items()}

    want = {m.name: grads(m) for m in (rr_osc, rr_pend)}
    out, errors = [None] * threads_n, []
    start = threading.Barrier(threads_n)
    before = _counters.snapshot()

    def work(i):
        try:
            start.wait()
            for _ in range(rounds):
                out[i] = grads(models[i])
        except BaseException as exc:      # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert errors == []
    assert out == [want[m.name] for m in models]
    after = _counters.snapshot()
    assert after["records"] - before["records"] == threads_n * rounds
    assert after["reverses"] - before["reverses"] == threads_n * rounds
