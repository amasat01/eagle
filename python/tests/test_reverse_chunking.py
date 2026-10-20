# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""memory_budget: the reverse runs the batch in passes of 4096 * 2^q samples
and combines the passes' (hi, comp) parameter partials with the canonical
pairwise tree, so every gradient is bitwise the single-pass one under the
ordered policy (the host's for every policy), for every batch size. A
planted left fold over the passes must differ, and the refusals name the
bytes and the fix."""

from __future__ import annotations

import numpy as np
import pytest

import eagle.diff
from eagle.diff import _run
from _run_reverse_fixtures import host, inputs, rr_pend, simulation, xp_for

LEAF = 4096


def _grads(n, budget=None, backend="cpu", dtype="float64", seed=1.0, policy="compensated"):
    xp = xp_for(backend)
    x0, v0, tf = inputs(xp, n=n, dtype=dtype)
    sim = simulation(rr_pend, x0, v0, tf, window=37, dtype=dtype)
    if budget is not None:
        per = sim.runner._recorder.per_sample_bytes()
        sim.runner._recorder.budget = int(budget * per)
    sim.run()
    grads = eagle.diff.reverse(sim.record, {"x": xp.full(n, seed, dtype=dtype),
                                            "v": xp.full(n, 0.5, dtype=dtype)}, policy=policy)
    return {k: host(v) for k, v in grads.items()}, grads.report.chunks


def _same(a, b):
    assert sorted(a) == sorted(b)
    return all(np.array_equal(a[k], b[k]) for k in a)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("n, passes", [(1, LEAF), (LEAF, LEAF), (4 * LEAF, LEAF), (4 * LEAF, 9600),
                                       (3 * LEAF, LEAF), (16145, LEAF), (16145, 2 * LEAF)])
def test_chunked_is_bitwise_unchunked(n, passes, dtype):
    want, one = _grads(n, dtype=dtype)
    got, chunks = _grads(n, budget=passes, dtype=dtype)
    assert one == 1
    size = n if passes >= n else (2 * LEAF if passes >= 2 * LEAF else LEAF)
    assert chunks == -(-n // size)
    assert _same(got, want)


def test_canaries_differ(monkeypatch):
    n = 8 * LEAF
    want, _ = _grads(n, dtype="float32", seed=1.37)
    left = lambda parts, _code: float(np.float32(sum(np.float32(h) + np.float32(c) for h, c in parts)))
    monkeypatch.setattr(_run, "_combine", left)
    folded, chunks = _grads(n, budget=LEAF, dtype="float32", seed=1.37)
    assert chunks == 8 and not np.array_equal(folded["c"], want["c"])
    monkeypatch.undo()
    monkeypatch.setattr(_run, "_chunk_samples", lambda *a: 3 * LEAF)
    thirds, chunks = _grads(n, dtype="float32", seed=1.37)
    # compensated partials make unaligned passes agree to the last bit here:
    # order bits are no witness for them, the contract rows of eagle.reduce are
    assert chunks == 3
    for k in ("x", "v"):                 # per-sample gradients never depend on passes
        np.testing.assert_array_equal(thirds[k], want[k])


def test_refusals_name_the_bytes_and_the_fix():
    x0, v0, tf = inputs(np)
    sim = simulation(rr_pend, x0, v0, tf, window=37)
    per = sim.runner._recorder.per_sample_bytes()
    sim.runner._recorder.budget = per - 1
    sim.run()
    with pytest.raises(ValueError, match=r"eagle\.diff\.reverse: one sample alone needs .* MB "
                       r"\(11 checkpoints and a 37-step tape of 3 values\), more than "
                       r"memory_budget=.*; raise memory_budget=, lower max_steps, or set window="):
        eagle.diff.reverse(sim.record, {"x": np.ones(len(x0))})
    n = 2 * LEAF
    x0, v0, tf = inputs(np, n=n)
    sim = simulation(rr_pend, x0, v0, tf, window=37)
    sim.runner._recorder.budget = 100 * per
    sim.run()
    with pytest.raises(ValueError, match=r"holds 100 samples per pass, fewer than the 4096 a "
                       r"pass needs to keep parameter gradients bitwise; raise memory_budget="):
        eagle.diff.reverse(sim.record, {"x": np.ones(n)}, policy="ordered")


def test_memory_budget_door_and_default():
    x0, v0, tf = inputs(np)
    sim = eagle.simulation(rr_pend, dt=0.01, c=0.3, t_final=tf, x=x0, v=v0,
                           t=np.zeros_like(x0), max_steps=400, record=True,
                           memory_budget=12345678)
    assert sim.runner._recorder.budget == 12345678
    sim = simulation(rr_pend, x0, v0, tf)
    assert sim.runner._recorder.budget > 0
    with pytest.raises(ValueError, match="memory_budget is a positive int of bytes"):
        eagle.simulation(rr_pend, dt=0.01, c=0.3, t_final=tf, x=x0, v=v0,
                         t=np.zeros_like(x0), max_steps=400, record=True, memory_budget=-1)


@pytest.mark.gpu
@pytest.mark.parametrize("n, passes", [(4 * LEAF, LEAF), (16145, 2 * LEAF)])
def test_chunked_is_bitwise_unchunked_gpu(n, passes):
    want, _ = _grads(n, backend="gpu", policy="ordered")
    got, chunks = _grads(n, budget=passes, backend="gpu", policy="ordered")
    assert chunks > 1 and _same(got, want)
