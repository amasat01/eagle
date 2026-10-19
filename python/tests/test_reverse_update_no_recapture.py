# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A shooting loop that changes a parameter every iteration: g.update writes
the parameter's cell, so the forward and its reverse are each built (captured,
on the GPU) exactly once over 100 iterations, and every gradient equals a fresh
eager reverse at that parameter."""

import numpy as np
import pytest

import eagle
from eagle import _counters
from _reverse_fixtures import chain, plans, xp_for

N, ITERS = 256, 100
CASES = [pytest.param("cpu", id="cpu"), pytest.param("gpu", id="gpu", marks=pytest.mark.gpu)]


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


@pytest.mark.parametrize("backend", CASES)
def test_update_in_a_loop_rebuilds_nothing(tmp_path, backend):
    xp = xp_for(backend)
    ps = plans(tmp_path, backend)
    x = xp.asarray(np.linspace(-1.2, 1.2, N))
    seed = {"sin.y": xp.asarray(np.linspace(0.5, 1.5, N))}
    ks = [0.3 + 0.01 * i for i in range(ITERS)]
    before = _counters.snapshot()
    g, _z = chain(ps, x, ks[0], record=True)
    rev = eagle.diff.Reverse(g)
    got = []
    for k in ks:
        g.update(a=k)
        g.launch()
        grads = rev(g.record, seed)
        got.append((_host(grads["x"]), float(grads["a"])))
    after = _counters.snapshot()
    assert after["updates"] - before["updates"] == ITERS
    assert after["rebinds"] - before["rebinds"] == 0
    assert after["graph_builds"] - before["graph_builds"] == 2      # forward + reverse
    if backend == "gpu":
        assert after["captures"] - before["captures"] == 2
    for k, (gx, ga) in zip(ks, got):
        fresh, _ = chain(ps, x.copy(), k, record=True)
        want = eagle.diff.reverse(fresh.launch().record, seed)
        if backend == "cpu":
            assert gx.tobytes() == _host(want["x"]).tobytes() and ga == float(want["a"])
        else:
            np.testing.assert_allclose(gx, _host(want["x"]), rtol=1e-13, atol=1e-15)
            np.testing.assert_allclose(ga, float(want["a"]), rtol=1e-13)
