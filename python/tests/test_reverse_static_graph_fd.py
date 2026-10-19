# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""eagle.diff.reverse on a chain and a fork/join graph equals central
differences, on the host and on the device."""

import numpy as np
import pytest

import eagle
from _reverse_fixtures import central_difference, chain, fork_join, plans, xp_for

N = 64
CASES = [pytest.param("cpu", id="cpu"), pytest.param("gpu", id="gpu", marks=pytest.mark.gpu)]


def _inputs(xp):
    rng = np.random.default_rng(7)
    x = rng.uniform(-1.5, 1.5, N)
    seed = rng.uniform(0.5, 2.0, N)
    return xp.asarray(x), 0.75, xp.asarray(seed)


def _host(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


@pytest.mark.parametrize("backend", CASES)
@pytest.mark.parametrize("make, out", [(chain, "sin.y"), (fork_join, "w")])
def test_reverse_matches_central_differences(tmp_path, backend, make, out):
    xp = xp_for(backend)
    ps = plans(tmp_path, backend)
    x, a, seed = _inputs(xp)
    g, _o = make(ps, x, a, record=True)
    g.launch()
    grads = eagle.diff.reverse(g.record, {out: seed})
    dx, da = central_difference(make, ps, x, a, seed)
    np.testing.assert_allclose(_host(grads["x"]), _host(dx), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(float(grads["a"]), da, rtol=1e-6)
    # a parameter's sum over samples is bitwise only under the ordered policy
    assert grads.report.bitwise == {"x": True, "a": False}


def test_fork_join_sums_both_branches_into_x(tmp_path):
    ps = plans(tmp_path, "cpu")
    x, a, seed = _inputs(np)
    g, _w = fork_join(ps, x, a, record=True)
    g.launch()
    rev = eagle.diff.Reverse(g)
    grads = rev(g.record, {"w": seed})
    u, v = a * x * x, np.sin(x)
    # w = u v + u: dw/dx = (v + 1) * 2 a x + u cos x
    np.testing.assert_allclose(grads["x"], seed * ((v + 1) * 2 * a * x + u * np.cos(x)),
                               rtol=1e-12)
