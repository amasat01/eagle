# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A second embedded pair, Dormand-Prince 5(4), on the adaptive card's problem.

``benchmarks/rk78_card/dp54.py`` writes one attempt kernel for any tableau.
These rows run 64 of the card's Kepler orbits (eccentricity up to 0.9, the
card's final time) at rtol = atol = 1e-9 under DP5(4) and under RKF7(8), on the
host threads and, under the ``gpu`` marker, on the GPU, and require each pair
to land within the card's derived truth bound of the analytic Kepler solution,
the two pairs to agree within the sum of their bounds, DP5(4) to take more
steps than RKF7(8) at this tolerance (the point of the higher order), and the
host and GPU step counts to agree.

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

pytestmark = pytest.mark.repo_local

_DP54 = pathlib.Path(__file__).resolve().parents[2] / "benchmarks" / "rk78_card" / "dp54.py"
N = 64
TOL = 1e-9


@pytest.fixture(scope="module")
def dp54():
    spec = importlib.util.spec_from_file_location("eagle_dp54", _DP54)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run(dp54, device):
    rc = dp54.rc
    inp = rc._inputs(N)
    pos, _ = rc.kepler_truth(inp, rc.T_FINAL)
    out = {}
    for name, kernel in (("dp54", dp54.dp54_kernel(TOL, TOL)),
                         ("rkf78", dp54.rkf78_kernel(TOL, TOL))):
        s, n_acc, n_rej = dp54.solve(kernel, inp, device)
        bound = rc.truth_bound(inp, n_acc) * TOL / rc.RTOL  # the bound is linear in the tolerance
        out[name] = dict(s=s, n_acc=n_acc, n_rej=n_rej, bound=bound,
                         err=np.linalg.norm(s[:3] - pos, axis=0))
    return out


def _check(out):
    for name, r in out.items():
        assert np.all(r["err"] < r["bound"]), f"{name}: a sample exceeds its truth bound"
        assert r["err"].max() < 1e-4, f"{name}: position error {r['err'].max():.2e}"
    pair_gap = np.linalg.norm(out["dp54"]["s"][:3] - out["rkf78"]["s"][:3], axis=0)
    assert np.all(pair_gap < out["dp54"]["bound"] + out["rkf78"]["bound"])
    assert out["dp54"]["n_acc"].sum() > out["rkf78"]["n_acc"].sum()
    for name, r in out.items():
        print(f"{name}: accepted {r['n_acc'].mean():.0f} mean / {r['n_acc'].max()} max, "
              f"rejected {r['n_rej'].mean():.1f} mean, max position error {r['err'].max():.2e}")


def test_dp54_and_rkf78_reach_the_kepler_truth_on_the_host(dp54):
    _check(_run(dp54, device=False))


@pytest.mark.gpu
def test_dp54_and_rkf78_reach_the_kepler_truth_on_the_gpu_and_match_the_host(dp54):
    host, gpu = _run(dp54, device=False), _run(dp54, device=True)
    _check(gpu)
    for name in host:
        assert np.array_equal(host[name]["n_acc"], gpu[name]["n_acc"]), (
            f"{name}: host and GPU accepted-step counts differ")
        np.testing.assert_allclose(host[name]["s"], gpu[name]["s"], rtol=1e-6, atol=1e-6)
