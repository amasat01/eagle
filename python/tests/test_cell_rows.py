# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A kernel that stores a reduction over samples: every door finishes its rows
into the cell, and allocates the row scratch once per kernel and sample count."""

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")
import eagle  # noqa: E402
import eagle.plan  # noqa: E402
from eagle import _cell_rows  # noqa: E402
from hawk import Mutable, Param, Scalar, Terminated  # noqa: E402

if not hasattr(hawk, "all_samples"):
    pytest.skip("this hawk has no reductions over samples", allow_module_level=True)


@hawk.kernel
def cell_sum(x: Scalar, t: Mutable[Scalar]):
    t = hawk.all_samples(x * x).sum()


@hawk.kernel
def cell_max(x: Scalar, t: Mutable[Scalar]):
    t = hawk.all_samples(x * 1.5).max()


@hawk.kernel
def sim_cell(x: Mutable[Scalar], s: Mutable[Scalar], terminated: Terminated):
    x = x + 1.0
    s = hawk.all_samples(x * x).sum()
    terminated = x >= 3.0


@hawk.kernel
def lin(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x


def _x(n):
    return np.random.default_rng(n).integers(-8, 9, n).astype(np.float64)


@pytest.mark.parametrize("n", [1, 31, 32, 33, 1000])
def test_plan_run_returns_the_finished_sum_and_max(tmp_path, n):
    x = _x(n)
    (ps,) = eagle.plan.auto([cell_sum], targets=("host",), cache_dir=str(tmp_path))
    (pm,) = eagle.plan.auto([cell_max], targets=("host",), cache_dir=str(tmp_path))
    got = ps.run(x=x)
    assert got.shape == (1,) and got[0] == (x * x).sum()
    got = pm.run(x=x)
    assert got.shape == (1,) and got[0] == (x * 1.5).max()


def test_row_scratch_is_made_once_per_kernel_and_n(tmp_path):
    (p,) = eagle.plan.auto([cell_sum], targets=("host",), cache_dir=str(tmp_path))
    p.run(x=_x(100))
    before = _cell_rows.made
    for _ in range(4):
        p.run(x=_x(100))
    assert _cell_rows.made == before
    p.run(x=_x(200))
    assert _cell_rows.made == before + 1


def test_simulate_returns_the_cell_of_the_last_step(tmp_path):
    n = 40
    res = eagle.simulate(sim_cell, x=np.zeros(n), max_steps=10)
    assert res["s"].shape == (1,)
    assert res["s"][0] == 9.0 * n
    assert "s_comp" not in res.state


@pytest.mark.parametrize("policy", ["ordered", "compensated"])
def test_reverse_param_gradient_is_the_finished_cell(tmp_path, policy):
    n = 100
    x, seed = _x(n), _x(n + 7)[:n]
    (p,) = eagle.plan.auto([lin], targets=("host",), cache_dir=str(tmp_path))
    g = eagle.Graph().add(p.bind(x=x, a=2.0, y=np.zeros(n)), name="s").build(record=True)
    g.launch()
    rev = eagle.diff.Reverse(g, policy=policy)
    grads = rev(g.record, {"y": seed})
    assert grads["a"] == (seed * x).sum()           # integer terms: exact in any order
    assert np.array_equal(grads["x"], 2.0 * seed)
    assert rev(g.record, {"y": seed})["a"] == grads["a"]
