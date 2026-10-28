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


@hawk.kernel
def cell_sum(x: Scalar, t: Mutable[Scalar]):
    t = hawk.sum(x * x)


@hawk.kernel
def cell_max(x: Scalar, t: Mutable[Scalar]):
    t = hawk.max(x * 1.5)


@hawk.kernel
def cell_min(x: Scalar, t: Mutable[Scalar]):
    t = hawk.min(x)


@hawk.kernel
def sim_cell(x: Mutable[Scalar], s: Mutable[Scalar], terminated: Terminated):
    x = x + 1.0
    s = hawk.sum(x * x)
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


@pytest.mark.parametrize("n", [1, 33, 1000])
def test_plan_run_returns_a_vector_cell_one_plane_per_component(tmp_path, n):
    def vcell(v: hawk.Vector[3], t: Mutable[hawk.Vector[3]]):
        t = hawk.sum(v)  # noqa: F841
    try:
        k = hawk.kernel(vcell)
        k.walk.cells
    except Exception:
        pytest.skip("this hawk has no vector cells")
    (p,) = eagle.plan.auto([k], targets=("host",), cache_dir=str(tmp_path))
    v = np.random.default_rng(n).uniform(-2, 2, (3, n))
    got = p.run(v=v)
    np.testing.assert_allclose([got[f"t_{i}"][0] for i in range(3)], v.sum(axis=1), rtol=1e-12)


@pytest.mark.parametrize("n", [1, 31, 33, 1000])
def test_plan_run_returns_the_finished_min(tmp_path, n):
    x = _x(n)
    (p,) = eagle.plan.auto([cell_min], targets=("host",), cache_dir=str(tmp_path))
    got = p.run(x=x)
    assert got.shape == (1,) and got[0] == x.min()
    x[n // 2] = -np.inf
    assert p.run(x=x)[0] == -np.inf


@pytest.mark.gpu
@pytest.mark.parametrize("n", [1, 33, 1000])
def test_plan_run_returns_the_finished_min_on_cuda(tmp_path, n):
    cp = pytest.importorskip("cupy")
    x = _x(n)
    x[n // 2] = -np.inf if n > 1 else x[0]
    (p,) = eagle.plan.auto([cell_min], targets=("cuda",), cache_dir=str(tmp_path))
    got = p.run(x=cp.asarray(x))
    assert got.shape == (1,) and float(got[0]) == x.min()


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


def test_returns_routes_plan_run_and_simulate_through_the_bridge(tmp_path):
    import warnings

    from eagle.frameworks import _bridge

    x = _x(33)
    (ps,) = eagle.plan.auto([cell_sum], targets=("host",), cache_dir=str(tmp_path))
    assert isinstance(ps.run(x=x), eagle.Array)                                  # the default
    got = ps.run(x=x, returns="numpy")
    assert isinstance(got, np.ndarray) and got[0] == (x * x).sum()
    res = eagle.simulate(sim_cell, x=np.zeros(40), max_steps=10, returns="numpy")
    assert isinstance(res["s"], np.ndarray) and res["s"][0] == 9.0 * 40 and isinstance(res.finished, np.ndarray)
    _bridge._WARNED.discard("torch")
    import sys
    saved = sys.modules.get("torch", 0)
    sys.modules["torch"] = None
    try:
        with warnings.catch_warnings(record=True) as seen:
            warnings.simplefilter("always")
            again = ps.run(x=x, returns="torch")
            ps.run(x=x, returns="torch")
    finally:
        if saved == 0:
            del sys.modules["torch"]
        else:
            sys.modules["torch"] = saved
    assert isinstance(again, eagle.Array) and len([w for w in seen if "pip install torch" in str(w.message)]) == 1
