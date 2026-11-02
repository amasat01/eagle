# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The opt-in GEMM route (``hawk.emit.policy.using(contraction="gemm")``):
forward within the ulp contract of the kernel's own loop, the reverse equal
to hawk's vjp and to finite differences, a second consumer's gradient summed
in, and the default untouched."""

import numpy as np
import pytest

import eagle
import eagle.diff
import eagle.plan

hawk = pytest.importorskip("hawk")
import hawk.math as m  # noqa: E402
from hawk import Mutable, Scalar, Table  # noqa: E402
from hawk.emit import policy  # noqa: E402

R, U, S = 8, 6, 5


@hawk.kernel
def contract(w: Table["row":R, "neuron":U], f: Table["row":R, "sample":S], y: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + w.at(row=r, neuron=unit) * f.at(row=r, sample=sample)
    y = acc


@hawk.kernel
def contract_two(w: Table["row":R, "neuron":U], f: Table["row":R, "sample":S], y: Mutable[Scalar],
                 z: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + w.at(row=r, neuron=unit) * f.at(row=r, sample=sample)
    y = acc
    z = acc


TARGETS = [pytest.param("host"), pytest.param("cuda", marks=pytest.mark.gpu)]


def _xp(target):
    return np if target == "host" else pytest.importorskip("cupy")


def _np(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _plans(target, dtype, tmp_path):
    direct = eagle.plan.auto(contract, targets=(target,), cache_dir=str(tmp_path), scalar_type=dtype)
    with policy.using(contraction="gemm"):
        routed = eagle.plan.auto(contract, targets=(target,), cache_dir=str(tmp_path), scalar_type=dtype)
    return direct, routed


def _planes(dtype, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((R, U)).astype(dtype), rng.standard_normal((R, S)).astype(dtype),
            rng.standard_normal(S * U).astype(dtype))


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("target", TARGETS)
def test_the_routed_forward_is_the_loop_within_the_ulp_contract(target, dtype, tmp_path):
    xp, (direct, routed) = _xp(target), _plans(target, dtype, tmp_path)
    assert routed.plugin._gemm is not None and getattr(direct.plugin, "_gemm", None) is None
    w, f, _cot = _planes(dtype)
    y_d, y_r, again = (xp.zeros(S * U, dtype) for _ in range(3))
    direct.run(w=xp.asarray(w), f=xp.asarray(f), y=y_d)
    routed.run(w=xp.asarray(w), f=xp.asarray(f), y=y_r)
    routed.run(w=xp.asarray(w), f=xp.asarray(f), y=again)
    assert _rel(_np(y_r), _np(y_d)) <= (1e-15 if dtype == "float64" else 2 * np.finfo(np.float32).eps)
    assert np.array_equal(_np(y_r), _np(again))         # one device, one library build: the same bits


def _reverse(plan, xp, w, f, cot):
    y1, y2 = xp.zeros(S * U, w.dtype), xp.zeros(S * U, w.dtype)
    g = eagle.Graph()
    shared = xp.asarray(w)      # one table read by both launches
    g.add(plan.bind(w=shared, f=xp.asarray(f), y=y1), name="c1")
    g.add(plan.bind(w=shared, f=xp.asarray(f[::-1].copy()), y=y2), name="c2")
    g.build(record=True)
    g.launch()
    out = eagle.diff.reverse(g.record, {"c1.y": xp.asarray(cot), "c2.y": xp.asarray(cot[::-1].copy())},
                             returns="numpy" if xp is np else "cupy")
    return {k: _np(out[k]) for k in ("w", "c1.f", "c2.f")}


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("target", TARGETS)
def test_the_reverse_through_two_routed_consumers_is_the_hawk_vjp(target, dtype, tmp_path):
    xp, (direct, routed) = _xp(target), _plans(target, dtype, tmp_path)
    w, f, cot = _planes(dtype)
    want, got = _reverse(direct, xp, w, f, cot), _reverse(routed, xp, w, f, cot)
    assert type(routed.plugin._vjp).__name__ == "_VjpPlugin"
    for name in want:
        assert _rel(got[name], want[name]) <= (1e-15 if dtype == "float64" else 4 * np.finfo(np.float32).eps)
    if dtype == "float64":      # w is read by both consumers: its gradient is their sum
        def loss(w_):
            return float(cot @ (f.T @ w_).reshape(-1) + cot[::-1] @ (f[::-1].T @ w_).reshape(-1))
        h, fd = 1e-6, np.zeros_like(w)
        for i in range(R):
            for j in range(U):
                e = np.zeros_like(w)
                e[i, j] = h
                fd[i, j] = (loss(w + e) - loss(w - e)) / (2 * h)
        assert _rel(got["w"], fd) < 1e-8


def test_the_adjoint_adds_into_its_gradient_planes(tmp_path):
    """``accum_out`` sums: the routed vjp adds onto what the planes hold (beta = 1), as hawk's does."""
    from hawk import Kernel
    from hawk.diff import vjp

    from eagle._plan_deploy import _hawk_plugins

    _direct, routed = _plans("host", "float64", tmp_path)
    kernel, unit, cache = routed.plugin._hawk_source
    (hawk_vjp,) = _hawk_plugins([Kernel("contract_vjp", vjp(kernel, primal_unit=unit))], ("host",), cache)
    w, f, cot = _planes("float64")
    out = {}
    for name, plugin in (("hawk", hawk_vjp), ("gemm", routed.plugin._vjp)):
        bar_w, bar_f = np.ones((R, U)), np.full((R, S), 2.0)
        bound = eagle.plan.plan(plugin, structure=eagle.exec.HostTeam).bind(
            w=w, f=f, bar_y=cot.copy(), bar_w=bar_w, bar_f=bar_f)
        bound.launch()
        out[name] = (bar_w, bar_f)
    assert _rel(out["gemm"][0], out["hawk"][0]) <= 1e-15 and _rel(out["gemm"][1], out["hawk"][1]) <= 1e-15
    assert _rel(out["gemm"][0] - 1.0, f @ cot.reshape(S, U)) <= 1e-15


def test_a_kernel_with_two_outputs_stays_the_loop(tmp_path):
    with policy.using(contraction="gemm"):
        p = eagle.plan.auto(contract_two, targets=("host",), cache_dir=str(tmp_path))
    assert getattr(p.plugin, "_gemm", None) is None


def test_the_default_and_direct_leave_every_value_bit_identical(tmp_path):
    w, f, _cot = _planes("float64")
    ys = []
    for form in (None, "direct"):
        y = np.zeros(S * U)
        if form is None:
            p = eagle.plan.auto(contract, targets=("host",), cache_dir=str(tmp_path))
        else:
            with policy.using(contraction=form):
                p = eagle.plan.auto(contract, targets=("host",), cache_dir=str(tmp_path))
        assert getattr(p.plugin, "_gemm", None) is None and getattr(p.plugin, "_vjp", None) is None
        p.run(w=w, f=f, y=y)
        ys.append(y)
    assert ys[0].tobytes() == ys[1].tobytes()
