# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The opt-in GEMM route (``contraction="gemm"``):
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
F32_ULP = np.finfo(np.float32).eps


def _xp(target):
    return np if target == "host" else pytest.importorskip("cupy")


def _np(a):
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _plans(kernel, target, dtype, tmp_path):
    direct = eagle.plan.auto(kernel, targets=(target,), cache_dir=str(tmp_path), scalar_type=dtype)
    routed = eagle.plan.auto(kernel, targets=(target,), cache_dir=str(tmp_path), scalar_type=dtype, contraction="gemm")
    return direct, routed


def _planes(dtype, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((R, U)).astype(dtype), rng.standard_normal((R, S)).astype(dtype),
            rng.standard_normal(S * U).astype(dtype))


def _rel(a, b):
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


def _same(target, dtype, got, want, ulps=2):
    """The device route within the ulp contract; the host keeps the kernel, bit for bit."""
    if target == "host":
        assert got.tobytes() == want.tobytes()
    else:
        assert _rel(got, want) <= (1e-15 if dtype == "float64" else ulps * F32_ULP)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("target", TARGETS)
def test_the_routed_forward_is_the_loop_within_the_ulp_contract(target, dtype, tmp_path):
    xp, (direct, routed) = _xp(target), _plans(contract, target, dtype, tmp_path)
    assert routed.plugin._gemm is not None and getattr(direct.plugin, "_gemm", None) is None
    w, f, _cot = _planes(dtype)
    y_d, y_r, again = (xp.zeros(S * U, dtype) for _ in range(3))
    direct.run(w=xp.asarray(w), f=xp.asarray(f), y=y_d)
    routed.run(w=xp.asarray(w), f=xp.asarray(f), y=y_r)
    routed.run(w=xp.asarray(w), f=xp.asarray(f), y=again)
    _same(target, dtype, _np(y_r), _np(y_d))
    assert np.array_equal(_np(y_r), _np(again))         # one device, one library build: the same bits


def _reverse(plan, xp, w, f, cot, wrap=lambda node: node):
    from eagle.graph import Node

    y1, y2 = xp.zeros(S * U, w.dtype), xp.zeros(S * U, w.dtype)
    shared = xp.asarray(w)      # one table read by both launches
    g = eagle.Graph()
    g.add(plan.bind(w=shared, f=xp.asarray(f), y=y1), name="c1")
    g.add_node(wrap(Node("launch", plan.bind(w=shared, f=xp.asarray(f[::-1].copy()), y=y2), "c2")), name="c2")
    g.build(record=True)
    g.launch()
    out = eagle.diff.reverse(g.record, {"c1.y": xp.asarray(cot), "c2.y": xp.asarray(cot[::-1].copy())},
                             returns="numpy" if xp is np else "cupy")
    return {k: _np(out[k]) for k in ("w", "c1.f", "c2.f")}, (_np(y1), _np(y2))


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("target", TARGETS)
def test_the_reverse_through_two_routed_consumers_is_the_hawk_vjp(target, dtype, tmp_path):
    xp, (direct, routed) = _xp(target), _plans(contract, target, dtype, tmp_path)
    w, f, cot = _planes(dtype)
    (want, _), (got, _) = _reverse(direct, xp, w, f, cot), _reverse(routed, xp, w, f, cot)
    for name in want:
        _same(target, dtype, got[name], want[name], ulps=4)
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


@pytest.mark.gpu
def test_a_routed_launch_inside_a_skippable_runs_and_reverses(tmp_path):
    """The route's cuBLAS handle is made before capture on a conditional's body stream too."""
    cp = _xp("cuda")
    direct, routed = _plans(contract, "cuda", "float64", tmp_path)
    w, f, cot = _planes("float64")
    on = cp.ones(1, dtype=cp.uint32)

    def wrap(node):
        return eagle.skippable(node, eagle.SkipGuard.nonzero(on))
    (want, y_want), (got, y_got) = _reverse(direct, cp, w, f, cot, wrap), _reverse(routed, cp, w, f, cot, wrap)
    for a, b in zip(y_got, y_want):
        _same("cuda", "float64", a, b)
    for name in want:
        _same("cuda", "float64", got[name], want[name])


@pytest.mark.gpu
def test_the_adjoint_adds_into_its_gradient_planes(tmp_path):
    """``accum_out`` sums: the routed vjp adds onto what the planes hold (beta = 1), as hawk's does."""
    from hawk import Kernel
    from hawk.diff import vjp

    from eagle._plan_deploy import _hawk_plugins

    cp = _xp("cuda")
    _direct, routed = _plans(contract, "cuda", "float64", tmp_path)
    kernel, unit, cache = routed.plugin._hawk_source
    (hawk_vjp,) = _hawk_plugins([Kernel("contract_vjp", vjp(kernel, primal_unit=unit))], ("cuda",), cache)
    w, f, cot = _planes("float64")
    out = {}
    for name, plugin in (("hawk", hawk_vjp), ("gemm", routed.plugin._gemm.vjp)):
        bar_w, bar_f = cp.ones((R, U)), cp.full((R, S), 2.0)
        eagle.plan.plan(plugin, structure=eagle.exec.DeviceKernel).bind(
            w=cp.asarray(w), f=cp.asarray(f), bar_y=cp.asarray(cot), bar_w=bar_w, bar_f=bar_f).launch()
        out[name] = (bar_w.get(), bar_f.get())
    assert _rel(out["gemm"][0], out["hawk"][0]) <= 1e-15 and _rel(out["gemm"][1], out["hawk"][1]) <= 1e-15
    assert _rel(out["gemm"][0] - 1.0, f @ cot.reshape(S, U)) <= 1e-15


def test_a_kernel_with_two_outputs_stays_the_loop(tmp_path):
    p = eagle.plan.auto(contract_two, targets=("host",), cache_dir=str(tmp_path), contraction="gemm")
    assert getattr(p.plugin, "_gemm", None) is None


def test_the_default_and_direct_leave_every_value_bit_identical(tmp_path):
    w, f, _cot = _planes("float64")
    ys = []
    for form in (None, "direct"):
        y = np.zeros(S * U)
        if form is None:
            p = eagle.plan.auto(contract, targets=("host",), cache_dir=str(tmp_path))
        else:
            p = eagle.plan.auto(contract, targets=("host",), cache_dir=str(tmp_path), contraction=form)
        assert getattr(p.plugin, "_gemm", None) is None and getattr(p.plugin, "_vjp", None) is None
        p.run(w=w, f=f, y=y)
        ys.append(y)
    assert ys[0].tobytes() == ys[1].tobytes()


@hawk.kernel
def contract_staged(w: Table["row":R, "neuron":U], feat: hawk.Staged[hawk.Vector[R], m.n_samples],
                    y: Mutable[Scalar]):
    sample, unit = m.split_index(U)
    acc = 0.0
    for r in range(R):
        acc = acc + w.at(row=r, neuron=unit) * feat.at(plane=r, sample=sample)
    y = acc


@pytest.mark.parametrize("dtype", ["float64", "float32"])
@pytest.mark.parametrize("target", TARGETS)
def test_a_staged_plane_counted_by_the_launch_routes_within_the_ulp_contract(target, dtype, tmp_path):
    """``Staged[..., n_samples]``: the free extent is the run's, ``n // U``; the plane's stride is ``n``."""
    xp, (direct, routed) = _xp(target), _plans(contract_staged, target, dtype, tmp_path)
    assert routed.plugin._gemm is not None
    w, f, _cot = _planes(dtype)
    n = S * U
    feat = np.zeros((R, n), dtype)
    feat[:, :S] = f
    y_d, y_r = xp.zeros(n, dtype), xp.zeros(n, dtype)
    direct.run(w=xp.asarray(w), feat=xp.asarray(feat), y=y_d)
    routed.run(w=xp.asarray(w), feat=xp.asarray(feat), y=y_r)
    _same(target, dtype, _np(y_r), _np(y_d))
    assert _rel(_np(y_r), (f.astype(np.float64).T @ w.astype(np.float64)).reshape(-1)) < 1e-6
