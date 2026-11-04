# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The eagle/hawk feature card: one row per backend feature, each at ONE size against its natural baseline.

Where ``perf_card.py`` times one workload (adaptive Kepler orbits) across arms and sizes, this card times eleven
features (``ROWS``), every arm on the kernel the feature is about:

``wide_forms``     hawk's wide-value kernels in the thread, team and staged forms, against an opaque NumPy mask and torch
``steps_fused``    a fused ``steps=K`` body (K = 1, 8, 32), its jvp, and its vjp fused and as the auto table picks it
``control_flow``   a runtime trip count with a 4000 cap: guard, no guard, a ``jump=`` switch, segmented units, against Warp
``linalg_lane``    per-lane solve + eigh at n = 8 and 32, against cuSOLVER (CuPy) and torch.linalg
``linalg_team``    a team solve at n = 512
``contraction``    64 x 64 tables, a shared-axis sum (the thread box shape): box, no box, the GEMM route, against einsum
``fft``            batched c2c (1e5 x 1024) against cuFFT
``lane_set``       10 % active lanes: dense, map, prefix
``scatter``        keyed, ordered and OR scatter against ``cupyx.scatter_add`` and ``np.add.at``
``sample_sum``     a sum over samples, keyed and as a cell (``hawk.sum(x)``): compensated, ordered, auto, against ``cupy.sum``
``mpi_partition``  the C1 ensemble through ``RankPartition`` on 2 GPUs (N = 2e6) against one rank (N = 1e6)

Every (row, case, dtype, arm) runs in a FRESH PROCESS on a clean device, so no arm inherits another library's context,
pool or fragmentation. An allocation failure is the arm ``out_of_memory``; any other failure is ``failed: <reason>``; a
warm-up run over 120 s is ``not_run: budget``; timed rounds stop at 300 s. The card always completes. Each arm checks its
values against an independent oracle (NumPy, a closed form, the dense launch) before it is timed. An arm whose library is
absent reads ``not_run: not installed: <library>``.

Usage::

    python benchmarks/perf_card/feature_card.py [--rows ROW ...] [--smoke] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import json
import math
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))  # benchmarks/: the cards' shared helpers
import _card_common as cc  # noqa: E402

SCRIPT_REL = "benchmarks/perf_card/feature_card.py"
RUN_BUDGET_S = 120.0     # a warm-up run over this is not_run: budget
CELL_CAP_S = 300.0       # timed rounds stop here (at least one runs)
MIN_ROUNDS, MAX_ROUNDS, IQR_STOP = 3, 9, 0.05
PROCESS_TIMEOUT_S = cc.BUILD_S + RUN_BUDGET_S + CELL_CAP_S + RUN_BUDGET_S   # setup, warm-up, rounds, a last round
MARK = "FEATURE_ARM "
PROFILE_ENV = "FEATURE_CARD_PROFILE"   # set by reproduce/gpu_profile.py: one timed call alone inside a profiler range
F64, F32 = "float64", "float32"


class NotRun(Exception):
    """An arm that cannot run here; the text is the reason."""


def _installed(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _need(module):
    if not _installed(module):
        raise NotRun(f"not installed: {module}")
    return importlib.import_module(module)


def _gpu_count():
    if not _installed("cupy"):
        return 0
    import cupy as cp

    try:
        return cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        return 0


def _hawk_tests():
    """hawk's ``tests/`` (the wide-value corpus lives there), on ``sys.path``."""
    tests = pathlib.Path(os.environ.get("HAWK_SRC", REPO.parent / "hawk")) / "tests"
    if not (tests / "_wide_kernels.py").exists():
        raise NotRun("hawk tests/_wide_kernels.py not found (set HAWK_SRC)")
    sys.path.insert(0, str(tests))
    return tests


def _wk():
    _hawk_tests()
    return importlib.import_module("_wide_kernels")


def _kernels():
    spec = importlib.util.spec_from_file_location("feature_kernels", HERE / "feature_kernels.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- #
# Arrays and launches
# --------------------------------------------------------------------------- #
def _np():
    import numpy as np

    return np


def _real(dtype):
    return _np().float32 if dtype == F32 else _np().float64


def _xp(ctx):
    if ctx.target == "cuda":
        import cupy

        return cupy
    return _np()


def _get(a):
    return _np().asarray(a.get() if hasattr(a, "get") else a)


def _sync(ctx):
    if ctx.target == "cuda":
        import cupy

        cupy.cuda.runtime.deviceSynchronize()


def _plan(kernel, ctx, form=None):
    import eagle.plan

    cache = os.environ.get("FEATURE_CARD_CACHE") or tempfile.mkdtemp(prefix="feature_cache_")
    if form is not None:
        from hawk.targets.cards import _probe_wide
        return _probe_wide._auto(kernel, ctx.dtype, cache, form)
    return eagle.plan.auto(kernel, targets=(ctx.target,), scalar_type=ctx.dtype, cache_dir=cache)


def _planes(ctx, ins, outs):
    """Device (or host) planes: ``ins`` with floats cast to the row's precision, and zeroed ``outs``
    (``{name: (shape, dtype or None for the real type)}``)."""
    np, xp, real = _np(), _xp(ctx), _real(ctx.dtype)
    planes = {k: xp.asarray(np.ascontiguousarray(v.astype(real) if v.dtype.kind == "f" else v)) for k, v in ins.items()}
    planes.update({k: xp.zeros(shape, real if dt is None else dt) for k, (shape, dt) in outs.items()})
    return planes


def _bound(kernel, planes, ctx, form=None, **extent):
    return _plan(kernel, ctx, form).bind(**planes, **extent).launch


def _launcher(kernel, ins, outs, ctx, form=None, **extent):
    """``(run, planes)``: ``kernel`` built for the target and bound; ``run`` launches and waits."""
    planes = _planes(ctx, ins, outs)
    launch = _bound(kernel, planes, ctx, form, **extent)

    def run():
        launch()
        _sync(ctx)
    return run, planes


def _discard(fn):
    """``fn`` as a timed callable that returns nothing."""
    def timed():
        fn()
    return timed


def _close(got, want, dtype, scale=1.0):
    np = _np()
    tol = (2e-4 if dtype == F32 else 1e-9) * scale
    return bool(np.allclose(got, want, rtol=tol, atol=tol * max(1.0, float(np.abs(want).max()))))


# --------------------------------------------------------------------------- #
# Rows: each setup returns (run, check); check() -> (ok, what)
# --------------------------------------------------------------------------- #
def setup_wide_forms(ctx):
    np = _np()
    wk, real = _wk(), _real(ctx.dtype)
    names = ("fill_mask", "matvec_in", "masked_sum")
    data = {nm: wk.CORPUS[nm].planes(np.random.default_rng(7), real, ctx.n, True) for nm in names}
    want = {nm: wk.CORPUS[nm].want(real, data[nm], ctx.n) for nm in names}
    if ctx.arm == "numpy_mask":
        return (lambda: [wk.CORPUS[nm].want(real, data[nm], ctx.n) for nm in names]), (lambda: (True, "the NumPy reference"))
    if ctx.arm == "torch":
        torch = _need("torch")
        dev = "cuda" if ctx.target == "cuda" else "cpu"
        t = {nm: {k: torch.as_tensor(v, device=dev) for k, v in data[nm].items()} for nm in names[:2]}
        scale = (torch.arange(wk.W, device=dev, dtype=t["fill_mask"]["a"].dtype) + 1.5)[:, None]

        def run():
            outs = ((t["fill_mask"]["a"][None, :] * scale) * t["fill_mask"]["swg"][:, None], t["matvec_in"]["w"] @ t["matvec_in"]["x"])
            if dev == "cuda":
                torch.cuda.synchronize()
            return outs
        return _discard(run), (lambda: (all(_close(o.cpu().numpy(), want[nm], ctx.dtype) for o, nm in zip(run(), names)),
                                                    "fill_mask and matvec_in match the reference (masked_sum has no torch form)"))
    # the arm is a named form on a GPU (the host build has one form)
    form, extra = {"thread": ("thread", {}), "team": ("team", {"TC": 1}), "team_k8": ("team", {"TC": 1, "WARPS": 8}),
                  "staged": ("team", {})}[ctx.arm]
    if ctx.target != "cuda":
        form = None
    from hawk.targets.cards import wide_override

    runs, outs = [], {}
    with wide_override({"SLICE_MIN": 32, "TEAM_MIN": 32, **extra}) if ctx.target == "cuda" else contextlib.nullcontext():
        for nm in names:
            entry = wk.CORPUS[nm]
            run, planes = _launcher(entry.kernel, data[nm], {entry.out: (entry.shape(ctx.n), None)}, ctx, form)
            run()
            runs.append(run)
            outs[nm] = planes[entry.out]
    return _discard(lambda: [r() for r in runs]), (lambda: (all(np.array_equal(_get(outs[nm]), want[nm]) for nm in names),
                                                              "every output equals the reference bit for bit"))


def setup_steps_fused(ctx):
    np, wk = _np(), _wk()
    import hawk
    from hawk.diff import step_jvp, step_vjp

    real, n, xp, arm = _real(ctx.dtype), ctx.n, _xp(ctx), ctx.arm
    p = wk.CORPUS["steps_matvec_k8"].planes(np.random.default_rng(7), real, n, True)
    x, w = p["x"], p["w"]
    base = w.astype(np.float64) @ x.astype(np.float64)
    rng = np.random.default_rng(11)

    def ints(*shape):
        return rng.integers(-3, 4, shape).astype(real)

    unit = "0" * 64
    k1 = wk.steps_matvec_k1
    state = {"x": x, "w": w, "c": np.zeros(n, real), "terminated": np.zeros(n, bool), "finished_count": np.zeros(1, np.uint32)}

    def resetter(planes):
        def reset():
            planes["c"][...] = 0
            planes["terminated"][...] = False
            planes["finished_count"][...] = 0
        return reset

    if arm in ("K1", "K8", "K32"):
        K = int(arm[1:])
        kernel = {1: k1, 8: wk.steps_matvec_k8}.get(K) or hawk.kernel(wk.steps_matvec_body, steps=K)
        planes = _planes(ctx, state, {"y": ((wk.ROWS, n), None)})
        launch, reset, launches = _bound(kernel, planes, ctx), resetter(planes), 8 if K == 1 else 1

        def run():
            reset()
            for _ in range(launches):
                launch()
            _sync(ctx)
        return run, (lambda: (run() or _close(_get(planes["y"]), base + 7, ctx.dtype), "y equals W x + c after 8 steps"))
    if arm == "jvp_K8":
        dx, dw, dc = ints(wk.W, n), ints(wk.ROWS, wk.W), ints(n)
        planes = _planes(ctx, {**state, "dot_x": dx, "dot_w": dw, "dot_c": dc}, {"y": ((wk.ROWS, n), None), "dot_y": ((wk.ROWS, n), None)})
        launch, reset, fresh = _bound(hawk.steps(step_jvp(k1), 8), planes, ctx), resetter(planes), xp.asarray(dc)

        def run():
            reset()
            planes["dot_c"][...] = fresh
            launch()
            _sync(ctx)
        want = w.astype(np.float64) @ dx + dw.astype(np.float64) @ x + dc
        return run, (lambda: (run() or _close(_get(planes["dot_y"]), want, ctx.dtype), "dot_y equals W dx + dW x + dc"))
    if arm in ("vjp_K8_fused", "vjp_K8_auto"):
        by, bc = ints(wk.ROWS, n), ints(n)
        want_x, want_c = w.astype(np.float64).T @ by, bc + by.sum(axis=0)
        choice = "fused"
        if arm.endswith("auto"):
            from hawk.ir import steps_class

            props = importlib.import_module("eagle.launch")._device_props(-1) if ctx.target == "cuda" else {}
            resident = int(props.get("multiProcessorCount", 0)) * int(props.get("maxThreadsPerMultiProcessor", 0))
            choice = steps_class._choose(ctx.target == "cuda", 2, 8, n, resident)
            ctx.note = f"steps_reverse auto picks {choice}"
        outs = {"bar_x": ((wk.W, n), None), "bar_c": ((n,), None)}
        if choice == "fused":
            fused = hawk.Kernel("steps_matvec_vjp_fused", step_vjp(hawk.steps(k1, 8), wrt=("c", "x"), steps_reverse="fused", primal_unit=unit))
            planes = _planes(ctx, {"x": x, "w": w, "c": np.zeros(n, real), "terminated": np.zeros(n, bool), "y": np.zeros((wk.ROWS, n), real),
                                   "bar_y_next": by, "bar_c_next": bc}, outs)
            launch = _bound(fused, planes, ctx)

            def run():
                planes["terminated"][...] = False
                launch()
                _sync(ctx)
            out_x = planes["bar_x"]
        else:
            single = hawk.Kernel("steps_matvec_vjp_step", step_vjp(k1, wrt=("c", "x"), primal_unit=unit))
            planes = _planes(ctx, {"w": w, "bar_y_next": by, "bar_c_next": bc}, outs)
            launch, seeds = _bound(single, planes, ctx), (xp.asarray(by), xp.zeros((wk.ROWS, n), real))
            out_x, first = xp.zeros((wk.W, n), real), xp.asarray(bc)

            def run():
                out_x[...] = 0
                planes["bar_c_next"][...] = first
                for k in reversed(range(8)):
                    planes["bar_y_next"][...] = seeds[0 if k == 7 else 1]
                    launch()
                    out_x[...] += planes["bar_x"]
                    planes["bar_c_next"][...] = planes["bar_c"]
                _sync(ctx)
        return run, (lambda: (run() or (_close(_get(out_x), want_x, ctx.dtype) and _close(_get(planes["bar_c"]), want_c, ctx.dtype)),
                              "bar_x equals W^T bar_y and bar_c equals bar_c_next + sum(bar_y)"))
    if arm == "warp_loop":
        return _warp_matvec(ctx, x, w, base)
    if arm == "torch_func":
        return _torch_func_vjp(ctx, x, w, ints(w.shape[0], n))
    raise NotRun(f"unknown arm {arm}")


def _warp_matvec(ctx, x, w, base):
    wp = _need("warp")
    if ctx.dtype != F64 or ctx.target != "cuda":
        raise NotRun("the Warp loop runs in float64 on a GPU")
    wp.config.quiet = True
    wp.init()
    rows, width = w.shape
    n = x.shape[1]

    @wp.kernel
    def matvec(x: wp.array2d(dtype=wp.float64), w: wp.array2d(dtype=wp.float64), y: wp.array2d(dtype=wp.float64)):
        i = wp.tid()
        for step in range(8):
            for r in range(rows):
                acc = wp.float64(0.0)
                for k in range(width):
                    acc += w[r, k] * x[k, i]
                y[r, i] = acc + wp.float64(step)

    xw, ww = wp.array(x, dtype=wp.float64, device="cuda"), wp.array(w, dtype=wp.float64, device="cuda")
    yw = wp.zeros((rows, n), dtype=wp.float64, device="cuda")

    def run():
        wp.launch(matvec, dim=n, inputs=[xw, ww, yw], device="cuda")
        wp.synchronize()
    return run, (lambda: (run() or _close(yw.numpy(), base + 7, ctx.dtype), "y equals W x + c after 8 steps"))


def _torch_func_vjp(ctx, x, w, by):
    torch = _need("torch")
    np = _np()
    dev = "cuda" if ctx.target == "cuda" else "cpu"
    tx, tw, tc, tby = (torch.as_tensor(a, device=dev) for a in (x, w, np.zeros(x.shape[1], x.dtype), by))

    def step_loop(xx, cc_):
        for k in range(8):
            y = tw @ xx + (cc_ + k)[None, :]
        return y

    def run():
        _, pull = torch.func.vjp(step_loop, tx, tc)
        out = pull(tby)
        if dev == "cuda":
            torch.cuda.synchronize()
        return out
    want_x = w.astype(np.float64).T @ by
    return _discard(run), (lambda: (_close(run()[0].cpu().numpy(), want_x, ctx.dtype), "bar_x equals W^T bar_y"))


def setup_control_flow(ctx):
    np, fk = _np(), _kernels()
    n, cap = ctx.n, (50 if ctx.smoke else _kernels().CAP)
    rng = np.random.default_rng(5)
    k = rng.integers(0, 3, n).astype(np.int32)
    steps = rng.integers(1, cap + 1, n).astype(np.int32)
    v = rng.uniform(-1.0, 1.0, n)
    head = min(n, 1024)
    want = fk.loop_want(v[:head], steps[:head], k[:head], jump=ctx.arm == "jump")
    what = "y equals the NumPy loop on the first 1024 lanes"
    if ctx.arm in ("guard", "no_guard", "jump"):
        kernel = {"guard": fk.loop_guard, "no_guard": fk.loop_noguard, "jump": fk.loop_jump}[ctx.arm]
        run, planes = _launcher(kernel, {"k": k, "n": steps, "v": v}, {"y": ((n,), None)}, ctx)
        return run, (lambda: (run() or _close(_get(planes["y"])[:head], want, ctx.dtype), what))
    if ctx.target != "cuda":
        raise NotRun("needs a GPU")
    if ctx.arm == "segmented":
        import eagle
        from eagle import arrays

        os.environ["HAWK_CACHE_DIR"] = os.environ.get("FEATURE_CARD_CACHE") or tempfile.mkdtemp(prefix="feature_cache_")
        got = {}

        def up(a):
            return arrays.asarray(_np().ascontiguousarray(a), device="cuda:0")

        def run():
            res = eagle.simulate(fk.loop_segmented, max_steps=int(cap), dispatch="segmented", scalar_type=ctx.dtype,
                                 k=up(k.astype(np.int64)), count=up(steps.astype(np.float64)), v=up(v), c=up(np.zeros(n)))
            got["v"] = _get(res.state["v"])
        return run, (lambda: (run() or _close(got["v"][:head], want, ctx.dtype), what))
    wp = _need("warp")
    wp.config.quiet = True
    wp.init()

    @wp.kernel
    def loop(kk: wp.array(dtype=wp.int32), nn: wp.array(dtype=wp.int32), vv: wp.array(dtype=wp.float64), y: wp.array(dtype=wp.float64)):
        i = wp.tid()
        a = vv[i]
        bias = wp.float64(kk[i]) * wp.float64(0.001)
        for _ in range(nn[i]):
            a = wp.sin(a) * wp.float64(0.5) + a * wp.float64(0.5) + wp.float64(0.01) + bias
        y[i] = a
    wk_, wn_, wv_ = (wp.array(a, device="cuda") for a in (k, steps, v))
    wy = wp.zeros(n, dtype=wp.float64, device="cuda")

    def run():
        wp.launch(loop, dim=n, inputs=[wk_, wn_, wv_, wy], device="cuda")
        wp.synchronize()
    return run, (lambda: (run() or _close(wy.numpy()[:head], want, ctx.dtype), what))


def _spd(rng, n, count):
    np = _np()
    a = rng.random((n, n, count))
    return (a + a.transpose(1, 0, 2)) / 2 + n * np.eye(n)[:, :, None]


def _linalg(ctx, n, solve_eigh):
    np, fk = _np(), _kernels()
    count, real = ctx.n, _real(ctx.dtype)
    rng = np.random.default_rng(n)
    A, b = _spd(rng, n, count), rng.random((n, count))
    stack, rhs = np.moveaxis(A, 2, 0), b.T[:, :, None]
    want_x = np.linalg.solve(stack, rhs)[:, :, 0].T
    if ctx.arm in ("hawk", "team"):
        outs = {"x": ((n, count), None), "info": ((count,), np.int32)}
        if solve_eigh:
            outs["w"] = ((n, count), None)
        run, planes = _launcher(fk.solve_eigh(n) if solve_eigh else fk.solve_only(n), {"A": A.reshape(n * n, count), "b": b}, outs, ctx)

        def check():
            run()
            ok = _close(_get(planes["x"]), want_x, ctx.dtype, 50.0)
            if solve_eigh:
                ok = ok and _close(np.sort(_get(planes["w"]), axis=0), np.linalg.eigvalsh(stack).T, ctx.dtype, 50.0)
            return ok, "x" + (" and eigenvalues" if solve_eigh else "") + " match NumPy"
        return run, check
    if ctx.arm == "cupy":
        if ctx.target != "cuda":
            raise NotRun("needs a GPU")
        cp = _need("cupy")
        sc, bc = cp.asarray(stack.astype(real)), cp.asarray(rhs.astype(real))

        def run():
            out = cp.linalg.solve(sc, bc)
            if solve_eigh:
                cp.linalg.eigvalsh(sc)
            cp.cuda.runtime.deviceSynchronize()
            return out.get()
    else:
        torch = _need("torch")
        dev = "cuda" if ctx.target == "cuda" else "cpu"
        sc, bc = torch.as_tensor(stack.astype(real), device=dev), torch.as_tensor(rhs.astype(real), device=dev)

        def run():
            out = torch.linalg.solve(sc, bc)
            if solve_eigh:
                torch.linalg.eigvalsh(sc)
            if dev == "cuda":
                torch.cuda.synchronize()
            return out.cpu().numpy()
    return _discard(run), (lambda: (_close(run()[:, :, 0].T, want_x, ctx.dtype, 50.0), "x matches NumPy"))


def setup_linalg_lane(ctx):
    return _linalg(ctx, int(ctx.case[1:]), True)


def setup_linalg_team(ctx):
    return _linalg(ctx, 512, False)


def setup_contraction(ctx):
    np, fk = _np(), _kernels()
    planes_in = fk.contract_planes(_real(ctx.dtype))
    want = fk.contract_want(planes_in)
    if ctx.arm in ("box", "no_box"):
        if ctx.arm == "no_box":
            os.environ["HAWK_BOX"] = "0"
        run, planes = _launcher(fk.contract, planes_in, {"y": ((fk.ROWS * fk.SAMPLES,), None)}, ctx)
        return run, (lambda: (run() or _close(_get(planes["y"]), want, ctx.dtype), "y equals the NumPy matrix product"))
    if ctx.arm == "gemm":
        from hawk.emit import policy

        with policy.using(contraction="gemm"):      # the opt-in route: one GEMM into the lane plane
            run, planes = _launcher(fk.contract, planes_in, {"y": ((fk.ROWS * fk.SAMPLES,), None)}, ctx)
        return run, (lambda: (run() or _close(_get(planes["y"]), want, ctx.dtype), "y equals the NumPy matrix product"))
    if ctx.target != "cuda":
        raise NotRun("needs a GPU")
    cp = _need("cupy")
    w, f = cp.asarray(planes_in["w"]), cp.asarray(planes_in["f"])
    out = {}

    def run():
        out["y"] = cp.einsum("ru,rs->su", w, f)
        cp.cuda.runtime.deviceSynchronize()
    return run, (lambda: (run() or _close(_get(out["y"]).reshape(-1), want, ctx.dtype), "einsum equals the NumPy matrix product"))


def setup_fft(ctx):
    np = _np()
    count, n = ctx.n, 1024
    pairs = np.random.default_rng(5).uniform(-1, 1, (count, n, 2))
    want = np.fft.fft(pairs[:8, :, 0] + 1j * pairs[:8, :, 1], axis=1)
    if ctx.target != "cuda":
        raise NotRun("needs a GPU")
    if ctx.arm == "fft":
        import eagle
        import eagle.fft as F
        from eagle import arrays

        out = arrays.zeros((count, n, 2), "float64", "cuda:0")
        graph = eagle.Graph().add_node(F.c2c(arrays.asarray(pairs, "float64", "cuda:0"), out, (1,), forward=True)).build()

        def run():
            graph.launch()
            _sync(ctx)

        def check():
            run()
            got = _get(out)[:8]
            return bool(np.abs(got[..., 0] + 1j * got[..., 1] - want).max() < 1e-9 * n), "eagle.fft equals numpy.fft on 8 rows"
        return run, check
    cp = _need("cupy")
    z = cp.asarray(pairs.view(np.complex128)[..., 0])
    out = {}

    def run():
        out["y"] = cp.fft.fft(z, axis=1)
        cp.cuda.runtime.deviceSynchronize()
    return run, (lambda: (run() or bool(np.abs(out["y"][:8].get() - want).max() < 1e-9 * n), "cupy.fft equals numpy.fft on 8 rows"))


def setup_lane_set(ctx):
    np, fk = _np(), _kernels()
    n = ctx.n
    live = max(1, n // 10)
    x = np.linspace(0.0, 3.0, n)
    want = x
    for _ in range(8):
        want = np.sin(want) * 0.5 + np.cos(want * 1.3) * 0.25 + want * 0.1
    ins, ids, kernel = {"x": x}, np.arange(n), fk.dense
    if ctx.arm == "map":
        ids = np.sort(np.random.default_rng(2).choice(n, live, replace=False)).astype(np.int32)
        kernel, ins = fk.mapped, {"x": x, "active_map": ids, "active_count": np.array([live], np.uint32)}
    elif ctx.arm == "prefix":
        ids, kernel = np.arange(live), fk.prefix
        ins = {"x": x, "active_count": np.array([live], np.uint32)}
    run, planes = _launcher(kernel, ins, {"y": ((n,), None)}, ctx)

    def check():
        run()
        y, rest = _get(planes["y"]), np.ones(n, bool)
        rest[ids] = False
        return bool(np.allclose(y[ids], want[ids], rtol=1e-12, atol=1e-12) and not y[rest].any()), "live lanes equal the NumPy body; the rest are untouched"
    return run, check


def setup_scatter(ctx):
    np, fk = _np(), _kernels()
    n, nodes = ctx.n, (64 if ctx.smoke else 1024)
    rng = np.random.default_rng(1)
    key, v = rng.integers(0, nodes, n).astype(np.int32), rng.random(n)
    want = np.zeros(nodes)
    np.add.at(want, key, v)
    if ctx.arm in ("keyed", "ordered"):
        launch, planes = _launcher(fk.keyed_sum if ctx.arm == "keyed" else fk.ordered_sum, {"key": key, "v": v}, {"rs": ((nodes,), None)}, ctx)

        def run():
            planes["rs"][...] = 0
            launch()
        return run, (lambda: (run() or _close(_get(planes["rs"]), want, ctx.dtype), "keyed sums match np.add.at"))
    if ctx.arm == "or":
        bits = (np.uint32(1) << rng.integers(0, 32, n).astype(np.uint32)).astype(np.uint32)
        launch, planes = _launcher(fk.or_mask, {"word": key, "bit": bits}, {"mask": ((nodes,), np.uint32)}, ctx)
        ref = np.zeros(nodes, np.uint32)
        np.bitwise_or.at(ref, key, bits)

        def run():
            planes["mask"][...] = 0
            launch()
        return run, (lambda: (run() or bool(np.array_equal(_get(planes["mask"]), ref)), "the OR mask equals np.bitwise_or.at"))
    if ctx.arm == "np_add_at":
        def run():
            np.add.at(np.zeros(nodes), key, v)
        return run, (lambda: (True, "the NumPy reference"))
    if ctx.target != "cuda":
        raise NotRun("needs a GPU")
    cp, cupyx = _need("cupy"), _need("cupyx")
    kc, vc, rc = cp.asarray(key), cp.asarray(v), cp.zeros(nodes)

    def run():
        rc[...] = 0
        cupyx.scatter_add(rc, kc, vc)
        cp.cuda.runtime.deviceSynchronize()
    return run, (lambda: (run() or _close(rc.get(), want, ctx.dtype), "scatter_add matches np.add.at"))


def setup_sample_sum(ctx):
    np, fk = _np(), _kernels()
    x = np.random.default_rng(9).uniform(-1.0, 1.0, ctx.n)
    exact, mag, eps = math.fsum(x), float(np.abs(x).sum()), float(np.finfo(np.float64).eps)
    if ctx.arm.startswith("cell_"):       # the public cell form ``hawk.sum(x)``: its rows are folded by eagle
        launch, planes = _launcher(fk.total_cell(ctx.arm[5:], ctx.n), {"x": x}, {"t": ((1,), None)}, ctx)
        bound = (1.0 if ctx.arm == "cell_compensated" else 8.0) * eps * mag

        def run():
            planes["t"][...] = 0
            launch()
        return run, (lambda: (run() or abs(float(_get(planes["t"])[0]) - exact) <= bound,
                              "the cell is within the policy's ULP bound of math.fsum"))
    if ctx.arm in ("compensated", "ordered", "auto"):
        outs = {"out": ((1,), None), **({"out_c": ((1,), None)} if ctx.arm == "compensated" else {})}
        launch, planes = _launcher(fk.total(ctx.arm, ctx.n), {"x": x}, outs, ctx, key=1)

        def run():
            for p in planes.values():
                if p.shape == (1,):
                    p[...] = 0
            launch()
        bound = (1.0 if ctx.arm == "compensated" else 8.0) * eps * mag
        return run, (lambda: (run() or abs(sum(float(_get(p)[0]) for k, p in planes.items() if k.startswith("out")) - exact) <= bound,
                              "the sum is within the policy's ULP bound of math.fsum"))
    if ctx.target != "cuda":
        raise NotRun("needs a GPU")
    cp = _need("cupy")
    xc, out = cp.asarray(x), {}

    def run():
        out["t"] = cp.sum(xc)
        cp.cuda.runtime.deviceSynchronize()
    return run, (lambda: (run() or abs(float(out["t"]) - exact) <= 8.0 * eps * mag, "cupy.sum is within 8 ULP of math.fsum"))


def _perf_card():
    sys.path.insert(0, str(HERE))
    return importlib.import_module("perf_card")


def _persist_arm(pc, n, inp):
    """The C1 workload's persist arm over ``n`` samples: (run, download) with the inputs uploaded."""
    arm = pc._make_eagle_arms(n, ["eagle_graph_persist"])["eagle_graph_persist"]
    arm["state"].upload(inp)

    def run():
        arm["state"].upload(inp)
        arm["run"]()
    return run, arm["download"]


def setup_mpi_partition(ctx):
    """The one-rank arm: the C1 ensemble (float64, spread) at N / 2 on one GPU; the two-rank arm is ``_mpi_worker``."""
    if ctx.arm != "one_rank":
        raise NotRun("run by the parent under mpirun")
    if ctx.target != "cuda":
        raise NotRun("needs a GPU")
    pc = _perf_card()
    inp = {k: v[: ctx.n // 2] for k, v in pc._inputs(ctx.n // 2).items() if hasattr(v, "__len__")}
    run, download = _persist_arm(pc, ctx.n // 2, inp)
    return run, (lambda: (run() or bool(_np().isfinite(download()["t"]).all()), "every sample reaches a finite time"))


# The order of rows and arms is the priority order: a session deadline drops the tail. Within a row the eagle arms come first, then
# the fast libraries, the slow ones last; the rows run headline first (the wide-value forms, then the fused steps) and the
# multi-GPU row last.
# name -> spec. ``cases``: {case: (N, smoke N)}; ``arms``: eagle arms then baselines; ``host``: the arms that run on the CPU
# session (empty: GPU only); ``ref``: the arm the others are compared with.
ROWS = {
    "wide_forms": dict(setup=setup_wide_forms, cases={"all": (65536, 256)}, dtypes=(F64, F32), ref="thread",
                       arms=("thread", "team", "team_k8", "staged", "numpy_mask", "torch"), host=("thread", "numpy_mask", "torch"),
                       baseline="opaque NumPy mask; torch"),
    "steps_fused": dict(setup=setup_steps_fused, cases={"all": (65536, 256)}, dtypes=(F64,), ref="K8",
                        arms=("K1", "K8", "K32", "jvp_K8", "vjp_K8_fused", "vjp_K8_auto", "warp_loop", "torch_func"),
                        host=("K1", "K8", "K32", "jvp_K8", "vjp_K8_fused", "vjp_K8_auto", "torch_func"),
                        baseline="Warp loop (primal); torch.func over a loop (derivatives)"),
    "control_flow": dict(setup=setup_control_flow, cases={"all": (1_000_000, 4096)}, dtypes=(F64,), ref="guard",
                         arms=("guard", "no_guard", "jump", "segmented", "warp"), host=("guard", "no_guard", "jump"),
                         baseline="Warp per-thread loop"),
    "linalg_lane": dict(setup=setup_linalg_lane, cases={"n8": (131072, 250), "n32": (8192, 250)}, dtypes=(F64, F32), ref="hawk",
                        arms=("hawk", "cupy", "torch"), host=("hawk", "torch"), baseline="cuSOLVER batched (CuPy); torch.linalg"),
    "linalg_team": dict(setup=setup_linalg_team, cases={"n512": (16, 2)}, dtypes=(F64,), ref="team",
                        arms=("team", "cupy", "torch"), host=(), baseline="CuPy; torch.linalg"),
    "contraction": dict(setup=setup_contraction, cases={"all": (65536, 65536)}, dtypes=(F64, F32), ref="box",
                        arms=("box", "no_box", "gemm", "einsum"), host=("box", "no_box", "gemm"), baseline="cupy.einsum"),
    "fft": dict(setup=setup_fft, cases={"all": (100_000, 64)}, dtypes=(F64,), ref="fft", arms=("fft", "cupy_fft"), host=(),
                baseline="cupy.fft (cuFFT)"),
    "lane_set": dict(setup=setup_lane_set, cases={"all": (1 << 22, 4096)}, dtypes=(F64,), ref="dense", arms=("dense", "map", "prefix"),
                     host=(), baseline="none: ratio to dense"),
    "scatter": dict(setup=setup_scatter, cases={"all": (1 << 22, 4096)}, dtypes=(F64,), ref="keyed",
                    arms=("keyed", "ordered", "or", "cupyx_scatter_add", "np_add_at"), host=("keyed", "ordered", "or", "np_add_at"),
                    baseline="cupyx.scatter_add; np.add.at"),
    "sample_sum": dict(setup=setup_sample_sum, cases={"all": (1 << 24, 4096)}, dtypes=(F64,), ref="auto",
                       arms=("compensated", "ordered", "auto", "cell_compensated", "cell_auto", "cupy_sum"),
                       host=("compensated", "ordered", "auto", "cell_compensated", "cell_auto"), baseline="cupy.sum"),
    "mpi_partition": dict(setup=setup_mpi_partition, cases={"all": (2_000_000, 8192)}, dtypes=(F64,), ref="one_rank",
                          arms=("two_ranks", "one_rank"), host=(), baseline="one rank (N = 1e6), the C1 persist arm"),
}
# Every arm is ours (eagle / hawk code: the profile kit captures its kernels) or external (a baseline library: never profiled);
# a new arm must be added to exactly one set (tests/test_profile_kit.py fails otherwise).
EXTERNAL_ARMS = frozenset({"numpy_mask", "torch", "warp_loop", "torch_func", "cupy", "einsum", "cupy_fft", "cupyx_scatter_add",
                           "np_add_at", "cupy_sum", "warp"})
OUR_ARMS = frozenset({"thread", "team", "team_k8", "staged", "K1", "K8", "K32", "jvp_K8", "vjp_K8_fused", "vjp_K8_auto", "guard", "no_guard",
                      "jump", "segmented", "hawk", "box", "no_box", "gemm", "fft", "dense", "map", "prefix", "keyed", "ordered",
                      "or", "compensated", "auto", "cell_compensated", "cell_auto", "two_ranks", "one_rank"})
# an arm named for its baseline library runs the library branch of its row's setup
_ALIAS = {"cupy_fft": "cupy", "cupyx_scatter_add": "cupyx", "cupy_sum": "cupy"}


def manifest():
    """What ``benchmarks/cards_manifest.json`` records for this script."""
    return {name: {"arms": list(r["arms"]), "baseline": r["baseline"], "cases": {c: v[0] for c, v in r["cases"].items()},
                   "dtypes": list(r["dtypes"]), "host_arms": list(r["host"])} for name, r in ROWS.items()}


# --------------------------------------------------------------------------- #
# One arm in its own process
# --------------------------------------------------------------------------- #
def _is_oom(exc):
    text = f"{type(exc).__name__} {exc}".lower()
    return isinstance(exc, MemoryError) or any(s in text for s in (
        "outofmemory", "out of memory", "resource_exhausted", "memory allocation", "cudaerrormemoryallocation"))


def _time(run, smoke, on_gpu, slow=False):
    """A warm-up, then rounds until the IQR/median is under 5 % (3 to 9 rounds), at most 300 s: ((stats, SM MHz), rule)."""
    def stop():       # the warm-up is stopped at the budget, not labelled afterwards
        print(MARK + json.dumps({"status": "not_run: budget", "rule": "budget",
                                 "detail": f"> {RUN_BUDGET_S:g} s per run (warm-up stopped at {RUN_BUDGET_S:g} s)"}), flush=True)
        os._exit(0)

    timer = threading.Timer(RUN_BUDGET_S, stop)
    timer.daemon = True
    timer.start()
    t0 = time.perf_counter()
    try:
        run()
    finally:
        timer.cancel()
    warm = time.perf_counter() - t0
    if slow:      # a test hook: the warm-up reads as 121 s
        warm = RUN_BUDGET_S + 1
    if warm > RUN_BUDGET_S:
        return None, "budget"
    bus = cc.device_facts()[3] if on_gpu else None
    samples, rule, clock, start = [], "max rounds", None, time.perf_counter()
    while len(samples) < (1 if smoke else MAX_ROUNDS):
        t0 = time.perf_counter()
        run()
        samples.append(time.perf_counter() - t0)
        cc.heartbeat()
        clock = cc.nvml_sm_clock_mhz(bus) if bus else None
        if smoke:
            rule = "smoke"
        elif len(samples) >= MIN_ROUNDS and cc.median_iqr(samples)["iqr"] <= IQR_STOP * cc.median_iqr(samples)["median"]:
            rule = "iqr"
            break
        elif time.perf_counter() - start > CELL_CAP_S:
            rule = "cell cap 300 s"
            break
    stats = cc.median_iqr(samples)
    stats.update(rounds=len(samples), warm_s=warm)
    return (stats, clock), rule


def child(row, case, dtype, arm, smoke):
    spec = ROWS[row]
    n = spec["cases"][case][1 if smoke else 0]
    on_gpu = _gpu_count() > 0
    ctx = SimpleNamespace(row=row, case=case, dtype=dtype, arm=_ALIAS.get(arm, arm), n=n, smoke=smoke, note=None,
                          target="cuda" if on_gpu else "host")
    rec = {"status": "ok", "n": n}
    try:
        if os.environ.get("FEATURE_CARD_FORCE") == f"{arm}:oom":       # a test hook: a failed allocation
            raise MemoryError("out of memory (forced by FEATURE_CARD_FORCE)")
        cc.heartbeat("build")
        run, check = spec["setup"](ctx)
        ok, what = check()
        cc.heartbeat()
        if os.environ.get(PROFILE_ENV):
            import cupy as cp

            run()
            cp.cuda.runtime.profilerStart()
            run()
            cp.cuda.runtime.profilerStop()
            print(MARK + json.dumps({"status": "ok", "n": n, "rule": "profiled", "checks": {what: bool(ok)}}), flush=True)
            return 0
        timed, rule = _time(run, smoke, on_gpu, os.environ.get("FEATURE_CARD_FORCE") == f"{arm}:slow")
        rec.update(checks={what: bool(ok)}, rule=rule, note=ctx.note)
        if timed is None:
            rec.update(status="not_run: budget", detail="> 120 s per run")
        else:
            stats, clock = timed
            rec.update(wall_s=stats, sm_mhz=clock, host_rss_kib=cc.status_kib("VmHWM"))
            if on_gpu:
                rec["device_bytes"] = cc.process_device_bytes(cc.device_facts()[3])[0]
            if not ok:
                rec["status"] = f"failed: value check: {what}"
    except NotRun as exc:
        rec = {"status": f"not_run: {exc}"}
    except Exception as exc:  # noqa: BLE001  the card completes whatever one arm does
        first = (str(exc).splitlines() or [""])[0][:160]
        rec = {"status": "out_of_memory" if _is_oom(exc) else f"failed: {type(exc).__name__}: {first}"}
    print(MARK + json.dumps(rec, default=str), flush=True)
    return 0


_PARTS = None   # --parts: every finished arm is saved here and a re-run reuses it
_ASSEMBLE = False   # --assemble: run nothing, write the card from the saved parts


def run_arm(row, case, dtype, arm, smoke):
    """One arm in a fresh process: the record it printed, or the reason it did not. With ``--parts`` the record is saved,
    and an earlier one is reused."""
    key = f"{row}_{case}_{dtype}_{arm}" + ("_smoke" if smoke else "")
    if cc.REFERENCE and arm in EXTERNAL_ARMS:   # ours-only: a baseline library's record is the reference run's
        return cc.ref_part(key) or {"status": cc.NO_REFERENCE}
    saved = cc.part_load(_PARTS, key, keep_failed=_ASSEMBLE)
    if saved is not None:
        return saved
    if _ASSEMBLE or cc.deadline_passed():
        return {"status": cc.SESSION_LIMIT}
    rec = _run_arm_process(row, case, dtype, arm, smoke)
    cc.part_save(_PARTS, key, rec)
    return rec


def _run_arm_process(row, case, dtype, arm, smoke):
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--arm", row, case, dtype, arm] + (["--smoke"] if smoke else [])
    try:
        done = cc.run_watched(cmd, timeout=PROCESS_TIMEOUT_S, progress=pathlib.Path(tempfile.mkdtemp(prefix="feature_alive_")) / "alive")
    except subprocess.TimeoutExpired:
        return {"status": f"failed: no result in {PROCESS_TIMEOUT_S:.0f} s"}
    except cc.Stalled:
        return {"status": "failed: no progress"}
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith(MARK)]
    if lines:
        return json.loads(lines[-1][len(MARK):])
    text = done.stderr + done.stdout
    if any(s in text.lower() for s in ("out of memory", "outofmemory", "resource_exhausted")):
        return {"status": "out_of_memory"}
    last = ([ln.strip() for ln in text.splitlines() if ln.strip()] or ["no output"])[-1][:160]
    return {"status": f"failed: process exit {done.returncode}: {last}"}


def _mpi(smoke):
    """The two-rank arm under mpirun (a fresh process pair), the one-rank arm as any other arm; both are saved as parts."""
    n = 8192 if smoke else 2_000_000
    key = lambda arm: f"mpi_partition_all_{F64}_{arm}" + ("_smoke" if smoke else "")
    rec = {a: cc.part_load(_PARTS, key(a), keep_failed=_ASSEMBLE) for a in ROWS["mpi_partition"]["arms"]}
    if None not in rec.values():
        return rec, n
    why = None
    if _ASSEMBLE or cc.deadline_passed():
        why = cc.SESSION_LIMIT.removeprefix("not_run: ")
    elif _gpu_count() < 2:
        why = f"needs 2 GPUs (this machine has {_gpu_count()}); a Kaggle T4 x2 row"
    elif not shutil.which("mpirun"):
        why = "mpirun not found"
    elif not _installed("eagle._mpi"):
        why = "this eagle build has no rank structure (eagle._mpi; -DEAGLE_PYTHON_MPI)"
    if why:
        return {a: r or {"status": f"not_run: {why}"} for a, r in rec.items()}, n
    if rec["two_ranks"] is None:
        done = subprocess.run(["mpirun", "--allow-run-as-root", "-n", "2", sys.executable, __file__, "--mpi-worker", str(n)]
                              + (["--smoke"] if smoke else []), capture_output=True, text=True, timeout=PROCESS_TIMEOUT_S)
        if done.returncode:
            said = [ln.strip() for ln in (done.stderr + done.stdout).splitlines() if ln.strip().strip("-")]
            rec["two_ranks"] = {"status": f"failed: the two-rank run failed (rc {done.returncode}): {(said or ['no output'])[-1][:160]}"}
        else:
            rec["two_ranks"] = json.loads(done.stdout.strip().splitlines()[-1])
        cc.part_save(_PARTS, key("two_ranks"), rec["two_ranks"])
    if rec["one_rank"] is None:
        rec["one_rank"] = run_arm("mpi_partition", "all", F64, "one_rank", smoke)
    return rec, n


def _mpi_worker(n, smoke):
    """Rank body: the C1 ensemble (float64) of ``n`` samples cut across the ranks by ``RankPartition.local``, each
    rank on its own GPU, the planes folded back by ``allgather_plane``: the record of the timed rounds on rank 0."""
    import ctypes

    import numpy as np
    import cupy as cp
    import eagle.exec as eexec

    rank, size = eexec.RankPartition.world()
    cp.cuda.Device(rank % cp.cuda.runtime.getDeviceCount()).use()
    pc = _perf_card()
    whole = eexec.Partition(0, n, n)
    mine = eexec.RankPartition.local(whole)
    inp = {k: v for k, v in pc._inputs(n).items() if hasattr(v, "__len__")}
    local = {k: v[mine.base: mine.base + mine.count] for k, v in inp.items()}
    run, download = _persist_arm(pc, mine.count, local)
    full = {k: np.zeros(n) for k in pc.STATE_NAMES}

    def fold():
        run()
        out = download()
        for k in full:
            full[k][mine.base: mine.base + mine.count] = out[k]
            eexec.RankPartition.allgather_plane(full[k].ctypes.data_as(ctypes.c_void_p).value, whole)
    fold()
    times = []
    while len(times) < (1 if smoke else MAX_ROUNDS):
        t0 = time.perf_counter()
        fold()
        times.append(time.perf_counter() - t0)
    ok = bool(all(np.isfinite(full[k]).all() for k in full))
    if rank == 0:
        print(json.dumps({"status": "ok" if ok else "failed: value check", "n": n, "wall_s": {**cc.median_iqr(times), "rounds": len(times)},
                          "sm_mhz": None, "checks": {"every folded plane is finite": ok}}))


# --------------------------------------------------------------------------- #
# The card
# --------------------------------------------------------------------------- #
def run_row(name, smoke, gpu):
    """The result records of one row: one per (case, dtype), each with its arms."""
    spec, out = ROWS[name], []
    for case, (full, small) in spec["cases"].items():
        for dtype in spec["dtypes"]:
            if name == "mpi_partition":
                arms, n = _mpi(smoke)
            else:
                n = small if smoke else full
                arms = {a: ({"status": "not_run: needs a GPU"} if not gpu and a not in spec["host"] else run_arm(name, case, dtype, a, smoke))
                        for a in spec["arms"]}
            ref = arms.get(spec["ref"], {})
            ratio = {a: r["wall_s"]["median"] / ref["wall_s"]["median"] for a, r in arms.items() if "wall_s" in r and "wall_s" in ref}
            out.append({"row": name, "case": case, "dtype": dtype, "n": n, "baseline": spec["baseline"],
                        "reference_arm": spec["ref"], "arms": arms, "ratio_to_reference": ratio})
            print(f"{name}[{case},{dtype}]: " + ", ".join(f"{a}={r['status']}" for a, r in arms.items()), flush=True)
    return out


def render_markdown(card):
    lines = [f"<!-- Generated by {SCRIPT_REL}; do not edit. -->", "",
             f"**Device:** {card['device_slug']}. {card['method']}", "",
             "| row | case | dtype | N | arm | status | wall (IQR) | rounds | SM MHz | vs reference |", "|---|---|---|---:|---|---|---:|---:|---:|---:|"]
    for r in card["rows"]:
        for a, rec in r["arms"].items():
            w, ratio = rec.get("wall_s"), r["ratio_to_reference"].get(a)
            wall = f"{cc.fmt_time(w['median'])} ({cc.fmt_time(w['iqr'])})" if w else "–"
            lines.append(f"| {r['row']} | {r['case']} | {r['dtype']} | {rec.get('n', r['n']):,} | {a} | {rec['status']} | {wall} | "
                         f"{w['rounds'] if w else '–'} | {rec.get('sm_mhz') or '–'} | {f'{ratio:.3g}x' if ratio else '–'} |")
    lines += cc.reference_note([rec for r in card["rows"] for rec in r["arms"].values()])
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--rows", nargs="+", choices=list(ROWS), default=list(ROWS))
    ap.add_argument("--arm", nargs=4, metavar=("ROW", "CASE", "DTYPE", "ARM"), help=argparse.SUPPRESS)
    ap.add_argument("--mpi-worker", metavar="N", help=argparse.SUPPRESS)
    ap.add_argument("--assemble", action="store_true",
                    help="run nothing: write the card from the saved parts (--parts), a missing arm reads not_run: step time limit")
    ap.add_argument("--parts", type=pathlib.Path,
                    help="save each finished arm here and reuse what is there: a cap or a preemption costs one arm")
    ap.add_argument("--reference", type=pathlib.Path,
                    help="ours-only: time only our arms (OUR_ARMS); the baseline libraries' records come from this earlier run "
                         "(its results directory, same device) and carry its label")
    ap.add_argument("--smoke", action="store_true", help="smallest sizes, one round, card in a temporary directory")
    args = ap.parse_args(argv)
    global _PARTS, _ASSEMBLE
    _PARTS, _ASSEMBLE = args.parts, args.assemble
    if args.reference and not (args.mpi_worker or args.arm):
        cc.set_reference(args.reference, args.parts, cc.device_facts()[0]["name"])
    if args.mpi_worker:
        _mpi_worker(int(args.mpi_worker), args.smoke)
        return 0
    if args.arm:
        return child(*args.arm, args.smoke)
    if args.smoke and args.out_dir == HERE:
        args.out_dir = pathlib.Path(tempfile.mkdtemp(prefix="feature_card_smoke_"))
    t_start = time.perf_counter()
    os.environ.setdefault("FEATURE_CARD_CACHE", tempfile.mkdtemp(prefix="feature_cache_"))   # one build cache for every arm's process
    gpu = _gpu_count() > 0
    rows = [rec for r in args.rows for rec in run_row(r, args.smoke, gpu)]
    try:
        device, peak, slug, _ = cc.device_facts()
    except Exception:  # no GPU: a host card
        device = peak = None
        slug = "cpu-" + "".join(c if c.isalnum() else "-" for c in (cc.cpu_model() or "host").lower()).strip("-")
    card = {"schema": "eagle-feature-card/2", "script": SCRIPT_REL, "script_md5": cc.md5(__file__), "common_md5": cc.md5(cc.__file__),
            "generated_utc": cc.utc_now_iso(), "release": cc.release_string(), "device_slug": slug, "device": device, "peak": peak,
            "software": cc.software(),
            "method": "One fresh process per arm; a warm-up, then 3 to 9 rounds until the IQR is under 5 % of the median "
                      "(at most 300 s); a warm-up over 120 s is not_run: budget.",
            "rows": rows}
    card["wall_s"] = time.perf_counter() - t_start
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"feature_card_{slug}.json"
    path.write_text(json.dumps(card, indent=1, default=str) + "\n")
    (args.out_dir / f"feature_card_{slug}.md").write_text(render_markdown(card))
    print(f"wrote {path}")
    if not args.smoke and args.out_dir.resolve().is_relative_to(HERE.parent):
        cc.record_provenance(path)
    return cc.no_arm_ran_exit(card)


if __name__ == "__main__":
    sys.exit(main())
