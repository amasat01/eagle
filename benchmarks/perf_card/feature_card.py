# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The eagle/hawk feature card: one row per feature, each at ONE size against its natural baseline.

Where ``perf_card.py`` times one workload (adaptive Kepler orbits) across arms and sizes,
this card times a handful of features, each on the kernel its example notebook shows:

``wide_value_corpus`` (the release headline)
    hawk's wide-value kernels (``hawk/tests/_wide_kernels.py``: a 384-term masked sum, a 512-wide
    masked fill, a 24 x 512 matrix-vector product) against the user's opaque NumPy mask and, when
    ``HAWK_TWIN_VENV`` names the 0.7.2 environment (hawk ``make compare-0.7.2``), the 0.7.2 build.
``linalg_lane`` / ``linalg_team``
    per-lane solve + eigh at n = 8, 16, 32 and a team solve at n = 512, against NumPy and CuPy (cuSOLVER).
``fft``
    batched c2c over 1e5 x 1024 (``eagle.fft`` graph step, replayed) against cupy.fft, numpy.fft as the oracle; the
    per-lane n = 64 butterfly is not run (hawk has no per-lane FFT in 0.7.3).
``rays_bvh``
    closest hit of a ray per lane against a BVH, device against host (a capability row).
``prefix_index_set``
    an ``IndexSet(prefix=True)`` launch of 10 % of the lanes against the dense launch.
``ordered_scatter``
    ``hawk.scatter`` keyed sum against ``np.add.at`` and ``cupyx.scatter_add``.
``vjp_vs_torch``, ``zero_copy_interop``, ``early_termination``
    the three example notebooks' kernels (see their examples); early termination also times the
    host-guarded eager loop.
``mpi_partition``
    the plan through ``RankPartition`` on 2 GPUs against one rank (Kaggle T4 x2 only).

A row whose library or hardware is missing is reported ``not run: <reason>``; it never fails the
card. A row that checks values against NumPy or another arm records it under ``checks``. Each row is
the median of ``REPS`` interleaved runs after one warm-up. ``--smoke`` runs every row at its smallest
size, once, and writes the card to a temporary directory.

Usage::

    python benchmarks/perf_card/feature_card.py [--rows ROW ...] [--smoke] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))  # benchmarks/: the cards' shared helpers
import _card_common as cc  # noqa: E402

SCRIPT_REL = "benchmarks/perf_card/feature_card.py"
REPS = 5
DT = 0.01

def _installed(module):
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _load(name, path):
    """Import a repo example by file path, or None when it is not in this checkout."""
    path = pathlib.Path(path)
    if not path.exists():
        return None
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hawk_examples():
    root = os.environ.get("HAWK_SRC")
    return pathlib.Path(root) / "examples" if root else REPO.parent / "hawk" / "examples"


def _has_gpu():
    if not _installed("cupy"):
        return False
    import cupy as cp

    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except cp.cuda.runtime.CUDARuntimeError:
        return False


def _interleaved(arms, reps):
    """Warm every arm once, then ``reps`` rounds in rotating order; the wall of each call."""
    for fn in arms.values():
        fn()
    names, times = list(arms), {a: [] for a in arms}
    for r in range(reps):
        for a in names[r % len(names):] + names[:r % len(names)]:
            t0 = time.perf_counter()
            arms[a]()
            times[a].append(time.perf_counter() - t0)
    return {a: cc.median_iqr(v) for a, v in times.items()}


def _row(name, n, baseline, wall, reference, note=None):
    """One result row; ``reference`` is the arm the others are compared with."""
    ref = wall[reference]["median"]
    return {"row": name, "status": "ok", "n": n, "baseline": baseline, "wall_s": wall,
            "reference_arm": reference,
            "ratio_to_reference": {a: w["median"] / ref for a, w in wall.items()}, "note": note}


def _not_run(name, why):
    return {"row": name, "status": f"not run: {why}"}


# --------------------------------------------------------------------------- #
# Rows
# --------------------------------------------------------------------------- #
def row_vjp_vs_torch(smoke, reps):
    ex = _load("autodiff_vs_torch", _hawk_examples() / "autodiff_vs_torch.py")
    if ex is None:
        return _not_run("vjp_vs_torch", "hawk examples not found (set HAWK_SRC)")
    batch = 8 if smoke else 65536
    with tempfile.TemporaryDirectory() as tmp:
        if _has_gpu() and _installed("torch"):
            bundle = ex.build_artifact(pathlib.Path(tmp) / "vjp", ("cuda",))
            raw = ex._time_derivatives(bundle, batch, DT, reps)
            return {"row": "vjp_vs_torch", "status": "ok", "n": batch,
                    "baseline": "PyTorch autograd, same RK4 step", "raw": raw}
        # host: hawk's three kernels; the torch arms need a torch install
        import numpy as np
        import eagle.exec as eexec
        from eagle import plan as eplan

        bundle = ex.build_artifact(pathlib.Path(tmp) / "vjp", ("host",))
        names = ex._plane_names()
        x = np.ascontiguousarray(ex.seed_states(batch, 0), dtype=np.float32)
        rng = np.random.default_rng(1)
        bar = rng.normal(size=(4, batch)).astype(np.float32)
        tan = rng.normal(size=(4, batch)).astype(np.float32)
        out, bar_x, dot = (np.zeros_like(x) for _ in range(3))

        def bind(kernel, **planes):
            plan = eplan.plan(ex.host_plugin(bundle, kernel), structure=eexec.HostTeam)
            return plan.bind(x=x, dt=DT, **planes).launch

        arms = {"hawk_primal": bind(ex.KERNEL_NAME, x_next=out),
                "hawk_vjp": bind(ex.VJP_NAME, **{names.bar_out: bar, names.bar_wrt: bar_x}),
                "hawk_jvp": bind(ex.JVP_NAME, **{names.dot_wrt: tan, names.dot_out: dot})}
        wall = _interleaved(arms, reps)
    row = _row("vjp_vs_torch", batch, "hawk primal (host); no GPU/PyTorch here", wall,
               "hawk_primal")
    row["note"] = "torch arms need a GPU and PyTorch: not run on this machine"
    return row


def _interop_module():
    return _load("zero_copy_interop", REPO / "examples" / "python" / "zero_copy_interop.py")


def row_zero_copy_interop(smoke, reps):
    ex = _interop_module()
    if ex is None:
        return _not_run("zero_copy_interop", "eagle examples not found")
    import numpy as np
    import eagle.exec as eexec
    from eagle import plan as eplan

    size_mb = 1.0 if smoke else 256.0
    batch = max(1, int(size_mb * 1024 * 1024 / 12))
    x32 = np.ascontiguousarray(ex.seed_positions(batch, 0), dtype=np.float32)
    on_gpu = _has_gpu() and _installed("torch")
    with tempfile.TemporaryDirectory() as tmp:
        if on_gpu:
            import cupy as cp
            import eagle
            import torch

            bundle = ex.build_artifact(pathlib.Path(tmp) / "zc", ("cuda",))
            plan = eplan.plan(ex.device_plugin(bundle, ex.KERNEL_NAME), structure=eexec.DeviceKernel)
            t_in = torch.from_numpy(x32.copy()).cuda().contiguous()
            t_out = torch.empty_like(t_in)

            def aliased():
                plan.bind(x=eagle.to_cupy(t_in), k=2.0, y=eagle.to_cupy(t_out)).launch()
                cp.cuda.runtime.deviceSynchronize()

            def copied():
                cu_in = cp.asarray(t_in.cpu().numpy())
                cu_out = cp.empty_like(cu_in)
                plan.bind(x=cu_in, k=2.0, y=cu_out).launch()
                cp.cuda.runtime.deviceSynchronize()
                torch.from_numpy(cp.asnumpy(cu_out)).cuda()

            wall = _interleaved({"aliased": aliased, "naive_copy": copied}, reps)
            return _row("zero_copy_interop", size_mb, "copy through the host", wall, "aliased")
        bundle = ex.build_artifact(pathlib.Path(tmp) / "zc", ("host",))
        plan = eplan.plan(ex.host_plugin(bundle, ex.KERNEL_NAME), structure=eexec.HostTeam)

        def direct():
            plan.bind(x=x32, k=2.0, y=np.zeros_like(x32)).launch()

        def copied_host():
            xc = x32.copy()
            y = np.zeros_like(xc)
            plan.bind(x=xc, k=2.0, y=y).launch()
            y.copy()

        wall = _interleaved({"direct": direct, "copied": copied_host}, reps)
    return _row("zero_copy_interop", size_mb, "the same launch preceded by np.copy (host analogue)",
                wall, "direct", "host analogue: no GPU/PyTorch on this machine")


def row_early_termination(smoke, reps):
    ex = _load("early_termination", REPO / "examples" / "python" / "early_termination.py")
    if ex is None:
        return _not_run("early_termination", "eagle examples not found")
    import numpy as np
    import eagle
    import eagle.exec as eexec
    from eagle import plan as eplan

    batch = 64 if smoke else 1_000_000
    steps = 20 if smoke else 4000
    replays = steps // 2
    r_esc2, r_col2 = float(ex.R_ESC) ** 2, float(ex.R_COL) ** 2
    state = ex.seed_states(batch, 0)
    gpu = _has_gpu()
    xp = __import__("cupy") if gpu else np
    with tempfile.TemporaryDirectory() as tmp:
        bundle = ex.build_artifact(pathlib.Path(tmp) / "et", ("cuda" if gpu else "host",))
        plan = eplan.plan(ex.device_plugin(bundle, ex.KERNEL_NAME) if gpu else ex.host_plugin(bundle, ex.KERNEL_NAME),
                          structure=eexec.DeviceKernel if gpu else eexec.HostTeam)

        def make(guarded):
            planes = ex._fresh_planes(state, batch, xp)
            bound = ex._bind_pair(plan, *planes, DT, r_esc2, r_col2)

            def pair():
                bound[0].launch()
                bound[1].launch()
            if guarded:
                guard, _ = ex._counted(planes[3].view(xp.uint32))
                pair = eagle.skippable(pair, guard)
            return planes, pair

        planes, arms = {}, {}
        # the guarded and the unguarded pair, each replayed `replays` times as one captured graph on a GPU
        # (a plain loop on the host, which has no graph); the eager arm is the host-guarded loop
        for name, guarded in (("guarded", True), ("unguarded", False), ("eager", True)):
            planes[name], step = make(guarded)
            if gpu and name != "eager":
                from eagle.pipeline import GraphPipeline

                pipe = GraphPipeline()
                pipe.add(step, name="pair")
                pipe.build()
                arms[name] = (lambda pipe=pipe: pipe.launch(replays))
            else:
                def run(step=step):
                    for _ in range(replays):
                        step()
                    if gpu:
                        xp.cuda.runtime.deviceSynchronize()
                arms[name] = run

        def reset_then(name):
            def go():
                ex._reset_planes_(planes[name], state, batch)
                arms[name]()
            return go

        wall = _interleaved({n: reset_then(n) for n in arms}, reps)
        same = all(np.array_equal(_host(a), _host(b)) for a, b in zip(planes["guarded"][:3], planes["unguarded"][:3]))
    row = _row("early_termination", batch, "the same graph with the guard removed; the host-guarded eager loop", wall,
               "guarded", None if gpu else "host analogue: no graph, so the guarded arm is also an eager loop")
    return _checked(row, same, "guarded and unguarded runs end in the same state")


# --------------------------------------------------------------------------- #
# Launch helpers for the hawk-kernel rows
# --------------------------------------------------------------------------- #
def _sync(target):
    if target == "cuda":
        from eagle._alloc import synchronize

        synchronize()


def _bind(kernel, planes, outs, target, **extra):
    """(launch, planes): ``kernel`` built for ``target`` and bound to ``planes`` and zeroed ``outs``."""
    import numpy as np
    import eagle.plan
    from eagle import arrays

    plan = eagle.plan.auto(kernel, targets=(target,), scalar_type="float64",
                           cache_dir=tempfile.mkdtemp(prefix="feature_cache_"))
    dev = "cuda:0" if target == "cuda" else "cpu"
    bound = {k: arrays.asarray(np.ascontiguousarray(v), device=dev)
             for k, v in {**planes, **{k: np.zeros(s, np.int32 if k == "info" else np.float64) for k, s in outs.items()}}.items()}
    run = plan.bind(**bound, **extra)

    def launch():
        run.launch()
        _sync(target)
    return launch, bound


def _host(x):
    """A NumPy copy of a plane, a CuPy array, or a sequence of them."""
    import numpy as np

    if hasattr(x, "get"):
        return np.asarray(x.get())
    if isinstance(x, (list, tuple)):
        return np.stack([_host(i) for i in x])
    return np.asarray(x)


def _get(plane):
    import numpy as np

    return np.asarray(plane.get())


def _kernels():
    return _load("feature_kernels", HERE / "feature_kernels.py")


def _checked(row, ok, what):
    row["checks"] = {what: bool(ok)}
    if not ok:
        row["status"] = f"FAILED value check: {what}"
    return row


def _cupy_sync():
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()


# --------------------------------------------------------------------------- #
# The wide-value corpus (the release headline)
# --------------------------------------------------------------------------- #
_TWIN_DRIVER = """
import json, sys, tempfile
sys.path.insert(0, sys.argv[1])
import _wide_kernels as wk
names, lanes, reps = json.loads(sys.argv[2])
print(json.dumps({n: wk.run_device(n, "float64", L, cache_dir=tempfile.mkdtemp(), repeats=reps)["ns"] * L / 1e9
                  for n, L in zip(names, lanes)}))
"""
WIDE_ROWS = ("masked_sum", "fill_mask", "matvec_in")


def row_wide_value_corpus(smoke, reps):
    tests = pathlib.Path(os.environ.get("HAWK_SRC", REPO.parent / "hawk")) / "tests"
    if not (tests / "_wide_kernels.py").exists():
        return _not_run("wide_value_corpus", "hawk tests/_wide_kernels.py not found (set HAWK_SRC)")
    if not _has_gpu():
        return _not_run("wide_value_corpus", "needs a GPU")
    import numpy as np

    sys.path.insert(0, str(tests))
    wk = importlib.import_module("_wide_kernels")
    lanes = [1000 if smoke else wk.LANES[n] for n in WIDE_ROWS]
    cache = tempfile.mkdtemp(prefix="wide_cache_")
    wall, ok = {}, True
    twin = os.environ.get("HAWK_TWIN_VENV")
    twin_s = None
    if twin:
        env = {k: v for k, v in os.environ.items()
               if k not in ("HAWK_AETHER_INCLUDE", "HAWK_EAGLE_INCLUDE", "PYTHONPATH")}
        env.update(HAWK_DEVICE_COMPILER="nvcc", HAWK_GATE_ARCH=wk.device_arch())
        done = subprocess.run([str(pathlib.Path(twin) / "bin" / "python"), "-c", _TWIN_DRIVER, str(tests),
                               json.dumps([list(WIDE_ROWS), lanes, reps])], env=env, capture_output=True, text=True)
        twin_s = json.loads(done.stdout.strip().splitlines()[-1]) if done.returncode == 0 else None
    for name, n in zip(WIDE_ROWS, lanes):
        got = wk.run_device(name, "float64", n, cache_dir=cache, repeats=reps)
        ok &= bool(np.array_equal(got["out"], got["want"]))
        entry = wk.CORPUS[name]
        planes = entry.planes(np.random.default_rng(7), np.float64, n, True)
        mask = _interleaved({"numpy_mask": lambda: entry.want(np.float64, planes, n)}, reps)["numpy_mask"]
        wall[f"{name}:hawk"] = cc.median_iqr([got["ns"] * n / 1e9])
        wall[f"{name}:numpy_mask"] = mask
        if twin_s:
            wall[f"{name}:hawk_0.7.2"] = cc.median_iqr([twin_s[name]])
    row = _row("wide_value_corpus", lanes[0], "an opaque numpy mask; the 0.7.2 build (HAWK_TWIN_VENV)",
               wall, "%s:hawk" % WIDE_ROWS[0],
               None if twin_s else "0.7.2 arm not run: set HAWK_TWIN_VENV (hawk `make compare-0.7.2` makes it)")
    row["lanes"] = dict(zip(WIDE_ROWS, lanes))
    row["ratio_to_reference"] = {a: w["median"] / wall[f"{a.split(':')[0]}:hawk"]["median"] for a, w in wall.items()}
    return _checked(row, ok, "device output equals the NumPy reference bit for bit")


# --------------------------------------------------------------------------- #
# Per-lane and team linear algebra
# --------------------------------------------------------------------------- #
def _spd(rng, n, count):
    import numpy as np

    a = rng.random((n, n, count))
    return (a + a.transpose(1, 0, 2)) / 2 + n * np.eye(n)[:, :, None]


def row_linalg_lane(smoke, reps):
    import numpy as np

    out = []
    for n, full in ((8, 131072), (16, 32768), (32, 8192)):
        count = 250 if smoke else full
        rng = np.random.default_rng(n)
        A, b = _spd(rng, n, count), rng.random((n, count))
        launch, planes = _bind(_kernels().solve_eigh(n), {"A": A.reshape(n * n, count), "b": b},
                               {"x": (n, count), "w": (n, count), "info": (count,)}, "cuda" if _has_gpu() else "host")
        stack = np.moveaxis(A, 2, 0)
        want_x = np.linalg.solve(stack, b.T[:, :, None])[:, :, 0].T
        want_w = np.linalg.eigvalsh(stack).T
        arms = {"hawk": launch,
                "numpy": lambda: (np.linalg.solve(stack, b.T[:, :, None]), np.linalg.eigvalsh(stack))}
        if _installed("cupy") and _has_gpu():
            import cupy as cp

            sc, bc = cp.asarray(stack), cp.asarray(b.T[:, :, None])

            def cupy_arm():
                cp.linalg.solve(sc, bc)
                cp.linalg.eigvalsh(sc)
                _cupy_sync()
            arms["cupy"] = cupy_arm
        wall = _interleaved(arms, reps)
        row = _row(f"linalg_lane_n{n}", count, "NumPy batched solve+eigvalsh; CuPy (cuSOLVER) when present", wall, "hawk")
        got_w = np.sort(_get(planes["w"]), axis=0)
        row = _checked(row, np.allclose(_get(planes["x"]), want_x, rtol=1e-8, atol=1e-10)
                       and np.allclose(got_w, want_w, rtol=1e-8, atol=1e-10), "x and eigenvalues match NumPy")
        row["note"] = "torch.linalg: " + ("not run here" if not _installed("torch") else "see torch arm")
        out.append(row)
    return out


def row_linalg_team(smoke, reps):
    import numpy as np

    n, count = 512, 2 if smoke else 16
    rng = np.random.default_rng(3)
    A, b = _spd(rng, n, count), rng.random((n, count))
    target = "cuda" if _has_gpu() else "host"
    launch, planes = _bind(_kernels().solve_only(n), {"A": A.reshape(n * n, count), "b": b},
                           {"x": (n, count), "info": (count,)}, target)
    stack = np.moveaxis(A, 2, 0)
    rhs = b.T[:, :, None]
    arms = {"hawk_team": launch, "numpy": lambda: np.linalg.solve(stack, rhs)}
    if _installed("cupy") and _has_gpu():
        import cupy as cp

        sc, bc = cp.asarray(stack), cp.asarray(rhs)

        def cupy_arm():
            cp.linalg.solve(sc, bc)
            _cupy_sync()
        arms["cupy"] = cupy_arm
    wall = _interleaved(arms, reps)
    row = _row("linalg_team_n512", count, "NumPy / CuPy (cuSOLVER) batched solve", wall, "hawk_team")
    return _checked(row, np.allclose(_get(planes["x"]), np.linalg.solve(stack, rhs)[:, :, 0].T, rtol=1e-8, atol=1e-10),
                    "x matches NumPy")


def row_fft(smoke, reps):
    """Batched c2c over the last axis (``eagle.fft`` graph step, replayed) against cupy.fft; numpy.fft is the oracle."""
    import numpy as np
    import eagle
    import eagle.fft as F
    from eagle import arrays

    count, n = (64, 1024) if smoke else (100_000, 1024)
    rng = np.random.default_rng(5)
    pairs = np.ascontiguousarray(rng.uniform(-1, 1, (count, n, 2)))
    z = pairs[..., 0] + 1j * pairs[..., 1]
    ok, graphs, outs = True, {}, {}
    devices = ["cpu"] + (["cuda:0"] if _has_gpu() else [])
    for dev in devices:
        out = arrays.zeros((count, n, 2), "float64", dev)
        graphs[dev] = eagle.Graph().add_node(F.c2c(arrays.asarray(pairs, "float64", dev), out, (1,), forward=True)).build()
        graphs[dev].launch()
        _sync("cuda" if dev != "cpu" else "host")
        got = _host(out)[:8]
        ok &= bool(np.abs(got[..., 0] + 1j * got[..., 1] - np.fft.fft(z[:8], axis=1)).max() < 1e-9 * n)
        outs[dev] = out
    best = devices[-1]

    def eagle_arm():
        graphs[best].launch()
        _sync("cuda" if best != "cpu" else "host")
    arms = {f"eagle_fft_{best}": eagle_arm, "numpy.fft": lambda: np.fft.fft(z, axis=1)}
    note = "per-lane n = 64 butterfly: not run, hawk has no per-lane FFT in 0.7.3"
    if _installed("cupy") and _has_gpu():
        import cupy as cp

        zc = cp.asarray(z)

        def cupy_arm():
            cp.fft.fft(zc, axis=1)
            _cupy_sync()
        arms["cupy.fft"] = cupy_arm
    else:
        note += "; cupy.fft: not installed"
    wall = _interleaved(arms, reps)
    row = _row("fft", count * n, "numpy.fft; cupy.fft (cuFFT) when present", wall, f"eagle_fft_{best}", note)
    return _checked(row, ok, "eagle.fft equals numpy.fft on every device it ran on")


# --------------------------------------------------------------------------- #
# Rays, index sets, keyed sums
# --------------------------------------------------------------------------- #
def row_rays_bvh(smoke, reps):
    import numpy as np
    from hawk.geometry import build_bvh

    ntri, nray = (400, 256) if smoke else (20000, 65536)
    rng = np.random.default_rng(0)
    centers = rng.random((ntri, 1, 3)) * 10
    bvh = build_bvh(centers + 0.15 * rng.random((ntri, 3, 3)))
    o = rng.random((3, nray)) * 10
    d = rng.normal(size=(3, nray))
    d /= np.linalg.norm(d, axis=0)
    planes = {"o": o, "d": d, "tmax": np.full(nray, 1e30), **bvh.planes()}
    arms, got = {}, {}
    for target in (("host", "cuda") if _has_gpu() else ("host",)):
        arms[f"hawk_{target}"], pl = _bind(_kernels().hits(bvh), planes, {"y": (5, nray)}, target)
        got[target] = pl
    wall = _interleaved(arms, reps)
    row = _row("rays_bvh", nray, "device vs host (a capability row: no library baseline)", wall,
               "hawk_cuda" if "hawk_cuda" in arms else "hawk_host")
    if len(got) == 2:
        host, dev = _get(got["host"]["y"]), _get(got["cuda"]["y"])
        row = _checked(row, np.allclose(host, dev, rtol=1e-9, atol=1e-9), "device hits equal host hits")
    return row


def row_prefix_index_set(smoke, reps):
    import numpy as np

    n = 4096 if smoke else 1 << 22
    live = max(1, n // 10)
    k = _kernels()
    x = np.linspace(0.0, 3.0, n)
    target = "cuda" if _has_gpu() else "host"
    pre, pp = _bind(k.prefix, {"x": x, "active_count": np.array([live], dtype=np.uint32)}, {"y": (n,)}, target)
    den, dp = _bind(k.dense, {"x": x}, {"y": (n,)}, target)
    wall = _interleaved({"prefix_10pct": pre, "dense": den}, reps)
    row = _row("prefix_index_set", n, "a dense launch over every lane", wall, "prefix_10pct")
    p, d = _get(pp["y"]), _get(dp["y"])
    return _checked(row, np.array_equal(p[:live], d[:live]) and not p[live:].any(),
                    "the live prefix equals the dense launch; the rest is untouched")


def row_ordered_scatter(smoke, reps):
    import numpy as np

    n, nodes = (4096, 64) if smoke else (1 << 22, 1024)
    rng = np.random.default_rng(1)
    key, v = rng.integers(0, nodes, n).astype(np.int32), rng.random(n)
    target = "cuda" if _has_gpu() else "host"
    launch, planes = _bind(_kernels().keyed_sum, {"key": key, "v": v}, {"rs": (nodes,)}, target)

    def numpy_add_at():
        ref = np.zeros(nodes)
        np.add.at(ref, key, v)
        return ref
    launch()  # a keyed sum accumulates: check the first launch, before the timed ones add to it
    matches = np.allclose(_get(planes["rs"]), numpy_add_at(), rtol=1e-9)
    arms = {"hawk_scatter": launch, "np.add.at": numpy_add_at}
    if _installed("cupy") and _has_gpu():
        import cupy as cp
        import cupyx

        kc, vc, rc = cp.asarray(key), cp.asarray(v), cp.zeros(nodes)

        def cupy_arm():
            cupyx.scatter_add(rc, kc, vc)
            _cupy_sync()
        arms["cupyx.scatter_add"] = cupy_arm
    wall = _interleaved(arms, reps)
    row = _row("ordered_scatter", n, "np.add.at; cupyx.scatter_add when present", wall, "hawk_scatter")
    return _checked(row, matches, "keyed sums match np.add.at (ULP contract)")


def _gpu_count():
    if not _installed("cupy"):
        return 0
    import cupy as cp

    try:
        return cp.cuda.runtime.getDeviceCount()
    except cp.cuda.runtime.CUDARuntimeError:
        return 0


def row_mpi_partition(smoke, reps):
    if _gpu_count() < 2:
        return _not_run("mpi_partition", f"needs 2 GPUs (this machine has {_gpu_count()}); a Kaggle T4 x2 row")
    if not shutil.which("mpirun"):
        return _not_run("mpi_partition", "mpirun not found")
    try:
        import eagle._mpi  # noqa: F401
    except ImportError:
        return _not_run("mpi_partition", "this eagle build has no rank structure (eagle._mpi; -DEAGLE_PYTHON_MPI)")
    done = subprocess.run(["mpirun", "--allow-run-as-root", "-n", "2", sys.executable, __file__, "--mpi-worker",
                           "1000000" if not smoke else "4096", str(reps)], capture_output=True, text=True, timeout=600)
    if done.returncode:
        return _not_run("mpi_partition", f"the two-rank run failed (rc {done.returncode})")
    walls = json.loads(done.stdout.strip().splitlines()[-1])
    wall = {a: cc.median_iqr(v) for a, v in walls.items()}
    return _row("mpi_partition", 4096 if smoke else 1_000_000, "the same plan on one rank", wall, "one_rank")


def _mpi_worker(n, reps):
    """Rank body: the zero-copy example's kernel through ``RankPartition`` (device inner) against one rank's whole run."""
    import numpy as np
    import eagle.exec as eexec
    from eagle import plan as eplan

    ex = _interop_module()
    x = np.ascontiguousarray(ex.seed_positions(n, 0), dtype=np.float32)
    with tempfile.TemporaryDirectory() as tmp:
        plugin = ex.device_plugin(ex.build_artifact(pathlib.Path(tmp) / "mpi", ("cuda",)), ex.KERNEL_NAME)
        whole = eplan.plan(plugin, structure=eexec.DeviceKernel)
        ranked = eplan.plan(plugin, structure=eexec.RankPartition, inner=eexec.DeviceKernel)
        arms = {"one_rank": lambda: whole.run(x=x, k=2.0, y=np.zeros_like(x)),
                "two_ranks": lambda: ranked.run(x=x, k=2.0, y=np.zeros_like(x))}
        times = _interleaved(arms, reps)
    if eexec.RankPartition.world()[0] == 0:
        print(json.dumps({a: w["samples"] for a, w in times.items()}))


ROWS = {"wide_value_corpus": row_wide_value_corpus, "linalg_lane": row_linalg_lane, "linalg_team": row_linalg_team,
        "fft": row_fft, "rays_bvh": row_rays_bvh, "prefix_index_set": row_prefix_index_set,
        "ordered_scatter": row_ordered_scatter, "vjp_vs_torch": row_vjp_vs_torch,
        "zero_copy_interop": row_zero_copy_interop, "early_termination": row_early_termination,
        "mpi_partition": row_mpi_partition}


def render_markdown(card):
    lines = [f"<!-- Generated by {SCRIPT_REL}; do not edit. -->", "",
             f"**Device:** {card['device_slug']}. {card['method']}", "",
             "| row | N | arm | wall (IQR) | vs reference |", "|---|---:|---|---:|---:|"]
    for r in card["rows"]:
        if r["status"] != "ok":
            lines.append(f"| {r['row']} | – | {r['status']} | – | – |")
        elif "wall_s" not in r:
            lines.append(f"| {r['row']} | {r['n']:,} | see JSON (`raw`) | – | – |")
        else:
            if r.get("checks"):
                lines.append(f"| {r['row']} | – | check: {next(iter(r['checks']))} | {next(iter(r['checks'].values()))} | – |")
            for a, w in r["wall_s"].items():
                lines.append(f"| {r['row']} | {r['n']:,} | {a} | {cc.fmt_time(w['median'])} "
                             f"({cc.fmt_time(w['iqr'])}) | {r['ratio_to_reference'][a]:.3g}x |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--rows", nargs="+", choices=list(ROWS), default=list(ROWS))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--mpi-worker", nargs=2, metavar=("N", "REPS"), help=argparse.SUPPRESS)
    ap.add_argument("--smoke", action="store_true",
                    help="smallest size, one repetition, card in a temporary directory")
    args = ap.parse_args(argv)
    if args.mpi_worker:
        _mpi_worker(int(args.mpi_worker[0]), int(args.mpi_worker[1]))
        return 0
    if args.smoke:
        args.reps = 1
        if args.out_dir == HERE:
            args.out_dir = pathlib.Path(tempfile.mkdtemp(prefix="feature_card_smoke_"))
    elif args.reps < 5:
        ap.error("--reps must be >= 5")
    t_start = time.perf_counter()
    rows = []
    for r in args.rows:
        got = ROWS[r](args.smoke, args.reps)
        rows += got if isinstance(got, list) else [got]
    for r in rows:
        print(f"{r['row']}: {r['status']}", flush=True)
    try:
        device, peak, slug, _ = cc.device_facts()
    except Exception:  # no GPU: a host card
        device = peak = None
        slug = "cpu-" + "".join(c if c.isalnum() else "-" for c in (cc.cpu_model() or "host").lower()).strip("-")
    card = {"schema": "eagle-feature-card/1", "script": SCRIPT_REL,
            "script_md5": cc.md5(__file__), "common_md5": cc.md5(cc.__file__),
            "generated_utc": cc.utc_now_iso(), "release": cc.release_string(),
            "device_slug": slug, "device": device, "peak": peak, "software": cc.software(),
            "method": f"Median of {args.reps} interleaved repetitions after one warm-up, "
                      "IQR in parentheses.",
            "repetitions": args.reps, "rows": rows}
    card["wall_s"] = time.perf_counter() - t_start
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"feature_card_{slug}.json"
    path.write_text(json.dumps(card, indent=1, default=str) + "\n")
    (args.out_dir / f"feature_card_{slug}.md").write_text(render_markdown(card))
    print(f"wrote {path}")
    if not args.smoke and args.out_dir.resolve().is_relative_to(HERE.parent):
        cc.record_provenance(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
