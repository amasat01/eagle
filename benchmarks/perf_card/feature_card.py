# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The eagle/hawk feature card: one row per feature, each at ONE size against its natural baseline.

Where ``perf_card.py`` times one workload (adaptive Kepler orbits) across arms and sizes,
this card times a handful of features, each on the kernel its example notebook shows:

``vjp_vs_torch``
    hawk's generated derivative kernels (primal, vjp, jvp) against PyTorch autograd on the
    same RK4 step (``hawk/examples/autodiff_vs_torch.py``).
``zero_copy_interop``
    a PyTorch tensor aliased into eagle (``eagle.to_cupy``) against the copy through the host
    (``eagle/examples/python/zero_copy_interop.py``). Host-only machines time a launch
    against the same launch preceded by an explicit copy.
``early_termination``
    the guarded graph (``eagle.skippable``) against the same graph unguarded and against
    the host-guarded eager loop (``eagle/examples/python/early_termination.py``).

A row whose library or hardware is missing is reported ``not run: <reason>``; it never fails
the card. Rows that exist as notebooks but have no row yet are listed in the JSON under
``rows_pending`` (never silently absent). Each row is the median of ``REPS`` interleaved runs
after one warm-up. ``--smoke`` runs every row at its smallest size, once, and writes the card to
a temporary directory.

Usage::

    python benchmarks/perf_card/feature_card.py [--rows ROW ...] [--smoke] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
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

#: Features with a notebook but no row yet (name -> baseline the row will use).
ROWS_PENDING = {
    "wide_value_corpus": "hawk 0.7.3 vs the 0.7.2 wheel vs an opaque numpy mask",
    "linear_algebra": "per-lane n = 8/16/32 solve+eigh and team n = 512 vs cuSOLVER / torch.linalg",
    "fft": "per-lane n = 64 and batched 1e5 x 1024 vs cuFFT / cupy",
    "rays_bvh": "closest-hit, device vs host",
    "prefix_index_set": "prefix index set vs dense launch",
    "ordered_scatter": "ordered scatter + keyed sum vs cupyx scatter_add / np.add.at",
    "mpi_partition": "rank partition on 2 GPUs (Kaggle only)",
}


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
    gpu = _has_gpu() and not smoke
    xp = __import__("cupy") if gpu else np
    with tempfile.TemporaryDirectory() as tmp:
        target = "cuda" if gpu else "host"
        bundle = ex.build_artifact(pathlib.Path(tmp) / "et", (target,))
        plan = eplan.plan(ex.device_plugin(bundle, ex.KERNEL_NAME) if gpu
                          else ex.host_plugin(bundle, ex.KERNEL_NAME),
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

        arms, planes = {}, {}
        for name, guarded in (("guarded", True), ("unguarded", False)):
            planes[name], step = make(guarded)

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
    return _row("early_termination", batch, "the same graph with the guard removed", wall,
                "guarded", None if gpu else "host analogue: eagle.skippable vs an unguarded relaunch")


ROWS = {"vjp_vs_torch": row_vjp_vs_torch, "zero_copy_interop": row_zero_copy_interop,
        "early_termination": row_early_termination}


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
            for a, w in r["wall_s"].items():
                lines.append(f"| {r['row']} | {r['n']:,} | {a} | {cc.fmt_time(w['median'])} "
                             f"({cc.fmt_time(w['iqr'])}) | {r['ratio_to_reference'][a]:.3g}x |")
    if card["rows_pending"]:
        lines += ["", "No row yet: " + "; ".join(f"{k} ({v})" for k, v in card["rows_pending"].items()) + "."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--rows", nargs="+", choices=list(ROWS), default=list(ROWS))
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--smoke", action="store_true",
                    help="smallest size, one repetition, card in a temporary directory")
    args = ap.parse_args(argv)
    if args.smoke:
        args.reps = 1
        if args.out_dir == HERE:
            args.out_dir = pathlib.Path(tempfile.mkdtemp(prefix="feature_card_smoke_"))
    elif args.reps < 5:
        ap.error("--reps must be >= 5")
    t_start = time.perf_counter()
    rows = [ROWS[r](args.smoke, args.reps) for r in args.rows]
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
            "repetitions": args.reps, "rows": rows,
            "rows_pending": {k: v for k, v in ROWS_PENDING.items()}}
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
