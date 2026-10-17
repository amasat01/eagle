#!/usr/bin/env python
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Card writer: the guarded-graph skip vs its own unguarded control, and the
guarded graph vs the equivalent host-guarded eager loop.

Backs two claims in the example notebook: "over 13x at a million samples over 4000 steps"
(guarded graph vs the identical captured graph with the guard removed) and
"the guarded graph beats this host-guarded eager loop by 1.3-1.65x at
small-to-medium batch" (guarded graph vs eager).

Real run (GPU, done separately from this script's ``--dry`` path, never on this check's own machine time):
mirrors ``examples/python/early_termination.py``'s own ``_time_arms`` (same three
arms -- guarded graph, unguarded graph, eager loop; same warm-up and
INTERLEAVED-repetition discipline) but, unlike that function, which returns
only the median per arm, keeps every repetition's raw wall time. Writes
``early_termination_card.json``/``.md`` next to this script.

``--dry`` (CPU, tiny batch/steps, no GPU, no cupy required): there is no
CUDA graph on the host, so "guarded graph vs eager" collapses to one arm
there (``examples/python/early_termination.py``'s own ``run_host`` docstring:
"there is no graph capture on the host arm, so eagle.skippable's eager form
IS the only form here"). This exercises the SAME card-writing mechanics
with the one comparison that DOES still make sense on a host: the guard
(``eagle.skippable``) against an unguarded control that always relaunches,
both through ``eagle.exec.HostTeam``. Marked ``"dry": true``, never written
to the committed filename.

Usage::

    python early_termination_card.py --dry
    python early_termination_card.py --device              # needs a visible GPU
"""

from __future__ import annotations

import argparse
import inspect
import pathlib
import sys
import tempfile
import time

_HERE = pathlib.Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (_HERE.parent, _REPO_ROOT / "examples" / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _card_common as cc              # noqa: E402
import early_termination as example    # noqa: E402

CARD_NAME = "early_termination"
DRY_BATCH = 64
DRY_STEPS = 20
DRY_REPS = 3


def _guard_vs_unguarded_host(batch: int, steps: int, reps: int, dt: float,
                             r_esc: float, r_col: float, seed: int = 0) -> dict:
    """Host-only analogue: the SAME guard (``eagle.skippable``) against an
    unguarded control that always relaunches, both through
    ``eagle.exec.HostTeam`` -- what ``run_host`` already builds, kept apart
    into two arms instead of one, with every repetition's raw wall time."""
    import eagle
    import numpy as np
    from eagle import exec as eexec
    from eagle import plan as eplan

    if steps % 2:
        steps += 1
    replays = steps // 2
    r_esc2, r_col2 = float(r_esc) ** 2, float(r_col) ** 2
    state = example.seed_states(batch, seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = example.build_artifact(
            pathlib.Path(tmpdir) / "early_termination_card_dry", ("host",))
        plan = eplan.plan(example.host_plugin(bundle, example.KERNEL_NAME),
                          structure=eexec.HostTeam)

        g_planes = example._fresh_planes(state, batch, np)
        g_bound = example._bind_pair(plan, *g_planes, dt, r_esc2, r_col2)
        g_guard, g_reads = example._counted(g_planes[3].view(np.uint32))

        def g_pair():
            g_bound[0].launch()
            g_bound[1].launch()

        g_step = eagle.skippable(g_pair, g_guard)

        u_planes = example._fresh_planes(state, batch, np)
        u_bound = example._bind_pair(plan, *u_planes, dt, r_esc2, r_col2)

        def u_step():
            u_bound[0].launch()
            u_bound[1].launch()

        arms = {"host_guarded": (g_planes, g_step), "host_unguarded": (u_planes, u_step)}
        for planes, step in arms.values():
            example._reset_planes_(planes, state, batch)
            step()  # warm-up, excluded

        times = {name: [] for name in arms}
        for _ in range(max(1, reps)):
            for name, (planes, step) in arms.items():
                example._reset_planes_(planes, state, batch)
                start = time.perf_counter()
                for _ in range(replays):
                    step()
                times[name].append((time.perf_counter() - start) / steps)

    return {"batch": batch, "steps": steps,
           "wall_s_per_step": {name: cc.median_iqr(v) for name, v in times.items()}}


def _facts() -> dict:
    try:
        device, peak = cc.device_facts()[:2]
    except Exception:  # no GPU or no cupy: the dry path
        device = peak = None
    return {
        "device": device, "peak": peak, "software": cc.software(),
        "script_md5": cc.md5(__file__),
        "example_script_md5": cc.md5(example.__file__),
        "example_script": "examples/python/early_termination.py",
        "generated_utc": cc.utc_now_iso(),
    }


def _code() -> dict:
    return {"kernel_source": inspect.getsource(example.step_kernel)}


def render_md(card: dict) -> str:
    import json

    lines = [f"# {CARD_NAME} card", ""]
    if card.get("dry"):
        lines += ["**DRY RUN -- not authoritative, and not the same "
                  "measurement as the real card** (there is no CUDA graph "
                  "on a host; see this script's own docstring).", ""]
    lines += [f"Schema: `{card['schema']}`  ·  generated {card['facts']['generated_utc']}",
             "", "## Facts", "", "```json", json.dumps(card["facts"], indent=2, sort_keys=True),
             "```", "", "## Results", "", "```json",
             json.dumps(card["results"], indent=2, sort_keys=True), "```", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry", action="store_true")
    p.add_argument("--device", action="store_true")
    p.add_argument("--out", default=None)
    p.add_argument("--reps", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    args = p.parse_args(argv)

    if args.dry == args.device:
        p.error("pass exactly one of --dry / --device")

    facts = _facts()
    if args.dry:
        reps = args.reps or DRY_REPS
        batch = args.batch or DRY_BATCH
        steps = args.steps or DRY_STEPS
        results = _guard_vs_unguarded_host(
            batch, steps, reps, dt=0.01, r_esc=example.R_ESC, r_col=example.R_COL)
        card = {"schema": cc.DEMO_SCHEMA, "name": CARD_NAME, "dry": True,
               "facts": facts, "code": _code(), "results": results}
        out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(
            tempfile.mkdtemp(prefix="demo_card_dry_"))
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{CARD_NAME}_card.DRY.json"
        md_path = out_dir / f"{CARD_NAME}_card.DRY.md"
        cc.write_card(card, json_path, render_md(card), md_path)
        print(f"[dry] wrote {json_path} and {md_path} (not committed)")
        return 0

    # --device: the real run, GPU only. Mirrors _time_arms's
    # own three arms, but keeps every repetition instead of the median alone.
    import cupy as cp
    import eagle
    import eagle.exec as eexec
    import numpy as np
    from eagle import plan as eplan
    from eagle.pipeline import GraphPipeline

    reps = args.reps or 3
    batch = args.batch or 1_000_000
    steps = args.steps or 4000
    if steps % 2:
        steps += 1
    replays = steps // 2
    dt = 0.01
    r_esc2, r_col2 = float(example.R_ESC) ** 2, float(example.R_COL) ** 2
    state = example.seed_states(batch, 0)

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = example.build_artifact(
            pathlib.Path(tmpdir) / "early_termination_card", ("cuda",))
        plan = eplan.plan(example.device_plugin(bundle, example.KERNEL_NAME),
                          structure=eexec.DeviceKernel)

        def make_pair(planes):
            bound = example._bind_pair(plan, *planes, dt, r_esc2, r_col2)

            def pair():
                bound[0].launch()
                bound[1].launch()
            return pair

        g_planes = example._fresh_planes(state, batch, cp)
        g_guard, g_reads = example._counted(g_planes[3].view(cp.uint32))
        g_pipe = GraphPipeline()
        g_pipe.add(eagle.skippable(make_pair(g_planes), g_guard), name="pair")
        g_pipe.build()

        u_planes = example._fresh_planes(state, batch, cp)
        u_pipe = GraphPipeline()
        u_pipe.add(make_pair(u_planes), name="pair")
        u_pipe.build()

        e_planes = example._fresh_planes(state, batch, cp)
        e_guard, e_reads = example._counted(e_planes[3].view(cp.uint32))
        e_step = eagle.skippable(make_pair(e_planes), e_guard)

        def e_run(n):
            for _ in range(n):
                e_step()
            cp.cuda.runtime.deviceSynchronize()

        arms = {"guarded": (g_planes, g_pipe.launch), "unguarded": (u_planes, u_pipe.launch),
               "eager": (e_planes, e_run)}
        for planes, run in arms.values():
            example._reset_planes_(planes, state, batch)
            run(1)  # warm-up, excluded

        times = {name: [] for name in arms}
        for _ in range(max(1, reps)):
            for name, (planes, run) in arms.items():
                example._reset_planes_(planes, state, batch)
                start = time.perf_counter()
                run(replays)
                times[name].append((time.perf_counter() - start) / steps)
        last_termination = int(cp.asnumpy(g_planes[2][0]).max())

    wall = {name: cc.median_iqr(v) for name, v in times.items()}
    ratio_skip = (wall["unguarded"]["median"] / wall["guarded"]["median"]
                 if wall["guarded"]["median"] else None)
    ratio_eager = (wall["eager"]["median"] / wall["guarded"]["median"]
                  if wall["guarded"]["median"] else None)
    results = {"batch": batch, "steps": steps, "last_termination": last_termination,
              "wall_s_per_step": wall,
              "ratio_unguarded_over_guarded": ratio_skip,
              "ratio_eager_over_guarded": ratio_eager}
    card = {"schema": cc.DEMO_SCHEMA, "name": CARD_NAME, "dry": False,
           "facts": facts, "code": _code(), "results": results}
    out_dir = pathlib.Path(args.out) if args.out else _HERE
    json_path, md_path = out_dir / f"{CARD_NAME}_card.json", out_dir / f"{CARD_NAME}_card.md"
    cc.write_card(card, json_path, render_md(card), md_path)
    print(f"wrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
