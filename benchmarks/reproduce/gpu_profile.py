#!/usr/bin/env python
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Profile eagle's launch decisions and kernels on this machine's GPU.

Stages (``--stage``; ``profile.sh`` runs them in order), each written as plain text so
nobody needs a GUI:

* **probe**: what eagle reads from the device and the launch mode it picks for the
  performance card's kernel (``probe.txt``, seconds);
* **nsys-grid, nsys-width**: Nsight Systems timelines of eagle's automatic, persistent and
  band launches and of Warp's kernel over N, S and the stop-step spread, summarised (kernel
  time, launch counts, API time);
* **ncu points**: Nsight Compute reports of one launch of each kernel (occupancy, registers,
  bandwidth, warp stalls) at three points, exported to text; only on compute capability >= 7.0;
* **ncu-ratio**: the persist kernels' threads per issued instruction (about 32 converged, under 16
  fragmented), FP64 pipe use and achieved warps, for the oscillator (uniform and spread, two step
  budgets) and the RK78 card kernel at N = 1e6;
* **persist-sweep, fused-sweep**: wall time (CUDA events) and per-lane utilisation of the persist
  launch over persist threads per SM, and of the fused one-launch entry, at the same points;
* **cadence-sweep**: the band path (step plus compaction) over the compaction cadence, as nsys sums;
* **ncu-band**: the band path's step and compaction kernels under Nsight Compute, run as an eager
  loop because the graph's conditional loop node cannot be profiled.

``--probe-only`` runs the first stage alone (with ``--target cpu``: on the host, no GPU).
Run through ``profile.sh``, which sets up the environment and packs the tarball.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import platform
import re
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

HERE = pathlib.Path(__file__).resolve().parent
CARD_PATH = HERE.parent / "perf_card" / "perf_card.py"
RK78_CARD_PATH = HERE.parent / "rk78_card" / "rk78_card.py"

#: The probe's sample counts and the workload step budget its real runs use.
PROBE_NS = (1_000, 100_000, 1_000_000)
PROBE_STEPS = 100

#: nsys grid: arms x distributions x sample counts x step budgets (large N first, so a
#: stage cut short by its time cap loses the cheapest-to-spare rows). "spread:D" draws the
#: stop steps log-uniformly in [S/D, S] (the card's spread is D = 100); "uniform" is D = 1.
NSYS_ARMS = ("eagle_graph_auto", "eagle_graph_persist", "eagle_graph_steps_compact", "warp_kernel")
NSYS_NS = (1_000_000, 100_000, 10_000, 1_000)
NSYS_DISTS = ("spread", "uniform")
NSYS_STEPS = (100, 1000)
#: the spread-width sweep at N = 1e6, S = 1000 (the grid already holds uniform and spread:100)
WIDTH_DISTS = ("spread:10", "spread:1000")
NSYS_QUICK = (("eagle_graph_auto", 100_000, 100, "spread"),)

#: ncu captures: (label, arm, kernel-name regex). The band arm's step kernel is the
#: 16-steps active-set kernel, its compaction kernel builds the active-set map; the regexes
#: are checked against the names nsys records (a band loop nsys cannot see into is reported
#: as unverified, and ncu then says whether it matched).
NCU_KERNELS = (
    ("band_step", "eagle_graph_steps_compact", r"regex:oscillator_step_x16"),
    ("band_compaction", "eagle_graph_steps_compact", r"regex:(compact|rebuildActive)"),
    ("persist", "eagle_graph_persist", r"regex:oscillator_step_xauto_persist"),
    ("warp", "warp_kernel", r"regex:oscillators"),
)
#: ncu points: stage -> (N, S, distribution): thinning efficiency, bandwidth baseline, launch regime
NCU_POINTS = {"ncu-spread-1e6": (1_000_000, 1000, "spread"),
              "ncu-uniform-1e6": (1_000_000, 1000, "uniform"),
              "ncu-spread-1e4": (10_000, 100, "spread"),
              "ncu-quick": (10_000, 100, "spread")}
NCU_QUICK_KERNELS = ("persist",)
NCU_SECTIONS = ("SpeedOfLight", "Occupancy", "LaunchStats", "WarpStateStats",
                "MemoryWorkloadAnalysis", "SchedulerStats")
NCU_KERNEL_CAP_S = 150
NCU_MIN_CC = 7.0

#: Arm specs are a card arm's name, or one of the kit's own variants ``base[@key=value,...]``:
#: ``osc_persist``, ``osc_fused``, ``osc_compact`` (oscillator, plain or active-set kernel on the persist,
#: fused one-launch or band path; options ``tps`` = persist threads per SM, ``every`` = compaction
#: cadence in steps, ``eager`` = run the band loop from the host so every kernel is profilable) and
#: ``rk78_persist``, ``rk78_fused`` (the RK78 card's kernel).
OSC_PERSIST_TPS = (320, 640, 960, 1280)
RK78_PERSIST_TPS = (256, 384)
SWEEP_STEPS = (1000, 100)
SWEEP_DISTS = ("uniform", "spread")
SWEEP_N = 1_000_000
CADENCE_EVERY = (4, 8, 16, 32)
#: what the ratio stage reads beside the sections: threads per issued instruction (about 32 converged,
#: under 16 fragmented), FP64 pipe use, achieved warps, and the stall classes that split latency from issue
RATIO_METRICS = (
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "sm__inst_executed_pipe_fp64.sum",
    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.per_cycle_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_wait_per_issue_active.ratio",
    "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
)
RK78_POINT = (1_000_000, 0, "orbits")
#: stage -> captures (label, arm spec, kernel-name regex, point, extra metrics)
NCU_BAND_EAGER = "osc_compact@every=16,eager=1"
NCU_BAND_KERNELS = (("band_step", r"regex:oscillator_step"),
                    ("band_compact_predicate", r"regex:compactPredicate"),
                    ("band_compact_scan", r"regex:inclusiveScanBlock"),
                    ("band_compact_scatter", r"regex:compactScatter"))
RATIO_CAPTURES = tuple(
    (f"osc_persist_{d}_s{steps}", "osc_persist", r"regex:oscillator_step_xauto_persist",
     (SWEEP_N, steps, d), RATIO_METRICS) for d in SWEEP_DISTS for steps in SWEEP_STEPS
) + (("rk78_persist", "rk78_persist", r"regex:rkf78_attempt_persist", RK78_POINT, RATIO_METRICS),)


def load_card():
    """The performance card's module: its kernels and arm builders, not re-implemented."""
    spec = importlib.util.spec_from_file_location("perf_card", CARD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["perf_card"] = mod
    spec.loader.exec_module(mod)
    return mod


def say(msg):
    print(f"[profile] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Decision probe
# --------------------------------------------------------------------------- #
def _attr(obj, name):
    """A device-properties key or kernel attribute, 0 when absent (as eagle reads them)."""
    value = obj.get(name, 0) if isinstance(obj, dict) else getattr(obj, name, 0)
    return value if isinstance(value, (int, float)) else 0


def _host_planes(n, steps):
    import numpy as np

    card = sys.modules["perf_card"]
    inp = card._inputs(n, "spread", steps)
    return dict(omega=inp["omega"], zeta=inp["zeta"], nstop=inp["nstop"],
                terminated=np.zeros(n, dtype=bool), x=inp["x0"].copy(),
                v=inp["v0"].copy(), k=np.zeros(n)), inp


def _probe_runner(card, kind, n, device):
    """An ``until_done`` runner for the card's automatic ('auto', active-set) or
    persistent ('persist', plain) kernel, with the arrays of its spread workload."""
    import eagle

    if kind == "auto":
        kernel, extra = card._define_auto_kernel(), {}
    else:
        kernel, extra = card._define_persist_kernel(), {"_fast_mode": "persist", "_lane_utilisation": True}
    if device:
        arm = card._GpuArm(n)
        inp = card._inputs(n, "spread", PROBE_STEPS)
        arm.upload(inp)
        planes = arm.planes()
    else:
        planes, inp = _host_planes(n, PROBE_STEPS)
        arm = None
    # the persist runner is forced only for its timed run: the decision itself is read from the plan
    runner = eagle.until_done(eagle.deploy(kernel), max_steps=PROBE_STEPS, dt=card.DT,
                              **(extra if device else {}), **planes)
    return runner, arm, inp


def _ask(fn):
    """``fn()``, or a note when this eagle lacks what the probe reads (it must run against
    the released wheels and against newer trees alike)."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 -- any failure of one field is that field's answer
        return f"unavailable in this eagle ({type(exc).__name__})"


def probe(target, out):
    """Write ``probe.txt`` (``key: value`` lines in sections) and return the dict."""
    import eagle  # noqa: F401
    import eagle.plan  # noqa: F401

    card = load_card()
    device = target == "gpu"
    rec: dict = {}
    lines: list = []

    def put(key, value):
        value = _ask(value) if callable(value) else value
        rec[key] = value
        lines.append(f"{key}: {value}")

    def head(title):
        lines.extend(["", f"[{title}]"])

    put("target", target)
    put("python", platform.python_version())
    put("eagle", lambda: importlib.metadata.version("raptor-eagle"))
    put("hawk", lambda: importlib.metadata.version("raptor-hawk"))
    lp = _ask(lambda: importlib.import_module("eagle._launch_policy"))
    ud = _ask(lambda: importlib.import_module("eagle._until_done"))
    dev = None
    if device:
        import cupy as cp

        launch_mod = importlib.import_module("eagle.launch")
        dev = _ask(launch_mod._device_props)
        if isinstance(dev, dict):
            put("device.name", lambda: dev["name"].decode() if isinstance(dev["name"], bytes) else dev["name"])
            put("device.compute_capability", lambda: f"{dev['major']}.{dev['minor']}")
            head("device properties eagle reads")
            for key in ("multiProcessorCount", "regsPerMultiprocessor", "maxThreadsPerMultiProcessor",
                        "maxThreadsPerBlock", "warpSize", "sharedMemPerMultiprocessor", "l2CacheSize",
                        "clockRate", "memoryClockRate", "memoryBusWidth"):
                put(f"device.{key}", dev.get(key, "absent"))
        else:
            put("device.name", dev)
            dev = {}
        put("device.cupy", cp.__version__)
    else:
        put("device.name", "none (host plans: capacity is 0 by definition)")

    for kind in ("auto", "persist"):
        runner, _arm, _inp = _probe_runner(card, kind, PROBE_NS[0], device)
        step = _ask(lambda: runner.step)
        head(f"kernel: {kind} ({'active-set kind, compacting' if kind == 'auto' else 'plain kind'})")
        compacting = kind == "auto"
        put(f"{kind}.persist_entry", lambda: step.persist_entry() is not None)
        put(f"{kind}.range_entry", lambda: step.range_entry() is not None)
        reports = _ask(lambda: step.entry_reports())
        put(f"{kind}.entry_reports", reports if isinstance(reports, str) else f"{len(reports)} entries")
        for rep in () if isinstance(reports, str) else reports:
            put(f"{kind}.entry[{rep.name}]",
                f"regs={rep.regs} local_bytes={rep.local_bytes} bounded={rep.bounded} "
                f"resident_threads_per_sm={rep.resident_threads}" + (f" ({rep.note})" if rep.note else ""))
        if device and isinstance(dev, dict):
            attrs = _ask(lambda: launch_mod._kernel_attrs(step._entry))
            if isinstance(attrs, str):
                put(f"{kind}.kernel_attributes", attrs)
            else:
                regs = _attr(attrs, "num_regs")
                put(f"{kind}.num_regs", regs)
                for name in ("local_size_bytes", "shared_size_bytes", "max_threads_per_block"):
                    put(f"{kind}.{name}", _attr(attrs, name))
                sms = _attr(dev, "multiProcessorCount")
                put(f"{kind}.regs_per_sm // num_regs", (_attr(dev, "regsPerMultiprocessor") // regs) if regs else 0)
                put(f"{kind}.resident_threads_per_sm", lambda: lp.resident_threads_per_sm(dev, attrs))
                put(f"{kind}.sms", sms)
                put(f"{kind}.capacity (sms * resident)", lambda: lp.capacity(dev, attrs))
                put(f"{kind}.w_knee (warps per SM)", lambda: lp.W_KNEE_DEFAULT)
                put(f"{kind}.issue_knee (sms * 32 * w_knee)", lambda: sms * 32 * lp.W_KNEE_DEFAULT)
                put(f"{kind}.latency_regime_capacity", lambda: lp.latency_regime_capacity(dev, attrs))
        put(f"{kind}.device_capacity()", lambda: step.device_capacity())
        for n in PROBE_NS:
            put(f"{kind}.fast_mode_for(n={n})", lambda: ud._fast_mode_for(step, n, compacting=compacting))
        head(f"real runs: {kind}, spread, S={PROBE_STEPS}")
        for n in PROBE_NS:
            runner, arm, inp = _probe_runner(card, kind, n, device)
            report = _ask(lambda: runner.run())
            report = getattr(report, "report", report)
            put(f"{kind}.run_mode(n={n})", lambda: report.mode)
            put(f"{kind}.run_launches(n={n})", lambda: report.launches)
            if getattr(report, "lane_utilisation", None) is not None:
                put(f"{kind}.lane_utilisation(n={n})", f"{report.lane_utilisation:.3f}")
            del runner, arm
            if device:
                cp.get_default_memory_pool().free_all_blocks()

    text = "\n".join(lines) + "\n"
    (out / "probe.txt").write_text(text)
    (out / "probe.json").write_text(json.dumps(rec, indent=1, default=str))
    return rec


# --------------------------------------------------------------------------- #
# Worker: one arm, one configuration, traced only around its measured run
# --------------------------------------------------------------------------- #
def make_inputs(card, n, steps, dist):
    """The card's inputs for ``uniform``, ``spread`` (its default width) or ``spread:D``
    (stop steps log-uniform in [S/D, S]; same generator and seed as the card)."""
    import math

    import numpy as np

    kind, _, width = dist.partition(":")
    if kind == "uniform" or (kind == "spread" and width in ("", "100")):
        return card._inputs(n, kind, steps)
    div = float(width)
    inp = card._inputs(n, "spread", steps)
    rng = np.random.default_rng([card.SEED, n, steps, 7, int(div)])
    nstop = np.ceil(np.exp(rng.uniform(math.log(steps / div), math.log(steps), n)))
    nstop = np.clip(nstop, max(1, steps / div), steps)
    nstop[0] = float(steps)
    inp["nstop"] = np.asarray(nstop, dtype="float64")
    return inp


def load_rk78():
    """The RK78 card's module (its kernel, inputs and planes), loaded like the oscillator card."""
    spec = importlib.util.spec_from_file_location("rk78_card", RK78_CARD_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rk78_card"] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_spec(spec):
    """``base[@key=value,...]`` -> (base, {key: value})."""
    base, _, opts = spec.partition("@")
    return base, dict(kv.split("=", 1) for kv in opts.split(",") if kv)


_OVERRIDE_LOCK = threading.Lock()


@contextlib.contextmanager
def persist_threads(tps):
    """Inside the block, a persist launch uses ``tps`` threads per SM instead of eagle's rule (``None``:
    the rule). eagle has no hook for this, so the sizing function its launch calls is replaced for the
    block -- a measurement tool, undone on exit; :func:`launched_threads_per_sm` reads what really launched.
    The replacement is process-wide, so a second block entered meanwhile (another thread, or nested)
    raises instead of racing."""
    if tps is None:
        yield
        return
    if not _OVERRIDE_LOCK.acquire(blocking=False):
        raise RuntimeError("persist_threads is already active in this process: run one sweep cell at a time")
    policy = importlib.import_module("eagle._launch_policy")
    original = policy.persistent_threads_per_sm
    policy.persistent_threads_per_sm = lambda *args, **kwargs: tps
    try:
        yield
    finally:
        policy.persistent_threads_per_sm = original
        _OVERRIDE_LOCK.release()


def launched_threads_per_sm(runner):
    """Threads per SM of the runner's persist launch (block x grid / SMs), or a note when this eagle
    keeps its launch geometry elsewhere."""
    try:
        sms = _attr(importlib.import_module("eagle.launch")._device_props(), "multiProcessorCount")
        return int(runner._fast_block * runner._fast_grid) // sms
    except Exception as exc:  # noqa: BLE001 -- the geometry attributes are private and differ across eagles
        return f"unavailable in this eagle ({type(exc).__name__})"


def _runner_arm(state, runner):
    """The arm dict the kit times: ``run`` keeps the report of the last run."""
    state.runner = runner
    arm = {"state": state, "download": state.download, "report": None}

    def run():
        report = runner.run()
        arm["report"] = getattr(report, "report", report)

    arm["run"] = run
    return arm


def _eager_band(arm, runner):
    """Replace the arm's graph replay with the band loop run from the host: the same kernels,
    launched one by one, so a profiler can see each (a graph's conditional loop node hides them)."""
    ud = importlib.import_module("eagle._until_done")

    def run():
        ud._mask_count(runner._xp, runner._device, runner.finished, runner.terminated)
        runner.loop()
        arm["report"] = SimpleNamespace(mode="band-eager", launches=int(runner.loop.iterations()),
                                        compactions=None)

    arm["run"] = run
    return arm


def build_arm(spec, n, steps, dist):
    """(arm, inputs) of one arm spec (see :data:`OSC_PERSIST_TPS`'s comment): a card arm, or a variant."""
    import eagle
    import eagle.plan  # noqa: F401  (eagle.simulate first in a process imports it circularly)

    base, opts = parse_spec(spec)
    if base.startswith("rk78_"):
        rk = load_rk78()
        mode = {"persist": "persist", "fused": "fused_one"}[base.split("_", 1)[1]]
        state = rk._GpuArm(n)
        runner = eagle.until_done(eagle.deploy(rk.ARM_KERNEL["eagle_graph_persist"]()), max_steps=rk.MAX_ATTEMPTS,
                                  t_final=rk.T_FINAL, terminated=state.terminated, **state.planes,
                                  _fast_mode=mode, _lane_utilisation=(mode == "persist"))
        return _runner_arm(state, runner), rk._inputs(n)
    card = load_card()
    inp = make_inputs(card, n, steps, dist)
    if base in ("osc_persist", "osc_fused"):
        mode = "persist" if base == "osc_persist" else "fused_one"
        state = card._GpuArm(n)
        runner = eagle.until_done(eagle.deploy(card._define_persist_kernel()), max_steps=steps, dt=card.DT,
                                  _fast_mode=mode, _lane_utilisation=(mode == "persist"), **state.planes())
        return _runner_arm(state, runner), inp
    if base == "osc_compact":
        state = card._GpuArm(n)
        runner = eagle.until_done(eagle.deploy(card._define_active_kernel()), max_steps=steps, dt=card.DT,
                                  every=int(opts.get("every", card.COMPACT_EVERY)), _fast_mode=False,
                                  **state.planes())
        arm = _runner_arm(state, runner)
        return (_eager_band(arm, runner) if opts.get("eager") == "1" else arm), inp
    return card._make_gpu_arms(n, steps, names=(base,))[base], inp


def worker(spec, n, steps, dist):
    """Build one arm of the card, warm it, then run it once between
    ``cudaProfilerStart/Stop`` so a profiler attached with a capture range
    sees that run alone."""
    import cupy as cp

    tps = parse_spec(spec)[1].get("tps")
    with persist_threads(int(tps) if tps else None):
        arm, inp = build_arm(spec, n, steps, dist)
        for _ in range(2):
            arm["state"].upload(inp)  # compile, graph capture, caches
            arm["run"]()
        arm["state"].upload(inp)
        cp.cuda.runtime.deviceSynchronize()
        start, stop = cp.cuda.Event(), cp.cuda.Event()
        cp.cuda.runtime.profilerStart()
        start.record()
        arm["run"]()
        stop.record()
        stop.synchronize()
        cp.cuda.runtime.profilerStop()
    report = arm.get("report")
    print(f"worker arm={spec} n={n} steps={steps} dist={dist} "
          f"mode={getattr(report, 'mode', None)} launches={getattr(report, 'launches', None)} "
          f"compactions={getattr(report, 'compactions', None)} "
          f"wall_ms={cp.cuda.get_elapsed_time(start, stop):.3f}")


def _self_cmd(arm, n, steps, dist):
    return [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", arm, str(n), str(steps), dist]


# --------------------------------------------------------------------------- #
# Sweeps: wall time of one arm over a parameter, in one process
# --------------------------------------------------------------------------- #
SWEEP_WARM, SWEEP_REPS = 2, 3


def time_cell(spec, n, steps, dist):
    """Median and best CUDA-event time (ms) of one arm over its inputs, the report of its last run
    and the persist threads per SM it launched. Built, warmed and timed inside the spec's override."""
    import cupy as cp

    base, opts = parse_spec(spec)
    with persist_threads(int(opts["tps"]) if "tps" in opts else None):
        arm, inp = build_arm(spec, n, steps, dist)
        for _ in range(SWEEP_WARM):
            arm["state"].upload(inp)
            arm["run"]()
        walls = []
        for _ in range(SWEEP_REPS):
            arm["state"].upload(inp)
            cp.cuda.runtime.deviceSynchronize()
            start, stop = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            arm["run"]()
            stop.record()
            stop.synchronize()
            walls.append(cp.cuda.get_elapsed_time(start, stop))
        launched = launched_threads_per_sm(arm["state"].runner) if base.endswith("persist") else "-"
    report = arm["report"]
    del arm
    cp.get_default_memory_pool().free_all_blocks()
    walls.sort()
    return walls[len(walls) // 2], walls[0], report, launched


def sweep_cells(stage):
    """(label, spec, S, dist, requested threads per SM) in run order; S = 0 and dist 'orbits' for the RK78 card."""
    if stage.endswith("-quick"):  # one point: S = 1000 spread; the persist sweep adds one override to prove it applies
        quick = [("osc", "osc_fused" if stage.startswith("fused") else "osc_persist", 1000, "spread", None)]
        if stage.startswith("persist"):
            quick.append(("osc", "osc_persist@tps=640", 1000, "spread", 640))
        return quick
    stage = stage.removesuffix("-quick")
    cells = []
    for steps in SWEEP_STEPS:
        for dist in SWEEP_DISTS:
            if stage == "fused-sweep":
                cells.append(("osc", "osc_fused", steps, dist, None))
            else:
                cells += [("osc", "osc_persist" if t is None else f"osc_persist@tps={t}", steps, dist, t)
                          for t in (None, *OSC_PERSIST_TPS)]
    if stage == "fused-sweep":
        cells.append(("rk78", "rk78_fused", 0, "orbits", None))
    else:
        cells += [("rk78", "rk78_persist" if t is None else f"rk78_persist@tps={t}", 0, "orbits", t)
                  for t in (None, *RK78_PERSIST_TPS)]
    return cells


def sweep_stage(out, stage):
    """Time every cell; the table is rewritten after each, so a stage cut short keeps what it finished.
    A cell this eagle refuses (no such entry, no override) is a row saying so, never a stage failure."""
    head = (f"{'kernel':6s} {'S':>5s} {'dist':8s} {'tps_req':>8s} {'tps_launched':>12s} {'mode':10s} "
            f"{'median_ms':>10s} {'best_ms':>9s} {'lane_util':>9s}")
    rows, refused = [], 0
    path = out / f"{stage.replace('-', '_')}.txt"
    note = ("Wall = CUDA events around one run (median of 3 after 2 warm runs). tps_req 'auto' = eagle's own rule; "
            "tps_launched = block x grid / SMs of the persist launch as built (the register-limited ceiling can "
            "lower a request). lane_util = active lanes / warp-iterations (persist only).\n")
    for label, spec, steps, dist, tps in sweep_cells(stage):
        n = RK78_POINT[0] if label == "rk78" else SWEEP_N
        try:
            median, best, report, launched = time_cell(spec, n, steps, dist)
            util = getattr(report, "lane_utilisation", None)
            row = (f"{label:6s} {steps:>5d} {dist:8s} {'auto' if tps is None else tps:>8} {str(launched):>12s} "
                   f"{str(getattr(report, 'mode', '?')):10s} {median:10.3f} {best:9.3f} "
                   f"{'-' if util is None else format(util, '.3f'):>9s}")
        except Exception as exc:  # noqa: BLE001 -- one refused cell must not end the sweep
            refused += 1
            row = (f"{label:6s} {steps:>5d} {dist:8s} {'auto' if tps is None else tps:>8} "
                   f"unavailable in this eagle: {type(exc).__name__}: {str(exc)[:120]}")
        rows.append(row)
        path.write_text(note + head + "\n" + "\n".join(rows) + "\n")
        say(f"{stage} {spec} S={steps} {dist}: {row.split(None, 5)[-1]}")
    if refused == len(rows):
        raise RuntimeError(f"every cell of {stage} was refused; see {path.name}")


# --------------------------------------------------------------------------- #
# nsys
# --------------------------------------------------------------------------- #
def _query(con, sql):
    try:
        return con.execute(sql).fetchone()
    except sqlite3.Error:
        return None


def nsys_stage(out, name, cells, nsys):
    """Trace every cell; the summary file is rewritten after each capture, so a stage cut
    short by its time cap still leaves everything it finished."""
    rows, texts, names = [], [], set()
    summary = out / f"{name}.txt"
    table_head = ("arm                        dist        N        S     mode       launches  "
                  "traced_kernels  kernel_ms  api_calls  api_ms  compactions  wall_ms")
    notes = ("Measured run only (profiler start/stop around one run after two warm runs).\n"
             "launches = what eagle reports for the run; traced_kernels / kernel_ms = what nsys recorded. A band run\n"
             "whose loop sits in a conditional graph can show fewer traced kernels than launches: the\n"
             "device-side loop's kernels are then missing from the trace, not absent from the run.\n")

    def flush():
        table = [table_head] + [
            f"{r[0]:26s} {r[1]:11s} {r[2]:<8d} {r[3]:<5d} {str(r[4]):10s} {str(r[5]):>8s}  {str(r[6]):>14s}  "
            f"{str(r[7]):>9s}  {str(r[8]):>9s}  {str(r[9]):>6s}  {str(r[10]):>11s}  {str(r[11]):>7s}" for r in rows]
        summary.write_text(notes + "\n".join(table) + "\n\n" + "\n".join(texts))

    for arm, n, steps, dist in cells:
        tag = re.sub(r"[^A-Za-z0-9_.-]", "", f"{arm.replace('@', '-')}-{dist.replace(':', '')}-n{n}-s{steps}")
        rep = out / f"nsys_{tag}"
        t0 = time.perf_counter()
        cmd = [nsys, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none",
               "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
               "-f", "true", "-o", str(rep)] + _self_cmd(arm, n, steps, dist)
        done = subprocess.run(cmd, capture_output=True, text=True)
        said = next((ln for ln in done.stdout.splitlines() if ln.startswith("worker ")), "")
        mode = (re.search(r"mode=(\S+)", said) or [None, "?"])[1]
        launches = (re.search(r"launches=(\S+)", said) or [None, "?"])[1]
        compactions = (re.search(r"compactions=(\S+)", said) or [None, "?"])[1]
        wall_ms = (re.search(r"wall_ms=(\S+)", said) or [None, "?"])[1]
        report = pathlib.Path(str(rep) + ".nsys-rep")
        if done.returncode != 0 or not report.exists():
            rows.append((arm, dist, n, steps, mode, launches, "FAILED", "", "", "", "", ""))
            texts.append(f"== {tag}: nsys failed (exit {done.returncode})\n{done.stderr[-800:]}\n")
            flush()
            continue
        db = rep.with_suffix(".sqlite")
        subprocess.run([nsys, "export", "--type", "sqlite", "-f", "true", "-o", str(db), str(report)],
                       capture_output=True, text=True)
        k = a = (None, None)
        if db.exists():
            con = sqlite3.connect(db)
            k = _query(con, "SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_KERNEL") or k
            a = _query(con, "SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_RUNTIME") or a
            for (kname,) in con.execute("SELECT DISTINCT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                                        "JOIN StringIds s ON k.demangledName = s.id").fetchall() if k[0] else []:
                names.add((arm, kname))
            con.close()
            db.unlink()
        stats = subprocess.run(
            [nsys, "stats", "--force-export=true", "-r", "cuda_gpu_kern_sum", "-r", "cuda_api_sum",
             "-r", "cuda_gpu_mem_time_sum", str(report)], capture_output=True, text=True)
        db.unlink(missing_ok=True)  # `nsys stats` exports its own copy
        wall = time.perf_counter() - t0
        texts.append(f"== {tag} (mode {mode}, {wall:.0f} s)\n{stats.stdout}\n")
        rows.append((arm, dist, n, steps, mode, launches, k[0], f"{(k[1] or 0) * 1e-6:.3f}", a[0],
                     f"{(a[1] or 0) * 1e-6:.3f}", compactions, wall_ms))
        flush()
        say(f"nsys {tag}: kernels={k[0]} kernel_ms={(k[1] or 0) * 1e-6:.3f} api_calls={a[0]} ({wall:.0f} s)")
    flush()
    return names


def nsys_cells(stage):
    if stage == "nsys-quick":
        return list(NSYS_QUICK)
    if stage == "cadence-sweep-quick":
        return [("osc_compact@every=16", SWEEP_N, 1000, "spread"), (f"{NCU_BAND_EAGER}", SWEEP_N, 1000, "spread")]
    if stage == "cadence-sweep":
        # the graph run gives the wall and the compaction count; the eager run (same kernels, launched from the host)
        # gives the per-kernel sums the graph's conditional loop node hides from nsys
        return [(f"osc_compact@every={e}{eager}", SWEEP_N, 1000, d)
                for d in SWEEP_DISTS for e in CADENCE_EVERY for eager in ("", ",eager=1")]
    if stage == "nsys-grid":
        return [(a, n, s, d) for n in NSYS_NS for d in NSYS_DISTS for s in NSYS_STEPS for a in NSYS_ARMS]
    return [(a, 1_000_000, 1000, d) for d in WIDTH_DISTS for a in NSYS_ARMS]


# --------------------------------------------------------------------------- #
# ncu
# --------------------------------------------------------------------------- #
def compute_capability():
    sel = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    try:
        done = subprocess.run(["nvidia-smi", "-i", sel, "--query-gpu=compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30)
        return float(done.stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def ncu_command(ncu, rep, regex, arm, point, sudo, metrics=()):
    """One kernel's capture: the first matching launch of the measured run, never more."""
    n, steps, dist = point
    cmd = [ncu, "--profile-from-start", "off", "--launch-skip", "0", "--launch-count", "1",
           "--kernel-name", regex, "--target-processes", "all", "-f", "-o", str(rep)]
    for section in NCU_SECTIONS:
        cmd += ["--section", section]
    if metrics:
        cmd += ["--metrics", ",".join(metrics)]
    cmd += _self_cmd(arm, n, steps, dist)
    if sudo:
        cmd = ["sudo", "-n", "-E", "env", f"PATH={os.environ['PATH']}"] + cmd
    return cmd


class Skip(Exception):
    """A stage that has nothing to do here (reason in the message)."""


def ncu_captures(stage):
    """The stage's captures: (label, arm spec, kernel-name regex, (N, S, dist), extra metrics)."""
    if stage == "ncu-ratio":
        return list(RATIO_CAPTURES)
    if stage == "ncu-ratio-quick":
        return [c for c in RATIO_CAPTURES if c[0] == f"osc_persist_spread_s{SWEEP_STEPS[0]}"]
    if stage in ("ncu-band", "ncu-band-quick"):
        kernels = NCU_BAND_KERNELS[:2] if stage.endswith("-quick") else NCU_BAND_KERNELS
        return [(label, NCU_BAND_EAGER, regex, (SWEEP_N, 1000, "spread"), ()) for label, regex in kernels]
    point = NCU_POINTS[stage]
    return [(label, arm, regex, point, ()) for label, arm, regex in NCU_KERNELS
            if stage != "ncu-quick" or label in NCU_QUICK_KERNELS]


def ncu_stage(out, stage, ncu, sudo):
    cc_ = compute_capability()
    if cc_ is None:
        raise Skip("compute capability not readable (nvidia-smi)")
    if cc_ < NCU_MIN_CC:
        raise Skip(f"compute capability {cc_} < {NCU_MIN_CC} (Nsight Compute needs Volta or newer)")
    chosen = ncu_captures(stage)
    summary, failed = [], 0
    for label, arm, regex, point, metrics in chosen:
        rep = out / f"{stage}_{label}"
        t0 = time.perf_counter()
        try:
            done = subprocess.run(ncu_command(ncu, rep, regex, arm, point, sudo, metrics), capture_output=True,
                                  text=True, timeout=NCU_KERNEL_CAP_S)
            code, tail = done.returncode, (done.stdout + done.stderr)[-1500:]
        except subprocess.TimeoutExpired:
            code, tail = 124, f"ncu exceeded {NCU_KERNEL_CAP_S} s"
        report = pathlib.Path(str(rep) + ".ncu-rep")
        if sudo and report.exists():
            subprocess.run(["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(report)])
        if code != 0 or not report.exists():
            failed += 1
            summary.append(f"== {label}: ncu failed (exit {code})\n{tail}\n")
            say(f"ncu {stage} {label}: failed")
        else:
            text = subprocess.run([ncu, "--import", str(report), "--csv", "--page", "details"],
                                  capture_output=True, text=True)
            (out / f"{stage}_{label}.csv").write_text(text.stdout)
            digest = _ncu_digest(text.stdout)
            if metrics:
                raw = subprocess.run([ncu, "--import", str(report), "--csv", "--page", "raw"],
                                     capture_output=True, text=True)
                (out / f"{stage}_{label}.raw.csv").write_text(raw.stdout)
                digest += "\n" + _ncu_metric_lines(raw.stdout, metrics)
            summary.append(f"== {label} ({arm}, {regex}, N={point[0]} S={point[1]} {point[2]}, "
                           f"{time.perf_counter() - t0:.0f} s)\n{digest}\n")
            say(f"ncu {stage} {label}: done ({time.perf_counter() - t0:.0f} s)")
        (out / f"{stage}.txt").write_text("\n".join(summary))
    if failed == len(chosen):
        raise RuntimeError(f"every ncu capture of {stage} failed; see {stage}.txt")


_DIGEST = ("Duration", "Registers Per Thread", "Theoretical Occupancy", "Achieved Occupancy",
           "Block Size", "Grid Size", "DRAM Throughput", "Memory Throughput", "Compute (SM) Throughput",
           "Achieved Active Warps Per SM", "Warp Cycles Per Issued Instruction", "Issue Slots Busy",
           "No Eligible", "Occupancy Limiter", "Block Limit Registers")


def _ncu_digest(csv_text):
    """The lines of the details page worth reading first (the full CSV is kept alongside)."""
    import csv

    keep = []
    for row in csv.DictReader(csv_text.splitlines()):
        name = row.get("Metric Name", "")
        if any(name.startswith(d) for d in _DIGEST):
            keep.append(f"  {row.get('Section Name', ''):24s} {name:44s} {row.get('Metric Value', '')} "
                        f"{row.get('Metric Unit', '')}")
    return "\n".join(keep)


def _ncu_metric_lines(raw_csv, metrics):
    """``name value unit`` for each requested metric from ncu's raw page (one column per metric; a
    units row follows the header). A metric the report lacks is named as missing, never skipped."""
    import csv

    rows = list(csv.reader(raw_csv.splitlines()))
    if len(rows) < 2:
        return "\n".join(f"  {m}: missing (empty raw page)" for m in metrics)
    header, body = rows[0], rows[1:]
    units = body[0] if body and not body[0][0].strip().isdigit() else [""] * len(header)
    values = next((r for r in body if r and r[0].strip().isdigit()), [""] * len(header))
    out = []
    for metric in metrics:
        if metric in header:
            col = header.index(metric)
            out.append(f"  {metric}: {values[col] if col < len(values) else '?'} "
                       f"{units[col] if col < len(units) else ''}".rstrip())
        else:
            out.append(f"  {metric}: missing from this report")
    return "\n".join(out)


def validate_ncu(ncu, nsys_names):
    """Check without a supported GPU: every chosen section and metric exists in this ncu, and each
    kernel-name regex matches a name nsys recorded for its arm. Returns (ok, lines)."""
    lines, ok = [], True
    listed = subprocess.run([ncu, "--list-sections"], capture_output=True, text=True).stdout
    for section in NCU_SECTIONS:
        found = re.search(rf"^{section}\s", listed, re.M) is not None
        ok &= found
        lines.append(f"section {section}: {'found' if found else 'MISSING'}")
    known = subprocess.run([ncu, "--query-metrics", "--chip", "ga100"], capture_output=True, text=True).stdout
    names = {ln.split()[0] for ln in known.splitlines() if ln.strip()}
    for metric in RATIO_METRICS:
        found = metric.split(".", 1)[0] in names
        ok &= found
        lines.append(f"metric {metric}: {'found' if found else 'MISSING'}")
    for label, arm, regex in validation_kernels():
        pattern = re.compile(regex.split("regex:", 1)[1])
        hits = sorted({n for a, n in nsys_names if a == arm and pattern.search(n)})
        if arm == "eagle_graph_steps_compact" and not hits:
            lines.append(f"kernel {label} ({arm}) {regex}: unverified (nsys does not record the band loop's kernels)")
            continue
        ok &= bool(hits)
        lines.append(f"kernel {label} ({arm}) {regex}: {len(hits)} match(es) {hits[:3]}")
    return ok, lines


def validation_kernels():
    """(label, arm spec, regex) of every distinct kernel the ncu stages will ask for."""
    seen = {}
    for stage in ("ncu-spread-1e6", "ncu-ratio", "ncu-band"):
        for label, arm, regex, _point, _metrics in ncu_captures(stage):
            seen.setdefault((arm, regex), f"{stage}:{label}")
    return [(label, arm, regex) for (arm, regex), label in seen.items()]


# --------------------------------------------------------------------------- #
STAGES = ("probe", "nsys-grid", "nsys-width", "nsys-quick", *NCU_POINTS, "ncu-ratio", "ncu-band",
          "persist-sweep", "fused-sweep", "cadence-sweep", "validate-ncu",
          *(f"{s}-quick" for s in ("ncu-ratio", "ncu-band", "persist-sweep", "fused-sweep", "cadence-sweep")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("profile-out"))
    ap.add_argument("--target", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--probe-only", action="store_true", help="the probe stage alone")
    ap.add_argument("--stage", choices=STAGES, help="run one stage (profile.sh runs them in order)")
    ap.add_argument("--nsys", help="path of nsys")
    ap.add_argument("--ncu", help="path of ncu")
    ap.add_argument("--ncu-sudo", action="store_true", help="run ncu under `sudo -n` (counters are admin-only)")
    ap.add_argument("--worker", nargs=4, metavar=("ARM", "N", "S", "DIST"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.worker:
        worker(args.worker[0], int(args.worker[1]), int(args.worker[2]), args.worker[3])
        return 0
    stage = "probe" if args.probe_only else args.stage
    if stage is None:
        ap.error("pass --stage NAME (or --probe-only); profile.sh runs the whole sequence")
    if args.target == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        if stage == "probe":
            probe(args.target, args.out)
        elif stage.removesuffix("-quick") in ("persist-sweep", "fused-sweep"):
            if args.target == "cpu":
                raise Skip("the sweeps time the device")
            sweep_stage(args.out, stage)
        elif stage.startswith("nsys") or stage in ("validate-ncu", "cadence-sweep", "cadence-sweep-quick"):
            if not args.nsys:
                raise Skip("nsys not found")
            if stage == "validate-ncu":
                cells = [(arm, 100_000, 100, "spread") for arm in dict.fromkeys(k[1] for k in validation_kernels())]
                names = nsys_stage(args.out, "validate_nsys", cells, args.nsys)
                ok, lines = validate_ncu(args.ncu, names)
                (args.out / "ncu_validation.txt").write_text("\n".join(lines) + f"\n{'OK' if ok else 'FAILED'}\n")
                print("\n".join(lines))
                return 0 if ok else 1
            nsys_stage(args.out, stage.replace("-", "_"), nsys_cells(stage), args.nsys)
        else:
            if not args.ncu:
                raise Skip("ncu not found")
            ncu_stage(args.out, stage, args.ncu, args.ncu_sudo)
    except Skip as why:
        print(f"SKIP: {why}")
        return 3
    return 0


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        sys.exit(main())
