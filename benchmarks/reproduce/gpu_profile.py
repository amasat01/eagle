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
* **compare**: every comparison case on one matrix, eagle against Warp: the oscillator and RK78 kernels over N and
  (oscillator) step budget and stop-step spread, each with eagle auto (and the mode it took), the forced fused
  one-launch entry, the persist launch over threads per SM, the band path over the compaction cadence, and Warp's kernel;
  CUDA-event walls, a correctness guard per arm, a table, a CSV and a summary of where auto trails Warp or the fastest
  eagle path (it covers the persist and fused sweeps of earlier kits, which it replaces);
  the oscillator also runs in float32 (``osc32``, ``precision=fp32``; arm specs carry ``fp=32``) at N = 1e5, 1e6,
  S = 100, 1000, spread and uniform, with no Warp arm; every table and CSV row is labelled with its precision;
* **crossover**: the fused one-launch entry against the persist launch at N = 3e4, 1e5, 3e5, S = 100, card spread,
  with eagle's probe on, and the entry auto settled on (its mode after each run);
* **band**: the band path over the compaction cadence (16, 32, 64, 128) beside eagle's own pick, at N = 1e5 and 1e6,
  and R = one compaction group's cost over one full-live step launch, fitted from the uniform cell;
* **ncu-tsweep**: achieved warps (and the ratio metrics) of the persist kernels at each persist size, N = 1e6;
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
OSC_PERSIST_TPS = (384, 640, 896, 1152, 1536, 2048)
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
#: FP32 variant of the oscillator (spec option ``fp=32``; same model, float32 scalar type): persist sizes of its compare
#: cells, its sample counts, and the ratio metrics plus the FP32 FMA pipe it is read against
OSC32_PERSIST_TPS = (640, 1024, 1536, 2048)
OSC32_NS = (100_000, 1_000_000)
OSC32_STEPS = (1000, 100)
OSC32_DISTS = ("spread", "uniform")
RATIO_METRICS_FP32 = RATIO_METRICS + (
    "sm__inst_executed_pipe_fma.sum",
    "sm__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active",
)
#: threads-per-SM sweep under ncu: achieved warps at each persist size, N = 1e6, spread
TSWEEP_STEPS = (100, 1000)
TSWEEP_FP32_STEPS = 1000  # the FP32 rows: N = 1e6, spread, this S
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


def precision_of(kernel_or_spec):
    """``fp32`` for the float32 oscillator (kernel ``osc32``, or an arm spec with ``fp=32``), else ``fp64``."""
    if kernel_or_spec == "osc32":
        return "fp32"
    return "fp32" if "@" in kernel_or_spec and parse_spec(kernel_or_spec)[1].get("fp") == "32" else "fp64"


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


def _persist_pick(card, n, fp32):
    """Threads per SM eagle's launch rule picks for the forced persist entry (spread, S = PROBE_STEPS) in float32
    or float64, read from the geometry of a real run."""
    arm, inp = build_arm("osc_persist@fp=32" if fp32 else "osc_persist", n, PROBE_STEPS, "spread")
    arm["state"].upload(inp)
    arm["run"]()
    return launched_threads_per_sm(arm["state"].runner)


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
    put("hawk.literal_table_option", lambda: ", ".join(
        n for n in dir(importlib.import_module("hawk")) if "literal" in n.lower() and "table" in n.lower())
        or "not exposed by this hawk: the literal-table ON/OFF comparison is skipped")
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
                f"resident_threads_per_sm={rep.resident_threads}" + (f" ({rep.note})" if rep.note else "")
                + "".join(f" {k}={v}" for k, v in read_fields(rep).items()))
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
            put(f"{kind}.launch_fields(n={n})",
                lambda: read_fields(runner, report) or "unavailable in this eagle (no T, c, f, R, k or text_bytes exposed)")
            if getattr(report, "lane_utilisation", None) is not None:
                put(f"{kind}.lane_utilisation(n={n})", f"{report.lane_utilisation:.3f}")
            del runner, arm
            if device:
                cp.get_default_memory_pool().free_all_blocks()

    head("persist entry, FP32 oscillator (precision=fp32) beside FP64 (precision=fp64)")
    if device:
        for n in PROBE_NS[1:]:
            for prec in ("fp32", "fp64"):
                put(f"persist.threads_per_sm[precision={prec}](n={n})", lambda: _persist_pick(card, n, prec == "fp32"))
    else:
        put("persist.threads_per_sm[precision=fp32]", "none (host plans launch no persist kernel)")

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
    policy = importlib.import_module("eagle._launch_policy")
    original = policy.persistent_threads_per_sm  # an eagle without the hook raises here, holding nothing
    if not _OVERRIDE_LOCK.acquire(blocking=False):
        raise RuntimeError("persist_threads is already active in this process: run one cell at a time")
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
        which = base.split("_", 1)[1]
        if which == "warp":
            return rk._make_warp_arm(n), rk._inputs(n)
        # auto keeps until_done's own pick; the others force one entry (band: the single-attempt kernel, compacted)
        extra = {"auto": {}, "persist": {"_fast_mode": "persist", "_lane_utilisation": True},
                 "fused": {"_fast_mode": "fused_one"},
                 "band": {"_fast_mode": False, **({"every": int(opts["every"])} if "every" in opts else {})}}[which]
        state = rk._GpuArm(n)
        kernel = rk._define_active_kernel() if which == "band" else rk._define_kernel()
        runner = eagle.until_done(eagle.deploy(kernel), max_steps=rk.MAX_ATTEMPTS, t_final=rk.T_FINAL,
                                  terminated=state.terminated, **state.planes, **extra)
        return _runner_arm(state, runner), rk._inputs(n)
    card = load_card()
    inp = make_inputs(card, n, steps, dist)
    fp32 = opts.get("fp") == "32"  # the same model in float32: planes, inputs and the kernel's scalar type
    dtype = "float32" if fp32 else "float64"
    if fp32:
        inp = {k: v.astype(dtype) for k, v in inp.items()}

    def deploy(kernel):
        return eagle.deploy(kernel, scalar_type="float32") if fp32 else eagle.deploy(kernel)

    if fp32 and base == "eagle_graph_auto":
        state = card._GpuArm(n, dtype)
        runner = eagle.until_done(deploy(card._define_auto_kernel()), max_steps=steps, dt=card.DT, **state.planes())
        return _runner_arm(state, runner), inp
    if base in ("osc_persist", "osc_fused"):
        mode = "persist" if base == "osc_persist" else "fused_one"
        state = card._GpuArm(n, dtype)
        runner = eagle.until_done(deploy(card._define_persist_kernel()), max_steps=steps, dt=card.DT,
                                  _fast_mode=mode, _lane_utilisation=(mode == "persist"), **state.planes())
        return _runner_arm(state, runner), inp
    if base == "osc_band":
        import hawk

        state = card._GpuArm(n, dtype)
        steps_kernel = hawk.steps(card._define_active_kernel(), BAND_STEPS_PER_LAUNCH)
        runner = eagle.until_done(deploy(steps_kernel), max_steps=steps, dt=card.DT, _fast_mode=False,
                                  **({"every": int(opts["every"])} if "every" in opts else {}), **state.planes())
        return _runner_arm(state, runner), inp
    if base == "osc_compact":
        state = card._GpuArm(n)
        runner = eagle.until_done(eagle.deploy(card._define_active_kernel()), max_steps=steps, dt=card.DT,
                                  every=int(opts.get("every", card.COMPACT_EVERY)), _fast_mode=False,
                                  **state.planes())
        arm = _runner_arm(state, runner)
        return (_eager_band(arm, runner) if opts.get("eager") == "1" else arm), inp
    return card._make_gpu_arms(n, steps, names=(base,))[base], inp


def _step_accounting(spec, arm, inp, steps):
    """``sample_steps`` (sample-steps the run executed), ``ops_per_step`` (FP64 instructions per sample-step) and
    where that came from, for the lane efficiency ncu reports are divided by. The count comes from the
    kernel's sidecar when this eagle exposes it, else from the card's own FP64 count of the kernel."""
    base = parse_spec(spec)[0]
    try:
        if base.startswith("rk78_"):
            out = arm["download"]()
            total, card_ops = float((out["n_acc"] + out["n_rej"]).sum()), load_rk78().FLOPS_PER_ATTEMPT
        elif base.startswith("osc_") or base == "eagle_graph_auto":
            total, card_ops = float(inp["nstop"].sum()), load_card().FLOPS_PER_SAMPLE_STEP
        else:
            return "sample_steps=- ops_per_step=- ops_source=-"
        runner = getattr(arm["state"], "runner", None)
        sidecar = next((getattr(o, "ops_per_step") for o in (getattr(runner, "step", None), runner)
                        if isinstance(getattr(o, "ops_per_step", None), (int, float))), None)
        ops, src = (sidecar, "sidecar") if sidecar is not None else (card_ops, "card-count")
        launched = launched_threads_per_sm(runner) if base.endswith("persist") else "-"
        return f"sample_steps={total:.0f} ops_per_step={ops} ops_source={src} launched_tps={launched}"
    except Exception as exc:  # noqa: BLE001 -- the accounting is optional
        return f"sample_steps=- ops_per_step=- ops_source=unavailable({type(exc).__name__}) launched_tps=-"


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
    accounting = _step_accounting(spec, arm, inp, steps)
    print(f"worker arm={spec} n={n} steps={steps} dist={dist} {accounting} "
          f"mode={getattr(report, 'mode', None)} launches={getattr(report, 'launches', None)} "
          f"compactions={getattr(report, 'compactions', None)} "
          f"wall_ms={cp.cuda.get_elapsed_time(start, stop):.3f}")


def _self_cmd(arm, n, steps, dist):
    return [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", arm, str(n), str(steps), dist]


# --------------------------------------------------------------------------- #
# compare: every arm of eagle against Warp on one matrix, in one process
# --------------------------------------------------------------------------- #
SWEEP_WARM, SWEEP_REPS = 2, 3
COMPARE_NS = (1_000, 10_000, 100_000, 1_000_000)
COMPARE_OSC_STEPS = (1000, 100)
#: stop-step distributions: uniform, the card's spread (D = 100), a narrow and a wide log-uniform width
COMPARE_OSC_DISTS = ("uniform", "spread", "spread:10", "spread:1000")
#: compaction cadences of the band arm (None: eagle's own default) and its steps per launch
BAND_EVERY = (None, 8, 16)
BAND_STEPS_PER_LAUNCH = 8
#: the crossover stage: fused_one against persist below the persist launch's capacity (S = 100, card spread)
CROSSOVER_NS = (30_000, 100_000, 300_000)
#: the band stage: N, the distributions, and the compaction cadences beside eagle's own pick; the fit of wall over
#: S / every on the uniform cell gives the compaction group's cost C and the step launch's cost, hence R
BAND_NS = (100_000, 1_000_000)
BAND_DISTS = ("uniform", "spread")
BAND_SWEEP_EVERY = (16, 32, 64, 128)
BAND_STEPS = 1000
#: what an eagle may expose about a launch (threads per SM, chunk, step fill, compaction ratio, steps per launch,
#: persist text size); read where found, never required
FIELD_NAMES = ("T", "c", "f", "R", "k", "text_bytes", "every", "steps_per_launch")
#: context arms (other frameworks), run once at the card's own N = 1e6, S = 1000 spread point and only while
#: the stage has used less than this share of its budget
CONTEXT_ARMS = ("torch_compiled", "jax_vmap")
CONTEXT_POINT = (1_000_000, 1000, "spread")
CONTEXT_BUDGET_SHARE = 0.6
#: auto counts as having picked the fastest eagle path when within this share of it (timer noise)
AUTO_TOLERANCE = 0.05
_WARP_SPECS = ("warp_kernel", "rk78_warp")
_WALL_SPECS = CONTEXT_ARMS


def compare_arms(kernel):
    """(label, arm spec) of every arm of one kernel, the reference (eagle auto) first and Warp last."""
    if kernel == "osc32":  # no Warp arm: the float32 variant is eagle-only (auto against its own entries)
        fp = "fp=32"
        arms = [("auto", f"eagle_graph_auto@{fp}"), ("fused_one", f"osc_fused@{fp}"), ("persist@auto", f"osc_persist@{fp}")]
        arms += [(f"persist@{t}", f"osc_persist@{fp},tps={t}") for t in OSC32_PERSIST_TPS]
        return arms + [("band@auto", f"osc_band@{fp}")]
    osc = kernel == "osc"
    pre = "osc" if osc else "rk78"
    tps = OSC_PERSIST_TPS if osc else RK78_PERSIST_TPS
    arms = [("auto", "eagle_graph_auto" if osc else "rk78_auto"), ("fused_one", f"{pre}_fused")]
    arms += [("persist@auto", f"{pre}_persist")] + [(f"persist@{t}", f"{pre}_persist@tps={t}") for t in tps]
    arms += [(f"band@{e or 'auto'}", f"{pre}_band" + (f"@every={e}" if e else "")) for e in BAND_EVERY]
    return arms + [("warp", "warp_kernel" if osc else "rk78_warp")]


def compare_cells(stage, kernels=("osc", "rk78", "osc32"), ns=COMPARE_NS):
    """(kernel, N, S, dist) in run order, large N first; S = 0 and dist 'orbits' for the RK78 card; kernel
    ``osc32`` is the oscillator in float32 (precision=fp32)."""
    base = stage.removesuffix("-quick")
    quick = stage.endswith("-quick")
    if base == "crossover":
        return [("osc", n, 100, "spread") for n in sorted(CROSSOVER_NS, reverse=True)][-1:] if quick else \
            [("osc", n, 100, "spread") for n in sorted(CROSSOVER_NS, reverse=True)]
    if base == "band":
        cells = [("osc", n, BAND_STEPS, d) for n in sorted(BAND_NS, reverse=True) for d in BAND_DISTS]
        return [("osc", BAND_NS[0], BAND_STEPS, "uniform")] if quick else cells
    if quick:
        quick_cells = {"osc": ("osc", 100_000, 100, "spread"), "rk78": ("rk78", 10_000, 0, "orbits"),
                       "osc32": ("osc32", 100_000, 100, "spread")}
        return [quick_cells[k] for k in kernels]
    cells = []
    for n in sorted(ns, reverse=True):
        if "rk78" in kernels:
            cells.append(("rk78", n, 0, "orbits"))
        if "osc" in kernels:
            cells += [("osc", n, s, d) for s in COMPARE_OSC_STEPS for d in COMPARE_OSC_DISTS]
        if "osc32" in kernels and n in OSC32_NS:
            cells += [("osc32", n, s, d) for s in OSC32_STEPS for d in OSC32_DISTS]
    return cells


def stage_arms(stage, kernel):
    """(label, spec) of the arms a stage runs in each cell."""
    base = stage.removesuffix("-quick")
    if base == "crossover":
        return [("auto", "eagle_graph_auto"), ("fused_one", "osc_fused"), ("persist@auto", "osc_persist")]
    if base == "band":
        return [("auto", "eagle_graph_auto"), ("band@auto", "osc_band")] + [
            (f"band@{e}", f"osc_band@every={e}") for e in BAND_SWEEP_EVERY]
    return compare_arms(kernel)


def _planes(out):
    """A card's download (x, v, k tuple, or the RK78 dict) as a dict of host arrays."""
    return dict(zip(("x", "v", "k"), out)) if isinstance(out, (tuple, list)) else dict(out)


def judge(kernel, out, ref, inp, same_entry, truth=None, atol=None):
    """Guard one arm's final state against the reference's: (ok, bitwise, max_diff, why).
    Within the card's tolerance always (oscillator: stop steps exact, x and v within the card's ATOL of the
    reference; RK78: every sample at the final time and inside the stated truth bound); bit for bit as well
    when the arm ran the same kernel through the same launch mode as the reference."""
    import numpy as np

    out, ref = _planes(out), _planes(ref)
    keys = [k for k in ref if k in out]
    bitwise = all(np.array_equal(out[k], ref[k], equal_nan=True) for k in keys)
    state = ("x", "v") if kernel == "osc" else ("x", "y", "z", "vx", "vy", "vz")
    diff = max(float(np.max(np.abs(out[k] - ref[k]))) for k in state)
    if not np.isfinite(diff) or not all(np.isfinite(out[k]).all() for k in state):
        return False, bitwise, diff, "non-finite state"
    if kernel == "osc":
        if not np.array_equal(out["k"], inp["nstop"]):
            return False, bitwise, diff, "step counts differ from the stop steps"
        card_atol = sys.modules["perf_card"].ATOL if atol is None else atol
        if diff > card_atol:
            return False, bitwise, diff, f"state differs from the reference by {diff:.2e} > {card_atol:.0e}"
    else:
        rk = sys.modules["rk78_card"]
        if not np.all(out["t"] == rk.T_FINAL):
            return False, bitwise, diff, "a sample did not reach the final time"
        pos_t, _ = truth
        pos = np.stack([out["x"], out["y"], out["z"]])
        err = np.sqrt(((pos - pos_t) ** 2).sum(axis=0))
        over = err > rk.truth_bound(inp, out["n_acc"])
        if over.any():
            return False, bitwise, diff, f"{int(over.sum())} samples exceed the truth bound (worst {err[over].max():.2e})"
    if same_entry and not bitwise:
        return False, bitwise, diff, "not bit for bit equal to the reference, though it ran the same entry"
    return True, bitwise, diff, ""


def read_fields(*objs):
    """The launch fields (:data:`FIELD_NAMES`) these objects expose by those exact names, as a dict;
    a name nothing exposes is absent (an eagle without it is not an error)."""
    found = {}
    for name in FIELD_NAMES:
        for obj in objs:
            try:
                value = getattr(obj, name)
            except Exception:  # noqa: BLE001 -- a property of an older eagle may raise
                continue
            if isinstance(value, (int, float, str, bool)):
                found[name] = value
                break
    return found


class ClockSampler:
    """Samples the timed GPU's SM clock through NVML while a block runs; ``median`` is the MHz the block
    ran at (None when NVML is not usable). Bounds are quoted at this clock, not the boost clock."""

    def __init__(self, period=0.01):
        self.samples, self.period, self._stop, self._thread, self._nv, self._h = [], period, threading.Event(), None, None, None
        try:
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                import pynvml
            pynvml.nvmlInit()
            sel = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
            self._h = pynvml.nvmlDeviceGetHandleByIndex(int(sel) if sel.isdigit() else 0)
            self._nv = pynvml
        except Exception:  # noqa: BLE001 -- no pynvml or no driver: the clock is simply not recorded
            self._h = None

    def _read(self):
        try:
            self.samples.append(self._nv.nvmlDeviceGetClockInfo(self._h, self._nv.NVML_CLOCK_SM))
        except Exception:  # noqa: BLE001
            pass

    def _poll(self):
        while not self._stop.wait(self.period):
            self._read()

    def __enter__(self):
        if self._h is not None:
            self._read()
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *exc):
        if self._thread is not None:
            self._stop.set()
            self._thread.join()
            self._read()

    def median(self):
        s = sorted(self.samples)
        return s[len(s) // 2] if s else None

    def max_mhz(self):
        try:
            return self._nv.nvmlDeviceGetMaxClockInfo(self._h, self._nv.NVML_CLOCK_SM) if self._h else None
        except Exception:  # noqa: BLE001
            return None


def _timer_kind(spec):
    base = parse_spec(spec)[0]
    return "warp" if base in _WARP_SPECS else "wall" if base in _WALL_SPECS else "event"


def timed_ms(kind, run):
    """One run of ``run`` in milliseconds: CUDA events around it (Warp: Warp's own events on its stream;
    other frameworks, which run on streams of their own: host clock between two device syncs)."""
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()
    if kind == "warp":
        try:
            import warp as wp

            start, stop = wp.Event(enable_timing=True), wp.Event(enable_timing=True)
            wp.record_event(start)
            run()
            wp.record_event(stop)
            return float(wp.get_event_elapsed_time(start, stop))
        except AttributeError:
            kind = "wall"
    if kind == "wall":
        t0 = time.perf_counter()
        run()
        cp.cuda.runtime.deviceSynchronize()
        return (time.perf_counter() - t0) * 1e3
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    start.record()
    run()
    stop.record()
    stop.synchronize()
    return cp.cuda.get_elapsed_time(start, stop)


def _corrupt(label, out):
    """Test hook: PROFILE_TEST_CORRUPT=<arm label> perturbs that arm's downloaded state (the guard's plant)."""
    if os.environ.get("PROFILE_TEST_CORRUPT") == label:
        out = _planes(out)
        out["x"] = out["x"].copy()
        out["x"][0] += 1e-6
    return out


def measure_arm(label, spec, kernel, n, steps, dist, judge_out):
    """Build, warm, guard and (if the guard passes) time one arm. Returns its record: status ok / wrong /
    skipped / unavailable (the reason in ``note``), median and best ms, the report's mode, the persist threads
    per SM launched and its lane utilisation. ``judge_out(out, same_key)`` returns the guard's verdict."""
    import cupy as cp

    base, opts = parse_spec(spec)
    tps = int(opts["tps"]) if "tps" in opts else None
    rec = dict(label=label, spec=spec, status="ok", note="", median=None, best=None, mode="-", launched="-",
               util=None, bitwise=None, diff=None, sm_mhz=None, modes="", fields={})
    arm = None
    clock = ClockSampler()
    try:
        with persist_threads(tps), clock:
            arm, inp = build_arm(spec, n, steps, dist)
            runs = []
            for _ in range(SWEEP_WARM):
                arm["state"].upload(inp)
                arm["run"]()
                runs.append(str(getattr(arm.get("report"), "mode", "-")))
            report = arm.get("report")
            rec["mode"] = "-" if report is None else str(getattr(report, "mode", "?"))
            rec["util"] = getattr(report, "lane_utilisation", None)
            if base.endswith("persist"):
                rec["launched"] = launched_threads_per_sm(arm["state"].runner)
            if tps is not None and isinstance(rec["launched"], int) and rec["launched"] < tps:
                rec.update(status="skipped", note=f"{tps} threads per SM is above the residency limit "
                                                  f"(launched {rec['launched']})")
                return rec
            out = _corrupt(label, arm["download"]())
            ok, bitwise, diff, why = judge_out(out, (base, rec["mode"]))
            rec.update(bitwise=bitwise, diff=diff)
            if not ok:
                rec.update(status="wrong", note=why)
                return rec
            kind, walls = _timer_kind(spec), []
            for _ in range(SWEEP_REPS):
                arm["state"].upload(inp)
                walls.append(timed_ms(kind, arm["run"]))
                runs.append(str(getattr(arm.get("report"), "mode", "-")))
            report = arm.get("report")
            rec["fields"] = read_fields(getattr(arm["state"], "runner", None), report)
            rec["util"] = getattr(report, "lane_utilisation", rec["util"])
            walls.sort()
            rec.update(median=walls[len(walls) // 2], best=walls[0])
        rec["modes"] = ">".join(runs)  # the mode of each run: the third is what eagle's probe settled on
        rec["sm_mhz"] = clock.median()
        rec["sm_max_mhz"] = clock.max_mhz()
    except Exception as exc:  # noqa: BLE001 -- one refused arm must not end the matrix
        rec.update(status="unavailable", note=f"{type(exc).__name__}: {str(exc)[:140]}")
    finally:
        del arm
        cp.get_default_memory_pool().free_all_blocks()
    return rec


_FAMILY = {"eagle_graph_auto": "osc-auto", "osc_fused": "osc-plain", "osc_persist": "osc-plain", "osc_band": "osc-band",
           "rk78_auto": "rk78-plain", "rk78_fused": "rk78-plain", "rk78_persist": "rk78-plain", "rk78_band": "rk78-active"}


def _entry_family(spec):
    """The kernel an arm launches: arms of one family and one launch mode must agree bit for bit."""
    base = parse_spec(spec)[0]
    return _FAMILY.get(base, base)


def run_cell(kernel, n, steps, dist, extra_arms=(), arms=None):
    """Every arm of one cell. The reference is eagle auto (else the first arm that ran). Returns the cell dict."""
    fp32 = kernel == "osc32"
    inp = make_inputs(load_card(), n, steps, dist) if kernel in ("osc", "osc32") else load_rk78()._inputs(n)
    if fp32:
        inp = {k: v.astype("float32") for k, v in inp.items()}
    atol = load_card().FP32_ATOL if fp32 else None
    truth = load_rk78().kepler_truth(inp, load_rk78().T_FINAL) if kernel == "rk78" else None
    ref = {}

    def make_judge(spec):
        def judge_out(out, key):
            if "out" not in ref:
                ref.update(out=out, key=(_entry_family(spec), key[1]), spec=spec)
                return True, True, 0.0, ""
            same = (_entry_family(spec), key[1]) == ref["key"]
            return judge("osc" if fp32 else kernel, out, ref["out"], inp, same, truth, atol)
        return judge_out

    recs = []
    for label, spec in [*(arms or compare_arms(kernel)), *extra_arms]:
        rec = measure_arm(label, spec, kernel, n, steps, dist, make_judge(spec))
        recs.append(rec)
        say(f"compare {kernel} N={n} S={steps} {dist} {label}: {rec['status']} "
            f"{'' if rec['median'] is None else format(rec['median'], '.3f') + ' ms '}{rec['mode']} {rec['note']}")
    return dict(kernel=kernel, n=n, steps=steps, dist=dist, arms=recs, reference=ref.get("spec"))


def _eagle_ok(cell):
    return [r for r in cell["arms"] if r["status"] == "ok" and r["label"] != "warp" and r["median"] is not None
            and r["spec"] not in CONTEXT_ARMS]


def cell_ratios(cell):
    """(auto ms, warp ms, fastest eagle record, auto/warp, best/warp); None where an arm has no time."""
    by = {r["label"]: r for r in cell["arms"]}
    auto = by.get("auto", {}).get("median")
    warp = by.get("warp", {}).get("median")
    eagle = _eagle_ok(cell)
    best = min(eagle, key=lambda r: r["median"]) if eagle else None
    return (auto, warp, best, auto / warp if auto and warp else None,
            best["median"] / warp if best and warp else None)


def _cell_name(kernel, n, steps, dist):
    return f"{kernel} N={n:.0e}".replace("e+0", "e").replace("e+", "e") + f" S={steps or '-'} {dist}"


def render_compare(cells, unreached=()):
    """(text, csv) of the matrix so far: the per-cell wall table, the notes, and the summary that lists every
    cell where eagle auto is slower than Warp and every cell where auto did not pick the fastest eagle path."""
    def cell_id(c):
        return _cell_name(c["kernel"], c["n"], c["steps"], c["dist"])

    def fmt(rec):
        if rec is None:
            return "-"
        return {"ok": f"{rec['median']:.3f}" if rec["median"] is not None else "?", "wrong": "WRONG",
                "skipped": "skip"}.get(rec["status"], "n/a")

    text = ["Wall = CUDA events around one run (median of 3 after 2 warm runs), ms. WRONG = the guard flagged the arm "
            "(not timed); skip = persist size above the residency limit; n/a = this eagle cannot build or run the arm.",
            "band = hawk.steps(kernel, %d) compacted every `every` steps (auto = eagle's default) for the oscillator, the "
            "single-attempt kernel for RK78; persist@T = T threads per SM." % BAND_STEPS_PER_LAUNCH,
            "Every row carries precision=fp64 or precision=fp32 (osc32: the oscillator in float32, same model; eagle "
            "arms only, no Warp arm; guard = float32 tolerance against its own auto).", ""]
    csv = ["kernel,n,steps,dist,arm,status,median_ms,best_ms,mode,threads_per_sm_launched,lane_utilisation,"
           "bitwise_vs_auto,max_diff_vs_auto,sm_mhz,modes_per_run,launch_fields,precision,note"]
    notes, slower, not_fastest = [], [], []
    for kernel in ("osc", "rk78", "osc32"):
        mine = [c for c in cells if c["kernel"] == kernel]
        if not mine:
            continue
        labels = [r["label"] for r in mine[0]["arms"]]
        extra = sorted({r["label"] for c in mine for r in c["arms"]} - set(labels))
        labels += extra
        text.append(f"{'cell':32s} {'precision':>14s} " + " ".join(f"{lab:>12s}" for lab in labels) + f" {'auto mode':>10s} {'fastest eagle':>14s} "
                    f"{'auto/warp':>9s} {'best/warp':>9s}")
        for c in mine:
            by = {r["label"]: r for r in c["arms"]}
            auto, warp, best, a_w, b_w = cell_ratios(c)
            text.append(f"{cell_id(c):32s} {'precision=' + precision_of(kernel):>14s} " + " ".join(f"{fmt(by.get(lab)):>12s}" for lab in labels)
                        + f" {by.get('auto', {}).get('mode', '-'):>10s} {best['label'] if best else '-':>14s} "
                        f"{'-' if a_w is None else format(a_w, '.2f'):>9s} {'-' if b_w is None else format(b_w, '.2f'):>9s}")
            for r in c["arms"]:
                csv.append(",".join(str(v) for v in (
                    c["kernel"], c["n"], c["steps"], c["dist"], r["label"], r["status"],
                    "" if r["median"] is None else f"{r['median']:.4f}", "" if r["best"] is None else f"{r['best']:.4f}",
                    r["mode"], r["launched"], "" if r["util"] is None else f"{r['util']:.3f}",
                    "" if r["bitwise"] is None else int(r["bitwise"]), "" if r["diff"] is None else f"{r['diff']:.3e}",
                    r.get("sm_mhz") or "", r.get("modes", ""),
                    ";".join(f"{k}={v}" for k, v in r.get("fields", {}).items()), precision_of(kernel),
                    '"' + r["note"].replace('"', "'") + '"')))
                if r["note"]:
                    notes.append(f"  {cell_id(c)} {r['label']}: {r['status']}: {r['note']}")
                shown = {k: v for k, v in r.get("fields", {}).items() if k in ("T", "c", "f", "R", "k", "text_bytes")}
                if shown:
                    notes.append(f"  {cell_id(c)} {r['label']}: launch fields " + " ".join(f"{k}={v}" for k, v in shown.items()))
            if a_w is not None and a_w > 1.0:
                slower.append(f"  {cell_id(c)}: auto {auto:.3f} ms ({by['auto']['mode']}) vs Warp {warp:.3f} ms = {a_w:.2f}x")
            if auto is not None and best is not None and auto > best["median"] * (1 + AUTO_TOLERANCE):
                not_fastest.append(f"  {cell_id(c)}: auto {auto:.3f} ms ({by['auto']['mode']}); fastest {best['label']} "
                                   f"{best['median']:.3f} ms ({best['mode']}), auto is {auto / best['median']:.2f}x")
            elif auto is None and best is not None:
                not_fastest.append(f"  {cell_id(c)}: auto has no time ({by.get('auto', {}).get('status', 'missing')}); "
                                   f"fastest {best['label']} {best['median']:.3f} ms")
        text.append("")
    ctx = [c for c in cells if any(r["spec"] in CONTEXT_ARMS for r in c["arms"])]
    for c in ctx:
        text.append(f"context ({cell_id(c)}), other frameworks, host clock between syncs: "
                    + ", ".join(f"{r['spec']} {fmt(r)}" for r in c["arms"] if r["spec"] in CONTEXT_ARMS))
    clocks = sorted(r["sm_mhz"] for c in cells for r in c["arms"] if r.get("sm_mhz"))
    peak = max((r.get("sm_max_mhz") or 0 for c in cells for r in c["arms"]), default=0)
    text += ["", "-- SM clock during the runs (NVML median per arm; quote bounds at it, not at the boost clock): "
             + (f"median {clocks[len(clocks) // 2]} MHz, lowest {clocks[0]}, highest {clocks[-1]}"
                + (f", device maximum {peak}" if peak else "") if clocks else "unavailable (no NVML)")]
    text += ["", "-- notes", *(notes or ["  none"]), "",
             f"-- cells where eagle auto is slower than Warp ({len(slower)})", *(slower or ["  none"]), "",
             f"-- cells where auto did not pick the fastest eagle path (more than {AUTO_TOLERANCE:.0%} slower; {len(not_fastest)})",
             *(not_fastest or ["  none"])]
    if unreached:
        text += ["", f"-- cells not reached ({len(unreached)}; the stage budget ended first)",
                 *(f"  {_cell_name(*cell)}" for cell in unreached)]
    return "\n".join(text) + "\n", "\n".join(csv) + "\n"


def band_fit(cells):
    """R per uniform band cell: wall = A + C * (S / every) over the swept cadences, so C is one compaction group's
    cost and A / (S / K) one full-live step launch's (K steps); R = C / that. Lines of text."""
    import numpy as np

    lines = []
    for c in cells:
        pts = [(c["steps"] / int(r["label"].split("@")[1]), r["median"]) for r in c["arms"]
               if r["label"].startswith("band@") and r["label"] != "band@auto" and r["median"] is not None]
        auto = {r["label"]: r for r in c["arms"]}
        pick = auto.get("band@auto", {}).get("fields", {}).get("every")
        head = f"osc N={c['n']:.0e} S={c['steps']} {c['dist']}".replace("e+0", "e")
        if len(pts) >= 3 and c["dist"] == "uniform":
            x, y = np.array(pts).T
            slope, icept = np.polyfit(x, y, 1)
            launch = icept * BAND_STEPS_PER_LAUNCH / c["steps"]
            lines.append(f"  {head}: compaction group C = {slope:.4f} ms, step launch (K={BAND_STEPS_PER_LAUNCH}, "
                         f"full live) = {launch:.4f} ms, R = C / launch = {slope / launch:.2f}"
                         if launch > 0 and slope > 0 else f"  {head}: the cadence fit gave no positive C and launch "
                         f"(slope {slope:.4g}, intercept {icept:.4g}): R is not measurable here")
        best = min(pts, key=lambda p: p[1]) if pts else None
        lines.append(f"  {head}: eagle's own band pick every={pick if pick is not None else 'unavailable in this eagle'}"
                     + (f"; fastest swept cadence every={int(c['steps'] / best[0])} ({best[1]:.3f} ms)" if best else ""))
    return lines or ["  no band cell ran"]


def crossover_lines(cells):
    """fused_one against persist per cell, and the entry auto settled on (its mode after each run)."""
    lines = []
    for c in cells:
        by = {r["label"]: r for r in c["arms"]}
        f, p, a = (by.get(k, {}) for k in ("fused_one", "persist@auto", "auto"))
        if f.get("median") and p.get("median"):
            verdict = "persist" if p["median"] < f["median"] else "fused_one"
            lines.append(f"  N={_cell_name('osc', c['n'], 0, '').split()[1][2:]}: fused_one {f['median']:.3f} ms, persist {p['median']:.3f} ms "
                         f"(faster: {verdict}, {max(f['median'], p['median']) / min(f['median'], p['median']):.2f}x); "
                         f"auto settled on {(a.get('modes', '').split('>') + ['?', '?', '?'])[2]} after 3 runs "
                         f"(modes per run {a.get('modes', '?')})")
    return lines or ["  no crossover cell ran"]


def compare_stage(out, stage, kernels=("osc", "rk78", "osc32"), ns=COMPARE_NS, budget_s=None):
    """Run every cell; the table and CSV are rewritten after each, so a stage cut short keeps what it finished,
    and no cell starts once the budget is spent. A stage in which no arm of any cell ran fails."""
    t0, cells = time.monotonic(), []
    todo = compare_cells(stage, kernels, ns)
    txt, csv = out / f"{stage.replace('-', '_')}.txt", out / f"{stage.replace('-', '_')}.csv"
    for i, (kernel, n, steps, dist) in enumerate(todo):
        if budget_s is not None and time.monotonic() - t0 > budget_s:
            todo = todo[i:]
            break
        extra = ()
        if (kernel, n, steps, dist) == ("osc", *CONTEXT_POINT) and (
                budget_s is None or time.monotonic() - t0 < CONTEXT_BUDGET_SHARE * budget_s):
            extra = tuple((a, a) for a in CONTEXT_ARMS)
        cells.append(run_cell(kernel, n, steps, dist, extra, stage_arms(stage, kernel)))
        text, table = render_compare(cells)
        txt.write_text(text)
        csv.write_text(table)
    else:
        todo = []
    text, table = render_compare(cells, todo)
    if stage.startswith("band"):
        text += "\n-- band cost model\n" + "\n".join(band_fit(cells)) + "\n"
    if stage.startswith("crossover"):
        text += "\n-- crossover\n" + "\n".join(crossover_lines(cells)) + "\n"
    txt.write_text(text)
    csv.write_text(table)
    say("\n" + text.split("-- notes")[-1])
    if not any(r["status"] == "ok" for c in cells for r in c["arms"]):
        raise RuntimeError(f"no arm of any cell of {stage} ran; see {txt.name}")


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
    if stage.startswith("ncu-tsweep"):
        caps = [(f"osc_persist_t{t}_spread_s{steps}", f"osc_persist@tps={t}", r"regex:oscillator_step_xauto_persist",
                 (SWEEP_N, steps, "spread"), RATIO_METRICS) for steps in TSWEEP_STEPS for t in OSC_PERSIST_TPS]
        caps += [(f"rk78_persist_t{t}", f"rk78_persist@tps={t}", r"regex:rkf78_attempt_persist", RK78_POINT,
                  RATIO_METRICS) for t in RK78_PERSIST_TPS]
        caps += [(f"osc32_persist_t{t}_spread_s{TSWEEP_FP32_STEPS}", f"osc_persist@fp=32,tps={t}",
                  r"regex:oscillator_step_xauto_persist", (SWEEP_N, TSWEEP_FP32_STEPS, "spread"), RATIO_METRICS_FP32)
                 for t in OSC_PERSIST_TPS]
        return caps[:1] if stage.endswith("-quick") else caps
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
        worker_line = ""
        try:
            done = subprocess.run(ncu_command(ncu, rep, regex, arm, point, sudo, metrics), capture_output=True,
                                  text=True, timeout=NCU_KERNEL_CAP_S)
            code, tail = done.returncode, (done.stdout + done.stderr)[-1500:]
            worker_line = next((ln for ln in done.stdout.splitlines() if "worker arm=" in ln), "")
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
                digest += "\n" + (f"  lane_efficiency: not computed for FP32 (the count above is the FP64 pipe's; read "
                                   f"{RATIO_METRICS_FP32[-2]})" if precision_of(arm) == "fp32"
                                   else lane_efficiency_lines(raw.stdout, worker_line))
            summary.append(f"== {label} ({arm}, precision={precision_of(arm)}, {regex}, N={point[0]} S={point[1]} {point[2]}, "
                           f"{time.perf_counter() - t0:.0f} s)\n  {worker_line[worker_line.find('sample_steps'):]}\n{digest}\n")
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


def _ncu_values(raw_csv, names):
    """{metric: float} from ncu's raw page for the metrics it holds (a missing or unreadable one is absent)."""
    import csv

    rows = list(csv.reader(raw_csv.splitlines()))
    if len(rows) < 2:
        return {}
    header = rows[0]
    values = next((r for r in rows[1:] if r and r[0].strip().isdigit()), None)
    found = {}
    for name in names:
        if values and name in header and header.index(name) < len(values):
            try:
                found[name] = float(values[header.index(name)].replace(",", ""))
            except ValueError:
                pass
    return found


def lane_efficiency_lines(raw_csv, worker_line):
    """Lane efficiency from ncu, never from the slot counter: sample-steps x FP64 instructions per step, over
    32 x the FP64 warp-instructions ncu counted; and those warp-instructions per 32 sample-steps (ideal: ops per step)."""
    got = dict(tok.split("=", 1) for tok in worker_line.split() if "=" in tok)
    fp64 = _ncu_values(raw_csv, ("sm__inst_executed_pipe_fp64.sum",)).get("sm__inst_executed_pipe_fp64.sum")
    try:
        steps, ops = float(got["sample_steps"]), float(got["ops_per_step"])
    except (KeyError, ValueError):
        return "  lane_efficiency: unavailable (no sample-step accounting from the worker)"
    if not fp64:
        return "  lane_efficiency: unavailable (sm__inst_executed_pipe_fp64.sum missing from the report)"
    return (f"  lane_efficiency = sample_steps x ops_per_step / (32 x fp64_warp_inst) = {steps * ops / (32 * fp64):.3f} "
            f"(ops_per_step {ops:g} from {got.get('ops_source', '?')})\n"
            f"  fp64_warp_inst_per_32_sample_steps = {fp64 * 32 / steps:.1f} (ideal {ops:g})")


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
    for metric in RATIO_METRICS_FP32:
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
          "ncu-tsweep", "compare", "crossover", "band", "cadence-sweep", "validate-ncu",
          *(f"{s}-quick" for s in ("ncu-ratio", "ncu-band", "ncu-tsweep", "compare", "crossover", "band",
                                   "cadence-sweep")))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("profile-out"))
    ap.add_argument("--target", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--probe-only", action="store_true", help="the probe stage alone")
    ap.add_argument("--stage", choices=STAGES, help="run one stage (profile.sh runs them in order)")
    ap.add_argument("--nsys", help="path of nsys")
    ap.add_argument("--ncu", help="path of ncu")
    ap.add_argument("--ncu-sudo", action="store_true", help="run ncu under `sudo -n` (counters are admin-only)")
    ap.add_argument("--kernels", default="osc,rk78,osc32",
                    help="compare: kernels to run (osc, rk78, osc32 = the oscillator in float32)")
    ap.add_argument("--ns", default=",".join(str(n) for n in COMPARE_NS), help="compare: sample counts")
    ap.add_argument("--budget-s", type=float, help="compare: start no cell after this many seconds")
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
        elif stage.removesuffix("-quick") in ("compare", "crossover", "band"):
            if args.target == "cpu":
                raise Skip("this stage times the device")
            compare_stage(args.out, stage, tuple(args.kernels.split(",")),
                          tuple(int(float(x)) for x in args.ns.split(",")), args.budget_s)
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
