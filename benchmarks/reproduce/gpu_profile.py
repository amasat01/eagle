#!/usr/bin/env python
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Profile eagle's kernels and launch shapes on this machine's GPU, for the performance card's workload.

The workload, its inputs, its arms and the FMA probe come from ``perf_card/perf_card.py`` (loaded, never copied).
Stages (``--stage``; ``profile.sh`` runs them in order); each writes plain text or CSV into ``--out``:

* **sol_probe**: the FP64 speed of light of this device (``sol_probe.txt``): the card's register-only FMA chain and the
  clocks;
* **wide_sweep**: hawk's wide-value sweep (``python -m hawk.targets.cards.sweep``) over the forms thread, team and
  staged: candidate rows for ``wide.json`` and ``thread_box.json`` in ``wide_sweep.json``;
* **ncu_corpus**: Nsight Compute on the sweep's ties and on the team and staged kernels (``ncu_corpus.csv``);
* **ncu_ensemble**: Nsight Compute on every eagle arm of the card at N = 1e6, spread and uniform: lane efficiency, FP64 pipe,
  issue, occupancy, registers, DRAM, stall classes, and the DP instructions executed per attempt (``ncu_ensemble.csv``);
* **ncu_features**: the same capture of every arm of ours (eagle / hawk code, never a baseline library) of every
  feature-card row, case and dtype; the kernels of one run, up to :data:`LAUNCH_CAP` (``ncu_features.csv``);
* **ncu_nvlibs**: counters only, never ``--set full``, never a source or SASS page (the CUDA EULA bars disassembling
  NVIDIA's libraries): the NVIDIA library kernels behind the feature rows eagle loses (:data:`NVLIB_CELLS`), through
  :data:`NVLIB_SECTIONS` and ``--import-source no`` (``ncu_nvlibs.csv``);
* **chunk_residency**: the persist launch over chunk x threads per SM x {f32, f64} at N = 1e6 and 1e7, spread and
  spread-short: wall and slot fill (``chunk_residency.csv``); Nsight Compute on a tie (``chunk_residency_ncu.csv``);
* **launch_shape**: launches and kernels per run for each card arm (Nsight Systems, one run per arm), and the fused
  against persist wall over N = 1e3..1e6 (``launch_shape.csv``, ``crossover.csv``);
* **band**: the compaction cadence of the compact arm (``band.csv``); run by hand, on starved cards.

Nsight Compute captures use ``--set full`` (``--quick``: the listed metrics only); each ``.ncu-rep`` is kept. Nsight
Compute needs compute capability 7.0 or newer and counter access. Run through ``profile.sh``, which sets up
the environment and packs the tarball.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import importlib
import importlib.util
import json
import os
import pathlib
import sqlite3
import statistics
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
CARD_PATH = HERE.parent / "perf_card" / "perf_card.py"
FEATURE_PATH = HERE.parent / "perf_card" / "feature_card.py"
NCU_MIN_CC = 7.0
NCU_CAP_S = 150                               # one --quick capture
NCU_FULL_CAP_S = 600                          # one --set full capture: ~40 replay passes of every kernel, built and run in it
LAUNCH_CAP = 32                               # kernels captured of an arm that launches many (a run's first 32)
TIE = 0.10                                    # within this fraction of the best is a tie
#: the closed NVIDIA library kernels behind the feature rows eagle loses: (row, case, dtype, arm)
NVLIB_CELLS = (("contraction", "all", "float64", "einsum"), ("contraction", "all", "float32", "einsum"),
               ("linalg_team", "n512", "float64", "cupy"), ("linalg_team", "n512", "float64", "torch"),
               *(("linalg_lane", "n32", d, a) for d in ("float64", "float32") for a in ("cupy", "torch")),
               ("fft", "all", "float64", "cupy_fft"), ("sample_sum", "all", "float64", "cupy_sum"),
               ("scatter", "all", "float64", "cupyx_scatter_add"))
#: counters sections of those captures: none carries instruction text (no SourceCounters, no InstructionStats)
NVLIB_SECTIONS = ("SpeedOfLight", "LaunchStats", "Occupancy", "ComputeWorkloadAnalysis", "MemoryWorkloadAnalysis",
                  "WarpStateStats", "SchedulerStats")
ENSEMBLE_ARMS = ("eagle_graph", "eagle_graph_compact", "eagle_graph_reorder", "eagle_graph_persist", "eagle_simulate")  # the card's eagle arms
ENSEMBLE_QUICK_ARM = "eagle_graph_persist"    # the smoke's arm: one plain launch, which Nsight Compute can always capture
#: Nsight Compute cannot capture a kernel inside a CUDA graph that has conditional nodes (the graph arms' device-side loop):
#: such a row is labelled, not failed; the graph arms' launch shape comes from launch_shape (Nsight Systems).
NCU_CONDITIONAL_GRAPH = "graph which can have conditional nodes are not supported for profiling"
NOT_PROFILABLE = "not profilable: Nsight Compute cannot capture kernels in a CUDA graph with conditional nodes"
LAUNCH_ARMS = ("eagle_graph", "eagle_graph_persist", "eagle_simulate")
KERNEL = "regex:rkf78"                        # the card's attempt kernel (every eagle arm's)

CHUNKS, TPS = (1, 4, 16, 32), (None, "2x")
CHUNK_NS, CHUNK_DISTS, CHUNK_DTYPES = (1_000_000, 10_000_000), ("spread", "spread-short"), ("float32", "float64")
CROSSOVER_NS = (1_000, 10_000, 100_000, 1_000_000)
BAND_EVERY = (4, 8, 16, 32)
POINT = (1_000_000, "spread")                 # the ensemble's N and distribution; also the band and launch point

#: lane efficiency (threads per issued instruction), FP64 pipe, issue, occupancy, registers, DRAM and the stall classes
ENSEMBLE_METRICS = (
    "smsp__thread_inst_executed_per_inst_executed.ratio",
    "sm__inst_executed_pipe_fp64.avg.pct_of_peak_sustained_active",
    "sm__inst_issued.avg.pct_of_peak_sustained_active",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "launch__registers_per_thread",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    *(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio"
      for s in ("long_scoreboard", "short_scoreboard", "wait", "math_pipe_throttle", "no_instruction")),
)
#: dynamic double-precision instructions executed (thread level)
DP_METRICS = tuple(f"smsp__sass_thread_inst_executed_op_{op}_pred_on.sum" for op in ("dfma", "dmul", "dadd"))
CORPUS_METRICS = (
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__pipe_fp64_cycles_active.avg.pct_of_peak_sustained_active",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
    "sm__inst_issued.avg.pct_of_peak_sustained_active",
    "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
)


class Skip(Exception):
    """A stage that has nothing to do here (reason in the message)."""


def say(msg):
    print(f"[profile] {msg}", flush=True)


def load_card(path=CARD_PATH):
    """The performance card's module (``path``: the feature card's): its kernel, inputs and arms."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def compute_capability():
    sel = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or "0"
    try:
        done = subprocess.run(["nvidia-smi", "-i", sel, "--query-gpu=compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30)
        return float(done.stdout.split()[0])
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def emit(path, row):
    """Append ``row`` to the CSV ``path`` (the header with the first row), flushed: a stage cut short keeps its rows."""
    new = not path.exists()
    with path.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------- #
# Persist launch geometry (a measurement tool: eagle has no hook for the threads per SM)
# --------------------------------------------------------------------------- #
_OVERRIDE_LOCK = threading.Lock()


@contextlib.contextmanager
def persist_threads(tps):
    """Inside the block a persist launch uses ``tps`` threads per SM instead of eagle's rule (``None``: the rule;
    ``"2x"``: twice what the rule picks). The sizing function the launch calls is replaced for the block and
    restored on exit; a second block entered meanwhile raises instead of racing."""
    if tps is None:
        yield
        return
    policy = importlib.import_module("eagle._launch_policy")
    original = policy.persistent_threads_per_sm
    if not _OVERRIDE_LOCK.acquire(blocking=False):
        raise RuntimeError("persist_threads is already active in this process: run one cell at a time")

    def request(*args, **kwargs):
        return 2 * original(*args, **kwargs) if tps == "2x" else tps

    policy.persistent_threads_per_sm = request
    try:
        yield
    finally:
        policy.persistent_threads_per_sm = original
        _OVERRIDE_LOCK.release()


def launched_threads_per_sm(runner):
    """Threads per SM of the runner's persist launch (block x grid / SMs), or None when this eagle does not say."""
    try:
        sms = importlib.import_module("eagle.launch")._device_props()
        sms = sms["multiProcessorCount"] if isinstance(sms, dict) else sms.multiProcessorCount
        return int(runner.launch_policy["block"] * runner.launch_policy["grid"]) // sms
    except Exception:  # noqa: BLE001 -- the geometry differs across eagles
        return None


# --------------------------------------------------------------------------- #
# One cell: build an arm of the card (or a variant of its persist/fused/compact launch), time it, or profile one run
# --------------------------------------------------------------------------- #
def build(card, cell):
    """(arm, host inputs) of ``cell`` = {arm, n, dist, dtype, chunk, tps, every}. The card's own arms are built by the
    card; ``persist`` (pinned chunk), ``fused`` and ``band`` (cadence ``every``) are the same kernel launched otherwise."""
    import eagle
    import eagle.plan  # noqa: F401  (eagle.simulate first in a process imports it circularly)

    n, name = cell["n"], cell["arm"]
    inp = card._inputs(n, card._set_cell(cell["dist"]))
    if name in ENSEMBLE_ARMS:
        return card._make_arms(n, [name])[name], inp
    f32 = cell.get("dtype") == "float32"
    st = card._GpuArm(n)
    if f32:  # the same model with float32 planes and scalar type
        st.planes = {k: v.astype("float32") if v.dtype == "float64" else v for k, v in st.planes.items()}
        inp = {k: v.astype("float32") for k, v in inp.items()}
    extra = {"persist": {"_fast_mode": "persist", "_slot_fill": True, "_chunk": cell.get("chunk", 32)},
             "fused": {"_fast_mode": "fused_one"},
             "band": {"_fast_mode": False, "every": cell.get("every", card.COMPACT_EVERY)}}[name]
    kernel = (card._define_active_kernel if name == "band" else card._define_kernel)()
    step = eagle.deploy(kernel, scalar_type="float32") if f32 else eagle.deploy(kernel)
    st.runner = eagle.until_done(step, max_steps=card.MAX_ATTEMPTS, t_final=card.T_FINAL,
                                 terminated=st.terminated, **st.planes, **extra)
    arm = {"state": st, "download": st.download, "report": None}

    def run():
        report = st.runner.run()
        arm["report"] = getattr(report, "report", report)

    arm["run"] = run
    return arm, inp


def run_cell(card, cell, reps=3, profile=False):
    """The cell's row: median CUDA-synchronised wall of ``reps`` runs after a warm one (``profile``: one run between
    ``cudaProfilerStart/Stop``, so a profiler with a capture range sees it alone), the launch the report names, the slot
    fill, the threads per SM that launched and the attempts the run made."""
    import cupy as cp

    sync = "eagle_graph"
    with persist_threads(cell.get("tps")):
        arm, inp = build(card, cell)
        arm["state"].upload(inp)
        arm["run"]()
        card._sync_all(sync)
        walls = []
        for _ in range(1 if profile else reps):
            arm["state"].upload(inp)
            card._sync_all(sync)
            if profile:
                cp.cuda.runtime.profilerStart()
            t0 = time.perf_counter()
            arm["run"]()
            card._sync_all(sync)
            walls.append((time.perf_counter() - t0) * 1e3)
            if profile:
                cp.cuda.runtime.profilerStop()
        report = arm.get("report")
        runner = getattr(arm["state"], "runner", None)
        row = {**cell, "wall_ms": round(statistics.median(walls), 3),
               "mode": getattr(report, "mode", None), "launches": getattr(report, "launches", None),
               "slot_fill": getattr(report, "slot_fill", None),
               "launched_tps": launched_threads_per_sm(runner) if runner is not None else None,
               "sm_mhz": card.cc.nvml_sm_clock_mhz(cp.cuda.Device().pci_bus_id)}
        if profile:
            out = arm["download"]()
            row["sample_steps"] = float((out["n_acc"] + out["n_rej"]).sum())
    del arm, inp
    gc.collect()
    card.cc.trim_pools()
    return row


def timed_cell(card, cell, reps):
    """:func:`run_cell`, a failure (out of memory included) recorded in the row and never retried."""
    try:
        return run_cell(card, cell, reps)
    except Exception as exc:  # noqa: BLE001 -- the row says what happened
        gc.collect()
        card.cc.trim_pools()
        return {**cell, "error": f"{type(exc).__name__}: {str(exc)[:120]}"}


def worker(card, cell):
    print("worker " + json.dumps(run_cell(card, cell, profile=True)), flush=True)


def worker_sweep(form, width, fold, dtype, cache):
    """One cell of hawk's wide sweep, built and launched (what Nsight Compute captures)."""
    from hawk.targets.cards import sweep

    sweep.measure([(form, int(width), int(fold), dtype)], pathlib.Path(cache), 1)


# --------------------------------------------------------------------------- #
# Nsight Compute and Nsight Systems
# --------------------------------------------------------------------------- #
NCU_TIME = "gpu__time_duration.sum"
_UNIT_S = {"nsecond": 1e-9, "usecond": 1e-6, "msecond": 1e-3, "second": 1.0}


def ncu_kernels(raw_csv):
    """One dict per kernel of ncu's raw page (header, a units row, then a row per kernel): every numeric column by name,
    ``name`` and ``gpu_time_s`` (the duration in seconds, when the page has it)."""
    rows = list(csv.reader(raw_csv.splitlines()))
    if len(rows) < 2:
        return []
    out = []
    for row in (r for r in rows[1:] if r and r[0].strip().isdigit()):
        kernel = {}
        for name, unit, value in zip(rows[0], rows[1], row):
            if name == "Kernel Name":
                kernel["name"] = value
            with contextlib.suppress(ValueError):
                kernel[name] = float(value.replace(",", ""))
                if name == NCU_TIME:
                    kernel["gpu_time_s"] = kernel[name] * _UNIT_S.get(unit, 1e-9)
        out.append(kernel)
    return out


def ncu_values(raw_csv):
    """{metric: float} of the first kernel of ncu's raw page."""
    kernels = ncu_kernels(raw_csv)
    return {k: v for k, v in kernels[0].items() if isinstance(v, float)} if kernels else {}


def kernel_summary(kernels, metrics):
    """The row of an arm's captured kernels: how many, the summed GPU time, the top kernel by time and its ``metrics``."""
    top = max(kernels, key=lambda k: k.get("gpu_time_s", 0.0), default={})
    return {"kernels": len(kernels), "gpu_time_s": sum(k.get("gpu_time_s", 0.0) for k in kernels) if kernels else None,
            "top_kernel": top.get("name", "")[:120] or None, **{m: top.get(m) for m in metrics}}


def ncu_run(args, rep, cmd_line, metrics, regex=None, count=1, ranged=True, sections=()):
    """One capture of the first ``count`` launches (matching ``regex``, else any) of ``cmd_line``, inside the worker's
    profiler range when ``ranged``: ([kernel dicts], its output). Full mode adds ``--set full`` to ``metrics``;
    ``sections`` (counters only: the nvlibs) replace it and keep the source out of the report (``--import-source no``)."""
    full = not getattr(args, "quick", False) or bool(sections)
    cap = NCU_FULL_CAP_S if full else NCU_CAP_S
    cmd = [args.ncu, "--profile-from-start", "off" if ranged else "on", "--launch-skip", "0", "--launch-count", str(count),
           *(["--kernel-name", regex] if regex else []), "--target-processes", "all", "-f", "-o", str(rep),
           *(["--set", "full"] if full and not sections else []),
           *(a for sec in sections for a in ("--section", sec)), *(["--import-source", "no"] if sections else []),
           "--metrics", ",".join(metrics)] + cmd_line
    if args.ncu_sudo:
        keep = [f"{k}={os.environ[k]}" for k in ("PATH", "LD_LIBRARY_PATH") if k in os.environ]   # sudo drops both
        cmd = ["sudo", "-n", "-E", "env", *keep] + cmd
    try:
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=cap)
    except subprocess.TimeoutExpired:
        return [], f"ncu exceeded {cap} s"
    report = pathlib.Path(str(rep) + ".ncu-rep")
    if args.ncu_sudo and report.exists():
        subprocess.run(["sudo", "-n", "chown", f"{os.getuid()}:{os.getgid()}", str(report)])
    if done.returncode != 0 or not report.exists():
        return [], ncu_failure_note(done.stdout + done.stderr)
    raw = subprocess.run([args.ncu, "--import", str(report), "--csv", "--page", "raw"],
                         capture_output=True, text=True).stdout
    return ncu_kernels(raw), done.stdout


def ncu_capture(args, rep, regex, cmd_line, metrics, ranged=True):
    """One capture of the first launch matching ``regex`` of ``cmd_line``: ({metric: value}, its output)."""
    kernels, out = ncu_run(args, rep, cmd_line, metrics, regex, ranged=ranged)
    return ({k: v for k, v in kernels[0].items() if isinstance(v, float)} if kernels else {}), out


def ncu_failure_note(text):
    """ncu's own ==ERROR== / ==WARNING== lines when it printed any (a library warning printed later must not hide
    them) followed by the profiled application's last lines, else the output's last 300 characters."""
    own = [ln.strip() for ln in text.splitlines() if ln.lstrip().startswith(("==ERROR==", "==WARNING=="))]
    if not own:
        return text[-300:]
    app = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("==")]
    return "\n".join(own[-5:] + ["-- the application's last lines:"] + app[-6:])   # why the profiled process failed


def ncu_stage_verdict(notes):
    """A stage whose every capture failed is failed, with ncu's reason: never ok over a table of empty rows."""
    if notes and all(notes):
        lines = notes[0].splitlines()
        why = next((ln.strip() for ln in lines if "==ERROR==" in ln or "ERR_" in ln), "")
        why = why or next((ln.strip() for ln in lines if ln.strip()), "no output")
        if "ERR_NVGPUCTRPERM" in why:
            why = "the GPU performance counters are not permitted here (ERR_NVGPUCTRPERM)"
        raise RuntimeError(f"every ncu capture failed ({len(notes)}): {why}")


def worker_cmd(cell):
    return [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", json.dumps(cell)]


def worker_row(text):
    line = next((ln for ln in text.splitlines() if ln.startswith("worker ")), None)
    return json.loads(line[7:]) if line else {}


def require_ncu(args):
    if not args.ncu:
        raise Skip("ncu not found")
    cc_ = compute_capability()
    if cc_ is None:
        raise Skip("compute capability not readable (nvidia-smi)")
    if cc_ < NCU_MIN_CC:
        raise Skip(f"Nsight Compute needs compute capability {NCU_MIN_CC} or newer; this GPU is sm_{int(cc_ * 10)}")


def nsys_run(args, rep, cell):
    """((kernel count, kernel ms, API calls, kernel names), the worker's output) of one profiled run of ``cell`` under
    nsys; the first is None with a note when nsys or its export failed."""
    cmd = [args.nsys, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none", "--cpuctxsw=none",
           "--capture-range=cudaProfilerApi", "--capture-range-end=stop", "-f", "true", "-o", str(rep)] + worker_cmd(cell)
    done = subprocess.run(cmd, capture_output=True, text=True)
    report = pathlib.Path(str(rep) + ".nsys-rep")
    if done.returncode != 0 or not report.exists():
        return None, (done.stdout + done.stderr)[-300:]
    db = report.with_suffix(".sqlite")
    subprocess.run([args.nsys, "export", "--type", "sqlite", "-f", "true", "-o", str(db), str(report)],
                   capture_output=True, text=True)
    try:
        con = sqlite3.connect(db)
        k = con.execute("SELECT COUNT(*), COALESCE(SUM(end - start), 0) FROM CUPTI_ACTIVITY_KIND_KERNEL").fetchone()
        calls = con.execute("SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME").fetchone()[0]
        names = [r[0] for r in con.execute("SELECT DISTINCT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                                           "JOIN StringIds s ON k.demangledName = s.id")]
        con.close()
    except sqlite3.Error as exc:
        return None, f"nsys export unreadable: {exc}"
    finally:
        db.unlink(missing_ok=True)
    return (k[0], round(k[1] * 1e-6, 3), calls, names), done.stdout


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def sol_probe(args):
    card = load_card()
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    sms = props["multiProcessorCount"]
    dfma = card._fma_probe(sms)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    clocks = subprocess.run(["nvidia-smi", "-q", "-d", "CLOCK"], capture_output=True, text=True).stdout
    (args.out / "sol_probe.txt").write_text(
        f"device: {name}\ncompute_capability: {compute_capability()}\nsm_count: {sms}\n"
        f"dfma_per_s: {dfma:.4e}\nfp64_flops: {2 * dfma:.4e}\n"
        f"sm_mhz_after: {card.cc.nvml_sm_clock_mhz(cp.cuda.Device().pci_bus_id)}\n-- nvidia-smi -q -d CLOCK\n{clocks}")
    say(f"FP64 {2 * dfma:.3e} flop/s on {sms} SMs")


def wide_sweep(args):
    hawk_src = os.environ.get("HAWK_SRC")
    path = f"{hawk_src}{os.pathsep}{os.environ.get('PYTHONPATH', '')}" if hawk_src else os.environ.get("PYTHONPATH", "")
    cmd = [sys.executable, "-m", "hawk.targets.cards.sweep", "--out", str(args.out / "wide_sweep.json"),
           "--cache", str(args.out.parent / "sweep_cache")] + (["--quick"] if args.quick else [])
    done = subprocess.run(cmd, env={**os.environ, "PYTHONPATH": path}, cwd=hawk_src or None)
    if done.returncode != 0:
        raise RuntimeError(f"the hawk sweep exited {done.returncode}")


def corpus_cells(sweep_json, quick):
    """(form, width, fold, dtype) to capture: the forms tied at each tie of the sweep, then the team and staged kernels
    at the widest point."""
    won = json.loads(sweep_json.read_text())["winners"]
    cells = [(f, w["width"], w["fold"], w["dtype"]) for w in won if w["tie"] for f in w["tied"]]
    top = max(won, key=lambda w: (w["width"], w["fold"]))
    cells += [(f, top["width"], top["fold"], top["dtype"]) for f in ("team", "staged")]
    return list(dict.fromkeys(cells))[: 2 if quick else 12]


def ncu_corpus(args):
    require_ncu(args)
    sweep_json = next((p for p in (args.out / "wide_sweep.json", args.out.parent / "wide_sweep.json") if p.exists()), None)
    if sweep_json is None:   # the stage's own, else the one run_cards.sh's sweep step wrote beside the data directory
        raise Skip("no wide_sweep.json: run wide_sweep first")
    me = str(pathlib.Path(__file__).resolve())
    notes = []
    for form, width, fold, dtype in corpus_cells(sweep_json, args.quick):
        label = f"{form}_{width}_{fold}_{dtype}"
        cmd = [sys.executable, me, "--worker-sweep", form, str(width), str(fold), dtype,
               str(args.out.parent / "sweep_cache" / "ncu")]
        got, note = ncu_capture(args, args.out / f"ncu_corpus_{label}", f"regex:sweep_{width}_{fold}", cmd, CORPUS_METRICS,
                               ranged=False)
        emit(args.out / "ncu_corpus.csv", {"form": form, "width": width, "fold": fold, "dtype": dtype,
                                           **{m: got.get(m) for m in CORPUS_METRICS}, "note": "" if got else note})
        say(f"ncu {label}: {'ok' if got else 'failed'}")
        notes.append("" if got else note or "no output")
    ncu_stage_verdict(notes)


def capture_row(args, rep, cmd_line, metrics, **kw):
    """(the row's kernels / gpu_time_s / top_kernel and ``metrics`` of the top kernel, the output, the note) of one capture."""
    kernels, out = ncu_run(args, rep, cmd_line, metrics, **kw)
    return kernel_summary(kernels, metrics), out, "" if kernels else out or "no output"


def ncu_ensemble(args):
    require_ncu(args)
    notes = []
    for arm in (ENSEMBLE_QUICK_ARM,) if args.quick else ENSEMBLE_ARMS:
        for dist in ("spread", "uniform"):
            cell = {"arm": arm, "n": 10_000 if args.quick else POINT[0], "dist": dist}
            row, out, note = capture_row(args, args.out / f"ncu_ensemble_{arm}_{dist}", worker_cmd(cell),
                                         ENSEMBLE_METRICS + DP_METRICS, regex=KERNEL)
            labelled = NCU_CONDITIONAL_GRAPH in note
            note = NOT_PROFILABLE if labelled else note
            attempts = worker_row(out).get("sample_steps")
            dp = sum(row[m] or 0.0 for m in DP_METRICS)
            emit(args.out / "ncu_ensemble.csv", {
                "arm": arm, "dist": dist, **{k: row[k] for k in ("kernels", "gpu_time_s", "top_kernel", *ENSEMBLE_METRICS)},
                "lane_efficiency": row[ENSEMBLE_METRICS[0]] / 32 if row[ENSEMBLE_METRICS[0]] is not None else None,
                "dp_inst_per_attempt": dp / attempts if row["kernels"] and attempts else None, "note": note})
            say(f"ncu {arm} {dist}: {'ok' if row['kernels'] else note.splitlines()[0][:60]}")
            if not labelled:
                notes.append(note)
    ncu_stage_verdict(notes)


def one_per_row(cells):
    """The first cell of each feature-card row (the smoke of the feature stages)."""
    return list({c[0]: c for c in reversed(cells)}.values())[::-1]


def feature_cells(quick):
    """(row, case, dtype, arm, N) of every arm of ours of every feature-card row, case and dtype (``quick``: one per row,
    at the smoke N)."""
    fc = load_card(FEATURE_PATH)
    cells = [(r, c, d, a, spec["cases"][c][1 if quick else 0]) for r, spec in fc.ROWS.items() for c in spec["cases"]
             for d in spec["dtypes"] for a in spec["arms"] if a in fc.OUR_ARMS]
    return one_per_row(cells) if quick else cells


def feature_capture(args, cells, name, sections=()):
    """Each cell's timed call, alone inside a profiler range (feature_card's own runner, ``FEATURE_CARD_PROFILE`` set)."""
    require_ncu(args)
    notes = []
    for row, case, dtype, arm, n in cells:
        cmd = ["env", "FEATURE_CARD_PROFILE=1", sys.executable, str(FEATURE_PATH), "--arm", row, case, dtype, arm,
               *(["--smoke"] if args.quick else [])]
        got, _, note = capture_row(args, args.out / f"{name}_{row}_{case}_{dtype}_{arm}", cmd, ENSEMBLE_METRICS,
                                   count=LAUNCH_CAP, sections=sections)
        emit(args.out / f"{name}.csv", {"row": row, "case": case, "dtype": dtype, "n": n, "arm": arm,
                                        **{k: got[k] for k in ("kernels", "gpu_time_s", "top_kernel", *ENSEMBLE_METRICS)},
                                        "note": note})
        say(f"ncu {row} {case} {dtype} {arm}: {got['kernels']} kernels")
        notes.append(note)
    ncu_stage_verdict(notes)


def ncu_features(args):
    feature_capture(args, feature_cells(args.quick), "ncu_features")


def ncu_nvlibs(args):
    cells = [(*c, load_card(FEATURE_PATH).ROWS[c[0]]["cases"][c[1]][1 if args.quick else 0]) for c in NVLIB_CELLS]
    feature_capture(args, one_per_row(cells) if args.quick else cells, "ncu_nvlibs", NVLIB_SECTIONS)


def chunk_cells(args):
    ns = args.ns or ((100_000,) if args.quick else CHUNK_NS)
    chunks, tps = ((8, 32), (None,)) if args.quick else (CHUNKS, TPS)
    dists = CHUNK_DISTS[:1] if args.quick else CHUNK_DISTS
    dtypes = CHUNK_DTYPES[1:] if args.quick else CHUNK_DTYPES
    return [{"arm": "persist", "n": n, "dist": d, "dtype": t, "chunk": c, "tps": p}
            for n in ns for p in tps for t in dtypes for d in dists for c in chunks]


def chunk_ties(rows):
    """The two fastest rows of each (N, distribution, dtype) group whose runner-up is within :data:`TIE` of the best."""
    groups: dict = {}
    for r in rows:
        if "wall_ms" in r:
            groups.setdefault((r["n"], r["dist"], r["dtype"]), []).append(r)
    ranked = [sorted(g, key=lambda r: r["wall_ms"])[:2] for g in groups.values() if len(g) > 1]
    return [p for p in ranked if p[1]["wall_ms"] <= p[0]["wall_ms"] * (1 + TIE)]


def chunk_residency(args):
    card, rows, start = load_card(), [], time.perf_counter()
    for cell in chunk_cells(args):
        if args.budget_s and time.perf_counter() - start > args.budget_s:
            row = {**cell, "error": "not run: the stage budget was used up"}
        else:
            row = timed_cell(card, cell, 1 if args.quick else 3)
        emit(args.out / "chunk_residency.csv", {k: row.get(k) for k in
                                                (*cell, "wall_ms", "mode", "launches", "slot_fill", "launched_tps", "sm_mhz", "error")})
        rows.append(row)
        say(f"chunk {cell['n']} {cell['dist']} {cell['dtype']} c={cell['chunk']} tps={cell['tps']}: "
            f"{row.get('wall_ms', row.get('error'))}")
    if not args.ncu or (compute_capability() or 0) < NCU_MIN_CC:
        return
    for pair in chunk_ties(rows)[:4]:
        for r in pair:
            cell = {k: r[k] for k in ("arm", "n", "dist", "dtype", "chunk", "tps")}
            tag = "_".join(str(cell[k]) for k in ("n", "dist", "dtype", "chunk", "tps"))
            got, note = ncu_capture(args, args.out / f"chunk_residency_ncu_{tag}", KERNEL, worker_cmd(cell),
                                    ENSEMBLE_METRICS)
            emit(args.out / "chunk_residency_ncu.csv", {**cell, **{m: got.get(m) for m in ENSEMBLE_METRICS},
                                                         "note": "" if got else note})


def launch_shape(args):
    card = load_card()
    n = 10_000 if args.quick else POINT[0]
    for arm in LAUNCH_ARMS[:2] if args.quick else LAUNCH_ARMS:
        cell = {"arm": arm, "n": n, "dist": POINT[1]}
        row = {"arm": arm, "n": n, "wall_ms": None, "mode": None, "launches": None, "kernels": None, "kernel_ms": None,
               "api_calls": None, "kernel_names": None, "note": ""}
        if args.nsys:
            got, out = nsys_run(args, args.out / f"nsys_{arm}", cell)
            if got:
                row.update(kernels=got[0], kernel_ms=got[1], api_calls=got[2], kernel_names=" | ".join(got[3][:4]))
                row.update({k: v for k, v in worker_row(out).items() if k in ("wall_ms", "mode", "launches")})
            else:
                row["note"] = f"nsys failed: {out}"
        else:
            row["note"] = "nsys not found: timing only"
        if row["wall_ms"] is None:
            row.update({k: v for k, v in timed_cell(card, cell, 1 if args.quick else 3).items()
                        if k in ("wall_ms", "mode", "launches")})
        emit(args.out / "launch_shape.csv", row)
        say(f"launch_shape {arm}: {row['wall_ms']} ms, {row['kernels']} kernels")
    for n in CROSSOVER_NS[:2] if args.quick else args.ns or CROSSOVER_NS:
        for arm in ("fused", "persist"):
            row = timed_cell(card, {"arm": arm, "n": n, "dist": POINT[1], "dtype": "float64", "chunk": 32},
                             1 if args.quick else 3)
            emit(args.out / "crossover.csv", {k: row.get(k) for k in ("arm", "n", "wall_ms", "mode", "launches", "error")})


def band(args):
    card = load_card()
    for every in BAND_EVERY:
        row = timed_cell(card, {"arm": "band", "n": args.ns[0] if args.ns else POINT[0], "dist": POINT[1],
                                "every": every}, 3)
        emit(args.out / "band.csv", {k: row.get(k) for k in ("n", "every", "wall_ms", "mode", "launches", "error")})


STAGES = {"sol_probe": sol_probe, "wide_sweep": wide_sweep, "ncu_corpus": ncu_corpus, "ncu_ensemble": ncu_ensemble,
          "ncu_features": ncu_features, "ncu_nvlibs": ncu_nvlibs,
          "chunk_residency": chunk_residency, "launch_shape": launch_shape, "band": band}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("profile-out"))
    ap.add_argument("--stage", choices=tuple(STAGES), help="run one stage (profile.sh runs them in order)")
    ap.add_argument("--quick", action="store_true", help="one small point of the stage")
    ap.add_argument("--nsys", help="path of nsys")
    ap.add_argument("--ncu", help="path of ncu")
    ap.add_argument("--ncu-sudo", action="store_true", help="run ncu under `sudo -n` (counters are admin-only)")
    ap.add_argument("--ns", type=lambda t: tuple(int(float(x)) for x in t.split(",") if x),
                    help="sample counts, comma separated (1e5,1e6): chunk_residency, launch_shape crossover, band")
    ap.add_argument("--budget-s", type=float, help="chunk_residency: start no cell after this many seconds")
    ap.add_argument("--worker", metavar="CELL_JSON", help=argparse.SUPPRESS)
    ap.add_argument("--worker-sweep", nargs=5, metavar=("FORM", "W", "F", "DTYPE", "CACHE"), help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        worker(load_card(), json.loads(args.worker))
        return 0
    if args.worker_sweep:
        worker_sweep(*args.worker_sweep)
        return 0
    if args.stage is None:
        ap.error("pass --stage NAME; profile.sh runs the whole sequence")
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        STAGES[args.stage](args)
    except Skip as why:
        print(f"SKIP: {why}")
        return 3
    return 0


if __name__ == "__main__":
    with contextlib.suppress(BrokenPipeError):
        sys.exit(main())
