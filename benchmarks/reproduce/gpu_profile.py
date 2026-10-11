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
  bandwidth, warp stalls) at three points, exported to text; only on compute capability >= 7.0.

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
import time

HERE = pathlib.Path(__file__).resolve().parent
CARD_PATH = HERE.parent / "perf_card" / "perf_card.py"

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


def worker(arm_name, n, steps, dist):
    """Build one arm of the card, warm it, then run it once between
    ``cudaProfilerStart/Stop`` so a profiler attached with a capture range
    sees that run alone."""
    import cupy as cp

    card = load_card()
    inp = make_inputs(card, n, steps, dist)
    arm = card._make_gpu_arms(n, steps, names=(arm_name,))[arm_name]
    for _ in range(2):
        card._timed(arm, inp, True)  # compile, graph capture, caches
    arm["state"].upload(inp)
    cp.cuda.runtime.deviceSynchronize()
    cp.cuda.runtime.profilerStart()
    arm["run"]()
    cp.cuda.runtime.deviceSynchronize()
    cp.cuda.runtime.profilerStop()
    report = arm.get("report")
    print(f"worker arm={arm_name} n={n} steps={steps} dist={dist} "
          f"mode={getattr(report, 'mode', None)} launches={getattr(report, 'launches', None)}")


def _self_cmd(arm, n, steps, dist):
    return [sys.executable, str(pathlib.Path(__file__).resolve()), "--worker", arm, str(n), str(steps), dist]


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
                  "traced_kernels  kernel_ms  api_calls  api_ms")
    notes = ("Measured run only (profiler start/stop around one run after two warm runs).\n"
             "launches = what eagle reports for the run; traced_kernels / kernel_ms = what nsys recorded. A band run\n"
             "whose loop sits in a conditional graph can show fewer traced kernels than launches: the\n"
             "device-side loop's kernels are then missing from the trace, not absent from the run.\n")

    def flush():
        table = [table_head] + [
            f"{r[0]:26s} {r[1]:11s} {r[2]:<8d} {r[3]:<5d} {str(r[4]):10s} {str(r[5]):>8s}  {str(r[6]):>14s}  "
            f"{str(r[7]):>9s}  {str(r[8]):>9s}  {str(r[9]):>6s}" for r in rows]
        summary.write_text(notes + "\n".join(table) + "\n\n" + "\n".join(texts))

    for arm, n, steps, dist in cells:
        tag = f"{arm}-{dist.replace(':', '')}-n{n}-s{steps}"
        rep = out / f"nsys_{tag}"
        t0 = time.perf_counter()
        cmd = [nsys, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node", "--sample=none",
               "--cpuctxsw=none", "--capture-range=cudaProfilerApi", "--capture-range-end=stop",
               "-f", "true", "-o", str(rep)] + _self_cmd(arm, n, steps, dist)
        done = subprocess.run(cmd, capture_output=True, text=True)
        said = next((ln for ln in done.stdout.splitlines() if ln.startswith("worker ")), "")
        mode = (re.search(r"mode=(\S+)", said) or [None, "?"])[1]
        launches = (re.search(r"launches=(\S+)", said) or [None, "?"])[1]
        report = pathlib.Path(str(rep) + ".nsys-rep")
        if done.returncode != 0 or not report.exists():
            rows.append((arm, dist, n, steps, mode, launches, "FAILED", "", "", ""))
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
                     f"{(a[1] or 0) * 1e-6:.3f}"))
        flush()
        say(f"nsys {tag}: kernels={k[0]} kernel_ms={(k[1] or 0) * 1e-6:.3f} api_calls={a[0]} ({wall:.0f} s)")
    flush()
    return names


def nsys_cells(stage):
    if stage == "nsys-quick":
        return list(NSYS_QUICK)
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


def ncu_command(ncu, rep, regex, arm, point, sudo):
    """One kernel's capture: the first matching launch of the measured run, never more."""
    n, steps, dist = point
    cmd = [ncu, "--profile-from-start", "off", "--launch-skip", "0", "--launch-count", "1",
           "--kernel-name", regex, "--target-processes", "all", "-f", "-o", str(rep)]
    for section in NCU_SECTIONS:
        cmd += ["--section", section]
    cmd += _self_cmd(arm, n, steps, dist)
    if sudo:
        cmd = ["sudo", "-n", "-E", "env", f"PATH={os.environ['PATH']}"] + cmd
    return cmd


class Skip(Exception):
    """A stage that has nothing to do here (reason in the message)."""


def ncu_stage(out, stage, ncu, sudo):
    cc_ = compute_capability()
    if cc_ is None:
        raise Skip("compute capability not readable (nvidia-smi)")
    if cc_ < NCU_MIN_CC:
        raise Skip(f"compute capability {cc_} < {NCU_MIN_CC} (Nsight Compute needs Volta or newer)")
    point = NCU_POINTS[stage]
    chosen = [k for k in NCU_KERNELS if stage != "ncu-quick" or k[0] in NCU_QUICK_KERNELS]
    summary, failed = [], 0
    for label, arm, regex in chosen:
        rep = out / f"{stage}_{label}"
        t0 = time.perf_counter()
        try:
            done = subprocess.run(ncu_command(ncu, rep, regex, arm, point, sudo), capture_output=True,
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
            summary.append(f"== {label} ({arm}, {regex}, N={point[0]} S={point[1]} {point[2]}, "
                           f"{time.perf_counter() - t0:.0f} s)\n{_ncu_digest(text.stdout)}\n")
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


def validate_ncu(ncu, nsys_names):
    """Check without a supported GPU: every chosen section exists in this ncu, and each
    kernel-name regex matches a name nsys recorded. Returns (ok, lines)."""
    lines, ok = [], True
    listed = subprocess.run([ncu, "--list-sections"], capture_output=True, text=True).stdout
    for section in NCU_SECTIONS:
        found = re.search(rf"^{section}\s", listed, re.M) is not None
        ok &= found
        lines.append(f"section {section}: {'found' if found else 'MISSING'}")
    for label, arm, regex in NCU_KERNELS:
        pattern = re.compile(regex.split("regex:", 1)[1])
        hits = sorted({n for a, n in nsys_names if a == arm and pattern.search(n)})
        if arm == "eagle_graph_steps_compact" and not hits:
            lines.append(f"kernel {label} ({arm}) {regex}: unverified (nsys does not record the band loop's kernels)")
            continue
        ok &= bool(hits)
        lines.append(f"kernel {label} ({arm}) {regex}: {len(hits)} match(es) {hits[:3]}")
    return ok, lines


# --------------------------------------------------------------------------- #
STAGES = ("probe", "nsys-grid", "nsys-width", "nsys-quick", *NCU_POINTS, "validate-ncu")


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
        elif stage.startswith("nsys") or stage == "validate-ncu":
            if not args.nsys:
                raise Skip("nsys not found")
            if stage == "validate-ncu":
                cells = [(a, 100_000, 100, "spread") for a in dict.fromkeys(k[1] for k in NCU_KERNELS)]
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
