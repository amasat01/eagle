# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The eagle CPU card: the performance card's workload, run the common CPU ways.

The workload is the one of ``perf_card.py`` (imported from it, not copied): a
batch of independent damped harmonic oscillators, classical RK4 in float64,
every sample stopping at its own step count ("spread": log-uniform stop steps;
"uniform": every sample runs the full count). This card runs it the ways a
Python user would on a CPU (the "arms"):

``eagle_term8`` / ``eagle_term1``
    The GPU card's ``eagle_graph`` arm on the host, through eagle's host dual
    mode: the same hawk step kernel (its ``Terminated`` guard skips finished
    samples, and it finishes its own samples: it sets the mask and counts
    each sample reaching its stop step), deployed by ``eagle.deploy`` (NumPy
    planes select the host build) and run by the same one call,
    ``eagle.until_done``, on eagle's OpenMP host team (8 and 1 threads).
``eagle_compact8`` / ``eagle_compact1``
    The GPU card's ``eagle_graph_compact`` on the host: the same call on the
    active-set step kernel (a NumPy mask, so the host compaction), compacting
    every 16 steps when a sample finished.
``eagle_reorder8`` / ``eagle_reorder1``
    The GPU card's ``eagle_graph_reorder`` on the host: plus the reorder of
    the planes and the final restore to sample order inside the wall.
``jax_shard8``
    The JAX arm below, sharded over 8 CPU devices (``shard_map``).
``numba_prange``
    Numba ``@njit(parallel=True)``: ``prange`` over samples, each sample runs
    its own steps in an inner loop and stops at its own step count.
``jax_vmap``
    JAX: a per-sample ``lax.while_loop`` (stop when the step count is
    reached, or the step cap), batched with ``vmap`` and compiled with ``jit``.
``torch_masked``
    PyTorch CPU, the masked array formulation (``torch.where`` keeps finished
    samples unchanged), ``torch.set_num_threads(8)``.
``numpy_masked``
    NumPy, the same masked array formulation (``np.where``).
``mp_numpy``
    The standard library's ``multiprocessing``: 8 processes, each running the
    masked NumPy formulation over one contiguous chunk of the batch.
``python_loops``
    Plain Python: a loop over samples, each running its own steps (small N
    only; its cells are not extrapolated).

Every arm runs in its own persistent worker process, so every library's
thread pool is pinned by that process's environment (``OMP_NUM_THREADS`` and
friends, ``torch.set_num_threads``, ``numba.set_num_threads``) and no two
pools are live at once. The driver process interleaves the repetitions across
the workers (the arm order rotates every repetition), checks that every arm
agrees with ``eagle_term8`` before any timing counts (identical per-sample step
counts, x and v within the performance card's tolerance), and reports
medians with interquartile ranges.

A separate MEMORY pass follows the timing pass: every (arm, N, configuration)
in a fresh process, peak resident memory of one run above a baseline taken
after the imports and the warm-up compile, before the batch is allocated.

Reproduce (from the eagle repository root, in an environment with eagle,
hawk, NumPy, Numba, JAX and PyTorch)::

    python benchmarks/perf_card/cpu_card.py                 # everything
    python benchmarks/perf_card/cpu_card.py --configs spread:100 --ns 1000000

A run writes one part file per cell into ``--parts-dir`` and then merges every
part present; the card (``cpu_card_<cpu-slug>.json`` + ``.md``) is written once
the matrix is complete (``--allow-partial`` writes it earlier, for dry runs).
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):  # perf_card.py; benchmarks/: the shared helpers
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _card_common as cc  # noqa: E402
import perf_card as pc  # noqa: E402  (the workload definition)

# --------------------------------------------------------------------------- #
# Card definition
# --------------------------------------------------------------------------- #
EAGLE_ARMS = ("eagle_term8", "eagle_term1", "eagle_compact8", "eagle_compact1",
              "eagle_reorder8", "eagle_reorder1")
ARMS = EAGLE_ARMS + ("numba_prange", "jax_shard8", "jax_vmap", "torch_masked",
                     "numpy_masked", "mp_numpy", "python_loops")
REFERENCE_ARM = "eagle_term8"
THREADS = 8
#: The arms run in three groups, each interleaved on its own, so one cell's
#: group fits a 10-minute command at N = 1e6. The reference arm runs in every
#: group (it is what each group verifies against), and its spread across the
#: groups is recorded as the between-group drift.
ARM_GROUPS = (EAGLE_ARMS + ("python_loops",),
              ("eagle_term8", "torch_masked", "mp_numpy"),
              ("eagle_term8", "numpy_masked", "numba_prange", "jax_shard8", "jax_vmap"))
ARM_LABELS = {
    "eagle_term8": "eagle host, termination loop, up to 8 threads",
    "eagle_term1": "eagle host, termination loop, 1 thread",
    "eagle_compact8": "eagle host, termination + compaction, up to 8 threads",
    "eagle_compact1": "eagle host, termination + compaction, 1 thread",
    "eagle_reorder8": "eagle host, termination + compaction + reorder, up to 8 threads",
    "eagle_reorder1": "eagle host, termination + compaction + reorder, 1 thread",
    "jax_shard8": "JAX sharded over 8 CPU devices",
    "numba_prange": "Numba prange, 8 threads",
    "jax_vmap": "JAX jit + vmap(while_loop), 1 device",
    "torch_masked": "PyTorch masked, 8 threads",
    "numpy_masked": "NumPy masked",
    "mp_numpy": "multiprocessing, 8 x NumPy",
    "python_loops": "plain Python loops",
}
#: Threads (or processes) each arm is pinned to.
ARM_THREADS = {"eagle_term8": THREADS, "eagle_term1": 1, "eagle_compact8": THREADS,
               "eagle_compact1": 1, "eagle_reorder8": THREADS, "eagle_reorder1": 1,
               "jax_shard8": THREADS, "numba_prange": THREADS,
               "jax_vmap": THREADS, "torch_masked": THREADS, "numpy_masked": 1,
               "mp_numpy": THREADS, "python_loops": 1}
ARM_LOOP = {
    "eagle_term8": "per sample: eagle.until_done, one host-team pass over dynamic tiles of samples, each sample stepped to its own stop",
    "eagle_term1": "per sample: eagle.until_done, one host-team pass over dynamic tiles of samples, each sample stepped to its own stop",
    "eagle_compact8": "as the termination loop, the step launched over the active-set map only (recomputed every 16 steps)",
    "eagle_compact1": "as the termination loop, the step launched over the active-set map only (recomputed every 16 steps)",
    "eagle_reorder8": "as the compaction loop, plus a reorder of the planes when the live samples spread thin, restored at the end",
    "eagle_reorder1": "as the compaction loop, plus a reorder of the planes when the live samples spread thin, restored at the end",
    "jax_shard8": "per step over a shard: 8 shards, each a vmapped while loop running until the shard's last sample stops or the step cap is reached",
    "numba_prange": "per sample: prange over samples, each runs its own steps to its stop step",
    "jax_vmap": "per step over the batch: the vmapped while loop runs until the last sample stops or the step cap is reached, finished samples held by a select",
    "torch_masked": "per step over the batch: whole-batch tensor expressions, finished samples held by torch.where",
    "numpy_masked": "per step over the batch: whole-batch array expressions, finished samples held by np.where",
    "mp_numpy": "per step over a chunk: 8 processes, each the NumPy masked loop over its chunk, stopping when its chunk is done",
    "python_loops": "per sample: a Python loop over samples, each runs its own steps to its stop step",
}
ARM_THREADING = {
    "eagle_term8": "OMP_NUM_THREADS=8 (eagle's OpenMP host team: up to 8 threads; eagle uses one per physical core for a run under eagle._host_loop.SMT_MIN_WORK sample-steps)",
    "eagle_term1": "OMP_NUM_THREADS=1 (eagle's OpenMP host team)",
    "eagle_compact8": "OMP_NUM_THREADS=8 (eagle's OpenMP host team: up to 8 threads; eagle uses one per physical core for a run under eagle._host_loop.SMT_MIN_WORK sample-steps)",
    "eagle_compact1": "OMP_NUM_THREADS=1 (eagle's OpenMP host team)",
    "eagle_reorder8": "OMP_NUM_THREADS=8 (eagle's OpenMP host team: up to 8 threads; eagle uses one per physical core for a run under eagle._host_loop.SMT_MIN_WORK sample-steps)",
    "eagle_reorder1": "OMP_NUM_THREADS=1 (eagle's OpenMP host team)",
    "jax_shard8": "XLA_FLAGS=--xla_force_host_platform_device_count=8, one shard per device",
    "numba_prange": "NUMBA_NUM_THREADS=8 and numba.set_num_threads(8)",
    "jax_vmap": "XLA's CPU client defaults, one device (CPU busy column shows the use)",
    "torch_masked": "torch.set_num_threads(8), OMP_NUM_THREADS=8",
    "numpy_masked": "single-threaded ufuncs; BLAS/OpenMP pools pinned to 1",
    "mp_numpy": "multiprocessing.Pool(8), fork; every process pinned to 1 thread",
    "python_loops": "one interpreter thread",
}
#: Plain Python runs only where a run stays short: N <= 10,000 and at most
#: this many (sample, step) pairs (about 10 s per run on the reference host).
PYTHON_MAX_N = 10_000
PYTHON_MAX_SAMPLE_STEPS = 2_500_000
#: FP64 fused multiply-add units per core assumed for the peak (Intel lists
#: two AVX-512 FMA units for the Xeon W-2125 this card was first run on).
FMA_UNITS_PER_CORE = 2
#: Pause before every run (seconds), so the thread pools of the arm that ran
#: last have finished spin-waiting and gone idle before the next arm starts.
SETTLE_S = float(os.environ.get("CPU_CARD_SETTLE_S", "0.2"))
#: A timed repetition of a short run repeats it back to back until it spans
#: at least MIN_TIMED_S (count fixed per arm and cell from a second, warm run's
#: wall, at most MAX_INNER) and reports the mean, so runs of a few ms measure
#: warm threads and a ramped clock for every arm alike; runs longer than
#: MIN_TIMED_S are timed once per repetition.
MIN_TIMED_S = 0.1
MAX_INNER = 2000
#: Untimed running before each timed block, so it starts on a ramped clock
#: (after SETTLE_S idle, intel_pstate/powersave takes ~40 ms to ramp up).
RAMP_S = 0.05
#: Back-to-back runs, after the ramp, whose mean sizes a cell's timed block.
CALIBRATE_S = 0.05


def _cpu_jiffies():
    """(busy, total) jiffies over all CPUs, from /proc/stat's first line."""
    try:
        f = [int(x) for x in pathlib.Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    except (OSError, ValueError):
        return None
    idle = f[3] + (f[4] if len(f) > 4 else 0)
    return sum(f) - idle, sum(f)


def _busy_cores(j0, j1):
    """Logical CPUs' worth of work the machine did between two readings."""
    if j0 is None or j1 is None or j1[1] == j0[1]:
        return None
    return round((j1[0] - j0[0]) / (j1[1] - j0[1]) * (os.cpu_count() or 1), 3)


def _cpu_mhz_now():
    """The highest current clock over the CPUs (sysfs scaling_cur_freq) -- the
    core the arm just ramped; idle cores read low -- or None."""
    vals = []
    for p in pathlib.Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            vals.append(int(p.read_text()) / 1000.0)
        except (OSError, ValueError):
            pass
    return round(max(vals), 1) if vals else None
#: Memory pass warm-up size (the warm-up compiles every JIT on a tiny batch).
WARM_N = 64
#: The state a sample needs: x, v (state width 2) plus the per-sample scalars
#: omega, zeta and the stop step, all float64.
STATE_WIDTH = 2
PER_SAMPLE_SCALARS = 3


def _included(arm, n, distribution, max_steps):
    """None if the arm runs this cell, else the reason it does not."""
    if arm != "python_loops":
        return None
    steps = _sample_steps(n, distribution, max_steps)
    if n > PYTHON_MAX_N or steps > PYTHON_MAX_SAMPLE_STEPS:
        return (f"plain Python runs only at N <= {PYTHON_MAX_N:,} with at most "
                f"{PYTHON_MAX_SAMPLE_STEPS:,} sample-steps per run (this cell: "
                f"{steps:,}); not extrapolated")
    return None


_STEPS_MEMO = {}


def _sample_steps(n, distribution, max_steps):
    key = (n, distribution, max_steps)
    if key not in _STEPS_MEMO:
        inp = pc._inputs(n, distribution, max_steps)
        _STEPS_MEMO[key] = int(pc._active_profile(inp["nstop"], max_steps).sum())
    return _STEPS_MEMO[key]


def _arm_env(arm):
    t = str(ARM_THREADS[arm]) if arm not in ("mp_numpy",) else "1"
    env = {k: t for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                          "NUMBA_NUM_THREADS")}
    env.update(CUDA_VISIBLE_DEVICES="", PYTHONWARNINGS="ignore",
               XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_PLATFORMS="cpu")
    if arm == "jax_shard8":
        env["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={THREADS}"
    return env


# --------------------------------------------------------------------------- #
# Rendering (pure: JSON in, Markdown out; imported by the drift test)
# --------------------------------------------------------------------------- #
_fmt_time, _fmt_rate, _fmt_pct = cc.fmt_time, cc.fmt_rate, cc.fmt_pct
_fmt_mib, _fmt_x = cc.fmt_mib, cc.fmt_x


#: The comparison contract, printed above the arm table.
CAP_CONTRACT = (
    "Every arm runs each sample to its own stop, and the stop steps here never "
    "exceed max_steps (S); the arms that carry a cap enforce it as well: eagle "
    "per sample, JAX in its while-loop condition, the masked arms by their "
    "outer step count (a batch-level cap).")
#: What each capped arm needed to meet it.
CAP_NEEDED = (
    "What each arm needed to get there: JAX, a cap in the while_loop condition; "
    "eagle, the kernel source unchanged, with its loop shape, unrolling and "
    "host entry chosen by hawk + eagle (per device, per batch).")


def render_markdown(card: dict) -> str:
    """The CPU card's tables and per-arm notes, as MyST Markdown.

    Deterministic in ``card`` alone, so a test can re-render the committed
    JSON and compare it with the committed Markdown byte for byte."""
    cpu, peak, method = card["cpu"], card["peak"], card["method"]
    lines = [
        f"<!-- Generated by benchmarks/perf_card/cpu_card.py from "
        f"cpu_card_{card['cpu_slug']}.json; do not edit. -->",
        "",
        f"**CPU:** {cpu['model']} ({cpu['cores']} cores, {cpu['logical_cpus']} "
        f"logical CPUs, {cpu['simd_doubles']} float64 lanes per vector "
        f"instruction). **Peak FP64:** {_fmt_rate(peak['fp64_flops'])} FLOP/s "
        f"({peak['assumption']}). Without SIMD the same cores peak at "
        f"{_fmt_rate(peak['fp64_flops_scalar'])} FLOP/s. Medians of "
        f"{method['repetitions']} interleaved repetitions; IQR in parentheses.",
        "",
        f"Same workload as the GPU card (damped oscillators, classical RK4, "
        f"float64, every sample stops at its own step). \"8 threads\" uses all "
        f"{cpu['logical_cpus']} logical CPUs ({cpu['cores']} cores × "
        f"{cpu['threads_per_core']} hardware threads). CPU busy = process CPU "
        f"seconds / wall (all threads and worker processes; spin-waiting "
        f"counts). Compile times are not in any wall; they have their own "
        f"table below. {method['settle'].capitalize()}; {method['inner_runs'].split(' (count')[0]}.",
        "",
        _groups_line(card),
        "",
        "**SIMD.** " + " ".join(card["simd"]["summary"]),
        "",
        "| arm | loop structure | threading |",
        "|---|---|---|",
    ]
    lines[-2:-2] = [CAP_CONTRACT, "", CAP_NEEDED, ""]
    for arm in card["arms"]:
        lines.append(f"| {ARM_LABELS[arm]} | {ARM_LOOP[arm]} | {ARM_THREADING[arm]} |")
    lines.append("")

    mem = card.get("memory") or {}
    mem_rows = {(r["distribution"], r["max_steps"], r["n"], r["arm"]): r
                for r in mem.get("rows", [])}
    by_cfg = {}
    for row in card["results"]:
        by_cfg.setdefault((row["distribution"], row["max_steps"]), []).append(row)
    for cfg in card["configs"]:
        key = (cfg["distribution"], cfg["max_steps"])
        rows = by_cfg.get(key, [])
        lines += [
            f"### {pc._config_title(cfg)}",
            "",
            "| N | arm | threads | wall (IQR) | sample·steps/s | useful FLOP/s (% peak) "
            "| CPU busy | peak memory (× minimum) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in sorted(rows, key=lambda r: (r["n"], ARMS.index(r["arm"]))):
            w = row["wall_s"]
            m = mem_rows.get((row["distribution"], row["max_steps"], row["n"], row["arm"]))
            mem_cell = ("–" if m is None else
                        f"{_fmt_mib(m['peak_bytes'])} ({_fmt_x(m['overhead_factor'])})")
            lines.append(
                f"| {row['n']:,} | {ARM_LABELS[row['arm']]} | {row['threads']} "
                f"| {_fmt_time(w['median'])} ({_fmt_time(w['iqr'])}) "
                f"| {_fmt_rate(row['sample_steps_per_s'])} "
                f"| {_fmt_rate(row['useful_flops_per_s'])} "
                f"({_fmt_pct(row['fraction_of_peak_fp64'])}) "
                f"| {row['cpu_busy']:.2g} "
                f"| {mem_cell} |"
            )
        lines += [""] + cc.fastest_lines(rows, ARM_LABELS) + [""]

    if card["dropped"]:
        lines += ["### Cells not run", ""]
        seen = {}
        for d in card["dropped"]:
            seen.setdefault((d["arm"], d["reason"].split(" (this cell")[0]), []).append(
                f"{d['distribution']} S={d['max_steps']} N={d['n']:,}")
        for (arm, reason), cells in seen.items():
            lines.append(f"- {ARM_LABELS[arm]}: {reason} ({'; '.join(cells)}).")
        lines.append("")

    lines += cc.fit_section(card, ARM_LABELS)

    comp_t = card.get("compile_time") or {}
    if comp_t.get("rows"):
        lines += ["### Compile time", "", comp_t["method"], "",
                  "| arm | compiled | cold (median of 3) | warm (median of 3) |",
                  "|---|---|---:|---:|"]
        for r in comp_t["rows"]:
            lines.append(f"| {ARM_LABELS[r['arm']]} | {r['what']} "
                         f"| {_fmt_time(r['cold_s']['median'])} "
                         f"| {_fmt_time(r['warm_s']['median'])} |")
        lines.append("")
        lines.append("No compile step: " + ", ".join(ARM_LABELS[a] for a in comp_t["none"])
                     + ".")
        lines.append("")
        shared = comp_t.get("shared") or {}
        if shared:
            lines.append("Same build as another row: " + "; ".join(
                f"{ARM_LABELS[a]} (as {ARM_LABELS[b]})" for a, b in shared.items()) + ".")
            lines.append("")
    if mem.get("rows"):
        lines += ["### Memory method", "", mem["method"], ""]
        comp = mem.get("compile_overhead_bytes") or {}
        if comp:
            lines.append("Compile/import overhead (resident memory added by the warm-up "
                         "compile, measured once per arm at the largest N): "
                         + "; ".join(f"{ARM_LABELS[a]} {_fmt_mib(v)}"
                                     for a, v in comp.items()) + ".")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _groups_line(card):
    groups = "; ".join("(" + ", ".join(f"`{a}`" for a in g) + ")"
                       for g in card["arm_groups"])
    line = (f"The arms run in {len(card['arm_groups'])} groups, each interleaved "
            f"on its own: {groups}. eagle at 8 threads runs in every group as the "
            f"common reference; its row below is from the first group.")
    reruns = card.get("reference_reruns") or []
    if reruns:
        worst = max(reruns, key=lambda r: r["max_over_min"])
        line += (f" Its median wall differed between groups by at most "
                 f"{100 * (worst['max_over_min'] - 1):.2g} % in any cell "
                 f"({worst['distribution']} S={worst['max_steps']}, N={worst['n']:,}).")
    return line


def _fit_notes(card):
    """One neutral, number-backed note per arm (written into the card, then
    rendered): what the tool brings, and its wall time against eagle CPU at
    8 threads at the largest N it ran, per configuration."""
    rows = {(r["distribution"], r["max_steps"], r["n"], r["arm"]): r
            for r in card["results"]}
    wins, cells = cc.wins(card)

    def vs_eagle(arm):
        parts = []
        for cfg in card["configs"]:
            d, s = cfg["distribution"], cfg["max_steps"]
            ns = sorted(n for (dd, ss, n, a) in rows if (dd, ss, a) == (d, s, arm))
            if not ns:
                continue
            n = ns[-1]
            me, ref = rows[(d, s, n, arm)], rows.get((d, s, n, REFERENCE_ARM))
            if ref is None:
                continue
            ratio = me["wall_s"]["median"] / ref["wall_s"]["median"]
            parts.append(f"{d} S={s}, N={n:,}: {_fmt_time(me['wall_s']['median'])} "
                         f"({ratio:.3g}× eagle 8-thread time)")
        return "; ".join(parts)

    strengths = {
        "eagle_term8": (
            "The GPU card's eagle graph arm, unchanged, on the host: the same hawk "
            "step kernel, eagle.deploy and eagle.until_done call, run by eagle's "
            "OpenMP host team; one code base for GPUs and CPUs."),
        "eagle_term1": (
            "The same loop on one thread: the difference to the 8-thread row is "
            "the threading alone."),
        "eagle_compact8": (
            "The GPU card's compaction arm on the host: the launch covers only the "
            "samples still running, the map rebuilt every 16 steps when a sample "
            "finished."),
        "eagle_compact1": (
            "The compaction loop on one thread: the difference to the 8-thread "
            "row is the threading alone."),
        "eagle_reorder8": (
            "The GPU card's reorder arm on the host: the live samples moved to "
            "the front of every plane when they spread thin, restored to sample "
            "order at the end (inside the wall)."),
        "eagle_reorder1": (
            "The reorder loop on one thread: the difference to the 8-thread row "
            "is the threading alone."),
        "jax_shard8": (
            "The JAX arm spread over 8 CPU devices: the batch is sharded and each "
            "shard runs its own vmapped while loop, with no cross-shard step "
            "synchronisation."),
        "numba_prange": (
            "Compiles a plain Python loop to machine code with no separate "
            "language; each sample keeps its state in registers across all of "
            "its steps and stops on its own, so finished samples cost nothing."),
        "jax_vmap": (
            "Compiles the whole loop once and batches a per-sample while loop "
            "with vmap; the same code runs on GPUs and differentiates with "
            "jax.grad."),
        "torch_masked": (
            "Runs the masked array version on a multi-threaded tensor library; "
            "it fits where the computation already lives next to a PyTorch "
            "model or needs autograd."),
        "numpy_masked": (
            "No compile step and no dependency beyond NumPy; masked array code "
            "computes every sample every step, so it fits dense batches where "
            "all samples run to the end (the uniform configuration)."),
        "mp_numpy": (
            "Spreads NumPy over processes with the standard library alone; "
            "each chunk stops once its own samples are done."),
        "python_loops": (
            "Nothing to install and each line can be stepped in a debugger; "
            "run only at small N here."),
    }
    notes = {}
    for arm in card["arms"]:
        cmp = "" if arm == REFERENCE_ARM else f" On this card: {vs_eagle(arm)}."
        if arm == REFERENCE_ARM:
            cmp = " On this card: " + "; ".join(
                f"{cfg['distribution']} S={cfg['max_steps']}, N={max(card['ns']):,}: "
                f"{_fmt_time(rows[(cfg['distribution'], cfg['max_steps'], max(card['ns']), arm)]['wall_s']['median'])}"
                for cfg in card["configs"]
                if (cfg["distribution"], cfg["max_steps"], max(card["ns"]), arm) in rows) + "."
        notes[arm] = (f"{strengths[arm]}{cmp} Fastest in {wins[arm]} of "
                      f"{cells} cells.")
    intro = (
        "Each tool below is written the way its users write it. Two loop "
        "structures appear: per step over the batch (eagle's termination loops, JAX, "
        "PyTorch, NumPy, multiprocessing) and per sample over its steps (Numba, "
        "plain Python). The per-sample structure touches each sample's state once "
        "for all its steps; the per-step structure streams the batch through "
        "memory every step, which is the structure the GPU card's kernels "
        "use. Ratios below are wall-time ratios at the largest N each arm ran "
        "(below 1 = less time than eagle at 8 threads).")
    return {"intro": intro, "arms": notes}


# --------------------------------------------------------------------------- #
# Arms (each instantiated inside its own worker process)
# --------------------------------------------------------------------------- #
DT = pc.DT
H2 = 0.5 * DT
H6 = DT * 0.16666666666666666


# >>> code:numpy_masked
def _masked_numpy(omega, zeta, x, v, nstop, max_steps):
    """The masked array formulation (the GPU card's cupy_masked, in NumPy)."""
    import numpy as np

    k = np.zeros_like(x)
    nw2 = -(omega * omega)
    cc = (2.0 * zeta) * omega
    for _ in range(max_steps):
        running = k < nstop
        if not running.any():
            break
        a1 = nw2 * x - cc * v
        v2 = v + H2 * a1
        a2 = nw2 * (x + H2 * v) - cc * v2
        v3 = v + H2 * a2
        a3 = nw2 * (x + H2 * v2) - cc * v3
        v4 = v + DT * a3
        a4 = nw2 * (x + DT * v3) - cc * v4
        xn = x + H6 * (((v + 2.0 * v2) + 2.0 * v3) + v4)
        vn = v + H6 * (((a1 + 2.0 * a2) + 2.0 * a3) + a4)
        x = np.where(running, xn, x)
        v = np.where(running, vn, v)
        k = k + running
    return x, v, k
# <<< code:numpy_masked


# >>> code:mp_chunk
def _mp_chunk(args):
    t0 = time.process_time()
    base = _rss_kib()
    _reset_peak()
    x, v, k = _masked_numpy(*args)
    peak = _peak_kib()
    return x, v, k, time.process_time() - t0, max(0, peak - base) * 1024
# <<< code:mp_chunk


# >>> code:python_loops
def _python_loops(omega, zeta, x0, v0, nstop):
    om, ze, xs, vs, ns = (a.tolist() for a in (omega, zeta, x0, v0, nstop))
    dt, t3, t7 = DT, H6, H2
    xo, vo, ko = [], [], []
    for i in range(len(om)):
        x, v, k, nst = xs[i], vs[i], 0.0, ns[i]
        w = om[i]
        t10 = -(w * w)
        t14 = (2.0 * ze[i]) * w
        while k < nst:
            t16 = t10 * x - t14 * v
            t18 = v + t7 * t16
            t25 = t10 * (x + t7 * v) - t14 * t18
            t27 = v + t7 * t25
            t34 = t10 * (x + t7 * t18) - t14 * t27
            t36 = v + dt * t34
            xn = x + t3 * (((v + 2.0 * t18) + 2.0 * t27) + t36)
            vn = v + t3 * (((t16 + 2.0 * t25) + 2.0 * t34) + (t10 * (x + dt * t27) - t14 * t36))
            x, v = xn, vn
            k += 1.0
        xo.append(x)
        vo.append(v)
        ko.append(k)
    return xo, vo, ko
# <<< code:python_loops


# >>> code:numba_prange
def _make_numba(cache=False):
    import numba
    import numpy as np
    from numba import njit, prange

    numba.set_num_threads(THREADS)

    @njit(parallel=True, cache=cache)
    def oscillators(omega, zeta, x0, v0, nstop, dt):
        n = x0.shape[0]
        xo = np.empty(n)
        vo = np.empty(n)
        ko = np.empty(n)
        t3 = dt * 0.16666666666666666
        t7 = 0.5 * dt
        for i in prange(n):
            x = x0[i]
            v = v0[i]
            k = 0.0
            nst = nstop[i]
            w = omega[i]
            t10 = -(w * w)
            t14 = (2.0 * zeta[i]) * w
            while k < nst:
                t16 = t10 * x - t14 * v
                t18 = v + t7 * t16
                t25 = t10 * (x + t7 * v) - t14 * t18
                t27 = v + t7 * t25
                t34 = t10 * (x + t7 * t18) - t14 * t27
                t36 = v + dt * t34
                xn = x + t3 * (((v + 2.0 * t18) + 2.0 * t27) + t36)
                vn = v + t3 * (((t16 + 2.0 * t25) + 2.0 * t34)
                               + (t10 * (x + dt * t27) - t14 * t36))
                x = xn
                v = vn
                k += 1.0
            xo[i] = x
            vo[i] = v
            ko[i] = k
        return xo, vo, ko

    return oscillators
# <<< code:numba_prange


# >>> code:jax_vmap
def _make_jax_fn(cap, jit=True):
    import jax
    import jax.numpy as jnp
    from jax import lax

    def one(omega, zeta, x0, v0, nstop):
        t10 = -(omega * omega)
        t14 = (2.0 * zeta) * omega

        def body(s):
            x, v, k = s
            t16 = t10 * x - t14 * v
            t18 = v + H2 * t16
            t25 = t10 * (x + H2 * v) - t14 * t18
            t27 = v + H2 * t25
            t34 = t10 * (x + H2 * t18) - t14 * t27
            t36 = v + DT * t34
            xn = x + H6 * (((v + 2.0 * t18) + 2.0 * t27) + t36)
            vn = v + H6 * (((t16 + 2.0 * t25) + 2.0 * t34) + (t10 * (x + DT * t27) - t14 * t36))
            return xn, vn, k + 1.0

        return lax.while_loop(lambda s: (s[2] < nstop) & (s[2] < cap), body,
                              (x0, v0, jnp.zeros_like(x0)))

    return jax.jit(jax.vmap(one)) if jit else jax.vmap(one)
# <<< code:jax_vmap


class _Arm:
    """One arm inside its worker: init() compiles what it can; setup() takes a
    cell's inputs (and compiles what is shape-specific); prepare() is the
    untimed reset before a run; run() is timed; download() returns x, v, k."""

    compile_s = None       # arm-level one-off compile, set by init()
    child_cpu_s = 0.0      # CPU seconds of helper processes in the last run
    child_peak_bytes = 0   # their summed peak memory above their own baseline

    def init(self):
        pass

    def setup(self, inp, max_steps):
        import numpy as np

        self.inp = {k: np.ascontiguousarray(v) for k, v in inp.items()}
        self.max_steps = max_steps
        return None

    def prepare(self):
        pass

    def facts(self):
        return {}


def _host_build_facts(kernel):
    """(.so path, entry symbol, source md5s) of ``kernel``'s host build, by
    hawk's own bundle builder (the content-addressed build ``eagle.deploy``
    uses, so a cache hit recompiles nothing)."""
    import atexit
    import shutil

    from hawk.artifact import build_bundle, plan_view

    work = pathlib.Path(tempfile.mkdtemp(prefix="cpu_card_"))
    atexit.register(shutil.rmtree, work, ignore_errors=True)
    bundle = build_bundle([kernel], work, targets=("host",))
    art = next(a for a in bundle.artifacts if a.name == kernel.name)
    sources = {p.name: cc.md5(p) for p in sorted(work.iterdir()) if p.suffix == ".cpp"}
    return pathlib.Path(art.entries["host"].artifact), plan_view(art.sidecar)["host_entry"], sources


class _EagleHostArm(_Arm):
    """The GPU card's termination-based eagle arms on the host (eagle's host
    dual mode): ``variant`` "term" (eagle_graph), "compact"
    (eagle_graph_compact) or "reorder" (eagle_graph_reorder)."""

    variant = "term"

    def init(self):
        import numpy as np

        import eagle

        self.np = np
        self.kernel = (pc._define_kernel() if self.variant == "term"
                       else pc._define_active_kernel())
        t0 = time.perf_counter()
        eagle.deploy(self.kernel)  # the build (the hawk compile cache may serve it)
        self.compile_s = time.perf_counter() - t0

    def setup(self, inp, max_steps):
        from types import SimpleNamespace

        import eagle

        np = self.np
        super().setup(inp, max_steps)
        n = len(inp["x0"])
        st = SimpleNamespace(**{k: np.empty(n) for k in ("omega", "zeta", "nstop", "x", "v")})
        st.k = np.zeros(n)
        st.terminated = np.zeros(n, dtype=bool)
        self.st = st
        planes = vars(st)
        oscillator_step = self.kernel
        if self.variant == "term":
            # >>> code:eagle_term
            self.runner = eagle.until_done(eagle.deploy(oscillator_step),
                                           max_steps=max_steps, dt=DT, **planes)
            # <<< code:eagle_term
        elif self.variant == "compact":
            # >>> code:eagle_compact
            self.runner = eagle.until_done(eagle.deploy(oscillator_step),
                                           max_steps=max_steps, dt=DT,
                                           every=pc.COMPACT_EVERY, **planes)
            # <<< code:eagle_compact
        else:
            # >>> code:eagle_reorder
            self.runner = eagle.until_done(eagle.deploy(oscillator_step),
                                           max_steps=max_steps, dt=DT,
                                           every=pc.COMPACT_EVERY,
                                           reorder=pc.REORDER_THETA, **planes)
            # <<< code:eagle_reorder
        return None

    def prepare(self):
        st = self.st
        for dst, key in ((st.omega, "omega"), (st.zeta, "zeta"), (st.nstop, "nstop"),
                         (st.x, "x0"), (st.v, "v0")):
            self.np.copyto(dst, self.inp[key])
        st.k.fill(0.0)
        self.runner.reset()

    def run(self):
        self.runner.run()  # a reordering run ends with the planes in sample order

    def download(self):
        return self.st.x.copy(), self.st.v.copy(), self.st.k.copy()

    def facts(self):
        import hawk
        import hawk.compile as hc

        so, entry, sources = _host_build_facts(self.kernel)
        return {"variant": self.variant,
                "simd": _objdump_mix(so, entry),
                # bare name, never the absolute discovery path (leaks the
                # build box; same reason _hawk_host_flags drops -I/-isystem)
                "compiler": cc.tool_name(hc.host_compiler()),
                "compile_flags": _hawk_host_flags(),
                "host_profile": _hawk_host_profile(),
                "kernel_sources_md5": sources,
                "loop": "eagle.until_done (host team), the kernel's own finished count",
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "hawk_commit": cc.git_commit(hawk, short=True)}


class _EagleHostCompactArm(_EagleHostArm):
    variant = "compact"


class _EagleHostReorderArm(_EagleHostArm):
    variant = "reorder"


class _NumbaArm(_Arm):
    def init(self):
        import numpy as np

        self.fn = _make_numba()
        a = np.ones(WARM_N)
        t0 = time.perf_counter()
        self.fn(a, a, a, a, a, DT)
        self.compile_s = time.perf_counter() - t0

    def run(self):
        i = self.inp
        self.out = self.fn(i["omega"], i["zeta"], i["x0"], i["v0"], i["nstop"], DT)

    def download(self):
        return self.out

    def facts(self):
        import numba

        asm = "\n".join(self.fn.inspect_asm().values())
        return {"simd": _mnemonic_mix(asm), "threading_layer": numba.threading_layer(),
                "num_threads": numba.get_num_threads(), "fastmath": False}


class _JaxArm(_Arm):
    def init(self):
        import jax

        jax.config.update("jax_enable_x64", True)
        self.first_compile_s = {}

    def make_fn(self, cap):
        return _make_jax_fn(cap)

    def compile_for(self, n):
        """Compile for shape (n,) and the step cap from abstract shapes alone
        (no arrays); a compile before any cell's setup uses a cap of 1."""
        import jax
        import jax.numpy as jnp

        cap = getattr(self, "max_steps", 1)
        self.fn = self.make_fn(cap)
        self.compiled_cap = cap
        spec = jax.ShapeDtypeStruct((n,), jnp.float64)
        t0 = time.perf_counter()
        self.compiled = self.fn.lower(spec, spec, spec, spec, spec).compile()
        self.compiled_n = n
        seconds = time.perf_counter() - t0
        # JAX caches executables: a later compile at the same N is a cache hit,
        # so the card reports the first (cold) compile at each N
        self.first_compile_s.setdefault(n, seconds)
        return seconds

    def setup(self, inp, max_steps):
        import jax.numpy as jnp

        super().setup(inp, max_steps)
        n = len(inp["x0"])
        if (getattr(self, "compiled_n", None), getattr(self, "compiled_cap", None)) != (n, max_steps):
            self.compile_for(n)
        cell_compile = self.first_compile_s[n]
        self.args = tuple(jnp.asarray(self.inp[k])
                          for k in ("omega", "zeta", "x0", "v0", "nstop"))
        return cell_compile

    def run(self):
        import jax

        self.out = jax.block_until_ready(self.compiled(*self.args))

    def download(self):
        import numpy as np

        return tuple(np.asarray(a) for a in self.out)

    def facts(self):
        import jax

        return {"devices": [str(d) for d in jax.devices()],
                "formulation": ("lax.while_loop per sample (stop when the step "
                                "count reaches the sample's stop step), vmap over "
                                "samples, jit; compiled once per N, timed apart")}


class _TorchArm(_Arm):
    def init(self):
        import torch

        torch.set_num_threads(THREADS)
        self.torch = torch

    def setup(self, inp, max_steps):
        super().setup(inp, max_steps)
        t = self.torch
        self.args = {k: t.from_numpy(v) for k, v in self.inp.items()}

    # >>> code:torch_masked
    def run(self):
        torch = self.torch
        a = self.args
        with torch.inference_mode():
            omega, zeta, x, v, nstop = a["omega"], a["zeta"], a["x0"], a["v0"], a["nstop"]
            k = torch.zeros_like(x)
            nw2 = -(omega * omega)
            cc = (2.0 * zeta) * omega
            for _ in range(self.max_steps):
                running = k < nstop
                if not bool(running.any()):
                    break
                a1 = nw2 * x - cc * v
                v2 = v + H2 * a1
                a2 = nw2 * (x + H2 * v) - cc * v2
                v3 = v + H2 * a2
                a3 = nw2 * (x + H2 * v2) - cc * v3
                v4 = v + DT * a3
                a4 = nw2 * (x + DT * v3) - cc * v4
                xn = x + H6 * (((v + 2.0 * v2) + 2.0 * v3) + v4)
                vn = v + H6 * (((a1 + 2.0 * a2) + 2.0 * a3) + a4)
                x = torch.where(running, xn, x)
                v = torch.where(running, vn, v)
                k = k + running
        self.out = (x, v, k)
    # <<< code:torch_masked

    def download(self):
        return tuple(t.numpy().copy() for t in self.out)

    def facts(self):
        return {"num_threads": self.torch.get_num_threads(),
                "parallel_info": self.torch.__config__.parallel_info().splitlines()[:6]}


class _NumpyArm(_Arm):
    def run(self):
        i = self.inp
        self.out = _masked_numpy(i["omega"], i["zeta"], i["x0"], i["v0"], i["nstop"],
                                 self.max_steps)

    def download(self):
        return self.out

    def facts(self):
        try:
            from numpy._core._multiarray_umath import __cpu_features__ as feats
            simd = sorted(k for k, on in feats.items() if on and k.startswith("AVX512"))
        except ImportError:
            simd = []
        return {"numpy_cpu_features_avx512": simd}


class _MpArm(_Arm):
    def init(self):
        import multiprocessing as mp

        t0 = time.perf_counter()
        self.pool = mp.get_context("fork").Pool(THREADS)
        self.pool.map(abs, range(THREADS))
        self.compile_s = time.perf_counter() - t0  # pool start-up

    # >>> code:mp_numpy
    def run(self):
        import numpy as np

        i = self.inp
        bounds = np.linspace(0, len(i["x0"]), THREADS + 1).astype(int)
        chunks = [tuple(i[k][a:b] for k in ("omega", "zeta", "x0", "v0", "nstop"))
                  + (self.max_steps,) for a, b in zip(bounds[:-1], bounds[1:])]
        res = self.pool.map(_mp_chunk, chunks)
        self.out = tuple(np.concatenate([r[j] for r in res]) for j in range(3))
        self.child_cpu_s = sum(r[3] for r in res)
        self.child_peak_bytes = sum(r[4] for r in res)
    # <<< code:mp_numpy

    def download(self):
        return self.out

    def facts(self):
        return {"processes": THREADS, "start_method": "fork",
                "chunking": "8 contiguous chunks, pickled to and from the pool (in the wall)"}


class _PythonArm(_Arm):
    def run(self):
        i = self.inp
        self.out = _python_loops(i["omega"], i["zeta"], i["x0"], i["v0"], i["nstop"])

    def download(self):
        import numpy as np

        return tuple(np.asarray(a, dtype=np.float64) for a in self.out)


class _JaxShardArm(_JaxArm):
    """JAX over all 8 CPU devices: the batch sharded along one mesh axis,
    each shard running the vmapped per-sample while loop on its own
    (``shard_map``), so no step synchronises across shards."""

    def init(self):
        import jax
        from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

        jax.config.update("jax_enable_x64", True)
        devs = jax.devices("cpu")
        self.mesh = Mesh(devs, ("s",))
        self.sharding = NamedSharding(self.mesh, P("s"))
        self.devices = len(devs)
        self.first_compile_s = {}

    def make_fn(self, cap):
        import jax
        from jax.sharding import PartitionSpec as P

        # >>> code:jax_shard8
        try:
            from jax import shard_map
        except ImportError:
            from jax.experimental.shard_map import shard_map
        body = _make_jax_fn(cap, jit=False)
        mapped = shard_map(body, mesh=self.mesh, in_specs=(P("s"),) * 5,
                           out_specs=(P("s"),) * 3)
        return jax.jit(mapped, in_shardings=(self.sharding,) * 5,
                       out_shardings=(self.sharding,) * 3)
        # <<< code:jax_shard8

    def compile_for(self, n):
        import jax
        import jax.numpy as jnp

        cap = getattr(self, "max_steps", 1)
        self.fn = self.make_fn(cap)
        self.compiled_cap = cap
        spec = jax.ShapeDtypeStruct((n,), jnp.float64, sharding=self.sharding)
        t0 = time.perf_counter()
        self.compiled = self.fn.lower(spec, spec, spec, spec, spec).compile()
        self.compiled_n = n
        seconds = time.perf_counter() - t0
        self.first_compile_s.setdefault(n, seconds)
        return seconds

    def setup(self, inp, max_steps):
        import jax

        cell_compile = super().setup(inp, max_steps)
        self.args = tuple(jax.device_put(a, self.sharding) for a in self.args)
        return cell_compile

    def facts(self):
        import jax

        return {"devices": [str(d) for d in jax.devices()],
                "xla_flags": os.environ.get("XLA_FLAGS"),
                "formulation": ("batch sharded over the CPU devices (NamedSharding, "
                                "one mesh axis); shard_map of vmap(lax.while_loop) "
                                "per shard, jit; compiled once per N, timed apart")}


def _hawk_host_flags():
    """hawk's host recipe as this hawk reports it (include paths dropped)."""
    import hawk.compile as hc

    try:
        flags, skip = [], False
        for f in hc.host_flags():
            if skip:
                skip = False
            elif f in ("-isystem", "-I", "-include"):
                skip = True
            elif not f.startswith(("-I", "-isystem")):
                flags.append(f)
        return flags
    except Exception as exc:  # a hawk whose recipe API differs
        return [f"unavailable: {exc}"]


def _hawk_host_profile():
    """hawk's host profile and opt level in effect, and the code-generation
    flags they produce, as this hawk reports them (``hawk.compile``), so the
    card records the flags the kernels were actually built with."""
    import hawk.compile as hc

    try:
        return {"profile": hc.host_profile(), "codegen_flags": hc.host_codegen_flags(),
                "profiles": list(hc.HOST_PROFILES),
                "opt_level": hc.opt_level(), "opt_levels": list(hc.OPT_LEVELS)}
    except Exception as exc:  # a hawk whose recipe API differs
        return {"error": f"unavailable: {exc}"}


#: The marked code blocks each arm's snippet is made of, as "file:block"
#: (the eagle arms run perf_card.py's hawk step kernels).
ARM_CODE = {
    **{a: ("perf_card.py:hawk_step", "cpu_card.py:eagle_term")
       for a in ("eagle_term8", "eagle_term1")},
    **{a: ("perf_card.py:hawk_kind", "perf_card.py:hawk_active", "cpu_card.py:eagle_compact")
       for a in ("eagle_compact8", "eagle_compact1")},
    **{a: ("perf_card.py:hawk_kind", "perf_card.py:hawk_active", "cpu_card.py:eagle_reorder")
       for a in ("eagle_reorder8", "eagle_reorder1")},
    "numba_prange": ("cpu_card.py:numba_prange",),
    "jax_shard8": ("cpu_card.py:jax_vmap", "cpu_card.py:jax_shard8"),
    "jax_vmap": ("cpu_card.py:jax_vmap",),
    "torch_masked": ("cpu_card.py:torch_masked",),
    "numpy_masked": ("cpu_card.py:numpy_masked",),
    "mp_numpy": ("cpu_card.py:numpy_masked", "cpu_card.py:mp_chunk", "cpu_card.py:mp_numpy"),
    "python_loops": ("cpu_card.py:python_loops",),
}


def _code_record(arms):
    blocks = {f"{pathlib.Path(f).name}:{k}": v
              for f in (pc.__file__, __file__) for k, v in pc._code_blocks(f).items()}
    return cc.arm_code(blocks, ARM_CODE, arms)


_ARM_CLASSES = {"eagle_term8": _EagleHostArm, "eagle_term1": _EagleHostArm,
                "eagle_compact8": _EagleHostCompactArm,
                "eagle_compact1": _EagleHostCompactArm,
                "eagle_reorder8": _EagleHostReorderArm,
                "eagle_reorder1": _EagleHostReorderArm,
                "jax_shard8": _JaxShardArm,
                "numba_prange": _NumbaArm, "jax_vmap": _JaxArm,
                "torch_masked": _TorchArm, "numpy_masked": _NumpyArm,
                "mp_numpy": _MpArm, "python_loops": _PythonArm}


# --------------------------------------------------------------------------- #
# Process-level measurement helpers
# --------------------------------------------------------------------------- #
_MNEMONIC = re.compile(r"\b(v?(?:add|sub|mul|div|fn?madd\d*|fn?msub\d*)([sp])d)\b")


def _mnemonic_mix(asm):
    scalar, packed = {}, {}
    for m in _MNEMONIC.finditer(asm):
        (packed if m.group(2) == "p" else scalar)[m.group(1)] = \
            (packed if m.group(2) == "p" else scalar).get(m.group(1), 0) + 1
    return {"scalar_double_ops": dict(sorted(scalar.items())),
            "packed_double_ops": dict(sorted(packed.items())),
            "vectorised": bool(packed)}


def _objdump_mix(so, symbol):
    try:
        out = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(so)],
                             capture_output=True, text=True, timeout=60, check=True).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"objdump failed: {exc}"}
    m = re.search(rf"<{re.escape(symbol)}>:\n(.*?)(?:\n\n|\Z)", out, re.S)
    mix = _mnemonic_mix(m.group(1) if m else "")
    mix["source"] = f"objdump -d of {so.name}, function {symbol}"
    return mix


def _rss_kib():
    return cc.status_kib("VmRSS")


def _peak_kib():
    # ru_maxrss (KiB on Linux): after _reset_peak it is the peak since the reset
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


_reset_peak = cc.reset_host_peak


def _cpu_s():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime + r.ru_stime


# --------------------------------------------------------------------------- #
# Worker (one per arm, persistent across a run) and memory probe
# --------------------------------------------------------------------------- #
def _worker(arm):
    reply_fd = cc.probe_reply_fd()  # library prints go to stderr
    reply = os.fdopen(reply_fd, "w", buffering=1)
    impl = _ARM_CLASSES[arm]()
    try:
        impl.init()
        init_error = None
    except Exception:  # reported on the first command
        init_error = traceback.format_exc()
    import numpy as np

    for line in sys.stdin:
        cmd = json.loads(line)
        try:
            if init_error:
                raise RuntimeError(init_error)
            op = cmd["cmd"]
            if op == "quit":
                break
            if op == "facts":
                out = {"facts": impl.facts(), "compile_s": impl.compile_s,
                       "versions": _versions()}
            elif op == "setup":
                inp = pc._inputs(cmd["n"], cmd["distribution"], cmd["max_steps"])
                out = {"cell_compile_s": impl.setup(inp, cmd["max_steps"])}
            elif op == "calibrate":
                r0 = time.perf_counter()
                while time.perf_counter() - r0 < RAMP_S:
                    impl.prepare()
                    impl.run()
                runs, spent = 0, 0.0
                while spent < CALIBRATE_S or runs == 0:
                    impl.prepare()
                    t0 = time.perf_counter()
                    impl.run()
                    spent += time.perf_counter() - t0
                    runs += 1
                out = {"wall": spent / runs}
            elif op == "run":
                wall = cpu = 0.0
                inner = int(cmd.get("inner", 1))
                if cmd.get("ramp"):
                    # after the driver's settle the clock idles low: run untimed for
                    # RAMP_S (at least one run) so the timed block starts ramped
                    r0 = time.perf_counter()
                    while True:
                        impl.prepare()
                        impl.run()
                        if time.perf_counter() - r0 >= RAMP_S:
                            break
                mhz = _cpu_mhz_now()
                for _ in range(inner):
                    impl.prepare()
                    c0 = _cpu_s()
                    t0 = time.perf_counter()
                    impl.run()
                    wall += time.perf_counter() - t0
                    cpu += _cpu_s() - c0 + impl.child_cpu_s
                wall, cpu = wall / inner, cpu / inner
                if cmd.get("save"):
                    x, v, k = impl.download()
                    np.savez(cmd["save"], x=x, v=v, k=k)
                out = {"wall": wall, "cpu_s": cpu, "mhz": mhz}
            else:
                raise ValueError(f"unknown command {op!r}")
        except Exception:
            out = {"error": traceback.format_exc()}
        reply.write(json.dumps(out) + "\n")


def _memory_probe(arm, distribution, max_steps, n):
    """One fresh process: imports, warm-up compile on a tiny batch (and, for
    JAX, the compile at shape N from abstract shapes), baseline, then one run
    of the full batch (inputs generated inside the measured span)."""
    reply_fd = cc.probe_reply_fd()
    rss_start = _rss_kib()
    impl = _ARM_CLASSES[arm]()
    impl.init()
    impl.setup(pc._inputs(WARM_N, distribution, max_steps), max_steps)
    impl.prepare()
    impl.run()
    impl.download()
    if arm in ("jax_vmap", "jax_shard8"):
        impl.compile_for(n)
    impl.inp = impl.out = impl.args = impl.arm = None  # drop warm-up arrays
    import gc

    gc.collect()
    rss_base = _rss_kib()
    reset_ok = _reset_peak()
    inp = pc._inputs(n, distribution, max_steps)
    impl.setup(inp, max_steps)
    impl.prepare()
    impl.run()
    impl.download()
    peak = _peak_kib()
    out = {"rss_start_bytes": rss_start * 1024, "rss_baseline_bytes": rss_base * 1024,
           "ru_maxrss_bytes": peak * 1024, "peak_reset": reset_ok,
           "peak_bytes": max(0, peak - rss_base) * 1024 + impl.child_peak_bytes,
           "child_peak_bytes": impl.child_peak_bytes,
           "compile_overhead_bytes": max(0, rss_base - rss_start) * 1024}
    cc.probe_reply(reply_fd, out)


#: The compile-time pass: (arm, key) pairs, each timed cold and warm.
COMPILE_REPS = 3
NO_COMPILE_ARMS = ("torch_masked", "numpy_masked", "mp_numpy", "python_loops")
#: Arms that run another arm's build (same kernel, different thread count or
#: loop around it).
SHARED_BUILD_ARMS = {"eagle_term1": "eagle_term8", "eagle_compact1": "eagle_compact8",
                     "eagle_reorder8": "eagle_compact8", "eagle_reorder1": "eagle_compact8"}


def _compile_keys(ns):
    keys = [("eagle_term8", "step", "hawk step kernel (host build)"),
            ("eagle_compact8", "step_active_set",
             "hawk active-set step kernel (host build)")]
    keys.append(("numba_prange", "numba", "Numba parallel loop (first call, 64 samples)"))
    for arm in ("jax_shard8", "jax_vmap"):
        for n in ns:
            keys.append((arm, f"jax:{n}", f"XLA executable for N = {n:,}"))
    return keys


def _compile_probe(arm, key):
    """One fresh process: imports (untimed), then the time from the first call
    to a ready kernel (trace + codegen + compile, or a cache hit)."""
    reply_fd = cc.probe_reply_fd()
    if arm == "numba_prange":
        import numpy as np

        import numba  # noqa: F401  (import untimed)
        a = np.ones(WARM_N)
        t0 = time.perf_counter()
        _make_numba(cache=True)(a, a, a, a, a, DT)
        seconds = time.perf_counter() - t0
    elif arm in ("jax_vmap", "jax_shard8"):
        cc.jax_persistent_cache(os.environ["CPU_CARD_JAX_CACHE"])
        impl = _ARM_CLASSES[arm]()
        impl.init()
        n = int(key.split(":")[1])
        t0 = time.perf_counter()
        impl.compile_for(n)
        seconds = time.perf_counter() - t0
    else:
        import eagle
        import eagle.plan  # noqa: F401  (imports untimed)
        import hawk  # noqa: F401
        import hawk.artifact  # noqa: F401
        import hawk.compile  # noqa: F401

        t0 = time.perf_counter()
        eagle.deploy(pc._define_kernel() if key == "step" else pc._define_active_kernel())
        seconds = time.perf_counter() - t0
    cc.probe_reply(reply_fd, {"seconds": seconds})


def _run_compile_pass(ns):
    rows = []
    for arm, key, what in _compile_keys(ns):
        env = dict(os.environ)
        env.update(_arm_env(arm))
        cold, warm = cc.cold_warm(
            __file__, ["--compile-probe", arm, key],
            {"HAWK_CACHE_DIR": "hawk", "NUMBA_CACHE_DIR": "numba",
             "CPU_CARD_JAX_CACHE": "jax"},
            reps=COMPILE_REPS, prefix="cpu_card_cache_", base_env=env,
            what=f"compile probe {arm} {key}")
        rows.append({"arm": arm, "key": key, "what": what,
                     "cold_s": cc.median_iqr(cold), "warm_s": cc.median_iqr(warm)})
        print(f"compile {arm:14s} {key:12s} cold {statistics.median(cold):.3g}s "
              f"warm {statistics.median(warm):.3g}s", flush=True)
    return rows


_COMPILE_METHOD = (
    "Its own pass, never part of any wall. Each measurement is a fresh process: "
    "the imports run untimed, then the timer spans the first call to a ready "
    "kernel (trace + code generation + compile, or the cache lookup). Cold: an "
    "empty cache (hawk's compile cache pointed at a fresh directory through "
    "HAWK_CACHE_DIR; Numba with cache=True into a fresh NUMBA_CACHE_DIR; JAX "
    "with its persistent compilation cache in a fresh directory). Warm: a new "
    "process on the cache the first cold run populated (hawk cache hit; Numba's "
    "on-disk cache; JAX's persistent cache with the minimum compile time and "
    "entry size set to 0). hawk: eagle.deploy of the kernel (tracing, C++ "
    "emission, the g++ host build, eagle's plan); the active-set variant is a "
    "second build of the same step. Numba: the first call on a 64-sample batch "
    "(compile plus a negligible run). JAX: lower + compile from abstract shapes, "
    "once per N (XLA specialises on the shape). Medians of 3 cold and 3 warm "
    "processes.")


def _versions():
    out = {"python": platform.python_version()}
    for mod in ("numpy", "numba", "jax", "jaxlib", "torch", "hawk"):
        m = sys.modules.get(mod)
        if m is not None:
            out[mod] = getattr(m, "__version__", None)
    return out


class _Worker(cc.LineWorker):
    def __init__(self, arm):
        env = dict(os.environ)
        env.update(_arm_env(arm))
        super().__init__(__file__, ["--worker", arm], env, arm)
        self.info = self.call({"cmd": "facts"})

    def close(self):
        super().close({"cmd": "quit"})


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _verify(outs, nstop):
    """Every arm reaches the stop steps exactly; x, v agree with eagle_term8."""
    return cc.verify_xvk(outs, REFERENCE_ARM, nstop, pc.ATOL)


def _run_cell(workers, arms, distribution, max_steps, n, reps, work):
    import numpy as np

    inp = pc._inputs(n, distribution, max_steps)
    sample_steps = int(pc._active_profile(inp["nstop"], max_steps).sum())
    here = [a for a in arms if _included(a, n, distribution, max_steps) is None]
    dropped = [{"arm": a, "n": n, "distribution": distribution, "max_steps": max_steps,
                "reason": _included(a, n, distribution, max_steps)}
               for a in arms if a not in here]
    load_before = os.getloadavg()
    cell_compile = {}
    for a in here:
        cell_compile[a] = workers[a].call({"cmd": "setup", "n": n,
                                           "distribution": distribution,
                                           "max_steps": max_steps})["cell_compile_s"]
    # warm-up + verification (excluded from timing)
    outs, inner = {}, {}
    for a in here:
        path = work / f"{a}.npz"
        warm = workers[a].call({"cmd": "run", "save": str(path)})["wall"]
        # the count comes from a calibration block (ramped, then back-to-back runs
        # for CALIBRATE_S): a first run in a process may carry one-off set-up, and a
        # single run on a cold clock reads slow, both of which leave the block short
        warm = workers[a].call({"cmd": "calibrate"})["wall"]
        inner[a] = int(max(1, min(MAX_INNER, -(-MIN_TIMED_S // max(warm, 1e-9)))))
        with np.load(path) as z:
            outs[a] = (z["x"], z["v"], z["k"])
        path.unlink()
    worst = _verify(outs, inp["nstop"])
    del outs
    walls = {a: [] for a in here}
    busy = {a: [] for a in here}
    settle_busy = {a: [] for a in here}
    mhz = {a: [] for a in here}
    for rep in range(reps):
        order = here[rep % len(here):] + here[:rep % len(here)]
        for a in order:
            j0 = _cpu_jiffies()
            time.sleep(SETTLE_S)
            j1 = _cpu_jiffies()
            settle_busy[a].append(_busy_cores(j0, j1))
            r = workers[a].call({"cmd": "run", "inner": inner[a], "ramp": True})
            walls[a].append(r["wall"])
            busy[a].append(r["cpu_s"] / r["wall"])
            mhz[a].append(r["mhz"])
    load_after = os.getloadavg()
    rows = []
    for a in here:
        wall = cc.median_iqr(walls[a])
        med = wall["median"]
        flops = pc.FLOPS_PER_SAMPLE_STEP * sample_steps
        arm_compile = workers[a].info["compile_s"]
        rows.append({
            "arm": a, "n": n, "distribution": distribution, "max_steps": max_steps,
            "threads": ARM_THREADS[a], "wall_s": wall, "inner_runs": int(inner[a]),
            "cpu_busy": statistics.median(busy[a]),
            "busy_cores_during_settle": settle_busy[a],
            "cpu_mhz_at_timer_start": (statistics.median(mhz[a])
                                       if all(m is not None for m in mhz[a]) else None),
            "sample_steps": sample_steps, "sample_steps_per_s": sample_steps / med,
            "useful_flops": flops, "useful_flops_per_s": flops / med,
            "compile_s": cell_compile[a] if cell_compile[a] is not None else arm_compile,
            "compile_scope": ("first compile for this cell's N (JAX)"
                              if cell_compile[a] is not None else
                              ("once per process" if arm_compile is not None else None)),
            "max_abs_diff_vs_reference": worst[a],
        })
    return {"rows": rows, "dropped": dropped,
            "load_avg_before": load_before, "load_avg_after": load_after,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def _run_memory(arm, distribution, max_steps, n):
    env = dict(os.environ)
    env.update(_arm_env(arm))
    return cc.run_probe(__file__, ["--memory-probe", arm, distribution, str(max_steps),
                                   str(n)], env=env,
                        what=f"memory probe {arm} {distribution} {max_steps} {n}")


def _min_bytes(n):
    return n * (STATE_WIDTH + PER_SAMPLE_SCALARS) * 8


def _cpufreq_facts():
    """The frequency governor and its limits (cpu0, sysfs), for reading the
    card's clock readings against."""
    base = pathlib.Path("/sys/devices/system/cpu/cpu0/cpufreq")
    out = {}
    for key, name, scale in (("scaling_governor", "scaling_governor", None),
                             ("cpuinfo_min_mhz", "cpuinfo_min_freq", 1000.0),
                             ("cpuinfo_max_mhz", "cpuinfo_max_freq", 1000.0)):
        try:
            v = (base / name).read_text().strip()
            out[key] = float(v) / scale if scale else v
        except (OSError, ValueError):
            out[key] = None
    return out


def _cpu_facts():
    info = {}
    try:
        for line in subprocess.run(["lscpu"], capture_output=True, text=True,
                                   timeout=30).stdout.splitlines():
            k, _, v = line.partition(":")
            info[k.strip()] = v.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    model = cc.cpu_model() or "unknown CPU"
    flags = set()
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("flags"):
                flags = set(line.split(":", 1)[1].split())
                break
    except OSError:
        pass
    cores = int(info.get("Core(s) per socket", "0")) * int(info.get("Socket(s)", "1"))
    logical = os.cpu_count()
    tpc = int(info.get("Thread(s) per core", "1"))
    m = re.search(r"@\s*([\d.]+)\s*GHz", model)
    base_hz = float(m.group(1)) * 1e9 if m else float(info.get("CPU max MHz", "0")) * 1e6
    simd = 8 if "avx512f" in flags else (4 if "avx" in flags else 2)
    fma = "fma" in flags
    flop_per_lane = 2 if fma else 1
    peak = cores * FMA_UNITS_PER_CORE * simd * flop_per_lane * base_hz
    peak_scalar = cores * FMA_UNITS_PER_CORE * flop_per_lane * base_hz
    clean = re.sub(r"\((R|TM)\)|\bCPU\b|@.*$", "", model)
    slug = re.sub(r"[^a-z0-9]+", "-", clean.lower()).strip("-")
    cpu = {"model": model, "cores": cores, "logical_cpus": logical,
           "threads_per_core": tpc, "base_clock_hz": base_hz,
           "max_clock_mhz": info.get("CPU max MHz"), "simd_doubles": simd,
           "fma": fma, "isa_flags": sorted(f for f in flags if f.startswith(("avx", "fma", "sse"))),
           **_cpufreq_facts()}
    peak_d = {
        "fp64_flops": peak, "fp64_flops_scalar": peak_scalar,
        "assumption": (f"{cores} cores × {FMA_UNITS_PER_CORE} FMA units × {simd} "
                       f"float64 lanes × {flop_per_lane} FLOP per FMA × "
                       f"{base_hz / 1e9:.3g} GHz base clock"),
        "source": ("cores and ISA from lscpu and /proc/cpuinfo; base clock from the "
                   "model name; FMA units per core assumed (Intel lists 2 AVX-512 "
                   "FMA units for the Xeon W-2125); turbo can exceed the base clock "
                   "and wide-vector load can run below it"),
    }
    return cpu, peak_d, slug


def _simd_summary(arm_info):
    out = []
    e = arm_info.get(REFERENCE_ARM, {}).get("facts", {})
    s = e.get("simd", {})
    if s and "error" not in s:
        sc = ", ".join(f"{k} ×{v}" for k, v in s["scalar_double_ops"].items()) or "none"
        pk = ", ".join(f"{k} ×{v}" for k, v in s["packed_double_ops"].items()) or "none"
        flags = " ".join(e.get("compile_flags", []))
        prof = (e.get("host_profile") or {}).get("profile")
        out.append(
            "eagle's host build of the step ("
            + (f"hawk host profile `{prof}`: " if prof else "") + f"{flags}"
            + ("" if "-march" in flags else ", no -march") + ") "
            + ("is vectorised" if s["vectorised"] else "is not vectorised")
            + f": its entry function holds scalar float64 instructions ({sc}) and "
            f"packed ones ({pk})"
            + ("." if s["vectorised"] else
               "; the 8-thread speed-up over 1 thread comes from the OpenMP team."))
    ac = arm_info.get("eagle_compact8", {}).get("facts", {}).get("simd", {})
    if ac and "error" not in ac:
        pk = ", ".join(f"{k} ×{v}" for k, v in ac["packed_double_ops"].items()) or "none"
        out.append("The active-set build of the step (compaction and reorder arms) "
                   + ("is vectorised" if ac["vectorised"] else "is not vectorised")
                   + f" (packed float64 instructions: {pk}).")
    commits = {i.get("facts", {}).get("hawk_commit") for i in arm_info.values()} - {None}
    if commits:
        out.append(f"hawk at commit {', '.join(sorted(commits))}.")
    nb = arm_info.get("numba_prange", {}).get("facts", {}).get("simd", {})
    if nb:
        out.append(
            "Numba's compiled loop "
            + ("uses packed float64 instructions" if nb["vectorised"] else
               "uses scalar float64 instructions only (each sample's steps are a "
               "dependent chain)")
            + ".")
    out.append("NumPy and PyTorch use their libraries' own SIMD kernels for each "
               "array operation; JAX's XLA code was not inspected.")
    return out


def _parts_key(distribution, max_steps, n):
    return f"{distribution}-{max_steps}-{n}"


def _merge(parts_dir, configs, ns, reps, slug, cpu, peak, out_dir, allow_partial):
    results, dropped, cells, arm_info, memory = [], [], [], {}, {}
    missing = []
    reference_reruns = []
    for d, s in configs:
        for n in ns:
            ref_walls = []
            for g in range(len(ARM_GROUPS)):
                key = f"{_parts_key(d, s, n)}_g{g}"
                p = parts_dir / f"cell_{key}.json"
                if not p.is_file():
                    missing.append(key)
                    continue
                part = json.loads(p.read_text())
                if part["reps"] != reps:
                    missing.append(key + f" (reps {part['reps']})")
                    continue
                for row in part["cell"]["rows"]:
                    if row["arm"] == REFERENCE_ARM:
                        ref_walls.append(row["wall_s"]["median"])
                        if g != 0:
                            continue
                    results.append(row)
                dropped += part["cell"]["dropped"]
                cells.append({"distribution": d, "max_steps": s, "n": n, "group": g,
                              **{k: part["cell"][k] for k in
                                 ("load_avg_before", "load_avg_after", "utc")}})
                arm_info.update(part["arm_info"])
            if len(ref_walls) > 1:
                reference_reruns.append({"distribution": d, "max_steps": s, "n": n,
                                         "reference_wall_medians_by_group": ref_walls,
                                         "max_over_min": max(ref_walls) / min(ref_walls)})
            for a in ARMS:
                if _included(a, n, d, s) is not None:
                    continue
                mp_ = parts_dir / f"memory_{_parts_key(d, s, n)}_{a}.json"
                if mp_.is_file():
                    memory.setdefault(_parts_key(d, s, n), []).append(
                        json.loads(mp_.read_text()))
                else:
                    missing.append(f"memory {_parts_key(d, s, n)} {a}")
    comp_path = parts_dir / "compile.json"
    compile_rows = json.loads(comp_path.read_text()) if comp_path.is_file() else []
    if not compile_rows:
        missing.append("compile.json")
    if missing and not allow_partial:
        print("card not written; missing parts: " + ", ".join(missing))
        return None
    for r in results:
        r["fraction_of_peak_fp64"] = r["useful_flops_per_s"] / peak["fp64_flops"]
    ns_present = sorted({r["n"] for r in results})
    cfg_present = [{"distribution": d, "max_steps": s} for d, s in configs
                   if any((r["distribution"], r["max_steps"]) == (d, s) for r in results)]
    arms = [a for a in ARMS if any(r["arm"] == a for r in results)]
    mem_rows, comp = [], {}
    for key, rows in memory.items():
        for r in rows:
            mem_rows.append(r)
    for r in sorted(mem_rows, key=lambda r: r["n"]):
        comp[r["arm"]] = r["compile_overhead_bytes"]
    card = {
        "schema": "eagle-cpu-card/2",
        "toolchain": cc.toolchain(),
        "cpu_slug": slug,
        "script": "benchmarks/perf_card/cpu_card.py",
        "script_md5": cc.md5(__file__),
        "workload_script": "benchmarks/perf_card/perf_card.py",
        "workload_script_md5": cc.md5(pc.__file__),
        "common_md5": cc.md5(cc.__file__),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cpu": cpu, "peak": peak,
        "platform": platform.platform(),
        "arm_info": arm_info,
        "workload": {
            "problem": ("damped harmonic oscillator x'' + 2 zeta omega x' + "
                        "omega^2 x = 0, classical RK4, fixed step (perf_card.py)"),
            "dt": pc.DT, "precision": "float64", "seed": pc.SEED,
            "flops_per_sample_step": pc.FLOPS_PER_SAMPLE_STEP,
            "counting_rule": pc._COUNTING_RULE["flops"],
        },
        "configs": cfg_present,
        "ns": ns_present,
        "arms": arms,
        "arm_labels": {a: ARM_LABELS[a] for a in arms},
        "code": _code_record(arms),
        "method": {
            "repetitions": reps,
            "statistic": "median, interquartile range (inclusive quartiles)",
            "warm_up": "one run per arm and cell, excluded; it is also the verification run",
            "interleaving": "the arm order rotates by one every repetition",
            "settle": (f"{SETTLE_S:g} s pause before every timed repetition, so the "
                       "previous arm's thread pools stop spin-waiting first, then "
                       f"{RAMP_S:g} s of untimed runs of the arm itself, so the timed "
                       "block starts on a ramped clock (the machine's busy cores during "
                       "the pause and the clock at the timer start are recorded per row)"),
            "inner_runs": (f"a repetition whose run is shorter than {MIN_TIMED_S:g} s "
                           f"repeats it back to back until the repetition spans at "
                           f"least {MIN_TIMED_S:g} s (count fixed per arm and cell from "
                           f"a {CALIBRATE_S:g} s calibration block of back-to-back runs after the ramp, at most {MAX_INNER}; recorded per row as "
                           "inner_runs) and records the mean per run; eagle's untimed "
                           "input copy runs between the repeats"),
            "isolation": ("every arm runs in its own persistent worker process with "
                          "its thread pools pinned by environment and API; only one "
                          "arm runs at a time"),
            "wall": ("the integration alone, inputs already in the arm's own arrays "
                     "or tensors (eagle: copied into its bound arrays before the "
                     "timer; multiprocessing: chunk transfer to and from the pool is "
                     "inside the timer; plain Python: list conversion inside)"),
            "cpu_busy": "process CPU seconds (getrusage, plus pool processes) / wall",
            "compile": ("eagle: eagle.deploy of the host kernel, plain or active-set "
                        "(the hawk compile cache may serve it); Numba: first call "
                        "on a tiny batch; JAX: "
                        "lower + compile per N from abstract shapes; multiprocessing: "
                        "pool start-up; NumPy, PyTorch, Python: none"),
            "verification": (f"all arms reach identical per-sample step counts; x and v "
                             f"agree with {REFERENCE_ARM} within absolute {pc.ATOL:g}"),
            "threads": dict(ARM_THREADS),
            "load_avg_1m": ("per_cell rows already carry load_avg_before/after "
                            "(os.getloadavg()); eagle_term8/eagle_term1 and "
                            "eagle_compact8/eagle_compact1/eagle_reorder8/"
                            "eagle_reorder1 are the 1-thread vs N-thread rows"),
            "lane_utilisation": "n/a on the host (device-only hook; see perf_card.py)",
            "nsys_counts": "n/a on the host (no nsys trace here)",
            "fixed_overhead": ("not re-measured as a dedicated cell on this card; "
                               "per_cell's inner_runs/wall already show per-call "
                               "overhead shrinking at small N"),
        },
        "results": results,
        "dropped": dropped,
        "arm_groups": [list(g) for g in ARM_GROUPS],
        "reference_reruns": reference_reruns,
        "per_cell": cells,
        "simd": {"summary": _simd_summary(arm_info)},
        "compile_time": {"method": _COMPILE_METHOD, "rows": compile_rows,
                         "none": list(NO_COMPILE_ARMS),
                         "shared": dict(SHARED_BUILD_ARMS)},
        "memory": {
            "method": (
                "A separate pass after the timing pass. Each (arm, N, configuration) "
                "runs in a fresh process: imports, then a warm-up run on a "
                f"{WARM_N}-sample batch (every JIT compiles here; JAX also compiles "
                "for shape N from abstract shapes), then the baseline (resident set "
                "size, VmRSS) and a reset of the process's peak mark "
                "(/proc/self/clear_refs, so ru_maxrss restarts from the baseline); "
                "then the batch's inputs are generated and one full run done. Peak "
                "memory = ru_maxrss at the end minus the baseline; for "
                "multiprocessing, plus each pool process's own peak above its "
                "baseline (shared pages counted once per process). Minimum = the "
                f"state the workload needs, N × ({STATE_WIDTH} state + "
                f"{PER_SAMPLE_SCALARS} per-sample scalars: omega, zeta, stop step) "
                "× 8 bytes; the factor is peak / minimum. Compile overhead = "
                "baseline minus the resident size before the arm's imports-and-"
                "warm-up. Below about 1 MiB the allocator reuses memory already resident at "
                "the baseline, so small-N peaks can read below the minimum."),
            "minimum_bytes_per_sample": (STATE_WIDTH + PER_SAMPLE_SCALARS) * 8,
            "rows": mem_rows,
            "compile_overhead_bytes": comp,
        },
    }
    card["fit"] = _fit_notes(card)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"cpu_card_{slug}.json"
    json_path.write_text(json.dumps(card, indent=1) + "\n")
    (out_dir / f"cpu_card_{slug}.md").write_text(render_markdown(card))
    print(f"wrote {json_path}" + (f" (PARTIAL: {len(missing)} parts missing, e.g. {missing[0]})" if missing else ""))
    return card


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--parts-dir", type=pathlib.Path, default=None)
    ap.add_argument("--reps", type=int, default=pc.REPS)
    ap.add_argument("--ns", type=int, nargs="+", default=list(pc.NS))
    ap.add_argument("--configs", nargs="+", default=[f"{d}:{s}" for d, s in pc.CONFIGS],
                    help="cells to run, e.g. spread:100 uniform:1000")
    ap.add_argument("--groups", type=int, nargs="+", default=list(range(len(ARM_GROUPS))),
                    choices=range(len(ARM_GROUPS)), help="arm groups to run (ARM_GROUPS)")
    ap.add_argument("--no-timing", action="store_true", help="skip the timing pass")
    ap.add_argument("--no-memory", action="store_true", help="skip the memory pass")
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--memory-probe", nargs=4, help=argparse.SUPPRESS)
    ap.add_argument("--compile-probe", nargs=2, help=argparse.SUPPRESS)
    ap.add_argument("--no-compile", action="store_true", help="skip the compile-time pass")
    args = ap.parse_args(argv)
    if args.worker:
        _worker(args.worker)
        return 0
    if args.compile_probe:
        _compile_probe(*args.compile_probe)
        return 0
    if args.memory_probe:
        a, d, s, n = args.memory_probe
        _memory_probe(a, d, int(s), int(n))
        return 0
    if args.reps < 5:
        ap.error("--reps must be >= 5")
    run_arms = [a for a in ARMS if any(a in ARM_GROUPS[g] for g in args.groups)]
    run_cfgs = [(c.split(":")[0], int(c.split(":")[1])) for c in args.configs]
    cpu, peak, slug = _cpu_facts()
    parts = args.parts_dir or (pathlib.Path(tempfile.gettempdir())
                               / f"eagle_cpu_card_parts_{slug}")
    parts.mkdir(parents=True, exist_ok=True)

    if not args.merge_only and not args.no_timing:
        work = pathlib.Path(tempfile.mkdtemp(prefix="cpu_card_run_"))
        workers = {}
        try:
            for a in run_arms:
                t0 = time.perf_counter()
                workers[a] = _Worker(a)
                print(f"worker {a:14s} ready ({time.perf_counter() - t0:.1f}s)", flush=True)
            arm_info = {a: w.info for a, w in workers.items()}
            for d, s in run_cfgs:
                for n, g in [(n, g) for n in args.ns for g in args.groups]:
                    t0 = time.perf_counter()
                    cell = _run_cell(workers, list(ARM_GROUPS[g]), d, s, n, args.reps, work)
                    (parts / f"cell_{_parts_key(d, s, n)}_g{g}.json").write_text(json.dumps(
                        {"reps": args.reps, "cell": cell,
                         "arm_info": {a: arm_info[a] for a in ARM_GROUPS[g]}}, indent=1))
                    best = min(cell["rows"], key=lambda r: r["wall_s"]["median"])
                    print(f"{d:8s} S={s:5d} N={n:>8,} g{g} fastest={best['arm']:14s} "
                          f"({time.perf_counter() - t0:.1f}s, load "
                          f"{cell['load_avg_before'][0]:.2f}->{cell['load_avg_after'][0]:.2f})",
                          flush=True)
        finally:
            for w in workers.values():
                w.close()
            import shutil

            shutil.rmtree(work, ignore_errors=True)
    if not args.merge_only and not args.no_compile:
        rows = _run_compile_pass(list(pc.NS) if not args.allow_partial else args.ns)
        (parts / "compile.json").write_text(json.dumps(rows, indent=1))
    if not args.merge_only and not args.no_memory:
        for d, s in run_cfgs:
            for n in args.ns:
                t0 = time.perf_counter()
                for a in run_arms:
                    if _included(a, n, d, s) is not None:
                        continue
                    r = _run_memory(a, d, s, n)
                    row = {"arm": a, "n": n, "distribution": d, "max_steps": s,
                           **r, "minimum_bytes": _min_bytes(n),
                           "overhead_factor": r["peak_bytes"] / _min_bytes(n)}
                    (parts / f"memory_{_parts_key(d, s, n)}_{a}.json").write_text(
                        json.dumps(row, indent=1))
                print(f"memory {d:8s} S={s:5d} N={n:>8,}  "
                      f"({time.perf_counter() - t0:.1f}s)", flush=True)
    _merge(parts, [(d, s) for d, s in pc.CONFIGS] if not args.allow_partial else run_cfgs,
           list(pc.NS) if not args.allow_partial else args.ns, args.reps, slug, cpu,
           peak, args.out_dir, args.allow_partial)
    return 0


if __name__ == "__main__":
    sys.exit(main())
