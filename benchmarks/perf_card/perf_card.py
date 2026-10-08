# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The eagle performance card: one workload, many ways of running it, one JSON.

The workload is a batch of independent damped harmonic oscillators, each
integrated with a fixed-step classical Runge-Kutta (RK4) scheme until its OWN
step count is reached. Every sample has its own frequency, damping ratio,
initial state and stop step, so the batch thins out as it runs: the
per-sample, irregular class (one thread per independent problem, own state,
own control flow, early termination).

The step is written once, as a hawk kernel (``oscillator_step`` below),
built with ``eagle.deploy`` and run through eagle several ways, and written as
each array library's users write it for the other arms (the "arms"):

``eagle_graph``
    The hawk kernel deployed (``eagle.deploy``) and run in one call
    (``eagle.until_done``): the step, which finishes its own samples (it
    assigns its ``Terminated`` mask once a sample reached its stop step and
    counts it), captured once as the body of a device-side loop (one CUDA
    graph with a WHILE conditional node), replayed with one launch. The plain
    ``@hawk.kernel`` decorator builds the automatic kernel, so each launch
    runs the steps eagle's policy picks (see ``eagle_graph_auto``), over the
    whole batch. A finished sample is skipped by the kernel's ``Terminated``
    guard; the loop ends on the device when every sample has finished.
``eagle_graph_compact``
    The same call with active-set compaction: the single step traced under
    a kind whose guard reads an index map (hawk ``Guard(active_set=True)``,
    ``@hawk.kernel(steps=1, kind=ACTIVE_SET)``), and every
    ``COMPACT_EVERY`` steps, when a sample finished since the last time, an
    ``eagle.ActiveSet`` recomputes the map of the samples still running (on the
    device, inside the same graph). The launch covers only the mapped samples,
    so live samples share warps instead of one live lane holding a warp.

    ``eagle_graph``'s loop checks its stop guard every step; this loop checks
    it once every ``COMPACT_EVERY`` steps (the cadence the compaction itself
    runs on). So the two rows in the main tables differ in BOTH the guard
    cadence and the active-set map -- two variables, not one. The
    "compaction cadence sweep" (:func:`_run_compaction_sweep`, rendered as
    its own table) holds the cadence fixed and varies only the map, by
    pairing ``eagle_graph_compact`` at a cadence K against a
    ``matched_cadence`` arm: the SAME plain step kernel as ``eagle_graph``,
    looped K times between guard checks, with no active-set map. The only
    structural difference within a pair is the map.
``eagle_graph_reorder``
    ``eagle_graph_compact`` plus the occasional physical reorder
    (``eagle.ActiveSet(mask, reorder=REORDER_THETA)``): when a compaction finds
    the live samples spread thin over the warps they occupy, a reorder moves
    them to the front of every per-sample plane (in place, inside the same
    graph, under a conditional node), so the map reads contiguous memory
    again. The run ends with every plane back in sample order
    (``ActiveSet.restore()``, its cost is in the wall).
``eagle_graph_steps``
    The same call on the step declared to take ``STEPS_PER_LAUNCH`` (16) steps
    per launch (``@hawk.kernel(steps=16)``, hawk's ``hawk.steps``): one launch
    runs each sample's own loop of up to 16 steps with its state in registers,
    and a sample leaves that loop at its own stop step, so the planes are read
    and written once per launch and the device loop checks its guard once per
    launch. Results are bit-identical to one step per launch on this kernel.
    K is fixed by hand for this arm: 16 runs within 1% of 64 on these
    workloads; ``eagle_graph_auto`` below picks it at run time.
``eagle_graph_steps_compact``
    ``hawk.steps`` of the active-set step (STEPS_PER_LAUNCH steps per
    launch, reading the active-set map), compacted every
    ``STEPS_COMPACT_EVERY`` (64) steps, i.e. every 4 launches
    (``until_done(every=...)`` counts steps).
``eagle_graph_auto``
    The same step declared ``@hawk.kernel(steps="auto")`` on the active-set
    kind, run by the same call: after each launch eagle's policy picks the
    next launch's steps (8 to 64, doubling while few samples finish and
    halving when many do) and compacts after every launch in which a sample
    finished, all on the device; the last launch takes exactly the steps
    left.
``eagle_simulate``
    The kernel of ``eagle_graph_auto`` run by ``eagle.simulate``'s bound form
    (``eagle.simulation``): the model, its state and its parameters by name;
    eagle deploys it and builds the same loop.
``eager_loop``
    The single step (the plain decorator's kernel's ``.step``: one step per
    launch, no steps-per-launch word, no loop in the kernel) launched from a
    Python loop, one launch per step, with a host read of the finished-sample
    count after every step: what launching every step yourself costs.
``cupy_masked``
    The same arithmetic written as CuPy array expressions over the whole
    batch, with ``cupy.where`` keeping finished samples unchanged, and a host
    check of "any sample still running" after every step.
``torch_masked``
    Eager PyTorch on the GPU: the same whole-batch masked step as
    ``cupy_masked`` written with tensors (``torch.where`` holds finished
    samples), under ``torch.inference_mode``, with a host check of "any
    sample still running" after every step.
``torch_graphed``
    The same masked step, the way PyTorch users remove launch overhead by
    hand: a block of ``GRAPH_BLOCK`` (16) steps captured once as a
    ``torch.cuda.CUDAGraph`` over static buffers (warmed up on a side stream
    before capture) and replayed, with one host check of "any sample still
    running" per replay. Finished samples are held by ``torch.where``, so the
    up to 15 steps a replay runs past the last stop change nothing.
``torch_compiled`` (optional)
    The same step under ``torch.compile(mode="reduce-overhead")`` (Inductor-
    fused kernels replayed as CUDA graphs), run only where Triton, Inductor's
    GPU code generator, runs (compute capability 7.0 or newer); elsewhere the
    card says it was not run.
``jax_vmap``
    JAX on the GPU in float64 (``jax_enable_x64``): a per-sample
    ``lax.while_loop`` batched with ``vmap`` and compiled with ``jit`` (the
    CPU card's formulation). Batched, the loop runs until the slowest sample
    stops; finished samples are held by the batched select. The whole loop is
    one compiled executable (no host round trip per step); it is compiled per
    N, before any timing.
``warp_kernel``
    NVIDIA Warp: an explicit per-thread ``@wp.kernel`` in float64, one thread
    per sample, each running its own RK4 steps in a ``while`` loop that ends
    at its own stop step (per-thread early exit), one launch per run.
``cpu_openmp``
    The same hawk kernel, the host side of the same ``eagle.deploy``
    (NumPy planes select it), run through eagle's OpenMP host team in one
    call (``eagle.run_until_done``).

All arms of the main pass do the same math in the same precision (float64) on
the same samples; their results are checked against each other before any timing
counts. Every arm is timed in its own block, one arm after the other in the
arm order, with no interleaving: the first two runs of an arm are a warm-up and
are excluded (eagle's automatic arms pick their device entry over those two
runs: the size rule, then a measured run of the other entry, keeping the
faster), then back-to-back untimed runs of that arm until RAMP_S = 0.5 s have
elapsed (at least one), then TIMED_RUNS = 5 timed runs. Every reported time is
the mean of the middle 3 of the 5 runs (the single highest and the single
lowest dropped), with the range of those 3; the median and interquartile range
of all 5 and every sample are in the JSON.

Before each arm's block a cool-down gate (:func:`_cool_down_gate`) waits, at
most GATE_MAX_S, until the GPU is within GATE_DELTA_C of the idle temperature
read at card start and reads idle; right after each timed run the SM clock and
the throttle reasons are read through NVML (outside the timed region) and
recorded per arm, with the wall time in SM clock cycles. Without NVML both
are skipped and recorded as such.

The FP32 pass (:func:`_run_fp32_pass`) repeats the oscillator in float32
for the three arms that compare across libraries: ``eagle_simulate``
(``scalar_type="float32"``), ``warp_kernel`` and ``jax_vmap``. It has its own
table (eagle's wall over Warp's, and the entry eagle settled on); each cell
checks the arms against each other before it is timed.

Two passes follow the timing pass, each in fresh processes: the MEMORY pass
(peak device memory, and peak host memory, of one run of every (arm, N,
configuration) above a baseline) and the COMPILE-TIME pass (every compile
step, cold on empty caches and warm on the populated ones). Neither is part
of any wall.

The run WRITES the card itself: ``card_<device-slug>.json`` (every number) and
``card_<device-slug>.md`` (the table and the "where each arm is fastest"
lines, rendered from the JSON by :func:`render_markdown`, which the docs page
includes and a test re-renders to prove the two have not drifted).

Reproduce (from the eagle repository root, in an environment with eagle,
hawk, CuPy, PyTorch (CUDA), JAX (CUDA) and a CUDA toolchain)::

    python benchmarks/perf_card/perf_card.py

``--quick`` runs a reduced matrix for a smoke test; ``--no-nsys`` skips the
kernel-time pass (Nsight Systems) and leaves those fields ``null``;
``--no-memory`` / ``--no-compile`` skip those passes (dry runs only: the
card's test requires both).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import sqlite3
import statistics
import subprocess
import sys
import tempfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))  # benchmarks/: the cards' shared helpers
import _card_common as cc  # noqa: E402

# --------------------------------------------------------------------------- #
# Workload definition (the numbers here are recorded in the card)
# --------------------------------------------------------------------------- #
DT = 0.01
SEED = 20261001
ARMS = ("eagle_graph", "eagle_graph_compact", "eagle_graph_reorder",
        "eagle_graph_steps", "eagle_graph_steps_compact", "eagle_graph_auto",
        "eagle_graph_persist",
        "eagle_simulate", "eager_loop",
        "cupy_masked", "torch_masked", "torch_graphed", "jax_vmap", "warp_kernel",
        "cpu_openmp")
#: Arms run only where their toolchain runs (see _optional_arms); a card that
#: did not run one names it, with the reason, under "arms_not_run".
OPTIONAL_ARMS = ("torch_compiled",)
#: Every arm in table order.
_AFTER_TORCH = ARMS.index("torch_graphed") + 1
ALL_ARMS = ARMS[:_AFTER_TORCH] + OPTIONAL_ARMS + ARMS[_AFTER_TORCH:]
GPU_ARMS = tuple(a for a in ALL_ARMS if a != "cpu_openmp")
EAGLE_GPU_ARMS = ARMS[:ARMS.index("eager_loop") + 1]
#: The arms built on the K-steps-per-launch kernel (STEPS_PER_LAUNCH).
STEPS_ARMS = ("eagle_graph_steps", "eagle_graph_steps_compact")
#: The arm built on the kernel that picks its steps per launch at run time
#: (``@hawk.kernel(steps="auto")``, active-set kind).
AUTO_ARM = "eagle_graph_auto"
#: The arm that forces the persist entry explicitly (the persistent
#: launch) -- "hawk + eagle persistent" in the
#: card table. Built with the SAME auto kernel as :data:`AUTO_ARM`, but
#: ``until_done``'s internal ``_fast_mode="persist"`` override (no public
#: API) forces the persist launch regardless of what the latency-regime
#: capacity would otherwise pick, so this row is comparable across every
#: N/S cell (including below capacity, where :data:`AUTO_ARM` would not
#: choose it).
PERSIST_ARM = "eagle_graph_persist"
#: The eagle arms that launch the single-step kernel once per step.
SINGLE_STEP_ARMS = ("eagle_graph", "eagle_graph_compact", "eagle_graph_reorder",
                    "eager_loop")
#: The arm that runs the automatic kernel through ``eagle.simulate``'s
#: bound form (``eagle.simulation``): the model, its state and its
#: parameters by name.
SIMULATE_ARM = "eagle_simulate"
ARM_LABELS = {
    "eagle_graph": "eagle graph (device loop)",
    "eagle_graph_compact": "eagle graph + compaction",
    "eagle_graph_reorder": "eagle graph + compaction + reorder",
    "eagle_graph_steps": "eagle graph, K fused steps per launch",
    "eagle_graph_steps_compact": "eagle graph, K fused steps per launch + compaction",
    "eagle_graph_auto": "eagle auto (eagle picks the launch mode)",
    "eagle_graph_persist": "hawk + eagle persistent",
    "eagle_simulate": "eagle.simulate (eagle picks the launch mode)",
    "eager_loop": "eager launch loop",
    "cupy_masked": "CuPy masked",
    "torch_masked": "PyTorch masked",
    "torch_graphed": "PyTorch, CUDA graph of 16 steps",
    "torch_compiled": "PyTorch torch.compile (reduce-overhead)",
    "jax_vmap": "JAX jit + vmap(while_loop)",
    "warp_kernel": "Warp per-thread kernel",
    "cpu_openmp": "CPU OpenMP",
}
#: Each arm's loop structure (rendered as the card's arm table).
ARM_LOOP = {
    "eagle_graph": "per launch over the batch, inside one CUDA graph: the plain decorator's kernel, which finishes its own samples and runs the steps per launch eagle's policy picks (8 to 64), guard checked on the device every launch",
    "eagle_graph_compact": "as the graph arm, the step launched over the active-set map only (recomputed every 16 steps)",
    "eagle_graph_reorder": "as the compaction arm, plus a physical reorder of the per-sample planes when the live samples spread thin",
    "eagle_graph_steps": "per launch over the batch, inside one CUDA graph: the step kernel runs up to K = 16 steps per sample with the state in registers, each sample leaving at its own stop step; guard checked on the device every launch",
    "eagle_graph_steps_compact": "as the K-steps arm, the launch over the active-set map only (recomputed every 64 steps = 4 launches)",
    "eagle_graph_auto": "the active-set kernel under eagle's automatic policy, which picks the launch mode from the batch: one launch running every sample to its end for a small batch, the persistent launch above that, and the compaction loop (active-set map, steps per launch picked on the device) only when the caller asks for compaction or reordering; the row records the mode taken",
    "eagle_graph_persist": "ONE launch, no graph, no map, no policy kernel: a grid sized from the SM count and the kernel's occupancy, each lane fetching base + atomicAdd(counter, 1) whenever its sample is done, running the whole budget's worth of steps per sample it picks up",
    "eagle_simulate": "as the automatic arm: eagle.simulate takes the kernel, its state and its parameters by name and eagle picks the launch mode the same way",
    "eager_loop": "per step over the batch: one kernel launch from Python, a host read of the finished count every step",
    "cupy_masked": "per step over the batch: whole-batch array expressions, finished samples held by cupy.where, a host check every step; the cap is batch-level (the outer step loop runs at most max_steps)",
    "torch_masked": "per step over the batch: whole-batch tensor expressions, finished samples held by torch.where, a host check every step; the cap is batch-level (the outer step loop runs at most max_steps)",
    "torch_graphed": "as PyTorch masked, 16 steps captured as one CUDA graph over static buffers, replayed with one host check per replay",
    "torch_compiled": "as PyTorch masked, the step compiled by Inductor and replayed as a CUDA graph, a host check every step",
    "jax_vmap": "per step over the batch, inside one compiled executable: the vmapped while loop runs until the slowest sample stops or the step cap is reached, finished samples held by a select",
    "warp_kernel": "per sample: one launch, each thread loops over its own sample's steps and stops at its own stop step or the step cap, whichever comes first",
    "cpu_openmp": "per step over the batch on the host: one host-team launch per step, the kernel marks and counts its finished samples",
}
NS = (1_000, 10_000, 100_000, 1_000_000)
#: Batch size of the cap fixture (some samples have a stop step beyond the cap).
CAP_FIXTURE_N = 4096
#: Its cap: a multiple of 64, so the arms that run whole blocks of steps (16 or
#: 64 per launch or replay) stop exactly at it.
CAP_FIXTURE_S = 128
#: (stop-step distribution, maximum step count S)
#: "spread": stop step log-uniform in [S/100, S] -- the irregular case
#: "uniform": every sample stops at S -- a dense, all-active batch
CONFIGS = (("spread", 100), ("spread", 1000), ("uniform", 1000))
#: Untimed ramp of back-to-back runs of one arm before its timed runs (seconds).
RAMP_S = 0.5
#: Timed runs per arm and cell (the statistic drops the highest and the lowest).
TIMED_RUNS = 5
REPS = TIMED_RUNS
#: Untimed runs before the repetitions: eagle's automatic arms run the size
#: pick, then measure the other device entry, and keep the faster from the
#: third run on -- two warm-up runs leave only the settled choice timed.
WARMUP_RUNS = 2
#: Kernel-pass runs per arm (the graph arm is repeated: a run whose kernel
#: count falls short of the expected count is discarded, see _nsys_kernel_times)
KERNEL_PASS_RUNS = {"eagle_graph": 3, "eagle_graph_compact": 3,
                    "eagle_graph_reorder": 3, "eagle_graph_steps": 3,
                    "eagle_graph_steps_compact": 3, "eagle_graph_auto": 3,
                    "eagle_graph_persist": 3,
                    "eagle_simulate": 3,
                    "eager_loop": 1, "cupy_masked": 1,
                    "torch_masked": 1, "torch_graphed": 1, "torch_compiled": 1,
                    "jax_vmap": 3, "warp_kernel": 3}
#: Steps between two compactions in the eagle_graph_compact arm (the eagle
#: default; a compaction runs only if a sample finished since the last one).
COMPACT_EVERY = 16
#: The reorder trigger's locality threshold in the eagle_graph_reorder arm
#: (the eagle default).
REORDER_THETA = 0.5
#: Steps one launch of the eagle_graph_steps arms advances (hawk
#: ``@hawk.kernel(steps=K)``), fixed by hand on this card: on the Quadro P2000
#: at N = 1e6, S = 1000 (bit-identical to K = 1 throughout), K = 16 and K = 64
#: run within 1% of each other (uniform 710 vs 702 ms, spread with compaction
#: every 64 steps 185 vs 182 ms; K = 1: 866 ms, 281 ms with compaction +
#: reorder), and K = 16 compacts at a 64-step cadence within eagle's minimum
#: of 4 launches between compactions.
STEPS_PER_LAUNCH = 16
#: Steps between two compactions in the eagle_graph_steps_compact arm
#: (``until_done(every=...)`` counts steps): 64 steps = 4 launches of 16.
STEPS_COMPACT_EVERY = 64
CPU_THREADS = 8

#: Compaction cadence sweep: isolates the active-set map as a single
#: variable by pairing eagle_graph_compact at cadence K against a
#: cadence-matched arm with no map (see _run_compaction_sweep). Run on one
#: representative config (the deeper spread case) at the larger Ns, where
#: the batch has thinned enough for compaction to matter.
COMPACTION_SWEEP_KS = (8, 16, 32)
COMPACTION_SWEEP_CONFIG = CONFIGS[1]  # ("spread", 1000)
COMPACTION_SWEEP_NS = (1_000_000, 100_000)

#: Arithmetic operations in the compiled step, per sample (add, subtract and
#: multiply each count 1; a negation counts 0), counted on the hawk-emitted
#: source. Recorded in the card with the rule.
FLOPS_PER_SAMPLE_STEP = 44
#: Bytes the compiled step moves per sample per launch, from its loads and
#: stores (float64 = 8 B, the bool mask = 1 B). Every sample loads x, v,
#: omega, zeta and its mask (33 B); a running sample also loads k and its stop
#: step and stores x, v, k (40 B). A finished sample's stores are skipped by
#: the guard; a sample finishing this step stores its mask (1 B; the one-word
#: atomic on the finished counter is not counted).
STEP_BYTES_ALL = 33
STEP_BYTES_ACTIVE = 40
STEP_BYTES_FINISH = 1
#: eagle_graph_compact: each launched position also loads its map entry (4 B);
#: one compaction moves about 30 B per sample (mask 1 B; keep flag, inclusive
#: scan and block sums written and read back, 4 B each; scatter reads and the
#: map store).
MAP_BYTES = 4
COMPACT_BYTES_PER_SAMPLE = 30
#: eagle_graph_reorder: one reorder stages and places every moved plane over
#: the span (each element read twice and written twice): omega, zeta, nstop,
#: x, v, k (8 B each), the mask (1 B) and perm (4 B) -- 53 B, so 212 B per
#: span sample -- plus inv (4 B read, 4 B written) and a scan of the mask
#: (30 B per sample, as a compaction).
REORDER_BYTES_PER_SPAN_SAMPLE = 4 * 53 + 8
#: The useful data a step needs, per running sample: load x, v, k, omega,
#: zeta; store x, v, k (64 B). The arm-independent yardstick.
USEFUL_BYTES_PER_SAMPLE_STEP = 64

_COUNTING_RULE = {
    "flops": (
        "Useful FLOPs = 44 per running sample per step: the add, subtract and "
        "multiply operations of the hawk-emitted step (negation not counted), "
        "times the number of (sample, step) pairs that are still running. The "
        "same count is used for every arm, so FLOP/s compares useful work."
    ),
    "useful_bytes": (
        "Useful bytes = 64 per running sample per step: load x, v, k, omega, "
        "zeta and store x, v, k as float64. The same count is used for every arm."
    ),
    "issued_bytes": (
        "Issued bytes = the arm's own analytic loads and stores. Step kernel: "
        "33 B per sample per launch (x, v, omega, zeta, mask) plus 40 B per "
        "running sample (k and stop-step loads, x/v/k stores), plus 1 B per "
        "sample finishing (its mask; the kernel finishes its own samples, so "
        "there is no stop-rule kernel); eagle_graph (the plain decorator's "
        "automatic kernel) and eager_loop count it per launch over the whole "
        "batch, eagle_graph over the launch sequence eagle's policy picks "
        "(replayed from the stop-step profile), eager_loop once per step. "
        "eagle_graph_compact: the step kernel's "
        "33 B + 4 B (the map entry) per MAPPED position per launch instead of "
        "per sample and the same 40 B per running sample, plus 30 B per "
        "sample per compaction. eagle_graph_reorder: as eagle_graph_compact, "
        "plus 220 B per span sample and 30 B per sample per reorder; "
        "the reorders are modelled from the trigger rule assuming the live "
        "samples sit at random inside the span. eagle_graph_steps: the step "
        "kernel's count per LAUNCH of 16 steps instead of per step (33 B per "
        "sample per launch, 40 B per sample running at the launch's start, "
        "1 B per sample finishing; the state stays in registers across the "
        "launch's steps). eagle_graph_steps_compact: eagle_graph_compact's "
        "count per launch, compacting every 4 launches. eagle_graph_auto: "
        "eagle_graph_compact's count per launch over the launch sequence its "
        "policy picks (replayed from the stop-step profile with eagle's own "
        "policy), compacting after every launch in which a sample finished; "
        "eagle_simulate: the same as eagle_graph_auto (the same kernel and loop). "
        "cupy_masked: each elementwise operation reads every array "
        "operand once and writes its result once over the whole batch (8 B "
        "per float64 element, 1 B per bool element; a reduction writes one "
        "element). torch_masked: the same elementwise count as cupy_masked (the "
        "same expressions, one kernel per operation). torch_graphed: the same "
        "count per step, over whole blocks of 16 steps, plus three state copies "
        "(x, v, k: 16 B per sample each) and the check's reduction (16 B per "
        "sample) per block. torch_compiled (Inductor): modelled as one fused "
        "pass per step, 88 B per sample (as jax_vmap's). warp_kernel: per "
        "sample once, loads omega, zeta, the stop step, x, v (40 B) and stores "
        "x, v, k (24 B); the state stays in registers across the steps. jax_vmap: from the compiled XLA "
        "program, per sample per loop iteration one fused body (loads x, v, k, "
        "the two per-sample coefficients and the stop step, 48 B; stores x, v, "
        "k, 24 B) and the loop condition's reduction (loads k and the stop "
        "step, 16 B), plus 32 B per sample once for the coefficients. "
        "cpu_openmp: host memory, the step kernel's count only."
    ),
}
#: jax_vmap's analytic bytes (see _COUNTING_RULE): per sample per iteration,
#: and once per sample (the loop-invariant coefficients).
JAX_BYTES_PER_SAMPLE_ITER = 48 + 24 + 16
JAX_BYTES_PER_SAMPLE_ONCE = 32
#: torch_graphed's per-block state copies and check reduction.
TORCH_BLOCK_BYTES_PER_SAMPLE = 3 * 16 + 16
#: warp_kernel: loads 5 float64 and stores 3 per sample, once.
WARP_BYTES_PER_SAMPLE = 64


# --------------------------------------------------------------------------- #
# Rendering (pure: JSON in, Markdown out; imported by the drift test)
# --------------------------------------------------------------------------- #
_fmt_time, _fmt_rate, _fmt_pct = cc.fmt_time, cc.fmt_rate, cc.fmt_pct
_fmt_mib, _fmt_x = cc.fmt_mib, cc.fmt_x


def _config_title(cfg):
    if cfg["distribution"] == "spread":
        return (f"Spread stop steps (log-uniform in [{cfg['max_steps'] // 100}, "
                f"{cfg['max_steps']}]), up to {cfg['max_steps']} steps")
    return f"Uniform stop step: every sample runs {cfg['max_steps']} steps"


#: The comparison contract, printed above the tables of both passes.
CAP_CONTRACT = (
    "Every arm runs each sample to its own stop and no further than max_steps "
    "(S): eagle and Warp enforce the cap per sample, JAX in its while-loop "
    "condition, PyTorch and CuPy by the outer step count (a batch-level cap); an "
    "arm that runs whole blocks of steps per launch or replay stops at the first "
    "block boundary at or after it.")
#: What each arm needed to meet it.
CAP_NEEDED = (
    "What each arm needed to get there: Warp, a hand-written step counter and "
    "cap inside the kernel; JAX, a cap in the while_loop condition; eagle, the "
    "kernel source unchanged, with its loop shape, unrolling and device entry "
    "chosen by hawk + eagle (per device, per batch).")


def _cap_fixture_line(fx):
    return (f"Cap fixture (checked before any timing counts): N = {fx['n']:,}, "
            f"{fx['capped_samples']} samples with stop step 2S = "
            f"{2 * fx['max_steps']}, S = {fx['max_steps']}; every arm "
            f"({len(fx['arms'])}) reported k = min(stop step, S) per sample and "
            f"the same finished count ({fx['finished']:,}): passed.")


def render_markdown(card: dict) -> str:
    """The card's tables and per-size fastest-arm lines, as MyST Markdown.

    Deterministic in ``card`` alone, so a test can re-render the committed
    JSON and compare it with the committed Markdown byte for byte."""
    dev = card["device"]
    spread = "range" if _trimmed_card(card) else "IQR"
    v2 = card["schema"] != "eagle-perf-card/1"  # schema/1: render the old shape, byte for byte
    lines = [
        f"<!-- Generated by benchmarks/perf_card/perf_card.py from "
        f"card_{card['device_slug']}.json; do not edit. -->",
        "",
        f"**Device:** {dev['name']} (compute capability {dev['compute_capability']}, "
        f"{dev['sm_count']} SMs). **Peak FP64:** "
        f"{_fmt_rate(card['peak']['fp64_flops'])} FLOP/s. **Peak DRAM bandwidth:** "
        f"{_fmt_rate(card['peak']['dram_bytes_per_s'])} B/s. "
        + (f"Mean of the middle {card['method']['repetitions'] - 2} of "
           f"{card['method']['repetitions']} runs of each arm (the highest and "
           f"the lowest dropped), range of those {card['method']['repetitions'] - 2} "
           f"in parentheses." if _trimmed_card(card) else
           f"Medians of {card['method']['repetitions']} interleaved repetitions; "
           f"IQR in parentheses."),
        "",
    ]
    gate = card["method"].get("cool_down_gate")
    if gate:
        lines += [
            (f"Each arm's block starts once the GPU is within {gate['delta_c']} °C of "
             f"its idle {gate['idle_baseline_c']:g} °C and reads idle, or after "
             f"{gate['max_s']:g} s; the SM clock is read right after each timed run "
             f"and its range is in the last column." if not gate.get("skipped") else
             f"No cool-down gate or clock record ({gate['skipped']})."),
            "",
        ]
    peak = card["peak"]
    if peak.get("observed_sm_clock_hz"):
        lines += [
            f"The FP64 peak above is at the device's reported maximum clock "
            f"({dev['clock_rate_hz'] / 1e6:.5g} MHz). During the "
            f"timed repetitions the SM clock read "
            f"{peak['observed_sm_clock_hz'] / 1e6:.4g} MHz; at that clock the FP64 "
            f"peak is {_fmt_rate(peak['fp64_flops_at_observed_sm_clock'])} FLOP/s, "
            f"and every FP64 percentage below is "
            f"{dev['clock_rate_hz'] / peak['observed_sm_clock_hz']:.3g}× larger "
            f"against it.",
            "",
        ]
        if v2 and peak.get("issue_peak_instr_per_s_at_observed_clock"):
            dpc = card["workload"].get("dp_instructions_per_step")
            lines += [
                f"\"% of issue peak\" (the eagle arms only) counts DP instructions "
                f"(DFMA/DMUL/DADD/DSETP/MUFU), not useful FLOPs: the step's SASS has "
                f"{dpc['count'] if dpc else '?'} of them, against the device's DP "
                f"issue-slot throughput at the observed clock -- a kernel that is "
                f"already at the issue pipe's limit can still read well under 100% "
                f"of FP64 peak, because useful FLOPs/step is smaller than issued "
                f"DP instructions/step.",
                "",
            ]
    every = card["workload"]["compaction_every_steps"]
    lines += [
        f"`eagle_graph` checks its stop guard every step; `eagle_graph_compact` "
        f"checks it once every {every} steps (the cadence its compaction runs "
        f"on) and reads an active-set map built from it. The two rows below "
        f"differ in both the cadence and the map; the compaction cadence sweep "
        f"further down holds the cadence fixed at a few values and isolates "
        f"the map alone.",
        "",
    ]
    block = card.get("torch_graph_block")
    if block:
        lines += [
            f"PyTorch, CUDA graph of {block} steps captures {block} masked steps once "
            "as a torch.cuda.CUDAGraph over static buffers and replays it, with one "
            "host check of \"any sample running\" per replay: it removes the "
            "per-kernel launch overhead and keeps the masking (every sample is "
            "computed every step, finished ones held by torch.where, so the steps "
            "a replay runs past the last stop change nothing). JAX's while loop is "
            "one compiled executable with no host check per step; batched by vmap, "
            "it runs until the slowest sample stops. Warp runs one thread per "
            "sample, each stopping at its own step.",
            "",
        ]
    for arm, why in (card.get("arms_not_run") or {}).items():
        lines += [f"{ARM_LABELS[arm]}: {why}.", ""]
    lines += [CAP_CONTRACT, "", CAP_NEEDED, ""]
    if card.get("cap_fixture"):
        lines += [_cap_fixture_line(card["cap_fixture"]), ""]
    lines += ["| arm | loop structure |", "|---|---|"]
    for arm in card["arms"]:
        lines.append(f"| {ARM_LABELS[arm]} | {ARM_LOOP[arm]} |")
    lines.append("")
    mem = card.get("memory") or {}
    mem_rows = {(r["distribution"], r["max_steps"], r["n"], r["arm"]): r
                for r in mem.get("rows", [])}
    by_cfg = {}
    for row in card["results"]:
        key = (row["distribution"], row["max_steps"])
        by_cfg.setdefault(key, []).append(row)
    for cfg in card["configs"]:
        key = (cfg["distribution"], cfg["max_steps"])
        rows = by_cfg.get(key, [])
        header = (f"| N | arm | wall ({spread}) | kernel-only | end-to-end | sample·steps/s "
                  "| useful FLOP/s (% peak) | issued B/s (% peak) "
                  "| device memory (× minimum) | host memory |")
        rule = "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        if v2:
            header += " % issue peak | lane util | launches (kernel/memcpy/API) |"
            rule += "---:|---:|---:|"
        clocks = any(r["wall_s"].get("sm_mhz") for r in rows)
        if clocks:
            header += " SM clock (MHz) |"
            rule += "---:|"
        lines += [f"### {_config_title(cfg)}", "", header, rule]
        for row in sorted(rows, key=lambda r: (r["n"], ALL_ARMS.index(r["arm"]))):
            w = row["wall_s"]
            m = mem_rows.get((row["distribution"], row["max_steps"], row["n"], row["arm"]))
            dev_cell = ("–" if m is None or m["device_peak_bytes"] is None else
                        f"{_fmt_mib(m['device_peak_bytes'])} ({_fmt_x(m['device_factor'])})")
            host_cell = "–" if m is None else _fmt_mib(m["host_peak_bytes"])
            cells = (
                f"| {row['n']:,} | {ARM_LABELS[row['arm']]} "
                f"| {_fmt_stat(w)} "
                f"| {_fmt_time(row['kernel_only_s'])} "
                f"| {_fmt_time(cc.center(row['end_to_end_s']))} "
                f"| {_fmt_rate(row['sample_steps_per_s'])} "
                f"| {_fmt_rate(row['useful_flops_per_s'])} "
                f"({_fmt_pct(row['fraction_of_peak_fp64'])}) "
                f"| {_fmt_rate(row['issued_bytes_per_s'])} "
                f"({_fmt_pct(row['fraction_of_peak_dram'])}) "
                f"| {dev_cell} | {host_cell} |"
            )
            if v2:
                nc = row.get("nsys_counts")
                launches = ("–" if not nc else
                           f"{nc.get('kernel_launches', '–')}/"
                           f"{nc.get('memcpy', '–')}/{nc.get('api_calls', '–')}")
                cells += (f" {_fmt_pct(row.get('fraction_of_issue_peak_at_observed_clock'))} "
                         f"| {row.get('lane_utilisation') or '–'} | {launches} |")
            if clocks:
                cells += f" {_clock_range(w) or '–'} |"
            lines.append(cells)
        lines.append("")
        lines += cc.fastest_lines(rows, ARM_LABELS)
        lines.append("")
    sweep = card.get("compaction_sweep")
    if sweep and sweep["rows"]:
        sweep_cfg_title = _config_title(
            {"distribution": sweep["distribution"], "max_steps": sweep["max_steps"]})
        lines += [
            "### Compaction cadence sweep (isolates the active-set map)",
            "",
            sweep["isolates"],
            "",
            f"Config: {sweep_cfg_title}.",
            "",
            "| N | K (steps between guard checks) | matched cadence, no map "
            f"({spread}) | eagle graph + compaction ({spread}) | ratio (matched/compact, "
            "`>1` = compaction faster) |",
            "|---:|---:|---:|---:|---:|",
        ]
        by_nk = {}
        for row in sweep["rows"]:
            by_nk.setdefault((row["n"], row["k"]), {})[row["variant"]] = row
        for n, k in sorted(by_nk):
            pair = by_nk[(n, k)]
            mc = pair["matched_cadence"]["wall_s"]
            co = pair["compact"]["wall_s"]
            ratio = cc.center(mc) / cc.center(co)
            lines.append(
                f"| {n:,} | {k} "
                f"| {_fmt_stat(mc)} "
                f"| {_fmt_stat(co)} "
                f"| {ratio:.3g}× |"
            )
        lines.append("")
    fp32 = card.get("fp32")
    if fp32 and fp32.get("rows"):
        cyc = any(w.get("cycles") for r in fp32["rows"] for w in r["wall_s"].values())
        lines += [
            f"### FP32: eagle vs Warp (float32, S = {fp32['max_steps']})",
            "",
            fp32["method"], "",
            CAP_CONTRACT, "", CAP_NEEDED, "",
            *([_cap_fixture_line(fp32["cap_fixture"]), ""] if fp32.get("cap_fixture") else []),
            f"| distribution | N | eagle.simulate ({spread}) | Warp ({spread}) | JAX ({spread}) "
            "| eagle / Warp |" + (" eagle / Warp (cycles) |" if cyc else "")
            + " eagle entry |",
            "|---|---:|---:|---:|---:|---:|" + ("---:|" if cyc else "") + "---|",
        ]
        for r in fp32["rows"]:
            w = r["wall_s"]
            cells = []
            for a in ("eagle_simulate", "warp_kernel", "jax_vmap"):
                rng = _clock_range(w[a])
                cells.append(_fmt_stat(w[a]) + (f", {rng} MHz" if rng else ""))
            ratio = cc.center(w["eagle_simulate"]) / cc.center(w["warp_kernel"])
            cyc_cell = ""
            if cyc:
                ce, cw = w["eagle_simulate"].get("cycles"), w["warp_kernel"].get("cycles")
                cyc_cell = (f" {cc.center(ce) / cc.center(cw):.3g}× |"
                            if ce and cw else " – |")
            lines.append(f"| {r['distribution']} | {r['n']:,} | " + " | ".join(cells)
                         + f" | {ratio:.3g}× |{cyc_cell} {r['eagle_run_mode']} |")
        lines.append("")
    fo = card.get("fixed_overhead")
    if v2 and fo and fo.get("rows"):
        lines += [
            "### Fixed overhead per run (N = 64, S = 100)",
            "",
            fo["method"], "",
            f"| arm | wall ({spread}) | kernel-only |",
            "|---|---:|---:|",
        ]
        for row in sorted(fo["rows"], key=lambda r: ALL_ARMS.index(r["arm"])):
            w = row["wall_s"]
            lines.append(f"| {ARM_LABELS[row['arm']]} "
                         f"| {_fmt_stat(w)} "
                         f"| {_fmt_time(row.get('kernel_only_s'))} |")
        lines.append("")
    lines += cc.fit_section(card, ARM_LABELS)
    lines += cc.compile_table(
        card.get("compile_time") or {}, ARM_LABELS, _fmt_time,
        " (PyTorch's eager kernels ship prebuilt; the CUDA-graph capture is per N and "
        "recorded per configuration as torch_graph_capture_s).")
    if mem.get("rows"):
        lines += ["### Memory method", "", mem["method"], ""]
        top = max(r["n"] for r in mem["rows"])
        base = {}
        for r in mem["rows"]:
            if r["n"] == top and r["device_baseline_bytes"] is not None:
                base.setdefault(r["arm"], r["device_baseline_bytes"])
        if base:
            lines += ["Device memory the process already held at the baseline (CUDA "
                      "context, loaded kernels and libraries, the warm-up's leftovers; "
                      f"at N = {top:,}, first configuration): "
                      + "; ".join(f"{ARM_LABELS[a]} {_fmt_mib(base[a])}"
                                  for a in card["arms"] if a in base) + ".", ""]
    return "\n".join(lines).rstrip() + "\n"


def _statistic_text(reps):
    return f"mean of the middle {reps - 2} of {reps}"


def _trimmed_card(card):
    """True for a card whose wall statistic is the trimmed mean (older cards
    carry the median and IQR and render as they always did)."""
    return str(card.get("method", {}).get("statistic", "")).startswith("mean")


def _fmt_stat(w):
    """A wall statistic as 'centre (spread)': the trimmed mean with the range
    of the kept runs, or the median with its IQR."""
    if "mean" in w:
        return f"{_fmt_time(w['mean'])} ({_fmt_time(w['lo'])}\u2013{_fmt_time(w['hi'])})"
    return f"{_fmt_time(w['median'])} ({_fmt_time(w['iqr'])})"


def _fmt_mib(nbytes):
    return "–" if nbytes is None else f"{nbytes / 2**20:.3g} MiB"


def _fmt_x(value):
    return "–" if value is None else f"{value:.3g}×"


def _fit_notes(card):
    """One neutral, number-backed note per arm (written into the card, then
    rendered): what the tool brings, and its wall time against the eagle
    graph arm at the largest N, per configuration."""
    ref_arm = "eagle_graph"
    rows = {(r["distribution"], r["max_steps"], r["n"], r["arm"]): r
            for r in card["results"]}
    wins, cells = cc.wins(card)
    top = max(card["ns"])
    per_config = card.get("per_config") or []
    identical = bool(per_config) and all(e.get("steps_bit_identical_to_graph")
                                         for e in per_config)

    def numbers(arm):
        parts = []
        for cfg in card["configs"]:
            d, s = cfg["distribution"], cfg["max_steps"]
            me, ref = rows.get((d, s, top, arm)), rows.get((d, s, top, ref_arm))
            if me is None or ref is None:
                continue
            t = _fmt_time(cc.center(me["wall_s"]))
            if arm == ref_arm:
                parts.append(f"{d} S={s}: {t}")
            else:
                parts.append(f"{d} S={s}: {t} "
                             f"({cc.center(me['wall_s']) / cc.center(ref['wall_s']):.3g}× "
                             f"the eagle graph time)")
        return f"N = {top:,}, " + "; ".join(parts) if parts else ""

    strengths = {
        "eagle_graph": (
            "The step written once as a hawk kernel; the whole loop runs on the "
            "device as one CUDA graph with a device-side stop guard, so a run is "
            "one launch with no host round trip per step."),
        "eagle_graph_compact": (
            "Adds an active-set map: the launch covers only the samples still "
            f"running, rebuilt every {card['workload']['compaction_every_steps']} "
            "steps; it is made for batches that thin out (the spread "
            "configurations)."),
        "eagle_graph_reorder": (
            "Adds a physical reorder when the live samples spread thin over "
            "their warps, so the mapped reads stay contiguous; the final restore "
            "to sample order is inside its wall."),
        "eagle_graph_steps": (
            "The same step with one decorator argument, "
            f"@hawk.kernel(steps={card['workload'].get('steps_per_launch')}): each "
            "launch runs that many steps per sample with the state kept in "
            "registers, as Warp's per-thread loop does, and the launches, guard "
            "checks and plane reads shrink by the same factor"
            + ("; its results are bit-identical to the eagle graph arm's in "
               "every cell" if identical else "")
            + ". K is chosen by hand here; the eagle_graph_auto arm picks it at "
            "run time."),
        "eagle_graph_steps_compact": (
            "The K-steps kernel over the active-set map: the register-resident "
            "steps of the arm above, and the launches cover only the samples "
            "still running, so it fits batches that thin out."),
        "eagle_graph_auto": (
            "The same step with @hawk.kernel(steps=\"auto\") on the active-set "
            "kind, and nothing to tune: it picks the steps per launch itself "
            "after every launch, on the device, so a batch whose samples all "
            "run to the end gets Warp-like fusion of up to 64 steps per launch, "
            "and a batch that thins out gets shorter launches with the active "
            "set compacted after each one; no sample takes more steps than the "
            "cap."),
        "eagle_graph_persist": (
            "The plain (not active-set) automatic kernel, forced through "
            "hawk's persist entry: one launch, no graph, no map, no policy "
            "kernel, no cap at 64 -- a grid of SMs blocks of 256 lanes each "
            "steal the next unfinished sample off a counter when their own "
            "finishes, so idle warps refill without ever rebuilding an "
            "index map; made for batches too large for one resident wave."),
        "eagle_simulate": (
            "The same automatic kernel through eagle.simulate: the model, its "
            "state and its parameters by name, with no plan or plane binding "
            "to write; eagle deploys the kernel and builds the same device "
            "loop."),
        "eager_loop": (
            "The same kernels launched from Python one step at a time; the host "
            "sees the state after every step."),
        "cupy_masked": (
            "NumPy-like array code on GPUs with no kernel to write; every sample "
            "is computed every step, so it fits dense batches where all samples "
            "run to the end (the uniform configuration)."),
        "torch_masked": (
            "The same array code in PyTorch tensors; it fits where the "
            "computation sits next to a PyTorch model or needs autograd."),
        "torch_graphed": (
            "The same PyTorch code with its launch overhead removed by hand: "
            f"{card.get('torch_graph_block')} steps captured as one CUDA graph and "
            "replayed, one host check per replay, the masking unchanged."),
        "torch_compiled": (
            "The PyTorch step under torch.compile(mode=\"reduce-overhead\"): "
            "Inductor-fused kernels replayed as a CUDA graph."),
        "warp_kernel": (
            "Explicit per-thread kernels written in Python, maintained by NVIDIA "
            "and differentiable through wp.Tape; each thread runs its own sample "
            "and stops at its own step, so finished samples cost nothing."),
        "jax_vmap": (
            "The whole loop compiled once, a per-sample while loop batched with "
            "vmap; the same code runs on CPUs and GPUs and differentiates with "
            "jax.grad."),
        "cpu_openmp": (
            "The same hawk kernel on the host through eagle's OpenMP team, for "
            "machines without a GPU."),
    }
    notes = {}
    for arm in card["arms"]:
        nums = numbers(arm)
        notes[arm] = (f"{strengths[arm]}" + (f" On this card: {nums}." if nums else "")
                      + f" Fastest in {wins[arm]} of {cells} cells.")
    intro = (
        "Each tool below is written the way its users write it, and each brings "
        "something of its own. Times are walls (mean of the middle 3 of 5 runs) at the largest N; ratios "
        "are against the eagle graph arm (below 1 = less time). The fastest arm "
        "per N is listed under each table above.")
    return {"intro": intro, "arms": notes}


# --------------------------------------------------------------------------- #
# The kernels (hawk)
# --------------------------------------------------------------------------- #
def _define_kernel():
    import hawk
    from hawk import Mutable, Param, Scalar, Terminated

    # >>> code:hawk_step
    @hawk.kernel
    def oscillator_step(omega: Scalar, zeta: Scalar, nstop: Scalar, dt: Param,
                        terminated: Terminated, x: Mutable[Scalar],
                        v: Mutable[Scalar], k: Mutable[Scalar]):
        # One classical RK4 step of x'' + 2 zeta omega x' + omega^2 x = 0.
        x0 = x
        v0 = v
        w2 = omega * omega
        c = 2.0 * zeta * omega
        h2 = 0.5 * dt
        h6 = dt * 0.16666666666666666
        a1 = -w2 * x0 - c * v0
        x2 = x0 + h2 * v0
        v2 = v0 + h2 * a1
        a2 = -w2 * x2 - c * v2
        x3 = x0 + h2 * v2
        v3 = v0 + h2 * a2
        a3 = -w2 * x3 - c * v3
        x4 = x0 + dt * v3
        v4 = v0 + dt * a3
        a4 = -w2 * x4 - c * v4
        x = x0 + h6 * (v0 + 2.0 * v2 + 2.0 * v3 + v4)
        v = v0 + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        k1 = k + 1.0
        k = k1
        terminated = k1 >= nstop  # the sample stops after its own nstop steps
    # <<< code:hawk_step

    return oscillator_step


def _define_steps_kernel():
    """The step of :func:`_define_kernel`, declared to take STEPS_PER_LAUNCH
    steps per launch (the decorator form of ``hawk.steps``)."""
    import hawk
    from hawk import Mutable, Param, Scalar, Terminated

    # >>> code:hawk_steps
    @hawk.kernel(steps=STEPS_PER_LAUNCH)
    def oscillator_step(omega: Scalar, zeta: Scalar, nstop: Scalar, dt: Param,
                        terminated: Terminated, x: Mutable[Scalar],
                        v: Mutable[Scalar], k: Mutable[Scalar]):
        # The RK4 step above, unchanged; one launch runs up to 16 of them.
        x0 = x
        v0 = v
        w2 = omega * omega
        c = 2.0 * zeta * omega
        h2 = 0.5 * dt
        h6 = dt * 0.16666666666666666
        a1 = -w2 * x0 - c * v0
        x2 = x0 + h2 * v0
        v2 = v0 + h2 * a1
        a2 = -w2 * x2 - c * v2
        x3 = x0 + h2 * v2
        v3 = v0 + h2 * a2
        a3 = -w2 * x3 - c * v3
        x4 = x0 + dt * v3
        v4 = v0 + dt * a3
        a4 = -w2 * x4 - c * v4
        x = x0 + h6 * (v0 + 2.0 * v2 + 2.0 * v3 + v4)
        v = v0 + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        k1 = k + 1.0
        k = k1
        terminated = k1 >= nstop
    # <<< code:hawk_steps

    return oscillator_step


def _define_step_kernel():
    """The single step of :func:`_define_kernel` (its ``.step``): the kernel the
    ``eager_loop`` arm launches itself, one step per launch."""
    return _define_kernel().step


def _active_set_kind():
    """The kind the compaction arms' kernels are traced under: their guard
    reads the active-set index map."""
    from hawk.ext import Guard, Kind

    # >>> code:hawk_kind
    ACTIVE_SET = Kind("active_set", guard=Guard(active_set=True))
    # <<< code:hawk_kind
    return ACTIVE_SET


def _define_active_kernel():
    """The single step of :func:`_define_kernel` under the active-set kind:
    the kernel the compaction arms launch, one step per launch at their own
    ``every`` cadence (``hawk.steps`` of it is the K-steps compaction kernel)."""
    import hawk
    from hawk import Mutable, Param, Scalar, Terminated

    ACTIVE_SET = _active_set_kind()

    # >>> code:hawk_active
    @hawk.kernel(steps=1, kind=ACTIVE_SET)
    def oscillator_step(omega: Scalar, zeta: Scalar, nstop: Scalar, dt: Param,
                        terminated: Terminated, x: Mutable[Scalar],
                        v: Mutable[Scalar], k: Mutable[Scalar]):
        # The RK4 step above, unchanged; its launch covers the active set.
        x0 = x
        v0 = v
        w2 = omega * omega
        c = 2.0 * zeta * omega
        h2 = 0.5 * dt
        h6 = dt * 0.16666666666666666
        a1 = -w2 * x0 - c * v0
        x2 = x0 + h2 * v0
        v2 = v0 + h2 * a1
        a2 = -w2 * x2 - c * v2
        x3 = x0 + h2 * v2
        v3 = v0 + h2 * a2
        a3 = -w2 * x3 - c * v3
        x4 = x0 + dt * v3
        v4 = v0 + dt * a3
        a4 = -w2 * x4 - c * v4
        x = x0 + h6 * (v0 + 2.0 * v2 + 2.0 * v3 + v4)
        v = v0 + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        k1 = k + 1.0
        k = k1
        terminated = k1 >= nstop
    # <<< code:hawk_active

    return oscillator_step


def _define_auto_kernel():
    """The step of :func:`_define_kernel` under the active-set kind, declared
    to pick its steps per launch at run time (``@hawk.kernel(steps="auto")``)."""
    import hawk
    from hawk import Mutable, Param, Scalar, Terminated

    ACTIVE_SET = _active_set_kind()

    # >>> code:hawk_auto
    @hawk.kernel(steps="auto", kind=ACTIVE_SET)
    def oscillator_step(omega: Scalar, zeta: Scalar, nstop: Scalar, dt: Param,
                        terminated: Terminated, x: Mutable[Scalar],
                        v: Mutable[Scalar], k: Mutable[Scalar]):
        # The RK4 step above, unchanged; eagle picks the steps per launch.
        x0 = x
        v0 = v
        w2 = omega * omega
        c = 2.0 * zeta * omega
        h2 = 0.5 * dt
        h6 = dt * 0.16666666666666666
        a1 = -w2 * x0 - c * v0
        x2 = x0 + h2 * v0
        v2 = v0 + h2 * a1
        a2 = -w2 * x2 - c * v2
        x3 = x0 + h2 * v2
        v3 = v0 + h2 * a2
        a3 = -w2 * x3 - c * v3
        x4 = x0 + dt * v3
        v4 = v0 + dt * a3
        a4 = -w2 * x4 - c * v4
        x = x0 + h6 * (v0 + 2.0 * v2 + 2.0 * v3 + v4)
        v = v0 + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        k1 = k + 1.0
        k = k1
        terminated = k1 >= nstop
    # <<< code:hawk_auto

    return oscillator_step


def _define_persist_kernel():
    """The step of :func:`_define_kernel` under the PLAIN kind (no
    active-set guard), declared to pick its steps per launch at run time
    (``@hawk.kernel(steps="auto")``) -- hawk's persist entry only exists
    for an automatic kernel with a contiguous range,
    which means the plain kind, not :func:`_define_auto_kernel`'s
    active-set one; :data:`PERSIST_ARM` forces eagle's persist launch over
    this kernel."""
    import hawk
    from hawk import Mutable, Param, Scalar, Terminated

    # >>> code:hawk_persist
    @hawk.kernel(steps="auto")
    def oscillator_step(omega: Scalar, zeta: Scalar, nstop: Scalar, dt: Param,
                        terminated: Terminated, x: Mutable[Scalar],
                        v: Mutable[Scalar], k: Mutable[Scalar]):
        # The RK4 step above, unchanged; eagle picks the steps per launch,
        # and (forced) runs them through the persist entry, no active set.
        x0 = x
        v0 = v
        w2 = omega * omega
        c = 2.0 * zeta * omega
        h2 = 0.5 * dt
        h6 = dt * 0.16666666666666666
        a1 = -w2 * x0 - c * v0
        x2 = x0 + h2 * v0
        v2 = v0 + h2 * a1
        a2 = -w2 * x2 - c * v2
        x3 = x0 + h2 * v2
        v3 = v0 + h2 * a2
        a3 = -w2 * x3 - c * v3
        x4 = x0 + dt * v3
        v4 = v0 + dt * a3
        a4 = -w2 * x4 - c * v4
        x = x0 + h6 * (v0 + 2.0 * v2 + 2.0 * v3 + v4)
        v = v0 + h6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
        k1 = k + 1.0
        k = k1
        terminated = k1 >= nstop
    # <<< code:hawk_persist

    return oscillator_step


def _check_kernel_bodies():
    """Every variant's body is the single step's, op for op: the K-steps
    kernel's traced DAG equals ``hawk.steps`` of :func:`_define_kernel`'s, the
    active-set step's equals its ``.step``, and the automatic kernel's equals
    ``hawk.steps(<active-set step>, "auto")``."""
    import hawk

    single, active = _define_kernel(), _define_active_kernel()
    assert _define_steps_kernel().walk.digest == hawk.steps(
        single, STEPS_PER_LAUNCH).walk.digest, (
        "the K-steps kernel's body drifted from the single step's")
    assert active.walk.digest == single.step.walk.digest, (
        "the active-set step's body drifted from the single step's")
    assert _define_auto_kernel().walk.digest == hawk.steps(active, "auto").walk.digest, (
        "the automatic kernel's body drifted from the single step's")


def _kernel_sources(work_dir):
    """md5 of every kernel source the eagle arms run, built by hawk's own
    bundle builder into ``work_dir`` (the content-addressed build
    ``eagle.deploy`` uses, so a cache hit recompiles nothing)."""
    import hawk
    from hawk.artifact import build_bundle

    kernels = {"step": _define_kernel(), "step_single": _define_step_kernel(),
               "steps": _define_steps_kernel(), "active": _define_active_kernel(),
               "steps_active": hawk.steps(_define_active_kernel(), STEPS_PER_LAUNCH),
               "auto": _define_auto_kernel()}
    sources = {}
    for prefix, kernel in kernels.items():
        d = work_dir / f"kernel_{prefix}"
        build_bundle([kernel], d, targets=("host", "cuda"))
        sources.update({f"{prefix}/{p.name}": cc.md5(p) for p in sorted(d.iterdir())
                        if p.suffix in (".cu", ".cpp", ".ptx")})
    return sources


def _dp_instruction_count(work_dir, device):
    """Static DP-instruction count of one step of the hawk-emitted body
    (DFMA/DMUL/DADD/DSETP/MUFU), from the SASS of the single-step kernel's
    PTX (``kernel_step_single``, already built by :func:`_kernel_sources`
    into ``work_dir``), compiled for this device's own compute capability.

    ``_check_kernel_bodies`` proves every eagle arm's traced body is this
    same step (bit for bit: the K-steps kernel is ``hawk.steps`` of it, the
    active-set step is its ``.step``, the automatic kernel is ``hawk.steps``
    of the active-set step). The fused kernels hoist per-sample work out of
    their step loop, so this count is the per-step count of the single-step
    arms only (``SINGLE_STEP_ARMS``); the fused arms' per-step count is lower
    and is not taken statically here. Returns the
    SASS count dict (see ``_card_common.sass_dp_instruction_count``) or None
    (toolchain unavailable)."""
    major, minor = (int(x) for x in device["compute_capability"].split("."))
    d = work_dir / "kernel_step_single"
    code = None
    if d.is_dir():
        files = list(d.iterdir())
        code = next((p for p in files if p.suffix == ".cubin"), None) or next(
            (p for p in files if p.suffix == ".ptx"), None)
    if code is None:
        return None
    return cc.sass_dp_instruction_count(code, f"sm_{major}{minor}")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def _inputs(n, distribution, max_steps, dtype="float64"):
    import numpy as np

    rng = np.random.default_rng([SEED, n, max_steps, distribution == "spread"])
    omega = rng.uniform(1.0, 10.0, n)
    zeta = rng.uniform(0.01, 0.2, n)
    x0 = rng.uniform(-1.0, 1.0, n)
    v0 = rng.uniform(-1.0, 1.0, n)
    if distribution == "spread":
        lo, hi = math.log(max_steps / 100), math.log(max_steps)
        nstop = np.ceil(np.exp(rng.uniform(lo, hi, n)))
        nstop = np.clip(nstop, max(1, max_steps // 100), max_steps)
    else:
        nstop = np.full(n, float(max_steps))
    nstop[0] = float(max_steps)  # the batch always runs exactly max_steps steps
    return {key: np.asarray(val, dtype=dtype) for key, val in
            dict(omega=omega, zeta=zeta, x0=x0, v0=v0, nstop=nstop).items()}


def _active_profile(nstop, max_steps):
    """running[s] = samples still running at step s (s = 0..max_steps-1)."""
    import numpy as np

    counts = np.bincount(nstop.astype(np.int64), minlength=max_steps + 1)
    finished_before = np.concatenate(([0], np.cumsum(counts)[:-1]))
    return (len(nstop) - finished_before)[:max_steps]


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
class _GpuArm:
    """Device buffers for one GPU arm (inputs uploaded in place, so captured
    pointers stay valid across repetitions)."""

    def __init__(self, n, dtype="float64"):
        import cupy as cp

        self.n = n
        self.omega = cp.empty(n, dtype=dtype)
        self.zeta = cp.empty(n, dtype=dtype)
        self.nstop = cp.empty(n, dtype=dtype)
        self.x = cp.empty(n, dtype=dtype)
        self.v = cp.empty(n, dtype=dtype)
        self.k = cp.zeros(n, dtype=dtype)
        self.terminated = cp.zeros(n, dtype=cp.bool_)
        self.runner = None  # the eagle.until_done runner (or eagle.simulation)

    def planes(self):
        """Every plane the step binds, by name."""
        return dict(omega=self.omega, zeta=self.zeta, nstop=self.nstop,
                    terminated=self.terminated, x=self.x, v=self.v, k=self.k)

    def upload(self, inp):
        self.omega.set(inp["omega"])
        self.zeta.set(inp["zeta"])
        self.nstop.set(inp["nstop"])
        self.x.set(inp["x0"])
        self.v.set(inp["v0"])
        self.k.fill(0.0)
        self.terminated.fill(False)
        if self.runner is not None:
            self.runner.reset()

    def download(self):
        return self.x.get(), self.v.get(), self.k.get()


def _runner_arm(st, fired=False):
    """The arm dict of an ``eagle.until_done`` runner or an
    ``eagle.simulation``: the build (the graph capture of the first run) and
    the reorder count are read off each run's report."""
    arm = {"state": st, "download": st.download,
           # .loop is None for a fast-mode run (one launch, no WHILE
           # graph) -- report.launches is the same number either way (the
           # band path's own .launches IS self.loop.iterations()).
           "iterations": lambda: arm["report"].launches if arm["report"] else 0,
           "build_s": None, "report": None}
    if fired:
        arm["fired"] = []

    def run():
        report = st.runner.run()  # a reordering run ends in sample order
        report = getattr(report, "report", report)  # a simulation's result
        arm["build_s"] = report.build_s
        arm["report"] = report
        if fired:
            arm["fired"].append(report.reorders)

    arm["run"] = run
    return arm


def _make_gpu_arms(n, max_steps, names=None):
    """The GPU arms for one (N, S) cell, as {arm: dict(state, run, download,
    ...)}; ``names`` limits the build to those arms (the memory and
    compile-time probes build one arm per process). Each eagle arm deploys
    its hawk kernel with ``eagle.deploy`` (a cache hit after the first
    build)."""
    names = tuple(a for a in _run_arms() if a != "cpu_openmp") if names is None else tuple(names)
    arms = {}
    if any(a in names for a in ("eagle_graph", "eagle_graph_compact",
                                "eagle_graph_reorder", "eager_loop", "cupy_masked")):
        arms.update(_make_cupy_arms(n, max_steps, names))
    if any(a in names for a in STEPS_ARMS + (AUTO_ARM, PERSIST_ARM, SIMULATE_ARM)):
        arms.update(_make_steps_arms(n, max_steps, names))
    if any(a in names for a in ("torch_masked", "torch_graphed", "torch_compiled")):
        arms.update(_make_torch_arms(n, max_steps, names))
    if "jax_vmap" in names:
        arms["jax_vmap"] = _make_jax_arm(n, max_steps)
    if "warp_kernel" in names:
        arms["warp_kernel"] = _make_warp_arm(n, max_steps)
    return {a: arms[a] for a in names}


def _make_steps_arms(n, max_steps, names):
    import eagle
    import eagle.plan  # noqa: F401  (eagle.simulate first in a process imports it circularly)
    import hawk

    arms = {}
    if "eagle_graph_steps" in names:
        oscillator_step = _define_steps_kernel()
        # >>> code:eagle_graph_steps
        # eagle_graph_steps: the same one call, on the K-steps kernel; forced
        # off the fast path (it is a fixed-K kernel, not "auto", so it would
        # never be fast-routed, but the row's device-loop labels/byte model
        # assume the band -- forced explicitly so a future kernel change
        # cannot silently drift this arm onto the fast path unnoticed)
        s = _GpuArm(n)
        s.runner = eagle.until_done(eagle.deploy(oscillator_step), max_steps=max_steps,
                                    dt=DT, _fast_mode=False, **s.planes())
        # <<< code:eagle_graph_steps
        arms["eagle_graph_steps"] = _runner_arm(s)

    if "eagle_graph_steps_compact" in names:
        oscillator_step = _define_active_kernel()
        # >>> code:eagle_graph_steps_compact
        # eagle_graph_steps_compact: K steps per launch of the active-set step,
        # compacted every 64 steps (4 launches); forced off the fast path for
        # the same reason eagle_graph_steps is (see there)
        c = _GpuArm(n)
        steps = hawk.steps(oscillator_step, STEPS_PER_LAUNCH)
        c.runner = eagle.until_done(eagle.deploy(steps), max_steps=max_steps, dt=DT,
                                    every=STEPS_COMPACT_EVERY, _fast_mode=False,
                                    **c.planes())
        # <<< code:eagle_graph_steps_compact
        arms["eagle_graph_steps_compact"] = _runner_arm(c)

    if AUTO_ARM in names:
        oscillator_step = _define_auto_kernel()
        # >>> code:eagle_graph_auto
        # eagle_graph_auto: the same one call; the kernel picks its steps per
        # launch and compacts as it goes
        a = _GpuArm(n)
        a.runner = eagle.until_done(eagle.deploy(oscillator_step), max_steps=max_steps,
                                    dt=DT, **a.planes())
        # <<< code:eagle_graph_auto
        arms[AUTO_ARM] = _runner_arm(a)

    if PERSIST_ARM in names:
        oscillator_step = _define_persist_kernel()
        # >>> code:eagle_graph_persist
        # eagle_graph_persist: the persist entry forced explicitly (device
        # rule (ii)) -- one launch, no graph, no map, no policy kernel;
        # lanes steal samples off a counter until every one is done
        p = _GpuArm(n)
        p.runner = eagle.until_done(eagle.deploy(oscillator_step), max_steps=max_steps,
                                    dt=DT, _fast_mode="persist",
                                    _lane_utilisation=True, **p.planes())
        # <<< code:eagle_graph_persist
        arms[PERSIST_ARM] = _runner_arm(p)

    if SIMULATE_ARM in names:
        oscillator_step = _define_auto_kernel()
        # >>> code:eagle_simulate
        # eagle_simulate: the model, its state and its parameters by name; it
        # runs until every sample has finished
        m = _GpuArm(n)
        m.runner = eagle.simulation(oscillator_step, omega=m.omega, zeta=m.zeta,
                                    nstop=m.nstop, dt=DT, x=m.x, v=m.v, k=m.k,
                                    max_steps=max_steps)
        # <<< code:eagle_simulate
        arms[SIMULATE_ARM] = _runner_arm(m)
    return arms


def _make_cupy_arms(n, max_steps, names):
    import cupy as cp

    import eagle

    arms = {}
    if "eagle_graph" in names:
        oscillator_step = _define_kernel()
        # >>> code:eagle_graph
        # eagle_graph: one call; one CUDA graph captured on the first run.
        # Forced off the fast path: this row's label, ARM_LOOP description,
        # issued-bytes model and expected-kernel count all assume the
        # device-loop/band path (eagle_graph_auto's row is the one that
        # follows auto's actual pick, see there).
        g = _GpuArm(n)
        g.runner = eagle.until_done(eagle.deploy(oscillator_step), max_steps=max_steps,
                                    dt=DT, _fast_mode=False, **g.planes())
        # <<< code:eagle_graph
        arms["eagle_graph"] = _runner_arm(g)

    if "eagle_graph_compact" in names:
        oscillator_step = _define_active_kernel()
        # >>> code:eagle_graph_compact
        # eagle_graph_compact: the same call on the active-set step; forced
        # off the fast path for the same reason eagle_graph is (see there)
        a = _GpuArm(n)
        a.runner = eagle.until_done(eagle.deploy(oscillator_step), max_steps=max_steps,
                                    dt=DT, every=COMPACT_EVERY, _fast_mode=False,
                                    **a.planes())
        # <<< code:eagle_graph_compact
        arms["eagle_graph_compact"] = _runner_arm(a)

    if "eagle_graph_reorder" in names:
        oscillator_step = _define_active_kernel()
        # >>> code:eagle_graph_reorder
        # eagle_graph_reorder: the same call, plus the conditional reorder of
        # every per-sample plane (restored to sample order at the end);
        # reorder= already keeps this off the fast path, forced here too for
        # the same reason eagle_graph is (see there)
        r = _GpuArm(n)
        r.runner = eagle.until_done(eagle.deploy(oscillator_step), max_steps=max_steps,
                                    dt=DT, every=COMPACT_EVERY, reorder=REORDER_THETA,
                                    _fast_mode=False,
                                    **r.planes())
        # <<< code:eagle_graph_reorder
        arms["eagle_graph_reorder"] = _runner_arm(r, fired=True)

    if "eager_loop" in names:
        oscillator_step = _define_kernel()
        # >>> code:eager_loop
        # eager_loop: the single step (no loop in the kernel), launched from
        # Python, a host check of the finished count per step
        e = _GpuArm(n)
        done = cp.zeros(1, dtype=cp.uint32)
        e_step = eagle.deploy(oscillator_step.step).bind(
            dt=DT, finished_count=done.view(cp.int32), **e.planes())

        def run_eager():
            done.fill(0)
            for _ in range(max_steps):
                e_step.launch()
                if int(done.get()[0]) == n:
                    break

        # <<< code:eager_loop
        arms["eager_loop"] = dict(state=e, run=run_eager, download=e.download)

    if "cupy_masked" in names:
        # >>> code:cupy_masked
        # cupy_masked: array expressions over the whole batch
        c = _GpuArm(n)
        result = {}

        def run_cupy():
            x, v, k = c.x, c.v, c.k
            nw2 = -(c.omega * c.omega)
            cc = (2.0 * c.zeta) * c.omega
            h2 = 0.5 * DT
            h6 = DT * 0.16666666666666666
            for _ in range(max_steps):
                running = k < c.nstop
                if not bool(running.any()):
                    break
                a1 = nw2 * x - cc * v
                v2 = v + h2 * a1
                a2 = nw2 * (x + h2 * v) - cc * v2
                v3 = v + h2 * a2
                a3 = nw2 * (x + h2 * v2) - cc * v3
                v4 = v + DT * a3
                a4 = nw2 * (x + DT * v3) - cc * v4
                xn = x + h6 * (((v + 2.0 * v2) + 2.0 * v3) + v4)
                vn = v + h6 * (((a1 + 2.0 * a2) + 2.0 * a3) + a4)
                x = cp.where(running, xn, x)
                v = cp.where(running, vn, v)
                k = k + running
            result["xvk"] = (x, v, k)

        def cupy_download():
            x, v, k = result["xvk"]
            return x.get(), v.get(), k.get()

        # <<< code:cupy_masked
        arms["cupy_masked"] = dict(state=c, run=run_cupy, download=cupy_download)
    return arms


# --------------------------------------------------------------------------- #
# PyTorch and JAX arms (each written the way that library's users write it)
# --------------------------------------------------------------------------- #
H2 = 0.5 * DT
H6 = DT * 0.16666666666666666


#: torch_graphed: masked steps per captured CUDA graph (one host check per
#: replay; the same cadence as the compaction arms' guard).
GRAPH_BLOCK = 16


def _triton_ok():
    """(True, None) where torch.compile's Inductor GPU backend can run
    (Triton needs compute capability >= 7.0), else (False, reason)."""
    import torch

    if not torch.cuda.is_available():
        return False, "needs PyTorch with CUDA and a GPU, so it was not run"
    cap = torch.cuda.get_device_capability()
    if cap >= (7, 0):
        return True, None
    return False, (f"needs compute capability 7.0 or newer (Triton, Inductor's GPU "
                   f"code generator); this GPU is {cap[0]}.{cap[1]}, so it was not run")


_RUN_ARMS = []


def _run_arms():
    """The arms this machine runs, in table order (ARMS plus the optional
    arms whose toolchain runs here)."""
    if not _RUN_ARMS:
        extra, _ = _optional_arms()
        _RUN_ARMS.append(tuple(a for a in ALL_ARMS if a in ARMS or a in extra))
    return _RUN_ARMS[0]


def _optional_arms():
    """The optional arms this machine runs, and the reason for each it does not."""
    ok, why = _triton_ok()
    return (("torch_compiled",) if ok else ()), ({} if ok else {"torch_compiled": why})


class _TorchState:
    """Device tensors of a PyTorch arm (inputs copied in place, so a captured
    graph's static buffers keep their addresses across runs)."""

    def __init__(self, n):
        import torch

        self.torch = torch
        dev = torch.device("cuda")
        self.omega, self.zeta, self.nstop, self.x, self.v, self.k, self.nw2, self.cc = (
            torch.empty(n, dtype=torch.float64, device=dev) for _ in range(8))
        self.out = None

    def upload(self, inp):
        t = self.torch
        for dst, key in ((self.omega, "omega"), (self.zeta, "zeta"),
                         (self.nstop, "nstop"), (self.x, "x0"), (self.v, "v0")):
            dst.copy_(t.from_numpy(inp[key]))
        self.k.zero_()

    def download(self):
        x, v, k = self.out
        return x.cpu().numpy(), v.cpu().numpy(), k.cpu().numpy()


# >>> code:torch_step
def _torch_step(x, v, k, nw2, cc, nstop):
    """One masked RK4 step on whole-batch tensors (torch_masked's expressions)."""
    import torch

    running = k < nstop
    a1 = nw2 * x - cc * v
    v2 = v + H2 * a1
    a2 = nw2 * (x + H2 * v) - cc * v2
    v3 = v + H2 * a2
    a3 = nw2 * (x + H2 * v2) - cc * v3
    v4 = v + DT * a3
    a4 = nw2 * (x + DT * v3) - cc * v4
    xn = x + H6 * (((v + 2.0 * v2) + 2.0 * v3) + v4)
    vn = v + H6 * (((a1 + 2.0 * a2) + 2.0 * a3) + a4)
    return torch.where(running, xn, x), torch.where(running, vn, v), k + running, running
# <<< code:torch_step


# >>> code:torch_graphed
def _torch_capture_block(s, block):
    """Capture ``block`` masked steps over the static buffers of ``s`` as one
    CUDA graph; returns (graph, flag), ``flag`` the captured "any sample
    still running" after the block."""
    import torch

    def steps():
        x, v, k = s.x, s.v, s.k
        for _ in range(block):
            x, v, k, _ = _torch_step(x, v, k, s.nw2, s.cc, s.nstop)
        s.x.copy_(x)
        s.v.copy_(v)
        s.k.copy_(k)
        return (s.k < s.nstop).any()

    side = torch.cuda.Stream()  # warm-up on a side stream before capture
    side.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(side):
        for _ in range(2):
            steps()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        flag = steps()
    return graph, flag


def _torch_graphed_run(s, graph, flag, max_steps, block):
    import torch

    with torch.no_grad():
        torch.mul(s.omega, s.omega, out=s.nw2)
        s.nw2.neg_()
        torch.mul(2.0 * s.zeta, s.omega, out=s.cc)
        for _ in range(-(-max_steps // block)):
            graph.replay()  # finished samples are held by torch.where
            if not bool(flag):
                break
    s.out = (s.x, s.v, s.k)
# <<< code:torch_graphed


# >>> code:torch_compiled
def _torch_step_inplace(x, v, k, nw2, cc, nstop):
    """torch_compiled's step: torch_masked's expressions, the state updated
    in place (static buffers, so one recorded graph serves every step);
    returns "any sample still running" as a device scalar."""
    xn, vn, kn, running = _torch_step(x, v, k, nw2, cc, nstop)
    x.copy_(xn)
    v.copy_(vn)
    k.copy_(kn)
    return running.any()
# <<< code:torch_compiled


def _make_torch_arms(n, max_steps, names):
    import torch

    arms = {}
    if "torch_masked" in names:
        s = _TorchState(n)

        # >>> code:torch_masked
        def run_masked():
            with torch.inference_mode():
                x, v, k = s.x, s.v, s.k
                nw2 = -(s.omega * s.omega)
                cc = (2.0 * s.zeta) * s.omega
                for _ in range(max_steps):
                    running = k < s.nstop
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
                s.out = (x, v, k)
        # <<< code:torch_masked

        arms["torch_masked"] = dict(state=s, run=run_masked, download=s.download)
    if "torch_graphed" in names:
        g = _TorchState(n)
        t0 = time.perf_counter()
        graph, flag = _torch_capture_block(g, GRAPH_BLOCK)
        torch.cuda.synchronize()
        build_s = time.perf_counter() - t0
        arms["torch_graphed"] = dict(
            state=g, run=lambda: _torch_graphed_run(g, graph, flag, max_steps, GRAPH_BLOCK),
            download=g.download, build_s=build_s)
    if "torch_compiled" in names:
        # only where Triton runs (see _triton_ok); a fresh compile per cell:
        # dynamo's per-function cache would otherwise fill up across the cells'
        # shapes and buffers and fall back to eager
        torch._dynamo.reset()
        c = _TorchState(n)
        for t in (c.x, c.v, c.k, c.nw2, c.cc, c.nstop):
            torch._dynamo.mark_static_address(t)
        step = torch.compile(_torch_step_inplace, dynamic=False, mode="reduce-overhead")

        def run_compiled():
            with torch.no_grad():
                torch.mul(c.omega, c.omega, out=c.nw2)
                c.nw2.neg_()
                torch.mul(2.0 * c.zeta, c.omega, out=c.cc)
                for _ in range(max_steps):
                    if not bool(step(c.x, c.v, c.k, c.nw2, c.cc, c.nstop)):
                        break
            c.out = (c.x, c.v, c.k)

        arms["torch_compiled"] = dict(state=c, run=run_compiled, download=c.download)
    return arms


# --------------------------------------------------------------------------- #
# Warp arm (an explicit per-thread kernel, written the way Warp users write it)
# --------------------------------------------------------------------------- #
_WARP_KERNEL = []


def _warp_kernel():
    """The Warp kernel (defined once per process; Warp compiles its module on
    first launch, into its kernel cache)."""
    if _WARP_KERNEL:
        return _WARP_KERNEL[0]
    wp = cc.warp_init()

    # >>> code:warp_kernel
    @wp.kernel
    def oscillators(omega: wp.array(dtype=wp.float64), zeta: wp.array(dtype=wp.float64),
                    nstop: wp.array(dtype=wp.float64), x: wp.array(dtype=wp.float64),
                    v: wp.array(dtype=wp.float64), k: wp.array(dtype=wp.float64),
                    dt: wp.float64, h2: wp.float64, h6: wp.float64, cap: int):
        # one thread per sample: its own RK4 steps, up to its own stop step
        # and no further than the step cap
        i = wp.tid()
        xs = x[i]
        vs = v[i]
        ks = wp.float64(0.0)
        ns = nstop[i]
        w2 = omega[i] * omega[i]
        c = wp.float64(2.0) * zeta[i] * omega[i]
        two = wp.float64(2.0)
        it = int(0)
        while ks < ns and it < cap:
            a1 = -w2 * xs - c * vs
            x2 = xs + h2 * vs
            v2 = vs + h2 * a1
            a2 = -w2 * x2 - c * v2
            x3 = xs + h2 * v2
            v3 = vs + h2 * a2
            a3 = -w2 * x3 - c * v3
            x4 = xs + dt * v3
            v4 = vs + dt * a3
            a4 = -w2 * x4 - c * v4
            xs = xs + h6 * (vs + two * v2 + two * v3 + v4)
            vs = vs + h6 * (a1 + two * a2 + two * a3 + a4)
            ks = ks + wp.float64(1.0)
            it = it + 1
        x[i] = xs
        v[i] = vs
        k[i] = ks
    # <<< code:warp_kernel

    _WARP_KERNEL.append(oscillators)
    return oscillators


_WARP_KERNEL32 = []


def _warp_kernel32():
    """The Warp kernel of :func:`_warp_kernel` in float32 (the FP32 pass)."""
    if _WARP_KERNEL32:
        return _WARP_KERNEL32[0]
    wp = cc.warp_init()

    @wp.kernel
    def oscillators(omega: wp.array(dtype=wp.float32), zeta: wp.array(dtype=wp.float32),
                    nstop: wp.array(dtype=wp.float32), x: wp.array(dtype=wp.float32),
                    v: wp.array(dtype=wp.float32), k: wp.array(dtype=wp.float32),
                    dt: wp.float32, h2: wp.float32, h6: wp.float32, cap: int):
        i = wp.tid()
        xs = x[i]
        vs = v[i]
        ks = wp.float32(0.0)
        ns = nstop[i]
        w2 = omega[i] * omega[i]
        c = wp.float32(2.0) * zeta[i] * omega[i]
        two = wp.float32(2.0)
        it = int(0)
        while ks < ns and it < cap:
            a1 = -w2 * xs - c * vs
            x2 = xs + h2 * vs
            v2 = vs + h2 * a1
            a2 = -w2 * x2 - c * v2
            x3 = xs + h2 * v2
            v3 = vs + h2 * a2
            a3 = -w2 * x3 - c * v3
            x4 = xs + dt * v3
            v4 = vs + dt * a3
            a4 = -w2 * x4 - c * v4
            xs = xs + h6 * (vs + two * v2 + two * v3 + v4)
            vs = vs + h6 * (a1 + two * a2 + two * a3 + a4)
            ks = ks + wp.float32(1.0)
            it = it + 1
        x[i] = xs
        v[i] = vs
        k[i] = ks

    _WARP_KERNEL32.append(oscillators)
    return oscillators


class _WarpState:
    def __init__(self, n, dtype="float64"):
        import warp as wp

        self.wp = wp
        self.omega, self.zeta, self.nstop, self.x, self.v, self.k = (
            wp.empty(n, dtype=getattr(wp, dtype), device="cuda:0") for _ in range(6))

    def upload(self, inp):
        for dst, key in ((self.omega, "omega"), (self.zeta, "zeta"),
                         (self.nstop, "nstop"), (self.x, "x0"), (self.v, "v0")):
            dst.assign(inp[key])
        self.wp.synchronize_device("cuda:0")

    def download(self):
        return self.x.numpy().copy(), self.v.numpy().copy(), self.k.numpy().copy()


def _make_warp_arm(n, max_steps, dtype="float64"):
    import warp as wp

    kernel = _warp_kernel() if dtype == "float64" else _warp_kernel32()
    s = _WarpState(n, dtype)
    real = getattr(wp, dtype)
    # >>> code:warp_launch
    args = [s.omega, s.zeta, s.nstop, s.x, s.v, s.k,
            real(DT), real(H2), real(H6), int(max_steps)]

    def run():
        wp.launch(kernel, dim=n, inputs=args, device="cuda:0")
        wp.synchronize_device("cuda:0")
    # <<< code:warp_launch

    return dict(state=s, run=run, download=s.download)


# >>> code:jax_vmap
def _jax_fn(cap):
    """jit(vmap(per-sample while_loop)) -- the CPU card's JAX formulation, the
    loop condition also holding the step cap."""
    import jax
    import jax.numpy as jnp
    from jax import lax

    jax.config.update("jax_enable_x64", True)

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

    return jax.jit(jax.vmap(one))
# <<< code:jax_vmap


def _jax_compile(n, dtype="float64", cap=1000):
    """The jax_vmap executable for shape (n,) and step cap ``cap``, from
    abstract shapes alone."""
    import jax
    import jax.numpy as jnp

    fn = _jax_fn(cap)
    spec = jax.ShapeDtypeStruct((n,), getattr(jnp, dtype), sharding=jax.sharding.SingleDeviceSharding(jax.devices("gpu")[0]))
    return fn.lower(spec, spec, spec, spec, spec).compile()


class _JaxState:
    def __init__(self, n):
        self.args = None
        self.out = None

    def upload(self, inp):
        import jax

        dev = jax.devices("gpu")[0]
        self.args = tuple(jax.device_put(inp[k], dev)
                          for k in ("omega", "zeta", "x0", "v0", "nstop"))
        jax.block_until_ready(self.args)

    def download(self):
        import numpy as np

        return tuple(np.asarray(a) for a in self.out)


def _make_jax_arm(n, max_steps, compiled=None, dtype="float64"):
    import jax

    t0 = time.perf_counter()
    if compiled is None:
        compiled = _jax_compile(n, dtype, max_steps)
    compile_s = time.perf_counter() - t0
    s = _JaxState(n)

    def run():
        s.out = jax.block_until_ready(compiled(*s.args))

    return dict(state=s, run=run, download=s.download, compile_s=compile_s)


#: Elementwise operations per step in run_cupy, as (float64 array operands
#: read, bool operands read, float64 results written, bool results written):
#: the analytic issued-bytes count for cupy_masked (see _COUNTING_RULE).
_CUPY_STEP_OPS = (
    # running = k < nstop ; running.any()
    (2, 0, 0, 1), (0, 1, 0, 0),
    # a1 = nw2*x - cc*v : 3 ops
    (2, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0),
    # v2 = v + h2*a1 : 2 ops
    (1, 0, 1, 0), (2, 0, 1, 0),
    # a2 = nw2*(x + h2*v) - cc*v2 : 5 ops
    (1, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0),
    # v3 : 2 ops
    (1, 0, 1, 0), (2, 0, 1, 0),
    # a3 : 5 ops
    (1, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0),
    # v4 : 2 ops
    (1, 0, 1, 0), (2, 0, 1, 0),
    # a4 : 5 ops
    (1, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0),
    # xn = x + h6*(((v + 2*v2) + 2*v3) + v4) : 7 ops
    (1, 0, 1, 0), (2, 0, 1, 0), (1, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0),
    (1, 0, 1, 0), (2, 0, 1, 0),
    # vn : 7 ops
    (1, 0, 1, 0), (2, 0, 1, 0), (1, 0, 1, 0), (2, 0, 1, 0), (2, 0, 1, 0),
    (1, 0, 1, 0), (2, 0, 1, 0),
    # x, v = where(running, ., .) ; k = k + running
    (2, 1, 1, 0), (2, 1, 1, 0), (1, 1, 1, 0),
)


def _cupy_issued_bytes(n, steps_launched):
    per_step = sum(8 * f + b + 8 * fw + bw for f, b, fw, bw in _CUPY_STEP_OPS) * n
    return per_step * steps_launched


def _kernel_issued_bytes(running, n_finishing, n):
    """Issued bytes of the step kernel over one run."""
    launches = len(running)
    return (STEP_BYTES_ALL * n * launches + STEP_BYTES_ACTIVE * int(running.sum())
            + STEP_BYTES_FINISH * n_finishing)


def _compact_issued_bytes(running, n_finishing, n, every):
    """Issued bytes of the eagle_graph_compact arm over one run: the loop runs
    whole blocks of ``every`` steps; the step kernel touches the MAPPED positions
    (all n until the first compaction, then the samples running at the last
    compaction); a compaction runs at the end of a block in which a sample
    finished."""
    import numpy as np

    steps = len(running)
    blocks = -(-steps // every)
    padded = np.concatenate((running, np.zeros(blocks * every - steps, dtype=running.dtype)))
    mapped, total, compactions = n, 0, 0
    for b in range(blocks):
        block = padded[b * every:(b + 1) * every]
        total += (STEP_BYTES_ALL + MAP_BYTES) * mapped * every
        total += STEP_BYTES_ACTIVE * int(block.sum())
        after = int(padded[(b + 1) * every]) if (b + 1) * every < len(padded) else 0
        start = int(block[0])
        if after < start or (b == blocks - 1 and start > 0):
            compactions += 1
            mapped = after
    return total + STEP_BYTES_FINISH * n_finishing + COMPACT_BYTES_PER_SAMPLE * n * compactions


#: eagle's automatic policy as its documentation states it (``eagle.until_done``):
#: the first launch takes AUTO_K0 steps; after each launch k doubles when
#: fewer than 1/16 of the live samples finished in it and halves when more
#: than 1/4 did, clamped to [AUTO_K_MIN, AUTO_K_MAX]; the last launch takes
#: exactly the steps left. The card replays it to count the arm's bytes, and
#: checks the replay against the launches each run reports
#: (``per_config.auto_schedule_matches_run``).
AUTO_K0, AUTO_K_MIN, AUTO_K_MAX = 16, 8, 64


def _next_k(k, newly, live):
    if newly * 16 < live:
        return min(2 * k, AUTO_K_MAX)
    if newly * 4 > live:
        return max(k // 2, AUTO_K_MIN)
    return k


def _auto_schedule(running, n, max_steps):
    """The launches of the eagle_graph_auto arm, ``[(first step, k), ...]``,
    replayed from the stop-step profile with eagle's policy (AUTO_K0)."""

    def finished_at(step):
        return n - (int(running[step]) if step < len(running) else 0)

    out, k, step = [], AUTO_K0, 0
    fin_at = finished_at(0)
    word = min(k, max_steps)
    while fin_at != n and step < max_steps:
        out.append((step, word))
        step += word
        fin = finished_at(step)
        k = _next_k(k, fin - fin_at, n - fin)
        fin_at = fin
        word = min(k, max_steps - step)
    return out


#: AUTO_ARM/SIMULATE_ARM auto-route past the band the same way until_done's
#: public ``_fast_mode_for`` does (capacity: below it, one fused launch;
#: above, the persist launch); only below some device-specific capacity AND
#: with no reorder requested does ``eagle_graph`` ever take the band these
#: two arms model with :func:`_auto_schedule`/:func:`_auto_issued_bytes` --
#: so every row below records the mode the runner actually took
#: (``report.mode``) and the model/schedule check follows it, instead of
#: assuming band unconditionally.
def _fast_schedule_matches(report, max_steps) -> bool:
    """A fast-path run's own one-launch "schedule": exactly one launch of
    the whole budget as its word -- bit-identical by construction, so
    there is no per-launch doubling sequence to replay against."""
    return report.launches == 1 and dict(report.launches_by_k) == {max_steps: 1}


def _schedule_matches_run(report, replay, max_steps) -> bool:
    """Whether the row's actual launch schedule matches what the card
    modelled for it: the band's doubling replay (``replay``, from
    :func:`_auto_schedule`) when the runner actually took the band path,
    the fast path's own one-launch contract otherwise."""
    if report.mode == "band":
        return replay == dict(report.launches_by_k)
    return _fast_schedule_matches(report, max_steps)


def _persist_issued_bytes(running, n_finishing, n):
    """Issued bytes of the persist arm: ONE launch, no map, no policy
    kernel -- every lane's planes load/store once per step it actually
    runs (summed over the batch, as the step kernel's active traffic),
    plus the one-time load/store the grid's lanes cover n with (one pass,
    not one per launch -- there is only the one launch) and the finish
    write."""
    return (STEP_BYTES_ALL * n + STEP_BYTES_ACTIVE * int(running.sum())
            + STEP_BYTES_FINISH * n_finishing)


def _auto_issued_bytes(running, n_finishing, n, max_steps):
    """Issued bytes of the eagle_graph_auto arm: per launch the step kernel's
    count over the MAPPED positions (as :func:`_compact_issued_bytes`, once per
    launch), and a compaction after each launch in which a sample finished."""
    mapped, total, compactions = n, 0, 0
    for first, k in _auto_schedule(running, n, max_steps):
        start = int(running[first])
        total += (STEP_BYTES_ALL + MAP_BYTES) * mapped + STEP_BYTES_ACTIVE * start
        after = int(running[first + k]) if first + k < len(running) else 0
        if after < start:
            compactions += 1
            mapped = after
    return total + STEP_BYTES_FINISH * n_finishing + COMPACT_BYTES_PER_SAMPLE * n * compactions


def _reorder_issued_bytes(running, n_finishing, n, every, theta):
    """Issued bytes of the eagle_graph_reorder arm over one run: as
    :func:`_compact_issued_bytes`, plus the moves of every reorder. A compaction fires a
    reorder by the trigger rule (``count < theta * 32 * live32`` and
    ``count < span``), with ``live32`` the expected number of occupied
    32-sample groups when the ``count`` live samples sit at random inside the
    ``span`` (the layout a reorder leaves, thinned at random); a fire moves
    ``span`` samples and sets ``span = count``. Returns (bytes, reorders)."""
    import numpy as np

    steps = len(running)
    blocks = -(-steps // every)
    padded = np.concatenate((running, np.zeros(blocks * every - steps, dtype=running.dtype)))
    mapped, span, total, compactions, reorders = n, n, 0, 0, 0
    for b in range(blocks):
        block = padded[b * every:(b + 1) * every]
        total += (STEP_BYTES_ALL + MAP_BYTES) * mapped * every
        total += STEP_BYTES_ACTIVE * int(block.sum())
        after = int(padded[(b + 1) * every]) if (b + 1) * every < len(padded) else 0
        start = int(block[0])
        if after < start or (b == blocks - 1 and start > 0):
            compactions += 1
            mapped = after
            groups = -(-span // 32)
            live32 = groups * (1.0 - (1.0 - after / span) ** 32) if span else 0.0
            if after < theta * 32 * live32 and after < span:
                reorders += 1
                total += REORDER_BYTES_PER_SPAN_SAMPLE * span + COMPACT_BYTES_PER_SAMPLE * n
                span = after
    total += STEP_BYTES_FINISH * n_finishing + COMPACT_BYTES_PER_SAMPLE * n * compactions
    return total, reorders


class _CpuArm:
    def __init__(self, n, max_steps):
        import numpy as np

        import eagle

        oscillator_step = _define_kernel()
        # >>> code:cpu_openmp_deploy
        # cpu_openmp: the same kernel; its planes on the host select the host build
        self.plan = eagle.deploy(oscillator_step)
        # <<< code:cpu_openmp_deploy
        self.n = n
        self.max_steps = max_steps
        self.omega = np.empty(n)
        self.zeta = np.empty(n)
        self.nstop = np.empty(n)
        self.x = np.empty(n)
        self.v = np.empty(n)
        self.k = np.zeros(n)
        self.terminated = np.zeros(n, dtype=bool)
        self.kernel_s = None  # the steps and the checks run inside one call

    def upload(self, inp):
        import numpy as np

        np.copyto(self.omega, inp["omega"])
        np.copyto(self.zeta, inp["zeta"])
        np.copyto(self.nstop, inp["nstop"])
        np.copyto(self.x, inp["x0"])
        np.copyto(self.v, inp["v0"])
        self.k.fill(0.0)
        self.terminated.fill(False)

    # >>> code:cpu_openmp
    def run(self):
        import eagle

        eagle.run_until_done(self.plan, max_steps=self.max_steps, dt=DT,
                             omega=self.omega, zeta=self.zeta, nstop=self.nstop,
                             terminated=self.terminated, x=self.x, v=self.v,
                             k=self.k)
    # <<< code:cpu_openmp

    def download(self):
        return self.x.copy(), self.v.copy(), self.k.copy()


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #
_sync = cc.sync


def _timed(arm, inp, is_gpu):
    """One run: (wall, end_to_end, outputs). end_to_end spans input upload,
    the run and the output download; wall spans the run alone."""
    t_a = time.perf_counter()
    arm["state"].upload(inp)
    if is_gpu:
        _sync()
    t_b = time.perf_counter()
    arm["run"]()
    if is_gpu:
        _sync()
    t_c = time.perf_counter()
    out = arm["download"]()
    t_d = time.perf_counter()
    return t_c - t_b, t_d - t_a, out


_median_iqr, _gpu_state, _ClockPoller = cc.median_iqr, cc.gpu_state, cc.ClockPoller
_trimmed = cc.trimmed


#: Cool-down gate before every arm block: wait until the GPU is within
#: GATE_DELTA_C of its idle temperature and reads 0 % utilisation, at most
#: GATE_MAX_S, polling every GATE_POLL_S. The idle baseline is read once at
#: card start (:func:`_gate_init`).
GATE_DELTA_C = 3
GATE_MAX_S = 20.0
GATE_POLL_S = 0.05

#: NVML clocks-throttle-reason bits (nvml.h), by name.
_THROTTLE_BITS = (
    (0x1, "idle"), (0x2, "applications clocks setting"), (0x4, "power cap"),
    (0x8, "HW slowdown"), (0x10, "sync boost"), (0x20, "SW thermal slowdown"),
    (0x40, "HW thermal slowdown"), (0x80, "HW power brake"),
    (0x100, "display clock setting"))

#: The NVML handle of the timed GPU (None: no NVML, the gate and the clock
#: record are skipped) and the idle temperature the gate compares against.
_NVML = {"handle": None, "baseline_c": None}
_now, _sleep = time.perf_counter, time.sleep


def _nvml_call(name, *extra):
    """One NVML reading on the timed GPU, or None when NVML is not usable."""
    h = _NVML["handle"]
    if h is None:
        return None
    try:
        import pynvml

        return getattr(pynvml, name)(h, *extra)
    except Exception:
        return None


def _nvml_temp():
    import pynvml

    t = _nvml_call("nvmlDeviceGetTemperature", pynvml.NVML_TEMPERATURE_GPU)
    return None if t is None else float(t)


def _nvml_util():
    u = _nvml_call("nvmlDeviceGetUtilizationRates")
    return None if u is None else int(u.gpu)


def _nvml_sm_mhz():
    import pynvml

    c = _nvml_call("nvmlDeviceGetClockInfo", pynvml.NVML_CLOCK_SM)
    return None if c is None else float(c)


def _nvml_throttle_mask():
    m = _nvml_call("nvmlDeviceGetCurrentClocksThrottleReasons")
    return None if m is None else int(m)


def throttle_names(mask):
    """The names of the throttle-reason bits set in ``mask`` (empty: none)."""
    return [name for bit, name in _THROTTLE_BITS if mask & bit]


def _gate_init(bus_id):
    """Resolve the NVML handle and read the idle temperature once (call at
    card start, device initialised and idle). Returns the gate's method record
    for the card's JSON; with no NVML the gate is recorded as skipped."""
    _NVML["handle"] = cc._nvml_handle(bus_id)
    _NVML["baseline_c"] = _nvml_temp() if _NVML["handle"] is not None else None
    return _gate_record()


def _gate_record():
    skipped = _NVML["baseline_c"] is None
    return {
        "delta_c": GATE_DELTA_C, "max_s": GATE_MAX_S, "poll_s": GATE_POLL_S,
        "idle_baseline_c": _NVML["baseline_c"],
        "skipped": "NVML unavailable" if skipped else None,
        "clock_record": (None if skipped else
                         "SM clock and throttle reasons read via NVML right after "
                         "each timed run, outside the timed region"),
    }


def _cool_down_gate():
    """Poll NVML every GATE_POLL_S until the temperature is at most the idle
    baseline + GATE_DELTA_C and the utilisation reads 0, or GATE_MAX_S has
    passed. Returns {waited_s, temp_c, gate_timed_out}, or None when the gate
    is skipped (no NVML). The polling is outside every timed region."""
    base = _NVML["baseline_c"]
    if _NVML["handle"] is None or base is None:
        return None
    t0 = _now()
    while True:
        temp, util = _nvml_temp(), _nvml_util()
        if temp is None or util is None:
            return None
        waited = _now() - t0
        if temp <= base + GATE_DELTA_C and util == 0:
            return {"waited_s": waited, "temp_c": temp, "gate_timed_out": False}
        if waited >= GATE_MAX_S:
            return {"waited_s": waited, "temp_c": temp, "gate_timed_out": True}
        _sleep(GATE_POLL_S)


def _clock_record():
    """(SM clock in MHz, throttle-reason names) now, or None without NVML."""
    mhz, mask = _nvml_sm_mhz(), _nvml_throttle_mask()
    if mhz is None or mask is None:
        return None
    return mhz, throttle_names(mask)


def _time_arm_block(arm, inp, is_gpu, runs, after_run=None, meta=None):
    """One arm's own timed block: the cool-down gate, an untimed ramp of
    back-to-back runs until RAMP_S has elapsed (at least one run), then
    ``runs`` timed runs. Returns ([wall], [end_to_end]); ``after_run`` is
    called after each timed run. When ``meta`` is a dict it receives the gate
    record (``gate``) and, for a GPU arm with NVML, the SM clock and throttle
    names read right after each timer stopped (``sm_mhz``, ``throttle``)."""
    gate = _cool_down_gate()
    if meta is not None and gate is not None:
        meta["gate"] = gate
    t0 = time.perf_counter()
    while True:
        _timed(arm, inp, is_gpu)
        if time.perf_counter() - t0 >= RAMP_S:
            break
    walls, e2es = [], []
    for _ in range(runs):
        w, e2e, _ = _timed(arm, inp, is_gpu)
        walls.append(w)
        e2es.append(e2e)
        if meta is not None and is_gpu:
            rec = _clock_record()  # after the timer stopped
            if rec is not None:
                meta.setdefault("sm_mhz", []).append(rec[0])
                meta.setdefault("throttle", []).append(rec[1])
        if after_run is not None:
            after_run()
    return walls, e2es


def _arm_stat(walls, meta):
    """The arm's wall statistic, plus (when recorded) its gate record, the SM
    clock and throttle names of each timed run, and ``cycles``: the trimmed
    statistic of wall x SM clock per run, the time in SM clock cycles."""
    stat = _trimmed(walls)
    if meta.get("gate"):
        stat["gate"] = meta["gate"]
    sm = meta.get("sm_mhz")
    if sm and len(sm) == len(walls):
        stat["sm_mhz"] = list(sm)
        stat["throttle"] = [list(t) for t in meta["throttle"]]
        stat["cycles"] = _trimmed([w * m * 1e6 for w, m in zip(walls, sm)])
    return stat


def _clock_range(stat):
    """'lo-hi' of the SM clocks (MHz) a wall statistic was timed at, or None
    for a card that did not record them."""
    sm = stat.get("sm_mhz")
    if not sm:
        return None
    return f"{min(sm):.0f}\u2013{max(sm):.0f}"


def _verify(outs, n, nstop):
    """All arms agree before timing counts: identical step counts, states
    within the stated tolerance of the cpu_openmp reference."""
    return cc.verify_xvk(outs, "cpu_openmp", nstop, ATOL)


#: Agreement tolerance on x and v (absolute; |x0|, |v0| <= 1 and the
#: oscillators are damped, so states stay O(1) or below). Arms may differ in
#: rounding (fused multiply-add contraction differs between compilers and
#: between one fused kernel and many elementwise ones).
ATOL = 1e-11


def _wait_for_host_builds():
    """Join eagle's background host builds (a deploy with a GPU leaves its
    host side building on its own thread), so none runs while an arm is
    timed."""
    for thread in threading.enumerate():
        if thread.name == "eagle-host-build":
            thread.join()


def _check_cap(arms, names, inp, max_steps, gpu_names):
    """The cap fixture: a few samples whose stop step is twice the cap. Every
    arm must report k == min(nstop, max_steps) per sample, and the same finished
    count (samples that reached their own stop), or the card run fails."""
    import numpy as np

    nstop = inp["nstop"]
    nstop[:8] = 2 * max_steps
    want = np.minimum(nstop, max_steps)
    want_done = int((want >= nstop).sum())
    for name in names:
        _, _, out = _timed(arms[name], inp, name in gpu_names)
        k = np.asarray(out[2])
        assert np.array_equal(k, want), \
            f"{name}: step counts differ from min(nstop, {max_steps}) in the cap fixture"
        done = int((k >= nstop).sum())
        assert done == want_done, \
            f"{name}: finished {done}, expected {want_done}, in the cap fixture"
    return {"n": len(nstop), "max_steps": max_steps, "capped_samples": 8,
            "finished": want_done, "arms": list(names), "passed": True}


def _cap_fixture_main(max_steps=CAP_FIXTURE_S, n=CAP_FIXTURE_N):
    import cupy as cp

    run_arms = _run_arms()
    inp = _inputs(n, "spread", max_steps)
    arms = _make_gpu_arms(n, max_steps)
    cpu = _CpuArm(n, max_steps)
    arms["cpu_openmp"] = dict(state=cpu, run=cpu.run, download=cpu.download)
    _wait_for_host_builds()
    row = _check_cap(arms, run_arms, inp, max_steps,
                     tuple(a for a in run_arms if a != "cpu_openmp"))
    del arms
    cp.get_default_memory_pool().free_all_blocks()
    cc.torch_empty_cache()
    return row


def _cap_fixture_fp32(max_steps=CAP_FIXTURE_S, n=CAP_FIXTURE_N):
    import cupy as cp

    inp = _inputs(n, "spread", max_steps, "float32")
    arms = _make_fp32_arms(n, max_steps)
    _wait_for_host_builds()
    row = _check_cap(arms, FP32_ARMS, inp, max_steps, FP32_ARMS)
    del arms
    cp.get_default_memory_pool().free_all_blocks()
    return row


def _run_config(n, distribution, max_steps, reps, bus_id, kernel_pass=False):
    import cupy as cp
    import numpy as np

    run_arms = _run_arms()
    inp = _inputs(n, distribution, max_steps)
    running = _active_profile(inp["nstop"], max_steps)
    sample_steps = int(running.sum())
    if kernel_pass:
        from cupy.cuda import nvtx

        arms = _make_gpu_arms(n, max_steps, names=kernel_pass)
        _wait_for_host_builds()
        for name in kernel_pass:
            for _ in range(WARMUP_RUNS):
                _timed(arms[name], inp, True)  # warm
            # the mode the last warm-up run took (None for an arm with no
            # report): settled after the warm-up, so reading it once here
            # covers every rep below -- carried in the NVTX tag so
            # _expected_kernels (the nsys pass) knows whether this row's
            # kernel count is the band's data-dependent lower bound or the
            # fast path's exact one launch.
            mode = getattr(arms[name].get("report"), "mode", None) or ""
            for r in range(KERNEL_PASS_RUNS[name]):
                arms[name]["state"].upload(inp)
                _sync()
                nvtx.RangePush(f"perfcard|{distribution}|{max_steps}|{n}|{name}|{mode}|{r}")
                arms[name]["run"]()
                _sync()
                nvtx.RangePop()
        return None

    arms = _make_gpu_arms(n, max_steps)
    cpu = _CpuArm(n, max_steps)
    arms["cpu_openmp"] = dict(state=cpu, run=cpu.run, download=cpu.download)

    # warm-up + verification (excluded from timing)
    outs = {}
    for name in run_arms:
        for _ in range(WARMUP_RUNS):
            _, _, outs[name] = _timed(arms[name], inp, name != "cpu_openmp")
    worst = _verify(outs, n, inp["nstop"])
    if not arms["eagle_graph"]["state"].runner.auto:
        # one step per launch; the plain decorator's automatic kernel instead
        # takes the steps its policy picks (its loop count is a cap, not a count)
        assert arms["eagle_graph"]["iterations"]() == max_steps
    launches = -(-max_steps // STEPS_PER_LAUNCH)
    assert arms["eagle_graph_steps"]["iterations"]() == launches
    steps_identical = all(np.array_equal(a, b) for a, b in
                          zip(outs["eagle_graph"], outs["eagle_graph_steps"]))

    _wait_for_host_builds()
    state_before = _gpu_state(bus_id)
    poller = _ClockPoller(bus_id)
    walls = {a: [] for a in run_arms}
    e2es = {a: [] for a in run_arms}
    metas = {a: {} for a in run_arms}
    cpu_kernel = []
    # every arm is timed in its own block, in the arm order (no interleaving)
    for name in run_arms:
        def _grab_cpu_kernel():
            if cpu.kernel_s is not None:
                cpu_kernel.append(cpu.kernel_s)
        walls[name], e2es[name] = _time_arm_block(
            arms[name], inp, name != "cpu_openmp", reps,
            _grab_cpu_kernel if name == "cpu_openmp" else None, metas[name])
    state_after = _gpu_state(bus_id)
    during = poller.stop()

    n_finishing = n
    # AUTO_ARM/SIMULATE_ARM model their issued bytes from whatever mode the
    # runner actually took (see _schedule_matches_run above): the band's
    # per-launch map/compaction count only applies when it ran the band;
    # a fast-path run issues the persist model's traffic (one pass over the
    # whole batch, no map) whichever of the two fast entries it used.
    auto_mode = arms[AUTO_ARM]["report"].mode
    sim_mode = arms[SIMULATE_ARM]["report"].mode
    issued = {
        "eagle_graph": _kernel_issued_bytes(
            running[[first for first, _k in _auto_schedule(running, n, max_steps)]],
            n_finishing, n),
        "eagle_graph_compact": _compact_issued_bytes(running, n_finishing, n,
                                                     COMPACT_EVERY),
        "eagle_graph_reorder": _reorder_issued_bytes(running, n_finishing, n,
                                                     COMPACT_EVERY, REORDER_THETA)[0],
        # per launch: the running count at each launch's first step
        "eagle_graph_steps": _kernel_issued_bytes(running[::STEPS_PER_LAUNCH],
                                                  n_finishing, n),
        "eagle_graph_steps_compact": _compact_issued_bytes(
            running[::STEPS_PER_LAUNCH], n_finishing, n,
            STEPS_COMPACT_EVERY // STEPS_PER_LAUNCH),
        AUTO_ARM: (_auto_issued_bytes(running, n_finishing, n, max_steps)
                  if auto_mode == "band" else _persist_issued_bytes(running, n_finishing, n)),
        PERSIST_ARM: _persist_issued_bytes(running, n_finishing, n),
        SIMULATE_ARM: (_auto_issued_bytes(running, n_finishing, n, max_steps)
                       if sim_mode == "band" else _persist_issued_bytes(running, n_finishing, n)),
        "eager_loop": _kernel_issued_bytes(running, n_finishing, n),
        "cupy_masked": _cupy_issued_bytes(n, max_steps),
        "torch_masked": _cupy_issued_bytes(n, max_steps),
        "torch_graphed": (_cupy_issued_bytes(n, -(-max_steps // GRAPH_BLOCK) * GRAPH_BLOCK)
                          + TORCH_BLOCK_BYTES_PER_SAMPLE * n * -(-max_steps // GRAPH_BLOCK)),
        "torch_compiled": JAX_BYTES_PER_SAMPLE_ITER * n * max_steps,
        "warp_kernel": WARP_BYTES_PER_SAMPLE * n,
        "jax_vmap": (JAX_BYTES_PER_SAMPLE_ITER * n * max_steps
                     + JAX_BYTES_PER_SAMPLE_ONCE * n),
        "cpu_openmp": (STEP_BYTES_ALL * n * max_steps + STEP_BYTES_ACTIVE * sample_steps
                       + STEP_BYTES_FINISH * n_finishing),
    }
    rows = []
    for name in run_arms:
        wall = _arm_stat(walls[name], metas[name])
        med = wall["mean"]
        flops = FLOPS_PER_SAMPLE_STEP * sample_steps
        useful_bytes = USEFUL_BYTES_PER_SAMPLE_STEP * sample_steps
        rows.append({
            "arm": name, "n": n, "distribution": distribution,
            "max_steps": max_steps,
            "wall_s": wall,
            "end_to_end_s": _trimmed(e2es[name]),
            "kernel_only_s": (statistics.median(cpu_kernel)
                              if name == "cpu_openmp" and cpu_kernel else None),
            "kernel_only_source": ("perf_counter around each host-team launch"
                                   if name == "cpu_openmp" and cpu_kernel else None),
            "sample_steps": sample_steps,
            "sample_steps_per_s": sample_steps / med,
            "useful_flops": flops,
            "useful_flops_per_s": flops / med,
            "useful_bytes": useful_bytes,
            "useful_bytes_per_s": useful_bytes / med,
            "issued_bytes": issued[name],
            "issued_bytes_per_s": issued[name] / med,
            "max_abs_diff_vs_cpu": worst[name],
            "lane_utilisation": (
                getattr(arms[name].get("report"), "lane_utilisation", None)
                if name in arms else None),
            # the launch shape the runner actually took this row ("band",
            # "fused_one" or "persist"; None for an arm with no report --
            # cupy/torch/jax/warp/cpu_openmp): AUTO_ARM/SIMULATE_ARM auto-pick
            # past the band at/above capacity.
            "run_mode": (getattr(arms[name].get("report"), "mode", None)
                        if name in arms else None),
        })
    replay = {}
    for _first, k in _auto_schedule(running, n, max_steps):
        replay[k] = replay.get(k, 0) + 1
    auto_report = arms[AUTO_ARM]["report"]
    graph_report = arms["eagle_graph"]["report"]
    sim_report = arms[SIMULATE_ARM]["report"]
    extra = {"graph_build_s": arms["eagle_graph"]["build_s"],
             "graph_compact_build_s": arms["eagle_graph_compact"]["build_s"],
             "graph_reorder_build_s": arms["eagle_graph_reorder"]["build_s"],
             "graph_steps_build_s": arms["eagle_graph_steps"]["build_s"],
             "graph_steps_compact_build_s":
                 arms["eagle_graph_steps_compact"]["build_s"],
             "graph_auto_build_s": arms[AUTO_ARM]["build_s"],
             "auto_run_mode": auto_report.mode,
             "auto_launches_by_k": {str(k): c for k, c in
                                    auto_report.launches_by_k.items()},
             "auto_compactions": auto_report.compactions,
             "auto_schedule_matches_run":
                 _schedule_matches_run(auto_report, replay, max_steps),
             "graph_run_mode": graph_report.mode,
             "graph_launches_by_k": {str(k): c for k, c in
                                     graph_report.launches_by_k.items()},
             "graph_schedule_matches_run":
                 _schedule_matches_run(graph_report, replay, max_steps),
             "graph_simulate_build_s": arms[SIMULATE_ARM]["build_s"],
             "simulate_run_mode": sim_report.mode,
             "simulate_schedule_matches_run":
                 _schedule_matches_run(sim_report, replay, max_steps),
             "steps_bit_identical_to_graph": steps_identical,
             "reorders_fired": arms["eagle_graph_reorder"]["fired"][-1],
             "jax_compile_s": arms["jax_vmap"]["compile_s"],
             "torch_graph_capture_s": arms["torch_graphed"]["build_s"],
             "gpu_state_before": state_before, "gpu_state_after": state_after,
             "gpu_state_during": during,
             "running_fraction_mean": sample_steps / (n * max_steps)}
    del arms
    cp.get_default_memory_pool().free_all_blocks()
    cc.torch_empty_cache()
    return rows, extra


# --------------------------------------------------------------------------- #
# FP32 pass (eagle vs Warp, JAX as a third reference, all in float32)
# --------------------------------------------------------------------------- #
FP32_ARMS = ("eagle_simulate", "warp_kernel", "jax_vmap")
FP32_NS = (10_000, 100_000, 1_000_000)
FP32_DISTRIBUTIONS = ("uniform", "spread")
FP32_MAX_STEPS = 1000
#: Agreement tolerance on x and v in float32 (absolute; the states are O(1)
#: or below, one rounding is ~6e-8, and a thousand RK4 steps accumulate it).
FP32_ATOL = 1e-4


def _make_fp32_arms(n, max_steps):
    """The FP32 pass's arms for one N: eagle.simulate with
    ``scalar_type="float32"`` over float32 planes, and Warp and JAX written
    in float32."""
    import eagle

    oscillator_step = _define_auto_kernel()
    m = _GpuArm(n, "float32")
    m.runner = eagle.simulation(oscillator_step, omega=m.omega, zeta=m.zeta,
                                nstop=m.nstop, dt=DT, x=m.x, v=m.v, k=m.k,
                                max_steps=max_steps, scalar_type="float32")
    return {"eagle_simulate": _runner_arm(m),
            "warp_kernel": _make_warp_arm(n, max_steps, "float32"),
            "jax_vmap": _make_jax_arm(n, max_steps, dtype="float32")}


def _run_fp32_cell(n, distribution, max_steps, reps):
    """One FP32 cell: the arms warmed up (WARMUP_RUNS untimed runs, over which
    eagle's automatic arm settles its entry), checked against each other,
    then, for each arm in its own block, a ramp and REPS timed runs."""
    import cupy as cp
    import numpy as np

    inp = _inputs(n, distribution, max_steps, "float32")
    arms = _make_fp32_arms(n, max_steps)
    _wait_for_host_builds()
    outs = {}
    for name in FP32_ARMS:
        for _ in range(WARMUP_RUNS):
            _, _, outs[name] = _timed(arms[name], inp, True)
    ref = outs["warp_kernel"]
    diffs = {}
    for name in ("eagle_simulate", "jax_vmap"):
        o = outs[name]
        assert np.array_equal(np.asarray(o[2]), np.asarray(ref[2])), \
            f"{name}: step counts differ from warp_kernel (n={n}, {distribution})"
        diffs[name] = max(float(np.max(np.abs(np.asarray(o[i]) - np.asarray(ref[i]))))
                          for i in (0, 1))
        assert diffs[name] <= FP32_ATOL, \
            f"{name}: max |diff| vs warp_kernel {diffs[name]:g} > {FP32_ATOL:g}"
    # every arm is timed in its own block, in the arm order (no interleaving)
    metas = {a: {} for a in FP32_ARMS}
    walls = {a: _time_arm_block(arms[a], inp, True, reps, meta=metas[a])[0]
             for a in FP32_ARMS}
    row = {"n": n, "distribution": distribution, "max_steps": max_steps,
           "wall_s": {a: _arm_stat(walls[a], metas[a]) for a in FP32_ARMS},
           "max_abs_diff_vs_warp": diffs,
           "eagle_run_mode": arms["eagle_simulate"]["report"].mode}
    del arms
    cp.get_default_memory_pool().free_all_blocks()
    return row


def _gate_sentence():
    """The gate and clock record in plain words, for a method text ('' when
    the gate was skipped)."""
    g = _gate_record()
    if g["skipped"]:
        return ""
    return (f" Before each arm's block the GPU is cooled down: the block starts "
            f"once the temperature is within {g['delta_c']} \u00b0C of the idle "
            f"{g['idle_baseline_c']:g} \u00b0C and the GPU is idle, or after "
            f"{g['max_s']:g} s. Right after each timed run the SM clock is read "
            f"(its range is shown per arm); cycles are the wall time times that "
            f"clock, and eagle / Warp (cycles) is their ratio.")


def _run_fp32_pass(ns, reps, max_steps=FP32_MAX_STEPS):
    """The FP32 pass: every (distribution, N) cell, as the card's ``fp32``
    section (rendered by :func:`render_markdown`)."""
    cap_fixture = _cap_fixture_fp32()
    rows = []
    for dist in FP32_DISTRIBUTIONS:
        for n in ns:
            t0 = time.perf_counter()
            rows.append(_run_fp32_cell(n, dist, max_steps, reps))
            print(f"fp32 {dist:8s} S={max_steps:5d} N={n:>8,}  "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
    return {
        "cap_fixture": cap_fixture,
        "precision": "float32", "arms": list(FP32_ARMS), "ns": list(ns),
        "max_steps": max_steps, "repetitions": reps,
        "ramp_s": RAMP_S, "statistic": _statistic_text(reps),
        "cool_down_gate": _gate_record(),
        "warmup_runs": WARMUP_RUNS, "atol": FP32_ATOL,
        "method": (
            "Every arm runs the same oscillator in float32: eagle.simulate with "
            "scalar_type=\"float32\", NVIDIA Warp's per-thread kernel and JAX's "
            "vmapped loop, all over the same float32 inputs. "
            f"Each cell runs {WARMUP_RUNS} untimed runs per arm first (eagle's "
            "automatic arm picks its entry over them: the size rule, then a "
            "measured run of the other entry, keeping the faster; the entry it "
            "settled on is in the last column), then each arm in its own block "
            f"(no interleaving): an untimed ramp of back-to-back runs for "
            f"{RAMP_S:g} s, then {reps} timed runs. Before any timing counts, "
            "eagle and JAX reach step counts identical to Warp's and states "
            f"within absolute {FP32_ATOL:g} of Warp's (the largest differences "
            "are in the JSON). Each time is the mean of the middle "
            f"{reps - 2} of the {reps} runs (highest and lowest dropped), "
            f"with the range of those {reps - 2}; eagle / Warp is the ratio of "
            "these means: below 1 eagle is faster." + _gate_sentence()),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Compaction cadence sweep (isolates the active-set map as one variable)
# --------------------------------------------------------------------------- #
def _make_compaction_sweep_arms(n, max_steps, ks):
    """For each K in ``ks``, a cadence-matched pair of device-loop arms that
    isolates the active-set map as the SOLE variable: ``matched_cadence`` (the
    plain single step, the ``.step`` of eagle_graph's kernel, no map) and
    ``compact`` (eagle_graph_compact's active-set step kernel), both checking
    the loop's stop guard once every K steps. The only structural difference
    between the two arms of a pair is whether the step kernel reads the
    active-set map."""
    import cupy as cp

    import eagle
    from eagle import GraphPipeline, SkipGuard, repeat_while

    step_plan = eagle.deploy(_define_step_kernel())
    compact_plan = eagle.deploy(_define_active_kernel())
    out = {}
    for k in ks:
        iters = -(-max_steps // k)

        # matched_cadence: eagle_graph's own step kernel (no map), looped k
        # times between guard checks -- the same cadence as `compact` below.
        m = _GpuArm(n)
        done = cp.zeros(1, dtype=cp.uint32)
        total = cp.asarray([n], dtype=cp.uint32)
        m_step = step_plan.bind(dt=DT, finished_count=done.view(cp.int32),
                                **m.planes())

        def m_steps(m_step=m_step, k=k):
            for _ in range(k):
                m_step.launch()

        m_loop = repeat_while(m_steps, SkipGuard(done, 0, total, 0), iters)
        m_pipe = GraphPipeline().add(m_loop)
        m_pipe.build()
        cp.cuda.Device().synchronize()

        def m_run(m_pipe=m_pipe, done=done):
            done.fill(0)
            m_pipe.launch()

        # compact: eagle_graph_compact's active-set step kernel at cadence k.
        a = _GpuArm(n)
        a.runner = eagle.until_done(compact_plan, max_steps=max_steps, dt=DT,
                                    every=k, **a.planes())

        out[k] = {
            "matched_cadence": dict(state=m, run=m_run, download=m.download,
                                    iterations=m_loop.iterations),
            "compact": _runner_arm(a),
        }
    return out


def _run_compaction_sweep(n, distribution, max_steps, reps, ks):
    """Time the cadence-matched compaction-isolation pair (see
    :func:`_make_compaction_sweep_arms`) for every K in ``ks``, verified
    against a cpu_openmp reference and timed like the main table's arms (each variant in its own block).
    One row per (K, variant); rendered by :func:`render_markdown` as the
    "compaction cadence sweep" table."""
    import cupy as cp
    import numpy as np

    inp = _inputs(n, distribution, max_steps)
    running = _active_profile(inp["nstop"], max_steps)
    sample_steps = int(running.sum())

    cpu = _CpuArm(n, max_steps)
    cpu.upload(inp)
    cpu.run()
    ref = cpu.download()

    sweep_arms = _make_compaction_sweep_arms(n, max_steps, ks)
    rows = []
    variants = ("matched_cadence", "compact")
    for k in ks:
        iters = -(-max_steps // k)
        # warm-up + verify (excluded from timing)
        for name in variants:
            arm = sweep_arms[k][name]
            _, _, (x, v, kk) = _timed(arm, inp, True)
            assert np.array_equal(kk, inp["nstop"]), (
                f"compaction sweep K={k} {name}: step counts differ from the stop steps")
            dx = float(np.max(np.abs(x - ref[0])))
            dv = float(np.max(np.abs(v - ref[1])))
            assert max(dx, dv) <= ATOL, (
                f"compaction sweep K={k} {name}: state differs from cpu_openmp by "
                f"{max(dx, dv):.3e} > {ATOL:.1e}")
            assert arm["iterations"]() == iters

        metas = {name: {} for name in variants}
        walls = {name: _time_arm_block(sweep_arms[k][name], inp, True, reps,
                                       meta=metas[name])[0]
                 for name in variants}

        for name in variants:
            wall = _arm_stat(walls[name], metas[name])
            rows.append({
                "k": k, "variant": name, "n": n, "distribution": distribution,
                "max_steps": max_steps, "wall_s": wall,
                "sample_steps": sample_steps,
                "sample_steps_per_s": sample_steps / wall["mean"],
            })
    del sweep_arms
    cp.get_default_memory_pool().free_all_blocks()
    return rows


# --------------------------------------------------------------------------- #
# Memory pass (fresh process per cell) and compile-time pass (fresh process
# per measurement); neither is part of any wall
# --------------------------------------------------------------------------- #
#: Warm-up batch of the memory and compile-time probes.
WARM_N = 64
#: The state a sample needs: x, v (state width 2) plus the per-sample scalars
#: omega, zeta and the stop step, all float64 (the CPU card's minimum).
STATE_WIDTH = 2
PER_SAMPLE_SCALARS = 3
COMPILE_REPS = 3
#: Arms with no compile step of their own (PyTorch's eager kernels ship
#: prebuilt).
NO_COMPILE_ARMS = ("torch_masked", "torch_graphed")


def _min_bytes(n):
    return n * (STATE_WIDTH + PER_SAMPLE_SCALARS) * 8


def _host_peak_kib():
    # VmHWM, not ru_maxrss: ru_maxrss also keeps the RSS of the image this
    # process was forked from before its exec (the main benchmark process's), which a
    # clear_refs reset does not lower
    return cc.status_kib("VmHWM")


def _library_counters():
    """Each library's own view of its device memory (a cross-check)."""
    out = {}
    if "cupy" in sys.modules:
        pool = sys.modules["cupy"].get_default_memory_pool()
        out["cupy_pool_total_bytes"] = int(pool.total_bytes())
        out["cupy_pool_used_bytes"] = int(pool.used_bytes())
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        out["torch_max_memory_reserved"] = int(torch.cuda.max_memory_reserved())
        out["torch_max_memory_allocated"] = int(torch.cuda.max_memory_allocated())
    if "jax" in sys.modules:
        try:
            stats = sys.modules["jax"].devices()[0].memory_stats() or {}
            out.update({f"jax_{k}": int(v) for k, v in stats.items()
                        if k in ("peak_bytes_in_use", "bytes_in_use", "bytes_reserved",
                                 "pool_bytes", "peak_pool_bytes")})
        except Exception:  # a JAX build without memory_stats
            pass
    return out


def _probe_arm(arm, n, max_steps, jax_compiled=None):
    """One arm for a probe process (an eagle arm deploys its kernel here)."""
    if arm == "cpu_openmp":
        cpu = _CpuArm(n, max_steps)
        return dict(state=cpu, run=cpu.run, download=cpu.download)
    if arm == "jax_vmap":
        return _make_jax_arm(n, max_steps, compiled=jax_compiled)
    return _make_gpu_arms(n, max_steps, names=(arm,))[arm]


def _probe_once(a, inp):
    a["state"].upload(inp)
    a["run"]()
    cc.probe_sync()
    return a["download"]()


def _memory_probe(arm, distribution, max_steps, n, bus_id):
    """One fresh process: imports, the arm's kernels built and a warm-up run
    on a WARM_N batch (every JIT compiles here; JAX also compiles for shape N
    from abstract shapes), the caching allocators trimmed, the baseline read;
    then the batch's inputs generated, the arm built for N (eagle's graph,
    PyTorch's recorded CUDA graph) and one full run with upload and download;
    then the reading again, with every allocator still holding its
    high-water reservation."""
    import gc

    reply_fd = cc.probe_reply_fd()
    gpu = arm != "cpu_openmp"
    _probe_once(_probe_arm(arm, WARM_N, max_steps), _inputs(WARM_N, distribution, max_steps))
    jax_compiled = _jax_compile(n, "float64", max_steps) if arm == "jax_vmap" else None
    gc.collect()
    cc.trim_pools()
    cc.probe_sync()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.reset_peak_memory_stats()
    dev_base, dev_source = cc.process_device_bytes(bus_id) if gpu else (None, None)
    lib_base = _library_counters() if gpu else {}
    rss_base = cc.status_kib("VmRSS")
    reset_ok = cc.reset_host_peak()
    inp = _inputs(n, distribution, max_steps)
    a = _probe_arm(arm, n, max_steps, jax_compiled)
    x, v, k = _probe_once(a, inp)
    assert _np_array_equal(k, inp["nstop"]), f"memory probe {arm}: wrong step counts"
    dev_after, _ = cc.process_device_bytes(bus_id) if gpu else (None, None)
    host_peak = _host_peak_kib()
    cc.probe_reply(reply_fd, {
        "device_baseline_bytes": dev_base, "device_after_bytes": dev_after,
        "device_peak_bytes": None if not gpu else max(0, dev_after - dev_base),
        "device_source": dev_source,
        "library_counters_baseline": lib_base,
        "library_counters_after": _library_counters() if gpu else {},
        "host_rss_baseline_bytes": rss_base * 1024,
        "host_peak_bytes": max(0, host_peak - rss_base) * 1024,
        "host_peak_reset": reset_ok})


def _np_array_equal(a, b):
    import numpy as np

    return bool(np.array_equal(a, b))


def _run_memory_pass(ns, bus_id):
    rows = []
    for dist, steps in CONFIGS:
        for n in ns:
            t0 = time.perf_counter()
            for arm in _run_arms():
                r = cc.run_probe(__file__, ["--memory-probe", arm, dist, str(steps),
                                            str(n), bus_id],
                                 what=f"memory probe {arm} {dist} {steps} {n}")
                peak = r["device_peak_bytes"]
                rows.append({"arm": arm, "n": n, "distribution": dist, "max_steps": steps,
                             **r, "minimum_bytes": _min_bytes(n),
                             "device_factor": (None if peak is None
                                               else peak / _min_bytes(n))})
            print(f"memory {dist:8s} S={steps:5d} N={n:>8,}  "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)
    return rows


def _compile_keys(ns, arms=None):
    """(key, arms, what) of every compile measurement (``arms``: the arms run,
    default this machine's)."""
    arms = _run_arms() if arms is None else arms
    deploy = "eagle.deploy of the hawk {} (the device build; the host side builds in the background)"
    keys = [
        ("eagle:eagle_graph", ["eagle_graph"],
         deploy.format("step kernel") + ", the device-loop graph"),
        ("eagle:eagle_graph_compact", ["eagle_graph_compact"],
         deploy.format("active-set step kernel") + ", the compaction-loop graph"),
        ("eagle:eagle_graph_reorder", ["eagle_graph_reorder"],
         deploy.format("active-set step kernel") + ", the compaction + reorder "
         "loop graph"),
        ("eagle:eagle_graph_steps", ["eagle_graph_steps"],
         deploy.format("K-steps kernel") + ", the device-loop graph"),
        ("eagle:eagle_graph_steps_compact", ["eagle_graph_steps_compact"],
         deploy.format("active-set K-steps kernel") + ", the compaction-loop graph"),
        ("eagle:eagle_graph_auto", ["eagle_graph_auto"],
         deploy.format("active-set automatic kernel") + ", the policy kernel, "
         "the policy-loop graph"),
        ("eagle:eagle_graph_persist", ["eagle_graph_persist"],
         deploy.format("automatic kernel") + ", the persistent launch"),
        ("eagle:eagle_simulate", ["eagle_simulate"],
         "eagle.simulate's deploy of the hawk active-set automatic kernel (the device "
         "build; the host side builds in the background), the policy kernel, the "
         "policy-loop graph"),
        ("eagle:eager_loop", ["eager_loop"],
         deploy.format("single step kernel")),
        ("cupy", ["cupy_masked"],
         "CuPy's elementwise and reduction kernels of one masked step"),
    ]
    if "torch_compiled" in arms:
        for n in ns:
            keys.append((f"torch:{n}", ["torch_compiled"],
                         f"torch.compile (Inductor + Triton) and CUDA-graph record, N = {n:,}"))
    keys.append(("warp", ["warp_kernel"], "Warp kernel module (code generation, "
                 "NVRTC compile, load)"))
    for n in ns:
        keys.append((f"jax:{n}", ["jax_vmap"], f"XLA executable for N = {n:,}"))
    keys.append(("hawk_host", ["cpu_openmp"],
                 "eagle.deploy of the hawk step kernel and its first host run, which "
                 "waits for the host build"))
    return keys


def _compile_probe(key):
    """One fresh process: imports (untimed), then the time from the first call
    to a ready kernel (trace + code generation + compile, or a cache hit)."""
    reply_fd = cc.probe_reply_fd()
    tiny = _inputs(WARM_N, "uniform", 1)  # one step: the run itself is negligible
    if key.startswith("jax:"):
        import jax

        cc.jax_persistent_cache(os.environ["PERF_CARD_JAX_CACHE"])
        jax.devices("gpu")  # backend start-up untimed
        t0 = time.perf_counter()
        _jax_compile(int(key.split(":")[1]), "float64", 1)
        seconds = time.perf_counter() - t0
    elif key == "warp":
        cc.warp_init()  # runtime start-up untimed
        t0 = time.perf_counter()
        _probe_once(_make_warp_arm(WARM_N, 1), tiny)
        seconds = time.perf_counter() - t0
    elif key.startswith("torch:"):
        import torch

        n = int(key.split(":")[1])
        inp = _inputs(n, "uniform", 3)
        torch.zeros(1, device="cuda")  # context untimed
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # three steps: the first call compiles, the second records the
        # CUDA graph, the third replays it
        _probe_once(_make_torch_arms(n, 3, ("torch_compiled",))["torch_compiled"], inp)
        seconds = time.perf_counter() - t0
    else:
        import cupy as cp

        import eagle  # noqa: F401  (imports untimed)
        import eagle.plan  # noqa: F401
        import hawk  # noqa: F401
        import hawk.artifact  # noqa: F401
        import hawk.compile  # noqa: F401
        cp.cuda.Device().synchronize()  # context untimed
        arm = {"cupy": "cupy_masked", "hawk_host": "cpu_openmp"}.get(
            key, key.split(":")[-1])
        t0 = time.perf_counter()
        _probe_once(_probe_arm(arm, WARM_N, 1), tiny)
        seconds = time.perf_counter() - t0
    cc.probe_reply(reply_fd, {"seconds": seconds})


#: The cache directories a compile-time measurement points at a fresh (cold)
#: or populated (warm) directory.
_CACHE_ENV = {"HAWK_CACHE_DIR": "hawk", "CUPY_CACHE_DIR": "cupy",
              "TORCHINDUCTOR_CACHE_DIR": "inductor", "TRITON_CACHE_DIR": "triton",
              "CUDA_CACHE_PATH": "cuda", "PERF_CARD_JAX_CACHE": "jax",
              "WARP_CACHE_PATH": "warp"}


def _run_compile_pass(ns):
    rows = []
    for key, arms, what in _compile_keys(ns):
        cold, warm = cc.cold_warm(__file__, ["--compile-probe", key], _CACHE_ENV,
                                  reps=COMPILE_REPS, prefix="perf_card_cache_",
                                  what=f"compile probe {key}")
        rows.append({"key": key, "arms": arms, "what": what,
                     "cold_s": _median_iqr(cold), "warm_s": _median_iqr(warm)})
        print(f"compile {key:28s} cold {statistics.median(cold):.3g}s "
              f"warm {statistics.median(warm):.3g}s", flush=True)
    return rows


_COMPILE_METHOD = (
    "Its own pass, never part of any wall. Each measurement is a fresh process: "
    "the imports and the GPU context start-up run untimed, then the timer spans "
    "the first call to a ready kernel (trace + code generation + compile + module "
    "load, or the cache lookup). Cold: every cache empty -- hawk's compile cache "
    "(HAWK_CACHE_DIR), CuPy's kernel cache (CUPY_CACHE_DIR), the CUDA driver's "
    "PTX cache (CUDA_CACHE_PATH), PyTorch Inductor's and Triton's caches "
    "(TORCHINDUCTOR_CACHE_DIR, TRITON_CACHE_DIR) and JAX's persistent "
    "compilation cache (minimum compile time and entry size 0), each a fresh "
    "directory. Warm: a new process on the caches the first cold run populated. "
    "eagle arms: eagle.deploy of the arm's hawk kernel, which returns once the "
    "device build is done (C++/CUDA emission, nvcc to PTX, driver load; eagle's "
    "own small CuPy kernels compile beside it) while the host compiler builds the "
    "host side on another thread, and the arm's set-up on a "
    f"{WARM_N}-sample batch (eagle's graph built, one step run). CuPy: one "
    f"masked step on {WARM_N} samples (every elementwise and reduction kernel "
    "compiled on first use). Warp: the kernel module's code generation, "
    "compile and load on the first launch (64 samples), cached in "
    "WARP_CACHE_PATH. PyTorch torch.compile, where it runs: three steps at N "
    "(the first call traces and compiles, the second records the CUDA graph, "
    "the third replays it), per N (dynamic=False). JAX: lower + compile from abstract "
    "shapes, per N (XLA specialises on the shape). CPU OpenMP: the same "
    "eagle.deploy as the eagle graph arm and one host run, which waits for the "
    "host build. Medians of 3 cold "
    "and 3 warm processes.")

_MEMORY_METHOD = (
    "A separate pass after the timing pass. Each (arm, N, configuration) runs "
    "in a fresh process: imports, the arm's kernels built and a warm-up run on a "
    f"{WARM_N}-sample batch (every compile happens here; JAX also compiles for "
    "shape N from abstract shapes), every caching allocator trimmed (CuPy's "
    "pools freed, torch.cuda.empty_cache(), Warp's pool release threshold set "
    "to 0), then the baseline; then the batch's "
    "inputs generated, the arm set up for N (eagle builds its graph, PyTorch "
    "captures its CUDA graph, whose private memory pool counts) and one full run with upload and download. "
    "Device memory = the memory the CUDA driver accounts to the process (NVML "
    "per-process used memory) after the run minus at the baseline. The caching "
    "allocators of CuPy, PyTorch and JAX are left at their defaults and are not "
    "trimmed after the baseline; each keeps the memory it reserved until "
    "trimmed, so the reading after the run is the allocator's high-water "
    "reservation, the same measure for every arm. A disabled pool returns "
    "memory between operations, so its peak could only be caught by sampling, "
    "which misses short peaks; the held reservation needs no sampling. Caveats: "
    "the reading is what the process holds, so each allocator's rounding and "
    "growth policy count (JAX's allocator grows in regions that can exceed the "
    "request; PyTorch's rounds blocks up; a captured CUDA graph keeps a private "
    "pool; Warp's stream-ordered pool reserves in chunks of tens of MiB); the driver accounts in pages of about 2 MiB, so small-N rows read "
    "0 or one page. The per-library counters (CuPy pool, "
    "torch.cuda.max_memory_reserved/allocated, JAX memory_stats) are recorded "
    "in the JSON as a cross-check. Host memory = peak resident set size above "
    "the baseline (/proc/self/clear_refs reset at the baseline, then "
    "VmHWM). Minimum = the state the workload needs, N × "
    f"({STATE_WIDTH} state + {PER_SAMPLE_SCALARS} per-sample scalars: omega, "
    "zeta, stop step) × 8 bytes; the factor is device memory / minimum.")


# --------------------------------------------------------------------------- #
# Kernel-only times (Nsight Systems pass)
# --------------------------------------------------------------------------- #
#: The kernel pass traces each group in its own process: one process holding
#: CuPy, PyTorch and JAX at once under nsys loses its kernel records
#: (each library alone traces cleanly).
KERNEL_PASS_GROUPS = (EAGLE_GPU_ARMS + ("cupy_masked",),
                      ("torch_masked", "torch_graphed", "torch_compiled"),
                      ("jax_vmap",), ("warp_kernel",))


def _nsys_kernel_times(ns, work):
    """Run this script's kernel pass under nsys, one traced process per
    (configuration, N); return ({(dist, S, n, arm): summed kernel seconds or
    None}, {same key: {"kernel_launches", "memcpy", "api_calls"} or None}, a
    note naming the nsys version and any traced process that failed)."""
    nsys = cc.which("nsys")
    if nsys is None:
        return None, None, "nsys not found"
    out, out_counts, failed = {}, {}, []
    for dist, steps in CONFIGS:
        for n in ns:
            for gi, group in enumerate(KERNEL_PASS_GROUPS):
                group = tuple(a for a in group if a in _run_arms())
                if not group:
                    continue
                tag = f"{dist}-{steps}-{n}-g{gi}"
                got = counts = None
                for _attempt in range(2):
                    try:
                        got, counts = _nsys_one(nsys, work / f"kernel_pass_{tag}",
                                        ["--kernel-pass", "--kp-config", dist, str(steps),
                                         "--ns", str(n), "--kp-arms", *group])
                    except (subprocess.SubprocessError, sqlite3.Error, OSError):
                        continue
                    if all(got.get((dist, steps, n, a)) is not None for a in group):
                        break  # every arm of the group has a complete traced run
                if got is None:
                    failed.append(tag)
                    got, counts = {}, {}
                out.update({(dist, steps, n, a): got.get((dist, steps, n, a)) for a in group})
                out_counts.update({(dist, steps, n, a): counts.get((dist, steps, n, a))
                                   for a in group})
    note = cc.nsys_version(nsys) + ("" if not failed else
                      "; traced process failed twice for: " + ", ".join(failed))
    incomplete = sorted("-".join(map(str, k)) for k, v in out.items() if v is None)
    if incomplete:
        note += "; no complete traced run (kernel-only left empty) for: " + ", ".join(incomplete)
    return out, out_counts, note


def _nsys_one(nsys, rep, args):
    runs, modes = {}, {}
    for text, count, seconds, memcpy, api in cc.nsys_ranges(
            nsys, rep, __file__, args, "perfcard|"):
        _, dist, steps, n, arm, mode, _r = text.split("|")
        key = (dist, int(steps), int(n), arm)
        runs.setdefault(key, []).append((count, seconds, memcpy, api))
        modes[key] = mode or None
    out, out_counts = {}, {}
    for key, got in runs.items():
        mode = modes[key]
        expected = _expected_kernels(key[3], key[1], mode)
        # The compaction and automatic arms' counts depend on the data (how
        # many compactions ran, the steps each launch took), so only a lower
        # bound is known: a run below it lost records. PERSIST_ARM is always
        # the fast path (forced); AUTO_ARM/SIMULATE_ARM are exact the same
        # way when the runner's own mode this row (`mode`) was a fast path,
        # not the band -- no data dependence either way (see
        # _expected_kernels).
        exact = (key[3] in ("eager_loop", "cupy_masked", PERSIST_ARM)
                or (key[3] in (AUTO_ARM, SIMULATE_ARM) and mode in ("fused_one", "persist")))
        valid = [(seconds, memcpy, api) for count, seconds, memcpy, api in got
                 if (count == expected if exact else count >= expected)]
        out[key] = statistics.median(s for s, _, _ in valid) if valid else None
        if valid:
            memcpys = [m for _, m, _ in valid if m is not None]
            apis = [a for _, _, a in valid if a is not None]
            out_counts[key] = {
                "kernel_launches": expected,
                "memcpy": statistics.median(memcpys) if memcpys else None,
                "api_calls": statistics.median(apis) if apis else None}
        else:
            out_counts[key] = None
    return out, out_counts


#: A fast-path run (fused_one or persist) issues exactly two kernels: the
#: mask-count kernel that re-syncs `finished` to the mask every run
#: (eagle._until_done._mask_count) and the one step
#: launch itself -- no matter max_steps.
_FAST_PATH_KERNELS = 2


def _expected_kernels(arm, max_steps, mode=None):
    """Kernels one run launches: the step kernel per step for the eager loop;
    44 elementwise kernels per CuPy step plus 4 set-up ones (both exact). A
    traced run with fewer kernels lost trace records and is discarded. The
    eagle graph arms' counts are data-dependent (compactions, the steps each
    automatic launch takes), and the PyTorch and JAX arms' depend on the
    library's own kernels; their entries are lower bounds -- EXCEPT
    AUTO_ARM/SIMULATE_ARM when ``mode`` (the runner's own
    :attr:`eagle.RunReport.mode` this row) says a fast-path run actually
    took it: :data:`_FAST_PATH_KERNELS`, exact, same as PERSIST_ARM's own
    row (the band's data-dependent bound does not
    apply once eagle auto-routed past the band)."""
    if arm in (AUTO_ARM, SIMULATE_ARM) and mode in ("fused_one", "persist"):
        return _FAST_PATH_KERNELS
    return {# a lower bound: the plain decorator's automatic kernel at its most
            # steps per launch
            "eagle_graph": -(-max_steps // AUTO_K_MAX),
            # a lower bound: the step kernel of every step (the loop runs
            # whole blocks of COMPACT_EVERY, and compactions add more)
            "eagle_graph_compact": max_steps,
            "eagle_graph_reorder": max_steps,
            # lower bounds: the K-steps kernel once per launch
            "eagle_graph_steps": -(-max_steps // STEPS_PER_LAUNCH),
            "eagle_graph_steps_compact": -(-max_steps // STEPS_PER_LAUNCH),
            # a lower bound: the automatic kernel at its most steps per
            # launch -- the band's bound (mode != fast path above)
            "eagle_graph_auto": -(-max_steps // AUTO_K_MAX),
            # the persist entry forced explicitly, always the fast path
            "eagle_graph_persist": _FAST_PATH_KERNELS,
            "eagle_simulate": -(-max_steps // AUTO_K_MAX),
            "eager_loop": max_steps,
            "cupy_masked": 44 * max_steps + 4,
            "torch_masked": max_steps,
            "torch_graphed": max_steps,
            "torch_compiled": max_steps,
            "jax_vmap": max_steps,
            "warp_kernel": 1}[arm]


_device_facts, _software = cc.device_facts, cc.software


#: The marked code blocks (``# >>> code:<name>``) each arm's snippet is made of.
ARM_CODE = {
    "eagle_graph": ("hawk_step", "eagle_graph"),
    "eagle_graph_compact": ("hawk_kind", "hawk_active", "eagle_graph_compact"),
    "eagle_graph_reorder": ("hawk_kind", "hawk_active", "eagle_graph_reorder"),
    "eagle_graph_steps": ("hawk_steps", "eagle_graph_steps"),
    "eagle_graph_steps_compact": ("hawk_kind", "hawk_active", "eagle_graph_steps_compact"),
    "eagle_graph_auto": ("hawk_kind", "hawk_auto", "eagle_graph_auto"),
    "eagle_graph_persist": ("hawk_persist", "eagle_graph_persist"),
    "eagle_simulate": ("hawk_kind", "hawk_auto", "eagle_simulate"),
    "eager_loop": ("hawk_step", "eager_loop"),
    "cupy_masked": ("cupy_masked",),
    "torch_masked": ("torch_masked",),
    "torch_graphed": ("torch_step", "torch_graphed"),
    "torch_compiled": ("torch_step", "torch_compiled"),
    "jax_vmap": ("jax_vmap",),
    "warp_kernel": ("warp_kernel", "warp_launch"),
    "cpu_openmp": ("hawk_step", "cpu_openmp_deploy", "cpu_openmp"),
}


def _code_blocks(path=None):
    """The ``# >>> code:<name>`` ... ``# <<< code:<name>`` blocks of a script
    (this one by default): {name: {"first_line", "last_line", "lines"}}, for
    the landing page's per-arm snippets."""
    return cc.code_spans(path or __file__)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--reps", type=int, default=REPS,
                    help="timed runs per arm and cell (default 12; the mean "
                         "drops the highest and the lowest)")
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--quick", action="store_true",
                    help="smoke test: N up to 1e4, no file written "
                         "unless --out-dir is given")
    ap.add_argument("--no-nsys", action="store_true")
    ap.add_argument("--no-sweep", action="store_true",
                    help="skip the compaction cadence sweep")
    ap.add_argument("--sweep-ns", type=int, nargs="+",
                    default=list(COMPACTION_SWEEP_NS),
                    help="Ns for the compaction cadence sweep (spread config, "
                         f"max_steps={COMPACTION_SWEEP_CONFIG[1]})")
    ap.add_argument("--no-memory", action="store_true",
                    help="skip the memory pass (dry runs only)")
    ap.add_argument("--no-compile", action="store_true",
                    help="skip the compile-time pass (dry runs only)")
    ap.add_argument("--no-fp32", action="store_true",
                    help="skip the FP32 pass (dry runs only)")
    ap.add_argument("--no-fixed-overhead", action="store_true",
                    help="skip the per-run fixed-overhead cell (N=64, S=100)")
    ap.add_argument("--kernel-pass", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--kp-config", nargs=2, help=argparse.SUPPRESS)
    ap.add_argument("--kp-arms", nargs="+", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--memory-probe", nargs=5, help=argparse.SUPPRESS)
    ap.add_argument("--compile-probe", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.memory_probe:
        a, d, s, n, bus = args.memory_probe
        _memory_probe(a, d, int(s), int(n), bus)
        return 0
    if args.compile_probe:
        _compile_probe(args.compile_probe)
        return 0
    if args.reps < 5:
        ap.error("--reps must be >= 5")
    if args.quick:
        args.ns = [n for n in args.ns if n <= 10_000]
        args.no_sweep = True

    if args.kernel_pass and args.kp_arms is None:
        args.kp_arms = [a for a in _run_arms() if a != "cpu_openmp"]
    device, peak, slug, bus_id = _device_facts()
    gate_record = _gate_init(bus_id)
    if args.kernel_pass:
        dist, steps = args.kp_config[0], int(args.kp_config[1])
        for n in args.ns:
            _run_config(n, dist, steps, args.reps, bus_id, kernel_pass=tuple(args.kp_arms))
        return 0

    _check_kernel_bodies()
    cap_fixture = _cap_fixture_main()
    print(f"cap fixture passed on {len(cap_fixture['arms'])} arms", flush=True)
    work = pathlib.Path(tempfile.mkdtemp(prefix="perf_card_"))
    sources = _kernel_sources(work)
    dp_count = _dp_instruction_count(work, device)

    results, extras = [], []
    for dist, steps in CONFIGS:
        for n in args.ns:
            t0 = time.perf_counter()
            rows, extra = _run_config(n, dist, steps, args.reps, bus_id)
            results += rows
            extras.append({"distribution": dist, "max_steps": steps, "n": n, **extra})
            best = min(rows, key=lambda r: r["wall_s"]["mean"])
            print(f"{dist:8s} S={steps:5d} N={n:>8,}  fastest={best['arm']:12s} "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

    fixed_overhead_rows = []
    if not args.no_fixed_overhead:
        t0 = time.perf_counter()
        fixed_overhead_rows, _ = _run_config(64, "uniform", 100, args.reps, bus_id)
        print(f"fixed overhead N=64 S=100 ({time.perf_counter() - t0:.1f}s)", flush=True)

    fp32 = None
    if not args.no_fp32:
        fp32 = _run_fp32_pass([n for n in FP32_NS if not args.quick or n <= 10_000],
                              args.reps)

    sweep_rows = []
    if not args.no_sweep:
        sdist, ssteps = COMPACTION_SWEEP_CONFIG
        for n in args.sweep_ns:
            t0 = time.perf_counter()
            sweep_rows += _run_compaction_sweep(n, sdist, ssteps, args.reps,
                                                COMPACTION_SWEEP_KS)
            print(f"compaction sweep {sdist:8s} S={ssteps:5d} N={n:>8,}  "
                  f"({time.perf_counter() - t0:.1f}s)", flush=True)

    kernel_times, nsys_counts, nsys_note = (None, None, "skipped (--no-nsys)")
    if not args.no_nsys:
        kernel_times, nsys_counts, nsys_note = _nsys_kernel_times(args.ns, work)
    memory_rows = [] if args.no_memory else _run_memory_pass(args.ns, bus_id)
    compile_rows = [] if args.no_compile else _run_compile_pass(args.ns)
    for row in results:
        key = (row["distribution"], row["max_steps"], row["n"], row["arm"])
        if row["arm"] != "cpu_openmp" and kernel_times is not None:
            row["kernel_only_s"] = kernel_times.get(key)
            row["kernel_only_source"] = (
                "Nsight Systems: sum of kernel durations over one run, in a "
                "separate traced pass (--cuda-graph-trace=node); median of the "
                "complete traced runs; includes eagle's loop-control kernels "
                "for the graph arm and every library kernel the PyTorch, JAX "
                "and Warp arms launch; node-level graph tracing adds a per-node "
                "cost, so for a graph of many short kernels (PyTorch's captured "
                "block) the traced sum can exceed the untraced wall")
        # Item 5: kernel-launch/memcpy/graph-node counts, from the same nsys
        # pass above; "--" (None) with --no-nsys or no complete traced run.
        row["nsys_counts"] = (nsys_counts.get(key) if row["arm"] != "cpu_openmp"
                              and nsys_counts is not None else None)
        # Item 4: lane utilisation -- set already, per-row, from the
        # persist arm's own report (the persistent launch); every other
        # arm's row has none, so this only fills the gap.
        row.setdefault("lane_utilisation", None)
    for row in results:
        row["fraction_of_peak_fp64"] = (
            None if row["arm"] == "cpu_openmp" or peak["fp64_flops"] is None
            else row["useful_flops_per_s"] / peak["fp64_flops"])
        row["fraction_of_peak_dram"] = (
            None if row["arm"] == "cpu_openmp"
            else row["issued_bytes_per_s"] / peak["dram_bytes_per_s"])
        ko = row["kernel_only_s"]
        row["fraction_of_peak_fp64_kernel_only"] = (
            None if not ko or row["fraction_of_peak_fp64"] is None
            else FLOPS_PER_SAMPLE_STEP * row["sample_steps"] / ko / peak["fp64_flops"])
    for row in fixed_overhead_rows:
        row["lane_utilisation"] = None
        row["fraction_of_peak_fp64"] = (
            None if row["arm"] == "cpu_openmp" or peak["fp64_flops"] is None
            else row["useful_flops_per_s"] / peak["fp64_flops"])

    # Item 1: the SM clock the repetitions actually ran at (NVML where
    # available, nvidia-smi otherwise -- see ClockPoller) and the FP64 peak
    # at that clock.
    clocks = [e["gpu_state_during"].get("sm_clock_mhz_median") for e in extras]
    clock_sources = {e["gpu_state_during"].get("source") for e in extras
                     if e["gpu_state_during"].get("source")}
    clocks = [c for c in clocks if c is not None]
    if clocks and peak["fp64_flops"] is not None:
        observed_hz = statistics.median(clocks) * 1e6
        peak["observed_sm_clock_hz"] = observed_hz
        peak["fp64_flops_at_observed_sm_clock"] = (
            peak["fp64_flops"] * observed_hz / device["clock_rate_hz"])
        peak["observed_clock_source"] = (
            "median over configurations of the median per-repetition clock "
            "reading (" + "; ".join(sorted(clock_sources)) + ")")
        # Items 1+2: "% of issue peak at the measured clock" -- DP
        # instructions/s (dp_instructions_per_step x sample-steps/s) over the
        # device's DP issue-slot peak at the observed clock (fp64_flops is 2
        # FLOP per FMA-equivalent issue slot, so issue slots/s = flops/2);
        # only for the hawk-emitted eagle arms the SASS count was taken on.
        peak["issue_peak_instr_per_s_at_observed_clock"] = (
            peak["fp64_flops_at_observed_sm_clock"] / 2.0)
        for row in results + fixed_overhead_rows:
            f = row["fraction_of_peak_fp64"] if "fraction_of_peak_fp64" in row else None
            row.setdefault("fraction_of_peak_fp64_at_observed_clock", None)
            if f is not None:
                row["fraction_of_peak_fp64_at_observed_clock"] = (
                    row["useful_flops_per_s"] / peak["fp64_flops_at_observed_sm_clock"])
            # Only the arms that launch the single-step kernel: the fused
            # kernels hoist per-sample work out of the step loop, so their
            # per-step DP count is lower than the single-step SASS count.
            row["fraction_of_issue_peak_at_observed_clock"] = (
                None if dp_count is None or row["arm"] not in SINGLE_STEP_ARMS
                else dp_count["count"] * row["sample_steps_per_s"]
                     / peak["issue_peak_instr_per_s_at_observed_clock"])
    card = {
        "schema": "eagle-perf-card/2",
        "toolchain": cc.toolchain(),
        "device_slug": slug,
        "script": "benchmarks/perf_card/perf_card.py",
        "script_md5": cc.md5(__file__),
        "common_md5": cc.md5(cc.__file__),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "peak": peak,
        "software": _software(),
        "kernel_sources_md5": sources,
        "workload": {
            "problem": ("damped harmonic oscillator x'' + 2 zeta omega x' + "
                        "omega^2 x = 0, classical RK4, fixed step"),
            "dt": DT, "precision": "float64",
            "omega_range": [1.0, 10.0], "zeta_range": [0.01, 0.2],
            "x0_v0_range": [-1.0, 1.0], "seed": SEED,
            "stop_rule": ("a sample stops after its own step count nstop; "
                          "spread: nstop log-uniform in [S/100, S]; uniform: "
                          "nstop = S; sample 0 always runs S steps"),
            "flops_per_sample_step": FLOPS_PER_SAMPLE_STEP,
            "compaction_every_steps": COMPACT_EVERY,
            "reorder_theta": REORDER_THETA,
            "steps_per_launch": STEPS_PER_LAUNCH,
            "steps_compaction_every_steps": STEPS_COMPACT_EVERY,
            "dp_instructions_per_step": dp_count,
        },
        "configs": [{"distribution": d, "max_steps": s} for d, s in CONFIGS],
        "cap_fixture": cap_fixture,
        "ns": list(args.ns),
        "arms": {a: ARM_LABELS[a] for a in _run_arms()},
        "arms_not_run": _optional_arms()[1],
        "code": cc.arm_code(_code_blocks(), ARM_CODE, _run_arms()),
        "method": {
            "repetitions": args.reps,
            "statistic": _statistic_text(args.reps),
            "ramp_s": RAMP_S,
            "cool_down_gate": gate_record,
            "warm_up": "one run per arm, excluded; it is also the verification run",
            "interleaving": ("none: every arm is timed in its own block, in the arm "
                             "order; a block is the warm-up and verification runs, "
                             f"an untimed ramp of back-to-back runs for {RAMP_S:g} s "
                             f"(at least one run), then {args.reps} timed runs"),
            "wall": "the integration loop alone, from inputs on the device to "
                    "the device synchronised",
            "end_to_end": "host inputs uploaded, integration, outputs downloaded",
            "verification": (
                f"all arms reach identical per-sample step counts; x and v agree "
                f"with cpu_openmp within absolute {ATOL:g}"),
            "cpu_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
            "frameworks": (
                "PyTorch and JAX share the process (and the GPU) with CuPy and "
                "eagle; JAX with XLA_PYTHON_CLIENT_PREALLOCATE="
                f"{os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')} and "
                "jax_enable_x64; the JAX executable compiled per cell before the "
                "warm-up (per_config.jax_compile_s); torch_graphed's CUDA graph "
                "captured per cell before the warm-up "
                "(per_config.torch_graph_capture_s); torch_compiled, where it "
                "runs, recompiled per cell (torch._dynamo.reset(), dynamic=False); "
                "Warp's kernel module compiled once per process, on the warm-up"),
            "kernel_only": nsys_note,
            "nsys_counts": ("kernel-launch/memcpy/runtime-API-call counts per run, "
                            "from the same traced pass as kernel_only; \"--\" (None) "
                            "with --no-nsys or when no complete traced run exists"
                            if not args.no_nsys else "skipped (--no-nsys)"),
            "lane_utilisation": ("active lanes / warp-iterations, in (0, 1]; "
                                 "filled for the persist arm (the persistent "
                                 "launch), \"--\" for every other "
                                 "arm (none counts it)"),
            "host_clock_source": ("cpu_openmp's row claims no host FP64 roofline on "
                                  "this card (see the CPU card, cpu_card.py, for the "
                                  "host peak and its clock source)"),
            "counting_rule": _COUNTING_RULE,
            "bytes_check": ("DRAM byte counters are not readable on compute "
                            "capability 6.1 (no Nsight Compute support); bytes "
                            "are analytic only on this device"),
            "compaction_sweep": (
                "wall time and sample-steps/s only (no kernel-only/nsys pass, "
                "no FLOP or byte accounting); each (K, variant) pair is "
                "verified against cpu_openmp the same way as the main arms "
                "before its repetitions are timed, and each of the two "
                "variants at a given K is timed in its own block the same way "
                "as the main table's arms"),
        },
        "results": results,
        "per_config": extras,
        "torch_graph_block": GRAPH_BLOCK,
        "memory": {"method": _MEMORY_METHOD,
                   "minimum_bytes_per_sample": (STATE_WIDTH + PER_SAMPLE_SCALARS) * 8,
                   "rows": memory_rows},
        "compile_time": {"method": _COMPILE_METHOD, "rows": compile_rows,
                         "none": list(NO_COMPILE_ARMS)},
        "fixed_overhead": {
            "n": 64, "max_steps": 100, "distribution": "uniform",
            "method": ("item 3: wall at a cell small enough that per-run overhead "
                       "dominates (N=64, S=100), each arm's own kernel-only vs wall "
                       "where the main nsys pass covers this cell's N (it does not "
                       "by default, so kernel_only_s is usually null here)"),
            "rows": fixed_overhead_rows,
        },
    }
    if sweep_rows:
        card["compaction_sweep"] = {
            "isolates": (
                "the active-set map alone. For a given K, `matched_cadence` "
                "(the plain per-sample step, no map) and `compact` "
                "(eagle_graph_compact's active-set step) check the loop's "
                "stop guard at the SAME cadence, once every K steps; the "
                "only structural difference between the two rows at a given "
                "K is whether the step kernel reads the active-set map. The "
                "main table's eagle_graph row checks its guard every step, "
                "so its difference from eagle_graph_compact there mixes the "
                "cadence change with the map -- this sweep does not."
            ),
            "distribution": COMPACTION_SWEEP_CONFIG[0],
            "max_steps": COMPACTION_SWEEP_CONFIG[1],
            "ks": list(COMPACTION_SWEEP_KS),
            "ns": list(args.sweep_ns),
            "rows": sweep_rows,
        }
    if fp32:
        card["fp32"] = fp32
    card["fit"] = _fit_notes(card)
    if args.quick and args.out_dir == HERE:
        print(render_markdown(card))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"card_{slug}.json"
    json_path.write_text(json.dumps(card, indent=1, sort_keys=False) + "\n")
    (args.out_dir / f"card_{slug}.md").write_text(render_markdown(card))
    print(f"wrote {json_path}")
    shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    # JAX shares the GPU with CuPy, PyTorch and eagle in one process: it must
    # not preallocate most of the device memory at start-up
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    _threads = int(os.environ.get("OMP_NUM_THREADS") or CPU_THREADS)
    os.environ["OMP_NUM_THREADS"] = str(min(_threads, CPU_THREADS))
    sys.exit(main())
