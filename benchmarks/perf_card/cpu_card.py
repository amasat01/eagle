# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The adaptive-step card on the CPU: the RKF7(8) Kepler workload, the common CPU ways.

The workload, the scheme, the truth and the agreement checks are those of
``perf_card.py`` (imported from it). The arms:

``eagle_cpu8`` / ``eagle_cpu1``
    The GPU card's hawk attempt kernel, deployed by ``eagle.deploy`` (NumPy
    planes select the host build) and run through eagle's OpenMP host team
    (8 and 1 threads); the kernel finishes its own samples.
``numba_prange``
    Numba ``@njit(parallel=True)``: ``prange`` over samples, each sample runs
    its whole adaptive loop to the final time (8 threads).
``jax_vmap``
    The GPU card's ``jit(vmap(lax.while_loop))`` on the CPU backend.
``torch_masked``
    The GPU card's masked whole-batch attempt on CPU tensors
    (``torch.set_num_threads(8)``).
``numpy_masked``
    The same masked attempt in NumPy (one thread).

The masked arms advance every sample every attempt until the slowest one
finishes; they stop at the per-run budget.

Every arm runs in its own persistent worker process (its thread pool pinned
by that process's environment); the driver process interleaves the repetitions
across the workers (the arm order rotates every repetition), checks every arm
against the analytic truth and against ``eagle_cpu8`` before timing counts,
and reports medians with interquartile ranges. Peak host memory of the
warm-up run above the worker's post-compile baseline, and each worker's
compile time on empty caches, are recorded per (arm, N).

Reproduce (from the eagle repository root)::

    python benchmarks/perf_card/cpu_card.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):  # perf_card.py; benchmarks/: the shared helpers
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _card_common as cc  # noqa: E402
import perf_card as rc  # noqa: E402

SCRIPT_REL = "benchmarks/perf_card/cpu_card.py"
THREADS = 8
ARMS = ("eagle_cpu8", "eagle_cpu1", "numba_prange", "jax_vmap", "torch_masked",
        "numpy_masked")
REFERENCE_ARM = "eagle_cpu8"
ARM_THREADS = {"eagle_cpu8": THREADS, "eagle_cpu1": 1, "numba_prange": THREADS,
               "jax_vmap": THREADS, "torch_masked": THREADS, "numpy_masked": 1}
ARM_LABELS = {
    "eagle_cpu8": "eagle CPU, 8 threads", "eagle_cpu1": "eagle CPU, 1 thread",
    "numba_prange": "Numba prange", "jax_vmap": "JAX jit(vmap(while_loop))",
    "torch_masked": "PyTorch masked", "numpy_masked": "NumPy masked",
}
ARM_CODE = {"eagle_cpu8": ("hawk_kernel", "eagle_cpu_deploy", "eagle_cpu"),
            "eagle_cpu1": ("hawk_kernel", "eagle_cpu_deploy", "eagle_cpu"),
            "numba_prange": ("numba_prange",), "jax_vmap": ("jax_vmap",),
            "torch_masked": ("torch_masked",), "numpy_masked": ("numpy_masked",)}
ARM_LIBRARY = {"numba_prange": "numba", "jax_vmap": "jax", "torch_masked": "torch"}
MASKED_ARMS = ("torch_masked", "numpy_masked")
#: Arms with no compile step (PyTorch's and NumPy's kernels ship prebuilt).
NO_COMPILE_ARMS = MASKED_ARMS
NS = (1_000_000,)
REPS = rc.ROUNDS_MAX
COMMITTED_ARMS = ("eagle_cpu8", "eagle_cpu1", "numba_prange", "jax_vmap", "numpy_masked")
WARM_N = 64
#: The fixed-overhead cell: the eagle arms alone at this N (launch and join cost).
OVERHEAD_N = 64


# >>> code:numpy_masked
def numpy_attempt(s, t, h, n_acc, n_rej):
    """One masked RKF7(8) attempt over the whole batch (NumPy arrays)."""
    import numpy as np

    running = t < rc.T_FINAL
    last = h >= rc.T_FINAL - t
    step = np.where(last, rc.T_FINAL - t, h)
    s8, err = rc.rkf78_stages(s, step, np.sqrt)
    with np.errstate(divide="ignore"):
        ratio = rc.error_ratio(s, s8, err, np.abs, np.maximum)
        accept = ratio < 1.0
        take = running & accept
        expo = np.where(accept, rc.EXP_ACCEPT, rc.EXP_REJECT)
        scale = np.clip(rc.SAFETY * ratio ** expo, rc.MIN_SCALE, rc.MAX_SCALE)
    s = [np.where(take, s8[d], s[d]) for d in range(6)]
    t = np.where(take, np.where(last, rc.T_FINAL, t + step), t)
    h = np.where(running, step * scale, h)
    return s, t, h, n_acc + take, n_rej + (running & ~accept), running


def numpy_run(s, t, h, n_acc, n_rej):
    for _ in range(rc.MAX_ATTEMPTS):
        s, t, h, n_acc, n_rej, running = numpy_attempt(s, t, h, n_acc, n_rej)
        if not running.any():
            break
    return s, t, h, n_acc, n_rej
# <<< code:numpy_masked


def _tableau_arrays():
    import numpy as np

    a = np.zeros((rc.N_STAGES, rc.N_STAGES))
    for i, row in enumerate(rc.RK_A):
        a[i + 1, :len(row)] = row
    return a, np.asarray(rc.RK_B, dtype=np.float64), np.asarray(rc.RK_BE, dtype=np.float64)


# >>> code:numba_prange
def make_numba():
    import numba
    import numpy as np
    from numba import njit, prange

    numba.set_num_threads(THREADS)
    A, B, BE = _tableau_arrays()
    tf, rtol, atol, h0 = rc.T_FINAL, rc.RTOL, rc.ATOL, rc.H0
    safety, lo, hi = rc.SAFETY, rc.MIN_SCALE, rc.MAX_SCALE
    e_acc, e_rej, max_att = rc.EXP_ACCEPT, rc.EXP_REJECT, rc.MAX_ATTEMPTS

    @njit(inline="always")
    def rhs(s, out):
        r2 = s[0] * s[0] + s[1] * s[1] + s[2] * s[2]
        ir3 = 1.0 / (r2 * np.sqrt(r2))
        out[0], out[1], out[2] = s[3], s[4], s[5]
        out[3], out[4], out[5] = -s[0] * ir3, -s[1] * ir3, -s[2] * ir3

    @njit(parallel=True)
    def integrate(state, t_out, h_out, acc_out, rej_out):
        for i in prange(state.shape[0]):
            s = state[i].copy()
            k = np.empty((13, 6))
            si = np.empty(6)
            s8 = np.empty(6)
            t, h, n_acc, n_rej = 0.0, h0, 0, 0
            while t < tf and n_acc + n_rej < max_att:
                last = h >= tf - t
                step = tf - t if last else h
                rhs(s, k[0])
                for st in range(1, 13):
                    for d in range(6):
                        acc = 0.0
                        for j in range(st):
                            if A[st, j] != 0.0:
                                acc += A[st, j] * k[j, d]
                        si[d] = s[d] + step * acc
                    rhs(si, k[st])
                ratio = 0.0
                for d in range(6):
                    acc = 0.0
                    err = 0.0
                    for st in range(13):
                        acc += B[st] * k[st, d]
                        err += BE[st] * k[st, d]
                    s8[d] = s[d] + step * acc
                    sc = atol + rtol * max(abs(s8[d]), abs(s[d]))
                    ratio = max(ratio, abs(step * err) / sc)
                accept = ratio < 1.0
                scale = min(hi, max(lo, safety * ratio ** (e_acc if accept else e_rej)))
                if accept:
                    s[:] = s8
                    t = tf if last else t + step
                    n_acc += 1
                else:
                    n_rej += 1
                h = step * scale
            state[i] = s
            t_out[i], h_out[i] = t, h
            acc_out[i], rej_out[i] = n_acc, n_rej

    return integrate
# <<< code:numba_prange


# --------------------------------------------------------------------------- #
# Arms inside a worker
# --------------------------------------------------------------------------- #
class _Worker:
    def __init__(self, arm):
        self.arm = arm
        self.compile_s = 0.0

    def setup(self, n, distribution="spread"):
        import numpy as np

        self.n = n
        self.inp = rc._inputs(n, distribution)
        arm = self.arm
        t0 = time.perf_counter()
        if arm.startswith("eagle_cpu"):
            self.cpu = rc._make_arms(n, ("eagle_cpu",))["eagle_cpu"]["state"]
        elif arm == "numba_prange":
            if not hasattr(self, "nb"):
                self.nb = make_numba()
                st = np.stack([rc._inputs(WARM_N)[k] for k in rc.STATE_NAMES], axis=1)
                z = np.empty(WARM_N)
                self.nb(st.copy(), z, z.copy(), z.copy(), z.copy())  # compile
        elif arm == "jax_vmap":
            import jax

            jax.config.update("jax_enable_x64", True)
            self.jfn = rc.jax_integrate_fn().lower(
                jax.ShapeDtypeStruct((n, 6), jax.numpy.float64)).compile()
        elif arm == "torch_masked":
            import torch

            torch.set_num_threads(THREADS)
        self.compile_s += time.perf_counter() - t0

    def run(self):
        """One timed run from host inputs; returns (seconds, outputs)."""
        import numpy as np

        arm, inp, n = self.arm, self.inp, self.n
        names = rc.PLANE_NAMES
        if arm.startswith("eagle_cpu"):
            self.cpu.upload(inp)
            t0 = time.perf_counter()
            self.cpu.run()
            sec = time.perf_counter() - t0
            return sec, self.cpu.download()
        if arm == "numba_prange":
            st = np.stack([inp[k] for k in rc.STATE_NAMES], axis=1)
            t, h, a, r = (np.empty(n) for _ in range(4))
            t0 = time.perf_counter()
            self.nb(st, t, h, a, r)
            sec = time.perf_counter() - t0
            out = {k: st[:, i].copy() for i, k in enumerate(rc.STATE_NAMES)}
            out.update(t=t, h=h, n_acc=a, n_rej=r)
            return sec, out
        if arm == "jax_vmap":
            import jax

            s0 = jax.device_put(np.stack([inp[k] for k in rc.STATE_NAMES], axis=1))
            s0.block_until_ready()
            t0 = time.perf_counter()
            res = self.jfn(s0)
            res[1].block_until_ready()
            sec = time.perf_counter() - t0
            s, t, h, a, r = (np.asarray(v) for v in res)
            out = {k: s[:, i].copy() for i, k in enumerate(rc.STATE_NAMES)}
            out.update(t=t, h=h, n_acc=a.astype(float), n_rej=r.astype(float))
            return sec, out
        if arm == "torch_masked":
            import torch

            st = rc._ArrayState.__new__(rc._ArrayState)
            st.planes = {k: torch.from_numpy(inp[k].copy()) for k in rc.STATE_NAMES}
            st.planes.update(t=torch.zeros(n, dtype=torch.float64),
                             h=torch.full((n,), rc.H0, dtype=torch.float64),
                             n_acc=torch.zeros(n, dtype=torch.float64),
                             n_rej=torch.zeros(n, dtype=torch.float64))
            t0 = time.perf_counter()
            s, t, h, a, r = rc.torch_run(st)
            sec = time.perf_counter() - t0
            return sec, {k: v.numpy().copy() for k, v in zip(names, (*s, t, h, a, r))}
        if arm == "numpy_masked":
            s = [inp[k].copy() for k in rc.STATE_NAMES]
            t, h = np.zeros(n), np.full(n, rc.H0)
            a, r = np.zeros(n), np.zeros(n)
            t0 = time.perf_counter()
            s, t, h, a, r = numpy_run(s, t, h, a, r)
            sec = time.perf_counter() - t0
            return sec, dict(zip(names, (*s, t, h, a.astype(float), r.astype(float))))
        raise ValueError(arm)


def _worker_main(arm):
    """Persistent worker: reads one JSON command per line on stdin."""
    import numpy as np

    reply = os.fdopen(cc.probe_reply_fd(), "w")
    w = _Worker(arm)
    for line in sys.stdin:
        cmd = json.loads(line)
        if cmd["op"] == "setup":
            base = rc._set_cell(cmd.get("distribution", "spread"))
            w.compile_s = 0.0
            w.setup(WARM_N)
            w.run()  # warm-up at WARM_N: every compile happens here
            w.setup(cmd["n"], base)
            rss = cc.status_kib("VmRSS")
            cc.reset_host_peak()
            sec, out = w.run()  # the verification run
            peak = max(0, cc.status_kib("VmHWM") - rss) * 1024
            np.savez(cmd["out"], **out)
            live = max(0, cc.status_kib("VmRSS") - rss) * 1024
            msg = {"warm_s": sec, "host_peak_bytes": peak, "host_live_bytes": live, "compile_s": w.compile_s}
        elif cmd["op"] == "run":
            sec, _ = w.run()
            msg = {"wall_s": sec}
        else:
            break
        reply.write(json.dumps(msg) + "\n")
        reply.flush()


def _env(arm, cache):
    env = dict(os.environ)
    th = str(ARM_THREADS[arm])
    env.update(OMP_NUM_THREADS=th, MKL_NUM_THREADS=th, OPENBLAS_NUM_THREADS=th,
               NUMBA_NUM_THREADS=str(THREADS), HAWK_CACHE_DIR=str(cache / "hawk"),
               NUMBA_CACHE_DIR=str(cache / "numba"), JAX_PLATFORMS="cpu",
               XLA_FLAGS=("--xla_cpu_multi_thread_eigen=false "
                          "intra_op_parallelism_threads=1" if th == "1" else ""))
    return env


def _proc(arm, cache):
    return cc.LineWorker(__file__, ["--worker", arm], _env(arm, cache), arm)


#: Register-only FMA chains, one OpenMP thread per core and a vector of lanes per chain: the host's FP64 FMA rate.
_CPU_FMA_SRC = r'''
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#define W 8
#define CH 8
int main(int argc, char** argv) {
  long iters = atol(argv[1]);
  int nt = atoi(argv[2]);
  double best = 1e30, sink = 0;
  for (int rep = 0; rep < 3; ++rep) {
    double t0 = omp_get_wtime();
    #pragma omp parallel num_threads(nt) reduction(+ : sink)
    {
      double a[CH][W];
      for (int c = 0; c < CH; ++c) for (int w = 0; w < W; ++w) a[c][w] = 1.0 + 0.1 * c + 0.01 * w;
      const double b = 1.0000001, d = 1e-9;
      for (long i = 0; i < iters; ++i)
        for (int c = 0; c < CH; ++c) {
          #pragma omp simd
          for (int w = 0; w < W; ++w) a[c][w] = a[c][w] * b + d;
        }
      double s = 0;
      for (int c = 0; c < CH; ++c) for (int w = 0; w < W; ++w) s += a[c][w];
      sink += s;
    }
    double t = omp_get_wtime() - t0;
    if (t < best) best = t;
  }
  std::fprintf(stderr, "%g\n", sink);
  std::printf("%.6e\n", (double)nt * CH * W * (double)iters / best);
  return 0;
}
'''
CPU_FMA_ITERS = 2_000_000


def _cpu_cores():
    """Physical cores this process may run on (logical CPUs over the SMT width)."""
    logical = len(os.sched_getaffinity(0))
    try:
        width = int(next(ln.split(":")[1] for ln in subprocess.run(["lscpu"], capture_output=True, text=True,
                                                                    timeout=30).stdout.splitlines()
                         if ln.startswith("Thread(s) per core")))
    except (OSError, subprocess.SubprocessError, StopIteration, ValueError):
        width = 1
    return max(1, logical // max(1, width))


def _cpu_fma_probe(work):
    """FP64 FMA lane-operations per second of the host: ``gcc -O3 -march=native -fopenmp`` register-only chains on
    one thread per core, W = 8 lanes per chain. None when no compiler is found."""
    compiler = os.environ.get("CXX") or shutil.which("g++")
    if not compiler:
        return None
    src, exe = work / "cpu_fma_probe.cpp", work / "cpu_fma_probe"
    src.write_text(_CPU_FMA_SRC)
    built = subprocess.run([compiler, "-O3", "-march=native", "-ffp-contract=fast", "-fopenmp", str(src), "-o", str(exe)],
                           capture_output=True, text=True, timeout=300)
    if built.returncode:
        return None
    done = subprocess.run([str(exe), str(CPU_FMA_ITERS), str(_cpu_cores())], capture_output=True, text=True, timeout=300)
    return float(done.stdout.split()[0]) if done.returncode == 0 else None


def _time_arm(a, n, distribution, work):
    """One arm in its own worker process: set up, warm up (the verification run), then rounds 3..9 until
    IQR/median <= 2 %. A failure is a status: ``out_of_memory``, ``not_run: budget`` or ``failed: <last error line>``."""
    import numpy as np

    p = _proc(a, work / f"cache_{a}_{n}")
    path = work / f"out_{a}_{n}.npz"
    reply = {"arm": a, "n": n, "distribution": distribution}
    walls, setup = [], None
    try:
        if rc._fault(a) == "oom":
            raise MemoryError("out of memory (forced by EAGLE_CARD_FAULT)")
        try:  # setup = build + the warm-up run: the warm-up is stopped at the per-run budget, as on the GPU card
            setup = p.call({"op": "setup", "n": n, "out": str(path), "distribution": distribution},
                           timeout=cc.BUILD_S + rc.ARM_RUN_BUDGET_S)
        except TimeoutError:
            return {**reply, "status": "not_run: budget", "warm_s": rc.ARM_RUN_BUDGET_S, "stopped": True}
        if rc._fault(a) == "slow":
            setup["warm_s"] = rc.ARM_RUN_BUDGET_S + 1.0
        if setup["warm_s"] > rc.ARM_RUN_BUDGET_S:
            return {**reply, "status": "not_run: budget", "warm_s": setup["warm_s"]}
        with np.load(path) as z:
            reply["out"] = {k: z[k] for k in z.files}
        walls, rule, started = [], "rounds_max", time.perf_counter()
        while len(walls) < REPS:
            walls.append(p.call({"op": "run"}, timeout=2 * rc.ARM_RUN_BUDGET_S)["wall_s"])
            if len(walls) >= min(rc.ROUNDS_MIN, REPS):
                q1, med, q3 = np.percentile(walls, [25, 50, 75])
                if q3 - q1 <= rc.IQR_STOP * med:
                    rule = "iqr<=2%"
                    break
            if time.perf_counter() - started > rc.CELL_BUDGET_S:
                rule = "cell budget 300 s"
                break
        return {**reply, "status": "ok", "walls": walls, "stop_rule": rule, "setup": setup}
    except TimeoutError as err:
        return {**reply, "status": f"failed: {err}"} if not walls else {
            **reply, "status": "ok", "walls": walls, "stop_rule": "a round over the budget was stopped", "setup": setup}
    except (RuntimeError, OSError, MemoryError, ValueError, KeyError) as err:
        text = str(err)
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return {**reply, "status": "out_of_memory" if any(k in text for k in rc._OOM_TEXT)
                else "failed: " + (lines[-1][:300] if lines else type(err).__name__)}
    finally:
        try:
            p.close({"op": "quit"})
        except Exception:  # noqa: BLE001  (a dead worker has nothing to close)
            pass


_PARTS = None       # --parts: every finished (arm, cell) is saved here and a re-run reuses it
_ASSEMBLE = False   # --assemble: run nothing, write the card from the saved parts


def _cached_arm(a, n, distribution, work):
    """``_time_arm`` with the saved parts and the session deadline: a finished arm is reused, an arm the deadline (or an
    ``--assemble``) reaches is ``not_run: session time limit``."""
    import numpy as np

    key = f"timing_{a}_{n}_{distribution}"
    saved = cc.part_load(_PARTS, key, keep_failed=_ASSEMBLE)
    if saved is not None:
        if (_PARTS / f"{key}.npz").exists():
            with np.load(_PARTS / f"{key}.npz") as z:
                saved["out"] = {k: z[k] for k in z.files}
        return saved
    if _ASSEMBLE or cc.deadline_passed():
        return {"arm": a, "n": n, "distribution": distribution, "status": cc.SESSION_LIMIT}
    r = _time_arm(a, n, distribution, work)
    if _PARTS is not None:
        if r.get("out"):
            _PARTS.mkdir(parents=True, exist_ok=True)
            np.savez(_PARTS / f"{key}.npz", **r["out"])
        cc.part_save(_PARTS, key, {k: v for k, v in r.items() if k != "out"})
    return r


def _run_cell(n, names, work, distribution, state):
    """All arms at one cell, one after another. ``state`` carries the arms already over budget (at a smaller N)."""
    base = rc._set_cell(distribution)
    replies = []
    for a in names:
        if a in state["budget"] and n >= state["budget"][a]:
            r = {"arm": a, "n": n, "distribution": distribution, "status": "not_run: budget"}
        else:
            r = _cached_arm(a, n, distribution, work)
            if r["status"] == "not_run: budget":
                state["budget"][a] = n
        replies.append(r)
        print(f"{distribution} N={n:>9,} {a:14s} {r['status']}", flush=True)
    ok = {r["arm"]: r["out"] for r in replies if r["status"] == "ok"}
    agreement, ref, steps = ({}, "none", {})
    if ok:
        ref = REFERENCE_ARM if REFERENCE_ARM in ok else next(iter(ok))
        agreement, steps = rc._verify(ok, rc._inputs(n, base), ref)
    rows = []
    for r in replies:
        row = {"n": n, "distribution": distribution, "arm": r["arm"], "status": r["status"],
               "threads": ARM_THREADS[r["arm"]]}
        if r["status"] != "ok":
            row["rule"] = (f"> {rc.ARM_RUN_BUDGET_S:g} s per run" if r["status"] == "not_run: budget" else r["status"])
        else:
            att = agreement[r["arm"]]["attempts_total"]
            wall = cc.median_iqr(r["walls"])
            row.update(wall_s=wall, rounds=len(r["walls"]), rule=r["stop_rule"], attempts=att,
                       attempt_samples_per_s=att / wall["median"],
                       useful_flops_per_s=rc.FLOPS_PER_ATTEMPT * att / wall["median"],
                       host_peak_bytes=r["setup"]["host_peak_bytes"], host_live_bytes=r["setup"]["host_live_bytes"],
                       compile_s_cold=(None if r["arm"] in NO_COMPILE_ARMS else r["setup"]["compile_s"]),
                       slot_fill=None)
        rows.append(row)
    control = next((r for r in rows if r["arm"] == "numpy_masked" and r["status"] == "ok"), None)
    for r in rows:  # memory net of the NumPy control: the harness bytes every arm shares are not the arm's
        if r["status"] == "ok":
            r["host_peak_net_of_control_bytes"] = (
                None if control is None or r is control else max(0, r["host_peak_bytes"] - control["host_peak_bytes"]))
    return rows, {"n": n, "distribution": distribution, "reference_arm": ref, "agreement": agreement, "steps": steps}


def code_line_counts():
    blocks = rc.code_blocks()
    blocks.update(rc.code_blocks(__file__))
    per_block = {k: cc.count_code_lines(v) for k, v in blocks.items()}
    return {"blocks": {k: per_block[k] for bs in ARM_CODE.values() for k in bs},
            "arms": {a: sum(per_block[b] for b in bs) for a, bs in ARM_CODE.items()},
            "arm_blocks": ARM_CODE, "shared_scheme_lines": per_block["shared_scheme"]}


def render_markdown(card):
    lines = [f"<!-- Generated by {SCRIPT_REL} from cpu_card_{card['cpu_slug']}.json; "
             "do not edit. -->", "",
             f"**CPU:** {card['cpu']}, {card['logical_cpus']} logical CPUs. "
             f"Medians of {min(card['method']['rounds']['min'], card['method']['rounds']['max'])} to "
             f"{card['method']['rounds']['max']} rounds per arm, each arm in its own process (rounds stop at "
             f"IQR/median <= 2 %); IQR in parentheses. Workload as the GPU card ({rc.FLOPS_PER_ATTEMPT} FLOP per "
             f"attempt).", "",
             "| N | arm | threads | wall (IQR) | rounds | attempts·samples/s | useful FLOP/s | eff_sol "
             "| host memory | host memory net of NumPy | lines of code |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(card["results"], key=lambda r: (r["n"], r.get("distribution", ""),
                                                   ARMS.index(r["arm"]))):
        head = f"| {r['n']:,} ({r.get('distribution', 'spread')}) | {ARM_LABELS[r['arm']]} | {r['threads']} "
        if r["status"] != "ok":
            lines.append(head + f"| {r['status'].replace('_', ' ')} ({r['rule']}) | – | – | – | – | – | – | – |")
            continue
        net = r.get("host_peak_net_of_control_bytes")
        lines.append(
            head + f"| {cc.fmt_time_si(r['wall_s']['median'])} ({cc.fmt_time_si(r['wall_s']['iqr'])}) "
            f"| {r['rounds']} | {cc.fmt_rate_si(r['attempt_samples_per_s'])} "
            f"| {cc.fmt_rate_si(r['useful_flops_per_s'])} "
            f"| {r.get('eff_sol_note') or ('–' if r.get('eff_sol') is None else format(r['eff_sol'], '.3f'))} "
            f"| {cc.fmt_mib(r['host_peak_bytes'])} | {'–' if net is None else cc.fmt_mib(net)} "
            f"| {card['code_lines']['arms'][r['arm']]} |")
    lines += ["", "| N | arm | max position error | max error / bound "
              "| samples with different step counts |", "|---:|---|---:|---:|---:|"]
    for e in card["per_n"]:
        for a, g in e["agreement"].items():
            lines.append(f"| {e['n']:,} | {ARM_LABELS[a]} "
                         f"| {g['max_position_error_vs_truth']:.2e} "
                         f"| {g['max_error_over_bound']:.2e} "
                         f"| {g['samples_with_step_counts_differing_from_reference']:,} |")
    return "\n".join(lines) + "\n"


def _cpu_slug(model):
    """The card's file stem: the CPU model, or ``aarch64`` where /proc/cpuinfo names none (Arm)."""
    if platform.machine() in ("aarch64", "arm64") and (not model or model.lower() in ("aarch64", "arm64", "arm")):
        return "aarch64"
    return re.sub(r"[^a-z0-9]+", "-", (model or "unknown-cpu").lower()).strip("-")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--reps", type=int, default=REPS, help="most rounds per (arm, cell); they stop at IQR/median <= 2 %%")
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--arms", nargs="+", default=list(COMMITTED_ARMS))
    ap.add_argument("--distributions", nargs="+", choices=rc.DISTRIBUTIONS,
                    default=list(rc.DISTRIBUTIONS))
    ap.add_argument("--smoke", action="store_true",
                    help="smallest N, one repetition, card written to a temporary directory")
    ap.add_argument("--assemble", action="store_true",
                    help="run nothing: write the card from the saved parts (--parts), a missing arm reads not_run: session time limit")
    ap.add_argument("--parts", type=pathlib.Path,
                    help="save each finished (arm, cell) here and reuse what is there: a preemption costs one arm")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        _worker_main(args.worker)
        return 0
    global _PARTS, _ASSEMBLE
    _PARTS, _ASSEMBLE = args.parts, args.assemble
    arms_not_run = {a: f"{ARM_LIBRARY[a]} not installed in this profile"
                    for a in ARMS if a in args.arms and a in ARM_LIBRARY
                    and not rc._lib_installed(ARM_LIBRARY[a])}
    for a, why in arms_not_run.items():
        print(f"arm {a} not run: {why}", flush=True)
    names = tuple(a for a in ARMS if a in args.arms and a not in arms_not_run)
    if args.smoke:
        args.ns, args.reps, args.distributions = [1_000], 1, ["spread"]
        if args.out_dir == HERE:
            args.out_dir = pathlib.Path(tempfile.mkdtemp(prefix="cpu_card_smoke_"))
    t_start = time.perf_counter()
    work = pathlib.Path(tempfile.mkdtemp(prefix="cpu_card_"))
    results, per_n = [], []
    try:
        # the headline cell (N = 1e6 spread) first, the method cell last: a session deadline drops the tail
        cells = sorted(((d, n, names) for d in args.distributions for n in args.ns), key=lambda c: (c[0], c[1]) != ("spread", 1_000_000))
        if not args.smoke:
            cells += [("spread", OVERHEAD_N, tuple(a for a in names if a.startswith("eagle")))]
        state = {"budget": {}}
        for dist, n, arms in cells:
            t0 = time.perf_counter()
            rows, extra = _run_cell(n, arms, work, dist, state)
            results += rows
            per_n.append(extra)
            print(f"{dist} N={n:>8,} ({time.perf_counter() - t0:.1f}s)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    try:
        fma = _cpu_fma_probe(pathlib.Path(tempfile.mkdtemp(prefix="cpu_fma_")))
    except (OSError, subprocess.SubprocessError, ValueError):
        fma = None
    ops = json.loads((HERE.parent / "sol_reference.json").read_text()).get("ops_per_attempt")
    for r in results:
        if r["status"] == "ok":
            r.update(cc.derive_eff_sol(r["attempt_samples_per_s"], fma, ops))
    cpu = cc.cpu_model()
    slug = _cpu_slug(cpu)
    card = {"schema": "eagle-rk78-cpu-card/2", "toolchain": cc.toolchain(), "cpu": cpu, "cpu_slug": slug,
            "logical_cpus": os.cpu_count(), "script": SCRIPT_REL,
            "script_md5": cc.md5(__file__), "workload_script_md5": cc.md5(rc.__file__),
            "common_md5": cc.md5(cc.__file__),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "software": {"python": platform.python_version(),
                         **{m: cc.version_of(m) for m in ("numpy", "numba", "jax",
                                                          "torch")}},
            "ns": list(args.ns), "distributions": list(args.distributions),
            "fixed_overhead_n": OVERHEAD_N, "release": cc.release_string(),
            "device_fma_per_s": fma, "ops_per_attempt": ops,
            "ops_per_attempt_source": "hand count; A100 dynamic count pending",
            "eff_sol_over_one": [[r["arm"], r["n"], r["distribution"]] for r in results
                                 if r.get("eff_sol_note")],
            "fma_probe": f"gcc -O3 -march=native -fopenmp register-only FMA chains, {_cpu_cores()} threads (cores), 8 lanes",
            "arms_not_run": arms_not_run,
            "arms": {a: ARM_LABELS[a] for a in names},
            "code_lines": code_line_counts(),
            "method": {"rounds": {"min": rc.ROUNDS_MIN, "max": args.reps, "stop": "IQR/median <= 2 %"}, "reference_arm": REFERENCE_ARM,
                       "memory": "per phase, in the worker, after its warm-up compile: the peak resident set "
                                 "(VmHWM) of the verification run above the RSS before it (host_peak_bytes) "
                                 "and the RSS the run leaves live above that baseline (host_live_bytes); "
                                 "host_peak_net_of_control_bytes subtracts the NumPy control's peak at the "
                                 "same N (null where the control did not run)",
                       "compile": "the worker's compile time on empty caches (hawk "
                                  "host build, Numba JIT, XLA compile at N)",
                       "threads": dict(ARM_THREADS),
                       "load_avg_1m": ("os.getloadavg()[0] recorded before and after "
                                       "each N's timed repetitions (per_n[*]."
                                       "load_avg_before/after); eagle_cpu8/eagle_cpu1 "
                                       "are the 1-thread vs N-thread rows"),
                       "simd": "not measured on this card (see cpu_card.py's simd.summary "
                               "for the same hawk host build's scalar/packed op counts)",
                       "slot_fill": "n/a on the host (device-only hook)",
                       "fixed_overhead": f"the eagle arms alone at N = {OVERHEAD_N}"},
            "results": results, "per_n": per_n,
            "wall_s": time.perf_counter() - t_start}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"cpu_card_{slug}.json"
    path.write_text(json.dumps(card, indent=1) + "\n")
    (args.out_dir / f"cpu_card_{slug}.md").write_text(render_markdown(card))
    print(f"wrote {path}")
    if not args.smoke and args.out_dir.resolve().is_relative_to(HERE.parent):
        cc.record_provenance(path)
    return cc.no_arm_ran_exit(card)


if __name__ == "__main__":
    sys.exit(main())
