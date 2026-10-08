# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The adaptive-step card on the CPU: the RKF7(8) Kepler workload, the common CPU ways.

The workload, the scheme, the truth and the agreement checks are those of
``rk78_card.py`` (imported from it). The arms:

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
finishes; they run up to ``MASKED_MAX_N`` samples.

Every arm runs in its own persistent worker process (its thread pool pinned
by that process's environment); the driver process interleaves the repetitions
across the workers (the arm order rotates every repetition), checks every arm
against the analytic truth and against ``eagle_cpu8`` before timing counts,
and reports medians with interquartile ranges. Peak host memory of the
warm-up run above the worker's post-compile baseline, and each worker's
compile time on empty caches, are recorded per (arm, N).

Reproduce (from the eagle repository root)::

    python benchmarks/rk78_card/cpu_rk78_card.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import platform
import re
import shutil
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):  # rk78_card.py; benchmarks/: the shared helpers
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _card_common as cc  # noqa: E402
import rk78_card as rc  # noqa: E402

SCRIPT_REL = "benchmarks/rk78_card/cpu_rk78_card.py"
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
MASKED_ARMS = ("torch_masked", "numpy_masked")
#: Arms with no compile step (PyTorch's and NumPy's kernels ship prebuilt).
NO_COMPILE_ARMS = MASKED_ARMS
MASKED_MAX_N = 100_000
NS = (1_000, 10_000, 100_000, 1_000_000)
REPS = 5
WARM_N = 64


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

    def setup(self, n):
        import numpy as np

        self.n = n
        self.inp = rc._inputs(n)
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
            w.compile_s = 0.0
            w.setup(WARM_N)
            w.run()  # warm-up at WARM_N: every compile happens here
            w.setup(cmd["n"])
            rss = cc.status_kib("VmRSS")
            cc.reset_host_peak()
            sec, out = w.run()  # the verification run
            peak = max(0, cc.status_kib("VmHWM") - rss) * 1024
            np.savez(cmd["out"], **out)
            msg = {"warm_s": sec, "host_peak_bytes": peak, "compile_s": w.compile_s}
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


def _run_n(n, reps, names, work):
    import numpy as np

    arms = [a for a in names if a not in MASKED_ARMS or n <= MASKED_MAX_N]
    procs = {a: _proc(a, work / f"cache_{a}_{n}") for a in arms}
    try:
        setup, outs = {}, {}
        for a in arms:
            path = work / f"out_{a}_{n}.npz"
            setup[a] = procs[a].call({"op": "setup", "n": n, "out": str(path)})
            with np.load(path) as z:
                outs[a] = {k: z[k] for k in z.files}
        ref = REFERENCE_ARM if REFERENCE_ARM in outs else arms[0]
        agreement, steps = rc._verify(outs, rc._inputs(n), ref)
        walls = {a: [] for a in arms}
        load_before = cc.load_avg_1m()
        for r in range(reps):
            rot = arms[r % len(arms):] + arms[:r % len(arms)]
            for a in rot:
                walls[a].append(procs[a].call({"op": "run"})["wall_s"])
        load_after = cc.load_avg_1m()
    finally:
        for p in procs.values():
            p.close({"op": "quit"})
    rows = []
    for a in arms:
        att = agreement[a]["attempts_total"]
        wall = cc.median_iqr(walls[a])
        rows.append({"n": n, "arm": a, "threads": ARM_THREADS[a], "wall_s": wall,
                     "attempts": att, "attempt_samples_per_s": att / wall["median"],
                     "useful_flops_per_s": rc.FLOPS_PER_ATTEMPT * att / wall["median"],
                     "host_peak_bytes": setup[a]["host_peak_bytes"],
                     "compile_s_cold": (None if a in NO_COMPILE_ARMS
                                        else setup[a]["compile_s"]),
                     "lane_utilisation": None})
    return rows, {"n": n, "reference_arm": ref, "agreement": agreement, "steps": steps,
                  "load_avg_before": load_before, "load_avg_after": load_after}


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
             f"Medians of {card['method']['repetitions']} interleaved repetitions; IQR "
             f"in parentheses. Workload as the GPU card ({rc.FLOPS_PER_ATTEMPT} FLOP per "
             f"attempt); masked arms up to N = {MASKED_MAX_N:,}.", "",
             "| N | arm | threads | wall (IQR) | attempts·samples/s | useful FLOP/s "
             "| host memory | compile (cold) | lines of code |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(card["results"], key=lambda r: (r["n"], ARMS.index(r["arm"]))):
        lines.append(
            f"| {r['n']:,} | {ARM_LABELS[r['arm']]} | {r['threads']} "
            f"| {cc.fmt_time_si(r['wall_s']['median'])} ({cc.fmt_time_si(r['wall_s']['iqr'])}) "
            f"| {cc.fmt_rate_si(r['attempt_samples_per_s'])} "
            f"| {cc.fmt_rate_si(r['useful_flops_per_s'])} "
            f"| {cc.fmt_mib(r['host_peak_bytes'])} | {cc.fmt_time_si(r['compile_s_cold'])} "
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path, default=HERE)
    ap.add_argument("--reps", type=int, default=REPS)
    ap.add_argument("--ns", type=int, nargs="+", default=list(NS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        _worker_main(args.worker)
        return 0
    names = tuple(a for a in ARMS if a in args.arms)
    work = pathlib.Path(tempfile.mkdtemp(prefix="rk78_cpucard_"))
    results, per_n = [], []
    try:
        for n in args.ns:
            t0 = time.perf_counter()
            rows, extra = _run_n(n, args.reps, names, work)
            results += rows
            per_n.append(extra)
            print(f"N={n:>8,} ({time.perf_counter() - t0:.1f}s)", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    cpu = cc.cpu_model()
    slug = re.sub(r"[^a-z0-9]+", "-", cpu.lower()).strip("-")
    card = {"schema": "eagle-rk78-cpu-card/2", "toolchain": cc.toolchain(), "cpu": cpu, "cpu_slug": slug,
            "logical_cpus": os.cpu_count(), "script": SCRIPT_REL,
            "script_md5": cc.md5(__file__), "workload_script_md5": cc.md5(rc.__file__),
            "common_md5": cc.md5(cc.__file__),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "software": {"python": platform.python_version(),
                         **{m: cc.version_of(m) for m in ("numpy", "numba", "jax",
                                                          "torch")}},
            "ns": list(args.ns), "arms": {a: ARM_LABELS[a] for a in names},
            "code_lines": code_line_counts(),
            "method": {"repetitions": args.reps, "reference_arm": REFERENCE_ARM,
                       "masked_max_n": MASKED_MAX_N,
                       "memory": "peak resident set (VmHWM) of the verification run "
                                 "above the worker's RSS after its warm-up compile",
                       "compile": "the worker's compile time on empty caches (hawk "
                                  "host build, Numba JIT, XLA compile at N)",
                       "threads": dict(ARM_THREADS),
                       "load_avg_1m": ("os.getloadavg()[0] recorded before and after "
                                       "each N's timed repetitions (per_n[*]."
                                       "load_avg_before/after); eagle_cpu8/eagle_cpu1 "
                                       "are the 1-thread vs N-thread rows"),
                       "simd": "not measured on this card (see cpu_card.py's simd.summary "
                               "for the same hawk host build's scalar/packed op counts)",
                       "lane_utilisation": "n/a on the host (device-only hook)",
                       "fixed_overhead": "not re-measured as a dedicated cell on this card"},
            "results": results, "per_n": per_n}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / f"cpu_card_{slug}.json"
    path.write_text(json.dumps(card, indent=1) + "\n")
    (args.out_dir / f"cpu_card_{slug}.md").write_text(render_markdown(card))
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
