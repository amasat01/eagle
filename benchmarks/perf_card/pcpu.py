# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The CPU profile step (pcpu): what CPU optimisation needs besides the cards' walls.

Items (each one labelled ``ok``, ``failed: ...`` or ``not_run: <reason>``; the others run whatever one does):

a  ISA A/B: the hawk host profile ``native`` vs ``portable`` (``$HAWK_HOST_PROFILE``) on the eagle ensemble arm.
b  thread sweep: eagle 1, 2, 4, 8, 16 threads and Numba prange 1 and 8 threads (``OMP_NUM_THREADS``, ``NUMBA_NUM_THREADS``).
c  ``perf stat`` (core events and the vector/FP retired events this CPU exposes) on the eagle, Numba and JAX ensemble arms
   and on the hawk feature rows (``feature_card.py --arm ROW CASE DTYPE ARM``).
d  ``perf record -g`` top 15 symbols of the eagle ensemble arm and the feature rows.
e  small-N baselines: NumPy masked and eagle 1 thread.
f  memory: RSS by phase and the top tracemalloc allocations of eagle and Numba at N = 1e6 and 1e5.

The ensemble arms are cpu_card's workers (its ``_Worker``) run as one-shot children (``--child``); perf counts only the
timed runs (``--control``), not the import and the compile. The feature rows are counted whole (their build is cached by an
earlier unprofiled run of the same child). Items are saved as parts under ``--parts`` (a re-run reuses them), and a pass of
``$CARD_DEADLINE`` labels the items not started ``not_run: session time limit``.

Reproduce (from the eagle repository root)::

    python benchmarks/perf_card/pcpu.py --out-dir DIR --parts DIR/parts
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time

HERE = pathlib.Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import _card_common as cc  # noqa: E402

MARK = "PCPU_CHILD "
N_BIG, N_MID = 1_000_000, 100_000
TOP = 15
#: the core events, then per vendor the vector/FP retired events (probed one by one: a missing name fails a whole perf stat)
CORE_EVENTS = ("cycles", "instructions", "branch-misses", "cache-misses", "LLC-load-misses")
VECTOR_EVENTS = (
    "fp_arith_inst_retired.scalar_double", "fp_arith_inst_retired.128b_packed_double",
    "fp_arith_inst_retired.256b_packed_double", "fp_arith_inst_retired.512b_packed_double",   # Intel
    "fp_ret_sse_avx_ops.all", "fp_ops_retired_by_type.all", "fp_ops_retired_by_width.scalar_uops_retired",
    "fp_ops_retired_by_width.pack_128_uops_retired", "fp_ops_retired_by_width.pack_256_uops_retired",
    "fp_ops_retired_by_width.pack_512_uops_retired")                                           # AMD Zen
F64 = "float64"
#: (row, case, dtype, arm) of the hawk feature rows profiled
FEATURE_ROWS = (("wide_forms", "all", F64, "thread"), ("steps_fused", "all", F64, "jvp_K8"),
                ("steps_fused", "all", F64, "vjp_K8_auto"), ("scatter", "all", F64, "keyed"),
                ("contraction", "all", F64, "gemm"))
ENSEMBLE = (("eagle", "eagle_cpu8"), ("numba", "numba_prange"), ("jax", "jax_vmap"))


# --------------------------------------------------------------------------- the one-shot child
def _rss_mib():
    return {"rss_mib": cc.status_kib("VmRSS") / 1024, "hwm_mib": cc.status_kib("VmHWM") / 1024}


def child(arm, n, reps, trace):
    """One ensemble arm: RSS by phase, ``reps`` timed runs (perf counts them when ``$PCPU_CTL`` names its control pipes)."""
    import tracemalloc

    if trace:
        tracemalloc.start()
    import cpu_card as cpc
    import numpy as np  # noqa: F401
    import perf_card as rc

    phases = {"after_import": _rss_mib()}
    base = rc._set_cell("spread")
    w = cpc._Worker(arm)
    w.inp = rc._inputs(n, base)
    phases["after_plane_alloc"] = _rss_mib()
    t0 = time.perf_counter()
    w.setup(cpc.WARM_N)
    w.run()                                   # every compile happens here
    compile_s = time.perf_counter() - t0
    phases["after_compile"] = _rss_mib()
    w.setup(n, base)
    phases["after_state_setup"] = _rss_mib()
    first, _ = w.run()
    phases["after_first_run"] = _rss_mib()
    ctl = os.environ.get("PCPU_CTL")
    if ctl:
        wfd, rfd = (int(x) for x in ctl.split(","))
        os.write(wfd, b"enable\n")
        os.read(rfd, 16)
    walls = [w.run()[0] for _ in range(reps)]
    if ctl:
        os.write(wfd, b"disable\n")
        os.read(rfd, 16)
    gc.collect()
    phases["steady"] = _rss_mib()
    rec = {"arm": arm, "n": n, "compile_s": compile_s, "first_run_s": first, "wall_s": walls,
           "wall_median_s": statistics.median(walls), "phases": phases,
           "threads": {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS")}}
    with contextlib.suppress(Exception):
        from hawk.compile import toolchain
        rec["host_profile"] = toolchain.host_profile()
    if trace:
        snap = tracemalloc.take_snapshot().statistics("lineno")[:10]
        rec["tracemalloc_top"] = [{"where": str(s.traceback), "mib": s.size / 2**20, "count": s.count} for s in snap]
    print(MARK + json.dumps(rec), flush=True)
    return 0


# --------------------------------------------------------------------------- the driver
class Ctx:
    def __init__(self, args, work):
        self.smoke, self.work, self.parts = args.smoke, work, args.parts
        self.reps = 1 if args.smoke else 3
        self.cap = 120 if args.smoke else 600
        self.perf = shutil.which("perf")
        self.events = None

    def env(self, arm, **extra):
        import cpu_card as cpc
        e = cpc._env(arm, self.work / "cache")
        e.update(CUDA_VISIBLE_DEVICES="", **extra)
        return e

    def timeout(self):
        left = float(os.environ["CARD_DEADLINE"]) - time.time() if os.environ.get("CARD_DEADLINE") else 1e9
        return max(30.0, min(self.cap, left + 60))


def run_child(cx, arm, n, env, trace=False):
    """The child's record, or a status string."""
    cmd = [sys.executable, str(pathlib.Path(__file__).resolve()), "--child", arm, str(n), "--reps", str(cx.reps)]
    if trace:
        cmd.append("--trace")
    try:
        done = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=cx.timeout())
    except subprocess.TimeoutExpired:
        return f"failed: no result in {cx.timeout():.0f} s"
    for ln in reversed(done.stdout.splitlines()):
        if ln.startswith(MARK):
            return json.loads(ln[len(MARK):])
    last = ([ln.strip() for ln in (done.stderr + done.stdout).splitlines() if ln.strip()] or ["no output"])[-1][:160]
    return f"failed: process exit {done.returncode}: {last}"


LIBRARY = {"numba_prange": "numba", "jax_vmap": "jax"}


def missing_lib(arm):
    import perf_card as rc
    lib = LIBRARY.get(arm)
    return f"not_run: {lib} not installed in this profile" if lib and not rc._lib_installed(lib) else None


def timed_item(cx, arm, n, **extra):
    if missing_lib(arm):
        return {"status": missing_lib(arm)}
    rec = run_child(cx, arm, n, cx.env(arm, **extra))
    if isinstance(rec, str):
        return {"status": rec}
    return {"status": "ok", **{k: rec[k] for k in ("n", "arm", "wall_s", "wall_median_s", "compile_s", "threads")},
            "host_profile": rec.get("host_profile")}


def probe_events(cx):
    """The events this CPU and kernel count: each name probed alone; None if perf cannot count at all."""
    if cx.events is None:
        ok = {}
        for ev in (*CORE_EVENTS, *VECTOR_EVENTS):
            done = subprocess.run([cx.perf, "stat", "-x,", "-e", ev, "--", "true"], capture_output=True, text=True)
            vals = [ln.split(",")[0] for ln in done.stderr.splitlines() if ev in ln]
            ok[ev] = done.returncode == 0 and bool(vals) and vals[0].strip().isdigit()
        cx.events = ok
    return cx.events


def no_perf(cx):
    """The reason perf cannot be used, or None."""
    if not cx.perf:
        return "not_run: perf not installed"
    if not any(probe_events(cx).values()):
        return "not_run: perf counters not permitted or not exposed (perf_event_paranoid / PMU)"
    return None


def perf_run(cx, kind, target, out_dir, tag, events=()):
    """``perf stat`` or ``perf record -g`` around ``target(wrap, ctl)``; returns (record or status, raw output path)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{tag}.{'stat' if kind == 'stat' else 'record'}"
    pipes = [os.pipe(), os.pipe()]        # perf reads pipes[0][0], acks on pipes[1][1]; the child writes [0][1], reads [1][0]
    base = [cx.perf, kind]
    base += ["-x,", "-e", ",".join(events), "-o", str(raw)] if kind == "stat" else ["-g", "-F", "999", "-o", str(raw)]
    base += ["-D", "-1", "--control", f"fd:{pipes[0][0]},{pipes[1][1]}", "--"]
    try:
        rec = target(lambda cmd: base + cmd, (pipes[0][1], pipes[1][0]), pass_extra=(pipes[0][0], pipes[1][1]))
    finally:
        for r, w in pipes:
            os.close(r)
            os.close(w)
    return rec, raw


def parse_stat(path):
    out = {}
    for ln in pathlib.Path(path).read_text().splitlines():
        f = ln.split(",")
        if len(f) > 2 and f[2] and not ln.startswith("#"):
            out[f[2]] = float(f[0]) if f[0].strip().replace(".", "", 1).isdigit() else f[0]
    cyc, ins = out.get("cycles"), out.get("instructions")
    if isinstance(cyc, float) and isinstance(ins, float) and cyc:
        out["ipc"] = ins / cyc
    return out


def top_symbols(raw, report):
    done = subprocess.run([shutil.which("perf"), "report", "-i", str(raw), "--no-children", "--stdio", "-g", "none",
                           "--sort", "symbol"], capture_output=True, text=True)
    pathlib.Path(report).write_text(done.stdout)
    rows = []
    for ln in done.stdout.splitlines():
        m = re.match(r"^\s+([\d.]+)%\s+\[.\]\s+(.*\S)", ln)
        if m:
            rows.append({"percent": float(m[1]), "symbol": m[2].split("  ")[0]})
    return rows[:TOP]


def perf_ensemble(cx, kind, arm, n, out_dir, tag, events=()):
    """perf around an ensemble child; its control pipes pass through perf to the child."""
    def target(wrap, ctl, pass_extra):
        cmd_env = cx.env(arm, PCPU_CTL=f"{ctl[0]},{ctl[1]}")
        cmd = wrap([sys.executable, str(pathlib.Path(__file__).resolve()), "--child", arm, str(n), "--reps", str(cx.reps)])
        try:
            done = subprocess.run(cmd, env=cmd_env, capture_output=True, text=True, timeout=cx.timeout(),
                                  pass_fds=(*ctl, *pass_extra))
        except subprocess.TimeoutExpired:
            return f"failed: no result in {cx.timeout():.0f} s"
        if done.returncode != 0:
            last = ([x.strip() for x in (done.stderr + done.stdout).splitlines() if x.strip()] or ["no output"])[-1][:160]
            return f"failed: process exit {done.returncode}: {last}"
        return "ok"
    return perf_run(cx, kind, target, out_dir, tag, events)


def feature_cmd(row, case, dtype, arm, smoke):
    return [sys.executable, str(HERE / "feature_card.py"), "--arm", row, case, dtype, arm] + (["--smoke"] if smoke else [])


def perf_feature(cx, kind, spec, out_dir, tag, events=()):
    """perf around a feature row's per-arm child (whole process: its build comes from a cache warmed by one plain run first)."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "FEATURE_CARD_CACHE": str(cx.work / "feature_cache")}
    cmd = feature_cmd(*spec, cx.smoke)
    warm = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=cx.timeout())
    status = next((json.loads(ln.split(" ", 1)[1])["status"] for ln in warm.stdout.splitlines() if ln.startswith("FEATURE_ARM ")),
                  f"failed: process exit {warm.returncode}")
    if status != "ok":
        return status, None
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = out_dir / f"{tag}.{'stat' if kind == 'stat' else 'record'}"
    base = [cx.perf, kind]
    base += ["-x,", "-e", ",".join(events), "-o", str(raw)] if kind == "stat" else ["-g", "-F", "999", "-o", str(raw)]
    try:
        done = subprocess.run([*base, "--", *cmd], env=env, capture_output=True, text=True, timeout=cx.timeout())
    except subprocess.TimeoutExpired:
        return f"failed: no result in {cx.timeout():.0f} s", None
    return ("ok" if done.returncode == 0 else f"failed: perf exit {done.returncode}"), raw


def stat_events(cx, which):
    ev = probe_events(cx)
    return [e for e in (CORE_EVENTS if which == "core" else VECTOR_EVENTS) if ev[e]]


def stat_item(cx, spec, out_dir, tag):
    """perf stat of one target in a core pass and a vector pass (no multiplexing); the events counted are recorded."""
    if spec[0] == "ens" and missing_lib(spec[1]):
        return {"status": missing_lib(spec[1])}
    rec = {"status": "ok", "events_unavailable": [e for e, ok in probe_events(cx).items() if not ok]}
    for which in ("core", "vector"):
        evs = stat_events(cx, which)
        if not evs:
            rec[which] = "not_run: no such event on this CPU"
            continue
        status, raw = (perf_ensemble(cx, "stat", *spec[1:], out_dir, f"{tag}.{which}", evs) if spec[0] == "ens"
                       else perf_feature(cx, "stat", spec[1], out_dir, f"{tag}.{which}", evs))
        if status != "ok":
            rec[which] = status
            continue
        rec[which] = parse_stat(raw)
    if not any(isinstance(rec[w], dict) for w in ("core", "vector")):
        rec["status"] = next(rec[w] for w in ("core", "vector"))
    return rec


def record_item(cx, spec, out_dir, tag):
    status, raw = (perf_ensemble(cx, "record", *spec[1:], out_dir, tag) if spec[0] == "ens"
                   else perf_feature(cx, "record", spec[1], out_dir, tag))
    if status != "ok":
        return {"status": status}
    top = top_symbols(raw, out_dir / f"{tag}.report.txt")
    raw.unlink(missing_ok=True)           # perf.data is large; the report text stays
    return {"status": "ok" if top else "failed: perf report listed no symbols", "top_symbols": top}


def mem_item(cx, arm, n):
    if missing_lib(arm):
        return {"status": missing_lib(arm)}
    plain = run_child(cx, arm, n, cx.env(arm))
    if isinstance(plain, str):
        return {"status": plain}
    traced = run_child(cx, arm, n, cx.env(arm), trace=True)
    return {"status": "ok", "n": n, "arm": arm, "phases": plain["phases"],
            "tracemalloc_top": traced.get("tracemalloc_top") if isinstance(traced, dict) else traced}


# --------------------------------------------------------------------------- the plan
def plan(cx):
    """(key, function) in priority order: a deadline drops the tail."""
    smoke = cx.smoke
    n = 1_000 if smoke else N_MID
    out = cx.out_dir
    items = []
    for prof in ("native", "portable"):
        items.append((f"a_isa_{prof}", lambda p=prof: timed_item(
            cx, "eagle_cpu8", n, HAWK_HOST_PROFILE=p, HAWK_CACHE_DIR=str(cx.work / f"hawk-{p}"))))
    for t in ((1, 2) if smoke else (1, 2, 4, 8, 16)):
        items.append((f"b_threads_eagle_{t}", lambda t=t: timed_item(cx, "eagle_cpu8", n, OMP_NUM_THREADS=str(t))))
    for t in ((1,) if smoke else (1, 8)):
        items.append((f"b_threads_numba_{t}", lambda t=t: timed_item(
            cx, "numba_prange", n, OMP_NUM_THREADS=str(t), NUMBA_NUM_THREADS=str(t))))
    for label, arm in ENSEMBLE:
        items.append((f"c_stat_{label}", lambda a=arm: stat_item(cx, ("ens", a, n), out / "perf", f"stat_{a}")))
    rows = FEATURE_ROWS[:1] if smoke else FEATURE_ROWS
    for spec in rows:
        tag = f"{spec[0]}_{spec[3]}"
        items.append((f"c_stat_{tag}", lambda s=spec, t=tag: stat_item(cx, ("row", s), out / "perf", f"stat_{t}")))
    items.append(("e_baseline_numpy_masked", lambda: timed_item(cx, "numpy_masked", n)))
    items.append(("e_baseline_eagle_1thread", lambda: timed_item(cx, "eagle_cpu1", n)))
    for arm in ("eagle_cpu8", "numba_prange"):
        for nn in ((1_000,) if smoke else (N_BIG, N_MID)):
            items.append((f"f_memory_{arm}_{nn}", lambda a=arm, m=nn: mem_item(cx, a, m)))
    if smoke:
        items.append(("d_record", lambda: {"status": "not_run: smoke (perf stat only)"}))
    else:
        items.append(("d_record_eagle", lambda: record_item(cx, ("ens", "eagle_cpu8", n), out / "perf", "record_eagle_cpu8")))
        for spec in rows:
            tag = f"{spec[0]}_{spec[3]}"
            items.append((f"d_record_{tag}", lambda s=spec, t=tag: record_item(cx, ("row", s), out / "perf", f"record_{t}")))
    return items


def needs_perf(key):
    return key.startswith(("c_", "d_"))


def render_markdown(summary):
    lines = ["# pcpu: CPU profile step", "", f"CPU: {summary['cpu']}; perf: {summary['perf'] or 'absent'}", "",
             "| item | status | wall median (s) |", "|---|---|---|"]
    for k, v in summary["items"].items():
        wall = v.get("wall_median_s")
        lines.append(f"| {k} | {v['status']} | {'' if wall is None else f'{wall:.3g}'} |")
    for k, v in summary["items"].items():
        if v.get("top_symbols"):
            lines += ["", f"## {k}: top symbols", ""] + [f"- {s['percent']:.1f} %  `{s['symbol']}`" for s in v["top_symbols"]]
    lines += ["", "Raw perf outputs: `perf/`. Counted: the timed runs of the ensemble arms; the whole feature-row child.", ""]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out-dir", type=pathlib.Path)
    ap.add_argument("--parts", type=pathlib.Path)
    ap.add_argument("--smoke", action="store_true", help="N = 1e3, 1 and 2 threads, perf stat only")
    ap.add_argument("--child", nargs=2, metavar=("ARM", "N"), help=argparse.SUPPRESS)
    ap.add_argument("--reps", type=int, default=3, help=argparse.SUPPRESS)
    ap.add_argument("--trace", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.child:
        return child(args.child[0], int(args.child[1]), args.reps, args.trace)
    out_dir = args.out_dir or pathlib.Path(tempfile.mkdtemp(prefix="pcpu_"))
    work = pathlib.Path(tempfile.mkdtemp(prefix="pcpu_work_"))
    cx = Ctx(args, work)
    cx.out_dir = out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    perf_why = no_perf(cx)
    items, t_start = {}, time.perf_counter()
    summary = {"schema": "eagle-pcpu/1", "cpu": cc.cpu_model(), "perf": cx.perf, "perf_events": cx.events,
               "smoke": args.smoke, "release": cc.release_string(), "items": items}

    def write():
        summary["wall_s"] = time.perf_counter() - t_start
        (out_dir / "pcpu_summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
        (out_dir / "pcpu_summary.md").write_text(render_markdown(summary))

    try:
        for key, fn in plan(cx):
            saved = cc.part_load(args.parts, key)
            if saved is not None:
                items[key] = saved
            elif cc.deadline_passed():
                items[key] = {"status": cc.SESSION_LIMIT}
            elif needs_perf(key) and perf_why:
                items[key] = {"status": perf_why}
            else:
                try:
                    items[key] = fn()
                except Exception as exc:  # noqa: BLE001  one item never stops the rest
                    items[key] = {"status": f"failed: {type(exc).__name__}: {str(exc).splitlines()[0][:160] if str(exc) else ''}"}
                cc.part_save(args.parts, key, items[key])
            print(f"{key}: {items[key]['status']}", flush=True)
            write()
    finally:
        shutil.rmtree(work, ignore_errors=True)
    summary["perf_events"] = cx.events
    write()
    print(f"wrote {out_dir / 'pcpu_summary.json'}")
    ran = any(v["status"] == "ok" for v in items.values())
    if not ran:
        print("no item of this step ran", file=sys.stderr)
    return 0 if ran else 1


if __name__ == "__main__":
    sys.exit(main())
