#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Calibration data for the dispatch cost model: raw timings, no fitting.

Runs five groups of isolated micro-cases on ONE target (the GPU the process sees,
or the host) and writes one CSV: ``# key=value`` header lines describing the machine
and the software, then one row per repetition (every repetition kept, the first of
each case marked ``warm=0``). Nothing on the machine is changed. The data is fitted
later, elsewhere.

  A  class slopes     one kernel per operation class (arith/special in two precisions,
                      plane loads, gathers), a loop of m operations, three m; and the
                      same arithmetic as 4 independent chains (separates dependent-chain
                      latency from throughput)
  B  launch + planes  an almost empty kernel, three batch sizes, 1 or 16 fused steps
  C  split overhead   the per-branch units of a split step, and the class sort
  D  divergence       (GPU) one fused cheap/heavy kernel at four heavy fractions
  E  held out         paired fused vs split steps at three heavy fractions, for a
                      library-call heavy branch and for a cheap vectorisable branch
                      against a heavy one that carries a long loop (scalar)

Usage: calibrate.py --target gpu|cpu [--quick] [--out PATH] [--reps N] [--seed S]
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import importlib
import importlib.util
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCHEMA = 2
CLASSES = ("a32", "a64", "s32", "s64", "ld", "ga")
MAX_BRANCHES = 4
COLUMNS = ["case_id", "group", "kind", "target", "kernel_src_md5", "codegen_md5", "n", "K",
           "fused_steps", "steps", "m", "counts",
           *[f"cost_b{j}" for j in range(MAX_BRANCHES)], "fused_vectorised", "chains",
           "held_out", "rep", "warm", "timer", "wall_s", "status"]

KERNELS_SRC = '''\
import hawk, hawk.math as M
from hawk import Index, Mutable, Scalar, Table, Terminated, Vector
'''


def md5(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


# --- the kernels (generated per trip count: a hawk loop's bound is fixed at trace time) --- #
def _plain(name: str, params: str, body: str, steps: bool = True) -> str:
    deco = "@hawk.kernel(steps=1)" if steps else "@hawk.kernel"
    lines = "\n".join("    " + ln for ln in body.strip("\n").splitlines())
    lead = f"{params}, " if params else ""
    return (f"{deco}\ndef {name}({lead}x: Mutable[Scalar], t: Mutable[Scalar], "
            f"terminated: Terminated):\n{lines}\n    t = t + 1.0\n    terminated = t >= 1e30\n\n")


def class_kernel(cls: str, m: int) -> tuple:
    """``(name, source)`` of the class-``cls`` kernel with a loop of ``m`` operations
    (``ld``: ``m`` plane loads)."""
    name = f"cal_{cls}_{m}"
    if cls == "arith":
        return name, _plain(name, "", f"a = x\nfor _i in range({m}):\n    a = a * 0.9999 + 0.01\nx = a")
    if cls == "arith4":     # the same m multiply-adds as 4 independent chains of m / 4
        return name, _plain(name, "", "a = x\nb = x * 1.1\nc = x * 1.2\nd = x * 1.3\n"
                            f"for _i in range({max(m // 4, 1)}):\n    a = a * 0.9999 + 0.01\n"
                            "    b = b * 0.9999 + 0.01\n    c = c * 0.9999 + 0.01\n"
                            "    d = d * 0.9999 + 0.01\nx = a + b + c + d")
    if cls == "special":    # the argument stays in a steady range (as the heavy branch of group E does)
        return name, _plain(name, "", f"a = x\nfor _i in range({m}):\n    a = M.sin(a) * 1.01\nx = a")
    if cls == "ld":
        return name, _plain(name, f"v: Vector[{m}]", "x = x + M.vsum(v)")
    return name, _plain(name, "tab: Table[Scalar], gi: Index",
                        f"a = x\nfor _i in range({m}):\n    a = a + tab.at(gi + _i)\nx = a")


def contrast_kernel(name: str, trips: int) -> tuple:
    """A two-way dispatch whose cheap branch vectorises and whose heavy branch carries a
    ``trips``-trip loop of plain arithmetic (rolled, so scalar, above the unroll limit)."""
    pre = f"a = x\nfor _i in range({trips}):\n    a = a * 0.9999 + 0.01\n"
    return name, _plain(name, "k: Index", pre + 'x = M.dispatch(k, [x * 1.25 + 0.5, a], policy="auto")')


def dispatch_kernel(name: str, k_branches: int, heavy: int) -> tuple:
    """A kernel with one dispatch over ``k_branches`` (a cheap multiply for every
    branch but the last, which is a ``heavy``-sine loop when ``heavy`` > 0)."""
    cheap = [f"x * {1.0 + 0.25 * j} + 0.5" for j in range(k_branches)]
    pre, last = "", cheap[-1]
    if heavy:
        pre = f"a = x\nfor _i in range({heavy}):\n    a = M.sin(a) * 1.01\n"
        last = "a"
    branches = ", ".join([*cheap[:-1], last])
    body = f"{pre}x = M.dispatch(k, [{branches}], policy=\"auto\")"
    return name, _plain(name, "k: Index", body)


EMPTY_SRC = ('@hawk.kernel\ndef cal_empty(x: Mutable[Scalar], t: Mutable[Scalar], a: Mutable[Scalar], '
             'b: Mutable[Scalar], terminated: Terminated):\n    x = x + 1.0\n    a = a + x\n    b = b + a\n'
             '    t = t + 1.0\n    terminated = t >= 1e30\n\n')


class Kernels:
    """Kernel sources written to a private module (hawk traces from source files)."""

    def __init__(self, workdir: Path):
        self.sources: dict = {}
        self.dir = workdir

    def add(self, name_src: tuple) -> str:
        self.sources[name_src[0]] = name_src[1]
        return name_src[0]

    def load(self):
        text = KERNELS_SRC + "".join(self.sources.values())
        path = self.dir / f"cal_kernels_{md5(text)[:10]}.py"
        path.write_text(text)
        spec = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = mod
        spec.loader.exec_module(mod)
        return mod

    def src_md5(self, name: str) -> str:
        return md5(self.sources[name])


# --- machine facts --- #
def sh(cmd: list) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def smi(index: int) -> str:
    out = sh(["nvidia-smi", "-i", str(index), "--query-gpu=clocks.sm,clocks.mem,temperature.gpu,"
              "power.draw,utilization.gpu", "--format=csv,noheader"])
    return out.replace(", ", ",") or "unavailable"


def git_sha(module) -> str:
    here = Path(module.__file__).resolve().parent
    for d in (here, *here.parents[:3]):
        if (d / ".git").exists():
            return sh(["git", "-C", str(d), "rev-parse", "--short=12", "HEAD"]) or "unknown"
    return os.environ.get("EAGLE_GIT_SHA", "release")


def version_of(dist: str) -> str:
    from importlib import metadata

    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return "unknown"


def isa_level() -> str:
    flags = ""
    try:
        flags = next(ln for ln in Path("/proc/cpuinfo").read_text().splitlines() if ln.startswith("flags"))
    except (OSError, StopIteration):
        pass
    for f in ("avx512f", "avx2", "avx", "sse4_2"):
        if re.search(rf"\b{f}\b", flags):
            return f
    return platform.machine()


def cpu_model() -> str:
    try:
        return next(ln.split(":", 1)[1].strip() for ln in Path("/proc/cpuinfo").read_text().splitlines()
                    if ln.startswith("model name"))
    except (OSError, StopIteration):
        return platform.processor() or platform.machine()


def openmp_runtime() -> str:
    try:
        libs = {Path(ln.split()[-1]).name for ln in Path("/proc/self/maps").read_text().splitlines()
                if re.search(r"lib(gomp|omp|iomp5)", ln)}
    except OSError:
        libs = set()
    return ",".join(sorted(libs)) or "none-loaded"


def header(args, device_index: int, hawk_mod, eagle_mod, opt: str) -> dict:
    from hawk.compile import toolchain

    h = {"schema": SCHEMA, "collector_md5": md5(Path(__file__).read_text()),
         "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
         "target": args.target, "mode": "quick" if args.quick else "full",
         "eagle_version": version_of("raptor-eagle"), "eagle_sha": git_sha(eagle_mod),
         "hawk_version": version_of("raptor-hawk"), "hawk_sha": git_sha(hawk_mod),
         "compiler": sh([os.environ.get("CXX", "c++"), "--version"]).splitlines()[0:1] or "unknown",
         "opt_level": opt, "host_profile": toolchain.host_profile(),
         "python": platform.python_version(), "cpu_model": cpu_model(), "isa": isa_level(),
         "threads": os.cpu_count(), "omp_proc_bind": os.environ.get("OMP_PROC_BIND", "unset"),
         "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "unset"), "seed": args.seed}
    h["compiler"] = h["compiler"][0] if isinstance(h["compiler"], list) and h["compiler"] else "unknown"
    if args.target == "gpu":
        from eagle._device_props import device_props

        props = device_props(0)
        for name in props.__dataclass_fields__:
            h[f"dev_{name}"] = getattr(props, name)
        h["driver"] = sh(["nvidia-smi", "-i", str(device_index), "--query-gpu=driver_version",
                          "--format=csv,noheader"]) or "unknown"
        import cupy

        h["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
        h["cuda_driver_api"] = cupy.cuda.runtime.driverGetVersion()
        h["gpu_name"] = sh(["nvidia-smi", "-i", str(device_index), "--query-gpu=name",
                            "--format=csv,noheader"])
        h["smi_before"] = smi(device_index)
    return h


# --- counts of an op mix, from the cost model itself --- #
class Costs(list):
    """The per-branch cost rows (6 counts, 4 chain lengths, vectorised, and the plane
    elements written when counted), plus hawk's fact whether the fused host loop vectorises."""

    vectorised = False


def _row(b) -> tuple:
    stores = getattr(b, "stores", None)
    return (*b.chain_tuple(), int(b.vectorised), *(() if stores is None else (stores,)))


def plain_costs(kernel, relabel32: bool) -> Costs:
    """The cost of one sample of a kernel without a dispatch, as hawk counts it for the
    host profile in effect; the unit rows record hawk's own ``vectorised``."""
    from eagle._dispatch_door import _step
    from hawk.ir.cost import kernel_cost

    b = kernel_cost(_step(kernel).walk)
    row = list(_row(b))
    if relabel32:           # a float32 build runs the model's 64-bit classes at 32-bit cost; the step keeps its own vectorised flag
        a32, a64, s32, s64, ld, ga, c32, c64, cs32, cs64, vec, *stores = row
        row = [a32 + a64, 0.0, s32 + s64, 0.0, ld, ga, c32 + c64, 0.0, cs32 + cs64, 0.0, vec, *stores]
    out = Costs([tuple(row)])
    out.vectorised = bool(row[10])
    return out


def fmt_cost(c) -> str:
    return ";".join(f"{v:g}" for v in c)


def branch_costs(kernel, group="dispatch0") -> Costs:
    from eagle._dispatch_door import _step
    from hawk.ir import dispatch_costs

    cost = dispatch_costs(_step(kernel).walk)[group]
    out = Costs(_row(b) for b in cost.branches)
    out.vectorised = bool(cost.fused_vectorised)
    return out


# --- planes and timing --- #
class Rig:
    """Allocates planes for a plugin's arg_spec on the target, binds, and times."""

    def __init__(self, target: str, seed: int):
        self.target = target
        self.rng = np.random.default_rng(seed)
        if target == "gpu":
            import cupy

            self.xp = cupy
        else:
            self.xp = np

    real = np.float64          # element type of the float planes of the build under test

    def host_to(self, arr):
        arr = np.asarray(arr)
        return self.xp.asarray(arr.astype(self.real) if arr.dtype.kind == "f" else arr)

    def planes(self, plugin, n: int, **given) -> dict:
        xp, out = self.xp, {}
        for role, name in plugin.arg_spec:
            if name in given:
                out[name] = given[name]
            elif role == "terminated":
                out[name] = xp.zeros(n, dtype=xp.bool_)
            elif name == "finished_count":
                out[name] = xp.zeros(1, dtype=xp.uint32)
            elif name == "fused_steps":
                out[name] = xp.ones(1, dtype=xp.int64)
            elif name.endswith("_offsets"):
                out[name] = xp.zeros(MAX_BRANCHES + 1, dtype=xp.int64)
            elif role in ("vec_in",):
                out[name] = self.host_to(self.rng.random((n, 1)))
            elif role in ("mutable", "per_sample", "lookup"):
                out[name] = self.host_to(self.rng.random(n))
            else:
                raise RuntimeError(f"calibrate: no plane rule for {role} {name!r}")
        return out

    def bind(self, plugin, n: int, **given):
        from eagle._plan_deploy import auto

        planes = self.planes(plugin, n, **given)
        plan = auto(plugin).select(planes)
        return plan.bind(**planes), planes

    def time(self, bound, count: int, reps: int, fused_steps: int = 1) -> list:
        """Seconds of ``reps`` launches over ``[0, count)`` of the bound plan."""
        from eagle import _dispatch_door as door
        from eagle import exec as eexec

        if count == bound._parts[0].count:
            return door.time_launch(bound, reps, fused_steps)
        from eagle import _core
        from eagle._plan_binding import DEVICE_CUDA

        word = bound._planes.get(door.FUSED_STEPS_PLANE) if isinstance(bound._planes, dict) else None
        if word is not None:
            word[...] = fused_steps
        part = eexec.Partition(0, int(count), bound._parts[0].nSamples if hasattr(
            bound._parts[0], "nSamples") else count)
        if bound._device_type != DEVICE_CUDA:
            return list(_core.time_launch(bound._entry, bound._addrs, part, reps))
        import cupy

        block = int(importlib.import_module("eagle.launch")._resolved_block(bound._entry, count, None))
        return list(_core.time_launch(bound._entry, bound._addrs, part, reps, device=True,
                                      stream=int(cupy.cuda.get_current_stream().ptr), block=block))

    def wall(self, fn) -> float:
        if self.target == "gpu":
            self.xp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        fn()
        if self.target == "gpu":
            self.xp.cuda.Device().synchronize()
        return time.perf_counter() - t0


class Writer:
    def __init__(self, fh, target: str, reps: int):
        self.w = csv.writer(fh)
        self.w.writerow(COLUMNS)
        self.target, self.reps, self.rows, self.case = target, reps, 0, 0
        self.timer = "event" if target == "gpu" else "steady_clock"

    def case_rows(self, group, kind, *, src, codegen, n, K, fused_steps, steps, m, counts, costs,
                  seconds, held_out=0, timer=None, status="ok", chains=0):
        self.case += 1
        cid = f"{group}{self.case:03d}"
        codegen = codegen if codegen in ("n/a", "unknown") else md5(codegen)   # of the bundle digest
        cols = [fmt_cost(c) for c in costs] + [""] * (MAX_BRANCHES - len(costs))
        for rep, s in enumerate(seconds):
            self.w.writerow([cid, group, kind, self.target, src, codegen, n, K, fused_steps, steps, m,
                             ";".join(str(c) for c in counts), *cols, int(getattr(costs, "vectorised", False)),
                             chains, held_out, rep,
                             int(rep > 0), timer or self.timer, f"{s:.9e}", status])
            self.rows += 1


def digest_of(plugin) -> str:
    src = getattr(plugin, "_hawk_source", None)
    return str(src[1]) if src else "unknown"


def build(ker_mod, names, scalar_type, targets, clock):
    """Build kernels into one bundle; seconds spent are added to ``clock``."""
    from eagle._plan_deploy import _hawk_plugins

    t0 = time.perf_counter()
    plugins = _hawk_plugins([getattr(ker_mod, n) for n in names], targets, None, scalar_type)
    clock[0] += time.perf_counter() - t0
    return dict(zip(names, plugins))


def split_units(ker_mod, name, targets, clock):
    """The segmented units and classify plugin of dispatch kernel ``name``."""
    import hawk
    from eagle._plan_deploy import _build_by_name

    seg = hawk.resolve_dispatch(getattr(ker_mod, name), {"dispatch0": "segmented"})
    t0 = time.perf_counter()
    by, bundle = _build_by_name([seg], targets, None)
    clock[0] += time.perf_counter() - t0
    units = sorted(n for n in by if "__k" in n)
    return [by[u] for u in units], by.get(f"{seg.name}__cls"), str(bundle.digest)


def fused_plugin(ker_mod, name, targets, clock):
    import hawk
    from eagle._plan_deploy import _build_by_name

    sw = hawk.resolve_dispatch(getattr(ker_mod, name), {"dispatch0": "switch"})
    t0 = time.perf_counter()
    by, bundle = _build_by_name([sw], targets, None)
    clock[0] += time.perf_counter() - t0
    return by[sw.name], str(bundle.digest)


# --- the groups --- #
def group_a(ctx):
    n, ms = ctx["n_a"], ctx["ms"]
    ker = Kernels(ctx["dir"])
    wanted = []        # (cls, m, scalar)
    for m in ms:
        for cls, scalars in (("arith", ("float64", "float32")), ("arith4", ("float64",)),
                             ("special", ("float64", "float32")), ("ld", ("float64",)), ("ga", ("float64",))):
            mm = max(m // 4, 1) if cls == "ld" else m
            ker.add(class_kernel(cls, mm))
            wanted += [(cls, mm, m, s) for s in scalars]
    mod = ker.load()
    clock = ctx["clock"]
    plugins = {}
    for scalar in ("float64", "float32"):
        for local in (True, False):          # gathers read other samples: a bundle of their own
            names = sorted({f"cal_{c}_{mm}" for c, mm, _m, s in wanted
                            if s == scalar and (c != "ga") == local})
            if names:
                for nm, p in build(mod, names, scalar, ctx["targets"], clock).items():
                    plugins[(nm, scalar)] = p
    rig, w = ctx["rig"], ctx["writer"]
    for cls, mm, m, scalar in wanted:
        name = f"cal_{cls}_{mm}"
        plugin = plugins[(name, scalar)]
        rig.real = np.float32 if scalar == "float32" else np.float64
        given = {}
        if cls == "ga":
            given = {"tab": rig.host_to(rig.rng.random(2048)),
                     "gi": rig.host_to(rig.rng.integers(0, 1024, n).astype(np.int64))}
        if cls == "ld":
            given = {"v": rig.host_to(rig.rng.random((mm, n)))}
        bound, _ = rig.bind(plugin, n, **given)
        secs = rig.time(bound, n, ctx["reps"] + 1)
        rig.real = np.float64
        costs = plain_costs(getattr(mod, name), scalar == "float32")
        w.case_rows("A", "unit", src=ker.src_md5(name), codegen=digest_of(plugin), n=n, K=1,
                    fused_steps=1, steps=1, m=mm, counts=[n], costs=costs, seconds=secs,
                    chains={"arith": 1, "arith4": 4}.get(cls, 0) if scalar == "float64" else 0)
        label = cls + ("32" if scalar == "float32" else "")
        ctx["slopes"].setdefault(label, {})[mm] = float(np.median(secs[1:]))


def group_b(ctx):
    ker = Kernels(ctx["dir"])
    ker.sources["cal_empty"] = EMPTY_SRC
    mod = ker.load()
    plugin = build(mod, ["cal_empty"], None, ctx["targets"], ctx["clock"])["cal_empty"]
    rig, w = ctx["rig"], ctx["writer"]
    costs = plain_costs(mod.cal_empty, False)
    for n in ctx["ns_b"]:
        bound, _ = rig.bind(plugin, n)
        for fs in (1, 16):
            secs = rig.time(bound, n, ctx["reps"] + 1, fs)
            w.case_rows("B", "launch", src=ker.src_md5("cal_empty"), codegen=digest_of(plugin), n=n,
                        K=1, fused_steps=fs, steps=fs, m=0, counts=[n], costs=costs, seconds=secs)


def _classes_for(rig, n, k_branches, shares):
    """A class plane with the given share per branch (iid), and its counts."""
    cls = rig.rng.choice(k_branches, size=n, p=shares).astype(np.int64)
    return cls, np.bincount(cls, minlength=k_branches)


def split_step(rig, units, n, counts, reps):
    """Per-repetition seconds of one split step: every unit over its own run."""
    off = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    total = np.zeros(reps)
    for j, plugin in enumerate(units):
        if counts[j] == 0:
            continue
        bound, planes = rig.bind(plugin, n)
        planes_off = [v for k, v in planes.items() if k.endswith("_offsets")][0]
        planes_off[: len(off)] = rig.host_to(off)[: len(planes_off)] if len(off) <= len(planes_off) \
            else rig.host_to(off[: len(planes_off)])
        total += np.array(rig.time(bound, int(counts[j]), reps))
    return list(total)


def group_c(ctx):
    n_list, rig, w = ctx["ns_c"], ctx["rig"], ctx["writer"]
    ker = Kernels(ctx["dir"])
    for k in (2, 4):
        ker.add(dispatch_kernel(f"cal_split{k}", k, 0))
    mod = ker.load()
    for k in (2, 4):
        name = f"cal_split{k}"
        units, _cls, digest = split_units(mod, name, ctx["targets"], ctx["clock"])
        costs = branch_costs(getattr(mod, name))
        for n in n_list:
            counts = np.full(k, n // k, dtype=np.int64)
            counts[-1] += n - counts.sum()
            secs = split_step(rig, units, n, counts, ctx["reps"] + 1)
            w.case_rows("C", "split", src=ker.src_md5(name), codegen=digest, n=n, K=k, fused_steps=1,
                        steps=1, m=0, counts=list(counts), costs=costs, seconds=secs)
    from eagle._active_set import ActiveSet

    for n in (n_list[0], n_list[-1]):
        k = 4
        mask = rig.xp.zeros(n, dtype=rig.xp.bool_)
        cls8 = rig.host_to(rig.rng.integers(0, k, n).astype(np.uint8))
        aset = ActiveSet(mask, reorder=0.5, classes=k, class_plane=cls8)
        secs = []
        for _ in range(ctx["reps"] + 1):
            cls8[...] = rig.host_to(rig.rng.integers(0, k, n).astype(np.uint8))
            secs.append(rig.wall(aset.class_sort))
        w.case_rows("C", "sort", src="n/a", codegen="n/a", n=n, K=k, fused_steps=1, steps=1, m=0,
                    counts=[n], costs=[], seconds=secs, timer="steady_clock")


def _pair(ctx, ker, mod, group, ps, held_out, heavy, kinds, name="cal_pair"):
    rig, w, n = ctx["rig"], ctx["writer"], ctx["n_e"]
    units, _cls, digest_s = split_units(mod, name, ctx["targets"], ctx["clock"])
    fused, digest_f = fused_plugin(mod, name, ctx["targets"], ctx["clock"])
    costs = branch_costs(getattr(mod, name))
    for p in ps:
        k, counts = _classes_for(rig, n, 2, [1 - p, p]) if p > 0 else (
            np.zeros(n, dtype=np.int64), np.array([n, 0]))
        if "fused" in kinds:
            bound, _ = rig.bind(fused, n, k=rig.host_to(k))
            secs = rig.time(bound, n, ctx["reps"] + 1)
            w.case_rows(group, "fused", src=ker.src_md5(name), codegen=digest_f, n=n, K=2,
                        fused_steps=1, steps=1, m=heavy, counts=list(counts), costs=costs,
                        seconds=secs, held_out=held_out)
        if "split" in kinds:
            secs = split_step(rig, units, n, counts, ctx["reps"] + 1)
            w.case_rows(group, "split", src=ker.src_md5(name), codegen=digest_s, n=n, K=2,
                        fused_steps=1, steps=1, m=heavy, counts=list(counts), costs=costs,
                        seconds=secs, held_out=held_out)


def group_d_e(ctx, groups):
    heavy = ctx["heavy"]
    ker = Kernels(ctx["dir"])
    ker.add(dispatch_kernel("cal_pair", 2, heavy))
    if "E" in groups:
        ker.add(contrast_kernel("cal_contrast", heavy))
    mod = ker.load()
    if "D" in groups:
        _pair(ctx, ker, mod, "D", (0.0, 1 / 32, 1 / 4, 1.0), 0, heavy, ("fused",))
    if "E" in groups:
        _pair(ctx, ker, mod, "E", (0.01, 0.1, 0.5), 1, heavy, ("fused", "split"))
        _pair(ctx, ker, mod, "E", (0.1, 0.5), 1, heavy, ("fused", "split"), "cal_contrast")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", choices=("gpu", "cpu"), required=True)
    ap.add_argument("--quick", action="store_true", help="smoke pass: small sizes, 3 repetitions")
    ap.add_argument("--out", help="CSV file, or an existing directory to receive a named file")
    ap.add_argument("--reps", type=int, help="repetitions per case after the first (default 6; quick 2)")
    ap.add_argument("--seed", type=int, default=20261011)
    ap.add_argument("--groups", default="ABCDE", help="subset of the groups to run (default ABCDE)")
    args = ap.parse_args(argv)

    if args.target == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import eagle
    import hawk

    dev_index = int((os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0] or 0)) \
        if args.target == "gpu" else 0
    gpu = args.target == "gpu"
    q = args.quick
    ctx = {"reps": args.reps or (2 if q else 6), "targets": ("cuda",) if gpu else ("host",),
           "ms": (16, 64) if q else (16, 64, 256),
           "n_a": 2**14 if q else (2**22 if gpu else 2**20),
           "ns_b": (2**12, 2**14, 2**16) if q else (2**16, 2**20, 2**22 if gpu else 2**21),
           "ns_c": (2**12, 2**14) if q else (2**18, 2**22 if gpu else 2**20),
           "n_e": 2**14 if q else (2**22 if gpu else 2**20), "heavy": 32 if q else 64,
           "clock": [0.0], "slopes": {}, "rig": Rig(args.target, args.seed)}
    from hawk.compile import toolchain

    opt = os.environ.get("HAWK_OPT_LEVEL") or str(toolchain.DEFAULT_OPT_LEVEL)
    hdr = header(args, dev_index, hawk, eagle, opt)
    out = Path(args.out) if args.out else Path.cwd()
    if out.is_dir():
        stamp = hdr["date_utc"].replace(":", "").replace("-", "")
        out = out / f"calibration-{args.target}-{stamp}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[calibrate] target={args.target} mode={hdr['mode']} reps={ctx['reps'] + 1} out={out}", flush=True)
    walls = {}
    with tempfile.TemporaryDirectory(prefix="calibrate-") as tmp, open(out, "w", newline="") as fh:
        ctx["dir"] = Path(tmp)
        body = Path(tmp) / "rows.csv"
        with open(body, "w", newline="") as bf:
            ctx["writer"] = Writer(bf, args.target, ctx["reps"] + 1)
            t_all = time.perf_counter()
            steps = [("A", lambda: group_a(ctx)), ("B", lambda: group_b(ctx)), ("C", lambda: group_c(ctx))]
            de = [g for g in "DE" if g in args.groups and (g != "D" or gpu)]
            if de:
                steps.append(("DE", lambda: group_d_e(ctx, de)))
            for label, fn in steps:
                if label in args.groups or label == "DE":
                    c0, t0 = ctx["clock"][0], time.perf_counter()
                    fn()
                    walls[label] = (time.perf_counter() - t0, ctx["clock"][0] - c0)
                    print(f"[calibrate] group {label}: wall {walls[label][0]:.1f} s "
                          f"(compile {walls[label][1]:.1f} s)", flush=True)
            total = time.perf_counter() - t_all
            hdr["openmp_runtime"] = openmp_runtime()
            hdr["compile_s"] = f"{ctx['clock'][0]:.1f}"
            hdr["total_wall_s"] = f"{total:.1f}"
            hdr["rows"] = ctx["writer"].rows
        if gpu:
            hdr["smi_after"] = smi(dev_index)
        for key, val in hdr.items():
            fh.write(f"# {key}={val}\n")
        fh.write(body.read_text())
    for label, by_m in sorted(ctx["slopes"].items()):
        walls_m = [by_m[m] for m in sorted(by_m)]
        verdict = "monotone" if all(a < b for a, b in zip(walls_m, walls_m[1:])) else "NOT monotone"
        print(f"[calibrate] A {label:8s} median s by m: "
              + "  ".join(f"{m}:{by_m[m]:.2e}" for m in sorted(by_m)) + f"  ({verdict})", flush=True)
    print(f"[calibrate] total wall {hdr['total_wall_s']} s, compile {hdr['compile_s']} s, "
          f"rows {hdr['rows']}", flush=True)
    print(f"[calibrate] wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
