# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Helpers the performance cards share.

``perf_card/perf_card.py`` (RK4, GPU), ``perf_card/cpu_card.py`` (RK4, CPU),
``rk78_card/rk78_card.py`` (RKF7(8), GPU) and ``rk78_card/cpu_rk78_card.py``
(RKF7(8), CPU) each define a workload, its arms and the card they write; what
they have in common lives here: number formatting, the median/IQR statistic,
the ``# >>> code:NAME`` snippet markers, the device and software facts, the
memory probes' process protocol, the cold/warm compile-time loop and the
Nsight Systems kernel pass.

The cards import this file by path (``sys.path`` gets the ``benchmarks``
directory, which is not a package), so it never shadows a module of another
repository. Nothing here imports a GPU library at import time.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import re
import sqlite3
import statistics
import subprocess
import sys
import tempfile

# --------------------------------------------------------------------------- #
# Formatting (pure: the rendered tables depend on these exactly)
# --------------------------------------------------------------------------- #
DASH = "–"


def fmt_time(seconds):
    """Three significant digits in s, ms, µs or ns (the RK4 cards)."""
    if seconds is None:
        return DASH
    for scale, unit in ((1.0, "s"), (1e-3, "ms"), (1e-6, "µs")):
        if float(f"{seconds:.3g}") >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds * 1e9:.3g} ns"


def fmt_rate(value):
    """Scientific notation, three significant digits (the RK4 cards)."""
    if value is None:
        return DASH
    mantissa, exp = f"{value:.2e}".split("e")
    return f"{mantissa}e{int(exp)}"


def fmt_pct(fraction):
    return DASH if fraction is None else f"{100.0 * fraction:.2g} %"


def fmt_time_si(seconds):
    """Three significant digits in s, ms or µs (the RK7(8) cards)."""
    if seconds is None:
        return DASH
    if seconds >= 1:
        return f"{seconds:.3g} s"
    if seconds >= 1e-3:
        return f"{seconds * 1e3:.3g} ms"
    return f"{seconds * 1e6:.3g} µs"


def fmt_rate_si(value):
    """Three significant digits with a k/M/G/T prefix (the RK7(8) cards)."""
    if value is None:
        return DASH
    for div, unit in ((1e12, "T"), (1e9, "G"), (1e6, "M"), (1e3, "k")):
        if value >= div:
            return f"{value / div:.3g} {unit}"
    return f"{value:.3g}"


def fmt_pct_si(fraction):
    return DASH if fraction is None else f"{100 * fraction:.2g}%"


def fmt_mib(nbytes):
    return DASH if nbytes is None else f"{nbytes / 2**20:.3g} MiB"


def fmt_x(value):
    return DASH if value is None else f"{value:.3g}×"


# --------------------------------------------------------------------------- #
# Statistics and small facts
# --------------------------------------------------------------------------- #
def median_iqr(values):
    """Median and interquartile range (inclusive quartiles), with the samples."""
    q = statistics.quantiles(values, n=4, method="inclusive")
    return {"median": statistics.median(values), "iqr": q[2] - q[0],
            "q1": q[0], "q3": q[2], "samples": list(values)}


def trimmed(values):
    """The GPU card's statistic over its timed runs: ``mean`` of the runs minus
    the single highest and the single lowest (12 runs -> the middle 10), with
    ``lo``/``hi`` the range of those kept runs; ``median`` and ``iqr``
    (inclusive quartiles) are over all the runs, ``samples`` is every run."""
    vals = sorted(values)
    if len(vals) < 3:
        raise ValueError("trimmed needs at least 3 values")
    kept = vals[1:-1]
    q = statistics.quantiles(vals, n=4, method="inclusive")
    return {"mean": statistics.fmean(kept), "lo": kept[0], "hi": kept[-1],
            "median": statistics.median(vals), "iqr": q[2] - q[0],
            "samples": list(values)}


def center(stat):
    """The one number a card's wall statistic stands for: the trimmed mean
    when the card carries one, else the median (CPU and RK7(8) cards)."""
    return stat["mean"] if "mean" in stat else stat["median"]


def md5(path):
    return hashlib.md5(pathlib.Path(path).read_bytes()).hexdigest()


def version_of(module):
    try:
        return __import__(module).__version__
    except Exception:  # not installed: the arm cannot have run
        return None


def which(name):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        p = pathlib.Path(d) / name
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return None


def tool_name(path):
    """The bare executable name (``g++``), never the local absolute path a
    discovery function (``which``, ``hawk.compile.host_compiler``) resolved
    it to -- a committed card must not leak the build box's directory layout
    (conda env path, username) the way a raw discovery path does. Callers
    that record a tool's identity in a card dict call this instead of
    storing the resolved path verbatim."""
    return pathlib.Path(path).name if path else path


def find_absolute_paths(value, _key=""):
    """[(dotted key path, value), ...] for every string anywhere inside
    `value` (recursively, through dicts/lists/tuples) that is an absolute
    filesystem path -- the generic half of the "no absolute path in a card"
    gate, so a future field that starts recording one has the same net
    `tool_name` gives the compiler field."""
    hits = []
    if isinstance(value, str):
        # a "||"-joined identity ("path||version||digest") is checked part by part
        if any(os.path.isabs(part) for part in value.split("||")):
            hits.append((_key or "<root>", value))
    elif isinstance(value, dict):
        for k, v in value.items():
            hits.extend(find_absolute_paths(v, f"{_key}.{k}" if _key else str(k)))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            hits.extend(find_absolute_paths(v, f"{_key}[{i}]"))
    return hits


def cpu_model():
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def git_commit(module, short=False):
    """``<commit>[-dirty]`` of the repository holding ``module``, or None."""
    root = pathlib.Path(module.__file__).resolve().parent
    try:
        rev = subprocess.run(["git", "-C", str(root), "rev-parse",
                              *(["--short"] if short else []), "HEAD"],
                             capture_output=True, text=True, timeout=30)
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain",
                                "--untracked-files=no"],
                               capture_output=True, text=True, timeout=30)
        if rev.returncode == 0:
            return rev.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def software(modules=("numpy", "cupy", "torch", "jax", "jaxlib", "warp")):
    """Versions and commits of the GPU cards' software."""
    import eagle
    import hawk

    return {"python": platform.python_version(),
            **{m: version_of(m) for m in modules},
            "hawk": getattr(hawk, "__version__", None),
            "eagle_commit": git_commit(eagle), "hawk_commit": git_commit(hawk),
            "platform": platform.platform(), "cpu": cpu_model(),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}


def verify_xvk(outs, ref_arm, nstop, atol):
    """The RK4 cards' agreement check, before any timing counts: every arm's
    ``(x, v, k)`` reaches exactly the stop steps, and its x and v agree with
    ``ref_arm``'s within ``atol``; returns {arm: its largest difference}."""
    import numpy as np

    ref = outs[ref_arm]
    worst = {}
    for arm, (x, v, k) in outs.items():
        assert np.array_equal(k, nstop), f"{arm}: step counts differ from the stop steps"
        d = max(float(np.max(np.abs(x - ref[0]))), float(np.max(np.abs(v - ref[1]))))
        assert d <= atol, f"{arm}: state differs from {ref_arm} by {d:.3e} > {atol:.1e}"
        worst[arm] = d
    return worst


# --------------------------------------------------------------------------- #
# Host memory (Linux /proc)
# --------------------------------------------------------------------------- #
def status_kib(field):
    m = re.search(rf"^{field}:\s+(\d+)", pathlib.Path("/proc/self/status").read_text(), re.M)
    return int(m.group(1)) if m else 0


def reset_host_peak():
    """Reset the process's peak-RSS mark (VmHWM, and with it ru_maxrss) to
    the current RSS (Linux ``/proc/self/clear_refs`` value 5)."""
    try:
        pathlib.Path("/proc/self/clear_refs").write_text("5")
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# The code snippets: ``# >>> code:NAME`` ... ``# <<< code:NAME`` blocks
# --------------------------------------------------------------------------- #
_CODE_MARK = re.compile(r"^\s*# (>>>|<<<) code:(\S+)\s*$")


def code_spans(path):
    """{name: {"first_line", "last_line", "lines"}} of every marked block of
    ``path`` (lines = the code between the markers)."""
    out, open_ = {}, {}
    for i, line in enumerate(pathlib.Path(path).read_text().splitlines(), 1):
        m = _CODE_MARK.match(line)
        if not m:
            continue
        kind, name = m.groups()
        if kind == ">>>":
            assert name not in open_ and name not in out, f"code block {name} opened twice"
            open_[name] = i
        else:
            start = open_.pop(name)
            out[name] = {"first_line": start + 1, "last_line": i - 1, "lines": i - start - 1}
    assert not open_, f"unclosed code blocks: {sorted(open_)}"
    return out


def code_text(path):
    """{name: source lines} of every marked block of ``path`` (markers excluded)."""
    text = pathlib.Path(path).read_text().splitlines()
    return {name: text[s["first_line"] - 1:s["last_line"]]
            for name, s in code_spans(path).items()}


def count_code_lines(lines):
    """Non-blank, non-comment lines."""
    return sum(1 for ln in lines if ln.strip() and not ln.strip().startswith("#"))


def arm_code(blocks, arm_blocks, arms):
    """{"blocks": every marked block, "arms": {arm: its blocks and their total
    line count}} -- the RK4 cards' record of each arm's snippet."""
    return {"blocks": blocks,
            "arms": {a: {"blocks": list(arm_blocks[a]),
                         "lines": sum(blocks[b]["lines"] for b in arm_blocks[a])}
                     for a in arms}}


# --------------------------------------------------------------------------- #
# Rendering pieces several cards share
# --------------------------------------------------------------------------- #
def fastest_lines(rows, labels, fmt=fmt_time):
    """The "Fastest arm by median wall time" list of one table's rows."""
    trimmed_stat = any("mean" in r["wall_s"] for r in rows)
    lines = ["Fastest arm by mean wall time (middle 10 of 12 runs):" if trimmed_stat
             else "Fastest arm by median wall time:", ""]
    for n in sorted({r["n"] for r in rows}):
        at_n = sorted((r for r in rows if r["n"] == n), key=lambda r: center(r["wall_s"]))
        best, second = at_n[0], at_n[1]
        ratio = center(second["wall_s"]) / center(best["wall_s"])
        lines.append(
            f"- N = {n:,}: {labels[best['arm']]} ({fmt(center(best['wall_s']))}), then "
            f"{labels[second['arm']]} ({fmt(center(second['wall_s']))}, "
            f"{ratio:.3g}× the time)")
    return lines


def fit_section(card, labels):
    """The "Where each tool fits" section (empty when the card has no notes)."""
    fit = card.get("fit")
    if not fit:
        return []
    lines = ["### Where each tool fits", "", fit["intro"], ""]
    lines += [f"- **{labels[arm]}.** {fit['arms'][arm]}" for arm in card["arms"]]
    return lines + [""]


def compile_table(comp, labels, fmt, trailer):
    """The GPU cards' "Compile time" section (rows naming several arms)."""
    if not comp.get("rows"):
        return []
    lines = ["### Compile time", "", comp["method"], "",
             "| arm | compiled | cold (median of 3) | warm (median of 3) |",
             "|---|---|---:|---:|"]
    for r in comp["rows"]:
        lines.append(f"| {', '.join(labels[a] for a in r['arms'])} | {r['what']} "
                     f"| {fmt(r['cold_s']['median'])} | {fmt(r['warm_s']['median'])} |")
    return lines + ["", "No compile step: " + ", ".join(labels[a] for a in comp["none"])
                    + trailer, ""]


def wins(card):
    """({arm: cells where it is fastest}, number of cells) over the card's
    (distribution, max_steps, n) cells."""
    cells = {}
    for r in card["results"]:
        cells.setdefault((r["distribution"], r["max_steps"], r["n"]), []).append(r)
    out = {a: 0 for a in card["arms"]}
    for rs in cells.values():
        out[min(rs, key=lambda r: center(r["wall_s"]))["arm"]] += 1
    return out, len(cells)


# --------------------------------------------------------------------------- #
# GPU state and facts
# --------------------------------------------------------------------------- #
def sync():
    import cupy as cp

    cp.cuda.Device().synchronize()


def torch_empty_cache():
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


def trim_pools():
    """Return every caching allocator's free blocks to the driver (Warp's
    stream-ordered pool: release threshold 0, so it keeps nothing it does not
    use past a synchronisation)."""
    if "warp" in sys.modules:
        wp = sys.modules["warp"]
        wp.set_mempool_release_threshold("cuda:0", 0)
        wp.synchronize()
    if "cupy" in sys.modules:
        cp = sys.modules["cupy"]
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    torch_empty_cache()


def probe_sync():
    """Synchronise every GPU library this process has imported."""
    if "cupy" in sys.modules:
        sys.modules["cupy"].cuda.Device().synchronize()
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def process_device_bytes(bus_id):
    """(bytes, source): the device memory the driver accounts to THIS process
    on the timed GPU (NVML's per-process used memory), or, where the driver
    does not report per process, the device's whole used memory."""
    import pynvml

    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByPciBusId(bus_id)
        pid = os.getpid()
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h):
            if p.pid == pid and p.usedGpuMemory is not None:
                return int(p.usedGpuMemory), "nvml per-process used memory"
        return int(pynvml.nvmlDeviceGetMemoryInfo(h).used), "nvml device used memory"
    finally:
        pynvml.nvmlShutdown()


GPU_STATE_FIELDS = ("clocks.sm,clocks.mem,clocks.max.sm,temperature.gpu,"
                    "utilization.gpu,power.draw,pstate")


def gpu_state(bus_id):
    """Clocks, temperature and utilisation of the timed GPU (nvidia-smi)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--id={bus_id}", f"--query-gpu={GPU_STATE_FIELDS}",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return dict(zip(GPU_STATE_FIELDS.split(","), [s.strip() for s in out.split(",")]))


def _nvml_handle(bus_id):
    """An NVML device handle for ``bus_id``, or None (NVML unavailable or the
    bus id does not resolve) -- the caller falls back to nvidia-smi."""
    try:
        import pynvml

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByPciBusId(bus_id)
    except Exception:  # pynvml missing, no driver, bad bus id, ...
        return None


def nvml_sm_clock_mhz(bus_id):
    """One-shot SM clock reading in MHz through NVML (``nvmlDeviceGetClockInfo``),
    or None when NVML is not usable here."""
    try:
        import pynvml

        h = _nvml_handle(bus_id)
        if h is None:
            return None
        return float(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
    except Exception:
        return None


class ClockPoller:
    """Samples the timed GPU's SM clock and temperature while the
    repetitions run, so the card records the clock the numbers were measured
    at, not an idle reading. Prefers NVML (``nvml_site`` on PYTHONPATH): an
    in-process call cheap enough to sample every 0.2 s, which still gets
    several samples inside a short window (N=64, S=100 timed cells can run
    under a second, where a 1 Hz nvidia-smi subprocess poll often gets none).
    Falls back to polling nvidia-smi once a second when NVML is not usable in
    this process; the returned dict's "source" field names which one ran."""

    def __init__(self, bus_id):
        import threading

        self._bus_id = bus_id
        self._handle = _nvml_handle(bus_id)
        self._samples = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def _poll_nvml(self):
        import pynvml

        while not self._stop.wait(0.2):
            try:
                sm = pynvml.nvmlDeviceGetClockInfo(self._handle, pynvml.NVML_CLOCK_SM)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                temp = pynvml.nvmlDeviceGetTemperature(
                    self._handle, pynvml.NVML_TEMPERATURE_GPU)
                self._samples.append({"clocks.sm": sm, "utilization.gpu": util,
                                      "temperature.gpu": temp})
            except Exception:
                pass

    def _poll(self):
        if self._handle is not None:
            self._poll_nvml()
            return
        while not self._stop.wait(1.0):
            st = gpu_state(self._bus_id)
            if st is not None:
                self._samples.append(st)

    def stop(self):
        self._stop.set()
        self._thread.join()

        def col(key):
            vals = []
            for st in self._samples:
                try:
                    vals.append(float(st[key]))
                except (KeyError, ValueError, TypeError):
                    pass
            return vals

        sm, temp, util = col("clocks.sm"), col("temperature.gpu"), col("utilization.gpu")
        source = ("NVML nvmlDeviceGetClockInfo(NVML_CLOCK_SM), sampled every 0.2 s"
                   if self._handle is not None else
                   "nvidia-smi clocks.sm, sampled every 1 s (NVML unavailable)")
        if not sm:
            return {"samples": 0, "source": source}
        return {"samples": len(sm), "sm_clock_mhz_min": min(sm),
                "sm_clock_mhz_median": statistics.median(sm),
                "sm_clock_mhz_max": max(sm),
                "temperature_c_max": max(temp) if temp else None,
                "utilization_pct_median": statistics.median(util) if util else None,
                "source": source}


def load_avg_1m():
    """The 1-minute load average (``os.getloadavg()``), or None on a platform
    that does not support it."""
    try:
        return os.getloadavg()[0]
    except OSError:
        return None


def toolchain():
    """What built and ran the card's kernels: the host C++ compiler hawk
    resolves (bare name, version, identity digest), nvcc, the Python, and the
    name of the environment file the run was started from (``$RAPTOR_ENV_FILE``, None
    when the run was not started from one). A number whose compiler is
    unknown cannot be reproduced, so every card records this."""
    out = {"env_file": tool_name(os.environ.get("RAPTOR_ENV_FILE")),
           "python": platform.python_version(),
           "host_compiler": None, "nvcc": None}
    try:
        from hawk.compile import toolchain as tc

        # the identity is "path||version||digest": keep the bare name
        path, sep, rest = tc.compiler_identity(tc.host_compiler()).partition("||")
        out["host_compiler"] = tool_name(path) + sep + rest
    except Exception as exc:  # the card still runs; the gap is recorded
        out["host_compiler"] = f"unavailable ({type(exc).__name__})"
    try:
        r = subprocess.run(["nvcc", "--version"], capture_output=True, text=True,
                           timeout=30)
        out["nvcc"] = r.stdout.strip().splitlines()[-1] if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        pass
    return out


#: SASS opcodes counted as a DP (double-precision) issue slot: fused
#: multiply-add, multiply, add, the predicate-setting compare the step/error
#: control uses, and the multi-function unit (reciprocal/sqrt family) --
#: the same opcode family cuobjdump prints for compute capability 6.x-9.x.
SASS_DP_OPCODES = ("DFMA", "DMUL", "DADD", "DSETP", "MUFU")
_SASS_OPCODE_RE = re.compile(
    r"\b(" + "|".join(SASS_DP_OPCODES) + r")(?:\.[A-Z0-9]+)*\b")


def sass_dp_instruction_count(code_path, arch):
    """Static DP-instruction count (DFMA/DMUL/DADD/DSETP/MUFU) of
    ``code_path`` -- a ``.cubin`` hawk/nvcc already produced is dumped as
    is; a ``.ptx`` is compiled for ``arch`` (e.g. "sm_61", the device's own
    compute capability) with the CUDA toolkit's own ``ptxas`` first, so the
    count is the SASS ptxas actually selects for this device, not a hand
    count of the hawk-emitted source. Returns ``{"count", "method"}`` or
    None when the toolchain (ptxas/cuobjdump) is not on PATH, no ``.cubin``/
    ``.ptx`` is given, or the compile/dump fails; never raises."""
    cuobjdump = which("cuobjdump")
    if cuobjdump is None:
        return None
    code_path = pathlib.Path(code_path)
    try:
        if code_path.suffix == ".cubin":
            cubin, built = code_path, "the cubin hawk's build already produced"
        elif code_path.suffix == ".ptx":
            ptxas = which("ptxas")
            if ptxas is None:
                return None
            tmp = tempfile.TemporaryDirectory(prefix="sass_count_")
            cubin = pathlib.Path(tmp.name) / "k.cubin"
            out = subprocess.run([ptxas, f"-arch={arch}", "-O3", str(code_path),
                                  "-o", str(cubin)], capture_output=True, text=True,
                                 timeout=120)
            if out.returncode != 0:
                return None
            built = f"ptxas -arch={arch} -O3 of the hawk-emitted PTX"
        else:
            return None
        dump = subprocess.run([cuobjdump, "--dump-sass", str(cubin)],
                              capture_output=True, text=True, timeout=120)
        if dump.returncode != 0:
            return None
    except (OSError, subprocess.SubprocessError):
        return None
    count = len(_SASS_OPCODE_RE.findall(dump.stdout))
    return {"count": count, "method": (
        f"cuobjdump --dump-sass of {built}; occurrences of "
        + "/".join(SASS_DP_OPCODES) + " (every opcode modifier, e.g. DFMA.RZ, "
        "counted once) over the whole compiled module")}


def host_hot_loop_fp_simd_counts(binary_path, symbol_substr):
    """Packed vs scalar double-precision FP instruction counts of the
    function(s) whose disassembly symbol contains ``symbol_substr`` in
    ``binary_path`` (``objdump -d``) -- the EXECUTED hot loop, not a
    whole-file vectorisation count (a whole-.so count can say "vectorised"
    while the hot loop runs the scalar remainder). Returns
    ``{"packed", "scalar", "method"}`` or None when ``objdump`` is missing,
    the binary cannot be disassembled, or no matching symbol is found."""
    objdump = which("objdump")
    if objdump is None or not pathlib.Path(binary_path).is_file():
        return None
    try:
        out = subprocess.run([objdump, "-d", "--demangle", str(binary_path)],
                             capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    lines = out.stdout.splitlines()
    blocks, cur = [], None
    for line in lines:
        if line.endswith(">:") and symbol_substr in line:
            cur = []
            blocks.append(cur)
        elif cur is not None and line and not line[:1].isspace() and ">:" in line:
            cur = None  # a new, unrelated symbol starts
        elif cur is not None:
            cur.append(line)
    if not blocks:
        return None
    body = "\n".join(line for block in blocks for line in block)
    packed = len(re.findall(r"\b(?:v?(?:fmadd|fmsub|mul|add|sub)[a-z]*pd)\b", body))
    scalar = len(re.findall(r"\b(?:v?(?:fmadd|fmsub|mul|add|sub)[a-z]*sd)\b", body))
    return {"packed": packed, "scalar": scalar, "method": (
        f"objdump -d --demangle; vfmadd*/vmul*/vadd*/vsub* with the pd "
        "(packed double) vs sd (scalar double) suffix, counted only inside "
        f"the symbol(s) matching '{symbol_substr}'")}


#: 64-bit floating-point fused multiply-add results per clock per SM, by
#: compute capability. Source: CUDA C++ Programming Guide, "Arithmetic
#: Instructions" throughput table (64-bit floating-point add, multiply,
#: multiply-add).
FP64_FMA_PER_CLK_PER_SM = {(6, 0): 32, (6, 1): 4, (6, 2): 4, (7, 0): 32,
                           (7, 5): 2, (8, 0): 32, (8, 6): 2, (8, 9): 2,
                           (9, 0): 64}


def _device_clock_khz(cp, dev_id, props, key, attr):
    """A clock in kHz: the device property when this runtime still reports it
    (CUDA 12), else ``cudaDeviceGetAttribute`` (the only route in CUDA 13,
    which removed ``clockRate`` and ``memoryClockRate`` from the properties)."""
    if key in props:
        return props[key]
    return cp.cuda.runtime.deviceGetAttribute(attr, dev_id)


def device_facts():
    """(device, peak, slug, PCI bus id) of the current CUDA device."""
    import cupy as cp

    dev = cp.cuda.Device()
    p = cp.cuda.runtime.getDeviceProperties(dev.id)
    name = p["name"].decode() if isinstance(p["name"], bytes) else p["name"]
    cc = (p["major"], p["minor"])
    # cudaDevAttrClockRate = 13, cudaDevAttrMemoryClockRate = 36
    clock_khz = _device_clock_khz(cp, dev.id, p, "clockRate", 13)
    mem_clock_khz = _device_clock_khz(cp, dev.id, p, "memoryClockRate", 36)
    clock_hz = clock_khz * 1e3
    fma = FP64_FMA_PER_CLK_PER_SM.get(cc)
    peak_fp64 = None if fma is None else p["multiProcessorCount"] * fma * 2 * clock_hz
    peak_bw = 2 * mem_clock_khz * 1e3 * p["memoryBusWidth"] / 8
    bus_id = f"{p['pciDomainID']:08x}:{p['pciBusID']:02x}:{p['pciDeviceID']:02x}.0"
    smi = {}
    try:
        line = subprocess.run(
            ["nvidia-smi", f"--id={bus_id}",
             "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,"
             "memory.total,power.limit", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        keys = ("name", "driver_version", "clocks_max_sm", "clocks_max_mem",
                "memory_total", "power_limit")
        smi = dict(zip(keys, [s.strip() for s in line.split(",")]))
    except (OSError, subprocess.SubprocessError):
        pass
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    device = {
        "name": name, "compute_capability": f"{cc[0]}.{cc[1]}",
        "sm_count": p["multiProcessorCount"], "clock_rate_hz": clock_hz,
        "memory_clock_hz": mem_clock_khz * 1e3,
        "memory_bus_width_bits": p["memoryBusWidth"],
        "total_memory_bytes": p["totalGlobalMem"], "pci_bus_id": bus_id,
        "driver_version": smi.get("driver_version"),
        "cuda_driver_api": cp.cuda.runtime.driverGetVersion(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "nvidia_smi": smi,
    }
    peak = {
        "fp64_flops": peak_fp64,
        "fp64_source": (
            "SM count x FP64 FMA per clock per SM (CUDA C++ Programming Guide, "
            "Arithmetic Instructions throughput table, compute capability "
            f"{cc[0]}.{cc[1]}: {fma}) x 2 FLOP per FMA x the device's reported "
            "maximum clock (cudaDeviceProp::clockRate)"),
        "dram_bytes_per_s": peak_bw,
        "dram_source": (
            "2 x memory clock (cudaDeviceProp::memoryClockRate) x bus width "
            "(cudaDeviceProp::memoryBusWidth) / 8"),
    }
    if name == "Quadro P2000":
        peak["datasheet_cross_check"] = (
            "Quadro P2000 data sheet: 3.0 TFLOPS peak FP32 (FP64 is 1/32 of "
            "FP32 on compute capability 6.1), 140 GB/s memory bandwidth")
    return device, peak, slug, bus_id


def warp_init():
    """Import and initialise Warp once, without its banner."""
    import warp as wp

    wp.config.quiet = True
    wp.init()
    return wp


def jax_persistent_cache(directory):
    """Point JAX's persistent compilation cache at ``directory``, caching every
    executable (minimum compile time and entry size 0)."""
    import jax

    jax.config.update("jax_compilation_cache_dir", directory)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


# --------------------------------------------------------------------------- #
# Probe processes: the memory and compile-time passes run one fresh process
# per measurement; it replies with one JSON line on the original stdout
# --------------------------------------------------------------------------- #
def probe_reply_fd():
    """Keep the process's stdout for the reply and send everything a library
    prints there to stderr; returns the reply descriptor."""
    fd = os.dup(1)
    os.dup2(2, 1)
    return fd


def probe_reply(fd, obj):
    os.write(fd, (json.dumps(obj) + "\n").encode())


def run_probe(script, args, *, env=None, timeout=600, what="probe"):
    """Run ``script args`` in a fresh interpreter; its last stdout line, parsed."""
    out = subprocess.run([sys.executable, str(pathlib.Path(script).resolve()), *args],
                         capture_output=True, text=True, env=env, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"{what}:\n{out.stderr[-4000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def cold_warm(script, args, cache_env, *, reps=3, prefix="card_cache_", base_env=None,
              timeout=600, what="compile probe"):
    """``reps`` cold processes (each on fresh, empty cache directories) and
    ``reps`` warm ones (on the first cold run's populated caches) of
    ``script args``; each replies ``{"seconds": ...}``. ``cache_env`` maps an
    environment variable to the sub-directory it names. Returns (cold, warm)."""
    import shutil

    cold, warm, dirs = [], [], []
    try:
        for phase in ("cold",) * reps + ("warm",) * reps:
            if phase == "cold":
                dirs.append(pathlib.Path(tempfile.mkdtemp(prefix=prefix)))
            cache = dirs[0] if phase == "warm" else dirs[-1]
            env = dict(os.environ if base_env is None else base_env)
            env.update({k: str(cache / v) for k, v in cache_env.items()})
            sec = run_probe(script, args, env=env, timeout=timeout, what=what)["seconds"]
            (cold if phase == "cold" else warm).append(sec)
    finally:
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
    return cold, warm


class LineWorker:
    """A persistent worker process of a CPU card: ``script args`` under
    ``env``, one JSON command per stdin line, one JSON reply per stdout line
    (a reply carrying ``"error"`` raises)."""

    def __init__(self, script, args, env, name):
        self.name = name
        self.p = subprocess.Popen([sys.executable, str(pathlib.Path(script).resolve()),
                                   *args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  text=True, env=env)

    def call(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError(f"worker {self.name} exited (rc={self.p.poll()}) on {obj}")
        out = json.loads(line)
        if "error" in out:
            raise RuntimeError(f"worker {self.name}:\n{out['error']}")
        return out

    def close(self, quit_obj):
        try:
            self.p.stdin.write(json.dumps(quit_obj) + "\n")
            self.p.stdin.close()
            self.p.wait(timeout=60)
        except (OSError, subprocess.SubprocessError):
            self.p.kill()


# --------------------------------------------------------------------------- #
# Kernel-only times: one script run traced by Nsight Systems
# --------------------------------------------------------------------------- #
def nsys_ranges(nsys, rep, script, args, prefix, timeout=None):
    """Trace ``script args`` under nsys (CUDA + NVTX, graph nodes traced one
    by one) and return ``[(range text, kernel count, summed kernel seconds,
    memcpy count, runtime-API call count)]`` for every NVTX range whose text
    starts with ``prefix`` -- the kernel-launch/memcpy/graph-node counts come
    from the same trace the kernel-only pass already runs, no second pass.
    The report files are removed. Raises on a failed trace or export."""
    rep = pathlib.Path(rep)
    cmd = [nsys, "profile", "--trace=cuda,nvtx", "--cuda-graph-trace=node",
           "--sample=none", "--cpuctxsw=none", "-f", "true", "-o", str(rep),
           sys.executable, str(pathlib.Path(script).resolve()), *args]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=timeout)
    db = rep.with_suffix(".sqlite")
    subprocess.run([nsys, "export", "--type", "sqlite", "-f", "true", "-o", str(db),
                    str(rep) + ".nsys-rep"], check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.DEVNULL)
    con = sqlite3.connect(db)
    try:
        tables = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        ranges = con.execute(
            "SELECT n.start, n.end, COALESCE(n.text, s.value) FROM NVTX_EVENTS n "
            "LEFT JOIN StringIds s ON n.textId = s.id WHERE n.end IS NOT NULL").fetchall()
        out = []
        for start, end, text in ranges:
            if not text or not text.startswith(prefix):
                continue
            count, total = con.execute(
                "SELECT COUNT(*), COALESCE(SUM(end - start), 0) "
                "FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ? AND start < ?",
                (start, end)).fetchone()
            memcpy = None
            if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables:
                (memcpy,) = con.execute(
                    "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_MEMCPY "
                    "WHERE start >= ? AND start < ?", (start, end)).fetchone()
            api = None
            if "CUPTI_ACTIVITY_KIND_RUNTIME" in tables:
                (api,) = con.execute(
                    "SELECT COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME "
                    "WHERE start >= ? AND start < ?", (start, end)).fetchone()
            out.append((text, count, total * 1e-9, memcpy, api))
    finally:
        con.close()
    db.unlink()
    pathlib.Path(str(rep) + ".nsys-rep").unlink()
    return out


def nsys_version(nsys):
    return subprocess.run([nsys, "--version"], capture_output=True, text=True).stdout.strip()
