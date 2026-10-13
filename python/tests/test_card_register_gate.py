# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The card kernels' device code, held to its register and instruction budget.

The persistent entry of an automatic kernel is the large-batch entry, and its
residency is set by its register count (per sub-partition, in 256-register
units: 168 registers hold 12 warps per SM, 40 hold 48). hawk keeps that
entry's own bookkeeping lean so it lands on those steps without a register
cap and without spilling. Compile-only (no GPU): each card kernel's CUDA unit
is built as ``eagle.deploy`` builds it, for every arch below the toolkit
supports, and ``ptxas -v`` plus the SASS are read back:

* ``rkf78_attempt_persist`` and the oscillator's
  ``oscillator_step_xauto_persist`` use at most their pinned per-arch counts
  (per CUDA major), each on or below its residency step;
* the persist entry is warp-converged: its source holds ONE
  ``atomicAdd(hawk_next`` (the chunk head's warp-wide claim) and its
  unrolled step loop no atomic and no warp collective;
* no entry spills, keeps a stack frame, or touches local memory;
* every entry's floating-point arithmetic (DFMA/DMUL/DADD/MUFU) is the
  pinned count: the bookkeeping changes, the maths never does;
* the fused RKF7(8) entries stay within 2 registers of their pinned counts
  (no cap reaches them spill-free, so they are left as they are);
* the oscillator's persist entry stays within 1000 instructions (its step
  loop is unrolled by 4, and growth there should be noticed).

Skipped where ``nvcc``/``cuobjdump`` or hawk are absent.
"""

from __future__ import annotations

import collections
import importlib.util
import pathlib
import re
import shutil
import subprocess
import sys

import pytest

pytestmark = pytest.mark.repo_local

_BENCH = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"
ARCHS = ("sm_61", "sm_80", "sm_89", "sm_90")
ARITH = ("DFMA", "DMUL", "DADD", "MUFU")

#: Register pins by CUDA major, then entry, then arch: register counts differ
#: between toolkits, so each is held to the pins of the toolkit that compiled it
#: (CUDA 12.9 pins cover the 12.x series). An arch missing under a major has no
#: pin there and is not checked. Persist entries: the most registers allowed
#: (the warp-converged loop holds no more than the per-lane loop it replaced).
PERSIST_REGS = {
    12: {
        "rkf78_attempt_persist": {"sm_61": 166, "sm_80": 164, "sm_89": 164, "sm_90": 164},
        "oscillator_step_xauto_persist": {"sm_61": 32, "sm_80": 30, "sm_89": 35, "sm_90": 32},
    },
    13: {
        "rkf78_attempt_persist": {"sm_80": 164, "sm_89": 164, "sm_90": 164},
        "oscillator_step_xauto_persist": {"sm_80": 30, "sm_89": 35, "sm_90": 32},
    },
}

#: Pinned register counts of the fused RKF7(8) entries, +-2 allowed, by major.
FUSED_REGS = {
    12: {
        "rkf78_attempt": {"sm_61": 188, "sm_80": 182, "sm_89": 182, "sm_90": 184},
        "rkf78_attempt_range": {"sm_61": 186, "sm_80": 184, "sm_89": 184, "sm_90": 186},
    },
    13: {
        "rkf78_attempt": {"sm_80": 182, "sm_89": 182, "sm_90": 184},
        "rkf78_attempt_range": {"sm_80": 184, "sm_89": 184, "sm_90": 186},
    },
}

#: Pinned arithmetic per entry by CUDA major, the same on every arch. The
#: toolkits differ in how many DADDs the RKF7(8) entries carry.
# The step and stop counters are uint32 planes: their increments are integer
# adds, so each entry carries fewer DADDs than the float64 counters did, and
# the range entry's loop unrolls one more trip (six copies of the fused step).
_OSC_ARITH = {
    "oscillator_step_xauto": (32, 16, 6, 0),
    "oscillator_step_xauto_range": (96, 32, 14, 0),
    # four unrolled trips, no peeled one: the step's maths is the fused one
    "oscillator_step_xauto_persist": (64, 20, 10, 0),
}
ARITH_COUNTS = {
    12: {
        **_OSC_ARITH,
        "rkf78_attempt": (1247, 400, 79, 69),
        "rkf78_attempt_range": (1247, 400, 79, 69),
        "rkf78_attempt_persist": (659, 210, 56, 37),
    },
    13: {
        **_OSC_ARITH,
        "rkf78_attempt": (1247, 400, 81, 69),
        "rkf78_attempt_range": (1247, 400, 81, 69),
        "rkf78_attempt_persist": (659, 210, 57, 37),
    },
}

OSC_PERSIST_MAX_INSTRUCTIONS = 1000


def _load(name, rel):
    if str(_BENCH) not in sys.path:
        sys.path.insert(0, str(_BENCH))
    spec = importlib.util.spec_from_file_location(name, _BENCH / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _units():
    """``{label: CUDA source}`` of the two card kernels, as deployed."""
    from hawk.emit import BACKENDS, render_source

    perf = _load("eagle_perf_card_gate", "perf_card/perf_card.py")
    rk78 = _load("eagle_rk78_card_gate", "rk78_card/rk78_card.py")
    out = {}
    for label, k in (("osc", perf._define_persist_kernel()), ("rk78", rk78._define_kernel())):
        out[label] = render_source(k.name, k.sinks, k.walk, BACKENDS["cuda"],
                                   kind=k.kind, one_step=k.one_step).text
    return out


def _compile(source, arch, tmp):
    """``{entry: {regs, stack, spill, n, ops}}`` of one unit on one arch."""
    from hawk.compile import toolchain as tc
    from hawk.emit import FAST_ENTRIES_MACRO

    cu, cubin = tmp / f"unit_{arch}.cu", tmp / f"unit_{arch}.cubin"
    cu.write_text(source)
    flags = tc.device_flags(arch=arch, target="cubin",
                            extra=(f"-D{FAST_ENTRIES_MACRO}=1", "-Xptxas", "-v"))
    run = subprocess.run(["nvcc", *flags, str(cu), "-o", str(cubin)],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr[-2000:]
    entries, cur = {}, None
    for line in run.stderr.splitlines():
        m = re.search(r"Compiling entry function '([^']+)'", line)
        if m:
            cur = entries.setdefault(m.group(1), {"ops": collections.Counter(), "n": 0})
            continue
        m = re.search(r"Used (\d+) registers", line)
        if m and cur is not None:
            cur["regs"] = int(m.group(1))
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads",
                      line)
        if m and cur is not None:
            cur["stack"], cur["spill"] = int(m.group(1)), int(m.group(2)) + int(m.group(3))
    sass = subprocess.run(["cuobjdump", "-sass", str(cubin)], capture_output=True,
                          text=True, check=True).stdout
    cur = None
    for line in sass.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = entries[m.group(1)]
            continue
        m = re.match(r"^\s*/\*[0-9a-f]{4,6}\*/\s+(?:@!?U?P[T0-9]+\s+)?([A-Z][A-Z0-9_]*)", line)
        if m and cur is not None:
            cur["n"] += 1
            cur["ops"][m.group(1)] += 1
    return entries


def _nvcc_major():
    """The CUDA major version of the ``nvcc`` on PATH (``None`` if unreadable)."""
    out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout
    m = re.search(r"release (\d+)\.", out)
    return int(m.group(1)) if m else None


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    pytest.importorskip("hawk")
    if not (shutil.which("nvcc") and shutil.which("cuobjdump")):
        pytest.skip("no CUDA toolkit (nvcc, cuobjdump) on PATH")
    from hawk.compile import toolchain as tc

    toolkit = tc.toolkit_archs("nvcc")
    if _nvcc_major() not in PERSIST_REGS:
        pytest.skip(f"no register pins for CUDA {_nvcc_major()}")
    archs = [a for a in ARCHS if toolkit is None or int(a[3:]) in toolkit[1]]
    units = _units()
    out = {}
    for label, source in units.items():
        tmp = tmp_path_factory.mktemp(label)
        for arch in archs:
            for entry, row in _compile(source, arch, tmp).items():
                out.setdefault(entry, {})[arch] = row
    return out


def test_every_card_entry_was_compiled(compiled):
    assert set(compiled) == set(ARITH_COUNTS[_nvcc_major()])


@pytest.mark.parametrize("entry", ["rkf78_attempt_persist", "oscillator_step_xauto_persist"])
def test_the_persist_entries_sit_on_their_residency_step(compiled, entry):
    pins = PERSIST_REGS[_nvcc_major()][entry]
    for arch, row in compiled[entry].items():
        if arch in pins:
            assert row["regs"] <= pins[arch], (entry, arch, row["regs"])


@pytest.mark.parametrize("label", ["osc", "rk78"])
def test_the_persist_entry_claims_once_per_chunk_head(label):
    pytest.importorskip("hawk")
    source = _units()[label]
    entry = source.split("_persist(", 1)[1].split("\n}\n", 1)[0]
    head, rest = entry.split("        do {", 1)
    step = rest.split("        } while (", 1)[0]
    assert entry.count("atomicAdd(hawk_next") == head.count("atomicAdd(hawk_next") == 1
    for token in ("atomic", "__syncwarp", "__ballot", "__shfl", "__any_sync", "__all_sync"):
        assert token not in step, (label, token)


def test_no_entry_spills_or_touches_local_memory(compiled):
    for entry, rows in compiled.items():
        for arch, row in rows.items():
            assert (row["spill"], row["stack"]) == (0, 0), (entry, arch)
            assert row["ops"]["LDL"] + row["ops"]["STL"] == 0, (entry, arch)


def test_the_arithmetic_of_every_entry_is_unchanged(compiled):
    for entry, rows in compiled.items():
        for arch, row in rows.items():
            got = tuple(row["ops"][op] for op in ARITH)
            want = ARITH_COUNTS[_nvcc_major()][entry]
            assert got == want, (entry, arch, dict(zip(ARITH, got)))


def test_the_fused_rkf78_entries_keep_their_registers(compiled):
    for entry, pinned in FUSED_REGS[_nvcc_major()].items():
        for arch, row in compiled[entry].items():
            if arch in pinned:
                assert abs(row["regs"] - pinned[arch]) <= 2, (entry, arch, row["regs"])


def test_the_oscillator_persist_entry_stays_small(compiled):
    for arch, row in compiled["oscillator_step_xauto_persist"].items():
        assert row["n"] <= OSC_PERSIST_MAX_INSTRUCTIONS, (arch, row["n"])
