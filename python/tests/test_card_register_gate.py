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
  (CUDA 12.6), each on or below its residency step;
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

#: The most registers a persist entry may use, per arch (CUDA 12.6): the
#: warp-converged loop holds no more than the per-lane loop it replaced.
PERSIST_REGS = {
    "rkf78_attempt_persist": {"sm_61": 166, "sm_80": 168, "sm_89": 168, "sm_90": 168},
    "oscillator_step_xauto_persist": {"sm_61": 40, "sm_80": 36, "sm_89": 36, "sm_90": 40},
}

#: Pinned register counts of the fused RKF7(8) entries (CUDA 12.6), +-2 allowed.
FUSED_REGS = {
    "rkf78_attempt": {"sm_61": 186, "sm_80": 186, "sm_89": 186, "sm_90": 188},
    "rkf78_attempt_range": {"sm_61": 188, "sm_80": 188, "sm_89": 188, "sm_90": 188},
}

#: Pinned arithmetic per entry, the same on every arch (CUDA 12.6).
ARITH_COUNTS = {
    "oscillator_step_xauto": (32, 16, 8, 0),
    "oscillator_step_xauto_range": (80, 28, 17, 0),
    # four unrolled trips, no peeled one: the step's maths is the fused one
    "oscillator_step_xauto_persist": (64, 20, 14, 0),
    "rkf78_attempt": (1247, 400, 84, 69),
    "rkf78_attempt_range": (1247, 400, 84, 69),
    "rkf78_attempt_persist": (659, 210, 58, 37),
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


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    pytest.importorskip("hawk")
    if not (shutil.which("nvcc") and shutil.which("cuobjdump")):
        pytest.skip("no CUDA toolkit (nvcc, cuobjdump) on PATH")
    from hawk.compile import toolchain as tc

    toolkit = tc.toolkit_archs("nvcc")
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
    assert set(compiled) == set(ARITH_COUNTS)


@pytest.mark.parametrize("entry", sorted(PERSIST_REGS))
def test_the_persist_entries_sit_on_their_residency_step(compiled, entry):
    for arch, row in compiled[entry].items():
        assert row["regs"] <= PERSIST_REGS[entry][arch], (entry, arch, row["regs"])


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
            assert got == ARITH_COUNTS[entry], (entry, arch, dict(zip(ARITH, got)))


def test_the_fused_rkf78_entries_keep_their_registers(compiled):
    for entry, pinned in FUSED_REGS.items():
        for arch, row in compiled[entry].items():
            assert abs(row["regs"] - pinned[arch]) <= 2, (entry, arch, row["regs"])


def test_the_oscillator_persist_entry_stays_small(compiled):
    for arch, row in compiled["oscillator_step_xauto_persist"].items():
        assert row["n"] <= OSC_PERSIST_MAX_INSTRUCTIONS, (arch, row["n"])
