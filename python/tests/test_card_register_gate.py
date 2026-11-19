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

* ``rkf78_attempt_persist`` uses at most their pinned per-arch counts
  (per CUDA release), each on or below its residency step;
* the persist entry is warp-converged: its source holds ONE
  ``atomicAdd(hawk_next`` (the chunk head's warp-wide claim) and its
  unrolled step loop no atomic and no warp collective;
* no entry spills, keeps a stack frame, or touches local memory;
* every entry's floating-point arithmetic (DFMA/DMUL/DADD/MUFU) is the
  pinned count: the bookkeeping changes, the maths never does;
* the fused RKF7(8) entries stay within 2 registers of their pinned counts
  (no cap reaches them spill-free, so they are left as they are);

The pins are keyed by the release of the CUDA compiler hawk builds with (``hawk.targets.machine``),
which need not be the ``nvcc`` on PATH. Skipped where hawk states no CUDA compiler.
"""

from __future__ import annotations

import collections
import importlib.util
import pathlib
import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.repo_local

_BENCH = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"
ARCHS = ("sm_61", "sm_80", "sm_89", "sm_90")
ARITH = ("DFMA", "DMUL", "DADD", "MUFU")

#: Register pins by CUDA release, then entry, then arch: register counts differ
#: between toolkits, so each is held to the pins of the toolkit that compiled it
#: (12.6, 12.9 and 13.4 are measured). An arch missing under a release has no
#: pin there and is not checked. Persist entries: the most registers allowed
#: (the warp-converged loop is held to the register count pinned here, per arch).
PERSIST_REGS = {
    "12.6": {
        "rkf78_attempt_persist": {"sm_61": 162, "sm_80": 164, "sm_89": 164, "sm_90": 164},
    },
    "12.9": {
        "rkf78_attempt_persist": {"sm_61": 166, "sm_80": 164, "sm_89": 164, "sm_90": 164},
    },
    "13.4": {
        "rkf78_attempt_persist": {"sm_80": 164, "sm_89": 164, "sm_90": 164},
    },
}

#: Pinned register counts of the fused RKF7(8) entries, +-2 allowed, by release.
FUSED_REGS = {
    "12.6": {
        "rkf78_attempt": {"sm_61": 184, "sm_80": 184, "sm_89": 184, "sm_90": 186},
        "rkf78_attempt_range": {"sm_61": 186, "sm_80": 186, "sm_89": 186, "sm_90": 186},
    },
    "12.9": {
        "rkf78_attempt": {"sm_61": 188, "sm_80": 182, "sm_89": 182, "sm_90": 184},
        "rkf78_attempt_range": {"sm_61": 186, "sm_80": 184, "sm_89": 184, "sm_90": 186},
    },
    "13.4": {
        "rkf78_attempt": {"sm_80": 182, "sm_89": 182, "sm_90": 184},
        "rkf78_attempt_range": {"sm_80": 184, "sm_89": 184, "sm_90": 186},
    },
}

#: Pinned arithmetic per entry by CUDA release, the same on every arch. The
#: toolkits differ in how many DADDs the RKF7(8) entries carry.
# The step and stop counters are uint32 planes, so their increments are integer
# adds (no DADD); the range entry's loop unrolls six copies of the fused step.
ARITH_COUNTS = {
    "12.6": {
        "rkf78_attempt": (1247, 400, 80, 69),
        "rkf78_attempt_range": (1247, 400, 80, 69),
        "rkf78_attempt_persist": (659, 210, 56, 37),
    },
    "12.9": {
        "rkf78_attempt": (1247, 400, 79, 69),
        "rkf78_attempt_range": (1247, 400, 79, 69),
        "rkf78_attempt_persist": (659, 210, 56, 37),
    },
    "13.4": {
        "rkf78_attempt": (1247, 400, 81, 69),
        "rkf78_attempt_range": (1247, 400, 81, 69),
        "rkf78_attempt_persist": (659, 210, 57, 37),
    },
}

def _load(name, rel):
    if str(_BENCH) not in sys.path:
        sys.path.insert(0, str(_BENCH))
    spec = importlib.util.spec_from_file_location(name, _BENCH / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _units():
    """``{label: kernel}``: the card kernels, as ``eagle.deploy`` builds them."""
    rk78 = _load("eagle_rk78_card_gate", "perf_card/perf_card.py")
    return {"rk78": rk78._define_kernel()}


def _compile(kernel, arch, tmp):
    """``{entry: {regs, stack, spill, n, ops}}`` of one kernel's cubin on one arch: the registers, stack and spills from the
    figures the build states (``sidecar["tool"]["cuda"]``, there only for a real arch's cubin image), the SASS read back."""
    from hawk.artifact import build_bundle
    from hawk.build_settings import BuildSettings

    bundle = build_bundle([kernel], tmp / arch, targets=("cuda",), settings=BuildSettings(fast_entries=True),
                          options={"cuda.arch": arch})
    art = next(a for a in bundle.artifacts if a.name == kernel.name)
    entries = {}
    for name, figures in art.sidecar["tool"]["cuda"].items():
        entries[name] = {"ops": collections.Counter(), "n": 0, "regs": figures["regs"], "stack": figures["stack_B"],
                         "spill": (figures["spill_store_B"] or 0) + (figures["spill_load_B"] or 0),
                         "local": art.sidecar["launch"].get("local_bytes")}
    cuobjdump = pathlib.Path(_cuda_compiler()["path"]).with_name("cuobjdump")
    sass = subprocess.run([str(cuobjdump), "-sass", str(art.entries["cuda"].artifact)], capture_output=True,
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


def _cuda_compiler():
    """hawk's stated CUDA compiler: ``path``, ``release`` (``"X.Y"``) and ``archs``; ``{}`` when it finds none."""
    from hawk.targets import machine

    return machine("cuda")["tools"].get("compiler") or {}


def _nvcc_release():
    """The ``"X.Y"`` release of the compiler hawk builds the cubins with: the key of every pin table."""
    return _cuda_compiler().get("release")


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    pytest.importorskip("hawk")
    from hawk.targets import buildable

    if not buildable("cuda"):
        pytest.skip("hawk states no CUDA toolchain on this machine")
    toolkit = _cuda_compiler()["archs"]
    if _nvcc_release() not in PERSIST_REGS:
        pytest.skip(f"no register pins for CUDA {_nvcc_release()}")
    archs = [a for a in ARCHS if int(a[3:]) in toolkit]
    out = {}
    for label, kernel in _units().items():
        tmp = tmp_path_factory.mktemp(label)
        for arch in archs:
            for entry, row in _compile(kernel, arch, tmp).items():
                out.setdefault(entry, {})[arch] = row
    return out


def test_every_card_entry_was_compiled(compiled):
    assert set(compiled) == set(ARITH_COUNTS[_nvcc_release()])


@pytest.mark.parametrize("entry", ["rkf78_attempt_persist"])
def test_the_persist_entries_sit_on_their_residency_step(compiled, entry):
    pins = PERSIST_REGS[_nvcc_release()][entry]
    for arch, row in compiled[entry].items():
        if arch in pins:
            assert row["regs"] <= pins[arch], (entry, arch, row["regs"])


@pytest.mark.parametrize("label", ["rk78"])
def test_the_persist_entry_claims_once_per_chunk_head(label):
    pytest.importorskip("hawk")
    from hawk.artifact import emit_source

    source = emit_source(_units()[label], targets=("cuda",)).sources["cuda"].text
    entry = source.split("_persist(", 1)[1].split("\n}\n", 1)[0]
    head, rest = entry.split("        do {", 1)
    step = rest.split("        } while (", 1)[0]
    assert entry.count("atomicAdd(hawk_next") == head.count("atomicAdd(hawk_next") == 1
    for token in ("atomic", "__syncwarp", "__ballot", "__shfl", "__any_sync", "__all_sync"):
        assert token not in step, (label, token)


def test_no_entry_spills_or_touches_local_memory(compiled):
    for entry, rows in compiled.items():
        for arch, row in rows.items():
            assert (row["spill"], row["stack"], row["local"] or 0) == (0, 0, 0), (entry, arch)
            assert row["ops"]["LDL"] + row["ops"]["STL"] == 0, (entry, arch)


def test_the_arithmetic_of_every_entry_is_unchanged(compiled):
    for entry, rows in compiled.items():
        for arch, row in rows.items():
            got = tuple(row["ops"][op] for op in ARITH)
            want = ARITH_COUNTS[_nvcc_release()][entry]
            assert got == want, (entry, arch, dict(zip(ARITH, got)))


def test_the_fused_rkf78_entries_keep_their_registers(compiled):
    for entry, pinned in FUSED_REGS[_nvcc_release()].items():
        for arch, row in compiled[entry].items():
            if arch in pinned:
                assert abs(row["regs"] - pinned[arch]) <= 2, (entry, arch, row["regs"])
