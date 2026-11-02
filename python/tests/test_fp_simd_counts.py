# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The packed-vs-scalar FP64 counter of the cards, on captured disassembly (static: no Arm machine needed),
and the CPU card's file stem on Arm."""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import pytest

pytestmark = pytest.mark.repo_local

_BENCH = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"


def _card_common():
    spec = importlib.util.spec_from_file_location("card_common_t", _BENCH / "_card_common.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

AARCH64 = """
  4005a0:	4e60cc20 	fmla	v0.2d, v1.2d, v2.2d
  4005a4:	6ee1d442 	fadd	v2.2d, v2.2d, v1.2d
  4005a8:	1f420c20 	fmadd	d0, d1, d2, d3
  4005ac:	1e620820 	fmul	d0, d1, d2
  4005b0:	65c2a020 	fmla	z0.d, p0/m, z1.d, z2.d
  4005b4:	1e622820 	fadd	d0, d1, d2
  4005b8:	0f801020 	fmla	v0.4s, v1.4s, v2.s[0]
  4005bc:	d65f03c0 	ret
"""
X86 = """
  401000:	c5 fd 59 c1          	vmulpd %ymm1,%ymm0,%ymm0
  401004:	c4 e2 f1 b9 c2       	vfmadd231sd %xmm2,%xmm1,%xmm0
  401009:	c5 f9 58 c1          	vaddpd %ymm1,%ymm0,%ymm0
"""


def test_aarch64_neon_sve_and_scalar_double_ops_are_told_apart():
    got = _card_common().fp_simd_counts(AARCH64)
    assert got == {"packed": 3, "scalar": 3, "sve": 1}  # the 4s single-precision fmla is none of them


def test_x86_packed_and_scalar_double_ops_are_still_counted():
    assert _card_common().fp_simd_counts(X86) == {"packed": 2, "scalar": 1, "sve": 0}


def test_an_arm_machine_without_a_model_name_gets_the_aarch64_stem(monkeypatch):
    spec = importlib.util.spec_from_file_location("cpu_card_t", _BENCH / "perf_card" / "cpu_card.py")
    sys.path.insert(0, str(_BENCH / "perf_card"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(mod.platform, "machine", lambda: "aarch64")
    assert mod._cpu_slug("aarch64") == mod._cpu_slug(None) == "aarch64"
    monkeypatch.setattr(mod.platform, "machine", lambda: "x86_64")
    assert mod._cpu_slug("AMD EPYC 9B45") == "amd-epyc-9b45"
