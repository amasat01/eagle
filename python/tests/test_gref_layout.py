# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Compile-time gate: eagle's GREF_DTYPE matches aether's View AND the host mirror.

The launcher passes array views to the kernel **by value** as ``aether::Array<
double, 3>::ViewT`` structs, marshalled from :data:`eagle.abi.GREF_DTYPE`; the C++
host mirrors the same layout as ``plugin/gref_abi.h``'s ``GRefMirror``. If aether's
View layout ever drifts, the by-value kernel-parameter ABI would silently corrupt.
This test shells out (through cupy's nvcc backend) to compile a tiny TU that
includes both ``<aether/aether.h>`` and the ported ``plugin/gref_abi.h``: a
``sizeof`` identity pin between the real ``aether::View`` and ``GRefMirror``, plus a
constexpr functional-equivalence check (same extents/strides/device fed to both a
real ``View`` and ``gref_abi.h``'s own ``make_gref`` must read back identically
through the public accessors) — so drift on either side fails the build loudly.

aether's ``View`` (unlike the prior flat-public-member ``GRef``) keeps its members
private and derives ``compStride_``/``sampleStride_`` from ``mapping().stride(i)``,
so a per-field ``offsetof`` cross-check (the earlier design) no longer applies;
the accessor-based equivalence check below is the aether-shaped replacement.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from eagle.abi import GREF_DTYPE


def _find_aether_include() -> str | None:
    """Return the directory containing aether/aether.h, checking env override then
    the active Python prefix (works when aether is installed into the conda env)."""
    import sys
    candidates = [
        os.environ.get("EAGLE_AETHER_INCLUDE"),
        os.path.join(sys.prefix, "include"),
    ]
    for d in filter(None, candidates):
        if os.path.isfile(os.path.join(d, "aether", "aether.h")):
            return d
    return None


AETHER_INCLUDE = _find_aether_include()
HOST_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "plugin"


def _nvcc_options():
    if not AETHER_INCLUDE:
        pytest.skip("aether/aether.h not found; set EAGLE_AETHER_INCLUDE env var")
    return ("-std=c++20", f"-I{AETHER_INCLUDE}", f"-I{HOST_DIR}")


@pytest.mark.gpu
def test_gref_dtype_matches_aether_layout():
    import cupy as cp

    asserts = [
        f'static_assert(sizeof(GRef) == {GREF_DTYPE.itemsize}, "GRef size drift");',
        # aether real type <-> the ported eagle/host mirror
        'static_assert(sizeof(GRef) == sizeof(GRefMirror), "aether View vs host mirror size");',
        'static_assert(alignof(GRef) == alignof(GRefMirror), "aether View vs host mirror alignment");',
        'static_assert(std::is_standard_layout<GRefMirror>::value, "GRefMirror not standard-layout");',
        'static_assert(std::is_trivially_copyable<GRefMirror>::value, "GRefMirror not trivially-copyable");',
        # Functional equivalence: the SAME (data=nullptr, n, device) fed to a real
        # constexpr aether::View and to a GRefMirror built the same way make_gref
        # computes it (gref_abi.h: compStride_ == samples_ == n, sampleStride_ ==
        # 1) must read back identically through the public accessors -- proving
        # GRefMirror's derived fields agree with aether's own layout math rather
        # than an independently-drifted guess. (make_gref itself is not constexpr,
        # so its formula is reproduced here rather than called.)
        "constexpr std::uint64_t kN = 7;",
        "constexpr GRef probe(nullptr, MapT(ExtT(kN), { std::size_t(kN), std::size_t{1} }), aether::Device(kDLCUDA));",
        "constexpr GRefMirror mirror{ nullptr, kN, kN, 1, kEagleAbiDeviceCUDA, 0 };",
        'static_assert(mirror.samples_ == probe.samples(), "GRefMirror.samples_ vs View::samples() drift");',
        'static_assert(mirror.compStride_ == probe.mapping().stride(0), "GRefMirror.compStride_ vs View mapping stride(0) drift");',
        'static_assert(mirror.sampleStride_ == probe.mapping().stride(1), "GRefMirror.sampleStride_ vs View mapping stride(1) drift");',
        'static_assert(static_cast<int>(mirror.deviceType_) == static_cast<int>(probe.device().type()), "GRefMirror.deviceType_ vs View::device() drift");',
    ]

    src = (
        "#include <aether/aether.h>\n"
        '#include "gref_abi.h"\n'
        "using namespace eagle::plugin;\n"  # the ABI PODs live in eagle::plugin
        "#include <cstddef>\n"
        "#include <cstdint>\n"
        "#include <type_traits>\n"
        "using Vec3dArray = aether::Array<double, 3>;\n"
        "using GRef = Vec3dArray::ViewT;\n"
        "using MapT = typename GRef::mapping_type;\n"
        "using ExtT = typename GRef::extents_type;\n" + "\n".join(asserts) + "\n"
        'extern "C" __global__ void probe_gref(Vec3dArray::ViewT) {}\n'
    )

    mod = cp.RawModule(code=src, backend="nvcc", options=_nvcc_options())
    fn = mod.get_function("probe_gref")  # triggers nvcc compile + static_asserts
    assert fn is not None
