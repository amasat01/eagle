# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Compile-time gate: eagle's HANDLE_DTYPE matches aether's View AND the host mirror.

Per-sample spacecraft scalars (mass/area/cr/cd) and the termination mask are passed
to the kernel **by value** as ``aether::Array<T>::ViewT`` structs (the rank-1
sibling of the vector GRef), marshalled from :data:`eagle.abi.HANDLE_DTYPE`; the C++
host mirrors the same layout as ``plugin/gref_abi.h``'s ``ScalarHandle`` /
``IntHandle``. This test shells out (through cupy's nvcc backend) to compile a tiny
TU that includes both ``<aether/aether.h>`` and the ported ``plugin/gref_abi.h`` and
pins ``sizeof`` for the ``double`` (scalar), ``bool`` (mask), and ``int`` (a pure
kernel's ``Mutable[int]``) view types against ``HANDLE_DTYPE`` and the host mirrors,
plus a constexpr functional-equivalence check between a real aether View and the
mirror built the same way ``gref_abi.h``'s ``make_handle``/``make_int_handle``
compute it — so drift on either side fails the build loudly.

aether's View (unlike the prior bare-pointer ``HandleT``) carries its own extent and
device tag and keeps its data members private, so a per-field ``offsetof`` cross-
check (the earlier design) no longer applies; the accessor-based equivalence check
below is the aether-shaped replacement.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from eagle.abi import HANDLE_DTYPE


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
def test_handle_dtype_matches_aether_layout():
    import cupy as cp

    size = HANDLE_DTYPE.itemsize
    asserts = [
        # aether real View <-> the ported eagle/host mirrors (Real/bool ScalarHandle, IntHandle)
        f'static_assert(sizeof(ScalarHandle) == {size}, "ScalarHandle size drift");',
        f'static_assert(sizeof(IntHandle) == {size}, "IntHandle size drift");',
        'static_assert(std::is_standard_layout<ScalarHandle>::value, "ScalarHandle not standard-layout");',
        'static_assert(std::is_trivially_copyable<ScalarHandle>::value, "ScalarHandle not trivially-copyable");',
        'static_assert(std::is_standard_layout<IntHandle>::value, "IntHandle not standard-layout");',
        'static_assert(std::is_trivially_copyable<IntHandle>::value, "IntHandle not trivially-copyable");',
    ]
    for tag, cxx in (("R", "double"), ("B", "bool"), ("I", "int")):
        h = f"H{tag}"
        asserts += [
            f"using {h} = aether::Array<{cxx}>::ViewT;",
            f'static_assert(sizeof({h}) == {size}, "{cxx} view size drift");',
            f'static_assert(alignof({h}) == alignof(ScalarHandle), "{cxx} view alignment drift");',
        ]
    # Functional equivalence (double lane only -- the mechanism is dtype-independent):
    # the SAME (data=nullptr, n, device) fed to a real constexpr aether::View and to
    # a ScalarHandle built the way make_handle computes it (gref_abi.h: stride == 1)
    # must read back identically through the public accessors. (make_handle itself
    # is not constexpr, so its formula is reproduced here rather than called.)
    asserts += [
        "constexpr std::uint64_t kN = 7;",
        "using RMapT = typename HR::mapping_type;",
        "using RExtT = typename HR::extents_type;",
        "constexpr HR probe(nullptr, RMapT(RExtT(kN), { std::size_t{1} }), aether::Device(kDLCUDA));",
        "constexpr ScalarHandle mirror{ nullptr, kN, 1, kEagleAbiDeviceCUDA, 0 };",
        'static_assert(mirror.samples == probe.samples(), "ScalarHandle.samples vs View::samples() drift");',
        'static_assert(mirror.stride == probe.mapping().stride(0), "ScalarHandle.stride vs View mapping stride(0) drift");',
        'static_assert(static_cast<int>(mirror.deviceType) == static_cast<int>(probe.device().type()), "ScalarHandle.deviceType vs View::device() drift");',
    ]

    src = (
        "#include <aether/aether.h>\n"
        '#include "gref_abi.h"\n'
        "using namespace eagle::plugin;\n"  # the ABI PODs live in eagle::plugin
        "#include <cstddef>\n"
        "#include <cstdint>\n"
        "#include <type_traits>\n" + "\n".join(asserts) + "\n"
        'extern "C" __global__ void probe_handle(\n'
        "    aether::Array<double>::ViewT,\n"
        "    aether::Array<bool>::ViewT,\n"
        "    aether::Array<int>::ViewT) {}\n"
    )

    mod = cp.RawModule(code=src, backend="nvcc", options=_nvcc_options())
    fn = mod.get_function("probe_handle")  # triggers nvcc compile + static_asserts
    assert fn is not None
