# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The shared role -> ABI-shape classifier (:func:`eagle.roles.
classify_arg`) and the three properties that actually matter once BOTH launch
paths (the cupy device path, ``eagle.launch.assemble_args``; the ctypes host
path, ``eagle.host_launch.HostPluginLibrary.run``) dispatch on it.

**The obvious gate is vacuous and is deliberately NOT here.** "Assert the two
paths classify every role identically" becomes a tautology the moment they
share one classifier — it would compare the classifier to itself and could
never fail. What is gated instead, each with a named injection that proves it
:

* **Coverage** — every role in the canonical vocabulary (:data:`eagle.roles.
  ROLES`) resolves to a tag, no exception, no ``None`` (delete one
  role's branch from the classifier).
* **Fail-loud, in BOTH paths** — an unrecognised role raises through
  ``assemble_args`` AND through ``HostPluginLibrary.run`` (THE
  load-bearing gate: before the fix, the device path silently produced a
  short argument list and launched a wrong kernel; the host path already
  raised. Both paths must raise now, and this test asserts the raise
  specifically, not that "an exception happened somewhere").
* **Tag exhaustiveness, PER PATH** — each path's OWN dispatch handles every
  tag the classifier can emit (remove one tag's branch from ONE
  path -> RED for that path only, a DIFFERENTIAL. Sharing the classifier does
  not mean sharing the per-tag construction — each path still needs its own
  complete switch over the tag space).

Pure-Python where possible (the classifier tests need no GPU); the two
per-path exhaustiveness/fail-loud tests need their real consumer (a handful
of small cupy arrays for the device path, a tiny g++-compiled host plugin —
mirroring ``test_host_launch_torch.py``'s pattern — for the host path).
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess

import numpy as np
import pytest

from eagle.roles import ARG_TAGS, ROLES, classify_arg


# --------------------------------------------------------------------------- #
# Coverage + golden mapping — pure Python, no GPU.
# --------------------------------------------------------------------------- #
def test_classify_arg_covers_every_canonical_role():
    """Every role in ROLES resolves to a tag in ARG_TAGS -- no exception, no
    None. 'mutable' is context-dependent (vec / mat / scalar), so all three
    contexts are exercised; every other role is context-free."""
    for role in sorted(ROLES):
        if role == "mutable":
            assert classify_arg("mutable", "m", vec_mutables={"m"}) in ARG_TAGS
            assert classify_arg("mutable", "m", mat_mutables={"m"}) in ARG_TAGS
            assert classify_arg("mutable", "m") in ARG_TAGS  # scalar/int
        else:
            assert classify_arg(role, "x") in ARG_TAGS


def test_classify_arg_golden_mapping():
    """The exact tag per role -- locks the mapping so a future edit that
    changes WHICH tag a role resolves to (not just deletes a branch) is
    caught too."""
    assert classify_arg("out", "x") == "GREF_VEC"
    assert classify_arg("vec_in", "x") == "GREF_VEC"
    assert classify_arg("mat_in", "x") == "GREF_MAT"
    assert classify_arg("per_sample", "x") == "HANDLE"
    assert classify_arg("lookup", "x") == "HANDLE"
    assert classify_arg("terminated", "x") == "HANDLE"
    assert classify_arg("nsamples", "x") == "NSAMPLES"
    assert classify_arg("uniform", "x") == "UNIFORM"
    assert classify_arg("mutable", "m", vec_mutables={"m"}) == "GREF_VEC"
    assert classify_arg("mutable", "m", mat_mutables={"m"}) == "GREF_MAT"
    assert classify_arg("mutable", "m") == "HANDLE"  # neither set -> scalar/int


def test_classify_arg_raises_on_unrecognized_role():
    with pytest.raises(ValueError, match="unknown arg role"):
        classify_arg("bogus_role", "x")


# --------------------------------------------------------------------------- #
# Fail-loud + tag exhaustiveness, cupy DEVICE path (assemble_args). Device-marked.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_assemble_args_covers_every_tag():
    """Tag exhaustiveness, device path: one representative role per tag
    (out=GREF_VEC, mat_in=GREF_MAT, per_sample=HANDLE, nsamples=NSAMPLES,
    uniform=UNIFORM), a real assemble_args call over small cupy arrays.
    On the device leg, removing one tag's branch from assemble_args must
    RED this test specifically."""
    import cupy as cp

    from eagle.launch import assemble_args

    n = 8
    out = cp.zeros((1, n))
    mat = cp.ones((1, n))
    per_sample_arr = cp.full(n, 2.0)
    term = cp.zeros(n, dtype=cp.bool_)

    args = assemble_args(
        [
            ("out", "out"),
            ("mat_in", "A"),
            ("per_sample", "s"),
            ("nsamples", "n"),
            ("uniform", "u"),
        ],
        out=out,
        vec={},
        per_sample={"s": per_sample_arr},
        terminated=term,
        uniforms={"u": 3.0},
        n=n,
        mats={"A": mat},
    )
    assert len(args) == 5


def test_assemble_args_raises_on_unrecognized_role():
    """On the device leg, assemble_args must raise for an unrecognised
    role -- never silently skip it (the pre-fix defect: a short argument
    list, then a wrong kernel call). No GPU needed: classify_arg raises
    before any cupy-array access happens."""
    from eagle.launch import assemble_args

    n = 4
    with pytest.raises(ValueError, match="unknown arg role"):
        assemble_args(
            [("bogus_role", "x")],
            out=None,
            vec={},
            per_sample={},
            terminated=None,
            uniforms={},
            n=n,
        )


# --------------------------------------------------------------------------- #
# Fail-loud + tag exhaustiveness, ctypes HOST path (HostPluginLibrary.run).
# CPU-only -- mirrors test_host_launch_torch.py's g++-compile-a-tiny-plugin
# pattern (no eagle/plugin/** C++ header, no nvcc, no cupy).
# --------------------------------------------------------------------------- #
_PLUGIN_SRC = r"""
#include <cstdint>
struct GRefMirror {
    void* data_;
    uint32_t nVecs_;
    uint32_t dimOffset_;
    uint64_t tex_;
    uint32_t texOffset_;
};
struct ScalarHandle { void* ptr; };

extern "C" void exhaust_host(void* const* p, int n) {
    double* m       = (double*)((const GRefMirror*)p[0])->data_;
    const double* A = (const double*)((const GRefMirror*)p[1])->data_;
    const double* s = (const double*)((const ScalarHandle*)p[2])->ptr;
    uint32_t nn     = *(const uint32_t*)p[3];
    double u        = *(const double*)p[4];
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        m[i] = A[i] + s[i] + u + (double)(nn - (uint32_t)n);
}
"""

# One "mutable" (vector) -> GREF_VEC, one "mat_in" (1x1, trivial) -> GREF_MAT,
# one "per_sample" -> HANDLE, "nsamples" -> NSAMPLES, "uniform" -> UNIFORM --
# the same five-tag coverage as the device-path exhaustiveness test above.
_TAG_SIDECAR = {
    "kernel": "exhaust",
    "aether_abi": "aether-abi/1",
    "host_entry": "exhaust_host",
    "arg_spec": [
        ["mutable", "m"],
        ["mat_in", "A"],
        ["per_sample", "s"],
        ["nsamples", "n"],
        ["uniform", "u"],
    ],
    "mutables": [{"name": "m", "dtype": "vector", "width": 1}],
    "mat_shapes": {"A": [1, 1]},
}


def _gxx() -> str | None:
    if shutil.which(os.environ.get("CXX", "")):
        return os.environ.get("CXX")
    return shutil.which("g++") or (
        "/usr/bin/g++" if os.path.exists("/usr/bin/g++") else None
    )


@pytest.fixture(scope="module")
def _tag_plugin_so(tmp_path_factory):
    gxx = _gxx()
    if gxx is None:
        pytest.skip("no g++ available to build the host plugin fixture")
    tmp = tmp_path_factory.mktemp("uA4_arg_classify")
    src = tmp / "exhaust_host.cpp"
    src.write_text(_PLUGIN_SRC)
    so = tmp / "exhaust_host.so"
    proc = subprocess.run(
        [gxx, "-O3", "-shared", "-fPIC", "-fopenmp", "-o", str(so), str(src)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"host plugin compile failed: {proc.stderr.strip()[:400]}")
    return str(so)


def _bound_lib(so_path):
    from eagle import interop
    from eagle.host_launch import HostPluginLibrary

    n = 6
    lib = HostPluginLibrary(so_path, _TAG_SIDECAR)
    m = np.zeros(n, dtype=np.float64)
    a = np.full(n, 5.0, dtype=np.float64)
    s = np.full(n, 2.0, dtype=np.float64)
    lib.bind_vector("m", interop.host_ptr(m), n)
    lib.bind_matrix("A", interop.host_ptr(a), n, 1, 1)
    lib.bind_handle("s", interop.host_ptr(s))
    lib.bind_uniform("u", 3.0)
    return lib, n, m


def test_host_plugin_run_covers_every_tag(_tag_plugin_so):
    """Tag exhaustiveness, host path: the SAME five tags as the device-path
    test above, via a real dlopen'd plugin call. On the host leg, removing
    one tag's branch from HostPluginLibrary.run must RED this test
    specifically, while the device-path test above stays green (the
    differential)."""
    lib, n, m = _bound_lib(_tag_plugin_so)
    assert lib.run(n) == n
    # m[i] = A[i] + s[i] + u + (nn - n) = 5 + 2 + 3 + 0 = 10 -- proves every
    # bound value (GREF_VEC mutable, GREF_MAT mat_in, HANDLE per_sample,
    # NSAMPLES, UNIFORM) actually reached the plugin, not just "didn't crash".
    np.testing.assert_allclose(m, np.full(n, 10.0))


def test_host_plugin_run_raises_on_unrecognized_role(_tag_plugin_so):
    """On the host leg, HostPluginLibrary.run must raise for an
    unrecognised role. ``_arg_spec`` is mutated AFTER construction (bypassing
    the upstream ``validate_sidecar`` gate at __init__, which would otherwise
    reject the bogus role before ``run`` is ever reached) so this pins
    ``run``'s OWN defense-in-depth fail-loud property in isolation."""
    lib, n, _m = _bound_lib(_tag_plugin_so)
    lib._arg_spec = [("bogus_role", "x")]
    with pytest.raises(ValueError, match="unknown arg role"):
        lib.run(n)


def test_both_paths_raise_on_the_same_unrecognized_role(_tag_plugin_so):
    """The direct proof that defect 2 (launch.py's
    silent fallthrough) is dead on BOTH paths for the identical bogus role,
    asserted specifically via pytest.raises on each call -- not "an
    exception happened somewhere" in a broad try/except."""
    from eagle.launch import assemble_args

    with pytest.raises(ValueError, match="unknown arg role"):
        assemble_args(
            [("bogus_role", "x")],
            out=None, vec={}, per_sample={}, terminated=None, uniforms={}, n=4,
        )

    lib, n, _m = _bound_lib(_tag_plugin_so)
    lib._arg_spec = [("bogus_role", "x")]
    with pytest.raises(ValueError, match="unknown arg role"):
        lib.run(n)


# --------------------------------------------------------------------------- #
# GRefMirror / ScalarHandle POD-layout sanity (mirrors
# test_host_launch_torch.py::test_abi_pod_layouts_match_cpp -- cheap, worth
# re-asserting here since this file constructs its own tiny plugin using the
# same layout).
# --------------------------------------------------------------------------- #
def test_grefmirror_scalarhandle_pod_sizes():
    from eagle.host_launch import GRefMirror, ScalarHandle

    assert ctypes.sizeof(GRefMirror) == 40
    assert ctypes.sizeof(ScalarHandle) == 32
