# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The four eagle-owned execution-structure certification rows
(``raptor.conformance.interop.ROWS``):
``execution_partition_identity``, ``execution_host_device_twin``,
``execution_illegal_placement_refused``, ``execution_layout_selfcheck_refused``.

raptor declared these four rows in ``raptor.conformance.interop.ROWS``
ahead of their implementation, banked here as ``xfail(strict=True)`` against
``eagle.exec``/``eagle.plan``, which did not exist yet. This file lands both
modules (``eagle/exec.py``, ``eagle/plan.py``) and un-xfails all four.

FIXTURES: rows 1/2 (partition identity, host/device twin) need a REAL
aether-abi/2 body run through ``eagle.plan``/``eagle.exec`` on both targets.
The committed v1 fixture (``fixtures/gravity.ptx``) cannot serve this any
more — a legacy v1 artifact (no partition triple) is restricted to a
whole-view, single-partition launch, so splitting it into two partitions
would be exactly the illegal placement that restriction exists to refuse, not a
bit-exact identity to prove. :data:`v2_sample_local` below is a HAND-MIRRORED
aether-abi/2 ``sample_local`` fixture (``y = a*x + b``) — the same body and
ABI shape as the repo-root C++ fixtures' ``exec_local``
(``tests/fixtures/host_plugin_execv2.cpp`` / ``device_plugin_execv2.cu``),
self-contained and compiled at test time (the existing python-fixture
convention: see ``test_int_uniform.py``'s ``int_plugin_so`` /
``test_arg_classify.py``'s ``_tag_plugin_so``, which re-declare the ABI
structs locally rather than including the C++ header). Rows 3/4 (illegal
placement, layout self-check) exercise ``eagle.plan``/``eagle.exec``
primitives directly and need no real kernel body at all — row 3 still uses
the committed v1 ``gravity.ptx`` purely as an inert plugin HANDLE (its
``exec_access`` is overridden explicitly, so ``eagle.plan.plan`` never
inspects the artifact's own body or ABI tag for that call).

Each row is registered via :func:`raptor.conformance.interop.register_row`
(the mandatory hygiene gate, ``assert_no_skip_machinery``, bans
skip/skipif/importorskip and the conftest availability markers on a
registered row) and MUST NOT skip -- the fixture below raises loudly rather
than ``pytest.skip``-ing when the g++/nvcc toolchain it needs is unavailable,
since a registered row silently skipping is exactly the defect class this
matrix's hygiene gate exists to catch.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest
from raptor.conformance.interop import register_row

from _device_file import compile_device
from eagle import LoadedVector

pytestmark = pytest.mark.interop_matrix

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"


# --------------------------------------------------------------------------- #
# The v2 `sample_local` fixture (y = a*x + b), both targets -- compiled at
# test time, module-scoped.
# --------------------------------------------------------------------------- #
_HOST_SRC = r"""
#include <cstdint>
// Self-contained ScalarHandle mirror -- SAME 32-byte layout as
// plugin/gref_abi.h::ScalarHandle / eagle.host_launch.ScalarHandle (the
// existing python-fixture convention: redeclare the ABI POD locally rather
// than including the C++ header -- see test_int_uniform.py / test_arg_classify.py).
struct ScalarHandle {
    void* data;
    unsigned long long samples;
    unsigned long long stride;
    int deviceType;
    int deviceId;
};

// aether-abi/2 HOST entry: a SERIAL range over
// [base, base+count) -- no internal threading, that is eagle's HostTeam.
// Mirrors tests/fixtures/host_plugin_execv2.cpp's `exec_local_host`.
extern "C" void exec_local_host(void* const* params, long long base,
                                 long long count, long long /*nSamples*/) {
    double* y       = (double*)((const ScalarHandle*)params[0])->data;
    const double* x = (const double*)((const ScalarHandle*)params[1])->data;
    const double a  = *(const double*)params[2];
    const double b  = *(const double*)params[3];
    for (long long i = base; i < base + count; ++i) y[i] = a * x[i] + b;
}
"""

_DEVICE_SRC = r"""
#include <cstdint>
struct ScalarHandle {
    void* data;
    unsigned long long samples;
    unsigned long long stride;
    int deviceType;
    int deviceId;
};

// aether-abi/2 DEVICE entry: a FLAT index + early-out on `count`, global
// sample index `base + flat` -- the grid is eagle's (DeviceKernel::grid).
// Mirrors tests/fixtures/device_plugin_execv2.cu's `exec_local`.
extern "C" __global__ void exec_local(ScalarHandle y, ScalarHandle x, double a,
                                       double b, long long base, long long count,
                                       long long /*nSamples*/) {
    long long flat = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    long long i = base + flat;
    ((double*)y.data)[i] = a * ((const double*)x.data)[i] + b;
}
"""


class _V2SampleLocal:
    """A loaded v2 ``sample_local`` fixture: ``.arg_spec`` (the
    ``eagle.roles`` vocabulary ``eagle.plan`` packs), ``.exec_access``, and
    the raw device/host entry points ``eagle.plan`` launches through
    ``eagle.exec``."""

    exec_access = "sample_local"
    exec_op = None
    # [mutable y, per_sample x, uniform a, uniform b] -- see host_entry/
    # device_function above; matches host_plugin_execv2.cpp's exec_local_host
    # comment (its own arg_spec additionally lists a trailing `nsamples` role
    # the body never dereferences -- omitted here since eagle.plan's packer
    # only emits params a body actually reads).
    arg_spec = (
        ("mutable", "y"),
        ("per_sample", "x"),
        ("uniform", "a"),
        ("uniform", "b"),
    )

    def __init__(self, so_path, device_image):
        self._lib = ctypes.CDLL(str(so_path))
        fn = self._lib.exec_local_host
        self.host_entry = ctypes.cast(fn, ctypes.c_void_p).value

        import cupy as cp

        self._module = cp.RawModule(path=str(device_image))
        self._kernel = self._module.get_function("exec_local")
        self.device_function = self._kernel.kernel.ptr


def _gxx() -> str | None:
    return shutil.which(os.environ.get("CXX", "")) or shutil.which("g++") or (
        "/usr/bin/g++" if os.path.exists("/usr/bin/g++") else None
    )


def _nvcc() -> str | None:
    found = shutil.which("nvcc")
    if found:
        return found
    cand = pathlib.Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")) / "bin" / "nvcc"
    return str(cand) if cand.exists() else None


@pytest.fixture(scope="module")
def v2_sample_local(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("execv2_sample_local")

    gxx = _gxx()
    if gxx is None:
        raise RuntimeError(
            "execution_partition_identity/execution_host_device_twin need g++ "
            "to build the v2 host fixture; none found on PATH or at /usr/bin/g++"
        )
    nvcc = _nvcc()
    if nvcc is None:
        raise RuntimeError(
            "execution_partition_identity/execution_host_device_twin need nvcc "
            "to build the v2 device fixture; none found on PATH or CUDA_PATH"
        )

    src = tmp / "exec_local_host.cpp"
    src.write_text(_HOST_SRC)
    so = tmp / "exec_local_host.so"
    proc = subprocess.run(
        [gxx, "-O2", "-shared", "-fPIC", "-o", str(so), str(src)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"v2 host fixture compile failed:\n{proc.stderr}"

    cu = tmp / "exec_local.cu"
    cu.write_text(_DEVICE_SRC)
    # PTX, or a CUBIN for this device when the toolkit is newer than the driver
    device_image = compile_device(nvcc, cu, tmp / "exec_local")

    return _V2SampleLocal(so, device_image)


# --------------------------------------------------------------------------- #
# The four rows.
# --------------------------------------------------------------------------- #
@register_row("execution_partition_identity")
def test_execution_partition_identity_bit_exact_under_split(v2_sample_local):
    """A ``sample_local`` body (``y = a*x + b``, no cross-sample reads/
    writes) run WHOLE under ``DeviceKernel`` must be BIT-IDENTICAL to the
    same body run as two ``eagle.exec`` partitions of the same sample set
    (the ``{base, count, nSamples}`` triple), driven through ``eagle.plan``."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    rng = np.random.default_rng(11)
    n = 64
    x = rng.normal(size=n)
    half = n // 2
    a, b = 2.0, -1.5

    whole = eplan.plan(v2_sample_local, structure=eexec.DeviceKernel).run(x=x, a=a, b=b)
    partitioned = eplan.plan(
        v2_sample_local,
        structure=eexec.DeviceKernel,
        partitions=[(0, half, n), (half, n - half, n)],
    ).run(x=x, a=a, b=b)

    np.testing.assert_array_equal(whole, partitioned)  # bit-exact, sample_local


@register_row("execution_host_device_twin")
def test_execution_host_device_twin_within_ruled_band(v2_sample_local):
    """The SAME ``sample_local`` body run under ``HostTeam`` (OpenMP)
    vs ``DeviceKernel`` (CUDA) must agree within the RULED host/device band
    (fork F-d default: ``S x 2 x eps(dtype)``, S = combined element count)."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    rng = np.random.default_rng(12)
    n = 64
    x = rng.normal(size=n)
    a, b = 2.0, -1.5

    device = eplan.plan(v2_sample_local, structure=eexec.DeviceKernel).run(
        x=x, a=a, b=b
    )
    host = eplan.plan(v2_sample_local, structure=eexec.HostTeam).run(x=x, a=a, b=b)

    band = x.size * 2 * np.finfo(np.float64).eps  # F-d default anchor
    assert np.max(np.abs(device - host)) <= band


@register_row("execution_illegal_placement_refused")
def test_execution_illegal_placement_cross_sample_write_multi_partition_refused():
    """``exec_access='cross_sample_write'`` is single-device only
    until a partial-accum combine is ruled — a plan spanning two partitions
    for such a body must be REFUSED, and the refusal must NAME the rule
    ("a refused plan names the rule"), not merely raise something."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    # an inert handle -- see module docstring
    plugin = LoadedVector(FIX / "gravity.ptx")

    with pytest.raises(ValueError, match="cross_sample_write"):
        eplan.plan(
            plugin,
            structure=eexec.DeviceKernel,
            partitions=[(0, 32, 64), (32, 32, 64)],
            _exec_access="cross_sample_write",
        )


@register_row("execution_layout_selfcheck_refused")
def test_execution_layout_selfcheck_refused_on_size_mismatch():
    """A plugin whose ``aether_abi`` tag correctly reads
    ``aether-abi/2`` but whose exported ``eagle_layout_sizes[]`` disagrees
    with this build's own sizes (GRefMirror 40 / ScalarHandle 32 / IntHandle
    32) must be refused at LOAD, not garbage-computed."""
    from eagle import exec as eexec

    bad_sizes = {"GRefMirror": 41, "ScalarHandle": 32, "IntHandle": 32}
    with pytest.raises(ValueError, match="layout"):
        eexec.check_layout_sizes(bad_sizes)
