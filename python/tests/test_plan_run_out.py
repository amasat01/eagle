# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``Plan.run`` writes its result into every SUPPLIED writable output/mutable
plane, not just returns it.

Bug (reproduced, and the exact row RED before this file's fix): the landing
page's own "Thirty seconds" snippet --
``eplan.plan(plugin, structure=eexec.DeviceKernel).run(x=..., a=..., b=...,
y=y)`` -- returned the correct answer but left the caller's numpy ``y``
untouched (still zeros): the device path uploads a FRESH cupy buffer from
``y`` and downloads a FRESH numpy array for the return value, never writing
through to ``y`` itself, unless ``y`` happened to need eagle's sample-major
layout fix (:mod:`eagle._layout`, already covered by ``test_layout.py``) --
the one case this bug never touched. ``HostTeam`` looked unaffected only
because ``np.ascontiguousarray`` is a no-op identity for an already-
contiguous array; a non-contiguous HOST output had the SAME bug.

Every row below is RED against the pre-fix code (confirmed by temporarily
disabling ``eagle.plan._write_back_outputs`` / ``_check_run_writable``, see
the module docstring of ``eagle.plan`` for the fix itself) -- a row that only
exercises the PRE-EXISTING sample-major path (already covered by
``test_layout.py``) is folded into the SAME test as a native-layout row, so
the one assertion this file's fix is actually responsible for still fails the
whole test when the fix is reverted.
"""

from __future__ import annotations

import ctypes
import pathlib
import subprocess

import numpy as np
import pytest

import eagle

EAGLE_ROOT = pathlib.Path(__file__).resolve().parents[2]  # holds plugin/gref_abi.h

# Reuse the already-compiled width-3 ``out`` kernel (e2_vec3) and its build
# plumbing from the E2 packer suite instead of re-deriving a second one: this
# file's new ground is the WRITE-BACK, not the packing, so the vector case
# below drives the SAME body those rows already certify.
from test_plan_packer_e2 import (  # noqa: E402
    _VEC3_SPEC,
    _device_plugin,
    _gxx,
    _host_plugin,
    _nvcc,
    _x,
    e2_device_module,  # noqa: F401 -- reused as a pytest fixture, by parameter name
)

# --------------------------------------------------------------------------- #
# A minimal scalar (width-1) ``out`` kernel: y = a*x + b -- the landing page's
# own ``scale`` body, by hand, so this file needs neither hawk nor a GPU
# compile of hawk's own artifact to reproduce the bug generically.
# --------------------------------------------------------------------------- #
_SCALE_HOST_SRC = r"""
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

extern "C" const std::uint64_t eagle_layout_sizes[5] = {
    sizeof(GRefMirror), sizeof(ScalarHandle), sizeof(IntHandle),
    sizeof(EAGLE_ABI_INDEX_T), sizeof(PartitionTriple),
};

extern "C" void scale_host(void* const* params, std::int64_t base,
                           std::int64_t count, std::int64_t /*nSamples*/)
{
    const GRefMirror* yv = static_cast<const GRefMirror*>(params[0]);
    const GRefMirror* xv = static_cast<const GRefMirror*>(params[1]);
    double* y       = reinterpret_cast<double*>(yv->data_);
    const double* x = reinterpret_cast<const double*>(xv->data_);
    const double a  = *static_cast<const double*>(params[2]);
    const double b  = *static_cast<const double*>(params[3]);
    for (std::int64_t i = base; i < base + count; ++i)
        y[i * yv->sampleStride_] = a * x[i * xv->sampleStride_] + b;
}
"""

_SCALE_DEVICE_SRC = r"""
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

extern "C" __device__ unsigned long long eagle_layout_sizes[5] = {
    sizeof(GRefMirror), sizeof(ScalarHandle), sizeof(IntHandle),
    sizeof(EAGLE_ABI_INDEX_T), sizeof(PartitionTriple),
};

__device__ __forceinline__ bool sample_index(long long base, long long count,
                                             long long& i)
{
    const long long flat =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return false;
    i = base + flat;
    return true;
}

extern "C" __global__ void scale(GRefMirror y, GRefMirror x, double a, double b,
                                 std::uint32_t /*n*/, long long base,
                                 long long count, long long /*nSamples*/)
{
    long long i;
    if (!sample_index(base, count, i)) return;
    reinterpret_cast<double*>(y.data_)[i * y.sampleStride_] =
        a * reinterpret_cast<const double*>(x.data_)[i * x.sampleStride_] + b;
}
"""

_SCALE_SPEC = (("out", "y"), ("vec_in", "x"), ("uniform", "a"), ("uniform", "b"),
               ("nsamples", "n"))


@pytest.fixture(scope="module")
def scale_host_lib(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("scale_out_host")
    src = tmp / "scale_host.cpp"
    src.write_text(_SCALE_HOST_SRC)
    so = tmp / "scale_host.so"
    proc = subprocess.run(
        [_gxx(), "-O2", "-std=c++17", "-shared", "-fPIC", f"-I{EAGLE_ROOT}",
         "-o", str(so), str(src)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"scale host fixture compile failed:\n{proc.stderr}"
    return ctypes.CDLL(str(so))


@pytest.fixture(scope="module")
def scale_device_module(tmp_path_factory):
    import cupy as cp

    tmp = tmp_path_factory.mktemp("scale_out_device")
    cu = tmp / "scale_device.cu"
    cu.write_text(_SCALE_DEVICE_SRC)
    cubin = tmp / "scale_device.cubin"
    cc = cp.cuda.Device(0).compute_capability
    proc = subprocess.run(
        [_nvcc(), "-cubin", "-std=c++17", f"-arch=sm_{cc}", f"-I{EAGLE_ROOT}",
         str(cu), "-o", str(cubin)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, f"scale device fixture compile failed:\n{proc.stderr}"
    return cp.RawModule(path=str(cubin))


def _expected_scale(x, a, b):
    return a * x + b


# --------------------------------------------------------------------------- #
# Row 1 -- the bug exactly as reported: a native, contiguous, numpy ``y``
# supplied to a DEVICE plan is now written through, not just returned.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
@pytest.mark.gpu
def test_device_plan_writes_native_scalar_y_back(scale_device_module):
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, a, b = 8, 2.0, 1.0
    x = np.arange(n, dtype=np.float64)
    y = np.zeros(n)
    plugin = _device_plugin(scale_device_module, "scale", _SCALE_SPEC,
                            arg_widths={"y": 1})

    result = eplan.plan(plugin, structure=eexec.DeviceKernel).run(x=x, a=a, b=b, y=y)

    expected = _expected_scale(x, a, b)
    np.testing.assert_array_equal(y, expected)          # written into the CALLER's y
    np.testing.assert_array_equal(result, y)             # returned value == written y


# --------------------------------------------------------------------------- #
# Row 2 -- the HOST twin: a non-contiguous native ``y`` (the one HostTeam case
# ``np.ascontiguousarray``'s identity shortcut does NOT cover -- it copies,
# so the pre-fix code wrote the answer into a throwaway buffer).
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
def test_host_plan_writes_a_noncontiguous_native_y_back(scale_host_lib):
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, a, b = 8, -3.0, 0.5
    x = np.arange(n, dtype=np.float64)
    y_buf = np.zeros(2 * n)
    y = y_buf[::2]                      # native shape (n,), strided: NOT contiguous
    assert not y.flags["C_CONTIGUOUS"]
    plugin = _host_plugin(scale_host_lib, "scale_host", _SCALE_SPEC,
                          arg_widths={"y": 1})

    result = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x, a=a, b=b, y=y)

    expected = _expected_scale(x, a, b)
    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(y_buf[1::2], 0.0)      # the untouched lanes untouched
    np.testing.assert_array_equal(result, y)


# --------------------------------------------------------------------------- #
# Row 3 -- the width>1 case: a native (w, N) ``y`` is the SAME gap (``_adapt_
# planes`` never touches it, so it was never routed through the sample-major
# fix either); a sample-major (N, w) ``y`` is folded into the SAME test as the
# non-regression half -- it already worked (``test_layout.py``'s own device
# row), and stays correct here too.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
@pytest.mark.gpu
def test_device_plan_writes_vector_y_back_native_and_sample_major(
    e2_device_module,  # noqa: F811 -- the fixture imported above, by parameter name
):
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, a = 16, 2.0
    x = _x(n)
    expected = a * x + np.arange(3, dtype=np.float64)[:, None]
    plugin = _device_plugin(e2_device_module, "e2_vec3", _VEC3_SPEC,
                            arg_widths={"y": 3})

    y_native = np.zeros((3, n))
    result = eplan.plan(plugin, structure=eexec.DeviceKernel).run(x=x, a=a, y=y_native)
    np.testing.assert_array_equal(y_native, expected)           # THE fix
    np.testing.assert_array_equal(result, y_native)

    y_major = np.zeros((n, 3))
    with pytest.warns(eagle.LayoutWarning):
        result2 = eplan.plan(
            plugin, structure=eexec.DeviceKernel
        ).run(x=x, a=a, y=y_major)
    np.testing.assert_array_equal(y_major, expected.T)          # unaffected by the fix
    np.testing.assert_array_equal(result2, y_major)


# --------------------------------------------------------------------------- #
# Row 4 -- a read-only supplied output is refused, naming the argument,
# instead of silently staying untouched.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
@pytest.mark.gpu
def test_device_plan_refuses_a_readonly_supplied_y(scale_device_module):
    import eagle.exec as eexec
    from eagle import plan as eplan

    n = 8
    x = np.arange(n, dtype=np.float64)
    y = np.zeros(n)
    y.flags.writeable = False
    plugin = _device_plugin(scale_device_module, "scale", _SCALE_SPEC,
                            arg_widths={"y": 1})

    with pytest.raises(ValueError, match=r"'y'.*read-only"):
        eplan.plan(plugin, structure=eexec.DeviceKernel).run(x=x, a=1.0, b=0.0, y=y)


@pytest.mark.repo_local
def test_host_plan_refuses_a_readonly_supplied_y(scale_host_lib):
    import eagle.exec as eexec
    from eagle import plan as eplan

    n = 8
    x = np.arange(n, dtype=np.float64)
    y = np.zeros(n)
    y.flags.writeable = False
    plugin = _host_plugin(scale_host_lib, "scale_host", _SCALE_SPEC,
                          arg_widths={"y": 1})

    with pytest.raises(ValueError, match=r"'y'.*read-only"):
        eplan.plan(plugin, structure=eexec.HostTeam).run(x=x, a=1.0, b=0.0, y=y)


# --------------------------------------------------------------------------- #
# Row 5 -- the exact landing-page snippet (amasat01.github.io/index.md,
# "Thirty seconds" section), copied verbatim past the imports: the literal
# bug report, end to end through hawk + a real compiled artifact.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
@pytest.mark.gpu
def test_landing_page_snippet_leaves_y_equal_to_result():
    import pathlib as _pathlib
    import tempfile
    from types import SimpleNamespace

    hawk = pytest.importorskip("hawk")
    from hawk import Mutable, Param, Scalar
    from hawk.artifact import build_bundle, plan_view

    import eagle.exec as eexec
    from eagle import plan as eplan
    from eagle.registry import load_manifest

    @hawk.kernel
    def scale(x: Scalar, a: Param, b: Param, y: Mutable[Scalar]):
        y = a * x + b

    bundle_dir = _pathlib.Path(tempfile.mkdtemp()) / "bundle"
    bundle = build_bundle([scale], bundle_dir, targets=("host", "cuda"))
    sidecar = next(art.sidecar for art in bundle.artifacts if art.name == "scale")
    registry = load_manifest(bundle_dir / "manifest.json")
    loaded = registry["scale"]
    view = plan_view(sidecar)
    view.pop("host_entry", None)
    plugin = SimpleNamespace(device_function=loaded.fn.kernel.ptr,
                              _keepalive=(registry, loaded), **view)

    y = np.zeros(8)
    result = eplan.plan(plugin, structure=eexec.DeviceKernel).run(
        x=np.arange(8.0), a=2.0, b=1.0, y=y
    )

    np.testing.assert_array_equal(y, np.array([1., 3., 5., 7., 9., 11., 13., 15.]))
    np.testing.assert_array_equal(result, y)
