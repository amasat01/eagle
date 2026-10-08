# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The three ``eagle.plan`` packer extensions this file exercises: the
GRef-shaped roles, dtype-parametric planes, and a multi-output return.

Each row below was RED against the pre-existing packer, which (a) refused every
GRef-shaped role outright ("arg_spec role 'out' is not supported by this
packer yet"), (b) hard-coded ``float64`` for both the allocated output
planes and the ``uniform`` box, and (c) returned only the FIRST output plane
from ``Plan.run``.

FIXTURES follow the file-local convention of ``test_exec_contract_rows.py`` /
``test_arg_classify.py``: a single self-contained TU per target, compiled at
test time. They differ from those in ONE deliberate way — they ``#include
"plugin/gref_abi.h"`` (the repo's own header, via ``-I <eagle root>``) instead
of re-declaring the PODs locally, because the byte-layout row's whole subject
is the agreement between what Python packs and what the C++ header DEFINES;
a hand-copied struct in the fixture would fix the copy, not the contract
(a generated TU includes that same header). The fixtures mirror
``tests/fixtures/{host_plugin_execv2.cpp,device_plugin_execv2.cu}``'s shape:
the int64 partition triple after the role args, a serial host range / flat
device index, and the ``eagle_layout_sizes`` export.
"""

from __future__ import annotations

import ctypes
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

EAGLE_ROOT = pathlib.Path(__file__).resolve().parents[2]  # holds plugin/gref_abi.h

# --------------------------------------------------------------------------- #
# The fixture bodies. arg_specs (mirroring the exemplars' comment style):
#   e2_vec3      [out y, vec_in x, uniform a, nsamples n]   y[c][i] = a*x[c][i] + c
#   e2_vec3_f32  the same body at float32
#   e2_multi     [out y, wide_out w, vec_in x, nsamples n]  TWO output planes
#   e2_decode    [wide_out probe, vec_in x, nsamples n]     the mirror, decoded
#                                                           BY THE C++ HEADER
# --------------------------------------------------------------------------- #
_HOST_SRC = r"""
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

// The layout self-check export, derived from this TU's own PODs (the
// exemplars derive it the same way).
extern "C" const std::uint64_t eagle_layout_sizes[5] = {
    sizeof(GRefMirror), sizeof(ScalarHandle), sizeof(IntHandle),
    sizeof(EAGLE_ABI_INDEX_T), sizeof(PartitionTriple),
};

// The GRefMirror field offsets, straight out of the header — what the packed
// Python box's field offsets are asserted against (the byte-layout row).
extern "C" const std::uint64_t e2_gref_offsets[6] = {
    offsetof(GRefMirror, data_),         offsetof(GRefMirror, samples_),
    offsetof(GRefMirror, compStride_),   offsetof(GRefMirror, sampleStride_),
    offsetof(GRefMirror, deviceType_),   offsetof(GRefMirror, deviceId_),
};

static constexpr std::uint64_t kWidth = 3;

extern "C" void e2_vec3_host(void* const* params, std::int64_t base,
                             std::int64_t count, std::int64_t /*nSamples*/)
{
    const GRefMirror* yv = static_cast<const GRefMirror*>(params[0]);
    const GRefMirror* xv = static_cast<const GRefMirror*>(params[1]);
    double* y            = reinterpret_cast<double*>(yv->data_);
    const double* x      = reinterpret_cast<const double*>(xv->data_);
    const double a       = *static_cast<const double*>(params[2]);
    for (std::int64_t i = base; i < base + count; ++i)
        for (std::uint64_t c = 0; c < kWidth; ++c)
            y[c * yv->compStride_ + i * yv->sampleStride_] =
                a * x[c * xv->compStride_ + i * xv->sampleStride_] + double(c);
}

// float32 twin: the MIRROR layout is unchanged (it is a view descriptor, not a
// value), only the pointee type and the by-value uniform's width follow the
// artifact's declared scalar mode.
extern "C" void e2_vec3_f32_host(void* const* params, std::int64_t base,
                                 std::int64_t count, std::int64_t /*nSamples*/)
{
    const GRefMirror* yv = static_cast<const GRefMirror*>(params[0]);
    const GRefMirror* xv = static_cast<const GRefMirror*>(params[1]);
    float* y             = reinterpret_cast<float*>(yv->data_);
    const float* x       = reinterpret_cast<const float*>(xv->data_);
    const float a        = *static_cast<const float*>(params[2]);
    for (std::int64_t i = base; i < base + count; ++i)
        for (std::uint64_t c = 0; c < kWidth; ++c)
            y[c * yv->compStride_ + i * yv->sampleStride_] =
                a * x[c * xv->compStride_ + i * xv->sampleStride_] + float(c);
}

extern "C" void e2_multi_host(void* const* params, std::int64_t base,
                              std::int64_t count, std::int64_t /*nSamples*/)
{
    const GRefMirror* yv = static_cast<const GRefMirror*>(params[0]);
    double* y            = reinterpret_cast<double*>(yv->data_);
    double* w = static_cast<double*>(static_cast<const ScalarHandle*>(params[1])->data);
    const GRefMirror* xv = static_cast<const GRefMirror*>(params[2]);
    const double* x      = reinterpret_cast<const double*>(xv->data_);
    for (std::int64_t i = base; i < base + count; ++i) {
        for (std::uint64_t c = 0; c < kWidth; ++c)
            y[c * yv->compStride_ + i * yv->sampleStride_] =
                x[c * xv->compStride_ + i * xv->sampleStride_] + double(c);
        w[i] = 10.0 * x[i * xv->sampleStride_];
    }
}

// The mirror DECODED by the C++ header's own struct definition: whatever
// Python packed lands here at the header's offsets, and the body reports the
// fields back through an ordinary output plane.
extern "C" void e2_decode_host(void* const* params, std::int64_t /*base*/,
                               std::int64_t /*count*/, std::int64_t /*nSamples*/)
{
    double* probe = static_cast<double*>(
        static_cast<const ScalarHandle*>(params[0])->data);
    const GRefMirror* xv = static_cast<const GRefMirror*>(params[1]);
    probe[0] = double(sizeof(GRefMirror));
    probe[1] = double(xv->samples_);
    probe[2] = double(xv->compStride_);
    probe[3] = double(xv->sampleStride_);
    probe[4] = double(xv->deviceType_);
    probe[5] = double(xv->deviceId_);
    probe[6] = double(reinterpret_cast<std::uint64_t>(xv->data_));
    // ... and the pointer actually resolves: the LAST component's first sample,
    // reached through the decoded strides only.
    probe[7] = reinterpret_cast<const double*>(xv->data_)[2 * xv->compStride_];
}
"""

_DEVICE_SRC = r"""
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

extern "C" __device__ unsigned long long eagle_layout_sizes[5] = {
    sizeof(GRefMirror), sizeof(ScalarHandle), sizeof(IntHandle),
    sizeof(EAGLE_ABI_INDEX_T), sizeof(PartitionTriple),
};

static constexpr unsigned long long kWidth = 3;

// The ONE index seam: flat index, early-out on `count`, global i = base + flat.
__device__ __forceinline__ bool sample_index(long long base, long long count,
                                             long long& i)
{
    const long long flat =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return false;
    i = base + flat;
    return true;
}

extern "C" __global__ void e2_vec3(GRefMirror y, GRefMirror x, double a,
                                   std::uint32_t /*n*/, long long base,
                                   long long count, long long /*nSamples*/)
{
    long long i;
    if (!sample_index(base, count, i)) return;
    for (unsigned long long c = 0; c < kWidth; ++c)
        reinterpret_cast<double*>(y.data_)[c * y.compStride_ + i * y.sampleStride_] =
            a * reinterpret_cast<const double*>(x.data_)
                    [c * x.compStride_ + i * x.sampleStride_] +
            double(c);
}

extern "C" __global__ void e2_vec3_f32(GRefMirror y, GRefMirror x, float a,
                                       std::uint32_t /*n*/, long long base,
                                       long long count, long long /*nSamples*/)
{
    long long i;
    if (!sample_index(base, count, i)) return;
    for (unsigned long long c = 0; c < kWidth; ++c)
        reinterpret_cast<float*>(y.data_)[c * y.compStride_ + i * y.sampleStride_] =
            a * reinterpret_cast<const float*>(x.data_)
                    [c * x.compStride_ + i * x.sampleStride_] +
            float(c);
}

extern "C" __global__ void e2_multi(GRefMirror y, ScalarHandle w, GRefMirror x,
                                    std::uint32_t /*n*/, long long base,
                                    long long count, long long /*nSamples*/)
{
    long long i;
    if (!sample_index(base, count, i)) return;
    const double* xp = reinterpret_cast<const double*>(x.data_);
    for (unsigned long long c = 0; c < kWidth; ++c)
        reinterpret_cast<double*>(y.data_)[c * y.compStride_ + i * y.sampleStride_] =
            xp[c * x.compStride_ + i * x.sampleStride_] + double(c);
    static_cast<double*>(w.data)[i] = 10.0 * xp[i * x.sampleStride_];
}
"""


# --------------------------------------------------------------------------- #
# Fixture plumbing.
# --------------------------------------------------------------------------- #
class _E2Plugin:
    """The minimal duck-typed v2 plugin ``eagle.plan`` drives: an
    ``arg_spec``, an ``exec_access`` declaration, the declared ``scalar_type``
    and output-plane ``arg_widths``, and one entry point per target."""

    exec_access = "sample_local"
    exec_op = None

    def __init__(self, arg_spec, *, scalar_type="float64", arg_widths=None,
                 host_entry=None, device_function=None, keepalive=None):
        self.arg_spec = tuple(arg_spec)
        self.scalar_type = scalar_type
        self.arg_widths = dict(arg_widths or {})
        if host_entry is not None:
            self.host_entry = host_entry
        if device_function is not None:
            self.device_function = device_function
        self._keepalive = keepalive


def _gxx() -> str:
    found = shutil.which(os.environ.get("CXX", "")) or shutil.which("g++")
    if found is None and os.path.exists("/usr/bin/g++"):
        found = "/usr/bin/g++"
    if found is None:
        raise RuntimeError("the E2 packer rows need g++ to build the host fixture")
    return found


def _nvcc() -> str:
    found = shutil.which("nvcc")
    if found:
        return found
    cand = pathlib.Path(os.environ.get("CUDA_PATH", "/usr/local/cuda")) / "bin" / "nvcc"
    if cand.exists():
        return str(cand)
    raise RuntimeError("the E2 device rows need nvcc to build the device fixture")


@pytest.fixture(scope="module")
def e2_host_lib(tmp_path_factory):
    """The compiled host fixture (``ctypes.CDLL``) — one TU, all host entries."""
    tmp = tmp_path_factory.mktemp("e2_packer_host")
    src = tmp / "e2_host.cpp"
    src.write_text(_HOST_SRC)
    so = tmp / "e2_host.so"
    proc = subprocess.run(
        [_gxx(), "-O2", "-std=c++17", "-shared", "-fPIC", f"-I{EAGLE_ROOT}",
         "-o", str(so), str(src)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"E2 host fixture compile failed:\n{proc.stderr}"
    return ctypes.CDLL(str(so))


@pytest.fixture(scope="module")
def e2_device_module(tmp_path_factory):
    """The compiled device fixture (a cupy ``RawModule`` over a cubin built for
    the ACTUAL visible device's SM).

    A ``-ptx`` build embeds whichever nvcc built it; loading raw PTX always
    goes through the driver's JIT, which rejects a newer nvcc's PTX outright
    (``CUDA_ERROR_UNSUPPORTED_PTX_VERSION``) even when the target SM is one the
    installed driver fully supports. A ``-cubin`` build for the device's own SM
    is real SASS and loads directly — no JIT, so the nvcc/driver version
    pairing cannot matter.
    """
    import cupy as cp

    tmp = tmp_path_factory.mktemp("e2_packer_device")
    cu = tmp / "e2_device.cu"
    cu.write_text(_DEVICE_SRC)
    cubin = tmp / "e2_device.cubin"
    cc = cp.cuda.Device(0).compute_capability
    proc = subprocess.run(
        [_nvcc(), "-cubin", "-std=c++17", f"-arch=sm_{cc}", f"-I{EAGLE_ROOT}",
         str(cu), "-o", str(cubin)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"E2 device fixture compile failed:\n{proc.stderr}"
    return cp.RawModule(path=str(cubin))


def _host_plugin(lib, symbol, arg_spec, **kw):
    fn = getattr(lib, symbol)
    return _E2Plugin(
        arg_spec, host_entry=ctypes.cast(fn, ctypes.c_void_p).value, **kw
    )


def _device_plugin(module, symbol, arg_spec, **kw):
    kernel = module.get_function(symbol)
    return _E2Plugin(
        arg_spec, device_function=kernel.kernel.ptr, keepalive=(module, kernel), **kw
    )


_VEC3_SPEC = (("out", "y"), ("vec_in", "x"), ("uniform", "a"), ("nsamples", "n"))
_MULTI_SPEC = (
    ("out", "y"),
    ("wide_out", "w"),
    ("vec_in", "x"),
    ("nsamples", "n"),
)


def _x(n, dtype=np.float64):
    return np.arange(3 * n, dtype=dtype).reshape(3, n) * 0.5


# --------------------------------------------------------------------------- #
# Row 1 — the GRef-shaped roles pack as the 40-byte mirror.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
def test_e2_gref_roles_pack_as_the_mirror_host(e2_host_lib):
    """``out``/``vec_in`` (``classify_arg`` -> ``GREF_VEC``) cross as the
    40-byte ``GRefMirror``, extents/strides taken from the bound array — the
    body reads its own ``compStride_``/``sampleStride_``, so a wrongly-shaped
    mirror cannot produce the right answer."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, a = 16, 2.0
    x = _x(n)
    plugin = _host_plugin(e2_host_lib, "e2_vec3_host", _VEC3_SPEC,
                          arg_widths={"y": 3})

    y = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x, a=a)

    expected = a * x + np.arange(3, dtype=np.float64)[:, None]
    assert y.shape == (3, n)
    np.testing.assert_array_equal(y, expected)


@pytest.mark.repo_local
def test_e2_gref_roles_pack_under_a_split_host(e2_host_lib):
    """The mirror is passed WHOLE while the partition triple slices the samples
    two partitions must reproduce the whole-view answer bit for bit."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, a = 16, -1.25
    x = _x(n)
    plugin = _host_plugin(e2_host_lib, "e2_vec3_host", _VEC3_SPEC,
                          arg_widths={"y": 3})

    whole = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x, a=a)
    split = eplan.plan(
        plugin, structure=eexec.HostTeam, partitions=[(0, 6, n), (6, n - 6, n)]
    ).run(x=x, a=a)

    np.testing.assert_array_equal(whole, split)


@pytest.mark.repo_local
@pytest.mark.gpu
def test_e2_gref_roles_pack_as_the_mirror_device(e2_device_module):
    """The device arm of the row above: the SAME 40-byte mirror rides
    ``cuLaunchKernel``'s ``kernelParams`` by value."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, a = 16, 2.0
    x = _x(n)
    plugin = _device_plugin(e2_device_module, "e2_vec3", _VEC3_SPEC,
                            arg_widths={"y": 3})

    y = eplan.plan(plugin, structure=eexec.DeviceKernel).run(x=x, a=a)

    expected = a * x + np.arange(3, dtype=np.float64)[:, None]
    assert y.shape == (3, n)
    np.testing.assert_array_equal(y, expected)


# --------------------------------------------------------------------------- #
# Row 2 — dtype-parametric planes (the artifact's declared scalar mode).
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
def test_e2_float32_planes_follow_the_declared_scalar_type_host(e2_host_lib):
    """A ``scalar_type='float32'`` artifact gets ``float32`` planes and a
    4-byte ``uniform`` box; the packer previously allocated ``float64`` planes and
    a ``c_double`` uniform regardless of the declaration."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, a = 16, 2.0
    x = _x(n)
    plugin = _host_plugin(e2_host_lib, "e2_vec3_f32_host", _VEC3_SPEC,
                          scalar_type="float32", arg_widths={"y": 3})

    y = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x, a=a)

    assert y.dtype == np.float32
    expected = (a * x.astype(np.float32)
                + np.arange(3, dtype=np.float32)[:, None]).astype(np.float32)
    np.testing.assert_array_equal(y, expected)


@pytest.mark.repo_local
def test_e2_float64_artifact_keeps_float64_planes(e2_host_lib):
    """The default arm of the row above: an artifact declaring nothing (or
    ``float64``) is unchanged — dtype-parametric, not dtype-guessed."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n = 16
    plugin = _host_plugin(e2_host_lib, "e2_vec3_host", _VEC3_SPEC,
                          arg_widths={"y": 3})
    y = eplan.plan(plugin, structure=eexec.HostTeam).run(x=_x(n), a=1.0)
    assert y.dtype == np.float64


@pytest.mark.repo_local
@pytest.mark.gpu
def test_e2_float32_planes_follow_the_declared_scalar_type_device(e2_device_module):
    """The device arm: ``cp.zeros(..., dtype=cp.float64)`` was hard-coded."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n, a = 16, 2.0
    x = _x(n)
    plugin = _device_plugin(e2_device_module, "e2_vec3_f32", _VEC3_SPEC,
                            scalar_type="float32", arg_widths={"y": 3})

    y = eplan.plan(plugin, structure=eexec.DeviceKernel).run(x=x, a=a)

    assert y.dtype == np.float32
    expected = (a * x.astype(np.float32)
                + np.arange(3, dtype=np.float32)[:, None]).astype(np.float32)
    np.testing.assert_array_equal(y, expected)


# --------------------------------------------------------------------------- #
# Row 3 — every output plane comes back, in arg_spec order.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
def test_e2_multi_output_returns_every_plane_host(e2_host_lib):
    """A body with TWO declared output roles returns BOTH planes, in a dict
    keyed by plane name (never a positional tuple -- ``arg_spec``'s own
    order is role-grouped then name-sorted, not the kernel's authored
    order); ``_output_role`` previously returned the first only."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n = 16
    x = _x(n)
    plugin = _host_plugin(e2_host_lib, "e2_multi_host", _MULTI_SPEC,
                          arg_widths={"y": 3})

    result = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x)

    assert isinstance(result, dict) and set(result) == {"y", "w"}
    y, w = result["y"], result["w"]
    np.testing.assert_array_equal(y, x + np.arange(3, dtype=np.float64)[:, None])
    np.testing.assert_array_equal(w, 10.0 * x[0])


@pytest.mark.repo_local
def test_e2_single_output_still_returns_a_bare_array(e2_host_lib):
    """The compatibility arm: ONE output role still returns the plane itself,
    never a 1-tuple (``test_exec_contract_rows.py``'s two rows compare the
    result to a numpy array directly)."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    plugin = _host_plugin(e2_host_lib, "e2_vec3_host", _VEC3_SPEC,
                          arg_widths={"y": 3})
    y = eplan.plan(plugin, structure=eexec.HostTeam).run(x=_x(8), a=1.0)
    assert isinstance(y, np.ndarray)


@pytest.mark.repo_local
@pytest.mark.gpu
def test_e2_multi_output_returns_every_plane_device(e2_device_module):
    """The device arm of the multi-output row."""
    import eagle.exec as eexec
    from eagle import plan as eplan

    n = 16
    x = _x(n)
    plugin = _device_plugin(e2_device_module, "e2_multi", _MULTI_SPEC,
                            arg_widths={"y": 3})

    result = eplan.plan(plugin, structure=eexec.DeviceKernel).run(x=x)
    y, w = result["y"], result["w"]

    np.testing.assert_array_equal(y, x + np.arange(3, dtype=np.float64)[:, None])
    np.testing.assert_array_equal(w, 10.0 * x[0])


# --------------------------------------------------------------------------- #
# Row 4 — the packed mirror's BYTES against the C++ header's own sizes/offsets.
# --------------------------------------------------------------------------- #
@pytest.mark.repo_local
def test_e2_packed_mirror_bytes_match_the_cpp_layout(e2_host_lib):
    """The box :func:`eagle.plan._pack_args` packs for a known array is
    exactly what the C++ ``GRefMirror`` decodes — its size and field offsets
    equal the fixture's own ``eagle_layout_sizes``/``offsetof`` exports (read
    out of the compiled ``.so``), and the fixture body, reading the bytes
    THROUGH that struct definition, reports back the extents, strides, device
    tag and data pointer the packer put in."""
    import eagle.exec as eexec
    from eagle import plan as eplan
    from eagle.abi import DEVICE_CPU
    from eagle.host_launch import GRefMirror

    sizes = (ctypes.c_uint64 * 5).in_dll(e2_host_lib, "eagle_layout_sizes")
    offsets = (ctypes.c_uint64 * 6).in_dll(e2_host_lib, "e2_gref_offsets")

    # The fixture's layout table agrees with this build's compiled core.
    eexec.check_layout_sizes(list(sizes))
    assert ctypes.sizeof(GRefMirror) == sizes[0] == 40
    assert [GRefMirror.__dict__[f].offset for f, _ in GRefMirror._fields_] == list(
        offsets
    )

    n = 16
    x = np.ascontiguousarray(_x(n))
    plugin = _host_plugin(
        e2_host_lib,
        "e2_decode_host",
        (("wide_out", "probe"), ("vec_in", "x"), ("nsamples", "n")),
    )
    probe = eplan.plan(plugin, structure=eexec.HostTeam).run(x=x)

    assert probe[0] == 40                      # sizeof(GRefMirror), C++ side
    assert probe[1] == n                       # samples_ = the run's sample count
    assert probe[2] == n                       # compStride_ = the (3, n) row pitch
    assert probe[3] == 1                       # sampleStride_
    assert probe[4] == DEVICE_CPU              # deviceType_
    assert probe[5] == 0                       # deviceId_
    assert int(probe[6]) == x.ctypes.data      # data_
    assert probe[7] == x[2, 0]                 # ... and it resolves to the data


# --------------------------------------------------------------------------- #
# Row 5 — the WHOLE role vocabulary packs, each at its own shape.
# --------------------------------------------------------------------------- #
def test_e2_every_role_packs_at_its_pinned_mirror_shape():
    """Every one of the 12 canonical roles resolves
    through :func:`eagle.roles.classify_arg` to a box of the pinned size — 40 B
    ``GRefMirror`` for the GRef-shaped roles, 32 B ``ScalarHandle`` for the
    handle-shaped ones, a by-value scalar for ``uniform``, 4 B for ``nsamples``
    and NO role is refused. A role added to the vocabulary without a
    packer branch fails here, not at a deployment."""
    from eagle import plan as eplan
    from eagle.roles import ROLES

    spec = [(role, f"{role}_arg") for role in sorted(ROLES)]
    values = {
        f"{role}_arg": (np.zeros((3, 8)) if role in ("out", "vec_in", "mat_in")
                        else np.zeros(8))
        for role in sorted(ROLES)
    }
    kw = {"uniform_arg": 1.0}
    expected = {
        "out": 40, "vec_in": 40, "mat_in": 40,
        "per_sample": 32, "lookup": 32, "terminated": 32, "mutable": 32,
        "wide_in": 32, "wide_out": 32, "accum_out": 32,
        "uniform": 8, "nsamples": 4,
    }

    boxes, addrs = eplan._pack_args(spec, values, kw, 8, 1)

    assert len(boxes) == len(addrs) == len(spec)
    assert {role: ctypes.sizeof(b) for (role, _), b in zip(spec, boxes)} == expected

    # ``mutable`` resolves BY CONTEXT: declared vector-/matrix-shaped, it
    # is the 40-byte mirror, not the 32-byte handle.
    for declared in ("vec_mutables", "mat_mutables"):
        boxes, _ = eplan._pack_args(
            [("mutable", "m")], {"m": np.zeros((3, 8))}, {}, 8, 1,
            **{declared: frozenset({"m"})},
        )
        assert ctypes.sizeof(boxes[0]) == 40


def test_e2_uniform_box_width_follows_the_declared_scalar_type():
    """The ``uniform`` role crosses at the artifact's declared width — 8 bytes
    for ``float64``/``softdouble``, 4 for ``float32`` — not always a
    ``ctypes.c_double``."""
    from eagle import plan as eplan

    spec = [("uniform", "a"), ("nsamples", "n")]
    for scalar_type, want in (("float64", 8), ("softdouble", 8), ("float32", 4)):
        dtype = eplan._plane_dtype(_E2Plugin(spec, scalar_type=scalar_type))
        boxes, _ = eplan._pack_args(spec, {}, {"a": 1.5}, 8, 1, dtype=dtype)
        assert ctypes.sizeof(boxes[0]) == want
        assert ctypes.sizeof(boxes[1]) == 4  # unchanged by the scalar mode


def test_e2_integer_uniform_boxes_as_long_long_only_when_declared():
    """An INTEGER ``uniform`` (sidecar v2 ``params`` entry ``dtype: "int"``)
    crosses as ``c_longlong`` — the ``long long`` its kernel takes — and a
    float one keeps its declared Real width. RED before the fix: the integer
    packed as ``c_double`` and read the double's bit
    pattern back on the device (7 -> 4619567317775286272)."""
    from types import SimpleNamespace

    import numpy as np

    from eagle import plan as eplan

    params = [{"name": "k", "dtype": "int"}, {"name": "mu", "dtype": "float"}]
    assert eplan._integer_uniforms(SimpleNamespace(params=params)) == {"k"}
    assert eplan._integer_uniforms(SimpleNamespace(params=["k", "mu"])) == frozenset()
    assert eplan._integer_uniforms(SimpleNamespace()) == frozenset()
    spec = [("uniform", "k"), ("uniform", "mu"), ("nsamples", "n")]
    kw = {"k": 7, "mu": 1.5}
    boxes, _ = eplan._pack_args(spec, {}, kw, 8, 1, dtype=np.dtype(np.float64),
                                int_uniforms=frozenset({"k"}))
    assert isinstance(boxes[0], ctypes.c_longlong) and boxes[0].value == 7
    assert isinstance(boxes[1], ctypes.c_double) and boxes[1].value == 1.5
    boxes, _ = eplan._pack_args(spec, {}, kw, 8, 1, dtype=np.dtype(np.float64))
    assert isinstance(boxes[0], ctypes.c_double)  # undeclared stays float: unchanged
