# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""torch-CPU node I/O: drive a dlopen'd CPU plugin over host tensors.

Compiles a tiny host plugin ``.so`` with g++ (the code generator's host-emission
toolchain in miniature — ABI-only), then runs it through
:class:`eagle.host_launch.HostPluginLibrary` over numpy arrays and torch **CPU**
tensors, zero-copy via :func:`eagle.interop.host_ptr`. Proves the CPU-plugin ABI
end to end from Python without cupy or CUDA: the plugin writes the caller's torch
tensor in place.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import numpy as np
import pytest

from eagle import interop
from eagle.host_launch import GRefMirror, HostPluginLibrary, ScalarHandle

# A self-contained host plugin: out[i] = a[i] + b[i]. Depends only on the ABI POD
# (a plain `struct ScalarHandle { void* ptr; }`) — exactly what a deployed plugin
# needs, no eagle / aether / CUDA headers. Its own OpenMP loop owns the parallelism.
_PLUGIN_SRC = r"""
struct ScalarHandle { void* ptr; };
extern "C" void addvec_host(void* const* p, int n) {
    double* out      = (double*)((const ScalarHandle*)p[0])->ptr;
    const double* a  = (const double*)((const ScalarHandle*)p[1])->ptr;
    const double* b  = (const double*)((const ScalarHandle*)p[2])->ptr;
#pragma omp parallel for
    for (int i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}
"""

# The sidecar the launcher packs against — what a code-generator deploy would stamp
# for a pure kernel: a scalar `mutable` output, two `per_sample` inputs, then
# `nsamples`.
_SIDECAR = {
    "kernel": "addvec",
    "aether_abi": "aether-abi/1",
    "host_entry": "addvec_host",
    "arg_spec": [
        ["mutable", "out"],
        ["per_sample", "a"],
        ["per_sample", "b"],
        ["nsamples", "n"],
    ],
    "mutables": [{"name": "out", "dtype": "float", "width": 1}],
}


def _gxx() -> str | None:
    if shutil.which(os.environ.get("CXX", "")):
        return os.environ.get("CXX")
    return shutil.which("g++") or (
        "/usr/bin/g++" if os.path.exists("/usr/bin/g++") else None
    )


def _compile_plugin(tmp_path) -> str:
    gxx = _gxx()
    if gxx is None:
        pytest.skip("no g++ available to build the host plugin fixture")
    src = tmp_path / "addvec_host.cpp"
    src.write_text(_PLUGIN_SRC)
    so = tmp_path / "addvec_host.so"
    proc = subprocess.run(
        [gxx, "-O3", "-shared", "-fPIC", "-fopenmp", "-o", str(so), str(src)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"host plugin compile failed: {proc.stderr.strip()[:400]}")
    return str(so)


def test_abi_pod_layouts_match_cpp():
    # The ctypes mirrors must match the C++ static_asserts in plugin/gref_abi.h.
    import ctypes

    assert ctypes.sizeof(GRefMirror) == 40
    assert ctypes.sizeof(ScalarHandle) == 32


def test_host_plugin_over_numpy(tmp_path):
    so = _compile_plugin(tmp_path)
    n = 4096
    a = np.arange(n, dtype=np.float64)
    b = 2.0 * np.arange(n, dtype=np.float64)
    out = np.full(n, -1.0, dtype=np.float64)

    lib = HostPluginLibrary(so, _SIDECAR)
    lib.bind_handle("out", interop.host_ptr(out))
    lib.bind_handle("a", interop.host_ptr(a))
    lib.bind_handle("b", interop.host_ptr(b))
    assert lib.run(n) == n

    np.testing.assert_array_equal(out, a + b)  # written in place


def test_host_plugin_over_torch_cpu(tmp_path):
    torch = pytest.importorskip("torch")
    so = _compile_plugin(tmp_path)
    n = 4096
    a = torch.arange(n, dtype=torch.float64)
    b = 2.0 * torch.arange(n, dtype=torch.float64)
    out = torch.full((n,), -1.0, dtype=torch.float64)

    lib = HostPluginLibrary(so, _SIDECAR)
    # Zero-copy: bind the torch tensors' own host storage by data pointer.
    lib.bind_handle("out", interop.host_ptr(out))
    lib.bind_handle("a", interop.host_ptr(a))
    lib.bind_handle("b", interop.host_ptr(b))
    lib.run(n)

    # The plugin wrote the caller's torch tensor in place (torch in -> torch out,
    # no copy, no device round-trip).
    assert torch.equal(out, a + b)


def test_host_ptr_rejects_cuda_tensor():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device to build a device tensor")
    t = torch.ones(8, dtype=torch.float64, device="cuda")
    with pytest.raises(TypeError):
        interop.host_ptr(t)


@pytest.mark.parametrize("scalar_type", ["float32", "softdouble"])
def test_host_plugin_rejects_nonfloat64_scalar_type(scalar_type):
    """The Python loader mirrors the C++ ``eagle::cpu::PluginRegistry::add_plugin``
    gate: the host runner executes native float64 only, so a sidecar declaring a
    non-``float64`` ``scalar_type`` (a device-only float32 / SoftDouble kernel) is
    rejected up front — before dlopen — with a clear error naming the offending type.
    Closes the dtype-asymmetry the parity audit flagged (C++ gated, Python did not)."""
    sidecar = dict(_SIDECAR, scalar_type=scalar_type)
    with pytest.raises(ValueError, match=r"float64 only.*" + scalar_type):
        HostPluginLibrary("/no/such/plugin.so", sidecar)


def test_host_plugin_accepts_absent_or_float64_scalar_type(tmp_path):
    """The gate is a no-op for the two valid states: an absent ``scalar_type`` (a
    pre-dtype sidecar, backward-lenient) and an explicit ``"float64"`` both load and
    run — proving the new check does not regress the float64 path."""
    so = _compile_plugin(tmp_path)
    HostPluginLibrary(so, _SIDECAR)  # no scalar_type -> treated as float64
    HostPluginLibrary(so, dict(_SIDECAR, scalar_type="float64"))
