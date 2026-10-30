# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A hawk-BUILT device bundle with finishing samples runs through the C++ device registry
with no caller-supplied tables: the registry owns the ``finished_count`` cell."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar, Terminated  # noqa: E402
from hawk.artifact import build_bundle  # noqa: E402

pytestmark = pytest.mark.gpu

ROOT = pathlib.Path(__file__).resolve().parents[2]

_DRIVER = r"""
#include "plugin/plugin_registry/registry.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char** argv) {
    const std::string dir = argv[1];
    const int n = std::atoi(argv[2]);
    const double g = std::atof(argv[3]);
    cudaSetDevice(0);
    cudaFree(0);
    cuInit(0);
    auto reg = eagle::cuda::PluginRegistry::from_manifest(dir + "/manifest.json");
    std::vector<double> k(n), k0;
    FILE* f = std::fopen(argv[4], "rb");
    if (!f || std::fread(k.data(), sizeof(double), n, f) != std::size_t(n)) return 3;
    std::fclose(f);
    k0 = k;
    double* dk; unsigned char* dm;
    cudaMalloc(&dk, n * sizeof(double));
    cudaMalloc(&dm, n);
    cudaMemcpy(dk, k.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemset(dm, 0, n);
    reg.bind_handle("k", dk);
    reg.bind_handle("terminated", dm);
    reg.bind_uniform("g", g);
    if (reg.inject(nullptr, n) != 1) return 4;   // no consolidate call anywhere
    cudaDeviceSynchronize();
    std::vector<unsigned char> m(n);
    cudaMemcpy(k.data(), dk, n * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(m.data(), dm, n, cudaMemcpyDeviceToHost);
    long long fin = 0;
    for (auto v : m) fin += v;
    std::printf("FINISHED_COUNT=%u MASK_SUM=%lld\n", reg.finished_count(), fin);
    // Two consecutive runs through pack() + DeviceKernel::run on a NON-default stream: each
    // counts from zero, with no caller step.
    CUstream st; cuStreamCreate(&st, CU_STREAM_NON_BLOCKING);
    unsigned per_run[2] = {0, 0};
    for (int r = 0; r < 2; ++r) {
        cudaMemcpy(dk, k0.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(dm, 0, n);
        cudaDeviceSynchronize();
        auto packed = reg.pack(0, n);
        eagle::exec::DeviceKernel::run(reg.function(0), packed,
                                       eagle::exec::Partition::whole(n), st, 256);
        cuStreamSynchronize(st);
        per_run[r] = reg.finished_count();
    }
    // inject_plugin on the same stream, twice: also per run.
    unsigned inj[2] = {0, 0};
    for (int r = 0; r < 2; ++r) {
        cudaMemcpy(dk, k0.data(), n * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemset(dm, 0, n);
        cudaDeviceSynchronize();
        reg.inject_plugin(st, 0, n);
        cuStreamSynchronize(st);
        inj[r] = reg.finished_count();
    }
    std::printf("PACK_RUN1=%u PACK_RUN2=%u INJ_RUN1=%u INJ_RUN2=%u\n", per_run[0], per_run[1], inj[0], inj[1]);
    f = std::fopen(argv[5], "wb");
    std::fwrite(k.data(), sizeof(double), n, f);
    std::fclose(f);
    return 0;
}
"""


def test_hawk_built_device_bundle_finishes_with_no_caller_tables(tmp_path):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        pytest.skip("no nvcc")
    (tmp_path / "drv.cu").write_text(_DRIVER)
    inc = os.environ.get("EAGLE_TEST_AETHER_INCLUDE")
    if not inc:
        pytest.skip("EAGLE_TEST_AETHER_INCLUDE (the directory holding aether/) is not set")
    proc = subprocess.run(
        [nvcc, "-std=c++20", "-arch=native", "-Wno-deprecated-gpu-targets", f"-I{ROOT}", f"-I{inc}",
         "-DAETHER_HAS_CUDA=1", "-DEAGLE_BLOCKSIZE=256", "-DNDEBUG", str(tmp_path / "drv.cu"),
         "-lcuda", "-o", str(tmp_path / "drv")], capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(proc.stderr[-2000:])

    @hawk.kernel(steps=1)
    def affine(g: Param, terminated: Terminated, k: Mutable[Scalar]):
        k = k * g + 0.5
        terminated = k >= 2.0

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    build_bundle([affine], bundle_dir, mode="float64", targets=("cuda",))
    assert json.loads((bundle_dir / "manifest.json").read_text())["plugins"]

    n, g = 4096, 1.37
    k0 = np.linspace(-3.0, 3.0, n)
    (tmp_path / "in.bin").write_bytes(k0.tobytes())
    run = subprocess.run([str(tmp_path / "drv"), str(bundle_dir), str(n), repr(g),
                          str(tmp_path / "in.bin"), str(tmp_path / "out.bin")],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stderr + run.stdout
    want = k0 * g + 0.5
    fields = dict(kv.split("=") for kv in run.stdout.split())
    assert int(fields["MASK_SUM"]) == int((want >= 2.0).sum()) > 0
    assert int(fields["FINISHED_COUNT"]) == int(fields["MASK_SUM"])
    expect = int(fields["MASK_SUM"])
    assert int(fields["PACK_RUN1"]) == expect and int(fields["PACK_RUN2"]) == expect
    assert int(fields["INJ_RUN1"]) == expect and int(fields["INJ_RUN2"]) == expect
    got = np.frombuffer((tmp_path / "out.bin").read_bytes(), dtype=np.float64)
    np.testing.assert_allclose(got, want, rtol=1e-14, atol=1e-15)
