# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A hawk-BUILT host bundle (float32 and float64) loads through the C++ host registry.

hawk builds one kernel for the host target; its ``manifest.json`` names the artifact
``"format": "host"``; a small C++ driver loads the manifest with
``eagle::cpu::PluginRegistry`` (float64) or ``PluginRegistry32`` (float32), runs it, and the
answer equals the NumPy law. CPU only; the driver is built with g++.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar, Terminated  # noqa: E402
from hawk.artifact import build_bundle  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]

_DRIVER = r"""
#include "plugin/host_registry.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

template <class Reg, class T>
int drive(const std::string& dir, std::uint32_t n, double g, const char* in, const char* out) {
    Reg reg;
    reg.load(eagle::plugin::parse_manifest(dir + "/manifest.json"), dir);
    std::vector<T> k(n);
    std::vector<unsigned char> term(n, 0);
    FILE* f = std::fopen(in, "rb");
    if (!f || std::fread(k.data(), sizeof(T), n, f) != n) return 3;
    std::fclose(f);
    reg.bind_handle("k", k.data());
    reg.bind_handle("terminated", term.data());
    reg.bind_uniform("g", g);
    if (reg.run(int(n)) != 1) return 4;
    f = std::fopen(out, "wb");
    std::fwrite(k.data(), sizeof(T), n, f);
    std::fclose(f);
    return 0;
}

int main(int argc, char** argv) {
    if (argc != 7) return 2;
    const std::string kind = argv[1];
    const std::uint32_t n = std::uint32_t(std::strtoul(argv[3], nullptr, 10));
    try {
        if (kind == "float32")
            return drive<eagle::cpu::PluginRegistry32, float>(argv[2], n, std::atof(argv[4]), argv[5], argv[6]);
        return drive<eagle::cpu::PluginRegistry, double>(argv[2], n, std::atof(argv[4]), argv[5], argv[6]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
}
"""


@pytest.fixture(scope="module")
def driver(tmp_path_factory):
    gxx = shutil.which("g++")
    if gxx is None:
        pytest.skip("no g++")
    d = tmp_path_factory.mktemp("hostdrv")
    (d / "drv.cpp").write_text(_DRIVER)
    proc = subprocess.run(
        [gxx, "-std=c++23", "-O2", "-fopenmp", "-DEAGLE_CPU_ONLY=1", f"-I{ROOT}",
         str(d / "drv.cpp"), "-ldl", "-o", str(d / "drv")],
        capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(proc.stderr[-1500:])
    return d / "drv"


@pytest.mark.parametrize("mode,dtype,rtol", [("float64", np.float64, 1e-14),
                                             ("float32", np.float32, 1e-6)])
def test_hawk_built_host_bundle_loads_and_matches_numpy(driver, tmp_path, mode, dtype, rtol):
    @hawk.kernel(steps=1)
    def affine(g: Param, terminated: Terminated, k: Mutable[Scalar]):
        k = k * g + 0.5
        terminated = k >= 1.0e30

    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    build_bundle([affine], bundle_dir, mode=mode, targets=("host",))
    manifest = json.loads((bundle_dir / "manifest.json").read_text())
    assert {e["format"] for e in manifest["plugins"]} == {"host"}

    n, g = 4096, 1.37
    k0 = np.linspace(-3.0, 3.0, n).astype(dtype)
    (tmp_path / "in.bin").write_bytes(k0.tobytes())
    proc = subprocess.run([str(driver), mode, str(bundle_dir), str(n), repr(g),
                           str(tmp_path / "in.bin"), str(tmp_path / "out.bin")],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    got = np.frombuffer((tmp_path / "out.bin").read_bytes(), dtype=dtype)
    np.testing.assert_allclose(got, k0 * dtype(g) + dtype(0.5), rtol=rtol, atol=100 * np.finfo(dtype).eps)
