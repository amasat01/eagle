# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Shared helpers for the C++/CUDA demo tests (build + fixture staging).

The demo executables under ``eagle/plugin/`` load a manifest + PTX artifacts + input
bins from an ``art`` directory. eagle's tests stage the *committed fixture* artifacts
into a tmp ``art`` dir, write a hand-built manifest (the schema is fixed and small) plus
the input bins + numpy goldens, and run the demo — so the suite drives the deployment
path with NO producer import. The fixture kernels' math is known here, so the goldens
are computed inline.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import numpy as np
import pytest

import eagle

HOST_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "plugin"
FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
ABI = eagle.ABI_VERSION


def _env() -> dict:
    env = {**os.environ, "MAKEFLAGS": "--jobserver-style=pipe"}
    env.setdefault("CUDA_PATH", "/usr/local/cuda")
    return env


def _device_compute_capability() -> str:
    """The visible device's compute capability (``CUDA_VISIBLE_DEVICES``-aware,
    via the Driver API directly — same ordinal-0 device any cupy/CUDA-runtime
    caller in this process would see), as ``"<major><minor>"``."""
    import ctypes

    lib = ctypes.CDLL("libcuda.so.1")
    if lib.cuInit(0) != 0:
        raise RuntimeError("cuInit failed")
    dev = ctypes.c_int()
    if lib.cuDeviceGet(ctypes.byref(dev), 0) != 0:
        raise RuntimeError("cuDeviceGet failed")
    major, minor = ctypes.c_int(), ctypes.c_int()
    # CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_{MAJOR,MINOR} = 75, 76.
    if lib.cuDeviceGetAttribute(ctypes.byref(major), 75, dev) != 0 or \
            lib.cuDeviceGetAttribute(ctypes.byref(minor), 76, dev) != 0:
        raise RuntimeError("cuDeviceGetAttribute failed")
    return f"{major.value}{minor.value}"


def build(src_dir, build_dir, exe_name, *, cuda: bool):
    """Configure + build a demo once; return the executable path (skips if the
    toolchain is unavailable)."""
    if shutil.which("cmake") is None:
        pytest.skip("cmake not available")
    # Prefer nvcc on PATH (set by any CUDA container); fall back to CUDA_PATH.
    nvcc = shutil.which("nvcc") or (_env()["CUDA_PATH"] + "/bin/nvcc")
    cfg = [
        "cmake", "-S", str(src_dir), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
    ]
    if cuda:
        if not pathlib.Path(nvcc).exists():
            pytest.skip("nvcc not available")
        cfg.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
        # Pin CMAKE_CUDA_ARCHITECTURES to the ACTUAL device's real (SASS-only)
        # architecture on the command line. CMake's own project(... CUDA) step
        # already defaults this variable (observed: "52", nvcc/CMake's stock
        # fallback) before the demo's CMakeLists.txt's own
        # "if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)" ever runs, so that default
        # is silently dead; without a real-SASS match the runtime falls back to
        # JIT-compiling the embedded virtual-PTX image, which carries whatever
        # nvcc-on-PATH's own (possibly newer-than-the-driver) PTX ISA version —
        # CUDA_ERROR_UNSUPPORTED_PTX_VERSION. A "-real" target embeds no PTX at
        # all, so no JIT, and the nvcc/driver version pairing cannot matter.
        try:
            cc = _device_compute_capability()
        except Exception as exc:  # pragma: no cover - environment dependent
            pytest.skip(f"could not query the device compute capability: {exc}")
        cfg.append(f"-DCMAKE_CUDA_ARCHITECTURES={cc}-real")
    for cmd in (cfg, ["cmake", "--build", str(build_dir), "-j8"]):
        r = subprocess.run(cmd, env=_env(), capture_output=True, text=True)
        assert r.returncode == 0, f"{cmd}\n{r.stdout}\n{r.stderr}"
    exe = pathlib.Path(build_dir) / exe_name
    assert exe.exists()
    return exe


def write_bin(path, arr, dtype="<f8"):
    np.asarray(arr).astype(dtype).tofile(path)


def unit_vecs(n, seed, lo, hi):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(3, n))
    d /= np.linalg.norm(d, axis=0)
    return np.ascontiguousarray((d * rng.uniform(lo, hi, n)).astype(np.float64))


def copy_fixture(art, stem, *, as_stem=None):
    """Stage a committed fixture artifact (``<stem>.ptx`` + ``<stem>.json``) into
    ``art``, optionally renaming both to ``as_stem``."""
    dst = as_stem or stem
    shutil.copy(FIX / f"{stem}.ptx", pathlib.Path(art) / f"{dst}.ptx")
    shutil.copy(FIX / f"{stem}.json", pathlib.Path(art) / f"{dst}.json")


def write_manifest(art, entries, pattern):
    """Write a manifest.json referencing staged fixtures.

    ``entries`` is a list of ``(id, artifact_stem, enabled)`` tuples in injection
    order; each stem's ``<stem>.ptx`` / ``<stem>.json`` must already be staged."""
    plugins = [
        {
            "id": pid,
            "order": order,
            "enabled": enabled,
            "artifact": f"{stem}.ptx",
            "sidecar": f"{stem}.json",
            "format": "ptx",
        }
        for order, (pid, stem, enabled) in enumerate(entries)
    ]
    (pathlib.Path(art) / "manifest.json").write_text(
        json.dumps(
            {"version": 1, "pattern": pattern, "aether_abi": ABI, "plugins": plugins},
            indent=2,
        )
    )
    return pathlib.Path(art) / "manifest.json"


def run(exe, *args):
    return subprocess.run(
        [str(exe), *[str(a) for a in args]], capture_output=True, text=True
    )


def parse_vals(stdout):
    return {k: v for k, v in (l.split("=", 1) for l in stdout.splitlines() if "=" in l)}


# --- inline numpy goldens for the fixture kernels ------------------------------
def gravity_ref(pos, mu):
    r = np.linalg.norm(pos, axis=0)
    return -mu * pos / r**3


def drag_ps_ref(vel, cd, area, mass):
    s = np.linalg.norm(vel, axis=0)
    return -0.5 * (cd * area / mass) * s * vel
