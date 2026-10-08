# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0


"""The compiled extensions must load on a machine without an NVIDIA driver.

eagle's host structures run anywhere; only device execution needs a driver and a
GPU, and ``eagle::exec::DeviceKernel`` resolves its Driver-API calls at first device
use for exactly that reason. A ``libcuda`` entry in an extension's dynamic-section
NEEDED list would undo it: the import itself would fail on a driver-less machine,
before any host path could run.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess

import pytest

_EXTENSIONS = ("eagle._core", "eagle._mpi")


def _needed(path: str) -> list[str]:
    readelf = shutil.which("readelf")
    if readelf is None:
        pytest.skip("readelf is not available to read the dynamic section")
    out = subprocess.run([readelf, "-d", path], capture_output=True, text=True,
                         check=True).stdout
    return [ln.split("[", 1)[1].rstrip("]") for ln in out.splitlines()
            if "(NEEDED)" in ln]


@pytest.mark.parametrize("module", _EXTENSIONS)
def test_the_extension_does_not_link_the_driver_library(module):
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        pytest.skip(f"{module} is not built in this environment")
    needed = _needed(spec.origin)
    assert needed, f"{module}: readelf listed no NEEDED entries at all"
    assert not [n for n in needed if n.startswith("libcuda.so")], (
        f"{module} links the NVIDIA driver library ({needed}); it would fail to "
        "import on a machine without a driver"
    )
