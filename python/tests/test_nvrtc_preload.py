# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle loads the NVIDIA wheels' libnvrtc-builtins by full path at import, so CuPy's first
compile finds it even after JAX's CUDA plugin started first (eagle._nvrtc)."""

import importlib.util
import os
import sys

import pytest

from eagle import _nvrtc


def test_preload_loads_only_builtins_libraries_that_exist():
    loaded = _nvrtc.preload_builtins()
    for path in loaded:
        assert os.path.isfile(path)
        assert os.path.basename(path).startswith("libnvrtc-builtins.so.")
        assert ".alt." not in os.path.basename(path)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the NVIDIA wheels' layout is Linux-only")
def test_a_wheel_builtins_library_is_resident_after_import_eagle():
    spec = importlib.util.find_spec("nvidia")
    roots = list(spec.submodule_search_locations) if spec else []
    wheel = [p for r in roots for p in __import__("glob").glob(os.path.join(r, "*", "lib", "libnvrtc-builtins.so.*"))
             if ".alt." not in p]
    if not wheel or importlib.util.find_spec("cupy") is None:
        pytest.skip("no NVIDIA-wheel NVRTC builtins and CuPy in this environment")
    import eagle  # noqa: F401  (the import preloads)

    resident = {line.split()[-1] for line in open("/proc/self/maps") if "libnvrtc-builtins" in line}
    assert resident & {os.path.realpath(p) for p in wheel} or resident & set(wheel)


def test_nothing_happens_without_cupy(monkeypatch):
    real = importlib.util.find_spec
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None if name == "cupy" else real(name, *a))
    assert _nvrtc.preload_builtins() == ()
