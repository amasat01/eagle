# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The wall between ``eagle._core`` and the CUDA backend plugin, read off the artifacts.

``libeagle_cuda.so`` exports exactly the committed seam manifests and nothing else;
``eagle._core`` neither needs nor references the CUDA runtime or driver; and the
plugin needs nothing outside glibc/OpenMP and carries the ship arch list
(``$EAGLE_PLUGIN_EXPECTED_ARCHS`` = ``"<sass,...>/<ptx,...>"`` for a dev build, e.g.
``"61/"``).

These read the INSTALLED artifacts (whatever ``eagle._core`` resolves to), so they
run unchanged in the repository suite and in an isolated wheel venv. The checks
themselves live in ``backend_wall.py`` and run standalone on any artifact.
"""

from __future__ import annotations

import importlib.util
import os
import pathlib

import backend_wall
import pytest


def _core_path() -> pathlib.Path:
    spec = importlib.util.find_spec("eagle._core")
    if spec is None or spec.origin is None:
        pytest.fail("eagle._core is not importable; the wall cannot be read")
    return pathlib.Path(spec.origin)


def _plugin_path() -> pathlib.Path:
    forced = os.environ.get("EAGLE_BACKEND_CUDA")
    path = pathlib.Path(forced) if forced else _core_path().parent / "libeagle_cuda.so"
    if not path.exists():
        pytest.skip(f"no CUDA plugin installed at {path} (a core-only build)")
    return path


def _tools(*names):
    for n in names:
        try:
            backend_wall._tool(n)
        except backend_wall.ToolMissing as e:
            pytest.skip(str(e))


def _expected_archs():
    raw = os.environ.get("EAGLE_PLUGIN_EXPECTED_ARCHS")
    if not raw:
        return None, None
    sass, _, ptx = raw.partition("/")
    return tuple(a for a in sass.split(",") if a), tuple(a for a in ptx.split(",") if a)


def test_plugin_exports_exactly_the_seam_manifests():
    _tools("nm")
    problems = backend_wall.check_symbols(_plugin_path())
    assert not problems, "\n".join(problems)


def test_manifests_are_sorted_and_disjoint():
    locked = backend_wall.read_manifest(backend_wall.LOCKED_MANIFEST)
    experimental = backend_wall.read_manifest(backend_wall.X_MANIFEST)
    assert locked == sorted(locked, key=str.encode)
    assert experimental == sorted(experimental, key=str.encode)
    assert all(n.startswith("eagle_backend_x_") for n in experimental)
    assert not [n for n in locked if n.startswith("eagle_backend_x_")]


def test_core_is_cuda_free():
    _tools("nm", "readelf")
    problems = backend_wall.check_core(_core_path())
    assert not problems, "\n".join(problems)


def test_plugin_needs_nothing_outside_glibc_and_openmp():
    _tools("readelf")
    bad = [
        n
        for n in backend_wall.needed(_plugin_path())
        if not n.startswith(backend_wall.PLUGIN_ALLOWED_NEEDED)
    ]
    assert not bad, f"libeagle_cuda.so needs {bad}"


def test_plugin_carries_the_expected_device_code():
    _tools("cuobjdump")
    sass, ptx = _expected_archs()
    problems = backend_wall.check_plugin(_plugin_path(), sass, ptx)
    assert not problems, "\n".join(problems)
