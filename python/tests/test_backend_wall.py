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
import re

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


def test_the_plugin_reports_the_seam_minor_of_the_header_it_was_built_from():
    header_file = pathlib.Path(__file__).parents[1] / "src" / "seam" / "eagle_backend.h"
    if not header_file.is_file():
        pytest.skip("repository files are not present")  # the suite copied outside the repository
    minor = re.search(r"#define EAGLE_BACKEND_VERSION_MINOR (\d+)u", header_file.read_text()).group(1)
    from eagle import _core

    if not _core.cuda_backend("cuda")["loaded"]:
        pytest.skip("no CUDA plugin loaded (a core-only build)")
    assert _core.cuda_backend("cuda")["version"] == f"1.{minor}"


def test_the_transfer_symbols_are_in_the_locked_manifest():
    locked = backend_wall.read_manifest(backend_wall.LOCKED_MANIFEST)
    for name in ("memcpy", "host_alloc", "host_free", "host_func"):
        assert f"eagle_backend_{name}" in locked


_MINOR_7 = (
    "device_count", "set_device", "get_device", "launch_status", "device_synchronize", "stream_is_capturing", "module_load", "module_load_jit",
    "module_function", "module_release", "link_ptx", "gemm", "gemm_release", "function_attribute", "launch_raw", "cell_op", "memcpy_strided",
    "mem_alloc", "mem_release", "mem_drain", "pointer_device", "stream_sync",
    "event_create", "event_create_timed", "event_elapsed", "event_record", "event_wait", "event_release",
)


def test_the_minor_7_symbols_are_in_the_locked_manifest():
    locked = backend_wall.read_manifest(backend_wall.LOCKED_MANIFEST)
    for name in _MINOR_7:
        assert f"eagle_backend_{name}" in locked


def test_each_minor_7_symbol_has_a_row_that_calls_it():
    # a symbol with no row that reaches it is dead weight in the wall: every one is bound in the core
    # (so its absence raises by name) and named by a seam-row test
    root = pathlib.Path(__file__).parent
    rows = "".join((root / f).read_text() for f in ("test_backend_seam_rows.py", "test_stream_scope.py", "test_array.py", "test_array_binding.py"))
    core_binding = {
        "device_count": "device_count", "set_device": "set_device", "get_device": "get_device", "launch_status": "launch_status", "device_synchronize": "device_synchronize",
        "stream_is_capturing": "stream_is_capturing", "module_load": "module_load", "module_load_jit": "module_load_jit",
        "module_function": "module_function", "module_release": "module_release", "link_ptx": "link_ptx", "gemm": "gemm", "gemm_release": "gemm_release", "function_attribute": "function_attribute", "launch_raw": "launch_raw",
        "cell_op": "cell_op", "memcpy_strided": "memcpy_strided",
        "mem_alloc": "mem_alloc", "mem_release": "mem_release", "mem_drain": "mem_drain",
        "pointer_device": "pointer_device", "stream_sync": "stream_sync",
        "event_create": "event_create", "event_create_timed": "event_create_timed", "event_elapsed": "event_elapsed",
        "event_record": "event_record", "event_wait": "event_wait",
        "event_release": "event_release",
    }
    missing = [n for n in _MINOR_7 if f"_core.{core_binding[n]}(" not in rows]
    assert not missing, missing


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


def test_plugin_archs_string_lists_the_whole_device_code():
    # the embedded archs string must carry every arch, not a cut-off prefix
    _tools("cuobjdump")
    plugin = _plugin_path()
    arch = rb"(?:[5-9][0-9]|1[0-9][0-9])"  # sm_50 .. sm_199: a lookup table's bytes can spell "1/56"
    listing = rb"(?<![^\x00])" + arch + rb"(?:," + arch + rb")*/" + arch + rb"(?:," + arch + rb")*(?=\x00)"
    found = re.findall(listing, plugin.read_bytes())
    assert len(found) == 1, found
    sass, ptx = re.fullmatch(r"([0-9,]+)/([0-9,]+)", found[0].decode()).groups()
    elf = backend_wall._run(backend_wall._tool("cuobjdump"), "--list-elf", str(plugin))
    ptxs = backend_wall._run(backend_wall._tool("cuobjdump"), "--list-ptx", str(plugin))
    assert sass.split(",") == sorted(
        set(re.findall(r"\.sm_(\d+)\.cubin", elf)), key=int
    )
    assert ptx.split(",") == sorted(set(re.findall(r"\.sm_(\d+)\.ptx", ptxs)), key=int)
