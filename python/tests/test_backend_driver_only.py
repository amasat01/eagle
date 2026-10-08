# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The CUDA plugin reaches CUDA through the driver library alone.

* No undefined CUDA symbol: every CUDA runtime call is served inside the plugin
  (its driver translation layer), so nothing binds to a CUDA runtime at load;
  and no CUDA runtime is linked IN either (a static ``cudart`` carries its
  ``cudaGetExportTable`` entry and ``cudart_*.cpp`` source markers).
* The only library name the plugin carries besides its DT_NEEDED entries is
  ``libcuda.so.1`` (a ``dlopen`` target is a string in the image).
* At run time the only library the plugin itself loads is ``libcuda.so.1``: a
  child interpreter starts the plugin BEFORE anything else touches CUDA, then runs
  the suites that drive every capability group (next to torch's and CuPy's own
  CUDA runtimes) under ``LD_DEBUG=files``; every library the loader reports as
  opened by ``libeagle_cuda.so`` is read back from its log.
* A driver missing a group's entry points clears that group only
  (``EAGLE_CUDA_DRIVER_HIDE`` hides a driver symbol from the plugin).

The DT_NEEDED half of the rule is covered by
``test_plugin_needs_nothing_outside_glibc_and_openmp`` (``test_backend_wall.py``).
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess
import sys
import textwrap

import backend_wall
import pytest

#: The suites whose rows exercise every capability group of the plugin.
GROUP_SUITES = (
    # stream, capturer, captured, fork, conditional, graph, launcher
    "test_conditional.py",
    "test_gc_capture_invalidation_guard.py",  # captured lifetimes, capture guard
    "test_netsparse_member_enable.py",  # attribution, node enable
    "test_netsparse_device_props.py",  # device
    "test_exec_contract_rows.py",  # exec
    "test_interop_buffer.py",  # interop (fence)
    "test_backend_seam_rows.py",  # composer
    "test_reorder.py",  # filtering (compaction with the trigger, reorder, restore)
)


def _plugin() -> pathlib.Path:
    import eagle._core

    forced = os.environ.get("EAGLE_BACKEND_CUDA")
    path = (
        pathlib.Path(forced)
        if forced
        else pathlib.Path(eagle._core.__file__).parent / "libeagle_cuda.so"
    )
    if not path.exists():
        pytest.skip(f"no CUDA plugin installed at {path} (a core-only build)")
    return path.resolve()


def _child(body: str, env: dict) -> subprocess.CompletedProcess:
    """Run ``body`` in a fresh interpreter that imports THIS process's eagle."""
    import eagle._core

    import eagle

    pkg_parent = str(pathlib.Path(eagle.__file__).resolve().parent.parent)
    core_file = str(pathlib.Path(eagle._core.__file__).resolve())
    preamble = f"""
import importlib.util, json, pathlib, sys
sys.path.insert(0, {pkg_parent!r})
spec = importlib.util.find_spec("eagle")
origin = None if spec is None else pathlib.Path(spec.origin).resolve().parent.parent
if str(origin) != {pkg_parent!r}:
    sys.meta_path[:] = [f for f in sys.meta_path
                        if isinstance(f, type) or f.find_spec("eagle", None) is None]
import eagle, eagle._core
assert str(pathlib.Path(eagle._core.__file__).resolve()) == {core_file!r}
"""
    return subprocess.run(
        [sys.executable, "-c", preamble + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )


def test_plugin_references_no_cuda_symbol():
    try:
        undefined = backend_wall.undefined(_plugin())
    except backend_wall.ToolMissing as e:
        pytest.skip(str(e))
    cuda = [s for s in undefined if backend_wall._CUDA_SYMBOL.match(s)]
    assert not cuda, f"libeagle_cuda.so binds CUDA symbols at load: {cuda}"


#: Markers a statically linked CUDA runtime leaves in an image.
_STATIC_CUDART = re.compile(rb"cudaGetExportTable|cudart_[a-z_]+\.cpp")


def test_plugin_links_no_cuda_runtime():
    found = sorted(set(_STATIC_CUDART.findall(_plugin().read_bytes())))
    assert not found, f"libeagle_cuda.so carries a CUDA runtime: {found}"


def test_plugin_names_no_library_but_the_driver():
    plugin = _plugin()
    try:
        needed = set(backend_wall.needed(plugin))
    except backend_wall.ToolMissing as e:
        pytest.skip(str(e))
    names = set(
        re.findall(rb"lib[A-Za-z0-9_+.-]*?\.so(?:\.[0-9]+)*", plugin.read_bytes())
    )
    # compared by library (``libgomp`` in ``libgomp.so.1``): a wheel's bundled copy
    # keeps the original soname in its version table beside the bundled name
    def lib(name):
        return backend_wall.unvendored(name).split(".so", 1)[0]

    allowed = {lib(n) for n in needed} | {"libcuda", lib(plugin.name)}
    extra = {n.decode() for n in names if lib(n.decode()) not in allowed}
    assert not extra, f"libeagle_cuda.so names libraries it may load: {sorted(extra)}"


@pytest.mark.gpu
def test_plugin_loads_only_the_driver(tmp_path):
    pytest.importorskip("cupy")
    plugin = _plugin()
    log = tmp_path / "ld"
    here = pathlib.Path(__file__).resolve().parent
    suites = [str(here / s) for s in GROUP_SUITES]
    env = dict(
        os.environ,
        EAGLE_BACKEND_CUDA=str(plugin),
        LD_DEBUG="files",
        LD_DEBUG_OUTPUT=str(log),
    )
    proc = _child(
        f"""
        # The plugin first, so the driver library is opened by IT (a library
        # already loaded by torch or CuPy would not be reopened, nor logged).
        eagle._core.Stream().synchronize()
        import pytest
        rc = pytest.main(["-q", "-p", "no:cacheprovider", "-x",
                          "--basetemp", {str(tmp_path / "bt")!r}, *{suites!r}])
        d = eagle._core.cuda_backend()
        print(json.dumps({{"rc": int(rc), "loaded": d["loaded"],
                           "capabilities": d["capabilities"]}}))
        """,
        env,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["rc"] == 0 and out["loaded"], (
        f"{out}\n--- the child's pytest run (last lines) ---\n" + proc.stdout[-4000:])

    # glibc's loader log: "file=<name> [0];  dynamically loaded by <path> [0]".
    pattern = re.compile(r"file=(\S+) \[\d+\];\s+dynamically loaded by (\S+) \[")
    loaded_by_plugin: set[str] = set()
    plugin_opened = False
    for part in tmp_path.glob("ld.*"):
        for name, by in pattern.findall(part.read_text(errors="replace")):
            if pathlib.Path(by).resolve() == plugin:
                loaded_by_plugin.add(name)
            if pathlib.Path(name).name == plugin.name:
                plugin_opened = True
    assert plugin_opened, "the loader log never shows libeagle_cuda.so being opened"
    assert loaded_by_plugin == {"libcuda.so.1"}, sorted(loaded_by_plugin)


@pytest.mark.gpu
def test_a_missing_driver_entry_point_clears_only_its_group():
    plugin = _plugin()
    env = dict(
        os.environ,
        EAGLE_BACKEND_CUDA=str(plugin),
        EAGLE_CUDA_DRIVER_HIDE="cuGraphConditionalHandleCreate",
    )
    proc = _child(
        """
        c = eagle._core
        d = c.cuda_backend()
        s = c.Stream()
        s.synchronize()
        try:
            c.CaptureConditional(1, 8)
            raised = None
        except eagle.BackendUnavailable as e:
            raised = str(e)
        print(json.dumps({"loaded": d["loaded"], "groups": d["groups"],
                          "raised": raised}))
        """,
        env,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["loaded"] is True, out
    unavailable = {g for g, why in out["groups"].items() if why}
    assert unavailable == {"conditional"}, out["groups"]
    assert out["raised"] is not None and "conditional" in out["raised"], out
