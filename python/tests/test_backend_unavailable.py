# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A missing, mismatched or partial CUDA backend is a TYPED refusal, never a crash.

Every row runs in a fresh subprocess (the backend is loaded once per process and
the outcome cached), pointed at a plugin through ``$EAGLE_BACKEND_CUDA``:

* a path that does not exist: ``import eagle`` / ``eagle.exec`` and the host
  rows still work; the backend reports the env path it was told to load, and
  ``Stream()`` raises ``eagle.BackendUnavailable`` naming it.
* the real plugin with no visible device: it loads, and ``Stream()`` raises
  ``eagle.BackendUnavailable`` naming ``cudaErrorNoDevice``.
* a stub exporting only ``eagle_backend_version`` = 2.0: refused, naming
  major 2 against major 1.
* a stub exporting ONLY the mandatory core with capabilities = 0: it loads,
  every capability group is unavailable naming the group, and every Python entry
  point of each group (and of the experimental symbols) raises
  ``eagle.BackendUnavailable`` naming it — nothing crashes.
* a stub whose ``eagle_backend_build_info`` predates ``backend_kind`` (a smaller
  ``struct_size``): it loads, and the core keeps only what the stub filled.

The stubs are compiled here with the system C compiler (``cc -shared``).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import textwrap

import pytest

GROUPS = (
    "stream",
    "capturer",
    "captured",
    "fork",
    "conditional",
    "attribution",
    "graph",
    "launcher",
    "exec",
    "device",
    "interop",
    "filtering",
)

_MANDATORY_STUB = r"""
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
struct info {{ uint32_t struct_size, abi_version; int32_t cudart_version, reserved0;
              char eagle_version[32]; char archs[128]; char backend_kind[16]; }};
uint32_t eagle_backend_version(void) {{ return 1u << 16; }}
const char* eagle_backend_last_error(void) {{ return ""; }}
int32_t eagle_backend_build_info(struct info* i) {{
    struct info full; memset(&full, 0, sizeof full);
    uint32_t n = {filled};
    full.abi_version = 1u << 16;
    strcpy(full.eagle_version, "stub"); strcpy(full.archs, "none");
    strcpy(full.backend_kind, "stub");
    if (i->struct_size < n) n = i->struct_size;
    full.struct_size = n; memcpy(i, &full, n); return 0; }}
int32_t eagle_backend_probe(void) {{ return 0; }}
void eagle_backend_free(void* p) {{ free(p); }}
int32_t eagle_backend_capabilities(uint64_t* g) {{ *g = 0; return 0; }}
int32_t eagle_backend_device_types(int32_t* t, int32_t cap, int32_t* n) {{
    if (cap > 0) t[0] = 2; *n = 1; return 0; }}
"""

_MAJOR2_STUB = r"""
#include <stdint.h>
uint32_t eagle_backend_version(void) { return 2u << 16; }
"""


def _compile(tmp_path: pathlib.Path, name: str, source: str) -> pathlib.Path:
    cc = shutil.which("cc") or shutil.which("gcc")
    if cc is None:
        pytest.skip("no C compiler to build the stub plugin")
    src = tmp_path / f"{name}.c"
    src.write_text(source)
    out = tmp_path / f"{name}.so"
    subprocess.run([cc, "-shared", "-fPIC", "-o", str(out), str(src)], check=True)
    return out


def _run_child(
    body: str, plugin: str | os.PathLike, extra_env: dict | None = None
) -> dict:
    """Run ``body`` in a fresh interpreter importing THIS process's eagle.

    Returns the JSON object the body prints last."""
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
core = str(pathlib.Path(eagle._core.__file__).resolve())
assert core == {core_file!r}, core
"""
    env = dict(os.environ, EAGLE_BACKEND_CUDA=str(plugin), **(extra_env or {}))
    proc = subprocess.run(
        [sys.executable, "-c", preamble + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_absent_plugin_keeps_import_and_host_rows(tmp_path):
    missing = tmp_path / "nonexistent" / "x.so"
    out = _run_child(
        """
        import eagle.exec
        assert issubclass(eagle.BackendUnavailable, RuntimeError)
        assert eagle._core.BackendUnavailable is eagle.BackendUnavailable
        part = eagle.exec.Partition(0, 10, 10)
        d = eagle._core.cuda_backend()
        try:
            eagle._core.Stream()
            raised = None
        except eagle.BackendUnavailable as e:
            raised = str(e)
        print(json.dumps({"fold": eagle.exec.fold("sum", [1.0, 2.0]),
                          "count": part.count, "loaded": d["loaded"],
                          "error": d["error"], "groups": d["groups"],
                          "raised": raised}))
        """,
        missing,
    )
    assert out["fold"] == 3.0 and out["count"] == 10
    assert out["loaded"] is False
    assert "EAGLE_BACKEND_CUDA" in out["error"] and str(missing) in out["error"]
    assert all(str(missing) in why for why in out["groups"].values())
    assert out["raised"] is not None and str(missing) in out["raised"]


def test_no_visible_device_is_typed(tmp_path):
    import eagle._core

    plugin = pathlib.Path(eagle._core.__file__).parent / "libeagle_cuda.so"
    if not plugin.exists():
        pytest.skip("no CUDA plugin installed (a core-only build)")
    out = _run_child(
        """
        d = eagle._core.cuda_backend()
        try:
            eagle._core.Stream()
            raised = None
        except eagle.BackendUnavailable as e:
            raised = str(e)
        print(json.dumps({"loaded": d["loaded"], "raised": raised}))
        """,
        plugin,
        extra_env={"CUDA_VISIBLE_DEVICES": ""},
    )
    assert out["loaded"] is True
    if out["raised"] and "no usable CUDA driver" in out["raised"]:
        pytest.skip("no CUDA driver on this machine: the driver-absent case is "
                    "the no-driver leg's, not this row's")
    assert out["raised"] is not None and "cudaErrorNoDevice" in out["raised"], out


def test_major_version_mismatch_is_refused_naming_both(tmp_path):
    stub = _compile(tmp_path, "major2", _MAJOR2_STUB)
    out = _run_child(
        """
        d = eagle._core.cuda_backend()
        print(json.dumps({"loaded": d["loaded"], "error": d["error"]}))
        """,
        stub,
    )
    assert out["loaded"] is False
    assert "major 2" in out["error"] and "needs major 1" in out["error"], out["error"]
    assert str(stub) in out["error"]


def test_mandatory_only_backend_refuses_every_group_by_name(tmp_path):
    stub = _compile(tmp_path, "mandatory", _MANDATORY_STUB.format(filled="sizeof full"))
    out = _run_child(
        """
        d = eagle._core.cuda_backend()
        print(json.dumps(d))
        """,
        stub,
    )
    assert out["loaded"] is True and out["capabilities"] == 0, out
    assert set(out["groups"]) == set(GROUPS)
    for name, why in out["groups"].items():
        assert why.endswith(f"does not provide {name}"), (name, why)
    assert not any(out["experimental"].values())


def test_mandatory_only_backend_python_entries_raise_naming_the_group(tmp_path):
    stub = _compile(tmp_path, "mandatory", _MANDATORY_STUB.format(filled="sizeof full"))
    out = _run_child(
        """
        c = eagle._core
        calls = {
            "stream": lambda: c.Stream(),
            "capturer": lambda: c.StreamCapturer(1),
            "fork": lambda: c.CaptureFork(1, 2),
            "conditional": lambda: c.CaptureConditional(1, 8),
            "attribution": lambda: c.is_node_toggleable(8),
            "graph": lambda: c.Graph(),
            "exec": lambda: c.run_device(8, [], c.Partition(0, 1, 1)),
            "device": lambda: c.device_props(0),
            "interop": lambda: c.event_pool_created(),
            "filtering": lambda: c.compact_device(8, 8, 8, 0, 1),
            "filtering:reorder": lambda: c.reorder_device([], 8, 8, 8, 8, 8, 8, 0, 0, 1),
            "filtering:restore": lambda: c.restore_device([], 8, 8, 0, 1),
            "x_composer": lambda: c.GraphComposer(),
            "x_guard": lambda: c.capture_guard_depth(),
        }
        res = {}
        for k, f in calls.items():
            try:
                f()
                res[k] = None
            except eagle.BackendUnavailable as e:
                res[k] = str(e)
            except Exception as e:  # anything untyped is a failure of the row
                res[k] = "UNTYPED " + type(e).__name__ + ": " + str(e)
        print(json.dumps(res))
        """,
        stub,
    )
    for key, why in out.items():
        assert why is not None and not why.startswith("UNTYPED"), (key, why)
        if key.startswith("x_"):
            assert "does not provide eagle_backend_x_" in why, (key, why)
        else:
            group = key.split(":")[0]
            assert why.endswith(f"does not provide {group}"), (key, why)


def test_smaller_build_info_struct_is_honoured(tmp_path):
    # An older backend: build_info ends before backend_kind (offset 176).
    stub = _compile(tmp_path, "oldinfo", _MANDATORY_STUB.format(filled="176u"))
    out = _run_child(
        """
        d = eagle._core.cuda_backend()
        print(json.dumps(d))
        """,
        stub,
    )
    assert out["loaded"] is True, out
    info = out["build_info"]
    assert info["struct_size"] == 176
    assert info["archs"] == "none" and info["eagle_version"] == "stub"
    assert "backend_kind" not in info  # past the size the backend wrote: ABSENT
