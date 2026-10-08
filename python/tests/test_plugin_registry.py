# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The independent multi-plugin registry injects a plugin SET into one graph.

Rebased onto committed fixtures (no producer import): the C++ ``PluginRegistry`` loads a
hand-built acceleration manifest of the fixture forces (gravity + per-sample drag) and
injects them, flag-gated and in manifest order, into the opening between two compiled-in
backend kernels of a captured graph. Each plugin accumulates into the shared outVec.
Three regimes: both on (5 nodes), the global flag off (baseline, 3 nodes), one disabled
in the manifest (4 nodes).

Skipped here (need a producer to build a ``Table`` force): the lookup-table
consolidation cases and the conflicting-table-shapes guard. Those stay in the producer's
suite.
"""

import json
import pathlib
import shutil

import cpp_demo as C
import numpy as np
import pytest

import eagle
from eagle.abi import ABI_VERSION

MU = 3.986004418e5
SCALE = 2.5  # plugin/graph_inject backend_pre
POST = 0.5  # plugin/graph_inject backend_post

pytestmark = pytest.mark.gpu

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def inject_exe(tmp_path_factory):
    build = tmp_path_factory.mktemp("registry_build")
    return C.build(C.HOST_DIR / "graph_inject", build, "inject_demo", cuda=True)


def _make_inputs(art, n):
    """Write the SoA input bins; return the host arrays for golden computation."""
    rng = np.random.default_rng(11)
    P = C.unit_vecs(n, seed=0, lo=7.0e3, hi=4.2e4)
    V = C.unit_vecs(n, seed=2, lo=1.0, hi=8.0)
    cd = rng.uniform(2.0, 2.4, n)
    area = rng.uniform(1.0, 20.0, n)
    mass = rng.uniform(100.0, 1200.0, n)
    C.write_bin(art / "position.bin", P)
    C.write_bin(art / "velocity.bin", V)
    C.write_bin(art / "mass.bin", mass)
    C.write_bin(art / "area.bin", area)
    C.write_bin(art / "cd.bin", cd)
    C.write_bin(art / "uniforms.bin", np.array([MU]))
    return P, V, cd, area, mass


def _run(exe, art, n, *extra):
    r = C.run(exe, art, n, *extra)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "PASS" in r.stdout, r.stdout
    return C.parse_vals(r.stdout)


def _stage_set(art):
    C.copy_fixture(art, "gravity")
    C.copy_fixture(art, "drag_ps")
    return C.write_manifest(
        art, [("gravity", "gravity", True), ("drag_ps", "drag_ps", True)],
        "vector",
    )


def test_two_plugins_inject_into_one_graph(inject_exe, tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    _stage_set(art)
    n = 4096
    P, V, cd, area, mass = _make_inputs(art, n)
    golden = POST * (
        C.gravity_ref(SCALE * P, MU) + C.drag_ps_ref(SCALE * V, cd, area, mass)
    )
    C.write_bin(art / "golden.bin", golden)

    vals = _run(inject_exe, art, n)
    assert int(vals["N_PLUGINS"]) == 2
    assert int(vals["ACTIVE"]) == 2
    assert int(vals["INJECTED"]) == 2
    assert int(vals["TOTAL_NODES"]) == 5  # pre, memset, gravity, drag, post
    assert float(vals["INJECT_MAX_REL"]) < 1e-12


def test_flag_off_is_a_noop_baseline(inject_exe, tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    _stage_set(art)
    n = 1024
    _make_inputs(art, n)
    C.write_bin(art / "golden.bin", np.zeros((3, n)))

    vals = _run(inject_exe, art, n, "off")
    assert int(vals["N_PLUGINS"]) == 2
    assert int(vals["ACTIVE"]) == 0
    assert int(vals["INJECTED"]) == 0
    assert int(vals["TOTAL_NODES"]) == 3  # pre, memset, post (no plugin nodes)
    assert float(vals["INJECT_MAX_REL"]) < 1e-12


def test_one_plugin_disabled_in_manifest(inject_exe, tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    manifest = _stage_set(art)
    m = json.loads(manifest.read_text())
    for p in m["plugins"]:
        if p["id"] == "drag_ps":
            p["enabled"] = False
    manifest.write_text(json.dumps(m, indent=2))
    n = 2048
    P, V, cd, area, mass = _make_inputs(art, n)
    golden = POST * C.gravity_ref(SCALE * P, MU)  # drag disabled
    C.write_bin(art / "golden.bin", golden)

    vals = _run(inject_exe, art, n)
    assert int(vals["N_PLUGINS"]) == 2
    assert int(vals["ACTIVE"]) == 1
    assert int(vals["INJECTED"]) == 1
    assert int(vals["TOTAL_NODES"]) == 4  # pre, memset, gravity, post
    assert float(vals["INJECT_MAX_REL"]) < 1e-12


# --------------------------------------------------------------------------- #
# Manifest-level duplicate plugin id rejection ("duplicate ids are
# rejected"), Python side. A Bundle-built manifest never carries a duplicate
# (the manifest builder auto-suffixes an auto-derived
# collision), so this guards the OTHER path: a manifest assembled/edited by
# hand. Companion: eagle/tests/test_PluginRegistryManifest.cu (same check, the
# CUDA ``PluginRegistry::from_manifest`` side). No GPU needed: the duplicate
# scan is its own pass over ``plugins[]`` ids, before ``load_manifest`` loads
# (or even attempts to import cupy for) any entry.
# --------------------------------------------------------------------------- #
def _write_dup_id_manifest(tmp_path, *, second_enabled=True):
    shutil.copy(FIX / "gravity.ptx", tmp_path / "gravity.ptx")
    shutil.copy(FIX / "gravity.json", tmp_path / "gravity.json")
    shutil.copy(FIX / "drag_ps.ptx", tmp_path / "drag_ps.ptx")
    shutil.copy(FIX / "drag_ps.json", tmp_path / "drag_ps.json")
    manifest = {
        "schema_version": 1,
        "pattern": "vector",
        "aether_abi": ABI_VERSION,
        "plugins": [
            {
                "id": "gravity",
                "order": 0,
                "enabled": True,
                "artifact": "gravity.ptx",
                "sidecar": "gravity.json",
                "format": "ptx",
            },
            {
                # same id as above, referencing a DIFFERENT artifact — exactly
                # the hand-built-manifest mistake the guard exists to catch.
                "id": "gravity",
                "order": 1,
                "enabled": second_enabled,
                "artifact": "drag_ps.ptx",
                "sidecar": "drag_ps.json",
                "format": "ptx",
            },
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def test_load_manifest_rejects_duplicate_plugin_id(tmp_path):
    manifest = _write_dup_id_manifest(tmp_path)
    with pytest.raises(ValueError, match="duplicate plugin id"):
        eagle.load_manifest(manifest)


def test_load_manifest_rejects_duplicate_plugin_id_even_if_one_is_disabled(tmp_path):
    # The duplicate is a structural property of the manifest itself, independent
    # of which entries would actually be loaded/injected.
    manifest = _write_dup_id_manifest(tmp_path, second_enabled=False)
    with pytest.raises(ValueError, match="duplicate plugin id"):
        eagle.load_manifest(manifest)
