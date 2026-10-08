# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The precompiled C++/CUDA plugin host loads an eagle PTX plugin and runs it.

Rebased onto committed fixtures (no producer import): the standalone Driver-API host
(``plugin/plugin_host.cpp``) driver-loads a fixture acceleration, builds GRef/HandleT
views by value, and launches it standalone, as a captured-graph node, and through the
GRef<->DLPack bridge — verifying against a numpy golden. The mid-pipeline injection
demo (``plugin/graph_inject``) drives the same fixture through the C++
``PluginRegistry`` via a hand-built 1-plugin manifest.
"""

import json

import cpp_demo as C
import numpy as np
import pytest

MU = 3.986004418e5
INJECT_SCALE = 2.5  # plugin/graph_inject backend_pre
INJECT_POST = 0.5  # plugin/graph_inject backend_post

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def host_exe(tmp_path_factory):
    build = tmp_path_factory.mktemp("host_build")
    return C.build(C.HOST_DIR, build, "plugin_host", cuda=False)


@pytest.fixture(scope="module")
def inject_exe(tmp_path_factory):
    build = tmp_path_factory.mktemp("inject_build")
    return C.build(C.HOST_DIR / "graph_inject", build, "inject_demo", cuda=True)


def _run_host(exe, art, n):
    r = C.run(exe, art, n)
    assert r.returncode == 0, f"host failed:\n{r.stdout}\n{r.stderr}"
    assert "PASS" in r.stdout, r.stdout
    return C.parse_vals(r.stdout)


def test_cpp_host_runs_gravity_plugin(host_exe, tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    C.copy_fixture(art, "gravity", as_stem="plugin")
    n = 4096
    P = C.unit_vecs(n, seed=0, lo=7.0e3, hi=4.2e4)
    C.write_bin(art / "position.bin", P)
    C.write_bin(art / "uniforms.bin", np.array([MU]))
    C.write_bin(art / "golden.bin", C.gravity_ref(P, MU))

    vals = _run_host(host_exe, art, n)
    assert float(vals["STANDALONE_MAX_REL"]) < 1e-12
    assert float(vals["GRAPH_MAX_REL"]) < 1e-12
    assert float(vals["DLPACK_MAX_REL"]) < 1e-12


def test_cpp_host_terminated_skips(host_exe, tmp_path):
    # the host's terminated.bin upload branch + the kernel skip guard: terminated
    # samples stay at the zero-initialized accumulator.
    art = tmp_path / "art"
    art.mkdir()
    C.copy_fixture(art, "gravity", as_stem="plugin")
    n = 1024
    P = C.unit_vecs(n, seed=4, lo=7.0e3, hi=4.2e4)
    C.write_bin(art / "position.bin", P)
    C.write_bin(art / "uniforms.bin", np.array([MU]))
    mask = np.zeros(n, dtype=np.uint8)
    mask[::2] = 1  # terminate every other
    mask.tofile(art / "terminated.bin")
    golden = C.gravity_ref(P, MU).copy()
    golden[:, ::2] = 0.0  # terminated -> untouched (0)
    C.write_bin(art / "golden.bin", golden)

    vals = _run_host(host_exe, art, n)
    assert float(vals["STANDALONE_MAX_REL"]) < 1e-12
    assert float(vals["GRAPH_MAX_REL"]) < 1e-12
    assert float(vals["DLPACK_MAX_REL"]) < 1e-12


def test_cpp_host_runs_per_sample_drag_plugin(host_exe, tmp_path):
    art = tmp_path / "art"
    art.mkdir()
    C.copy_fixture(art, "drag_ps", as_stem="plugin")
    n = 4096
    rng = np.random.default_rng(11)
    V = C.unit_vecs(n, seed=2, lo=1.0, hi=8.0)
    cd = rng.uniform(2.0, 2.4, n)
    area = rng.uniform(1.0, 20.0, n)
    mass = rng.uniform(100.0, 1200.0, n)
    C.write_bin(art / "velocity.bin", V)
    C.write_bin(art / "mass.bin", mass)
    C.write_bin(art / "area.bin", area)
    C.write_bin(art / "cd.bin", cd)
    # no uniforms.bin: this force has zero broadcast constants
    C.write_bin(art / "golden.bin", C.drag_ps_ref(V, cd, area, mass))

    vals = _run_host(host_exe, art, n)
    assert float(vals["STANDALONE_MAX_REL"]) < 1e-12
    assert float(vals["GRAPH_MAX_REL"]) < 1e-12
    assert float(vals["DLPACK_MAX_REL"]) < 1e-12


def _doctored_artifact(tmp_path, **sidecar_overrides):
    """Stage the gravity fixture, then rewrite its sidecar in the tmp artifact
    dir. The committed fixture is never touched — only the staged copy."""
    art = tmp_path / "art"
    art.mkdir()
    C.copy_fixture(art, "gravity", as_stem="plugin")
    sc = json.loads((art / "plugin.json").read_text())
    sc.update(sidecar_overrides)
    (art / "plugin.json").write_text(json.dumps(sc, indent=2))
    return art


def test_cpp_host_refuses_recognized_but_unlaunchable_family(host_exe, tmp_path):
    """Door #7's family dispatch: a sidecar of a family this host does
    not bind is refused as the wrong KIND, before any CUDA call.

    Before this dispatch existed, the standalone host had no gates at all: a ``pure``
    sidecar fell through to the arg_spec loop and died with ``unknown arg role:
    mutable`` — naming a perfectly valid schema-v1 role as though the artifact were
    malformed. It is not malformed; it is not launchable here, and the gate must
    say so. GPU-free in substance (the refusal precedes ``cuInit``); the module's
    ``gpu`` marker is inherited, not required."""
    art = _doctored_artifact(tmp_path, pattern="pure")
    r = C.run(host_exe, art, 16)
    assert r.returncode != 0, f"expected a refusal:\n{r.stdout}\n{r.stderr}"
    assert "recognized but not launchable" in r.stderr, r.stderr
    # the not-launchable branch carries NO upgrade suffix: `pure` is a family
    # this build knows perfectly well, so telling the caller to upgrade eagle
    # would send them after a fix that does not exist.
    assert "upgrade eagle" not in r.stderr, r.stderr
    assert "unknown arg role" not in r.stderr, r.stderr


def test_cpp_host_refuses_unrecognized_family(host_exe, tmp_path):
    """Door #7's shared ``validate_sidecar``, which runs first: a family no
    eagle build recognizes gets the OTHER branch — the one that does say
    ``upgrade eagle``, because a newer eagle is exactly what would load it."""
    art = _doctored_artifact(tmp_path, pattern="not_a_registered_family")
    r = C.run(host_exe, art, 16)
    assert r.returncode != 0, f"expected a refusal:\n{r.stdout}\n{r.stderr}"
    assert "unknown sidecar pattern" in r.stderr, r.stderr
    assert "upgrade eagle" in r.stderr, r.stderr


def test_cpp_host_injects_jit_into_backend_graph(inject_exe, tmp_path):
    # a Driver-API-loaded PTX plugin captured as a node INSIDE a Runtime-API graph,
    # driven through the registry via a single-plugin acceleration manifest.
    art = tmp_path / "art"
    art.mkdir()
    C.copy_fixture(art, "gravity")
    C.write_manifest(art, [("gravity", "gravity", True)], "vector")
    n = 4096
    P = C.unit_vecs(n, seed=0, lo=7.0e3, hi=4.2e4)
    C.write_bin(art / "position.bin", P)
    C.write_bin(art / "uniforms.bin", np.array([MU]))
    C.write_bin(art / "golden.bin", INJECT_POST * C.gravity_ref(INJECT_SCALE * P, MU))

    r = C.run(inject_exe, art, n)
    assert r.returncode == 0, f"{r.stdout}\n{r.stderr}"
    assert "PASS" in r.stdout, r.stdout
    vals = C.parse_vals(r.stdout)
    assert int(vals["TOTAL_NODES"]) == 4  # pre, memset, JIT, post
    assert float(vals["INJECT_MAX_REL"]) < 1e-12
