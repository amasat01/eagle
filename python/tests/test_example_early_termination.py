# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The early-termination example's host arm runs everywhere; the device arm
(the graph-vs-eager comparison the example is really about) runs wherever a
GPU and cupy are both present and skips cleanly otherwise.

The example needs hawk, so this row skips cleanly when hawk is absent.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "python" / "early_termination.py"

# cupy >= 14 deprecated ExternalStream in favor of Stream.from_external, but
# GraphPipeline still needs the raw-handle bridge ExternalStream gives it (see
# eagle's own pyproject.toml, which filters the same warning for the same
# reason). Only this one warning is silenced -- nothing else.
pytestmark = [
    pytest.mark.repo_local,
    pytest.mark.filterwarnings("ignore:ExternalStream is deprecated:DeprecationWarning"),
]


def _load_example():
    """Import the example script by path; it is not part of the package."""
    spec = importlib.util.spec_from_file_location("early_termination", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def example():
    """The example module, or a clean skip when its toolchain is absent."""
    pytest.importorskip("numpy")
    pytest.importorskip("eagle")
    pytest.importorskip("hawk")
    if shutil.which("g++") is None and importlib.util.find_spec("ziglang") is None:
        pytest.skip("no host C++ compiler available to build the kernel")
    assert EXAMPLE.is_file(), f"the example is missing: {EXAMPLE}"
    return _load_example()


def _gpu_available() -> bool:
    cupy = pytest.importorskip("cupy")
    try:
        return cupy.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _torch_available() -> bool:
    return importlib.util.find_spec("torch") is not None


def test_the_host_arm_runs_without_cupy(example, monkeypatch):
    """The host arm never imports cupy: block it and the run must still succeed."""
    monkeypatch.setitem(sys.modules, "cupy", None)

    results = example.run_demo(steps=200, batch=48, dt=0.02, seed=0,
                               target="host", verbose=False)

    assert results["target"] == "host"
    assert results["exec_targets"] == ["host"]
    assert results["active_after"] == 0, (
        "not every sample terminated within the horizon")
    assert results["skip_frozen"], (
        "a replay after all-terminated changed the state")
    assert results["skip_count_zero"]
    assert results["skip_control_runs"]
    assert results["host_syncs"] == 100  # measured guard reads, one per pair
    assert results["wall_per_step"] > 0.0
    assert sum(results["lifetime_histogram"]) == 48


@pytest.mark.gpu
def test_the_device_arm_matches_the_eager_arm_and_skips_when_done(example):
    """The acceptance checks (a)-(d), restated against the returned numbers.

    ``run_demo`` already asserts every one of these itself; this row
    re-checks them so a failure names which one gave way, and this is the
    row that proves the device arm actually ran (not skipped).
    """
    if not _gpu_available():
        pytest.skip("no CUDA device visible to this process")

    results = example.run_demo(steps=200, batch=48, dt=0.02, seed=0,
                               target="device", reps=1, verbose=False)

    assert results["target"] == "device"
    assert results["exec_targets"] == ["device"]
    # (a) the graph path never touches the host mid-horizon.
    assert results["host_syncs_graph"] == 1, (
        f"graph-path host syncs: expected 1, got {results['host_syncs_graph']}")
    assert results["host_syncs_eager"] == 100  # steps // 2, one per pair
    # (b) the two arms agree on WHEN every sample terminated, exactly.
    assert results["termination_match"], (
        "graph and eager arms disagree on some sample's termination step")
    # (c) and on the final states, to float32 tolerance.
    assert results["state_rel_diff"] <= example.STATE_REL_TOL, (
        f"graph vs eager final states differ by {results['state_rel_diff']:.3e}")
    # the unguarded control (same captured pair, no skippable) must land on
    # the exact same result, bit-for-bit -- the guard only skips no-op work.
    assert results["unguarded_exact_match"], (
        "the unguarded control disagrees with the guarded graph -- the "
        "guard changed the RESULT, not just whether the step ran")
    # (d) once every sample is done, further replays are provably no-ops.
    assert results["active_after"] == 0
    assert results["skip_frozen"], (
        "a replay after all-terminated changed the state -- the guard did "
        "not skip the step")
    assert results["skip_count_zero"]
    assert results["skip_control_runs"], "a nonzero count did not let the step run"
    assert sum(results["lifetime_histogram"]) == 48


def test_the_torch_comparison_agrees_and_every_arm_terminates(example):
    """``--torch``'s four arms, at a tiny size: skip without torch or a GPU.

    No timing assertions (this is a demo measurement, not a benchmark) --
    only the agreement check (state within
    ``STATE_REL_TOL``, termination mismatches reported not hidden) and that
    every arm's own physics actually terminates the whole batch.
    """
    if not _gpu_available():
        pytest.skip("no CUDA device visible to this process")
    if not _torch_available():
        pytest.skip("torch is not installed")
    import torch
    if not torch.cuda.is_available():
        pytest.skip("torch is installed but not built with CUDA support")

    results = example.run_demo(steps=200, batch=48, dt=0.02, seed=0,
                               target="device", reps=1, torch_compare=True,
                               verbose=False)

    torch_results = results["torch"]
    assert set(torch_results["agreement"]) == {
        "eager_masked", "eager_hostcheck", "graph_masked", "graph_hostcheck"}
    for name, agreement in torch_results["agreement"].items():
        # Hard: state must agree with our guarded graph to STATE_REL_TOL.
        assert agreement["state_rel_diff"] <= example.STATE_REL_TOL, (
            f"torch arm {name!r} disagrees with the guarded graph by "
            f"{agreement['state_rel_diff']:.3e}")
        # Every sample's own physics must have terminated by step 200.
        assert agreement["final_alive"] == 0, (
            f"torch arm {name!r} left {agreement['final_alive']} sample(s) "
            f"still alive after the horizon")
        # Termination-step mismatches (float32 op-order) are REPORTED, not
        # asserted to zero -- a real mismatch here is not a bug to hide.
        print(f"torch arm {name}: termination mismatches = "
              f"{agreement['termination_mismatches']}/{torch_results['batch']}")


def test_the_traced_kernel_declares_the_ping_pong_planes_and_the_live_count(example):
    """The kernel is ONE gated RK4 step: state, alive and age all ping-pong,
    guarded by one accumulator."""
    kernel = example.step_kernel()
    roles = {name: role for role, name in kernel.arg_spec}

    assert kernel.name == example.KERNEL_NAME
    assert set(roles) == {
        "x", "alive", "age", "dt", "r_esc2", "r_col2",
        "x_next", "alive_next", "age_next", "n_active",
    }
    assert roles["x"] == "vec_in" and roles["x_next"] == "mutable"
    assert roles["alive"] == "per_sample" and roles["alive_next"] == "mutable"
    assert roles["age"] == "per_sample" and roles["age_next"] == "mutable"
    assert roles["dt"] == roles["r_esc2"] == roles["r_col2"] == "uniform"
    assert roles["n_active"] == "accum_out"
