# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The zero-copy interop example's host arm runs everywhere; the device arm
(torch tensors crossing into eagle) runs wherever a GPU and torch are both
present and skips cleanly otherwise.

The example needs hawk, so this row skips cleanly when hawk is absent.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest
from conftest import CUPY_AVAILABLE

pytestmark = pytest.mark.repo_local

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "python" / "zero_copy_interop.py"


def _load_example():
    """Import the example script by path; it is not part of the package."""
    spec = importlib.util.spec_from_file_location("zero_copy_interop", EXAMPLE)
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


def _torch_and_gpu_available() -> bool:
    """Torch installed, a CUDA device visible to cupy, AND torch itself
    built with CUDA support -- the third check is the one that was missing:
    a pip CPU-only torch wheel installs cleanly and imports cleanly beside a
    real GPU cupy can see, so the first two checks alone said "yes" and
    every ``.cuda()`` call downstream raised "Torch not compiled with CUDA
    enabled" instead of a clean skip."""
    if importlib.util.find_spec("torch") is None:
        return False
    if not CUPY_AVAILABLE:
        return False
    import torch
    return torch.cuda.is_available()


def test_the_host_arm_runs_without_torch_or_cupy(example, monkeypatch):
    """The host arm never imports torch or cupy: block both, it must still work."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "cupy", None)

    results = example.run_demo(batch=32, k=3.0, seed=0, target="host", verbose=False)

    assert results["target"] == "host"
    assert results["exec_targets"] == ["host"]
    assert results["ptr_stable"]
    assert results["max_rel"] <= example.REL_TOL


@pytest.mark.gpu
def test_the_device_arm_exercises_the_certified_rows(example):
    """Pointer identity exactly where a row is ALIAS, a copy where it is COPY,
    checked here against ``ROWS`` directly (not just the example's own
    internal assertion) -- and the results agree with a numpy reference.
    """
    if not _torch_and_gpu_available():
        pytest.skip("no CUDA-capable torch installation visible to this process")
    from raptor.conformance.interop import ROWS

    results = example.run_demo(batch=32, k=3.0, seed=0, target="device", verbose=False)

    assert results["target"] == "device"
    assert results["exec_targets"] == ["device"]
    assert ROWS["T-IN-CUDA-ALIAS"].kind == "ALIAS"
    assert results["in_cuda_alias"] is True
    assert results["in_cuda_alias_behavioural"] is True
    assert ROWS["T-OUT-CUDA-ALIAS"].kind == "ALIAS"
    assert results["out_cuda_alias"] is True
    assert ROWS["T-IN-CPU-COPY"].kind == "COPY"
    assert results["in_cpu_alias"] is False  # COPY means NOT aliased
    assert results["max_rel_cuda"] <= example.REL_TOL
    assert results["max_rel_cpu"] <= example.REL_TOL


def test_a_deliberately_copied_input_fails_the_alias_check(example):
    """Non-vacuity: the SAME identity check the example uses must be able to
    say "no" -- a tensor round-tripped through a fresh host buffer and
    re-uploaded is provably a different device allocation.
    """
    import numpy as np

    if not _torch_and_gpu_available():
        pytest.skip("no CUDA-capable torch installation visible to this process")
    import cupy as cp
    import eagle
    import torch

    t = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4).contiguous()
    cu_alias = eagle.to_cupy(t)
    assert cu_alias.data.ptr == t.data_ptr()  # the real alias, for contrast

    forced_copy = cp.asarray(t.clone().cpu().numpy())  # host round-trip, fresh allocation
    assert forced_copy.data.ptr != t.data_ptr(), (
        "a deliberately copied input passed the alias check -- the check "
        "used by the example is vacuous")
    np.testing.assert_array_equal(cp.asnumpy(forced_copy), t.cpu().numpy())


def test_a_copied_input_does_not_see_an_in_place_mutation(example):
    """Non-vacuity for the T-IN-CUDA-ALIAS BEHAVIOURAL check: bind a COPY of
    the input instead of the alias, mutate the ORIGINAL tensor in place,
    relaunch, and the output must NOT track the mutation -- if it did, the
    behavioural check the example runs would be proving nothing.
    """
    import numpy as np

    if not _torch_and_gpu_available():
        pytest.skip("no CUDA-capable torch installation visible to this process")
    import pathlib
    import tempfile

    import cupy as cp
    import eagle
    import eagle.exec as eexec
    import torch
    from eagle import plan as eplan

    k = 3.0
    with tempfile.TemporaryDirectory() as tmp:
        bundle = example.build_artifact(pathlib.Path(tmp) / "zc", ("cuda",))
        plan = eplan.plan(example.device_plugin(bundle, example.KERNEL_NAME),
                          structure=eexec.DeviceKernel)

        t_in = torch.arange(12, dtype=torch.float32, device="cuda").reshape(3, 4).contiguous()
        cu_in_copy = cp.asarray(eagle.to_cupy(t_in).get())  # a real, disconnected copy
        cu_out = cp.empty_like(cu_in_copy)
        bound = plan.bind(x=cu_in_copy, k=k, y=cu_out)
        bound.launch()
        cp.cuda.runtime.deviceSynchronize()
        before = cp.asnumpy(cu_out).copy()

        t_in.mul_(3)  # mutate the ORIGINAL tensor; cu_in_copy is a separate allocation
        bound.launch()
        cp.cuda.runtime.deviceSynchronize()
        after = cp.asnumpy(cu_out)

        np.testing.assert_array_equal(before, after)  # unaffected: proves it is disconnected


def test_the_copy_cost_demo_agrees_at_a_tiny_size(example):
    """The copy-cost demo, at a tiny size: assert agreement only, no timing."""
    if not _torch_and_gpu_available():
        pytest.skip("no CUDA-capable torch installation visible to this process")

    results = example.run_copy_cost_demo(size_mb=0.01, k=3.0, reps=1, verbose=False)

    assert results["batch"] > 0
    assert results["agreement"] <= example.REL_TOL


def test_the_traced_kernel_declares_its_planes(example):
    """The kernel is ONE per-sample scale: a vector in, a uniform, a vector out."""
    kernel = example.step_kernel()
    roles = {name: role for role, name in kernel.arg_spec}

    assert kernel.name == example.KERNEL_NAME
    assert set(roles) == {"x", "k", "y"}
    assert roles["x"] == "vec_in"
    assert roles["k"] == "uniform"
    assert roles["y"] == "mutable"
