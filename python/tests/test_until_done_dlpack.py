# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.until_done`` over any DLPack producer: planes of torch (and JAX, in
``test_frameworks_jax.py``) are read in place, the caller's array is never
written, and ``Runner.result`` returns the answer as a NEW array of the caller's
type. The device follows the data; a host plane beside a device plane is
refused by name."""

from __future__ import annotations

import numpy as np
import pytest

hawk = pytest.importorskip("hawk")

import eagle  # noqa: E402
import eagle.plan as eplan  # noqa: E402
from hawk import Mutable, Param, Scalar, Terminated  # noqa: E402
from hawk.ext import Guard, Kind  # noqa: E402

STOP, N = 500.0, 1 << 12


@pytest.fixture(scope="module")
def count_compact():
    @hawk.kernel(kind=Kind("dlpack_compact", guard=Guard(active_set=True)), steps=1)
    def count_compact(nstop: Param, terminated: Terminated, k: Mutable[Scalar]):
        k = k + 1.0
        terminated = k >= nstop

    return count_compact


def _k0():
    return np.random.default_rng(0).uniform(0, 490, N)


def _check(rep, out, k0):
    assert rep.done and rep.finished == N
    assert rep.compactions >= 1
    np.testing.assert_allclose(np.asarray(out.cpu() if hasattr(out, "cpu") else out),
                               k0 + np.ceil(STOP - k0), atol=1e-9)


def test_torch_cpu_tensor_is_read_in_place_and_the_result_is_a_new_tensor(count_compact):
    torch = pytest.importorskip("torch")
    k0 = _k0()
    k_in = torch.from_numpy(k0.copy())
    runner = eagle.until_done(eplan.auto(count_compact), max_steps=1000, k=k_in, nstop=STOP)
    rep = runner.run()
    out = runner.result("k")
    _check(rep, out, k0)
    assert isinstance(out, torch.Tensor) and out.data_ptr() != k_in.data_ptr()
    np.testing.assert_array_equal(k_in.numpy(), k0)            # the caller's tensor is untouched
    again = runner.result("k")
    assert again.data_ptr() != out.data_ptr()                  # a fresh array per call


@pytest.mark.gpu
def test_cupy_array_runs_on_the_device_in_place(count_compact):
    cp = pytest.importorskip("cupy")
    k0 = _k0()
    k_in = cp.asarray(k0)
    runner = eagle.until_done(eplan.auto(count_compact), max_steps=1000, k=k_in, nstop=STOP)
    rep = runner.run()
    assert runner.result("k") is k_in                          # numpy and cupy keep the in-place contract
    _check(rep, k_in.get(), k0)


@pytest.mark.gpu
def test_torch_cuda_tensor_gets_a_new_cuda_tensor(count_compact):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    k0 = _k0()
    k_in = torch.from_numpy(k0.copy()).cuda()
    runner = eagle.until_done(eplan.auto(count_compact), max_steps=1000, k=k_in, nstop=STOP)
    rep = runner.run()
    out = runner.result("k")
    _check(rep, out, k0)
    assert out.is_cuda and out.data_ptr() != k_in.data_ptr()
    np.testing.assert_array_equal(k_in.cpu().numpy(), k0)


@pytest.mark.gpu
def test_a_host_plane_beside_a_device_plane_is_refused_by_name(count_compact):
    torch = pytest.importorskip("torch")
    cp = pytest.importorskip("cupy")
    k0 = _k0()
    host_mask = torch.zeros(N, dtype=torch.bool)
    with pytest.raises(ValueError, match=r"'k' is on the device but 'terminated' is on the host"):
        eagle.until_done(eplan.auto(count_compact), max_steps=1000, nstop=STOP,
                         k=cp.asarray(k0), terminated=host_mask)
