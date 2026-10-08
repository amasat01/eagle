# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.frameworks.torch``: a hawk kernel as a ``torch.autograd.Function``.

The subject is one per-sample kernel with every role the bridge serves (a
vector plane, a ``Param``, a ``Terminated`` mask, a vector output), compiled
once per target for the whole module. The CPU rows run wherever torch imports;
the CUDA rows carry the ``gpu`` and ``torch_gpu`` markers and skip where torch is
a CPU-only build.

LAYOUT rows prove the bridge hands outputs back in the CALLER's
layout: sample-major in gives sample-major out (view and copy alike, the
latter warning once per call and never a second time for the output), the
derivative checks still pass in that layout, a component-major call is
unchanged, and per-sample inputs that disagree on layout are refused rather
than guessed (:func:`vec_sum`, the one two-vector-input kernel that scenario
needs).
"""

from __future__ import annotations

import subprocess
import sys
import warnings

import pytest

torch = pytest.importorskip("torch")
hawk = pytest.importorskip("hawk")

from hawk import Mutable, Param, Scalar, Terminated, Vector  # noqa: E402
from hawk.math import dot, sqrt  # noqa: E402

import eagle  # noqa: E402
from eagle import _layout  # noqa: E402
from eagle.frameworks import torch as eagle_torch  # noqa: E402
from eagle.interop import BufferRefused  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_call_sites():
    """Forget every call site a prior test already warned for (same dedup
    state :mod:`eagle._layout` keeps across the whole process)."""
    _layout._reset_seen()
    yield
    _layout._reset_seen()


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.torch_gpu]),
]


@hawk.kernel
def drag(v: Vector[3], cd: Param, terminated: Terminated, a: Mutable[Vector[3]]):
    a = -cd * sqrt(dot(v, v)) * v


def _reference(v, cd, terminated):
    """The same law in plain torch, terminated samples left at zero."""
    a = -cd * v.norm(dim=0) * v
    return a.masked_fill(terminated, 0.0)


@hawk.kernel
def vec_sum(v: Vector[3], w: Vector[3], a: Mutable[Vector[3]]):
    a = v + w


@hawk.kernel(steps=1)
def step_and_finish(k: Scalar, nstop: Param, terminated: Terminated,
                    k_next: Mutable[Scalar]):
    """A counter step that finishes its own samples once it reaches
    ``nstop`` -- the shape the torch bridge hands the updated mask back
    for, instead of the caller recomputing the same decision."""
    k1 = k + 1.0
    k_next = k1
    terminated = k1 >= nstop


@pytest.fixture(scope="module")
def op(tmp_path_factory):
    return eagle_torch.function(drag, cache_dir=tmp_path_factory.mktemp("torch_fn"))


@pytest.fixture(scope="module")
def op_two_vectors(tmp_path_factory):
    return eagle_torch.function(
        vec_sum, cache_dir=tmp_path_factory.mktemp("torch_fn_two_vectors"))


@pytest.fixture(scope="module")
def op_finishing(tmp_path_factory):
    return eagle_torch.function(
        step_and_finish, cache_dir=tmp_path_factory.mktemp("torch_fn_finishing"))


def _device(name):
    if name == "cuda" and not torch.cuda.is_available():
        pytest.skip("torch is a CPU-only build here")
    return torch.device(name)


def _inputs(device, n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(3, n, dtype=torch.float64, generator=g).to(device)
    cd = torch.tensor(0.3, dtype=torch.float64, device=device)
    terminated = torch.zeros(n, dtype=torch.bool, device=device)
    terminated[1::3] = True
    return v, cd, terminated


def _sample_major(v):
    """``v`` (component-major, ``(w, n)``) as an INDEPENDENTLY allocated
    sample-major ``(n, w)`` tensor of equal values — not a transposed view of
    ``v``, so adapting it needs a genuine copy (and a ``LayoutWarning``),
    unlike ``v.T`` which is already the zero-copy case."""
    return v.detach().clone().T.contiguous()


def test_import_eagle_stays_torch_free():
    code = ("import sys, eagle, eagle.frameworks; "
            "bad = sorted(m for m in ('torch', 'hawk') if m in sys.modules); "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_signature_is_read_off_the_kernel(op):
    assert op.inputs == ("v", "cd", "terminated")
    assert op.outputs == ("a",)
    assert set(op.wrt) == {"v", "cd"}
    assert op.finishes_terminated is False


def test_signature_surfaces_the_finished_mask_as_an_extra_output(op_finishing):
    assert op_finishing.inputs == ("k", "nstop", "terminated")
    assert op_finishing.outputs == ("k_next", "terminated")
    assert op_finishing.finishes_terminated is True


@pytest.mark.parametrize("dev", DEVICES)
def test_a_kernel_that_finishes_its_mask_returns_the_updated_mask(op_finishing, dev):
    """eagle API finding: ``_OUTPUT_ROLES`` used to drop ``terminated``, so a
    kernel's own stop decision never came back to the torch caller -- a
    stepping loop had to recompute it with a second, unreconciled masking
    rule. Now the updated mask comes back as the kernel's own extra return
    value, feedable straight into the next step as its input mask."""
    device = _device(dev)
    n = 5
    k = torch.zeros(n, dtype=torch.float64, device=device)
    nstop = torch.tensor(2.0, dtype=torch.float64, device=device)
    terminated = torch.zeros(n, dtype=torch.bool, device=device)

    k, terminated = op_finishing(k, nstop, terminated)
    assert terminated.dtype == torch.bool and terminated.device.type == device.type
    torch.testing.assert_close(k, torch.ones(n, dtype=torch.float64, device=device))
    assert not torch.any(terminated)

    k, terminated = op_finishing(k, nstop, terminated)
    torch.testing.assert_close(
        k, torch.full((n,), 2.0, dtype=torch.float64, device=device))
    assert torch.all(terminated)


@pytest.mark.parametrize("dev", DEVICES)
def test_forward_matches_reference(op, dev):
    v, cd, terminated = _inputs(_device(dev))
    a = op(v, cd, terminated)
    assert a.device == v.device and a.shape == v.shape
    torch.testing.assert_close(a, _reference(v, cd, terminated), rtol=1e-14, atol=0)
    torch.testing.assert_close(op(v=v, cd=0.3, terminated=terminated), a)


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_float64(op, dev):
    v, cd, terminated = _inputs(_device(dev))
    v.requires_grad_()
    cd.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda v, cd: op(v, cd, terminated), (v, cd), check_forward_ad=True,
        check_batched_grad=False)


@pytest.mark.parametrize("dev", DEVICES)
def test_terminated_sample_contributes_zero_gradient(op, dev):
    v, cd, terminated = _inputs(_device(dev))
    v.requires_grad_()
    cd.requires_grad_()
    op(v, cd, terminated).sum().backward()
    assert torch.all(v.grad[:, terminated] == 0)
    assert torch.all(v.grad[:, ~terminated] != 0)

    v_ref = v.detach().clone().requires_grad_()
    cd_ref = cd.detach().clone().requires_grad_()
    _reference(v_ref, cd_ref, terminated).sum().backward()
    torch.testing.assert_close(v.grad, v_ref.grad, rtol=1e-13, atol=0)
    torch.testing.assert_close(cd.grad, cd_ref.grad, rtol=1e-13, atol=0)


@pytest.mark.parametrize("dev", DEVICES)
def test_forward_mode_tangent_runs_the_derived_kernel(op, dev):
    import torch.autograd.forward_ad as fwAD

    v, cd, terminated = _inputs(_device(dev))
    dv = torch.ones_like(v)
    with fwAD.dual_level():
        a = op(fwAD.make_dual(v, dv), cd, terminated)
        tangent = fwAD.unpack_dual(a).tangent
    _, expect = torch.func.jvp(lambda v: _reference(v, cd, terminated), (v,), (dv,))
    torch.testing.assert_close(tangent, expect, rtol=1e-13, atol=0)
    assert torch.all(tangent[:, terminated] == 0)


# --------------------------------------------------------------------------- #
# LAYOUT: the bridge returns outputs in the CALLER's layout.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("dev", DEVICES)
def test_sample_major_input_gives_sample_major_output(op, dev):
    """A sample-major input (here a zero-copy view) brings its output back
    sample-major too, equal to the component-major call transposed; neither
    call copies, so strict mode (every LayoutWarning an error) stays quiet —
    which also proves the component-major call is unchanged."""
    device = _device(dev)
    v, cd, terminated = _inputs(device)
    with warnings.catch_warnings():
        warnings.simplefilter("error", eagle.LayoutWarning)
        a_component = op(v, cd, terminated)
        a_sample = op(v.T, cd, terminated)
    assert a_component.shape == v.shape
    assert a_sample.shape == (v.shape[1], v.shape[0])
    torch.testing.assert_close(a_sample, a_component.T)


@pytest.mark.parametrize("dev", DEVICES)
def test_sample_major_copy_warns_once_per_call(op, dev):
    """A sample-major input that is NOT a zero-copy view is copied in with
    one ``LayoutWarning``; bringing the output back sample-major is always a
    transpose of a buffer this module just allocated, so it never copies and
    never raises a second warning for the same call."""
    device = _device(dev)
    v, cd, terminated = _inputs(device)
    v_sample = _sample_major(v)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        a_sample = op(v_sample, cd, terminated)
    hits = [w for w in record if issubclass(w.category, eagle.LayoutWarning)]
    assert len(hits) == 1
    torch.testing.assert_close(a_sample, op(v, cd, terminated).T)


@pytest.mark.parametrize("dev", DEVICES)
def test_gradcheck_and_forward_ad_sample_major(op, dev):
    """gradcheck (reverse-mode) and ``check_forward_ad`` both still pass when
    the per-sample input — and hence the output — is sample-major."""
    device = _device(dev)
    v, cd, terminated = _inputs(device)
    v_sample = _sample_major(v).requires_grad_()
    cd.requires_grad_()
    assert torch.autograd.gradcheck(
        lambda v_sample, cd: op(v_sample, cd, terminated), (v_sample, cd),
        check_forward_ad=True, check_batched_grad=False)


def test_mixed_layout_across_inputs_is_refused(op_two_vectors):
    """No documented rule says which layout wins when per-sample inputs
    disagree, so the call is refused, naming each input and its layout."""
    v = torch.randn(3, 6, dtype=torch.float64)              # component-major
    w = _sample_major(torch.randn(3, 6, dtype=torch.float64))  # sample-major
    with pytest.raises(ValueError) as excinfo:
        op_two_vectors(v, w)
    message = str(excinfo.value)
    assert "disagree on layout" in message
    assert "'v' (component-major)" in message
    assert "'w' (sample-major)" in message


@pytest.mark.parametrize("dev", DEVICES)
def test_planes_cross_zero_copy(op, dev, monkeypatch):
    """The kernel reads the caller's tensor and writes the returned one: the
    addresses bound to the launch ARE the tensors' storage."""
    device = _device(dev)
    v, cd, terminated = _inputs(device)
    bound = {}
    if dev == "cpu":
        import hawk.runtime

        real = hawk.runtime.run

        def spy(kernel, **arrays):
            bound.update({k: a.ctypes.data for k, a in arrays.items()
                          if hasattr(a, "ctypes")})
            return real(kernel, **arrays)

        monkeypatch.setattr(hawk.runtime, "run", spy)
    else:
        from eagle.plan import Plan

        real = Plan.bind

        def spy(self, /, **planes):
            bound.update({k: a.data.ptr for k, a in planes.items()
                          if hasattr(a, "data")})
            return real(self, **planes)

        monkeypatch.setattr(Plan, "bind", spy)
        torch.cuda.set_sync_debug_mode("error")  # any torch-side sync raises
    try:
        a = op(v, 0.3, terminated)
    finally:
        if dev == "cuda":
            torch.cuda.set_sync_debug_mode("default")
    assert bound == {"v": v.data_ptr(), "terminated": terminated.data_ptr(),
                     "a": a.data_ptr()}


def test_wrong_dtype_is_refused_not_cast(op):
    v, cd, terminated = _inputs("cpu")
    with pytest.raises(BufferRefused, match="dtype"):
        op(v.float(), cd, terminated)


def test_unsupported_role_is_refused():
    from hawk import Accum, Index, Scalar

    @hawk.kernel
    def scatter(x: Scalar, lane: Index, acc: Accum[Scalar]):
        acc.add(x, at=lane)

    with pytest.raises(ValueError, match="acc"):
        eagle_torch.function(scatter)


# --------------------------------------------------------------------------- #
# Stream rows: an adversarial schedule that reads stale data unless the bridge
# rides torch's CURRENT stream (pattern of the interop matrix's STREAM-* rows).
# --------------------------------------------------------------------------- #
def _sleep_cycles(seconds):
    """Cycles for a ``~seconds`` ``torch.cuda._sleep``, calibrated on this device."""
    import time

    probe = 2_000_000
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda._sleep(probe)
    torch.cuda.synchronize()
    return max(int(probe * seconds / max(time.perf_counter() - t0, 1e-6)), probe)


def _adversarial_schedule(op, cycles, busy=None):
    """On a non-default torch stream, no synchronization anywhere: a ~60 ms sleep
    occupies the stream, the input is filled with a sentinel BEHIND it, the bridge
    runs, and a torch op consumes the output immediately. Riding the stream, the
    consumer sees the kernel's answer for the sentinel; launched anywhere else, it
    reads the output before the kernel has written it."""
    n, sentinel = 1 << 12, 2.0
    s = torch.cuda.Stream()
    v = torch.zeros(3, n, dtype=torch.float64, device="cuda")
    terminated = torch.zeros(n, dtype=torch.bool, device="cuda")
    torch.cuda.synchronize()
    if busy is not None:  # the RED twin's stream, occupied for twice as long
        with torch.cuda.stream(busy):
            torch.cuda._sleep(2 * cycles)
    with torch.cuda.stream(s):
        torch.cuda._sleep(cycles)
        v.fill_(sentinel)
        seen = op(v, 0.5, terminated).clone()
    torch.cuda.synchronize()
    return seen, _reference(torch.full_like(v, sentinel), 0.5, terminated)


@pytest.mark.gpu
@pytest.mark.torch_gpu
def test_bridge_rides_torch_current_stream(op):
    _device("cuda")
    seen, expect = _adversarial_schedule(op, _sleep_cycles(0.06))
    torch.testing.assert_close(seen, expect, rtol=1e-14, atol=0)


@pytest.mark.gpu
@pytest.mark.torch_gpu
def test_bridge_on_a_wrong_stream_reads_stale_data(op, monkeypatch):
    """The RED twin: the same schedule with the bridge forced onto ANOTHER stream,
    itself busy for twice as long. The consumer then runs before the kernel, so the
    schedule above is able to fail."""
    _device("cuda")
    cycles = _sleep_cycles(0.06)
    wrong = torch.cuda.Stream()
    monkeypatch.setattr(eagle_torch, "_current_stream",
                        lambda index: wrong.cuda_stream)
    seen, expect = _adversarial_schedule(op, cycles, busy=wrong)
    assert not torch.equal(seen, expect)
    assert torch.all(seen == 0), "the consumer read the output before the launch"
