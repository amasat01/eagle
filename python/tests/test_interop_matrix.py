# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle's interop certification matrix: the 8 eagle-owned rows.

The row catalogue is declared once, in raptor
(``raptor.conformance.interop.ROWS``) — that module is the sole authority
on which rows this package owns; this file does not redesign them. Filtering ``ROWS`` by
``owner_repo == "eagle"`` gives exactly these eight, and this module registers all of
them
(``@register_row``) via :func:`raptor.conformance.interop.register_row`:

* ``T-IN-CUDA-ALIAS``, ``T-OUT-CUDA-ALIAS``, ``T-IN-CPU-COPY`` — the torch
  adapter rows (``eagle.interop._TorchAdapter``).
* ``CP-CAPTURE-REJECT`` — the cupy adapter/pipeline row
  (``eagle.launch.require_cupy`` inside ``GraphPipeline`` capture).
* ``STREAM-TORCH-PRODUCER-ORDER``, ``STREAM-CUPY-PRODUCER-ORDER``,
  ``STREAM-TORCH-CONSUMER-ORDER``, ``STREAM-IDENTITY`` — the stream rows
  (``_TorchAdapter.launch_context`` / ``GraphPipeline.launch``'s event
  fix).

**Hygiene, mechanically enforced by the companion module
(``test_interop_matrix_conformance.py``)**: this file must contain ZERO
``pytest.skip`` / ``skipif`` / ``importorskip`` tokens and must not use any
conftest availability marker (``gpu`` / ``torch`` / ``torch_gpu`` /
``jax_gpu`` / ``tf_gpu`` — ``eagle/python/conftest.py`` auto-skips marked
items, and a silently-skipped matrix row certifies nothing). Every row instead
opens with :func:`~raptor.conformance.interop.certified_framework`, which
``pytest.fail``s (never skips) when a CERTIFIED framework is absent from the
gate environment. Nothing framework-specific is imported at module load (the
same "GPU-free collection" contract ``eagle.interop`` itself documents) — each
test imports torch/cupy lazily, after its ``certified_framework`` guard.

The only marker this module carries is ``interop_matrix``
(``pyproject.toml``), a plain CI-deselection tag — NOT wired into
``conftest.py``'s skip machinery, so it cannot silently skip anything;
``-m "not interop_matrix"`` is the one sanctioned way a row may not run.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from raptor.conformance.interop import certified_framework, register_row

from eagle import GraphPipeline, LoadedVector, detect, to_cupy

pytestmark = pytest.mark.interop_matrix

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
MU = 3.986004418e5  # Earth GM, km^3/s^2


def _make_positions(n=64, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(3, n))
    d /= np.linalg.norm(d, axis=0)
    return np.ascontiguousarray((d * rng.uniform(7.0e3, 4.2e4, n)).astype(np.float64))


def _gravity_ref(pos, mu):
    r = np.linalg.norm(pos, axis=0)
    return -mu * pos / r**3


def _max_rel(g, ref):
    return float(np.max(np.linalg.norm(g - ref, axis=0) / np.linalg.norm(ref, axis=0)))


# --------------------------------------------------------------------------- #
# torch adapter rows (T-IN-CUDA-ALIAS, T-OUT-CUDA-ALIAS, T-IN-CPU-COPY)
# --------------------------------------------------------------------------- #
@register_row("T-IN-CUDA-ALIAS")
def test_torch_cuda_input_dlpack_alias_and_write_through():
    """``eagle.to_cupy(t)`` for a CUDA torch tensor is ``_TorchAdapter.to_cupy``,
    which is literally ``cp.from_dlpack(t)`` (``_TorchAdapter.to_cupy``): a
    zero-copy device view. Three legs: pointer-alias, ``__dlpack_device__``
    (device ordinal) agreement, and write-through (a write on the cupy side is
    visible on the torch side with no copy in between)."""
    certified_framework("torch")
    certified_framework("cupy")
    import cupy as cp
    import torch

    t = torch.arange(24, dtype=torch.float64, device="cuda").reshape(3, 8).contiguous()
    cu = to_cupy(t)
    assert isinstance(cu, cp.ndarray)
    assert cu.data.ptr == t.data_ptr()  # zero-copy device view, shares storage
    # same device ordinal
    assert tuple(cu.__dlpack_device__()) == tuple(t.__dlpack_device__())

    sentinel = 7.25
    cu[0, 0] = sentinel
    torch.cuda.synchronize()
    # write-through: t sees the cupy-side write
    assert float(t[0, 0].item()) == sentinel


@register_row("T-OUT-CUDA-ALIAS")
def test_torch_cuda_output_handoff_and_export_alias():
    """The ``out=`` handoff leg (a caller-provided torch CUDA buffer filled in
    place, no reallocation) PLUS the export leg: the engine's raw cupy result,
    exported back through ``_TorchAdapter.from_cupy`` (== ``torch.from_dlpack``),
    is pointer-equal to the cupy array and write-through in both directions."""
    certified_framework("torch")
    certified_framework("cupy")
    import cupy as cp
    import torch

    plugin = LoadedVector(FIX / "gravity.ptx")
    P = _make_positions(64, seed=1)

    # out= handoff leg.
    t_in = torch.from_numpy(P.copy()).cuda()
    buf = torch.empty((3, P.shape[1]), dtype=torch.float64, device="cuda")
    ptr = buf.data_ptr()
    ret = plugin(position=t_in, mu=MU, out=buf)
    assert ret is buf  # same torch object back
    assert buf.data_ptr() == ptr  # filled in place, no reallocation
    assert _max_rel(cp.asnumpy(cp.asarray(buf)), _gravity_ref(P, MU)) < 1e-12

    # export leg: the engine's raw cupy result, exported via the adapter.
    y_cp = cp.asarray(plugin(position=cp.asarray(P), mu=MU))
    y_t = detect(t_in).from_cupy(y_cp)
    assert isinstance(y_t, torch.Tensor)
    assert y_t.data_ptr() == y_cp.data.ptr  # pointer-equal, no copy on export
    sentinel = 3.5
    y_cp[0, 0] = sentinel
    torch.cuda.synchronize()
    assert float(y_t[0, 0].item()) == sentinel  # write-through


@register_row("T-IN-CPU-COPY")
def test_torch_cpu_input_copies_with_write_isolation():
    """A torch CPU tensor comes back as a torch tensor (type fidelity) via a
    host upload — and, being a copy (not an alias), is write-isolated in BOTH
    directions: mutating the input after the call leaves the output unchanged,
    and vice versa."""
    certified_framework("torch")
    certified_framework("cupy")
    import torch

    plugin = LoadedVector(FIX / "gravity.ptx")
    P = _make_positions(64, seed=2)
    t = torch.from_numpy(P.copy())  # CPU tensor
    out = plugin(position=t, mu=MU)
    assert isinstance(out, torch.Tensor)
    assert not out.is_cuda
    assert _max_rel(out.numpy(), _gravity_ref(P, MU)) < 1e-12

    out_snapshot = out.clone()
    t[0, 0] = 555.0  # mutate the input after the call
    assert torch.equal(out, out_snapshot)  # output unaffected -> not aliased to t

    t_snapshot = t.clone()
    out[0, 0] = -555.0  # mutate the output after the call
    assert torch.equal(t, t_snapshot)  # input unaffected -> not aliased to out


# --------------------------------------------------------------------------- #
# cupy adapter/pipeline row (CP-CAPTURE-REJECT)
# --------------------------------------------------------------------------- #
@register_row("CP-CAPTURE-REJECT")
def test_graph_pipeline_capture_rejects_non_cupy_input():
    """Inside ``GraphPipeline`` capture, a non-cupy input (here: a plain numpy
    array, which allocation-during-capture rules forbid) is rejected by
    ``eagle.launch.require_cupy`` — the documented capturable-path
    restriction, positively tested (not merely inferred from the docstring)."""
    certified_framework("cupy")
    import cupy as cp

    plugin = LoadedVector(FIX / "gravity.ptx")
    n = 32
    P = _make_positions(n, seed=3)
    out = cp.zeros((3, n), dtype=cp.float64)
    term = cp.zeros(n, dtype=cp.bool_)

    pipe = GraphPipeline()
    # numpy, not cupy -- the capturable-path rejection under test
    pipe.add(lambda: plugin.launch(out=out, position=P, mu=MU, terminated=term))
    with pytest.raises(TypeError, match="pre-allocated cupy"):
        pipe.build()


# --------------------------------------------------------------------------- #
# the STREAM-* rows — serial on the device under test; torch._sleep / a
# cupy RawKernel spin manufacture the adversarial schedule instead of
# hoping ambient load triggers it.
# --------------------------------------------------------------------------- #
def _torch_sleep_cycles(seconds: float) -> int:
    """Cycle count for an ``~seconds``-long ``torch.cuda._sleep``, calibrated
    from a timed probe sleep — parametric in the device, no hard-coded clock
    (no-single-target-hardware rule)."""
    import time

    import torch

    probe = 2_000_000
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    torch.cuda._sleep(probe)
    torch.cuda.synchronize()
    dt = max(time.perf_counter() - t0, 1e-6)
    return max(int(probe * (seconds / dt)), probe)


@register_row("STREAM-TORCH-PRODUCER-ORDER")
def test_torch_stream_producer_order_engine_reads_post_sentinel():
    """On a NON-default torch stream: a calibrated ``_sleep`` (>=~50ms),
    then ``t.fill_(sentinel)`` (queued behind the sleep, not yet executed),
    then the engine launch inside the SAME stream context. If the launch does
    not ride ``s``, it runs concurrently with the sleep (sm_61 concurrent
    kernels) and reads pre-sentinel data with certainty — deterministically
    RED-able."""
    certified_framework("torch")
    certified_framework("cupy")
    import torch

    plugin = LoadedVector(FIX / "gravity.ptx")
    n = 128
    sentinel = 11.0
    cycles = _torch_sleep_cycles(0.06)  # >= ~50ms
    s = torch.cuda.Stream()
    t = torch.zeros((3, n), dtype=torch.float64, device="cuda")
    with torch.cuda.stream(s):
        torch.cuda._sleep(cycles)  # occupies s first
        t.fill_(sentinel)  # enqueued behind the sleep
        y = plugin(position=t, mu=MU)  # engine launch inside the same stream context
    torch.cuda.synchronize()
    expected = _gravity_ref(np.full((3, n), sentinel), MU)
    assert _max_rel(y.cpu().numpy(), expected) < 1e-12


_S2_SPIN_SRC = r"""
extern "C" __global__ void s2_producer_spin(long long cycles)
{
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}
"""
_s2_spin_kernel_cache = None


def _s2_spin_kernel():
    global _s2_spin_kernel_cache
    if _s2_spin_kernel_cache is None:
        import cupy as cp

        _s2_spin_kernel_cache = cp.RawKernel(_S2_SPIN_SRC, "s2_producer_spin")
    return _s2_spin_kernel_cache


def _cupy_spin_cycles(seconds: float) -> int:
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(0)
    # CUDA 13 removed the property; cudaDevAttrClockRate (13) is the stable route.
    khz = int(props["clockRate"] if "clockRate" in props
              else cp.cuda.runtime.deviceGetAttribute(13, 0))
    return int(seconds * khz * 1000)


def _s2_leg(stream, sentinel: float, n: int = 64) -> float:
    """Build a fresh one-node pipeline, seed ``position`` on ``stream`` behind
    a >=~50ms spin with NO sync, replay once on ``stream``, and return the
    max-relative error against the analytic reference. The fix
    (``pipeline.py::GraphPipeline.launch``) must make replay-1 see the seed
    regardless of whether ``stream`` is a non-default caller stream or the
    legacy ``Stream.null`` — that asymmetry is exactly what the two
    parametrized legs below probe."""
    import cupy as cp

    plugin = LoadedVector(FIX / "gravity.ptx")
    pos = cp.zeros((3, n), dtype=cp.float64)  # pre-seed value: wrong on purpose
    out = cp.zeros((3, n), dtype=cp.float64)
    term = cp.zeros(n, dtype=cp.bool_)
    pipe = GraphPipeline()
    pipe.add(lambda: plugin.launch(out=out, position=pos, mu=MU, terminated=term))
    pipe.build()

    cycles = _cupy_spin_cycles(0.06)
    spin = _s2_spin_kernel()
    with stream:
        spin((1,), (1,), (cycles,))
        pos[...] = sentinel  # queued behind the spin
        pipe.launch(1)  # NO sync between the seed write and this replay

    expected = _gravity_ref(np.full((3, n), sentinel), MU)
    return _max_rel(cp.asnumpy(out), expected)


@register_row("STREAM-CUPY-PRODUCER-ORDER")
@pytest.mark.parametrize("leg", ["non_default_stream", "legacy_null"])
def test_cupy_pipeline_launch_reflects_seed_on_either_stream(leg):
    certified_framework("cupy")
    import cupy as cp

    stream = (
        cp.cuda.Stream(non_blocking=True)
        if leg == "non_default_stream"
        else cp.cuda.Stream.null
    )
    diff = _s2_leg(stream, sentinel=6.5, n=64)
    assert diff < 1e-10


#: The race window this row tests. The engine's own kernel must still be RUNNING when
#: the consumer op is enqueued, or the row is UNFALSIFIABLE: a fast kernel finishes
#: before any consumer can overtake it, so the result is correct whether or not
#: anything ordered it. Measured on the gate device (P2000): ~3.8 ms at this
#: size vs ~0.02 ms at the original n=128 — which is why the original form could
#: not be reddened by ANY injection tried (not by removing
#: stream adoption, nor either DLPack handshake, nor all three together).
_S3_N = 1 << 22


@register_row("STREAM-TORCH-CONSUMER-ORDER")
def test_torch_stream_consumer_order_same_stream_fifo():
    """Inside the SAME non-default torch stream, consume the engine's CUDA
    output IMMEDIATELY with a torch op, no sync in between.

    The engine's kernel is sized (:data:`_S3_N`) so it is still RUNNING when the
    consumer op is enqueued: if the launch + export did not ride torch's current
    stream, the consumer reads the output buffer MID-WRITE and the assertion
    fails. Proven RED-able by an injection test (stream
    adoption AND both DLPack handshakes removed).

    The earlier form used ``n=128`` and asserted that "same-stream FIFO is the
    only thing making this correct" — that claim was FALSE. At that size the
    kernel simply completed first, and no injection could redden the row."""
    certified_framework("torch")
    certified_framework("cupy")
    import torch

    plugin = LoadedVector(FIX / "gravity.ptx")
    n = _S3_N
    P = _make_positions(n, seed=5)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        t = torch.from_numpy(P.copy()).cuda()
        y = plugin(position=t, mu=MU)  # engine launch + export, same stream
        y2 = y * 2.0  # immediate torch consumer op, NO sync
    torch.cuda.synchronize()
    expected = _gravity_ref(P, MU)
    assert _max_rel(y.cpu().numpy(), expected) < 1e-12
    assert _max_rel(y2.cpu().numpy(), 2.0 * expected) < 1e-12


@register_row("STREAM-IDENTITY")
def test_torch_adapter_launch_context_stream_identity():
    """The mechanism-level identity behind ``_TorchAdapter.launch_context``
    (``_TorchAdapter.launch_context``): inside the launch context, cupy's current
    stream handle IS torch's current stream handle."""
    certified_framework("torch")
    certified_framework("cupy")
    import cupy as cp
    import torch

    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        t = torch.zeros(3, dtype=torch.float64, device="cuda")
        adapter = detect(t)
        assert adapter.name == "torch" and adapter.cuda
        with adapter.launch_context():
            cupy_ptr = cp.cuda.get_current_stream().ptr
            torch_ptr = torch.cuda.current_stream().cuda_stream
            assert cupy_ptr == torch_ptr
