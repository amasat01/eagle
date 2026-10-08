# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The NVIDIA Warp interop certification rows (eagle-owned, declared in
``raptor.conformance.interop.ROWS``): ``WP-IN-CUDA-ALIAS``,
``WP-OUT-CUDA-ALIAS`` and ``STREAM-WARP-PRODUCER-ORDER``.

Kept separate from ``test_interop_matrix.py`` rather than appended to it:
the two files' rows are independent and additive, and nothing here touches
that module. Warp crosses through eagle's GENERIC zero-copy buffer layer,
:func:`eagle.interop.import_buffer` — the same layer numpy/cupy/torch/a C++
producer use (``docs/content/interop_contract.rst``) — not the
plugin-launch convenience ``Adapter``/``to_cupy`` path
(``eagle.interop.detect``), which never needed a dedicated Warp adapter:
any ``__dlpack__`` + ``__dlpack_device__`` producer already routes through
``_DlpackAdapter`` there. ``import_buffer`` is what these rows certify.

Warp speaks **legacy (pre-1.0) DLPack**: ``warp.array.__dlpack__`` takes no
``max_version`` keyword, so ``eagle.interop._core_buffer_from``'s versioned
call raises ``TypeError`` and falls back to the legacy call — the same path
every pre-1.0 producer takes. A legacy capsule carries no read/write flag, so
``BufferView.access`` reports ``"unknown"`` (never upgraded by the layer
itself; see ``docs/content/interop_contract.rst``'s buffer-model table).

Three rows:

* ``WP-IN-CUDA-ALIAS`` — ``import_buffer(warp_array)`` is a zero-copy device
  view: pointer identity, ``access == "unknown"``, and write-through.
* ``WP-OUT-CUDA-ALIAS`` — the reverse crossing, ``wp.from_dlpack(view)`` of
  an eagle-side (cupy) buffer: pointer-equal, write-through in BOTH
  directions (a cupy-side write is visible to warp, and a warp-side write —
  via ``wp.array.fill_``, no copy — is visible to cupy).
* ``STREAM-WARP-PRODUCER-ORDER`` — the stream contract. A write enqueued on a
  dedicated ``wp.Stream``, with no host synchronization, is visible to a
  reader on a caller-controlled consumer stream exactly when
  ``import_buffer`` is told that stream (the DLPack stream-code contract: an
  explicit ``cudaStream_t`` handle orders the producer's pending work before
  it) — manufactured the same way as the torch and cupy producer-order
  rows (a calibrated occupier, then a same-stream write, no sync). Also exercises
  ``wp.Stream(cuda_stream=...)`` directly, wrapping the consumer's own raw
  handle (the warp mirror of ``eagle.interop.external_stream``'s
  ``cupy.cuda.ExternalStream`` wrap of a torch handle) and checking the
  identity round-trips — kept off the adversarial path itself (see below).
  Verified adversarially during development (not committed, to match
  the other stream rows' own precedent): the documented opt-out ``stream=-1`` ("no
  ordering: the caller takes responsibility") in place of the real
  consumer-stream handle makes warp's own ``__dlpack__`` skip its internal
  wait, and the read races ahead and deterministically reads the PRE-write
  value instead of the sentinel — so this row is falsifiable, not vacuously
  green. (A variant producer built via
  ``wp.Stream(cuda_stream=<a cupy-owned handle>)`` was tried and measured to
  NOT reproduce that falsifiability reliably — root cause not fully
  isolated, so the row keeps the proven-falsifiable native ``wp.Stream``
  producer instead.)

  Why the producer runs under ``wp.ScopedStream``: ``warp.array.__dlpack__``
  fences the GIVEN stream
  against ``self.device.stream`` — warp's own notion of "this device's
  CURRENT stream" — never against whatever stream last actually wrote the
  array (warp tracks no per-array producer stream). So the dedicated
  ``producer`` must ALSO be made the device's current stream (``with
  wp.ScopedStream(producer):``, spanning the real spin/fill launches AND the
  ``import_buffer`` call itself, since that is where ``__dlpack__`` reads
  ``self.device.stream``) or the fence targets an unrelated, already-idle
  stream and establishes no real dependency at all. Without that scope the
  row still PASSED in isolation — by accident, not by the DLPack contract:
  ``cp.from_dlpack(view) * 1.0``'s first-use NVRTC compile of cupy's generic
  multiply happened to outlast the spin+fill on the GPU, masking the missing
  fence. Once any earlier test in the same process had already warmed that
  exact compile path (two independent ``cp.ElementwiseKernel`` compiles were
  enough to reproduce it standalone — nothing eagle- or
  GraphPipeline-specific), the padding vanished and the row read the
  pre-write sentinel deterministically. ``wp.ScopedStream`` fixes this with a
  device-side (non-host-blocking) current-stream switch, so the row no
  longer depends on incidental compile-timing padding.

Hygiene mirrors ``test_interop_matrix.py`` exactly: zero
``pytest.skip``/``skipif``/``importorskip`` tokens, no conftest
availability marker on these functions (only the plain ``interop_matrix``
CI-deselection tag — not wired into ``conftest.py``'s skip machinery), and
every row opens with :func:`~raptor.conformance.interop.certified_framework`,
which ``pytest.fail``s (never skips) when a certified framework is absent.
Nothing warp- or cupy-specific is imported at module load — the two warp
kernels the stream row needs are built lazily, inside the test, by
:func:`_warp_kernels` below.
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from raptor.conformance.interop import certified_framework, register_row

from eagle.interop import import_buffer

pytestmark = pytest.mark.interop_matrix

#: the CUDA device this file exercises -- the only one a CUDA_VISIBLE_DEVICES
#: gate exposes, matching every other row in this suite (no device is
#: hard-coded by name or model).
_DEVICE = "cuda:0"


@register_row("WP-IN-CUDA-ALIAS")
def test_warp_cuda_input_import_buffer_alias_access_and_write_through():
    """``import_buffer`` on a CUDA ``wp.array`` is a zero-copy device view:
    pointer-alias, device agreement, the legacy-DLPack access flag
    (``"unknown"``), and write-through (a write through the returned cupy
    view is visible on the warp side, with no copy in between)."""
    certified_framework("warp")
    certified_framework("cupy")
    import cupy as cp
    import warp as wp

    wp.init()
    n = 64
    rng = np.random.default_rng(0)
    host = rng.normal(size=n).astype(np.float64)
    a = wp.array(host, dtype=wp.float64, device=_DEVICE)

    view = import_buffer(a)
    assert view.ptr == a.ptr  # zero-copy device view, shares storage
    assert view.access == "unknown"  # legacy DLPack: no read/write flag
    assert view.owner == "external"
    assert view.producer == "warp.array"
    assert view.device == a.__dlpack_device__()  # same (device_type, device_id)

    sentinel = 7.25
    cu = cp.from_dlpack(view)
    cu[0] = sentinel
    cp.cuda.Stream.null.synchronize()
    assert float(a.numpy()[0]) == sentinel  # write-through: warp sees it


@register_row("WP-OUT-CUDA-ALIAS")
def test_warp_cuda_output_export_wp_from_dlpack_alias_and_write_through():
    """The export leg: an eagle-side (cupy) buffer, imported and handed to
    ``wp.from_dlpack``, is pointer-equal and write-through in both
    directions."""
    certified_framework("warp")
    certified_framework("cupy")
    import cupy as cp
    import warp as wp

    wp.init()
    n = 64
    out_cp = cp.arange(n, dtype=cp.float64)
    view = import_buffer(out_cp)
    w = wp.from_dlpack(view)
    assert isinstance(w, wp.array)
    assert w.ptr == out_cp.data.ptr  # zero-copy device view, shares storage

    sentinel = -3.5
    out_cp[1] = sentinel
    cp.cuda.Stream.null.synchronize()
    assert float(w.numpy()[1]) == sentinel  # cupy write visible on the warp side

    sentinel2 = 42.0
    w.fill_(sentinel2)  # warp-side write, no copy
    cp.cuda.Stream.null.synchronize()
    assert bool((cp.asnumpy(out_cp) == sentinel2).all())  # visible on the cupy side


# --------------------------------------------------------------------------- #
# STREAM-WARP-PRODUCER-ORDER -- serial on the device under test, same recipe as
# the torch (a calibrated occupier) and cupy (a RawKernel spin)
# producer-order rows: the two warp kernels below are built lazily (first
# call), cached module-globally by warp's own source-hash cache, never
# imported/compiled at collection.
# --------------------------------------------------------------------------- #
_wp_kernel_cache: dict = {}


def _warp_kernels():
    """``(spin, fill)`` warp kernels, built once and cached.

    ``spin`` is a single-thread busy loop over ``iters`` float64 adds,
    written to ``out`` so the compiler cannot discard it as dead code --
    real, calibratable GPU occupancy, the warp-kernel equivalent of
    ``torch.cuda._sleep``/the cupy ``RawKernel`` spin the torch and cupy
    producer-order rows use.
    ``fill`` writes ``value`` into every element of ``out``."""
    if not _wp_kernel_cache:
        import warp as wp

        @wp.kernel
        def spin(iters: int, out: wp.array(dtype=wp.float64)):
            i = wp.tid()
            x = wp.float64(0.0)
            for _ in range(iters):
                x = x + wp.float64(1.0)
            out[i] = x

        @wp.kernel
        def fill(value: wp.float64, out: wp.array(dtype=wp.float64)):
            i = wp.tid()
            out[i] = value

        _wp_kernel_cache["spin"] = spin
        _wp_kernel_cache["fill"] = fill
    return _wp_kernel_cache["spin"], _wp_kernel_cache["fill"]


def _warp_spin_iters(device: str, stream, seconds: float) -> int:
    """Iteration count for an ``~seconds``-long spin, calibrated from a timed
    probe launch on ``stream`` -- parametric in the device, no hard-coded
    cycle count (mirrors eagle's own ``_torch_sleep_cycles``/
    ``_cupy_spin_cycles`` in ``test_interop_matrix.py``). A throwaway warm-up
    launch absorbs the kernel's first-use JIT compile so it never pollutes
    the timed probe."""
    import warp as wp

    spin, _ = _warp_kernels()
    scratch = wp.zeros(1, dtype=wp.float64, device=device)
    probe = 200_000
    kw = dict(dim=1, inputs=[probe], outputs=[scratch], device=device, stream=stream)
    wp.launch(spin, **kw)
    wp.synchronize_stream(stream)  # warm-up: absorb first-use JIT compile
    t0 = time.perf_counter()
    wp.launch(spin, **kw)
    wp.synchronize_stream(stream)
    dt = max(time.perf_counter() - t0, 1e-6)
    return max(int(probe * (seconds / dt)), probe)


@register_row("STREAM-WARP-PRODUCER-ORDER")
def test_warp_stream_producer_order_consumer_reads_post_sentinel():
    """A calibrated spin, then a fill (queued behind it), both enqueued on a
    dedicated ``wp.Stream`` with no synchronization; a reader on a
    caller-controlled cupy stream must see the sentinel exactly because
    ``import_buffer`` is told that consumer stream (an explicit
    ``cudaStream_t`` handle in the DLPack stream-code contract), which makes
    warp's own ``__dlpack__`` wait for the spin+fill before the consumer
    stream proceeds -- no host synchronization anywhere in between.

    Also exercises ``wp.Stream(cuda_stream=...)`` directly -- the idiomatic
    way to run warp kernels on a stream another library owns (the warp
    mirror of ``eagle.interop.external_stream``'s
    ``cupy.cuda.ExternalStream`` wrap of a torch handle): wrapping the
    consumer's own raw handle must round-trip its identity exactly."""
    certified_framework("warp")
    certified_framework("cupy")
    import cupy as cp
    import warp as wp

    wp.init()
    spin, fill = _warp_kernels()
    n = 128
    sentinel = 11.0

    producer = wp.Stream(_DEVICE)
    iters = _warp_spin_iters(_DEVICE, producer, 0.06)  # >= ~50ms
    data = wp.zeros(n, dtype=wp.float64, device=_DEVICE)
    scratch = wp.zeros(1, dtype=wp.float64, device=_DEVICE)

    consumer = cp.cuda.Stream(non_blocking=True)
    # wp.Stream(cuda_stream=...) wraps the consumer's own raw handle: the
    # idiomatic way a warp kernel would ride a stream eagle/cupy owns.
    # Wrapping must round-trip the identity exactly -- checked directly,
    # off the adversarial path below (it is the import_buffer stream=
    # parameter, not this wrap, that the ordering assertion depends on).
    wrapped_consumer = wp.Stream(_DEVICE, cuda_stream=int(consumer.ptr))
    assert wrapped_consumer.cuda_stream == int(consumer.ptr)

    # wp.ScopedStream makes `producer` warp's CURRENT stream for this device
    # (a device-side, non-host-blocking current-stream switch) -- required
    # because warp.array.__dlpack__ fences its given stream against
    # self.device.stream, never against whichever stream last wrote the
    # array (see this module's docstring). The scope must cover
    # the real spin/fill launches AND the import_buffer call itself, since
    # that is where __dlpack__ reads self.device.stream.
    with wp.ScopedStream(producer):
        wp.launch(
            spin, dim=1, inputs=[iters], outputs=[scratch], device=_DEVICE,
            stream=producer,
        )
        wp.launch(
            fill, dim=n, inputs=[sentinel], outputs=[data], device=_DEVICE,
            stream=producer,
        )

        # the DLPack stream-code contract: pass the real consumer handle so
        # warp's own __dlpack__ fences the consumer stream against the
        # producer (self.device.stream, which producer now is, for as long
        # as this scope is open).
        view = import_buffer(data, stream=int(consumer.ptr))
    with consumer:
        out = cp.from_dlpack(view) * 1.0  # a real op on the consumer stream
        result = out.get(stream=consumer)  # read + sync, same stream throughout
    assert np.all(result == sentinel), result[:8]
