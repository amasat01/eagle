# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""(a) + (c) — the eagle small caches.

Carried forward from an earlier window, which landed the code generator's own
mirror of (a) in its host-provider ``HostLibrary``:

* **(a) the host ctypes ARG-PACK, cached per (binding generation, n)**
  (:class:`eagle.host_launch.HostPluginLibrary`). ``run()`` rebuilt the whole
  ``params[]`` array and re-boxed every argument on every call — the engine
  review measured that at ~54% of a WARM host call. It is a pure function of
  ``(the bindings, n)`` and a warm host loop rebinds the SAME pointers call
  after call, so it was being rebuilt for nothing. NOTE this is the class the
  ★ provider dispatch prefers whenever eagle is importable, so it is the
  production host path, not a spare one.
* **(c) the ``ExternalStream`` cache** (:mod:`eagle.interop`).
  ``_TorchAdapter.launch_context`` runs ONCE PER LAUNCH and built a fresh cupy
  ``ExternalStream`` wrapper around torch's current stream every time; the
  wrapper creates and owns nothing, so one per raw handle is enough.

Host-only: (a) compiles a tiny plugin ``.so`` with g++ (the same miniature
host-emission toolchain ``test_host_launch_torch.py`` uses —
ABI-only), and (c) is exercised through a stand-in ``cupy``. Each cache has a
PARITY claim, a COUNTER proving it is actually hit, and an executable RED leg.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import types

import numpy as np
import pytest

from eagle import interop
from eagle.host_launch import HostPluginLibrary

# --------------------------------------------------------------------------- #
# (a) — the host arg-pack cache.
# --------------------------------------------------------------------------- #
# out[i] = a[i] * k. Depends only on the ABI POD, exactly like a deployed
# plugin: a scalar ``mutable`` output, a ``per_sample`` input, a by-value
# ``uniform``, and the ``nsamples`` count.
_PLUGIN_SRC = r"""
#include <stdint.h>
struct ScalarHandle { void* ptr; };
extern "C" void eh15_scale_host(void* const* p, int n) {
    double* out       = (double*)((const ScalarHandle*)p[0])->ptr;
    const double* a   = (const double*)((const ScalarHandle*)p[1])->ptr;
    const double k    = *(const double*)p[2];
    const uint32_t ns = *(const uint32_t*)p[3];
    const int lim = (int)ns < n ? (int)ns : n;
    for (int i = 0; i < lim; ++i)
        out[i] = a[i] * k;
}
"""

_SIDECAR = {
    "kernel": "eh15_scale",
    "aether_abi": "aether-abi/1",
    "host_entry": "eh15_scale_host",
    "arg_spec": [
        ["mutable", "out"],
        ["per_sample", "a"],
        ["uniform", "k"],
        ["nsamples", "n"],
    ],
    "mutables": [{"name": "out", "dtype": "float", "width": 1}],
}


def _gxx():
    if shutil.which(os.environ.get("CXX", "")):
        return os.environ.get("CXX")
    return shutil.which("g++") or (
        "/usr/bin/g++" if os.path.exists("/usr/bin/g++") else None
    )


@pytest.fixture(scope="module")
def plugin_so(tmp_path_factory):
    gxx = _gxx()
    if gxx is None:
        pytest.skip("no g++ available to build the host plugin fixture")
    tmp_path = tmp_path_factory.mktemp("eh15_plugin")
    src = tmp_path / "eh15_scale_host.cpp"
    src.write_text(_PLUGIN_SRC)
    so = tmp_path / "eh15_scale_host.so"
    proc = subprocess.run(
        [gxx, "-O2", "-shared", "-fPIC", "-o", str(so), str(src)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"host plugin compile failed: {proc.stderr.strip()[:400]}")
    return str(so)


def _bind_and_run(lib, out, a, k, n):
    """One warm-loop iteration, shaped exactly like the provider layer above:
    rebind everything (it does not track what changed), then run."""
    lib.bind_handle("out", interop.host_ptr(out))
    lib.bind_handle("a", interop.host_ptr(a))
    lib.bind_uniform("k", k)
    return lib.run(n)


def test_the_arg_pack_is_packed_once_across_six_identical_launches(plugin_so):
    """THE counter gate (mirrors the code generator's landed twin): six launches that
    rebind the SAME pointers and the SAME uniform must pack exactly once."""
    lib = HostPluginLibrary(plugin_so, _SIDECAR)
    n = 512
    a = np.arange(n, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)

    reps = 6
    for _ in range(reps):
        _bind_and_run(lib, out, a, 3.0, n)

    stats = lib._pack_stats()
    assert stats["misses"] == 1, (
        f"the arg pack was rebuilt {stats['misses']} times for {reps} identical "
        "launches — the generation counter is moving on an idempotent rebind"
    )
    assert stats["hits"] == reps - 1
    assert np.array_equal(out, a * 3.0)


def test_warm_is_byte_equal_to_cold(plugin_so):
    """PARITY: the cached pack must produce the SAME bytes a cold pack does."""
    n = 257
    a = np.linspace(-3.0, 4.0, n)

    cold = np.zeros(n, dtype=np.float64)
    cold_lib = HostPluginLibrary(plugin_so, _SIDECAR)
    _bind_and_run(cold_lib, cold, a, -1.25, n)
    assert cold_lib._pack_stats() == {"hits": 0, "misses": 1}

    warm = np.zeros(n, dtype=np.float64)
    warm_lib = HostPluginLibrary(plugin_so, _SIDECAR)
    for _ in range(8):
        warm.fill(0.0)
        _bind_and_run(warm_lib, warm, a, -1.25, n)
    assert warm_lib._pack_stats()["hits"] == 7
    assert cold.tobytes() == warm.tobytes()


def test_red_leg_a_different_output_buffer_repacks(plugin_so):
    """RED LEG, executable: the cache must NOT survive a binding that actually
    changed. Two different buffers -> two packs; if the counter did not move
    here, the cached pack would be writing through the PREVIOUS buffer's
    pointer — a silent wrong-memory write, the worst failure this cache could
    have."""
    lib = HostPluginLibrary(plugin_so, _SIDECAR)
    n = 64
    a = np.arange(n, dtype=np.float64)
    first = np.zeros(n, dtype=np.float64)
    second = np.zeros(n, dtype=np.float64)
    assert first.ctypes.data != second.ctypes.data

    _bind_and_run(lib, first, a, 2.0, n)
    before = lib._pack_stats()["misses"]
    _bind_and_run(lib, second, a, 2.0, n)
    assert lib._pack_stats()["misses"] > before, (
        "a DIFFERENT output buffer reused the cached pack"
    )
    assert np.array_equal(second, a * 2.0)
    assert np.array_equal(first, a * 2.0)  # untouched by the second launch


def test_a_changed_uniform_repacks_and_the_answer_follows_it(plugin_so):
    """A ``uniform`` is boxed BY VALUE, so a changed Param must repack — and
    the result must follow the new value, not the packed old one."""
    lib = HostPluginLibrary(plugin_so, _SIDECAR)
    n = 32
    a = np.ones(n, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)

    _bind_and_run(lib, out, a, 5.0, n)
    assert np.array_equal(out, a * 5.0)
    before = lib._pack_stats()["misses"]
    _bind_and_run(lib, out, a, -2.0, n)
    assert lib._pack_stats()["misses"] > before, "a changed uniform reused the pack"
    assert np.array_equal(out, a * -2.0)


def test_a_changed_batch_size_repacks(plugin_so):
    """``n`` is part of the pack (it is boxed as the ``nsamples`` argument), so
    it is part of the cache key."""
    lib = HostPluginLibrary(plugin_so, _SIDECAR)
    a = np.arange(128, dtype=np.float64)
    out = np.zeros(128, dtype=np.float64)

    _bind_and_run(lib, out, a, 1.0, 128)
    before = lib._pack_stats()["misses"]
    _bind_and_run(lib, out, a, 1.0, 64)
    assert lib._pack_stats()["misses"] > before, "a changed n reused the pack"


def test_signed_zero_rebind_invalidates_the_pack(plugin_so):
    """``-0.0 == 0.0`` is True in Python and the two are NOT interchangeable in
    a kernel, so the change test is BIT-exact (the same guard the code generator's twin
    carries). A plain ``==`` test would silently keep the +0.0 pack."""
    lib = HostPluginLibrary(plugin_so, _SIDECAR)
    n = 8
    a = np.ones(n, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)

    _bind_and_run(lib, out, a, 0.0, n)
    before = lib._pack_stats()["misses"]
    _bind_and_run(lib, out, a, -0.0, n)
    assert lib._pack_stats()["misses"] > before, (
        "a rebind from +0.0 to -0.0 reused the cached pack — the change test "
        "is comparing with == instead of bit-exactly"
    )
    assert np.array_equal(np.copysign(1.0, out), np.full(n, -1.0))


def test_red_leg_defeating_the_cache_zeroes_the_hits(plugin_so):
    """RED LEG, executable: with the generation counter defeated (bumped on
    every rebind, as an unconditional ``self._X[name] = value`` would do) the
    hit counter the gate above reads stays at zero."""
    lib = HostPluginLibrary(plugin_so, _SIDECAR)
    n = 16
    a = np.arange(n, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    for _ in range(6):
        lib._pack_gen += 1  # <- the defect: an idempotent rebind invalidates
        _bind_and_run(lib, out, a, 1.5, n)
    assert lib._pack_stats()["hits"] == 0
    assert np.array_equal(out, a * 1.5)  # ...and it is still CORRECT, just slow


def test_an_unconsolidated_lookup_still_raises_on_the_miss_path(plugin_so):
    """The validation the miss path performs is not re-run on a hit (it cannot
    newly fail), but it must still fire on the FIRST pack."""
    sidecar = dict(_SIDECAR)
    sidecar["arg_spec"] = [["mutable", "out"], ["lookup", "tab"], ["nsamples", "n"]]
    sidecar["buffers"] = [{"name": "tab", "kind": "lookup", "count": 4}]
    lib = HostPluginLibrary(plugin_so, sidecar)
    out = np.zeros(4, dtype=np.float64)
    lib.bind_handle("out", interop.host_ptr(out))
    with pytest.raises(ValueError, match="was not supplied at consolidation"):
        lib.run(4)
    assert lib._pack_stats() == {"hits": 0, "misses": 1}


# --------------------------------------------------------------------------- #
# (c) — the ExternalStream cache.
# --------------------------------------------------------------------------- #
class _FakeExternalStream:
    constructed = 0

    def __init__(self, ptr):
        _FakeExternalStream.constructed += 1
        self.ptr = ptr


@pytest.fixture
def fake_cupy_streams(monkeypatch):
    _FakeExternalStream.constructed = 0
    mod = types.ModuleType("cupy")
    mod.cuda = types.SimpleNamespace(ExternalStream=_FakeExternalStream)
    monkeypatch.setitem(sys.modules, "cupy", mod)
    interop._reset_external_stream_cache()
    yield _FakeExternalStream
    interop._reset_external_stream_cache()


def test_the_same_stream_handle_returns_the_same_wrapper(fake_cupy_streams):
    first = interop.external_stream(0xDEAD)
    for _ in range(9):
        assert interop.external_stream(0xDEAD) is first
    assert fake_cupy_streams.constructed == 1
    assert interop._external_stream_stats() == {"hits": 9, "misses": 1}


def test_a_different_stream_handle_gets_its_own_wrapper(fake_cupy_streams):
    a = interop.external_stream(0x1000)
    b = interop.external_stream(0x2000)
    assert a is not b
    assert a.ptr == 0x1000 and b.ptr == 0x2000
    assert fake_cupy_streams.constructed == 2


def test_the_wrapper_wraps_exactly_the_requested_handle(fake_cupy_streams):
    """PARITY: a cached wrapper must be indistinguishable from a fresh one —
    it wraps the handle it was asked for, and nothing else."""
    for ptr in (0, 1, 0xFFFF_FFFF, 0x7FFF_FFFF_FFFF):
        assert interop.external_stream(ptr).ptr == ptr


def test_red_leg_defeating_the_stream_cache_zeroes_the_hits(fake_cupy_streams):
    """RED LEG, executable: with the cache cleared between calls (i.e. the
    uncached per-call construction) the hit counter stays at zero and a
    new wrapper is built every time."""
    for _ in range(9):
        interop._reset_external_stream_cache()
        interop.external_stream(0xDEAD)
    assert interop._external_stream_stats()["hits"] == 0
    assert fake_cupy_streams.constructed == 9


@pytest.mark.torch
def test_torch_launch_context_reuses_the_cached_external_stream():
    """The real consumer: ``_TorchAdapter.launch_context`` is called once per
    launch, so on a steady stream it must stop constructing wrappers."""
    import torch

    from eagle.interop import _TorchAdapter

    if not torch.cuda.is_available():  # pragma: no cover - marker-gated
        pytest.skip("torch CUDA not available")
    interop._reset_external_stream_cache()
    adapter = _TorchAdapter(cuda=True)
    first = adapter.launch_context()
    for _ in range(5):
        assert adapter.launch_context() is first
    assert interop._external_stream_stats()["hits"] == 5
