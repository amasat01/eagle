# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The JAX interop certification rows (eagle-owned, declared in
``raptor.conformance.interop.ROWS``): ``JX-IN-CUDA-ALIAS``,
``JX-IN-WRITE-REFUSED``, ``JX-OUT-CUDA-ALIAS``, ``JX-IN-CPU-COPY`` and
``STREAM-JAX-PRODUCER-ORDER``.

JAX crosses through eagle's DLPack layer (``eagle.interop``), the same way
every other framework does. JAX arrays are immutable and JAX speaks only
legacy DLPack (no read-only flag), which shapes the contract:

* In: a jax array in a read-only role is aliased on CUDA (uploaded on the
  CPU). A jax array in a role the kernel writes is refused by name, because
  a copy would silently drop the kernel's update.
* Out: on CUDA the jax result IS eagle's result buffer, and eagle never
  writes it again. On the CPU the result is a copy.
* Stream: eagle imports a jax array with its current stream passed to
  ``__dlpack__`` and launches on that stream, so a jax operation still
  running on the GPU is waited for on the device, not by the host. The
  output is synchronized before it goes back to jax.

``STREAM-JAX-PRODUCER-ORDER`` needs a jax operation that is really still
running. ``fixtures/jax_spin.cu`` is a small ``jax.ffi`` kernel that spins
for a set number of GPU cycles, built with nvcc into the pytest basetemp the
first time it is needed. A missing nvcc fails the row.

Hygiene is the same as ``test_interop_matrix.py``: no skip tokens, no
availability marker, only the ``interop_matrix`` tag, and every row opens with
``certified_framework``, which fails (never skips) when a certified framework
is absent. Nothing framework-specific is imported at module load.
"""

from __future__ import annotations

import ctypes
import gc
import json
import os
import pathlib
import shutil
import subprocess
import time

# JAX takes most of the GPU's memory unless told otherwise, which would starve
# the cupy and torch rows sharing this process. Set before any jax import.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest
from raptor.conformance.interop import certified_framework, register_row

from eagle import LoadedPure, LoadedVector
from eagle.frameworks.cupy import to_cupy
from eagle.interop import BufferRefused, host_ptr, import_buffer

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


def _jax():
    import jax

    jax.config.update("jax_enable_x64", True)  # the kernels are float64
    return jax


def _gpu(jax):
    return jax.devices("gpu")[0]


def _cpu(jax):
    return jax.devices("cpu")[0]


@register_row("JX-IN-CUDA-ALIAS")
def test_jax_cuda_input_aliases_in_a_read_only_role():
    """A CUDA jax array passed as a kernel input is imported without a copy:
    pointer-equal, same device ordinal, the access flag is the legacy-DLPack
    ``"unknown"`` (``"read-only"`` once jax ships versioned capsules), the
    result matches the analytic reference, and the input is unchanged."""
    certified_framework("jax")
    certified_framework("cupy")
    import cupy as cp

    jax = _jax()
    plugin = LoadedVector(FIX / "gravity.ptx")
    P = _make_positions(64, seed=1)
    x = jax.device_put(P, _gpu(jax))
    ptr = x.unsafe_buffer_pointer()

    view = import_buffer(x)
    assert view.ptr == ptr
    assert view.access in ("unknown", "read-only")
    assert view.producer == "jaxlib.ArrayImpl"

    cu = to_cupy(x)
    assert isinstance(cu, cp.ndarray)
    assert cu.data.ptr == ptr  # shares storage
    assert tuple(cu.__dlpack_device__()) == tuple(x.__dlpack_device__())

    y = plugin(position=x, mu=MU)
    assert _max_rel(np.asarray(y), _gravity_ref(P, MU)) < 1e-12
    assert np.array_equal(np.asarray(x), P)  # the input was only read
    assert x.unsafe_buffer_pointer() == ptr


@register_row("JX-IN-WRITE-REFUSED")
def test_jax_array_in_a_written_role_is_refused_by_name(tmp_path):
    """A jax array passed where the kernel writes (``out=``, ``terminated``,
    a ``Mutable`` plane, a written host buffer) raises ``BufferRefused``
    naming the role and jax's immutability, and is left byte-unchanged. A
    ``terminated`` mask the plugin declares read-only is not a written role
    and is accepted."""
    certified_framework("jax")
    certified_framework("cupy")
    import cupy as cp

    jax = _jax()
    gravity = LoadedVector(FIX / "gravity.ptx")
    P = _make_positions(64, seed=3)
    n = P.shape[1]
    pos = cp.asarray(P)

    # out= (device)
    out = jax.device_put(np.full((3, n), 5.0), _gpu(jax))
    with pytest.raises(BufferRefused, match=r"jax array.*out=.*immutable"):
        gravity(position=pos, mu=MU, out=out)
    assert np.all(np.asarray(out) == 5.0)

    # out= (CPU jax array)
    out_cpu = jax.device_put(np.full((3, n), 5.0), _cpu(jax))
    with pytest.raises(BufferRefused, match=r"jax array.*out="):
        gravity(position=pos, mu=MU, out=out_cpu)
    assert np.all(np.asarray(out_cpu) == 5.0)

    # terminated, on a plugin that does not declare it read-only
    mask = jax.device_put(np.zeros(n, dtype=bool), _gpu(jax))
    with pytest.raises(BufferRefused, match=r"jax array.*terminated mask"):
        gravity(position=pos, mu=MU, terminated=mask)
    assert not np.asarray(mask).any()

    # a Mutable plane (the pure launcher)
    bump = LoadedPure(FIX / "bump.ptx")
    counter = jax.device_put(np.linspace(0.0, 1.0, n), _gpu(jax))
    before = np.asarray(counter).copy()
    with pytest.raises(BufferRefused, match=r"jax array.*Mutable 'counter'"):
        bump(counter=counter, rate=2.5)
    assert np.array_equal(np.asarray(counter), before)

    # a written host buffer (the host-plugin launch path)
    host = jax.device_put(np.arange(8.0), _cpu(jax))
    assert not np.asarray(host).flags.writeable  # jax's CPU view is read-only
    assert host_ptr(host) == int(np.asarray(host).ctypes.data)  # reading is fine
    with pytest.raises(BufferRefused, match=r"jax array.*written host buffer"):
        host_ptr(host, writable=True)
    frozen = np.arange(8.0)
    frozen.flags.writeable = False
    with pytest.raises(BufferRefused, match="read-only"):
        host_ptr(frozen, writable=True)
    assert np.array_equal(np.asarray(host), np.arange(8.0))

    # a plugin that declares `terminated` read-only takes the mask
    meta = json.loads((FIX / "gravity.json").read_text())
    meta["terminated_readonly"] = True
    shutil.copy(FIX / "gravity.ptx", tmp_path / "gravity.ptx")
    (tmp_path / "gravity.json").write_text(json.dumps(meta))
    readonly = LoadedVector(tmp_path / "gravity.ptx")
    y = readonly(position=pos, mu=MU, terminated=mask)
    assert _max_rel(cp.asnumpy(y), _gravity_ref(P, MU)) < 1e-12


@register_row("JX-OUT-CUDA-ALIAS")
def test_jax_cuda_result_is_eagles_result_buffer(monkeypatch):
    """The jax result IS eagle's fresh result buffer: pointer-equal, complete
    at return, owned only by the jax array (eagle's array, allocated through
    aether, dies with it), never written again, and a second call returns a
    distinct buffer."""
    certified_framework("jax")
    certified_framework("cupy")
    import cupy as cp

    from eagle import interop

    jax = _jax()
    plugin = LoadedVector(FIX / "gravity.ptx")
    P = _make_positions(64, seed=4)
    ref = _gravity_ref(P, MU)
    x = jax.device_put(P, _gpu(jax))
    plugin(position=x, mu=MU)  # warm up: load the module, fill the caches

    seen = {}
    original = interop._JaxAdapter.from_array

    def spy(self, a):
        seen["ptr"] = a.ptr
        return original(self, a)

    monkeypatch.setattr(interop._JaxAdapter, "from_array", spy)
    cp.cuda.runtime.deviceSynchronize()

    y = plugin(position=x, mu=MU)
    assert y.unsafe_buffer_pointer() == seen["ptr"]  # the same buffer
    gc.collect()
    assert _max_rel(np.asarray(y), ref) < 1e-12  # complete at return, and the jax array alone keeps eagle's buffer alive
    snapshot = np.asarray(y).copy()

    y2 = plugin(position=x, mu=MU)
    assert y2.unsafe_buffer_pointer() != y.unsafe_buffer_pointer()
    assert np.array_equal(np.asarray(y), snapshot)  # never written again

    del y, y2
    cp.cuda.runtime.deviceSynchronize()


@register_row("JX-IN-CPU-COPY")
def test_jax_cpu_input_comes_back_as_a_cpu_jax_array_via_a_copy():
    """A CPU jax array comes back as a CPU jax array through an upload and a
    download: correct values, and the output does not alias the input. Both
    sides are immutable, so there are no mutation legs."""
    certified_framework("jax")
    certified_framework("cupy")
    jax = _jax()
    plugin = LoadedVector(FIX / "gravity.ptx")
    P = _make_positions(64, seed=2)
    x = jax.device_put(P, _cpu(jax))

    out = plugin(position=x, mu=MU)
    assert isinstance(out, jax.Array)
    assert out.devices() == {_cpu(jax)}
    assert _max_rel(np.asarray(out), _gravity_ref(P, MU)) < 1e-12
    assert out.unsafe_buffer_pointer() != x.unsafe_buffer_pointer()
    assert np.array_equal(np.asarray(x), P)


# --------------------------------------------------------------------------- #
# STREAM-JAX-PRODUCER-ORDER
# --------------------------------------------------------------------------- #
_spin_lib: dict = {}


def _spin_op(tmp_path_factory):
    """``spin(x, cycles)``: a jax operation that keeps the GPU busy for
    ``cycles`` clock cycles and returns ``x + 1`` (see ``fixtures/jax_spin.cu``).
    Built once with nvcc for the device in use, into the pytest basetemp."""
    if not _spin_lib:
        import cupy as cp

        jax = _jax()
        nvcc = shutil.which("nvcc")
        if nvcc is None:
            pytest.fail("nvcc is needed to build the jax.ffi spin kernel and was not found")
        so = tmp_path_factory.mktemp("jaxspin") / "libjax_spin.so"
        arch = cp.cuda.Device().compute_capability
        build = subprocess.run(
            [nvcc, "-std=c++17", "-O1", f"-arch=sm_{arch}", "--compiler-options=-fPIC",
             "--shared", "-isystem", jax.ffi.include_dir(), str(FIX / "jax_spin.cu"),
             "-o", str(so)],
            capture_output=True, text=True,
        )
        assert build.returncode == 0, build.stderr
        lib = ctypes.cdll.LoadLibrary(str(so))
        jax.ffi.register_ffi_target(
            "eagle_test_jax_spin", jax.ffi.pycapsule(lib.SpinHandler), platform="CUDA"
        )
        _spin_lib["lib"] = lib  # keep the library loaded

    jax = _jax()

    def spin(x, cycles):
        call = jax.ffi.ffi_call(
            "eagle_test_jax_spin",
            jax.ShapeDtypeStruct(x.shape, x.dtype),
            vmap_method="broadcast_all",
        )
        return call(x, cycles=np.int64(cycles))

    return spin


def _spin_cycles(spin, x, seconds: float) -> int:
    """Cycle count for a spin of about ``seconds``, from a timed probe (no
    clock rate is hard-coded)."""
    spin(x, 1000).block_until_ready()  # compile and warm up
    probe = 100_000_000
    t0 = time.perf_counter()
    spin(x, probe).block_until_ready()
    dt = max(time.perf_counter() - t0, 1e-6)
    return max(int(probe * seconds / dt), probe)


@register_row("STREAM-JAX-PRODUCER-ORDER")
def test_jax_stream_producer_order_import_waits_on_the_device(tmp_path_factory):
    """With a jax operation still running on the GPU, eagle's import passes
    its stream to ``__dlpack__``: the import returns at once, the launch is
    enqueued behind the operation without a host wait, and it reads the
    finished values.

    The operation is in flight for real: an export with no stream argument
    blocks the host for most of the operation's wall time. The operation
    writes -1 first and ``x + 1`` last, so an early read is visible."""
    certified_framework("jax")
    certified_framework("cupy")
    import cupy as cp

    jax = _jax()
    spin = _spin_op(tmp_path_factory)
    plugin = LoadedVector(FIX / "gravity.ptx")
    P = _make_positions(64, seed=5)
    x = jax.device_put(P - 1.0, _gpu(jax))  # the operation returns P
    cycles = _spin_cycles(spin, x, 0.4)

    # warm up eagle's side so only the import and enqueue are timed below
    plugin(position=cp.asarray(P), mu=MU)
    t0 = time.perf_counter()
    spin(x, cycles).block_until_ready()
    wall = time.perf_counter() - t0  # the operation's wall time

    # the operation really is in flight: an export with no stream waits for it
    busy = spin(x, cycles)
    t0 = time.perf_counter()
    capsule = busy.__dlpack__()
    blocked = time.perf_counter() - t0
    del capsule
    busy.block_until_ready()
    assert blocked >= 0.5 * wall, (blocked, wall)

    # eagle's import under its own stream: no host wait, correct values
    busy = spin(x, cycles)
    t0 = time.perf_counter()
    cu = to_cupy(busy)  # passes cupy's current stream to __dlpack__
    imported = time.perf_counter() - t0
    y = plugin(position=cu, mu=MU)  # cupy in, cupy out: no synchronization
    enqueued = time.perf_counter() - t0
    assert imported < 0.1 * wall, (imported, wall)
    assert enqueued < 0.1 * wall, (enqueued, wall)
    assert _max_rel(cp.asnumpy(y), _gravity_ref(P, MU)) < 1e-12
