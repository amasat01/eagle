# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The JAX FFI certification rows (eagle-owned, declared in
``raptor.conformance.interop.ROWS``): ``JX-FFI-CPU-ALIAS``, ``JX-OUT-XLA-OWNED``,
``JX-JIT-INVARIANCE`` and ``STREAM-JAX-FFI-ORDER``.

These certify :func:`eagle.frameworks.jax.function`: a hawk kernel run as an XLA
custom call on XLA's own buffers and stream. A row with two legs has one test per
leg (``register_row(..., leg=...)``); ``assert_rows_complete("eagle", ran=True)``
needs each of them to have run and passed. Every row carries a falsifier in the
test itself, a known-bad variant the same instrument must catch:

* ``JX-FFI-CPU-ALIAS``: the addresses the kernel receives are compared with
  ``unsafe_buffer_pointer()``; a copied array's address must NOT match.
* ``JX-OUT-XLA-OWNED``: the same kernel registered with the handler's zero flag
  off reads stale bytes on the dirtied allocator, so the allocator really was dirty.
* ``JX-JIT-INVARIANCE``: one folded sample is perturbed and the comparison must
  see it.
* ``STREAM-JAX-FFI-ORDER``: a spin handler that waits for the stream on the host
  (``fixtures/jax_spin.cu``, ``SpinSyncHandler``) must read as a blocked dispatch.

Hygiene is the same as the other matrix modules: no skip tokens, no availability
marker, only the ``interop_matrix`` tag, and every row opens with
``certified_framework``. A leg with no device of its platform fails. JAX and hawk
are imported inside the tests.
"""

from __future__ import annotations

import ctypes
import functools
import os
import pathlib
import shutil
import subprocess
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest
from raptor.conformance.interop import certified_framework, register_row

pytestmark = pytest.mark.interop_matrix

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"


@functools.cache
def _kernels():
    from hawk import Mutable, Param, Scalar, Terminated, Vector, kernel
    from hawk.math import dot, sqrt

    @kernel
    def drag(v: Vector[3], cd: Param, terminated: Terminated, a: Mutable[Vector[3]]):
        a = -cd * sqrt(dot(v, v)) * v

    @kernel(steps=1)
    def step_and_finish(k: Scalar, nstop: Param, terminated: Terminated,
                        k_next: Mutable[Scalar]):
        k1 = k + 1.0
        k_next = k1
        terminated = k1 >= nstop

    return {"drag": drag, "step_and_finish": step_and_finish}


_OPS = {}


def _ops(tmp_path_factory):
    """The two kernels as JAX functions, built once per session (and float64 on)."""
    if not _OPS:
        from eagle.frameworks import jax as eagle_jax
        from test_interop_jax import _jax

        _jax()
        root = tmp_path_factory.mktemp("jax_rows")
        _OPS.update({name: eagle_jax.function(k, cache_dir=root / name)
                     for name, k in _kernels().items()})
    return _OPS


def _device(jax, platform):
    """The platform's first device. A missing one raises: the leg fails, it never skips."""
    return jax.devices(platform)[0]


def _digest(K, platform):
    return K.level.slots[("digest", platform, "float64")]


def _ffi(jax, platform, digest, results, *args):
    from eagle.frameworks import jax as eagle_jax

    eagle_jax._xla(platform)
    return jax.ffi.ffi_call(eagle_jax.TARGET, results)(*args, digest=digest)


def _register_twin(K, platform, digest, *, zero=True, entry=None, device=None):
    """Register ``K``'s float64 recipe again under ``digest``: with the handler's
    zero flags off, and/or with another launch entry in place of the kernel's."""
    from eagle.frameworks import jax as eagle_jax

    level = K.level
    arch = _digest(K, platform).rsplit("cuda-", 1)[1] if platform == "cuda" else ""
    unit = level.unit(K._dir, "host" if platform == "cpu" else "cuda", "float64", arch)
    fields = dict(eagle_jax._recipe_fields(level, level.artifact(unit), "float64"))
    if not zero:
        fields["zero"] = [0] * len(fields["zero"])
    if platform == "cpu":
        keep = level.slots[("keep", "cpu", "float64")]
        eagle_jax._install(digest, fields, platform="cpu", entry=entry or keep.host_entry)
    else:
        seam = eagle_jax._seam()
        keep = level.slots[("keep", "cuda", "float64")]
        eagle_jax._install(digest, fields, platform="cuda", entry=entry or seam["run"],
                           ordinal=device.local_hardware_id, function=keep.device_function,
                           seam=seam)
    return fields


# --------------------------------------------------------------------------- #
# JX-FFI-CPU-ALIAS
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    """``fixtures/xla_pack_probe.cpp``: a launch entry that records the argument
    structures (and so the buffer addresses) it is handed instead of running a kernel."""
    cxx = shutil.which(os.environ.get("CXX", "") or "c++") or shutil.which("g++")
    if cxx is None:
        pytest.fail("a C++ compiler is needed to build fixtures/xla_pack_probe.cpp")
    so = tmp_path_factory.mktemp("xla_probe") / "libxla_pack_probe.so"
    build = subprocess.run([cxx, "-std=c++17", "-O1", "-fPIC", "-shared",
                            str(FIX / "xla_pack_probe.cpp"), "-o", str(so)],
                           capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    lib = ctypes.CDLL(str(so))
    lib.probe_read.restype = ctypes.c_int32
    return lib


def _seen_addresses(lib):
    """The buffer address of each slot of the last recorded call (field 0 of each box)."""
    buf = (ctypes.c_ubyte * 8192)()
    got = lib.probe_read(buf, 8192)
    assert got > 0
    nslots = int.from_bytes(bytes(buf[:4]), "little", signed=True)
    return [int.from_bytes(bytes(buf[4 + 40 * s: 12 + 40 * s]), "little") for s in range(nslots)]


@register_row("JX-FFI-CPU-ALIAS", leg="cpu")
def test_cpu_bridge_reads_xla_host_buffers_in_place(tmp_path_factory, probe):
    """On the CPU platform the kernel's entry is handed XLA's own buffers: each
    input's address is the jax array's ``unsafe_buffer_pointer()`` (no staging copy)
    and the result buffer is a fresh one that XLA owns (not an input's address), so
    the array that comes back is that same memory. A copied array's address does
    not match, so the comparison can fail."""
    certified_framework("jax")
    import jax
    import jax.numpy as jnp

    K = _ops(tmp_path_factory)["drag"]
    n = 5
    rng = np.random.default_rng(0)
    with jax.default_device(_device(jax, "cpu")):
        v = jnp.asarray(rng.normal(size=(3, n)))
        cd = jnp.asarray([0.3])
        done = jnp.asarray(np.arange(n) % 3 == 1)
        K(v, cd, done)  # builds the kernel and registers its recipe
        probe.probe_set_slots(len(K.level.sig.arg_spec))
        _register_twin(K, "cpu", "eagle-test/rows/cpu-alias",
                       entry=ctypes.cast(probe.probe_host, ctypes.c_void_p).value)
        out = _ffi(jax, "cpu", "eagle-test/rows/cpu-alias",
                   [jax.ShapeDtypeStruct((3, n), jnp.float64)], v, cd, done)[0]
        jax.block_until_ready(out)
        copy = jnp.array(v, copy=True)
    seen = dict(zip((name for _, name in K.level.sig.arg_spec), _seen_addresses(probe)))
    inputs = {"v": v, "cd": cd, "terminated": done}
    for name, array in inputs.items():
        assert seen[name] == array.unsafe_buffer_pointer(), name
    assert seen["a"] == out.unsafe_buffer_pointer()
    assert seen["a"] not in {a.unsafe_buffer_pointer() for a in inputs.values()}
    assert seen["v"] != copy.unsafe_buffer_pointer()  # the falsifier: a copy is told apart
    # the product path: the real result is a fresh buffer too
    res = K(v, cd, done)
    assert res.unsafe_buffer_pointer() not in {a.unsafe_buffer_pointer() for a in inputs.values()}


# --------------------------------------------------------------------------- #
# JX-OUT-XLA-OWNED
# --------------------------------------------------------------------------- #
def _dirty(jax, jnp, shape):
    """Fill and free several arrays of ``shape``, so the allocator's next blocks
    of that size hold old data."""
    for rounds in range(3):
        junk = [jnp.full(shape, 7.0 + i + rounds) for i in range(4)]
        jax.block_until_ready(junk)
        del junk


def _outputs_are_xla_owned(tmp_path_factory, platform):
    certified_framework("jax")
    import jax
    import jax.numpy as jnp

    ops = _ops(tmp_path_factory)
    K = ops["drag"]
    n = 1 << 18
    device = _device(jax, platform)
    with jax.default_device(device):
        x = jnp.ones((3, n))
        cd = jnp.asarray([0.5])
        done = jnp.asarray(np.arange(n) % 2 == 1)
        # terminated samples: the kernel writes nothing, the bridge returns zeros
        _dirty(jax, jnp, (3, n))
        a = np.asarray(jax.jit(lambda x, c, t: K(x, c, t))(x, cd, done))
        assert np.all(a[:, 1::2] == 0.0)
        np.testing.assert_allclose(a[:, 0::2], -0.5 * np.sqrt(3.0))
        # the falsifier: the same kernel with the zero flag off reads what was there
        _register_twin(K, platform, f"eagle-test/rows/{platform}-zero-off", zero=False,
                       device=device)
        _dirty(jax, jnp, (3, n))
        stale = np.asarray(_ffi(jax, platform, f"eagle-test/rows/{platform}-zero-off",
                                [jax.ShapeDtypeStruct((3, n), jnp.float64)], x, cd, done)[0])
        assert np.any(stale[:, 1::2] != 0.0), "the allocator was not dirty: the check proves nothing"

        # a donated mask comes back at the same address; one not donated is left intact
        S = ops["step_and_finish"]
        k = jnp.arange(6.0)
        mask = jnp.asarray([False, True, False, False, False, False])
        before = mask.unsafe_buffer_pointer()
        step = jax.jit(lambda k, t: S(k, 3.0, t), donate_argnums=1)
        k_next, finished = step(k, mask)
        assert finished.unsafe_buffer_pointer() == before
        np.testing.assert_array_equal(np.asarray(finished), [False, True, True, True, True, True])
        np.testing.assert_array_equal(np.asarray(k_next), [1.0, 0.0, 3.0, 4.0, 5.0, 6.0])
        kept = jnp.asarray([False, True, False, False, False, False])
        jax.jit(lambda k, t: S(k, 3.0, t))(k, kept)
        np.testing.assert_array_equal(np.asarray(kept), [False, True, False, False, False, False])


@register_row("JX-OUT-XLA-OWNED", leg="cpu")
def test_outputs_are_xla_owned_and_zero_where_unwritten_cpu(tmp_path_factory):
    """CPU leg: see ``_outputs_are_xla_owned``."""
    _outputs_are_xla_owned(tmp_path_factory, "cpu")


@register_row("JX-OUT-XLA-OWNED", leg="cuda")
def test_outputs_are_xla_owned_and_zero_where_unwritten_cuda(tmp_path_factory):
    """CUDA leg: outputs are XLA-allocated and exactly zero for terminated samples
    on a dirtied allocator (fill and free before the call, never fresh pages); the
    zero-flag-off twin shows the dirt; a donated mask is returned pointer-equal and
    a mask that was not donated is left intact."""
    _outputs_are_xla_owned(tmp_path_factory, "cuda")


# --------------------------------------------------------------------------- #
# JX-JIT-INVARIANCE
# --------------------------------------------------------------------------- #
#: The ULP contract of the default host profile (fused multiply-add, vector math):
#: a result may differ from another structure's by a few units in the last place.
ULP_BOUND = 8


def _ulps(a, b):
    """Units in the last place between two float64 arrays (an ordered integer map)."""
    def key(x):
        i = np.ascontiguousarray(x, np.float64).view(np.int64)
        return np.where(i < 0, np.int64(-(2**63)) - i, i)

    return np.abs(key(a) - key(b))


def _agree(got, want, bitwise):
    got, want = np.asarray(got), np.asarray(want)
    if bitwise:
        return got.shape == want.shape and np.array_equal(got.view(np.uint64), want.view(np.uint64))
    return got.shape == want.shape and int(_ulps(got, want).max()) <= ULP_BOUND


def _invariance(jax, jnp, K, device, bitwise):
    """Eager, jit, a vmap-folded batch and a scan of steps give one answer."""
    n, batch, steps = 37, 3, 4          # n is not a multiple of any vector width
    rng = np.random.default_rng(3)
    with jax.default_device(device):
        xs = jnp.asarray(rng.uniform(0.5, 0.6, size=(batch, 3, n)))
        cd = jnp.asarray([1.0])
        done = jnp.asarray(np.arange(n) % 4 == 1)
        eager = [K(xs[i], cd, done) for i in range(batch)]
        jitted = jax.jit(K)
        assert all(_agree(jitted(xs[i], cd, done), eager[i], bitwise) for i in range(batch))

        folded = jax.vmap(lambda y: K(y, cd, done))(xs)
        assert all(_agree(folded[i], eager[i], bitwise) for i in range(batch))

        # steps: the kernel applied to its own result, as a Python loop, under jit, and as a scan
        loop = xs[0]
        for _ in range(steps):
            loop = K(loop, cd, done)
        scanned = jax.lax.scan(lambda c, _: (K(c, cd, done), None), xs[0], None, length=steps)[0]
        looped_jit = jax.jit(lambda c: jax.lax.fori_loop(0, steps, lambda _, c: K(c, cd, done), c))(xs[0])
        assert _agree(scanned, loop, bitwise)
        assert _agree(looped_jit, loop, bitwise)

        # the falsifier: one folded sample off by 64 ulp must not agree
        bumped = np.array(xs)
        bumped[1, 2, 7] *= 1.0 + 64 * np.finfo(np.float64).eps
        folded_bad = jax.vmap(lambda y: K(y, cd, done))(jnp.asarray(bumped))
        assert not _agree(folded_bad[1], eager[1], bitwise)
        assert _agree(folded_bad[0], eager[0], bitwise)


@register_row("JX-JIT-INVARIANCE", leg="cpu")
def test_jit_invariance_on_the_cpu(tmp_path, monkeypatch):
    """CPU leg: bitwise on the exact host profile; within ``ULP_BOUND`` ulp on the
    default profile when that one is ``native-vector-math`` (it is the bitwise
    comparison again wherever the default is already exact)."""
    certified_framework("jax")
    import jax
    import jax.numpy as jnp
    from eagle.frameworks import jax as eagle_jax
    from hawk.compile.toolchain import EXACT_HOST_PROFILES, host_profile
    from test_interop_jax import _jax

    _jax()
    drag = _kernels()["drag"]
    device = _device(jax, "cpu")
    for name in ("native", None):
        if name:
            monkeypatch.setenv("HAWK_HOST_PROFILE", name)
        else:
            monkeypatch.delenv("HAWK_HOST_PROFILE", raising=False)
        K = eagle_jax.function(drag, cache_dir=tmp_path / (name or "default"))
        _invariance(jax, jnp, K, device, bitwise=host_profile() in EXACT_HOST_PROFILES)


@register_row("JX-JIT-INVARIANCE", leg="cuda")
def test_jit_invariance_on_cuda(tmp_path_factory):
    """CUDA leg: bitwise (one device kernel, whatever XLA wraps around it)."""
    certified_framework("jax")
    import jax
    import jax.numpy as jnp

    _invariance(jax, jnp, _ops(tmp_path_factory)["drag"], _device(jax, "cuda"), bitwise=True)


# --------------------------------------------------------------------------- #
# STREAM-JAX-FFI-ORDER
# --------------------------------------------------------------------------- #
_sync = {}


def _spin_sync_op():
    """``spin`` with a host wait after its launch: built into the library
    ``test_interop_jax`` already built (``SpinSyncHandler``)."""
    import jax
    from test_interop_jax import _spin_lib

    if "registered" not in _sync:
        jax.ffi.register_ffi_target(
            "eagle_test_jax_spin_sync", jax.ffi.pycapsule(_spin_lib["lib"].SpinSyncHandler),
            platform="CUDA")
        _sync["registered"] = True

    def spin(x, cycles):
        return jax.ffi.ffi_call("eagle_test_jax_spin_sync",
                                jax.ShapeDtypeStruct(x.shape, x.dtype),
                                vmap_method="broadcast_all")(x, cycles=np.int64(cycles))

    return spin


def _dispatch_ratio(run, x, wall):
    """The time ``run(x)`` takes to RETURN, over the operation's wall time. Values
    are read after the clock stops."""
    t0 = time.perf_counter()
    result = run(x)
    returned = time.perf_counter() - t0
    return returned / wall, np.asarray(result)


@register_row("STREAM-JAX-FFI-ORDER", leg="cuda")
def test_bridge_after_an_xla_op_rides_xla_stream(tmp_path_factory):
    """In one ``jit``, a long-running XLA operation (a spin that writes -1 first and
    ``x + 1`` last) is followed by the bridge. The call returns in under 5 % of the
    operation's wall time (the host never waits) and the bridge read the finished
    values. The same chain with a spin that waits on the host after its launch
    returns in at least half of it: the instrument reads a blocked dispatch as one."""
    certified_framework("jax")
    certified_framework("cupy")
    import jax
    import jax.numpy as jnp
    from test_interop_jax import _spin_cycles, _spin_op

    K = _ops(tmp_path_factory)["drag"]
    spin = _spin_op(tmp_path_factory)
    spin_sync = _spin_sync_op()
    n = 64
    device = _device(jax, "cuda")
    with jax.default_device(device):
        base = jnp.asarray(np.random.default_rng(5).uniform(0.5, 1.0, size=(3, n)))
        cd = jnp.asarray([0.5])
        done = jnp.asarray(np.arange(n) % 4 == 1)
        cycles = _spin_cycles(spin, base, 0.4)
        t0 = time.perf_counter()
        spin(base, cycles).block_until_ready()
        wall = time.perf_counter() - t0

        y = np.asarray(base) + 1.0
        want = np.where(np.asarray(done)[None, :], 0.0, -0.5 * np.linalg.norm(y, axis=0) * y)

        def chain(op):
            run = jax.jit(lambda x: K(op(x, cycles), cd, done))
            run(base).block_until_ready()  # compile and warm up
            return [_dispatch_ratio(run, base, wall) for _ in range(3)]

        ordered = chain(spin)
        blocked = chain(spin_sync)
    for _, values in ordered + blocked:
        np.testing.assert_allclose(values, want, rtol=1e-12)
    assert max(r for r, _ in ordered) < 0.05, [r for r, _ in ordered]
    assert min(r for r, _ in blocked) >= 0.5, [r for r, _ in blocked]
