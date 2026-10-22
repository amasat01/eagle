# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.frameworks.jax``: a hawk kernel as a JAX function.

Three subjects, each a per-sample kernel compiled once per (level, platform,
mode) for the whole module: ``drag`` (a vector plane, a ``Param``, a
``Terminated`` mask, a vector output), ``vec_sum`` (two vector planes, no
``Param``) and ``step_and_finish`` (a kernel that finishes its own mask, which
comes back as an extra output).

The CORE rows run eighteen transformations (forward, jit, grad, jvp, linearize,
jacfwd, jacrev, hessian, nested and third-order derivatives, vmap over planes and
over Params) against the same law written in ``jax.numpy``, for each subject, in
float64 and float32, on the CPU platform and on a CUDA device (``gpu`` marker).
The rest pin the handler: digest-only lowering, named refusals (unknown kernel,
wrong dtype, wrong device, mixed float dtypes), zero-filled outputs on a dirtied
allocator, donation of the finished mask, the argument PODs against eagle's own
packer (through a test-only recording entry built from ``fixtures/``), the
library's exports, the vendored FFI API version, the persistent compilation
cache across processes, ``jax.export`` + :func:`~eagle.frameworks.jax.preload`,
and thread safety (8 threads x 50 calls, and 8 threads tracing a fresh kernel
at a barrier).
"""

from __future__ import annotations

import json
import ctypes
import os
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap
import threading

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

jax = pytest.importorskip("jax",
    reason="jax is not installed here (jax's jaxlib ships free-threaded wheels for 3.14t only (none for 3.13t))")
hawk = pytest.importorskip("hawk")

import jax.numpy as jnp  # noqa: E402
from hawk import Mutable, Param, Scalar, Terminated, Vector  # noqa: E402
from hawk.math import dot, sqrt  # noqa: E402

from eagle.frameworks import jax as eagle_jax  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FIX = HERE / "fixtures"

PLATFORMS = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
MODES = ["float64", "float32"]


@pytest.fixture(autouse=True)
def _x64():
    """float64 arrays exist only with x64 on; restored after each test."""
    before = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", before)


@hawk.kernel
def drag(v: Vector[3], cd: Param, terminated: Terminated, a: Mutable[Vector[3]]):
    a = -cd * sqrt(dot(v, v)) * v


if hasattr(hawk, "all_samples"):
    @hawk.kernel
    def cell_min(x: Scalar, t: Mutable[Scalar]):
        t = hawk.all_samples(x).min()  # noqa: F841

    @hawk.kernel
    def cell_sum(x: Scalar, t: Mutable[Scalar]):
        t = hawk.all_samples(x).sum()  # noqa: F841


@hawk.kernel
def vec_sum(v: Vector[3], w: Vector[3], a: Mutable[Vector[3]]):
    a = v + w


@hawk.kernel(steps=1)
def step_and_finish(k: Scalar, nstop: Param, terminated: Terminated,
                    k_next: Mutable[Scalar]):
    k1 = k + 1.0
    k_next = k1
    terminated = k1 >= nstop


def _device(platform):
    try:
        devices = jax.devices(platform)
    except RuntimeError:
        devices = []
    if not devices:
        pytest.skip(f"JAX has no {platform} device in this process")
    return devices[0]


# -- the three subjects: inputs, the bridge call and the jnp law ----------------

def _mask(n):
    return jnp.asarray(np.arange(n) % 3 == 1)


class _Case:
    def __init__(self, kernel, make, call, ref):
        self.kernel, self.make, self.call, self.ref = kernel, make, call, ref


def _drag_make(n, dt, rng):
    return (jnp.asarray(rng.normal(size=(3, n)), dt), jnp.asarray([0.3], dt),
            {"t": _mask(n)})


def _drag_ref(x, p, e):
    return jnp.where(e["t"][None, :], 0.0, -p[0] * jnp.sqrt(jnp.sum(x * x, axis=0)) * x)


def _sum_make(n, dt, rng):
    return (jnp.asarray(rng.normal(size=(3, n)), dt),
            jnp.asarray(rng.normal(size=(3, n)), dt), {})


def _step_make(n, dt, rng):
    return (jnp.asarray(rng.uniform(0.0, 4.0, size=n), dt), jnp.asarray([2.5], dt),
            {"t": _mask(n)})


CASES = {
    "drag": _Case(drag, _drag_make, lambda K, x, p, e: K(x, p, e["t"]), _drag_ref),
    "vec_sum": _Case(vec_sum, _sum_make, lambda K, x, p, e: K(x, p),
                     lambda x, p, e: x + p),
    "step_and_finish": _Case(step_and_finish, _step_make,
                             lambda K, x, p, e: K(x, p, e["t"])[0],
                             lambda x, p, e: jnp.where(e["t"], 0.0, x + 1.0)),
}


@pytest.fixture(scope="module")
def ops(tmp_path_factory):
    root = tmp_path_factory.mktemp("jax_fn")
    return {name: eagle_jax.function(case.kernel, cache_dir=root / name)
            for name, case in CASES.items()}


def _checks(f, ref, x, p, vb, pb):
    loss = lambda x, p: jnp.sum(f(x, p) ** 2)        # noqa: E731
    rloss = lambda x, p: jnp.sum(ref(x, p) ** 2)     # noqa: E731
    one = jnp.ones_like(p)

    def g3(L):
        return lambda x, p: jnp.sum(jax.grad(L, 1)(x, p))

    return {
        "forward": (f(x, p), ref(x, p)),
        "jit": (jax.jit(f)(x, p), ref(x, p)),
        "grad_plane": (jax.grad(loss, 0)(x, p), jax.grad(rloss, 0)(x, p)),
        "grad_param": (jax.grad(loss, 1)(x, p), jax.grad(rloss, 1)(x, p)),
        "jit_grad": (jax.jit(jax.grad(loss, (0, 1)))(x, p)[1], jax.grad(rloss, (0, 1))(x, p)[1]),
        "jvp_param": (jax.jvp(lambda c: f(x, c), (p,), (one,))[1],
                      jax.jvp(lambda c: ref(x, c), (p,), (one,))[1]),
        "linearize": (jax.linearize(lambda y: f(y, p), x)[1](x),
                      jax.linearize(lambda y: ref(y, p), x)[1](x)),
        "jacfwd": (jax.jacfwd(lambda y: f(y, p))(x), jax.jacfwd(lambda y: ref(y, p))(x)),
        "jacrev": (jax.jacrev(lambda y: f(y, p))(x), jax.jacrev(lambda y: ref(y, p))(x)),
        "hessian_plane": (jax.hessian(loss, 0)(x, p), jax.hessian(rloss, 0)(x, p)),
        "hessian_param": (jax.hessian(loss, 1)(x, p), jax.hessian(rloss, 1)(x, p)),
        "jacrev_jacrev": (jax.jacrev(jax.jacrev(loss, 0), 0)(x, p),
                          jax.jacrev(jax.jacrev(rloss, 0), 0)(x, p)),
        "jacfwd_jacfwd": (jax.jacfwd(jax.jacfwd(loss, 1), 1)(x, p),
                          jax.jacfwd(jax.jacfwd(rloss, 1), 1)(x, p)),
        "grad3_param": (jax.grad(g3(g3(loss)), 1)(x, p), jax.grad(g3(g3(rloss)), 1)(x, p)),
        "vmap_plane": (jax.vmap(lambda y: f(y, p))(vb), jax.vmap(lambda y: ref(y, p))(vb)),
        "vmap_param": (jax.vmap(lambda c: f(x, c))(pb), jax.vmap(lambda c: ref(x, c))(pb)),
        "vmap_grad": (jax.vmap(jax.grad(loss, 0), in_axes=(0, None))(vb, p),
                      jax.vmap(jax.grad(rloss, 0), in_axes=(0, None))(vb, p)),
        "jit_vmap_jacfwd": (jax.jit(jax.vmap(jax.jacfwd(lambda y: f(y, p))))(vb),
                            jax.vmap(jax.jacfwd(lambda y: ref(y, p)))(vb)),
    }


CHECKS = ["forward", "jit", "grad_plane", "grad_param", "jit_grad", "jvp_param",
          "linearize", "jacfwd", "jacrev", "hessian_plane", "hessian_param",
          "jacrev_jacrev", "jacfwd_jacfwd", "grad3_param", "vmap_plane", "vmap_param",
          "vmap_grad", "jit_vmap_jacfwd"]

_RESULTS = {}


def _close(got, want, mode):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    assert got.shape == want.shape
    tol = 1e-10 if mode == "float64" else 2e-4
    scale = 1.0 + float(np.max(np.abs(want))) if want.size else 1.0
    err = float(np.max(np.abs(got - want))) if want.size else 0.0
    assert err <= tol * scale, f"max abs error {err:.3e} > {tol:.0e} x {scale:.3g}"


@pytest.mark.parametrize("check", CHECKS)
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("case", list(CASES))
@pytest.mark.parametrize("platform", PLATFORMS)
def test_core_transformations_match_jnp(ops, platform, case, mode, check):
    device = _device(platform)
    key = (platform, case, mode)
    if key not in _RESULTS:
        c = CASES[case]
        K = ops[case]
        dt = np.float64 if mode == "float64" else np.float32
        rng = np.random.default_rng(0)
        with jax.default_device(device):
            x, p, e = c.make(6, dt, rng)
            vb = jnp.stack([x * 0.5, x, x * 1.5])
            pb = jnp.stack([p * 0.5, p * 2.0])
            f = lambda x, p: c.call(K, x, p, e)          # noqa: E731
            ref = lambda x, p: c.ref(x, p, e)            # noqa: E731
            _RESULTS[key] = {name: (np.asarray(a), np.asarray(b))
                             for name, (a, b) in _checks(f, ref, x, p, vb, pb).items()}
    got, want = _RESULTS[key][check]
    _close(got, want, mode)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_check_grads_orders_one_and_two(ops, platform):
    from jax.test_util import check_grads

    with jax.default_device(_device(platform)):
        x, p, e = _drag_make(5, np.float64, np.random.default_rng(1))
        f = lambda x, p: ops["drag"](x, p, e["t"])         # noqa: E731
        check_grads(f, (x, p), order=2, modes=("fwd", "rev"))


def test_import_eagle_stays_jax_free():
    code = ("import sys, eagle, eagle.frameworks; "
            "bad = sorted(m for m in ('jax', 'hawk') if m in sys.modules); "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_signature_is_read_off_the_kernel(ops):
    K = ops["drag"]
    assert K.inputs == ("v", "cd", "terminated")
    assert K.outputs == ("a",)
    assert set(K.wrt) == {"v", "cd"}
    assert not K.finishes_terminated
    S = ops["step_and_finish"]
    assert S.outputs == ("k_next", "terminated") and S.finishes_terminated


# -- refusals ----------------------------------------------------------------------

def test_mixed_float_dtypes_are_refused_by_name(ops):
    with jax.default_device(_device("cpu")):
        x, _, e = _drag_make(4, np.float32, np.random.default_rng(0))
        with pytest.raises(TypeError, match="never a mix"):
            ops["drag"](x, jnp.asarray([0.3], jnp.float64), e["t"])
        with pytest.raises(TypeError, match="float32 or float64"):
            ops["vec_sum"](x.astype(jnp.int32), x.astype(jnp.int32))


def _digest(K, platform, mode):
    level = K.level
    return level.slots[("digest", platform, mode)]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_lowering_carries_the_digest_and_no_address(ops, platform):
    with jax.default_device(_device(platform)):
        x, p, e = _drag_make(4, np.float64, np.random.default_rng(0))
        text = jax.jit(lambda x, p: ops["drag"](x, p, e["t"])).lower(x, p).as_text()
    digest = _digest(ops["drag"], platform, "float64")
    assert digest in text
    keep = ops["drag"].level.slots[("keep", platform, "float64")]
    address = getattr(keep, "host_entry", None) or keep.device_function
    assert str(address) not in text and hex(address) not in text


def _ffi(platform, digest, results, *args, aliases=None):
    target = eagle_jax.TARGET
    eagle_jax._xla(platform)
    return jax.ffi.ffi_call(target, results, input_output_aliases=aliases)(*args, digest=digest)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_unknown_kernel_is_refused_by_name(platform):
    with jax.default_device(_device(platform)):
        x = jnp.ones(4)
        with pytest.raises(Exception, match="is not loaded in this process"):
            jax.block_until_ready(_ffi(platform, "eagle-test/nope", jax.ShapeDtypeStruct((4,), x.dtype), x))


@pytest.mark.parametrize("platform", PLATFORMS)
def test_handler_refuses_a_buffer_of_the_wrong_dtype(ops, platform):
    with jax.default_device(_device(platform)):
        x, p, e = _drag_make(4, np.float64, np.random.default_rng(0))
        ops["drag"](x, p, e["t"])
        digest = _digest(ops["drag"], platform, "float64")
        out = jax.ShapeDtypeStruct((3, 4), jnp.float32)
        with pytest.raises(Exception, match="element type"):
            jax.block_until_ready(_ffi(platform, digest, [out], x.astype(jnp.float32),
                                       p.astype(jnp.float32), e["t"]))


@pytest.mark.gpu
def test_a_call_on_another_gpu_is_refused_by_name(ops):
    device = _device("cuda")
    with jax.default_device(device):
        x, p, e = _drag_make(4, np.float64, np.random.default_rng(0))
        ops["drag"](x, p, e["t"])
        level = ops["drag"].level
        unit = level.unit(ops["drag"]._dir, "cuda", "float64",
                          _digest(ops["drag"], "cuda", "float64").rsplit("cuda-", 1)[1])
        fields = eagle_jax._recipe_fields(level, level.artifact(unit), "float64")
        seam = eagle_jax._seam()
        keep = level.slots[("keep", "cuda", "float64")]
        eagle_jax._install("eagle-test/other-gpu", fields, platform="cuda", entry=seam["run"],
                           ordinal=device.local_hardware_id + 1,
                           function=keep.device_function, seam=seam)
        out = jax.ShapeDtypeStruct((3, 4), jnp.float64)
        with pytest.raises(Exception, match="one GPU per process until 0.7"):
            jax.block_until_ready(_ffi("cuda", "eagle-test/other-gpu", [out], x, p, e["t"]))


def test_a_cuda_call_without_the_cuda_backend_raises_the_typed_error(monkeypatch):
    """The GPU-less wheel leg: no CUDA plugin means BackendUnavailable, by name."""
    from eagle import BackendUnavailable, _core

    monkeypatch.setattr(eagle_jax, "_SEAM", {})
    monkeypatch.setattr(_core, "cuda_backend",
                        lambda: {"loaded": False, "error": "no libeagle_cuda.so here"})
    with pytest.raises(BackendUnavailable, match="no libeagle_cuda.so here"):
        eagle_jax._seam()


# -- outputs -------------------------------------------------------------------------

@pytest.mark.parametrize("platform", PLATFORMS)
def test_unwritten_outputs_are_zero_on_a_dirtied_allocator(ops, platform):
    n = 1 << 18
    with jax.default_device(_device(platform)):
        junk = [jnp.full((3, n), 7.0 + i) for i in range(4)]
        jax.block_until_ready(junk)
        del junk
        x = jnp.ones((3, n))
        t = jnp.asarray(np.arange(n) % 2 == 1)
        a = np.asarray(jax.jit(lambda x, t: ops["drag"](x, 0.5, t))(x, t))
    assert np.all(a[:, 1::2] == 0.0)
    np.testing.assert_allclose(a[:, 0::2], -0.5 * np.sqrt(3.0))


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_donated_finished_mask_comes_back_in_place(ops, platform):
    with jax.default_device(_device(platform)):
        k = jnp.arange(6.0)
        t = jnp.asarray([False, True, False, False, False, False])
        before = t.unsafe_buffer_pointer()
        f = jax.jit(lambda k, t: ops["step_and_finish"](k, 3.0, t), donate_argnums=1)
        k_next, done = f(k, t)
        assert done.unsafe_buffer_pointer() == before
        np.testing.assert_array_equal(np.asarray(done), [False, True, True, True, True, True])
        np.testing.assert_array_equal(np.asarray(k_next), [1.0, 0.0, 3.0, 4.0, 5.0, 6.0])
        kept = jnp.asarray([False, True, False, False, False, False])
        jax.jit(lambda k, t: ops["step_and_finish"](k, 3.0, t))(k, kept)
        np.testing.assert_array_equal(np.asarray(kept), [False, True, False, False, False, False])


# -- carried state, finishing kernels, the user's kernel name -------------------------

@hawk.kernel(steps=1)
def count_up(nstop: Param, terminated: Terminated, k: Mutable[Scalar]):
    k = k + 1.0
    terminated = k >= nstop


@hawk.kernel
def finish_plain(k: Scalar, nstop: Param, terminated: Terminated, k_next: Mutable[Scalar]):
    k1 = k + 1.0
    k_next = k1
    terminated = k1 >= nstop


def _until_done_state(platform, k0, stop):
    """``until_done`` on the same kernel: the answer a JAX loop must reproduce."""
    import eagle
    import eagle.plan as eplan

    if platform == "cuda":
        import cupy as cp

        k = cp.asarray(k0)
        eagle.until_done(eplan.auto(count_up), max_steps=1000, k=k, nstop=stop).run()
        return cp.asnumpy(k)
    k = np.array(k0)
    eagle.until_done(eplan.auto(count_up), max_steps=1000, k=k, nstop=stop).run()
    return k


@pytest.mark.parametrize("platform", PLATFORMS)
def test_until_done_compacts_on_jax_arrays_outside_jit(platform):
    """Compaction runs on the non-jit path: a JAX array is read in place (DLPack), the
    caller's buffer is never written, and ``result`` returns the answer as a new JAX array."""
    import eagle
    import eagle.plan as eplan
    from hawk.ext import Guard, Kind

    @hawk.kernel(kind=Kind("jax_compact", guard=Guard(active_set=True)), steps=1)
    def count_compact(nstop: Param, terminated: Terminated, k: Mutable[Scalar]):
        k = k + 1.0
        terminated = k >= nstop

    stop, n = 500.0, 1 << 12
    k0 = np.random.default_rng(0).uniform(0, 490, n)
    with jax.default_device(_device(platform)):
        k_in = jnp.asarray(k0)
        runner = eagle.until_done(eplan.auto(count_compact), max_steps=1000, k=k_in, nstop=stop)
        rep = runner.run()
        k_out = runner.result("k")
    assert rep.done and rep.finished == n
    assert rep.compactions >= 1                       # the loop really compacted
    assert isinstance(k_out, jax.Array) and k_out is not k_in
    np.testing.assert_array_equal(np.asarray(k_in), k0)           # the input is untouched
    np.testing.assert_allclose(np.asarray(k_out), k0 + np.ceil(stop - k0), atol=1e-9)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_carried_state_plane_keeps_its_value_in_a_loop_with_no_where(tmp_path, platform):
    from jax import lax

    stop, n = 500.0, 1 << 12
    k0 = np.random.default_rng(0).uniform(0, 50, n)
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(count_up, wrt=(), cache_dir=tmp_path)
        m0 = jnp.zeros(n, bool)

        def body(c):
            k, m, i = c
            k, m = K(stop, m, k)
            return k, m, i + 1

        loop = jax.jit(lambda k, m: lax.while_loop(lambda c: ~c[1].all(), body, (k, m, 0)))
        k, m, steps = loop(jnp.asarray(k0), m0)
        want = _until_done_state(platform, k0, stop)
        assert bool(m.all()) and int(steps) == 500
        np.testing.assert_array_equal(np.asarray(k), want)

        def step(c, _):
            k, m = c
            k, m = K(stop, m, k)
            return (k, m), None

        scanned = jax.jit(lambda k: lax.scan(step, (k, m0), None, length=600)[0])
        np.testing.assert_array_equal(np.asarray(scanned(jnp.asarray(k0))[0]), want)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_wrt_empty_wraps_a_state_kernel_forward_only(tmp_path, platform):
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(count_up, wrt=(), cache_dir=tmp_path)
        k, m = jnp.arange(4.0), jnp.zeros(4, bool)
        np.testing.assert_array_equal(np.asarray(jax.jit(lambda k, m: K(2.5, m, k)[0])(k, m)),
                                      [1.0, 2.0, 3.0, 4.0])
        batched = jax.vmap(lambda k: K(2.5, m, k)[0])(jnp.stack([k, k + 1]))
        assert batched.shape == (2, 4)
        with pytest.raises(NotImplementedError, match="count_up.*forward-only"):
            jax.grad(lambda k: K(2.5, m, k)[0].sum())(k)


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("jit", [False, True])
def test_min_cell_is_the_exact_minimum(tmp_path, platform, jit):
    if not hasattr(hawk, "all_samples"):
        pytest.skip("this hawk has no reductions over samples")
    for n in (1, 33, 1000):
        x = np.random.default_rng(n).uniform(-5, 5, n)
        with jax.default_device(_device(platform)):
            K = eagle_jax.function(cell_min, wrt=(), cache_dir=tmp_path)
            f = jax.jit(K) if jit else K
            got = np.asarray(f(jnp.asarray(x)))
            assert got.shape == (1,) and got[0] == x.min()
            x[n // 2] = -np.inf
            assert np.asarray(f(jnp.asarray(x)))[0] == -np.inf


def test_float64_sum_cell_on_the_cpu_matches_eagle_reduce(tmp_path):
    """The host finish of a float64 sum cell takes the arithmetic eagle.reduce
    takes on this machine (the fixed-point leaf where the GPU has slow FP64),
    so it equals the GPU's finish bit for bit."""
    if not hasattr(hawk, "all_samples"):
        pytest.skip("this hawk has no reductions over samples")
    import eagle
    from eagle import _cell_rows
    from eagle._graph_reduce import fixed_f64

    rng = np.random.default_rng(0)
    big = rng.uniform(1, 2, 500) * 2.0 ** rng.integers(-80, 80, 500)
    v = np.concatenate([big, -big[::-1]])
    v[::3] *= 1 + 2.0 ** -30          # wide spread: the two arithmetics differ here
    x = np.zeros(v.size * _cell_rows.ROW)
    x[::_cell_rows.ROW] = v           # one term per row: the rows are v exactly
    want = np.zeros(1)
    eagle.Graph().add_node(eagle.reduce(v, want)).build().launch()
    with jax.default_device(_device("cpu")):
        got = np.asarray(eagle_jax.function(cell_sum, wrt=(), cache_dir=tmp_path)(jnp.asarray(x)))
    assert got.tobytes() == want.tobytes()
    if fixed_f64():                   # non-vacuity: the (hi, comp) finish gives other bits
        from eagle import _core

        comp = np.zeros(1)
        _core.reduce_canonical_host(v.ctypes.data, v.size, 1, 0, comp.ctypes.data)
        assert comp.tobytes() != want.tobytes()


@pytest.mark.parametrize("policy", ["compensated", "ordered"])
def test_policy_reaches_the_cell_recipe(tmp_path, policy):
    """The recipe the handler runs carries the policy given, and the sum is right."""
    if not hasattr(hawk, "all_samples"):
        pytest.skip("this hawk has no reductions over samples")
    from eagle import _cell_rows
    from eagle._graph_reduce import POLICIES

    x = np.random.default_rng(0).uniform(-1, 1, 3 * _cell_rows.ROW + 5)
    with jax.default_device(_device("cpu")):
        got = np.asarray(eagle_jax.function(cell_sum, wrt=(), cache_dir=tmp_path,
                                            policy=policy)(jnp.asarray(x)))
    np.testing.assert_allclose(got, [x.sum()], rtol=1e-12)
    recipes = [json.loads(p.read_text()) for p in tmp_path.rglob(eagle_jax.RECIPE_FILE)]
    assert [r["policy"] for r in recipes] == [POLICIES[policy]]


def test_unknown_policy_is_refused_like_eagle_reduce(tmp_path):
    with pytest.raises(ValueError, match=r"policy 'fast' is not one of \['auto', 'compensated', 'ordered'\]"):
        eagle_jax.function(drag, cache_dir=tmp_path, policy="fast")


@pytest.mark.parametrize("platform", PLATFORMS)
def test_grad_through_scan_of_a_finishing_kernel_is_exact(tmp_path, platform):
    from jax import lax

    stop, n = 500.0, 1 << 10
    k0 = jnp.asarray(np.random.default_rng(0).uniform(0, 50, n))
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(step_and_finish, cache_dir=tmp_path)
        m0 = jnp.zeros(n, bool)

        def run(k):
            def step(c, _):
                k, m = c
                k_next, m = K(k, stop, m)
                return (jnp.where(m & (k_next == 0.0), k, k_next), m), None

            return lax.scan(step, (k, m0), None, length=600)[0][0].sum()

        assert float(jax.grad(run)(k0).sum()) == n


@pytest.mark.parametrize("platform", PLATFORMS)
def test_a_finishing_kernel_without_steps_one_runs_and_differentiates(tmp_path, platform):
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(finish_plain, cache_dir=tmp_path)
        k, m = jnp.arange(4.0) + 0.5, jnp.zeros(4, bool)
        k_next, done = K(k, 3.0, m)
        np.testing.assert_array_equal(np.asarray(k_next), [1.5, 2.5, 3.5, 4.5])
        np.testing.assert_array_equal(np.asarray(done), [False, False, True, True])
        np.testing.assert_array_equal(np.asarray(jax.grad(lambda k: K(k, 3.0, m)[0].sum())(k)),
                                      np.ones(4))


def test_refusals_name_the_users_kernel(ops):
    x = jnp.ones((3, 4), jnp.float64)
    with pytest.raises(ValueError) as err:
        ops["drag"](x[:, :3], 0.3, jnp.zeros(4, bool))
    assert "'drag'" in str(err.value) or "drag:" in str(err.value)
    assert "_params" not in str(err.value)


# -- the argument PODs against eagle's own packer -------------------------------------

@pytest.fixture(scope="module")
def probe(tmp_path_factory):
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


def _recorded(lib):
    buf = (ctypes.c_ubyte * 8192)()
    got = lib.probe_read(buf, 8192)
    assert got > 0
    raw = bytes(buf[:got])
    nslots = int.from_bytes(raw[:4], "little", signed=True)
    boxes = [raw[4 + 40 * s: 44 + 40 * s] for s in range(nslots)]
    triple = np.frombuffer(raw[4 + 40 * nslots:], np.int64)
    return boxes, triple


@pytest.mark.parametrize("platform", PLATFORMS)
def test_handler_pods_match_eagles_packer(ops, probe, platform):
    from eagle._plan_pack import _pack_args

    device = _device(platform)
    K = ops["drag"]
    level = K.level
    n = 5
    with jax.default_device(device):
        x, p, e = _drag_make(n, np.float64, np.random.default_rng(2))
        K(x, p, e["t"])
        digest = _digest(K, platform, "float64")
        arch = digest.rsplit("cuda-", 1)[1] if platform == "cuda" else ""
        unit = level.unit(K._dir, "host" if platform == "cpu" else "cuda", "float64", arch)
        fields = eagle_jax._recipe_fields(level, level.artifact(unit), "float64")
        test_digest = f"eagle-test/parity/{platform}"
        if platform == "cpu":
            probe.probe_set_slots(len(fields["slot_kind"]))
            eagle_jax._install(test_digest, fields, platform="cpu",
                               entry=ctypes.cast(probe.probe_host, ctypes.c_void_p).value)
        else:
            seam = dict(eagle_jax._seam())
            eagle_jax._install(test_digest, fields, platform="cuda",
                               entry=ctypes.cast(probe.probe_run_device, ctypes.c_void_p).value,
                               ordinal=device.local_hardware_id, function=1, seam=seam)
        out = _ffi(platform, test_digest, [jax.ShapeDtypeStruct((3, n), jnp.float64)],
                   x, p, e["t"])
        jax.block_until_ready(out)
    boxes, triple = _recorded(probe)
    np.testing.assert_array_equal(triple, [0, n, n])
    spec = level.sig.arg_spec
    host = {"a": np.zeros((3, n)), "v": np.asarray(x), "cd": np.asarray(p),
            "terminated": np.asarray(e["t"])}
    device_type = 1 if platform == "cpu" else 2
    want, _ = _pack_args(spec, host, {}, n, device_type, vec_mutables=frozenset({"a"}),
                         dtype=np.float64)
    pointers = {"v": x.unsafe_buffer_pointer(), "cd": p.unsafe_buffer_pointer(),
                "terminated": e["t"].unsafe_buffer_pointer(), "a": out[0].unsafe_buffer_pointer()}
    assert len(boxes) == len(spec)
    for (role, name), got, box in zip(spec, boxes, want):
        packed = bytes(box)
        # field 0 is the buffer address: XLA's buffer here, numpy's in the packer
        assert int.from_bytes(got[:8], "little") == pointers[name], name
        # the last field is the device id: the call's device, 0 in the packer
        assert got[8:len(packed) - 4] == packed[8:len(packed) - 4], (role, name)
        device_id = int.from_bytes(got[len(packed) - 4:len(packed)], "little")
        assert device_id == (0 if platform == "cpu" else device.local_hardware_id), name


# -- the library -------------------------------------------------------------------------

def _xla_path():
    from eagle import _core

    return pathlib.Path(_core.__file__).resolve().parent / "libeagle_xla.so"


def test_library_exports_three_symbols_and_needs_no_cuda():
    nm, ldd = shutil.which("nm"), shutil.which("ldd")
    if nm is None or ldd is None:
        pytest.skip("binutils nm/ldd not on PATH")
    out = subprocess.run([nm, "-D", "--defined-only", str(_xla_path())], capture_output=True,
                         text=True, check=True).stdout
    names = sorted({line.split()[-1].split("@")[0] for line in out.splitlines()
                    if line.split() and line.split()[-2] in ("T", "W", "D", "B")})
    assert names == ["eagle_xla_launch_cpu", "eagle_xla_launch_cuda", "eagle_xla_register"]
    needed = subprocess.run([ldd, str(_xla_path())], capture_output=True, text=True).stdout
    assert not re.search(r"libcuda|libcudart|libnvrtc", needed), needed


@pytest.mark.repo_local
def test_pinned_ffi_api_matches_the_vendored_header():
    header = (HERE.parent / "src" / "xla" / "vendor" / "xla" / "ffi" / "api" / "c_api.h").read_text()
    major = int(re.search(r"#define XLA_FFI_API_MAJOR (\d+)", header).group(1))
    minor = int(re.search(r"#define XLA_FFI_API_MINOR (\d+)", header).group(1))
    assert eagle_jax.XLA_FFI_API == (major, minor)


# -- processes: persistent cache, export + preload ----------------------------------------

_CHILD = textwrap.dedent("""
    import os, sys
    os.environ["JAX_PLATFORMS"] = "cpu"
    import jax, jax.numpy as jnp, numpy as np
    jax.config.update("jax_enable_x64", True)
    import hawk
    from hawk import Mutable, Param, Terminated, Vector
    from hawk.math import dot, sqrt
    from eagle.frameworks import jax as ej

    @hawk.kernel
    def drag(v: Vector[3], cd: Param, terminated: Terminated, a: Mutable[Vector[3]]):
        a = -cd * sqrt(dot(v, v)) * v

    cache, what = sys.argv[1], sys.argv[2]
    v = jnp.ones((3, 4)); cd = jnp.asarray([0.5]); t = jnp.zeros(4, bool)
    want = -0.5 * np.sqrt(3.0)
    if what == "cache":
        jax.config.update("jax_compilation_cache_dir", sys.argv[3])
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        K = ej.function(drag, cache_dir=cache)
        assert np.allclose(jax.jit(lambda v, cd, t: K(v, cd, t))(v, cd, t), want)
    elif what == "export":
        K = ej.function(drag, cache_dir=cache)
        f = jax.jit(lambda v, cd, t: K(v, cd, t))
        checks = [jax.export.DisabledSafetyCheck.custom_call(ej.TARGET)]
        exp = jax.export.export(f, disabled_checks=checks)(v, cd, t)
        open(sys.argv[3], "wb").write(exp.serialize())
    else:
        exp = jax.export.deserialize(bytearray(open(sys.argv[3], "rb").read()))
        try:
            exp.call(v, cd, t).block_until_ready()
            raise SystemExit("ran before preload")
        except Exception as err:
            assert "is not loaded in this process" in str(err), err
        assert ej.preload(cache)
        assert np.allclose(exp.call(v, cd, t), want)
    print("OK")
""")


def _child(tmp_path, *args):
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    run = subprocess.run([sys.executable, str(script), *map(str, args)], capture_output=True,
                         text=True, env=env, timeout=600)
    assert run.returncode == 0 and "OK" in run.stdout, run.stdout + run.stderr


def test_persistent_cache_hits_across_processes(tmp_path):
    cache, xla_cache = tmp_path / "build", tmp_path / "xla"
    _child(tmp_path, cache, "cache", xla_cache)
    first = sorted(os.listdir(xla_cache))
    _child(tmp_path, cache, "cache", xla_cache)
    assert first and sorted(os.listdir(xla_cache)) == first


def test_an_exported_module_runs_after_preload(tmp_path):
    pytest.importorskip("flatbuffers", reason="jax.export serialization needs flatbuffers")
    cache, blob = tmp_path / "build", tmp_path / "drag.exported"
    _child(tmp_path, cache, "export", blob)
    _child(tmp_path, cache, "load", blob)


# -- threads ----------------------------------------------------------------------------

@pytest.mark.ft
@pytest.mark.parametrize("platform", PLATFORMS)
def test_threads_calling_one_function_on_distinct_inputs(ops, platform):
    device = _device(platform)
    K = ops["drag"]
    f = jax.jit(lambda x, p, t: K(x, p, t))
    errors = []

    def work(i):
        try:
            with jax.default_device(device):
                for j in range(50):
                    x, p, e = _drag_make(8, np.float64, np.random.default_rng(100 * i + j))
                    _close(f(x, p, e["t"]), _drag_ref(x, p, e), "float64")
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[0]


@pytest.mark.ft
@pytest.mark.parametrize("platform", PLATFORMS)
def test_threads_tracing_a_fresh_kernel_at_a_barrier(tmp_path, platform):
    device = _device(platform)
    K = eagle_jax.function(drag, cache_dir=tmp_path)
    barrier = threading.Barrier(8)
    errors, prims = [], []

    def work(i):
        try:
            with jax.default_device(device):
                x, p, e = _drag_make(6, np.float64, np.random.default_rng(i))
                loss = lambda x, p: jnp.sum(K(x, p, e["t"]) ** 2)        # noqa: E731
                rloss = lambda x, p: jnp.sum(_drag_ref(x, p, e) ** 2)    # noqa: E731
                barrier.wait()
                _close(jax.grad(loss, 1)(x, p), jax.grad(rloss, 1)(x, p), "float64")
                prims.append((id(K.level.slots["jax"]), id(K.level.derived("jvp")),
                              id(K.level.derived("vjp"))))
        except Exception as err:  # noqa: BLE001
            errors.append(err)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[0]
    assert len(set(prims)) == 1


# -- the handler honours the alias itself; vmap of a carried-state kernel -----------------

N_OSC, DT = 512, 0.01


@hawk.kernel(steps=1)
def osc(dt: Param, c: Param, t_final: Scalar, terminated: Terminated,
        x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    vn = v - dt * (x + c * v)
    x = x + dt * v
    v = vn
    tn = t + dt
    t = tn
    terminated = tn >= t_final


def _osc_start():
    n = N_OSC
    return (jnp.linspace(0.5, 1.5, n), jnp.linspace(-0.2, 0.3, n), jnp.zeros(n),
            jnp.zeros(n, bool), jnp.linspace(0.5, 2.0, n))


def _osc_advance(K, c, s, tf):
    x, v, t, done = s
    t, v, x, done = K(DT, c, tf, done, x, v, t)         # outputs are t, v, x, terminated
    return x, v, t, done


@pytest.mark.parametrize("platform", PLATFORMS)
def test_vmap_of_two_steps_carries_the_state(tmp_path, platform):
    drags = jnp.linspace(0.0, 1.5, 4)
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(osc, cache_dir=tmp_path)
        x, v, t, done, tf = _osc_start()

        def two(c):
            s = (x, v, t, done)
            for _ in range(2):
                s = _osc_advance(K, c, s, tf)
            return s[2]

        got = np.asarray(jax.jit(jax.vmap(two))(drags))
        assert got.shape == (4, N_OSC)
        np.testing.assert_array_equal(got, np.full((4, N_OSC), 2 * DT))


@pytest.mark.parametrize("platform", PLATFORMS)
def test_vmap_of_a_capped_while_finishes(tmp_path, platform):
    drags = jnp.linspace(0.0, 1.5, 4)
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(osc, cache_dir=tmp_path)
        x, v, t, done, tf = _osc_start()

        def one(c):
            def cond(s):
                return ~jnp.all(s[0][3]) & (s[1] < 400)

            s, k = jax.lax.while_loop(cond, lambda s: (_osc_advance(K, c, s[0], tf), s[1] + 1),
                                      ((x, v, t, done), 0))
            return s[3].sum(), k

        finished, steps = jax.jit(jax.vmap(one))(drags)
        np.testing.assert_array_equal(np.asarray(finished), np.full(4, N_OSC))
        assert np.all(np.abs(np.asarray(steps) - 200) <= 1)


def _osc_rollout(K, steps, tf):
    """``x`` after ``steps`` steps of ``osc`` as a function of ``(c, x0, v0)``."""
    n = tf.shape[0]
    t0, done0 = jnp.zeros(n), jnp.zeros(n, bool)

    def run(c, x0, v0):
        x, v, t, done = x0, v0, t0, done0
        for _ in range(steps):
            t, v, x, done = K(DT, c, tf, done, x, v, t)
        return x

    return run


@pytest.mark.parametrize("platform", PLATFORMS)
@pytest.mark.parametrize("finishing", [False, True], ids=["live", "terminated"])
def test_forward_mode_of_a_carried_state_kernel_matches_differences_and_reverse(
        tmp_path, platform, finishing):
    n = 6
    # finishing: samples 0-1 finish after step 1, 2-3 after step 2, the rest never
    tf = (jnp.asarray([0.005, 0.005, 0.015, 0.015, 5.0, 5.0]) if finishing
          else jnp.full(n, 5.0))
    x0, v0 = jnp.linspace(0.5, 1.5, n), jnp.linspace(-0.2, 0.3, n)
    c = jnp.asarray([0.3])
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(osc, cache_dir=tmp_path)
        run = _osc_rollout(K, 4, tf)
        f = lambda x: run(c, x, v0)
        tx = jnp.linspace(1.0, 2.0, n)
        out, tan = jax.jvp(f, (x0,), (tx,))
        eps = 1e-6
        diff = (f(x0 + eps * tx) - f(x0 - eps * tx)) / (2 * eps)
        np.testing.assert_allclose(np.asarray(tan), np.asarray(diff), atol=1e-8)
        fwd, rev = jax.jacfwd(f)(x0), jax.jacrev(f)(x0)
        np.testing.assert_allclose(np.asarray(fwd), np.asarray(rev), atol=1e-12)
        g = lambda cc: run(cc, x0, v0)
        np.testing.assert_allclose(np.asarray(jax.jit(jax.jacfwd(g))(c)),
                                   np.asarray(jax.jacrev(g)(c)), atol=1e-12)
        if finishing:
            # a sample finished after step 1 ran ONE step: dx/dx0 = 1 + dt * (dv/dx0 = 0)
            np.testing.assert_allclose(np.asarray(jnp.diag(fwd))[:2], 1.0, atol=1e-14)
            assert np.asarray(jnp.diag(fwd))[4] != 1.0


# the reverse of a carried-state kernel, as the step-vjp path (custom_vjp) gave it before the
# transposed-jvp rule; the plane gradients are pinned to it, the Param's is not (see below)
_OSC_GRAD_X0 = [0.986, 1.3880000000000001, 1.7796700180000002, 2.1835836226000005,
                2.572987863442422, 2.9804674789531442]
_OSC_GRAD_V0 = [1.00696, 1.01098, 1.0298514054, 1.03791836678, 1.0915509991095553,
                1.1077850773265565]


@pytest.mark.parametrize("platform", PLATFORMS)
def test_reverse_of_a_carried_state_kernel_equals_the_step_vjp_path(tmp_path, platform):
    """``jax.grad`` of a rollout with samples finishing at different steps: the
    plane gradients equal what the earlier step-vjp reverse gave (within a few ulp);
    the ``Param`` gradient is the finite difference (the earlier path also summed the
    steps a finished sample did not take: it gave -0.0096379 here)."""
    tf = jnp.asarray([0.005, 0.005, 0.015, 0.015, 5.0, 5.0])
    x0, v0, c = jnp.linspace(0.5, 1.5, 6), jnp.linspace(-0.2, 0.3, 6), jnp.asarray([0.3])
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(osc, cache_dir=tmp_path)
        def loss(c, x, v):
            n = 6
            t, d = jnp.zeros(n), jnp.zeros(n, bool)
            for _ in range(4):
                t, v, x, d = K(DT, c, tf, d, x, v, t)
            return (x * x + v).sum()

        gc, gx, gv = jax.grad(loss, (0, 1, 2))(c, x0, v0)
        np.testing.assert_allclose(np.asarray(gx), _OSC_GRAD_X0, rtol=1e-14)
        np.testing.assert_allclose(np.asarray(gv), _OSC_GRAD_V0, rtol=1e-14)
        eps = 1e-6
        fd = (loss(c + eps, x0, v0) - loss(c - eps, x0, v0)) / (2 * eps)
        np.testing.assert_allclose(np.asarray(gc), [float(fd)], rtol=1e-6)


# -- a batch of Param values under vmap: one launch over n * B samples ----------------

def _eagle_primitives(f, *args):
    """The names of eagle's primitives and whether a ``scan``/``while`` loop is in ``f``'s jaxpr."""
    text = str(jax.make_jaxpr(f)(*args))
    names = re.findall(r"eagle_\w+", text)
    return names, ("scan" in text or "while" in text)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_vmap_over_a_param_is_one_launch_equal_to_the_loop(tmp_path, platform):
    x, gs = jnp.linspace(0.1, 1.0, 7), jnp.asarray([0.5, 1.5, -0.8])
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(drag, cache_dir=tmp_path)
        m = jnp.zeros(7, bool)
        f = lambda c: K(jnp.stack([x, x * 2, x * 3]), c.reshape(1), m)
        got = jax.vmap(f)(gs)
        loop = jnp.stack([f(c) for c in gs])
        assert got.shape == (3, 3, 7)
        np.testing.assert_array_equal(np.asarray(got), np.asarray(loop))
        names, looped = _eagle_primitives(jax.vmap(f), gs)
        assert names == ["eagle_drag_params_cells"] and not looped
        loss = lambda c: (f(c) ** 2).sum()
        # the entry's gradient sums its samples in the kernel's scatter, the loop's in the
        # finished cell: the same terms, a different order
        np.testing.assert_allclose(np.asarray(jax.vmap(jax.grad(loss))(gs)),
                                   np.asarray(jnp.stack([jax.grad(loss)(c) for c in gs])), rtol=1e-13)
        eps = 1e-6
        fd = (jax.vmap(loss)(gs + eps) - jax.vmap(loss)(gs - eps)) / (2 * eps)
        np.testing.assert_allclose(np.asarray(jax.vmap(jax.grad(loss))(gs)), np.asarray(fd), rtol=1e-6)
        np.testing.assert_allclose(np.asarray(jax.vmap(jax.jacfwd(loss))(gs)),
                                   np.asarray(jax.vmap(jax.grad(loss))(gs)), rtol=1e-13)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_jacrev_over_planes_of_a_kernel_with_a_param_is_one_launch(tmp_path, platform):
    """The reverse Jacobian maps the vjp over cotangents; the ``Param``'s gradient is one
    entry per cotangent, not a loop."""
    x = jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, (3, 6)))
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(drag, cache_dir=tmp_path)
        m = jnp.zeros(6, bool)
        f = lambda v: K(v, jnp.asarray([0.05]), m)
        _, looped = _eagle_primitives(jax.jacrev(f), x)
        assert not looped
        np.testing.assert_allclose(np.asarray(jax.jacrev(f)(x)), np.asarray(jax.jacfwd(f)(x)),
                                   rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("platform", PLATFORMS)
def test_vmap_over_a_param_of_a_carried_state_kernel_with_finished_samples(tmp_path, platform):
    tf = jnp.asarray([0.005, 0.005, 0.015, 0.015, 5.0, 5.0])
    x0, v0, cs = jnp.linspace(0.5, 1.5, 6), jnp.linspace(-0.2, 0.3, 6), jnp.asarray([0.3, 0.0, 1.2])
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(osc, cache_dir=tmp_path)

        def loss(c):
            x, v, t, done = x0, v0, jnp.zeros(6), jnp.zeros(6, bool)
            for _ in range(4):
                t, v, x, done = K(DT, c.reshape(1), tf, done, x, v, t)
            return (x * x + v).sum()

        np.testing.assert_array_equal(np.asarray(jax.vmap(loss)(cs)),
                                      np.asarray(jnp.stack([loss(c) for c in cs])))
        grads = np.asarray(jax.vmap(jax.grad(loss))(cs))
        np.testing.assert_allclose(grads, np.asarray(jnp.stack([jax.grad(loss)(c) for c in cs])),
                                   rtol=1e-13)
        eps = 1e-6
        np.testing.assert_allclose(
            grads, np.asarray((jax.vmap(loss)(cs + eps) - jax.vmap(loss)(cs - eps)) / (2 * eps)),
            rtol=1e-6)
        np.testing.assert_allclose(np.asarray(jax.vmap(jax.jacfwd(loss))(cs)), grads, rtol=1e-13)
        names, looped = _eagle_primitives(jax.vmap(loss), cs)
        assert set(names) == {"eagle_osc_params_cells"} and not looped


@pytest.mark.skipif(not hasattr(hawk, "all_samples"), reason="this hawk has no reductions over samples")
def test_vmap_over_a_param_of_a_kernel_with_cells_keeps_the_loop(tmp_path):
    @hawk.kernel
    def scaled_sum(x: Scalar, k: Param, t: Mutable[Scalar]):
        t = hawk.all_samples(k * x).sum()  # noqa: F841

    x, ks = jnp.linspace(0.1, 1.0, 9), jnp.asarray([0.5, 1.5, -0.8])
    with jax.default_device(_device("cpu")):
        K = eagle_jax.function(scaled_sum, cache_dir=tmp_path)
        f = lambda k: K(x, k.reshape(1))
        np.testing.assert_array_equal(np.asarray(jax.vmap(f)(ks)),
                                      np.asarray(jnp.stack([f(k) for k in ks])))
        names, looped = _eagle_primitives(jax.vmap(f), ks)
        assert names == ["eagle_scaled_sum_params"] and looped


@pytest.mark.parametrize("platform", PLATFORMS)
def test_outputs_follow_the_kernels_slot_order_not_the_declaration(tmp_path, platform):
    with jax.default_device(_device(platform)):
        K = eagle_jax.function(osc, cache_dir=tmp_path)
        assert K.outputs == ("t", "v", "x", "terminated")


def _active_kernel():
    from hawk.ext import Guard, Kind

    @hawk.kernel(kind=Kind("jax_active", guard=Guard(active_set=True)))
    def spring(w: Scalar, a: Param, terminated: Terminated, y: Mutable[Scalar]):
        y = a * w * w + sqrt(w)

    return spring


@pytest.mark.parametrize("platform", PLATFORMS)
def test_an_active_set_kernel_runs_every_sample_and_differentiates(tmp_path, platform):
    """The bridge binds the identity index map: the answer is eagle.plan's with
    every sample live, and the derivative kernels read the same planes."""
    import eagle.plan as eplan
    from eagle import ActiveSet

    n = 1000
    w = np.random.default_rng(3).uniform(0.5, 2.0, n)
    with jax.default_device(_device(platform)):
        spring = _active_kernel()
        K = eagle_jax.function(spring, cache_dir=tmp_path)
        assert K.inputs == ("w", "a", "terminated")
        m = jnp.zeros(n, bool)
        got = np.asarray(jax.jit(lambda w: K(w, 0.7, m))(jnp.asarray(w)))
        y, term = np.zeros(n), np.zeros(n, dtype=bool)
        eplan.auto(spring).bind(w=w, a=0.7, terminated=term, y=y, **ActiveSet(term).planes()).launch()
        np.testing.assert_array_equal(got, y)
        np.testing.assert_allclose(got, 0.7 * w * w + np.sqrt(w), rtol=1e-14)
        gw = jax.grad(lambda w, a: K(w, a, m).sum(), (0, 1))(jnp.asarray(w), 0.7)
        np.testing.assert_allclose(np.asarray(gw[0]), 1.4 * w + 0.5 / np.sqrt(w), rtol=1e-13)
        np.testing.assert_allclose(float(gw[1]), (w * w).sum(), rtol=1e-13)
        tangent = jax.jvp(lambda w: K(w, 0.7, m), (jnp.asarray(w),), (jnp.ones(n),))[1]
        np.testing.assert_allclose(np.asarray(tangent), 1.4 * w + 0.5 / np.sqrt(w), rtol=1e-13)


def test_a_finished_count_of_the_wrong_width_is_refused_by_name():
    import eagle.plan as eplan

    plan = eplan.auto(osc)
    n = 4
    with pytest.raises(ValueError, match=r"'finished_count'.*uint32.*finished_count\.astype\(np\.uint32\)"):
        plan.bind(dt=0.01, c=0.1, t_final=np.ones(n), terminated=np.zeros(n, bool), x=np.ones(n),
                  v=np.ones(n), t=np.zeros(n), finished_count=np.zeros(1, np.int64))


def test_a_jax_node_name_is_the_vjps_key():
    import eagle

    w, y = np.arange(4.0), np.zeros(4)
    node = eagle_jax.node(lambda a: a * 2.0, reads={"w": w}, writes={"y": y}, name="model")
    g = eagle.Graph().add_node(node).build(record=True)
    g.launch()
    np.testing.assert_array_equal(y, 2.0 * w)
    with pytest.raises(ValueError, match=r"node 'model' is opaque: pass vjps=\{'model'"):
        eagle.diff.reverse(g.record, {"model.y": np.ones(4)})
    grads = eagle.diff.reverse(g.record, {"model.y": np.ones(4)},
                               vjps={"model": lambda primals, cts: {"w": 2.0 * cts["y"]}})
    np.testing.assert_array_equal(grads["w"], np.full(4, 2.0))
