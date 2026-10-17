# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``eagle.frameworks.warp``: a hawk kernel inside a Warp program.

One subject kernel (a vector plane, a ``Param``, a ``Terminated`` mask, a
vector output) runs on Warp CPU arrays through hawk's host runtime and on Warp
CUDA arrays through eagle's launch path, in float32 and float64. The rows:
values against a NumPy reference; ``wp.Tape`` gradients against central finite
differences (planes and the ``Param``); the ``Param`` gradient is a
fixed-order sum; capture of forward and backward in one CUDA graph (a sync
anywhere would break it); stream order on Warp's stream, with the instrument
shown to see a violation; eight threads.

CUDA rows carry the ``gpu`` marker and skip where no CUDA Warp device exists.
"""

import threading
import time
import traceback
from typing import Any

import numpy as np
import pytest

wp = pytest.importorskip("warp")
hawk = pytest.importorskip("hawk")

from hawk import Mutable, Param, Scalar, Terminated, Vector  # noqa: E402
from hawk.math import dot, sqrt  # noqa: E402

from eagle.frameworks import warp as eagle_warp  # noqa: E402

if hasattr(wp, "LOG_WARNING"):
    wp.config.log_level = wp.LOG_WARNING              # no init banner
else:
    wp.config.quiet = True
wp.init()

DEVICES = [
    "cpu",
    pytest.param("cuda:0", marks=pytest.mark.gpu),
]
MODES = {"float32": (wp.float32, np.float32), "float64": (wp.float64, np.float64)}
#: gradient tolerance against float64 finite differences, per mode
GRAD_RTOL = {"float32": 5e-4, "float64": 1e-7}

@hawk.kernel
def drag(v: Vector[3], cd: Param, terminated: Terminated, a: Mutable[Vector[3]]):
    a = -cd * sqrt(dot(v, v)) * v


@hawk.kernel(steps=1)
def step_and_finish(k: Scalar, nstop: Param, terminated: Terminated,
                    k_next: Mutable[Scalar]):
    k1 = k + 1.0
    k_next = k1
    terminated = k1 >= nstop


@wp.kernel
def weighted_sum(a: wp.array2d(dtype=Any), w: wp.array2d(dtype=Any), loss: wp.array(dtype=Any)):
    i, j = wp.tid()
    wp.atomic_add(loss, 0, a[i, j] * w[i, j])


@wp.kernel(module="unique")
def spin_write(x: wp.array2d(dtype=wp.float64), cycles: wp.int64, value: wp.float64):
    i, j = wp.tid()
    _spin(cycles)
    x[i, j] = value


@wp.func_native("""
long long t0 = clock64();
while (clock64() - t0 < cycles) {}
""")
def _spin(cycles: wp.int64):
    ...


@hawk.kernel
def finish_plain(k: Scalar, nstop: Param, terminated: Terminated, k_next: Mutable[Scalar]):
    k1 = k + 1.0
    k_next = k1
    terminated = k1 >= nstop


@hawk.kernel(steps=1)
def count_up(nstop: Param, terminated: Terminated, k: Mutable[Scalar]):
    k = k + 1.0
    terminated = k >= nstop


@hawk.kernel(steps=1)
def grow(rate: Param, nstop: Param, terminated: Terminated, k: Mutable[Scalar], n: Mutable[Scalar]):
    k = k + rate * k * k
    n = n + 1.0
    terminated = n >= nstop


@pytest.fixture(scope="module")
def op(tmp_path_factory):
    return eagle_warp.function(drag, cache_dir=tmp_path_factory.mktemp("warp_drag"))


def _skip_without(dev):
    if dev.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("no CUDA Warp device")


def _case(dev, mode, n=7, seed=0, grad=True):
    wdt, ndt = MODES[mode]
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(3, n)).astype(ndt)
    w = rng.normal(size=(3, n)).astype(ndt)
    term = np.zeros(n, dtype=bool)
    term[2] = True
    return dict(
        v_np=v, w_np=w, term_np=term, cd_np=0.3,
        v=wp.array(v, dtype=wdt, device=dev, requires_grad=grad),
        cd=wp.array([0.3], dtype=wdt, device=dev, requires_grad=grad),
        t=wp.array(term, dtype=wp.bool, device=dev),
        w=wp.array(w, dtype=wdt, device=dev))


def _law(v, cd, term):
    a = -cd * np.linalg.norm(v, axis=0) * v
    a[:, term] = 0.0
    return a


def _fd_grads(c, h=1e-6):
    """Central finite differences of ``sum(w * law)`` in float64."""
    v = c["v_np"].astype(np.float64)
    w, term = c["w_np"].astype(np.float64), c["term_np"]

    def loss(vv, cd):
        return (_law(vv, cd, term) * w).sum()

    gv = np.zeros_like(v)
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            up, dn = v.copy(), v.copy()
            up[i, j] += h
            dn[i, j] -= h
            gv[i, j] = (loss(up, c["cd_np"]) - loss(dn, c["cd_np"])) / (2 * h)
    gc = (loss(v, c["cd_np"] + h) - loss(v, c["cd_np"] - h)) / (2 * h)
    return gv, gc


def _tape_grads(op, c, dev, mode):
    loss = wp.zeros(1, dtype=MODES[mode][0], device=dev, requires_grad=True)
    with wp.Tape() as tape:
        a = op(c["v"], c["cd"], c["t"])
        wp.launch(weighted_sum, dim=a.shape, inputs=[a, c["w"]], outputs=[loss], device=dev)
    tape.backward(loss=loss)
    return loss, tape


# --------------------------------------------------------------------------- #
# values and gradients
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dev", DEVICES)
def test_forward_matches_the_reference(op, dev, mode):
    _skip_without(dev)
    c = _case(dev, mode, grad=False)
    a = op(c["v"], c["cd"], c["t"])
    rtol = 1e-5 if mode == "float32" else 1e-13
    np.testing.assert_allclose(
        a.numpy(), _law(c["v_np"].astype(np.float64), c["cd_np"], c["term_np"]), rtol=rtol, atol=0)
    assert not a.numpy()[:, 2].any()                  # the terminated sample stays zero


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dev", DEVICES)
def test_tape_gradients_match_finite_differences(op, dev, mode):
    _skip_without(dev)
    c = _case(dev, mode)
    _, tape = _tape_grads(op, c, dev, mode)
    gv, gc = _fd_grads(c)
    rtol = GRAD_RTOL[mode]
    np.testing.assert_allclose(c["v"].grad.numpy(), gv, rtol=rtol, atol=rtol * np.abs(gv).max())
    np.testing.assert_allclose(c["cd"].grad.numpy()[0], gc, rtol=rtol)
    assert not c["v"].grad.numpy()[:, 2].any()        # no gradient through a terminated sample


@pytest.mark.parametrize("dev", DEVICES)
def test_tape_zero_resets_and_a_second_backward_repeats(op, dev):
    _skip_without(dev)
    c = _case(dev, "float64")
    loss, tape = _tape_grads(op, c, dev, "float64")
    once = c["v"].grad.numpy().copy(), c["cd"].grad.numpy().copy()
    tape.zero()
    assert not c["v"].grad.numpy().any() and not c["cd"].grad.numpy().any()
    tape.backward(loss=loss)
    np.testing.assert_array_equal(c["v"].grad.numpy(), once[0])
    np.testing.assert_array_equal(c["cd"].grad.numpy(), once[1])


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dev", DEVICES)
def test_param_gradient_is_a_fixed_order_sum(op, dev, mode):
    """The Param's gradient is the per-sample partials summed in chunks of a
    fixed size, each chunk then the chunk totals in order: bitwise the same as
    that order in plain NumPy, and bitwise repeatable."""
    _skip_without(dev)
    n = 1000
    wdt, ndt = MODES[mode]
    c = _case(dev, mode, n=n, seed=3)
    c["term_np"][:] = False
    c["t"] = wp.array(c["term_np"], dtype=wp.bool, device=dev)
    runs = []
    for _ in range(2):
        c["cd"].grad.zero_()
        _tape_grads(op, c, dev, mode)
        runs.append(c["cd"].grad.numpy().copy())
    assert (runs[0] == runs[1]).all()
    # partials, as hawk's reverse kernel gives them, in the call's dtype
    v = c["v_np"]
    partial = (-(np.linalg.norm(v.astype(np.float64), axis=0) * (v * c["w_np"]).sum(axis=0))
               ).astype(ndt)
    chunk = -(-n // eagle_warp._CHUNKS)
    total = ndt(0)
    for lo in range(0, n, chunk):
        s = ndt(0)
        for x in partial[lo:lo + chunk]:
            s = ndt(s + x)
        total = ndt(total + s)
    np.testing.assert_allclose(runs[0][0], total, rtol=2e-4 if mode == "float32" else 1e-12)


@pytest.mark.parametrize("dev", DEVICES)
def test_a_finishing_kernel_returns_its_mask(tmp_path, dev):
    _skip_without(dev)
    step = eagle_warp.function(step_and_finish, cache_dir=tmp_path)
    k = wp.array(np.array([0.0, 1.0, 2.0, 3.0]), dtype=wp.float64, device=dev)
    nstop = wp.array([3.0], dtype=wp.float64, device=dev)
    term = wp.zeros(4, dtype=wp.bool, device=dev)
    k_next, done = step(k, nstop, term)
    np.testing.assert_array_equal(k_next.numpy(), [1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(done.numpy(), [False, False, True, True])
    assert not term.numpy().any()                     # the caller's own mask is untouched


@pytest.mark.parametrize("dev", DEVICES)
def test_named_refusals(op, dev):
    _skip_without(dev)
    c = _case(dev, "float64", grad=False)
    v32 = wp.array(c["v_np"].astype(np.float32), dtype=wp.float32, device=dev)
    with pytest.raises(TypeError, match="float32 build or the float64 one"):
        op(v32, c["cd"], c["t"])
    with pytest.raises(TypeError, match="must be a wp.array"):
        op(c["v_np"], c["cd"], c["t"])
    with pytest.raises(ValueError, match="expected"):
        op(wp.zeros((2, 7), dtype=wp.float64, device=dev), c["cd"], c["t"])


@pytest.mark.gpu
def test_arrays_on_two_devices_are_refused(op):
    _skip_without("cuda:0")
    c = _case("cuda:0", "float64", grad=False)
    cpu = wp.array(c["v_np"], dtype=wp.float64, device="cpu")
    with pytest.raises(ValueError, match="the call is on"):
        op(c["v"], c["cd"], wp.array(c["term_np"], dtype=wp.bool, device="cpu"))
    with pytest.raises(ValueError, match="the call is on"):
        op(cpu, c["cd"], c["t"])


# --------------------------------------------------------------------------- #
# CUDA graph capture
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
def test_graph_capture_replays_forward_and_backward(tmp_path, mode):
    """Forward, a Warp loss and the tape backward in ONE graph: a host sync
    anywhere would fail the capture. Replays follow the arrays' new contents."""
    _skip_without("cuda:0")
    dev = "cuda:0"
    fn = eagle_warp.function(drag, cache_dir=tmp_path)
    c = _case(dev, mode)
    wdt = MODES[mode][0]
    fn.prepare(mode, dev, backward=True)
    loss = wp.zeros(1, dtype=wdt, device=dev, requires_grad=True)
    out = wp.zeros((3, 7), dtype=wdt, device=dev, requires_grad=True)
    with wp.ScopedCapture(device=dev) as capture:
        with wp.Tape() as tape:
            fn(c["v"], c["cd"], c["t"], out=(out,))
            wp.launch(weighted_sum, dim=out.shape, inputs=[out, c["w"]], outputs=[loss], device=dev)
        tape.zero()
        loss.grad.fill_(1.0)
        tape.backward()
    rtol = GRAD_RTOL[mode]
    for scale in (1.0, 2.5):
        c["v"].assign(wp.array(c["v_np"] * scale, dtype=wdt, device=dev))
        wp.capture_launch(capture.graph)
        wp.synchronize_device(dev)
        c2 = dict(c, v_np=(c["v_np"] * scale))
        gv, gc = _fd_grads(c2)
        np.testing.assert_allclose(out.numpy(), _law(c2["v_np"].astype(np.float64), 0.3, c["term_np"]),
                                   rtol=1e-5 if mode == "float32" else 1e-13, atol=0)
        np.testing.assert_allclose(c["v"].grad.numpy(), gv, rtol=rtol, atol=rtol * np.abs(gv).max())
        np.testing.assert_allclose(c["cd"].grad.numpy()[0], gc, rtol=rtol)


@pytest.mark.gpu
@pytest.mark.parametrize("mode", MODES)
def test_graph_capture_replays_a_state_loop_and_its_backward(tmp_path, mode):
    _skip_without("cuda:0")
    dev = "cuda:0"
    fn = eagle_warp.function(grow, cache_dir=tmp_path)
    wdt = MODES[mode][0]
    w = np.random.default_rng(2).normal(size=_N)
    fn.prepare(mode, dev, backward=True)
    c = _grow_arrays(dev, mode)
    mk = lambda dt, grad=False: wp.zeros(_N, dtype=dt, device=dev, requires_grad=grad)
    outs = [(mk(wdt, True), mk(wdt), mk(wp.bool)) for _ in range(_STEPS)]
    loss = wp.zeros(1, dtype=wdt, device=dev, requires_grad=True)
    warr = wp.array(w, dtype=wdt, device=dev)
    with wp.ScopedCapture(device=dev) as capture:
        with wp.Tape() as tape:
            k, n, m = c["k"], c["n"], c["m"]
            for out in outs:
                k, n, m = fn(k, n, c["nstop"], c["rate"], m, out=out)
            wp.launch(_weighted_sum_1d, dim=_N, inputs=[k, warr], outputs=[loss], device=dev)
        tape.zero()
        loss.grad.fill_(1.0)
        tape.backward()
    rtol = GRAD_RTOL[mode]
    ref_k, ref_r = _grow_grads(w)
    for _ in range(2):                                                  # a replay repeats exactly
        wp.capture_launch(capture.graph)
        wp.synchronize_device(dev)
        np.testing.assert_allclose(outs[-1][0].numpy(), _grow_np(_K0, _RATE)[0],
                                   rtol=1e-5 if mode == "float32" else 1e-13, atol=0)
        np.testing.assert_allclose(c["k"].grad.numpy(), ref_k, rtol=rtol, atol=rtol * np.abs(ref_k).max())
        np.testing.assert_allclose(c["rate"].grad.numpy()[0], ref_r, rtol=rtol)


@pytest.mark.gpu
def test_an_unbuilt_kernel_inside_a_capture_is_refused_by_name(tmp_path):
    _skip_without("cuda:0")
    dev = "cuda:0"
    fresh = eagle_warp.function(drag, cache_dir=tmp_path)
    c = _case(dev, "float64", grad=False)
    with pytest.raises(RuntimeError, match="not built.*capture is open.*prepare"):
        with wp.ScopedCapture(device=dev):
            fresh(c["v"], c["cd"], c["t"])


# --------------------------------------------------------------------------- #
# stream order
# --------------------------------------------------------------------------- #
def _spin_cycles(seconds):
    import cupy

    khz = int(cupy.cuda.runtime.deviceGetAttribute(13, 0))   # cudaDevAttrClockRate
    return int(seconds * khz * 1000)


def _spin_then(op, c, stream, cycles, *, sync):
    """A Warp spin writes the sentinel input behind ~cycles; the eagle call
    follows on the same Warp stream with NO sync (or, for the known-answer
    variant, a host sync). Returns (output array, seconds the call took)."""
    dev = "cuda:0"
    c["v"].zero_()
    with wp.ScopedStream(stream):
        wp.launch(spin_write, dim=c["v"].shape, inputs=[c["v"], cycles, 1.0], device=dev)
        t0 = time.perf_counter()
        out = op(c["v"], c["cd"], c["t"])
        if sync:
            wp.synchronize_stream(stream)
        took = time.perf_counter() - t0
    return out, took


@pytest.mark.gpu
def test_a_call_rides_warps_stream_and_never_syncs(op):
    _skip_without("cuda:0")
    dev = "cuda:0"
    c = _case(dev, "float64", grad=False)
    op(c["v"], c["cd"], c["t"])                        # build and load outside the timing
    cycles = _spin_cycles(0.25)
    stream = wp.Stream(dev)
    expect = _law(np.ones((3, 7)), 0.3, c["term_np"])

    t0 = time.perf_counter()
    out, took = _spin_then(op, c, stream, cycles, sync=False)
    wp.synchronize_stream(stream)
    wall = time.perf_counter() - t0
    np.testing.assert_allclose(out.numpy(), expect, rtol=1e-13, atol=0)
    assert wall > 0.2, "the spin did not run; the instrument is blind"
    assert took < 0.05 * wall, f"the call blocked the host ({took:.3f}s of {wall:.3f}s)"

    # known answers of the same instrument: a host-synced call reads the whole spin ...
    _, synced = _spin_then(op, c, stream, cycles, sync=True)
    assert synced > 0.5 * wall
    # ... and the same call on ANOTHER Warp stream is unordered after the spin: stale input
    other = wp.Stream(dev)
    c["v"].zero_()
    with wp.ScopedStream(stream):
        wp.launch(spin_write, dim=c["v"].shape, inputs=[c["v"], cycles, 1.0], device=dev)
    with wp.ScopedStream(other):
        stale = op(c["v"], c["cd"], c["t"])
    wp.synchronize_stream(other)
    assert not stale.numpy().any()
    wp.synchronize_stream(stream)


# --------------------------------------------------------------------------- #
# threads
# --------------------------------------------------------------------------- #
def _in_threads(count, work, dev="cpu"):
    # Warp creates a device's CUDA context lazily and not thread-safely: touch it first
    wp.zeros(1, device=dev)
    barrier = threading.Barrier(count)
    errors, results = [], [None] * count

    def run(i):
        try:
            barrier.wait()
            results[i] = work(i)
        except BaseException:  # noqa: BLE001 - surfaced below
            errors.append(traceback.format_exc())

    # Warp's current stream is per device, not per thread: workers must not change it
    stream = wp.get_device(dev).stream if dev != "cpu" else None
    threads = [threading.Thread(target=run, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, "\n".join(errors)
    assert dev == "cpu" or wp.get_device(dev).stream is stream, "a worker left the device on another stream"
    return results


def _grad_of_sum_w(fn, c, dev):
    """Forward and backward through an explicit tape (no global tape, so
    threads do not contend), the output adjoint seeded by hand."""
    tape = wp.Tape()
    a = fn(c["v"], c["cd"], c["t"], tape=tape)
    a.grad.assign(c["w"])
    tape.backward()
    wp.synchronize_device(dev)
    return a, c["v"].grad, c["cd"].grad


@pytest.mark.ft
@pytest.mark.parametrize("dev", DEVICES)
def test_the_first_call_from_eight_threads_builds_each_level_once(tmp_path, dev):
    _skip_without(dev)
    fresh = eagle_warp.function(drag, cache_dir=tmp_path)
    cases = [_case(dev, "float64", seed=i) for i in range(8)]
    results = _in_threads(8, lambda i: _grad_of_sum_w(fresh, cases[i], dev), dev)
    for c, (a, gv, gc) in zip(cases, results):
        a, gv, gc = a.numpy(), gv.numpy(), gc.numpy()   # read back on this thread
        ref = _law(c["v_np"], 0.3, c["term_np"])
        np.testing.assert_allclose(a, ref, rtol=1e-13, atol=0)
        want_v, want_c = _fd_grads(c)
        np.testing.assert_allclose(gv, want_v, rtol=1e-7, atol=1e-7 * np.abs(want_v).max())
        np.testing.assert_allclose(gc[0], want_c, rtol=1e-7)


@pytest.mark.ft
@pytest.mark.parametrize("dev", DEVICES)
def test_twenty_calls_each_from_eight_threads(op, dev):
    _skip_without(dev)

    def work(i):
        runs = []
        for j in range(20):
            c = _case(dev, "float64", seed=1000 * i + j, grad=False)
            runs.append((c, op(c["v"], c["cd"], c["t"])))
        wp.synchronize_device(dev)
        return runs

    for runs in _in_threads(8, work, dev):             # read back on this thread
        for c, a in runs:
            np.testing.assert_allclose(a.numpy(), _law(c["v_np"], 0.3, c["term_np"]), rtol=1e-13, atol=0)


@pytest.mark.parametrize("dev", DEVICES)
def test_a_finishing_kernel_without_steps_one_runs(tmp_path, dev):
    _skip_without(dev)
    fn = eagle_warp.function(finish_plain, cache_dir=tmp_path)
    k = wp.array(np.arange(4.0) + 0.5, dtype=wp.float64, device=dev)
    nstop = wp.array([3.0], dtype=wp.float64, device=dev)
    mask = wp.zeros(4, dtype=wp.bool, device=dev)
    k_next, done = fn(k, nstop, mask)
    assert k_next.numpy().tolist() == [1.5, 2.5, 3.5, 4.5]
    assert done.numpy().tolist() == [False, False, True, True]


# --------------------------------------------------------------------------- #
# carried state: Mutable planes the kernel reads, then writes
# --------------------------------------------------------------------------- #
_N, _STEPS, _RATE, _NSTOP = 6, 6, 0.1, 4.0
_K0 = np.linspace(0.2, 0.7, _N)
_N0 = np.arange(_N, dtype=np.float64)
_M0 = np.array([False, True, False, False, False, False])   # sample 1 never steps


def _grow_np(k0, rate, steps=_STEPS):
    """The loop in NumPy (float64): final state, and each step's active samples."""
    k, n, m = k0.astype(np.float64).copy(), _N0.copy(), _M0.copy()
    trace = []
    for _ in range(steps):
        trace.append((k.copy(), ~m))
        k = np.where(m, k, k + rate * k * k)
        n = np.where(m, n, n + 1.0)
        m = m | (n >= _NSTOP)
    return k, n, m, trace


def _grow_grads(w, rate=_RATE):
    """d(sum w*k_final)/d(k0, rate) by a hand-written reverse sweep."""
    _, _, _, trace = _grow_np(_K0, rate)
    bar, g_rate = w.astype(np.float64).copy(), 0.0
    for k, active in reversed(trace):
        g_rate += (bar * k * k)[active].sum()
        bar = np.where(active, bar * (1 + 2 * rate * k), bar)
    return bar, g_rate


def _grow_fd(w, h=1e-6):
    loss = lambda k0, r: (w * _grow_np(k0, r)[0]).sum()
    gk = np.empty(_N)
    for j in range(_N):
        up, dn = _K0.copy(), _K0.copy()
        up[j] += h
        dn[j] -= h
        gk[j] = (loss(up, _RATE) - loss(dn, _RATE)) / (2 * h)
    return gk, (loss(_K0, _RATE + h) - loss(_K0, _RATE - h)) / (2 * h)


def _grow_arrays(dev, mode, grad=True):
    wdt, ndt = MODES[mode]
    return dict(
        k=wp.array(_K0, dtype=wdt, device=dev, requires_grad=grad),
        n=wp.array(_N0, dtype=wdt, device=dev),
        rate=wp.array([_RATE], dtype=wdt, device=dev, requires_grad=grad),
        nstop=wp.array([_NSTOP], dtype=wdt, device=dev),
        m=wp.array(_M0, dtype=wp.bool, device=dev))


@wp.kernel
def _weighted_sum_1d(a: wp.array(dtype=Any), w: wp.array(dtype=Any), loss: wp.array(dtype=Any)):
    i = wp.tid()
    wp.atomic_add(loss, 0, a[i] * w[i])


@pytest.fixture(scope="module")
def grow_fn(tmp_path_factory):
    return eagle_warp.function(grow, cache_dir=tmp_path_factory.mktemp("warp_grow"))


def _loop(fn, c, steps=_STEPS):
    k, n, m = c["k"], c["n"], c["m"]
    for _ in range(steps):
        k, n, m = fn(k, n, c["nstop"], c["rate"], m)
    return k, n, m


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dev", DEVICES)
def test_a_state_kernel_keeps_unstepped_samples_over_a_loop(grow_fn, dev, mode):
    _skip_without(dev)
    c = _grow_arrays(dev, mode, grad=False)
    assert grow_fn.inputs == ("k", "n", "nstop", "rate", "terminated")
    one_k, _, _ = grow_fn(c["k"], c["n"], c["nstop"], c["rate"], c["m"])
    assert c["k"].numpy().tolist() == pytest.approx(_K0.tolist(), rel=1e-6)   # the input is not overwritten
    k, n, m = _loop(grow_fn, c)
    want_k, want_n, want_m, _ = _grow_np(_K0, _RATE)
    rtol = 1e-5 if mode == "float32" else 1e-13
    np.testing.assert_allclose(k.numpy(), want_k, rtol=rtol, atol=0)
    np.testing.assert_array_equal(n.numpy(), want_n)
    np.testing.assert_array_equal(m.numpy(), want_m)
    assert n.numpy()[1] == 1.0 and k.numpy()[1] == pytest.approx(_K0[1], rel=1e-6)   # sample 1 never stepped
    assert one_k.numpy().shape == (_N,)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("dev", DEVICES)
def test_tape_gradients_through_state_match_the_reverse_and_finite_differences(grow_fn, dev, mode):
    """Six steps, samples finishing at different steps, no ``where`` anywhere: a
    finished sample's gradient passes through its skipped steps unchanged."""
    _skip_without(dev)
    w = np.random.default_rng(1).normal(size=_N)
    c = _grow_arrays(dev, mode)
    wdt = MODES[mode][0]
    loss = wp.zeros(1, dtype=wdt, device=dev, requires_grad=True)
    with wp.Tape() as tape:
        k, _, _ = _loop(grow_fn, c)
        wp.launch(_weighted_sum_1d, dim=_N, inputs=[k, wp.array(w, dtype=wdt, device=dev)],
                  outputs=[loss], device=dev)
    tape.backward(loss=loss)
    ref_k, ref_r = _grow_grads(w)
    fd_k, fd_r = _grow_fd(w)
    np.testing.assert_allclose(ref_k, fd_k, rtol=1e-7)                 # the references agree
    assert ref_r == pytest.approx(fd_r, rel=1e-7)
    rtol = GRAD_RTOL[mode]
    np.testing.assert_allclose(c["k"].grad.numpy(), ref_k, rtol=rtol, atol=rtol * np.abs(ref_k).max())
    np.testing.assert_allclose(c["rate"].grad.numpy()[0], ref_r, rtol=rtol)
    assert c["k"].grad.numpy()[1] == pytest.approx(w[1], rel=1e-6)     # never stepped: the identity


def test_stepping_a_state_plane_in_place_is_allowed_forward_and_refused_under_a_tape(grow_fn):
    c = _grow_arrays("cpu", "float64", grad=False)
    k, n = c["k"], c["n"]
    grow_fn(k, n, c["nstop"], c["rate"], c["m"], out=(k, n, wp.zeros(_N, dtype=wp.bool, device="cpu")))
    np.testing.assert_allclose(k.numpy(), np.where(_M0, _K0, _K0 + _RATE * _K0 ** 2), rtol=1e-15)
    c = _grow_arrays("cpu", "float64")
    with pytest.raises(ValueError, match=r"out\['k'\] is the input array.*gradient needs"):
        grow_fn(c["k"], c["n"], c["nstop"], c["rate"], c["m"],
                out=(c["k"], c["n"], wp.zeros(_N, dtype=wp.bool, device="cpu")), tape=wp.Tape())


def test_a_warp_without_the_private_tape_attribute_fails_by_name(op, monkeypatch):
    """The active tape is read off ``wp._src.context.runtime``; if that moves, the call says so."""
    c = _case("cpu", "float64")
    monkeypatch.delattr(wp._src.context.runtime, "tape")
    with pytest.raises(RuntimeError, match=r"runtime\.tape.*tape="):
        op(c["v"], c["cd"], c["t"])
    op(c["v"], c["cd"], c["t"], tape=wp.Tape())             # an explicit tape needs no lookup
