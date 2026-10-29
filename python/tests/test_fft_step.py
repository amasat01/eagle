# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``eagle.fft``: batched C2C / R2C / C2R graph steps and their derivatives, host and device.

Values against ``numpy.fft`` (the oracle; eagle itself uses no numpy), vjp and jvp against central differences, one
captured graph holding a forward and its vjp replayed three times bit for bit, a plan built inside a capture refused,
and the missing-cuFFT error."""

import numpy as np
import pytest

import eagle
import eagle.fft as F
from eagle import _stream, arrays

aether = pytest.importorskip("aether")
if not hasattr(pytest.importorskip("aether.fft"), "plan"):
    pytest.skip("needs aether-dsc >= 0.7.2 (aether.fft)", allow_module_level=True)

DEVICES = ["cpu", pytest.param("cuda:0", marks=pytest.mark.gpu)]
DTYPES = ["float32", "float64"]


def pairs(c, dtype):
    return np.ascontiguousarray(np.stack([c.real, c.imag], -1).astype(dtype))


def cplx(p):
    p = np.asarray(p, np.float64)
    return p[..., 0] + 1j * p[..., 1]


def rand_c(shape, dtype, seed=0):
    r = np.random.default_rng(seed)
    return pairs(r.uniform(-1, 1, shape) + 1j * r.uniform(-1, 1, shape), dtype)


def put(x, device, dtype=None):
    return arrays.asarray(np.ascontiguousarray(x), dtype, device)


def get(a):
    eagle._alloc.synchronize()
    return np.asarray(a.get() if a.device != "cpu" else a)


def run(node, device):
    """One eager run of ``node`` on the eagle graph (built and launched)."""
    g = eagle.Graph().add_node(node).build()
    g.launch()
    return g


def tol(dtype, n=64):
    return (8 + 4 * np.log2(n + 1)) * np.finfo(dtype).eps


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,axes", [((3, 8), (1,)), ((2, 9), (1,)), ((4, 6, 5), (1, 2)), ((7,), (0,))])
def test_c2c_values_equal_numpy_forward_and_inverse(device, dtype, shape, axes):
    x = rand_c(shape, dtype)
    for forward in (True, False):
        out = arrays.zeros(shape + (2,), dtype, device)
        run(F.c2c(put(x, device), out, axes, forward=forward), device)
        want = (np.fft.fftn if forward else np.fft.ifftn)(cplx(x), axes=axes)
        n = int(np.prod([shape[a] for a in axes]))
        err = np.abs(cplx(get(out)) - want).max()
        assert err <= tol(dtype, n) * np.linalg.norm(cplx(x)) * (1 if forward else 1 / n)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,axes", [((3, 8), (1,)), ((2, 9), (1,)), ((4, 6, 5), (1, 2))])
def test_r2c_and_c2r_values_equal_numpy(device, dtype, shape, axes):
    r = np.random.default_rng(1).uniform(-1, 1, shape).astype(dtype)
    half = list(shape)
    half[axes[-1]] = shape[axes[-1]] // 2 + 1
    spec = arrays.zeros(tuple(half) + (2,), dtype, device)
    run(F.r2c(put(r, device), spec, axes), device)
    n = int(np.prod([shape[a] for a in axes]))
    assert np.abs(cplx(get(spec)) - np.fft.rfftn(r.astype(np.float64), axes=axes)).max() <= tol(dtype, n) * np.linalg.norm(r)
    back = arrays.zeros(shape, dtype, device)
    run(F.c2r(spec, back, axes), device)
    assert np.abs(get(back) - r).max() <= 4 * tol(dtype, n) * np.linalg.norm(r)


def _fd(f, x, seed, h=1e-6):
    """Central differences of ``<seed, f(x)>`` along each entry of x."""
    g = np.zeros_like(x, dtype=np.float64)
    for idx in np.ndindex(x.shape):
        up, dn = x.copy(), x.copy()
        up[idx] += h
        dn[idx] -= h
        g[idx] = (np.sum(seed * f(up)) - np.sum(seed * f(dn))) / (2 * h)
    return g


def _apply(make, x, out_shape, device):
    out = arrays.zeros(out_shape, "float64", device)
    run(make(put(x, device), out), device)
    return get(out).copy()


CASES = [
    ("c2c_fwd", lambda x, o: F.c2c(x, o, (1,)), (2, 6, 2), (2, 6, 2)),
    ("c2c_inv_ortho", lambda x, o: F.c2c(x, o, (1,), forward=False, norm="ortho"), (2, 5, 2), (2, 5, 2)),
    ("c2c_2d_forward_norm", lambda x, o: F.c2c(x, o, (0, 1), norm="forward"), (3, 4, 2), (3, 4, 2)),
    ("r2c_even", lambda x, o: F.r2c(x, o, (1,)), (2, 8), (2, 5, 2)),
    ("r2c_odd", lambda x, o: F.r2c(x, o, (1,)), (2, 9), (2, 5, 2)),
    ("r2c_2d", lambda x, o: F.r2c(x, o, (0, 1)), (3, 6), (3, 4, 2)),
    ("c2r_even", lambda x, o: F.c2r(x, o, (1,)), (2, 5, 2), (2, 8)),
    ("c2r_odd", lambda x, o: F.c2r(x, o, (1,)), (2, 5, 2), (2, 9)),
]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,make,in_shape,out_shape", CASES, ids=[c[0] for c in CASES])
def test_vjp_and_jvp_equal_central_differences(device, name, make, in_shape, out_shape):
    rng = np.random.default_rng(5)
    x = rng.uniform(-1, 1, in_shape)
    if name.startswith("c2r"):                   # c2r reads only the Hermitian part: keep DC and Nyquist imaginary parts 0
        x[..., 0, 1] = 0.0
        if out_shape[-1] % 2 == 0:
            x[..., -1, 1] = 0.0
    seed = rng.uniform(-1, 1, out_shape)
    f = lambda v: _apply(make, v, out_shape, device)             # noqa: E731
    # vjp: the gradient of <seed, f(x)> is vjp(f)(seed)
    node = make(put(x, device), arrays.zeros(out_shape, "float64", device))
    bar = arrays.zeros(in_shape, "float64", device)
    run(F.vjp(node, put(seed, device), bar), device)
    want = _fd(f, x, seed)
    if name.startswith("c2r"):                   # the imaginary parts of DC / Nyquist are not read: no gradient there
        want[..., 0, 1] = 0.0
        if out_shape[-1] % 2 == 0:
            want[..., -1, 1] = 0.0
    np.testing.assert_allclose(get(bar), want, rtol=1e-6, atol=1e-7)
    # jvp: the directional derivative along t is f(t) (the map is linear)
    t = rng.uniform(-1, 1, in_shape)
    jv = arrays.zeros(out_shape, "float64", device)
    run(F.jvp(node, put(t, device), jv), device)
    h = 1e-6
    np.testing.assert_allclose(get(jv), (f(x + h * t) - f(x - h * t)) / (2 * h), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
def test_a_forward_and_its_vjp_in_one_captured_graph_replay_bitwise(device):
    shape = (3, 8)
    x = rand_c(shape, "float64")
    seed = rand_c(shape, "float64", 9)
    y, bar = arrays.zeros(shape + (2,), "float64", device), arrays.zeros(shape + (2,), "float64", device)
    node = F.c2c(put(x, device), y, (1,))
    g = eagle.Graph().add_node(node).add_node(F.vjp(node, put(seed, device), bar)).build()
    g.launch()
    first = (get(y).copy(), get(bar).copy())
    for _ in range(3):
        bar.fill(0), y.fill(0)
        g.launch()
        assert get(y).tobytes() == first[0].tobytes() and get(bar).tobytes() == first[1].tobytes()
    want = np.fft.ifft(cplx(seed), axis=1) * shape[1]                    # the unscaled inverse
    np.testing.assert_allclose(cplx(first[1]), want, rtol=1e-12, atol=1e-12)


@pytest.mark.gpu
def test_a_plan_built_inside_a_capture_is_refused_by_name():
    from eagle import _core

    x, out = arrays.zeros((2, 8, 2), "float64", "cuda:0"), arrays.zeros((2, 8, 2), "float64", "cuda:0")
    stream = _core.Stream(non_blocking=True)
    capturer = _core.StreamCapturer(stream.ptr())
    with _stream.scope(stream.ptr()):
        capturer.begin()
        try:
            with pytest.raises(RuntimeError, match=r"eagle.fft: a plan is built outside a stream capture"):
                F.c2c(x, out, (1,))
        finally:
            capturer.end()


def test_a_complex_array_is_refused_by_name():
    with pytest.raises(TypeError, match=r"pass a float64 view of \(re, im\) pairs"):
        F.c2c(np.zeros(4, np.complex128), np.zeros(4, np.complex128))


@pytest.mark.gpu
def test_a_device_transform_without_cufft_names_the_extra(monkeypatch):
    from aether import fft as aether_fft

    def missing(*a, **k):
        raise aether.FftUnavailable("cuFFT is not installed: pip install 'aether-dsc[cufft12]'")

    monkeypatch.setattr(aether_fft, "plan", missing)
    x, out = arrays.zeros((2, 8, 2), "float64", "cuda:0"), arrays.zeros((2, 8, 2), "float64", "cuda:0")
    with pytest.raises(aether.FftUnavailable, match=r"aether-dsc\[cufft12\]"):
        F.c2c(x, out, (1,))


@pytest.mark.parametrize("device", DEVICES)
def test_eagle_diff_reverse_pulls_a_transform_back_through_the_same_rule(device):
    x = np.random.default_rng(1).uniform(-1, 1, (2, 6, 2))
    seed = np.random.default_rng(2).uniform(-1, 1, (2, 6, 2))
    y = arrays.zeros((2, 6, 2), "float64", device)
    g = eagle.Graph().add_node(F.c2c(put(x, device), y, (1,)), name="f").build(record=True)
    g.launch()
    grads = eagle.diff.Reverse(g)(g.record, {"f.out": put(seed, device)})
    np.testing.assert_allclose(cplx(get(grads["x"])), np.fft.ifft(cplx(seed), axis=1) * 6, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("device", DEVICES)
def test_framework_arrays_in_and_out_equal_the_eagle_array_path_and_write_in_place(device):
    x = rand_c((3, 8), "float64")
    want = arrays.zeros((3, 8, 2), "float64", device)
    run(F.c2c(put(x, device), want, (1,)), device)
    if device == "cpu":
        xin, out = x.copy(), np.zeros((3, 8, 2))
        host = out
    else:
        cp = pytest.importorskip("cupy")
        xin, out = cp.asarray(x), cp.zeros((3, 8, 2))
        host = None
    run(F.c2c(xin, out, (1,)), device)
    eagle._alloc.synchronize()
    got = out if device == "cpu" else out.get()
    assert np.array_equal(got, get(want))            # bitwise, and the caller's own buffer holds it
    assert host is None or host is out


def test_a_read_only_numpy_out_is_refused_by_name():
    out = np.zeros((3, 8, 2))
    out.flags.writeable = False
    with pytest.raises(ValueError, match=r"eagle\.fft: out is read-only"):
        F.c2c(rand_c((3, 8), "float64"), out, (1,))


def test_a_complex_numpy_array_is_still_refused_by_name():
    with pytest.raises(TypeError, match="complex array"):
        F.c2c(np.zeros((3, 8), np.complex128), np.zeros((3, 8, 2)), (1,))
