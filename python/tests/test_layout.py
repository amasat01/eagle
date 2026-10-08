# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Sample-major ``(N, w)`` planes beside the native component-major ``(w, N)``.

The rules live in :mod:`eagle._layout`; every eagle door applies them through
:func:`eagle._layout.adapt`. What each group of rows proves:

* CLASSIFY: the decision table (native / zero-copy view / copy), including the
  ambiguous ``N == w`` case, scalar planes and matrices; and that HAWK's
  restatement (``hawk.runtime.plane_layout``) agrees on every corpus array.
* PLAN.RUN: a sample-major input or output gives the native answer, the copy
  warns once per call site naming the argument, a written plane lands back in
  the caller's array, and a transposed view passes with no warning and no copy.
  Plus the regression row for the missing shape check: a wrong-width plane used
  to run silently and is now refused.
* PLAN.BIND: the copy is made once at bind, refreshed before every launch and
  copied back after, on host and device, and inside a captured graph replay.
* STRICT MODE: ``filterwarnings("error")`` refuses at bind, before any copy.
* TORCH: a sample-major input runs, its gradient comes back sample-major, and
  so does the output itself.
* MARSHAL (Loaded* doors): inputs and Mutables are adapted, and a copied
  Mutable is written back into the caller's device array.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import eagle
from eagle import _layout

hawk_trace = pytest.importorskip("hawk.trace")
hawk_artifact = pytest.importorskip("hawk.artifact")

from test_plan_bind_launch import (  # noqa: E402
    _bundle,
    _device_subject,
    _host_subject,
    e4_vec3_scale,
)

W, N = 3, 8


@pytest.fixture(autouse=True)
def _fresh_call_sites():
    _layout._reset_seen()
    yield
    _layout._reset_seen()


@pytest.fixture(scope="module")
def vec3_unit(tmp_path_factory):
    return _bundle(e4_vec3_scale, tmp_path_factory, "layout_vec3", ("cuda", "host"))


def _x(n=N):
    return np.arange(W * n, dtype=np.float64).reshape(W, n) * 0.25 + 1.0


def _host_plan(unit):
    import eagle.exec as eexec
    from eagle import plan as eplan

    return eplan.plan(_host_subject(unit, "e4_vec3_scale"), structure=eexec.HostTeam)


def _device_plan(unit):
    import eagle.exec as eexec
    from eagle import plan as eplan

    return eplan.plan(_device_subject(unit, "e4_vec3_scale"),
                      structure=eexec.DeviceKernel)


def _layout_warnings(record):
    return [w for w in record if issubclass(w.category, eagle.LayoutWarning)]


# --------------------------------------------------------------------------- #
# CLASSIFY
# --------------------------------------------------------------------------- #
def _cls(a, head=(W,)):
    return _layout.classify(a.shape, [s // a.itemsize for s in a.strides], head)


def test_classify_table():
    c = np.zeros((W, N))
    assert _cls(c) == "native"
    assert _cls(c.T) == "view"
    assert _cls(np.zeros((N, W))) == "copy"
    assert _cls(np.zeros((W, W))) == "native"        # ambiguous: native
    assert _cls(np.zeros(N), (1,)) == "native"       # scalar plane
    assert _cls(np.zeros((N, W)), ()) == "native"    # nothing declared
    assert _cls(np.zeros((2, N))) == "native"        # wrong width: left to the check
    m = np.zeros((3, 3, N))
    assert _cls(m, (3, 3)) == "native"
    assert _cls(np.zeros((N, 3, 3)), (3, 3)) == "copy"
    assert _cls(np.moveaxis(m, -1, 0), (3, 3)) == "view"


def test_hawk_and_eagle_classify_every_corpus_array_identically():
    """hawk restates the rule (it imports no eagle); the two must not drift."""
    runtime = pytest.importorskip("hawk.runtime")
    base = np.zeros((W, N))
    corpus = [base, base.T, np.zeros((N, W)), np.zeros((W, W)),
              np.zeros((W, W)).T, np.zeros((1, W)), np.zeros((W, 1)),
              np.zeros((N, 2 * W))[:, ::2], np.zeros((2 * W, N))[::2].T,
              np.zeros((2, N)), np.zeros(N)]
    for a in corpus:
        eagle_says = _cls(a)
        hawk_says = runtime.plane_layout(memoryview(a), W)
        assert eagle_says == hawk_says, (a.shape, a.strides, eagle_says, hawk_says)


#: ONE SAMPLE: ``(head, flat width, shape, expected)``. A shape that is both a
#: batch and one sample is a batch (native is never guessed away).
_SINGLE_CORPUS = [
    ((), 1, (), "single"),               # a 0-d array (a number's shape)
    ((), 1, (1,), "native"),             # a batch of one
    ((), 1, (N,), "native"),
    ((3,), 3, (3,), "single"),
    ((3,), 3, (3, 1), "native"),         # a batch of one
    ((3,), 3, (3, 3), "native"),         # a batch of w
    ((3,), 3, (1, 3), "view"),           # sample-major batch of one
    ((4,), 4, (4,), "single"),           # a quaternion
    ((4,), 4, (4, 1), "native"),
    ((4,), 4, (1, 4), "view"),
    ((2, 3), 6, (2, 3), "single"),       # a matrix as (r, c)
    ((2, 3), 6, (6,), "single"),         # ... or flat (r*c,)
    ((2, 3), 6, (6, 1), "native"),
    ((3, 1), 3, (3, 1), "native"),       # Matrix[r, 1] given (r, 1): a batch
]


@pytest.mark.parametrize("head,width,shape,expected", _SINGLE_CORPUS)
def test_hawk_and_eagle_classify_every_single_sample_shape_identically(
        head, width, shape, expected):
    """One sample is the plane's head; hawk restates the rule and must agree."""
    runtime = pytest.importorskip("hawk.runtime")
    a = np.zeros(shape)
    assert _layout.classify(a.shape, [s // a.itemsize for s in a.strides],
                            (width,)) == expected
    assert runtime.plane_layout(memoryview(a), width) == expected
    if len(head) == 2 and shape == head:
        assert _layout.classify(a.shape, (shape[1], 1), head) == expected


# --------------------------------------------------------------------------- #
# PLAN.RUN (host)
# --------------------------------------------------------------------------- #
def test_run_sample_major_input_is_copied_with_a_warning(vec3_unit):
    p = _host_plan(vec3_unit)
    x = _x()
    with pytest.warns(eagle.LayoutWarning,
                      match=r"(?s)'x'.*\(8, 3\).*\(3, N\).*read-only.*zero-copy"):
        y = p.run(x=np.ascontiguousarray(x.T), a=2.0)
    np.testing.assert_array_equal(y, 2.0 * x)


def test_run_sample_major_output_is_written_back_and_returned(vec3_unit):
    p = _host_plan(vec3_unit)
    x = _x()
    y = np.zeros((N, W))
    with pytest.warns(eagle.LayoutWarning, match=r"'y'.*copied back"):
        got = p.run(x=x, y=y, a=2.0)
    assert got is y
    np.testing.assert_array_equal(y, (2.0 * x).T)


def test_run_transposed_view_is_zero_copy_with_no_warning(vec3_unit):
    p = _host_plan(vec3_unit)
    x = _x()
    y_native = np.zeros((W, N))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = p.run(x=x.T, y=y_native.T, a=2.0)
    np.testing.assert_array_equal(y_native, 2.0 * x)
    assert np.shares_memory(got, y_native)


def test_run_refuses_a_wrong_width_plane(vec3_unit):
    """Regression: Plan.run checked no shape at all, so this ran silently."""
    p = _host_plan(vec3_unit)
    with pytest.raises(ValueError, match=r"'x' is declared 3 component"):
        p.run(x=np.ones((W + 1, N)), a=2.0)


def test_run_refuses_a_short_output_plane(vec3_unit):
    p = _host_plan(vec3_unit)
    with pytest.raises(ValueError, match=r"'y' is declared 3 component"):
        p.run(x=_x(), y=np.zeros((W, N - 1)), a=2.0)


def test_warning_fires_once_per_call_site():
    a = np.zeros((N, W))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for _ in range(3):
            _layout.adapt("x", a, (W,), writes=False)       # one site
        _layout.adapt("x", a, (W,), writes=False)           # another site
        _layout.adapt("z", a, (W,), writes=False)           # another argument
    hits = _layout_warnings(record)
    assert len(hits) == 3
    assert all(h.filename == __file__ for h in hits), "names the caller's line"


def test_strict_mode_refuses_at_bind_before_any_copy(vec3_unit):
    p = _host_plan(vec3_unit)
    y = np.zeros((N, W))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=eagle.LayoutWarning)
        for _ in range(2):      # strict on the second call at the same site too
            with pytest.raises(eagle.LayoutWarning, match="'y'"):
                p.bind(x=_x(), y=y, a=2.0)
    np.testing.assert_array_equal(y, 0.0)


# --------------------------------------------------------------------------- #
# PLAN.BIND (host)
# --------------------------------------------------------------------------- #
def test_bind_copies_refresh_before_and_write_back_after_each_launch(vec3_unit):
    p = _host_plan(vec3_unit)
    x = np.ascontiguousarray(_x().T)              # (N, 3), read-only role
    y = np.zeros((N, W))                          # (N, 3), written role
    with pytest.warns(eagle.LayoutWarning):
        bound = p.bind(x=x, y=y, a=2.0)
    assert bound.n == N
    bound.launch()
    np.testing.assert_array_equal(y, 2.0 * x)
    x *= 3.0                                      # the caller's array is live
    bound.launch()
    np.testing.assert_array_equal(y, 2.0 * x)


def test_rebind_adapts_the_new_plane(vec3_unit):
    p = _host_plan(vec3_unit)
    x = _x()
    y1 = np.zeros((W, N))
    bound = p.bind(x=x, y=y1, a=2.0)
    y2 = np.zeros((N, W))
    with pytest.warns(eagle.LayoutWarning, match="'y'"):
        bound.rebind(y=y2)
    bound.launch()
    np.testing.assert_array_equal(y2, (2.0 * x).T)
    np.testing.assert_array_equal(y1, 0.0)


# --------------------------------------------------------------------------- #
# DEVICE
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_device_run_with_sample_major_planes(vec3_unit):
    import cupy as cp

    p = _device_plan(vec3_unit)
    x = _x()
    y = np.zeros((N, W))
    with pytest.warns(eagle.LayoutWarning):
        got = p.run(x=np.ascontiguousarray(x.T), y=y, a=2.0)
    assert got is y
    np.testing.assert_array_equal(y, (2.0 * x).T)
    y_dev = cp.zeros((N, W))
    with pytest.warns(eagle.LayoutWarning):
        got = p.run(x=x, y=y_dev, a=2.0)
    np.testing.assert_array_equal(cp.asnumpy(y_dev), (2.0 * x).T)
    np.testing.assert_array_equal(got, (2.0 * x).T)


@pytest.mark.gpu
def test_device_bind_copies_ride_the_stream_and_replay_in_a_graph(vec3_unit):
    import cupy as cp

    p = _device_plan(vec3_unit)
    x = cp.asarray(np.ascontiguousarray(_x().T))
    y = cp.zeros((N, W))
    with pytest.warns(eagle.LayoutWarning):
        bound = p.bind(x=x, y=y, a=2.0)
    bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    np.testing.assert_array_equal(cp.asnumpy(y), 2.0 * cp.asnumpy(x))

    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        bound.launch(stream=stream)
        stream.synchronize()
        y.fill(0.0)
        stream.begin_capture()
        bound.launch()
        graph = stream.end_capture()
        x *= 3.0                                  # the replay re-reads the input
        graph.launch(stream=stream)
    stream.synchronize()
    np.testing.assert_array_equal(cp.asnumpy(y), 2.0 * cp.asnumpy(x))


@pytest.mark.gpu
def test_device_transposed_view_binds_zero_copy(vec3_unit):
    import cupy as cp

    p = _device_plan(vec3_unit)
    x = cp.asarray(_x())
    y_native = cp.zeros((W, N))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bound = p.bind(x=x.T, y=y_native.T, a=2.0)
    assert not bound._copies
    bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    np.testing.assert_array_equal(cp.asnumpy(y_native), 2.0 * cp.asnumpy(x))


@pytest.mark.gpu
def test_marshal_adapts_inputs_and_writes_mutables_back():
    import cupy as cp

    from eagle.marshal import coerce_mutable, coerce_vec_inputs

    x = _x()
    with pytest.warns(eagle.LayoutWarning, match="'v'"):
        out, n = coerce_vec_inputs({"v": np.ascontiguousarray(x.T)}, ["v"])
    assert n == N
    np.testing.assert_array_equal(cp.asnumpy(out["v"]), x)

    class Decl:
        dtype, width = "vector", W

    caller = cp.zeros((N, W))
    adapted = {}
    with pytest.warns(eagle.LayoutWarning, match="'m'"):
        arr, n = coerce_mutable("m", caller, Decl, adapted=adapted)
    assert n == N and arr.shape == (W, N)
    arr[...] = cp.asarray(x)                     # what a kernel would write
    adapted["m"].write_back_from(arr)
    np.testing.assert_array_equal(cp.asnumpy(caller), x.T)


# --------------------------------------------------------------------------- #
# TORCH
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def torch_op(tmp_path_factory):
    pytest.importorskip("torch")
    from eagle.frameworks import torch as eagle_torch

    return eagle_torch.function(e4_vec3_scale,
                                cache_dir=tmp_path_factory.mktemp("layout_torch"))


def test_torch_sample_major_input_and_its_gradient(torch_op):
    """A sample-major input's output comes back sample-major too."""
    import torch

    x = torch.tensor(_x())
    native = torch_op(x.clone().requires_grad_(), 2.0)

    xs = x.T.contiguous().requires_grad_()         # (N, 3), contiguous
    with pytest.warns(eagle.LayoutWarning, match="'x'"):
        y = torch_op(xs, 2.0)
    assert y.shape == (N, W)
    torch.testing.assert_close(y, native.detach().T)
    y.sum().backward()
    assert xs.grad.shape == (N, W)
    torch.testing.assert_close(xs.grad, torch.full((N, W), 2.0, dtype=torch.float64))


def test_torch_transposed_view_needs_no_copy(torch_op):
    """The zero-copy sample-major view's output is sample-major too,
    itself a zero-copy view, so strict mode still raises nothing."""
    import torch

    x = torch.tensor(_x())
    with warnings.catch_warnings():
        warnings.simplefilter("error", eagle.LayoutWarning)
        y = torch_op(x.T, 2.0)
    assert y.shape == (N, W)
    torch.testing.assert_close(y, (2.0 * x).T)
