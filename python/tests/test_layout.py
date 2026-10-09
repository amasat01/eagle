# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Sample-major ``(N, w)`` planes beside the native component-major ``(w, N)``.

The rules live in :mod:`eagle._layout`; every eagle door applies them through
:func:`eagle._layout.adapt`. What each group of rows proves:

* CLASSIFY: the decision table (native / zero-copy view / copy / ambiguous),
  including the ambiguous ``N == w`` case (refused, resolved by ``layout=`` or a
  marker), scalar planes and matrices; and that HAWK's
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
    assert _cls(np.zeros((W, W))) == "ambiguous"     # both readings: refused
    assert _cls(np.zeros(N), (1,)) == "native"       # scalar plane
    assert _cls(np.zeros((N, W)), ()) == "native"    # nothing declared
    assert _cls(np.zeros((2, N))) == "native"        # wrong width: left to the check
    m = np.zeros((3, 3, N))
    assert _cls(m, (3, 3)) == "native"
    assert _cls(np.zeros((3, 3, 3)), (3, 3)) == "ambiguous"   # N == R == C
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
    ((3,), 3, (3, 3), "ambiguous"),      # a batch of w: both readings, refused
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


# --------------------------------------------------------------------------- #
# AMBIGUOUS (w, w) planes, per-call layout= and per-array markers
# --------------------------------------------------------------------------- #
def _sm(n):
    """Sample-major (n, 3) data."""
    return np.arange(W * n, dtype=np.float64).reshape(n, W) * 0.5 + 1.0


def test_classify_resolves_an_ambiguous_plane_by_marker_and_by_layout():
    sq = np.zeros((W, W))
    st = [s // sq.itemsize for s in sq.strides]
    assert _layout.classify(sq.shape, st, (W,), axis="last") == "native"
    assert _layout.classify(sq.shape, st, (W,), axis="first") == "copy"
    assert _layout.classify(sq.shape, st, (W,), layout="samples_last") == "native"
    assert _layout.classify(sq.shape, st, (W,), layout="samples_first") == "copy"
    ft = np.asfortranarray(sq)
    assert _layout.classify(ft.shape, [s // 8 for s in ft.strides], (W,),
                            axis="first") == "view"
    # a marker the shape contradicts; and a marker beats the call's layout
    c = np.zeros((W, N))
    cs = [s // 8 for s in c.strides]
    assert _layout.classify(c.shape, cs, (W,), axis="first") == "conflict"
    assert _layout.classify(c.T.shape, [s // 8 for s in c.T.strides], (W,),
                            axis="last") == "conflict"
    assert _layout.classify(sq.shape, st, (W,), axis="last",
                            layout="samples_first") == "native"


def test_hawk_and_eagle_agree_on_the_ambiguous_cases_and_markers():
    runtime = pytest.importorskip("hawk.runtime")
    for a in (np.zeros((W, W)), np.zeros((W, W)).T, np.asfortranarray(np.zeros((W, W)))):
        st = [s // 8 for s in a.strides]
        for axis in (None, "first", "last"):
            for layout in (None, "samples_first", "samples_last"):
                assert _layout.classify(a.shape, st, (W,), axis, layout) == \
                    runtime.plane_layout(memoryview(a), W, axis, layout), \
                    (a.strides, axis, layout)
    for a in (np.zeros((W, N)), np.zeros((W, N)).T, np.zeros((N, W))):
        st = [s // 8 for s in a.strides]
        for axis in (None, "first", "last"):
            assert _layout.classify(a.shape, st, (W,), axis) == \
                runtime.plane_layout(memoryview(a), W, axis)


def test_the_marker_protocol_is_shared_with_hawk():
    hawk = pytest.importorskip("hawk")
    x = np.zeros((W, W))
    for mine, theirs, axis in ((eagle.samples_first, hawk.samples_first, "first"),
                               (eagle.samples_last, hawk.samples_last, "last")):
        a, b = mine(x), theirs(x)
        assert a.__raptor_samples_axis__ == b.__raptor_samples_axis__ == axis
        assert a.array is b.array is x
        assert _layout.split_marks({"x": b})[1] == {"x": axis}     # eagle takes hawk's


def test_run_refuses_an_ambiguous_plane_naming_both_fixes(vec3_unit):
    p = _host_plan(vec3_unit)
    x = np.ascontiguousarray(_sm(W))
    with pytest.raises(ValueError) as err:
        p.run(x=x, a=2.0)
    msg = str(err.value)
    for needle in ("'x'", "(3, 3)", "component-major (3, N)", "sample-major (N, 3)",
                   'layout="samples_first"', 'layout="samples_last"',
                   "eagle.samples_first(x)", "eagle.samples_last(x)"):
        assert needle in msg, (needle, msg)


def test_run_layout_samples_first_matches_the_unambiguous_n4_run(vec3_unit):
    p = _host_plan(vec3_unit)
    sm4 = _sm(4)
    ref = p.run(x=np.asfortranarray(sm4), a=2.0)            # (N, 3) view: unambiguous
    ref = np.asarray(ref)
    with pytest.warns(eagle.LayoutWarning):                 # C-contiguous: a copy
        got = p.run(x=np.ascontiguousarray(sm4[:3]), a=2.0, layout="samples_first")
    np.testing.assert_array_equal(got, ref[:, :3])


def test_run_layout_samples_last_binds_as_native(vec3_unit):
    p = _host_plan(vec3_unit)
    x = _x(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = p.run(x=x, a=2.0, layout="samples_last")
    np.testing.assert_array_equal(y, 2.0 * x)


def test_run_marker_samples_first_is_zero_copy(vec3_unit):
    p = _host_plan(vec3_unit)
    sm4 = _sm(4)
    ref = np.asarray(p.run(x=np.asfortranarray(sm4), a=2.0))
    xf = np.asfortranarray(sm4[:3])
    y = np.zeros((W, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")                      # no LayoutWarning: a view
        got = p.run(x=eagle.samples_first(xf), y=eagle.samples_first(y.T), a=2.0)
    np.testing.assert_array_equal(y, ref[:, :3])
    assert np.shares_memory(got, y)


def test_bound_buffer_address_is_the_inputs_for_a_marked_plane(vec3_unit):
    import ctypes

    p = _host_plan(vec3_unit)
    xf = np.asfortranarray(_sm(3))
    y = np.zeros((W, 3))
    b = p.bind(x=eagle.samples_first(xf), y=eagle.samples_last(y), a=2.0)
    assert b._planes["x"].ctypes.data == xf.ctypes.data     # the caller's own bytes
    assert not b._copies


@pytest.mark.parametrize("marker,shape", [("samples_first", (W, N)),
                                          ("samples_last", (N, W))])
def test_a_marker_contradicting_the_shape_is_refused(vec3_unit, marker, shape):
    p = _host_plan(vec3_unit)
    x = getattr(eagle, marker)(np.ones(shape))
    with pytest.raises(ValueError, match=rf"'x'.*{marker}"):
        p.run(x=x, a=2.0)


def test_marker_beats_the_per_call_layout(vec3_unit):
    p = _host_plan(vec3_unit)
    x = _x(3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        y = p.run(x=eagle.samples_last(x), a=2.0, layout="samples_first")
    np.testing.assert_array_equal(y, 2.0 * x)


def test_a_bad_layout_value_is_refused(vec3_unit):
    with pytest.raises(ValueError, match="samples_first.*samples_last"):
        _host_plan(vec3_unit).run(x=_x(3), a=2.0, layout="rows")


def test_bind_and_rebind_take_layout_and_markers(vec3_unit):
    p = _host_plan(vec3_unit)
    x = _x(3)
    y = np.zeros((W, 3))
    with pytest.raises(ValueError, match="samples_first"):
        p.bind(x=x, y=y, a=2.0)
    b = p.bind(x=x, y=y, a=2.0, layout="samples_last")
    b.launch() if hasattr(b, "launch") else b()
    np.testing.assert_array_equal(y, 2.0 * x)
    x2 = _x(3) + 1.0
    with pytest.raises(ValueError, match="samples_first"):
        b.rebind(x=x2)
    b.rebind(x=eagle.samples_last(x2))


def test_matrix_head_with_n_equal_r_equal_c_is_refused():
    m = np.zeros((3, 3, 3))
    with pytest.raises(_layout.LayoutError, match=r"'m'.*\(3, 3, 3\).*layout="):
        _layout.adapt("m", m, (3, 3), writes=False)
    assert _layout.adapt("m", m, (3, 3), writes=False, axis="last") is None
    with pytest.warns(eagle.LayoutWarning):
        a = _layout.adapt("m", m, (3, 3), writes=False, layout="samples_first")
    assert a is not None and a.layout == "copy"


def test_hawks_markers_are_accepted_by_eagle(vec3_unit):
    hawk = pytest.importorskip("hawk")
    p = _host_plan(vec3_unit)
    x = _x(3)
    y = p.run(x=hawk.samples_last(x), a=2.0)
    np.testing.assert_array_equal(y, 2.0 * x)


@hawk_trace.kernel
def lay_step(x: hawk_trace.Vector[3], t_end: hawk_trace.Param,
             dt: hawk_trace.Param, terminated: hawk_trace.Terminated,
             y: hawk_trace.Mutable[hawk_trace.Vector[3]],
             t: hawk_trace.Mutable[hawk_trace.Scalar]):
    y = y + dt * x
    t1 = t + dt
    t = t1
    terminated = t1 >= t_end


def _simulate(x, y, **opts):
    return eagle.simulate(lay_step, x=x, y=y, t=np.zeros(3), t_end=3.0, dt=1.0,
                          max_steps=10, **opts)


def test_simulate_refuses_resolves_and_marks_with_a_hawk_kernel():
    sm4 = _sm(4)
    x4, y4 = np.asfortranarray(sm4), np.zeros((4, W), order="F")
    eagle.simulate(lay_step, x=x4, y=y4, t=np.zeros(4), t_end=3.0, dt=1.0,
                   max_steps=10)               # unambiguous (4, 3) views
    ref = y4[:3].copy()
    np.testing.assert_array_equal(ref, 3.0 * sm4[:3])
    x = np.ascontiguousarray(sm4[:3])
    with pytest.raises(ValueError, match=r"'x'.*samples_first"):
        _simulate(x, np.zeros((W, 3)))
    # per call: the (3, 3) C-contiguous state is native (samples_last)
    y = np.zeros((W, 3))
    xn = np.ascontiguousarray(sm4[:3].T)
    r = _simulate(xn, y, layout="samples_last")
    np.testing.assert_array_equal(y, 3.0 * sm4[:3].T)
    assert r.y is y
    # per array, zero-copy views of F-contiguous sample-major data
    xf = np.asfortranarray(sm4[:3])
    yn = np.zeros((W, 3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _simulate(eagle.samples_first(xf), eagle.samples_first(yn.T))
    np.testing.assert_array_equal(yn.T, ref)
    # a hawk marker is accepted, and beats the call's layout
    hawk = pytest.importorskip("hawk")
    y = np.zeros((W, 3))
    _simulate(hawk.samples_last(xn), hawk.samples_last(y), layout="samples_first")
    np.testing.assert_array_equal(y, 3.0 * sm4[:3].T)
