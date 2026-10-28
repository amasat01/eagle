"""``eagle.Array`` / ``eagle.arrays``: views, dtypes (f16, bf16), interop, the no-CUDA and no-numpy builds, and the
capture rows (each op adds one kernel node, replays, and a synchronisation inside a capture fails)."""
from __future__ import annotations

import ast
import inspect
import os
import subprocess
import sys

import pytest

import eagle.arrays as xp
from eagle import _core

SITE = os.path.dirname(os.path.dirname(_core.__file__))
STUB = (
    "import sys, types, os\n"
    "sys.modules['numpy'] = sys.modules['cupy'] = None\n"
    "pkg = types.ModuleType('eagle'); pkg.__path__ = [%r]; sys.modules['eagle'] = pkg\n"
    "import eagle.arrays as xp\n" % os.path.join(SITE, "eagle")
)


def _run(body, **env):
    environ = dict(os.environ, **env)
    r = subprocess.run([sys.executable, "-I", "-u", "-c", STUB + body], capture_output=True, text=True, env=environ)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


HOST_ROWS = """
a = xp.arange(100000, dtype=xp.int64)
b = xp.add(a, a)
assert b.tolist()[-1] == 2 * 99999 and int(b.sum()) == 2 * (99999 * 100000 // 2)
m = xp.asarray([[1, 2, 3], [4, 5, 6]])
assert m.T.tolist() == [[1, 4], [2, 5], [3, 6]] and m[:, 1].tolist() == [2, 5] and m[..., 0].tolist() == [1, 4]
assert memoryview(m).tolist() == [[1, 2, 3], [4, 5, 6]] and memoryview(m.T).strides == (8, 24)
assert xp.asarray(memoryview(bytearray(16)).cast('d')).shape == (2,)
v = xp.asarray(m); assert v.ptr == m.ptr
assert _c.array_from_dlpack(m).ptr == m.ptr
assert xp.shares_memory(m, m[0]) and not xp.shares_memory(m, xp.zeros(3))
assert xp.float64 == 'float64' and xp.float64 == float and hash(xp.float64) == hash('float64')
z = xp.asarray([0.0, -0.0, float('nan'), 2.5] * 25000)
assert int(xp.count_nonzero(z)) == 50000
ix, cnt = xp.compact_indices(z)
assert int(cnt) == 50000 and ix[:3].tolist() == [2, 3, 6] and ix[49999].item() == 99999 and xp.flatnonzero(z).shape == (50000,)
assert float(xp.sum(xp.asarray([0.25] * 100000))) == 25000.0 and xp.sum(xp.asarray([1, 2, 3], xp.uint8)).dtype == 'int64'
print('rows ok')
"""


def test_host_rows_run_with_numpy_hidden_and_four_threads():
    out = _run("import importlib\n_c = importlib.import_module('eagle._core')\n" + HOST_ROWS, OMP_NUM_THREADS="4")
    assert "rows ok" in out


def test_no_cuda_build_imports_and_runs_the_host_rows_and_core_links_no_cudart():
    out = _run("import importlib\n_c = importlib.import_module('eagle._core')\n" + HOST_ROWS,
               CUDA_VISIBLE_DEVICES="", EAGLE_BACKEND_CUDA="/nonexistent/libeagle_cuda.so")
    assert "rows ok" in out
    so = _core.__file__
    assert "cudart" not in subprocess.run(["ldd", so], capture_output=True, text=True).stdout
    nm = subprocess.run(["nm", "-D", "--undefined-only", so], capture_output=True, text=True).stdout
    assert " cuda" not in nm and "cuMem" not in nm


def test_views_share_storage_and_refuse_what_cannot_be_viewed():
    a = xp.arange(12, dtype=xp.int32).reshape((3, 4))
    v = a[1:, ::2]
    assert v.shape == (2, 2) and v.strides == (16, 8) and v.tolist() == [[4, 6], [8, 10]]
    v[0, 0] = 99
    assert a[1, 0].item() == 99
    with pytest.raises(ValueError):
        v.reshape((4,))
    a[0] = [7, 7, 7, 7]
    assert a.tolist()[0] == [7, 7, 7, 7] and a.view(xp.uint8).shape == (3, 16)


def test_half_and_bfloat16_cast_round_to_nearest_even_and_do_arithmetic_on_the_host():
    f = xp.asarray([2049.0, 2051.0, 65519.0, 65520.0, 1 / 3], xp.float64)
    assert f.astype(xp.float16).tolist() == [2048.0, 2052.0, 65504.0, float("inf"), 0.333251953125]
    assert xp.asarray([1 / 3, 257.0], xp.float64).astype(xp.bfloat16).tolist() == [0.333984375, 256.0]
    h = xp.asarray([1.5, 2.5, 60000.0], xp.float16)
    assert xp.add(h, h).tolist() == [3.0, 5.0, float("inf")]
    assert xp.multiply(h[:2], h[:2]).tolist() == [2.25, 6.25] and xp.greater(h, h).tolist() == [False] * 3
    assert xp.linspace(0, 1, 5, dtype=xp.float16).tolist() == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert xp.add(h[:2], xp.asarray([1.0, 1.0], xp.float32)).dtype == "float32"
    assert str(xp.add(xp.asarray([1], xp.int32), h[:1]).dtype) == "float64"


def test_a_scalar_operand_on_either_side_matches_the_array_spelling_and_allocates_nothing():
    a = xp.asarray([3, -1, 2, 0, 5], xp.int32)
    three = xp.full_like(a, 3)
    for name in ("add", "subtract", "multiply", "maximum", "equal", "not_equal", "greater"):
        f = getattr(xp, name)
        assert f(a, 3).tolist() == f(a, three).tolist() and f(3, a).tolist() == f(three, a).tolist(), name
    assert (a < 3).tolist() == [False, True, True, True, False] and (3 > a).tolist() == (a < 3).tolist()
    assert xp.subtract(10, a).tolist() == [7, 11, 8, 10, 5] and xp.add(a, 2.5).dtype == "int32"
    f = xp.asarray([1.5, -0.0, float("nan")], xp.float64)
    assert xp.maximum(f, 0.0).tolist()[:2] == [1.5, 0.0] and xp.equal(f, 1.5).tolist() == [True, False, False]
    out = xp.zeros((5,), xp.int32)
    assert xp.add(a, 1, out=out) is out and out.tolist() == [4, 0, 3, 1, 6]


def test_complex_data_is_refused_by_name_with_the_pair_view_remedy():
    np = pytest.importorskip("numpy")
    for make in (lambda z: z, lambda z: z.__array__ if False else memoryview(z)):
        with pytest.raises(ValueError, match="complex128 is not an eagle dtype; pass a float64 view of"):
            xp.asarray(make(np.zeros(3, np.complex128)))
    with pytest.raises(ValueError, match="complex64 is not an eagle dtype; pass a float32 view of"):
        xp.asarray(np.zeros(3, np.complex64))


def test_the_host_cumsum_flatnonzero_take_scatter_where_clip_match_python():
    a = xp.asarray([3, 1, 2, 0, 5], xp.int32)
    assert xp.cumsum(a).tolist() == [3, 4, 6, 6, 11]
    idx = xp.flatnonzero(a)
    assert idx.tolist() == [0, 1, 2, 4] and xp.take(a, idx).tolist() == [3, 1, 2, 5]
    out = xp.zeros((5,), xp.int32)
    xp.scatter(out, xp.asarray([4, 3, 2, 1, 0], xp.int32), a)
    assert out.tolist() == [5, 0, 2, 1, 3]
    assert xp.where(xp.greater(a, xp.full_like(a, 1)), a, xp.zeros_like(a)).tolist() == [3, 0, 2, 0, 5]
    assert xp.clip(a, 1, 3).tolist() == [3, 1, 2, 1, 3] and int(xp.count_nonzero(a)) == 4
    assert xp.concatenate([a, a]).shape == (10,) and xp.stack([a, a]).shape == (2, 5) and xp.diff(a).tolist() == [-2, 1, -2, 5]
    assert xp.bincount(xp.asarray([0, 2, 2], xp.int32)).tolist() == [1, 0, 2] and xp.unique(a).tolist() == [0, 1, 2, 3, 5]


def test_array_ops_reachable_from_a_capture_hold_no_host_read_or_sync():
    banned = {"synchronize", "device_synchronize", "get", "item", "tolist"}
    for fn in (xp._binary, xp.where, xp.clip, xp.take, xp.scatter, xp.cumsum, xp.copyto, xp.arange, xp.linspace, xp.count_nonzero,
               xp.compact_indices, xp.sum):
        names = {n.attr for n in ast.walk(ast.parse(inspect.getsource(fn))) if isinstance(n, ast.Attribute)}
        assert not names & banned, fn.__name__


# ---- GPU rows -----------------------------------------------------------------------------------------

def _dot_kernel_nodes(captured, tag):
    path = f"/local/scratch/amasat/tmp/w072/scratch/S1a/tmp/arr_{tag}.dot"
    return captured.debug_dot(path, 0).count('shape="octagon"')


def _capture(s, enqueue):
    cap = _core.StreamCapturer(s.ptr())
    cap.begin()
    try:
        enqueue()
    finally:
        captured = cap.end()
    return captured


@pytest.fixture()
def stream():
    s = _core.Stream(non_blocking=True)
    yield s


def _to(arr, dev, st):
    out = xp.empty(arr.shape, arr.dtype, dev, st)
    _core.array_copyto(out, arr)
    _core.device_synchronize()
    return out


@pytest.mark.gpu
def test_device_arrays_copy_view_cast_and_round_trip_through_the_host(stream):
    dev, st = "cuda:0", stream.ptr()
    h = xp.arange(24, dtype=xp.float64).reshape((4, 6))
    d = _to(h, dev, st)
    assert d.device == dev and d.stream == st and d.get().tolist() == h.tolist()
    assert d[1:, ::2].get().tolist() == h[1:, ::2].tolist() and d.T.get().tolist() == h.T.tolist()
    d[1:3, 1:5] = 0.5                       # pitched device destination
    h[1:3, 1:5] = 0.5
    assert d.get().tolist() == h.tolist()
    assert d.astype(xp.float32).get().tolist() == h.astype(xp.float32).tolist()
    assert d.astype(xp.float16).get().tolist() == h.astype(xp.float16).tolist()
    assert float(d.sum()) == float(h.sum()) and d.sum().device == dev and xp.add(d, d).get().tolist() == xp.add(h, h).tolist()
    assert _core.pointer_device(d.ptr) == 0
    with pytest.raises(BufferError):
        memoryview(d)


@pytest.mark.gpu
def test_fill_setitem_and_copy_without_a_stream_follow_the_scope_into_a_capture(stream):
    from eagle import _stream
    a = xp.zeros((16,), xp.float32, "cuda:0")           # allocated on the default stream, not ``stream``
    b = xp.zeros((16,), xp.float32, "cuda:0")
    _core.device_synchronize()

    def enqueue():
        with _stream.scope(stream.ptr()):
            a.fill(7)
            b[...] = 5
            with pytest.raises(RuntimeError, match="allocation inside a stream capture"):
                a.copy()                         # a copy allocates: refused in a capture, on the scope stream outside one

    captured = _capture(stream, enqueue)
    _core.device_synchronize()
    assert a.get().tolist() == [0.0] * 16 and b.get().tolist() == [0.0] * 16   # nothing ran at build
    assert _dot_kernel_nodes(captured, "scope_default") == 2
    launcher = _core.Graph.from_captured(captured).launcher()
    launcher.stream(stream.ptr())
    launcher.launch()
    launcher.synchronize()
    assert a.get().tolist() == [7.0] * 16 and b.get().tolist() == [5.0] * 16
    with _stream.scope(stream.ptr()):
        assert a.copy().stream == stream.ptr() and a.astype(xp.float64).stream == stream.ptr()


@pytest.mark.gpu
def test_with_no_scope_an_op_uses_the_destination_arrays_own_stream(stream):
    """A non-blocking-stream array is ordered by its own stream, not the legacy one, when no scope is open."""
    st = stream.ptr()
    a = xp.zeros((1 << 20,), xp.float32, "cuda:0", st)
    b = xp.empty((1 << 20,), xp.float32, "cuda:0", st)
    for i in range(20):                                   # a chain: each op reads the previous result
        a.fill(float(i + 1))
        xp.add(a, a, out=b)
        xp.copyto(a, b)
    stream.synchronize()
    assert a.get().tolist()[:2] == [40.0, 40.0] and float(a.sum(st)) == 40.0 * (1 << 20)


@pytest.mark.gpu
def test_setitem_and_asarray_take_a_cai_or_dlpack_device_source(stream):
    class Cai:                                           # a CUDA-array-interface producer over an eagle array
        def __init__(self, a): self._a, self.__cuda_array_interface__ = a, a.__cuda_array_interface__
    class Dl:                                            # a DLPack producer: the consumer stream must arrive
        def __init__(self, a): self._a, self.seen = a, []
        def __dlpack_device__(self): return self._a.__dlpack_device__()
        def __dlpack__(self, stream=None, **kw): self.seen.append(stream); return self._a.__dlpack__(stream=stream, **kw)
    src = _to(xp.asarray([1.0, 2.0, 3.0, 4.0], xp.float64), "cuda:0", stream.ptr())
    dst = xp.zeros((4,), xp.float64, "cuda:0", stream.ptr())
    dst[...] = Cai(src)
    dl = Dl(src)
    dst2 = xp.zeros((4,), xp.float64, "cuda:0", stream.ptr())
    dst2[:] = dl
    assert dl.seen and dl.seen[0] is not None
    _core.device_synchronize()
    assert dst.get().tolist() == dst2.get().tolist() == [1.0, 2.0, 3.0, 4.0]
    assert xp.asarray(Cai(src)).ptr == src.ptr


@pytest.mark.gpu
def test_half_and_bfloat16_on_the_device_run_or_are_refused_by_name_below_their_hardware(stream):
    dev, st = "cuda:0", stream.ptr()
    major = _core.device_props(0)["cc_major"] if hasattr(_core, "device_props") else 6
    for dt, need in ((xp.float16, 5), (xp.bfloat16, 8)):
        d = _to(xp.asarray([1.5, 2.5], dt), dev, st)
        if major >= need:
            assert xp.add(d, d).get().tolist() == [3.0, 5.0]
        else:
            with pytest.raises(Exception, match="compute capability"):
                xp.add(d, d)
        assert d.astype(xp.float64).get().tolist() == [1.5, 2.5]
        assert xp.where(_to(xp.asarray([True, False], xp.bool_), dev, st), d, d).get().tolist() == [1.5, 2.5]


@pytest.mark.gpu
def test_dlpack_and_cai_round_trips_with_cupy_and_a_capture_inside_a_cupy_stream(stream):
    cp = pytest.importorskip("cupy")
    dev, st = "cuda:0", stream.ptr()
    d = _to(xp.arange(8, dtype=xp.float64), dev, st)
    _core.device_synchronize()
    c = cp.from_dlpack(d)
    assert int(c.data.ptr) == d.ptr and c.dtype == cp.float64
    assert int(cp.asarray(d).data.ptr) == d.ptr
    c[:] = 7.0
    cp.cuda.Device().synchronize()
    assert d.get().tolist() == [7.0] * 8
    back = _core.array_from_dlpack(c)
    assert back.ptr == d.ptr and _core.array_from_cai(c).ptr == d.ptr and back.device == "cuda:0"
    x, y, z = cp.full((8,), 10.0), cp.full((8,), 5.0), cp.zeros((8,))
    ex, ey, ez = xp.asarray(x), xp.asarray(y), xp.asarray(z)
    cs = cp.cuda.Stream(non_blocking=True)
    with cs:
        cs.begin_capture()
        _core.array_binary(_core.CELL_ADD, ez, ex, ey, int(cs.ptr))
        g = cs.end_capture()
    assert float(z[0]) == 0.0
    g.launch(stream=cs)
    cs.synchronize()
    assert bool((z == 15.0).all())


OPS_UNDER_CAPTURE = {
    "add": lambda o, a, b, i, sc, s: _core.array_binary(_core.CELL_ADD, o, a, b, s),
    "sub": lambda o, a, b, i, sc, s: _core.array_binary(_core.CELL_SUB, o, a, b, s),
    "mul": lambda o, a, b, i, sc, s: _core.array_binary(_core.CELL_MUL, o, a, b, s),
    "max": lambda o, a, b, i, sc, s: _core.array_binary(_core.CELL_MAX, o, a, b, s),
    "add_scalar": lambda o, a, b, i, sc, s: _core.array_binary_scalar(_core.CELL_ADDS, o, a, 3, s),
    "rsub_scalar": lambda o, a, b, i, sc, s: _core.array_binary_scalar(_core.CELL_RSUBS, o, a, 3, s),
    "clip": lambda o, a, b, i, sc, s: _core.array_clip(o, a, -2.0, 4.0, s),
    "take": lambda o, a, b, i, sc, s: _core.array_take(o, a, i, s),
    "scatter": lambda o, a, b, i, sc, s: _core.array_scatter(o, i, a, s),
    "cumsum": lambda o, a, b, i, sc, s: _core.array_cumsum(o, a, sc, s),
    "arange": lambda o, a, b, i, sc, s: _core.array_arange(o, 3, 2, s),
    "linspace": lambda o, a, b, i, sc, s: _core.array_linspace(o, 1.0, 0.5, s),
    "fill": lambda o, a, b, i, sc, s: o.fill(9, s),
    "astype": lambda o, a, b, i, sc, s: _core.array_copyto(o, a, s),
}


@pytest.mark.gpu
@pytest.mark.parametrize("name", sorted(OPS_UNDER_CAPTURE))
def test_each_binding_op_adds_one_kernel_node_replays_and_equals_the_host_twin(stream, name):
    dev, st = "cuda:0", stream.ptr()
    ha = xp.asarray([5, -3, 8, 1, 0, 7, -2, 4], xp.float64 if name in ("linspace", "astype") else xp.int32)
    hb = xp.asarray([1, 2, 3, 4, 5, 6, 7, 8], ha.dtype)
    hi = xp.asarray([7, 6, 5, 4, 3, 2, 1, 0], xp.int32)
    odt = xp.float32 if name == "astype" else (xp.float64 if name == "linspace" else ha.dtype)
    hout = xp.zeros((8,), odt)
    da, db, di, do = (_to(h, dev, st) for h in (ha, hb, hi, hout))
    scratch = xp.empty((xp.cumsum_scratch_bytes(8),), xp.uint8, dev, st)
    fn = OPS_UNDER_CAPTURE[name]
    host = {"add_scalar": lambda: xp.add(ha, xp.full_like(ha, 3)), "rsub_scalar": lambda: xp.subtract(xp.full_like(ha, 3), ha),
            "add": lambda: xp.add(ha, hb), "sub": lambda: xp.subtract(ha, hb), "mul": lambda: xp.multiply(ha, hb),
            "max": lambda: xp.maximum(ha, hb), "clip": lambda: xp.clip(ha, -2.0, 4.0), "take": lambda: xp.take(ha, hi),
            "scatter": None, "cumsum": lambda: xp.cumsum(ha), "arange": lambda: xp.arange(3, 19, 2, ha.dtype),
            "linspace": lambda: xp.linspace(1.0, 4.5, 8, xp.float64), "fill": lambda: xp.full((8,), 9, ha.dtype),
            "astype": lambda: ha.astype(xp.float32)}[name]
    if host is None:
        want = xp.zeros((8,), ha.dtype)
        xp.scatter(want, hi, ha)
    else:
        want = host()
    captured = _capture(stream, lambda: fn(do, da, db, di, scratch, st))
    assert _dot_kernel_nodes(captured, name) == 1
    launcher = _core.Graph.from_captured(captured).launcher()
    launcher.stream(st)
    for _ in range(3):
        launcher.launch()
    launcher.synchronize()
    assert do.get().tolist() == want.tolist()


@pytest.mark.gpu
def test_a_host_read_or_an_allocation_inside_a_capture_fails_by_name(stream):
    dev, st = "cuda:0", stream.ptr()
    d = _to(xp.arange(4, dtype=xp.int32), dev, st)
    cap = _core.StreamCapturer(st)
    cap.begin()
    try:
        with pytest.raises(Exception):
            d.get()
        with pytest.raises(Exception):
            xp.empty((4,), xp.int32, dev, st)
        with pytest.raises(Exception):
            d.sum(st)                     # allocates its result
        with pytest.raises(Exception):
            _core.stream_sync(st)         # the stream sync is not capture-legal either
        with pytest.raises(Exception):
            d.item()
    finally:
        try:
            cap.end()
        except Exception:
            pass


@pytest.mark.gpu
@pytest.mark.parametrize("name", ["count_nonzero", "compact_indices", "sum_f64", "sum_i32"])
def test_reduction_and_compaction_ops_capture_replay_and_equal_the_host_twin(stream, name):
    dev, st = "cuda:0", stream.ptr()
    values = [0.0, 2.0, float("nan"), -0.0, 4.0, 0.0, 1.5, 8.0] * 700            # 5600: past one scan tile
    values = [v if v == v else 3.0 for v in values] if name.startswith("sum") else values
    h = xp.asarray([int(v) for v in values] if name == "sum_i32" else values, xp.int32 if name == "sum_i32" else xp.float64)
    d = _to(h, dev, st)
    if name == "count_nonzero":
        out, want = xp.empty((), xp.int64, dev, st), xp.count_nonzero(h)
        run = lambda: _core.array_count_nonzero(out, d, st)
    elif name == "compact_indices":
        index, count = xp.empty((h.size,), xp.int32, dev, st), xp.empty((), xp.int64, dev, st)
        scratch = xp.empty((xp.cumsum_scratch_bytes(h.size),), xp.uint8, dev, st)
        want = xp.compact_indices(h)
        out = index
        run = lambda: _core.array_flatnonzero(index, count, d, scratch, st)
    else:
        out, want = xp.empty((), h.dtype, dev, st), xp.sum(h)
        wa, wb = xp.sum_work(d)
        run = lambda: _core.array_sum(out, d, wa, wb, 1, st)
    captured = _capture(stream, run)
    nodes = _dot_kernel_nodes(captured, name)
    assert nodes == 1 if name != "sum_f64" and name != "sum_i32" else 1 <= nodes <= 6
    launcher = _core.Graph.from_captured(captured).launcher()
    launcher.stream(st)
    for _ in range(3):
        launcher.launch()
    launcher.synchronize()
    if name == "compact_indices":
        k = int(count)
        assert k == int(want[1]) and index[:k].tolist() == want[0][:k].tolist()
    else:
        assert float(out) == float(want)


_NAN, _INF = float("nan"), float("inf")


def test_comparison_operators_are_elementwise_and_nan_is_false():
    x = xp.asarray([1.0, _NAN, 3.0, -_INF])
    got = {e: eval(e).tolist() for e in ("x == 3.0", "x != 3.0", "x < 3", "x <= 3", "x > 3", "x >= 3", "3 <= x")}
    assert got["x == 3.0"] == [False, False, True, False] and got["x != 3.0"] == [True, True, False, True]
    assert got["x < 3"] == [True, False, False, True] and got["x <= 3"] == [True, False, True, True]
    assert got["x > 3"] == [False] * 4 and got["x >= 3"] == [False, False, True, False] and got["3 <= x"] == got["x >= 3"]
    assert (xp.asarray([1, 2, 3]) == 2.0).tolist() == [False, True, False]
    assert (xp.asarray(-_INF) == -_INF) and xp.asarray([5]) > 0 and not (x == None)  # noqa: E711


def test_truth_value_of_a_size_one_array_and_the_ambiguity_refusal():
    assert bool(xp.asarray([2.0])) and not bool(xp.asarray(0))
    with pytest.raises(ValueError, match="truth value of an array with more than one element is ambiguous"):
        bool(xp.asarray([1, 2]))
    with pytest.raises(TypeError):
        hash(xp.asarray([1]))


@pytest.mark.gpu
def test_comparisons_on_the_device_match_the_host():
    host = xp.asarray([1.0, _NAN, 3.0, -_INF])
    dev = xp.asarray(host, device="cuda:0")
    for e in ("x == 3.0", "x != 3.0", "x < 3", "x <= 3", "x >= 3"):
        assert eval(e, {"x": dev}).tolist() == eval(e, {"x": host}).tolist(), e


@pytest.mark.gpu
def test_an_array_on_a_cupy_stream_object_keeps_the_stream_alive_until_its_free():
    import gc
    import weakref

    cp = pytest.importorskip("cupy")
    s = cp.cuda.Stream(non_blocking=True)
    alive = weakref.ref(s)
    a = xp.full((1 << 18,), 7, xp.int32, "cuda:0", s)
    handle = int(s.ptr)
    assert a.stream == handle
    del s
    gc.collect()
    assert alive() is not None                           # the array holds the stream object
    assert a.get().tolist()[:2] == [7, 7]
    del a
    gc.collect()
    assert alive() is None                               # released after the free was enqueued: no crash on the dead stream
    _core.device_synchronize()


def test_to_names_the_framework_and_raises_with_an_install_hint_when_it_is_missing(monkeypatch):
    a = xp.asarray([1.0, 2.0, 3.0], xp.float64)
    np = pytest.importorskip("numpy")
    out = a.to("numpy")
    assert isinstance(out, np.ndarray) and out.tolist() == [1.0, 2.0, 3.0]
    monkeypatch.setitem(sys.modules, "torch", None)             # torch cannot be imported
    with pytest.raises(ImportError, match=r"torch.*pip install torch"):
        a.to("torch")
    with pytest.raises(ValueError, match="unknown framework"):
        a.to("tensorflow")


def test_returns_routes_results_and_a_missing_framework_warns_once_and_keeps_eagle_arrays(monkeypatch):
    import warnings

    from eagle.frameworks import _bridge

    monkeypatch.setitem(sys.modules, "jax", None)
    monkeypatch.setattr(_bridge, "_WARNED", set())
    a = xp.asarray([1.0, 2.0], xp.float64)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        first = _bridge.route({"y": a}, "jax")
        second = _bridge.route({"y": a}, "jax")
    assert first["y"] is a and second["y"] is a
    assert len([w for w in seen if "pip install jax" in str(w.message)]) == 1
    np = pytest.importorskip("numpy")
    assert isinstance(_bridge.route((a, {"y": a}), "numpy")[1]["y"], np.ndarray)
    assert _bridge.route(a, None) is a


def test_gradients_come_back_through_the_bridge_on_request():
    np = pytest.importorskip("numpy")
    from eagle.diff._reverse import Gradients, Report, _returned

    g = Gradients({"x": xp.asarray([1.0, 2.0], xp.float64)}, Report(0, {}, False))
    assert isinstance(_returned(g, None)["x"], type(g["x"]))
    got = _returned(g, "numpy")
    assert isinstance(got["x"], np.ndarray) and got["x"].tolist() == [1.0, 2.0] and got.report is g.report


@pytest.mark.gpu
def test_to_cupy_aliases_device_memory(stream):
    cp = pytest.importorskip("cupy")
    d = _to(xp.arange(8, dtype=xp.float64), "cuda:0", stream.ptr())
    c = d.to("cupy")
    assert isinstance(c, cp.ndarray) and int(c.data.ptr) == d.ptr
