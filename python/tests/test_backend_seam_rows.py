# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Behaviour that crosses the eagle._core / CUDA-backend seam, on a real device.

* A Python callback the backend calls back into (a composer step) that raises
  surfaces as ITS OWN exception, not a generic error.
* A handle consumed by another call (a Launcher registered into a composer) refuses
  later use with a RuntimeError instead of touching a moved-from object.
* Device agreement (two GPUs): with the launch framework's device set to 1, the
  stream, the replayed graph and the fence eagle creates all land on device 1, and
  an explicit ``device=1`` does the same while device 0 is current. eagle's backend
  runs on the driver API alone and follows the CUDA runtime's context rules (the
  driver's per-thread current context, the device's primary context), which is
  what this row pins.
"""

from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")


def _device_count() -> int:
    try:
        return cp.cuda.runtime.getDeviceCount()
    except Exception:
        return 0


@pytest.mark.gpu
def test_a_raising_composer_step_surfaces_its_own_exception():
    from eagle import _core

    def step(stream):
        raise KeyError("raised inside a composer step")

    c = _core.GraphComposer("sequenced")
    c.register_callable(step, name="boom")
    c.build()
    with pytest.raises(KeyError, match="raised inside a composer step"):
        c.launch()


@pytest.mark.gpu
def test_a_consumed_launcher_refuses_reuse():
    from eagle import _core
    from eagle.frameworks.cupy import wrap_external_stream

    s = _core.Stream(non_blocking=True)
    a = cp.arange(8, dtype=cp.float64)
    cap = _core.StreamCapturer(s.ptr())
    with wrap_external_stream(s.ptr()):
        cap.begin()
        a *= 2.0
        captured = cap.end()
    launcher = _core.Graph.from_captured(captured).launcher()
    assert not captured  # consumed by from_captured
    c = _core.GraphComposer("sequenced")
    c.register_launcher(launcher, name="m")
    with pytest.raises(RuntimeError, match="consumed"):
        launcher.launch()


@pytest.mark.gpu2
@pytest.mark.skipif(
    _device_count() < 2, reason="needs two CUDA devices (skipped, never passed, on one)"
)
def test_eagle_objects_follow_the_frameworks_device():
    from eagle import _core
    from eagle.frameworks.cupy import wrap_external_stream

    def roundtrip(stream_ptr):
        x = cp.arange(16, dtype=cp.float64)
        cap = _core.StreamCapturer(stream_ptr)
        with wrap_external_stream(stream_ptr):
            cap.begin()
            x *= 3.0
            graph = cap.end()
        launcher = _core.Graph.from_captured(graph).launcher()
        launcher.stream(stream_ptr)
        launcher.launch()
        launcher.synchronize()
        return float(x[1])

    with cp.cuda.Device(1):
        s = _core.Stream(non_blocking=True)  # device -1: the framework's (1)
        assert (
            roundtrip(s.ptr()) == 3.0
        )  # a device-0 stream could not run device-1 work
        other = cp.cuda.Stream(non_blocking=True)
        _core.fence(s.ptr(), other.ptr, 1)
        other.synchronize()

    with cp.cuda.Device(0):
        explicit = _core.Stream(non_blocking=True, device=1)
    with cp.cuda.Device(1):
        assert roundtrip(explicit.ptr()) == 3.0


# ---- seam minor 7: built-in cell ops, bit-equal to the kernels they replace -------------------
# (the cupy kernels below are the references the ops must equal; integer ops, so bit-for-bit)


def _fresh_stream():
    from eagle import _core

    return _core.Stream(non_blocking=True)


def _run_op(op, stream, n=0, p=(), i=()):
    from eagle import _core

    cp.cuda.Device().synchronize()  # the operands were written on cupy's stream; this one does not wait for it
    _core.cell_op(op, stream.ptr(), n, [int(x) for x in p], [int(x) for x in i])
    stream.synchronize()


@pytest.mark.gpu
def test_cell_op_bump_mask_count_identity_count_fired_scale_count_equal_their_cupy_kernels():
    from eagle import _core

    s = _fresh_stream()
    cell = cp.asarray([41], dtype=cp.uint32)
    cp.ElementwiseKernel("", "raw uint32 cell", "if (i == 0) cell[0] += 1u;", "ref_bump")(cell, size=1)
    other = cp.asarray([41], dtype=cp.uint32)
    _run_op(_core.CELL_BUMP, s, p=[other.data.ptr])
    assert int(other[0]) == int(cell[0]) == 42

    rng = cp.random.default_rng(7)
    mask = (rng.random(100003) < 0.3).astype(cp.bool_)
    ref = cp.zeros(1, dtype=cp.uint32)
    cp.ElementwiseKernel("bool mask", "raw uint32 cell", "if (mask) atomicAdd(&cell[0], 1u);", "ref_count")(mask, ref)
    got = cp.zeros(1, dtype=cp.uint32)
    _run_op(_core.CELL_MASK_COUNT, s, mask.size, [mask.data.ptr, got.data.ptr])
    assert int(got[0]) == int(ref[0]) == int(cp.count_nonzero(mask))

    a, b = cp.full(1000, -1, dtype=cp.int32), cp.full(1000, -1, dtype=cp.int32)
    cp.ElementwiseKernel("", "raw int32 a", "a[i] = (int)i;", "ref_identity")(a, size=1000)
    _run_op(_core.CELL_IDENTITY, s, 1000, [b.data.ptr])
    assert bool((a == b).all())

    for fire in (0, 1, 5):
        f = cp.asarray([fire], dtype=cp.uint32)
        r1, r2 = cp.asarray([3], dtype=cp.uint32), cp.asarray([3], dtype=cp.uint32)
        cp.ElementwiseKernel("raw uint32 fire", "uint32 reorders",
                             "reorders += fire[0] != 0u ? 1u : 0u;", "ref_fired")(f, r1)
        _run_op(_core.CELL_COUNT_FIRED, s, p=[f.data.ptr, r2.data.ptr])
        assert int(r1[0]) == int(r2[0])

    c = cp.asarray([37], dtype=cp.uint32)
    o1, o2 = cp.zeros(1, dtype=cp.uint32), cp.zeros(1, dtype=cp.uint32)
    cp.ElementwiseKernel("raw uint32 c, int32 group", "raw uint32 o", "o[0] = c[0] * (unsigned)group;",
                         "ref_scale")(c, 32, o1, size=1)
    _run_op(_core.CELL_SCALE_COUNT, s, p=[c.data.ptr, o2.data.ptr], i=[32])
    assert int(o1[0]) == int(o2[0]) == 37 * 32


@pytest.mark.gpu
@pytest.mark.parametrize("lanes,group", [(1000, 32), (1024, 32), (7, 4), (96, 1), (33, 32)])
def test_cell_op_group_and_equals_its_cupy_kernel(lanes, group):
    from eagle import _core

    s = _fresh_stream()
    rng = cp.random.default_rng(lanes)
    mask = (rng.random(lanes) < 0.9).astype(cp.uint8)
    groups = -(-lanes // group)
    ref, got = cp.zeros(groups, dtype=cp.uint8), cp.zeros(groups, dtype=cp.uint8)
    cp.ElementwiseKernel(
        "raw uint8 m, int64 n, int32 group", "raw uint8 d",
        "unsigned char a = 1; long long s = (long long)i * group;"
        " for (int j = 0; j < group && s + j < n; ++j) a &= (m[s + j] != 0);"
        " d[i] = a;", "ref_group_and")(mask, lanes, group, ref, size=groups)
    _run_op(_core.CELL_GROUP_AND, s, groups, [mask.data.ptr, got.data.ptr], [lanes, group])
    assert bool((ref == got).all())


_REFERENCE_POLICY = r"""
extern "C" __global__ void ref_policy(
        const unsigned int* finished, const unsigned int* total, long long* cells,
        long long* word, unsigned int* go, long long* hist) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long KMIN = %(kmin)d, KMAX = %(kmax)d;
    const long long fin = finished[0], all = total[0];
    const long long live = all - fin, newly = fin - cells[%(fin_at)d];
    const long long done = cells[%(done)d] + word[0];
    cells[%(mapped)d] += all - cells[%(fin_at)d];
    cells[%(fin_at)d] = fin;
    cells[%(done)d] = done;
    const long long k = cells[%(k)d];
    const double rate = (double)cells[%(r)d] / %(scale)d.0;
    long long k_next = k;
    if (newly <= 0) k_next = KMAX;
    else {
        const double e2 = 2.0 * rate * (double)live * (double)k / (double)newly;
        long long p = 1;
        while (p < (1LL << 30) && e2 >= 2.0 * (double)p * (double)p) p *= 2;
        k_next = p < KMIN ? KMIN : (p > KMAX ? KMAX : p);
    }
    cells[%(k)d] = k_next;
    const long long left = cells[%(cap)d] - done;
    const unsigned int g = (fin != all) && (left > 0);
    go[0] = g;
    if (g) {
        const long long w = k_next < left ? k_next : left;
        word[0] = w;
        hist[w] += 1;
    }
}
"""


def _reference_policy_kernel(k_max, k_min, ud):
    """The policy as an NVRTC kernel, the independent reference the built-in op is compared to."""
    src = _REFERENCE_POLICY % {"kmin": k_min, "kmax": k_max, "k": ud._K, "fin_at": ud._FIN_AT, "done": ud._DONE,
                               "cap": ud._CAP, "r": ud._R, "mapped": ud._MAPPED, "scale": ud.RATE_SCALE}
    return cp.RawKernel(src, "ref_policy")


@pytest.mark.gpu
@pytest.mark.parametrize("k_min,k_max", [(8, 64), (1, 1024), (16, 16)])
def test_cell_op_policy_equals_the_cupy_kernel_over_a_random_corpus(k_min, k_max):
    from eagle import _core, _until_done as ud

    s = _fresh_stream()
    ref_kernel = _reference_policy_kernel(k_max, k_min, ud)
    rng = cp.random.default_rng(k_max)
    for _ in range(60):
        total = int(rng.integers(1, 5000))
        fin = int(rng.integers(0, total + 1))
        cells = rng.integers(0, 3000, 6).astype(cp.int64)
        cells[ud._FIN_AT] = int(rng.integers(0, fin + 1))
        cells[ud._CAP] = int(rng.integers(0, 5000))
        cells[ud._K] = int(rng.integers(k_min, k_max + 1))
        cells[ud._R] = int(rng.integers(100, 5000))
        word = rng.integers(0, 64, 1).astype(cp.int64)
        hist0 = rng.integers(0, 9, k_max + 2).astype(cp.int64)
        finished, tot = cp.asarray([fin], dtype=cp.uint32), cp.asarray([total], dtype=cp.uint32)
        r_cells, r_word, r_hist, r_go = cells.copy(), word.copy(), hist0.copy(), cp.zeros(1, dtype=cp.uint32)
        g_cells, g_word, g_hist, g_go = cells.copy(), word.copy(), hist0.copy(), cp.zeros(1, dtype=cp.uint32)
        ref_kernel((1,), (1,), (finished, tot, r_cells, r_word, r_go, r_hist))
        _run_op(_core.CELL_POLICY, s,
                p=[finished.data.ptr, tot.data.ptr, g_cells.data.ptr, g_word.data.ptr, g_go.data.ptr,
                   g_hist.data.ptr],
                i=[k_min, k_max, ud._K, ud._FIN_AT, ud._DONE, ud._CAP, ud._R, ud._MAPPED, ud.RATE_SCALE])
        for ref, got in ((r_cells, g_cells), (r_word, g_word), (r_hist, g_hist), (r_go, g_go)):
            assert bool((ref == got).all())


_DTYPES = ("bool", "uint8", "int32", "uint32", "int64", "uint64", "float32", "float64")


@pytest.mark.gpu
@pytest.mark.parametrize("dst", _DTYPES)
@pytest.mark.parametrize("src", _DTYPES)
def test_cell_op_cast_equals_astype(src, dst):
    from eagle import _core

    s = _fresh_stream()
    rng = cp.random.default_rng(3)
    if src.startswith("float"):
        x = ((rng.random(257) - 0.5) * 300).astype(src)
    elif src == "bool":
        x = rng.random(257) < 0.5
    else:
        x = rng.integers(0, 120, 257).astype(src)
    ref = x.astype(dst)
    got = cp.empty_like(ref)
    code = lambda name: getattr(_core, "DTYPE_" + name.upper())  # noqa: E731
    _run_op(_core.CELL_CAST, s, x.size, [got.data.ptr, x.data.ptr], [code(dst), code(src)])
    assert got.tobytes() == ref.tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["uint8", "float32", "float64", "int64"])
def test_cell_op_contiguous_gather_equals_ascontiguousarray(dtype):
    from eagle import _core

    s = _fresh_stream()
    base = cp.arange(4 * 5 * 6 * 7, dtype=dtype).reshape(4, 5, 6, 7)
    for view in (base.T, base[::2, 1:, :, ::3], base[:, 2], base[1].transpose(2, 0, 1), base[0, 0, 0]):
        ref = cp.ascontiguousarray(view)
        got = cp.empty(view.shape, dtype=dtype)
        shape = list(view.shape) + [0] * (4 - view.ndim)
        strides = list(view.strides) + [0] * (4 - view.ndim)
        _run_op(_core.CELL_CONTIGUOUS_GATHER, s, view.size, [got.data.ptr, view.data.ptr],
                [view.itemsize, view.ndim, *shape, *strides])
        assert got.tobytes() == ref.tobytes()


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", ["float32", "float64", "int32", "int64"])
def test_cell_op_add_equals_cupy_add(dtype):
    from eagle import _core

    s = _fresh_stream()
    rng = cp.random.default_rng(5)
    a = (rng.random(1001) * 100).astype(dtype)
    b = (rng.random(1001) * 100).astype(dtype)
    got = cp.empty_like(a)
    _run_op(_core.CELL_ADD, s, a.size, [got.data.ptr, a.data.ptr, b.data.ptr],
            [getattr(_core, "DTYPE_" + dtype.upper())])
    assert got.tobytes() == cp.add(a, b).tobytes()


@pytest.mark.gpu
def test_cell_op_refuses_bad_arguments_by_name():
    from eagle import _core

    s = _fresh_stream()
    with pytest.raises(ValueError, match="CAST needs dtype codes"):
        _run_op(_core.CELL_CAST, s, 4, [8, 8], [99, 0])
    with pytest.raises(ValueError, match="p1 must be a device address"):
        _run_op(_core.CELL_MASK_COUNT, s, 4, [8])
    with pytest.raises(ValueError, match="CONTIGUOUS_GATHER n must be the product"):
        _run_op(_core.CELL_CONTIGUOUS_GATHER, s, 5, [8, 8], [4, 1, 4, 0, 0, 0, 4, 0, 0, 0])


# ---- seam minor 7: device scope, modules, raw launch, strided copy ----------------------------

_KERNEL = 'extern "C" __global__ void fill(int* a, int base) { a[threadIdx.x] = base + (int)threadIdx.x; }'


@pytest.mark.gpu
def test_device_scope_symbols_match_cupy():
    from eagle import _core

    assert _core.device_count() == cp.cuda.runtime.getDeviceCount()
    _core.set_device(0)
    assert cp.cuda.runtime.getDevice() == 0
    free, total = cp.cuda.runtime.memGetInfo()
    got_total, got_free = _core.device_mem_info(0)
    assert got_total == total and abs(got_free - free) < 256 << 20
    s = _fresh_stream()
    assert _core.stream_is_capturing(s.ptr()) is False
    assert _core.stream_is_capturing(0) is False
    _core.device_synchronize()


_NOOP_PTX = b""".version 7.0
.target sm_52
.address_size 64
.visible .entry noop() { ret; }
"""


@pytest.mark.gpu
def test_ptx_loads_jit_under_a_register_cap():
    from eagle import _core

    module = _core.module_load_jit(_NOOP_PTX, 16)
    fn = _core.module_function(module, "noop")
    assert 0 <= _core.function_attribute(fn, 4) <= 16
    _core.module_release(module)
    with pytest.raises(Exception):
        _core.module_load_jit(b"not ptx", 16)


@pytest.mark.gpu
def test_ptx_links_to_a_cubin_under_a_register_cap():
    from eagle import _core

    cubin = _core.link_ptx(_NOOP_PTX, 16)
    assert cubin[:4] == b"\x7fELF"
    with pytest.raises(Exception):
        _core.link_ptx(b"not ptx", 16)


@pytest.mark.gpu
def test_a_module_loads_its_function_launches_raw_and_unloads():
    import ctypes

    from eagle import _core

    from cupy.cuda import compiler

    image = compiler.compile_using_nvrtc(_KERNEL)[0]  # a cubin (or PTX text): either loads
    module = _core.module_load(image)
    fn = _core.module_function(module, "fill")
    out = cp.zeros(8, dtype=cp.int32)
    cp.cuda.Device().synchronize()  # the zero fill was written on cupy's stream; the raw launch below does not wait for it
    ptr, base = ctypes.c_void_p(out.data.ptr), ctypes.c_int(100)
    s = _fresh_stream()
    _core.launch_raw(fn, (1, 1, 1), (8, 1, 1), [ctypes.addressof(ptr), ctypes.addressof(base)], s.ptr())
    s.synchronize()
    assert out.get().tolist() == list(range(100, 108))
    with pytest.raises(IndexError, match="no function named 'missing'"):
        _core.module_function(module, "missing")
    assert _core.function_attribute(fn, 0) >= 32 and _core.function_attribute(fn, 4) > 0
    _core.module_release(module)
    _core.module_release(0)


@pytest.mark.gpu
def test_memcpy_strided_equals_copyto_on_strided_planes():
    from eagle import _core

    s = _fresh_stream()
    src = cp.arange(5 * 9, dtype=cp.float32).reshape(5, 9)
    dst = cp.zeros((5, 12), dtype=cp.float32)
    # the 5x4 block at column 2 of src into column 3 of dst
    ref = dst.copy()
    cp.copyto(ref[:, 3:7], src[:, 2:6])
    _core.memcpy_strided(dst[:, 3:].data.ptr, dst.strides[0], src[:, 2:].data.ptr, src.strides[0], 4 * 4, 5,
                         _core.COPY_D2D, s.ptr())
    s.synchronize()
    assert bool((dst == ref).all())
    host = cp.asnumpy(src)
    got = __import__("numpy").zeros((5, 4), dtype="float32")
    _core.memcpy_strided(got.ctypes.data, 16, src[:, 2:].data.ptr, src.strides[0], 16, 5, _core.COPY_D2H, s.ptr())
    s.synchronize()
    assert (got == host[:, 2:6]).all()
    with pytest.raises(ValueError, match="pitch is smaller"):
        _core.memcpy_strided(dst.data.ptr, 4, src.data.ptr, 64, 16, 2, _core.COPY_D2D, s.ptr())


@pytest.mark.gpu
def test_an_event_orders_one_stream_after_another_and_is_released():
    from eagle import _core, arrays

    a, b = _core.Stream(non_blocking=True), _core.Stream(non_blocking=True)
    event = _core.event_create()
    src = arrays.full((1 << 20,), 7, arrays.int32, "cuda:0", a.ptr())
    _core.event_record(event, a.ptr())
    _core.event_wait(b.ptr(), event)
    dst = arrays.empty((1 << 20,), arrays.int32, "cuda:0", b.ptr())
    _core.array_copyto(dst, src, b.ptr())
    b.synchronize()
    assert dst.get().tolist() == [7] * (1 << 20)
    del src, dst                    # their frees are queued on the two streams: release them first
    _core.event_release(event)
    _core.event_release(0)
    with pytest.raises(Exception, match="null event"):
        _core.event_record(0, a.ptr())


@pytest.mark.gpu
def test_two_timed_events_measure_the_work_between_them():
    from eagle import _core, arrays

    s = _core.Stream(non_blocking=True)
    t0, t1 = _core.event_create_timed(), _core.event_create_timed()
    _core.event_record(t0, s.ptr())
    a = arrays.full((1 << 22,), 1, arrays.int32, "cuda:0", s.ptr())
    _core.event_record(t1, s.ptr())
    s.synchronize()
    assert _core.event_elapsed(t0, t1) >= 0.0
    del a
    _core.event_release(t0)
    _core.event_release(t1)
    with pytest.raises(Exception, match="null event"):
        _core.event_elapsed(0, t1)


@pytest.mark.gpu
def test_get_device_reads_the_current_device_ordinal():
    from eagle import _core
    from eagle.launch import _current_device_ordinal

    _core.set_device(0)
    assert _core.get_device() == cp.cuda.runtime.getDevice() == 0
    assert _current_device_ordinal() == 0


def test_gemm_on_the_host_through_the_seam_binding_and_the_release_of_an_unused_handle_is_quiet():
    import struct

    from eagle import _core

    a = struct.pack("6f", 1, 2, 3, 4, 5, 6)        # (2, 3) column-major
    b = struct.pack("6f", 1, 0, 0, 1, 1, 1)        # (3, 2) column-major
    import ctypes
    ba, bb, bc = (ctypes.create_string_buffer(buf, len(buf)) for buf in (a, b, bytes(16)))
    _core.gemm(0, 0, 0, 2, 2, 3, ctypes.addressof(ba), 2, ctypes.addressof(bb), 3, ctypes.addressof(bc), 2, host=True)
    assert struct.unpack("4f", bc.raw) == (1.0, 2.0, 9.0, 12.0)
    _core.gemm_release(0, 0)                         # nothing was created for the legacy stream: a no-op, no error


@pytest.mark.gpu
def test_gemm_on_the_device_through_the_seam_with_a_batch():
    import eagle.arrays as xp
    from eagle import _core

    two = xp.asarray([[1.0, 2.0], [3.0, 4.0]], xp.float32, "cuda:0")
    out = xp.zeros((2, 2, 2), xp.float32, "cuda:0")
    batch = xp.asarray([[[1.0, 0.0], [0.0, 1.0]], [[2.0, 0.0], [0.0, 2.0]]], xp.float32, "cuda:0")
    s = _core.Stream(non_blocking=True)
    _core.gemm(0, 0, 0, 2, 2, 2, batch.ptr, 2, two.ptr, 2, out.ptr, 2, 2, 4, 0, 4, 1.0, 0.0, s.ptr(), 0, False)
    s.synchronize()
    _core.gemm_release(s.ptr(), 0)
    got = out.get().tolist()
    assert got[0] == [[1.0, 3.0], [2.0, 4.0]] or got[0] == [[1.0, 2.0], [3.0, 4.0]]
    assert got[1] == [[2 * v for v in row] for row in got[0]]


@pytest.mark.gpu
def test_launch_status_is_a_query_that_finds_nothing_pending_after_good_launches():
    from eagle import _core

    cp.arange(10).sum()                                  # work in flight: the query does not wait for it
    assert _core.launch_status() is None
