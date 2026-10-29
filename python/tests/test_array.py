# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The array layer's seam rows (SA1): ``cell_op`` 10-23 and ``mem_*``, with no numpy and no cupy.

Every op: enqueued in a capture it adds exactly one kernel node, the graph replays, and the
device result equals the host twin (OpenMP + SIMD over the same body). A synchronisation inside a
capture fails, and so does an allocation. Buffers are ctypes arrays and ``_core.mem_alloc`` memory.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys

import pytest

from eagle import _core

I64, F64, I32, U8 = _core.DTYPE_INT64, _core.DTYPE_FLOAT64, _core.DTYPE_INT32, _core.DTYPE_UINT8
SIZE = {_core.DTYPE_UINT8: 1, I32: 4, I64: 8, F64: 8, _core.DTYPE_FLOAT32: 4, _core.DTYPE_UINT32: 4,
        _core.DTYPE_UINT64: 8, _core.DTYPE_BOOL: 1}
CTYPE = {I64: ctypes.c_int64, F64: ctypes.c_double, I32: ctypes.c_int32, U8: ctypes.c_uint8}
N = 1500


class Dev:
    """A device buffer of ``nbytes`` from ``mem_alloc``, filled from / read to a ctypes array."""

    def __init__(self, nbytes):
        self.chunk, self.ptr = _core.mem_alloc(_core.MEM_DEVICE, -1, nbytes, 0)
        self.nbytes = nbytes

    def put(self, arr):
        _core.memcpy(self.ptr, ctypes.addressof(arr), self.nbytes, _core.COPY_H2D, 0)
        _core.device_synchronize()
        return self

    def get(self, ctype, n):
        out = (ctype * n)()
        _core.memcpy(ctypes.addressof(out), self.ptr, self.nbytes, _core.COPY_D2H, 0)
        _core.device_synchronize()
        return list(out)

    def free(self):
        _core.mem_release(self.chunk)


def _arr(ctype, values):
    return (ctype * len(values))(*values)


def _case(op):
    """(ctype, dtype, device inputs, host inputs, i, f, out ctype) of a representative case of ``op``."""
    ct = CTYPE[I64]
    x = [(j * 31) % 97 - 48 for j in range(N)]
    y = [(j * 17) % 89 - 44 for j in range(N)]
    idx = [(j * 7 + 3) % N for j in range(N)]       # N = 1500 is not prime: use a coprime step for scatter
    idx = [(j * 7 + 3) % N for j in range(N)] if N % 7 else idx
    mask = [1 if j % 3 == 0 else 0 for j in range(N)]
    c = _core
    table = {
        c.CELL_FILL: (I64, [], [], [7], [], ct),
        c.CELL_SUB: (I64, [x, y], [], [], [], ct),
        c.CELL_MUL: (I64, [x, y], [], [], [], ct),
        c.CELL_MAX: (I64, [x, y], [], [], [], ct),
        c.CELL_EQ: (I64, [x, y], [], [], [], ctypes.c_uint8),
        c.CELL_NE: (I64, [x, y], [], [], [], ctypes.c_uint8),
        c.CELL_GT: (I64, [x, y], [], [], [], ctypes.c_uint8),
        c.CELL_WHERE: (I64, [x, y, mask], [], [], [], ct),
        c.CELL_CLIP: (I64, [x], [], [-5, 40], [], ct),
        c.CELL_ARANGE: (I64, [], [], [-3, 2], [], ct),
        c.CELL_LINSPACE: (F64, [], [], [], [-1.5, 0.37], ctypes.c_double),
        c.CELL_TAKE: (I64, [x, idx], [], [], [], ct),
        c.CELL_CUMSUM: (I64, [x], [], [], [], ct),
        c.CELL_SCATTER: (I64, [x, idx], [], [], [], ct),
    }
    for name in ("ADDS", "SUBS", "MULS", "MAXS", "RSUBS"):     # the scalar-operand forms: -5 is in the operand
        table[getattr(c, "CELL_" + name)] = (I64, [x], [], [-5], [], ct)
    for name in ("EQS", "NES", "GTS", "LTS"):
        table[getattr(c, "CELL_" + name)] = (I64, [x], [], [-5], [], ctypes.c_uint8)
    return table[op]


OPS = [getattr(_core, n) for n in ("CELL_FILL", "CELL_SUB", "CELL_MUL", "CELL_MAX", "CELL_EQ", "CELL_NE", "CELL_GT",
                                   "CELL_WHERE", "CELL_CLIP", "CELL_ARANGE", "CELL_LINSPACE", "CELL_TAKE",
                                   "CELL_CUMSUM", "CELL_SCATTER", "CELL_ADDS", "CELL_SUBS", "CELL_MULS", "CELL_MAXS",
                                   "CELL_EQS", "CELL_NES", "CELL_GTS", "CELL_RSUBS", "CELL_LTS")]


def _prepare(op):
    """Device buffers + the cell_op / host_op arguments of ``op``; the caller frees the buffers."""
    dtype, ins, _, ints, floats, out_ct = _case(op)
    elem = SIZE[dtype]
    bufs = []
    for k, vals in enumerate(ins):
        ct = ctypes.c_int32 if (op in (_core.CELL_TAKE, _core.CELL_SCATTER) and k == 1) else (
            ctypes.c_uint8 if (op == _core.CELL_WHERE and k == 2) else CTYPE[dtype])
        bufs.append(Dev(len(vals) * ctypes.sizeof(ct)).put(_arr(ct, vals)))
    out = Dev(N * ctypes.sizeof(out_ct))
    if op == _core.CELL_SCATTER:
        out.put(_arr(out_ct, [0] * N))
    p = [out.ptr] + [b.ptr for b in bufs]
    if op == _core.CELL_WHERE:      # p0 dst, p1 x, p2 y, p3 mask
        p = [out.ptr, bufs[0].ptr, bufs[1].ptr, bufs[2].ptr]
    scratch = None
    if op == _core.CELL_CUMSUM:
        scratch = Dev(_core.cumsum_scratch_bytes(N))
        p = [out.ptr, bufs[0].ptr, scratch.ptr]
    i = list(ints) + [0] * (9 - len(ints)) + [dtype]
    return dtype, out, out_ct, bufs + ([scratch] if scratch else []), p, i, floats


def _capture(stream, enqueue):
    cap = _core.StreamCapturer(stream.ptr())
    cap.begin()
    enqueue()
    return cap.end()


def _kernel_nodes(captured, tmp="/local/scratch/amasat/tmp/w072/scratch/S1a/tmp/array_graph.dot") -> int:
    """Kernel nodes of a captured graph, read off its Graphviz dump (a memset or copy node is not one)."""
    dot = captured.debug_dot(tmp, 0)
    return dot.count('shape="octagon"')      # CUDA draws kernel nodes as octagons (memset / copy nodes are not)


@pytest.mark.gpu
@pytest.mark.parametrize("op", OPS)
def test_each_op_adds_one_kernel_node_replays_and_equals_its_host_twin(op):
    dtype, out, out_ct, keep, p, i, floats = _prepare(op)
    s = _core.Stream(non_blocking=True)
    captured = _capture(s, lambda: _core.cell_op(op, s.ptr(), N, p, i, -1, floats))
    assert _kernel_nodes(captured) == 1
    launcher = _core.Graph.from_captured(captured).launcher()
    launcher.stream(s.ptr())
    for _ in range(3):
        launcher.launch()
    launcher.synchronize()
    got = out.get(out_ct, N)
    # the host twin on host copies of the same operands
    host = [(_arr(ctypes.c_uint8, [0] * 0), 0)]
    hp = []
    for b in [out] + keep:
        n = b.nbytes
        h = (ctypes.c_uint8 * n)()
        _core.memcpy(ctypes.addressof(h), b.ptr, n, _core.COPY_D2H, 0)
        hp.append(h)
    _core.device_synchronize()
    ref_dst = (ctypes.c_uint8 * out.nbytes)()
    if op == _core.CELL_CUMSUM:
        hp_ptrs = [ctypes.addressof(ref_dst), ctypes.addressof(hp[1])]
    elif op == _core.CELL_SCATTER:
        hp_ptrs = [ctypes.addressof(ref_dst)] + [ctypes.addressof(h) for h in hp[1:]]
    else:
        hp_ptrs = [ctypes.addressof(ref_dst)] + [ctypes.addressof(h) for h in hp[1:]]
    if op == _core.CELL_CUMSUM:
        hp_ptrs.append(ctypes.addressof((ctypes.c_uint8 * 8)()))
    _core.host_op(op, N, hp_ptrs, i, floats)
    ref = list((out_ct * N).from_buffer(ref_dst))
    assert got == ref, f"op {op}: device {got[:6]}... != host {ref[:6]}..."
    assert any(v != 0 for v in got) or op == _core.CELL_FILL
    del host
    for b in [out] + keep:
        b.free()


@pytest.mark.gpu
def test_the_multi_block_cumsum_has_no_lane_ceiling():
    n = (1 << 21) + 5
    x = Dev(n * 8).put(_arr(ctypes.c_int64, [1] * n))
    out, scratch = Dev(n * 8), Dev(_core.cumsum_scratch_bytes(n))
    s = _core.Stream(non_blocking=True)
    _core.cell_op(_core.CELL_CUMSUM, s.ptr(), n, [out.ptr, x.ptr, scratch.ptr], [0] * 9 + [I64])
    s.synchronize()
    tail = (ctypes.c_int64 * 3)()
    _core.memcpy(ctypes.addressof(tail), out.ptr + (n - 3) * 8, 24, _core.COPY_D2H, 0)
    _core.device_synchronize()
    assert list(tail) == [n - 2, n - 1, n]
    for b in (x, out, scratch):
        b.free()


@pytest.mark.gpu
def test_a_synchronisation_inside_a_capture_fails():
    buf = Dev(64)
    s = _core.Stream(non_blocking=True)
    cap = _core.StreamCapturer(s.ptr())
    cap.begin()
    try:
        _core.cell_op(_core.CELL_FILL, s.ptr(), 8, [buf.ptr], [3] + [0] * 8 + [I64])
        assert _core.stream_is_capturing(s.ptr()) is True
        with pytest.raises(Exception):
            s.synchronize()           # not capture-legal: the non-vacuity row of every op above
        with pytest.raises(Exception):
            _core.device_synchronize()
    finally:
        try:
            cap.end()
        except Exception:
            pass
    buf.free()


@pytest.mark.gpu
def test_memory_is_stream_ordered_pointer_device_knows_it_and_a_capture_defers_the_free():
    chunk, ptr = _core.mem_alloc(_core.MEM_DEVICE, -1, 4096, 0)
    assert ptr % 256 == 0 and _core.pointer_device(ptr) == 0
    host_chunk, host_ptr = _core.mem_alloc(_core.MEM_PINNED, -1, 128, 0)
    assert _core.pointer_device(host_ptr) == -1
    assert _core.mem_alloc(_core.MEM_DEVICE, -1, 0, 0) == (0, 0)
    s = _core.Stream(non_blocking=True)
    victim_chunk, _ = _core.mem_alloc(_core.MEM_DEVICE, -1, 1024, s.ptr())
    cap = _core.StreamCapturer(s.ptr())
    cap.begin()
    try:
        with pytest.raises(Exception, match="allocation inside a stream capture is refused"):
            _core.mem_alloc(_core.MEM_DEVICE, -1, 64, s.ptr())
        _core.mem_release(victim_chunk)               # deferred: the stream is capturing
        assert _core.mem_drain() == 1
    finally:
        cap.end()
    assert _core.mem_drain() == 0
    _core.mem_release(chunk)
    _core.mem_release(host_chunk)
    _core.mem_release(0)


_HOST_CHILD = """
import ctypes, sys
from eagle import _core
sys.modules.setdefault("numpy", None)
n = 5003
def buf(ct, vals): return (ct * len(vals))(*vals)
x = buf(ctypes.c_int64, [(j * 31) % 97 - 48 for j in range(n)])
y = buf(ctypes.c_int64, [(j * 17) % 89 - 44 for j in range(n)])
out = (ctypes.c_int64 * n)()
I = lambda extra=(): list(extra) + [0] * (9 - len(extra)) + [@DT@]
def run(op, ptrs, ints=(), floats=()):
    _core.host_op(op, n, [ctypes.addressof(p) for p in ptrs], I(ints), list(floats)); return list(out)
assert run(_core.CELL_SUB, [out, x, y]) == [a - b for a, b in zip(x, y)]
assert run(_core.CELL_MUL, [out, x, y]) == [a * b for a, b in zip(x, y)]
assert run(_core.CELL_MAX, [out, x, y]) == [max(a, b) for a, b in zip(x, y)]
assert run(_core.CELL_ARANGE, [out], (-3, 2)) == [-3 + 2 * j for j in range(n)]
assert run(_core.CELL_CLIP, [out, x], (-5, 40)) == [min(max(a, -5), 40) for a in x]
scratch = (ctypes.c_uint8 * 8)()
acc, want = 0, []
for a in x: acc += a; want.append(acc)
assert run(_core.CELL_CUMSUM, [out, x, scratch]) == want
print("ok")
""".replace("@DT@", str(I64))


def test_the_host_twins_run_under_four_threads_and_match_a_pure_python_reference():
    env = dict(os.environ, OMP_NUM_THREADS="4", CUDA_VISIBLE_DEVICES="")
    done = subprocess.run([sys.executable, "-I", "-c", _HOST_CHILD], capture_output=True, text=True, env=env,
                          cwd=os.getcwd())
    assert done.returncode == 0 and done.stdout.strip() == "ok", done.stderr[-800:]


@pytest.mark.gpu
def test_a_captured_zero_is_one_memset_node_and_no_kernel():
    from eagle import _alloc, arrays

    s = _core.Stream(non_blocking=True)
    a = arrays.full((1 << 12,), 7, arrays.int32, "cuda:0", s.ptr())
    s.synchronize()
    captured = _capture(s, lambda a=a: _alloc.zero("cuda:0", a, s.ptr()))
    dot = captured.debug_dot("/local/scratch/amasat/tmp/w072/scratch/S1a/tmp/zero_graph.dot", 0)
    assert dot.count('shape="octagon"') == 0            # no kernel
    assert dot.count("MEMSET") + dot.count("Memset") + dot.count("memset") == 1
    _core.Graph.from_captured(captured).launcher().launch()
    _core.device_synchronize()
    assert a.get().tolist() == [0] * (1 << 12)
    del a


@pytest.mark.gpu
@pytest.mark.parametrize("sides", ["into_view", "from_view", "both_strided"])
def test_a_captured_strided_copy_is_one_kernel_node_and_replays_correctly(sides):
    from eagle import arrays

    s = _core.Stream(non_blocking=True)
    flat = arrays.arange(24, dtype=arrays.int64, device="cuda:0", stream=s.ptr())
    out = arrays.zeros((4, 6), arrays.int64, "cuda:0", s.ptr())
    if sides == "into_view":         # a contiguous (6, 4) source into the transposed view of a (4, 6) array
        dst, src = out.T, flat.reshape((6, 4))
    elif sides == "from_view":       # the transposed view of a (4, 6) source into a contiguous (6, 4) array
        out = arrays.zeros((6, 4), arrays.int64, "cuda:0", s.ptr())
        dst, src = out, flat.reshape((4, 6)).T
    else:                            # a column slice of one array into a transposed slice of another
        dst, src = out.T[1:5, 1:3], flat.reshape((4, 6))[1:3, 0:4].T
    s.synchronize()
    captured = _capture(s, lambda dst=dst, src=src: arrays.copyto(dst, src, s.ptr()))
    assert _kernel_nodes(captured) == 1
    launcher = _core.Graph.from_captured(captured).launcher()
    launcher.stream(s.ptr())
    for _ in range(2):
        launcher.launch()
    launcher.synchronize()
    assert dst.get().tolist() == src.get().tolist()
    del dst, src, out, flat


def test_a_view_names_another_shape_over_the_same_memory():
    from eagle import arrays

    base = arrays.arange(6, dtype=arrays.int64)
    view = arrays.view_as(base, (3, 2))
    assert view.shape == (3, 2) and view.tolist() == [[0, 1], [2, 3], [4, 5]]
    wide = arrays.view_as(base, (5, 6), (0, 1))            # rows alias: one row of storage, five rows named
    assert wide[4].tolist() == base.tolist()
    base.fill(7)
    assert view.tolist() == [[7, 7]] * 3


@pytest.mark.gpu
def test_a_strided_host_side_copies_with_a_pitched_transfer_either_way():
    from eagle import arrays

    host = arrays.arange(24, dtype=arrays.int64).reshape((4, 6))
    dev = arrays.zeros((4, 3), arrays.int64, "cuda:0")
    arrays.copyto(dev, host[:, 1:4])                          # strided host source, contiguous device target
    assert dev.get().tolist() == host[:, 1:4].tolist()
    back = arrays.zeros((4, 6), arrays.int64)
    arrays.copyto(back[:, 2:5], dev)                          # contiguous device source, strided host target
    dev.synchronize()
    assert back[:, 2:5].tolist() == dev.get().tolist()
