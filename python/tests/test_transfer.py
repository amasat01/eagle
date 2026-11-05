# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``eagle.copy``, ``eagle.pinned`` and host nodes: a graph across the host and the
device, and the same builder over numpy planes on a machine without a GPU."""

import numpy as np
import pytest

import eagle
from eagle.graph import opaque


# --- host only (no GPU needed) --- #
def test_copy_over_numpy_planes_is_a_host_copy_and_x_into_x_does_nothing():
    a, b = np.arange(4.0), np.zeros(4)
    g = eagle.Graph().add_node(eagle.copy(a, into=b)).add_node(eagle.copy(b, into=b)).build()
    g.launch()
    assert g.backend == "cpu"
    np.testing.assert_array_equal(b, a)


def test_the_builder_runs_over_numpy_planes():
    @eagle.Graph.builder
    def step(h_in, h_mid, h_out):
        g = eagle.Graph()
        eagle.opaque(lambda: h_in.__imul__(2.0), reads=(h_in,), writes=(h_in,))
        h_mid = eagle.copy(h_in, into=h_mid)
        g.result = eagle.copy(h_mid, into=h_out)
        return g

    h_in, h_mid, h_out = np.arange(4.0), np.zeros(4), np.zeros(4)
    g = step(h_in, h_mid, h_out)
    g.launch(2)
    np.testing.assert_array_equal(h_out, np.arange(4.0) * 4)


def test_copy_refuses_a_shape_a_dtype_and_a_strided_array():
    with pytest.raises(ValueError, match="one shape and dtype"):
        eagle.copy(np.zeros(4), into=np.zeros(5))
    with pytest.raises(ValueError, match="one shape and dtype"):
        eagle.copy(np.zeros(4), into=np.zeros(4, np.float32))
    with pytest.raises(ValueError, match="not contiguous"):
        eagle.copy(np.zeros(8)[::2], into=np.zeros(4))
    with pytest.raises(TypeError, match="must be an array"):
        eagle.copy(1.0, into=np.zeros(4))


def test_pinned_is_an_eagle_array_of_the_shape_and_dtype():
    p = eagle.pinned((3, 5), "float32")
    assert p.shape == (3, 5) and p.dtype == "float32" and p.is_contiguous
    p.fill(1.5)
    assert np.asarray(p).sum() == 3 * 5 * 1.5
    assert eagle.pinned(0).shape == (0,)


def test_the_reverse_of_a_copy_is_the_copy_back():
    from eagle.diff import reverse  # noqa: F401  (the transpose is registered with the kind)
    from eagle.graph import get_node_kind

    assert get_node_kind("copy").transpose is not None


# --- GPU --- #
try:
    import cupy as cp
    _devices = cp.cuda.runtime.getDeviceCount()
except Exception:                      # no cupy, or no device: the GPU rows skip
    cp, _devices = None, 0
gpu = pytest.mark.gpu
needs_gpu = pytest.mark.skipif(not _devices, reason="needs a GPU")


def _bump(arr):
    return opaque(lambda: cp.add(arr, 1.0, out=arr), reads=[arr], writes=[arr], name="bump")


def _host_scale(h, k):
    return opaque(lambda: eagle.arrays.multiply(h, k, out=h), reads=[h], writes=[h], name="host_scale")


@gpu
@needs_gpu
def test_a_device_host_device_graph_is_one_capture_and_correct():
    d = cp.arange(8.0)
    h = eagle.pinned(8)
    g = eagle.Graph()
    g.add_node(_bump(d))
    g.add_node(eagle.copy(d, into=h))
    g.add_node(_host_scale(h, 2.0))
    g.add_node(eagle.copy(h, into=d))
    g.add_node(_bump(d))
    g.build()
    assert g.backend == "gpu"
    assert g.schedule.count("capture") == 5 and "eager_host" not in g.schedule
    g.launch()
    np.testing.assert_array_equal(d.get(), (np.arange(8.0) + 1) * 2 + 1)
    g.launch(3)       # three replays equal three single launches
    ref = (np.arange(8.0) + 1) * 2 + 1
    for _ in range(3):
        ref = (ref + 1) * 2 + 1
    np.testing.assert_array_equal(d.get(), ref)


@gpu
@needs_gpu
def test_a_pageable_numpy_array_copies_too():
    d, h = cp.zeros(4), np.arange(4.0)
    g = eagle.Graph().add_node(eagle.copy(h, into=d)).build()
    g.launch()
    np.testing.assert_array_equal(d.get(), h)


@gpu
@needs_gpu
def test_the_source_of_a_copy_rebinds_without_a_recapture():
    d, h1, h2 = cp.zeros(4), np.arange(4.0), np.arange(4.0) + 10
    g = eagle.Graph().add_node(eagle.copy(h1, into=d)).build()
    g.launch()
    g._rebind([(h1, h2)])
    g.launch()
    np.testing.assert_array_equal(d.get(), h2)


@gpu
@needs_gpu
def test_a_copy_in_a_parallel_branch_and_the_builder_spelling():
    @eagle.Graph.builder
    def pipelined(h_next, staged, batch, out):
        g = eagle.Graph()
        with g.parallel() as p:
            with p.branch():
                eagle.copy(h_next, into=staged)
            with p.branch():
                eagle.opaque(lambda: cp.multiply(batch, 2.0, out=out), reads=[batch], writes=[out])
        return g

    h = eagle.pinned(4)
    h[...] = [1, 2, 3, 4]
    staged, batch, out = cp.zeros(4), cp.arange(4.0), cp.zeros(4)
    g = pipelined(h, staged, batch, out)
    g.launch()
    np.testing.assert_array_equal(staged.get(), h)
    np.testing.assert_array_equal(out.get(), np.arange(4.0) * 2)


@gpu
@needs_gpu
def test_a_host_node_that_raises_is_raised_by_the_next_launch():
    d, h = cp.zeros(4), eagle.pinned(4)

    def boom():
        raise RuntimeError("check failed")

    g = (eagle.Graph().add_node(eagle.copy(d, into=h))
         .add_node(opaque(boom, reads=[h], writes=[h])).build())
    with pytest.raises(RuntimeError, match="check failed"):
        g.launch()


@gpu
@needs_gpu
def test_a_host_node_failing_on_every_replay_raises_once_and_the_next_launch_is_clean():
    d, h = cp.zeros(4), eagle.pinned(4)
    broken = [True]

    def check():
        if broken[0]:
            raise RuntimeError("check failed")

    g = (eagle.Graph().add_node(eagle.copy(d, into=h))
         .add_node(opaque(check, reads=[h], writes=[h])).build())
    with pytest.raises(RuntimeError, match="check failed"):
        g.launch(3)
    broken[0] = False
    g.launch(2)          # nothing left over from the three failed replays
    g.synchronize()


@gpu
@needs_gpu
def test_a_node_with_planes_on_both_sides_is_refused_with_the_copy_to_add():
    d, h = cp.zeros(4), np.zeros(4)
    g = eagle.Graph().add_node(opaque(lambda: None, reads=[h], writes=[d], name="mix"))
    with pytest.raises(ValueError, match=r"node 'mix'.*eagle\.copy\(<.*>, into=<a device array>\)"):
        g.build()


@gpu
@needs_gpu
def test_a_host_node_in_a_device_body_is_refused_a_copy_is_not():
    d, h, cell = cp.zeros(4), np.zeros(4), cp.ones(1, dtype=cp.uint32)
    inner = opaque(lambda: None, reads=[h], writes=[h], name="check")
    g = eagle.Graph().add_node(eagle.skippable(inner, eagle.SkipGuard.nonzero(cell)))
    with pytest.raises(ValueError, match=r"inside the device if/for body.*eagle\.copy is allowed"):
        g.build()
    pinned = eagle.pinned(4)
    pinned[...] = 3.0
    ok = eagle.Graph().add_node(eagle.skippable(eagle.copy(pinned, into=d), eagle.SkipGuard.nonzero(cell)))
    ok.build().launch()
    np.testing.assert_array_equal(d.get(), pinned)


@gpu
@needs_gpu
def test_a_pageable_copy_in_a_device_body_says_to_pin():
    d, h, cell = cp.zeros(4), np.zeros(4), cp.ones(1, dtype=cp.uint32)
    g = eagle.Graph().add_node(eagle.skippable(eagle.copy(h, into=d), eagle.SkipGuard.nonzero(cell)))
    with pytest.raises(ValueError, match=r"eagle\.pinned"):
        g.build()
    # the refusal is reported once: no CUDA error is left pending for the next launch to be blamed for
    eagle._stream.check_launch("the launch after the refusal")


@gpu
@needs_gpu
def test_one_hawk_kernel_runs_on_the_device_and_on_the_host_by_its_planes():
    hawk = pytest.importorskip("hawk")

    @hawk.kernel
    def tr_twice(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = 2.0 * x + 1.0

    twice = eagle.deploy(tr_twice, targets=("host", "cuda"),
                         cache_dir=str(__import__("os").environ.get("STREAM_SHM", "/tmp")) + "/transfer_hawk")

    @eagle.Graph.builder
    def across(d_in, d_mid, h_mid, h_out, d_out):
        g = eagle.Graph()
        twice(d_in, d_mid)                              # a device node
        eagle.copy(d_mid, into=h_mid)
        twice(h_mid, h_out)                             # the same kernel on host planes: a host node
        eagle.copy(h_out, into=d_out)
        return g

    n = 1 << 12
    x = np.random.default_rng(0).random(n)
    d_in, d_mid, d_out = cp.asarray(x), cp.zeros(n), cp.zeros(n)
    h_mid, h_out = eagle.pinned(n), eagle.pinned(n)
    g = across(d_in, d_mid, h_mid, h_out, d_out)
    assert g.backend == "gpu" and g.schedule.count("capture") == 4
    g.launch(3)
    np.testing.assert_allclose(d_out.get(), 2.0 * (2.0 * x + 1.0) + 1.0, rtol=1e-15)


def _child(code, **env):
    import os
    import subprocess
    import sys

    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env={**os.environ, **env})


def test_pinned_is_a_host_array_when_no_device_is_visible():
    done = _child("import eagle\np = eagle.pinned((2, 3), 'float32')\n"
                  "assert type(p) is eagle.arrays.Array and p.device == 'cpu' and p.shape == (2, 3)\n",
                  CUDA_VISIBLE_DEVICES="")
    assert done.returncode == 0, done.stderr


MIXED_AT_EXIT = (
    "import cupy as cp, eagle\n"
    "{unhook}"
    "d, h = cp.zeros(4), eagle.pinned(4)\n"
    "g = (eagle.Graph().add_node(eagle.copy(d, into=h))\n"
    "     .add_node(eagle.opaque(lambda: None, reads=[h], writes=[h])).build())\n"
    "g.launch(2)\n")


@gpu
@needs_gpu
def test_a_mixed_graph_leaves_nothing_leaked_at_interpreter_exit():
    done = _child(MIXED_AT_EXIT.format(unhook=""))
    assert done.returncode == 0 and "leaked" not in done.stderr, done.stderr
