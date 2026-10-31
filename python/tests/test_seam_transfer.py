# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Copies, pinned host memory and host functions through the backend seam.

On a GPU: every direction of ``memcpy`` (also as a node of a capture, replayed), the
no-op of a copy onto itself, pinned allocation, and a host function that runs on every
replay. Without a backend (``EAGLE_BACKEND_CUDA`` naming no file, in a child process)
the same calls are plain host memory: the host twin.
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

from eagle import _core
from eagle.frameworks.cupy import wrap_external_stream


def _host_array(ptr: int, n: int) -> np.ndarray:
    return np.ctypeslib.as_array((ctypes.c_double * n).from_address(ptr))


def test_the_copy_kinds_are_the_seam_values():
    assert (_core.COPY_H2D, _core.COPY_D2H, _core.COPY_D2D, _core.COPY_H2H) == (0, 1, 2, 3)


@pytest.mark.gpu
def test_every_direction_copies_and_a_copy_onto_itself_is_nothing():
    import cupy as cp

    n = 64
    ptr = _core.host_alloc(n * 8)
    try:
        h = _host_array(ptr, n)
        h[:] = np.arange(n)
        d, d2 = cp.zeros(n), cp.zeros(n)
        back = np.zeros(n)
        _core.memcpy(d.data.ptr, ptr, n * 8, _core.COPY_H2D)
        _core.memcpy(d2.data.ptr, d.data.ptr, n * 8, _core.COPY_D2D)
        _core.memcpy(back.ctypes.data, d2.data.ptr, n * 8, _core.COPY_D2H)
        cp.cuda.Device().synchronize()
        assert (back == np.arange(n)).all()
        h2 = np.zeros(n)
        _core.memcpy(h2.ctypes.data, ptr, n * 8, _core.COPY_H2H)
        cp.cuda.Device().synchronize()
        assert (h2 == np.arange(n)).all()
        _core.memcpy(ptr, ptr, n * 8, _core.COPY_H2H)  # the same storage: nothing to do
        _core.memcpy(ptr, 0, 0, _core.COPY_H2H)  # no bytes: nothing to do
        with pytest.raises(ValueError, match="kind"):
            _core.memcpy(ptr, h2.ctypes.data, 8, 9)
    finally:
        _core.host_free(ptr)


@pytest.mark.gpu
def test_a_captured_copy_and_host_function_run_on_every_replay():
    import cupy as cp

    n = 32
    ptr = _core.host_alloc(n * 8)
    try:
        h = _host_array(ptr, n)
        h[:] = 1.0
        d = cp.zeros(n)
        inc = cp.ElementwiseKernel("", "float64 x", "x += 1.0", "eagle_test_inc")
        hits = []

        def host(*_):
            h[:] *= 10.0  # host memory only: a host function must not call CUDA
            hits.append(1)

        fn = _core.HostFunc(host)
        s = _core.Stream(non_blocking=True)
        cap = _core.StreamCapturer(s.ptr())
        with wrap_external_stream(s.ptr()):
            cap.begin()
            _core.memcpy(d.data.ptr, ptr, n * 8, _core.COPY_H2D, s.ptr())
            inc(d)
            _core.memcpy(ptr, d.data.ptr, n * 8, _core.COPY_D2H, s.ptr())
            fn.enqueue(s.ptr())
            captured = cap.end()
        assert hits == []  # captured, not run
        launcher = _core.Graph.from_captured(captured).launcher()
        launcher.stream(s.ptr())
        for k in range(3):
            launcher.launch()
            launcher.synchronize()
            assert len(hits) == k + 1
        # h: 1 -> (1+1)*10 = 20 -> (20+1)*10 = 210 -> 2110
        assert (h == 2110.0).all()
    finally:
        _core.host_free(ptr)


HOST_TWIN = textwrap.dedent(
    """
    import ctypes
    import numpy as np
    from eagle import _core

    n = 16
    ptr = _core.host_alloc(n * 8)
    assert ptr and ptr % 64 == 0
    a = np.ctypeslib.as_array((ctypes.c_double * n).from_address(ptr))
    a[:] = np.arange(n)
    b = np.zeros(n)
    for kind in (_core.COPY_H2D, _core.COPY_D2H, _core.COPY_D2D, _core.COPY_H2H):
        b[:] = 0
        _core.memcpy(b.ctypes.data, ptr, n * 8, kind)
        assert (b == np.arange(n)).all()
    _core.memcpy(ptr, ptr, n * 8, _core.COPY_H2H)
    assert _core.host_alloc(0) == 0
    hits = []
    _core.HostFunc(lambda: hits.append(1)).enqueue(0)
    assert hits == [1]
    try:
        _core.HostFunc(lambda: 1 / 0).enqueue(0)
    except ZeroDivisionError:
        pass
    else:
        raise SystemExit("the host twin must raise a host function's exception")
    _core.host_free(ptr)
    _core.host_free(0)
    print("ok")
    """
)


def test_without_a_backend_the_same_calls_are_plain_host_memory():
    env = dict(os.environ, EAGLE_BACKEND_CUDA="/nonexistent/libeagle_cuda.so")
    run = subprocess.run(
        [sys.executable, "-c", HOST_TWIN], capture_output=True, text=True, env=env, timeout=60
    )
    assert run.returncode == 0 and run.stdout.strip() == "ok", run.stdout + run.stderr
