# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""One OpenMP runtime per process: the one aether-dsc ships, shared by aether and eagle, running threads."""
from __future__ import annotations

import ctypes
import os
import re
import sys

import pytest

import eagle
import eagle.arrays as xp
import eagle.fft as efft

_RUNTIME = re.compile(r"lib(gomp|omp|iomp5)[^/]*\.(so(\.\d+)*|dylib)$")


def _loaded_runtimes():
    """The distinct OpenMP runtime files mapped into this process."""
    if sys.platform == "darwin":
        dyld = ctypes.CDLL(None)
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        names = [dyld._dyld_get_image_name(i).decode() for i in range(dyld._dyld_image_count())]
    else:
        names = [line.split()[-1] for line in open("/proc/self/maps") if "/" in line.split()[-1]]
    return {os.path.realpath(n) for n in names if _RUNTIME.search(n)}


def test_aether_and_eagle_share_one_openmp_runtime_and_use_threads():
    aether_dsc = pytest.importorskip("aether_dsc")
    import aether

    shipped = aether_dsc.openmp()
    if shipped is None:
        pytest.skip("this aether install ships no OpenMP runtime (a source checkout)")
    # an eagle host graph around aether's host FFT, big enough to fan out over the team
    shape = (256, 4096)
    x = xp.ones(shape + (2,), "float64", "cpu")
    out = xp.zeros(shape + (2,), "float64", "cpu")
    eagle.Graph().add_node(efft.c2c(x, out, (1,), forward=True)).build().launch()
    assert eagle.exec.fold("sum", [1.0, 2.0]) == 3.0
    assert aether.__file__
    runtimes = _loaded_runtimes()
    assert runtimes == {os.path.realpath(str(shipped.library))}, runtimes
    openmp = ctypes.CDLL(str(shipped.library))
    threads = openmp.omp_get_max_threads()
    if (os.cpu_count() or 1) > 1 and "OMP_NUM_THREADS" not in os.environ:
        assert threads > 1, threads
