# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Make CuPy's run-time compiler find its builtins library.

NVIDIA's pip wheels put ``libnvrtc`` and ``libnvrtc-builtins`` in a folder that is not on
the loader's search path. NVRTC opens the builtins library by name on its first compile,
which works only while nothing has disturbed the lookup: starting JAX's CUDA plugin first
breaks it ("failed to open libnvrtc-builtins.so.13.0"). Loading the builtins library once,
by full path, before any compile makes that name lookup find the library already loaded.
"""

import ctypes
import glob
import importlib.util
import os
import sys


def preload_builtins() -> tuple:
    """Load every ``libnvrtc-builtins`` the NVIDIA wheels installed; the paths loaded.

    Linux only, and only when CuPy is installed; nothing to do with a system CUDA toolkit
    (its libraries are on the search path already). A library that will not load is skipped:
    CuPy then reports its own error on the first compile, as it would without eagle."""
    if not sys.platform.startswith("linux") or importlib.util.find_spec("cupy") is None:
        return ()
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return ()
    loaded = []
    for root in spec.submodule_search_locations:
        # CUDA 13 wheels: nvidia/cu13/lib; CUDA 12 wheels: nvidia/cuda_nvrtc/lib
        for path in sorted(glob.glob(os.path.join(root, "*", "lib", "libnvrtc-builtins.so.*"))):
            if ".alt." in os.path.basename(path):
                continue
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                continue
            loaded.append(path)
    return tuple(loaded)
