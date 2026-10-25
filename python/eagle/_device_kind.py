# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Which backend kind an array lives under, and which GPU kind is installed.

The kind of an array is read from its DLPack device type, so the core makes no
assumption about the framework that owns it. ``"host"`` is any CPU memory; a
GPU kind names the backend plugin (``libeagle_<kind>``) that serves it."""

import os
import sys
from pathlib import Path

#: DLPack device types: host memory (``kDLCPU``, ``kDLCUDAHost``) and the GPU kinds.
_DLPACK_KINDS = {1: "host", 3: "host", 2: "cuda", 8: "metal", 10: "hip"}
GPU_KINDS = ("cuda", "hip", "metal")
_EXT = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")


def device_kind(array) -> str:
    """``"host"`` or the GPU kind (``"cuda"``, ``"hip"``, ``"metal"``) ``array``
    lives on, from its ``__dlpack_device__``; an object without one is host
    memory."""
    if type(array).__module__.split(".")[0] == "cupy":
        return "cuda"
    dlpack_device = getattr(array, "__dlpack_device__", None)
    if dlpack_device is None:
        return "host"
    code = int(dlpack_device()[0])
    try:
        return _DLPACK_KINDS[code]
    except KeyError:
        raise ValueError(f"eagle: DLPack device type {code} has no eagle backend") from None


def installed_gpu_kinds() -> tuple:
    """The GPU kinds with a backend plugin on hand: an ``$EAGLE_BACKEND_<KIND>``
    file or a ``libeagle_<kind>`` beside the core."""
    here = Path(__file__).parent
    return tuple(k for k in GPU_KINDS
                 if Path(os.environ.get(f"EAGLE_BACKEND_{k.upper()}", here / f"libeagle_{k}{_EXT}")).is_file())


def gpu_kind() -> str:
    """The one installed GPU kind (``"cuda"`` when none is found, the only kind
    shipped today). Two installed kinds are refused: the caller names one."""
    found = installed_gpu_kinds()
    if len(found) > 1:
        raise ValueError(f"eagle: backend='gpu' is ambiguous, {', '.join(found)} are installed; "
                         f"name one of them as the backend")
    return found[0] if found else "cuda"
