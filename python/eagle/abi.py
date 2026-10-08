# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Host-side ABI mirrors, by-value argument builders, and the ABI version tag.

The numpy POD mirrors of the by-value kernel-parameter types -- ``GRef`` and
``HandleT`` -- plus the ``make_*`` helpers that pack a cupy array's device
pointer into one. The C++ twin is ``plugin/gref_abi.h``; every loader in
both languages validates an artifact's ABI tag via :func:`check_aether_abi`
before binding any by-value struct.
"""

from __future__ import annotations

import numpy as np

#: The by-value POD layout contract between producers and loaders. Changing
#: it needs a coordinated bump of the C++ ``EAGLE_AETHER_ABI`` macro.
ABI_VERSION = "aether-abi/1"

#: The two generations by name (C++ ``EAGLE_AETHER_ABI`` /
#: ``EAGLE_AETHER_ABI_V2``). ``ABI_TAG_V2`` adds the int64 partition triple
#: after the role args and an exported ``eagle_layout_sizes``.
ABI_TAG_V1 = ABI_VERSION
ABI_TAG_V2 = "aether-abi/2"

#: The tags :func:`check_aether_abi` accepts today (both generations).
ACCEPTED_ABI_TAGS = (ABI_TAG_V1, ABI_TAG_V2)


def abi_version_of(tag) -> int:
    """The ABI generation ``tag`` names: 1, 2, or 0 for anything else. A
    plain Python reimplementation of the C++ ``abi_version_of``, so that
    :mod:`eagle.abi` never forces the compiled extension to load."""
    if tag == ABI_TAG_V1:
        return 1
    if tag == ABI_TAG_V2:
        return 2
    return 0


def check_aether_abi(meta: dict, *, kind: str, name: str) -> None:
    """Reject an artifact whose ``meta["aether_abi"]`` is not one of
    :data:`ACCEPTED_ABI_TAGS` (absent/empty included). ``kind`` names the
    artifact in the error (e.g. 'vector plugin', 'plugin manifest')."""
    abi = meta.get("aether_abi")
    if abi_version_of(abi) == 0:
        raise ValueError(
            f"{kind} {name!r} was built for AETHER ABI {abi!r}, but this "
            f"loader expects one of {ACCEPTED_ABI_TAGS!r}; rebuild the plugin"
        )


#: DLPack device-type codes a mirror's device tag carries (``DLDeviceType``).
DEVICE_CPU = 1
DEVICE_CUDA = 2

# Host mirror of ``aether::Array<double,3>::ViewT`` (sizeof == 40, alignof
# == 8; verified by ``tests/test_gref_layout.py``). Width-independent.
GREF_DTYPE = np.dtype(
    {
        "names": [
            "data_",
            "samples_",
            "compStride_",
            "sampleStride_",
            "deviceType_",
            "deviceId_",
        ],
        "formats": [np.uintp, np.uint64, np.uint64, np.uint64, np.int32, np.int32],
        "offsets": [0, 8, 16, 24, 32, 36],
        "itemsize": 40,
    }
)

# Host mirror of ``aether::Array<T>::ViewT``, GREF_DTYPE's rank-1 sibling
# (sizeof == 32; verified by ``tests/test_handle_layout.py``).
HANDLE_DTYPE = np.dtype(
    {
        "names": ["data", "samples", "stride", "deviceType", "deviceId"],
        "formats": [np.uintp, np.uint64, np.uint64, np.int32, np.int32],
        "offsets": [0, 8, 16, 24, 28],
        "itemsize": 32,
    }
)

def make_gref(arr, n):
    """Build a by-value ``GRef`` struct over a contiguous ``(3, N)`` cupy
    array: ``compStride_ == samples_ == n``, ``sampleStride_ == 1``."""
    g = np.zeros((), dtype=GREF_DTYPE)
    g["data_"] = arr.data.ptr
    g["samples_"] = n
    g["compStride_"] = n
    g["sampleStride_"] = 1
    g["deviceType_"] = DEVICE_CUDA
    return g


def make_handle(arr, n):
    """Build a by-value scalar-array ``HandleT`` over a contiguous ``(N,)``
    array; ``n`` is required since the struct carries its own extent."""
    h = np.zeros((), dtype=HANDLE_DTYPE)
    h["data"] = arr.data.ptr
    h["samples"] = n
    h["stride"] = 1
    h["deviceType"] = DEVICE_CUDA
    return h
