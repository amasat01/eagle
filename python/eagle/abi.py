# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Host-side ABI mirrors, by-value argument builders, and the ABI version tag.

The ctypes POD mirrors of the by-value kernel-parameter types -- ``GRef`` and
``HandleT`` -- plus the ``make_*`` helpers that pack an array's device
pointer into one. The C++ twin is ``plugin/gref_abi.h``; every loader in
both languages validates an artifact's ABI tag via :func:`check_aether_abi`
before binding any by-value struct.
"""

from __future__ import annotations

import ctypes

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

class GRef(ctypes.Structure):
    """Host mirror of ``aether::Array<double,3>::ViewT`` (sizeof == 40, alignof == 8; verified by
    ``tests/test_gref_layout.py``). Width-independent."""

    _fields_ = [
        ("data_", ctypes.c_size_t),
        ("samples_", ctypes.c_uint64),
        ("compStride_", ctypes.c_uint64),
        ("sampleStride_", ctypes.c_uint64),
        ("deviceType_", ctypes.c_int32),
        ("deviceId_", ctypes.c_int32),
    ]


class HandleT(ctypes.Structure):
    """Host mirror of ``aether::Array<T>::ViewT``, GRef's rank-1 sibling (sizeof == 32; verified by
    ``tests/test_handle_layout.py``)."""

    _fields_ = [
        ("data", ctypes.c_size_t),
        ("samples", ctypes.c_uint64),
        ("stride", ctypes.c_uint64),
        ("deviceType", ctypes.c_int32),
        ("deviceId", ctypes.c_int32),
    ]


GREF_SIZE = ctypes.sizeof(GRef)
HANDLE_SIZE = ctypes.sizeof(HandleT)


def device_ptr(arr) -> int:
    """The device address of ``arr``: an :class:`eagle.Array` (``.ptr``) or a cupy array (``.data.ptr``)."""
    ptr = getattr(arr, "ptr", None)
    return arr.data.ptr if ptr is None else ptr


def make_gref(arr, n):
    """Build a by-value ``GRef`` over a contiguous ``(3, N)`` device array (anything with ``.ptr``):
    ``compStride_ == samples_ == n``, ``sampleStride_ == 1``."""
    return GRef(device_ptr(arr), n, n, 1, DEVICE_CUDA, 0)


def make_handle(arr, n):
    """Build a by-value scalar-array ``HandleT`` over a contiguous ``(N,)`` device array; ``n`` is
    required since the struct carries its own extent."""
    return HandleT(device_ptr(arr), n, 1, DEVICE_CUDA, 0)
