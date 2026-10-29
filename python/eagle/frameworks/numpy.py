# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The numpy bridge: the only place eagle's interop touches numpy.

numpy is imported inside the functions below. A host :class:`eagle.Array` speaks the buffer protocol, so
numpy reads it zero-copy; a device array is copied to the host first."""

from __future__ import annotations


def from_array(a):
    """The :class:`eagle.Array` ``a`` as a numpy array (a device array is copied to the host)."""
    import numpy as np

    return np.asarray(a if a.device in ("cpu", "pinned") else a.get())


def host_ptr(x, writable: bool) -> int:
    """Raw data pointer of a C-contiguous host buffer (the numpy half of :func:`eagle.interop.host_ptr`)."""
    import numpy as np

    a = np.asarray(x)
    if not a.flags.c_contiguous:
        raise ValueError("host_ptr: array must be C-contiguous")
    if writable and not a.flags.writeable:
        from ..interop import BufferRefused

        raise BufferRefused("host_ptr: the buffer is read-only, so a kernel cannot write it")
    return int(a.ctypes.data)


def from_eagle(a):
    """The :class:`eagle.Array` ``a`` as a numpy array (zero-copy for host memory; a device array is copied to the host)."""
    return from_array(a)
