# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A register bound, attached only when the compiler proves it free.

An entry that sits a few registers above a residency edge (46 registers hold
1280 threads on an A100, 40 hold 1536) can often be recompiled under a cap at
the edge. Whether it can is a compiler fact, so it is asked, never assumed: the
module's PTX is JIT-compiled again with ``CU_JIT_MAX_REGISTERS`` set to the
edge, and the bounded entry replaces the plain one only when it keeps the same
local-memory size (no new spill) and really fits the edge
(:func:`eagle._core.keep_bounded`). Otherwise the plain entry stands, and
:func:`entry_report` says why. A cubin image has nothing left to compile and is
never bounded.

The edge differs by device (an A100 gives 40 for the oscillator, an L4 already
holds it at 46), which is why this runs at load time on the device in use
rather than as a constant in the generated code.
"""

from __future__ import annotations

import ctypes
import importlib
import threading
from types import SimpleNamespace
from typing import NamedTuple

from . import _core
from ._launch_policy import _attr, _residency

__all__ = ["EntryReport", "entry", "entry_report"]

_CU_JIT_MAX_REGISTERS = 0
_CU_FUNC_ATTR_REGS = 4
_CU_FUNC_ATTR_LOCAL = 3
_ELF = b"\x7fELF"

_LOCK = threading.Lock()
_driver = None


class EntryReport(NamedTuple):
    """What the launch policy knows about one device entry."""

    name: str
    regs: int            #: registers per thread of the entry that launches
    local_bytes: int     #: its local (spill + stack) memory per thread
    bounded: bool        #: True when a register bound is in force
    edge: int            #: the residency edge in reach (0: none within 8 registers)
    resident_threads: int  #: threads per SM at ``regs``, 0 when unknown
    note: str            #: why a bound was not attached, empty when bounded or not tried


class _Bounded:
    """A function loaded under a register bound, in the shape a launch reads
    (``.kernel.ptr``) and the policy queries (``.attributes``)."""

    def __init__(self, module, handle):
        self._module = module
        self.kernel = SimpleNamespace(ptr=handle)

    @property
    def attributes(self) -> dict:
        return {"max_threads_per_block": _func_attr(self.kernel.ptr, 0),
                "shared_size_bytes": _func_attr(self.kernel.ptr, 1),
                "local_size_bytes": _func_attr(self.kernel.ptr, _CU_FUNC_ATTR_LOCAL),
                "num_regs": _func_attr(self.kernel.ptr, _CU_FUNC_ATTR_REGS)}

    def __del__(self):
        try:
            _cuda().cuModuleUnload(self._module)
        except Exception:
            pass


def _cuda():
    global _driver
    if _driver is None:
        _driver = ctypes.CDLL("libcuda.so.1")
        _driver.cuModuleLoadDataEx.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                                               ctypes.c_uint, ctypes.POINTER(ctypes.c_int),
                                               ctypes.POINTER(ctypes.c_void_p)]
        _driver.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                                                ctypes.c_char_p]
        _driver.cuModuleUnload.argtypes = [ctypes.c_void_p]
    return _driver


def _func_attr(handle, attr: int) -> int:
    from cupy_backends.cuda.api import driver

    return int(driver.funcGetAttribute(attr, handle))


def _plain_resources(fn) -> tuple:
    """``(registers, local bytes)`` of a loaded function."""
    attrs = fn.attributes
    return int(attrs["num_regs"]), int(attrs.get("local_size_bytes", 0))


def _load_bounded(loaded, name: str, cap: int):
    """``name`` from ``loaded``'s PTX JIT-compiled under ``cap`` registers, or
    ``None`` when the image cannot be (a cubin, an unreadable file, a driver
    refusal)."""
    import cupy

    try:
        image = open(loaded.image_path, "rb").read()
    except OSError:
        return None
    if image.startswith(_ELF):
        return None
    image += b"\0"
    cuda = _cuda()
    module, func = ctypes.c_void_p(), ctypes.c_void_p()
    options = (ctypes.c_int * 1)(_CU_JIT_MAX_REGISTERS)
    values = (ctypes.c_void_p * 1)(cap)
    with cupy.cuda.Device(loaded.device_id):
        if cuda.cuModuleLoadDataEx(ctypes.byref(module), image, 1, options, values) != 0:
            return None
        if cuda.cuModuleGetFunction(ctypes.byref(func), module, name.encode()) != 0:
            cuda.cuModuleUnload(module)
            return None
    return _Bounded(module, func.value)


def _decide(loaded, name: str, plain):
    """``(function that launches, EntryReport)`` for entry ``name``."""
    dev = importlib.import_module("eagle.launch")._device_props()
    regs, local = _plain_resources(plain)
    residency = _residency(dev)
    threads = _core.resident_warps(*residency, regs) * 32 if _attr(dev, "regsPerMultiprocessor") else 0
    edge = _core.register_edge(*residency, regs)
    if edge == 0:
        return plain, EntryReport(name, regs, local, False, 0, threads, "")
    bounded = _load_bounded(loaded, name, edge)
    if bounded is None:
        return plain, EntryReport(name, regs, local, False, edge, threads,
                                  "the image cannot be recompiled under a bound (a cubin)")
    b_regs, b_local = _plain_resources(bounded)
    if not _core.keep_bounded(regs, local, b_regs, b_local, edge):
        why = (f"bound {edge} gives {b_regs} registers and {b_local} B local "
               f"against {regs} and {local} B unbounded")
        return plain, EntryReport(name, regs, local, False, edge, threads, why)
    return bounded, EntryReport(name, b_regs, b_local, True, edge,
                                _core.resident_warps(*residency, b_regs) * 32, "")


def entry(loaded, name: str, plain):
    """The function to launch for entry ``name`` of ``loaded``'s module: its
    register-bounded recompile when one was proven spill-free, else ``plain``.
    Decided once per entry and kept with ``loaded``."""
    return _resolved(loaded, name, plain)[0]


def entry_report(loaded, name: str, plain) -> EntryReport:
    """The :class:`EntryReport` of entry ``name`` (see :func:`entry`)."""
    return _resolved(loaded, name, plain)[1]


def _resolved(loaded, name: str, plain):
    with _LOCK:
        cache = loaded.__dict__.setdefault("_bounded_entries", {})
        if name not in cache:
            cache[name] = _decide(loaded, name, plain)
        return cache[name]
