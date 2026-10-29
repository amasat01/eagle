# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A register bound, attached only when the launch needs it and the compiler proves it free.

An entry that sits a few registers above a residency edge (46 registers hold
1280 threads on an A100, 40 hold 1536) can often be recompiled under a cap at
the edge. A bound is tried only when the launch asks for more threads per SM
than the unbounded entry holds (:func:`eagle._core.bound_demanded`): an entry
whose demand already fits runs as compiled. Whether a bound is free is a
compiler fact, so it is asked, never assumed: the module's PTX is JIT-compiled
again with ``CU_JIT_MAX_REGISTERS`` set to the edge, and the bounded entry
replaces the plain one only when it keeps the same local-memory size (no new
spill), really fits the edge and its code grew by at most
``BOUND_TEXT_GROWTH_PERCENT`` (:func:`eagle._core.keep_bounded`) -- a bound
that rematerialises values instead of spilling them keeps the local size and
issues more instructions. Otherwise the plain entry stands, and
:func:`entry_report` says why. A cubin image has nothing left to compile and is
never bounded.

The edge differs by device (an A100 gives 40 for the oscillator, an L4 already
holds it at 46), which is why this runs at load time on the device in use
rather than as a constant in the generated code.
"""

from __future__ import annotations

import importlib
import struct
import threading
from types import SimpleNamespace
from typing import NamedTuple

from . import _core
from ._launch_policy import _attr, _residency

__all__ = ["EntryReport", "elf_section_size", "entry", "entry_report"]

_CU_FUNC_ATTR_REGS = 4
_CU_FUNC_ATTR_LOCAL = 3
_ELF = b"\x7fELF"

_LOCK = threading.Lock()


class EntryReport(NamedTuple):
    """What the launch policy knows about one device entry."""

    name: str
    regs: int            #: registers per thread of the entry that launches
    local_bytes: int     #: its local (spill + stack) memory per thread
    bounded: bool        #: True when a register bound is in force
    edge: int            #: the residency edge in reach (0: none within 8 registers)
    resident_threads: int  #: threads per SM at ``regs``, 0 when unknown
    text_bytes: int      #: the entry's ELF ``.text`` size, 0 when not measured
    demand: int          #: threads per SM the launch asks for, 0 when the entry reports none
    note: str            #: why a bound was not attached, empty when bounded or not tried


class _Bounded:
    """A function loaded under a register bound, in the shape a launch reads
    (``.kernel.ptr``) and the policy queries (``.attributes``)."""

    def __init__(self, module, handle, device):
        self._module, self._device = module, device
        self.kernel = SimpleNamespace(ptr=handle)

    @property
    def attributes(self) -> dict:
        return {"max_threads_per_block": _func_attr(self.kernel.ptr, 0),
                "shared_size_bytes": _func_attr(self.kernel.ptr, 1),
                "local_size_bytes": _func_attr(self.kernel.ptr, _CU_FUNC_ATTR_LOCAL),
                "num_regs": _func_attr(self.kernel.ptr, _CU_FUNC_ATTR_REGS)}

    def __del__(self):
        try:
            _core.module_release(self._module, self._device)
        except Exception:
            pass


def _func_attr(handle, attr: int) -> int:
    from . import _core

    return int(_core.function_attribute(int(handle), attr))


def _plain_resources(fn) -> tuple:
    """``(registers, local bytes)`` of a loaded function."""
    attrs = fn.attributes
    return int(attrs["num_regs"]), int(attrs.get("local_size_bytes", 0))


def _load_bounded(loaded, name: str, cap: int, device=None):
    """``name`` from ``loaded``'s PTX JIT-compiled under ``cap`` registers, or
    ``None`` when the image cannot be (a cubin, an unreadable file, a driver
    refusal)."""
    try:
        image = open(loaded.image_path, "rb").read()
    except OSError:
        return None
    if image.startswith(_ELF):
        return None
    device = loaded.device_id if device is None else device
    try:
        module = _core.module_load_jit(image + b"\0", cap, device)
    except Exception:           # a driver refusal
        return None
    try:
        func = _core.module_function(module, name)
    except Exception:
        _core.module_release(module, device)
        return None
    return _Bounded(module, func, device)


def elf_section_size(image: bytes, section: str) -> int:
    """Size in bytes of ELF64 section ``section`` in ``image``, ``0`` when the
    image is not a little-endian ELF64 or has no such section."""
    if image[:4] != _ELF or len(image) < 64 or image[4] != 2 or image[5] != 1:
        return 0
    shoff, = struct.unpack_from("<Q", image, 0x28)
    shentsize, shnum, shstrndx = struct.unpack_from("<HHH", image, 0x3A)

    def header(i):
        return struct.unpack_from("<IIQQQQ", image, shoff + i * shentsize)  # name type flags addr offset size

    try:
        names_offset = header(shstrndx)[4]
        for i in range(shnum):
            name, _, _, _, _, size = header(i)
            start = names_offset + name
            if image[start:image.index(b"\0", start)] == section.encode():
                return size
    except (struct.error, ValueError):
        return 0
    return 0


def _text_bytes(loaded, name: str, cap: int, device=None) -> int:
    """The ``.text`` size of entry ``name`` compiled from ``loaded``'s PTX under
    ``cap`` registers (0: unbounded), or ``0`` when it cannot be measured."""
    try:
        image = open(loaded.image_path, "rb").read()
    except (OSError, AttributeError):
        return 0
    if image.startswith(_ELF):
        return elf_section_size(image, f".text.{name}")
    try:
        cubin = _core.link_ptx(image + b"\0", cap, loaded.device_id if device is None else device)
    except Exception:       # noqa: BLE001 -- an unmeasurable entry reads as 0
        return 0
    return elf_section_size(cubin, f".text.{name}")


def _decide(loaded, name: str, plain, demand=None, device=None):
    """``(function that launches, EntryReport)`` for entry ``name``. ``demand``
    is the threads per SM the launch asks for; ``None`` asks for the whole
    residency the edge would give (a hardware-scheduled entry)."""
    props = importlib.import_module("eagle.launch")._device_props
    dev = props() if device is None else props(device)
    on = () if device is None else (device,)    # None: the device the kernel was loaded on
    regs, local = _plain_resources(plain)
    residency = _residency(dev)
    threads = _core.resident_warps(*residency, regs) * 32 if _attr(dev, "regsPerMultiprocessor") else 0
    edge = _core.register_edge(*residency, regs)
    if edge == 0:
        return plain, EntryReport(name, regs, local, False, 0, threads, 0, demand or 0, "")
    if demand is None:
        demand = _core.resident_warps(*residency, edge) * 32
    if not _core.bound_demanded(demand, threads):
        return plain, EntryReport(name, regs, local, False, edge, threads, 0, demand,
                                  f"no bound: the launch asks for {demand} threads per SM and "
                                  f"{threads} are resident")
    bounded = _load_bounded(loaded, name, edge, *on)
    if bounded is None:
        return plain, EntryReport(name, regs, local, False, edge, threads, 0, demand,
                                  "the image cannot be recompiled under a bound (a cubin)")
    b_regs, b_local = _plain_resources(bounded)
    text, b_text = _text_bytes(loaded, name, 0, *on), _text_bytes(loaded, name, edge, *on)
    if not _core.keep_bounded(regs, local, text, b_regs, b_local, b_text, edge):
        why = (f"bound {edge} gives {b_regs} registers, {b_local} B local and {b_text} B of code "
               f"against {regs}, {local} B and {text} B unbounded")
        return plain, EntryReport(name, regs, local, False, edge, threads, text, demand, why)
    return bounded, EntryReport(name, b_regs, b_local, True, edge,
                                _core.resident_warps(*residency, b_regs) * 32, b_text, demand, "")


def entry(loaded, name: str, plain, demand=None, device=None):
    """The function to launch for entry ``name`` of ``loaded``'s module: its
    register-bounded recompile when the launch's ``demand`` (threads per SM;
    ``None``: the whole residency a bound would give) exceeds what ``plain``
    holds and a bound was proven free, else ``plain``. Decided once per entry
    and kept with ``loaded``."""
    return _resolved(loaded, name, plain, demand, device)[0]


def entry_report(loaded, name: str, plain, demand=None, device=None) -> EntryReport:
    """The :class:`EntryReport` of entry ``name`` (see :func:`entry`)."""
    return _resolved(loaded, name, plain, demand, device)[1]


def _resolved(loaded, name: str, plain, demand=None, device=None):
    with _LOCK:
        cache = loaded.__dict__.setdefault("_bounded_entries", {})
        if (name, device) not in cache:
            cache[(name, device)] = _decide(loaded, name, plain, demand, device)
        return cache[(name, device)]
