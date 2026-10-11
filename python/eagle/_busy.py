# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The per-object busy guard: one object shared across threads is
memory-safe and caller-ordered, and every method that can detach the GIL
(``Graph.build/launch/enqueue``) holds a NON-BLOCKING guard so a second
thread entering the same object is told, never queued behind it and never
handed a half-built state.

The flag itself is C++ (``eagle/util/Busy.h``, an atomic flag plus the name
of the method in flight); this module is the Python face of it: the
:class:`Busy` exception type and a context-manager wrapper."""

from __future__ import annotations

__all__ = ["Busy", "BusyGuard"]


class Busy(RuntimeError):
    """A method was entered while another is in flight on the SAME object.
    Distinct objects run concurrently; one object is for one caller at a
    time."""


class _Held:
    __slots__ = ("_guard", "_method")

    def __init__(self, guard, method):
        self._guard = guard
        self._method = method

    def __enter__(self):
        self._guard.enter(self._method)
        return self

    def __exit__(self, *exc):
        self._guard.leave()
        return False


class BusyGuard:
    """``with guard("launch"):`` raises :class:`Busy` when the object is
    already inside a guarded method. The message reads ``<owner>.<method>:
    another call is still running on this <Type>; give each thread its own
    <Type>, or wait for it``."""

    __slots__ = ("_g",)

    def __init__(self, owner: str):
        from ._core import BusyGuard as _Native

        self._g = _Native(owner)

    def __call__(self, method: str) -> _Held:
        return _Held(self._g, method)
