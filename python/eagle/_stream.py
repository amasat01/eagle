# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The stream eagle issues work on when the caller names none.

With no scope open the current stream is the legacy default stream (handle
``1``), whatever array library the caller uses. A caller's own stream enters
through ``stream=`` or ``with eagle.stream(s):`` and is translated once, here, to
its raw handle (:func:`eagle.interop._stream_handle`: an int, ``__cuda_stream__``,
``.cuda_stream`` or ``.ptr``). eagle's own capturers open a scope on their stream,
so every implicit read inside a build or a run resolves to the stream being
captured. A launch on a raw handle joins the capture that handle is in, so
``with eagle.stream(s):`` inside a user's capture region puts eagle's work in
that capture.

The stack is per thread.
"""

from __future__ import annotations

import contextlib
import threading

__all__ = ["LEGACY", "current", "scope", "stream"]

#: The legacy default stream, as the seam's stream codes spell it (``0`` is
#: ambiguous there and refused, so a raw ``0`` maps to this).
LEGACY = 1


class _Scopes(threading.local):
    """The open scopes of one thread, innermost last: ``(handle, device or None)``."""

    def __init__(self) -> None:
        self.stack = []


_SCOPES = _Scopes()


def current(device: int | None = None) -> int:
    """The raw stream handle work for ``device`` goes on: the innermost open scope
    that is for ``device`` (or for no particular device), else :data:`LEGACY`."""
    for handle, owner in reversed(_SCOPES.stack):
        if owner is None or device is None or owner == device:
            return handle
    return LEGACY


def scoped(device: int | None = None) -> int:
    """The innermost open scope's handle for ``device``, or ``0`` when no scope is open (an op then uses its own
    array's stream)."""
    for handle, owner in reversed(_SCOPES.stack):
        if owner is None or device is None or owner == device:
            return handle
    return 0


@contextlib.contextmanager
def scope(handle: int, device: int | None = None):
    """Open a scope on the raw ``handle`` (``0`` is the legacy default stream)."""
    stack = _SCOPES.stack
    stack.append((LEGACY if int(handle) == 0 else int(handle), device))
    try:
        yield
    finally:
        stack.pop()


def stream(s, device: int | None = None):
    """``with eagle.stream(s):`` runs eagle's work on ``s`` until the block ends.

    ``s`` is a raw handle, a ``__cuda_stream__`` object, a torch or Warp stream
    (``.cuda_stream``) or a CuPy stream (``.ptr``); ``device`` limits the scope to
    that device's work (default: all)."""
    from .interop import _stream_handle

    return scope(_stream_handle(s), device)


class Event:
    """A CUDA event (timing-free unless ``timed=True``), made and ordered through the backend seam (no framework stream or event
    API). ``record`` it on a raw stream handle, make another stream ``wait`` for it; the last stream it was
    recorded on is what :meth:`synchronize` waits for."""

    __slots__ = ("_handle", "_device", "_stream")

    def __init__(self, device: int = -1, timed: bool = False) -> None:
        from . import _core

        self._device = int(device)
        self._handle = (_core.event_create_timed if timed else _core.event_create)(self._device)
        self._stream = LEGACY

    def record(self, stream: int) -> None:
        from . import _core

        self._stream = LEGACY if int(stream) == 0 else int(stream)
        _core.event_record(self._handle, self._stream)

    def wait(self, stream: int) -> None:
        """Make raw stream ``stream`` wait for what this event last recorded."""
        from . import _core

        _core.event_wait(int(stream), self._handle)

    def synchronize(self) -> None:
        """Block the host until the stream this event was last recorded on is idle."""
        from . import _core

        _core.stream_sync(self._stream, self._device)

    def elapsed_ms(self, start: "Event") -> float:
        """Milliseconds from ``start`` to this event (both timed and recorded); waits for this one first."""
        from . import _core

        self.synchronize()
        return float(_core.event_elapsed(start._handle, self._handle))

    def __del__(self) -> None:
        try:
            from . import _core

            _core.event_release(self._handle, self._device)
        except Exception:
            pass


def seed_events(cache: list) -> list:
    """Record the two ordering events of a replay and return the ones to wait on: one on the stream the
    caller works on (:func:`current`) and, when that is not the legacy stream, one on the legacy stream.
    ``cache`` holds the two :class:`Event` objects, made here on first use and re-recorded after."""
    from .launch import _current_device_ordinal

    here = current()
    while len(cache) < 2:
        cache.append(None)
    for i, handle in enumerate((here, LEGACY)):
        if i == 1 and here == LEGACY:
            break
        if cache[i] is None:
            cache[i] = Event(_current_device_ordinal())
        cache[i].record(handle)
    return [e for e in cache[:1 if here == LEGACY else 2]]


@contextlib.contextmanager
def on_device(device):
    """Make ``device`` (a CUDA ordinal; ``None`` for a host graph) the calling thread's current device for the
    block, restoring the previous one after."""
    if device is None:
        yield
        return
    from . import _core
    from .launch import _current_device_ordinal

    previous = _current_device_ordinal()     # 0 on a thread that has no context yet: the CUDA default
    if previous != device:
        _core.set_device(int(device))
    try:
        yield
    finally:
        if previous != device:
            _core.set_device(previous)


class _Callback(threading.local):
    on = False


_CALLBACK = _Callback()


@contextlib.contextmanager
def host_callback():
    """Mark the calling thread as running a host-function callback (a graph's host node): the backend forbids every CUDA call
    there, so :func:`check_launch` reads no status while it is open."""
    before, _CALLBACK.on = _CALLBACK.on, True
    try:
        yield
    finally:
        _CALLBACK.on = before


def check_launch(kernel: str, device: int = -1) -> None:
    """Read the launch error the device launches issued so far left pending (a query, never a wait) and raise
    it by name: ``kernel`` is the kernel (or graph) whose launch it was."""
    from . import _core

    if _CALLBACK.on:
        return
    from ._backend import BackendUnavailable

    try:
        _core.launch_status(device)
    except BackendUnavailable:                  # no usable driver: no device launch could have been issued
        return
    except RuntimeError as exc:
        if "cudaErrorNoDevice" in str(exc):     # a pipeline of host steps: there is no launch to read
            return
        raise RuntimeError(f"eagle: kernel {kernel!r}: the launch failed ({exc})") from None
