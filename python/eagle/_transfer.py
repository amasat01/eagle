# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Host and device in one graph: ``eagle.copy`` (a transfer node), ``eagle.pinned``
(page-locked host arrays) and the host node of a captured GPU graph.

Every call goes through the backend seam (``eagle._core``): no array library's runtime
is used for a copy, a pinned allocation or a host function. A copy is one memcpy node
in the capture; a node whose planes are all host arrays runs as a host function node
(its CPU lowering, called from the stream)."""

from __future__ import annotations

import atexit
import contextvars
import weakref

from . import _core, _recording, _stream, arrays
from ._active_set import _address
from ._backend import BackendUnavailable
from ._device_kind import device_kind
from .graph import Node

__all__ = ["copy", "pinned"]

#: Where a GPU build collects the host nodes it lowers (see :class:`HostNode`).
HOST_SINK = contextvars.ContextVar("eagle_graph_host_sink", default=None)


# --- pinned host arrays --- #
def pinned(shape, dtype="float64"):
    """An :class:`eagle.Array` in page-locked host memory, uninitialised.

    A copy between the device and a pinned array is a plain DMA transfer; from an
    ordinary host array the driver stages it first (it works, without overlap).
    Pinning is yours: eagle never pins an array behind your back. The memory is
    freed with the array. Without a GPU backend, or with no device visible, this is an ordinary
    host array (not page-locked). ``numpy.asarray(h)`` views it with no copy.

    Example::

        h_probe = eagle.pinned((3, n))
        eagle.copy(probe, into=h_probe)     # device -> host"""
    if _core.cuda_backend()["loaded"]:
        try:
            return arrays.empty(shape, dtype, device="pinned")
        except BackendUnavailable:       # the plugin is installed and no device is visible
            pass
    return arrays.empty(shape, dtype)


# --- copy --- #
def _contiguous(arr) -> bool:
    flags = getattr(arr, "flags", None)
    return flags is None or bool(flags.c_contiguous)


def same_storage(a, b) -> bool:
    """Whether ``a`` and ``b`` are one range of memory (a copy of one onto the
    other is nothing)."""
    where = _address(a)
    return where is not None and where == _address(b)


def copy(src, *, into, name=None):
    """``into[...] = src`` as a graph node, where either array may live on the host
    or on the device: the direction (host to device, device to host, device to
    device, host to host) follows from where the two live. Both are your arrays, of
    one shape and dtype, contiguous; eagle allocates nothing. ``src`` and ``into``
    sharing their storage make a node that does nothing (the edges stay).

    Inside a builder the node is recorded and ``into`` is returned, so
    ``h = eagle.copy(d, into=h)`` reads as the data flow. Elsewhere the node is
    returned, for ``Graph.add_node``. ``eagle.copy(x, into=eagle.pinned(...))`` is
    torch's ``.cpu()`` with the buffer in your hands.

    Example::

        h_probe = eagle.copy(probe, into=h_probe)     # device -> host
        flags = eagle.copy(h_flags, into=flags)       # host -> device"""
    for label, arr in (("src", src), ("into", into)):
        if not (hasattr(arr, "shape") and hasattr(arr, "dtype")):
            raise TypeError(f"eagle.copy: {label} must be an array, got {type(arr).__name__}")
    if tuple(src.shape) != tuple(into.shape) or src.dtype != into.dtype:
        raise ValueError(
            f"eagle.copy: src is {tuple(src.shape)} {src.dtype} and into is "
            f"{tuple(into.shape)} {into.dtype}; a copy moves bytes, so make them one shape and "
            "dtype (convert on the side that owns the array)")
    for label, arr in (("src", src), ("into", into)):
        if not _contiguous(arr):
            raise ValueError(f"eagle.copy: {label} is not contiguous; copy a contiguous array "
                             "(eagle.arrays.ascontiguousarray, or a kernel that writes one)")
    node = Node("copy", [src, into], name)
    return into if record(node) else node


def record(node, graph=None) -> bool:
    """Append ``node`` to the builder being run, if one is, once; whether a builder is
    running. With ``graph`` (``Graph.add_node``), only when ``graph`` is the handle of the
    scope being recorded, so a node lands in the order the builder's calls ran."""
    rec = _recording.current()
    if rec is None or not hasattr(rec, "cur") or (graph is not None and rec.handle() is not graph):
        return False
    if not any(item is node for item in rec.cur):
        rec.cur.append(node)
    return True


def _direction(src, dst) -> int:
    on_device = (device_kind(src) != "host", device_kind(dst) != "host")
    return {(False, True): _core.COPY_H2D, (True, False): _core.COPY_D2H,
            (True, True): _core.COPY_D2D, (False, False): _core.COPY_H2H}[on_device]


def current_stream() -> int:
    """The stream a capture is recording on (the one a node issues its work to)."""
    from ._stream import current

    return current()


def copy_run(payload):
    """The capture-time call of a copy node over ``payload = [src, dst]``: one
    memcpy node on the current stream. Arrays that are not one contiguous shape
    (an older internal broadcast) fall back to the assignment."""
    def run():
        src, dst = payload
        if same_storage(src, dst):
            return
        if tuple(src.shape) != tuple(dst.shape) or src.dtype != dst.dtype \
                or not (_contiguous(src) and _contiguous(dst)):
            dst[...] = src
            return
        (s, nbytes), (d, _) = _address(src), _address(dst)
        _core.memcpy(d, s, nbytes, _direction(src, dst), current_stream())
    return run


def copy_run_host(payload):
    """The host arm of a copy: the assignment, nothing when the storage is shared."""
    def run():
        src, dst = payload
        if not same_storage(src, dst):
            dst[...] = src
    return run


# --- host nodes of a captured GPU graph --- #
#: The host nodes alive: ``call`` reaches the user's module, which reaches the graph, a loop the
#: binding's HostFunc cannot be traversed for. Interpreter exit lets go of them (see below).
_LIVE = weakref.WeakSet()


@atexit.register
def _release_host_funcs():
    """Nothing replays at exit: drop each node's function so the loop is broken before the
    interpreter tears the binding down (else it reports leaked objects)."""
    for node in list(_LIVE):
        node.func = None


class HostNode:
    """A node of host arrays in a GPU capture: ``call`` (its CPU lowering) runs on
    every replay, on a backend thread, once the work before it is done. Capturing it
    adds one host function node. An exception ``call`` raises is kept and re-raised by
    the graph's next ``launch`` / ``synchronize``."""

    __slots__ = ("func", "errors", "__weakref__")

    def __init__(self, call):
        errors = self.errors = []

        def run():
            try:
                with _stream.host_callback():
                    call()
            except BaseException as exc:    # the backend thread cannot raise it
                errors.append(exc)
        self.func = _core.HostFunc(run)
        _LIVE.add(self)
        sink = HOST_SINK.get()
        if sink is not None:
            sink.append(self)

    def __call__(self):
        self.func.enqueue(current_stream())


def raise_host_errors(nodes):
    """Raise the first exception the host nodes kept since the last call, and forget every
    one of them (a node that fails on each replay must not spill into the next launch);
    the others are counted in a note on the raised one."""
    errors = [exc for node in nodes for exc in node.errors]
    if not errors:
        return
    for node in nodes:
        node.errors.clear()
    first = errors[0]
    if len(errors) > 1:
        first.add_note(f"{len(errors) - 1} more host node failure(s) in the same launch were dropped")
    raise first
