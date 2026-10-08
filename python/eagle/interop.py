# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""DLPack multi-framework tensor I/O for the kernel launchers.

A compiled kernel's inputs must reach the launch as contiguous ``(3, N)`` /
``(N,)`` float64 cupy arrays. This module lets a caller pass a PyTorch /
JAX / TensorFlow / cupy / numpy tensor (or any DLPack producer) instead:
import to cupy (zero-copy via DLPack when already on CUDA, a host upload
otherwise), remember the *origin* framework, and export the result back to
it (``torch`` in -> ``torch`` out). ``Device::CUDA_DEVICE == 2 == DLPack
kDLCUDA`` underpins the exchange.

Nothing is imported at module load, not even cupy: :func:`detect`
identifies a framework from ``type(x).__module__`` and DLPack/CUDA-array
attributes alone, so the real import happens inside an adapter's
``to_cupy``/``from_cupy``, preserving the GPU-free CPU test contract.
:func:`to_cupy` performs only the import; each launcher's own ``(3, N)`` /
float64 coercion on top is a no-op view when already conforming, a copy
otherwise.

The zero-copy contract itself is :func:`import_buffer` / :class:`BufferView`:
any DLPack / ``__cuda_array_interface__`` / ``__array_interface__``
producer becomes a view reporting its ``access``, ``owner``, ``producer``
and ``stream``, validating :class:`Requirements` with named refusals, and
re-exporting through ``__dlpack__`` under the DLPack 1.0 stream contract.
"""

from __future__ import annotations

# kDLCUDA in the DLPack device taxonomy (matching aether's own device tag).
_KDLCUDA = 2

# Frameworks we round-trip to on output ("torch in -> torch out"). cupy/numpy
# are the neutral internal/host currencies, not "strong" owners of a call's
# output type.
_STRONG = ("torch", "jax", "tensorflow")


def _root_module(x) -> str:
    """Top-level package a value's type comes from (``"torch"``, ``"numpy"``, ...)."""
    return type(x).__module__.partition(".")[0]


def _is_cuda(x) -> bool:
    """Whether ``x`` is device-resident, probed without importing its
    framework: the DLPack device tag first, else attribute sniffing (torch
    ``is_cuda``, a ``device`` repr containing cuda/gpu). A probe failure is
    conservative: treated as host-resident (the slow upload path)."""
    dd = getattr(x, "__dlpack_device__", None)
    if dd is not None:
        try:
            return int(dd()[0]) == _KDLCUDA
        except Exception:
            pass
    if getattr(x, "is_cuda", False):
        return True
    dev = getattr(x, "device", None)
    if dev is not None:
        try:
            s = str(dev() if callable(dev) else dev).lower()
            return "cuda" in s or "gpu" in s
        except Exception:
            return False
    return False


# ExternalStream cache: a pure wrapper over a raw cudaStream_t that creates,
# owns and destroys nothing, so one per stream handle suffices for the
# process (and cuts a cupy DeprecationWarning per construction to one).
# Keyed on the raw handle, which stays correct even if torch reallocates a
# stream at the same address.
_external_stream_cache: dict = {}
_EXTERNAL_STREAM_COUNTS = {"hits": 0, "misses": 0}


def _external_stream_stats() -> dict:
    """A copy of the :func:`external_stream` cache's hit/miss counters
    (record-only; the gate reads them)."""
    return dict(_EXTERNAL_STREAM_COUNTS)


def _reset_external_stream_cache() -> None:
    """Drop the cached ``ExternalStream`` wrappers and zero their counters
    (tests / teardown)."""
    _external_stream_cache.clear()
    _EXTERNAL_STREAM_COUNTS["hits"] = 0
    _EXTERNAL_STREAM_COUNTS["misses"] = 0


def external_stream(ptr: int):
    """The cached ``cupy.cuda.ExternalStream`` wrapping raw handle ``ptr``
    (same handle -> same wrapper object, identity not equality)."""
    key = int(ptr)
    stream = _external_stream_cache.get(key)
    if stream is None:
        _EXTERNAL_STREAM_COUNTS["misses"] += 1
        import cupy as cp

        stream = cp.cuda.ExternalStream(key)
        _external_stream_cache[key] = stream
    else:
        _EXTERNAL_STREAM_COUNTS["hits"] += 1
    return stream


class Adapter:
    """Import a framework's tensor to cupy and export a cupy result back
    to it. ``name`` is the framework tag (:func:`origin_adapter`); ``cuda``
    records whether the origin tensor was device-resident."""

    name = "base"
    cuda = True
    # True (deviceSynchronize before returning) for a host result or an
    # undriven async stream; False where the device stream carries order.
    blocking = True

    def launch_context(self):
        """The cupy stream context the kernel launches *within*. Defaults
        to a no-op (cupy's current stream); an async device adapter
        overrides this to make its own current stream the cupy one."""
        import contextlib

        return contextlib.nullcontext()

    def to_cupy(self, x):
        """Import ``x`` to a ``cupy.ndarray`` (zero-copy when possible)."""
        raise NotImplementedError

    def from_cupy(self, a):
        """Export the cupy result ``a`` back to this adapter's framework."""
        raise NotImplementedError


class _CupyAdapter(Adapter):
    name, cuda = "cupy", True
    blocking = False  # async on cupy's current stream (its default semantics)

    def to_cupy(self, x):
        return x

    def from_cupy(self, a):
        return a


class _NumpyAdapter(Adapter):
    name, cuda = "numpy", False

    def to_cupy(self, x):
        import cupy as cp

        return cp.asarray(x)  # host -> device upload (the historical numpy path)

    def from_cupy(self, a):
        import cupy as cp

        return cp.asnumpy(a)  # device -> host (the historical numpy-out path)


class _CaiAdapter(Adapter):
    """A ``__cuda_array_interface__`` producer with no framework to
    round-trip to; imported zero-copy, exported as cupy."""

    name, cuda = "cai", True
    blocking = False  # cupy-backed device buffer -> async on cupy's current stream

    def to_cupy(self, x):
        import cupy as cp

        return cp.asarray(x)  # cupy consumes __cuda_array_interface__ zero-copy

    def from_cupy(self, a):
        return a


class _DlpackAdapter(Adapter):
    """A generic ``__dlpack__`` producer with no framework we specifically
    know; exported as cupy (device) or numpy (host), since no owning
    framework exists to make."""

    name = "dlpack"

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.blocking = not cuda  # CUDA -> async on cupy's current stream; CPU -> sync

    def to_cupy(self, x):
        import cupy as cp

        if self.cuda:
            return cp.from_dlpack(x)  # zero-copy device view
        import numpy as np

        return cp.asarray(np.from_dlpack(x))  # cupy rejects CPU DLPack -> host upload

    def from_cupy(self, a):
        _refuse_permuted_export(a, self.name)
        if self.cuda:
            return a
        import cupy as cp

        return cp.asnumpy(a)


class _TorchAdapter(Adapter):
    name = "torch"

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.blocking = not cuda  # CUDA -> async on torch's current stream; CPU -> sync

    def launch_context(self):
        if not self.cuda:
            import contextlib

            return contextlib.nullcontext()
        import torch

        # Make torch's current stream the cupy current stream so the
        # launch inherits torch's ordering, no host sync.
        return external_stream(torch.cuda.current_stream().cuda_stream)

    def to_cupy(self, x):
        import cupy as cp

        if self.cuda:
            return cp.from_dlpack(x)  # zero-copy device view (shares x's storage)
        import numpy as np

        return cp.asarray(np.from_dlpack(x))  # CPU tensor -> host upload

    def from_cupy(self, a):
        _refuse_permuted_export(a, self.name)
        import torch

        if self.cuda:
            return torch.from_dlpack(a)  # zero-copy device view of the cupy result
        import cupy as cp

        return torch.from_numpy(cp.asnumpy(a))  # device result -> host torch tensor


class _JaxAdapter(Adapter):
    name = "jax"

    def __init__(self, cuda: bool):
        self.cuda = cuda
        # v1: sync (jax's XLA stream isn't a simple injectable cudaStream_t);
        # the DLPack round-trip still works, just synchronously.
        self.blocking = True

    def to_cupy(self, x):
        import cupy as cp

        if self.cuda:
            return cp.from_dlpack(x)
        import numpy as np

        return cp.asarray(np.from_dlpack(x))

    def from_cupy(self, a):
        _refuse_permuted_export(a, self.name)
        import jax

        if self.cuda:
            return jax.dlpack.from_dlpack(a)  # zero-copy device view
        import jax.numpy as jnp

        return jnp.asarray(a.get())  # host result -> jax (default device; best-effort)


class _TfAdapter(Adapter):
    name = "tensorflow"

    def __init__(self, cuda: bool):
        self.cuda = cuda
        self.blocking = True  # v1: sync (TF stream not injected yet); see _JaxAdapter

    def to_cupy(self, x):
        import cupy as cp

        if self.cuda:
            return cp.from_dlpack(x)
        import numpy as np

        return cp.asarray(np.from_dlpack(x))

    def from_cupy(self, a):
        _refuse_permuted_export(a, self.name)
        import tensorflow as tf

        if self.cuda:
            # TF's from_dlpack consumes a PyCapsule, not the array object.
            return tf.experimental.dlpack.from_dlpack(a.__dlpack__())
        import cupy as cp

        return tf.constant(cp.asnumpy(a))


# Singletons for the stateless adapters (the framework-tagged ones carry a
# per-tensor ``cuda`` flag and are built fresh in ``detect``).
_CUPY = _CupyAdapter()
_NUMPY = _NumpyAdapter()
_CAI = _CaiAdapter()


def detect(x) -> Adapter | None:
    """Identify ``x``'s framework adapter without importing the framework.
    ``None`` for a value we don't recognize as a tensor (:func:`to_cupy`
    then treats it as a host upload)."""
    root = _root_module(x)
    if root == "cupy":
        return _CUPY
    if root == "numpy":
        return _NUMPY
    if root == "torch":
        return _TorchAdapter(_is_cuda(x))
    if root in ("jax", "jaxlib"):
        return _JaxAdapter(_is_cuda(x))
    if root == "tensorflow":
        return _TfAdapter(_is_cuda(x))
    if hasattr(x, "__dlpack__") and hasattr(x, "__dlpack_device__"):
        return _DlpackAdapter(_is_cuda(x))
    if hasattr(x, "__cuda_array_interface__"):
        return _CAI
    return None


def _refuse_permuted_export(a, framework: str) -> None:
    """A framework bridge never exports a plane an :class:`eagle.ActiveSet`
    holds permuted (see :func:`eagle._active_set.refuse_if_permuted`)."""
    from ._active_set import refuse_if_permuted

    refuse_if_permuted(a, f"eagle.interop: the {framework} bridge")


def to_cupy(x):
    """Import a framework / DLPack tensor to a cupy array: zero-copy for a
    CUDA-resident producer, a host upload otherwise. Also the pre-import
    helper for the capturable ``launch`` path, since a DLPack import can
    synchronize and is unsafe mid-capture."""
    from ._active_set import refuse_if_permuted

    refuse_if_permuted(x, "eagle.interop.to_cupy")
    ad = detect(x)
    if ad is None:
        import cupy as cp

        return cp.asarray(x)
    return ad.to_cupy(x)


def host_ptr(x) -> int:
    """Raw host data pointer of a contiguous CPU tensor, for a zero-copy
    host-plugin launch (:class:`eagle.host_launch.HostPluginLibrary`) --
    no import, no copy, no cupy. Accepts a torch CPU tensor or a numpy
    array; a device tensor or non-contiguous buffer raises."""
    root = _root_module(x)
    if root == "torch":
        if _is_cuda(x):
            raise TypeError("host_ptr: tensor is device-resident, not host")
        if not x.is_contiguous():
            raise ValueError("host_ptr: tensor must be contiguous")
        return int(x.data_ptr())
    import numpy as np

    a = np.asarray(x)
    if not a.flags.c_contiguous:
        raise ValueError("host_ptr: array must be C-contiguous")
    return int(a.ctypes.data)


def as_out_buffer(out, shape, dtype=None):
    """Import a caller-provided device buffer as a zero-copy cupy view to
    fill in place -- strict where :func:`to_cupy` is lenient: a wrong
    dtype/shape/stride raises rather than copies (the caller's buffer would
    never get the result). ``out`` must be device-resident and a
    C-contiguous ``shape`` buffer of ``dtype`` (default float64)."""
    import cupy as cp

    if dtype is None:
        dtype = cp.float64
    ad = detect(out)
    if ad is None or not getattr(ad, "cuda", False):
        raise TypeError(
            "out= must be a device tensor (cupy / CUDA torch / CUDA-DLPack); a host "
            "buffer cannot be filled in place"
        )
    cu = to_cupy(out)  # zero-copy device view (no cast, no contiguation)
    if (
        cu.dtype != dtype
        or tuple(cu.shape) != tuple(shape)
        or not cu.flags.c_contiguous
    ):
        raise ValueError(
            f"out= must be a C-contiguous {tuple(shape)} {cp.dtype(dtype).name} CUDA "
            f"buffer; got shape {tuple(cu.shape)} dtype {cu.dtype}"
        )
    return cu


def origin_adapter(candidates) -> Adapter:
    """Pick the adapter the kernel's output should be returned as, over
    the state-vector inputs in signature order: the first strong framework
    (torch/jax/tensorflow) wins (two distinct ones raise ``TypeError``),
    else the first device adapter, else numpy."""
    ads = [detect(c) for c in candidates]
    strong = [a for a in ads if a is not None and a.name in _STRONG]
    if len({a.name for a in strong}) > 1:
        raise TypeError(
            "mixed tensor frameworks in one call: "
            f"{sorted({a.name for a in strong})}; pass all state-vector inputs as the "
            "same framework"
        )
    if strong:
        return strong[0]
    for a in ads:
        if a is not None and a.name in ("cupy", "dlpack", "cai") and a.cuda:
            return a
    return _NUMPY


# The generic buffer layer: import_buffer / BufferView -- the zero-copy
# contract every family consumer shares, implemented once in C++
# (aether/interop/Buffer.h + plugin/interop.h) and bound in eagle._core.
# Tracks ownership x access (owner/producer/access, legacy DLPack staying
# "unknown" unless asserted writable), the DLPack 1.0 stream contract on
# export, and keeps the producer alive for as long as any consumer holds
# a re-export.

_KDLCPU = 1

# DLPack dtype codes (kDLInt, kDLUInt, kDLFloat, kDLBool) by name.
_DTYPE_CODES = {
    "float64": (2, 64), "float32": (2, 32),
    "int64": (0, 64), "int32": (0, 32),
    "uint64": (1, 64), "uint32": (1, 32), "uint8": (1, 8),
    "bool": (6, 8),
}

_DEVICE_TYPES = {"cpu": 1, "cuda": 2}


class BufferRefused(ValueError):
    """A buffer failed a :class:`Requirements` check; the message carries
    one ``"<check>: required <want>, got <have>"`` clause per failed check."""


class Requirements:
    """What a consumer needs from a buffer; every argument left at its
    default is not checked: ``dtype``, ``count``, ``shape``, ``contiguous``,
    ``alignment``, ``device`` (``"cuda"``/``"cpu"``), ``device_id``, and
    ``writable`` (leave off for a read-only input, which accepts any access)."""

    __slots__ = ("dtype", "count", "shape", "contiguous", "alignment", "device",
                 "device_id", "writable")

    def __init__(self, *, dtype=None, count=None, shape=None, contiguous=False,
                 alignment=0, device=None, device_id=None, writable=False):
        if dtype is not None and str(dtype) not in _DTYPE_CODES:
            raise ValueError(f"Requirements: unknown dtype {dtype!r}")
        if device is not None and device not in _DEVICE_TYPES:
            raise ValueError(
                f"Requirements: device must be 'cuda' or 'cpu', got {device!r}")
        self.dtype = None if dtype is None else str(dtype)
        self.count = count
        self.shape = None if shape is None else tuple(int(s) for s in shape)
        self.contiguous = bool(contiguous)
        self.alignment = int(alignment)
        self.device = device
        self.device_id = device_id
        self.writable = bool(writable)


def _producer_name(obj) -> str:
    t = type(obj)
    return f"{t.__module__.partition('.')[0]}.{t.__qualname__}"


def _core_buffer_from(obj, stream):
    """The bound record for ``obj``: DLPack first (versioned, then legacy),
    then ``__cuda_array_interface__``, then ``__array_interface__``."""
    from eagle import _core

    from ._active_set import refuse_if_permuted

    refuse_if_permuted(obj, "eagle.interop.import_buffer")
    producer = _producer_name(obj)
    if hasattr(obj, "__dlpack__"):
        on_cpu = False
        dd = getattr(obj, "__dlpack_device__", None)
        if dd is not None:
            on_cpu = int(dd()[0]) == _KDLCPU
        if on_cpu and stream is not None:
            raise _core.InteropError(
                f"stream: a CPU buffer takes no stream, got {stream}")
        kw = {} if on_cpu else {"stream": stream}
        try:
            capsule = obj.__dlpack__(max_version=(1, 0), **kw)
        except TypeError:
            capsule = obj.__dlpack__(**kw)
        return _core.InteropBuffer.from_capsule(capsule, producer, stream)
    cai = getattr(obj, "__cuda_array_interface__", None)
    if cai is not None:
        ptr, read_only = cai["data"]
        buf = _core.InteropBuffer.from_interface(
            obj, int(ptr), bool(read_only), list(cai["shape"]),
            None if cai.get("strides") is None else list(cai["strides"]),
            cai["typestr"], 2, _cai_device(ptr), producer, stream)
        # CAI v3: work the producer queued on its stream is ordered before ours.
        producer_stream = cai.get("stream")
        if producer_stream is not None:
            _core.fence(int(producer_stream), stream, buf.device_id)
        return buf
    ai = getattr(obj, "__array_interface__", None)
    if ai is not None:
        ptr, read_only = ai["data"]
        return _core.InteropBuffer.from_interface(
            obj, int(ptr), bool(read_only), list(ai["shape"]),
            None if ai.get("strides") is None else list(ai["strides"]),
            ai["typestr"], _KDLCPU, 0, producer, stream)
    raise TypeError(
        f"import_buffer: {producer} speaks none of __dlpack__, "
        "__cuda_array_interface__ or __array_interface__")


def _cai_device(ptr) -> int:
    """The ordinal owning a CAI pointer (the protocol does not carry it)."""
    if not ptr:
        return 0
    import cupy as cp

    return int(cp.cuda.runtime.pointerGetAttributes(int(ptr)).device)


class BufferView:
    """A zero-copy view of another library's buffer (see
    :func:`import_buffer`). Holds the producer object so the memory stays
    valid after the caller drops its own reference; re-exportable through
    ``__dlpack__`` / ``__dlpack_device__``."""

    __slots__ = ("_buf", "_obj", "__weakref__")

    def __init__(self, buf, obj):
        self._buf = buf
        self._obj = obj

    @property
    def access(self) -> str:
        """``"read-write"``, ``"read-only"`` or ``"unknown"`` (legacy DLPack)."""
        return self._buf.access

    @property
    def owner(self) -> str:
        """``"external"``: the memory belongs to the producer."""
        return self._buf.owner

    @property
    def producer(self):
        """The producer's type name, e.g. ``"numpy.ndarray"``; ``None`` when owned."""
        return self._buf.producer or None

    @property
    def stream(self):
        """The CUDA stream the contents are ordered on (int), ``None`` on the host."""
        return self._buf.stream

    @property
    def assumed_writable(self) -> bool:
        """Whether the caller asserted writability at import."""
        return self._buf.assumed_writable

    @property
    def writable(self) -> bool:
        """Declared read-write, or unknown with an explicit assertion."""
        return self._buf.writable

    @property
    def ptr(self) -> int:
        """The data address (the producer's byte offset included)."""
        return self._buf.ptr

    @property
    def shape(self) -> tuple:
        return tuple(self._buf.shape)

    @property
    def strides(self) -> tuple:
        """Strides in elements."""
        return tuple(self._buf.strides)

    @property
    def dtype(self) -> str:
        """The element type name (``"float64"``, ...)."""
        return self._buf.dtype

    @property
    def device(self) -> tuple:
        """``(device_type, device_id)`` in DLPack's taxonomy (1 = CPU, 2 = CUDA)."""
        return (self._buf.device_type, self._buf.device_id)

    def refusals(self, require: Requirements) -> list:
        """Every refusal ``require`` raises against this view (empty if none)."""
        code, bits = _DTYPE_CODES[require.dtype] if require.dtype else (None, None)
        return list(self._buf.refusals(
            code, bits, require.count,
            None if require.shape is None else list(require.shape),
            require.contiguous, require.alignment,
            None if require.device is None else _DEVICE_TYPES[require.device],
            require.device_id, require.writable))

    def check(self, require: Requirements) -> None:
        """Raise :class:`BufferRefused` naming every failed check."""
        r = self.refusals(require)
        if r:
            raise BufferRefused("; ".join(r))

    def refence(self, stream) -> None:
        """Order this view's stream before ``stream`` again: call it before
        each launch that reads the buffer when the producer keeps writing
        between launches. Same stream codes as ``__dlpack__``."""
        self._buf.refence(stream)

    def __dlpack__(self, *, stream=None, max_version=None, dl_device=None, copy=None):
        if copy:
            raise BufferError("BufferView exports zero-copy only (copy=True refused)")
        if dl_device is not None and tuple(int(v) for v in dl_device) != self.device:
            raise BufferError(
                f"BufferView lives on {self.device}, not {tuple(dl_device)}")
        if max_version is None or int(max_version[0]) < 1:
            return self._buf.export_legacy(stream)
        return self._buf.export_versioned(stream)

    def __dlpack_device__(self) -> tuple:
        return self.device

    def __repr__(self) -> str:
        return (f"BufferView(shape={self.shape}, dtype={self.dtype}, "
                f"device={self.device}, access={self.access!r}, "
                f"owner={self.owner!r}, producer={self.producer!r})")


def import_buffer(obj, *, stream=None, require=None,
                  assume_writable=False) -> BufferView:
    """Import any DLPack / CUDA-array-interface / array-interface producer
    as a zero-copy :class:`BufferView`.

    ``stream`` is the CUDA stream the caller will use the buffer on (DLPack
    codes: ``None``/``1`` legacy default, ``2`` per-thread, ``-1`` no
    ordering, ``0`` refused, any other int a ``cudaStream_t``); must be
    ``None`` for a host buffer. ``require`` is an optional
    :class:`Requirements`, raising :class:`BufferRefused` naming every
    failed check. ``assume_writable`` asserts writability for a buffer
    whose producer declared no access, never overriding a declared
    read-only."""
    buf = _core_buffer_from(obj, stream)
    if assume_writable:
        buf.assume_writable()
    view = BufferView(buf, obj)
    if require is not None:
        view.check(require)
    return view


def refence(producer_stream, consumer_stream, device: int = 0) -> None:
    """Order ``producer_stream`` before ``consumer_stream`` (DLPack stream
    codes) on ``device`` without blocking the host: one event record + one
    stream wait."""
    from eagle import _core

    _core.fence(producer_stream, consumer_stream, device)
