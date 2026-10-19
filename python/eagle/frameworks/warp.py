# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A hawk per-sample kernel inside a Warp program: :func:`function`.

``fn = function(kernel)`` is a callable over ``wp.array`` planes. It runs the
compiled kernel on Warp's current stream of the arrays' device (no sync, so it
orders after earlier Warp launches and before later ones, and a
``wp.ScopedCapture`` records it), or on hawk's host runtime for CPU arrays.

``wp.Tape``: inside ``with wp.Tape()`` (or with ``tape=``) a call records a
backward on the tape, and ``tape.backward()`` runs the reverse-mode kernel hawk
derives from the kernel (:func:`hawk.diff.vjp`), adding each plane's gradient
to its ``.grad``. A ``Param`` is read from a one-cell array
(:func:`hawk.diff.param_planes`); give it ``requires_grad=True`` and its
gradient is the per-sample partials summed in a fixed order. Reverse mode only:
``Tape.record_func`` has no forward mode and no second derivative.

Carried state: a ``Mutable`` plane the kernel reads before it writes is an input
too (in declared order) and comes back as a new array that starts as a copy of
it, so a sample the kernel does not step (a terminated one) keeps its value and
the input still holds the value before the step. Under a tape, a state plane's
gradient follows hawk's one-step reverse (:func:`hawk.diff.step_vjp`), and a
terminated sample passes its incoming gradient through unchanged, so a
multi-step loop needs no ``where``. Passing the input array itself as ``out=``
steps in place (no gradient through that call).

Planes are component-major, as everywhere in eagle: ``(n,)`` for a scalar,
``(width, n)`` for a vector, ``(1,)`` for a ``Param``; float32 or float64,
never mixed in one call. The build follows the call's dtype. Outputs are zeroed
before the launch (a terminated sample stays zero); a ``Terminated`` mask the
kernel finishes comes back as the last output, a copy of the input mask.

Capture: build the kernel before ``wp.capture_begin`` (one eager call, or
:meth:`WarpFunction.prepare`); inside the capture a call only enqueues work.
Pass ``out=`` to avoid allocating inside the capture on a device without a
memory pool.

warp is imported by this submodule only: ``import eagle`` stays warp-free.
"""

import tempfile

import numpy as np
import warp as wp

from .. import FINISHED_PLANE, _cell_rows
from ._kernel_planes import call_mode, promote
from ._tower import LOCK, Level

__all__ = ["WarpFunction", "function"]

#: Chunks of the fixed-order Param reduction (partial sums, then their sum).
_CHUNKS = 256


def function(kernel, *, cache_dir=None, device_arch=None):
    """Wrap ``kernel`` (a hawk per-sample kernel) as a callable over Warp arrays.

    :param cache_dir: where artifacts publish (default: a private temp dir).
    :param device_arch: CUDA arch to compile for (default: the arrays' device).
    :returns: a :class:`WarpFunction`.
    """
    return WarpFunction(kernel, cache_dir=cache_dir, device_arch=device_arch)


# -- the accumulate / reduce kernels (per float dtype, built once) --------------
_KERNELS = {}


def _kernels(dtype):
    got = _KERNELS.get(dtype)
    if got is None:
        with LOCK:
            got = _KERNELS.get(dtype)
            if got is None:
                got = _KERNELS[dtype] = _make_kernels(dtype)
    return got


def _make_kernels(dtype):
    @wp.kernel(module="unique")
    def accumulate(dst: wp.array2d(dtype=dtype), src: wp.array2d(dtype=dtype),
                   nxt: wp.array2d(dtype=dtype), mask: wp.array(dtype=wp.bool), use_mask: int):
        # use_mask: 0 = add src; 1 = add src unless the sample is finished;
        # 2 = a finished sample adds its incoming gradient nxt instead (a skipped step is the identity)
        i, j = wp.tid()
        if use_mask == 0 or not mask[j]:
            dst[i, j] = dst[i, j] + src[i, j]
        elif use_mask == 2:
            dst[i, j] = dst[i, j] + nxt[i, j]

    @wp.kernel(module="unique")
    def chunk_sums(src: wp.array(dtype=dtype), mask: wp.array(dtype=wp.bool), use_mask: int,
                   part: wp.array(dtype=dtype), chunk: int):
        c = wp.tid()
        lo = c * chunk
        hi = wp.min(lo + chunk, src.shape[0])
        total = dtype(0.0)
        for i in range(lo, hi):
            if use_mask == 0 or not mask[i]:
                total = total + src[i]
        part[c] = total

    @wp.kernel(module="unique")
    def add_total(part: wp.array(dtype=dtype), dst: wp.array(dtype=dtype)):
        total = dtype(0.0)
        for c in range(part.shape[0]):
            total = total + part[c]
        dst[0] = dst[0] + total

    @wp.kernel(module="unique")
    def unfinished(dst: wp.array2d(dtype=dtype), src: wp.array2d(dtype=dtype),
                   mask: wp.array(dtype=wp.bool)):
        # a finished sample did not step: it seeds no gradient
        i, j = wp.tid()
        dst[i, j] = dtype(0.0) if mask[j] else src[i, j]

    return accumulate, unfinished, chunk_sums, add_total


_NO_MASK = {}


def _no_mask(device):
    """A one-cell bool array standing in for an absent mask (never read)."""
    got = _NO_MASK.get(device.ordinal)
    if got is None:
        with LOCK:
            got = _NO_MASK.get(device.ordinal)
            if got is None:
                got = _NO_MASK[device.ordinal] = wp.zeros(1, dtype=wp.bool, device=device)
    return got


# -- helpers ------------------------------------------------------------------
def _active_tape():
    """The ``wp.Tape`` currently recording, or ``None``. Warp keeps it on its private
    runtime object (``wp._src.context.runtime.tape``); if a Warp release moves it,
    this fails by name rather than silently recording nothing."""
    runtime = getattr(getattr(getattr(wp, "_src", None), "context", None), "runtime", None)
    if runtime is None or not hasattr(runtime, "tape"):
        raise RuntimeError(
            "eagle.frameworks.warp: cannot find Warp's active tape (wp._src.context.runtime.tape "
            f"is missing in warp {getattr(wp, '__version__', '?')}); this bridge is built for "
            "warp 1.17. Pass tape= to the call, or upgrade eagle")
    return runtime.tape


def _cupy_view(array, device):
    """``array`` as a cupy array over the same memory, no copy, no query of the
    pointer (memory allocated inside a graph capture has no valid pointer
    attributes), the device named outright."""
    import cupy

    nbytes = array.size * wp.types.type_size_in_bytes(array.dtype)
    memory = cupy.cuda.UnownedMemory(array.ptr, nbytes, array, device.ordinal)
    return cupy.ndarray(array.shape, dtype=_np_dtype(array),
                        memptr=cupy.cuda.MemoryPointer(memory, 0))


def _copy_values(dst, src):
    """Copy ``src``'s values into ``dst`` and nothing else: ``wp.copy`` also copies
    the gradient arrays and records itself on a tape, which the state's own
    backward already accounts for."""
    plain = lambda a: wp.array(ptr=a.ptr, dtype=a.dtype, shape=a.shape, device=a.device, copy=False)
    wp.copy(plain(dst), plain(src))


def _planes_2d(array):
    return array if array.ndim == 2 else array.reshape((1, array.shape[0]))


def _np_dtype(array):
    return np.dtype(wp.dtype_to_numpy(array.dtype))


class WarpFunction:
    """A hawk kernel as a callable over Warp arrays (see the module doc).

    :ivar inputs: the bound input names, in call order.
    :ivar outputs: the returned names: every ``Mutable`` plane, then the
        ``Terminated`` mask when the kernel finishes it.
    """

    def __init__(self, kernel, *, cache_dir=None, device_arch=None):
        self.name = kernel.name
        self._private = None
        if cache_dir is None:
            self._private = tempfile.TemporaryDirectory(prefix="eagle_warp_")
            cache_dir = self._private.name
        self._dir = cache_dir
        self._arch = device_arch
        self._root = Level(promote(kernel),
                           who="eagle.frameworks.warp", label=kernel.name)
        self.sig = self._root.sig
        self.inputs = self.sig.inputs
        self.outputs = self.sig.outputs

    def __repr__(self) -> str:
        return f"WarpFunction({self.name!r})"

    # -- the call -------------------------------------------------------------
    def __call__(self, *args, out=None, tape=None, **kwargs):
        """Run the kernel over ``args`` / ``kwargs`` (arrays, in input order or
        by name). Returns the output array (bare if one) or a tuple.

        :param out: preallocated output arrays (a tuple in output order, or a
            dict by name); they are zeroed first. Default: allocated here.
        :param tape: the ``wp.Tape`` to record on (default: the active one).
        """
        sig = self.sig
        values = self._bind(args, kwargs)
        device, n = self._check(values)
        mode = call_mode("eagle.frameworks.warp",
                         [(k, _np_dtype(v)) for k, v in values.items() if sig.is_float(k)])
        wants = any(v.requires_grad for k, v in values.items() if sig.differentiable(k))
        if wants and tape is None:
            tape = _active_tape()
        wants = wants and tape is not None
        outs = self._outputs(values, out, device, n, mode, wants)
        self._launch(self._root, mode, {**values, **dict(zip(sig.outputs, outs))}, device)
        if wants:
            self._record(tape, values, outs, device, n, mode)
        return outs[0] if len(outs) == 1 else tuple(outs)

    def prepare(self, mode, device, *, backward=False, samples=None):
        """Build and load the kernel (and its reverse-mode kernel when
        ``backward``) for ``mode`` (``"float32"``/``"float64"``) on ``device``,
        so a later call can sit inside ``wp.capture_begin``. A kernel that
        reduces over samples also needs ``samples``, the batch size the capture
        will run (the row scratch cannot be allocated inside a capture)."""
        device = wp.get_device(device)
        levels = [self._root] + ([self._adjoint()[0]] if backward else [])
        for level in levels:
            self._built(level, mode, device)
            if samples is not None:
                self._rows(level, int(samples), mode, device)

    @staticmethod
    def _rows(level, n, mode, device):
        """The row scratch of ``level`` at ``n`` samples (``None`` without a reduction)."""
        if not level.sig.cells:
            return None
        if device.is_cuda:
            import cupy as xp
        else:
            xp = np
        if device.is_cuda and device.is_capturing and not _cell_rows.has(level, n, xp, mode):
            raise RuntimeError(
                f"eagle.frameworks.warp: {level.label!r} reduces over samples and a graph "
                "capture is open (its row scratch cannot be allocated inside one); call "
                f"fn.prepare(mode, device, backward=..., samples={n}) or run the call once "
                "before wp.capture_begin")
        return _cell_rows.scratch(level, n, xp, np.dtype(mode), cells=level.sig.cells)

    def _bind(self, args, kwargs):
        names = self.inputs
        if len(args) > len(names):
            raise TypeError(f"{self.name}: takes {len(names)} planes {names}, got {len(args)}")
        values = dict(zip(names, args))
        for k, v in kwargs.items():
            if k not in names:
                raise TypeError(f"{self.name}: no input {k!r}; inputs are {names}")
            if k in values:
                raise TypeError(f"{self.name}: input {k!r} given twice")
            values[k] = v
        missing = [k for k in names if k not in values]
        if missing:
            raise TypeError(f"{self.name}: missing inputs {missing}")
        return {k: values[k] for k in names}

    def _check(self, values):
        sig = self.sig
        device = n = None
        for name, a in values.items():
            if not isinstance(a, wp.array):
                raise TypeError(f"{self.name}: {name!r} must be a wp.array, got {type(a).__name__}")
            if device is None:
                device = a.device
            elif a.device != device:
                raise ValueError(f"{self.name}: {name!r} is on {a.device}, the call is on {device}")
            if not a.is_contiguous:
                raise ValueError(f"{self.name}: {name!r} must be contiguous")
            if not sig.is_cell(name):
                have = a.shape[-1]
                if n is not None and have != n:
                    raise ValueError(f"{self.name}: {name!r} has {have} samples, others {n}")
                n = have
        for name, a in values.items():
            want = sig.shape(name, n)
            if tuple(a.shape) != want:
                raise ValueError(f"{self.name}: {name!r} has shape {tuple(a.shape)}, expected {want}")
            if not sig.is_float(name) and _np_dtype(a) != sig.dtype(name, "float64"):
                raise TypeError(f"{self.name}: {name!r} is {_np_dtype(a).name}, "
                                f"expected {sig.dtype(name, 'float64').name}")
        return device, n

    def _outputs(self, values, out, device, n, mode, grad):
        sig = self.sig
        if out is not None and not isinstance(out, dict):
            if len(out) != len(sig.outputs):
                raise ValueError(f"{self.name}: out= needs {len(sig.outputs)} arrays {sig.outputs}")
            out = dict(zip(sig.outputs, out))
        outs = []
        for name in sig.outputs:
            if out is not None and name in out:
                arr = out[name]
                if tuple(arr.shape) != sig.shape(name, n) or arr.device != device \
                        or _np_dtype(arr) != sig.dtype(name, mode):
                    raise ValueError(f"{self.name}: out[{name!r}] must be "
                                     f"{sig.shape(name, n)} {sig.dtype(name, mode).name} on {device}")
            else:
                arr = wp.empty(sig.shape(name, n), dtype=wp.dtype_from_numpy(sig.dtype(name, mode)),
                               device=device, requires_grad=grad and sig.is_float(name))
            if name == sig.terminated or name in sig.state:
                if arr is not values[name]:
                    _copy_values(arr, values[name])
                elif grad and name in sig.state:
                    raise ValueError(
                        f"{self.name}: out[{name!r}] is the input array, and the step overwrites "
                        "the value its gradient needs; pass a new array")
            else:
                arr.zero_()
            outs.append(arr)
        return outs

    def _adjoint(self):
        """``(level, wrt, name)``: the reverse kernel, the inputs it differentiates,
        and the spelling of a plane's gradient out and incoming adjoint. A stateless
        kernel uses :func:`hawk.diff.vjp`; a state kernel reads its step as a pure
        function of the values before it (:func:`hawk.diff.step_vjp`), which reads
        ``<plane>`` (before the step) and ``bar_<plane>_next`` and writes ``bar_<plane>``."""
        root = self._root
        if not root.sig.state:
            return root.derived("vjp"), root.wrt, root.names("vjp"), None
        nexts, grads = root.step_names()
        return root.stepped(), root.wrt, grads, {plane: o for o, plane in nexts.items()}

    # -- the backward ---------------------------------------------------------
    def _record(self, tape, values, outs, device, n, mode):
        sig = self.sig
        vjp, wrt, adjoint, seeds = self._adjoint()
        outputs = dict(zip(sig.outputs, outs))
        wanted = [k for k in wrt if k in values and values[k].requires_grad]
        tracked = [a for a in (*(values[k] for k in wanted),
                               *(o for k, o in outputs.items() if sig.is_float(k)))
                   if a.requires_grad]
        if seeds is None:
            seeds = {adjoint[o]: o for o in outputs if o in adjoint}

        def backward():
            planes = {}
            for name in vjp.sig.inputs:
                if name in values:
                    planes[name] = values[name]
                else:
                    seed = outputs[seeds[name]]
                    # an output the caller did not ask a gradient for carries a zero adjoint
                    planes[name] = seed.grad if seed.grad is not None else wp.zeros_like(seed)
                    if vjp.sig.cells and sig.terminated:
                        # the rows sum every sample: a finished one must seed nothing
                        planes[name] = self._unfinished(planes[name], values[sig.terminated],
                                                        device, mode)
            for name in vjp.sig.outputs:
                planes[name] = wp.zeros(vjp.sig.shape(name, n),
                                        dtype=wp.dtype_from_numpy(vjp.sig.dtype(name, mode)),
                                        device=device)
            self._launch(vjp, mode, planes, device)
            mask = values[sig.terminated] if sig.terminated else None
            for name in wanted:
                # a finished sample did not step: its incoming gradient passes through
                carried = outputs[name].grad if name in sig.state else None
                self._add(values[name].grad, planes[adjoint[name]], mask, device, mode,
                          cell=sig.is_cell(name), through=carried,
                          finished=adjoint[name] in vjp.sig.cells)

        tape.record_func(backward, tracked)

    @staticmethod
    def _add(dst, src, mask, device, mode, *, cell, finished=False, through=None):
        accumulate, _unfinished, chunk_sums, add_total = _kernels(
            wp.float32 if mode == "float32" else wp.float64)
        if cell and finished:     # a cell arrives finished (eagle.reduce of the rows): add it as is
            mask = None
        flag = 0 if mask is None else (1 if through is None else 2)
        mask = _no_mask(device) if mask is None else mask
        if cell and not finished:     # per-sample terms (no reduction over samples in the kernel)
            n = src.shape[0]
            count = min(_CHUNKS, n)
            part = wp.empty(count, dtype=src.dtype, device=device)
            wp.launch(chunk_sums, dim=count, inputs=[src, mask, flag, part, -(-n // count)],
                      device=device, record_tape=False)
            wp.launch(add_total, dim=1, inputs=[part, dst], device=device, record_tape=False)
            return
        d, s = _planes_2d(dst), _planes_2d(src)
        nxt = s if through is None else _planes_2d(through)
        wp.launch(accumulate, dim=d.shape, inputs=[d, s, nxt, mask, flag], device=device,
                  record_tape=False)

    @staticmethod
    def _unfinished(seed, mask, device, mode):
        _acc, unfinished, _cs, _at = _kernels(wp.float32 if mode == "float32" else wp.float64)
        out = wp.zeros_like(seed)
        wp.launch(unfinished, dim=_planes_2d(out).shape, inputs=[_planes_2d(out), _planes_2d(seed), mask],
                  device=device, record_tape=False)
        return out

    # -- the launch itself ------------------------------------------------------
    def _launch(self, level, mode, planes, device):
        """Run ``level`` over ``planes`` (name -> wp.array): on Warp's current
        stream for a CUDA device, on the host runtime for a CPU one."""
        spec = level.sig.arg_spec
        cells = level.sig.cells
        if not cells:
            if device.is_cuda:
                self._launch_device(level, mode, spec, planes, device)
            else:
                self._launch_host(level, mode, spec, planes, device)
            return
        # a reduction over samples: the kernel writes rows into eagle's scratch,
        # which are finished into the cell plane
        sig = level.sig
        n = next(int(planes[nm].shape[-1]) for r, nm in spec
                 if nm in planes and nm not in sig.rows and r in ("per_sample", "vec_in", "mat_in"))
        rows = self._rows(level, n, mode, device)
        with rows:
            if device.is_cuda:
                self._launch_device(level, mode, spec, planes, device, rows=rows.rows)
                stream = device.stream.cuda_stream or 0
                import cupy
                from eagle.interop import external_stream

                with cupy.cuda.Device(device.ordinal), external_stream(stream):
                    for nm, cell in rows.finish().items():
                        _cupy_view(planes[nm], device)[...] = cell
            else:
                self._launch_host(level, mode, spec, planes, device, rows=rows.rows)
                for nm, cell in rows.finish().items():
                    planes[nm].numpy()[...] = cell

    def _launch_host(self, level, mode, spec, planes, device, rows=None):
        from hawk.runtime import run

        host = self._built(level, mode, device)
        arrays = {}
        for role, name in spec:
            if role == "lookup" and name == FINISHED_PLANE:
                arrays[name] = np.zeros(1, dtype=np.int32)
            elif rows is not None and name in rows:
                arrays[name] = rows[name]
            elif name in planes:
                arrays[name] = planes[name].numpy()
        run(host, **arrays)

    def _launch_device(self, level, mode, spec, planes, device, rows=None):
        import cupy

        from eagle.interop import external_stream

        plan = self._built(level, mode, device)
        # a device's default stream carries no handle: it is the CUDA legacy stream, 0
        stream = device.stream.cuda_stream or 0
        with cupy.cuda.Device(device.ordinal), external_stream(stream):
            arrays = {}
            for role, name in spec:
                if role == "lookup" and name == FINISHED_PLANE:
                    arrays[name] = _cupy_view(wp.zeros(1, dtype=wp.int32, device=device), device)
                elif rows is not None and name in rows:
                    arrays[name] = rows[name]
                elif name in planes:
                    arrays[name] = _cupy_view(planes[name], device)
            plan.bind(**arrays).launch(stream=stream)

    def _built(self, level, mode, device):
        key = (mode, device.ordinal if device.is_cuda else None)
        with LOCK:
            built = level.slots.setdefault("warp", {})
        got = built.get(key)
        if got is None:
            if device.is_cuda and device.is_capturing:
                raise RuntimeError(
                    f"eagle.frameworks.warp: {level.label!r} is not built for {mode} on {device} "
                    "and a graph capture is open (building cannot be captured); call "
                    "fn.prepare(mode, device, backward=...) or run the call once before "
                    "wp.capture_begin")
            with LOCK:
                got = built.get(key)
                if got is None:
                    got = built[key] = (self._load_device(level, mode, device) if device.is_cuda
                                        else self._load_host(level, mode))
        return got

    def _load_host(self, level, mode):
        from hawk.runtime import load

        unit = level.unit(self._dir, "host", mode)
        return load(unit.directory, level.name)

    def _load_device(self, level, mode, device):
        from hawk.artifact import plugins

        from eagle import exec as eexec
        from eagle import plan as eplan
        from eagle.registry import load_manifest

        arch = self._arch or f"sm_{device.arch}"
        unit = level.unit(self._dir, "cuda", mode, arch)
        built = plugins(unit, device_loader=load_manifest)
        return eplan.plan(built[level.name], structure=eexec.DeviceKernel)
