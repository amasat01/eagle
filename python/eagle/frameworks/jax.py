# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A hawk per-sample kernel as a JAX function: :func:`function`.

The kernel becomes a JAX primitive lowered to ONE generic XLA FFI custom call
(``libeagle_xla.so``) that runs the compiled kernel in place on XLA's own
buffers -- on the CPU inline on XLA's thread, on a GPU on XLA's stream through
eagle's backend seam. Eager calls and ``jax.jit`` take the same path.

Derivatives come from hawk, never from a tape: the forward-mode rule runs the
kernel hawk derives with :func:`hawk.diff.jvp`, that kernel is linear in its
tangents and its transpose runs the :func:`hawk.diff.vjp` kernel, and every
further level is derived on demand. So ``jax.grad``, ``jvp``, ``vjp``,
``linearize``, ``jacfwd``, ``jacrev``, ``hessian`` and nested derivatives of
any order work, under ``jit`` or not. ``jax.vmap`` folds the batch into the
sample axis (one launch; zero-copy for ``in_axes=-1``, a leading batch axis
costs one transpose); a batched ``Param`` runs one launch per batch entry
(``lax.map``).

Conventions, off the kernel's own declaration (as for the torch bridge):
inputs are its planes, ``Param`` uniforms and ``Terminated`` mask in declared
order; outputs are its ``Mutable`` planes (bare if one), plus the updated mask
last when the kernel's body finishes it (pass it with ``donate_argnums`` to
update it in place). A per-sample plane is ``(n,)`` or ``(width, n)``
(component-major); a ``Param`` is a number or a one-cell array. Outputs are
allocated by XLA and zero where the kernel does not write (a terminated
sample). A ``Mutable`` plane the kernel also reads (carried state, the shape
``until_done`` steps) is an input too, in declared order, and comes back aliased
to it: a sample the kernel does not write keeps its input value, so a loop body
carries it with no ``jnp.where``. Its reverse runs the vjp of one step
(``hawk.diff.step_vjp``) at the values the step read, and a sample already
finished passes its adjoint through, so ``jax.grad`` of a ``lax.scan`` over it
needs no ``where`` either (``grad`` through ``while_loop`` is JAX's own refusal;
forward mode is refused too). ``function(kernel, wrt=())`` wraps any kernel
forward-only. The mask carries no derivative; a terminated sample's
derivatives are zero.

dtype: a call runs the float32 build or the float64 one, chosen by the dtype
of its first floating plane; a plane of another float dtype is refused by
name. Without ``jax_enable_x64`` every JAX array is float32, so the float32
build is what runs.

Devices: the CPU platform and ONE CUDA device per process (a call on another
GPU is refused by name; multi-GPU arrives in 0.7). Modules carry only a
kernel's digest, never an address, so the persistent compilation cache hits
across processes and an exported module is portable; running one in a fresh
process needs its kernels loaded first (:func:`preload`).

Thread safety: tracing, deriving, building and registering a kernel run under
one process-wide lock (:data:`eagle.frameworks._tower.LOCK`), so concurrent
first calls of a fresh kernel build each level once; launches take no lock
(the handler's recipe table is behind a reader-writer lock).

JAX is imported by this submodule only: ``import eagle`` stays JAX-free.
"""

from __future__ import annotations

import ctypes
import functools
import json
import tempfile
import threading
from pathlib import Path
from types import MappingProxyType

import jax
import jax.numpy as jnp
import numpy as np
from jax import extend as jex
from jax.interpreters import ad, batching, mlir

from .. import FINISHED_PLANE
from ._kernel_planes import FUSED_STEPS, SAMPLE_ROLES, call_mode, promote
from ._tower import LOCK, Level, finished_identity

__all__ = ["KernelFunction", "XLA_FFI_API", "function", "node", "preload"]

#: The XLA FFI API version (major, minor) of the vendored headers
#: ``libeagle_xla.so`` is built against (jaxlib 0.11.2). The JAX floor is the
#: release those headers come from: an older runtime with newer headers is
#: the unsupported direction.
XLA_FFI_API = (0, 4)

#: The FFI target both platforms register.
TARGET = "eagle_xla_launch"
#: The file next to each built unit that lets :func:`preload` re-register it.
RECIPE_FILE = "eagle_jax_recipe.json"

_WHO = "eagle.frameworks.jax"

# XLA_FFI_DataType codes (xla/ffi/api/c_api.h).
_XLA_DTYPE = MappingProxyType({np.dtype(np.bool_): 1, np.dtype(np.int32): 4, np.dtype(np.int64): 5,
                               np.dtype(np.uint8): 6, np.dtype(np.uint32): 8, np.dtype(np.float32): 11,
                               np.dtype(np.float64): 12})
# eagle_xla.h slot kinds, by hawk.runtime descriptor kind.
_SLOT = MappingProxyType({"gref": 1, "handle": 2, "int_handle": 3, "nsamples": 4})
_INOUT = 5


class _Recipe(ctypes.Structure):
    """``struct eagle_xla_recipe`` (``python/src/xla/eagle_xla.h``)."""

    _fields_ = [("struct_size", ctypes.c_uint32), ("platform", ctypes.c_uint32),
                ("ordinal", ctypes.c_int32), ("block", ctypes.c_uint32),
                ("entry", ctypes.c_uint64), ("last_error", ctypes.c_uint64),
                ("memset", ctypes.c_uint64), ("memcpy", ctypes.c_uint64),
                ("function", ctypes.c_uint64),
                ("nargs", ctypes.c_int32), ("nrets", ctypes.c_int32),
                ("dtype", ctypes.POINTER(ctypes.c_int32)),
                ("zero", ctypes.POINTER(ctypes.c_uint8)),
                ("nslots", ctypes.c_int32), ("sample_buffer", ctypes.c_int32),
                ("slot_kind", ctypes.POINTER(ctypes.c_int32)),
                ("slot_buffer", ctypes.POINTER(ctypes.c_int32)),
                ("slot_cell", ctypes.POINTER(ctypes.c_uint8)),
                ("index_bytes", ctypes.c_int32), ("nalias", ctypes.c_int32),
                ("alias_operand", ctypes.POINTER(ctypes.c_int32)),
                ("alias_result", ctypes.POINTER(ctypes.c_int32))]


def function(kernel, *, wrt=None, cache_dir=None, device_arch=None):
    """Wrap ``kernel`` as a JAX-differentiable, jit-able, vmap-able callable.

    :param kernel: a per-sample hawk kernel (``@hawk.kernel``). Its inputs may
        be ``Scalar``/``Vector``/``Matrix`` planes, ``Param`` uniforms and a
        ``Terminated`` mask; its outputs are ``Mutable`` planes.
    :param wrt: names to differentiate (default: every float input it reads).
    :param cache_dir: where builds publish (default: a private temp dir). Pass
        one directory to every process that should share builds (and to
        :func:`preload` to run exported modules).
    :param device_arch: CUDA arch to compile for (default: the GPU's own).
    :returns: a :class:`KernelFunction`.
    """
    return KernelFunction(kernel, wrt=wrt, cache_dir=cache_dir, device_arch=device_arch)


class KernelFunction:
    """A hawk kernel as a JAX function (see :func:`function`).

    :ivar inputs: the input names, in positional order.
    :ivar outputs: the output names, in return order: the kernel's own slot
        order (``arg_spec``, which lists the ``Mutable`` planes sorted by name,
        not as declared), the ``Terminated`` mask last when
        :attr:`finishes_terminated`. Read it here instead of assuming.
    :ivar wrt: the inputs that carry a derivative.
    :ivar finishes_terminated: whether the kernel's body finishes its mask.
    """

    def __init__(self, kernel, *, wrt=None, cache_dir=None, device_arch=None):
        if getattr(getattr(getattr(kernel, "kind", None), "guard", None), "active_set", False):
            raise ValueError(
                f"{_WHO}: kernel {kernel.name!r} is built with Guard(active_set=True), whose index "
                "map is eagle's launcher's input, not a JAX array; run it with eagle "
                "(eagle.plan / eagle.until_done), or build it without active_set")
        self._declared = kernel
        promoted = promote(kernel)
        self._tmp = None
        if cache_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="eagle_jax_")
            cache_dir = self._tmp.name
        self._dir = Path(cache_dir)
        self._device_arch = device_arch
        with LOCK:
            self.level = Level(promoted, wrt=wrt, who=_WHO, label=kernel.name)
            self.level.slots["owner"] = self
        sig = self.level.sig
        self.name = kernel.name
        role = {n: r for r, n in kernel.arg_spec}
        #: positional order: the kernel's declared planes, as the torch bridge
        self.inputs = tuple(n for n in kernel.planes if n in sig.role
                            and n in sig.inputs)
        self._params = frozenset(n for n in self.inputs if role.get(n) == "uniform")
        self.outputs = sig.outputs
        self.finishes_terminated = sig.finishes
        self.wrt = tuple(self.level.wrt) if any(sig.differentiable(n) for n in sig.inputs) else ()
        #: a differentiable carried-state kernel runs through its step vjp
        self._stepped = _stepped(self) if sig.state and self.wrt else None

    def __repr__(self) -> str:
        return (f"KernelFunction({self.name!r}, inputs={list(self.inputs)}, "
                f"outputs={list(self.outputs)}, wrt={list(self.wrt)})")

    def __call__(self, *args, **kwargs):
        """Run the kernel on ``args``/``kwargs``; its outputs as JAX arrays
        (bare when there is one)."""
        if len(args) > len(self.inputs):
            raise TypeError(f"{self.name}: takes {len(self.inputs)} inputs "
                            f"{list(self.inputs)}, got {len(args)}")
        bound = dict(zip(self.inputs, args))
        for name, value in kwargs.items():
            if name not in self.inputs:
                raise TypeError(f"{self.name}: {name!r} is not an input (inputs: "
                                f"{list(self.inputs)})")
            if name in bound:
                raise TypeError(f"{self.name}: input {name!r} given twice")
            bound[name] = value
        missing = [n for n in self.inputs if n not in bound]
        if missing:
            raise TypeError(f"{self.name}: missing input(s) {missing}")
        sig = self.level.sig
        planes = {}
        for name in self.inputs:
            value = bound[name]
            if name in self._params and not hasattr(value, "dtype"):
                continue                     # a Python number: cast to the mode below
            planes[name] = value if isinstance(value, jax.Array) or hasattr(
                value, "aval") else jnp.asarray(value)
        floats = [(n, v.dtype) for n, v in planes.items()
                  if sig.is_float(n) and n not in self._params]
        mode = call_mode(f"{_WHO} {self.name}", floats or
                         [(n, v.dtype) for n, v in planes.items() if sig.is_float(n)])
        for name in self._params:
            value = bound[name] if name not in planes else planes[name]
            if not hasattr(value, "dtype"):
                value = jnp.asarray([value], dtype=sig.dtype(name, mode))
            planes[name] = jnp.reshape(value, (1,)) if jnp.ndim(value) == 0 else value
        call = self._stepped or _primitive(self.level, self)
        out = call(*(planes[n] for n in sig.inputs))
        return out[0] if len(out) == 1 else tuple(out)


def _stepped(owner):
    """The call of a carried-state kernel with its reverse rule: the primitive
    forward (state planes alias their operands), and backward the vjp of ONE
    step (``hawk.diff.step_vjp``) at the values the step read, where a sample
    that was already finished is the identity (its step changed nothing).
    Forward mode (``jvp``) has no rule here and JAX refuses it."""
    level, sig = owner.level, owner.level.sig
    prim = _primitive(level, owner)

    @jax.custom_vjp
    def call(*operands):
        return tuple(prim(*operands))

    def forward(*operands):
        return tuple(prim(*operands)), operands      # the residual: the state before the step

    def backward(operands, cts):
        back = level.stepped()
        nexts, grads = level.step_names()
        by = dict(zip(sig.inputs, operands))
        after = {o: ct for o, ct in zip(sig.outputs, cts) if o in nexts}
        by.update({nexts[o]: ct for o, ct in after.items()})
        out = _primitive(back, owner)(*(by[n] for n in back.sig.inputs))
        mask = by.get(sig.terminated) if sig.finishes else None
        res = []
        for name, a in zip(sig.inputs, operands):
            g_name = grads.get(name)
            if g_name not in back.sig.outputs:
                res.append(None)
                continue
            g = out[back.sig.outputs.index(g_name)]
            if mask is not None:
                g = (finished_identity(jnp.where, mask, after[name], g) if name in sig.state
                     else jnp.where(mask, jnp.zeros_like(g), g))
            if tuple(g.shape) != tuple(a.shape):
                g = jnp.sum(g).reshape(a.shape)      # a cell's gradient: the batch sum
            res.append(g.astype(a.dtype))
        return tuple(res)

    call.defvjp(forward, backward)
    return call


# -- a JAX step between eagle stages -----------------------------------------------

class _Retain:
    """The JAX results a node's landing copies still read: ``(event, view)``
    pairs, the event recorded on the node's stream after the copy. JAX frees a
    dropped result on its own streams, not on the node's, so a result is kept
    until its copy ran (pruned at the node's next call)."""

    def __init__(self):
        self._held = []
        self._lock = threading.Lock()

    def keep(self, event, view):
        with self._lock:
            self._held.append((event, view))

    def prune(self):
        with self._lock:
            self._held = [(e, v) for e, v in self._held if not e.done]


def node(fn, *, reads, writes, name=None):
    """A JAX step as a node of an :class:`eagle.Graph`, stream-ordered: it
    builds the ``stream=`` tier of :func:`eagle.graph.opaque` for you (``name``
    is its label, which ``vjps=`` keys on). ``fn``
    takes one JAX array per plane of ``reads`` (a ``{name: array}`` mapping,
    in order) and returns one per plane of ``writes`` (jit it: an un-jitted
    call costs host time). Each launch hands eagle's planes to JAX zero-copy,
    fenced on the device, and copies JAX's results into ``writes`` on the
    node's own stream; the host never waits. On the CPU it is an eager call.
    For a JAX call that reads values back on the host use
    ``opaque(fn, eager=True)``."""
    from ..graph import _on_device, opaque
    from ..interop import import_buffer, to_cupy

    reads, writes = dict(reads), dict(writes)
    ordinals = {a.device.id for a in (*reads.values(), *writes.values())
                if type(a).__module__.split(".")[0] == "cupy"}
    if len(ordinals) > 1:
        raise ValueError(f"{_WHO}.node: the planes span devices {sorted(ordinals)}; "
                         "a node runs on one device")
    ordinal = next(iter(ordinals), None)
    stream = None
    if ordinal is not None:
        import cupy
        with _on_device(ordinal):               # the stream belongs to the planes' device
            stream = cupy.cuda.Stream(non_blocking=True)
    retain = _Retain()

    def run():
        if stream is not None:
            retain.prune()
        code = None if stream is None else stream.ptr
        out = fn(*(jax.dlpack.from_dlpack(import_buffer(a, stream=code))
                   for a in reads.values()))
        out = out if isinstance(out, (tuple, list)) else (out,)
        if len(out) != len(writes):
            raise ValueError(f"{_WHO}.node: fn returned {len(out)} array(s) for "
                             f"{len(writes)} write(s) {list(writes)}")
        with _on_device(ordinal):               # JAX leaves its own context current
            for dst, result in zip(writes.values(), out):
                view = import_buffer(result, stream=code)
                if stream is None:
                    dst[...] = np.from_dlpack(view)
                    continue
                cupy.copyto(dst, to_cupy(view))
                event = cupy.cuda.Event(disable_timing=True)
                event.record(stream)
                retain.keep(event, view)

    return opaque(run, reads=reads, writes=writes, name=name,
                  **({"stream": stream} if stream is not None else {}))


# -- the primitive per level ---------------------------------------------------

def _primitive(level, owner):
    """The JAX primitive of ``level``, made (and its rules registered) once."""
    with LOCK:
        made = level.slots.get("jax")
        if made is None:
            made = _Prim(level, owner)
            level.slots["jax"] = made
        return made


class _Prim:
    def __init__(self, level, owner):
        self.level, self.owner = level, owner
        sig = level.sig
        self.sig = sig
        p = jex.core.Primitive(f"eagle_{level.name}")
        p.multiple_results = True
        eager = jax.jit(lambda *a: p.bind(*a))
        p.def_impl(lambda *a: eager(*a))
        p.def_abstract_eval(self._abstract)
        ad.primitive_jvps[p] = self._jvp
        if level.kind == "jvp":
            ad.primitive_transposes[p] = self._transpose
        batching.primitive_batchers[p] = self._batch
        mlir.register_lowering(p, functools.partial(self._lower, "cpu"), platform="cpu")
        mlir.register_lowering(p, functools.partial(self._lower, "cuda"), platform="cuda")
        self.prim = p

    def __call__(self, *operands):
        return self.prim.bind(*operands)

    # -- shapes ---------------------------------------------------------------
    def _mode_n(self, avals):
        sig = self.sig
        mode = call_mode(f"{_WHO} {self.level.label}",
                         [(n, a.dtype) for n, a in zip(sig.inputs, avals) if sig.is_float(n)])
        n = next((a.shape[-1] for nm, a in zip(sig.inputs, avals)
                  if (sig.role[nm] in SAMPLE_ROLES or nm in sig.state) and a.ndim), None)
        if n is None:
            raise ValueError(f"{_WHO}: {self.level.label!r} needs a per-sample input plane")
        for name, a in zip(sig.inputs, avals):
            want = sig.shape(name, n)
            dt = sig.dtype(name, mode)
            if tuple(a.shape) != want:
                raise ValueError(f"{_WHO}: {self.level.label}: input {name!r} must have "
                                 f"shape {want} for n={n}, got {tuple(a.shape)}")
            if np.dtype(a.dtype) != dt:
                raise TypeError(f"{_WHO}: {self.level.label}: input {name!r} must be "
                                f"{dt.name} in a {mode} call, got {np.dtype(a.dtype).name}")
        return mode, n

    def _abstract(self, *avals):
        mode, n = self._mode_n(avals)
        return [jax.core.ShapedArray(self.sig.shape(o, n), self.sig.dtype(o, mode))
                for o in self.sig.outputs]

    # -- lowering -------------------------------------------------------------
    def _lower(self, platform, ctx, *args):
        mode, n = self._mode_n(ctx.avals_in)
        digest = _register(self.level, self.owner, platform, mode)
        conv = _convention(self.sig)
        shapes = [jax.ShapeDtypeStruct(self.sig.shape(r, n), self.sig.dtype(r, mode))
                  for r in conv["results"]]
        call = jax.ffi.ffi_call(TARGET, shapes, input_output_aliases=conv["aliases"] or None)

        def run(*operands):
            if self.sig.fused:
                # the step-count word, always 1: an int64 cell. Passed as its two
                # little-endian int32 halves, since without x64 JAX has no int64
                operands = (*operands, jnp.asarray([1, 0], np.int32))
            out = call(*operands, digest=digest)
            return [out[i] for i in conv["visible"]]

        return mlir.lower_fun(run, multiple_results=True)(ctx, *args)

    # -- forward mode ----------------------------------------------------------
    def _jvp(self, primals, tangents):
        level, sig = self.level, self.sig
        outs = self.prim.bind(*primals)
        by = dict(zip(sig.inputs, primals))
        live = [(n, t) for n, t in zip(sig.inputs, tangents)
                if n in level.wrt and type(t) is not ad.Zero]
        zero_out = [_zero(o) for o in outs]
        if level.wrt == () and any(type(t) is not ad.Zero and sig.differentiable(n)
                                   for n, t in zip(sig.inputs, tangents)):
            raise NotImplementedError(
                f"{_WHO}: {level.label!r} was wrapped with wrt=() (forward-only): it has no "
                "derivative, so jax.grad / jvp / vjp through it is refused; "
                "wrap it without wrt=() to differentiate it")
        if not live:
            return outs, zero_out
        derived = level.derived("jvp")
        names = level.names("jvp")
        given = dict(live)
        for w in level.wrt:
            p = by[w]
            by[names[w]] = given.get(w, jnp.zeros(p.shape, p.dtype))
        touts = _primitive(derived, self.owner)(*(by[n] for n in derived.sig.inputs))
        res = []
        for o, z in zip(sig.outputs, zero_out):
            t = names.get(o) if o != sig.terminated else None
            res.append(touts[derived.sig.outputs.index(t)] if t in derived.sig.outputs else z)
        return outs, res

    # -- reverse mode: the transpose of a jvp level is the vjp of its base ------
    def _transpose(self, cts, *args):
        level = self.level
        base = level.base
        tangent_of = {d: w for w, d in base.names("jvp").items() if w in base.wrt}
        known = {}
        for name, a in zip(level.sig.inputs, args):
            if ad.is_undefined_primal(a):
                if name not in tangent_of:
                    raise NotImplementedError(
                        f"{_WHO}: {level.label!r} is linear in its tangents only; "
                        f"{name!r} cannot be transposed")
            else:
                known[name] = a
        reverse = base.derived("vjp")
        bars = base.names("vjp")
        jnames = base.names("jvp")
        by = dict(known)
        for o in base.sig.outputs:
            if o == base.sig.terminated or o not in bars:
                continue
            t = jnames.get(o)
            if t not in level.sig.outputs:      # no tangent output: a zero cotangent
                sized = [(nm, a) for nm, a in zip(level.sig.inputs, args)
                         if level.sig.role[nm] in SAMPLE_ROLES]
                _, a = sized[0]
                aval = a.aval if ad.is_undefined_primal(a) else a
                mode = next(np.dtype(x.aval.dtype if ad.is_undefined_primal(x) else x.dtype)
                            for k, x in zip(level.sig.inputs, args) if level.sig.is_float(k))
                by[bars[o]] = jnp.zeros(base.sig.shape(o, aval.shape[-1]), mode)
                continue
            ct = cts[level.sig.outputs.index(t)]
            if type(ct) is ad.Zero:
                aval = ct.aval
                ct = jnp.zeros(aval.shape, aval.dtype)
            by[bars[o]] = ct
        missing = [n for n in reverse.sig.inputs if n not in by]
        if missing:
            raise NotImplementedError(
                f"{_WHO}: transposing {level.label!r} needs {missing}, which are "
                "not known primal values here")
        grads = _primitive(reverse, self.owner)(*(by[n] for n in reverse.sig.inputs))
        res = []
        for name, a in zip(level.sig.inputs, args):
            if not ad.is_undefined_primal(a):
                res.append(None)
                continue
            w = tangent_of[name]
            aval = a.aval
            g_name = bars.get(w)
            if g_name not in reverse.sig.outputs:
                res.append(ad.Zero(aval))
                continue
            g = grads[reverse.sig.outputs.index(g_name)]
            if tuple(g.shape) != tuple(aval.shape):
                g = jnp.sum(g).reshape(aval.shape)       # a cell's gradient: the batch sum
            res.append(g.astype(aval.dtype))
        return res

    # -- vmap -----------------------------------------------------------------
    def _batch(self, args, dims):
        sig = self.sig
        size = next(a.shape[d] for a, d in zip(args, dims) if d is not None)
        if any(d is not None and sig.is_cell(n)
               for n, d in zip(sig.inputs, dims)):
            moved = [jnp.moveaxis(a, d, 0) if d is not None else None
                     for a, d in zip(args, dims)]

            def one(xs):
                return self.prim.bind(*(x if x is not None else a
                                        for x, a in zip(xs, args)))

            outs = jax.lax.map(one, moved)
            return outs, [0] * len(outs)
        folded = []
        for name, a, d in zip(sig.inputs, args, dims):
            if sig.is_cell(name):
                folded.append(a)
                continue
            if d is None:
                a = jnp.broadcast_to(a[..., None], a.shape + (size,))
            else:
                a = jnp.moveaxis(a, d, -1)
            folded.append(a.reshape(a.shape[:-2] + (a.shape[-2] * size,)))
        outs = self.prim.bind(*folded)
        res = [o.reshape(o.shape[:-1] + (o.shape[-1] // size, size)) for o in outs]
        return res, [o.ndim - 1 for o in res]


def _zero(x):
    """The symbolic zero tangent of ``x`` (float0 for a bool or integer)."""
    return ad.Zero(jax.typeof(x).to_tangent_aval())


# -- the call convention and the recipe ------------------------------------------

def _carried(sig):
    """The planes a call carries in place: its carried state, aliased to the
    operand and never zero-filled."""
    return sig.state


def _convention(sig):
    """Operands, FFI results, the visible results and the mask alias of ``sig``:
    operands = its inputs; results = its output planes, then the
    ``finished_count`` cell (zeroed, discarded), then the finished mask. The
    finished mask and every carried-state plane alias their operand and are never
    zeroed: a sample the kernel does not write (a terminated one) keeps its input
    value. A pure output is zero where the kernel does not write."""
    planes = [o for o in sig.outputs if o != sig.terminated or not sig.finishes]
    results = planes + ([FINISHED_PLANE] if sig.counter else []) + (
        [sig.terminated] if sig.finishes else [])
    visible = [results.index(o) for o in sig.outputs]
    aliases = {sig.inputs.index(s): results.index(s) for s in _carried(sig)}
    if sig.finishes:
        aliases[sig.inputs.index(sig.terminated)] = len(results) - 1
    return {"operands": list(sig.inputs) + ([FUSED_STEPS] if sig.fused else []),
            "results": results, "visible": visible,
            "aliases": aliases}


def _recipe_fields(level, artifact, mode):
    """The platform-free half of ``level``'s recipe, from its built sidecar."""
    from hawk.runtime import descriptor_for_sidecar

    sig = level.sig
    conv = _convention(sig)
    sidecar = artifact.sidecar
    spec = [tuple(p) for p in sidecar["arg_spec"]]
    if spec != list(sig.arg_spec):
        raise RuntimeError(f"{_WHO}: {level.label}: the built artifact's arg_spec "
                           f"{spec} is not the kernel's {list(sig.arg_spec)}")
    kinds = descriptor_for_sidecar(sidecar)
    ops, rets = conv["operands"], conv["results"]
    declared = sidecar.get("arg_dtypes", {})
    dtype = []
    for name in ops + rets:
        dt = sig.dtype(name, mode)
        if name in declared and name != FINISHED_PLANE and np.dtype(declared[name]) != dt:
            raise RuntimeError(f"{_WHO}: {level.label}: slot {name!r} is built as "
                               f"{declared[name]}, the bridge expects {dt.name}")
        dtype.append(_XLA_DTYPE[np.dtype(np.int32) if name == FUSED_STEPS else dt])
    zero = [0 if (sig.finishes and r == sig.terminated) or r in _carried(sig) else 1 for r in rets]
    slot_kind, slot_buffer, slot_cell = [], [], []
    for (role, name), kind in zip(spec, kinds):
        if kind not in _SLOT:
            raise ValueError(f"{_WHO}: {level.label}: slot {name!r} is a by-value "
                             f"{kind!r}; a framework bridge binds planes only")
        if kind == "nsamples":
            slot_kind.append(_SLOT[kind]), slot_buffer.append(-1), slot_cell.append(0)
            continue
        if sig.finishes and name == sig.terminated:
            slot_kind.append(_INOUT)
            slot_buffer.append(len(ops) + rets.index(name))
        elif name in rets:
            slot_kind.append(_SLOT[kind])
            slot_buffer.append(len(ops) + rets.index(name))
        else:
            slot_kind.append(_SLOT[kind])
            slot_buffer.append(ops.index(name))
        slot_cell.append(1 if role == "lookup" else 0)
    sample = next((i for i, n in enumerate(ops) if sig.role[n] in SAMPLE_ROLES), None)
    if sample is None:
        sample = len(ops) + next(i for i, r in enumerate(rets) if r != FINISHED_PLANE)
    from hawk import _core as hawk_core

    return {"nargs": len(ops), "nrets": len(rets), "dtype": dtype, "zero": zero,
            "slot_kind": slot_kind, "slot_buffer": slot_buffer, "slot_cell": slot_cell,
            "sample_buffer": sample,
            "aliases": sorted(conv["aliases"].items()),
            "index_bytes": int(hawk_core.build_info()["index_type_bytes"])}


def _install(digest, fields, *, platform, entry, ordinal=0, block=256, function=0,
             seam=None):
    """Register one recipe in ``libeagle_xla.so``'s table."""
    lib = _xla()
    arr = {k: (ctypes.c_int32 * len(fields[k]))(*fields[k])
           for k in ("dtype", "slot_kind", "slot_buffer")}
    u8 = {k: (ctypes.c_uint8 * max(1, len(fields[k])))(*fields[k])
          for k in ("zero", "slot_cell")}
    r = _Recipe()
    r.struct_size = ctypes.sizeof(_Recipe)
    r.platform = 0 if platform == "cpu" else 1
    r.ordinal, r.block, r.entry, r.function = ordinal, block, entry, function
    if seam is not None:
        r.last_error, r.memset, r.memcpy = seam["last_error"], seam["memset"], seam["memcpy"]
    r.nargs, r.nrets = fields["nargs"], fields["nrets"]
    r.dtype, r.zero = arr["dtype"], u8["zero"]
    r.nslots = len(fields["slot_kind"])
    r.sample_buffer = fields["sample_buffer"]
    r.slot_kind, r.slot_buffer, r.slot_cell = (arr["slot_kind"], arr["slot_buffer"],
                                              u8["slot_cell"])
    r.index_bytes = fields["index_bytes"]
    pairs = fields["aliases"]
    a_op, a_res = [(ctypes.c_int32 * max(1, len(pairs)))(*column) for column in zip(*pairs)] or [None, None]
    r.nalias, r.alias_operand, r.alias_result = len(pairs), a_op, a_res
    status = lib.eagle_xla_register(digest.encode(), ctypes.byref(r))
    if status != 0:
        raise RuntimeError(f"{_WHO}: libeagle_xla.so refused the recipe of {digest!r} "
                           f"(status {status})")


def _register(level, owner, platform, mode):
    """Build ``level`` for ``platform`` in ``mode`` (once), register its recipe
    and return its digest -- the only thing the lowered module carries."""
    with LOCK:
        key = ("digest", platform, mode)
        digest = level.slots.get(key)
        if digest is not None:
            return digest
        if platform == "cpu":
            unit = level.unit(owner._dir, "host", mode)
            from hawk.artifact import plugins

            plugin = plugins(unit)[level.name]
            digest = f"{unit.digest}/{level.name}/{mode}/cpu"
            fields = _recipe_fields(level, level.artifact(unit), mode)
            _install(digest, fields, platform="cpu", entry=plugin.host_entry)
            level.slots[("keep", platform, mode)] = plugin
            _write_recipe(unit, level, digest, fields, "cpu", mode, "")
        else:
            seam = _seam()
            ordinal, arch = _cuda_device(owner._device_arch)
            unit = level.unit(owner._dir, "cuda", mode, arch)
            plugin = _load_device(unit, level.name, ordinal)
            digest = f"{unit.digest}/{level.name}/{mode}/cuda-{arch}"
            fields = _recipe_fields(level, level.artifact(unit), mode)
            _install(digest, fields, platform="cuda", entry=seam["run"], ordinal=ordinal,
                     function=plugin.device_function, seam=seam)
            level.slots[("keep", platform, mode)] = plugin
            _write_recipe(unit, level, digest, fields, "cuda", mode, arch)
        level.slots[key] = digest
        return digest


def _write_recipe(unit, level, digest, fields, platform, mode, arch):
    art = level.artifact(unit)
    record = {"digest": digest, "kernel": level.name, "platform": platform,
              "mode": mode, "arch": arch, "manifest": str(unit.manifest_path),
              "fields": fields}
    if platform == "cpu":
        record["host"] = {"library": str(art.entries["host"].artifact),
                          "entry": art.sidecar.get("host_entry")}
    (Path(unit.directory) / RECIPE_FILE).write_text(json.dumps(record))


def preload(cache_dir, *, platform=None):
    """Register every kernel built under ``cache_dir`` by this bridge, so a
    module that names them (a ``jax.export`` artifact, a persistent-cache hit
    in a fresh process) runs without tracing them again. CUDA kernels load on
    this process's GPU. Returns the digests registered."""
    done = []
    with LOCK:
        for path in sorted(Path(cache_dir).rglob(RECIPE_FILE)):
            record = json.loads(path.read_text())
            if platform is not None and record["platform"] != platform:
                continue
            fields = record["fields"]
            if record["platform"] == "cpu":
                from hawk import _core as hawk_core

                library = hawk_core.HostLibrary(record["host"]["library"])  # ABI check
                cdll = ctypes.CDLL(record["host"]["library"])
                entry = ctypes.cast(getattr(cdll, record["host"]["entry"]),
                                    ctypes.c_void_p).value
                _KEEP.append((library, cdll))
                _install(record["digest"], fields, platform="cpu", entry=entry)
            else:
                from eagle.registry import load_manifest

                seam = _seam()
                ordinal, _ = _cuda_device(record["arch"])
                import cupy

                with cupy.cuda.Device(ordinal):
                    registry = load_manifest(record["manifest"])
                loaded = registry[record["kernel"]]
                _KEEP.append((registry, loaded))
                _install(record["digest"], fields, platform="cuda", entry=seam["run"],
                         ordinal=ordinal, function=loaded.fn.kernel.ptr, seam=seam)
            done.append(record["digest"])
    return done


_KEEP = []          # modules preload loaded: never unloaded (a module may name them)


# -- libraries and devices --------------------------------------------------------

@functools.cache
def _xla_library():
    from .. import _core

    path = Path(_core.__file__).resolve().parent / "libeagle_xla.so"   # beside _core, also in an editable install
    if not path.is_file():
        raise ImportError(f"{_WHO}: {path} is missing; reinstall eagle")
    lib = ctypes.CDLL(str(path))
    lib.eagle_xla_register.restype = ctypes.c_int32
    lib.eagle_xla_register.argtypes = (ctypes.c_char_p, ctypes.POINTER(_Recipe))
    return lib


_TARGETS = set()


def _xla(platform="cpu"):
    """``libeagle_xla.so``, its FFI target registered for ``platform`` once."""
    lib = _xla_library()
    with LOCK:
        if platform not in _TARGETS:
            symbol = "eagle_xla_launch_cpu" if platform == "cpu" else "eagle_xla_launch_cuda"
            jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(getattr(lib, symbol)),
                                        platform="cpu" if platform == "cpu" else "CUDA")
            _TARGETS.add(platform)
    return lib


def _seam():
    """The CUDA plugin's launch, error, memset and memcpy entries, as addresses."""
    from eagle import BackendUnavailable, _core

    with LOCK:
        cached = _SEAM.get("cuda")
        if cached is not None:
            return cached
        info = _core.cuda_backend()
        if not info.get("loaded"):
            raise BackendUnavailable(
                f"{_WHO}: a CUDA call needs eagle's CUDA backend: "
                f"{info.get('error') or 'libeagle_cuda.so is not loaded'}")
        lib = ctypes.CDLL(info["path"])
        try:
            got = {key: ctypes.cast(getattr(lib, sym), ctypes.c_void_p).value
                   for key, sym in (("run", "eagle_backend_run_device"),
                                    ("last_error", "eagle_backend_last_error"),
                                    ("memset", "eagle_backend_memset_device"),
                                    ("memcpy", "eagle_backend_memcpy_device"))}
        except AttributeError as err:
            raise BackendUnavailable(
                f"{_WHO}: the CUDA backend {info['path']} predates the JAX bridge "
                f"({err}); reinstall eagle") from None
        got["lib"] = lib
        _xla("cuda")
        _SEAM["cuda"] = got
        return got


_SEAM = {}


def _cuda_device(device_arch):
    """The process's CUDA device: (ordinal, arch)."""
    try:
        devices = jax.local_devices(backend="cuda")
    except RuntimeError:
        devices = []
    if not devices:
        from eagle import BackendUnavailable

        raise BackendUnavailable(f"{_WHO}: JAX has no CUDA device in this process")
    default = jax.config.jax_default_device
    device = default if default is not None and default.platform == "gpu" else devices[0]
    ordinal = int(device.local_hardware_id)
    arch = device_arch
    if not arch:
        major, minor = str(device.compute_capability).split(".")
        arch = f"sm_{major}{minor}"
    return ordinal, arch


def _load_device(unit, name, ordinal):
    import cupy
    from hawk.artifact import plugins

    from eagle.registry import load_manifest

    with cupy.cuda.Device(ordinal):
        return plugins(unit, device_loader=load_manifest)[name]


def _register_targets():
    """Register the FFI target on import, so a module that names a kernel
    (an exported one, a cache hit) reaches the handler, which then names a
    kernel this process has not loaded instead of XLA naming a missing target."""
    _xla("cpu")
    if _has_gpu():
        _xla("cuda")


def _has_gpu():
    try:
        return bool(jax.devices("cuda"))
    except RuntimeError:
        return False


_register_targets()
