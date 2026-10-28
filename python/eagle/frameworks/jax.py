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
costs one transpose); a batched ``Param`` is one launch too (each sample
reads its entry of a table; a kernel that reduces over samples itself, or has a
``jump=`` plan, runs one launch per entry with ``lax.map``).

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
carries it with no ``jnp.where``. Its derivatives are those of ONE step at the
values the step read (hawk's jvp of the step; its transpose is the vjp of one
step), and a sample already finished passes its tangent and its adjoint
through, so ``jax.grad`` or ``jacfwd`` of a ``lax.scan`` over it needs no
``where`` either (``grad`` through ``while_loop`` is JAX's own refusal).
``function(kernel, wrt=())`` wraps any kernel
forward-only. The mask carries no derivative; a terminated sample's
derivatives are zero.

dtype: a call runs the float32 build or the float64 one, chosen by the dtype
of its first floating plane; a plane of another float dtype is refused by
name. Without ``jax_enable_x64`` every JAX array is float32, so the float32
build is what runs.

Devices: the CPU platform and CUDA. A function runs on the JAX device it was
built under (build one per GPU; a call on a GPU it was not built for is refused
by name). Modules carry only a
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

from .. import FINISHED_PLANE, _cell_rows, _layout, _split
from .._graph_reduce import POLICIES, check_policy
from .._graph_reduce import fixed_f64 as _fixed_f64
from ._kernel_planes import ACTIVE_PLANES, FUSED_STEPS, SAMPLE_ROLES, call_mode, promote
from ._tower import LOCK, Level, finished_identity, per_cell_source

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
                ("alias_result", ctypes.POINTER(ctypes.c_int32)),
                ("reduce", ctypes.c_uint64), ("ncells", ctypes.c_int32),
                ("cell_row", ctypes.c_int32), ("policy", ctypes.c_int32),
                ("cell_rows", ctypes.POINTER(ctypes.c_int32)),
                ("cell_comp", ctypes.POINTER(ctypes.c_int32)),
                ("cell_out", ctypes.POINTER(ctypes.c_int32)),
                ("cell_op", ctypes.POINTER(ctypes.c_int32)),
                ("work_a", ctypes.c_int32), ("work_b", ctypes.c_int32)]


def function(kernel, *, wrt=None, cache_dir=None, device_arch=None, policy="compensated"):
    """Wrap ``kernel`` as a JAX-differentiable, jit-able, vmap-able callable.

    :param kernel: a per-sample hawk kernel (``@hawk.kernel``). Its inputs may
        be ``Scalar``/``Vector``/``Matrix`` planes, ``Param`` uniforms and a
        ``Terminated`` mask; its outputs are ``Mutable`` planes.
    :param wrt: names to differentiate (default: every float input it reads).
    :param cache_dir: where builds publish (default: a private temp dir). Pass
        one directory to every process that should share builds (and to
        :func:`preload` to run exported modules).
    :param device_arch: CUDA arch to compile for (default: the GPU's own).
    :param policy: how a sum over samples (a cell) is finished: ``"compensated"``
        (default), ``"ordered"`` or ``"auto"``, as for :func:`eagle.reduce`.
    :returns: a :class:`KernelFunction`.
    """
    return KernelFunction(kernel, wrt=wrt, cache_dir=cache_dir, device_arch=device_arch,
                          policy=policy)


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

    def __init__(self, kernel, *, wrt=None, cache_dir=None, device_arch=None,
                 policy="compensated"):
        self._policy = POLICIES[check_policy(policy, _WHO)]
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
            self.level.cell_source = per_cell_source(kernel)
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
                         [(n, v.dtype) for n, v in planes.items() if sig.is_float(n)],
                         promote=True)
        for n, v in planes.items():
            if sig.is_float(n) and v.dtype != sig.dtype(n, mode):
                _layout.warn_promoted(n, v.dtype, sig.dtype(n, mode))
                planes[n] = v.astype(sig.dtype(n, mode))
        for name in self._params:
            value = bound[name] if name not in planes else planes[name]
            if not hasattr(value, "dtype"):
                value = jnp.asarray([value], dtype=sig.dtype(name, mode))
            planes[name] = jnp.reshape(value, (1,)) if jnp.ndim(value) == 0 else value
        call = self._stepped or _primitive(self.level, self)
        out = call(*(planes[n] for n in sig.inputs))
        return out[0] if len(out) == 1 else tuple(out)


def _stepped(owner):
    """The call of a carried-state kernel with its derivative rule: the
    primitive forward (state planes alias their operands), and its tangent is
    the jvp of ONE step (the kernel's pure step function, whose transpose is
    the vjp of one step) at the values the step read. A sample that was already
    finished is the identity (its step changed nothing): its state tangent
    passes through and its other outputs have none. Reverse mode is that rule
    transposed."""
    level, sig = owner.level, owner.level.sig
    prim = _primitive(level, owner)
    step = level.step_function()
    fstep = _primitive(step, owner)
    wrt = tuple(step.wrt)

    @jax.custom_jvp
    def call(*operands):
        return tuple(prim(*operands))

    @call.defjvp
    def rule(primals, tangents):
        outs = call(*primals)
        by, tan = dict(zip(sig.inputs, primals)), dict(zip(sig.inputs, tangents))
        fixed = {n: by[n] for n in step.sig.inputs}

        def run(*moving):
            return tuple(fstep(*(dict(fixed, **dict(zip(wrt, moving)))[n] for n in step.sig.inputs)))

        _, moved = jax.jvp(run, tuple(by[n] for n in wrt), tuple(tan[n] for n in wrt))
        mask = by[sig.terminated] if sig.finishes else None
        res = []
        for name, o in zip(sig.outputs, outs):
            nxt = name + "_next"
            if not sig.is_float(name) or nxt not in step.sig.outputs:
                res.append(np.zeros(o.shape, jax.dtypes.float0) if not sig.is_float(name)
                           else jnp.zeros_like(o))
                continue
            t = moved[step.sig.outputs.index(nxt)]
            if mask is not None:
                t = finished_identity(jnp.where, mask, tan[name], t) if name in sig.state \
                    else jnp.where(mask, jnp.zeros_like(t), t)
            res.append(t)
        return outs, tuple(res)

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
    from ..interop import import_buffer
    from .cupy import to_cupy

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
        with _on_device(ordinal):               # a previous call left JAX's device current
            args = [jax.dlpack.from_dlpack(import_buffer(a, stream=code))
                    for a in reads.values()]
        out = fn(*args)
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
        #: a kernel hawk splits in stages lowers to one ordinary call per stage,
        #: each stage's cells passed to the next as one-cell planes
        self.stages = tuple(Level(k, wrt=(), who=_WHO, label=f"{level.label} ({k.name})")
                            for k in _split.kernels_of(level.kernel))
        p = jex.core.Primitive(f"eagle_{level.name}")
        p.multiple_results = True
        # ``n``: the sample count, given when no input plane is per sample (the
        # vjp of a kernel whose only output is a cell reads the cell's cotangent)
        eager = jax.jit(lambda *a, n=None: p.bind(*a, n=n), static_argnames=("n",))
        p.def_impl(lambda *a, n=None: eager(*a, n=n))
        p.def_abstract_eval(self._abstract)
        ad.primitive_jvps[p] = self._jvp
        if level.kind == "jvp":
            ad.primitive_transposes[p] = self._transpose
        batching.primitive_batchers[p] = self._batch
        mlir.register_lowering(p, functools.partial(self._lower, "cpu"), platform="cpu")
        mlir.register_lowering(p, functools.partial(self._lower, "cuda"), platform="cuda")
        self.prim = p

    def __call__(self, *operands, n=None):
        return self.prim.bind(*operands, n=n)

    # -- shapes ---------------------------------------------------------------
    def _mode_n(self, avals, n=None):
        sig = self.sig
        given = n
        mode = call_mode(f"{_WHO} {self.level.label}",
                         [(n, a.dtype) for n, a in zip(sig.inputs, avals) if sig.is_float(n)])
        n = next((a.shape[-1] for nm, a in zip(sig.inputs, avals)
                  if (sig.role[nm] in SAMPLE_ROLES or nm in sig.state) and a.ndim), given)
        if n is None:
            raise ValueError(f"{_WHO}: {self.level.label!r} needs a per-sample input plane")
        cells = self._cells(avals)
        for name, a in zip(sig.inputs, avals):
            want = sig.shape(name, n, cells)
            dt = sig.dtype(name, mode)
            if tuple(a.shape) != want:
                raise ValueError(f"{_WHO}: {self.level.label}: input {name!r} must have "
                                 f"shape {want} for n={n}, got {tuple(a.shape)}")
            if np.dtype(a.dtype) != dt:
                raise TypeError(f"{_WHO}: {self.level.label}: input {name!r} must be "
                                f"{dt.name} in a {mode} call, got {np.dtype(a.dtype).name}")
        return mode, n

    def _cells(self, avals):
        """The entries of a per-cell kernel's parameter tables (1 for any other kernel)."""
        sig = self.sig
        return next((a.shape[0] for nm, a in zip(sig.inputs, avals) if sig.is_table(nm)), 1)

    def _abstract(self, *avals, n=None):
        mode, n = self._mode_n(avals, n)
        cells = self._cells(avals)
        return [jax.core.ShapedArray(self.sig.shape(o, n, cells), self.sig.dtype(o, mode))
                for o in self.sig.outputs]

    # -- lowering -------------------------------------------------------------
    def _lower(self, platform, ctx, *args, n=None):
        mode, n = self._mode_n(ctx.avals_in, n)
        cells = self._cells(ctx.avals_in)
        if not self.stages:
            run = self._ffi(self.level, platform, mode, n, cells)
        else:
            calls = [(st, self._ffi(st, platform, mode, n, cells)) for st in self.stages]

            def run(*operands):
                values = dict(zip(self.sig.inputs, operands))
                for st, call in calls:
                    values.update(zip(st.sig.outputs, call(*(values[x] for x in st.sig.inputs))))
                return [values[o] for o in self.sig.outputs]

        return mlir.lower_fun(run, multiple_results=True)(ctx, *args)

    def _ffi(self, level, platform, mode, n, cells=1):
        """One kernel's FFI call: its input planes in, its outputs out."""
        sig = level.sig
        device = platform == "cuda"
        work = _work_elems(sig, n, mode, self.owner._policy) if device else 0
        digest = _register(level, self.owner, platform, mode)
        conv = _convention(sig, device)
        # a cell's rows sit in the first ceil(n / ROW) of an n-sample plane (the handler sizes every
        # per-sample slot at n); the handler finishes them into the (1,) cell result
        shapes = [jax.ShapeDtypeStruct(_result_shape(sig, r, n, work, cells),
                                       _result_dtype(sig, r, mode))
                  for r in conv["results"]]
        call = jax.ffi.ffi_call(TARGET, shapes, input_output_aliases=conv["aliases"] or None)

        def run(*operands):
            if sig.fused:
                # the step-count word, always 1: an int64 cell. Passed as its two
                # little-endian int32 halves, since without x64 JAX has no int64
                operands = (*operands, jnp.asarray([1, 0], np.int32))
            if sig.active:
                # the identity index map over all n samples: the unguarded answer
                operands = (*operands, jnp.arange(n, dtype=np.int32), jnp.asarray([n], np.uint32))
            if sig.per_cell:
                # each sample's parameter cell, sample ``i`` reading cell ``i % cells``: the
                # int64 plane as its little-endian int32 halves (JAX has no int64 without x64)
                index = jnp.tile(jnp.arange(cells, dtype=np.int32), n // cells)
                operands = (*operands, jnp.stack([index, jnp.zeros_like(index)], -1).reshape(-1))
            out = call(*operands, digest=digest)
            return [out[i] for i in conv["visible"]]

        return run

    # -- forward mode ----------------------------------------------------------
    def _jvp(self, primals, tangents, *, n=None):
        level, sig = self.level, self.sig
        outs = self.prim.bind(*primals, n=n)
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
        touts = _primitive(derived, self.owner)(*(by[k] for k in derived.sig.inputs), n=n)
        res = []
        for o, z in zip(sig.outputs, zero_out):
            t = names.get(o) if o != sig.terminated else None
            res.append(touts[derived.sig.outputs.index(t)] if t in derived.sig.outputs else z)
        return outs, res

    # -- reverse mode: the transpose of a jvp level is the vjp of its base ------
    def _transpose(self, cts, *args, n=None):
        level = self.level
        # the vjp may read no per-sample plane (a cell's cotangent only): it is
        # told the sample count this call ran over
        _, n = self._mode_n([a.aval if ad.is_undefined_primal(a) else jax.typeof(a)
                             for a in args], n)
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
        grads = _primitive(reverse, self.owner)(*(by[k] for k in reverse.sig.inputs), n=n)
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
            if tuple(g.shape) != tuple(aval.shape):   # a cell arrives finished (hawk main: per-sample terms)
                g = (g if g_name in reverse.sig.cells else jnp.sum(g)).reshape(aval.shape)
            res.append(g.astype(aval.dtype))
        return res

    # -- vmap -----------------------------------------------------------------
    def _batch(self, args, dims, *, n=None):
        sig = self.sig
        size = next(a.shape[d] for a, d in zip(args, dims) if d is not None)
        batched_param = any(d is not None and sig.is_cell(nm) for nm, d in zip(sig.inputs, dims))
        # a Param's gradient is a sum over the folded samples: one entry per batch entry
        twin = self.level.cell_twin() if (batched_param or sig.cells) and not sig.per_cell else None
        if twin is not None and set(sig.inputs) <= set(twin.sig.inputs) \
                and set(sig.outputs) <= set(twin.sig.outputs):
            return self._batch_cells(twin, args, dims, size, n)
        # a batched cell input, or a cell output (a sum over the folded samples), cannot fold;
        # nor can a per-cell kernel (its cell index counts the samples of one entry)
        if sig.cells or batched_param or sig.per_cell:
            moved = [jnp.moveaxis(a, d, 0) if d is not None else None
                     for a, d in zip(args, dims)]

            def one(xs):
                return self.prim.bind(*(x if x is not None else a
                                        for x, a in zip(xs, args)), n=n)

            outs = jax.lax.map(one, moved)
            return outs, [0] * len(outs)
        folded = [a if sig.is_cell(nm) else _fold(a, d, size)
                  for nm, a, d in zip(sig.inputs, args, dims)]
        outs = self.prim.bind(*folded, n=None if n is None else n * size)
        res = [o.reshape(o.shape[:-1] + (o.shape[-1] // size, size)) for o in outs]
        return res, [o.ndim - 1 for o in res]

    def _batch_cells(self, twin, args, dims, size, n):
        """A batch of ``Param`` values as ONE launch over ``n * size`` samples, sample ``i``
        of entry ``b`` at ``i * size + b`` reading cell ``b`` of the parameter tables
        (entry ``b`` of a table is its gradient too: the table's scatter is the sum over
        the samples that read it)."""
        sig, by = self.sig, {nm: (a, d) for nm, a, d in zip(self.sig.inputs, args, dims)}
        folded = []
        for name in twin.sig.inputs:
            a, d = by[name]
            if sig.is_cell(name):
                a = (jnp.broadcast_to(a, (size,)) if d is None
                     else jnp.moveaxis(a, d, 0).reshape(size))
            else:
                a = _fold(a, d, size)
            folded.append(a)
        outs = _primitive(twin, self.owner)(*folded, n=None if n is None else n * size)
        res, bdims = [], []
        for name in sig.outputs:
            o = outs[twin.sig.outputs.index(name)]
            if name in sig.cells:                  # a sum over the samples: one entry per cell
                res.append(o.reshape(size, 1))
                bdims.append(0)
            else:
                res.append(o.reshape(o.shape[:-1] + (o.shape[-1] // size, size)))
                bdims.append(o.ndim)
        return res, bdims


def _fold(a, d, size):
    """A plane batched along axis ``d`` (or not at all) as one sample axis, sample-major."""
    if d is None:
        a = jnp.broadcast_to(a[..., None], a.shape + (size,))
    else:
        a = jnp.moveaxis(a, d, -1)
    return a.reshape(a.shape[:-2] + (a.shape[-2] * size,))


def _zero(x):
    """The symbolic zero tangent of ``x`` (float0 for a bool or integer)."""
    return ad.Zero(jax.typeof(x).to_tangent_aval())


# -- the call convention and the recipe ------------------------------------------

def _carried(sig):
    """The planes a call carries in place: its carried state, aliased to the
    operand and never zero-filled."""
    return sig.state


def _result_dtype(sig, name, mode):
    """The element type of a result of the call: the cells and the work results are float."""
    if isinstance(name, tuple):
        return sig.dtype(next(iter(sig.cells)), mode)
    return sig.dtype(name, mode)


def _result_shape(sig, name, n, work, cells=1):
    """The shape of a result: a cell's (1,), a work buffer's own, the rows' n samples."""
    if isinstance(name, tuple):
        return (1,) if name[0] == "cell" else (work,)
    return (n,) if name in sig.rows else sig.shape(name, n, cells)


def _work_elems(sig, n, mode, policy):
    """Elements each CUDA reduction work result holds (0 with no cells)."""
    if not sig.cells:
        return 0
    from .. import _core

    dtype = 0 if sig.dtype(next(iter(sig.cells)), mode) == np.float32 else 1
    return int(_core.reduce_canonical_device(0, -(-n // _cell_rows.ROW), dtype, 0, 0, 0, 0, 0, -1,
                                             policy, 1))


def _convention(sig, device=False):
    """Operands, FFI results, the visible results and the mask alias of ``sig``:
    operands = its inputs; results = its output planes, then the
    ``finished_count`` cell (zeroed, discarded), then the finished mask. A cell's
    rows stay results; each cell adds a (1,) result the handler finishes them into,
    and a CUDA call (``device``) two reduction work results. The
    finished mask and every carried-state plane alias their operand and are never
    zeroed: a sample the kernel does not write (a terminated one) keeps its input
    value. A pure output is zero where the kernel does not write."""
    planes = [o for o in sig.outputs if o != sig.terminated or not sig.finishes]
    planes += sorted(sig.rows - set(sig.cells))      # the cells' companion rows
    planes += [("cell", o) for o in sig.cells] + ([("work", 0), ("work", 1)] if device and sig.cells else [])
    results = planes + ([FINISHED_PLANE] if sig.counter else []) + (
        [sig.terminated] if sig.finishes else [])
    visible = [results.index(("cell", o) if o in sig.cells else o) for o in sig.outputs]
    #: a cell: (rows, companion rows or None, the cell result, op)
    cells = [(results.index(o), results.index(_cell_rows.comp_name(o)) if
              _cell_rows.comp_name(o) in results else None, results.index(("cell", o)),
              sig.cells[o]) for o in sig.cells]
    aliases = {sig.inputs.index(s): results.index(s) for s in _carried(sig)}
    if sig.finishes:
        aliases[sig.inputs.index(sig.terminated)] = len(results) - 1
    return {"operands": list(sig.inputs) + ([FUSED_STEPS] if sig.fused else [])
            + (list(ACTIVE_PLANES) if sig.active else [])
            + ([sig.cell_plane] if sig.per_cell else []),
            "results": results, "visible": visible, "cells": cells,
            "work": [results.index(("work", k)) for k in range(2 if device and sig.cells else 0)],
            "aliases": aliases}


def _recipe_fields(level, artifact, mode, device=False):
    """The platform-free half of ``level``'s recipe, from its built sidecar."""
    from hawk.runtime import descriptor_for_sidecar

    sig = level.sig
    conv = _convention(sig, device)
    sidecar = artifact.sidecar
    spec = [tuple(p) for p in sidecar["arg_spec"]]
    own = [p for p in spec if not (sig.active and p[1] in ACTIVE_PLANES)]
    if own != list(sig.arg_spec):
        raise RuntimeError(f"{_WHO}: {level.label}: the built artifact's arg_spec "
                           f"{spec} is not the kernel's {list(sig.arg_spec)}")
    kinds = descriptor_for_sidecar(sidecar)
    ops, rets = conv["operands"], conv["results"]
    declared = sidecar.get("arg_dtypes", {})
    dtype = []
    for name in ops + rets:
        dt = _result_dtype(sig, name, mode)
        if not isinstance(name, tuple) and name in declared and name != FINISHED_PLANE and np.dtype(declared[name]) != dt:
            raise RuntimeError(f"{_WHO}: {level.label}: slot {name!r} is built as "
                               f"{declared[name]}, the bridge expects {dt.name}")
        dtype.append(_XLA_DTYPE[np.dtype(np.int32) if name in (FUSED_STEPS, sig.cell_plane) else dt])
    zero = [0 if (sig.finishes and r == sig.terminated) or r in _carried(sig) or isinstance(r, tuple) else 1
            for r in rets]
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
        # the cell plane (int64 as two int32 halves) is checked against no sample count
        slot_cell.append(1 if role in ("lookup", "accum_out") or name == sig.cell_plane else 0)
    sample = next((i for i, n in enumerate(ops) if sig.role[n] in SAMPLE_ROLES), None)
    if sample is None:
        sample = len(ops) + next(i for i, r in enumerate(rets) if r != FINISHED_PLANE)
    from hawk import _core as hawk_core

    return {"nargs": len(ops), "nrets": len(rets), "dtype": dtype, "zero": zero,
            "slot_kind": slot_kind, "slot_buffer": slot_buffer, "slot_cell": slot_cell,
            "sample_buffer": sample,
            #: buffer indices (operands, then results)
            "cells": [(len(ops) + r, -1 if c is None else len(ops) + c, len(ops) + o, _CELL_OP[op])
                      for r, c, o, op in conv["cells"]],
            "work": [len(ops) + w for w in conv["work"]],
            "aliases": sorted(conv["aliases"].items()),
            "index_bytes": int(hawk_core.build_info()["index_type_bytes"])}


def _install(digest, fields, *, platform, entry, ordinal=0, block=256, function=0,
             seam=None, policy=POLICIES["compensated"]):
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
    cells = fields["cells"]
    cols = {k: (ctypes.c_int32 * max(1, len(cells)))(*[c[i] for c in cells])
            for i, k in enumerate(("cell_rows", "cell_comp", "cell_out", "cell_op"))}
    # the host finishes float64 sums with the machine GPU's arithmetic (bitwise host == device)
    fixed = platform == "cpu" and fields["cells"] and _fixed_f64()
    r.ncells, r.cell_row, r.policy = len(cells), _cell_rows.ROW, policy | (_FIXED_F64 if fixed else 0)
    r.cell_rows, r.cell_comp, r.cell_out, r.cell_op = (cols[k] for k in ("cell_rows", "cell_comp", "cell_out", "cell_op"))
    r.work_a, r.work_b = fields["work"] or (-1, -1)
    if seam is not None:
        r.reduce = seam["reduce"]
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
            digest = f"{unit.digest}/{level.name}/{mode}/p{owner._policy}/cpu"
            fields = _recipe_fields(level, level.artifact(unit), mode)
            _install(digest, fields, policy=owner._policy, platform="cpu", entry=plugin.host_entry)
            level.slots[("keep", platform, mode)] = plugin
            _write_recipe(unit, level, digest, fields, owner._policy, "cpu", mode, "")
        else:
            seam = _seam()
            ordinal, arch = _cuda_device(owner._device_arch)
            unit = level.unit(owner._dir, "cuda", mode, arch)
            plugin = _load_device(unit, level.name, ordinal)
            digest = f"{unit.digest}/{level.name}/{mode}/p{owner._policy}/cuda-{arch}"
            fields = _recipe_fields(level, level.artifact(unit), mode, device=True)
            _install(digest, fields, policy=owner._policy, platform="cuda", entry=seam["run"],
                     ordinal=ordinal, function=plugin.device_function, seam=seam)
            level.slots[("keep", platform, mode)] = plugin
            _write_recipe(unit, level, digest, fields, owner._policy, "cuda", mode, arch)
        level.slots[key] = digest
        return digest


def _write_recipe(unit, level, digest, fields, policy, platform, mode, arch):
    art = level.artifact(unit)
    record = {"digest": digest, "kernel": level.name, "platform": platform,
              "mode": mode, "arch": arch, "manifest": str(unit.manifest_path),
              "policy": policy, "fields": fields}
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
                _install(record["digest"], fields, policy=record["policy"], platform="cpu", entry=entry)
            else:
                from eagle.registry import load_manifest

                seam = _seam()
                ordinal, _ = _cuda_device(record["arch"])
                import cupy

                with cupy.cuda.Device(ordinal):
                    registry = load_manifest(record["manifest"])
                loaded = registry[record["kernel"]]
                _KEEP.append((registry, loaded))
                _install(record["digest"], fields, policy=record["policy"], platform="cuda",
                         entry=seam["run"], ordinal=ordinal, function=int(loaded.fn), seam=seam)
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

#: the cells' ops as the handler's EAGLE_XLA_CELL_* (mean rows already carry 1/n)
_CELL_OP = MappingProxyType({"sum": 0, "mean": 0, "max": 1, "min": 2})
#: EAGLE_XLA_POLICY_FIXED_F64 (eagle_xla.h)
_FIXED_F64 = 0x100


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
                                    ("memcpy", "eagle_backend_memcpy_device"),
                                    ("reduce", "eagle_backend_x_reduce_canonical"))}
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


def from_eagle(a):
    """The :class:`eagle.Array` ``a`` as a jax array (DLPack)."""
    return jax.dlpack.from_dlpack(a)
