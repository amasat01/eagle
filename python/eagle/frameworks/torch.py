"""A hawk per-sample kernel as a ``torch.autograd.Function``: :func:`function`.

Forward runs the kernel; backward runs the reverse-mode kernel hawk
derives from it (:func:`hawk.diff.vjp`); forward-mode AD runs the derived
tangent kernel (:func:`hawk.diff.jvp`). Nothing is taped: each pass is one
kernel launch, and a derivative of a derivative is one more launch of the
kernel hawk derives from the derived kernel, so ``create_graph=True``,
``torch.autograd.functional.hessian`` and ``gradgradcheck`` work to any
depth. ``torch.func.grad``/``jvp``/``vmap`` run through the same functions
(``vmap`` folds the batch into the sample axis; a batched ``Param`` loops).

Every launch is ONE generic ``torch.library.custom_op``, ``eagle::launch``
(registered once per process; the kernel is named by a digest string, so there
is no per-kernel op). ``torch.compile(fullgraph=True)`` traces through it with
no graph break, forward and backward. The op needs torch 2.4; below that the
bridge runs the same launches directly, without compile support (a call under
``torch.compile`` is refused by name).

CPU tensors run through hawk's host runtime; CUDA tensors through eagle's
launch path (:func:`eagle.plan.plan`) on torch's current stream, no sync.
Every plane crosses through :func:`eagle.interop.import_buffer`
zero-copy; a non-contiguous input is made contiguous, a wrong-dtype one
refused. The build follows the call: a float32 call runs the float32 build of
the kernel, a float64 call the float64 one; a call mixing the two is refused.

Conventions, off the kernel's own declaration: inputs are its planes and
``Param`` uniforms in order; outputs are its ``Mutable`` planes (bare if
one), PLUS the ``Terminated`` mask itself when the kernel's own body
finishes it (``terminated = cond``) -- the updated mask comes back as an
extra return value, the last one, so a stepping loop reuses the SAME stop
decision the kernel made instead of recomputing it with a second masking
rule of its own (the caller's own mask tensor is left untouched). A per-sample
plane is ``(n,)``/``(width, n)``; a sample-major ``(n, width)`` input is
accepted too (zero-copy transpose where contiguous), and outputs come back in
that same layout -- mixing layouts in one call is refused. A ``Param``'s
gradient is the batch sum of its per-sample partials. A kernel reads its
Params from one-cell device planes (:func:`hawk.diff.param_planes`), so a
launch never reads a Param back to the host and ``torch.cuda.graph`` can
capture it. A ``Terminated`` mask is never
differentiated: a marked sample's outputs and gradient/tangent stay zero, and
the mask itself carries no gradient/tangent of its own.

torch is imported by this submodule only: ``import eagle`` stays torch-free.
"""

from __future__ import annotations

import itertools
import math
import tempfile
from pathlib import Path
from types import MappingProxyType

import torch

from .. import FINISHED_PLANE, _cell_rows, _split
from ._kernel_planes import INPUT_ROLES, OUTPUT_ROLES, _prior_reads, call_mode, promote
from ._tower import LOCK, finished_identity, plane_names

__all__ = ["KernelFunction", "function"]

_WHO = "eagle.frameworks.torch"

#: Roles the bridge binds as an input: the shared ones, plus the by-value uniforms
#: of a kernel as the caller wrote it (a promoted Param is a one-cell ``lookup`` plane).
_INPUT_ROLES = INPUT_ROLES | {"uniform"}
#: A Param, read by value (``uniform``) or from a one-cell device plane (``lookup``).
_PARAM_ROLES = frozenset({"uniform", "lookup"})

#: torch floating dtype of each hawk build mode.
_DTYPE_OF = MappingProxyType({"float64": torch.float64, "float32": torch.float32})

#: digest -> (level, mode): what the generic ``eagle::launch`` op runs.
_REGISTRY = {}
_UIDS = itertools.count()
#: ``None`` until the first call asks; then the op (torch >= 2.4) or ``False``.
_OP = None


def function(kernel, *, wrt=None, cache_dir=None, device_arch=None):
    """Wrap ``kernel`` as a torch-differentiable callable.

    :param kernel: a per-sample hawk kernel (``@hawk.kernel``). Its inputs
        may be ``Scalar``/``Vector``/``Matrix`` planes, ``Param`` uniforms
        and a ``Terminated`` mask; its outputs are ``Mutable`` planes.
    :param wrt: names to differentiate (default: every differentiable input).
    :param cache_dir: where artifacts publish (default: a private temp dir).
    :param device_arch: CUDA arch to compile for (default: the first call's device).
    :returns: a :class:`KernelFunction`.
    """
    return KernelFunction(kernel, wrt=wrt, cache_dir=cache_dir, device_arch=device_arch)


# -- the generic op ----------------------------------------------------------
def _as_tensor(array, device):
    """A tensor over the same memory as the numpy/cupy ``array``."""
    if device.type == "cuda":
        return torch.as_tensor(array, device=device)
    return torch.from_numpy(array)


def _launch(digest, planes):
    """What ``eagle::launch`` does: run the level ``digest`` names over ``planes``."""
    level, mode = _REGISTRY[digest]
    return level.launch(mode, planes)


def _fake(digest, planes):
    level, mode = _REGISTRY[digest]
    return level.fake(mode, planes)


def _generic_op():
    """The one ``eagle::launch`` custom op (``False`` below torch 2.4)."""
    global _OP
    if _OP is not None:
        return _OP
    with LOCK:
        if _OP is not None:
            return _OP
        if not hasattr(torch.library, "custom_op"):
            _OP = False
            return _OP

        def launch(digest, planes):
            return _launch(digest, planes)

        launch.__annotations__ = {
            "digest": str,
            "planes": list[torch.Tensor],
            "return": list[torch.Tensor],
        }
        op = torch.library.custom_op("eagle::launch", mutates_args=())(launch)
        op.register_fake(_fake)
        _OP = op
        return _OP


def _apply_op(digests, args):
    """One launch of the level whose digests (per build mode) are ``digests``:
    the generic op, or a direct call where torch has no ``custom_op``. Takes
    only plain data, so a ``torch.compile`` trace goes straight through it."""
    mode = _mode_of(args)
    if _OP is False:
        if _is_compiling():
            raise RuntimeError(
                "eagle.frameworks.torch: torch.compile support needs torch 2.4+ "
                f"(this is torch {torch.__version__})"
            )
        return tuple(_launch(digests[mode], list(args)))
    return tuple(torch.ops.eagle.launch(digests[mode], list(args)))


def _gradients(got, saved, needs, grad_plan, mask_pos, grad_outs=(), carried=()):
    """The gradients of a level out of the outputs ``got`` of its vjp level:
    each wanted input's gradient plane, masked, a Param's summed over samples.
    A carried-state input (``carried[i]`` = its output's position) of a
    finished sample passes its output's gradient through: that step was the
    identity."""
    grads = []
    for i, need in enumerate(needs):
        pos, is_param = grad_plan[i]
        if not need or pos < 0:
            grads.append(None)
            continue
        grad = got[pos]
        out = carried[i] if carried else -1
        if is_param == 2:       # a cell: it arrives finished (eagle.reduce of the rows)
            grads.append(grad.reshape(saved[i].shape).to(saved[i].device))
            continue
        if mask_pos >= 0 and out >= 0:
            grad = finished_identity(torch.where, saved[mask_pos], grad_outs[out], grad)
        elif mask_pos >= 0:
            grad = grad.masked_fill(saved[mask_pos], 0.0)
        if is_param:     # per-sample terms (no reduction over samples in the kernel)
            grad = grad.sum().reshape(saved[i].shape).to(saved[i].device)
        grads.append(grad)
    return tuple(grads)


def _backward(saved, grad_outs, needs, plan, traced):
    """One level's backward: a launch of its vjp level (``plan`` is the static
    wiring, see :meth:`_Level._plan`: plain tuples, nothing to look up). Eager,
    the launch is the vjp level's own Function, so the result is differentiable
    again; traced (``torch.compile``), it is the generic op itself, not a nested
    Function, so it traces with no graph break (a second derivative through a
    compiled region is not supported)."""
    target, args_plan, grad_plan, mask_pos, carried = plan
    args = [saved[i] if kind == 0 else grad_outs[i] for kind, i in args_plan]
    got = _apply_op(target, args) if traced else target.apply(*args)
    return _gradients(got, saved, needs, grad_plan, mask_pos, grad_outs, carried)


def _is_compiling():
    check = getattr(torch.compiler, "is_compiling", None)
    return bool(check()) if check is not None else False


def _mode_of(tensors):
    """The build mode a call selects: the dtype of its floating tensors
    (float64 if it has none, which the buffer check then refuses)."""
    floats = [
        (f"input {i}", t.dtype)
        for i, t in enumerate(tensors)
        if isinstance(t, torch.Tensor) and t.dtype.is_floating_point
    ]
    return call_mode("eagle.frameworks.torch", floats) if floats else "float64"


class _Level:
    """One hawk kernel of the derivative tower (the primal, a vjp, a jvp of a
    jvp ...) and the ``torch.autograd.Function`` that applies it.

    A level's derivatives are derived on demand, once, under the process lock;
    the planes a derived kernel declares are found from its own ``arg_spec``
    by role and name (``bar<L>_x`` / ``dot<L>_x``: the level number counts
    nesting, so it is read off the kernel, never assumed)."""

    def __init__(self, owner, kernel, *, chain, path, order=None, state=()):
        self.owner, self.kernel, self.chain, self.path = (
            owner,
            kernel,
            tuple(chain),
            path,
        )
        self.spec = tuple(kernel.arg_spec)
        self.role = {name: role for role, name in self.spec}
        #: Mutable planes read before written (carried state): inputs as well as outputs
        self.state = tuple(state)
        self.finish = kernel.walk.finish
        internal = {FINISHED_PLANE} if self.finish is not None else frozenset()
        order = list(order) if order is not None else [n for _, n in self.spec]
        self.inputs = tuple(
            n
            for n in order
            if n in self.role
            and n not in internal
            and (
                self.role[n] in _INPUT_ROLES
                or n in self.state
                or (self.role[n] == "lookup" and n != FINISHED_PLANE)
            )
        )
        self.params = frozenset(n for n in self.inputs if self.role[n] in _PARAM_ROLES)
        self.terminated = next(
            (n for n in self.inputs if self.role[n] == "terminated"), None
        )
        self.finishes_terminated = (
            self.terminated is not None
            and self.finish is not None
            and self.finish[0] == self.terminated
        )
        #: a reduction over samples: its rows are eagle's scratch, the cell is the output
        self.cells = _cell_rows.kernel_cells(kernel)
        companions = _cell_rows.row_names(self.cells) - frozenset(self.cells)
        outputs = [n for n in order if (self.role.get(n) in OUTPUT_ROLES or n in self.cells)
                   and n not in companions]
        if self.finishes_terminated:
            outputs.append(self.terminated)
        self.outputs = tuple(outputs)
        slot = kernel.walk.slot_types
        self.width = {
            n: math.prod(slot[(self.role[n], n)].shape)
            for n in (*self.inputs, *self.outputs)
            if (self.role[n], n) in slot
        }
        self.wrt = None
        self._children = {}
        #: one digest per build mode, registered at creation (the op looks the
        #: level up by it; a lock-free read keeps a traced call lock-free)
        self.digests = {}
        for mode in _DTYPE_OF:
            digest = f"{owner.uid}/{owner.name}/{self.path or 'primal'}/{mode}"
            self.digests[mode] = digest
            with LOCK:
                _REGISTRY[digest] = (self, mode)
        self._fn = self._fn_traced = None
        self.links = {}
        self._built = {}  # (mode, device index or None) -> runner

    # -- derivation ---------------------------------------------------------
    def _wrt_names(self):
        owner = self.owner
        return [
            n
            for n in self.inputs
            if self.role[n] != "terminated"
            and (n not in owner.root_inputs or n in owner.root_wrt)
        ]

    def child(self, kind):
        """The ``"vjp"`` or ``"jvp"`` level of this one."""
        got = self._children.get(kind)
        if got is None:
            with LOCK:
                got = self._children.get(kind)
                if got is None:
                    got = self._children[kind] = self._derive(kind)
        return got

    def _derive(self, kind):
        from hawk.diff import jvp, step_vjp, vjp
        from hawk.trace import Kernel

        reverse = kind == "vjp"
        if self.state and not reverse:
            raise NotImplementedError(
                f"{_WHO}: {self.owner.name!r} carries state ({list(self.state)}); its "
                "reverse runs the vjp of one step, and forward mode has no rule"
            )
        derived = ((step_vjp if self.state else vjp) if reverse else jvp)(
            self.kernel, wrt=self.wrt
        )
        kernel = Kernel(f"{self.owner.name}_{self.path}{kind[0]}", derived)
        floats = [o for o in self.outputs if o != self.terminated]
        nexts = {o: f"bar_{o}_next" for o in floats if self.state}
        named = plane_names(
            _WHO,
            self.owner.name,
            kind,
            derived.wrt,
            [] if self.state else floats,
            {n for _, n in self.spec} | set(nexts.values()),
            {n for _, n in kernel.arg_spec},
        )
        child = _Level(
            self.owner, kernel, chain=(*self.chain, kernel), path=self.path + kind[0]
        )
        child.fn  # built now, so a traced pass never builds under the lock
        child.wrt = tuple(child._wrt_names())
        if reverse:
            child.seed = nexts or {o: named[o] for o in self.outputs if o in named}
            child.grad = {w: named[w] for w in derived.wrt if w in named}
            self.links.update(
                {traced: self._plan(child, traced) for traced in (False, True)}
            )
        else:
            child.tan_in = {w: named[w] for w in derived.wrt}
            child.tan_out = {o: named[o] for o in self.outputs if o in named}
        return child

    def _plan(self, child, traced):
        """The static wiring of this level's backward through its vjp level
        ``child`` (plain tuples, so a traced backward needs no object access)."""
        in_at = {n: i for i, n in enumerate(self.inputs)}
        out_at = {o: j for j, o in enumerate(self.outputs)}
        seeded = {plane: o for o, plane in child.seed.items()}
        args = tuple(
            (0, in_at[n]) if n in in_at else (1, out_at[seeded[n]])
            for n in child.inputs
        )
        child_at = {n: j for j, n in enumerate(child.outputs)}
        grads = tuple(
            (child_at.get(child.grad.get(n), -1),
             (2 if child.grad.get(n) in child.cells else 1) if n in self.params else 0)
            for n in self.inputs
        )
        mask = in_at.get(self.terminated, -1) if self.terminated is not None else -1
        carried = tuple(out_at[n] if n in self.state else -1 for n in self.inputs)
        return (child.digests if traced else child.fn), args, grads, mask, carried

    def link(self):
        """This level's eager backward wiring (its vjp level, derived on demand)."""
        plan = self.links.get(False)
        if plan is None:
            self.child("vjp")
            plan = self.links[False]
        return plan

    @property
    def fn(self):
        """The Function that serves every eager pass (forward AD included)."""
        if self._fn is None:
            with LOCK:
                if self._fn is None:
                    self._fn_traced = _autograd_function(self, forward_ad=False)
                    self._fn = _autograd_function(self, forward_ad=True)
        return self._fn

    @property
    def fn_for_call(self):
        """The Function a call applies: under ``torch.compile`` the twin that
        has no ``jvp``, since dynamo cannot trace a Function that defines one
        (forward-mode AD through a compiled region is not supported)."""
        fn = self.fn
        return self._fn_traced if _is_compiling() else fn

    # -- shapes -------------------------------------------------------------
    def plane_shape(self, name, n):
        if name in self.cells:
            return (1,)
        return (n,) if self.width.get(name, 1) == 1 else (self.width[name], n)

    def _samples(self, values):
        """The sample count and device, read off the per-sample input planes."""
        planes = [(n, v) for n, v in values.items() if n not in self.params]
        if not planes:
            raise ValueError(f"{self.owner.name}: no per-sample input plane bound")
        n, device = None, None
        for name, value in planes:
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{self.owner.name}: input {name!r} is a per-sample plane and must "
                    f"be a torch.Tensor, got {type(value).__name__}"
                )
            if n is None:
                n, device = value.shape[-1], value.device
            if value.device != device:
                raise ValueError(
                    f"{self.owner.name}: input {name!r} is on {value.device}, "
                    f"the other planes on {device}"
                )
            if tuple(value.shape) != self.plane_shape(name, n):
                raise ValueError(
                    f"{self.owner.name}: input {name!r} must have shape "
                    f"{self.plane_shape(name, n)} for n={n}, got {tuple(value.shape)}"
                )
        return n, device

    # -- the Function's three passes ------------------------------------------
    def digest(self, mode):
        return self.digests[mode]

    def fake(self, mode, planes):
        values = dict(zip(self.inputs, planes))
        n, device = self._samples(values)
        dtype = _DTYPE_OF[mode]
        return [
            torch.empty(
                self.plane_shape(name, n),
                dtype=torch.bool if name == self.terminated else dtype,
                device=device,
            )
            for name in self.outputs
        ]

    def launch(self, mode, planes):
        """The kernel itself, over concrete tensors: fresh zeroed outputs (a
        FINISHED mask is a private copy of the input mask, updated in place by
        the kernel and returned), written by one launch."""
        values = dict(zip(self.inputs, planes))
        n, device = self._samples(values)
        dtype = _DTYPE_OF[mode]
        outs = {
            name: torch.zeros(self.plane_shape(name, n), dtype=dtype, device=device)
            for name in self.outputs
            if name != self.terminated
        }
        if self.finishes_terminated:
            outs[self.terminated] = (
                values[self.terminated].detach().contiguous().clone()
            )
        for name in self.state:  # carried state: stepped in place on a copy
            outs[name] = values[name].detach().contiguous().clone()
        rows = self._cell_rows(n, mode, device)
        if rows is None:
            self._run(mode, {**values, **outs}, device)
        else:
            with rows:
                views = {nm: _as_tensor(a, device) for nm, a in rows.rows.items()}
                self._run(mode, {**values, **outs, **views}, device)
                for nm, cell in self._finish(rows, device).items():
                    outs[nm] = _as_tensor(cell, device).clone()
        return [outs[name] for name in self.outputs]

    @staticmethod
    def _finish(rows, device):
        """``rows.finish()`` on the stream the launch ran on (a capture sees it)."""
        if device.type != "cuda":
            return rows.finish()
        import cupy

        from eagle.interop import _stream_handle, external_stream

        index = device.index if device.index is not None else torch.cuda.current_device()
        with cupy.cuda.Device(index), external_stream(_stream_handle(torch.cuda.current_stream(index))):
            return rows.finish()

    def _cell_rows(self, n, mode, device):
        """The row scratch of this kernel at ``n`` samples (``None`` without a reduction)."""
        import numpy

        xp = numpy
        if device.type == "cuda":
            import cupy as xp
        return _cell_rows.scratch(self, n, xp, numpy.dtype(mode), cells=self.cells)

    def _mask(self, plane, values):
        if self.terminated is None:
            return plane
        return plane.masked_fill(values[self.terminated], 0.0)

    def jvp_pass(self, saved, tangents):
        child = self.child("jvp")
        values = dict(zip(self.inputs, saved))
        tangent = dict(zip(self.inputs, tangents))
        by_tangent = {plane: w for w, plane in child.tan_in.items()}
        args = []
        for name in child.inputs:
            if name in values:
                args.append(values[name])
            else:
                t = tangent[by_tangent[name]]
                args.append(
                    torch.zeros_like(values[by_tangent[name]]) if t is None else t
                )
        got = dict(zip(child.outputs, child.fn_for_call.apply(*args)))
        return tuple(
            (got[child.tan_out[o]] if o in self.cells else self._mask(got[child.tan_out[o]], values))
            if o != self.terminated and child.tan_out.get(o) in got
            else None
            for o in self.outputs
        )

    def vmap_pass(self, info, in_dims, args):
        """``torch.func.vmap``: a batch of calls is ONE call over
        ``n * batch`` samples (the batch folds into the sample axis); a
        batched ``Param`` cannot, so that case is a loop of calls."""
        size = info.batch_size
        # a batched Param, or a cell (a sum over the folded samples), cannot fold
        if self.cells or any(d is not None for n, d in zip(self.inputs, in_dims)
                             if n in self.params):
            rows = []
            for i in range(size):
                picked = [
                    a if d is None else a.select(d, i) for a, d in zip(args, in_dims)
                ]
                rows.append(self.fn_for_call.apply(*picked))
            outs = tuple(
                torch.stack([r[j] for r in rows]) for j in range(len(self.outputs))
            )
            return outs, (0,) * len(outs)
        folded = []
        for name, a, d in zip(self.inputs, args, in_dims):
            if name in self.params:
                folded.append(a)
                continue
            a = (
                a.movedim(d, -1)
                if d is not None
                else a.unsqueeze(-1).expand(*a.shape, size)
            )
            folded.append(a.reshape(*a.shape[:-2], -1))
        outs = tuple(
            o.reshape(*o.shape[:-1], -1, size) for o in self.fn_for_call.apply(*folded)
        )
        return outs, tuple(o.ndim - 1 for o in outs)

    # -- the launch itself ------------------------------------------------------
    def _run(self, mode, planes, device):
        """Run this kernel over ``planes`` (name -> tensor); the output planes
        in ``planes`` are written in place. A declared ``finished_count`` cell
        is synthesised HERE, fresh and zeroed, never read back (one launch has
        no running total for it to accumulate into)."""
        values = {}
        for role, name in self.spec:
            if role == "lookup" and name == FINISHED_PLANE:
                # one 4-byte counter cell (declared uint32; torch's int32
                # carries the same bytes on every device)
                values[name] = torch.zeros(1, dtype=torch.int32, device=device)
                continue
            if name not in planes:
                continue
            value = planes[name].detach()
            if role == "uniform":
                values[name] = float(value)
            elif role == "lookup":
                values[name] = value.reshape(1).contiguous()
            else:
                values[name] = value if role in OUTPUT_ROLES else value.contiguous()
        if device.type == "cuda":
            self._run_device(mode, values, device)
        else:
            self._run_host(mode, values)

    def _built_for(self, mode, index, make):
        key = (mode, index)
        got = self._built.get(key)
        if got is None:
            with LOCK:
                got = self._built.get(key)
                if got is None:
                    got = self._built[key] = make()
        return got

    def _run_host(self, mode, values):
        from hawk.runtime import run

        from eagle.interop import Requirements, import_buffer

        host = self._built_for(mode, None, lambda: self._load_host(mode))
        arrays = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                # the buffer check validates dtype, layout and device; the array
                # itself is torch's own NumPy view (same memory, always writable),
                # since NumPy < 2.2 imports every torch tensor read-only
                # over DLPack and the kernel writes its mutable planes
                import_buffer(
                    value,
                    require=_requirements(
                        Requirements, value, "cpu", name=name, mode=mode
                    ),
                )
                value = value.detach().numpy()
            arrays[name] = value
        run(host, **arrays)

    def _load_host(self, mode):
        from hawk.artifact import build_bundle
        from hawk.runtime import load

        unit = build_bundle(
            list(self.chain),
            self.owner.dir / "host" / mode / self.path_dir,
            mode=mode,
            targets=("host",),
        )
        return load(unit.directory, self.kernel.name)

    def _run_device(self, mode, values, device):
        import cupy

        from eagle.interop import (
            Requirements,
            _stream_handle,
            external_stream,
            import_buffer,
        )

        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        stream = _stream_handle(torch.cuda.current_stream(index))
        plan = self._built_for(mode, index, lambda: self._load_device(mode, index))
        with cupy.cuda.Device(index), external_stream(stream):
            arrays = {}
            for name, value in values.items():
                if isinstance(value, torch.Tensor):
                    view = import_buffer(
                        value,
                        stream=stream or 1,
                        require=_requirements(
                            Requirements, value, "cuda", index, name=name, mode=mode
                        ),
                    )
                    value = cupy.from_dlpack(view)
                arrays[name] = value
            plan.bind(**arrays).launch(stream=stream)

    def _load_device(self, mode, index):
        from hawk.artifact import build_bundle, plugins

        from eagle import exec as eexec
        from eagle import plan as eplan
        from eagle.registry import load_manifest

        arch = self.owner.device_arch
        if arch is None:
            major, minor = torch.cuda.get_device_capability(index)
            arch = f"sm_{major}{minor}"
        unit = build_bundle(
            list(self.chain),
            self.owner.dir / arch / mode / self.path_dir,
            mode=mode,
            targets=("cuda",),
            device_arch=arch,
        )
        built = plugins(unit, device_loader=load_manifest)
        _split.link(built, unit)
        return eplan.plan(built[self.kernel.name], structure=eexec.DeviceKernel)

    @property
    def path_dir(self):
        return self.path or "primal"


class KernelFunction:
    """A hawk kernel, its derived reverse- and forward-mode kernels (to any
    depth, derived when a higher derivative asks), and the
    ``torch.autograd.Function`` that applies them (see :func:`function`).

    :ivar inputs: the input names, in positional order.
    :ivar outputs: the output names, in return order -- the ``Terminated``
        mask's own name is last when :attr:`finishes_terminated` is true.
    :ivar wrt: the inputs that receive a gradient.
    :ivar finishes_terminated: whether the kernel's own body finishes the
        ``Terminated`` mask (``terminated = cond``), rather than only
        reading it.
    """

    def __init__(self, kernel, *, wrt=None, cache_dir=None, device_arch=None):
        from hawk.diff import step_vjp, vjp

        role_of = {name: role for role, name in kernel.arg_spec}
        finish = kernel.walk.finish
        #: A kernel that finishes a mask always carries the reserved
        #: `finished_count` lookup cell (the newly-finished counter) in its
        #: own arg_spec, whether or not it is wrapped here -- a launch
        #: synthesises it, fresh and zeroed (never read back: one launch has
        #: no running total for it to accumulate into), so it never needs a
        #: caller-facing role of its own.
        internal = {FINISHED_PLANE} if finish is not None else frozenset()
        unsupported = sorted(
            f"{name} ({role})"
            for name, role in role_of.items()
            if role not in _INPUT_ROLES | OUTPUT_ROLES and name not in internal
        )
        if unsupported:
            raise ValueError(
                f"eagle.frameworks.torch: kernel {kernel.name!r} declares "
                f"{', '.join(unsupported)}; the torch bridge serves per-sample "
                "kernels (Scalar/Vector/Matrix inputs, Param, Terminated, Mutable "
                "outputs)"
            )
        order = [name for name in kernel.planes if name in role_of]
        if not any(role_of[n] in OUTPUT_ROLES for n in order) and not (
            finish is not None and finish[0] in role_of
        ):
            raise ValueError(
                f"eagle.frameworks.torch: kernel {kernel.name!r} writes no output"
            )
        state = tuple(
            n for n in order if role_of[n] == "mutable" and n in _prior_reads(kernel)
        )
        self.name = kernel.name
        self.uid = next(_UIDS)
        _generic_op()
        base = promote(kernel)
        self.promoted = base if base is not kernel else None
        self._tmp = None
        if cache_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="eagle_torch_")
            cache_dir = self._tmp.name
        self.dir = Path(cache_dir)
        self.device_arch = device_arch
        root_inputs = [n for n in order if role_of[n] in _INPUT_ROLES or n in state]
        self.root_inputs = frozenset(root_inputs)
        self.root_wrt = frozenset(root_inputs)
        self._root = _Level(
            self, base, chain=(base,), path="", order=order, state=state
        )
        reverse = (step_vjp if state else vjp)(base, wrt=wrt)
        self._root.wrt = tuple(reverse.wrt)
        self.root_wrt = frozenset(reverse.wrt)
        self.inputs = self._root.inputs
        self.outputs = self._root.outputs
        self.wrt = self._root.wrt
        self.finishes_terminated = self._root.finishes_terminated
        self._terminated = self._root.terminated
        self._role = {n: role_of[n] for n in self.inputs}
        self._shape = {
            n: tuple(kernel.walk.slot_types[(role_of[n], n)].shape) for n in self.inputs
        }
        # derive both first-order levels now: a kernel the transforms refuse fails here
        self._root.child("vjp")
        if not state:
            self._root.child("jvp")
        self._root.fn  # both Functions exist before any (possibly traced) call

    def __repr__(self) -> str:
        return (
            f"KernelFunction({self.name!r}, inputs={list(self.inputs)}, "
            f"outputs={list(self.outputs)}, wrt={list(self.wrt)})"
        )

    def __call__(self, *args, layout=None, **kwargs):
        """Run the kernel on ``args``/``kwargs`` and return its outputs as
        tensors, differentiable through torch, in the caller's layout (a
        mix of sample-major and component-major inputs is refused, see
        :meth:`_call_layout`). ``layout`` (``"samples_first"`` /
        ``"samples_last"``) and ``eagle.samples_first(x)`` /
        ``eagle.samples_last(x)`` say which axis holds the samples of an input
        whose shape reads both ways (``(w, w)``), which is otherwise refused."""
        if len(args) > len(self.inputs):
            raise TypeError(
                f"{self.name}: takes {len(self.inputs)} inputs "
                f"{list(self.inputs)}, got {len(args)}"
            )
        bound = dict(zip(self.inputs, args))
        for name, value in kwargs.items():
            if name not in self.inputs:
                raise TypeError(
                    f"{self.name}: {name!r} is not an input (inputs: "
                    f"{list(self.inputs)})"
                )
            if name in bound:
                raise TypeError(f"{self.name}: input {name!r} given twice")
            bound[name] = value
        missing = [n for n in self.inputs if n not in bound]
        if missing:
            raise TypeError(f"{self.name}: missing input(s) {missing}")
        from eagle import _layout

        bound, axes = _layout.split_marks(bound, layout)
        if layout is None and not axes:
            # the common call: no layout said anywhere, so nothing consults the
            # layout scope (a ContextVar, which torch.compile cannot trace)
            return self._call_bound(bound, plain=True)
        with _layout.scope(layout, axes):
            return self._call_bound(bound, plain=False)

    def _call_bound(self, bound, plain=False):
        layouts = {}
        for name in self.inputs:
            bound[name], layout = self._component_major(name, bound[name], plain)
            if layout is not None:
                layouts[name] = layout
        sample_major = self._call_layout(layouts)
        out = self._root.fn_for_call.apply(*self._tensors(bound))
        if sample_major:
            out = tuple(_sample_major_view(o) for o in out)
        return out[0] if len(out) == 1 else out

    def _tensors(self, bound):
        """The call's inputs as the tensors the Function takes: a Param is a
        one-element tensor (of the call's dtype, on the samples' device when
        it is read from a device plane), the build mode is the call's dtype."""
        root = self._root
        planes = [bound[n] for n in self.inputs if n not in root.params]
        params = [bound[n] for n in self.inputs if n in root.params]
        mode = _mode_of([*planes, *params])
        dtype = _DTYPE_OF[mode]
        device = next((p.device for p in planes if isinstance(p, torch.Tensor)), None)
        promoted = self.promoted is not None
        out = []
        for name in self.inputs:
            value = bound[name]
            if name in root.params:
                if not isinstance(value, torch.Tensor):
                    # a fill kernel on the device, not a host-to-device copy (a sync)
                    value = torch.full(
                        (1,),
                        float(value),
                        dtype=dtype,
                        device=device if promoted else None,
                    )
                else:
                    if value.numel() != 1:
                        raise ValueError(
                            f"{self.name}: Param {name!r} is one number, got shape "
                            f"{tuple(value.shape)}"
                        )
                    value = value.reshape(1)
                    if promoted and device is not None and value.device != device:
                        value = value.to(device)
            out.append(value)
        return out

    # -- planes -------------------------------------------------------------
    def _component_major(self, name, value, plain=False):
        """A sample-major ``(n, width)`` input in ``(width, n)`` form: a
        zero-copy ``.T`` when contiguous, else a copy with a warning (both
        torch ops, so autograd routes gradients back correctly). Returns
        ``(native_value, layout)`` -- ``layout`` is ``"sample"``,
        ``"component"``, or ``None`` for a uniform/scalar/non-tensor."""
        if self._role[name] == "uniform" or not isinstance(value, torch.Tensor):
            return value, None
        width = math.prod(self._shape[name])
        if width <= 1:
            return value, None
        shape = value.shape
        if plain and value.ndim == 2 and shape[0] != shape[1]:
            # the unambiguous shapes, read straight off the tensor (traceable)
            if shape[0] == width:
                return value, "component"
            if shape[1] == width and _is_compiling():
                return value.T, "sample"
        from eagle import _layout

        adapted = _layout.adapt(name, value, (width,), writes=False)
        if adapted is None:
            return value, "component"
        return adapted.native, "sample"

    def _call_layout(self, layouts) -> bool:
        """Whether outputs should come back sample-major: ``True`` only
        when every layout-bearing input was sample-major, ``False`` for
        all-component-major (or none at all); a mix is refused."""
        kinds = set(layouts.values())
        if kinds <= {"component"}:
            return False
        if kinds == {"sample"}:
            return True
        by_name = ", ".join(
            f"{name!r} ({kind}-major)" for name, kind in sorted(layouts.items())
        )
        raise ValueError(
            f"{self.name}: per-sample inputs disagree on layout ({by_name}); "
            "pass every per-sample input sample-major (n, width), or every "
            "one component-major (width, n) — not a mix of the two"
        )


def _sample_major_view(value):
    """A fresh, module-allocated output in its sample-major ``(n, width)``
    form. Always a zero-copy ``.T`` (never a copy or warning, unlike an
    input plane's adapt) -- a plain torch op, so both AD directions
    propagate through it like any other view."""
    return value.T if value.ndim == 2 else value


def _requirements(cls, tensor, device, index=None, name=None, mode="float64"):
    """What a bound plane must be: for the synthesised ``finished_count``
    lookup cell, a 4-byte integer (see ``_Level._run``); the tensor's own dtype
    when it is a ``Terminated`` mask (``bool``), the call's build mode
    otherwise; contiguous; on ``device``."""
    if name == FINISHED_PLANE:
        dtype = "int32"
    else:
        dtype = "bool" if tensor.dtype == torch.bool else mode
    return cls(dtype=dtype, contiguous=True, device=device, device_id=index)


def _autograd_function(level, forward_ad):
    """The ``torch.autograd.Function`` behind ``level`` (one class per level;
    the eager one also defines ``jvp``, its ``torch.compile`` twin does not,
    since dynamo cannot trace a Function that does). Every pass is one launch
    of the generic op; the backward is the Function of the level's own vjp,
    which is why derivatives nest. The compile twin's ``forward`` and
    ``backward`` close over plain data only (digests, index tuples), which is
    what dynamo can trace."""
    # `forward` takes the inputs by position, one named parameter each: a
    # varargs `forward` is mistaken for the old ctx-first signature by dynamo
    names = ", ".join(f"x{i}" for i in range(len(level.inputs)))
    scope = {"_apply_op": _apply_op, "digests": level.digests}
    exec(
        f"def forward({names}):\n"
        f"    return _apply_op(digests, ({names}{',' if names else ''}))",
        scope,
    )
    links = level.links  # the traced backward closes over this, never the level

    def backward_traced(ctx, *grad_outs):
        return _backward(
            ctx.saved_tensors, grad_outs, ctx.needs_input_grad, links[True], True
        )

    def backward_eager(ctx, *grad_outs):
        return _backward(
            ctx.saved_tensors, grad_outs, ctx.needs_input_grad, level.link(), False
        )

    class _HawkKernel(torch.autograd.Function):
        forward = staticmethod(scope["forward"])
        backward = staticmethod(backward_eager if forward_ad else backward_traced)

        @staticmethod
        def setup_context(ctx, inputs, output):
            ctx.save_for_backward(*inputs)
            ctx.save_for_forward(*inputs)

        @staticmethod
        def vmap(info, in_dims, *args):
            return level.vmap_pass(info, in_dims, args)

    if forward_ad:
        _HawkKernel.jvp = staticmethod(
            lambda ctx, *tangents: level.jvp_pass(ctx.saved_tensors, tangents)
        )
    _HawkKernel.__name__ = _HawkKernel.__qualname__ = (
        f"HawkKernel_{level.owner.name}_{level.path_dir}"
    )
    return _HawkKernel
