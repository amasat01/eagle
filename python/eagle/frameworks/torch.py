# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A hawk per-sample kernel as a ``torch.autograd.Function``: :func:`function`.

Forward runs the kernel; backward runs the reverse-mode kernel hawk
derives from it (:func:`hawk.diff.vjp`); forward-mode AD runs the derived
tangent kernel (:func:`hawk.diff.jvp`) -- nothing is taped, each pass is
one kernel launch. ``torch.func`` transforms are not supported: they hand
derivative rules wrapped tensors exposing no storage, and a kernel can
only read a buffer.

CPU tensors run through hawk's host runtime; CUDA tensors through eagle's
launch path (:func:`eagle.plan.plan`) on torch's current stream, no sync.
Every plane crosses through :func:`eagle.interop.import_buffer`
zero-copy; a non-contiguous input is made contiguous, a wrong-dtype one
refused.

Conventions, off the kernel's own declaration: inputs are its planes and
``Param`` uniforms in order; outputs are its ``Mutable`` planes (bare if
one), PLUS the ``Terminated`` mask itself when the kernel's own body
finishes it (``terminated = cond``) -- the updated mask comes back as an
extra return value, the last one, so a stepping loop reuses the SAME stop
decision the kernel made instead of recomputing it with a second masking
rule of its own. A per-sample plane is ``(n,)``/``(width, n)``; a
sample-major ``(n, width)`` input is accepted too (zero-copy transpose where
contiguous), and outputs come back in that same layout -- mixing layouts
in one call is refused. A ``Param``'s gradient is the batch sum of its
per-sample partials (held in CUDA, read back to host once per launch). A
``Terminated`` mask is never differentiated: a marked sample's outputs and
gradient/tangent stay zero, and the mask itself carries no gradient/tangent
of its own.

torch is imported by this submodule only: ``import eagle`` stays torch-free.
"""

from __future__ import annotations

import math
import tempfile
import weakref
from pathlib import Path

import torch

from .. import FINISHED_PLANE

__all__ = ["KernelFunction", "function"]

#: Roles the bridge binds as an input: per-sample planes, the termination
#: mask and by-value uniforms.
_INPUT_ROLES = frozenset({"per_sample", "vec_in", "mat_in", "terminated", "uniform"})
#: Roles the bridge allocates and returns as an output.
_OUTPUT_ROLES = frozenset({"mutable", "out"})


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
    return KernelFunction(kernel, wrt=wrt, cache_dir=cache_dir,
                          device_arch=device_arch)


class KernelFunction:
    """A hawk kernel, its derived reverse- and forward-mode kernels, and the
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
        from hawk.diff import jvp, vjp
        from hawk.trace import Kernel

        role_of = {name: role for role, name in kernel.arg_spec}
        finish = kernel.walk.finish
        #: A kernel that finishes a mask always carries the reserved
        #: `finished_count` lookup cell (the newly-finished counter) in its
        #: own arg_spec, whether or not it is wrapped here -- `_run`
        #: synthesises it, fresh and zeroed, on every launch (never read
        #: back: one launch has no running total for it to accumulate
        #: into), so it never needs a caller-facing role of its own.
        internal = {FINISHED_PLANE} if finish is not None else frozenset()
        unsupported = sorted(
            f"{name} ({role})" for name, role in role_of.items()
            if role not in _INPUT_ROLES | _OUTPUT_ROLES and name not in internal
        )
        if unsupported:
            raise ValueError(
                f"eagle.frameworks.torch: kernel {kernel.name!r} declares "
                f"{', '.join(unsupported)}; the torch bridge serves per-sample "
                "kernels (Scalar/Vector/Matrix inputs, Param, Terminated, Mutable "
                "outputs)"
            )
        order = [name for name in kernel.planes if name in role_of]
        self.name = kernel.name
        self.inputs = tuple(n for n in order if role_of[n] in _INPUT_ROLES)
        self._terminated = next(
            (n for n in self.inputs if role_of[n] == "terminated"), None)
        #: The kernel's own finish, surfaced: a mask it only READS stays
        #: input-only (unchanged), but one it FINISHES is also an output,
        #: so the caller's next step reuses the SAME decision instead of
        #: recomputing it with a second masking rule of its own.
        self.finishes_terminated = (
            self._terminated is not None and finish is not None
            and finish[0] == self._terminated)
        outputs = [n for n in order if role_of[n] in _OUTPUT_ROLES]
        if self.finishes_terminated:
            outputs.append(self._terminated)
        self.outputs = tuple(outputs)
        if not self.outputs:
            raise ValueError(
                f"eagle.frameworks.torch: kernel {kernel.name!r} writes no output")
        self._role = role_of
        self._shape = {n: tuple(kernel.planes[n].ttype.shape) for n in order}

        reverse = vjp(kernel, wrt=wrt)
        self.wrt = tuple(reverse.wrt)
        forward = jvp(kernel, wrt=self.wrt)
        self._kernels = {
            "primal": kernel,
            "vjp": Kernel(f"{kernel.name}_vjp", reverse),
            "jvp": Kernel(f"{kernel.name}_jvp", forward),
        }
        self._spec = {key: tuple(k.arg_spec) for key, k in self._kernels.items()}

        self._tmp = None
        if cache_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="eagle_torch_")
            cache_dir = self._tmp.name
        self._dir = Path(cache_dir)
        self._device_arch = device_arch
        self._host = None          # {key: hawk.runtime.HostKernel}
        self._device = {}          # device index -> {key: eagle.plan.Plan}
        self._fn = _autograd_function(self)

    def __repr__(self) -> str:
        return (f"KernelFunction({self.name!r}, inputs={list(self.inputs)}, "
                f"outputs={list(self.outputs)}, wrt={list(self.wrt)})")

    def __call__(self, *args, **kwargs):
        """Run the kernel on ``args``/``kwargs`` and return its outputs as
        tensors, differentiable through torch, in the caller's layout (a
        mix of sample-major and component-major inputs is refused, see
        :meth:`_call_layout`)."""
        if len(args) > len(self.inputs):
            raise TypeError(
                f"{self.name}: takes {len(self.inputs)} inputs "
                f"{list(self.inputs)}, got {len(args)}")
        bound = dict(zip(self.inputs, args))
        for name, value in kwargs.items():
            if name not in self.inputs:
                raise TypeError(
                    f"{self.name}: {name!r} is not an input (inputs: "
                    f"{list(self.inputs)})")
            if name in bound:
                raise TypeError(f"{self.name}: input {name!r} given twice")
            bound[name] = value
        missing = [n for n in self.inputs if n not in bound]
        if missing:
            raise TypeError(f"{self.name}: missing input(s) {missing}")
        layouts = {}
        for name in self.inputs:
            bound[name], layout = self._component_major(name, bound[name])
            if layout is not None:
                layouts[name] = layout
        sample_major = self._call_layout(layouts)
        out = self._fn.apply(*(bound[n] for n in self.inputs))
        if sample_major:
            out = tuple(_sample_major_view(o) for o in out)
        return out[0] if len(out) == 1 else out

    # -- planes -------------------------------------------------------------
    def _component_major(self, name, value):
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
            f"{name!r} ({kind}-major)" for name, kind in sorted(layouts.items()))
        raise ValueError(
            f"{self.name}: per-sample inputs disagree on layout ({by_name}); "
            "pass every per-sample input sample-major (n, width), or every "
            "one component-major (width, n) — not a mix of the two"
        )

    def _plane_shape(self, name, n):
        width = math.prod(self._shape[name])
        return (n,) if width == 1 else (width, n)

    def _samples(self, values):
        """The sample count and device, read off the per-sample input planes."""
        planes = [(n, v) for n, v in values.items()
                  if self._role[n] != "uniform"]
        if not planes:
            raise ValueError(f"{self.name}: no per-sample input plane bound")
        n, device = None, None
        for name, value in planes:
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{self.name}: input {name!r} is a per-sample plane and must be "
                    f"a torch.Tensor, got {type(value).__name__}")
            if n is None:
                n, device = value.shape[-1], value.device
            if value.device != device:
                raise ValueError(
                    f"{self.name}: input {name!r} is on {value.device}, "
                    f"the other planes on {device}")
            if tuple(value.shape) != self._plane_shape(name, n):
                raise ValueError(
                    f"{self.name}: input {name!r} must have shape "
                    f"{self._plane_shape(name, n)} for n={n}, got "
                    f"{tuple(value.shape)}")
        return n, device

    def _zeros(self, name, n, device):
        return torch.zeros(self._plane_shape(name, n), dtype=torch.float64,
                           device=device)

    # -- launches -----------------------------------------------------------
    def _run(self, key, planes, device):
        """Run kernel ``key`` over ``planes`` (name -> tensor or number); the
        output planes in ``planes`` are written in place. A declared
        ``finished_count`` lookup cell (any kernel that finishes a mask
        carries one, see ``__init__``) is synthesised HERE, fresh and
        zeroed, never read back -- this bridge is one launch per call,
        with no running total for it to accumulate into."""
        values = {}
        for role, name in self._spec[key]:
            if role == "lookup" and name == FINISHED_PLANE:
                # the host runtime's own bind check wants int64 for an
                # i32/i64-wired lookup plane; the device plan path wants
                # the device-native int32 it atomically adds into.
                dtype = torch.int32 if device.type == "cuda" else torch.int64
                values[name] = torch.zeros(1, dtype=dtype, device=device)
                continue
            value = planes[name]
            if role == "uniform":
                values[name] = float(value)
            else:
                # a plane crosses as a plain buffer: autograd sees this
                # Function, never the launch inside it
                value = value.detach()
                values[name] = value if role in _OUTPUT_ROLES else value.contiguous()
        if device.type == "cuda":
            self._run_device(key, values, device)
        else:
            self._run_host(key, values)

    def _run_host(self, key, values):
        import numpy as np
        from hawk.artifact import build_bundle
        from hawk.runtime import load, run

        from eagle.interop import Requirements, import_buffer

        if self._host is None:
            unit = build_bundle(list(self._kernels.values()), self._dir / "host",
                                targets=("host",))
            self._host = {key: load(unit.directory, k.name)
                          for key, k in self._kernels.items()}
        arrays = {}
        for name, value in values.items():
            if isinstance(value, torch.Tensor):
                view = import_buffer(value, require=_requirements(
                    Requirements, value, "cpu", name=name))
                value = np.from_dlpack(view)
            arrays[name] = value
        run(self._host[key], **arrays)

    def _run_device(self, key, values, device):
        import cupy

        from eagle.interop import Requirements, external_stream, import_buffer

        index = device.index if device.index is not None \
            else torch.cuda.current_device()
        stream = _current_stream(index)
        with cupy.cuda.Device(index), external_stream(stream):
            plans = self._device.get(index) or self._load_device(index)
            arrays = {}
            for name, value in values.items():
                if isinstance(value, torch.Tensor):
                    view = import_buffer(value, stream=stream or 1,
                                         require=_requirements(
                                             Requirements, value, "cuda", index,
                                             name=name))
                    value = cupy.from_dlpack(view)
                arrays[name] = value
            plans[key].bind(**arrays).launch(stream=stream)

    def _load_device(self, index):
        from hawk.artifact import build_bundle, plugins

        from eagle import exec as eexec
        from eagle import plan as eplan
        from eagle.registry import load_manifest

        arch = self._device_arch
        if arch is None:
            major, minor = torch.cuda.get_device_capability(index)
            arch = f"sm_{major}{minor}"
        unit = build_bundle(list(self._kernels.values()), self._dir / arch,
                            targets=("cuda",), device_arch=arch)
        built = plugins(unit, device_loader=load_manifest)
        plans = {key: eplan.plan(built[k.name], structure=eexec.DeviceKernel)
                 for key, k in self._kernels.items()}
        self._device[index] = plans
        return plans

    # -- the three passes ---------------------------------------------------
    def _output_seeds(self, values, n, device):
        """Fresh zero buffers for every output, except a FINISHED
        ``Terminated`` mask: the finish is ``mask |= cond``, in place on
        the SAME tensor the caller passed in as input -- there is no
        separate output buffer to seed for it."""
        outs = {name: self._zeros(name, n, device) for name in self.outputs
                if name != self._terminated}
        if self.finishes_terminated:
            mask = values[self._terminated]
            # made contiguous HERE (not left to `_run`'s own pass), so the
            # storage this returns a VIEW of is the exact one the launch
            # wrote into, even when the caller's own mask was not already
            # contiguous. A VIEW, never the input tensor itself: torch
            # refuses handing an autograd input straight back as an output
            # of a `Function` whose `setup_context` also saves it.
            mask = mask if mask.is_contiguous() else mask.contiguous()
            outs[self._terminated] = mask.view_as(mask)
        return outs

    def _forward(self, values):
        n, device = self._samples(values)
        outs = self._output_seeds(values, n, device)
        self._run("primal", {**values, **outs}, device)
        return tuple(outs[name] for name in self.outputs)

    def _mask(self, plane, values):
        """Zero the samples the ``Terminated`` mask marks (a no-op without one)."""
        if self._terminated is None:
            return plane
        return plane.masked_fill(values[self._terminated], 0.0)

    def _backward(self, values, grad_outs, needs):
        from hawk.diff import ADJOINT_PREFIX

        n, device = self._samples(values)
        planes = dict(values)
        for name, grad in zip(self.outputs, grad_outs):
            planes[f"{ADJOINT_PREFIX}_{name}"] = grad
        bars = {}
        for name in self.wrt:
            shape_of = "uniform" if self._role[name] == "uniform" else name
            bars[name] = (torch.zeros(n, dtype=torch.float64, device=device)
                          if shape_of == "uniform" else self._zeros(name, n, device))
            planes[f"{ADJOINT_PREFIX}_{name}"] = bars[name]
        self._run("vjp", planes, device)
        grads = []
        for name, need in zip(self.inputs, needs):
            if not need or name not in bars:
                grads.append(None)
                continue
            grad = self._mask(bars[name], values)
            if self._role[name] == "uniform":
                param = values[name]
                grad = grad.sum().to(param.device)
            grads.append(grad)
        return tuple(grads)

    def _jvp(self, values, tangents):
        from hawk.diff import TANGENT_PREFIX

        n, device = self._samples(values)
        planes = dict(values)
        for name, tangent in zip(self.inputs, tangents):
            if name not in self.wrt:
                continue
            key = f"{TANGENT_PREFIX}_{name}"
            if self._role[name] == "uniform":
                planes[key] = 0.0 if tangent is None else float(tangent)
            else:
                planes[key] = (self._zeros(name, n, device) if tangent is None
                               else tangent)
        outs = self._output_seeds(values, n, device)
        for name in self.outputs:
            if name != self._terminated:
                planes[f"{TANGENT_PREFIX}_{name}"] = outs[name]
        self._run("jvp", planes, device)
        return tuple(outs[name] if name == self._terminated
                    else self._mask(outs[name], values) for name in self.outputs)


def _sample_major_view(value):
    """A fresh, module-allocated output in its sample-major ``(n, width)``
    form. Always a zero-copy ``.T`` (never a copy or warning, unlike an
    input plane's adapt) -- a plain torch op, so both AD directions
    propagate through it like any other view."""
    return value.T if value.ndim == 2 else value


def _current_stream(index):
    """The raw handle of torch's current stream on device ``index``: the
    stream every launch of a CUDA call rides (``0`` is the legacy default
    stream)."""
    return torch.cuda.current_stream(index).cuda_stream


def _requirements(cls, tensor, device, index=None, name=None):
    """What a bound plane must be: for the synthesised ``finished_count``
    lookup cell, int64 on the host runtime's own bind check or int32 on
    the device plan path (see ``_run``); the tensor's own dtype when it
    is a ``Terminated`` mask (``bool``), float64 otherwise; contiguous;
    on ``device``."""
    if name == FINISHED_PLANE:
        dtype = "int32" if device == "cuda" else "int64"
    else:
        dtype = "bool" if tensor.dtype == torch.bool else "float64"
    return cls(dtype=dtype, contiguous=True, device=device, device_id=index)


def _autograd_function(op):
    """The ``torch.autograd.Function`` behind ``op`` (one class per op),
    held through a weak proxy since ``op`` owns the class and a plain
    reference back would cycle."""
    op = weakref.proxy(op)

    class _HawkKernel(torch.autograd.Function):
        @staticmethod
        def forward(*args):
            return op._forward(dict(zip(op.inputs, args)))

        @staticmethod
        def setup_context(ctx, inputs, output):
            tensors = [v for v in inputs if isinstance(v, torch.Tensor)]
            ctx.layout = [isinstance(v, torch.Tensor) for v in inputs]
            ctx.scalars = [v for v in inputs if not isinstance(v, torch.Tensor)]
            ctx.save_for_backward(*tensors)
            ctx.save_for_forward(*tensors)

        @staticmethod
        def backward(ctx, *grad_outs):
            return op._backward(_saved(ctx), grad_outs, ctx.needs_input_grad)

        @staticmethod
        def jvp(ctx, *tangents):
            return op._jvp(_saved(ctx), tangents)

    def _saved(ctx):
        tensors, scalars = iter(ctx.saved_tensors), iter(ctx.scalars)
        return {name: next(tensors) if is_tensor else next(scalars)
                for name, is_tensor in zip(op.inputs, ctx.layout)}

    _HawkKernel.__name__ = _HawkKernel.__qualname__ = f"HawkKernel_{op.name}"
    return _HawkKernel
