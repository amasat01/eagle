# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The gradient certification rows (eagle-owned, declared in
``raptor.conformance.interop.ROWS``): ``JX-GRAD-TRANSFORMS``, ``T-GRAD-AUTOGRAD``,
``T-GRAD-FUNC`` and ``WP-GRAD-TAPE``.

Each row runs a hawk kernel (the drag law, with a finished-sample mask) through a
framework's own gradient machinery and compares it with the same law written in that
framework alone (Warp: central finite differences of the NumPy law). A row with two
legs has one test per leg. Every row carries a falsifier: the reference gradient,
scaled by one part in a million, must be told apart from the kernel's.

* ``JX-GRAD-TRANSFORMS``: ``grad``, ``jit(grad)``, ``vmap(grad)``, ``jacfwd`` and
  ``jacrev`` of ``eagle.frameworks.jax.function``.
* ``T-GRAD-AUTOGRAD``: ``backward`` and ``torch.autograd.gradcheck`` (forward AD too).
* ``T-GRAD-FUNC``: ``torch.func.grad``, ``jvp`` and ``vmap`` of ``grad``.
* ``WP-GRAD-TAPE``: ``wp.Tape`` through ``eagle.frameworks.warp.function``.

Hygiene is the same as the other matrix modules: no skip tokens, no availability
marker, only the ``interop_matrix`` tag, and every row opens with
``certified_framework``. The kernels and references come from the framework test
modules, imported inside the tests after that guard. A leg with no device of its
platform fails; it never skips.
"""

from __future__ import annotations

import os

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pytest
from raptor.conformance.interop import certified_framework, register_row

pytestmark = pytest.mark.interop_matrix

#: one part in a million: the falsifier's perturbation, far above the tolerances below
_BUMP = 1.0 + 1e-6


def _same(got, want, rtol=1e-10):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    return got.shape == want.shape and bool(
        np.allclose(got, want, rtol=rtol, atol=rtol * max(np.abs(want).max(), 1.0)))


# --------------------------------------------------------------------------- #
# JX-GRAD-TRANSFORMS
# --------------------------------------------------------------------------- #
def _jax_grad_transforms(tmp_path_factory, platform):
    certified_framework("jax")
    import jax
    import jax.numpy as jnp
    from test_interop_jax_ffi import _device, _ops

    K = _ops(tmp_path_factory)["drag"]
    n = 6
    rng = np.random.default_rng(0)
    with jax.default_device(_device(jax, platform)):
        x = jnp.asarray(rng.normal(size=(3, n)))
        cd = jnp.asarray([0.3])
        done = jnp.asarray(np.arange(n) % 3 == 1)
        batch = jnp.stack([x * 0.5, x, x * 1.5])

        f = lambda y, c: K(y, c, done)                                           # noqa: E731
        ref = lambda y, c: jnp.where(                                            # noqa: E731
            done[None, :], 0.0, -c[0] * jnp.sqrt(jnp.sum(y * y, axis=0)) * y)
        loss = lambda g: (lambda y, c: jnp.sum(g(y, c) ** 2))                    # noqa: E731

        pairs = {
            "grad": (jax.grad(loss(f), (0, 1))(x, cd), jax.grad(loss(ref), (0, 1))(x, cd)),
            "jit(grad)": (jax.jit(jax.grad(loss(f), (0, 1)))(x, cd),
                          jax.grad(loss(ref), (0, 1))(x, cd)),
            "vmap(grad)": (jax.vmap(jax.grad(loss(f), 0), in_axes=(0, None))(batch, cd),
                           jax.vmap(jax.grad(loss(ref), 0), in_axes=(0, None))(batch, cd)),
            "jacfwd": (jax.jacfwd(lambda y: f(y, cd))(x), jax.jacfwd(lambda y: ref(y, cd))(x)),
            "jacrev": (jax.jacrev(lambda y: f(y, cd))(x), jax.jacrev(lambda y: ref(y, cd))(x)),
        }
        for name, (got, want) in pairs.items():
            for g, w in zip(jax.tree.leaves(got), jax.tree.leaves(want)):
                assert _same(g, w), f"{name} disagrees with the pure-JAX law"
                assert not _same(g, np.asarray(w) * _BUMP), f"{name}: the comparison is blind"


@register_row("JX-GRAD-TRANSFORMS", leg="cpu")
def test_jax_gradient_transforms_on_the_cpu(tmp_path_factory):
    _jax_grad_transforms(tmp_path_factory, "cpu")


@register_row("JX-GRAD-TRANSFORMS", leg="cuda")
def test_jax_gradient_transforms_on_cuda(tmp_path_factory):
    _jax_grad_transforms(tmp_path_factory, "cuda")


# --------------------------------------------------------------------------- #
# T-GRAD-AUTOGRAD and T-GRAD-FUNC
# --------------------------------------------------------------------------- #
def _torch_case(tmp_path_factory, device):
    certified_framework("torch")
    import torch
    from test_frameworks_torch import _inputs, _reference, drag

    from eagle.frameworks import torch as eagle_torch

    op = eagle_torch.function(drag, cache_dir=tmp_path_factory.mktemp("grad_torch"))
    v, cd, done = _inputs(torch.device(device))
    return torch, op, _reference, v, cd, done


def _torch_autograd(tmp_path_factory, device):
    torch, op, ref, v, cd, done = _torch_case(tmp_path_factory, device)
    loss = lambda f, v, cd: (f(v, cd, done) ** 2).sum()                          # noqa: E731

    def grads(f):
        a, b = v.clone().requires_grad_(), cd.clone().requires_grad_()
        loss(f, a, b).backward()
        return a.grad, b.grad

    got, want = grads(op), grads(ref)
    torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-13)
    assert not got[0][:, done].any()                       # nothing flows through a finished sample
    with pytest.raises(AssertionError):                    # the instrument sees a 1e-6 error
        torch.testing.assert_close(got[0], want[0] * _BUMP, rtol=1e-11, atol=1e-13)
    a, b = v.clone().requires_grad_(), cd.clone().requires_grad_()
    assert torch.autograd.gradcheck(lambda a, b: op(a, b, done), (a, b),
                                    check_forward_ad=True, check_batched_grad=False)


@register_row("T-GRAD-AUTOGRAD", leg="cpu")
def test_torch_autograd_on_the_cpu(tmp_path_factory):
    _torch_autograd(tmp_path_factory, "cpu")


@register_row("T-GRAD-AUTOGRAD", leg="cuda")
def test_torch_autograd_on_cuda(tmp_path_factory):
    _torch_autograd(tmp_path_factory, "cuda")


def _torch_func(tmp_path_factory, device):
    torch, op, ref, v, cd, done = _torch_case(tmp_path_factory, device)
    loss = lambda f: (lambda v, cd: (f(v, cd, done) ** 2).sum())                 # noqa: E731
    got = torch.func.grad(loss(op), argnums=(0, 1))(v, cd)
    want = torch.func.grad(loss(ref), argnums=(0, 1))(v, cd)
    torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-13)

    one = torch.ones_like(v)
    torch.testing.assert_close(
        torch.func.jvp(lambda v: op(v, cd, done), (v,), (one,)),
        torch.func.jvp(lambda v: ref(v, cd, done), (v,), (one,)), rtol=1e-11, atol=1e-13)

    batch = torch.stack([v, 2 * v])
    got = torch.func.vmap(torch.func.grad(lambda v: loss(op)(v, cd)))(batch)
    want = torch.func.vmap(torch.func.grad(lambda v: loss(ref)(v, cd)))(batch)
    torch.testing.assert_close(got, want, rtol=1e-11, atol=1e-13)
    with pytest.raises(AssertionError):
        torch.testing.assert_close(got, want * _BUMP, rtol=1e-11, atol=1e-13)


@register_row("T-GRAD-FUNC", leg="cpu")
def test_torch_func_transforms_on_the_cpu(tmp_path_factory):
    _torch_func(tmp_path_factory, "cpu")


@register_row("T-GRAD-FUNC", leg="cuda")
def test_torch_func_transforms_on_cuda(tmp_path_factory):
    _torch_func(tmp_path_factory, "cuda")


# --------------------------------------------------------------------------- #
# WP-GRAD-TAPE
# --------------------------------------------------------------------------- #
def _warp_tape(tmp_path_factory, device):
    certified_framework("warp")
    from test_frameworks_warp import _case, _fd_grads, _tape_grads, drag

    from eagle.frameworks import warp as eagle_warp

    op = eagle_warp.function(drag, cache_dir=tmp_path_factory.mktemp("grad_warp"))
    for mode, rtol in (("float64", 1e-7), ("float32", 5e-4)):
        c = _case(device, mode)
        _tape_grads(op, c, device, mode)
        gv, gc = _fd_grads(c)                         # central differences of the NumPy law
        got_v, got_c = c["v"].grad.numpy(), c["cd"].grad.numpy()[0]
        tol = rtol * np.abs(gv).max()
        np.testing.assert_allclose(got_v, gv, rtol=rtol, atol=tol)
        np.testing.assert_allclose(got_c, gc, rtol=rtol)
        assert not got_v[:, c["term_np"]].any()       # nothing flows through a finished sample
        assert not np.allclose(got_v * (1 + 100 * rtol), gv, rtol=rtol, atol=tol)  # not blind


@register_row("WP-GRAD-TAPE", leg="cpu")
def test_warp_tape_on_the_cpu(tmp_path_factory):
    _warp_tape(tmp_path_factory, "cpu")


@register_row("WP-GRAD-TAPE", leg="cuda")
def test_warp_tape_on_cuda(tmp_path_factory):
    _warp_tape(tmp_path_factory, "cuda:0")
