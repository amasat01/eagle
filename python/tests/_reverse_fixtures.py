# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Shared graphs for the eagle.diff rows: a chain and a fork/join over small
hawk kernels, buildable on the host or the device, and a central-difference
reference for their gradients."""

from __future__ import annotations

import numpy as np
import pytest

import eagle
import eagle.plan

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar  # noqa: E402
from hawk import math as m  # noqa: E402


@hawk.kernel
def rv_square(x: Scalar, a: Param, y: Mutable[Scalar]):
    y = a * x * x


@hawk.kernel
def rv_sin(x: Scalar, y: Mutable[Scalar]):
    y = m.sin(x)


@hawk.kernel
def rv_join(u: Scalar, v: Scalar, w: Mutable[Scalar]):
    w = u * v + u


def xp_for(backend):
    if backend == "gpu":
        return pytest.importorskip("cupy")
    return np


def xp_of(x):
    return xp_for("gpu" if type(x).__module__.split(".")[0] == "cupy" else "cpu")


def plans(tmp_path, backend):
    """``(square, sin, join)`` auto plans for ``backend``."""
    targets = ("cuda",) if backend == "gpu" else ("host",)
    return eagle.plan.auto([rv_square, rv_sin, rv_join], targets=targets,
                           cache_dir=str(tmp_path / "hawk_cache"))


def chain(ps, x, a, *, record=False):
    """``z = sin(a x^2)``: ``(graph, z)``."""
    xp = xp_of(x)
    sq, sn, _join = ps
    y, z = xp.zeros_like(x), xp.zeros_like(x)
    g = (eagle.Graph().add(sq.bind(x=x, a=a, y=y), name="square")
         .add(sn.bind(x=y, y=z), name="sin").build(record=record))
    return g, z


def fork_join(ps, x, a, *, record=False):
    """``u = a x^2`` and ``v = sin x`` side by side, then ``w = u v + u``:
    ``(graph, w)``; ``x``'s adjoint is the sum of both branches'."""
    xp = xp_of(x)
    sq, sn, join = ps
    u, v, w = xp.zeros_like(x), xp.zeros_like(x), xp.zeros_like(x)
    g = (eagle.Graph()
         .add_group([sq.bind(x=x, a=a, y=u), sn.bind(x=x, y=v)], names=["square", "sin"])
         .add(join.bind(u=u, v=v, w=w), name="join").build(record=record))
    return g, w


def central_difference(make, ps, x, a, seed, h=1e-6):
    """``(dJ/dx per sample, dJ/da)`` of ``J = sum(seed * out)`` by central
    differences (samples are independent, so one shift of every ``x`` at once
    gives every per-sample derivative)."""
    def out(x_, a_):
        g, o = make(ps, x_.copy(), a_)
        g.launch()
        return o

    dx = (out(x + h, a) - out(x - h, a)) * seed / (2 * h)
    da = float(((out(x, a + h) - out(x, a - h)) * seed).sum() / (2 * h))
    return dx, da
