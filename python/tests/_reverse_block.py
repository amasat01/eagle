# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A test node kind ``y = x`` whose REVERSE step blocks until released: it
forces two calls to overlap, so the busy rows test a real overlap rather than
a lucky interleaving."""

from __future__ import annotations

import threading

import numpy as np

from eagle.graph import Node, register_node_kind


class Gate:
    """``entered`` is set once the reverse step runs; it waits for ``go``."""

    def __init__(self):
        self.entered, self.go = threading.Event(), threading.Event()


def block(x, y, gate):
    return Node("test_reverse_block", (x, y, gate), name="block")


def _transpose(node, incoming, ctx):
    _x, _y, gate = node.payload
    bar_y, bar_x = incoming["y"], np.empty_like(incoming["y"])

    def step():
        gate.entered.set()
        gate.go.wait(30)
        np.copyto(bar_x, bar_y)
    ctx.add(step, reads=[bar_y], writes=[bar_x])
    return {"x": bar_x}


def _copy(node, ctx):
    x, y, _gate = node.payload
    return lambda: np.copyto(y, x)


register_node_kind(
    "test_reverse_block",
    reads=lambda n: (("x", n.payload[0], "per_sample"),),
    writes=lambda n: (("y", n.payload[1], "mutable"),),
    lower_gpu=_copy, lower_cpu=_copy, transpose=_transpose)
