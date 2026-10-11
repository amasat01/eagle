# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A plane a node read and a later node overwrites is copied once, just before
the overwrite, so its reverse still sees the value it read; a graph that
overwrites nothing copies nothing."""

import numpy as np

import eagle
from _reverse_fixtures import central_difference, chain, plans

N = 48


def overwrite_chain(ps, x, a, *, record=False):
    """``y = a x^2``, then ``x = sin y`` written back over ``x``: ``(graph, x)``."""
    sq, sn, _join = ps
    y = np.zeros_like(x)
    g = (eagle.Graph().add(sq.bind(x=x, a=a, y=y), name="square")
         .add(sn.bind(x=y, y=x), name="sin").build(record=record))
    return g, x


def test_an_overwritten_input_is_copied_once_and_the_gradient_is_right(tmp_path):
    ps = plans(tmp_path, "cpu")
    x0 = np.linspace(-1.2, 1.3, N)
    seed = np.linspace(0.5, 1.5, N)
    g, _x = overwrite_chain(ps, x0.copy(), 0.6, record=True)
    g.launch()
    assert len(g.record._tape.copies) == 1
    grads = eagle.diff.reverse(g.record, {"sin.y": seed})
    dx, da = central_difference(overwrite_chain, ps, x0, 0.6, seed)
    np.testing.assert_allclose(grads["x"], dx, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(float(grads["a"]), da, rtol=1e-6)


def test_a_graph_that_overwrites_nothing_copies_nothing(tmp_path):
    ps = plans(tmp_path, "cpu")
    g, _z = chain(ps, np.linspace(-1, 1, N), 0.6, record=True)
    g.launch()
    assert g.record._tape.copies == []
