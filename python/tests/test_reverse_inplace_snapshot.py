# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A node writing over a plane it reads (bound in place) has that plane copied
just before it runs: one copy per in-place node, and the gradient through it
equals central differences."""

import numpy as np

import eagle
from _reverse_fixtures import central_difference, plans

N = 40


def in_place(ps, x, a, *, record=False):
    """``x = a x^2`` then ``x = sin x``, both in place: ``(graph, x)``."""
    sq, sn, _join = ps
    g = (eagle.Graph().add(sq.bind(x=x, a=a, y=x), name="square")
         .add(sn.bind(x=x, y=x), name="sin").build(record=record))
    return g, x


def test_one_copy_per_in_place_node_and_fd_equal(tmp_path):
    ps = plans(tmp_path, "cpu")
    x0 = np.linspace(-1.1, 0.9, N)
    seed = np.linspace(1.0, 2.0, N)
    g, _x = in_place(ps, x0.copy(), 0.8, record=True)
    g.launch()
    assert len(g.record._tape.copies) == 2
    grads = eagle.diff.reverse(g.record, {"y": seed})   # x, written as y
    dx, da = central_difference(in_place, ps, x0, 0.8, seed)
    np.testing.assert_allclose(grads["x"], dx, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(float(grads["a"]), da, rtol=1e-6)
