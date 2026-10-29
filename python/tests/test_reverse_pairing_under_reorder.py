# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Sample pairing through a reverse: a graph whose planes an ActiveSet has
physically reordered still returns every gradient in SAMPLE order, bit for
bit the unreordered run, and the sample-id plane that rode the reorder is the
Record's slot -> sample map."""

import numpy as np

import eagle
from _reverse_fixtures import chain, plans

def _bytes(a):
    return np.asarray(a.get() if hasattr(a, "get") else a).tobytes()


N = 6000


def _forward(ps, x, seed, *, reorder):
    """The chain over ``x`` (owned by a reordering set when ``reorder``):
    ``(grads, sample ids as they sit in the slots, record perm)``."""
    x = x.copy()
    sid = np.arange(N, dtype=np.int32)
    mask = np.zeros(N, dtype=np.uint8)
    mask[::3] = 1                                   # every third sample finished
    g, z = chain(ps, x, 0.7)
    y = g._nodes[0].payload._planes["y"]
    aset = None
    if reorder:
        aset = eagle.ActiveSet(mask, reorder=True).own(x, y, z, sid)
    g = (eagle.Graph().add(g._nodes[0].payload, name="square")
         .add(g._nodes[1].payload, name="sin").build(record=True))
    if reorder:
        aset.fire[0] = 1
        aset.reorder()
        assert aset.permuted
    g.launch()
    grads = eagle.diff.reverse(g.record, {"sin.y": seed})
    return grads, sid, g.record._tape.perm


def test_gradients_come_back_in_sample_order_bitwise(tmp_path):
    ps = plans(tmp_path, "cpu")
    rng = np.random.default_rng(11)
    x = rng.uniform(-2, 2, N)
    seed = rng.uniform(0.1, 3.0, N)
    ref, _sid, perm0 = _forward(ps, x, seed, reorder=False)
    got, sid, perm = _forward(ps, x, seed, reorder=True)
    assert perm0 is None
    assert not np.array_equal(perm, np.arange(N))   # the reorder really moved samples
    np.testing.assert_array_equal(sid, perm)        # the sample-id plane rode with them
    assert _bytes(got["x"]) == _bytes(ref["x"])
    assert _bytes(got["a"]) == _bytes(ref["a"])  # summed in sample order, not slot order
