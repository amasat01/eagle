# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""With autodiff unused (record=False, the default) a graph's forward does
exactly what it did before: no recorder, no copies, no extra step, and its
wall stays within a few percent of launching the same plans by hand."""

import time

import numpy as np

from _reverse_fixtures import chain, plans

N = 1 << 18
FLOOR = 1.10


def _paired_best(a, b, reps=41):
    """Best wall of each arm, the arms alternating so machine load hits both."""
    best = [float("inf"), float("inf")]
    for _ in range(reps):
        for i, fn in enumerate((a, b)):
            t0 = time.perf_counter()
            fn()
            best[i] = min(best[i], time.perf_counter() - t0)
    return best


def test_a_plain_graph_records_nothing(tmp_path):
    g, _z = chain(plans(tmp_path, "cpu"), np.linspace(-1, 1, 64), 0.5)
    assert g._recorder is None and g.record is None
    assert len(g._cpu_steps) == len(g._nodes)          # one step per stage, no copies
    g.launch()
    assert g.record is None


def test_plain_graph_forward_wall_matches_the_plans_launched_by_hand(tmp_path):
    g, _z = chain(plans(tmp_path, "cpu"), np.linspace(-1, 1, N), 0.5)
    bounds = [n.payload for n in g._nodes]

    def by_hand():
        for b in bounds:
            b.launch()

    g.launch(); by_hand()                               # warm both arms
    graph, hand = _paired_best(g.launch, by_hand)
    print(f"\nN={N} graph={graph * 1e6:.0f}us by_hand={hand * 1e6:.0f}us "
          f"ratio={graph / hand:.3f}")
    assert graph / hand <= FLOOR
