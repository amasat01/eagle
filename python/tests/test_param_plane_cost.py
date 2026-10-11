# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The price of a parameter plane: the same kernel reading its parameter from
a one-cell plane (a recorded graph's form) against the by-value immediate,
one launch against the other and nothing else, within the forward floor."""

import time

import numpy as np

from _reverse_fixtures import chain, plans

N = 1 << 18
FLOOR = 1.10


def _paired_best(a, b, reps=41):
    best = [float("inf"), float("inf")]
    for _ in range(reps):
        for i, fn in enumerate((a, b)):
            t0 = time.perf_counter()
            fn()
            best[i] = min(best[i], time.perf_counter() - t0)
    return best


def test_a_parameter_plane_costs_no_more_than_the_floor(tmp_path):
    ps = plans(tmp_path, "cpu")
    x = np.linspace(-1, 1, N)
    plain, _ = chain(ps, x, 0.5)
    recorded, _ = chain(ps, x, 0.5, record=True)
    cell, immediate = recorded._nodes[0].payload, plain._nodes[0].payload
    cell.launch(); immediate.launch()
    t_cell, t_imm = _paired_best(cell.launch, immediate.launch)
    print(f"\nN={N} cell={t_cell * 1e6:.0f}us immediate={t_imm * 1e6:.0f}us "
          f"ratio={t_cell / t_imm:.3f}")
    assert t_cell / t_imm <= FLOOR
