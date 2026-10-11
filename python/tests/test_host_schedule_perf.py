# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Paired wall rows: branch_tasks beats work_share in its regime (small
batches, several independent branches) and is not chosen outside it."""

import time

import pytest

import eagle
from eagle import _core

from _graph_fixtures import host_fork


def _best_of(fn, reps=7):
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _paired(tmp_path, n, branches=8, depth=8):
    """``(graph, tasks_seconds, work_share_seconds)``: the same plans, run by
    the built graph and node by node over the whole team."""
    chains, _arrays, _ = host_fork(tmp_path, branches=branches, n=n, depth=depth)
    g = eagle.Graph().add_group(chains).build()

    def work_share():
        for chain in chains:
            for bound in chain:
                bound.launch()

    g.launch(); work_share()                      # warm both arms
    return g, _best_of(lambda: g.launch()), _best_of(work_share)


def test_branch_tasks_is_at_least_twice_as_fast_in_its_regime(tmp_path):
    if _core.host_threads() < 4:
        pytest.skip("needs a host team of at least 4 threads")
    g, tasks, shared = _paired(tmp_path, n=1024)
    print(f"\nN=1024 B=8 tasks={tasks * 1e6:.0f}us work_share={shared * 1e6:.0f}us "
          f"ratio={shared / tasks:.1f}x")
    assert g.schedule == ("branch_tasks",)
    assert shared / tasks >= 2.0


def test_work_share_is_kept_outside_the_regime(tmp_path):
    g, tasks, shared = _paired(tmp_path, n=65536, branches=2, depth=2)
    print(f"\nN=65536 B=2 graph={tasks * 1e6:.0f}us work_share={shared * 1e6:.0f}us")
    assert g.schedule == ("work_share",)
