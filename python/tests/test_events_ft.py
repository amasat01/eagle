# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Thread-safety of ``events="refine"``: two Simulations refining and
reversing at once in two threads give exactly the single-thread results (each
refinement belongs to its own runner)."""

from __future__ import annotations

import threading

import numpy as np

import eagle
import eagle.diff
from _events_fixtures import ev_osc

N = 16


def _one(a):
    res = eagle.simulate(ev_osc, dt=0.1, w=2.0, a=a, x=np.linspace(0.8, 1.2, N),
                         v=np.zeros(N), t=np.zeros(N), max_steps=100, events="refine",
                         record=True)
    grads = eagle.diff.reverse(res.record, {"t": np.ones(N)})
    return res.t_stop.copy(), float(grads["w"]), float(grads["a"])


def test_two_threads_two_simulations():
    want = [_one(0.3), _one(0.2)]
    for _ in range(3):
        got, errors = [None, None], []

        def work(i, a):
            try:
                got[i] = _one(a)
            except BaseException as exc:    # surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=work, args=(i, a)) for i, a in enumerate((0.3, 0.2))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors, errors
        for g, w in zip(got, want):
            np.testing.assert_array_equal(g[0], w[0])
            assert g[1:] == w[1:]
