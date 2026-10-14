# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Thread-safety of ``events="refine"``: two Simulations refining and
reversing at once in two threads give exactly the single-thread results (each
refinement belongs to its own runner)."""

from __future__ import annotations

import threading

import numpy as np
import pytest

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


def test_n_threads_one_model_refine_and_reverse_equal_the_serial_run():
    want = {a: _one(a) for a in (0.2, 0.3)}
    threads_n = 6
    got, errors = [None] * threads_n, []
    start = threading.Barrier(threads_n)

    def work(i):
        try:
            start.wait()
            got[i] = _one((0.3, 0.2)[i % 2])
        except BaseException as exc:    # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(threads_n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors
    for i, g in enumerate(got):
        w = want[(0.3, 0.2)[i % 2]]
        np.testing.assert_array_equal(g[0], w[0])
        assert g[1:] == w[1:]


@pytest.mark.gpu
def test_device_n_threads_one_model_refine_equal_the_serial_run():
    cp = pytest.importorskip("cupy")

    def one(a):
        res = eagle.simulate(ev_osc, dt=0.1, w=2.0, a=a, x=cp.linspace(0.8, 1.2, N),
                             v=cp.zeros(N), t=cp.zeros(N), max_steps=100, events="refine")
        return cp.asnumpy(res.t_stop)

    want = {a: one(a) for a in (0.2, 0.3)}
    got, errors = [None] * 4, []
    start = threading.Barrier(4)

    def work(i):
        try:
            start.wait()
            got[i] = one((0.3, 0.2)[i % 2])
        except BaseException as exc:    # surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert not errors, errors
    for i, g in enumerate(got):
        np.testing.assert_array_equal(g, want[(0.3, 0.2)[i % 2]])
