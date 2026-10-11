# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``events="refine"`` on a compacting run, and on one that also reorders its
planes: the event times, the refined state and the gradients equal the
plain (uncompacted) run's, on the host and on the device. The stops are
scattered over the batch so finished samples leave holes everywhere."""

from __future__ import annotations

import numpy as np
import pytest

import eagle
import eagle.diff
from _events_fixtures import BACKENDS, ev_pend, ev_pend_map, host, xp_for

N = 4096          # the smallest span a reorder fires on


def _run(xp, model, **kw):
    rng = np.random.default_rng(11)
    x0 = xp.asarray(rng.uniform(0.05, 1.5, N))
    v0 = xp.asarray(rng.uniform(-1.0, 1.5, N))
    sim = eagle.simulation(model, dt=0.05, c=0.1, a=0.05, x=x0, v=v0, t=xp.zeros(N),
                           max_steps=400, events="refine", record=True, **kw)
    res = sim.run()
    out = {k: host(res[k]).copy() for k in ("x", "v", "t")}
    out["t_stop"] = host(res.t_stop).copy()
    grads = eagle.diff.reverse(res.record, {"t": xp.ones(N), "v": xp.ones(N)})
    return sim, out, {k: host(g) for k, g in grads.items()}


@pytest.mark.parametrize("kw", [dict(every=4), dict(every=4, reorder=0.5)],
                         ids=["compacting", "reordering"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_a_compacting_refined_run_equals_the_plain_one(backend, kw):
    xp = xp_for(backend)
    _p, plain, plain_g = _run(xp, ev_pend)
    sim, got, got_g = _run(xp, ev_pend_map, **kw)
    if "reorder" in kw:
        epochs = [c["slot_samples"] for c in sim.record._tape.checkpoints]
        assert any((host(e) != np.arange(N)).any() for e in epochs)   # it did reorder
    for k in plain:
        np.testing.assert_allclose(got[k], plain[k], rtol=1e-13, atol=1e-15, err_msg=k)
    assert sorted(got_g) == sorted(plain_g)
    for k in plain_g:
        np.testing.assert_allclose(got_g[k], plain_g[k], rtol=1e-11, atol=1e-13, err_msg=k)
