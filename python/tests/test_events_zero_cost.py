# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``events="step"`` (the default) costs nothing: with every event entry
point patched to raise, a plain and a recorded run still run and give the
same bits as before; the runner binds the same planes and allocates no
refinement."""

from __future__ import annotations

import numpy as np

import eagle
import eagle.diff
import eagle.diff._run as run_mod
import eagle.diff.events as events_mod
from _events_fixtures import ev_osc

N = 8


def _run(**kw):
    sim = eagle.simulation(ev_osc, dt=0.1, w=2.0, a=0.3, x=np.ones(N), v=np.zeros(N),
                           t=np.zeros(N), max_steps=50, **kw)
    return sim, sim.run()


def test_the_default_path_never_touches_event_code(monkeypatch):
    plain_before = _run()[1].x.copy()
    _sim, rec = _run(record=True)
    rec_before = rec.x.copy()
    eagle.diff.reverse(rec.record, {"x": np.ones(N)})

    def boom(*_a, **_k):
        raise AssertionError("event code ran on the default path")

    monkeypatch.setattr(events_mod, "RefinePass", boom)
    monkeypatch.setattr(events_mod, "RefineReverse", boom)
    monkeypatch.setattr(events_mod, "event_kernels", boom)
    for name in ("events_apply_host", "events_refine_host", "events_reverse_steps"):
        monkeypatch.setattr(events_mod._core, name, boom)
    sim, res = _run()
    np.testing.assert_array_equal(res.x, plain_before)
    assert sim.runner._recorder is None
    assert res.t_stop is None and res.event_status is None
    _sim, rec = _run(record=True)
    np.testing.assert_array_equal(rec.x, rec_before)
    assert rec.record._tape.events is None
    eagle.diff.reverse(rec.record, {"x": np.ones(N)})


def test_the_default_binds_the_same_planes():
    a, _ = _run()
    b, _ = _run(events="step")
    assert sorted(a.runner.step._planes) == sorted(b.runner.step._planes)
    assert run_mod.RunTape(None, None, None, None, None).events is None
