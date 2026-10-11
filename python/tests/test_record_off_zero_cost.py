# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""record=False leaves the forward exactly as it was: no taping twin is
built, nothing extra is bound, no Record is made, and the run takes the
same launch path. The same counters move under record=True, so the row
can see a change."""

from __future__ import annotations

import numpy as np
import pytest

from eagle import _counters
from _run_reverse_fixtures import inputs, rr_osc, simulation

pytest.importorskip("hawk")


def _delta(record):
    before = _counters.snapshot()
    sim = simulation(rr_osc, *inputs(np), record=record)
    report = sim.run()
    after = _counters.snapshot()
    return sim, report, {k: after[k] - before[k] for k in after}


def test_record_off_builds_and_binds_nothing_extra():
    simulation(rr_osc, *inputs(np), record=True)      # warm the plugin caches
    sim, report, off = _delta(False)
    assert sim.runner._recorder is None and sim.record is None
    assert report.record is None and report.report.mode != "record"
    assert off["records"] == 0 and off["reverses"] == 0
    assert off["binds"] == 1                           # the plan itself, nothing more
    _sim, rec_report, on = _delta(True)
    assert rec_report.report.mode == "record"
    assert on["records"] == 1 and on["binds"] > off["binds"]
