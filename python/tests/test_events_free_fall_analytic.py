# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Free fall refined at the ground, against the closed form (the integration
is exact, so only round-off separates them): t* = sqrt(2 y0 / g),
dt*/dy0 = 1 / (g t*), dv*/dy0 = -1 / t*. The bound is the refine's own
round-off (8 Newton rounds: 64 u) plus a few u per step of the stepped
path. A single-step and an automatic (several steps per launch) kernel
alike."""

from __future__ import annotations

import numpy as np
import pytest

import eagle.diff
from _events_fixtures import BACKENDS, GRAVITY, ev_fall, ev_fall_auto, fall, host, xp_for

N = 32


def _within(got, want, rtol):
    ratio = np.abs(got - want) / (np.abs(want) * rtol)
    assert ratio.max() <= 1.0, f"worst error is {ratio.max():.3g}x the bound"


@pytest.mark.parametrize("model", [ev_fall, ev_fall_auto], ids=["single_step", "automatic"])
@pytest.mark.parametrize("backend", BACKENDS)
def test_free_fall_matches_the_closed_form(backend, model):
    xp = xp_for(backend)
    y0 = np.linspace(1.0, 20.0, N)
    res = fall(xp, y0, dt=0.1, model=model)
    t_star = np.sqrt(2.0 * y0 / GRAVITY)
    steps = host(res.record.steps)
    rtol = np.finfo(float).eps * (64 + 4 * steps)
    _within(host(res.t_stop), t_star, rtol)
    assert set(host(res.event_status).tolist()) <= {1, 2}
    _within(host(res.v), -GRAVITY * t_star, rtol * 4)
    assert np.abs(host(res.y)).max() < 1e-12
    dt_dy = host(eagle.diff.reverse(res.record, {"t": xp.ones(N)})["y"])
    _within(dt_dy, 1.0 / (GRAVITY * t_star), rtol * 16)
    dv_dy = host(eagle.diff.reverse(res.record, {"v": xp.ones(N)})["y"])
    _within(dv_dy, -1.0 / t_star, rtol * 16)
