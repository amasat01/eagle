# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""What ``events="refine"`` refuses, each message pinned: a stop rule with no
smooth crossing, a sample with no crossing in its stop step (unless
``on_no_event="zero"``, which keeps its stepped stop with zero event
sensitivity), a grazing crossing, two stop rules crossing in one step, and
the door's own value checks. (A phase inside a loop is pinned in
test_events_bouncing_ball.py; a reset kernel reading other samples meets the
existing placement refusal.)"""

from __future__ import annotations

import numpy as np
import pytest

import eagle
import eagle.diff
from _events_fixtures import ev_fall, ev_osc, host

hawk = pytest.importorskip("hawk")
from hawk import Mutable, Param, Scalar, Terminated  # noqa: E402
from hawk import math as hm  # noqa: E402

N = 4


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_stepwise(dt: Param, terminated: Terminated, x: Mutable[Scalar], t: Mutable[Scalar]):
    xn = x - dt
    x = xn
    t = t + dt
    terminated = hm.floor(xn) < 0.0


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_tangent(dt: Param, terminated: Terminated, x: Mutable[Scalar], t: Mutable[Scalar]):
    x = x + dt
    tn = t + dt
    t = tn
    d = tn - 1.0
    terminated = d * d * d >= 0.0


@hawk.kernel(steps=1, time="t", time_step="dt")
def ev_two(dt: Param, big_t: Param, terminated: Terminated,
           x: Mutable[Scalar], v: Mutable[Scalar], t: Mutable[Scalar]):
    vn = v - dt * 4.0 * x
    xn = x + dt * v
    x = xn
    v = vn
    tn = t + dt
    t = tn
    terminated = hm.lor(xn <= 0.0, tn >= big_t)


@hawk.kernel(steps=1)
def ev_untimed(dt: Param, terminated: Terminated, y: Mutable[Scalar]):
    yn = y - dt
    y = yn
    terminated = yn < 0.0


def test_a_stop_rule_with_no_smooth_crossing_is_refused():
    with pytest.raises(Exception, match=r'events="refine": the stop rule of .ev_stepwise. is '
                                        r"not a smooth crossing \(.*\); refine needs g\(x\) "
                                        "that changes sign, not a count or a flag"):
        eagle.simulate(ev_stepwise, dt=0.1, x=np.ones(N), t=np.zeros(N), max_steps=50,
                       events="refine")


def test_a_sample_with_no_crossing_is_refused():
    y = np.array([0.5, 1.0, 1e6, 1e6])
    with pytest.raises(ValueError, match=r'events="refine": 2 samples \(first: 2\) have no '
                                         "crossing in their stop step; they hit max_steps or "
                                         'jumped the surface — pass on_no_event="zero" to '
                                         "take a zero event sensitivity for them"):
        eagle.simulate(ev_fall, dt=0.05, g=9.81, y=y, v=np.zeros(N), t=np.zeros(N),
                       max_steps=40, events="refine")


def test_on_no_event_zero_keeps_the_stepped_stop():
    y = np.array([0.5, 1.0, 1e6, 1e6])
    res = eagle.simulate(ev_fall, dt=0.05, g=9.81, y=y, v=np.zeros(N), t=np.zeros(N),
                         max_steps=40, events="refine", on_no_event="zero", record=True)
    assert host(res.event_status).tolist()[2:] == [3, 3]
    np.testing.assert_array_equal(res.t_stop[2:], 40 * 0.05)
    grads = eagle.diff.reverse(res.record, {"t": np.ones(N)})
    assert (host(grads["y"])[2:] == 0).all()
    assert (host(grads["y"])[:2] != 0).all()


def test_a_grazing_crossing_is_refused():
    with pytest.raises(ValueError, match=r'events="refine": 4 samples \(first: 0\) graze the '
                                         r"surface \(\|dg/dτ\| below GRAZE_TOL\); the event "
                                         "time is not differentiable there"):
        eagle.simulate(ev_tangent, dt=0.125, x=np.zeros(N), t=np.zeros(N), max_steps=50,
                       events="refine")


def test_two_rules_crossing_in_one_step_are_refused():
    with pytest.raises(ValueError, match='events="refine": two stop rules cross in the same '
                                         "step for sample 0"):
        eagle.simulate(ev_two, dt=0.1, big_t=0.8, x=np.ones(N), v=np.zeros(N),
                       t=np.zeros(N), max_steps=50, events="refine")


def test_the_door_checks_its_values():
    with pytest.raises(ValueError, match="events is 'step' .* or 'refine'"):
        eagle.simulate(ev_osc, dt=0.1, w=2.0, a=0.3, x=np.ones(N), v=np.zeros(N),
                       t=np.zeros(N), max_steps=10, events="exact")
    with pytest.raises(ValueError, match="on_no_event is 'raise'"):
        eagle.simulate(ev_osc, dt=0.1, w=2.0, a=0.3, x=np.ones(N), v=np.zeros(N),
                       t=np.zeros(N), max_steps=10, events="refine", on_no_event="skip")
    with pytest.raises(ValueError, match='on_no_event= applies to events="refine" only'):
        eagle.simulate(ev_osc, dt=0.1, w=2.0, a=0.3, x=np.ones(N), v=np.zeros(N),
                       t=np.zeros(N), max_steps=10, on_no_event="zero")
    with pytest.raises(Exception, match=r"time_step="):
        eagle.simulate(ev_untimed, dt=0.1, y=np.ones(N), max_steps=50, events="refine")
