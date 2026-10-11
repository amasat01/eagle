# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The splitting rule seen from Python: the decision table of
``tests/test_DispatchPolicy.cpp`` reproduced through the binding, the static
selector check, the look cadence and a shared state object."""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from eagle import _core
from eagle import _dispatch_policy as dp

N = 1 << 20
BATCH = 1 << 21  # the measured batch


def _cost(*branches, vectorised=False):
    return SimpleNamespace(branches=[SimpleNamespace(as_tuple=lambda b=b: b) for b in branches],
                           fused_vectorised=vectorised)


# (arith32, arith64, special32, special64, loads, gathers)
CHEAP = (10, 0, 0, 0, 0, 0)
PLAIN40 = (40, 0, 0, 0, 0, 0)
HEAVY = (0, 0, 30, 0, 0, 0)
HEAVY64 = (0, 0, 0, 30, 0, 0)
# the measured pair: x * -0.5 + 1 against exp(sin x cos x) + log(|x| + 2) + tanh x, fp64
PROBE = _cost((0, 2, 0, 0, 1, 0), (0, 5, 0, 5, 1, 0), vectorised=True)


def _props(major_lanes, sms, clock_khz, mem_khz, bus, ratio):
    return {"peak_flops_sp": sms * clock_khz * 1000.0 * major_lanes * 2.0,
            "peak_bytes_per_s": 2.0 * mem_khz * 1000.0 * bus / 8.0, "fp64_ratio": ratio}


DATACENTER = _props(64, 108, 1_410_000, 1_215_000, 5120, 2)
CONSUMER = _props(128, 28, 1_700_000, 9_500_000, 192, 64)
P2000 = _props(128, 8, 1_480_000, 3_504_000, 160, 32)


def _heavy(percent):
    return [BATCH - BATCH * percent // 100, BATCH * percent // 100]


# name, target, props, cost, counts, fused_steps, remaining_steps, current, expected.
# "measured" rows match paired walls on this machine (host: 64 steps per fused
# launch; device: the whole 400-step run in one launch); the rest are the prior.
TABLE = [
    ("measured host 0% heavy", "host", None, PROBE, _heavy(0), 64, 400, "switch", "segmented"),
    ("measured host 1% heavy", "host", None, PROBE, _heavy(1), 64, 400, "switch", "segmented"),
    ("measured host 5% heavy", "host", None, PROBE, _heavy(5), 64, 400, "switch", "segmented"),
    ("measured host 50% heavy", "host", None, PROBE, _heavy(50), 64, 400, "switch", "segmented"),
    ("measured host 100% heavy", "host", None, PROBE, _heavy(100), 64, 400, "segmented", "switch"),
    ("measured device 0% heavy", "device", P2000, PROBE, _heavy(0), 400, 400, "segmented", "switch"),
    ("measured device 1% heavy", "device", P2000, PROBE, _heavy(1), 400, 400, "switch", "segmented"),
    ("measured device 5% heavy", "device", P2000, PROBE, _heavy(5), 400, 400, "switch", "segmented"),
    ("measured device 50% heavy", "device", P2000, PROBE, _heavy(50), 400, 400, "switch", "segmented"),
    ("measured device 100% heavy", "device", P2000, PROBE, _heavy(100), 400, 400, "segmented", "switch"),
    ("host unvectorised 1% heavy", "host", None, _cost(CHEAP, HEAVY), [N * 99 // 100, N // 100], 64, 400, "segmented", "switch"),
    ("host empty mix keeps segmented", "host", None, PROBE, [0, 0], 1, 16, "segmented", "segmented"),
    ("device four equal cheap branches", "device", CONSUMER, _cost(*[CHEAP] * 4), [N << 4] * 4, 400, 400, "switch", "switch"),
    ("device 64-bit heavy, consumer", "device", CONSUMER, _cost(CHEAP, HEAVY64), [N // 2, N // 2], 400, 400, "switch", "segmented"),
    ("device small batch keeps fused", "device", DATACENTER, _cost(CHEAP, HEAVY), [500, 500], 1, 16, "switch", "switch"),
    ("device short run pays its sort", "device", DATACENTER, _cost(CHEAP, PLAIN40), [N // 2, N // 2], 1, 1, "switch", "switch"),
    ("device one branch takes all", "device", DATACENTER, _cost(CHEAP, PLAIN40), [N, 0], 400, 400, "segmented", "switch"),
]


@pytest.mark.parametrize("row", TABLE, ids=[r[0] for r in TABLE])
def test_decision_table(row):
    _, target, props, cost, counts, fused, remaining, current, expected = row
    got = dp.resolve(cost, counts, target=target, props=props, current=current,
                     fused_steps=fused, remaining_steps=remaining)
    assert got == expected


def _in_band():
    """A host heavy share of the probe whose gain lies inside the band."""
    for percent in range(1, 101):
        gain = dp.predicted_gain(PROBE, _heavy(percent), fused_steps=64, remaining_steps=400)
        if dp.DISPATCH_LEAVE_GAIN <= gain < dp.DISPATCH_ENTER_GAIN:
            return _heavy(percent)
    raise AssertionError("no heavy share of the probe lands inside the band")


RUN = {"fused_steps": 64, "remaining_steps": 400}


def test_a_non_static_selector_is_never_split():
    assert dp.resolve(PROBE, _heavy(1), static=True, **RUN) == "segmented"
    assert dp.resolve(PROBE, _heavy(1), static=False, current="segmented", **RUN) == "switch"


def test_gain_matches_the_band_the_table_relies_on():
    gain = dp.predicted_gain(PROBE, _in_band(), **RUN)
    assert dp.DISPATCH_LEAVE_GAIN <= gain < dp.DISPATCH_ENTER_GAIN
    assert dp.predicted_gain(PROBE, [0, 0]) == 1.0


def test_the_measured_divergence_replaces_the_prior():
    rare = _heavy(1)
    prior = dp.predicted_gain(PROBE, rare, target="device", props=P2000, fused_steps=400,
                              remaining_steps=400)
    grouped = dp.predicted_gain(PROBE, rare, target="device", props=P2000, fused_steps=400,
                                remaining_steps=400, mixed32=0.0)
    assert grouped < prior


def test_refusals_name_the_fix():
    cost = _cost(CHEAP, HEAVY)
    with pytest.raises(ValueError, match="device properties"):
        dp.resolve(cost, [1, 1], target="device")
    with pytest.raises(ValueError, match="'host' or 'device'"):
        dp.resolve(cost, [1, 1], target="gpu")
    with pytest.raises(ValueError, match="2 to 16"):
        dp.resolve(_cost(CHEAP), [1])
    with pytest.raises(ValueError, match="'switch' or 'segmented'"):
        dp.resolve(cost, [1, 1], current="auto")


def test_a_device_props_view_is_accepted():
    view = SimpleNamespace(**P2000)
    assert dp.resolve(PROBE, _heavy(50), target="device", props=view, fused_steps=400,
                      remaining_steps=400) == "segmented"


@pytest.mark.parametrize("reads, written, expected", [
    ([("per_sample", "kind")], {"state", "t"}, True),
    ([("mutable", "mode")], {"mode"}, False),
    ([("per_sample", "a"), ("lookup", "tab")], {"tab"}, False),
    ([], {"x"}, True),
    ([["per_sample", "k"]], set(), True),
])
def test_static_selector(reads, written, expected):
    assert dp.static_selector(reads, written) is expected


def test_the_look_comes_every_sixteen_steps():
    state = dp.DispatchState()
    assert [state.due() for _ in range(16)] == [False] * 15 + [True]
    assert state.due(15) is False
    assert state.due(1) is True
    assert state.due(40) is True


def test_the_state_keeps_the_policy_through_the_band():
    state = dp.DispatchState()
    band = _in_band()
    assert state.update(PROBE, _heavy(1), **RUN) == "segmented"
    assert state.update(PROBE, band, **RUN) == "segmented"
    assert state.update(PROBE, _heavy(100), **RUN) == "switch"
    assert state.update(PROBE, band, **RUN) == "switch"
    assert state.update(PROBE, _heavy(1), static=False, **RUN) == "switch"


def test_the_state_is_safe_to_share_between_threads():
    state = dp.DispatchState()
    looks = []

    def work():
        looks.append(sum(state.due() for _ in range(1600)))

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(looks) == 8 * 1600 // dp.DISPATCH_EVERY


def test_a_real_hawk_cost_goes_through_the_rule():
    hawk = pytest.importorskip("hawk")
    import hawk.math as M
    from hawk.ir import canonical, dispatch_costs

    @hawk.kernel
    def rare(k: hawk.Index, x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
        y = M.dispatch(k, [x + 1.0, M.exp(M.sin(x) * M.cos(x)), x], policy="auto")

    cost = dispatch_costs(canonical(rare.sinks))["dispatch0"]
    assert dp.static_selector(cost.selector_reads, {"y"})
    assert cost.fused_vectorised      # sample-local: the fused host loop runs every branch
    big = 1 << 26  # the host's launch terms are noise at this size
    assert dp.resolve(cost, [big * 98 // 100, big // 100, big // 100], **RUN) == "segmented"
    assert dp.resolve(cost, [0, big, 0], **RUN) == "switch"


def test_the_bound_constants_are_the_ruled_ones():
    assert (_core.DISPATCH_ENTER_GAIN, _core.DISPATCH_LEAVE_GAIN, _core.DISPATCH_EVERY) == (1.15, 1.05, 16)


def test_a_fitted_profile_replaces_named_weights_and_refuses_unknown_ones():
    base = dp.predicted_gain(PROBE, _heavy(5), fused_steps=64, remaining_steps=400)
    slow_sort = dp.predicted_gain(PROBE, _heavy(5), fused_steps=64, remaining_steps=400,
                                  weights={"sort_seconds": 1e-6})
    assert slow_sort < base
    with pytest.raises(ValueError, match="unknown cost-model weight"):
        dp.predicted_gain(PROBE, _heavy(5), weights={"simd_loss": 3})


def test_cost_model_is_thread_safe_to_call_concurrently():
    from concurrent.futures import ThreadPoolExecutor

    want = dp.predicted_gain(PROBE, _heavy(5), fused_steps=64, remaining_steps=400)
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda _: dp.predicted_gain(PROBE, _heavy(5), fused_steps=64,
                                                        remaining_steps=400), range(64)))
    assert got == [want] * 64
