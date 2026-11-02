# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eff_sol (attempts per second x ops per attempt / the device's FMA rate) of the ensemble cards: the constant is the
hand count of the step, and no committed card row is above the speed of light (a row above 1 blocks publishing)."""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

_BENCH = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"


def test_ops_per_attempt_is_the_hand_count_of_the_step():
    spec = importlib.util.spec_from_file_location("eagle_perf_card_eff", _BENCH / "perf_card" / "perf_card.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ref = json.loads((_BENCH / "sol_reference.json").read_text())
    assert ref["ops_per_attempt"] == module.FLOPS_PER_ATTEMPT


def test_no_committed_card_row_is_above_the_speed_of_light():
    cards = sorted((_BENCH / "perf_card").glob("*card_*.json"))
    cards = [c for c in cards if not c.name.startswith("feature_card")]
    if not cards:
        pytest.skip("no committed ensemble card yet")
    for path in cards:
        card = json.loads(path.read_text())
        rows = [r for r in card["results"] if r.get("status") == "ok"]
        assert rows and all(r.get("eff_sol") is not None and 0 < r["eff_sol"] <= 1.0
                            for r in rows if r["arm"] != "eagle_cpu"), path.name
        assert not card.get("eff_sol_over_one"), path.name
