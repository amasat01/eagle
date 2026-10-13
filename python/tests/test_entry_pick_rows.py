"""The automatic entry pick against the measured walls of five devices.

``data/entry_pick_rows.csv`` holds, per device and cell, the wall of the fused
and of the persistent entry (ms) with the device facts the pick reads, so no GPU
is needed. Where the regime rule names an entry without timing (a batch of at
least 8 or at most 1/4 of the capacity) that entry must be within 5 % of the
faster one; between the two the model's predicted persist-over-fused ratio must
be within a factor of two of the measured one.

The walls come from the oscillator profile tables of an H100, an A100, an RTX
PRO 6000, a T4 (the oscillator rows of the quick table) and, for the P2000, the
committed performance card, whose cells time the persistent entry and the
entry the automatic pick chose, so only the cells where that was the fused entry
carry both. The step fill of a ``spread:k`` batch is the mean over the maximum of
a log-uniform stop step in ``[S/k, S]``, ``(1 - 1/k) / ln k`` (``uniform``: 1).
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from eagle import _core
from eagle import _until_done as ud

_ROWS = Path(__file__).resolve().parent / "data" / "entry_pick_rows.csv"
_AGREE_WITHIN = 0.05
_MODEL_WITHIN = 2.0


def _rows():
    with open(_ROWS, newline="") as fh:
        return list(csv.DictReader(fh))


def _fill(dist: str) -> float:
    if dist == "uniform":
        return 1.0
    k = 100 if dist == "spread" else int(dist.split(":")[1])
    return (1.0 - 1.0 / k) / math.log(k)


def _capacity(row) -> tuple:
    """``(latency-regime capacity, resident threads of the device)``."""
    sms, per_sm = int(row["sms"]), int(row["resident_per_sm"])
    return min(sms * per_sm, sms * 32 * 8), sms * per_sm


def test_the_table_covers_every_device_and_both_regimes():
    rows = _rows()
    assert {r["device"] for r in rows} == {"H100", "A100", "RTX", "T4", "P2000"}
    regimes = {ud._entry_regime(_capacity(r)[0], int(r["n"])) for r in rows}
    assert regimes == {"fused", "persist", "band"}


def test_model_pick_agrees_with_the_profiles():
    misses = []
    for row in _rows():
        cap, resident = _capacity(row)
        n = int(row["n"])
        fused, persist = float(row["fused_ms"]), float(row["persist_ms"])
        regime = ud._entry_regime(cap, n)
        if regime == "persist":
            slower = persist / min(fused, persist) - 1.0
        elif regime == "fused":
            slower = fused / min(fused, persist) - 1.0
        else:
            ratio = _core.entry_ratio(
                int(row["major"]), int(row["minor"]), int(row["sms"]), int(row["fp64_ratio"]),
                int(row["clock_khz"]), row["precision"] == "fp64", resident, n, int(row["S"]),
                _fill(row["dist"]), float(row["step_ops"]))
            measured = persist / fused
            if not (1 / _MODEL_WITHIN <= ratio / measured <= _MODEL_WITHIN):
                misses.append((row["device"], row["cell"], f"model {ratio:.2f} vs measured {measured:.2f}"))
            continue
        if slower > _AGREE_WITHIN:
            misses.append((row["device"], row["cell"], f"{regime} entry {slower:.1%} slower"))
    assert not misses, misses


def test_the_model_is_unknown_without_a_clock_or_a_step_cost():
    args = (9, 0, 132, 2, 1980000, True, 202752, 100000, 1000, 1.0, 41.0)
    assert _core.entry_ratio(*args) > 0.0
    assert _core.entry_ratio(*args[:4], 0, *args[5:]) == 0.0
    assert _core.entry_ratio(*args[:10], 0.0) == 0.0


@pytest.mark.parametrize("capacity, n, want", [
    (1000, 250, "fused"), (1000, 251, "band"), (1000, 7999, "band"), (1000, 8000, "persist"),
    (0, 10, "band"), (-1, 10_000_000, "band")])
def test_the_regime_names_the_sure_batches(capacity, n, want):
    assert ud._entry_regime(capacity, n) == want
