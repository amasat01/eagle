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
    """``(persist entry's resident threads, fused entry's resident threads)`` of
    the kernel the row timed: its own threads per SM times the SMs. The first
    sizes the regime rule, the second the fused entry's waves in the model."""
    return int(row["capacity"]), int(row["sms"]) * int(row["resident_per_sm"])


def _judge(row):
    """``(kind, factor)`` of a row: how far the pick the rule or the model makes
    is from the faster entry (``"slower"``: the pick's wall over the faster
    wall), or how far the model's ratio is from the measured one (``"model"``:
    the larger of the two over the smaller)."""
    cap, resident = _capacity(row)
    n = int(row["n"])
    fused, persist = float(row["fused_ms"]), float(row["persist_ms"])
    regime = ud._entry_regime(cap, n)
    if regime == "persist":
        return "slower", persist / min(fused, persist)
    if regime == "fused":
        return "slower", fused / min(fused, persist)
    ratio = _core.entry_ratio(
        int(row["major"]), int(row["minor"]), int(row["sms"]), int(row["fp64_ratio"]),
        int(row["clock_khz"]), row["precision"] == "fp64", resident, n, int(row["S"]),
        _fill(row["dist"]), float(row["step_ops"]))
    measured = persist / fused
    return "model", max(ratio / measured, measured / ratio), ratio, measured


def _comparable(rows):
    return [r for r in rows if not r["not_comparable"]]


def test_every_row_is_judged_or_says_why_not():
    for row in _rows():
        assert row["not_comparable"] or (row["fused_ms"] and row["persist_ms"]), row["cell"]
        assert not (row["known_miss"] and row["not_comparable"]), row["cell"]


def test_model_pick_agrees_with_the_profiles():
    misses = []
    for row in _comparable(_rows()):
        if row["known_miss"]:
            continue
        verdict = _judge(row)
        limit = 1.0 + _AGREE_WITHIN if verdict[0] == "slower" else _MODEL_WITHIN
        if verdict[1] > limit:
            misses.append((row["device"], row["cell"], verdict[:2]))
    assert not misses, misses


def test_a_known_miss_still_fails_its_limit_and_stays_within_one_and_a_half():
    """A row listed as a known miss must still miss (a fix retires the entry) and
    must not get worse than a pick 1.5 times the faster entry."""
    stale, worse = [], []
    for row in _comparable(_rows()):
        if not row["known_miss"]:
            continue
        verdict = _judge(row)
        limit = 1.0 + _AGREE_WITHIN if verdict[0] == "slower" else _MODEL_WITHIN
        if verdict[1] <= limit:
            stale.append((row["device"], row["cell"]))
        if verdict[0] == "slower":
            worst = verdict[1]
        else:
            # a band row the model calls wrongly takes the slower entry once
            ratio, measured = verdict[2], verdict[3]
            pick = "persist" if ratio < 0.8 else "fused" if ratio > 1.25 else None
            worst = 1.0 if pick is None else (measured if pick == "persist" else 1 / measured)
        if worst > 1.5:
            worse.append((row["device"], row["cell"], worst))
    assert not stale, stale
    assert not worse, worse


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
