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


def _capacities(row) -> tuple:
    """``(persist entry's resident threads, one fused wave, fused entry's resident
    threads)`` of the kernel the row timed: its own threads per SM times the SMs.
    The first two size the regime rule, the last the fused entry's waves in the
    model."""
    return (int(row["capacity"]), int(row["wave_capacity"]),
            int(row["sms"]) * int(row["resident_per_sm"]))


def _regime(row) -> str:
    persist_cap, wave_cap, _ = _capacities(row)
    return ud._entry_regime(persist_cap, wave_cap, int(row["n"]))


def _slower(row) -> float:
    """For a row in a sure regime, the wall of the entry the rule names over the
    faster wall."""
    fused, persist = float(row["fused_ms"]), float(row["persist_ms"])
    named = persist if _regime(row) == "persist" else fused
    return named / min(fused, persist)


def _probed_pick(row) -> str:
    """The entry the probe settles on when the two entries take the row's walls:
    the first run takes the size pick, the second the other entry."""
    walls = {"fused_one": float(row["fused_ms"]), "persist": float(row["persist_ms"])}
    runner = ud.Runner.__new__(ud.Runner)
    first = "persist" if int(row["n"]) > int(row["wave_capacity"]) else "fused_one"
    runner._fast_state = ud._NO_PICK._replace(pick=first)
    runner._fast_mode, runner._fast_prep, runner._fast_fit = first, {}, None
    runner.n = int(row["n"])
    for _ in range(3):
        mode, probing = runner._pick_entry()
        timed = runner._carries_timing(probing)
        runner._note_run(mode, walls[mode] if timed else None, int(row["S"]), 1, 0.5)
    return runner._fast_pick


def _comparable(rows):
    return [r for r in rows if not r["not_comparable"]]


def test_every_row_is_judged_or_says_why_not():
    for row in _rows():
        assert row["not_comparable"] or (row["fused_ms"] and row["persist_ms"]), row["cell"]
        assert not (row["known_miss"] and row["not_comparable"]), row["cell"]


def test_model_pick_agrees_with_the_profiles():
    """A sure batch names an entry within 5 % of the faster one; a batch in the band
    is timed, and the pick is the faster entry (to the 2 % hysteresis)."""
    misses = []
    for row in _comparable(_rows()):
        if row["known_miss"]:
            continue
        if _regime(row) == "band":
            fused, persist = float(row["fused_ms"]), float(row["persist_ms"])
            pick = _probed_pick(row)
            picked = persist if pick == "persist" else fused
            if picked > min(fused, persist) / ud.PICK_HYSTERESIS:
                misses.append((row["device"], row["cell"], pick))
        elif _slower(row) > 1.0 + _AGREE_WITHIN:
            misses.append((row["device"], row["cell"], _slower(row)))
    assert not misses, misses


def test_a_known_miss_still_fails_its_limit_and_stays_within_one_and_a_half():
    """A row listed as a known miss must still miss (a fix retires the entry) and
    must not get worse than a pick 1.5 times the faster entry."""
    stale, worse = [], []
    for row in _comparable(_rows()):
        if not row["known_miss"]:
            continue
        assert _regime(row) != "band", row["cell"]
        slower = _slower(row)
        if slower <= 1.0 + _AGREE_WITHIN:
            stale.append((row["device"], row["cell"]))
        if slower > 1.5:
            worse.append((row["device"], row["cell"], slower))
    assert not stale, stale
    assert not worse, worse


def test_report_the_model_against_the_measured_ratio_in_the_band(capsys):
    """Not a gate: print the model's persist-over-fused ratio beside the measured
    one for every band row, so a better model can be judged against them."""
    lines = []
    for row in _comparable(_rows()):
        if _regime(row) != "band":
            continue
        ratio = _core.entry_ratio(
            int(row["major"]), int(row["minor"]), int(row["sms"]), int(row["fp64_ratio"]),
            int(row["clock_khz"]), row["precision"] == "fp64", _capacities(row)[2],
            int(row["n"]), int(row["S"]), _fill(row["dist"]), float(row["step_ops"]))
        measured = float(row["persist_ms"]) / float(row["fused_ms"])
        lines.append(f"{row['device']:6} {row['cell']:34} model {ratio:5.2f} measured {measured:5.2f}"
                     f" x{max(ratio / measured, measured / ratio):4.2f}")
    with capsys.disabled():
        print("\nentry model against the profiles (band rows):\n" + "\n".join(lines))
    assert lines


def test_the_model_is_unknown_without_a_clock_or_a_step_cost():
    args = (9, 0, 132, 2, 1980000, True, 202752, 100000, 1000, 1.0, 41.0)
    assert _core.entry_ratio(*args) > 0.0
    assert _core.entry_ratio(*args[:4], 0, *args[5:]) == 0.0
    assert _core.entry_ratio(*args[:10], 0.0) == 0.0


@pytest.mark.parametrize("persist_capacity, wave_capacity, n, want", [
    (4000, 1000, 250, "fused"), (4000, 1000, 251, "band"), (4000, 1000, 31999, "band"),
    (4000, 1000, 32000, "persist"), (0, 0, 10, "band"), (-1, -1, 10_000_000, "band"),
    (0, 1000, 10, "fused"), (4000, 0, 32000, "persist")])
def test_the_regime_names_the_sure_batches(persist_capacity, wave_capacity, n, want):
    assert ud._entry_regime(persist_capacity, wave_capacity, n) == want
