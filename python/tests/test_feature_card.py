# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The feature card (``benchmarks/perf_card/feature_card.py``): its rows are its manifest, and a failing arm never stops it.

The card runs every arm in a fresh process. These tests pin the eleven rows and their arms to
``benchmarks/cards_manifest.json`` (take a row out of ``ROWS`` and the first test is RED), and force the
two labelled paths of a run: a failed allocation in one arm reads ``out_of_memory``, a 121 s warm-up reads
``not_run: budget``, and in both the card still completes and is written.

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

pytestmark = pytest.mark.repo_local

_BENCH = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"
ROW_NAMES = {"wide_forms", "steps_fused", "control_flow", "linalg_lane", "linalg_team", "contraction", "fft", "lane_set",
             "scatter", "sample_sum", "mpi_partition"}


@pytest.fixture(scope="module")
def card_module():
    import sys

    sys.path.insert(0, str(_BENCH))
    spec = importlib.util.spec_from_file_location("feature_card", _BENCH / "perf_card" / "feature_card.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_card_has_the_eleven_rows(card_module):
    assert set(card_module.ROWS) == ROW_NAMES
    for name, spec in card_module.ROWS.items():
        assert spec["ref"] in spec["arms"], name
        assert set(spec["host"]) <= set(spec["arms"]), name


def test_the_manifest_is_the_script(card_module):
    recorded = json.loads((_BENCH / "cards_manifest.json").read_text())["c2_feature_card"]
    assert recorded == json.loads(json.dumps(card_module.manifest()))


def test_a_row_taken_out_is_not_the_manifest(card_module, monkeypatch):
    recorded = json.loads((_BENCH / "cards_manifest.json").read_text())["c2_feature_card"]
    monkeypatch.delitem(card_module.ROWS, "scatter")
    assert recorded != json.loads(json.dumps(card_module.manifest()))


@pytest.mark.parametrize("force, status", [("np_add_at:oom", "out_of_memory"), ("np_add_at:slow", "not_run: budget")])
def test_a_failed_arm_is_labelled_and_the_card_completes(card_module, monkeypatch, tmp_path, force, status):
    pytest.importorskip("hawk")
    monkeypatch.setenv("FEATURE_CARD_FORCE", force)
    monkeypatch.setitem(card_module.ROWS["scatter"], "arms", ("np_add_at",))
    monkeypatch.setitem(card_module.ROWS["scatter"], "ref", "np_add_at")
    assert card_module.main(["--smoke", "--rows", "scatter", "--out-dir", str(tmp_path)]) == 0
    (written,) = tmp_path.glob("feature_card_*.json")
    (row,) = json.loads(written.read_text())["rows"]
    assert row["arms"]["np_add_at"]["status"] == status
    assert list(tmp_path.glob("feature_card_*.md"))


def test_an_arm_that_runs_records_its_clock_and_rounds(card_module, monkeypatch, tmp_path):
    pytest.importorskip("hawk")
    monkeypatch.delenv("FEATURE_CARD_FORCE", raising=False)
    monkeypatch.setitem(card_module.ROWS["scatter"], "arms", ("np_add_at",))
    monkeypatch.setitem(card_module.ROWS["scatter"], "ref", "np_add_at")
    card_module.main(["--smoke", "--rows", "scatter", "--out-dir", str(tmp_path)])
    arm = json.loads(next(tmp_path.glob("feature_card_*.json")).read_text())["rows"][0]["arms"]["np_add_at"]
    assert arm["status"] == "ok" and {"median", "iqr", "rounds"} <= set(arm["wall_s"]) and "sm_mhz" in arm
