# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""What keeps a cap from forcing a re-run: a stalled arm is killed alone, the card is assembled from its saved parts with the
missing arms labelled, a warm-up is stopped at its budget, the budget labels name their source, and eff_sol is derived from the
card's raw fields (a value over 1 is pending, a better count re-scores it).

``repo_local``: reads repository files, never shipped in the wheel."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.repo_local

_BENCH = pathlib.Path(__file__).resolve().parents[2] / "benchmarks"


def _load(name, path):
    sys.path.insert(0, str(_BENCH))
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cc():
    return _load("_card_common", _BENCH / "_card_common.py")


@pytest.fixture(scope="module")
def feature():
    return _load("feature_card", _BENCH / "perf_card" / "feature_card.py")


def test_a_child_with_no_progress_is_killed_alone(cc, monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "BUILD_S", 1.0)
    with pytest.raises(cc.Stalled):
        cc.run_watched([sys.executable, "-c", "import time; time.sleep(60)"], timeout=60, progress=tmp_path / "alive")


def test_a_child_that_reports_progress_runs_to_its_end(cc, monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "BUILD_S", 1.0)
    monkeypatch.setattr(cc, "STALL_S", 3.0)
    code = f"import sys,time; sys.path.insert(0, {str(_BENCH)!r}); import _card_common as c\nfor _ in range(4): time.sleep(1.5); c.heartbeat()\nprint('done')"
    done = cc.run_watched([sys.executable, "-c", code], timeout=60, progress=tmp_path / "alive")
    assert done.stdout.strip() == "done"


def test_the_card_is_assembled_from_saved_parts_and_missing_arms_are_labelled(feature, monkeypatch, tmp_path):
    pytest.importorskip("hawk")
    monkeypatch.delenv("FEATURE_CARD_FORCE", raising=False)
    row = feature.ROWS["scatter"]
    monkeypatch.setitem(row, "arms", ("np_add_at",))
    monkeypatch.setitem(row, "ref", "np_add_at")
    parts = ["--parts", str(tmp_path / "parts")]
    assert feature.main(["--smoke", "--rows", "scatter", "--out-dir", str(tmp_path / "a"), *parts]) == 0
    monkeypatch.setitem(row, "arms", ("np_add_at", "keyed"))
    monkeypatch.setattr(feature, "_run_arm_process", lambda *a: pytest.fail("an assemble run starts no process"))
    assert feature.main(["--smoke", "--rows", "scatter", "--out-dir", str(tmp_path / "b"), "--assemble", *parts]) == 0
    arms = json.loads(next((tmp_path / "b").glob("feature_card_*.json")).read_text())["rows"][0]["arms"]
    assert arms["np_add_at"]["status"] == "ok" and arms["keyed"]["status"] == "not_run: step time limit"


def test_a_warm_up_is_stopped_at_the_budget_not_labelled_afterwards(tmp_path):
    code = (f"import sys,time; sys.path.insert(0, {str(_BENCH)!r}); sys.path.insert(0, {str(_BENCH / 'perf_card')!r})\n"
            "import feature_card as f\nf.RUN_BUDGET_S = 1.0\nf._time(lambda: time.sleep(30), False, False)\nprint('NOT STOPPED')")
    t0 = __import__("time").time()
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=25)
    assert __import__("time").time() - t0 < 20 and "NOT STOPPED" not in done.stdout
    rec = json.loads(done.stdout.split("FEATURE_ARM ", 1)[1])
    assert rec["status"] == "not_run: budget" and "warm-up stopped at 1 s" in rec["detail"]


def test_the_budget_labels_name_their_source():
    pc = _load("perf_card", _BENCH / "perf_card" / "perf_card.py")
    base = {"status": "not_run: budget", "n": 1_000_000, "distribution": "uniform", "arm": "cupy_masked"}
    predicted = pc._row_of({**base, "predicted_from": 1_000_000, "predicted_dist": "spread"}, None, 1106, 1.0)["rule"]
    assert predicted == "> 120 s per run, predicted from spread N=1,000,000"
    assert pc._row_of({**base, "stopped": True, "warm_s": 120.0}, None, 1106, 1.0)["rule"] == \
        "> 120 s per run (warm-up stopped at 120 s)"


def test_eff_sol_is_derived_from_the_raw_fields_and_a_better_count_rescores_it(cc):
    assert cc.derive_eff_sol(0.5e9, 1.0e12, 1106)["eff_sol"] == pytest.approx(0.553)
    over = cc.derive_eff_sol(1.0e9, 1.0e12, 1106)
    assert over["eff_sol"] is None and over["eff_sol_note"] == "pending (reference count)"
    card = {"device_fma_per_s": 1.0e12, "device_fma_per_s_float32": 2.0e12,
            "results": [{"status": "ok", "arm": "eagle_graph", "distribution": "spread", "attempt_samples_per_s": 1.0e9}]}
    assert cc.rescore_card(card, ops=1106)["results"][0]["eff_sol_note"] == "pending (reference count)"
    assert cc.rescore_card(card, ops=900)["results"][0]["eff_sol"] == pytest.approx(0.9)
