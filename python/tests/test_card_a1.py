# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Amendment A1 of the card ruling, forced on a real device: an arm that runs out of memory, one whose warm-up is
over the per-run budget and one whose process aborts each cost their own row, never the card; the card still ends
with exit code 0, every ok row has ``eff_sol <= 1`` and the other arms are timed."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import pytest

_SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "benchmarks" / "perf_card" / "perf_card.py"


@pytest.mark.gpu
def test_a_failing_arm_is_a_row_and_the_card_completes(tmp_path):
    pytest.importorskip("cupy")
    env = {**os.environ, "EAGLE_CARD_FAULT": "oom:cupy_masked,slow:warp_kernel,abort:eagle_graph_persist"}
    done = subprocess.run([sys.executable, str(_SCRIPT), "--smoke", "--out-dir", str(tmp_path),
                           "--arms", "eagle_graph", "eagle_graph_persist", "cupy_masked", "warp_kernel"],
                          env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stderr[-2000:]
    card = json.loads(next(tmp_path.glob("card_*.json")).read_text())
    status = {r["arm"]: r["status"] for r in card["results"]}
    assert status["cupy_masked"] == "out_of_memory"
    assert status["warp_kernel"] == "not_run: budget"
    assert status["eagle_graph_persist"].startswith("failed:")
    assert status["eagle_graph"] == "ok"
    row = next(r for r in card["results"] if r["arm"] == "warp_kernel")
    assert "> 120 s per run" in row["rule"]
    ok = [r for r in card["results"] if r["status"] == "ok"]
    assert ok and all((r["eff_sol"] is None and r.get("eff_sol_note")) or (0 < r["eff_sol"] <= 1.0) for r in ok) and all(1 <= r["rounds"] <= 9 for r in ok)
    assert card["tail_floor_ms"]["spread"] is not None
