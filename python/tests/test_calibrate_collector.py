# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The calibration collector (``benchmarks/reproduce/calibrate.py``): its quick pass
writes a CSV that follows the raw-data contract. The GPU twin runs the same pass on
the visible device."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("hawk")

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "reproduce" / "calibrate.py"
HEADER_KEYS = ("schema", "collector_md5", "date_utc", "target", "eagle_version", "eagle_sha",
               "hawk_version", "hawk_sha", "compiler", "opt_level", "host_profile", "cpu_model", "isa",
               "threads", "omp_proc_bind", "openmp_runtime", "compile_s", "total_wall_s", "rows")
GPU_HEADER_KEYS = ("gpu_name", "driver", "cuda_runtime", "smi_before", "smi_after")
COLUMNS = ("case_id", "group", "kind", "target", "kernel_src_md5", "codegen_md5", "n", "K",
           "fused_steps", "steps", "m", "counts", "cost_b0", "cost_b1", "cost_b2", "cost_b3",
           "fused_vectorised", "chains", "held_out", "rep", "warm", "timer", "wall_s", "status")
QUICK_REPS = 3          # --quick: the first (cold) repetition plus two


def run_quick(target: str, out: Path, env_extra: dict) -> tuple:
    env = {**os.environ, **env_extra}
    done = subprocess.run([sys.executable, str(SCRIPT), "--target", target, "--quick", "--out", str(out)],
                          capture_output=True, text=True, env=env, timeout=900)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    (csv_path,) = out.glob("calibration-*.csv")
    lines = csv_path.read_text().splitlines()
    head = dict(ln[2:].split("=", 1) for ln in lines if ln.startswith("# "))
    rows = list(csv.DictReader(ln for ln in lines if not ln.startswith("#")))
    return done.stdout, head, rows


def check_contract(head: dict, rows: list, target: str) -> None:
    for key in HEADER_KEYS + (GPU_HEADER_KEYS if target == "gpu" else ()):
        assert head.get(key), f"header key {key!r} missing or empty"
    assert head["schema"] == "2" and head["target"] == target
    assert int(head["rows"]) == len(rows) > 0
    assert tuple(rows[0]) == COLUMNS
    cases: dict = {}
    for row in rows:
        assert row["group"] in "ABCDE" and row["kind"] in ("unit", "fused", "split", "sort", "launch")
        assert row["target"] == target and row["status"] == "ok"
        assert row["kernel_src_md5"] and row["codegen_md5"], "md5 fields are never empty"
        assert row["timer"] == ("event" if target == "gpu" and row["kind"] != "sort" else "steady_clock")
        assert float(row["wall_s"]) > 0.0
        assert row["held_out"] == ("1" if row["group"] == "E" else "0")
        cases.setdefault(row["case_id"], []).append(row)
    for case_id, reps in cases.items():
        assert [r["rep"] for r in reps] == [str(i) for i in range(QUICK_REPS)], case_id
        assert [r["warm"] for r in reps] == ["0"] + ["1"] * (QUICK_REPS - 1), case_id
    groups = {r["group"] for r in rows}
    assert groups >= set("ABCE") and ("D" in groups) == (target == "gpu")
    check_cost_cells(rows)
    if target == "cpu":
        check_chains(head, rows)


def check_cost_cells(rows: list) -> None:
    """Every cost cell is 6 counts, 4 chain lengths and hawk's vectorised flag."""
    for row in rows:
        for key in ("cost_b0", "cost_b1", "cost_b2", "cost_b3"):
            if row[key]:
                cell = [float(x) for x in row[key].split(";")]
                assert len(cell) == 11 and cell[-1] in (0.0, 1.0), (row["case_id"], row[key])


def check_chains(head: dict, rows: list) -> None:
    """Group A holds the 1-chain / 4-chain pair at each m, with truthful chains and flags."""
    from importlib import import_module

    assert head["host_profile"] in import_module("hawk.compile.toolchain").HOST_PROFILES
    unit = {(r["chains"], r["m"]): [float(x) for x in r["cost_b0"].split(";")]
            for r in rows if r["group"] == "A" and r["kind"] == "unit" and r["chains"] != "0"}
    ms = {m for _, m in unit}
    assert ms and all(("1", m) in unit and ("4", m) in unit for m in ms)
    for m in ms:
        one, four = unit[("1", m)], unit[("4", m)]
        assert abs(one[1] - four[1]) <= 8 and one[7] > 3 * (four[7] - 2), (m, one, four)  # same ops, shorter path
    long_one = [unit[("1", m)] for m in ms if int(m) > 16]
    assert long_one and all(c[10] == 0.0 for c in long_one), "a rolled loop is not vectorised"
    contrast = [r for r in rows if r["group"] == "E" and r["kind"] == "split"]
    cheap, heavy = ([float(x) for x in contrast[-1][k].split(";")] for k in ("cost_b0", "cost_b1"))
    assert cheap[10] == 1.0 and heavy[10] == 0.0, "the contrast pair: cheap vectorises, heavy loop does not"


def test_quick_pass_on_the_host_writes_a_csv_that_follows_the_contract(tmp_path):
    out_text, head, rows = run_quick("cpu", tmp_path, {"CUDA_VISIBLE_DEVICES": ""})
    check_contract(head, rows, "cpu")
    assert "total wall" in out_text and "group A" in out_text


@pytest.mark.gpu
def test_device_quick_pass_writes_a_csv_that_follows_the_contract(tmp_path):
    pytest.importorskip("cupy")
    _, head, rows = run_quick("gpu", tmp_path, {})
    check_contract(head, rows, "gpu")
