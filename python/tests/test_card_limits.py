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


def test_a_compiling_child_gets_the_compile_limit_and_is_named_by_its_phase(cc, monkeypatch, tmp_path):
    """A child in its "compile" phase outlives the build limit, is killed at COMPILE_S, and the Stalled names the phase.
    RED: drop "compile" from run_watched's limits and the child dies at the 1 s build limit (elapsed < 2 s)."""
    monkeypatch.setattr(cc, "BUILD_S", 1.0)
    monkeypatch.setattr(cc, "COMPILE_S", 4.0)
    code = f"import sys,time; sys.path.insert(0, {str(_BENCH)!r}); import _card_common as c\nc.heartbeat('compile'); time.sleep(60)"
    t0 = __import__("time").monotonic()
    with pytest.raises(cc.Stalled) as err:
        cc.run_watched([sys.executable, "-c", code], timeout=60, progress=tmp_path / "alive")
    assert err.value.phase == "compile" and __import__("time").monotonic() - t0 > 3.0


def test_a_compile_over_its_limit_is_a_labelled_row_not_a_failure(monkeypatch, tmp_path):
    """The timing parent turns a compile-phase stall into the COMPILE_LIMIT row (a build stall stays "failed: no progress")."""
    pc = _load("perf_card", _BENCH / "perf_card" / "perf_card.py")

    def stall(phase):
        def run_probe(*a, **k):
            raise pc.cc.Stalled(phase, "no progress")
        return run_probe

    monkeypatch.setattr(pc, "_PARTS", None)
    monkeypatch.setattr(pc, "_ASSEMBLE", False)
    monkeypatch.setattr(pc.cc, "deadline_passed", lambda: False)
    monkeypatch.setattr(pc.cc, "run_probe", stall("compile"))
    assert pc._time_cell_arm("torch_compiled", 1000, "spread", "bus", 3, tmp_path)["status"] == pc.COMPILE_LIMIT
    assert pc.COMPILE_LIMIT == "not_run: compile > 10 min"
    monkeypatch.setattr(pc.cc, "run_probe", stall("build"))
    assert pc._time_cell_arm("torch_compiled", 1000, "spread", "bus", 3, tmp_path)["status"] == "failed: no progress"


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
    assert arms["np_add_at"]["status"] == "ok" and arms["keyed"]["status"] == "not_run: session time limit"


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


def test_a_deadline_that_passes_drops_only_the_tail_and_the_card_assembles(feature, monkeypatch, tmp_path):
    """After the first arm the session deadline passes: the next arms are ``not_run: session time limit``, the card is written."""
    import os
    import time

    pytest.importorskip("hawk")
    monkeypatch.delenv("FEATURE_CARD_FORCE", raising=False)
    row = feature.ROWS["scatter"]
    monkeypatch.setitem(row, "arms", ("np_add_at",))
    monkeypatch.setitem(row, "ref", "np_add_at")
    monkeypatch.setitem(row, "dtypes", ("float64", "float32"))
    real = feature._run_arm_process

    def run_then_expire(*args):
        rec = real(*args)
        monkeypatch.setenv("CARD_DEADLINE", str(time.time() - 1))
        return rec

    monkeypatch.setattr(feature, "_run_arm_process", run_then_expire)
    assert feature.main(["--smoke", "--rows", "scatter", "--out-dir", str(tmp_path)]) == 0
    rows = json.loads(next(tmp_path.glob("feature_card_*.json")).read_text())["rows"]
    assert [r["arms"]["np_add_at"]["status"] for r in rows] == ["ok", "not_run: session time limit"]
    os.environ.pop("CARD_DEADLINE", None)


def test_the_priority_order_runs_eagle_first_the_slow_arms_last_and_the_headline_cell_first():
    pc = _load("perf_card", _BENCH / "perf_card" / "perf_card.py")
    order = pc.ARM_PRIORITY
    assert all(order.index(a) < order.index("warp_kernel") for a in order if a.startswith("eagle_graph") or a == "eagle_simulate")
    assert order.index("jax_vmap") < order.index("cupy_masked") and order.index("torch_compiled") < order.index("torch_graphed")
    assert pc._ordered_cells(list(pc.CELLS))[0] == (10_000_000, "spread")
    assert pc._ordered_cells([(1_000_000, "uniform"), (1_000_000, "spread")])[0] == (1_000_000, "spread")


def test_the_cpu_card_resumes_from_its_saved_parts(monkeypatch, tmp_path):
    """Kill the CPU card after one (arm, cell), run it again: that part is reused and the card completes."""
    cpu = _load("cpu_card", _BENCH / "perf_card" / "cpu_card.py")
    real, started = cpu._time_arm, []

    def die_on_second(*args):
        started.append(args)
        if len(started) == 2:
            raise KeyboardInterrupt
        return real(*args)

    parts = ["--parts", str(tmp_path / "parts")]
    cli = ["--ns", "1000", "2000", "--arms", "numpy_masked", "--distributions", "spread", "--reps", "3", "--out-dir", str(tmp_path / "out")]
    monkeypatch.setattr(cpu, "_time_arm", die_on_second)
    with pytest.raises(KeyboardInterrupt):
        cpu.main([*cli, *parts])
    assert len(list((tmp_path / "parts").glob("*.json"))) == 1
    started.clear()
    monkeypatch.setattr(cpu, "_time_arm", lambda *a: started.append(a) or real(*a))
    assert cpu.main([*cli, *parts]) == 0
    assert [a[1] for a in started] == [2000]            # N = 1000 came from its saved part
    assert list((tmp_path / "out").glob("cpu_card_*.json"))


def test_a_worker_that_does_not_reply_in_time_is_stopped(tmp_path):
    """A CPU card worker past its call's time limit is killed and the call raises ``TimeoutError``, so an over-budget
    warm-up never holds the card (the CPU warm-up is stopped like the GPU one).

    RED: drop the ``timeout`` handling in ``LineWorker.call`` and this hangs for the sleep, then fails."""
    import time

    cc = _load("eagle_card_common_limits", _BENCH / "_card_common.py")
    script = tmp_path / "slow_worker.py"
    script.write_text("import sys, time\nfor line in sys.stdin:\n    time.sleep(60)\n")
    worker = cc.LineWorker(script, [], None, "slow")
    started = time.perf_counter()
    with pytest.raises(TimeoutError, match="within 1 s"):
        worker.call({"op": "setup"}, timeout=1.0)
    assert time.perf_counter() - started < 10 and worker.p.poll() is not None
