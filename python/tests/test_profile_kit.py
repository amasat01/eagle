# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The profiling kit (``benchmarks/reproduce/gpu_profile.py`` and ``profile.sh``): the decision
probe on the host, the probe and one Nsight Systems capture on the device (Nsight Compute
is skipped cleanly where the GPU is too old for it), and the script's error paths -- a
failing stage, a timeout, a kill followed by a resume, a missing ncu -- each of which still
leaves a tarball with an accurate ``status.txt``."""

from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("hawk")

REPRO = Path(__file__).resolve().parents[2] / "benchmarks" / "reproduce"
PROFILE_PY = REPRO / "gpu_profile.py"
PROFILE_SH = REPRO / "profile.sh"
PROBE_KEYS = ("target:", "device.name:", "auto.persist_entry:", "auto.entry_reports:", "auto.device_capacity():",
              "auto.fast_mode_for(n=1000):", "auto.fast_mode_for(n=1000000):",
              "auto.run_mode(n=1000):", "persist.run_mode(n=1000000):")
DEVICE_KEYS = ("device.multiProcessorCount:", "auto.num_regs:", "auto.resident_threads_per_sm:",
               "auto.capacity (sms * resident):", "auto.latency_regime_capacity:",
               "persist.lane_utilisation(n=1000000):")


def _kit_module():
    spec = importlib.util.spec_from_file_location("gpu_profile_under_test", PROFILE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_py(*args, env=None, timeout=600):
    return subprocess.run([sys.executable, str(PROFILE_PY), *args], capture_output=True, text=True,
                          env={**os.environ, **(env or {})}, timeout=timeout, cwd=str(REPRO.parents[1]))


def test_probe_on_the_host(tmp_path):
    done = run_py("--probe-only", "--target", "cpu", "--out", str(tmp_path))
    assert done.returncode == 0, done.stderr[-2000:]
    text = (tmp_path / "probe.txt").read_text()
    for key in PROBE_KEYS:
        assert key in text, key
    assert "auto.device_capacity(): 0" in text  # a host plan has no device capacity


def _one_gpu_env():
    count = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU ")
    if count > 1 and not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
        pytest.skip("several GPUs and CUDA_VISIBLE_DEVICES does not name one")
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


@pytest.mark.gpu
def test_probe_and_one_nsys_capture_on_the_device(tmp_path):
    env = _one_gpu_env()
    done = run_py("--probe-only", "--out", str(tmp_path), env=env)
    assert done.returncode == 0, done.stderr[-2000:]
    text = (tmp_path / "probe.txt").read_text()
    for key in PROBE_KEYS + DEVICE_KEYS:
        assert key in text, key
    nsys = shutil.which("nsys")
    if nsys is None:
        pytest.skip("nsys not installed")
    done = run_py("--stage", "nsys-quick", "--nsys", nsys, "--out", str(tmp_path), env=env)
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    summary = (tmp_path / "nsys_quick.txt").read_text()
    assert "eagle_graph_auto" in summary and "oscillator_step" in summary
    ncu = shutil.which("ncu")
    done = run_py("--stage", "ncu-quick", "--ncu", ncu or "ncu", "--out", str(tmp_path), env=env)
    cc = float(subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                               "--query-gpu=compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.split()[0])
    if cc < 7.0:
        assert done.returncode == 3 and "SKIP: compute capability" in done.stdout
    else:
        assert done.returncode in (0, 3)


# --- profile.sh error paths ---------------------------------------------------------------
@pytest.fixture
def kit(tmp_path):
    """A runner of ``profile.sh --quick`` over a throwaway env script and output directory."""
    env_file = tmp_path / "env.sh"
    env_file.write_text(f'export PATH="{Path(sys.executable).parent}:$PATH"\n')
    base = {**_one_gpu_env(), "PROFILE_MIN_DISK_GB": "1"}

    def start(extra_env=None, *flags):
        env = {**os.environ, **base, **(extra_env or {})}
        return subprocess.Popen(["bash", str(PROFILE_SH), "--env", str(env_file), "--out", str(tmp_path / "out"),
                                 "--quick", *flags], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, env=env)

    def run(extra_env=None, *flags):
        proc = start(extra_env, *flags)
        out, _ = proc.communicate(timeout=600)
        return proc.returncode, out

    def results():
        out = tmp_path / "out"
        (tarball,) = out.glob("profile-*.tgz")
        with tarfile.open(tarball) as tar:
            status = tar.extractfile("status.txt").read().decode()
            names = tar.getnames()
        assert status == (out / "status.txt").read_text()
        return status, names

    run.start, run.results, run.out = start, results, tmp_path / "out"
    return run


@pytest.mark.gpu
def test_a_failing_stage_does_not_stop_the_next(kit):
    rc, out = kit({"PROFILE_TEST_FAIL": "nsys-quick"})
    status, names = kit.results()
    assert rc == 0
    assert "probe: ok" in status
    assert "nsys-quick: failed: exit 1" in status and "forced failure" in status
    assert "ncu-quick: " in status and "not reached" not in status  # it was reached and decided
    assert "data/probe.txt" in names
    assert out.strip().splitlines()[-1].startswith("[repro] results:")


@pytest.mark.gpu
def test_a_stage_over_its_cap_is_failed_and_the_tarball_still_lands(kit):
    rc, out = kit({"PROFILE_TEST_CAP_S": "1"})
    status, _ = kit.results()
    assert rc == 0
    assert "probe: failed: exceeded its 1 s time cap" in status
    assert out.strip().splitlines()[-1].startswith("[repro] results:")


@pytest.mark.gpu
def test_a_killed_run_resumes_from_its_markers(kit):
    proc = kit.start()
    deadline = time.time() + 300
    probe_ok = Path(kit.out) / ".stages" / "probe.ok"
    while not probe_ok.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert probe_ok.exists()
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 143
    status, _ = kit.results()
    assert "probe: ok" in status
    assert out.strip().splitlines()[-1].startswith("[repro] results:")
    for stale in Path(kit.out).glob("profile-*.tgz"):
        stale.unlink()
    rc, out = kit()
    status, _ = kit.results()
    assert rc == 0
    assert "probe: ok (finished in an earlier run)" in status
    assert "nsys-quick: ok" in status


@pytest.mark.gpu
def test_a_missing_ncu_is_a_skipped_stage(kit, tmp_path):
    shim = tmp_path / "shim"
    shim.mkdir()
    nsys = shutil.which("nsys")
    if nsys:
        (shim / "nsys").symlink_to(nsys)
    keep = [d for d in os.environ["PATH"].split(os.pathsep) if not (Path(d) / "ncu").exists()
            and not (Path(d) / "nsys").exists()]
    rc, _ = kit({"PATH": os.pathsep.join([str(shim), *keep]), "PROFILE_TOOLS_PATH_ONLY": "1"})
    status, _ = kit.results()
    assert rc == 0
    assert "ncu-quick: skipped: ncu not found" in status
    assert "probe: ok" in status


# --- the stages that answer the launch-sizing questions: plans, parsing, the override -----
NEW_STAGES = ("compare", "crossover", "band", "cadence-sweep", "ncu-ratio", "ncu-tsweep", "ncu-band")


def test_the_new_stages_are_in_the_full_and_the_quick_plan():
    for flags, suffix in (((), ""), (("--quick",), "-quick")):
        done = subprocess.run(["bash", str(PROFILE_SH), "--dry-run", *flags], capture_output=True, text=True,
                              env={**os.environ, "REPRO_TARGET": "devbox"})
        assert done.returncode == 0, done.stderr[-1500:]
        for stage in NEW_STAGES:
            assert f"stage {stage}{suffix} (cap" in done.stderr, (stage, done.stderr[-800:])
        assert "deadline=50 min" in done.stderr
        assert "persist-sweep" not in done.stderr and "fused-sweep" not in done.stderr


def test_every_stage_profile_sh_plans_is_one_gpu_profile_py_accepts():
    kit = _kit_module()
    planned = subprocess.run(["bash", str(PROFILE_SH), "--dry-run"], capture_output=True, text=True,
                             env={**os.environ, "REPRO_TARGET": "devbox"}).stderr
    quick = subprocess.run(["bash", str(PROFILE_SH), "--dry-run", "--quick"], capture_output=True, text=True,
                           env={**os.environ, "REPRO_TARGET": "devbox"}).stderr
    names = {ln.split()[2] for ln in (planned + quick).splitlines() if " stage " in f" {ln} " and "(cap" in ln}
    assert len(names) >= 10 and "compare" in names and "compare-quick" in names and names <= set(kit.STAGES), names - set(kit.STAGES)


def _plan(*flags):
    done = subprocess.run(["bash", str(PROFILE_SH), "--dry-run", *flags], capture_output=True, text=True,
                          env={**os.environ, "REPRO_TARGET": "devbox"})
    return done.returncode, done.stderr


def test_only_selects_stages_by_name_or_prefix_and_refuses_an_unknown_one():
    rc, err = _plan("--only", "compare,probe")
    assert rc == 0 and [ln.split()[2] for ln in err.splitlines() if "(cap" in ln] == ["probe", "compare"]
    rc, err = _plan("--quick", "--only", "compare")
    assert rc == 0 and "stage compare-quick (cap" in err and "probe" not in err
    rc, err = _plan("--only", "ncu")
    assert rc == 0 and sorted(ln.split()[2] for ln in err.splitlines() if "(cap" in ln) == [
        "ncu-band", "ncu-ratio", "ncu-spread-1e4", "ncu-spread-1e6", "ncu-tsweep", "ncu-uniform-1e6"]
    rc, err = _plan("--only", "compar")
    assert rc != 0 and "matches no stage" in err


def test_the_cadence_sweep_cells_cover_the_asked_grid():
    kit = _kit_module()
    cadence = kit.nsys_cells("cadence-sweep")
    assert {kit.parse_spec(c[0])[1]["every"] for c in cadence} == {"4", "8", "16", "32"}
    assert {c[3] for c in cadence} == {"uniform", "spread"} and {c[1:3] for c in cadence} == {(10**6, 1000)}
    assert sum(kit.parse_spec(c[0])[1].get("eager") == "1" for c in cadence) == len(cadence) // 2


def test_compare_covers_every_case_of_the_matrix():
    kit = _kit_module()
    cells = kit.compare_cells("compare")
    osc = [c for c in cells if c[0] == "osc"]
    assert len(cells) == 4 * (1 + 2 * 4) + 2 * 4 and len([c for c in cells if c[0] == "rk78"]) == 4
    assert {c[1] for c in cells} == {10**3, 10**4, 10**5, 10**6} and cells[0][1] == 10**6  # large N first
    assert {(c[2], c[3]) for c in osc} == {(s, d) for s in (100, 1000)
                                           for d in ("uniform", "spread", "spread:10", "spread:1000")}
    assert [c[:2] for c in kit.compare_cells("compare", kernels=("rk78",), ns=(1000,))] == [("rk78", 1000)]
    assert len(kit.compare_cells("compare-quick")) == 3
    assert [c for c in kit.compare_cells("compare-quick", kernels=("osc", "rk78"))] == [
        ("osc", 100_000, 100, "spread"), ("rk78", 10_000, 0, "orbits")]
    labels = [lab for lab, _ in kit.compare_arms("osc")]
    assert labels == ["auto", "fused_one", "persist@auto", *(f"persist@{t}" for t in (384, 640, 896, 1152, 1536, 2048)),
                      "band@auto", "band@8", "band@16", "warp"]
    assert [lab for lab, _ in kit.compare_arms("rk78")] == [
        "auto", "fused_one", "persist@auto", "persist@256", "persist@384", "band@auto", "band@8", "band@16", "warp"]
    specs = dict(kit.compare_arms("osc"))
    assert specs["persist@640"] == "osc_persist@tps=640" and specs["band@8"] == "osc_band@every=8"
    assert specs["warp"] == "warp_kernel" and dict(kit.compare_arms("rk78"))["warp"] == "rk78_warp"


def test_the_fp32_cells_and_arms_cover_the_asked_matrix_and_are_labelled():
    kit = _kit_module()
    cells = [c for c in kit.compare_cells("compare") if c[0] == "osc32"]
    assert {(c[1], c[2], c[3]) for c in cells} == {(n, s, d) for n in (10**5, 10**6) for s in (100, 1000)
                                                   for d in ("spread", "uniform")} and len(cells) == 8
    assert [c for c in kit.compare_cells("compare-quick", kernels=("osc32",))] == [("osc32", 100_000, 100, "spread")]
    arms = kit.compare_arms("osc32")
    assert [lab for lab, _ in arms] == ["auto", "fused_one", "persist@auto", "persist@640", "persist@1024",
                                        "persist@1536", "persist@2048", "band@auto"]  # no Warp arm
    assert all(kit.precision_of(spec) == "fp32" for _, spec in arms)
    assert all(kit.precision_of(spec) == "fp64" for _, spec in kit.compare_arms("osc"))
    assert kit.precision_of("osc32") == "fp32" and kit.precision_of("osc") == "fp64"
    tsweep = [c for c in kit.ncu_captures("ncu-tsweep") if kit.precision_of(c[1]) == "fp32"]
    assert [kit.parse_spec(c[1])[1]["tps"] for c in tsweep] == ["384", "640", "896", "1152", "1536", "2048"]
    assert all(c[3] == (10**6, 1000, "spread") and "sm__inst_executed_pipe_fma.sum" in c[4] for c in tsweep)
    text, csv = kit.render_compare([dict(kernel="osc32", n=10**5, steps=100, dist="spread", reference="x", arms=[
        dict(label="auto", spec="eagle_graph_auto@fp=32", status="ok", note="", median=1.0, best=1.0, mode="persist",
             launched="-", util=None, bitwise=True, diff=0.0, sm_mhz=None, modes="", fields={})])])
    assert "precision=fp32" in text and csv.splitlines()[0].endswith("launch_fields,precision,note")
    assert csv.splitlines()[1].split(",")[-2] == "fp32"


def test_crossover_band_and_tsweep_cover_the_asked_cases():
    kit = _kit_module()
    cross = kit.compare_cells("crossover")
    assert sorted(c[1] for c in cross) == [30_000, 100_000, 300_000] and {c[2:] for c in cross} == {(100, "spread")}
    assert [l for l, _ in kit.stage_arms("crossover", "osc")] == ["auto", "fused_one", "persist@auto"]
    band = kit.compare_cells("band")
    assert {(c[1], c[3]) for c in band} == {(n, d) for n in (10**5, 10**6) for d in ("uniform", "spread")}
    assert [l for l, _ in kit.stage_arms("band", "osc")] == ["auto", "band@auto", "band@16", "band@32", "band@64", "band@128"]
    assert len(kit.compare_cells("crossover-quick")) == 1 and len(kit.compare_cells("band-quick")) == 1
    tsweep = kit.ncu_captures("ncu-tsweep")
    osc = {kit.parse_spec(c[1])[1]["tps"] for c in tsweep if c[1].startswith("osc")}
    assert osc == {"384", "640", "896", "1152", "1536", "2048"} and len(tsweep) == 2 * 6 + 2 + 6
    assert all(c[3][0] == 10**6 and c[4] in (kit.RATIO_METRICS, kit.RATIO_METRICS_FP32) for c in tsweep)
    assert len(kit.ncu_captures("ncu-tsweep-quick")) == 1


def test_lane_efficiency_comes_from_the_ncu_fp64_count_and_the_workers_step_accounting():
    kit = _kit_module()
    raw = '"ID","Kernel Name","sm__inst_executed_pipe_fp64.sum"\n"","","inst"\n"0","k","44000"\n'
    line = "worker arm=osc_persist n=1 steps=1 dist=spread sample_steps=32000 ops_per_step=44 ops_source=card-count"
    text = kit.lane_efficiency_lines(raw, line)
    assert "= 1.000" in text and "card-count" in text and "(ideal 44)" in text and "= 44.0" in text
    assert "unavailable" in kit.lane_efficiency_lines(raw, "worker arm=x")
    assert "unavailable" in kit.lane_efficiency_lines('"ID","Kernel Name"\n"0","k"\n', line)
    assert kit._ncu_values("", ("a",)) == {}


def test_launch_fields_are_read_by_exact_name_where_an_eagle_exposes_them():
    kit = _kit_module()

    class Report:
        T, text_bytes, hidden = 640, 1234, 7

    class Runner:
        every = 16
        k = "not-a-number-kept-as-text"

        @property
        def c(self):
            raise RuntimeError("older eagle")

    got = kit.read_fields(Runner(), Report())
    assert got == {"T": 640, "text_bytes": 1234, "every": 16, "k": "not-a-number-kept-as-text"}
    assert kit.read_fields(object()) == {}


def test_the_band_fit_recovers_the_compaction_cost_and_the_crossover_names_the_settled_entry():
    kit = _kit_module()
    steps, launch, group = 1000, 0.5, 0.05  # ms per K-step launch, per compaction group
    arms = [_rec("band@auto", 1.0, "band")]
    arms[0]["fields"] = {"every": 16}
    for e in (16, 32, 64, 128):
        arms.append(_rec(f"band@{e}", steps / kit.BAND_STEPS_PER_LAUNCH * launch + steps / e * group, "band"))
    cell = dict(kernel="osc", n=10**6, steps=steps, dist="uniform", reference="x", arms=arms)
    text = "\n".join(kit.band_fit([cell]))
    assert "R = C / launch = 0.10" in text and "every=16" in text
    assert "R is not measurable" in "\n".join(kit.band_fit([dict(cell, arms=[_rec(f"band@{e}", 5.0, "band")
                                                                              for e in (16, 32, 64, 128)])]))
    auto = _rec("auto", 1.0, "persist")
    auto["modes"] = "fused_one>fused_one>persist>persist>persist"
    cross = dict(kernel="osc", n=10**5, steps=100, dist="spread", reference="x", arms=[
        auto, _rec("fused_one", 2.0, "fused_one"), _rec("persist@auto", 1.0)])
    line = kit.crossover_lines([cross])[0]
    assert "N=1e5" in line and "faster: persist, 2.00x" in line and "settled on persist after 3 runs" in line


def _inputs_and_outputs(n=64):
    import numpy as np

    rng = np.random.default_rng(1)
    inp = {"nstop": np.full(n, 10.0)}
    ref = {"x": rng.uniform(-1, 1, n), "v": rng.uniform(-1, 1, n), "k": np.full(n, 10.0)}
    return inp, ref


def test_the_guard_flags_a_wrong_arm_and_demands_bits_only_of_the_same_entry():
    kit = _kit_module()
    kit.load_card()
    inp, ref = _inputs_and_outputs()
    same = {k: v.copy() for k, v in ref.items()}
    assert kit.judge("osc", same, ref, inp, True)[:2] == (True, True)
    near = {**same, "x": ref["x"] + 1e-13}  # inside the card's tolerance
    ok, bitwise, diff, _ = kit.judge("osc", near, ref, inp, False)
    assert ok and not bitwise and diff > 0
    ok, _, _, why = kit.judge("osc", near, ref, inp, True)  # the same entry must agree bit for bit
    assert not ok and "bit for bit" in why
    far = {**same, "x": ref["x"] + 1e-6}
    ok, _, _, why = kit.judge("osc", far, ref, inp, False)
    assert not ok and "differs from the reference" in why
    ok, _, _, why = kit.judge("osc", {**same, "k": ref["k"] - 1}, ref, inp, False)
    assert not ok and "step counts" in why
    ok, _, _, why = kit.judge("osc", {**same, "v": ref["v"] * float("nan")}, ref, inp, False)
    assert not ok and "non-finite" in why
    # a tuple download (the oscillator card's) is read like the dict
    assert kit.judge("osc", (same["x"], same["v"], same["k"]), ref, inp, True)[0]


def _rec(label, median, mode="persist", status="ok", note=""):
    return dict(label=label, spec=label, status=status, note=note, median=median, best=median, mode=mode,
                launched="-", util=None, bitwise=True, diff=0.0)


def test_the_summary_lists_where_auto_trails_warp_or_misses_the_fastest_path():
    kit = _kit_module()
    fast = dict(kernel="osc", n=10**6, steps=1000, dist="spread", reference="x", arms=[
        _rec("auto", 5.0), _rec("persist@640", 4.0), _rec("band@8", 9.0, "band"), _rec("warp", 4.5, "-")])
    ok = dict(kernel="osc", n=10**3, steps=100, dist="uniform", reference="x", arms=[
        _rec("auto", 1.0, "fused_one"), _rec("fused_one", 1.0, "fused_one"), _rec("warp", 2.0, "-")])
    wrong = dict(kernel="rk78", n=10**4, steps=0, dist="orbits", reference="x", arms=[
        _rec("auto", 100.0), _rec("persist@256", None, status="wrong", note="3 samples exceed the truth bound"),
        _rec("persist@384", None, status="skipped", note="384 threads per SM is above the residency limit"),
        _rec("band@8", None, status="unavailable", note="TypeError: x"), _rec("warp", 150.0, "-")])
    text, csv = kit.render_compare([fast, ok, wrong], unreached=[("osc", 10**3, 1000, "uniform")])
    slower = text.split("auto is slower than Warp")[1].split("--")[0]
    assert "osc N=1e6 S=1000 spread" in slower and "1.11x" in slower and "N=1e3" not in slower
    missed = text.split("did not pick the fastest")[1].split("--")[0]
    assert "persist@640" in missed and "1.25x" in missed and "rk78" not in missed
    assert "WRONG" in text and "skip" in text and "n/a" in text
    assert "3 samples exceed the truth bound" in text and "cells not reached (1" in text
    rows = csv.splitlines()
    assert rows[0].startswith("kernel,n,steps,dist,arm,status,median_ms") and len(rows) == 1 + 4 + 3 + 5
    assert any(r.startswith("rk78,10000,0,orbits,persist@256,wrong,") for r in rows)


def test_compare_refuses_the_cpu_target(tmp_path):
    done = run_py("--stage", "compare-quick", "--target", "cpu", "--out", str(tmp_path))
    assert done.returncode == 3 and "SKIP" in done.stdout


def test_ncu_captures_ask_for_the_ratio_metrics_and_the_band_kernels():
    kit = _kit_module()
    ratio = kit.ncu_captures("ncu-ratio")
    assert [c[0] for c in ratio][-1] == "rk78_persist" and len(ratio) == 5
    assert {(c[3][1], c[3][2]) for c in ratio[:4]} == {(s, d) for s in (100, 1000) for d in ("uniform", "spread")}
    assert all(c[4] == kit.RATIO_METRICS and c[3][0] == 10**6 for c in ratio)
    assert "smsp__thread_inst_executed_per_inst_executed.ratio" in kit.RATIO_METRICS
    assert any(m.startswith("sm__inst_executed_pipe_fp64") for m in kit.RATIO_METRICS)
    band = kit.ncu_captures("ncu-band")
    assert {c[1] for c in band} == {kit.NCU_BAND_EAGER} and len(band) == 4 and len(kit.ncu_captures("ncu-band-quick")) == 2
    cmd = kit.ncu_command("ncu", Path("rep"), ratio[0][2], ratio[0][1], ratio[0][3], False, ratio[0][4])
    assert cmd[cmd.index("--metrics") + 1] == ",".join(kit.RATIO_METRICS)
    assert "--metrics" not in kit.ncu_command("ncu", Path("rep"), "regex:x", "osc_persist", (1, 1, "spread"), False)


def test_the_ratio_metrics_are_read_from_ncus_raw_page_and_a_missing_one_is_named():
    kit = _kit_module()
    raw = ('"ID","Kernel Name","a.sum","b.ratio"\n"","","inst",""\n"0","k","12","31.5"\n')
    lines = kit._ncu_metric_lines(raw, ("a.sum", "b.ratio", "c.sum"))
    assert "a.sum: 12 inst" in lines and "b.ratio: 31.5" in lines and "c.sum: missing from this report" in lines
    assert "missing" in kit._ncu_metric_lines("", ("a.sum",))


def test_the_threads_per_sm_override_applies_inside_the_block_and_is_undone():
    kit = _kit_module()
    policy = pytest.importorskip("eagle._launch_policy")
    original = policy.persistent_threads_per_sm
    with kit.persist_threads(640):
        assert policy.persistent_threads_per_sm(None, None, 1, 1) == 640
    assert policy.persistent_threads_per_sm is original
    with kit.persist_threads(None):
        assert policy.persistent_threads_per_sm is original
    with pytest.raises(ZeroDivisionError), kit.persist_threads(320):
        raise ZeroDivisionError
    assert policy.persistent_threads_per_sm is original


def test_the_override_refuses_a_second_block_instead_of_racing():
    kit = _kit_module()
    pytest.importorskip("eagle._launch_policy")
    inside, release, outcome = threading.Event(), threading.Event(), {}

    def holder():
        with kit.persist_threads(320):
            inside.set()
            release.wait(30)

    thread = threading.Thread(target=holder)
    thread.start()
    assert inside.wait(30)
    try:
        with kit.persist_threads(640):
            outcome["entered"] = True
    except RuntimeError as exc:
        outcome["error"] = str(exc)
    finally:
        release.set()
        thread.join()
    assert "entered" not in outcome and "one cell at a time" in outcome["error"]
    with kit.persist_threads(960):  # free again once the first block ended
        pass


def test_an_eagle_without_the_override_hook_leaves_the_lock_free(monkeypatch):
    kit = _kit_module()
    policy = pytest.importorskip("eagle._launch_policy")
    monkeypatch.delattr(policy, "persistent_threads_per_sm", raising=False)
    for _ in range(2):  # the second refusal would be "already active" had the first kept the lock
        with pytest.raises(AttributeError), kit.persist_threads(320):
            pass


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_the_kit_scripts_are_shellcheck_clean():
    done = subprocess.run(["shellcheck", "-x", str(PROFILE_SH)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-1500:]


@pytest.mark.gpu
def test_compare_times_every_arm_and_the_override_really_changes_the_launch(tmp_path):
    done = run_py("--stage", "compare-quick", "--kernels", "osc,rk78", "--out", str(tmp_path), env=_one_gpu_env())
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    rows = [ln.split(",") for ln in (tmp_path / "compare_quick.csv").read_text().splitlines()[1:]]
    by = {(r[0], r[4]): r for r in rows}
    assert {r[0] for r in rows} == {"osc", "rk78"} and len(rows) == 13 + 9
    for key in (("osc", "auto"), ("osc", "warp"), ("osc", "fused_one"), ("rk78", "auto"), ("rk78", "warp")):
        assert by[key][5] == "ok" and float(by[key][6]) > 0, key
    assert by[("osc", "fused_one")][8] == "fused_one" and by[("osc", "persist@640")][8] == "persist"
    assert by[("osc", "persist@640")][9] == "640"  # launched = requested
    assert not any(r[5] == "wrong" for r in rows)
    text = (tmp_path / "compare_quick.txt").read_text()
    assert "cells where eagle auto is slower than Warp" in text and "did not pick the fastest eagle path" in text


@pytest.mark.gpu
def test_compare_runs_the_fp32_cells_and_labels_every_row(tmp_path):
    done = run_py("--stage", "compare-quick", "--kernels", "osc32", "--out", str(tmp_path), env=_one_gpu_env())
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    lines = (tmp_path / "compare_quick.csv").read_text().splitlines()
    rows = [ln.split(",") for ln in lines[1:]]
    assert len(rows) == 8 and {r[0] for r in rows} == {"osc32"} and {r[-2] for r in rows} == {"fp32"}
    by = {r[4]: r for r in rows}
    assert "warp" not in by and not any(r[5] == "wrong" for r in rows)
    for label in ("auto", "fused_one", "persist@auto", "persist@1024", "band@auto"):
        assert by[label][5] == "ok" and float(by[label][6]) > 0, label
    assert "precision=fp32" in (tmp_path / "compare_quick.txt").read_text()


@pytest.mark.gpu
def test_crossover_and_band_stages_run_and_record_the_clock(tmp_path):
    env = _one_gpu_env()
    for stage in ("crossover-quick", "band-quick"):
        done = run_py("--stage", stage, "--out", str(tmp_path), env=env)
        assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    cross = (tmp_path / "crossover_quick.txt").read_text()
    assert "-- crossover" in cross and "after 3 runs" in cross and "-- SM clock during the runs" in cross
    band = (tmp_path / "band_quick.txt").read_text()
    assert "-- band cost model" in band and "eagle's own band pick every=" in band
    header = (tmp_path / "band_quick.csv").read_text().splitlines()[0]
    assert "sm_mhz,modes_per_run,launch_fields,precision,note" in header


@pytest.mark.gpu
def test_compare_flags_a_planted_wrong_arm_and_does_not_time_it(tmp_path):
    done = run_py("--stage", "compare-quick", "--out", str(tmp_path),
                  env={**_one_gpu_env(), "PROFILE_TEST_CORRUPT": "persist@640"})
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    rows = {(r.split(",")[0], r.split(",")[4]): r.split(",") for r in
            (tmp_path / "compare_quick.csv").read_text().splitlines()[1:]}
    planted = rows[("osc", "persist@640")]
    assert planted[5] == "wrong" and planted[6] == "" and rows[("osc", "persist@384")][5] == "ok"
    assert "WRONG" in (tmp_path / "compare_quick.txt").read_text()


@pytest.mark.gpu
def test_the_cadence_sweep_sees_the_band_kernels_through_the_eager_run(tmp_path):
    nsys = shutil.which("nsys")
    if nsys is None:
        pytest.skip("nsys not installed")
    done = run_py("--stage", "cadence-sweep-quick", "--nsys", nsys, "--out", str(tmp_path), env=_one_gpu_env())
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    text = (tmp_path / "cadence_sweep_quick.txt").read_text()
    assert "osc_compact@every=16 " in text and "eager=1" in text and "compactPredicate" in text


@pytest.mark.gpu
def test_the_ncu_stages_are_validated_without_a_supported_gpu(tmp_path):
    nsys, ncu = shutil.which("nsys"), shutil.which("ncu")
    if nsys is None or ncu is None:
        pytest.skip("nsys and ncu are needed")
    done = run_py("--stage", "validate-ncu", "--nsys", nsys, "--ncu", ncu, "--out", str(tmp_path), env=_one_gpu_env())
    assert done.returncode == 0, done.stdout[-2500:] + done.stderr[-1500:]
    text = (tmp_path / "ncu_validation.txt").read_text()
    assert text.rstrip().endswith("OK") and "rkf78_attempt_persist" in text and "MISSING" not in text


@pytest.mark.gpu
@pytest.mark.parametrize("stage", ["ncu-ratio-quick", "ncu-band-quick", "ncu-tsweep-quick"])
def test_the_new_ncu_stages_skip_on_an_old_gpu_and_run_on_a_new_one(tmp_path, stage):
    ncu = shutil.which("ncu")
    if ncu is None:
        pytest.skip("ncu not installed")
    env = _one_gpu_env()
    done = run_py("--stage", stage, "--ncu", ncu, "--out", str(tmp_path), env=env)
    cc = float(subprocess.run(["nvidia-smi", "-i", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
                               "--query-gpu=compute_cap", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.split()[0])
    if cc < 7.0:
        assert done.returncode == 3 and "SKIP: compute capability" in done.stdout
    else:
        assert done.returncode == 0, done.stdout[-1500:]
        text = (tmp_path / f"{stage.replace('-quick', '')}.txt").read_text()
        assert ("smsp__thread_inst_executed_per_inst_executed" in text) == (not stage.startswith("ncu-band"))
