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
import re
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
@pytest.mark.interop_matrix  # needs Warp (the kit's preflight and comparison arm): the gate venv has none, the Kaggle leg installs it
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
@pytest.mark.interop_matrix  # needs Warp (the kit's preflight and comparison arm): the gate venv has none, the Kaggle leg installs it
def test_a_stage_over_its_cap_is_failed_and_the_tarball_still_lands(kit):
    rc, out = kit({"PROFILE_TEST_CAP_S": "1"})
    status, _ = kit.results()
    assert rc == 0
    assert "probe: failed: exceeded its 1 s time cap" in status
    assert out.strip().splitlines()[-1].startswith("[repro] results:")


@pytest.mark.gpu
@pytest.mark.interop_matrix  # needs Warp (the kit's preflight and comparison arm): the gate venv has none, the Kaggle leg installs it
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
@pytest.mark.interop_matrix  # needs Warp (the kit's preflight and comparison arm): the gate venv has none, the Kaggle leg installs it
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
NEW_STAGES = ("compare", "crossover", "band", "chunk-sweep", "cadence-sweep", "ncu-ratio", "ncu-tsweep", "ncu-chunk",
              "ncu-rk78-small", "ncu-band")


def test_the_new_stages_are_in_the_full_and_the_quick_plan():
    for flags, suffix in (((), ""), (("--quick",), "-quick")):
        done = subprocess.run(["bash", str(PROFILE_SH), "--dry-run", *flags], capture_output=True, text=True,
                              env={**os.environ, "REPRO_TARGET": "devbox"})
        assert done.returncode == 0, done.stderr[-1500:]
        for stage in NEW_STAGES:
            assert f"stage {stage}{suffix} (cap" in done.stderr, (stage, done.stderr[-800:])
        assert "deadline=50 min" in done.stderr
        assert "persist-sweep" not in done.stderr and "fused-sweep" not in done.stderr


def test_the_dry_run_prints_the_time_estimate_and_the_clock_policy():
    rc, err = _plan()
    assert rc == 0 and "estimate on an A100/H100:" in err and "35 min budget" in err and "SM clock: auto" in err
    rc, err = _plan("--lock-clock", "off")
    assert rc == 0 and "SM clock: off" in err
    rc, err = _plan("--lock-clock", "fast")
    assert rc != 0 and "--lock-clock takes MHz, auto or off" in err
    rc, err = _plan("--quick", "--only", "chunk-sweep,ncu-chunk,ncu-rk78-small")
    assert rc == 0 and [ln.split()[2] for ln in err.splitlines() if "(cap" in ln] == [
        "chunk-sweep-quick", "ncu-chunk-quick", "ncu-rk78-small-quick"]


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
        "ncu-band", "ncu-chunk", "ncu-ratio", "ncu-rk78-small", "ncu-spread-1e4", "ncu-spread-1e6", "ncu-tsweep",
        "ncu-uniform-1e6"]
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
                      "band@auto", "band@8", "band@16", "warp", "auto_last"]  # auto first and last
    assert [lab for lab, _ in kit.compare_arms("rk78")] == [
        "auto", "fused_one", "persist@auto", "persist@256", "persist@384", "band@auto", "band@8", "band@16", "warp", "auto_last"]
    specs = dict(kit.compare_arms("osc"))
    assert specs["persist@640"] == "osc_persist@chunk=rule,tps=640" and specs["band@8"] == "osc_band@every=8"
    assert specs["auto_last"] == specs["auto"]
    # every forced persist arm passes its chunk, and none counts the slot fill
    forced = [spec for k in ("osc", "rk78", "osc32") for _, spec in kit.compare_arms(k) if "persist" in spec]
    assert forced and all("chunk=" in spec and "util" not in spec for spec in forced)
    assert specs["warp"] == "warp_kernel" and dict(kit.compare_arms("rk78"))["warp"] == "rk78_warp"


def test_the_fp32_cells_and_arms_cover_the_asked_matrix_and_are_labelled():
    kit = _kit_module()
    cells = [c for c in kit.compare_cells("compare") if c[0] == "osc32"]
    assert {(c[1], c[2], c[3]) for c in cells} == {(n, s, d) for n in (10**5, 10**6) for s in (100, 1000)
                                                   for d in ("spread", "uniform")} and len(cells) == 8
    assert [c for c in kit.compare_cells("compare-quick", kernels=("osc32",))] == [("osc32", 100_000, 100, "spread")]
    arms = kit.compare_arms("osc32")
    assert [lab for lab, _ in arms] == ["auto", "fused_one", "persist@auto", "persist@640", "persist@1024",
                                        "persist@1536", "persist@2048", "band@auto", "auto_last"]  # no Warp arm
    assert all(kit.precision_of(spec) == "fp32" for _, spec in arms)
    assert all(kit.precision_of(spec) == "fp64" for _, spec in kit.compare_arms("osc"))
    assert kit.precision_of("osc32") == "fp32" and kit.precision_of("osc") == "fp64"
    tsweep = [c for c in kit.ncu_captures("ncu-tsweep") if kit.precision_of(c[1]) == "fp32"]
    assert [kit.parse_spec(c[1])[1]["tps"] for c in tsweep] == ["384", "640", "896", "1152", "1536", "2048"]
    assert all(c[3] == (10**6, 1000, "spread") and "sm__inst_executed_pipe_fma.sum" in c[4] for c in tsweep)
    text, csv = kit.render_compare([dict(kernel="osc32", n=10**5, steps=100, dist="spread", reference="x", arms=[
        dict(label="auto", spec="eagle_graph_auto@fp=32", status="ok", note="", median=1.0, best=1.0, mode="persist",
             launched="-", util=None, bitwise=True, diff=0.0, sm_mhz=None, modes="", fields={})])])
    assert "precision=fp32" in text and csv.splitlines()[0].endswith("launch_fields,precision,note") and "SM clock" in text
    assert csv.splitlines()[1].split(",")[-2] == "fp32"


def test_crossover_band_and_tsweep_cover_the_asked_cases():
    kit = _kit_module()
    cross = kit.compare_cells("crossover")
    assert sorted(c[1] for c in cross) == [30_000, 100_000, 300_000] and {c[2:] for c in cross} == {(100, "spread")}
    assert [l for l, _ in kit.stage_arms("crossover", "osc")] == ["auto", "fused_one", "persist@auto", "auto_last"]
    band = kit.compare_cells("band")
    assert {(c[1], c[3]) for c in band} == {(n, d) for n in (10**5, 10**6) for d in ("uniform", "spread")}
    assert [l for l, _ in kit.stage_arms("band", "osc")] == [
        "auto", "band@auto", "band@16", "band@32", "band@64", "band@128", "auto_last"]
    assert len(kit.compare_cells("crossover-quick")) == 1 and len(kit.compare_cells("band-quick")) == 1
    tsweep = kit.ncu_captures("ncu-tsweep")
    osc = {kit.parse_spec(c[1])[1]["tps"] for c in tsweep if c[1].startswith("osc")}
    assert osc == {"384", "640", "896", "1152", "1536", "2048"} and len(tsweep) == 2 * 6 + 2 + 6
    assert all(c[3][0] == 10**6 and c[4] in (kit.RATIO_METRICS, kit.RATIO_METRICS_FP32) for c in tsweep)
    assert len(kit.ncu_captures("ncu-tsweep-quick")) == 1


def _raw(ratio, fp64, fma=0):
    return ('"ID","Kernel Name","smsp__thread_inst_executed_per_inst_executed.ratio","sm__inst_executed_pipe_fp64.sum",'
            f'"sm__inst_executed_pipe_fma.sum"\n"","","","inst","inst"\n"0","k","{ratio}","{fp64}","{fma}"\n')


def test_the_lane_metrics_come_from_ncu_with_the_step_cost_calibrated_on_the_uniform_cell():
    kit = _kit_module()
    line = "worker arm=osc_persist n=1 steps=1 dist=uniform sample_steps=32000 fp64_ops_ir=44 ops_source=sidecar"
    uniform = kit.lane_metrics(_raw(32.0, 24000), line, "fp64")
    assert uniform["lane_efficiency"] == 1.0 and uniform["i_step"] == 24.0  # SASS issues 24, not the IR's 44
    spread = kit.lane_metrics(_raw(20.8, 36000), line, "fp64", i_step=uniform["i_step"])
    assert abs(spread["lane_efficiency"] - 0.65) < 1e-9 and abs(spread["inflation"] - 1.5) < 1e-9
    assert abs(kit.lane_metrics(_raw(32.0, 24000), line, "fp64", i_step=24.0)["inflation"] - 1.0) < 1e-9  # the calibration cell
    text = kit.lane_lines(_raw(20.8, 36000), line, "fp64", 24.0, "the uniform cell")
    assert "lane_efficiency" in text and "= 0.650" in text and "fp64_inflation" in text and "= 1.500" in text
    assert "calibrated on the uniform cell" in text and "44" not in text  # the IR count is never divided in
    fp32 = kit.lane_lines(_raw(32.0, 0, 8000), line, "fp32", 8.0, "x")
    assert "fp32_inflation" in fp32 and "sm__inst_executed_pipe_fma.sum" in fp32
    assert "unavailable" in kit.lane_lines(_raw(32.0, 24000), "worker arm=x", "fp64")
    assert "no calibrated i_step" in kit.lane_lines(_raw(32.0, 24000), line, "fp64")
    assert kit._ncu_values("", ("a",)) == {}


def test_the_chunk_sweep_covers_the_asked_grid_and_every_arm_passes_its_chunk():
    kit = _kit_module()
    cells = kit.chunk_cells("chunk-sweep")
    assert {(c[0], c[2]) for c in cells} >= {(k, s) for k in ("osc", "osc32") for s in (100, 1000)}
    assert all(c[1] == 10**6 and c[3] in ("spread", "orbits") for c in cells) and ("rk78", 10**6, 0, "orbits") in cells
    arms = dict(kit.chunk_arms("osc"))
    walls = [lab for lab in arms if lab.startswith(("T1", "T2"))]
    assert sorted({int(lab.split("/c")[1]) for lab in walls}) == [1, 2, 4, 8, 16, 32] and len(walls) == 12
    assert all("chunk=" in spec for spec in arms.values())
    assert arms["T2/c8"].endswith("tps=2x") and "tps" not in arms["T1/c8"]
    assert [lab for lab in arms if lab.startswith("fill")] == [f"fill/c{c}" for c in (1, 2, 4, 8, 16, 32)]
    assert all(("util=1" in spec) == lab.startswith("fill") for lab, spec in arms.items())  # a wall never counts the fill
    assert not any(lab.startswith("T2") for lab in dict(kit.chunk_arms("osc", with_t2=False)))
    assert "fp=32" in dict(kit.chunk_arms("osc32"))["T1/c4"] and kit.precision_of(dict(kit.chunk_arms("osc32"))["T1/c4"]) == "fp32"
    assert [lab for lab in dict(kit.chunk_arms("rk78")) if lab.startswith("T1")] == ["T1/c8", "T1/c32"]
    caps = kit.ncu_captures("ncu-chunk")
    assert len(caps) == 2 * 2 * 6 + 2 and all(c[4] == kit.RATIO_METRICS_FP32 for c in caps)
    small = kit.ncu_captures("ncu-rk78-small")
    assert [c[1] for c in small] == ["rk78_auto", "rk78_fused", "rk78_persist@chunk=rule"] and {c[3] for c in small} == {(10**4, 0, "orbits")}
    assert len(kit.ncu_captures("ncu-chunk-quick")) == 1 and len(kit.ncu_captures("ncu-rk78-small-quick")) == 1


def test_the_nsys_grid_gets_the_forced_fused_arm_and_the_forced_persist_arm_takes_its_chunk():
    kit = _kit_module()
    assert "osc_fused" in kit.NSYS_ARMS and "osc_persist@chunk=rule" in kit.NSYS_ARMS
    assert not any(a == "eagle_graph_persist" for a in kit.NSYS_ARMS) and ("persist", "osc_persist@chunk=rule") == kit.NCU_KERNELS[2][:2]


def test_walls_are_normalised_to_the_maximum_clock_unless_it_was_locked(monkeypatch):
    kit = _kit_module()
    monkeypatch.delenv("PROFILE_CLOCK_MODE", raising=False)
    assert abs(kit.normalised(18.77, 1095, 1410) - 14.58) < 0.01  # the case that read as a crossover artefact
    assert kit.normalised(5.0, None, 1410) == 5.0 and kit.normalised(None, 1095, 1410) is None
    monkeypatch.setenv("PROFILE_CLOCK_MODE", "locked:1410")
    assert kit.normalised(18.77, 1095, 1410) == 18.77
    assert "locked for the run" in kit.clock_line()
    monkeypatch.delenv("PROFILE_CLOCK_MODE")
    assert "NOT locked" in kit.clock_line()


def test_a_request_above_the_residency_limit_is_skipped_by_what_launched():
    kit = _kit_module()
    assert "above the residency limit (launched 2048)" in kit.residency_skip(4096, 2048)
    assert kit.residency_skip(2048, 2048) == "" and kit.residency_skip(None, 2048) == "" and kit.residency_skip(4096, "n/a") == ""

    class Runner:
        launch_policy = {"block": 128, "grid": 128}

    class Old:
        _fast_block, _fast_grid = 64, 64

    kit._attr = lambda *_: 8
    assert kit.launched_threads_per_sm(Runner()) == 128 * 128 // 8 and kit.launched_threads_per_sm(Old()) == 64 * 64 // 8


def test_the_rule_chunk_and_the_threads_option_parse(monkeypatch):
    kit = _kit_module()
    assert kit.tps_option({}) is None and kit.tps_option({"tps": "640"}) == 640 and kit.tps_option({"tps": "2x"}) == "2x"
    policy = importlib.import_module("eagle._launch_policy")
    monkeypatch.setattr(policy, "persistent_threads_per_sm", lambda *a, **k: 300)
    with kit.persist_threads("2x"):
        assert policy.persistent_threads_per_sm() == 600
    assert policy.persistent_threads_per_sm() == 300
    pin, chunk, _ = kit._chunk_pin({"chunk": "8"}, None, 100.0, False)
    assert pin == {"_chunk": 8} and chunk == 8


def test_render_chunk_sweep_tabulates_walls_slot_fill_and_the_rule_against_the_fastest():
    kit = _kit_module()
    arms = []
    for thr, base in (("T1", 4.0), ("T2", 4.5)):
        for c, scale in zip((1, 2, 4, 8), (1.5, 1.2, 1.0, 1.1)):
            r = _rec(f"{thr}/c{c}", base * scale)
            r.update(chunk=c, launched=384, chunk_rule=4, raw=base * scale)
            arms.append(r)
    for c, f in zip((1, 2, 4, 8), (0.99, 0.95, 0.9, 0.8)):
        r = _rec(f"fill/c{c}", 4.1)
        r.update(chunk=c, util=f, chunk_rule=4)
        arms.append(r)
    text, csv = kit.render_chunk_sweep([dict(kernel="osc", n=10**6, steps=100, dist="spread", reference="x", arms=arms)])
    assert "rule chunk 4" in text and "slot fill" in text and "fastest at T1: chunk 4" in text and "+0.0% against the fastest" in text
    assert "SM clock" in text and csv.splitlines()[0].startswith("kernel,n,steps,dist,precision,threads,chunk,")
    assert len(csv.splitlines()) == 1 + len(arms)


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
    assert "N=1e5" in line and "faster: persist, 2.00x" in line and "settled on persist after 5 runs" in line
    assert "agrees" in line
    auto["probe_walls"] = "persist=0.4;fused_one=0.7>persist=0.4"
    assert "persist=0.4;fused_one=0.7>persist=0.4" in kit.crossover_lines([cross])[1]
    auto["modes"] = "persist>persist>persist>fused_one>fused_one"  # the last run decides, not the third
    assert "settled on fused_one" in kit.crossover_lines([cross])[0] and "DISAGREES" in kit.crossover_lines([cross])[0]


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
@pytest.mark.interop_matrix  # needs Warp (the kit's preflight and comparison arm): the gate venv has none, the Kaggle leg installs it
def test_compare_times_every_arm_and_the_override_really_changes_the_launch(tmp_path):
    done = run_py("--stage", "compare-quick", "--kernels", "osc,rk78", "--out", str(tmp_path), env=_one_gpu_env())
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    rows = [ln.split(",") for ln in (tmp_path / "compare_quick.csv").read_text().splitlines()[1:]]
    by = {(r[0], r[4].split("/")[0]): r for r in rows}  # a persist label carries its chunk: persist@640/4
    kit = _kit_module()
    assert {r[0] for r in rows} == {"osc", "rk78"}
    assert len(rows) == len(kit.compare_arms("osc")) + len(kit.compare_arms("rk78"))
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
    assert len(rows) == 9 and {r[0] for r in rows} == {"osc32"} and {r[-2] for r in rows} == {"fp32"}
    by = {r[4].split("/")[0]: r for r in rows}
    assert "warp" not in by and not any(r[5] == "wrong" for r in rows)
    for label in ("auto", "fused_one", "persist@auto", "persist@1024", "band@auto", "auto_last"):
        assert by[label][5] == "ok" and float(by[label][6]) > 0, label
    assert "precision=fp32" in (tmp_path / "compare_quick.txt").read_text()


@pytest.mark.gpu
def test_crossover_and_band_stages_run_and_record_the_clock(tmp_path):
    env = _one_gpu_env()
    for stage in ("crossover-quick", "band-quick"):
        done = run_py("--stage", stage, "--out", str(tmp_path), env=env)
        assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    cross = (tmp_path / "crossover_quick.txt").read_text()
    assert "-- crossover" in cross and re.search(r"after \d+ runs", cross) and "-- SM clock during the runs" in cross
    assert "eagle's own walls per timed run" in cross and "auto_last" in cross
    band = (tmp_path / "band_quick.txt").read_text()
    assert "-- band cost model" in band and "eagle's own band pick every=" in band
    header = (tmp_path / "band_quick.csv").read_text().splitlines()[0]
    assert {"sm_mhz", "modes_per_run", "probe_walls", "launch_fields", "precision", "note"} <= set(header.split(","))


@pytest.mark.gpu
def test_compare_flags_a_planted_wrong_arm_and_does_not_time_it(tmp_path):
    done = run_py("--stage", "compare-quick", "--out", str(tmp_path),
                  env={**_one_gpu_env(), "PROFILE_TEST_CORRUPT": "persist@640"})
    assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    rows = {(r.split(",")[0], r.split(",")[4].split("/")[0]): r.split(",") for r in
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
@pytest.mark.interop_matrix  # its Warp arm: the gate venv has no Warp, the Kaggle leg installs it
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


# --------------------------------------------------------------------------- #
# the CUDA profile switch (--cuda 12|13) and the Warp-only comparison arms
# --------------------------------------------------------------------------- #
BOOTSTRAP_SH = REPRO / "bootstrap.sh"
# these runs check which locks and environments are picked; the lock-version guard has its own test below
BOOT_ENV = {**os.environ, "REPRO_TARGET": "devbox", "REPRO_GPUS": "NVIDIA RTX PRO 6000,12.0,580.65",
            "REPRO_MAMBA": "/bin/true", "REPRO_ALLOW_LOCK_MISMATCH": "1"}


def _bootstrap(*flags, env=None):
    done = subprocess.run(["bash", str(BOOTSTRAP_SH), "--dry-run", "--work", "/nonexistent/work", "--src", "/nonexistent/src",
                           *flags], capture_output=True, text=True, env={**BOOT_ENV, **(env or {})})
    return done.returncode, done.stdout + done.stderr


@pytest.mark.parametrize("extra", [(), ("--from-pypi",)])
def test_cuda_13_selects_the_cuda13_locks_and_cuda_12_is_the_default(extra):
    rc13, out13 = _bootstrap("--cuda", "13", *extra)
    assert rc13 == 0, out13[-800:]
    lock = "pypi-gpu-cuda13.txt" if extra else "cards-gpu-cuda13.txt"
    assert lock in out13 and "cards-gpu-cuda13" in out13 and "env-gpu-cuda13.sh" in out13
    if not extra:
        assert "toolchain-linux-64-cuda13.lock" in out13
    rc, default = _bootstrap(*extra)
    rc12, out12 = _bootstrap("--cuda", "12", *extra)
    rce, via_env = _bootstrap(*extra, env={"REPRO_CUDA": "13"})
    assert rc == rc12 == rce == 0
    assert default == out12 and "cuda13" not in default and "torch" in default
    assert via_env == out13
    # the smoke test of the CUDA 13 profile imports no PyTorch, JAX or CuPy comparison arm
    smoke = [ln for ln in out13.splitlines() if "-W ignore -c" in ln][0]
    assert "torch" not in smoke.split("not in this profile")[0] and "jax.devices" not in smoke and "import cupy" not in smoke


def test_from_pypi_refuses_a_lock_older_than_the_checkout(tmp_path):
    # a lock that pins another eagle than this checkout would run new card scripts against old wheels
    locks = tmp_path / "locks"
    shutil.copytree(BOOTSTRAP_SH.parent / "locks", locks)
    lock = locks / "pypi-gpu.txt"
    lock.write_text(re.sub(r"raptor-eagle\[cuda12\]==\S+", "raptor-eagle[cuda12]==0.0.1", lock.read_text()))
    env = {"REPRO_ALLOW_LOCK_MISMATCH": "0", "REPRO_LOCKS": str(locks)}
    rc, out = _bootstrap("--from-pypi", env=env)
    assert rc != 0 and "pins raptor-eagle 0.0.1" in out, out[-800:]

def test_cuda_13_leaves_the_cpu_profile_alone():
    rc, cpu13 = _bootstrap("--profile", "cpu", "--cuda", "13")
    rc2, cpu12 = _bootstrap("--profile", "cpu")
    assert rc == rc2 == 0 and cpu13 == cpu12 and "cuda13" not in cpu13


def test_an_aarch64_machine_gets_the_aarch64_micromamba_and_toolchain_lock():
    arm = {"REPRO_MACHINE": "aarch64", "REPRO_MAMBA": "download"}
    rc, out = _bootstrap("--profile", "cpu", env=arm)
    assert rc == 0, out[-800:]
    assert "micromamba-linux-aarch64" in out and "toolchain-linux-aarch64.lock" in out
    assert "cards-cpu-linux-aarch64.txt" in out
    assert "micromamba-linux-64" not in out and "toolchain-linux-64" not in out
    rc, x86 = _bootstrap("--profile", "cpu", env={**arm, "REPRO_MACHINE": "x86_64"})
    assert rc == 0 and "micromamba-linux-64" in x86 and "toolchain-linux-64.lock" in x86
    rc, other = _bootstrap("--profile", "cpu", env={**arm, "REPRO_MACHINE": "riscv64"})
    assert rc != 0 and "unsupported CPU architecture 'riscv64'" in other
    rc, gpu = _bootstrap(env={**arm, "REPRO_MAMBA": "/bin/true"})
    assert rc != 0 and "no pip lock for profile gpu on linux-aarch64" in gpu


@pytest.mark.parametrize("script_args", [("bootstrap",), ("profile",)])
def test_an_unknown_cuda_version_is_refused_with_the_choices(script_args):
    script = BOOTSTRAP_SH if script_args[0] == "bootstrap" else PROFILE_SH
    for flags, env in ((("--cuda", "11"), {}), ((), {"REPRO_CUDA": "latest"})):
        done = subprocess.run(["bash", str(script), "--dry-run", *flags], capture_output=True, text=True,
                              env={**BOOT_ENV, **env})
        assert done.returncode != 0 and "takes 12" in done.stderr, done.stderr[-400:]
        assert "unknown CUDA version" in done.stderr


def test_profile_sh_takes_the_cuda_switch_through_to_its_environment_script():
    rc, err = _plan("--cuda", "13", "--work", "/nonexistent/work")
    assert rc == 0 and "env=/nonexistent/work/env-gpu-cuda13.sh" in err
    rc, err = _plan("--work", "/nonexistent/work")
    assert rc == 0 and "env=/nonexistent/work/env-gpu.sh" in err
    rc, err = _plan("--cuda", "12", "--work", "/nonexistent/work", "--env", "/x/custom.sh")
    assert rc == 0 and "env=/x/custom.sh" in err


def test_the_cuda13_locks_pin_every_package_and_carry_no_torch_jax_or_cupy12():
    pip = (REPRO / "locks" / "cards-gpu-cuda13.txt").read_text()
    names = [ln.split("==")[0].lower() for ln in pip.splitlines() if "==" in ln and not ln.startswith(" ")]
    assert len(names) >= 10 and pip.count("--hash=sha256:") == len(names)
    for needed in ("cupy-cuda13x", "warp-lang", "cuda-bindings", "nvidia-ml-py", "numpy"):
        assert needed in names, needed
    assert not {"torch", "jax", "jaxlib", "cupy-cuda12x"} & set(names)
    assert "sm_75" in pip and ">= 580" in pip
    tool = [ln for ln in (REPRO / "locks" / "toolchain-linux-64-cuda13.lock").read_text().splitlines()
            if ln.startswith("https://")]
    assert len(tool) > 50 and all(len(ln.rsplit("#", 1)[1]) == 32 for ln in tool)
    assert any("/cuda-nvcc-13." in ln for ln in tool) and not any("/cuda-nvcc-12." in ln for ln in tool)
    wheels = (REPRO / "locks" / "pypi-gpu-cuda13.txt").read_text()
    assert "raptor-eagle[cuda13]" in wheels and "raptor-hawk[cuda13]" in wheels
    assert not [ln for ln in wheels.splitlines() if not ln.startswith("#") and ("torch" in ln or "jax" in ln)]


def test_torch_and_jax_arms_are_not_in_the_profile_and_are_skipped_not_failed(tmp_path):
    kit = _kit_module()
    assert kit.CONTEXT_ARMS == () and set(kit.DROPPED_CONTEXT_ARMS) == {"torch_compiled", "jax_vmap"}
    assert kit.REFERENCE_ARMS == ("warp",)
    for stage in ("compare", "compare-quick", "crossover", "band", "chunk-sweep"):
        for kernel in ("osc", "rk78", "osc32"):
            specs = [spec for _label, spec in kit.stage_arms(stage, kernel)]
            assert not [s for s in specs if "torch" in s or "jax" in s], (stage, kernel)
    # the compare stage adds no context arm at the context point: eagle and Warp arms only
    seen = []
    orig = kit.run_cell
    kit.run_cell = lambda kernel, n, steps, dist, extra, arms: (seen.append(extra), {"arms": [], "kernel": kernel})[1]
    kit.render_compare = lambda cells, todo=(): ("", "")
    try:
        with pytest.raises(RuntimeError):  # no arm ran: the stage's own refusal, not a missing import
            kit.compare_stage(tmp_path, "compare", kernels=("osc",), ns=(1_000_000,))
    finally:
        kit.run_cell = orig
    assert seen and all(extra == () for extra in seen)


def test_warm_runs_until_eagle_stops_measuring_and_reports_an_unsettled_arm():
    from types import SimpleNamespace

    kit = _kit_module()

    def arm_of(flags):
        seq = iter(flags)
        arm = {"state": SimpleNamespace(upload=lambda inp: None), "report": None}

        def run():
            probe, timed = next(seq)
            arm["report"] = SimpleNamespace(mode="persist", probe=probe, timed=timed)

        arm["run"] = run
        return arm

    runs = []
    assert kit.warm_until_settled(arm_of([(True, True), (True, True), (False, False)]), None, runs)
    assert len(runs) == 3  # past the two-run floor, one more until a run is neither probe nor timed
    runs = []
    assert kit.warm_until_settled(arm_of([(False, False)] * 5), None, runs) and len(runs) == 2
    runs = []
    assert not kit.warm_until_settled(arm_of([(False, True)] * 20), None, runs)
    assert len(runs) == kit.WARM_CAP
    runs = []  # a report without the flags (forced mode, older eagle): the fixed floor
    bare = {"state": SimpleNamespace(upload=lambda inp: None), "report": SimpleNamespace(mode="x")}
    bare["run"] = lambda: None
    assert kit.warm_until_settled(bare, None, runs) and len(runs) == kit.SWEEP_WARM


def test_every_arm_is_timed_by_the_same_event_pair_whatever_its_launch_mode():
    kit = _kit_module()
    kinds = {kit._timer_kind(spec) for spec in ("eagle_graph_auto", "osc_fused", "osc_persist", "osc_persist@chunk=rule")}
    assert len(kinds) == 1 and kinds == {"event"}


def test_an_unrelated_mamba_on_path_is_not_taken_for_the_conda_frontend(tmp_path):
    """A tool that only shares the name (Kaggle ships a test runner called mamba) is skipped:
    the bootstrap falls back to its own pinned micromamba; a real conda-family frontend is used."""
    shim = tmp_path / "bin"
    shim.mkdir()
    (shim / "mamba").write_text("#!/bin/sh\necho 'usage: mamba [-h] [specs ...]' >&2\nexit 2\n")
    (shim / "mamba").chmod(0o755)
    env = {"PATH": f"{shim}:/usr/bin:/bin", "REPRO_MAMBA": ""}
    rc, out = _bootstrap(env=env)
    assert rc == 0, out[-800:]
    assert "conda frontend: /nonexistent/work/bin/micromamba" in out
    (shim / "micromamba").write_text("#!/bin/sh\n[ \"$1\" = info ] && { echo '{}'; exit 0; }\nexit 0\n")
    (shim / "micromamba").chmod(0o755)
    rc, out = _bootstrap(env=env)
    assert rc == 0, out[-800:]
    assert f"conda frontend: {shim / 'micromamba'}" in out


def test_cuda_12_is_the_cu129_runtime_and_12_6_keeps_the_pre_turing_lock():
    new = (REPRO / "locks" / "cards-gpu.txt").read_text()
    old = (REPRO / "locks" / "cards-gpu-cuda126.txt").read_text()
    assert "torch==2.12.1+cu129" in new and "whl/cu129" in new and "12.6." not in new
    assert "torch==2.12.1+cu126" in old and "whl/cu126" in old
    assert "whl/cu129" in (REPRO / "locks" / "pypi-gpu.txt").read_text()
    assert "whl/cu126" in (REPRO / "locks" / "pypi-gpu-cuda126.txt").read_text()
    rc, out = _bootstrap("--cuda", "12.6")
    assert rc == 0 and "cards-gpu-cuda126.txt" in out and "env-gpu-cuda126.sh" in out
    assert "toolchain-linux-64.lock" in out and "toolchain-linux-64-cuda126" not in out


def test_ns_and_every_options_parse_and_N_is_in_the_cells_labels_and_file_names():
    kit = _kit_module()
    # chunk-sweep / ncu-chunk / band: the default is today's N; an explicit list repeats the cells per N
    assert {c[1] for c in kit.chunk_cells("chunk-sweep")} == {10**6}
    two = kit.chunk_cells("chunk-sweep", (10**6, 10**7))
    assert {c[1] for c in two} == {10**6, 10**7} and len(two) == 2 * len(kit.chunk_cells("chunk-sweep"))
    assert {c[1] for c in kit.compare_cells("band", band_ns=(10**7,))} == {10**7}
    assert {c[1] for c in kit.compare_cells("band")} == set(kit.BAND_NS)
    default = kit.ncu_captures("ncu-chunk")
    caps = kit.ncu_captures("ncu-chunk", (10**6, 10**7))
    assert len(caps) == 2 * len(default) and len({c[0] for c in caps}) == len(caps)  # labels do not collide
    assert all(c[0].endswith(("_n1e6", "_n1e7")) and f"n{kit.n_tag(c[3][0])}" in c[0] for c in caps)
    assert {c[3][0] for c in caps} == {10**6, 10**7} and not any("_n1e" in c[0] for c in default)
    assert kit.ns_suffix(None) == "" and kit.ns_suffix((10**6, 10**7)) == "_n1e6_1e7" and kit.n_tag(1_234_567) == "1234567"
    # band cadences
    assert [lab for lab, _ in kit.stage_arms("band", "osc")][2:-1] == [f"band@{e}" for e in kit.BAND_SWEEP_EVERY]
    arms = kit.stage_arms("band", "osc", (16, 32, 64, 128, 256, 512))
    assert [lab for lab, _ in arms][2:-1] == [f"band@{e}" for e in (16, 32, 64, 128, 256, 512)]
    assert dict(arms)["band@512"] == "osc_band@every=512"


def test_gpu_profile_cli_takes_ns_and_every_lists(monkeypatch, tmp_path):
    kit = _kit_module()
    seen = {}
    monkeypatch.setattr(kit, "compare_stage", lambda *a, **k: seen.update(a=a, k=k))
    monkeypatch.setattr(kit, "chunk_sweep_stage", lambda *a, **k: seen.update(chunk=a))
    assert kit.main(["--out", str(tmp_path), "--stage", "band", "--ns", "1e6", "1e7", "--every", "16", "32", "512"]) == 0
    assert seen["k"] == {"band_ns": (10**6, 10**7), "every": (16, 32, 512)}
    assert kit.main(["--out", str(tmp_path), "--stage", "chunk-sweep", "--ns", "10000000"]) == 0
    assert seen["chunk"][-1] == (10**7,)
    assert kit.main(["--out", str(tmp_path), "--stage", "band"]) == 0
    assert seen["k"] == {"band_ns": None, "every": None} and seen["a"][3] == kit.COMPARE_NS
