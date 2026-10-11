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
NEW_STAGES = ("persist-sweep", "fused-sweep", "cadence-sweep", "ncu-ratio", "ncu-band")


def test_the_new_stages_are_in_the_full_and_the_quick_plan():
    for flags, suffix in (((), ""), (("--quick",), "-quick")):
        done = subprocess.run(["bash", str(PROFILE_SH), "--dry-run", *flags], capture_output=True, text=True,
                              env={**os.environ, "REPRO_TARGET": "devbox"})
        assert done.returncode == 0, done.stderr[-1500:]
        for stage in NEW_STAGES:
            assert f"stage {stage}{suffix} (cap" in done.stderr, (stage, done.stderr[-800:])
        assert "deadline=35 min" in done.stderr


def test_every_stage_profile_sh_plans_is_one_gpu_profile_py_accepts():
    kit = _kit_module()
    planned = subprocess.run(["bash", str(PROFILE_SH), "--dry-run"], capture_output=True, text=True,
                             env={**os.environ, "REPRO_TARGET": "devbox"}).stderr
    quick = subprocess.run(["bash", str(PROFILE_SH), "--dry-run", "--quick"], capture_output=True, text=True,
                           env={**os.environ, "REPRO_TARGET": "devbox"}).stderr
    names = {ln.split()[2] for ln in (planned + quick).splitlines() if " stage " in f" {ln} " and "(cap" in ln}
    assert len(names) >= 10 and names <= set(kit.STAGES), names - set(kit.STAGES)


def test_sweep_cells_cover_the_asked_grid():
    kit = _kit_module()
    persist = kit.sweep_cells("persist-sweep")
    osc = [c for c in persist if c[0] == "osc"]
    assert {c[4] for c in osc} == {None, *kit.OSC_PERSIST_TPS}
    assert {(c[2], c[3]) for c in osc} == {(s, d) for s in (100, 1000) for d in ("uniform", "spread")}
    assert {c[4] for c in persist if c[0] == "rk78"} == {None, *kit.RK78_PERSIST_TPS}
    assert len(persist) == 4 * 5 + 3
    assert len(kit.sweep_cells("fused-sweep")) == 4 + 1
    assert len(kit.sweep_cells("persist-sweep-quick")) == 2 and len(kit.sweep_cells("fused-sweep-quick")) == 1
    cadence = kit.nsys_cells("cadence-sweep")
    assert {kit.parse_spec(c[0])[1]["every"] for c in cadence} == {"4", "8", "16", "32"}
    assert {c[3] for c in cadence} == {"uniform", "spread"} and {c[1:3] for c in cadence} == {(10**6, 1000)}
    assert sum(kit.parse_spec(c[0])[1].get("eager") == "1" for c in cadence) == len(cadence) // 2


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
    assert "entered" not in outcome and "one sweep cell at a time" in outcome["error"]
    with kit.persist_threads(960):  # free again once the first block ended
        pass


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_the_kit_scripts_are_shellcheck_clean():
    done = subprocess.run(["shellcheck", "-x", str(PROFILE_SH)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-1500:]


@pytest.mark.gpu
def test_the_sweeps_run_and_the_override_really_changes_the_launch(tmp_path):
    env = _one_gpu_env()
    for stage in ("persist-sweep-quick", "fused-sweep-quick"):
        done = run_py("--stage", stage, "--out", str(tmp_path), env=env)
        assert done.returncode == 0, done.stdout[-1500:] + done.stderr[-1500:]
    rows = (tmp_path / "persist_sweep_quick.txt").read_text().splitlines()[-2:]
    auto, forced = (r.split() for r in rows)
    assert auto[3] == "auto" and forced[3] == "640" and forced[4] == "640"  # launched = requested
    assert forced[5] == "persist" and 0.0 < float(forced[8]) <= 1.0
    fused = (tmp_path / "fused_sweep_quick.txt").read_text().splitlines()[-1].split()
    assert fused[5] == "fused_one" and float(fused[6]) > 0


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
@pytest.mark.parametrize("stage", ["ncu-ratio-quick", "ncu-band-quick"])
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
        assert ("smsp__thread_inst_executed_per_inst_executed" in text) == stage.startswith("ncu-ratio")
