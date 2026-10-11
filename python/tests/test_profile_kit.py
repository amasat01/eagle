# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The profiling kit (``benchmarks/reproduce/gpu_profile.py`` and ``profile.sh``): the decision
probe on the host, the probe and one Nsight Systems capture on the device (Nsight Compute
is skipped cleanly where the GPU is too old for it), and the script's error paths -- a
failing stage, a timeout, a kill followed by a resume, a missing ncu -- each of which still
leaves a tarball with an accurate ``status.txt``."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import pytest

pytest.importorskip("hawk")

REPRO = Path(__file__).resolve().parents[2] / "benchmarks" / "reproduce"
PROFILE_PY = REPRO / "gpu_profile.py"
PROFILE_SH = REPRO / "profile.sh"
PROBE_KEYS = ("target:", "device.name:", "auto.persist_entry:", "auto.device_capacity():",
              "auto.fast_mode_for(n=1000):", "auto.fast_mode_for(n=1000000):",
              "auto.run_mode(n=1000):", "persist.run_mode(n=1000000):")
DEVICE_KEYS = ("device.multiProcessorCount:", "auto.num_regs:", "auto.resident_threads_per_sm:",
               "auto.capacity (sms * resident):", "auto.latency_regime_capacity:",
               "persist.lane_utilisation(n=1000000):")


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
