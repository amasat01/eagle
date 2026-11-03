# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The profiling kit (``benchmarks/reproduce/gpu_profile.py`` and ``profile.sh``): the plan and its estimates, the
stage cells and parsers, the persist-size override, the script's error paths -- a failing stage, a timeout, a kill
followed by a resume, a missing ncu -- each of which still leaves a tarball with an accurate ``status.txt``, and the
bootstrap's CUDA profile switch."""

from __future__ import annotations

import csv
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import types
import tarfile
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("hawk")

REPRO = Path(__file__).resolve().parents[2] / "benchmarks" / "reproduce"
PROFILE_PY = REPRO / "gpu_profile.py"
PROFILE_SH = REPRO / "profile.sh"
STAGES = ("sol_probe", "wide_sweep", "ncu_corpus", "ncu_ensemble", "chunk_residency", "launch_shape")


def _kit_module():
    spec = importlib.util.spec_from_file_location("gpu_profile_under_test", PROFILE_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _one_gpu_env():
    count = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.count("GPU ")
    if count > 1 and not os.environ.get("CUDA_VISIBLE_DEVICES", "").isdigit():
        pytest.skip("several GPUs and CUDA_VISIBLE_DEVICES does not name one")
    return {"CUDA_DEVICE_ORDER": "PCI_BUS_ID"}


def _plan(*flags):
    done = subprocess.run(["bash", str(PROFILE_SH), "--dry-run", *flags], capture_output=True, text=True,
                          env={**os.environ, "REPRO_TARGET": "devbox"})
    return done.returncode, done.stderr


def _planned(err):
    return [ln.split()[2] for ln in err.splitlines() if "(cap" in ln]


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
    rc, out = kit({"PROFILE_TEST_FAIL": "wide_sweep"}, "--only", "sol_probe,wide_sweep,launch_shape")
    status, names = kit.results()
    assert rc == 1                                   # every stage ran; the failed one fails the run (run_cards' step)
    assert "sol_probe: ok" in status
    assert "wide_sweep: failed: exit 1" in status and "forced failure" in status
    assert "launch_shape: ok" in status
    assert "data/sol_probe.txt" in names
    assert out.strip().splitlines()[-1].startswith("[repro] results:")


@pytest.mark.gpu
def test_a_stage_over_its_cap_is_failed_and_the_tarball_still_lands(kit):
    rc, out = kit({"PROFILE_TEST_CAP_S": "1"}, "--only", "launch_shape")
    status, _ = kit.results()
    assert rc == 1
    assert "launch_shape: failed: exceeded its 1 s time cap" in status
    assert out.strip().splitlines()[-1].startswith("[repro] results:")


@pytest.mark.gpu
def test_a_killed_run_resumes_from_its_markers(kit):
    proc = kit.start(None, "--only", "sol_probe,launch_shape")
    deadline = time.time() + 300
    done_marker = Path(kit.out) / ".stages" / "sol_probe.ok"
    while not done_marker.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert done_marker.exists()
    proc.send_signal(signal.SIGTERM)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 143
    status, _ = kit.results()
    assert "sol_probe: ok" in status
    assert out.strip().splitlines()[-1].startswith("[repro] results:")
    for stale in Path(kit.out).glob("profile-*.tgz"):
        stale.unlink()
    rc, out = kit(None, "--only", "sol_probe,launch_shape")
    status, _ = kit.results()
    assert rc == 0
    assert "sol_probe: ok (finished in an earlier run)" in status
    assert "launch_shape: ok" in status


@pytest.mark.gpu
def test_the_ncu_stages_are_skipped_by_reason_and_the_rest_run(kit, tmp_path):
    """No ncu on the path, or a GPU older than compute capability 7.0 (the P2000): the reason is in status.txt."""
    shim = tmp_path / "shim"
    shim.mkdir()
    nsys = shutil.which("nsys")
    if nsys:
        (shim / "nsys").symlink_to(nsys)
    keep = [d for d in os.environ["PATH"].split(os.pathsep) if not (Path(d) / "ncu").exists()
            and not (Path(d) / "nsys").exists()]
    env = {"PATH": os.pathsep.join([str(shim), *keep]), "PROFILE_TOOLS_PATH_ONLY": "1"}
    nvcc = os.environ.get("HAWK_NVCC") or shutil.which("nvcc")
    if nvcc:  # a toolkit's bin holds ncu next to nvcc: dropping that dir must not drop the kernel compiler too
        env["HAWK_NVCC"] = nvcc
    rc, _ = kit(env, "--only", "sol_probe,ncu")
    status, _ = kit.results()
    assert rc == 0
    assert "ncu_corpus: skipped: ncu not found" in status and "ncu_ensemble: skipped: ncu not found" in status
    assert "sol_probe: ok" in status


@pytest.mark.gpu
def test_the_chunk_residency_and_launch_shape_stages_run_and_write_their_tables(kit):
    rc, _ = kit(None, "--only", "chunk_residency,launch_shape")
    status, names = kit.results()
    assert rc == 0 and "chunk_residency: ok" in status and "launch_shape: ok" in status
    assert {"data/chunk_residency.csv", "data/launch_shape.csv", "data/crossover.csv"} <= set(names)
    with (kit.out / "data" / "chunk_residency.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows and all(r["wall_ms"] and not r["error"] for r in rows)


# --- the plan, the cells and the parsers (no device) ---------------------------------------
def test_the_full_and_the_quick_plan_are_the_six_stages():
    for flags in ((), ("--quick",)):
        rc, err = _plan(*flags)
        assert rc == 0 and _planned(err) == list(STAGES), err[-800:]
    assert "deadline=30 min" in _plan()[1]


def test_the_dry_run_prints_the_a100_estimate_the_p2000_plan_and_the_clock_policy():
    rc, err = _plan()
    est = re.search(r"estimate on an A100: (\d+) min of the (\d+) min budget", err)
    assert rc == 0 and est and int(est[1]) <= int(est[2]) <= 25 and "SM clock: auto" in err
    assert "compute capability 7.0 (P2000): the ncu_* stages are skipped" in err
    rc, err = _plan("--quick")
    assert rc == 0 and int(re.search(r"estimate on an A100: (\d+) min", err)[1]) < 8
    rc, err = _plan("--lock-clock", "off")
    assert rc == 0 and "SM clock: off" in err
    rc, err = _plan("--lock-clock", "fast")
    assert rc != 0 and "--lock-clock takes MHz, auto or off" in err


def test_every_stage_profile_sh_plans_is_one_gpu_profile_py_accepts():
    kit = _kit_module()
    names = set(_planned(_plan()[1]) + _planned(_plan("--quick")[1]))
    assert names == set(STAGES) and names <= set(kit.STAGES)


def test_only_selects_stages_by_name_or_prefix_and_refuses_an_unknown_one():
    rc, err = _plan("--only", "launch_shape,sol_probe")
    assert rc == 0 and _planned(err) == ["sol_probe", "launch_shape"]
    rc, err = _plan("--only", "ncu")
    assert rc == 0 and _planned(err) == ["ncu_corpus", "ncu_ensemble"]
    rc, err = _plan("--only", "chunk_res")
    assert rc != 0 and "matches no stage" in err


def test_no_oscillator_is_left_in_the_kit():
    hits = [p.name for p in REPRO.iterdir() if p.is_file() and p.suffix in (".py", ".sh", ".md")
            and re.search("oscillator|osc32", p.read_text(), re.I)]
    assert hits == []


def test_the_chunk_grid_covers_chunk_threads_dtype_n_and_distribution():
    kit = _kit_module()
    cells = kit.chunk_cells(type("A", (), {"ns": None, "quick": False})())
    assert len(cells) == 4 * 2 * 2 * 2 * 2 and all(c["arm"] == "persist" for c in cells)
    assert {c["n"] for c in cells} == {1_000_000, 10_000_000}
    assert {c["dist"] for c in cells} == {"spread", "spread-short"} and {c["dtype"] for c in cells} == {"float32", "float64"}
    assert {c["chunk"] for c in cells} == {1, 4, 16, 32} and {c["tps"] for c in cells} == {None, "2x"}
    quick = kit.chunk_cells(type("A", (), {"ns": None, "quick": True})())
    assert 0 < len(quick) < 8


def test_a_tie_is_a_runner_up_within_ten_percent_of_the_best():
    kit = _kit_module()
    rows = [{"n": 1, "dist": "s", "dtype": "d", "chunk": c, "wall_ms": w} for c, w in ((1, 10.0), (4, 10.9), (16, 30.0))]
    rows += [{"n": 2, "dist": "s", "dtype": "d", "chunk": c, "wall_ms": w} for c, w in ((1, 10.0), (4, 12.0))]
    rows += [{"n": 3, "dist": "s", "dtype": "d", "error": "OutOfMemoryError"}]
    (pair,) = kit.chunk_ties(rows)
    assert [r["chunk"] for r in pair] == [1, 4] and pair[0]["n"] == 1


def test_the_corpus_captures_the_ties_then_team_and_staged_at_the_widest_point(tmp_path):
    kit = _kit_module()
    won = [{"dtype": "float64", "width": 64, "fold": 128, "form": "thread", "tied": ["thread", "team"], "tie": True},
           {"dtype": "float64", "width": 256, "fold": 128, "form": "team", "tied": ["team"], "tie": False}]
    path = tmp_path / "wide_sweep.json"
    path.write_text(json.dumps({"winners": won}))
    cells = kit.corpus_cells(path, False)
    assert cells == [("thread", 64, 128, "float64"), ("team", 64, 128, "float64"),
                     ("team", 256, 128, "float64"), ("staged", 256, 128, "float64")]
    assert len(kit.corpus_cells(path, True)) == 2


def test_ncu_values_reads_the_first_kernel_of_the_raw_page_and_skips_the_units_row():
    kit = _kit_module()
    raw = '"ID","m_a","m_b","m_c"\n"","%","inst","x"\n"0","1,500.5","2","n/a"\n"1","9","9","9"\n'
    got = kit.ncu_values(raw)
    assert got["m_a"] == 1500.5 and got["m_b"] == 2.0 and "m_c" not in got
    assert kit.ncu_values("") == {}


def test_the_ensemble_asks_for_the_lane_pipe_issue_occupancy_register_dram_and_stall_metrics():
    kit = _kit_module()
    text = " ".join(kit.ENSEMBLE_METRICS)
    for part in ("thread_inst_executed_per_inst_executed", "pipe_fp64", "inst_issued", "warps_active",
                 "registers_per_thread", "dram__throughput", "long_scoreboard"):
        assert part in text, part
    assert all(op in " ".join(kit.DP_METRICS) for op in ("dfma", "dmul", "dadd"))


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


# --------------------------------------------------------------------------- #
# the CUDA profile switch (--cuda 12|13)
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




def test_an_ncu_stage_whose_every_capture_failed_is_failed_with_the_reason(tmp_path, monkeypatch):
    """Counters not permitted (a container, or no admin rights): the stage fails naming ERR_NVGPUCTRPERM, the rows stay."""
    mod = _kit_module()
    denied = "==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters"
    monkeypatch.setattr(mod, "require_ncu", lambda args: None)
    monkeypatch.setattr(mod, "ncu_capture", lambda *a, **k: ({}, denied))
    args = types.SimpleNamespace(quick=True, out=tmp_path)
    with pytest.raises(RuntimeError, match=r"every ncu capture failed \(2\): .*ERR_NVGPUCTRPERM"):
        mod.ncu_ensemble(args)
    assert (tmp_path / "ncu_ensemble.csv").read_text().count("ERR_NVGPUCTRPERM") == 2
    mod.ncu_stage_verdict(["", denied])   # one capture succeeded: the stage is ok, the failed row keeps its note
    mod.ncu_stage_verdict([])             # nothing to capture: nothing to judge


def test_every_card_lock_carries_setuptools_scm_for_the_editable_builds():
    """bootstrap.sh installs aether-dsc, raptor and hawk with --no-build-isolation; their version comes from setuptools-scm."""
    for lock in sorted((REPRO / "locks").glob("cards-*.txt")):
        assert "setuptools-scm==" in lock.read_text(), lock.name


def test_an_ncu_failure_note_keeps_ncus_own_lines_ahead_of_a_later_library_warning():
    mod = _kit_module()
    text = ("==PROF== Connected to process 1\n==ERROR== Profiling is not supported on this device\n"
            "worker {}\n" + "x.py:600: FutureWarning: The pynvml package is deprecated. " * 8 + "\n")
    note = mod.ncu_failure_note(text)
    assert note.splitlines()[0] == "==ERROR== Profiling is not supported on this device"
    assert "worker {}" in note                       # the profiled application's last lines follow ncu's own
    with pytest.raises(RuntimeError, match="not supported on this device"):
        mod.ncu_stage_verdict([note])
    assert mod.ncu_failure_note("plain output") == "plain output"          # nothing of ncu's own: the tail
    assert "No kernels were profiled" in mod.ncu_failure_note("==WARNING== No kernels were profiled.\n")


def _run_cards(*flags):
    done = subprocess.run(["bash", str(REPRO / "run_cards.sh"), "--dry-run", "--work", "/nonexistent/work", "--slug", "m",
                           *flags], capture_output=True, text=True)
    return done.returncode, done.stdout + done.stderr


@pytest.mark.parametrize("plan", ["l4", "a100", "zen5"])
def test_a_smoke_session_runs_every_step_small_and_apart_from_the_full_one(plan):
    """--smoke: the plan's steps in order at their smallest, results and markers under <slug>-smoke (so a full run later
    on the same machine skips nothing), and the sweep's rows never written into hawk."""
    rc_full, full = _run_cards("--plan", plan)
    rc, smoke = _run_cards("--plan", plan, "--smoke")
    assert rc == rc_full == 0
    steps = lambda out: re.findall(r"\[repro\] step (\w+)", out)  # noqa: E731
    assert steps(smoke) == steps(full) and steps(full)
    assert "/cards/m-smoke/" in smoke and "/cards/m-smoke" not in full and "/cards/m/" in full
    commands = [ln for ln in smoke.splitlines() if ln.startswith("+ (")]
    for ln in commands:
        if "perf_card/" in ln:
            assert "--smoke" in ln, ln
        if "hawk.targets.cards.sweep" in ln:
            assert "--quick" in ln and "--write" not in ln, ln
        if "profile.sh" in ln:
            assert "--quick" in ln, ln
    if plan != "zen5":
        assert any("hawk.targets.cards.sweep" in ln and "--write" in ln for ln in full.splitlines())
    assert int(re.search(r"session deadline (\d+) min", smoke)[1]) <= 18


def test_a_card_in_which_no_arm_ran_fails_its_step_and_one_ok_arm_passes_it(capsys):
    spec = importlib.util.spec_from_file_location("card_common_under_test", REPRO.parent / "_card_common.py")
    cc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cc)
    failed = {"rows": [{"arms": {"thread": {"status": "failed: ModuleNotFoundError: No module named 'pynvml'"},
                                 "torch": {"status": "not_run: not installed: torch"}}}]}
    assert cc.no_arm_ran(failed) and cc.no_arm_ran_exit(failed) == 1
    assert "no arm of this card ran" in capsys.readouterr().err
    one_ok = {"rows": failed["rows"] + [{"arms": {"team": {"status": "ok"}}}]}
    assert not cc.no_arm_ran(one_ok) and cc.no_arm_ran_exit(one_ok) == 0
    assert cc.no_arm_ran({"results": []})                       # nothing measured is no card
