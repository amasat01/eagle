# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ensemble card script (``benchmarks/perf_card/perf_card.py``): the order of its memory probe, a fit note for every
arm it can run, and arm selection without the optional libraries."""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import pathlib
import types

import pytest

_REPO = pathlib.Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.repo_local     # the card script lives in the repository, not in the wheel
_CARD_DIR = _REPO / "benchmarks" / "perf_card"


def _rk78_script():
    spec = importlib.util.spec_from_file_location("eagle_perf_card", _CARD_DIR / "perf_card.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_memory_probe_run(module, monkeypatch, probe, patches):
    """Run ``module._memory_probe`` with every device and process reading faked;
    returns the ordered list of the calls that matter to the baseline."""
    calls = []

    def note(name, ret=None):
        def fake(*a, **k):
            calls.append(name)
            return ret
        return fake

    cc = module.cc
    monkeypatch.setattr(cc, "probe_reply_fd", note("reply_fd", 1))
    monkeypatch.setattr(cc, "probe_reply", note("reply"))
    monkeypatch.setattr(cc, "process_device_bytes", note("baseline", (123, "fake")))
    monkeypatch.setattr(cc, "DevicePeak", lambda bus_id: contextlib.nullcontext(types.SimpleNamespace(peak=0)))
    monkeypatch.setattr(cc, "trim_pools", note("trim"))
    monkeypatch.setattr(cc, "probe_sync", note("sync"))
    monkeypatch.setattr(cc, "status_kib", note("rss", 1))
    monkeypatch.setattr(cc, "reset_host_peak", note("reset", True))
    monkeypatch.setattr(module, "_touch_device", note("touch"))
    for name, ret in patches.items():
        monkeypatch.setattr(module, name, note(name, ret))
    probe()
    return calls


def _assert_baseline_before_warmup(calls, warm_call):
    assert calls.count("baseline") == 2, calls  # the baseline, then the reading after the run
    first = calls.index("baseline")
    assert "touch" in calls[:first], (
        f"the CUDA context must exist (one tiny device op) before the baseline: {calls}")
    assert first < calls.index(warm_call), (
        f"the device baseline must be read before the warm-up call {warm_call}: {calls}")
    assert first < calls.index("trim"), (
        f"the device baseline must be read before any allocator is trimmed: {calls}")
    assert calls.index("trim") < calls.index("baseline", first + 1), calls


def test_rk78_card_memory_baseline_is_read_before_the_warm_up(monkeypatch):
    """Same order check for the second card's memory probe.

    RED: read the baseline after the warm-up and this fails."""
    import numpy as np

    module = _rk78_script()
    out = (0.0, 0.0, {"t": np.full(2, module.T_FINAL)})
    patches = {"_probe_arm": {}, "_timed": out, "_inputs": {}, "_sync": None}
    calls = _fake_memory_probe_run(
        module, monkeypatch,
        lambda: module._memory_probe("eagle_graph", 8, "0000:00:00.0"), patches)
    _assert_baseline_before_warmup(calls, "_timed")


def test_every_arm_has_a_fit_note(monkeypatch):
    """'Where each tool fits' names every arm the script can run.

    RED: delete one arm's entry from ``perf_card._fit_notes`` and this fails with a ``KeyError`` naming the arm."""
    monkeypatch.syspath_prepend(str(_CARD_DIR))
    card_mod = importlib.import_module("perf_card")
    card = {"results": [], "ns": [1000], "arms": list(card_mod.ARMS),
            "code_lines": {"arms": dict.fromkeys(card_mod.ARMS, 1)},
            "workload": {"n_stages": 13, "graph_block": 16}}
    assert set(card_mod._fit_notes(card)["arms"]) == set(card_mod.ARMS)


def test_missing_optional_libraries_become_arms_not_run(monkeypatch):
    """With torch and jax blocked arm selection raises nothing and lists them, with the reason, under arms_not_run."""
    import sys

    module = _rk78_script()
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "jax", None)
    names, not_run = module._select_arms(module.COMMITTED_ARMS)
    for arm in ("torch_graphed", "torch_compiled"):
        assert not_run[arm] == "torch not installed in this profile"
    assert not_run["jax_vmap"] == "jax not installed in this profile"
    assert not set(not_run) & set(names) and "eagle_graph" in names


def test_a_float32_device_arm_uploads_its_inputs_in_the_plane_dtype():
    """RED: the cast to the hard-coded float64 ``SCALAR`` raises TypeError on a float32 plane (needs a device)."""
    import numpy as np
    import pytest

    cp = pytest.importorskip("cupy")

    module = _rk78_script()
    arm = module._GpuArm(8)
    arm.planes = {k: v.astype("float32") if v.dtype == "float64" else v for k, v in arm.planes.items()}
    arm.upload({name: np.ones(8) for name in module.STATE_NAMES})
    assert arm.planes["s"].dtype == cp.float32 and float(arm.planes["s"][0, 0]) == 1.0


def test_a_memory_probe_predicted_over_its_cap_is_not_started(monkeypatch, tmp_path):
    """An arm whose warm run at the smaller N, scaled to the larger N, exceeds the cap (or that already timed out) is
    labelled without starting a probe; a fast arm still runs at both N.
    RED: make ``_memory_probe_predicted_over`` return False and the slow arm's 10x probe is started."""
    module = _rk78_script()
    cc = module.cc
    cap = module.MEMORY_PROBE_TIMEOUT_S
    run_s = {"fast": 1.0, "slow": cap / 5}          # x10 N: 10 s (runs), 2 x cap (skipped)
    started = []

    def fake_probe(script, args, **kw):
        arm, n = args[1], int(args[2])
        started.append((arm, n))
        if arm == "dead":
            raise module.subprocess.TimeoutExpired("probe", cap)
        return {"device_peak_bytes": n, "host_peak_bytes": 0, "run_seconds": run_s[arm]}

    monkeypatch.setattr(cc, "run_probe", fake_probe)
    monkeypatch.setattr(cc, "part_load", lambda *a, **k: None)
    monkeypatch.setattr(cc, "part_save", lambda *a, **k: None)
    monkeypatch.setattr(cc, "deadline_passed", lambda: False)
    monkeypatch.setattr(module, "_ASSEMBLE", False)
    rows = module._run_memory_pass([1000, 10000], "bus", ["fast", "slow", "dead"])
    assert started == [("fast", 1000), ("slow", 1000), ("dead", 1000), ("fast", 10000)], started
    big = {r["arm"]: r for r in rows if r["n"] == 10000}
    assert big["fast"]["device_peak_bytes"] == 10000
    for arm in ("slow", "dead"):
        assert "predicted >" in module._dev_cell(big[arm]), big[arm]
