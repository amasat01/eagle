# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The uneven-batch efficiency is bounded by a committed speed-of-light reference.

``eff = R_spread / R_SOL`` with ``R_SOL = F_dev / 22`` (``benchmarks/sol_reference.json``).
Nothing can run faster than the speed of light, so every committed RK4 card row must sit at or
below 1.00 x R_SOL, a row above it is refused (never clamped), and an entry that is still
pending publishes nothing.

RED: plant a row at 110 % of R_SOL in a card, or mark a reference ``pending``, and the claim
raises ``ClaimError`` (``test_a_planted_row_above_the_speed_of_light_is_refused``).

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("eagle_pcl_eff", _REPO / "tools" / "prose_claims.py")
pcl = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pcl)

_CARDS = _REPO / "benchmarks" / "perf_card"
_GPU_CARDS = sorted(_CARDS.glob("card_*.json"))
_CPU_CARDS = sorted(_CARDS.glob("cpu_card_*.json"))
_REF = json.loads((_REPO / "benchmarks" / "sol_reference.json").read_text())


def _entry(path):
    slug = pcl.card_slug(path)
    return _REF["devices"][_REF.get("aliases", {}).get(slug, slug)]


def test_the_constant_is_the_audited_instruction_count():
    assert pcl.SOL_OPS_PER_STEP == _REF["ops_per_step"] == 16 + 4 + 2


@pytest.mark.parametrize("path", _GPU_CARDS, ids=lambda p: p.name)
def test_every_gpu_row_is_at_or_below_the_speed_of_light(path):
    card = json.loads(path.read_text())
    fma = pcl.sol_fma_per_s(_REPO, path)
    pcl.check_rows_under_sol(card, fma)
    top = max(r["sample_steps_per_s"] * pcl.SOL_OPS_PER_STEP / fma for r in card["results"] if r["sample_steps_per_s"])
    assert 0.0 < top <= 1.0


@pytest.mark.parametrize("path", _CPU_CARDS, ids=lambda p: p.name)
def test_a_cpu_card_is_bounded_when_measured_and_refused_when_pending(path):
    entry = _entry(path)
    if entry["source"] == "pending":
        with pytest.raises(pcl.ClaimError, match="pending"):
            pcl.sol_fma_per_s(_REPO, path)
        return
    card = json.loads(path.read_text())
    pcl.check_rows_under_sol(card, pcl.sol_fma_per_s(_REPO, path))


@pytest.mark.parametrize("path", _GPU_CARDS, ids=lambda p: p.name)
def test_a_spec_entry_equals_the_card_derived_value(path):
    entry = _entry(path)
    if entry["source"] != "spec":
        pytest.skip("measured entry")
    fma, clock = pcl.sol_spec_fma_per_s(json.loads(path.read_text()))
    assert entry["fma_per_s"] == pytest.approx(fma, rel=1e-9)
    assert entry["clock_mhz"] == clock


def test_measured_and_spec_agree_within_five_percent_where_both_exist():
    seen = 0
    for path in _GPU_CARDS:
        entry = _entry(path)
        if "cross_check" not in entry:
            continue
        seen += 1
        fma, _clock = pcl.sol_spec_fma_per_s(json.loads(path.read_text()))
        assert entry["cross_check"]["measured_fma_per_s"] == pytest.approx(fma, rel=0.05)
    assert seen >= 1


def test_the_efficiency_is_floored_and_never_rounds_up():
    assert pcl.fmt_eff(0.9999) == "99%"
    assert pcl.fmt_eff(0.923) == "92%"
    assert pcl.fmt_eff(0.004) == "<1%"
    assert pcl.fmt_eff(1.0) == "100%"


def _tree(tmp_path, card_name="card_quadro-p2000.json"):
    (tmp_path / "benchmarks" / "perf_card").mkdir(parents=True)
    shutil.copy(_REPO / "benchmarks" / "sol_reference.json", tmp_path / "benchmarks")
    shutil.copy(_CARDS / card_name, tmp_path / "benchmarks" / "perf_card")
    return tmp_path


def _eff(root):
    return pcl.process("<!--c eff dev=p2000 tool=eagle dist=spread-->x<!--/c-->", pcl.Cards(root), "x.md")


def test_the_committed_card_renders_an_efficiency(tmp_path):
    out = _eff(_tree(tmp_path))
    assert out.endswith("%<!--/c-->") and "92%" in out


def test_a_planted_row_above_the_speed_of_light_is_refused(tmp_path):
    root = _tree(tmp_path)
    path = root / "benchmarks" / "perf_card" / "card_quadro-p2000.json"
    card = json.loads(path.read_text())
    fma = pcl.sol_fma_per_s(root, path)
    row = next(r for r in card["results"] if r["arm"] == "warp_kernel")
    row["sample_steps_per_s"] = 1.10 * fma / pcl.SOL_OPS_PER_STEP
    path.write_text(json.dumps(card))
    with pytest.raises(pcl.ClaimError, match="speed of light"):
        _eff(root)


def test_a_pending_reference_is_refused(tmp_path):
    root = _tree(tmp_path)
    ref_path = root / "benchmarks" / "sol_reference.json"
    ref = json.loads(ref_path.read_text())
    ref["devices"]["quadro-p2000"]["source"] = "pending"
    ref_path.write_text(json.dumps(ref))
    with pytest.raises(pcl.ClaimError, match="pending"):
        _eff(root)


def test_an_unknown_tool_is_refused(tmp_path):
    with pytest.raises(pcl.ClaimError, match="needs arm="):
        pcl.process(
            "<!--c eff dev=p2000 tool=nonesuch dist=spread-->x<!--/c-->", pcl.Cards(_tree(tmp_path)), "x.md"
        )
