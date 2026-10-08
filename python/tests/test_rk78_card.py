# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The adaptive-step (RKF7(8)) card's table is the committed card, re-rendered.

``benchmarks/rk78_card/rk78_card.py`` writes ``card_<device>.json`` (every
number) and ``card_<device>.md`` (its tables), both from one run. These tests
re-render every committed card JSON with the script's own
:func:`render_markdown` and require the committed Markdown to match it byte
for byte; require each card to name the committed script by md5; require it
to cover every arm at every N, with an agreement entry, a memory row and a
compile-time entry per arm; and require the card's recorded lines of code per
arm to be the ones the script's code markers give today.

The script-level checks (balanced code markers, every arm's code blocks
present, the eagle arms' snippets deploying their kernel through
``eagle.deploy``/``eagle.simulate``, the FLOP count derived from the scheme
code) run with no card.

RED: edit one number in a committed ``card_*.md`` (or ``card_*.json``), or
edit the script without re-running it, and this fails naming the file.

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import pathlib

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
_CARD_DIR = _REPO / "benchmarks" / "rk78_card"
_SCRIPT = _CARD_DIR / "rk78_card.py"


def _script():
    spec = importlib.util.spec_from_file_location("eagle_rk78_card", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cards():
    cards = sorted(_CARD_DIR.glob("card_*.json"))
    if not cards:
        pytest.skip(f"no committed card_*.json under {_CARD_DIR} yet (the timed run "
                    "writes it)")
    return cards


def test_code_markers_cover_every_arm():
    """Every ``# >>> code:NAME`` has its ``# <<< code:NAME``, and every arm's
    listed blocks exist and are non-empty."""
    module = _script()
    text = _SCRIPT.read_text()
    opens = text.count("# >>> code:")
    closes = text.count("# <<< code:")
    assert opens == closes, f"{opens} opening vs {closes} closing code markers"
    blocks = module.code_blocks()
    assert "shared_scheme" in blocks
    assert set(module.ARM_CODE) == set(module.ARMS)
    for arm, names in module.ARM_CODE.items():
        for name in names:
            assert name in blocks, f"{arm}: no code block {name!r}"
            assert any(ln.strip() for ln in blocks[name]), f"{name!r} is empty"
    # the eagle arms show the user-facing doors: eagle.deploy / eagle.simulate
    for arm in module.EAGLE_GPU_ARMS + ("eagle_cpu",):
        text = "\n".join(ln for b in module.ARM_CODE[arm] for ln in blocks[b])
        assert "eagle.deploy(" in text or "eagle.simulation(" in text, (
            f"{arm}: its snippet does not deploy its kernel")


def test_flop_count_comes_from_the_scheme():
    """The per-attempt FLOP count is the counted scheme plus the controller,
    and the scheme count grows with the tableau's non-zero entries."""
    module = _script()
    scheme, controller = module._count_flops()
    assert module.FLOPS_PER_ATTEMPT == scheme + controller
    nnz = sum(1 for row in module.RK_A for a in row if a != 0)
    # at least one multiply and one add per non-zero a_ij per state component
    assert scheme > 2 * 6 * nnz


def test_committed_tables_match_their_cards():
    render = _script().render_markdown
    for card_path in _cards():
        card = json.loads(card_path.read_text())
        table_path = card_path.with_suffix(".md")
        assert table_path.is_file(), f"{card_path.name}: no rendered {table_path.name}"
        assert table_path.read_text() == render(card), (
            f"{table_path.name} is not the rendering of {card_path.name}: re-run "
            "benchmarks/rk78_card/rk78_card.py instead of editing either file")


def test_cards_name_their_script_and_are_complete():
    module = _script()
    md5 = hashlib.md5(_SCRIPT.read_bytes()).hexdigest()
    for card_path in _cards():
        card = json.loads(card_path.read_text())
        assert card["script"] == module.SCRIPT_REL
        assert card["script_md5"] == md5, (
            f"{card_path.name} was produced by a different rk78_card.py: re-run it")
        assert set(card["arms"]) == set(module.ARMS), f"{card_path.name}: arms differ"
        expected = {(n, a) for n in card["ns"] for a in card["arms"]}
        got = {(r["n"], r["arm"]) for r in card["results"]}
        assert got == expected, f"{card_path.name}: result rows do not cover the matrix"
        assert card["method"]["repetitions"] >= 5
        for extra in card["per_n"]:
            assert set(extra["agreement"]) == set(card["arms"])
            for arm, a in extra["agreement"].items():
                assert a["max_error_over_bound"] < 1.0, (
                    f"{card_path.name}: {arm} at N={extra['n']} exceeds its truth "
                    "bound")
        mem = {(r["n"], r["arm"]): r for r in card["memory"]["rows"]}
        assert set(mem) == expected, (
            f"{card_path.name}: memory rows do not cover the rows")
        for (n, arm), r in mem.items():
            assert (r["device_peak_bytes"] is not None) == (arm in module.GPU_ARMS)
            assert r["host_peak_bytes"] is not None
        comp = card["compile_time"]
        covered = {a for r in comp["rows"] for a in r["arms"]} | set(comp["none"])
        assert covered == set(module.ARMS), (
            f"{card_path.name}: compile-time table misses "
            f"{sorted(set(module.ARMS) - covered)}")
        assert set(card["fit"]["arms"]) == set(module.ARMS)


def test_card_line_counts_match_the_script():
    """The card's lines of code per arm are what the script's markers give."""
    module = _script()
    now = module.code_line_counts()
    for card_path in _cards():
        card = json.loads(card_path.read_text())
        assert card["code_lines"]["arms"] == now["arms"], (
            f"{card_path.name}: recorded lines of code differ from the script's "
            "markers")
