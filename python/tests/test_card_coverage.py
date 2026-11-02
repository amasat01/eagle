# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The coverage table on the performance pages: every feature names its card row, profile stage or reason, and every row of
the feature card appears in it, so a row added to the manifest without a coverage line is red."""

from __future__ import annotations

import json
import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_PAGE = _ROOT / "docs" / "content" / "performance" / "coverage.md"
_MANIFEST = _ROOT / "benchmarks" / "cards_manifest.json"
_ROWS = [line.split("|")[1:-1] for line in _PAGE.read_text().splitlines()
         if line.startswith("| ") and not line.startswith(("| group", "|---"))]
_C2 = set(json.loads(_MANIFEST.read_text())["c2_feature_card"])
_STAGE = re.compile(r"\bP[1-5]\b")


def test_every_row_has_a_card_stage_or_reason():
    assert len(_ROWS) >= 25
    for group, item, covered, reason in (tuple(c.strip() for c in row) for row in _ROWS):
        assert group and item
        assert reason, f"{item}: no reason"
        named = re.search(r"\bC[12]\b|\bcovered by\b|\bno timing row\b", covered) or _STAGE.search(covered)
        assert named, f"{item}: names no card, stage or reason"


def test_every_feature_card_row_is_covered():
    text = _PAGE.read_text()
    missing = sorted(r for r in _C2 if f"`{r}`" not in text)
    assert not missing, f"feature-card rows with no coverage line: {missing}"


def test_every_named_feature_row_is_in_the_manifest():
    named = {m for row in _ROWS for m in re.findall(r"C2 `(\w+)`", row[2])}
    assert named <= _C2, sorted(named - _C2)
