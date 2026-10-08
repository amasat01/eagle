# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""README.md's Performance block is `tools/sync_readme_cards.py`'s own output.

The headline table and the "Measured on" list live between
``<!-- cards:begin -->``/``<!-- cards:end -->`` markers; this test re-renders
them from the committed cards with that script's own ``render_block`` and
requires the committed block to match byte for byte, so a hand edit (or a
stale block left behind by a card re-run) is caught here instead of drifting
silently. It also checks the ``stale-prose-gate`` comment just above the
block: the surrounding hand-written paragraphs (compaction, reorder, the
hardware caveat) cite numbers from the reference card by hand, and that
comment's md5 is what ties them to it.

RED: edit one number inside the ``cards:begin``/``cards:end`` block, or one
byte of the reference card without updating the comment above it, and the
matching test here fails naming the file.

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import hashlib
import importlib.util
import pathlib
import re

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
_README = _REPO / "README.md"
_SCRIPT = _REPO / "tools" / "sync_readme_cards.py"
_BEGIN, _END = "<!-- cards:begin -->", "<!-- cards:end -->"
_STALE_GATE_RE = re.compile(r"<!-- stale-prose-gate:\n(.*?)-->", re.S)
_STALE_LINE_RE = re.compile(r"^(\S+\.json):\s*(\S+)\s*$", re.M)


def _script():
    spec = importlib.util.spec_from_file_location("eagle_sync_readme_cards", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _committed_block(text: str) -> str:
    start = text.index(_BEGIN) + len(_BEGIN)
    end = text.index(_END, start)
    return text[start:end].strip("\n")


def test_readme_block_matches_a_fresh_render():
    module = _script()
    text = _README.read_text()
    assert _BEGIN in text and _END in text, (
        "README.md has no cards:begin/cards:end block")
    committed = _committed_block(text)
    fresh = module.render_block(_REPO).strip("\n")
    assert committed == fresh, (
        "README.md's cards:begin/cards:end block is not tools/sync_readme_cards.py's "
        "current render: re-run it and commit the diff instead of editing either by hand")


def test_measured_on_anchors_match_the_built_pages_hyphenated_ids():
    """Sphinx/docutils normalizes a MyST target label's underscores to
    hyphens when it emits the built page's HTML id; this is a literal
    external link (performance.html#...), never rewritten for that the way
    an in-build MyST cross-reference would be, so it must already spell the
    normalized, hyphenated form -- the anchor/underscore bug `render_block`
    matching the committed block above cannot catch on its own, since both
    sides can drift to the SAME wrong spelling together."""
    module = _script()
    rendered = module.render_block(_REPO)
    assert "#perf-card-gpu-quadro-p2000" in rendered
    assert "#perf-card-cpu-intel-xeon-w-2125" in rendered
    assert "#perf_card_gpu-" not in rendered
    assert "#perf_card_cpu-" not in rendered


def test_readme_stale_prose_gate_matches_the_reference_card():
    """The comment right above the block names the reference card this
    section's hand-written paragraphs were checked against, by md5."""
    text = _README.read_text()
    m = _STALE_GATE_RE.search(text)
    assert m, "README.md's Performance section has no stale-prose-gate comment"
    entries = dict(_STALE_LINE_RE.findall(m.group(1)))
    assert entries, "the stale-prose-gate comment has no card lines"
    for relpath, recorded in entries.items():
        if recorded == "PENDING":
            continue
        path = _REPO / relpath
        assert path.is_file(), f"{relpath} is not a committed card"
        actual = hashlib.md5(path.read_bytes()).hexdigest()
        assert actual == recorded, (
            f"{relpath} has changed (md5 {actual}, comment says {recorded}) -- "
            "re-check the Performance section's prose against the new card and "
            "update the stale-prose-gate comment")
