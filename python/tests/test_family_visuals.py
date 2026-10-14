# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The family-visuals bundle under ``docs/_static/family_visuals/`` is the landing site's own output.

``sync_family_visuals.py`` (in the landing site) writes the bundle and ``LOCK.json``, which records the md5 of
every bundle file, of the performance cards the site generated from, and of the generator. This test checks:

* always: every bundle file matches ``LOCK.json`` byte for byte, nothing unlisted sits in the folder, every
  script and stylesheet ``assets.json`` names exists, and every fragment a docs page includes exists;
* with ``RAPTOR_SITE_ROOT`` set (a checkout of the landing site): ``LOCK.json``'s generator md5s match the
  site's files and the generator's own ``--check`` finds this bundle current;
* with ``RAPTOR_EAGLE_ROOT`` set (an eagle checkout): ``LOCK.json``'s card md5s match the cards.

A set-but-unusable root fails rather than skips.

RED: edit one byte of a bundle file and the first test fails naming it; re-run a card, or edit the site's
generator, without re-running ``sync_family_visuals.py`` and the matching test fails.

``repo_local``: reads repository files, never shipped in the wheel.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.repo_local

_REPO = pathlib.Path(__file__).resolve().parents[2]
_BUNDLE = _REPO / "docs" / "_static" / "family_visuals"
_PROFILE = "eagle"
_REGEN = "python sync_family_visuals.py --site-root <landing site> --target eagle=<this repo>"


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _lock() -> dict:
    return json.loads((_BUNDLE / "LOCK.json").read_text())


def _env_root(name: str) -> pathlib.Path | None:
    raw = os.environ.get(name)
    if not raw:
        return None
    root = pathlib.Path(raw)
    assert root.is_dir(), f"{name}={raw} is not a directory"
    return root


def test_bundle_files_match_the_lock():
    lock = _lock()
    present = {p.relative_to(_BUNDLE).as_posix() for p in _BUNDLE.rglob("*") if p.is_file()} - {"LOCK.json"}
    assert present == set(lock["files"]), (
        f"bundle folder and LOCK.json disagree: {sorted(present ^ set(lock['files']))}; regenerate with: {_REGEN}")
    for name, want in lock["files"].items():
        assert _md5((_BUNDLE / name).read_bytes()) == want, f"{name} differs from LOCK.json; regenerate with: {_REGEN}"


def test_assets_json_names_files_that_exist():
    assets = json.loads((_BUNDLE / "assets.json").read_text())
    for name in assets["css"] + assets["js"]:
        assert (_BUNDLE / name).is_file(), f"assets.json names {name}, which is not in the bundle"


def test_included_fragments_exist():
    docs = _REPO / "docs"
    seen = 0
    for page in list(docs.rglob("*.md")) + list(docs.rglob("*.rst")):
        if "_build" in page.parts:
            continue
        for rel in re.findall(r":file:\s*(\S*family_visuals/\S+)", page.read_text()):
            seen += 1
            assert (page.parent / rel).is_file(), f"{page.relative_to(_REPO)} includes {rel}, which does not exist"
    assert seen, "no docs page includes a family-visuals fragment"


def test_lock_matches_the_generator_and_site():
    site = _env_root("RAPTOR_SITE_ROOT")
    if site is None:
        pytest.skip("RAPTOR_SITE_ROOT not set")
    script = next((c for c in (site / "tools" / "sync_family_visuals.py", site / "sync_family_visuals.py") if c.is_file()), None)
    assert script is not None, f"no sync_family_visuals.py under {site}"
    for name, want in _lock()["generator"].items():
        path = script if name == "sync_family_visuals.py" else site / name
        assert path.is_file() and _md5(path.read_bytes()) == want, f"generator file {name} changed; regenerate with: {_REGEN}"
    done = subprocess.run([sys.executable, str(script), "--check", "--site-root", str(site), "--target", f"{_PROFILE}={_REPO}"],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr or done.stdout


def test_lock_matches_the_cards():
    eagle = _env_root("RAPTOR_EAGLE_ROOT")
    if eagle is None:
        pytest.skip("RAPTOR_EAGLE_ROOT not set")
    for rel, want in _lock()["cards"].items():
        card = eagle / rel
        assert card.is_file() and _md5(card.read_bytes()) == want, f"card {rel} changed; regenerate with: {_REGEN}"
