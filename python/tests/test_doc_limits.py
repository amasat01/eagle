# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The limits the interop pages state are tracked in one registry and expire.

``docs/content/_shared/limits.json`` holds every limit (``id``, ``text``, ``way_out``, ``cause``: ``ours``
or ``framework``, ``lifted_in``: a release or ``none``); a page states one as
``- <!-- limit: id --> text`` and the text must equal the registry's. The
tests fail when a page states a limit the registry lacks or words differently,
when the registry holds one no page states, and when the package version has
reached an entry's ``lifted_in`` -- the release that lifts a limit cannot ship
until every page that states it is edited.

``repo_local``: docs source-as-text, never shipped in the wheel.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import tomllib

import pytest

pytestmark = pytest.mark.repo_local

ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "docs/content/_shared/limits.json"
PAGES = sorted((ROOT / "docs/content/interop").glob("*.ipynb")) + sorted(
    (ROOT / "docs/content/userguide/tutorials").glob("*.ipynb"))
LINE = re.compile(r"^- <!-- limit: (\S+) --> (.*)$")


def _registry():
    return {e["id"]: e for e in json.loads(REGISTRY.read_text(encoding="utf-8"))}


def _stated():
    """``[(page name, id, text)]`` for every marked line in a notebook's markdown."""
    found = []
    for page in PAGES:
        for cell in json.loads(page.read_text(encoding="utf-8"))["cells"]:
            if cell["cell_type"] == "markdown":
                for line in "".join(cell["source"]).splitlines():
                    if m := LINE.match(line):
                        found.append((page.name, m[1], m[2]))
    return found


def _version():
    text = (ROOT / "python/pyproject.toml").read_bytes()
    return tomllib.loads(text.decode())["project"]["version"]


def _tuple(version):
    return tuple(int(re.match(r"\d+", p)[0]) for p in version.split("."))  # 0.6.2rc2 counts as 0.6.2


def _expired(registry, version):
    return [i for i, e in registry.items()
            if str(e["lifted_in"]) != "none" and _tuple(version) >= _tuple(str(e["lifted_in"]))]


def test_every_stated_limit_is_registered_with_the_same_text():
    registry = _registry()
    bad = [f"{page}: {i}" for page, i, text in _stated()
           if i not in registry or registry[i]["text"] != text]
    assert not bad, f"limit lines missing from {REGISTRY.name} or worded differently: {bad}"


def test_every_registered_limit_is_stated_on_a_page():
    stated = {i for _, i, _ in _stated()}
    unused = sorted(set(_registry()) - stated)
    assert not unused, f"registered limits no page states (remove the entry): {unused}"


def test_no_limit_outlives_its_lifted_in_release():
    assert not _expired(_registry(), _version()), (
        "this release lifts a limit: edit every page that states it and drop the entry")


def test_the_expiry_check_fires_when_the_version_reaches_lifted_in(monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_version", lambda: "0.7.0")
    with pytest.raises(AssertionError, match="lifts a limit"):
        test_no_limit_outlives_its_lifted_in_release()
