# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The shared conformance corpus — Python driver.

``eagle/tests/conformance/`` holds language-neutral fixture JSONs, each paired with an
expectation file (``{level, expect, error_substring, loaders, overrides}``). TWO drivers
read that one corpus and assert IDENTICAL outcomes: this file, and the gtest
``eagle/tests/test_ConformanceCorpus.cpp``. A divergence between the C++ and Python
validators — a check present in one language and missing in the other, or the same
rejection carrying a different message — therefore becomes a mechanical test failure
rather than an inspection finding.

Each reject fixture stores ONE ``error_substring`` that BOTH drivers assert, which is
what pins message parity across the language boundary.

Validation tier only: no GPU, no cupy, no real artifact bytes. That is a property of the
product, not a convenience of the test — the shared validator / manifest pre-load checks
run BEFORE any ``RawModule`` / ``dlopen``, so a rejected artifact is refused without
loading anything . A reject case can therefore point at a stub ``.so``
path that does not exist: if the gate ever regresses to validate-after-load, the loader
reaches the stub and raises ``OSError`` instead of the expected ``ValueError``, and this
driver fails.

Two fixture levels: ``sidecar`` (``{stem}.sidecar.json``) and ``manifest``
(``{stem}.manifest.json``). Loader classes are named ABSTRACTLY where both languages
share behaviour (``shared_validator``, ``host_add_plugin``), and per-LANGUAGE
(``cpp_load_manifest`` / ``py_load_manifest``) where the two are LOCKED
languages to *different* outcomes for the same fixture (an asymmetry pinned via the
``overrides`` mechanism, keyed by loader name). A driver silently SKIPS a loader that
names the other language's exclusive class — it is that driver's job, not this one's.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

from eagle.host_launch import HostPluginLibrary
from eagle.registry import load_manifest
from eagle.sidecar import validate_sidecar

# Every row reads the repository's conformance corpus (tests/conformance), which a
# wheel does not carry.
pytestmark = pytest.mark.repo_local

CORPUS = pathlib.Path(__file__).resolve().parent.parent.parent / "tests" / "conformance"

# The corpus rows landed so far. Step 0 landed {2, 4, 6, 7, 8, 11}; a later atomic
# widening added row 1 (the `neural_block` golden + its VJP companion) and row 5 (the
# unknown `scatter_policy` acceptance case); a later volume dispatch adds rows {3, 9,
# 10, 17, 18, 19, 20, 21} (the flagged-call fixtures pin) and repairs the gap by adding
# {13, 14, 15, 16} — those four sat on disk, unnamed by this floor, since step 0. A
# corpus that silently collects nothing reports green and proves nothing — precisely the
# defect class this machinery exists to close — so both drivers assert a floor, and the
# floor names its rows.
#
# KEEP IN SYNC with test_ConformanceCorpus.cpp: the cross-language
# message contract is exactly one shared substring per row, drawn from the
# language-invariant LEADING clause of each message. Full-message renderings (set
# orderings, `got <value>` tails, etc.) are per-language presentation and are
# PERMANENTLY non-contractual — no `error_substring` may ever span one.
#
# a later package adds rows {22, 23, 24}: the exec-ref closed-set
# hardening, the neural_block integer-field uint32 cap, and the `accumulate`
# scatter_policy acceptance golden.
#
# a later phase mints row 25 (two stems, the
# row02/row02b precedent): the neural-block MANIFEST carriage's wire dangling-ref
# reject (`row25_neural_manifest_dangling_exec_ref`, a LOCKED cross-language
# asymmetry — the C++ default vs. the Python override) and the orphaned-`blocks[]`
# guard (`row25b_orphaned_blocks_on_pure_manifest`, message-parity, no override).
REQUIRED_ROWS = frozenset(
    {
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
    }
)

# A path that deliberately does not exist. Reaching it means validate-before-load
# regressed; the loader then raises OSError, not the expected ValueError.
STUB_SO = "/nonexistent/conformance-stub.so"

# Loader-class names that belong exclusively to the OTHER driver (the C++ gtest). A
# fixture pinning a cross-language asymmetry lists BOTH this
# driver's and the C++ driver's loader class in the same ``loaders`` array; each
# driver silently skips the entry that isn't its own.
_NOT_MINE = frozenset({"cpp_load_manifest"})


def _fixture_path(stem: str, level: str) -> pathlib.Path:
    suffix = "manifest" if level == "manifest" else "sidecar"
    return CORPUS / f"{stem}.{suffix}.json"


def _load_corpus():
    """Return ``[(name, level, fixture, expectation), ...]`` for every fixture.

    ``fixture`` is the parsed sidecar dict for a ``sidecar``-level row, or the
    fixture's own :class:`pathlib.Path` for a ``manifest``-level row (Python's
    :func:`eagle.registry.load_manifest` reads a path, not a dict, so a manifest
    fixture is handed through unparsed)."""
    cases = []
    for expect_path in sorted(CORPUS.glob("*.expect.json")):
        stem = expect_path.name[: -len(".expect.json")]
        expectation = json.loads(expect_path.read_text())
        level = expectation["level"]
        fixture_path = _fixture_path(stem, level)
        if not fixture_path.exists():
            continue
        fixture = (
            fixture_path
            if level == "manifest"
            else json.loads(fixture_path.read_text())
        )
        cases.append((stem, level, fixture, expectation))
    return cases


_CASES = _load_corpus()
_PARAMS = [pytest.param(c, id=c[0]) for c in _CASES] or [
    pytest.param(None, id="EMPTY-CORPUS")
]


def test_corpus_directory_exists():
    assert CORPUS.is_dir(), f"conformance corpus directory is missing: {CORPUS}"


def test_corpus_collects_its_required_rows():
    """Fail loudly if the corpus collected nothing, or lost a row that was added."""
    assert _CASES, (
        f"the conformance corpus collected ZERO fixtures from {CORPUS}. A corpus that "
        "collects nothing reports green and proves nothing — check the path and the "
        "*.expect.json / *.sidecar.json|*.manifest.json pairing."
    )
    rows = {expectation["row"] for _, _, _, expectation in _CASES}
    missing = REQUIRED_ROWS - rows
    assert not missing, (
        f"conformance corpus lost required rows: {sorted(missing)}"
    )


def test_every_fixture_pairs_with_an_expectation():
    """Every ``*.sidecar.json`` / ``*.manifest.json`` has an ``*.expect.json`` (an
    orphan would go unrun)."""
    sidecars = {p.name[: -len(".sidecar.json")] for p in CORPUS.glob("*.sidecar.json")}
    manifests = {
        p.name[: -len(".manifest.json")] for p in CORPUS.glob("*.manifest.json")
    }
    fixtures = sidecars | manifests
    expects = {p.name[: -len(".expect.json")] for p in CORPUS.glob("*.expect.json")}
    assert fixtures == expects, (
        f"unpaired conformance fixtures: fixtures without expectations "
        f"{sorted(fixtures - expects)}, expectations without fixtures "
        f"{sorted(expects - fixtures)}"
    )


def _run_loader(loader: str, level: str, fixture, name: str) -> None:
    """Drive one named loader class over ``fixture``; raise whatever it raises.

    The loader names are shared with the C++ driver so a fixture's ``loaders`` list and
    its per-loader ``overrides`` are portable across both languages. Only called for a
    loader NOT in :data:`_NOT_MINE` — the caller filters those out first.
    """
    if level == "sidecar":
        if loader == "shared_validator":
            validate_sidecar(fixture, name=name)
        elif loader == "host_add_plugin":
            # The manifest-BYPASSING entry point — the door a conformance probe found
            # open.
            HostPluginLibrary(STUB_SO, fixture)
        else:
            raise AssertionError(
                f"unknown conformance loader class {loader!r} at sidecar level"
            )
    elif level == "manifest":
        if loader == "py_load_manifest":
            load_manifest(fixture)
        else:
            raise AssertionError(
                f"unknown conformance loader class {loader!r} at manifest level"
            )
    else:
        raise AssertionError(f"unknown conformance fixture level {level!r}")



def _provider_absent(manifest_path) -> bool:
    """True when the manifest's pattern is served only by a downstream provider
    package (``neural_block``) and none is registered in this environment: the
    Python loader then stops at pattern resolution, before the check the row pins."""
    pattern = json.loads(pathlib.Path(manifest_path).read_text()).get("pattern")
    if pattern != "neural_block":
        return False
    from eagle.registry import UnknownPatternError, _resolve_pattern_loader

    try:
        _resolve_pattern_loader(pattern)
    except UnknownPatternError:
        return True
    return False


@pytest.mark.parametrize("case", _PARAMS)
def test_conformance_case(case):
    assert case is not None, (
        "the conformance corpus collected ZERO fixtures — this parametrization ran on "
        "a sentinel. See test_corpus_collects_its_required_rows."
    )
    name, level, fixture, expectation = case
    assert level in ("sidecar", "manifest"), f"{name}: unknown fixture level {level!r}"
    loaders = expectation["loaders"]
    assert loaders, f"{name}: fixture declares no loaders, so it would assert nothing"

    ran_any = False
    for loader in loaders:
        if loader in _NOT_MINE:
            continue  # the C++ driver's exclusive loader class; not this driver's job
        if level == "manifest" and _provider_absent(fixture):
            pytest.skip(f"{name}: needs a registered 'neural_block' loader "
                        "(a downstream provider package); none is installed here")
        ran_any = True
        override = expectation.get("overrides", {}).get(loader, {})
        expect = override.get("expect", expectation["expect"])
        substring = override.get("error_substring", expectation["error_substring"])

        if expect == "accept":
            _run_loader(loader, level, fixture, name)  # must not raise
        elif expect == "reject":
            assert substring, f"{name}/{loader}: a reject row needs an error_substring"
            with pytest.raises(ValueError, match=re.escape(substring)):
                _run_loader(loader, level, fixture, name)
        else:
            raise AssertionError(f"{name}: unknown expect value {expect!r}")
    assert ran_any, (
        f"{name}: every declared loader belongs to the other driver — this fixture "
        "asserts nothing in Python; check its `loaders` list"
    )
