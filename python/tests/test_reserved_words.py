# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Reserved-word absence test -- the eagle half of a two-repo lock.

Three spellings are reserved as future "aggregate kernelization" (AK)
vocabulary that must be provably ABSENT from sources until that lane is
explicitly activated::

    AggregateRegion
    aggregate_kernel
    hawk.aggregate

These are RESERVED words: they must not appear in the code generator's or
eagle's sources until AK activation. This locks the words out of **both**
trees. The code generator's half (an AST-based token matcher over its own
package) lives in its own test suite; this file is eagle's half —
previously unowned anywhere.

**Plain-text scanning, not AST.** Unlike an AST scanner (which parses each
file with :mod:`ast` and scans identifiers + token-like string literals so
that comments/docstrings/prose are exempt by construction), this file does
a line-by-line, case-insensitive **substring** scan over raw file text.
That is deliberate and it is weaker in one specific way: a comment or
docstring mentioning one of the reserved spellings would trip this scanner
where an AST scanner would exempt it as prose. This is acceptable here
because (a) eagle's C++ headers have no AST-accessible Python-style
docstring concept to exempt in the first place, (b) a plain grep is far
cheaper to reason about for a mixed Python+C++ tree, and (c) the three
reserved spellings are distinctive enough (not generic English words) that
a false positive on this codebase today is very unlikely — and if the
scanner ever does trip on a legitimate comment *documenting* the
reservation, the fix is to reword that one comment, not to build
AST-exemption machinery here. Nobody should assume this scanner has an AST
scanner's rigor; it does not, on purpose.

Scan roots (walked recursively, never an enumerated file list — a
hardcoded list silently stops covering new files, which is how this class
of gate rots):

* ``eagle/python/eagle/`` — the Python package.
* ``eagle/python/src/``    — the nanobind C++ binding source
  (``eagle_core.cu``).
* ``eagle/eagle/``         — the header-only C++ core library (``cpu/``,
  ``cuda/``, ``filtering/``, ``reduce/``, ``util/``, ...).
* ``eagle/plugin/``        — the plugin-registry protocol headers + demos
  (``plugin_registry/``, ``graph_inject/``, ``pure_inject/``).
* Anything else directly under the ``eagle/`` repo root that is source code
  (e.g. ``eagle_demo.cpp`` / ``eagle_demo.cu``) — the walk starts at the
  repo root and prunes non-source directories by name (below), rather than
  enumerating source roots, precisely so a newly added top-level source
  directory is covered automatically instead of silently skipped.

**Exclusions, explicit:**

* This test file itself (``test_reserved_words.py``) — it necessarily
  contains the three reserved spellings verbatim, as the denylist and the
  self-falsification fixtures below.
* Any directory named ``docs`` (documentation, not source) or ``tests``
  (covers both ``eagle/tests/`` and ``eagle/python/tests/`` — mirrors the
  convention of scanning core only, not the test suite itself).
* Any directory whose name starts with ``build`` (``build``, ``build_cpp``,
  ``build_cuda``, ``build_ctest``, ``build_hdrinstall``,
  ``build_cpp_install``, ``build_cuda_install``, ``python/build``, ...) —
  generated CMake/nanobind build trees, not checked-in source.
* Any hidden directory (name starts with ``.``: ``.git``, ``.cache``,
  ``.pytest_cache``, ``.ruff_cache``, ...) and ``__pycache__``.
* Any directory ending in ``.egg-info`` (packaging metadata).
* File extensions outside ``{.py, .h, .hpp, .cuh, .cu, .cc, .cpp}`` — this
  is a source-code scan, not a repository-wide grep (so ``.so``, ``.md``,
  ``.rst``, license files, etc. are never opened).

Directory pruning happens at every depth via ``os.walk``'s in-place
``dirnames`` filtering, so an excluded directory is skipped no matter how
deep it is nested (e.g. ``eagle/python/build/...`` or a stray
``__pycache__`` under any package).
"""

from __future__ import annotations

import dataclasses
import os
import pathlib

import pytest

THIS_FILE = pathlib.Path(__file__).resolve()
EAGLE_ROOT = THIS_FILE.parents[2]  # eagle/python/tests -> eagle/python -> eagle

RESERVED_WORDS = ("AggregateRegion", "aggregate_kernel", "hawk.aggregate")

SOURCE_EXTENSIONS = frozenset({".py", ".h", ".hpp", ".cuh", ".cu", ".cc", ".cpp"})

# A floor, not an equality: new source files must not break the gate, but a
# walk that silently resolves to an empty or wrong root must. Actual count
# at authoring time (see test_eagle_walk_actually_covers_the_real_tree): 72.
MIN_FILES_WALKED = 60

RATIONALE = (
    "these three spellings (AggregateRegion, aggregate_kernel, "
    "hawk.aggregate) are reserved pre-activation words for the "
    "not-yet-activated aggregate-kernelization (AK) lane and must not "
    "appear in eagle sources until that lane is explicitly activated. See "
    "the code generator's own denylist for its half of the same lock."
)


def _is_excluded_dir(name: str) -> bool:
    """Directory-name prune rule applied at every depth of the walk."""
    if name.startswith("."):
        return True
    if name == "__pycache__":
        return True
    if name.startswith("build"):
        return True
    if name in ("docs", "tests"):
        return True
    if name.endswith(".egg-info"):
        return True
    return False


def iter_eagle_source_files(root: pathlib.Path = EAGLE_ROOT):
    """Every source file under ``root``, recursively, minus excluded dirs.

    ``root`` defaults to the real eagle repo root but is a plain parameter
    precisely so the self-falsification tests can point the same walker at
    a scratch tree outside the repo instead.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not _is_excluded_dir(d))
        for fname in sorted(filenames):
            path = pathlib.Path(dirpath) / fname
            if path.suffix not in SOURCE_EXTENSIONS:
                continue
            if path.resolve() == THIS_FILE:
                continue
            yield path


@dataclasses.dataclass(frozen=True)
class Violation:
    path: pathlib.Path
    lineno: int
    word: str
    line_text: str

    def message(self) -> str:
        try:
            rel = self.path.relative_to(EAGLE_ROOT)
        except ValueError:
            rel = self.path
        snippet = self.line_text.strip()
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        return (
            f"{rel}:{self.lineno}: found reserved word {self.word!r} "
            f"({snippet!r}). {RATIONALE}"
        )


def scan_file(path: pathlib.Path) -> list[Violation]:
    """Line-by-line, case-insensitive substring scan (plain text, not AST)."""
    violations: list[Violation] = []
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return violations
    for lineno, line in enumerate(text.splitlines(), start=1):
        lowered = line.lower()
        for word in RESERVED_WORDS:
            if word.lower() in lowered:
                violations.append(Violation(path, lineno, word, line))
    return violations


def scan_tree(root: pathlib.Path) -> list[Violation]:
    violations: list[Violation] = []
    for path in iter_eagle_source_files(root):
        violations.extend(scan_file(path))
    return violations


def format_violations(violations: list[Violation]) -> str:
    body = "\n".join(v.message() for v in violations)
    return f"{len(violations)} violation(s) found:\n{body}"


# ---------------------------------------------------------------------------
# 0. Walk coverage — the assertion that keeps the rest honest
# ---------------------------------------------------------------------------


@pytest.mark.repo_local
def test_eagle_walk_actually_covers_the_real_tree():
    """The walk resolves to the real eagle tree: a file-count floor + named files.

    Without this, a walk that resolved to an empty or wrong root would make
    the denylist test below pass green while guarding nothing (same
    rationale as the code generator's own coverage test).
    """
    walked = list(iter_eagle_source_files())
    assert len(walked) >= MIN_FILES_WALKED, (
        f"eagle source walk found only {len(walked)} file(s) under "
        f"{EAGLE_ROOT} (expected >= {MIN_FILES_WALKED}). The scan scope has "
        "broken: the denylist test below is now vacuously green. Fix the "
        "walk before trusting any other result here."
    )
    rel = {p.relative_to(EAGLE_ROOT).as_posix() for p in walked}
    for expected in (
        "python/eagle/registry.py",
        "plugin/plugin_registry/registry.h",
        "eagle/cuda/Graph.h",
    ):
        assert expected in rel, (
            f"{expected} is missing from the walked eagle source file list "
            f"({len(walked)} files walked) — the scan scope no longer "
            "covers the modules this gate was built around."
        )


# ---------------------------------------------------------------------------
# 1. The real scan
# ---------------------------------------------------------------------------


@pytest.mark.repo_local
def test_reserved_words_absent_from_eagle_sources():
    """The aggregate-kernelization vocabulary must stay absent from eagle."""
    violations = scan_tree(EAGLE_ROOT)
    if violations:
        pytest.fail(format_violations(violations))


# ---------------------------------------------------------------------------
# 2. Self-falsification — the scan must actually be able to fail
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reserved_word,planted_line",
    [
        pytest.param(
            "AggregateRegion",
            "class AggregateRegion {  // planted for self-falsification\n",
            id="AggregateRegion",
        ),
        pytest.param(
            "aggregate_kernel",
            "void aggregate_kernel(double* out) {}  // planted\n",
            id="aggregate_kernel",
        ),
        pytest.param(
            "hawk.aggregate",
            '# MOD = "hawk.aggregate"  planted\n',
            id="hawk.aggregate",
        ),
    ],
)
def test_scan_bites_on_a_planted_occurrence(tmp_path, reserved_word, planted_line):
    """Prove the scan actually bites, against a scratch copy outside the repo.

    ``tmp_path`` is pytest's per-test scratch directory — never the repo.
    Runs through ``iter_eagle_source_files`` + ``scan_tree``, the same code
    path the real gate uses, so this falsifies the walk and the matcher
    together. A scan that cannot fail is worthless; this one currently
    passes trivially against the real tree (0 hits tree-wide, see
    ``test_reserved_words_absent_from_eagle_sources`` above), which is
    exactly why this proof is mandatory.
    """
    src_dir = tmp_path / "plugin"
    src_dir.mkdir()
    planted = src_dir / "injected.h"
    planted.write_text(planted_line)

    violations = scan_tree(tmp_path)

    assert violations, (
        f"the scan did NOT bite on a planted occurrence of {reserved_word!r}. "
        "This gate cannot fail and is therefore worthless."
    )
    hit = next(v for v in violations if v.word == reserved_word)
    assert hit.path == planted.resolve() or hit.path == planted
    assert hit.lineno == 1
    msg = hit.message()
    assert "injected.h" in msg
    assert reserved_word in msg
    assert "reserved" in msg
    assert "pre-activation" in msg


def test_self_falsification_excluded_dir_is_actually_pruned(tmp_path):
    """The exclusion machinery is real: a planted word inside an excluded
    directory (``tests/``) must NOT fire, while a sibling planted word in a
    normal source path DOES fire. Proves the walker isn't just "scan
    everything" dressed up as an exclusion list.
    """
    excluded_dir = tmp_path / "tests"
    excluded_dir.mkdir()
    (excluded_dir / "should_be_ignored.h").write_text(
        "class AggregateRegion {};  // inside an excluded tests/ dir\n"
    )

    included_dir = tmp_path / "plugin"
    included_dir.mkdir()
    (included_dir / "real_source.h").write_text(
        "class AggregateRegion {};  // inside a normal source dir\n"
    )

    violations = scan_tree(tmp_path)
    hit_paths = {v.path.name for v in violations}
    assert "should_be_ignored.h" not in hit_paths, (
        "the tests/ exclusion did not prune — a planted word inside an "
        "excluded directory was still reported"
    )
    assert "real_source.h" in hit_paths, (
        "the scan missed a planted word in a normal (non-excluded) source "
        "directory — the walker is broken, not just over-exclusive"
    )


def test_falsification_case_word_list_matches_denylist():
    """Guards the falsification fixture list itself against silently
    drifting out of sync with RESERVED_WORDS (e.g. a fourth reserved word
    added to the denylist without a matching falsification case).
    """
    tested = {"AggregateRegion", "aggregate_kernel", "hawk.aggregate"}
    assert tested == set(RESERVED_WORDS)
