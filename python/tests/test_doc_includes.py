# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle/python/tests/test_doc_includes.py — certifies every
`include`/`literalinclude` directive under `eagle/docs` resolves.

`eagle/docs/conf.py`'s `suppress_warnings` sets it to `[..., "docutils", ...]` --
the category Sphinx uses for "Include file ... not found" -- so a broken
`literalinclude` target prints one warning line and the build still exits 0
(24 of eagle's 35 surfaced warnings are pre-existing doxygen duplicate-ID
noise under the SAME category, so the suppression cannot be narrowed without
also un-suppressing that noise). This is the check instead: a plain
filesystem scan (`raptor.conformance.doc_includes`, path-free -- it takes
`docs_root` as an argument and knows nothing about eagle) that reproduces
Sphinx's own `include`/`literalinclude` path-resolution rules.

**Why here, not downstream**: the
claim "eagle's includes resolve" is eagle's own claim. It was gated only
in a downstream sibling's suite, and that placement broke silently
within one day (the sibling's isolated-venv leg could no longer resolve
`tools.check_doc_includes` at all -- the file landed blocked-at-collection
instead of running, corrupting the leg's collection pins). The check now
lives next to the code that makes this claim and reads `eagle/docs` by
same-repo relative path only -- no sibling/repo knowledge.

`repo_local` (declared in this repo's pyproject.toml):
a repository claim (docs source-as-text, never shipped in the wheel), not a
distribution claim -- deselected only in the isolated-venv/packaging leg,
always run repo-rooted (CI, dev runs).
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest
from raptor.conformance.doc_includes import scan_docs

pytestmark = pytest.mark.repo_local

_DOCS_ROOT = pathlib.Path(__file__).resolve().parents[2] / "docs"


def _generate_perf_pages():
    """Writes ``content/_generated/*.md`` (gitignored) the way the docs build
    does: ``conf.py`` loads ``_tools/gen_perf_pages.py`` as an extension whose
    builder-inited hook calls ``generate`` before any page is read. A fresh
    checkout has no such files, and the pages that include them resolve only
    after this step."""
    path = _DOCS_ROOT / "_tools" / "gen_perf_pages.py"
    spec = importlib.util.spec_from_file_location("eagle_docs_gen_perf_pages", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.generate(module.EAGLE_ROOT, _DOCS_ROOT / "content" / "_generated")


def test_eagle_docs_includes_resolve():
    assert _DOCS_ROOT.is_dir(), f"missing eagle docs tree: {_DOCS_ROOT}"
    _generate_perf_pages()
    report = scan_docs(_DOCS_ROOT)
    # Trap #6 closer: a scan pointed at an empty/mis-rooted docs dir would
    # report broken == [] vacuously. Assert directives were actually found.
    assert report.n_directives >= 1, (
        f"0 include/literalinclude directives scanned under {_DOCS_ROOT} -- "
        "an empty or mis-rooted scan must not pass silently"
    )
    assert report.broken == [], "broken include(s) in eagle/docs:\n" + "\n".join(
        str(b) for b in report.broken
    )
