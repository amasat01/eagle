# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The registered pattern-loader hook (registry inversion).

``eagle.neural`` was extracted in full to a standalone downstream
package, and ``_LAZY_PROVIDERS`` is now permanently ``{}`` (registry.py carries no
in-tree-but-optional provider left to lazy-import by name). That package's own suite
exercises the "neural_block" dispatch
end-to-end through its explicit registration leg; THIS suite covers what is
specific to the hook mechanism itself, independent of any one provider:

* item 2 (AST half): registry.py carries zero ``neural`` import statements.
* item 3: an unresolvable pattern fails LOUD with ``UnknownPatternError`` naming the
  pattern, never silently falling back — exercised both against the private resolver
  directly and end-to-end through :func:`eagle.registry.load_manifest` (the REAL
  post-extraction state: lazy-default map empty, nothing registered in THIS process,
  no entry-point provider installed).
* item 4a: explicit registration (:func:`eagle.registry.register_pattern_loader`) is
  the first-priority discovery leg and is honored by the resolver.

Item 4b (entry-point dual discovery in a FRESH install) is a one-off environment-level
verification, not a repeatable in-suite pytest (a committed test here always runs
against an already-editable-installed eagle, so it cannot exercise "before eagle is
ever imported" the way a truly fresh install can).
"""

from __future__ import annotations

import ast
import json
import pathlib

import pytest

from eagle import registry
from eagle.abi import ABI_VERSION
from eagle.registry import UnknownPatternError, load_manifest, register_pattern_loader

REGISTRY_PY = pathlib.Path(registry.__file__)


# --------------------------------------------------------------------------- #
# item 2 (AST half): zero `neural` import statements in registry.py.
# --------------------------------------------------------------------------- #
def _imported_names(tree: ast.Module) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.append(module)
            names.extend(f"{module}.{alias.name}" for alias in node.names)
    return names


def test_registry_has_zero_neural_import_statements():
    tree = ast.parse(REGISTRY_PY.read_text())
    names = _imported_names(tree)
    hits = [n for n in names if "neural" in n]
    assert not hits, (
        f"registry.py must carry zero `neural` import statements (AST-level); "
        f"found {hits} among {names}"
    )


# --------------------------------------------------------------------------- #
# item 4a: explicit registration is the first-priority discovery leg.
# --------------------------------------------------------------------------- #
def test_register_pattern_loader_is_explicit_and_resolves():
    calls = []

    def _loader(manifest, path, directory):
        calls.append((manifest, path, directory))
        return "sentinel-result"

    register_pattern_loader("_r2_test_explicit_pattern", _loader)
    try:
        resolved = registry._resolve_pattern_loader("_r2_test_explicit_pattern")
        assert resolved is _loader
        assert resolved({"pattern": "_r2_test_explicit_pattern"}, "p", "d") == (
            "sentinel-result"
        )
        assert calls == [({"pattern": "_r2_test_explicit_pattern"}, "p", "d")]
    finally:
        registry._PATTERN_LOADERS.pop("_r2_test_explicit_pattern", None)


def test_registered_loader_takes_priority_over_a_stale_lazy_default(monkeypatch):
    """Registered-first: even if a (bogus) lazy-default entry exists for the same
    pattern, an already-registered loader must win without importing anything."""

    def _loader(manifest, path, directory):
        return "explicit-wins"

    monkeypatch.setitem(
        registry._LAZY_PROVIDERS,
        "_r2_test_priority_pattern",
        "a.module.that.does.not.exist",
    )
    register_pattern_loader("_r2_test_priority_pattern", _loader)
    try:
        assert registry._resolve_pattern_loader("_r2_test_priority_pattern") is _loader
    finally:
        registry._PATTERN_LOADERS.pop("_r2_test_priority_pattern", None)


# --------------------------------------------------------------------------- #
# item 3: fail-loud, never silently skip.
# --------------------------------------------------------------------------- #
def test_unknown_pattern_error_is_a_key_error():
    assert issubclass(UnknownPatternError, KeyError)


def test_resolve_pattern_loader_fails_loud_naming_the_pattern(monkeypatch):
    monkeypatch.setattr(registry, "_LAZY_PROVIDERS", {})
    monkeypatch.setattr(registry, "_PATTERN_LOADERS", {})
    monkeypatch.setattr(registry, "_entry_points_scanned", False)
    with pytest.raises(UnknownPatternError, match="_r2_never_registered_pattern"):
        registry._resolve_pattern_loader("_r2_never_registered_pattern")


def test_load_manifest_fails_loud_for_neural_block_with_nothing_resolvable(
    tmp_path, monkeypatch
):
    """Current reality (the lazy-default map is empty by default now, so this
    monkeypatching is belt-and-suspenders, not a simulation): clear registration +
    no entry-point provider (true in this sys.path dev env regardless, since no
    optional provider is imported anywhere in this process) must fail LOUD through the
    real ``load_manifest`` entry point — never silently fall back to the generic "not a
    supported plugin family" message, which stays reserved for patterns outside
    RECOGNIZED_PATTERNS entirely (see
    test_schema_hardening.py::test_frozen_field_pattern_rejects_unknown_at_manifest_level,
    UNCHANGED by this)."""
    monkeypatch.setattr(registry, "_LAZY_PROVIDERS", {})
    monkeypatch.setattr(registry, "_PATTERN_LOADERS", {})
    monkeypatch.setattr(registry, "_entry_points_scanned", True)

    manifest = {
        "schema_version": 1,
        "pattern": "neural_block",
        "aether_abi": ABI_VERSION,
        "plugins": [],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(UnknownPatternError, match="neural_block"):
        load_manifest(path)
