# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle.roles' schema constants re-point at raptor.

Verifies, for every re-pointed name: (a) VALUE equality with raptor's copy (the
existing ``test_verbatim_against_eagle.py`` on raptor's side already does this
direction); (b) OBJECT IDENTITY (``is``) with raptor's own object, the acceptance
gate's "re-exports verified is-identical to raptor's objects for every re-pointed
name".

``SCHEMA_VERSION`` is a further special case: eagle.roles now DERIVES it
(``SCHEMA_VERSION = manifest.SCHEMA_VERSION``, an import, not a copy), which makes
an equality/identity assert on it TAUTOLOGICAL — it can no longer fail no matter how
raptor's value drifts, because both names are the same object by construction. The
replacement, ``test_schema_version_has_exactly_one_integer_literal_home`` below, is
a STRUCTURAL test instead: an AST scan of the scanned repos' Python packages, which
fails if the derivation is ever undone (a literal reintroduced in ``roles.py``) or a
third undeclared copy appears anywhere. It discovers its targets by WALKING the
package trees, never from a hand-passed file list (a scanner fed its own expected
answers finds nothing wrong by construction).

Purely-eagle vocabulary (``LAUNCH_CERTIFIED_PATTERNS``, ``ROLES``) has no raptor
counterpart; ``MANIFEST_FORMATS`` and ``RECOGNIZED_PATTERNS`` are raptor's own
objects.

``EXEC_REF_KINDS`` was re-pointed too: eagle's copy had exactly one reader, its
own ``neural_block`` clause, and that clause moved to ``raptor.schema.blocks``. A
vocabulary twin with no reader is dead weight that can only drift, so it is now a
re-point like the neural-family names — pinned by
``test_exec_ref_kinds_is_repointed_at_raptor`` below, which is the successor to the
``is not`` assert this file used to carry.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from raptor.schema import blocks, manifest

import eagle.roles as roles

ROLES_PY = pathlib.Path(roles.__file__)

# The workspace root: sibling checkouts (eagle, raptor) live directly under
# it. This file is eagle/python/tests/test_roles_raptor_repoint.py, so
# .parents[3] is the workspace root.
_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: The two public repos' importable top-level packages (not their
#: tests/tools/docs) — the surface the structural scan below walks. Private
#: sibling repos are deliberately not walked here (not present in this
#: checkout layout) and are skipped rather than required.
_FAMILY_PACKAGES = {
    "eagle": _WORKSPACE_ROOT / "eagle" / "python" / "eagle",
    "raptor": _WORKSPACE_ROOT / "raptor" / "raptor",
}

#: The one SANCTIONED literal home: raptor's own number of record. Any
#: OTHER integer-literal ``SCHEMA_VERSION`` binding anywhere in the scanned
#: packages — including a reintroduced one in ``eagle.roles`` — is a
#: regression this test must catch.
_SANCTIONED_LITERAL_HOMES = {
    _FAMILY_PACKAGES["raptor"] / "schema" / "manifest.py",
}


def _integer_literal_schema_version_bindings() -> set[pathlib.Path]:
    """AST-walk every ``.py`` file under the scanned repos' packages (discovered by
    directory walk, never a hand-passed list) for a module-level
    ``SCHEMA_VERSION = <int literal>`` binding. Post-derivation, eagle's
    ``SCHEMA_VERSION = manifest.SCHEMA_VERSION`` is an ``ast.Attribute`` value, not
    an ``ast.Constant`` int, so it does not match — that asymmetry is exactly what
    makes a reintroduced literal mechanically detectable."""
    hits: set[pathlib.Path] = set()
    for package_root in _FAMILY_PACKAGES.values():
        for path in sorted(package_root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in tree.body:
                if not (
                    isinstance(node, ast.Assign)
                    and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "SCHEMA_VERSION"
                ):
                    continue
                value = node.value
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, int)
                    and not isinstance(value.value, bool)
                ):
                    hits.add(path)
    return hits


def test_schema_version_has_exactly_one_integer_literal_home():
    """Deriving eagle.roles.SCHEMA_VERSION from raptor makes the
    raptor<->eagle-Py edge unrepresentable — an integer-literal SCHEMA_VERSION may
    now exist in exactly one place among the packages scanned here: raptor's
    number of record. A reintroduced literal in roles.py (or any other
    undeclared copy) changes this set and REDs here.

    Deliberately NOT also pinned by an ``is``/``==`` identity assert on
    ``roles.SCHEMA_VERSION`` vs ``manifest.SCHEMA_VERSION``: CPython's small-int
    cache (-5..256) makes a reintroduced literal ``SCHEMA_VERSION = 1`` compare
    identical to raptor's object too (the exact confound an earlier version of
    this file's docstring already flagged), so that assert cannot fail in the one
    scenario this test exists to catch — decorative, not a guard. This structural
    scan is the real one."""
    for name, package_root in _FAMILY_PACKAGES.items():
        if not package_root.is_dir():
            pytest.skip(f"{name} sibling checkout not present at {package_root}")
    assert _integer_literal_schema_version_bindings() == _SANCTIONED_LITERAL_HOMES


def test_neural_family_constants_are_identical_objects_to_raptor():
    for name in (
        "NEURAL_REQUIRED_FIELDS",
        "NEURAL_EXEC_REF_FIELDS",
        "NEURAL_FORBIDDEN_FIELDS",
        "SCATTER_POLICIES",
        "BUFFER_KINDS",
    ):
        eagle_obj = getattr(roles, name)
        raptor_obj = getattr(blocks, name)
        assert eagle_obj is raptor_obj, (
            f"eagle.roles.{name} is not raptor.schema.blocks.{name}"
        )


def test_recognized_patterns_sources_neural_block_from_raptor():
    assert roles.RECOGNIZED_PATTERNS == {"vector", "pure", blocks.NEURAL_BLOCK_PATTERN}
    assert roles.RECOGNIZED_PATTERNS == manifest.BASE_PATTERNS | {
        blocks.NEURAL_BLOCK_PATTERN
    }
    # Source-level: the set is raptor's, never a second hand-spelled literal.
    text = ROLES_PY.read_text()
    assert "blocks.ALL_PATTERNS" in text
    assert '"vector", "pure", "neural_block"' not in text


def test_exec_ref_kinds_is_repointed_at_raptor():
    """Supersedes this file's original ``EXEC_REF_KINDS`` pin.

    ``EXEC_REF_KINDS`` was originally left eagle-owned as "schema-general, not
    neural-family" vocabulary, and this test used to certify exactly that — that
    eagle's copy was deliberately NOT raptor's object. The one reader of eagle's
    copy was its own ``neural_block`` clause; that clause moved to
    ``raptor.schema.blocks`` and reads raptor's copy, so eagle's literal had
    no reader left. A hand-spelled vocabulary twin nobody reads is a
    drift class, so it is now re-pointed. The successor claim is the re-point itself:
    same VALUE **and** the same object."""
    assert roles.EXEC_REF_KINDS == manifest.EXEC_REF_KINDS
    assert roles.EXEC_REF_KINDS is manifest.EXEC_REF_KINDS


def test_purely_eagle_vocabulary_stays_un_repointed():
    """The schema vocabulary raptor defines is raptor's object in eagle
    (``MANIFEST_FORMATS``); LAUNCH_CERTIFIED_PATTERNS/ROLES have no raptor
    counterpart at all."""
    assert roles.MANIFEST_FORMATS is manifest.MANIFEST_FORMATS
    assert roles.RECOGNIZED_PATTERNS is blocks.ALL_PATTERNS
    assert not hasattr(blocks, "LAUNCH_CERTIFIED_PATTERNS")
    assert not hasattr(manifest, "ROLES")
