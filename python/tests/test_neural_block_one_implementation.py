# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The ``neural_block`` clause exists in
exactly ONE place.

Follows the ``test_roles_raptor_repoint.py`` AST-scan precedent: the claim is
structural, discovered by WALKING the four repos' package trees, never a
hand-passed file list (a scanner fed its own expected answers finds nothing
wrong by construction). Two independent scans:

* no ``def _validate_neural_block`` anywhere outside ``raptor``: the clause
  function lives in ``raptor.schema.blocks`` and must not exist as a second
  copy anywhere else in the family.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

# The workspace root: the sibling checkouts (eagle, raptor) live directly
# under it. This file is eagle/python/tests/test_neural_block_one_implementation.py,
# so .parents[3] is the workspace root (mirrors test_roles_raptor_repoint.py).
_WORKSPACE_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: The repos' importable top-level packages (not their tests/tools/docs) —
#: the surface the scan below walks. The scan is skipped when a sibling
#: checkout is not present next to this one.
_FAMILY_PACKAGES = {
    "eagle": _WORKSPACE_ROOT / "eagle" / "python" / "eagle",
    "raptor": _WORKSPACE_ROOT / "raptor" / "raptor",
}


def _iter_py_files(root: pathlib.Path):
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _defines_validate_neural_block(path: pathlib.Path) -> bool:
    tree = ast.parse(path.read_text(), filename=str(path))
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "_validate_neural_block"
        for node in ast.walk(tree)
    )


def test_validate_neural_block_defined_only_in_raptor():
    for repo, root in _FAMILY_PACKAGES.items():
        if not root.is_dir():
            pytest.skip(f"sibling checkout {repo} not present at {root}")
    """No ``def _validate_neural_block`` anywhere outside raptor's package
    tree — a second definition (e.g. pasted back into eagle's sidecar.py)
    would be exactly the two-copies drift this move was meant to prevent."""
    offenders = []
    for repo, root in _FAMILY_PACKAGES.items():
        if repo == "raptor":
            continue
        for path in _iter_py_files(root):
            if _defines_validate_neural_block(path):
                offenders.append(str(path))
    assert not offenders, (
        f"_validate_neural_block is defined outside raptor: {offenders}"
    )
    # Sanity: raptor DOES define it — a vacuous pass elsewhere would certify
    # nothing (the "instrument needs a known-answer case" lesson).
    assert _defines_validate_neural_block(
        _FAMILY_PACKAGES["raptor"] / "schema" / "blocks.py"
    ), (
        "sanity check failed: raptor.schema.blocks no longer defines "
        "_validate_neural_block"
    )

