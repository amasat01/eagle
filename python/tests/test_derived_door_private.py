# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The door that lets a derivative kernel into a graph is eagle.diff's alone:
no other module of the package names it (every public door refuses a
derivative)."""

import ast
import pathlib

import eagle

ROOT = pathlib.Path(eagle.__file__).parent
DOOR = {"_add_derived", "_DERIVED_DOOR"}
ALLOWED = {ROOT / "graph.py", ROOT / "diff" / "_reverse.py"}


def test_only_eagle_diff_uses_the_derived_door():
    users = set()
    for path in ROOT.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            name = getattr(node, "attr", None) or getattr(node, "id", None) or getattr(node, "name", None)
            if name in DOOR:
                users.add(path)
    assert users == ALLOWED
