#!/usr/bin/env python3
# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Every public name (python/eagle/__init__.py __all__) is named in at least one test under python/tests.

A static check. Names that predate it sit in tools/untested_public_names.txt ("name  # reason"); the list may only
shrink: an entry that is now tested, or no longer public, is an error (delete the line), and with --base REF an
entry added since REF is one too.   usage: public_name_coverage.py [--base REF]"""
import ast
import re
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent.parent
allow_file = root / "tools" / "untested_public_names.txt"


def names_of(text):
    return {ln.split("#")[0].strip() for ln in text.splitlines() if ln.split("#")[0].strip()}


tree = ast.parse((root / "python/eagle/__init__.py").read_text())
public = set()
for node in tree.body:
    if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "__all__" for t in node.targets):
        public = {e.value for e in node.value.elts}
tests = "\n".join(p.read_text() for p in (root / "python/tests").rglob("*.py"))
untested = {n for n in public if not re.search(rf"\b{re.escape(n)}\b", tests)}
allowed = names_of(allow_file.read_text())
errors = [f"public name {n!r} is named by no test: add a test, or (for a name that predates this check) never a new line in {allow_file.name}"
          for n in sorted(untested - allowed)]
errors += [f"{n!r} is in {allow_file.name} but {'is tested now' if n in public else 'is no longer public'}: delete the line"
           for n in sorted(allowed - untested)]
if "--base" in sys.argv:
    ref = sys.argv[sys.argv.index("--base") + 1]
    old = subprocess.run(["git", "show", f"{ref}:tools/{allow_file.name}"], cwd=root, capture_output=True, text=True)
    if old.returncode == 0:
        errors += [f"{n!r} was added to {allow_file.name}; the list may only shrink" for n in sorted(allowed - names_of(old.stdout))]
print(f"public names: {len(public)}, untested and allowed: {len(untested & allowed)}")
for e in errors:
    print("FAIL", e, file=sys.stderr)
sys.exit(1 if errors else 0)
