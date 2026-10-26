# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""eagle allocates its arrays through ``aether`` (``eagle/_alloc.py``), never straight from cupy.

AST gate over ``python/eagle``. Two rules, both RED on a regression:

1. No call ``<cupy|cp|xp|self._xp|self._cupy>.<empty|zeros|full|ones|empty_like|zeros_like|
   ones_like>(...)`` and no ``memsetAsync`` outside ``_alloc.py`` (the numpy host path) and one
   cupy warm-up in ``gemm/capture.py``.
2. ``asarray`` (input coercion: upload or cast of what a caller passed) is counted per file. Those
   are the ones the 0.7.2 launch move replaces with ``aether.asarray``; the table only shrinks, and a
   stale entry is RED too."""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1] / "eagle"
ALLOC = {"empty", "zeros", "full", "ones", "empty_like", "zeros_like", "ones_like"}
BASES = {"cupy", "cp", "xp", "_xp", "_cupy"}

#: file -> allocation calls allowed (host path / warm-up)
ALLOC_ALLOWED = {"_alloc.py": 3, "gemm/capture.py": 1}
#: file -> ``asarray`` coercions that remain until 0.7.2
ASARRAY_ALLOWED = {
    "_alloc.py": 3, "_layout.py": 1, "_plan_planes.py": 5, "_simulate.py": 1, "_until_done.py": 2,
    "diff/_reverse.py": 1, "diff/_run.py": 8, "diff/events.py": 1, "interop.py": 4, "marshal.py": 1,
    "plan.py": 1,
}


def _base(fn):
    value = fn.value
    if isinstance(value, ast.Name):
        return value.id
    if isinstance(value, ast.Attribute):
        return value.attr
    return None


def _count(path):
    alloc = asarray = 0
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            fn = node.func
            if fn.attr == "memsetAsync" or (fn.attr in ALLOC and _base(fn) in BASES):
                alloc += 1
            elif fn.attr == "asarray" and _base(fn) in BASES:
                asarray += 1
    return alloc, asarray


def test_no_cupy_allocation_outside_the_allowance():
    problems = []
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        alloc, asarray = _count(path)
        if alloc != ALLOC_ALLOWED.get(rel, 0):
            problems.append(f"{rel}: {alloc} direct allocation/memset call(s), allowed {ALLOC_ALLOWED.get(rel, 0)}")
        if asarray != ASARRAY_ALLOWED.get(rel, 0):
            problems.append(f"{rel}: {asarray} asarray call(s), allowed {ASARRAY_ALLOWED.get(rel, 0)}")
    assert not problems, "\n".join(problems)
