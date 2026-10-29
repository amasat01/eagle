# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""eagle imports no numpy or cupy outside ``frameworks/``.

An AST walk over ``python/eagle``: no ``import``/``from`` of either library, anywhere (module level or inside a
function), and no ``importlib.import_module``/``__import__`` of them; the framework bridges under ``frameworks/``
import them lazily, inside functions, never at module level. numpy is not a required dependency. A subprocess with
CUDA hidden and numpy blocked imports eagle and runs host array ops."""

import ast
import os
import pathlib
import subprocess
import sys


import eagle

ROOT = pathlib.Path(eagle.__file__).resolve().parent     # the package under test, installed or in-tree
BANNED = {"numpy", "cupy"}


def _hits(tree):
    """``(line, module, at_module_level)`` of every import of a banned library."""
    out = []

    def visit(node, top):
        for child in ast.iter_child_nodes(node):
            inner = top and not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
            if isinstance(child, ast.Import):
                out.extend([(child.lineno, a.name, top) for a in child.names if a.name.partition(".")[0] in BANNED])
            elif isinstance(child, ast.ImportFrom) and child.level == 0 and (child.module or "").partition(".")[0] in BANNED:
                out.append((child.lineno, child.module, top))
            elif isinstance(child, ast.Call):
                fn = child.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
                if name in ("import_module", "__import__") and child.args and isinstance(child.args[0], ast.Constant) \
                        and str(child.args[0].value).partition(".")[0] in BANNED:
                    out.append((child.lineno, str(child.args[0].value), top))
            visit(child, inner)

    visit(tree, True)
    return out


def _scan():
    files = sorted(ROOT.rglob("*.py"))
    assert len(files) > 50 and (ROOT / "frameworks").is_dir(), "the walk found no package: a gate that scans nothing passes"
    for path in files:
        yield path.relative_to(ROOT).as_posix(), _hits(ast.parse(path.read_text()))


def test_nothing_outside_frameworks_imports_numpy_or_cupy():
    bad = [f"{rel}:{line} imports {mod}" for rel, hits in _scan() if not rel.startswith("frameworks/") for line, mod, _t in hits]
    assert not bad, "\n".join(bad)


#: bridges whose framework (jax, warp) itself requires numpy and whose dtype tables are numpy dtypes at import time
MODULE_LEVEL_NUMPY = {"frameworks/jax.py", "frameworks/warp.py", "frameworks/_kernel_planes.py"}


def test_the_framework_bridges_import_numpy_and_cupy_lazily_only():
    bad = [f"{rel}:{line} imports {mod} at module level" for rel, hits in _scan() if rel.startswith("frameworks/")
           for line, mod, top in hits if top and not (rel in MODULE_LEVEL_NUMPY and mod == "numpy")]
    assert not bad, "\n".join(bad)


def test_numpy_is_not_a_required_dependency():
    from importlib import metadata

    required = [r for r in (metadata.requires("raptor-eagle") or []) if "extra ==" not in r]
    assert required and not [r for r in required if r.lower().startswith(("numpy", "cupy"))], required


_HOST_OPS = """
import sys
sys.modules['numpy'] = sys.modules['cupy'] = None
import eagle
import eagle.arrays as xp
a = xp.asarray([3, 1, 2], xp.int32)
assert xp.add(a, a).tolist() == [6, 2, 4] and xp.cumsum(a).tolist() == [3, 4, 6]
assert float(xp.full((4,), 2.5, xp.float64).sum()) == 10.0
assert xp.take(a, xp.asarray([2, 0], xp.int32)).tolist() == [2, 3]
a[1:] = 9
assert a.tolist() == [3, 9, 9]
a[0:1] = xp.asarray([7], xp.int32)
assert a.tolist() == [7, 9, 9]
assert not hasattr(eagle, 'to_cupy') and not hasattr(eagle, 'np_dtype')
print('OK')
"""


def test_a_cuda_hidden_process_with_numpy_blocked_imports_eagle_and_runs_host_ops():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, "-I", "-c", _HOST_OPS], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and r.stdout.strip() == "OK", r.stdout + r.stderr
