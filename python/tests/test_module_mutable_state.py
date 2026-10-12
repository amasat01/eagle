# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A new module-level mutable (dict, list, set, deque, ...) in the ``eagle``
package fails this test until it is written into ``ALLOWED`` with the reason
it is safe to share between threads: a read-only table, or a lock that guards
it, or an idempotent memo whose every operation is a single atomic dict or set
call. ``__all__`` is exempt. The scan reads source only (no import)."""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "eagle"

MUTABLE_CALLS = {"dict", "list", "set", "bytearray", "defaultdict", "OrderedDict", "deque",
                 "Counter", "ChainMap", "WeakSet", "WeakKeyDictionary", "WeakValueDictionary",
                 "array", "local"}

TABLE = "read-only table, never mutated after import"
MEMO = "memo of a pure function: single atomic dict calls, a lost race recomputes the same value"

ALLOWED = {
    ("_active_set.py", "_REORDERING"): "guarded by _REORDER_LOCK",
    ("_counters.py", "_COUNTERS"): "guarded by _LOCK",
    ("_dispatch_door.py", "_PARTS"): "guarded by _PARTS_LOCK",
    ("_graph_reduce.py", "_OPS"): TABLE,
    ("_graph_reduce.py", "_DTYPES"): TABLE,
    ("_graph_view.py", "_STYLES"): TABLE,
    ("_graph_view.py", "_COND_FRAMES"): TABLE,
    ("_host_loop.py", "_TILES"): MEMO + ", size-capped by clear-when-full",
    ("_host_loop.py", "_RUNS"): MEMO + ", size-capped by clear-when-full",
    ("_host_loop.py", "_RUN_LOCKS"): "one lock per artifact via atomic setdefault; the lock is the guard",
    ("_layout.py", "_seen"): "warning de-duplication: atomic set add, worst case one repeated warning",
    ("_plan_deploy.py", "_AUTO_TARGETS"): TABLE,
    ("_plan_deploy.py", "_WARMED"): "atomic set add of an idempotent warm-up",
    ("_plan_deploy.py", "_BUNDLE_LOCKS"): "one lock per bundle folder via atomic setdefault; the lock is the guard",
    ("_plan_planes.py", "_BIND_TARGETS"): TABLE,
    ("_simulate.py", "_FAMILY"): TABLE,
    ("_until_done.py", "_STRUCTURES"): TABLE,
    ("exec.py", "_BY_NAME"): TABLE,
    ("gemm/capture.py", "CUBLAS_OP"): TABLE,
    ("gemm/plan.py", "_DTYPE_NAME_CACHE"): MEMO,
    ("gemm/plan.py", "_TERM_FORM_FOR_CLASS"): TABLE,
    ("graph.py", "_REGISTRY"): "guarded by _REGISTRY_LOCK",
    ("interop.py", "_external_stream_cache"): "guarded by _EXTERNAL_STREAM_LOCK",
    ("interop.py", "_EXTERNAL_STREAM_COUNTS"): "guarded by _EXTERNAL_STREAM_LOCK",
    ("interop.py", "_DTYPE_CODES"): TABLE,
    ("interop.py", "_DEVICE_TYPES"): TABLE,
    ("launch.py", "_kernel_attrs_cache"): "guarded by _KERNEL_ATTRS_LOCK (lock-free reads of immutable entries)",
    ("launch.py", "_launch_plan_cache"): "guarded by _LAUNCH_PLAN_LOCK",
    ("launch.py", "_LAUNCH_PLAN_COUNTS"): "guarded by _LAUNCH_PLAN_LOCK",
    ("marshal.py", "_coerce_stats"): "guarded by _COERCE_LOCK",
    ("marshal.py", "_zero_mask_cache"): "guarded by _ZERO_MASK_LOCK",
    ("marshal.py", "_ZERO_MASK_COUNTS"): "guarded by _ZERO_MASK_LOCK",
    ("registry.py", "_PLUGIN_MEMO"): MEMO + ", size-capped by clear-when-full",
    ("registry.py", "_PATTERN_LOADERS"): "written at provider import and under _ENTRY_POINT_LOCK; atomic dict calls",
    ("registry.py", "_LAZY_PROVIDERS"): "empty read-only map",
}


def _is_mutable(value) -> bool:
    if isinstance(value, (ast.Dict, ast.List, ast.Set, ast.DictComp, ast.ListComp, ast.SetComp)):
        return True
    if isinstance(value, ast.Call):
        f = value.func
        name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
        return name in MUTABLE_CALLS
    return False


def _module_level(body):
    for node in body:
        if isinstance(node, (ast.If, ast.Try)):
            for part in (node.body, node.orelse, *(h.body for h in getattr(node, "handlers", ()))):
                yield from _module_level(part)
        elif isinstance(node, ast.With):
            yield from _module_level(node.body)
        else:
            yield node


def found_mutables() -> set:
    found = set()
    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        for node in _module_level(ast.parse(path.read_text()).body):
            if isinstance(node, ast.Assign):
                targets, value = node.targets, node.value
            elif isinstance(node, ast.AnnAssign) and node.value is not None:
                targets, value = [node.target], node.value
            else:
                continue
            if _is_mutable(value):
                found |= {(rel, t.id) for t in targets
                          if isinstance(t, ast.Name) and t.id != "__all__"}
    return found


def test_every_module_level_mutable_is_allow_listed_with_a_reason():
    found = found_mutables()
    new = sorted(found - ALLOWED.keys())
    assert not new, (
        f"new module-level mutable(s) {new}: make them per-object state, or add each to "
        "ALLOWED in tests/test_module_mutable_state.py with the reason it is thread-safe")


def test_the_allow_list_has_no_stale_entries_and_every_entry_a_reason():
    stale = sorted(ALLOWED.keys() - found_mutables())
    assert not stale, f"ALLOWED names mutables that no longer exist: {stale}"
    assert all(len(reason) > 8 for reason in ALLOWED.values())


def test_the_scan_sees_a_planted_mutable():
    tree = ast.parse("import threading\n_cache = {}\nif True:\n    _more: list = []\nX = (1, 2)\n")
    names = [t.id for n in _module_level(tree.body) if isinstance(n, (ast.Assign, ast.AnnAssign))
             for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
             if _is_mutable(n.value)]
    assert names == ["_cache", "_more"]
