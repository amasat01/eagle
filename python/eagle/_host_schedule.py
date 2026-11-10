# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Lower one :class:`eagle.Graph` stage for the host and say how it runs.

``work_share`` runs a stage's nodes in order, each spread over the host team.
``branch_tasks`` runs a fork/join stage's independent branches at the same
time, one per thread: for small batches, where a barrier per node costs more
than the node. The choice is the C++ rule ``_core.resolve_host_schedule``.
A stage of native nodes becomes one ``_core.HostStage``: one launch of eagle's
host pool (GIL released), sized by its own first launch (``HostStage.lanes``).
"""

from __future__ import annotations

from . import _core
from .graph import get_node_kind


def _native_entry(node, ctx):
    hook = get_node_kind(node.kind).native_entry
    return hook(node, ctx) if hook is not None else None


def lower_cpu_stage(node, ctx):
    """``(step, schedule, stage)`` for one top-level node: ``step`` runs the stage,
    ``schedule`` is ``"work_share"`` or ``"branch_tasks"``, ``stage`` is the
    ``_core.HostStage`` behind ``step`` (``None`` for a Python step)."""
    step = ctx.lower(node)
    branches = node.payload if node.kind == "group" else [[node]]
    stage = [[_native_entry(n, ctx) for n in branch] for branch in branches]
    entries = [e for branch in stage for e in branch]
    native_only = None not in entries
    samples = max((p.n_samples for e in entries if e for p in e[2]), default=0)
    schedule = "work_share" if node.kind != "group" else _core.resolve_host_schedule(
        samples, len(stage), _core.host_threads(), native_only)
    if not native_only:
        return step, schedule, None
    host = _core.HostStage(stage, schedule == "branch_tasks")
    return host.run, schedule, host
