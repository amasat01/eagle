# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle._counters — process-global, monotonic static-graph counters.

Four counters, each bumped exactly once per EVENT (never per byte of work the
event does, and never on a failed attempt — see each hook site's own comment
for why success-only is the right call there): ``captures`` (a CUDA stream
capture completed by :meth:`~eagle.pipeline.GraphPipeline.build`),
``instantiations`` (a captured graph turned into a replayable exec graph, the
same method, the instantiate half), ``binds`` (a completed
:meth:`~eagle.plan.Plan.bind` call), ``rebinds`` (a completed
:meth:`~eagle.plan.BoundPlan.rebind` call).

The counters registry names ``captures``/``instantiations``/``rebinds`` plus hawk's own
``builds`` (``hawk.artifact.bundle.COUNTERS`` — hawk must not import eagle, so it keeps
its counter there, not here). ``binds`` is a FIFTH counter: a FRESH ``Plan.bind`` inside
a routing loop is the same class of static-graph violation as a recapture or a rebuild
(the invariant is "resolve once, replay many"), so it belongs beside the other three.
The merge of all five into one mapping, plus the diff/assert-static helper the
certification rows actually call, lives in a separate package that already depends on
both eagle and hawk.

NO RESET. A caller wanting "did anything move" takes two :func:`snapshot`
calls and diffs them; a reset would let two concurrent readers race over
which one observes a given increment (the same reasoning
``hawk.artifact.bundle.reset_unit_memo`` does NOT apply to these counters —
that reset is a different, test-scoped mechanism over a different dict).

Reading never touches the device: this is pure Python bookkeeping,
incremented at Python call sites that already exist. It is not a new
mechanism layered on top of capture/instantiate/bind/rebind — it only counts
calls to mechanisms this module does not otherwise change.
"""

from __future__ import annotations

import threading

#: Serialises :func:`bump`: ``+= 1`` on a dict slot is a read-modify-write, and
#: on a free-threaded interpreter two threads would lose increments.
_LOCK = threading.Lock()

_COUNTERS: dict = {
    "captures": 0,
    "instantiations": 0,
    "binds": 0,
    "rebinds": 0,
}


def bump(name: str) -> None:
    """Increment counter ``name`` by 1.

    Internal: only the hook sites in :mod:`eagle.pipeline`
    (:meth:`~eagle.pipeline.GraphPipeline.build`) and :mod:`eagle.plan`
    (:meth:`~eagle.plan.Plan.bind`, :meth:`~eagle.plan.BoundPlan.rebind`)
    call this. Raises ``KeyError`` naming ``name`` for anything else — a typo
    at a hook site should fail loudly, not silently mint a counter nothing
    ever reads."""
    if name not in _COUNTERS:
        raise KeyError(
            f"eagle._counters.bump: {name!r} is not one of "
            f"{sorted(_COUNTERS)}"
        )
    with _LOCK:
        _COUNTERS[name] += 1


def snapshot() -> dict:
    """A fresh ``dict`` copy of the four counters' current values.

    Reading never touches the device — this is pure Python bookkeeping, so a
    caller may snapshot from any thread/process context that can import this
    module at all."""
    with _LOCK:
        return dict(_COUNTERS)
