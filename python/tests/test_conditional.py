# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Acceptance: :mod:`eagle.pipeline`'s ``Skippable``/``SkipGuard``/
``skippable()`` Python surface over ``eagle::cuda::CaptureConditional``
-- mirrors ``eagle/tests/test_ConditionalGroup.cu``'s three-leg
gate shape at the Python layer.

G-BEHAV's central trap: an output-value assertion cannot distinguish "the
node was skipped" from "the node fired and its threads early-returned" -- it
passes identically either way. The skip-proof is therefore an UNGUARDED
TRACER: a kernel inside the guarded body that increments a counter with NO
liveness check of its own. A fired body cannot leave it silent; a skipped
body cannot advance it. ``test_skippable_three_leg_add`` and
``test_skippable_three_leg_add_concurrent`` build ONE graph, instantiate it
ONCE (``GraphPipeline.build()``), and replay the SAME launcher three times,
mutating only the guard's device word between legs -- never rebuilding.
Leg 3 (restore TRUE, same launcher, no rebuild) is what kills a build-time
host-read of the predicate: a build-time-baked "true" would still show
tracer==1 after leg 2 (vacuously), but could not re-fire on leg 3.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from eagle import SkipGuard, skippable
from eagle.pipeline import GraphPipeline
from _graph_dot import conditional_count

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

_TRACER_SRC = r"""
extern "C" __global__ void r8_tracer_kernel(int* counter) {
    atomicAdd(counter, 1);
}
"""
_tracer_kernel = cp.RawKernel(_TRACER_SRC, "r8_tracer_kernel")


def _fresh_tracer():
    """A device int32 counter, NVRTC-warmed (compilation is illegal
    mid-capture) and zeroed, ready to be incremented unconditionally."""
    t = cp.zeros(1, dtype=cp.int32)
    _tracer_kernel((1,), (1,), (t,))
    cp.cuda.runtime.deviceSynchronize()
    t[:] = 0
    return t


def _tracer_value(t) -> int:
    return int(t.get()[0])


# --------------------------------------------------------------------------- #
# G-BEHAV primary: three-leg skip-proof.
# --------------------------------------------------------------------------- #
def test_skippable_three_leg_add():
    """Skippable registered via plain add() -- the top-level weave."""
    count = cp.ones(1, dtype=cp.uint32)
    tracer = _fresh_tracer()
    guard = SkipGuard.nonzero(count)
    step = skippable(lambda: _tracer_kernel((1,), (1,), (tracer,)), guard)

    pipe = GraphPipeline()
    pipe.add(step)
    pipe.build()

    # Leg 1: guard TRUE (count=1 != baseline 0) -> body fires -> tracer==1.
    pipe.launch()
    assert _tracer_value(tracer) == 1, "leg 1 (guard TRUE) must fire exactly once"

    # Leg 2: mutate ONLY the guard's device word -> body must NOT fire.
    count[:] = 0
    pipe.launch()
    assert _tracer_value(tracer) == 1, "leg 2 (guard FALSE) must leave tracer silent"

    # Leg 3: restore TRUE, SAME launcher, no rebuild -> tracer==2.
    count[:] = 1
    pipe.launch()
    assert _tracer_value(tracer) == 2, "leg 3 (guard restored TRUE) must re-fire"


def test_skippable_three_leg_add_concurrent():
    """Same three-leg proof with the Skippable as one member of
    add_concurrent([...]), alongside an ordinary sibling member -- weaving on
    a CaptureFork branch stream ('probe-proven' claim)."""
    count = cp.ones(1, dtype=cp.uint32)
    tracer = _fresh_tracer()
    guard = SkipGuard.nonzero(count)
    gated = skippable(lambda: _tracer_kernel((1,), (1,), (tracer,)), guard)

    sibling_in = cp.arange(4, dtype=cp.float64)
    sibling_out = cp.zeros(4, dtype=cp.float64)
    kB = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 1.0", "r8_sibling_kernel"
    )
    kB(sibling_in, sibling_out)  # NVRTC warmup, outside capture
    cp.cuda.runtime.deviceSynchronize()
    sibling_out.fill(0.0)

    pipe = GraphPipeline()
    pipe.add_concurrent([gated, lambda: kB(sibling_in, sibling_out)])
    pipe.build()

    pipe.launch()
    assert _tracer_value(tracer) == 1, "leg 1: concurrent-member guard TRUE must fire"
    assert np.array_equal(cp.asnumpy(sibling_out), cp.asnumpy(sibling_in) + 1.0)

    count[:] = 0
    pipe.launch()
    assert _tracer_value(tracer) == 1, "leg 2: concurrent-member skip must also hold"

    count[:] = 1
    pipe.launch()
    assert _tracer_value(tracer) == 2, "leg 3: concurrent-member must re-fire"


# --------------------------------------------------------------------------- #
# G-STRUCT (secondary; explicitly insufficient alone).
# --------------------------------------------------------------------------- #
def test_graph_struct_conditional_node_and_body_cluster():
    """The built graph's dot-dump contains exactly one IF (conditional) node,
    and the guarded body's kernel sits inside ITS OWN subgraph cluster --
    distinct from the outer graph's -- not as a top-level node ([PD]
    verified ``Type: IF`` live, re-checked here against the ACTUAL flags
    GraphPipeline._debug_dot passes, i.e. the default, non-verbose dot).
    Also re-pins num_nodes()/node_labels() semantics for a
    conditional-bearing capture: num_nodes() counts only octagon-shaped
    kernel nodes (2: the eagle-owned setCond guard kernel + the tracer
    kernel) -- the IF node itself is shape="rectangle" and excluded, and
    node_labels()'s regex still finds the body kernel even though it now
    sits inside a cluster (label tests must not silently change
    meaning)."""
    count = cp.ones(1, dtype=cp.uint32)
    tracer = _fresh_tracer()
    guard = SkipGuard.nonzero(count)
    step = skippable(lambda: _tracer_kernel((1,), (1,), (tracer,)), guard)

    pipe = GraphPipeline()
    pipe.add(step)
    pipe.build()

    dot = pipe._dot()
    assert conditional_count(dot, "IF") in (None, 1), "exactly one conditional (IF) node"
    assert pipe.num_nodes() == 2, (
        "octagon count: setCond guard kernel + r8_tracer_kernel -- the IF "
        "node is shape=rectangle and must not be counted"
    )
    labels = pipe.node_labels()
    assert any("r8_tracer_kernel" in label for label in labels)
    assert any("setCountGuardKernel" in label for label in labels)

    clusters = re.findall(r"subgraph cluster_\d+ \{(.*?)\n\}", dot, re.S)
    assert len(clusters) == 2, "outer graph cluster + the IF-body child cluster"
    outer, body = clusters
    assert "r8_tracer_kernel" not in outer, "body kernel must NOT be top-level"
    assert (
        "r8_tracer_kernel" in body
    ), "body kernel must sit inside the IF's child cluster"


# --------------------------------------------------------------------------- #
# G-VALUE (labeled NOT-a-skip-proof): correctness only.
# --------------------------------------------------------------------------- #
def test_g_value_eager_vs_conditional_replay_bit_identical():
    """A VALUE-producing kernel (not the tracer) run through a
    conditional-guarded GraphPipeline replay must match an eager/host
    computation of the same operation, bit-identical, across an
    empty -> live -> empty transition of the guarded population
    (``SkipGuard.nonempty_bucket``'s real fencepost shape). This is
    correctness only -- see the module docstring for why a value assertion
    alone cannot prove a SKIP happened."""
    type_offsets = cp.asarray([0, 0], dtype=cp.uint32)  # bucket empty initially
    data = cp.arange(8, dtype=cp.float64)
    out = cp.zeros(8, dtype=cp.float64)
    kD = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 2.0", "r8_value_kernel"
    )
    kD(data, out)  # NVRTC warmup, outside capture
    cp.cuda.runtime.deviceSynchronize()

    guard = SkipGuard.nonempty_bucket(type_offsets, 0)
    step = skippable(lambda: kD(data, out), guard)

    pipe = GraphPipeline()
    pipe.add(step)
    pipe.build()

    def eager_expected():
        offsets = cp.asnumpy(type_offsets)
        if int(offsets[1]) != int(offsets[0]):
            return cp.asnumpy(data) * 2.0
        return np.zeros(8)

    for offset_1 in (0, 5, 0):  # empty -> live -> empty
        type_offsets[1] = offset_1
        out.fill(0.0)
        pipe.launch()
        assert np.array_equal(cp.asnumpy(out), eager_expected())


# --------------------------------------------------------------------------- #
# SkipGuard must RETAIN its array references.
# --------------------------------------------------------------------------- #
def test_skip_guard_retains_array_reference():
    """A dropped cupy array would leave setCond reading a dangling device
    pointer on every replay. Build a guard from an array with NO other live
    reference in the caller, force a GC pass, and confirm replays still see
    a correctly-gated result."""
    import gc

    tracer = _fresh_tracer()

    def make_guard():
        count = cp.ones(1, dtype=cp.uint32)  # no other reference kept below
        return SkipGuard.nonzero(count)

    guard = make_guard()
    gc.collect()

    step = skippable(lambda: _tracer_kernel((1,), (1,), (tracer,)), guard)
    pipe = GraphPipeline()
    pipe.add(step)
    pipe.build()
    gc.collect()

    pipe.launch()
    assert (
        _tracer_value(tracer) == 1
    ), "guard's retained array ref must keep the pointer alive"


# --------------------------------------------------------------------------- #
# numpy parity: Skippable is itself callable (eager conforming form).
# --------------------------------------------------------------------------- #
def test_skippable_eager_numpy_parity():
    """The form a downstream numpy sequential schedule will run unchanged --
    no cupy, no CUDA graph: ``Skippable.__call__`` evaluates its guard
    eagerly on plain numpy arrays."""
    calls = []
    n_active = np.array([0], dtype=np.uint32)
    guard = SkipGuard.nonzero(n_active)
    step = skippable(lambda: calls.append(1), guard)

    step()  # guard false (0 != 0) -> must not call
    assert calls == []

    n_active[0] = 3
    step()  # guard true -> must call
    assert calls == [1]

    n_active[0] = 0
    step()  # guard false again -> must not call
    assert calls == [1]


# --------------------------------------------------------------------------- #
# An IF node whose body is empty never completes when it fires (the stream
# hangs), so a skippable step that launches nothing is refused at build time.
# --------------------------------------------------------------------------- #
def test_skippable_empty_body_is_refused_at_build():
    count = cp.ones(1, dtype=cp.uint32)
    pipe = GraphPipeline()
    pipe.add(skippable(lambda: None, SkipGuard.nonzero(count)))
    with pytest.raises(ValueError, match="empty body"):
        pipe.build()

    # The refusal leaves no capture open: a well-formed pipeline still builds
    # and its guarded body fires.
    tracer = _fresh_tracer()
    ok = GraphPipeline()
    ok.add(skippable(lambda: _tracer_kernel((1,), (1,), (tracer,)),
                     SkipGuard.nonzero(count)))
    ok.build()
    ok.launch()
    assert _tracer_value(tracer) == 1
