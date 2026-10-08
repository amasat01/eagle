# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Certification: :class:`eagle.compose.GraphComposer`
— the GENERIC (AI-agnostic) recursive launch-list composition machinery
migrated out of a downstream expert-composer module. Everything here is
EAGLE-NATIVE: plain :class:`~eagle.pipeline.
GraphPipeline` instances built from simple cupy ``ElementwiseKernel``s, no
any neural-runtime import anywhere — proving the composer serves any ensemble of
independently-captured launchables, not just neural experts (mirrors
``test_pipeline_concurrent.py``/``test_netsparse_member_enable.py``'s own
graph/binding-tier style: function-style ``def test_<behavior>:``,
module-level ``pytestmark = pytest.mark.gpu``).

Coverage:

1. composer-of-composers nesting, BITWISE vs a flat-dense reference (the
   eagle-native mirror of a downstream sibling suite's
   ``test_nested_sequenced_composer_two_level_bitwise`` — the RECURSION
   LOCK's highest-value proof).
2. per-mode topology + routing behavior for ``"enabled"``/``"rebuild"``/
   ``"conditional"``.
3. fired-record correctness + reset.
4. loud rejections: unknown mode, non-sequenced nesting (both directions),
   duplicate registration, the flag-array contract (length/dtype/device).
5. the retired ``expert_composer.py`` reach-in
   (``entry.obj._pipe``/``pipe._ext``/``pipe._launcher``), now formalized as
   :meth:`~eagle.pipeline.GraphPipeline._launch_handles` — behavioral proof
   it exists and powers :meth:`~eagle.compose.GraphComposer.
   _async_launch_group`, not attribute-poking.
"""

from __future__ import annotations

import numpy as np
import pytest

from eagle.compose import GraphComposer
from eagle.pipeline import GraphPipeline

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

_SENTINEL = -777.0


def _warm(kernel, *args):
    """NVRTC-compile ``kernel`` outside capture (compilation is illegal
    mid-capture); callers pass scratch args so real buffers stay untouched."""
    kernel(*args)
    cp.cuda.runtime.deviceSynchronize()


def _dense_pipeline_run(kernel, x, out_shape):
    """The flat-dense reference: a FRESH, standalone, single-kernel
    ``GraphPipeline`` — not a bare numpy formula — so the composer-of-
    composers test below compares graph-captured output against another
    graph-captured output, the same standalone-dense shape
    a downstream sibling suite's own ``_dense_oracle`` uses."""
    out = cp.empty(out_shape, dtype=cp.float64)
    pipe = GraphPipeline()
    pipe.add(lambda: kernel(x, out))
    pipe.build()
    pipe.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(out).copy()


# ============================================================================
# 1. Composer-of-composers (the RECURSION LOCK), BITWISE vs flat-dense.
# ============================================================================
def test_nested_sequenced_composer_two_level_bitwise():
    """2-level composer-of-composers: outer flags x inner flags, bitwise vs
    the equivalent flat-dense runs. The outer composer's ONLY registered
    member is the (already-built) inner composer -- "outer flags" therefore
    reduces to one on/off slot, crossed against the inner composer's OWN
    2-member routing pattern (mirrors a downstream combo structure):

    1. outer=ON,  inner=(1,1) -- both inner members fire, bitwise dense.
    2. outer=ON,  inner=(1,0) -- only inner member 0 fires.
    3. outer=OFF, inner=(1,1) -- NOTHING fires despite inner being all-on
       (the outer gate wins: the inner composer's launch() is never even
       called -- verified via both members' output AND the inner
       composer's own fired_history never growing on that combo).
    """
    n = 64
    a = cp.arange(n, dtype=cp.float64)
    b = cp.arange(n, dtype=cp.float64) * 2.0 - 1.0
    kA = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 3.0 + 1.0", "compose_nest_A"
    )
    kB = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x - 5.0", "compose_nest_B"
    )

    outA = cp.full(n, _SENTINEL, dtype=cp.float64)
    outB = cp.full(n, _SENTINEL, dtype=cp.float64)
    _warm(kA, a, cp.empty_like(outA))
    _warm(kB, b, cp.empty_like(outB))

    pipeA = GraphPipeline()
    pipeA.add(lambda: kA(a, outA))
    pipeA.build()
    pipeB = GraphPipeline()
    pipeB.add(lambda: kB(b, outB))
    pipeB.build()

    inner = GraphComposer([1, 1], mode="sequenced")
    idxA = inner.register(pipeA)
    idxB = inner.register(pipeB)
    assert (idxA, idxB) == (0, 1)
    inner.build()

    outer = GraphComposer([1], mode="sequenced")
    outer_idx = outer.register(inner)
    assert outer_idx == 0
    outer.build()

    expectedA = _dense_pipeline_run(kA, a, n)
    expectedB = _dense_pipeline_run(kB, b, n)

    # 1. outer=ON, inner=(1,1) -- both fire.
    outA.fill(_SENTINEL)
    outB.fill(_SENTINEL)
    outer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(outA), expectedA), "combo1: A"
    assert np.array_equal(cp.asnumpy(outB), expectedB), "combo1: B"

    # 2. outer=ON, inner=(1,0) -- only A fires.
    outA.fill(_SENTINEL)
    outB.fill(_SENTINEL)
    inner.set_routing([1, 0])
    outer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(outA), expectedA), "combo2: A"
    assert np.all(cp.asnumpy(outB) == _SENTINEL), "combo2: B untouched"

    # 3. outer=OFF, inner=(1,1) -- NOTHING fires, despite inner all-on.
    outA.fill(_SENTINEL)
    outB.fill(_SENTINEL)
    inner.set_routing([1, 1])
    outer.set_routing([0])
    outer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.all(cp.asnumpy(outA) == _SENTINEL), "combo3: outer gate wins (A)"
    assert np.all(cp.asnumpy(outB) == _SENTINEL), "combo3: outer gate wins (B)"

    assert outer.fired_history() == [frozenset({0}), frozenset({0}), frozenset()]
    # inner.launch() was called on combos 1 and 2 only -- combo 3's outer
    # gate never reached it, so inner's own fired_history stops growing.
    assert inner.fired_history() == [frozenset({0, 1}), frozenset({0})]


# ============================================================================
# 2. Per-mode topology + routing: "enabled" / "rebuild" / "conditional".
# ============================================================================
def test_enabled_mode_toggle_between_replays_no_recapture():
    n = 64
    a = cp.arange(n, dtype=cp.float64)
    outA = cp.full(n, _SENTINEL, dtype=cp.float64)
    outB = cp.full(n, _SENTINEL, dtype=cp.float64)
    kA = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 2.0", "compose_enabled_A"
    )
    kB = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 3.0", "compose_enabled_B"
    )
    _warm(kA, a, cp.empty_like(outA))
    _warm(kB, a, cp.empty_like(outB))

    composer = GraphComposer(None, mode="enabled")  # default: all-active
    composer.register(lambda: kA(a, outA), name="A")
    composer.register(lambda: kB(a, outB), name="B")
    composer.build()
    pipe_identity = id(composer._pipe)

    composer.set_routing([1, 0])
    outA.fill(_SENTINEL)
    outB.fill(_SENTINEL)
    composer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(outA), cp.asnumpy(a) * 2.0)
    assert np.all(cp.asnumpy(outB) == _SENTINEL), "disabled member untouched"

    composer.set_routing([0, 1])
    outA.fill(_SENTINEL)
    outB.fill(_SENTINEL)
    composer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.all(cp.asnumpy(outA) == _SENTINEL), "disabled member untouched"
    assert np.array_equal(cp.asnumpy(outB), cp.asnumpy(a) + 3.0)

    assert id(composer._pipe) == pipe_identity, "mode='enabled' must never recapture"


def test_rebuild_mode_recaptures_each_routing_change_and_all_inactive_is_noop():
    n = 64
    a = cp.arange(n, dtype=cp.float64)
    outA = cp.full(n, _SENTINEL, dtype=cp.float64)
    outB = cp.full(n, _SENTINEL, dtype=cp.float64)
    kA = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x / 2.0", "compose_rebuild_A"
    )
    kB = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x - 1.0", "compose_rebuild_B"
    )
    _warm(kA, a, cp.empty_like(outA))
    _warm(kB, a, cp.empty_like(outB))

    composer = GraphComposer(None, mode="rebuild")
    composer.register(lambda: kA(a, outA), name="A")
    composer.register(lambda: kB(a, outB), name="B")
    composer.build()
    prev_id = id(composer._pipe)

    patterns = [(True, False), (False, True), (True, True)]  # >= 3, no repeats
    for on_a, on_b in patterns:
        composer.set_routing([on_a, on_b])
        assert id(composer._pipe) != prev_id, "mode='rebuild' must recapture"
        prev_id = id(composer._pipe)
        outA.fill(_SENTINEL)
        outB.fill(_SENTINEL)
        composer.launch(1)
        cp.cuda.runtime.deviceSynchronize()
        tag = f"pattern {(on_a, on_b)}"
        if on_a:
            assert np.array_equal(cp.asnumpy(outA), cp.asnumpy(a) / 2.0), tag
        else:
            assert np.all(cp.asnumpy(outA) == _SENTINEL), tag
        if on_b:
            assert np.array_equal(cp.asnumpy(outB), cp.asnumpy(a) - 1.0), tag
        else:
            assert np.all(cp.asnumpy(outB) == _SENTINEL), tag

    # All-inactive: captures nothing, launch() is a no-op, introspection
    # names the reason.
    composer.set_routing([False, False])
    assert composer._pipe is None
    composer.launch(1)  # must not raise
    assert composer.fired_history()[-1] == frozenset()
    with pytest.raises(RuntimeError, match="no members are currently active"):
        composer.num_nodes()


def test_conditional_mode_bitwise_guard_and_no_recapture():
    n = 32
    a = cp.arange(n, dtype=cp.float64)
    outA = cp.full(n, _SENTINEL, dtype=cp.float64)
    outB = cp.full(n, _SENTINEL, dtype=cp.float64)
    kA = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 7.0", "compose_cond_A"
    )
    kB = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 4.0", "compose_cond_B"
    )
    _warm(kA, a, cp.empty_like(outA))
    _warm(kB, a, cp.empty_like(outB))

    flags = cp.asarray([1, 0], dtype=cp.uint32)
    composer = GraphComposer(flags, mode="conditional")
    composer.register(lambda: kA(a, outA))
    composer.register(lambda: kB(a, outB))
    composer.build()
    pipe_identity = id(composer._pipe)

    composer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(outA), cp.asnumpy(a) + 7.0)
    assert np.all(cp.asnumpy(outB) == _SENTINEL), "guarded-off member untouched"

    # Direct device-flag mutation (no set_routing()) -- the conditional
    # path's own backward-compatible contract; ONE capture throughout.
    outA.fill(_SENTINEL)
    flags[0] = 0
    flags[1] = 1
    composer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.all(cp.asnumpy(outA) == _SENTINEL), "guarded-off member untouched"
    assert np.array_equal(cp.asnumpy(outB), cp.asnumpy(a) * 4.0)
    assert id(composer._pipe) == pipe_identity, "ONE captured graph, never rebuilt"
    assert composer.fired_history() == [frozenset({0}), frozenset({1})]


# ============================================================================
# 3. Fired-record correctness + reset (generic observability, standalone).
# ============================================================================
def test_fired_history_records_and_resets():
    n = 32
    a = cp.arange(n, dtype=cp.float64)
    outA = cp.empty(n, dtype=cp.float64)
    outB = cp.empty(n, dtype=cp.float64)
    kA = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 1.0", "compose_fired_A"
    )
    kB = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 2.0", "compose_fired_B"
    )
    _warm(kA, a, cp.empty_like(outA))
    _warm(kB, a, cp.empty_like(outB))

    composer = GraphComposer([1, 1], mode="enabled")
    composer.register(lambda: kA(a, outA), name="A")
    composer.register(lambda: kB(a, outB), name="B")
    composer.build()

    composer.launch(1)
    composer.set_routing([1, 0])
    composer.launch(1)
    composer.set_routing([0, 1])
    composer.launch(1)
    cp.cuda.runtime.deviceSynchronize()

    assert composer.fired_history() == [
        frozenset({0, 1}),
        frozenset({0}),
        frozenset({1}),
    ]
    composer.reset_fired_history()
    assert composer.fired_history() == []


# ============================================================================
# 4. Loud rejections.
# ============================================================================
def test_reject_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        GraphComposer(None, mode="bogus")


def test_reject_nesting_non_sequenced_composer_under_sequenced_outer():
    """One RECURSION LOCK direction: the OUTER composer is mode='sequenced'
    but the composer being nested is NOT -- rejected loudly, by name,
    rather than falling through to a confusing AttributeError."""
    n = 16
    a = cp.arange(n, dtype=cp.float64)
    out = cp.empty(n, dtype=cp.float64)
    k = cp.ElementwiseKernel("float64 x", "float64 y", "y = x", "compose_lock1")
    _warm(k, a, out)

    flags = cp.ones(1, dtype=cp.uint32)
    inner_conditional = GraphComposer(flags, mode="conditional")
    inner_conditional.register(lambda: k(a, out))
    inner_conditional.build()

    outer = GraphComposer([1], mode="sequenced")
    with pytest.raises(ValueError, match="mode='sequenced'"):
        outer.register(inner_conditional)


def test_reject_nesting_any_composer_under_a_non_sequenced_outer():
    """The OTHER RECURSION LOCK direction: registering a composer (even one
    that is itself mode='sequenced') under an OUTER composer whose own mode
    is anything BUT 'sequenced' is rejected loudly -- the flat modes have
    no per-member launch list to select a nested composer into at that
    granularity."""
    inner_sequenced = GraphComposer([1], mode="sequenced")
    inner_sequenced.register(lambda: None)
    inner_sequenced.build()

    outer_conditional = GraphComposer(cp.ones(1, dtype=cp.uint32), mode="conditional")
    with pytest.raises(ValueError, match=r"only supported under a mode='sequenced'"):
        outer_conditional.register(inner_sequenced)


def test_reject_duplicate_registration():
    n = 16
    a = cp.arange(n, dtype=cp.float64)
    out = cp.empty(n, dtype=cp.float64)
    k = cp.ElementwiseKernel("float64 x", "float64 y", "y = x", "compose_dup")
    _warm(k, a, out)

    pipe = GraphPipeline()
    pipe.add(lambda: k(a, out))
    pipe.build()

    composer = GraphComposer([1], mode="sequenced")
    composer.register(pipe)
    with pytest.raises(ValueError, match="already registered"):
        composer.register(pipe)


def test_reject_flag_array_wrong_dtype():
    with pytest.raises(TypeError, match="uint32"):
        GraphComposer(cp.ones(2, dtype=cp.int32), mode="conditional")


def test_reject_flag_array_wrong_device():
    with pytest.raises(TypeError, match="cupy"):
        GraphComposer(np.ones(2, dtype=np.uint32), mode="conditional")


def test_reject_flag_array_length_mismatch():
    composer = GraphComposer(cp.ones(1, dtype=cp.uint32), mode="conditional")
    composer.register(lambda: None)
    composer.register(lambda: None)  # 2 members, flags length 1
    with pytest.raises(ValueError, match="flags array length"):
        composer.build()


# ============================================================================
# 5. The retired reach-in, formalized: GraphPipeline._launch_handles().
# ============================================================================
def test_launch_handles_exists_and_powers_the_overlap_group():
    """``expert_composer.py``'s old :715-727 DEVIATION reached directly into
    a built ``GraphPipeline``'s private ``_ext``/``_launcher`` attributes to
    drive enqueue-without-sync replay across several pipelines. That reach-
    in is RETIRED by relocation: :meth:`GraphPipeline._launch_handles` is
    now the ONE sanctioned (still module-private -- no new PUBLIC surface)
    accessor, and :meth:`GraphComposer._async_launch_group` is its ONE
    consumer. Proven BEHAVIORALLY here: it raises before build(), stays
    stable across calls, and two independently-built pipelines registered
    under one mode='sequenced' composer replay correctly THROUGH it."""
    unbuilt = GraphPipeline()
    unbuilt.add(lambda: None)
    with pytest.raises(RuntimeError):
        unbuilt._launch_handles()

    n = 64
    a = cp.arange(n, dtype=cp.float64)
    b = cp.arange(n, dtype=cp.float64) * 5.0
    outA = cp.full(n, _SENTINEL, dtype=cp.float64)
    outB = cp.full(n, _SENTINEL, dtype=cp.float64)
    kA = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 2.0", "compose_handles_A"
    )
    kB = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x - 9.0", "compose_handles_B"
    )
    _warm(kA, a, cp.empty_like(outA))
    _warm(kB, b, cp.empty_like(outB))

    pipeA = GraphPipeline()
    pipeA.add(lambda: kA(a, outA))
    pipeA.build()
    pipeB = GraphPipeline()
    pipeB.add(lambda: kB(b, outB))
    pipeB.build()

    handles = pipeA._launch_handles()
    assert hasattr(handles, "ext")
    assert hasattr(handles, "launcher")
    assert pipeA._launch_handles().launcher is handles.launcher, (
        "stable handles, not rebuilt per call"
    )

    composer = GraphComposer([1, 1], mode="sequenced")
    composer.register(pipeA)
    composer.register(pipeB)
    composer.build()
    composer.launch(1)
    cp.cuda.runtime.deviceSynchronize()

    assert np.array_equal(cp.asnumpy(outA), cp.asnumpy(a) + 2.0)
    assert np.array_equal(cp.asnumpy(outB), cp.asnumpy(b) - 9.0)
