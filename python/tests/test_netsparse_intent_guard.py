# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Certification: :meth:`SkipGuard.intent` -- the
new named constructor for a router/user-owned per-lane intent flag -- plus
the composed-callable region it wraps ("compose inside the
callable" contract, gated here at ``GraphPipeline.add_concurrent``'s granularity).

Distinct from ``test_conditional.py``'s G-BEHAV tracer proof (which certifies
the underlying ``Skippable``/``CaptureConditional`` MECHANISM once, via
``SkipGuard.nonzero``): this file certifies the NEW ``intent`` constructor's
value-level contract -- (a) flag=1 bitwise-matches an
unguarded run, (b) flag=0 leaves prefilled sentinel buffers untouched, (c)
>=3 distinct flag patterns replay against ONE captured graph with no
recapture, (d) the eager/numpy twin of (a)+(b), and (e) a THREE-kernel-launch
composed callable wrapped in ONE ``skippable`` and registered as an
``add_concurrent`` member alongside an unguarded sibling, asserting
whole-region skip/run semantics (not just the last stage) in both flag
states. A standing red-probe (last) proves the buffer-untouched assertion
these tests rely on is not vacuous: it monkeypatches
``SkipGuard._device_ptrs`` (pytest's ``monkeypatch`` fixture, auto-reverted
at teardown -- nothing here is a permanent change to production code) so the
guard ALWAYS reads as firing, and shows the untouched-assertion goes RED
under that simulated regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from eagle import SkipGuard, skippable
from eagle.pipeline import GraphPipeline

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

_SENTINEL = -777.0


def _value_kernel():
    return cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 3.0 + 1.0", "netsparse_intent_value_kernel"
    )


def _warm(kernel, *args):
    """NVRTC-compile ``kernel`` outside capture (compilation is illegal
    mid-capture); callers pass scratch args so real buffers stay untouched."""
    kernel(*args)
    cp.cuda.runtime.deviceSynchronize()


# --------------------------------------------------------------------------- #
# Constructor shape: SkipGuard.intent(flags, index).
# --------------------------------------------------------------------------- #
def test_intent_constructor_shape():
    """``intent`` mirrors the general ``__init__``: retains ``flags`` by
    reference at ``count_idx=index``, no baseline (plain on/off, not a
    fencepost liveness check)."""
    flags = np.zeros(4, dtype=np.uint32)
    guard = SkipGuard.intent(flags, 2)
    assert guard.count_arr is flags
    assert guard.count_idx == 2
    assert guard.baseline_arr is None
    assert guard.evaluate() is False

    flags[2] = 1
    assert guard.evaluate() is True
    # Other lanes must not leak into this guard's decision.
    flags[0] = 1
    flags[3] = 1
    assert guard.evaluate() is True
    flags[2] = 0
    assert guard.evaluate() is False


# --------------------------------------------------------------------------- #
# (a) flag=1 -> wrapped callable's effects BITWISE identical to unguarded.
# --------------------------------------------------------------------------- #
def test_intent_flag_true_matches_unguarded_bitwise():
    n = 256
    data = cp.arange(n, dtype=cp.float64)
    k = _value_kernel()

    out_guarded = cp.full(n, _SENTINEL, dtype=cp.float64)
    _warm(k, data, out_guarded)
    out_guarded.fill(_SENTINEL)

    flags = cp.ones(1, dtype=cp.uint32)
    guard = SkipGuard.intent(flags, 0)
    step = skippable(lambda: k(data, out_guarded), guard)

    pipe = GraphPipeline()
    pipe.add(step)
    pipe.build()
    pipe.launch()

    out_unguarded = cp.zeros(n, dtype=cp.float64)
    k(data, out_unguarded)
    cp.cuda.runtime.deviceSynchronize()

    assert np.array_equal(cp.asnumpy(out_guarded), cp.asnumpy(out_unguarded)), (
        "flag=1 must reproduce the unguarded run bit-for-bit"
    )


# --------------------------------------------------------------------------- #
# (b) flag=0 -> output buffer untouched (prefilled sentinel survives).
# --------------------------------------------------------------------------- #
def test_intent_flag_false_leaves_sentinel_untouched():
    n = 256
    data = cp.arange(n, dtype=cp.float64)
    k = _value_kernel()
    out = cp.full(n, _SENTINEL, dtype=cp.float64)
    _warm(k, data, cp.empty_like(out))  # scratch target: `out` stays sentinel

    flags = cp.zeros(1, dtype=cp.uint32)
    guard = SkipGuard.intent(flags, 0)
    step = skippable(lambda: k(data, out), guard)

    pipe = GraphPipeline()
    pipe.add(step)
    pipe.build()
    pipe.launch()

    msg = "flag=0 must leave the buffer untouched"
    assert np.all(cp.asnumpy(out) == _SENTINEL), msg


# --------------------------------------------------------------------------- #
# (c) >=3 distinct flag patterns, one captured graph, no recapture.
# --------------------------------------------------------------------------- #
def test_intent_three_distinct_flag_patterns_no_recapture():
    """Two independent ``SkipGuard.intent`` lanes sharing one 2-element
    flags array ('flags + index' shape: ``index`` selects the lane).
    Three distinct ON/OFF combinations replay against the SAME instantiated
    graph -- each pattern's per-lane output exact: fired = bitwise unguarded
    match, skipped = sentinel survives."""
    n = 64
    dataA = cp.arange(n, dtype=cp.float64)
    dataB = cp.arange(n, dtype=cp.float64) * 2.0 - 1.0
    kA = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 5.0", "ns_intent_laneA"
    )
    kB = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 7.0", "ns_intent_laneB"
    )

    outA = cp.full(n, _SENTINEL, dtype=cp.float64)
    outB = cp.full(n, _SENTINEL, dtype=cp.float64)
    _warm(kA, dataA, cp.empty_like(outA))
    _warm(kB, dataB, cp.empty_like(outB))

    flags = cp.zeros(2, dtype=cp.uint32)
    guardA = SkipGuard.intent(flags, 0)
    guardB = SkipGuard.intent(flags, 1)
    stepA = skippable(lambda: kA(dataA, outA), guardA)
    stepB = skippable(lambda: kB(dataB, outB), guardB)

    pipe = GraphPipeline()
    pipe.add(stepA)
    pipe.add(stepB)
    pipe.build()

    expectedA = cp.asnumpy(dataA) + 5.0
    expectedB = cp.asnumpy(dataB) * 7.0

    patterns = [(1, 0), (0, 1), (1, 1)]  # 3 distinct patterns, no repeats
    for onA, onB in patterns:
        outA.fill(_SENTINEL)
        outB.fill(_SENTINEL)
        flags[0] = onA
        flags[1] = onB
        pipe.launch()  # SAME launcher/instantiated graph every iteration
        tag = f"pattern {(onA, onB)}"

        if onA:
            assert np.array_equal(cp.asnumpy(outA), expectedA), f"{tag}: lane A"
        else:
            assert np.all(cp.asnumpy(outA) == _SENTINEL), f"{tag}: lane A"

        if onB:
            assert np.array_equal(cp.asnumpy(outB), expectedB), f"{tag}: lane B"
        else:
            assert np.all(cp.asnumpy(outB) == _SENTINEL), f"{tag}: lane B"


# --------------------------------------------------------------------------- #
# (d) eager/numpy-mode twin of (a)+(b): Skippable.__call__, no cupy/capture.
# --------------------------------------------------------------------------- #
def test_intent_eager_numpy_twin_of_bitwise_and_untouched():
    calls = []
    flags = np.array([0], dtype=np.uint32)
    guard = SkipGuard.intent(flags, 0)
    buf = np.full(4, _SENTINEL)

    def body():
        calls.append(1)
        buf[:] = np.arange(4) * 3.0 + 1.0  # same op as _value_kernel

    step = skippable(body, guard)

    step()  # flag=0 -> must not call; buffer untouched (twin of (b))
    assert calls == []
    assert np.all(buf == _SENTINEL)

    flags[0] = 1
    step()  # flag=1 -> must call; matches the unguarded computation (twin of (a))
    assert calls == [1]
    assert np.array_equal(buf, np.arange(4) * 3.0 + 1.0)

    flags[0] = 0
    step()  # flag flips back to 0, SAME Skippable object -> silent again
    assert calls == [1]


# --------------------------------------------------------------------------- #
# (e) composed-callable certification: >=3 launches, ONE skippable, member of
#     add_concurrent alongside an unguarded sibling -- whole-region semantics.
# --------------------------------------------------------------------------- #
def test_intent_composed_multi_launch_region_add_concurrent_whole_region_semantics():
    """Gates ``GraphPipeline.add_concurrent`` (``Skippable`` accepted as an
    ``add_concurrent`` member) at the composed-callable granularity: a THREE
    -kernel-launch closure wrapped in ONE ``skippable``, alongside an
    unguarded sibling member. Both flag states assert on ALL THREE stage
    buffers (not just the last), proving the whole region skips/fires
    together -- not a partial-launch artifact -- while the sibling always
    runs."""
    n = 128
    a = cp.arange(n, dtype=cp.float64)

    stage1 = cp.full(n, _SENTINEL, dtype=cp.float64)
    stage2 = cp.full(n, _SENTINEL, dtype=cp.float64)
    stage3 = cp.full(n, _SENTINEL, dtype=cp.float64)

    k1 = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 1.0", "ns_intent_stage1"
    )
    k2 = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 2.0", "ns_intent_stage2"
    )
    k3 = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x - 3.0", "ns_intent_stage3"
    )
    _warm(k1, a, cp.empty_like(stage1))
    _warm(k2, a, cp.empty_like(stage2))
    _warm(k3, a, cp.empty_like(stage3))

    def composed_region():
        k1(a, stage1)
        k2(stage1, stage2)
        k3(stage2, stage3)

    flags = cp.zeros(1, dtype=cp.uint32)
    guard = SkipGuard.intent(flags, 0)
    gated = skippable(composed_region, guard)

    sibling_in = cp.arange(n, dtype=cp.float64)
    sibling_out = cp.zeros(n, dtype=cp.float64)
    k_sib = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 100.0", "ns_intent_sibling"
    )
    _warm(k_sib, sibling_in, cp.empty_like(sibling_out))

    pipe = GraphPipeline()
    pipe.add_concurrent([gated, lambda: k_sib(sibling_in, sibling_out)])
    pipe.build()

    # flag=0: the WHOLE 3-launch region must be skipped -- all three stage
    # buffers untouched, not just the last -- sibling still fires.
    pipe.launch()
    assert np.all(cp.asnumpy(stage1) == _SENTINEL), "region skip: stage1 untouched"
    assert np.all(cp.asnumpy(stage2) == _SENTINEL), "region skip: stage2 untouched"
    assert np.all(cp.asnumpy(stage3) == _SENTINEL), "region skip: stage3 untouched"
    assert np.array_equal(cp.asnumpy(sibling_out), cp.asnumpy(sibling_in) + 100.0), (
        "unguarded sibling must run regardless of the guarded member's state"
    )

    # flag=1, SAME launcher, no rebuild: whole region fires, each stage exact.
    flags[:] = 1
    pipe.launch()
    a_np = cp.asnumpy(a)
    expected1 = a_np + 1.0
    expected2 = expected1 * 2.0
    expected3 = expected2 - 3.0
    assert np.array_equal(cp.asnumpy(stage1), expected1)
    assert np.array_equal(cp.asnumpy(stage2), expected2)
    assert np.array_equal(cp.asnumpy(stage3), expected3)


# --------------------------------------------------------------------------- #
# Red-probe (nonvacuity): the buffer-untouched assertion above must be able
# to go RED. Never a permanent change to production code -- monkeypatch is
# reverted automatically at test teardown.
# --------------------------------------------------------------------------- #
def test_intent_untouched_check_is_nonvacuous_red_probe(monkeypatch):
    """Monkeypatch ``SkipGuard._device_ptrs`` so the guard ALWAYS reports
    'firing', regardless of the real flags array -- simulating a regression
    in the device-read seam :meth:`GraphPipeline.build` relies on. Replay
    with the REAL flag left at 0 (intent OFF) and show the buffer-untouched
    assertion (b)/(e) depend on actually trips: it is not a vacuous check
    that would pass no matter what the region does."""
    n = 64
    data = cp.arange(n, dtype=cp.float64)
    k = _value_kernel()
    out = cp.full(n, _SENTINEL, dtype=cp.float64)
    _warm(k, data, cp.empty_like(out))

    flags = cp.zeros(1, dtype=cp.uint32)  # real intent: OFF
    guard = SkipGuard.intent(flags, 0)

    always_on = cp.ones(1, dtype=cp.uint32)

    def _broken_device_ptrs(self):
        # Ignores `self` (the real, OFF guard) entirely -- simulates a
        # regressed seam that always reports "on".
        return int(always_on.data.ptr), 0

    monkeypatch.setattr(SkipGuard, "_device_ptrs", _broken_device_ptrs)

    step = skippable(lambda: k(data, out), guard)
    pipe = GraphPipeline()
    pipe.add(step)
    pipe.build()
    pipe.launch()

    # Under the broken seam the region fires even though flags[0] == 0 --
    # the buffer is NOT untouched. The assertion (b) relies on must catch
    # this, not pass vacuously.
    msg = "skip must leave the buffer untouched"
    with pytest.raises(AssertionError):
        assert np.all(cp.asnumpy(out) == _SENTINEL), msg
