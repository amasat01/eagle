# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Certification: the member-enable surface (Composition
MODE ``"enabled"`` -- the "Composition MODES" section)
built on ``eagle._core.capture_snapshot_nodes`` / ``is_node_toggleable`` /
``Launcher.set_node_enabled``: ``GraphPipeline.add(name=)`` /
``.add_concurrent(names=)`` / ``.set_member_enabled`` / ``.member`` /
``.is_member_toggleable`` / ``.member_nodes`` (CUDA
capture path), plus its host/numpy dual-mode twin :class:`RecordedSchedule`
and the shared ``eagle._member_enable`` machinery (``_MemberRegistry``,
``MemberHandle``, ``NonToggleableMemberError``, ``UnknownMemberError``).

Also ``GraphPipeline.set_node_enabled(node, enabled)`` -- an additional route:
a public, registry-BYPASSING passthrough to the
exact same ``Launcher.set_node_enabled`` call ``set_member_enabled`` already
makes per node, for a caller (a concurrent-member door elsewhere) that tracks its
OWN node attribution outside this pipeline's ``add``/``add_concurrent``
bookkeeping. No new ``_core`` surface; no behavior change to
``set_member_enabled``.

Two tiers, split by whether they can run yet:

  (a) HOST-MODE -- pure Python, no cupy, no CUDA graph, no GPU. Runnable
      right now. Covers :class:`RecordedSchedule`'s schedule-filter
      semantics and the bad-handle / typed-error contract of the SHARED
      ``_MemberRegistry`` (:class:`~eagle.pipeline.GraphPipeline` composes
      the exact same class -- see ``eagle._member_enable``'s module
      docstring -- so exercising it here through ``RecordedSchedule``'s
      host-reachable door tests the real production code, not a proxy).

  (b) CUDA-ARM -- ``pytest.mark.gpu``, AUTHORED but NOT YET RUN: the
      installed ``eagle._core`` predates these C++ additions
      (``capture_snapshot_nodes`` /
      ``is_node_toggleable`` / ``Launcher.set_node_enabled`` do not exist
      in it yet), so running these now would fail on an unrelated
      ``AttributeError`` rather than exercise anything. cupy and
      ``eagle.pipeline``/``eagle._conditional`` symbols that need a real
      capture are imported LOCALLY inside each such test (not at module
      top), so this file's (a) tests stay collectible and runnable even in
      a cupy-less environment.
"""

from __future__ import annotations

import numpy as np
import pytest

from eagle._member_enable import (
    MemberHandle,
    NonToggleableMemberError,
    RecordedSchedule,
    UnknownMemberError,
    _MemberRegistry,
)

_SENTINEL = -777.0

# ============================================================================
# (a) HOST-MODE -- no GPU, no cupy, no CUDA graph. Runnable right now.
# ============================================================================


def test_recorded_schedule_disabled_member_is_skipped():
    """A disabled member's callable does not run; an enabled sibling does --
    the schedule-filter semantics the class docstring promises."""
    calls = []
    sched = RecordedSchedule()
    sched.add(lambda: calls.append("a"), name="a")
    sched.add(lambda: calls.append("b"), name="b")

    sched.set_member_enabled("a", False)
    sched()
    assert calls == ["b"]


def test_recorded_schedule_enabled_equals_unfiltered():
    """enabled == unfiltered: with every member left at its default
    (enabled), the schedule's output equals plain sequential execution of
    the same callables in the same order."""
    calls = []
    sched = RecordedSchedule()
    sched.add(lambda: calls.append("a"))
    sched.add(lambda: calls.append("b"))
    sched.add(lambda: calls.append("c"))

    sched()  # no toggling at all
    assert calls == ["a", "b", "c"]


def test_recorded_schedule_toggle_between_eager_replays():
    """Same schedule object, no rebuild: flipping a member's enabled state
    between successive ``__call__`` invocations changes the schedule's
    output immediately each time -- the eager/host analogue of the CUDA
    path's ">= 3 toggle patterns against one captured graph, no recapture"
    certification."""
    calls = []
    sched = RecordedSchedule()
    sched.add(lambda: calls.append("a"), name="a")
    sched.add(lambda: calls.append("b"), name="b")

    patterns = [(True, True), (False, True), (True, False), (False, False)]
    observed = []
    for on_a, on_b in patterns:
        calls.clear()
        sched.set_member_enabled("a", on_a)
        sched.set_member_enabled("b", on_b)
        sched()
        observed.append(tuple(calls))

    assert observed == [("a", "b"), ("b",), ("a",), ()]


def test_recorded_schedule_add_concurrent_runs_all_members_serially():
    """Host mode always serializes ``add_concurrent``'s members, in
    registration order -- a conforming schedule per
    ``GraphPipeline.add_concurrent``'s own permission-not-obligation
    contract (co-execution is PERMITTED, never required)."""
    calls = []
    sched = RecordedSchedule()
    h1, h2, h3 = sched.add_concurrent(
        [lambda: calls.append(1), lambda: calls.append(2), lambda: calls.append(3)]
    )
    sched()
    assert calls == [1, 2, 3]
    assert [h1.index, h2.index, h3.index] == [0, 1, 2]


def test_recorded_schedule_add_concurrent_rejects_too_few_members():
    sched = RecordedSchedule()
    with pytest.raises(ValueError):
        sched.add_concurrent([lambda: None])


def test_recorded_schedule_add_concurrent_names_length_mismatch():
    sched = RecordedSchedule()
    with pytest.raises(ValueError):
        sched.add_concurrent([lambda: None, lambda: None], names=["only_one"])


def test_recorded_schedule_add_concurrent_names_alias_correctly():
    sched = RecordedSchedule()
    sched.add_concurrent(
        [lambda: None, lambda: None, lambda: None], names=["x", None, "z"]
    )
    assert sched.member("x").index == 0
    assert sched.member("z").index == 2
    # the middle member (index 1) is still addressable BY INDEX -- every
    # member is a member regardless of naming, only a NAME lookup can fail.
    assert sched.member(1).index == 1
    assert sched.member(1).name is None
    with pytest.raises(UnknownMemberError):
        sched.member("y")  # never registered under this name


def test_member_handle_equality_and_hash_by_index_only():
    a = MemberHandle(2, "expert2")
    b = MemberHandle(2, None)  # same index, no name -- still the same member
    c = MemberHandle(3, "expert2")  # same name text, different index
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != "not a handle"


# -- bad-handle / typed-error contract (shared _MemberRegistry) ------------


def test_registry_resolve_unknown_name_raises_unknown_member_error():
    reg = _MemberRegistry()
    reg.register("only")
    with pytest.raises(UnknownMemberError):
        reg.resolve("nope")


def test_registry_resolve_out_of_range_index_raises_unknown_member_error():
    reg = _MemberRegistry()
    reg.register(None)
    reg.register(None)
    with pytest.raises(UnknownMemberError):
        reg.resolve(2)
    with pytest.raises(UnknownMemberError):
        reg.resolve(-1)


def test_registry_duplicate_name_raises_value_error():
    reg = _MemberRegistry()
    reg.register("dup")
    with pytest.raises(ValueError):
        reg.register("dup")


def test_recorded_schedule_set_member_enabled_bad_handle_is_loud():
    """A toggle attempt against an unregistered name or an out-of-range
    index fails loudly, by name/index, never silently -- ``RecordedSchedule``
    resolves through the SAME ``_MemberRegistry.resolve`` GraphPipeline
    uses."""
    sched = RecordedSchedule()
    sched.add(lambda: None, name="only")
    with pytest.raises(UnknownMemberError):
        sched.set_member_enabled("nonexistent", True)
    with pytest.raises(UnknownMemberError):
        sched.set_member_enabled(7, True)


def test_unknown_member_error_is_a_lookup_error():
    """``LookupError``, not ``KeyError``/``IndexError`` alone --
    ``_MemberRegistry.resolve`` accepts BOTH a name (dict-style) and an
    index (sequence-style) lookup; a caller catching "any resolution
    failure" should not need two ``except`` clauses."""
    assert issubclass(UnknownMemberError, LookupError)
    err = UnknownMemberError("member index 3 out of range (0..1)")
    assert "3" in str(err)


def test_non_toggleable_member_error_is_a_type_error():
    """Type-level contract only. The REAL trigger -- a member containing a
    :class:`~eagle._conditional.Skippable`'s conditional node, which makes
    :meth:`~eagle.pipeline.GraphPipeline.set_member_enabled` refuse the
    toggle -- is CUDA-only and lives in the CUDA-ARM section below; this
    just proves the exception it raises there is loud and correctly typed:
    a ``TypeError`` ("this member is the wrong KIND of thing to toggle"),
    not a ``KeyError``/``IndexError`` (which would wrongly suggest a lookup
    problem)."""
    assert issubclass(NonToggleableMemberError, TypeError)
    err = NonToggleableMemberError("member 0 ('gated') is not toggleable: ...")
    assert "not toggleable" in str(err)


# ============================================================================
# (b) CUDA-ARM -- pytest.mark.gpu. AUTHORED, NOT YET RUN (see
# module docstring): eagle._core predates capture_snapshot_nodes /
# is_node_toggleable / Launcher.set_node_enabled. cupy and the
# capture-needing eagle symbols are imported LOCALLY, inside each test, so
# collecting this file never requires cupy.
# ============================================================================


def _warm(kernel, *args):
    """NVRTC-compile ``kernel`` outside capture (compilation is illegal
    mid-capture); callers pass scratch args so real buffers stay untouched."""
    import cupy as cp

    kernel(*args)
    cp.cuda.runtime.deviceSynchronize()


@pytest.mark.gpu
def test_member_enable_bitwise_twin_disabled_untouched_enabled_matches_never_toggled():
    """mode="enabled" bitwise twin (mirrors the corresponding certification for
    mode="conditional"): a member's launches are captured ONCE; disabling it leaves its
    buffer exactly as it was before that replay (a driver-level no-op, not a conditional
    skip); re-enabling it -- SAME captured graph, no rebuild -- reproduces bit-for-bit
    what a pipeline that was NEVER
    toggled would have produced. Toggling perturbs only WHETHER a member's
    nodes run, never WHAT they compute."""
    import cupy as cp

    from eagle.pipeline import GraphPipeline

    n = 256
    a = cp.arange(n, dtype=cp.float64)
    k = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 2.0 + 1.0", "ns_enable_twin"
    )

    out = cp.full(n, _SENTINEL, dtype=cp.float64)
    _warm(k, a, cp.empty_like(out))

    pipe = GraphPipeline()
    pipe.add(lambda: k(a, out), name="only")
    pipe.build()

    # disabled: buffer must stay exactly as pre-replay (sentinel survives).
    out.fill(_SENTINEL)
    pipe.set_member_enabled("only", False)
    pipe.launch()
    assert np.all(cp.asnumpy(out) == _SENTINEL), (
        "disabled member must leave its buffer untouched"
    )

    # re-enabled: SAME captured graph, no rebuild.
    pipe.set_member_enabled("only", True)
    pipe.launch()
    expected = cp.asnumpy(a) * 2.0 + 1.0
    assert np.array_equal(cp.asnumpy(out), expected)

    # A sibling pipeline that was NEVER toggled must match bit-for-bit.
    never_toggled_out = cp.full(n, _SENTINEL, dtype=cp.float64)
    never_toggled_pipe = GraphPipeline()
    never_toggled_pipe.add(lambda: k(a, never_toggled_out))
    never_toggled_pipe.build()
    never_toggled_pipe.launch()
    assert np.array_equal(cp.asnumpy(out), cp.asnumpy(never_toggled_out))


@pytest.mark.gpu
def test_member_enable_three_toggle_patterns_no_recapture():
    """>= 3 distinct toggle patterns replay against ONE captured/instantiated
    graph -- no recapture, no re-instantiate -- each pattern's per-member
    output exact: enabled = matches the unguarded computation, disabled =
    sentinel survives."""
    import cupy as cp

    from eagle.pipeline import GraphPipeline

    n = 64
    dataA = cp.arange(n, dtype=cp.float64)
    dataB = cp.arange(n, dtype=cp.float64) * 3.0 - 2.0
    kA = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 10.0", "ns_enable_laneA"
    )
    kB = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 5.0", "ns_enable_laneB"
    )

    outA = cp.full(n, _SENTINEL, dtype=cp.float64)
    outB = cp.full(n, _SENTINEL, dtype=cp.float64)
    _warm(kA, dataA, cp.empty_like(outA))
    _warm(kB, dataB, cp.empty_like(outB))

    pipe = GraphPipeline()
    pipe.add(lambda: kA(dataA, outA), name="A")
    pipe.add(lambda: kB(dataB, outB), name="B")
    pipe.build()  # ONE capture for the whole test

    expectedA = cp.asnumpy(dataA) + 10.0
    expectedB = cp.asnumpy(dataB) * 5.0

    patterns = [(True, False), (False, True), (True, True)]  # >= 3, no repeats
    for on_a, on_b in patterns:
        outA.fill(_SENTINEL)
        outB.fill(_SENTINEL)
        pipe.set_member_enabled("A", on_a)
        pipe.set_member_enabled("B", on_b)
        pipe.launch()  # SAME launcher/instantiated graph throughout
        tag = f"pattern {(on_a, on_b)}"

        if on_a:
            assert np.array_equal(cp.asnumpy(outA), expectedA), f"{tag}: A"
        else:
            assert np.all(cp.asnumpy(outA) == _SENTINEL), f"{tag}: A"
        if on_b:
            assert np.array_equal(cp.asnumpy(outB), expectedB), f"{tag}: B"
        else:
            assert np.all(cp.asnumpy(outB) == _SENTINEL), f"{tag}: B"


@pytest.mark.gpu
def test_member_enable_toggle_then_launch_ordering():
    """``set_member_enabled`` takes effect on the NEXT :meth:`launch`, never
    a stale state from a replay already in flight: alternating
    disable/enable across four successive launches on ONE captured graph
    produces the expected buffer state every single time, in order."""
    import cupy as cp

    from eagle.pipeline import GraphPipeline

    n = 32
    a = cp.arange(n, dtype=cp.float64)
    k = cp.ElementwiseKernel("float64 x", "float64 y", "y = x + 1.0", "ns_enable_order")
    out = cp.full(n, _SENTINEL, dtype=cp.float64)
    _warm(k, a, cp.empty_like(out))

    pipe = GraphPipeline()
    pipe.add(lambda: k(a, out), name="only")
    pipe.build()

    for enabled in (False, True, False, True):
        out.fill(_SENTINEL)
        pipe.set_member_enabled("only", enabled)
        pipe.launch()
        if enabled:
            assert np.array_equal(cp.asnumpy(out), cp.asnumpy(a) + 1.0)
        else:
            assert np.all(cp.asnumpy(out) == _SENTINEL)


@pytest.mark.gpu
def test_member_enable_attribution_disjoint_and_covers_whole_graph():
    """Node attribution (``GraphPipeline.build()``): every member's
    captured node-set delta has >= 1 node, no two members share a node
    (disjoint), and the union across all members equals the graph's total
    node count -- attribution neither drops nor double-counts a node. Uses
    a two-launch composed member (matching the "compose inside the
    callable" contract) alongside a one-launch member so the delta sizes
    differ."""
    import cupy as cp

    from eagle.pipeline import GraphPipeline

    n = 48
    a = cp.arange(n, dtype=cp.float64)
    b = cp.arange(n, dtype=cp.float64) * 2.0

    k1 = cp.ElementwiseKernel("float64 x", "float64 y", "y = x + 1.0", "ns_attr_k1")
    k2 = cp.ElementwiseKernel("float64 x", "float64 y", "y = x * 2.0", "ns_attr_k2")
    k3 = cp.ElementwiseKernel("float64 x", "float64 y", "y = x - 3.0", "ns_attr_k3")
    out1 = cp.empty(n, dtype=cp.float64)
    out2 = cp.empty(n, dtype=cp.float64)
    out3 = cp.empty(n, dtype=cp.float64)
    _warm(k1, a, out1)
    _warm(k2, a, out2)
    _warm(k3, b, out3)

    def composed_two_launch_member():
        k1(a, out1)
        k2(out1, out2)

    pipe = GraphPipeline()
    pipe.add(composed_two_launch_member, name="composed")  # 2 kernel nodes
    pipe.add(lambda: k3(b, out3), name="single")  # 1 kernel node
    pipe.build()

    composed_nodes = set(pipe.member_nodes("composed"))
    single_nodes = set(pipe.member_nodes("single"))

    assert len(composed_nodes) >= 1
    assert len(single_nodes) >= 1
    assert composed_nodes.isdisjoint(single_nodes), "members must not share a node"
    assert len(composed_nodes | single_nodes) == pipe.num_nodes(), (
        "union of every member's attributed nodes must equal the graph's total"
    )
    assert pipe.is_member_toggleable("composed")
    assert pipe.is_member_toggleable("single")


@pytest.mark.gpu
def test_member_enable_rejects_toggle_on_member_containing_skippable():
    """A member whose captured node set includes a
    :class:`~eagle._conditional.Skippable`'s conditional (IF) node --
    composing mode="conditional" and mode="enabled" on the SAME member --
    is recorded non-toggleable at :meth:`~eagle.pipeline.GraphPipeline.build`
    time, and :meth:`~eagle.pipeline.GraphPipeline.set_member_enabled`
    refuses it loudly, by name, rather than silently mis-toggling at the
    wrong granularity or failing deep inside a raw
    ``cudaGraphNodeSetEnabled`` call."""
    import cupy as cp

    from eagle import SkipGuard, skippable
    from eagle.pipeline import GraphPipeline

    n = 32
    a = cp.arange(n, dtype=cp.float64)
    k = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 1.0", "ns_enable_reject"
    )
    out = cp.full(n, _SENTINEL, dtype=cp.float64)
    _warm(k, a, cp.empty_like(out))

    flags = cp.ones(1, dtype=cp.uint32)  # always-on intent guard
    guard = SkipGuard.intent(flags, 0)
    gated = skippable(lambda: k(a, out), guard)

    pipe = GraphPipeline()
    pipe.add(gated, name="gated")
    pipe.build()

    assert not pipe.is_member_toggleable("gated")
    with pytest.raises(NonToggleableMemberError):
        pipe.set_member_enabled("gated", False)


@pytest.mark.gpu
def test_member_enable_before_build_is_loud():
    """``set_member_enabled`` before ``build()`` raises ``RuntimeError`` --
    the same discipline as :meth:`~eagle.pipeline.GraphPipeline.launch`'s
    existing pre-build guard, never a silent no-op or an ``AttributeError``
    from touching an as-yet-``None`` launcher."""
    from eagle.pipeline import GraphPipeline

    pipe = GraphPipeline()
    pipe.add(lambda: None, name="x")
    with pytest.raises(RuntimeError):
        pipe.set_member_enabled("x", True)


@pytest.mark.gpu
def test_set_node_enabled_before_build_is_loud():
    """The new registry-bypassing passthrough carries the SAME pre-build
    discipline as ``set_member_enabled`` -- ``RuntimeError``, never an
    ``AttributeError`` from touching an as-yet-``None`` launcher."""
    from eagle.pipeline import GraphPipeline

    pipe = GraphPipeline()
    pipe.add(lambda: None, name="x")
    with pytest.raises(RuntimeError):
        pipe.set_node_enabled(0, True)


@pytest.mark.gpu
def test_set_node_enabled_toggles_a_raw_node_the_same_as_set_member_enabled():
    """``set_node_enabled`` reaches the SAME driver call
    ``set_member_enabled`` does, just addressed by raw node handle instead
    of by registry name/index -- disabling ``only``'s own node through the
    new door leaves its buffer untouched exactly like disabling the member
    through the old one; re-enabling resumes updates; and driving BOTH
    doors against the SAME node produces the SAME final state, because
    they are the same call underneath."""
    import cupy as cp

    from eagle.pipeline import GraphPipeline

    n = 128
    a = cp.arange(n, dtype=cp.float64)
    k = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 3.0 - 1.0", "ns_set_node_enabled"
    )

    out = cp.full(n, _SENTINEL, dtype=cp.float64)
    _warm(k, a, cp.empty_like(out))

    pipe = GraphPipeline()
    pipe.add(lambda: k(a, out), name="only")
    pipe.build()
    (node,) = pipe.member_nodes("only")

    # Disable via the RAW-NODE door -- buffer must stay untouched.
    out.fill(_SENTINEL)
    pipe.set_node_enabled(node, False)
    pipe.launch()
    assert np.all(cp.asnumpy(out) == _SENTINEL)

    # Re-enable via the RAW-NODE door -- same captured graph, no rebuild.
    pipe.set_node_enabled(node, True)
    pipe.launch()
    expected = cp.asnumpy(a) * 3.0 - 1.0
    assert np.array_equal(cp.asnumpy(out), expected)

    # Cross-check: disabling via the OLD (registry) door and re-enabling via
    # the NEW (raw-node) door lands in the same enabled state either way.
    out.fill(_SENTINEL)
    pipe.set_member_enabled("only", False)
    pipe.launch()
    assert np.all(cp.asnumpy(out) == _SENTINEL)
    pipe.set_node_enabled(node, True)
    pipe.launch()
    assert np.array_equal(cp.asnumpy(out), expected)
