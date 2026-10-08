# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""``CaptureFork`` + ``GraphPipeline.add_concurrent``.

Graph/binding-tier tests (mirrors ``test_graph_plugin.py`` / ``test_pipeline_
parity.py``: function-style ``def test_<behavior>:``, module-level
``pytestmark = pytest.mark.gpu``) rather than ``test_neural_*.py``, since
``CaptureFork`` is a bound ``eagle::cuda`` primitive and ``add_concurrent`` is
``GraphPipeline``'s Python-surface fork/join sugar over it -- neither is
neural-engine-specific.

``test_capture_fork_round_trip_two_branches`` is PROMOTED from the working
scratch round-trip (spec acceptance #3):
fork two branches inside one capture, launch pre-allocated cupy work on each,
join, adopt via ``from_captured``, replay twice without rebuild, match a numpy
oracle, and assert the captured DAG has NO sibling-sibling edge between the
two branch kernels (transitive-redundant edges are PERMITTED -- probe alpha
saw a benign origin-frontier edge; this file does not assert against it).
"""

from __future__ import annotations

import re

import eagle._core as _core
import numpy as np
import pytest

from eagle.pipeline import GraphPipeline, _concurrent_write_conflict_check, _debug_dot

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

N = 1024


def _branch_kernels():
    kA = cp.ElementwiseKernel("float64 x", "float64 y", "y = x*2.0", "branchA")
    kB = cp.ElementwiseKernel("float64 x", "float64 y", "y = x+1.0", "branchB")
    return kA, kB


def _dot_edges_and_labels(dot: str):
    """Parse a ``cudaGraphDebugDotPrint`` dot string into ``(labels, edges)``."""
    labels = dict(re.findall(r'"([^"]+)"\s*\[[^\]]*label="([^"]*)"', dot, re.S))
    edges = re.findall(r'"([^"]+)"\s*->\s*"([^"]+)"', dot)
    return labels, edges


def _assert_no_sibling_edge(dot: str, name_a: str, name_b: str):
    labels, edges = _dot_edges_and_labels(dot)
    nA = [k for k, v in labels.items() if name_a in v]
    nB = [k for k, v in labels.items() if name_b in v]
    assert nA and nB, f"both {name_a!r} and {name_b!r} must appear as graph nodes"
    sib = [
        (u, v) for (u, v) in edges if (u in nA and v in nB) or (u in nB and v in nA)
    ]
    assert not sib, f"sibling-sibling edge present between {name_a}/{name_b}: {sib}"


# --------------------------------------------------------------------------- #
# 1. Promoted round-trip: raw CaptureFork binding, two branches, one capture.
# --------------------------------------------------------------------------- #
def test_capture_fork_round_trip_two_branches():
    a = cp.arange(N, dtype=cp.float64)
    b = cp.arange(N, dtype=cp.float64) * 3.0
    outA = cp.empty_like(a)
    outB = cp.empty_like(b)
    kA, kB = _branch_kernels()
    # Warm up OUTSIDE the capture: the first call NVRTC-compiles the kernel,
    # and neither compilation nor allocation is legal mid-capture.
    kA(a, outA)
    kB(b, outB)
    cp.cuda.runtime.deviceSynchronize()

    oracleA = cp.asnumpy(a) * 2.0
    oracleB = cp.asnumpy(b) + 1.0

    st = _core.Stream(non_blocking=True)
    ext = cp.cuda.ExternalStream(st.ptr())

    # CaptureFork MUST be constructed before begin() -- it creates streams/events.
    fork = _core.CaptureFork(st.ptr(), 2)
    assert len(fork) == 2
    assert fork.size() == 2
    assert not fork.forked()
    assert fork.origin() == st.ptr()

    br0 = cp.cuda.ExternalStream(fork.branch(0))
    br1 = cp.cuda.ExternalStream(fork.branch(1))

    cap = _core.StreamCapturer(st.ptr())
    with ext:
        cap.begin()
        fork.fork()
        assert fork.forked()
        with br0:
            kA(a, outA)
        with br1:
            kB(b, outB)
        fork.join()
        assert not fork.forked()
        captured = cap.end()

    assert bool(captured)

    # Dot BEFORE from_captured consumes the CapturedGraph.
    dot = _debug_dot(captured)
    _assert_no_sibling_edge(dot, "branchA", "branchB")

    g = _core.Graph.from_captured(captured)
    g.stream(st.ptr())
    launcher = g.launcher()
    launcher.stream(st.ptr())

    for replay in (1, 2):
        outA.fill(0.0)
        outB.fill(0.0)
        cp.cuda.runtime.deviceSynchronize()
        launcher.launch()
        launcher.synchronize()
        gotA = cp.asnumpy(outA)
        gotB = cp.asnumpy(outB)
        assert np.array_equal(gotA, oracleA), f"replay {replay}: branch A mismatch"
        assert np.array_equal(gotB, oracleB), f"replay {replay}: branch B mismatch"

    # A consumed (falsy) CapturedGraph must be rejected by from_captured.
    assert not bool(captured)
    with pytest.raises(ValueError):
        _core.Graph.from_captured(captured)


# --------------------------------------------------------------------------- #
# 2. GraphPipeline.add_concurrent -- fork topology + replay conformance.
# --------------------------------------------------------------------------- #
def test_add_concurrent_fork_topology_no_sibling_edge():
    a = cp.arange(N, dtype=cp.float64)
    b = cp.arange(N, dtype=cp.float64) * 3.0
    outA = cp.empty_like(a)
    outB = cp.empty_like(b)
    kA, kB = _branch_kernels()
    kA(a, outA)
    kB(b, outB)
    cp.cuda.runtime.deviceSynchronize()

    pipe = GraphPipeline()
    pipe.add_concurrent([lambda: kA(a, outA), lambda: kB(b, outB)])
    pipe.build()

    assert pipe.introspection_available()
    assert pipe.num_nodes() == 2
    _assert_no_sibling_edge(pipe._dot(), "branchA", "branchB")


def test_add_concurrent_replay_conformance():
    a = cp.arange(N, dtype=cp.float64)
    b = cp.arange(N, dtype=cp.float64) * 3.0
    outA = cp.empty_like(a)
    outB = cp.empty_like(b)
    kA, kB = _branch_kernels()
    kA(a, outA)
    kB(b, outB)
    cp.cuda.runtime.deviceSynchronize()
    oracleA = cp.asnumpy(a) * 2.0
    oracleB = cp.asnumpy(b) + 1.0

    pipe = GraphPipeline()
    pipe.add_concurrent([lambda: kA(a, outA), lambda: kB(b, outB)])
    pipe.build()

    outA.fill(0.0)
    outB.fill(0.0)
    pipe.launch(2)  # build once, replay twice without rebuild
    assert np.array_equal(cp.asnumpy(outA), oracleA)
    assert np.array_equal(cp.asnumpy(outB), oracleB)


def test_add_concurrent_requires_at_least_two_members():
    pipe = GraphPipeline()
    with pytest.raises(ValueError):
        pipe.add_concurrent([])
    with pytest.raises(ValueError):
        pipe.add_concurrent([lambda: None])


# --------------------------------------------------------------------------- #
# 3. Nonvacuity: the topology assert helper itself must be able to go RED.
#    (Deliberate-break/restore for THIS specific check is done by hand during
#    review, not pinned as a permanent test.)
# --------------------------------------------------------------------------- #
def test_assert_no_sibling_edge_is_nonvacuous_on_a_synthetic_sibling_dot():
    """Self-falsification for the topology-assert HELPER: a synthetic dot with
    a genuine A->B edge must make :func:`_assert_no_sibling_edge` fail -- proves
    the helper can go RED, not just that it happened to pass above."""
    synthetic_dot = (
        '"1"[shape="octagon",label="1\\nbranchA\\n"];\n'
        '"2"[shape="octagon",label="2\\nbranchB\\n"];\n'
        '"1"->"2";\n'
    )
    with pytest.raises(AssertionError):
        _assert_no_sibling_edge(synthetic_dot, "branchA", "branchB")


# --------------------------------------------------------------------------- #
# 4. _concurrent_write_conflict_check -- the self-falsifying validator test.
# --------------------------------------------------------------------------- #
def test_concurrent_write_conflict_check_catches_shared_target():
    shared = cp.zeros(8, dtype=cp.float64)
    with pytest.raises(AssertionError):
        _concurrent_write_conflict_check([[shared], [shared]])


def test_concurrent_write_conflict_check_accepts_disjoint_targets():
    a = cp.zeros(8, dtype=cp.float64)
    b = cp.zeros(8, dtype=cp.float64)
    _concurrent_write_conflict_check([[a], [b]])  # must not raise


def test_concurrent_write_conflict_check_allows_overlap_within_one_member():
    a = cp.zeros(8, dtype=cp.float64)
    b = cp.zeros(8, dtype=cp.float64)
    # a repeated twice inside member 0 (legal: a member's own launches
    # serialize with each other) alongside a disjoint member 1.
    _concurrent_write_conflict_check([[a, a], [b]])
