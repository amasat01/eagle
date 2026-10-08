# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Gate C — the nonvacuity ritual.

AUTHORED, NOT RUN here (same HARD CONSTRAINT as Gate B — one
half of this file needs a CUDA device). Both reds below must be recorded
when this file is run.

Two independent deliberate-breaks, proving each acceptance gate can actually
go RED (not just that it happened to pass):

1. **Gate B(ii) nonvacuity (GPU)**: with the ambient sibling count forced to
   1 (as if the context-manager line in
   :meth:`eagle.pipeline.GraphPipeline._run_concurrent` were hard-coded to 1,
   the block-size formula's own phrasing), an S>=2 ``add_concurrent`` group's
   sibling nodes would resolve to ``resolve_block(n, 1, dev_q, kern_q)``
   (the small-N formula, no longer always 256) instead of the S-aware
   value — Gate B(ii)'s own comparison (observed block ==
   ``resolve_block(n, S, dev_q, kern_q)``) must then FAIL. Forced via
   monkeypatching :func:`eagle._launch_policy.current_siblings` (the
   ambient-siblings READ side) rather than literally editing ``pipeline.py``
   — same "reach the identical broken state with zero production-code edits"
   idiom Gate B itself uses for its dot capture.

2. **Gate A reg-infeasible-row nonvacuity (pure, no GPU)**: with the
   register-feasibility search deleted from the resolver's formula (~:142-147 of
   ``_launch_policy.py``, replaced by an unconditional ``return bs``, no
   fallback), Gate A's own reg-infeasible pinned row
   (``test_launch_policy.py::test_reg_infeasible_row_falls_back_to_256``)
   would see something OTHER than 256 — proving that row is doing real work,
   not passing vacuously. The assertion below calls
   the REAL ``eagle._launch_policy.resolve_block`` on the row's own inputs
   (not a local hand-written twin) — a production change to the clamp flips
   THIS test's own pass/fail, not just a fact about a reimplementation kept
   on the side.
"""

from __future__ import annotations

import contextlib
import os
import tempfile

import pytest

# Only test 1 below needs a CUDA device; test 2 is pure. The file is marked
# `gpu` as a whole (matching Gate B's file-level marker) because it exists to
# be authored-and-run as ONE unit, following the
# "AUTHOR (do not run) Gate B, Gate C" pairing -- not because every test in
# it individually needs a GPU.
pytestmark = pytest.mark.gpu


@contextlib.contextmanager
def _verbose_dot_capture():
    """Monkeypatch ``eagle.pipeline._debug_dot`` for the extent of ONE
    ``GraphPipeline.build()`` call to request the VERBOSE (``flags=1``) dot
    instead of the module's own default (``flags=0``) -- restored on exit.
    ``GraphPipeline._dot()`` only ever returns what ``build()`` already
    computed and stored (never re-dots), so a flags=0 capture would carry NO
    ``<<<grid,block,smem>>>`` kernel-launch-config data at all (verified: it
    degrades to a plain ``shape="octagon"`` node with no launch config) --
    this test needs the SAME verbose-dot idiom Gate B uses (module docstring
    point 1), reimplemented locally (not imported from
    ``test_launch_policy_gate_b.py``) so this nonvacuity check has no
    dependency on that file staying in sync with it, matching how the kernel
    source above is also kept local."""
    import eagle.pipeline as pipeline_mod

    original = pipeline_mod._debug_dot

    def _flags1(captured):
        fd, path = tempfile.mkstemp(suffix=".dot")
        os.close(fd)
        try:
            return captured.debug_dot(path, 1)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    pipeline_mod._debug_dot = _flags1
    try:
        yield
    finally:
        pipeline_mod._debug_dot = original


# --------------------------------------------------------------------------- #
# 1. Gate B(ii) nonvacuity: siblings forced to 1 must make the comparison FAIL.
# --------------------------------------------------------------------------- #
def test_gate_b_ii_nonvacuous_when_siblings_forced_to_one(monkeypatch):
    """Reruns Gate B(ii)'s own shape (an S=2 ``add_concurrent`` group) with
    the ambient sibling count pinned to 1 throughout -- the same broken state
    "the context-manager line hard-coded to siblings=1" describes. The
    REAL comparison Gate B(ii) makes (observed node block ==
    ``resolve_block(n, S, dev_q, kern_q)`` with the REAL S=4) must then
    mismatch, because every launch under the hard-coded-1 regime actually
    resolves to ``resolve_block(n, 1, dev_q, kern_q)`` instead -- a distinct
    value at this pinned ``n`` (chosen, like Gate B's own ``_pinned_n()``, so
    S=1 and S=4 do not coincidentally land on the same block size)."""
    import cupy as cp

    import eagle._launch_policy as launch_policy
    from eagle._launch_policy import resolve_block
    from eagle.launch import launch
    from eagle.pipeline import GraphPipeline

    # Same throwaway kernel shape as Gate B (kept local + minimal so this
    # nonvacuity check has no dependency on Gate B's own file staying in
    # sync with it).
    # Field names/order/sizes MUST match plugin/gref_abi.h's GRefMirror exactly
    # (40 B): this struct is passed by value through the REAL eagle.launch.launch
    # entry point below, which packs eagle.abi.GREF_DTYPE boxes, not a hand-picked
    # shape (see test_launch_policy_gate_b.py's _gref_mirror_struct_src()).
    src = """
struct GRefMirror {
    double*            data_         = nullptr;
    unsigned long long samples_      = 0;
    unsigned long long compStride_   = 0;
    unsigned long long sampleStride_ = 1;
    int                deviceType_   = 2;
    int                deviceId_     = 0;
};
extern "C" __global__ void gate_c_add(GRefMirror out) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out.samples_) { out.data_[i] = 1.0; }
}
"""
    mod = cp.RawModule(code=src)
    fn = mod.get_function("gate_c_add")

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    # Same pinned n as test_launch_policy_gate_b.py's `_pinned_n()` (SMs*64):
    # that file's own comment already verified S=2 and S=4 resolve to
    # DISTINCT blocks (32 vs 64) at this n on an 8-SM device, which is also
    # what this test needs now that siblings=1 is n-dependent --
    # SMs*32 used to work only because siblings=1 was a flat 256, always
    # distinct from any small-n S>=2 value; it no longer is.
    n = int(props["multiProcessorCount"]) * 64
    siblings = 4

    # _launch_vector unconditionally requires a pre-allocated cupy
    # `terminated` mask (eagle/launch.py's require_cupy check) even though
    # this minimal arg_spec never binds it into the kernel's actual argument
    # list (same convention as test_graph_plugin.py/test_pipeline_parity.py).
    # Allocated ONCE, outside any capture: `_launch` below is re-invoked,
    # unchanged, from inside GraphPipeline.build()'s stream capture to record
    # each node, and allocating mid-capture is illegal
    # (cudaErrorStreamCaptureUnsupported) -- it must not allocate a fresh
    # array on every call.
    term = cp.zeros(n, dtype=cp.bool_)

    def _launch(out):
        launch(
            "vector", fn, [("out", "out")], (), (), (),
            out=out, kw={"terminated": term}, grid=None, block=None,
        )

    outs = [cp.zeros((1, n), dtype=cp.float64) for _ in range(siblings)]
    for out in outs:
        _launch(out)
    cp.cuda.runtime.deviceSynchronize()

    # The REAL expectation Gate B(ii) would assert against, with the REAL
    # siblings=4 -- computed once, BEFORE the deliberate break, from the same
    # independently-queried dev_q/kern_q Gate B itself uses. And the value the
    # hard-coded-1 break would force every node to, computed the same way
    # (no longer a bare 256 literal: siblings=1 is now n-dependent).
    expected_at_real_siblings = resolve_block(n, siblings, props, fn.attributes)
    expected_at_forced_one = resolve_block(n, 1, props, fn.attributes)

    # The deliberate break: force the ambient read to ALWAYS see 1, exactly
    # as if GraphPipeline._run_concurrent's `with sibling_context(siblings):`
    # line were hard-coded to `with sibling_context(1):`.
    monkeypatch.setattr(launch_policy, "current_siblings", lambda: 1)
    # eagle.launch imported `current_siblings` by name at module load time;
    # patch that binding too so the broken read actually reaches `launch()`.
    #
    # NOTE: `import eagle.launch as eagle_launch` would silently bind the
    # `launch` FUNCTION here, not the `eagle.launch` module -- `eagle/__init__.py`
    # does `from .launch import (..., launch, ...)`, which overwrites the
    # `launch` submodule attribute the import system placed on the `eagle`
    # package with the function of the same name, and `import a.b as c` is
    # sugar for `import a.b; c = a.b` (an ATTRIBUTE read on `a`, not a
    # `sys.modules["a.b"]` lookup) -- so `c` would end up the function, and
    # `setattr(c, "current_siblings", ...)` would be a silent no-op (functions
    # accept arbitrary attributes) that never reaches the real module's
    # binding `_resolved_block` actually reads. `importlib.import_module` is a
    # `sys.modules` lookup, immune to the shadowing.
    import importlib

    eagle_launch = importlib.import_module("eagle.launch")

    monkeypatch.setattr(eagle_launch, "current_siblings", lambda: 1)

    pipe = GraphPipeline()
    pipe.add_concurrent([(lambda out=out: _launch(out)) for out in outs])
    with _verbose_dot_capture():
        pipe.build()

    import re

    # cudaGraphDebugDotPrint's KERNEL record shape escapes the angle brackets
    # (Graphviz record-syntax metacharacters) -- verified against a real
    # flags=1 capture: `gate_b_add\<\<\<1,256,0\>\>\>`, not an unescaped
    # `<<<1,256,0>>>`. Each `\\?` tolerates the escape without requiring it.
    observed_blocks = {
        int(b)
        for _g, b, _s in re.findall(
            r"\\?<\\?<\\?<(\d+),\s*(\d+),\s*(\d+)\\?>\\?>\\?>", pipe._dot()
        )
    }

    if expected_at_real_siblings == expected_at_forced_one:
        pytest.skip(
            "this device/kernel combination resolves to the same block at "
            "the real siblings=4 and at the forced siblings=1 (e.g. a "
            "reg/smem-infeasible co-residency, or a device-SM-count "
            "coincidence) -- the hard-coded-1 break is not observable here; "
            "rerun with a kernel/n shaped like Gate B's reg-feasible pinned "
            "row"
        )
    assert observed_blocks != {expected_at_real_siblings}, (
        "Gate B(ii) NONVACUITY FAILURE: forcing siblings to 1 should have "
        f"made every node resolve to {expected_at_forced_one} (not the real "
        f"siblings=4 value {expected_at_real_siblings}), but the observed "
        f"block(s) {observed_blocks} still match it -- the sibling-count "
        "plumbing is not actually reaching the resolver, and Gate B(ii) "
        "would not catch a real regression here"
    )
    assert observed_blocks == {expected_at_forced_one}, (
        "expected the hard-coded-1 break to force every node to "
        f"resolve_block(n, 1, ...)={expected_at_forced_one}, got "
        f"{observed_blocks}"
    )


# --------------------------------------------------------------------------- #
# 2. Gate A reg-infeasible-row nonvacuity: deleting the clamp must go RED.
# --------------------------------------------------------------------------- #
def _hand_derived_value_without_reg_clamp(n, siblings, dev, kern):
    """Hand-derivation ONLY of what Gate A's reg-infeasible row would be if
    the reg-feasibility search were deleted from ``resolve_block`` (the raw
    clamped ``bs``, unconditionally, no fallback) -- NOT exercised by the
    nonvacuity assertion below (the previous
    version of this file asserted against this kind of local twin instead of
    the real implementation, so no production change could ever turn it
    red). Kept only to document the expected "without the clamp" number
    inline instead of a bare magic constant."""
    if siblings <= 1:
        return 256
    n_sms = dev["multiProcessorCount"]
    share = max(1, (n_sms * 4) // siblings)
    ideal = -(-n // share)
    bs = ((ideal + 31) // 32) * 32
    upper = min(256, (kern["max_threads_per_block"] // 32) * 32)
    return min(max(bs, 32), upper)


def test_gate_a_reg_infeasible_row_nonvacuous_without_clamp():
    """Gate A's own reg-infeasible pinned row
    (``test_launch_policy.py::test_reg_infeasible_row_falls_back_to_256``):
    n=512, S=4, dev={nSMs 8, regsPerSM 65536, smemPerSM 98304},
    kern={regs 600, smem 0, cap 1024} -> the REAL resolver returns 256
    because NO warp-multiple block keeps 4 co-resident blocks within the
    65536-register file (4*32*600 = 76800 > 65536, and it only gets worse at
    larger block sizes). Without the reg-feasibility search, the SAME inputs
    hand-derive to a DIFFERENT number:

        share = max(1, 8*4 // 4) = 8
        ideal = ceil(512/8) = 64  (already a warp multiple)
        clamp = min(max(64, 32), min(256, floor(1024/32)*32)) = 64

    -> 64, not 256.

    Fix: the assertion below calls the REAL
    ``eagle._launch_policy.resolve_block`` -- not the hand-derived twin
    above -- on exactly these inputs. Deleting the real reg-feasibility
    search (``_launch_policy.py``~:142-147, e.g. replacing the while-loop
    with an unconditional ``return bs``) makes the REAL resolver return 64
    here instead of 256, so THIS assertion goes RED, not just a fact about a
    reimplementation kept on the side."""
    dev = {
        "multiProcessorCount": 8,
        "regsPerMultiprocessor": 65536,
        "sharedMemPerMultiprocessor": 98304,
    }
    kern = {"num_regs": 600, "shared_size_bytes": 0, "max_threads_per_block": 1024}

    without_clamp = _hand_derived_value_without_reg_clamp(512, 4, dev, kern)
    assert without_clamp == 64, (
        f"hand-derivation says 64 without the clamp, got {without_clamp}"
    )

    from eagle._launch_policy import resolve_block

    real = resolve_block(512, 4, dev, kern)
    assert real == 256, (
        "Gate A's reg-infeasible row expects the REAL resolve_block to "
        f"return 256 (the saturation fallback), got {real} -- if the "
        "reg-feasibility search was removed from _launch_policy.py, this is "
        "the NONVACUITY FAILURE this test exists to catch"
    )
    assert real != without_clamp, (
        "Gate A NONVACUITY FAILURE: the reg-infeasible row's hand-derived "
        "without-clamp value (64) coincides with the real fallback (256) -- "
        "these pinned numbers no longer distinguish 'clamp present' from "
        "'clamp deleted'; re-pin the row's inputs"
    )
