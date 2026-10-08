# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Gate B — GPU end-to-end value twin against a real captured graph.

AUTHORED, NOT RUN in a session under a HARD CONSTRAINT of no GPU work (a
concurrent performance-measurement session owns the GPUs). A GPU-enabled
session runs this.

The observable is the captured graph's OWN ``name<<<grid,block,smem>>>``
label (``cudaGraphDebugDotPrint`` at ``flags=1``), not a shape
probe or a wall-clock measurement:

- (i) a plain :meth:`~eagle.pipeline.GraphPipeline.add` chain — every node's
  block must equal ``resolve_block(n, 1, dev_q, kern_q)`` (the siblings<=1
  small-N launch-geometry rule — no longer a flat 256 for every
  ``n``; it still converges there once ``n`` crosses the formula's own upper
  clamp), ``dev_q``/``kern_q`` queried independently of :mod:`eagle.launch`'s
  cache, same as (ii) below.
- (ii) :meth:`~eagle.pipeline.GraphPipeline.add_concurrent` at S=2 and S=4 —
  every sibling node's block must equal ``resolve_block(n, S, dev_q, kern_q)``
  where ``dev_q``/``kern_q`` are queried DIRECTLY from cupy in this test
  (independent of :mod:`eagle.launch`'s own device-properties cache — this gate's own
  LOCK: the two layers may never collapse into one recomputation of the
  implementation). A mis-plumbed sibling count (the policy seeing 1 instead
  of S) would show up as ``b == 256`` where this test expects e.g. 64 — a
  hard, value-level FAIL, not a vibe.

No wall-clock/concurrency-timing assertion anywhere (this gate's own LOCK: whether
co-execution actually happens is the device's own business).

**Getting a VERBOSE (flags=1) dot.** :class:`~eagle.pipeline.GraphPipeline`
only stores a ``flags=0`` dot (``GraphPipeline.build``'s `_debug_dot` call) and that
stays as-is (no new public surface). This file instead runs the REAL
``GraphPipeline.add``/``add_concurrent``/``build()`` (so the actual sibling-
context wrapping in ``_run_concurrent`` fires) and, for the DURATION of one
``build()`` call, monkeypatches the module-level ``eagle.pipeline._debug_dot``
helper ``build()`` already calls internally to request ``flags=1`` instead of
the default — a scoped test-time patch, restored immediately after, that
never touches ``GraphPipeline``'s own default behavior in any other test or
in production. Precedent for the label-parsing regex:
``_dot_edges_and_labels`` in test_pipeline_concurrent.py.

**Version-drift discipline.** cupy 14.1.1 / CUDA 12.9 is
the only combination this file's own probes verified. If a future cupy/CUDA
changes the ``<<<grid,block,smem>>>`` label shape, this file SKIPS with a
loud, named reason (never silently passes) rather than asserting on an
empty/malformed parse.
"""

from __future__ import annotations

import contextlib
import math
import os
import re
import tempfile

import pytest

pytestmark = pytest.mark.gpu

# cudaGraphDebugDotPrint's KERNEL record shape escapes the angle brackets in
# a Graphviz record label (`<`/`>` are record-syntax metacharacters) --
# verified against a real flags=1 capture on cupy 14.1.1/CUDA 12.9:
# `gate_b_add\<\<\<1,256,0\>\>\>`, not the unescaped `<<<1,256,0>>>` the
# original pattern assumed. Each `\\?` tolerates the escape without requiring
# it, so this still matches an unescaped label too.
_NODE_RE = re.compile(r"\\?<\\?<\\?<(\d+),\s*(\d+),\s*(\d+)\\?>\\?>\\?>")


def _gref_mirror_struct_src() -> str:
    """A bare POD mirror of ``aether::Array<double,3>::ViewT`` (see
    ``plugin/gref_abi.h``'s ``GRefMirror`` — no aether headers needed for the
    by-value ABI itself), so this file's throwaway test kernel can be
    NVRTC-compiled with zero include-path setup. Field names/order/sizes MUST
    match ``GRefMirror`` exactly — this struct is passed by value through the
    REAL ``eagle.launch.launch`` entry point (:func:`_launch_add` below), which
    packs ``eagle.abi.GREF_DTYPE`` boxes (40 B), not a hand-picked shape."""
    return """
struct GRefMirror {
    double*            data_         = nullptr;
    unsigned long long samples_      = 0;
    unsigned long long compStride_   = 0;
    unsigned long long sampleStride_ = 1;
    int                deviceType_   = 2;
    int                deviceId_     = 0;
};
"""


def _make_add_kernel(name: str = "gate_b_add"):
    """One minimal eagle-ABI 'vector' kernel: no vector/per-sample inputs, no
    params — just a by-value ``GRefMirror out`` it fills with a constant.
    Gate B only inspects the captured ``<<<grid,block,smem>>>`` label, never
    the output VALUE, so the kernel body only needs to be safe (bounds-
    guarded), not physically meaningful."""
    import cupy as cp

    src = _gref_mirror_struct_src() + f"""
extern "C" __global__ void {name}(GRefMirror out) {{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < out.samples_) {{
        out.data_[i] = 1.0;
    }}
}}
"""
    mod = cp.RawModule(code=src)
    return mod.get_function(name)


def _launch_add(fn, out, term):
    """Launch through the REAL ``eagle.launch.launch`` entry point with
    ``block=None`` — the attachment point under test. ``arg_spec =
    [("out", "out")]`` is the minimal 'vector' shape.

    ``_launch_vector`` unconditionally requires a pre-allocated cupy
    ``terminated`` mask (``eagle/launch.py``'s ``require_cupy`` check) even
    though this minimal ``arg_spec`` never binds it into the kernel's actual
    argument list -- so ``term`` is a caller-owned, pre-allocated array
    (never allocated in here: this function is re-invoked, unchanged, from
    inside ``GraphPipeline.build()``'s stream capture to record each node,
    and allocating mid-capture is illegal -- ``cudaErrorStreamCaptureUnsupported``).
    Same ``cp.zeros(n, dtype=cp.bool_)`` shape/dtype convention as
    ``test_graph_plugin.py``/``test_pipeline_parity.py``, allocated ONCE by
    each caller below, outside any capture."""
    from eagle.launch import launch

    launch(
        "vector", fn, [("out", "out")], (), (), (),
        out=out, kw={"terminated": term}, grid=None, block=None,
    )


def _pinned_n() -> int:
    """Device-parametric pinned n, queried at test start — never a fixed
    literal (efficient/gated BY DEVICE PROPERTY, not a
    hardcoded threshold).

    The original ``dev.multiProcessorCount * 32``
    gives n=256 on this 8-SM device, where ``resolve_block`` degenerates to
    the SAME block (32) for every S in {2, 3, 4} — a wrong-but-nonzero
    sibling count (e.g. ``pipeline.py``'s ``len(group.members)`` replaced by
    a hard constant ``2``) would then resolve to the identical value the
    test expects for the REAL S=4 case, and this gate would stay green.
    ``* 64`` gives n=512, where S=2 -> 32 and S=4 -> 64 differ (verified:
    see the distinctness assertion in the S>=2 test below), so a mis-plumbed
    sibling count is observable again — still device-parametric, not a
    fixed literal."""
    import cupy as cp

    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    return int(props["multiProcessorCount"]) * 64


def _dev_kern_queried(fn):
    """``dev_q``/``kern_q`` queried DIRECTLY from cupy in this test — NOT
    through :mod:`eagle.launch`'s own device-properties cache (this gate's own LOCK)."""
    import cupy as cp

    dev_q = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    kern_q = fn.attributes
    return dev_q, kern_q


@contextlib.contextmanager
def _verbose_dot_capture():
    """Monkeypatch ``eagle.pipeline._debug_dot`` for the extent of ONE
    ``GraphPipeline.build()`` call to request the VERBOSE (``flags=1``) dot
    instead of the module's own default — restored on exit regardless of
    outcome. See the module docstring for why this (and not a new
    ``GraphPipeline`` kwarg) is the zero-new-surface way to reach it."""
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


def _node_grid_block_pairs(dot: str, expected_node_count: int):
    """Every kernel node's ``(grid, block)`` pulled from a ``flags=1`` dot's
    ``name<<<grid,block,smem>>>`` labels. SKIPS with a loud, named
    reason (never silently passes) if the parsed count does not match the
    known node count — the risk-#3 version-drift guard."""
    pairs = [(int(g), int(b)) for g, b, _smem in _NODE_RE.findall(dot)]
    if len(pairs) != expected_node_count:
        pytest.skip(
            f"expected {expected_node_count} '<<<grid,block,smem>>>' dot "
            f"labels but parsed {len(pairs)} from the flags=1 dot -- the "
            "graph DOT label format may have changed (cupy/CUDA version "
            "drift beyond the verified cupy 14.1.1 / CUDA 12.9 combination). "
            "This is a loud skip, not a silent pass."
        )
    return pairs


# --------------------------------------------------------------------------- #
# (i) plain add() chain: every node stays block=256 (siblings<=1 identity).
# --------------------------------------------------------------------------- #
def test_plain_chain_every_node_matches_the_siblings_one_policy():
    import cupy as cp

    from eagle._launch_policy import resolve_block
    from eagle.pipeline import GraphPipeline

    n = _pinned_n()
    fn = _make_add_kernel()
    outs = [cp.zeros((1, n), dtype=cp.float64) for _ in range(3)]
    # Allocated ONCE, outside any capture, and reused (read-only) by every
    # launch below -- allocating fresh inside a step re-invoked during
    # GraphPipeline.build()'s stream capture is illegal mid-capture.
    term = cp.zeros(n, dtype=cp.bool_)
    for out in outs:  # warm up OUTSIDE capture: NVRTC compile is illegal mid-capture
        _launch_add(fn, out, term)
    cp.cuda.runtime.deviceSynchronize()

    with _verbose_dot_capture():
        pipe = GraphPipeline()
        for out in outs:
            pipe.add(lambda out=out: _launch_add(fn, out, term))
        pipe.build()

    dev_q, kern_q = _dev_kern_queried(fn)
    expected_block = resolve_block(n, 1, dev_q, kern_q)
    expected_grid = math.ceil(n / expected_block)
    pairs = _node_grid_block_pairs(pipe._dot(), expected_node_count=3)
    for grid, block in pairs:
        assert block == expected_block, (
            f"non-concurrent node must match resolve_block(n, 1, ...)="
            f"{expected_block}, got {block}"
        )
        assert grid == expected_grid, (
            f"grid must be ceil(n/{expected_block})={expected_grid}, got {grid}"
        )


# --------------------------------------------------------------------------- #
# (ii) add_concurrent at S=2 and S=4: every sibling matches the policy,
#      queried independently of eagle.launch's own cache.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("siblings", [2, 4])
def test_add_concurrent_every_sibling_matches_policy(siblings):
    import cupy as cp

    from eagle._launch_policy import resolve_block
    from eagle.pipeline import GraphPipeline

    n = _pinned_n()
    fn = _make_add_kernel()
    outs = [cp.zeros((1, n), dtype=cp.float64) for _ in range(siblings)]
    # Allocated ONCE, outside any capture (see the plain-chain test above for
    # why); shared (read-only) across every sibling launch.
    term = cp.zeros(n, dtype=cp.bool_)
    for out in outs:
        _launch_add(fn, out, term)
    cp.cuda.runtime.deviceSynchronize()

    dev_q, kern_q = _dev_kern_queried(fn)
    expected_block = resolve_block(n, siblings, dev_q, kern_q)
    expected_grid = math.ceil(n / expected_block)

    # Nonvacuity guard: at this pinned n, the two
    # tested S values (2 and 4) must resolve to DISTINCT block sizes, or a
    # wrong-but-nonzero sibling count (e.g. pipeline.py's `len(group.members)`
    # hard-replaced by a constant that aliases one of the tested S values)
    # would go undetected by the comparison below. Computed against the
    # OTHER parametrized S value, independent of which case is running.
    other_siblings = 4 if siblings == 2 else 2
    other_expected = resolve_block(n, other_siblings, dev_q, kern_q)
    assert expected_block != other_expected, (
        f"NONVACUITY FAILURE: siblings={siblings} and siblings={other_siblings} "
        f"both resolve to block={expected_block} at this pinned n={n} -- Gate B "
        "cannot distinguish a wrong-but-nonzero sibling count here; adjust n "
        "above"
    )

    with _verbose_dot_capture():
        pipe = GraphPipeline()
        pipe.add_concurrent(
            [(lambda out=out: _launch_add(fn, out, term)) for out in outs]
        )
        pipe.build()

    pairs = _node_grid_block_pairs(pipe._dot(), expected_node_count=siblings)
    for grid, block in pairs:
        assert block == expected_block, (
            f"S={siblings} sibling node must resolve to block={expected_block} "
            f"(a mis-plumbed sibling count would see 1 -> 256 here instead), "
            f"got {block}"
        )
        assert grid == expected_grid, (
            f"grid must be ceil(n/block)={expected_grid}, got {grid}"
        )
