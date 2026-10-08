# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""STOP-THE-LINE incident fix: GC-triggered CUDA teardown mid-capture
invalidates an unrelated active capture.

**Mechanism** (``eagle/util/CaptureGuard.h`` carries the full writeup): a
Python-bound object (``Launcher``, ``Stream``, ``Event`` via ``CaptureFork``,
``CapturedGraph``, ``Graph`` via ``Graph::Storage``, ``ScratchArena``) whose
C++ destructor calls a CUDA-touching teardown (``cudaGraphExecDestroy``,
``cudaStreamDestroy``, ``cudaGraphDestroy``, ``cudaFree``, ...) can become
garbage at a time entirely governed by Python's GC -- including DURING a
LATER, completely unrelated, still-active capture elsewhere in the process.
That teardown call always swallows its OWN error (``EAGLE_CHECK_NOTHROW``,
correct, by design -- a throwing destructor unwinding during
teardown would abort the process rather than report) but the driver call
still RUNS and returns "operation not permitted when stream is capturing",
which INVALIDATES the ambient capture (``cudaErrorStreamCaptureInvalidated``,
901) -- the next operation on it fails, and the poisoned native state can
crash much later, at an unrelated GC or interpreter exit (the exact shape
this incident's original segfault took, three frames of pure-Python
the code generator's tracing away from anything touching CUDA).

**Fix**: ``eagle::util::CaptureGuardState`` -- a process-wide capture-depth
counter (entered/exited by ``StreamCapturer::begin/end`` and
``CaptureConditional::begin/end``, the two capture-opening entry points in
this codebase) plus a deferred-destroy queue every CUDA-touching teardown
site now routes through (``destroyOrDefer``): runs immediately if no capture
is active anywhere in the process (today's exact timing, unchanged), or
queues to run once the OUTERMOST capture scope exits (drained also as a
process-exit backstop). The ``EAGLE_CHECK_NOTHROW`` swallow paths are
UNCHANGED -- the fix is never making the illegal call while a capture is in
flight, not unswallowing its error.

**Red-proof shape**: force the exact scenario -- build a throwaway engine,
wrap it in a reference CYCLE (survives refcounting, reachable only via
cyclic GC -- the same shape a dropped neural-layer object takes), start a
SECOND, unrelated pipeline's capture, and call ``gc.collect()`` explicitly
from INSIDE that active capture window. Assert, via
``eagle._core.capture_guard_depth`` / ``capture_guard_pending_count``
(test/diagnostic-only introspection) and OS-file-descriptor-level stderr
capture (``EAGLE_CHECK_NOTHROW`` prints through C's ``fprintf(stderr, ...)``,
which bypasses Python's ``sys.stderr`` / ``contextlib.redirect_stderr``
entirely -- confirmed while red-proofing this exact test, see below): (a) no
``EAGLE_CHECK_NOTHROW`` line fires during the ``gc.collect()`` window, (b)
the dead engine's teardown is QUEUED (pending count rises) rather than run,
(c) the queue drains (pending count returns to 0) once the second pipeline's
capture ends, and (d) that capture completes and replays bit-for-bit
correct -- proving the deferred teardown never touched the ambient capture.

**Confirmed RED against the pre-fix build** (this incident's diagnosis):
a hand-driven repro of this exact scenario (same reference-cycle
+ mid-capture ``gc.collect()`` shape, without the ``capture_guard_*``
introspection this file also needs -- that binding does not exist pre-fix)
showed ``capturer.end()`` raise
``RuntimeError: ... cudaErrorStreamCaptureInvalidated (901) ...`` and the fd
2-level capture contained the EXACT ``EAGLE_CHECK_NOTHROW: CUDA error
(swallowed, teardown) at eagle/cuda/Launcher.h's `destroyInstance_`:
operation not permitted when stream is capturing`` line the incident report
named. This file itself requires the rebuilt ``eagle._core`` (it calls the
new ``capture_guard_depth``/``capture_guard_pending_count`` bindings, which
do not exist in the pre-fix binary) -- authored to run GREEN once eagle._core
is rebuilt with the fix in ``eagle/util/CaptureGuard.h`` and the deferred
call sites it guards (``Stream.h`` / ``Event.h`` / ``CapturedGraph.h`` /
``Launcher.h`` / ``Graph.h`` / ``ScratchArena.h``).
"""

from __future__ import annotations

import gc
import os
import tempfile

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


class _CapturedFd2:
    """Redirect the RAW OS file descriptor 2 (stderr) to a temp file for the
    scope of the ``with`` block. ``EAGLE_CHECK_NOTHROW`` prints via C's
    ``fprintf(stderr, ...)``, which writes directly to fd 2 -- Python's
    ``sys.stderr`` / ``contextlib.redirect_stderr`` do NOT intercept that
    (verified directly while building this test's red-proof: the
    ``contextlib`` form silently missed the line; this form catches it)."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryFile(mode="w+b")
        self._saved_fd2 = os.dup(2)
        os.dup2(self._tmp.fileno(), 2)
        return self

    def __exit__(self, *exc):
        os.dup2(self._saved_fd2, 2)
        os.close(self._saved_fd2)
        self._tmp.seek(0)
        self.text = self._tmp.read().decode(errors="replace")
        self._tmp.close()


def _build_dead_cyclic_engine(cp):
    """Build a GraphPipeline+Launcher and wrap it in a reference cycle so it
    survives refcounting and is reachable ONLY via cyclic GC -- the shape a
    dropped neural-layer object takes in real use (forward/backward/JVP
    ``GraphPipeline``s go out of scope together, often via a cycle through
    the owning Python wrapper)."""
    from eagle.pipeline import GraphPipeline

    a = cp.arange(64, dtype=cp.float64)
    out = cp.empty(64, dtype=cp.float64)
    k = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x + 1.0", "stopline_dead_engine_k"
    )
    k(a, out)
    cp.cuda.runtime.deviceSynchronize()

    pipe = GraphPipeline()
    pipe.add(lambda: k(a, out))
    pipe.build()

    cycle = {"self": None, "pipe": pipe}
    cycle["self"] = cycle  # cycle -> cycle, cycle -> pipe: needs cyclic GC
    return cycle


def test_gc_during_active_capture_defers_teardown_and_capture_survives():
    """Force a dead engine's Launcher to be collected DURING a second,
    unrelated pipeline's active capture; assert the deferred-teardown guard
    keeps that capture intact end-to-end (the four-part red-proof described
    in the module docstring)."""
    import cupy as cp

    from eagle import _core
    from eagle.pipeline import GraphPipeline

    gc.disable()  # only this test's own gc.collect() call drives cyclic GC
    try:
        cycle = _build_dead_cyclic_engine(cp)
        del cycle  # refcount drop only -- still alive via the self-cycle

        assert _core.capture_guard_depth() == 0, (
            "no capture should be active before this test's own capture begins"
        )
        assert _core.capture_guard_pending_count() == 0

        n = 128
        a2 = cp.arange(n, dtype=cp.float64)
        out2 = cp.empty(n, dtype=cp.float64)
        k2 = cp.ElementwiseKernel(
            "float64 x", "float64 y", "y = x * 2.0", "stopline_second_k"
        )
        k2(a2, out2)
        cp.cuda.runtime.deviceSynchronize()

        pipe2 = GraphPipeline()
        capturer = _core.StreamCapturer(pipe2._sptr)
        fdcap = _CapturedFd2()
        with pipe2._ext:
            capturer.begin()
            assert _core.capture_guard_depth() >= 1, (
                "depth must be >=1 once this capture is open"
            )
            k2(a2, out2)
            with fdcap:
                gc.collect()  # forces the dead engine's Launcher dtor HERE
            pending_during_capture = _core.capture_guard_pending_count()
            captured = capturer.end()

        # (a) the illegal teardown call must never have RUN at all during
        # the gc.collect() window -- deferred, not attempted-then-swallowed.
        assert "EAGLE_CHECK_NOTHROW" not in fdcap.text, (
            "a CUDA teardown call ran mid-capture (should have been "
            f"deferred instead); captured stderr: {fdcap.text!r}"
        )
        # (b) it was actually QUEUED, not silently dropped or run early.
        assert pending_during_capture >= 1, (
            "the dead engine's Launcher teardown should have been queued "
            f"while the capture was active (saw {pending_during_capture})"
        )
        # (c) the queue drains once the capture ends.
        assert _core.capture_guard_pending_count() == 0, (
            "deferred teardown(s) must have run by the time capture ended"
        )
        assert _core.capture_guard_depth() == 0

        # (d) the capture itself is undamaged: instantiate, replay, check
        # bitwise correctness.
        g = _core.Graph.from_captured(captured)
        g.stream(pipe2._sptr)
        launcher = g.launcher()
        launcher.stream(pipe2._sptr)
        out2.fill(-777.0)
        launcher.launch()
        launcher.synchronize()
        expected = cp.asnumpy(a2) * 2.0
        assert np.array_equal(cp.asnumpy(out2), expected), (
            "the second pipeline's capture must replay bit-for-bit correct "
            "despite the dead engine's teardown landing mid-capture"
        )
    finally:
        gc.enable()


def test_gc_capture_guard_no_active_capture_runs_teardown_immediately():
    """Control: with NO capture active anywhere, a dropped engine's teardown
    must run RIGHT AWAY (pending count returns to 0 immediately after
    ``gc.collect()``, no artificial delay). The fix changes WHEN a teardown
    runs only while a capture is in flight; it must not turn every teardown
    into a permanently-queued one."""
    import cupy as cp

    from eagle import _core

    gc.disable()
    try:
        assert _core.capture_guard_depth() == 0
        cycle = _build_dead_cyclic_engine(cp)
        del cycle
        gc.collect()
        assert _core.capture_guard_pending_count() == 0, (
            "with no active capture, a collected engine's teardown must run "
            "immediately, not sit queued"
        )
    finally:
        gc.enable()
