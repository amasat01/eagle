# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Acceptance: :func:`eagle.repeat_while` / :class:`eagle.RepeatWhile` -- the
device-side loop (one CUDA WHILE conditional node) over
``eagle::cuda::CaptureConditional``'s loop kind. Mirrors
``test_conditional.py``'s shape: an UNGUARDED TRACER inside the body counts
iterations independently of eagle's own ``ran`` cell, so "the loop ran N
times" is proven by two independent witnesses (tracer and ``iterations()``),
never by an output value alone.

Acceptance rows (a)-(h) of the device-side loop, each with its RED noted in the
test docstring: how to make it fail by deleting the mechanism under test.

Lands in: this pytest file (no name manifest for the Python suite; the gate
is the ``pytest python/tests`` run in CI).
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

from eagle import RepeatWhile, SkipGuard, Skippable, repeat_while, skippable
from eagle._member_enable import NonToggleableMemberError
from eagle.compose import GraphComposer
from eagle.pipeline import GraphPipeline
from _graph_dot import clusters, conditional_count

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

_SRC = r"""
extern "C" __global__ void rw_tracer_kernel(int* counter) { atomicAdd(counter, 1); }
extern "C" __global__ void rw_bump_kernel(float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] * 1.0001f + 1.0f;
}
extern "C" __global__ void rw_bump_dec_kernel(float* y, int n, unsigned int* live) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = y[i] * 1.0001f + 1.0f;
    if (i == 0 && *live) --(*live);
}
extern "C" __global__ void rw_add_index_kernel(
    float* y, int n, const unsigned int* ran) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += float(ran[0]);
}
"""
# RawKernel compiles on first launch, so collecting this file needs no GPU.
_tracer = cp.RawKernel(_SRC, "rw_tracer_kernel")
_bump = cp.RawKernel(_SRC, "rw_bump_kernel")
_bump_dec = cp.RawKernel(_SRC, "rw_bump_dec_kernel")
_add_index = cp.RawKernel(_SRC, "rw_add_index_kernel")
_N = 256
_GRID = ((_N + 127) // 128,)
_BLOCK = (128,)


def _warm():
    """NVRTC-compile every kernel outside capture (compilation is illegal
    mid-capture), on throwaway buffers."""
    y = cp.zeros(_N, dtype=cp.float32)
    live = cp.ones(1, dtype=cp.uint32)
    t = cp.zeros(1, dtype=cp.int32)
    _tracer((1,), (1,), (t,))
    _bump(_GRID, _BLOCK, (y, np.int32(_N)))
    _bump_dec(_GRID, _BLOCK, (y, np.int32(_N), live))
    _add_index(_GRID, _BLOCK, (y, np.int32(_N), live))
    cp.cuda.runtime.deviceSynchronize()


@pytest.fixture(autouse=True)
def _warm_kernels():
    _warm()


def _tracer_value(t) -> int:
    return int(t.get()[0])


def _bump_dec_step(y, live):
    return lambda: _bump_dec(_GRID, _BLOCK, (y, np.int32(_N), live))


def _eager_steps(n):
    """The reference: n plain eager bump steps on a fresh buffer."""
    y = cp.zeros(_N, dtype=cp.float32)
    for _ in range(n):
        _bump(_GRID, _BLOCK, (y, np.int32(_N)))
    cp.cuda.runtime.deviceSynchronize()
    return cp.asnumpy(y)


# --------------------------------------------------------------------------- #
# (a) stop-at-guard. RED: tail ignores the guard (drop `*count != base` from
#     setLoopTailKernel) -> runs to the cap: iterations() == 64.
# --------------------------------------------------------------------------- #
def test_repeat_while_stops_at_guard():
    y = cp.zeros(_N, dtype=cp.float32)
    live = cp.asarray([5], dtype=cp.uint32)
    tracer = cp.zeros(1, dtype=cp.int32)

    def body():
        _bump_dec(_GRID, _BLOCK, (y, np.int32(_N), live))
        _tracer((1,), (1,), (tracer,))

    loop = repeat_while(body, SkipGuard.nonzero(live), 64)
    pipe = GraphPipeline().add(loop)
    pipe.build()
    pipe.launch()
    assert loop.iterations() == 5
    assert _tracer_value(tracer) == 5, "unguarded tracer: the body ran exactly 5 times"
    assert int(live.get()[0]) == 0
    assert np.array_equal(cp.asnumpy(y), _eager_steps(5)), "state == 5 eager steps"


# --------------------------------------------------------------------------- #
# (b) stop-at-cap + per-launch reset. RED: drop `counter[0] = cap` from
#     setLoopHeadKernel -> the second launch finds remaining == 0 -> 0 iterations.
# --------------------------------------------------------------------------- #
def test_repeat_while_stops_at_cap_and_resets_per_launch():
    y = cp.zeros(_N, dtype=cp.float32)
    live = cp.asarray([1000], dtype=cp.uint32)
    loop = repeat_while(_bump_dec_step(y, live), SkipGuard.nonzero(live), 64)
    pipe = GraphPipeline().add(loop)
    pipe.build()
    pipe.launch()
    assert loop.iterations() == 64
    pipe.launch()
    assert loop.iterations() == 64, (
        "second launch: head reset the cell, no accumulation"
    )
    assert int(live.get()[0]) == 1000 - 128
    assert np.array_equal(cp.asnumpy(y), _eager_steps(128))


# --------------------------------------------------------------------------- #
# (c) no host round trip. RED: replace the loop by a skippable launched 1000
#     times -> the counting launcher sees 1000 launches.
# --------------------------------------------------------------------------- #
class _CountingLauncher:
    def __init__(self, inner):
        self._inner = inner
        self.launches = 0

    def launch(self):
        self.launches += 1
        self._inner.launch()

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_repeat_while_is_one_graph_launch_and_never_reads_the_guard_on_host(
    monkeypatch,
):
    y = cp.zeros(_N, dtype=cp.float32)
    live = cp.asarray([1000], dtype=cp.uint32)
    loop = repeat_while(_bump_dec_step(y, live), SkipGuard.nonzero(live), 1000)
    pipe = GraphPipeline().add(loop)
    pipe.build()
    counting = _CountingLauncher(pipe._launcher)
    pipe._launcher = counting

    def _no_host_read(self):
        raise AssertionError("SkipGuard.evaluate must never run under a captured loop")

    monkeypatch.setattr(SkipGuard, "evaluate", _no_host_read)
    pipe.launch(1)
    assert counting.launches == 1, "1000 iterations from ONE cudaGraphLaunch"
    assert loop.iterations() == 1000
    assert np.array_equal(cp.asnumpy(y), _eager_steps(1000))


# --------------------------------------------------------------------------- #
# (d) bit identity across the three forms. RED: perturb one arm (e.g. an
#     extra bump in the eager arm) -> array_equal fails.
# --------------------------------------------------------------------------- #
def test_repeat_while_bit_identical_to_skippable_replays_and_host_loop():
    n = 16
    guard_word = cp.ones(1, dtype=cp.uint32)

    y_loop = cp.zeros(_N, dtype=cp.float32)
    loop = repeat_while(
        lambda: _bump(_GRID, _BLOCK, (y_loop, np.int32(_N))),
        SkipGuard.nonzero(guard_word),
        n,
    )
    p_loop = GraphPipeline().add(loop)
    p_loop.build()
    p_loop.launch(1)

    y_skip = cp.zeros(_N, dtype=cp.float32)
    p_skip = GraphPipeline().add(
        skippable(
            lambda: _bump(_GRID, _BLOCK, (y_skip, np.int32(_N))),
            SkipGuard.nonzero(guard_word),
        )
    )
    p_skip.build()
    p_skip.launch(n)

    y_host = cp.zeros(_N, dtype=cp.float32)
    host = repeat_while(
        lambda: _bump(_GRID, _BLOCK, (y_host, np.int32(_N))),
        SkipGuard.nonzero(guard_word),
        n,
    )
    host()  # eager host arm: while ran < cap and guard.evaluate()
    cp.cuda.runtime.deviceSynchronize()

    assert loop.iterations() == n == host.iterations()
    assert np.array_equal(cp.asnumpy(y_loop), cp.asnumpy(y_skip))
    assert np.array_equal(cp.asnumpy(y_loop), cp.asnumpy(y_host))


# --------------------------------------------------------------------------- #
# (e) empty body refused at build, in a CHILD PROCESS under a timeout: the RED
#     (refusal deleted) is an infinite loop on launch, caught by the timeout,
#     never waited on by the test process. The child builds AND launches.
# --------------------------------------------------------------------------- #
_EMPTY_BODY_CHILD = textwrap.dedent(
    """
    import cupy as cp
    from eagle import SkipGuard, repeat_while
    from eagle.pipeline import GraphPipeline
    guard = cp.ones(1, dtype=cp.uint32)
    pipe = GraphPipeline().add(repeat_while(lambda: None, SkipGuard.nonzero(guard), 64))
    try:
        pipe.build()
    except ValueError as e:
        assert "empty body" in str(e) or "launched no work" in str(e), e
        print("REFUSED")
    else:
        pipe.launch()  # RED path: an empty WHILE body loops forever here
        print("LAUNCHED")
    """
)


def test_repeat_while_empty_body_is_refused_at_build_under_process_timeout():
    proc = subprocess.run(
        [sys.executable, "-c", _EMPTY_BODY_CHILD],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert "REFUSED" in proc.stdout, proc.stdout

    # In-process: the refusal leaves no capture open -- the next build works.
    guard = cp.ones(1, dtype=cp.uint32)
    bad = GraphPipeline().add(repeat_while(lambda: None, SkipGuard.nonzero(guard), 4))
    with pytest.raises(ValueError):
        bad.build()
    y = cp.zeros(_N, dtype=cp.float32)
    ok_loop = repeat_while(
        lambda: _bump(_GRID, _BLOCK, (y, np.int32(_N))), SkipGuard.nonzero(guard), 4
    )
    ok = GraphPipeline().add(ok_loop)
    ok.build()
    ok.launch()
    assert ok_loop.iterations() == 4


# --------------------------------------------------------------------------- #
# (f) nesting. RED 1: drop the inner weave (run the Skippable's step bare) ->
#     flag 0 still fires the tracer. RED 2: drop the while-in-while refusal.
# --------------------------------------------------------------------------- #
def test_repeat_while_with_skippable_inside_body_follows_its_flag():
    y = cp.zeros(_N, dtype=cp.float32)
    guard = cp.ones(1, dtype=cp.uint32)
    flag = cp.zeros(1, dtype=cp.uint32)
    tracer = cp.zeros(1, dtype=cp.int32)
    inner = skippable(lambda: _tracer((1,), (1,), (tracer,)), SkipGuard.intent(flag, 0))
    # body: one plain kernel, then the guarded region -- the Skippable IS the
    # loop's step here (structural nesting), so wrap the plain kernel in it too
    # or register both as one Skippable body; v1 supports step=Skippable.
    loop = repeat_while(inner, SkipGuard.nonzero(guard), 8)
    pipe = GraphPipeline().add(loop)
    pipe.build()
    dot = pipe._dot()
    if conditional_count(dot, "WHILE") is None:  # this driver labels no node types
        assert len(clusters(dot)) == 3, "outer graph + the WHILE body + the IF body"
    else:
        assert conditional_count(dot, "WHILE") == 1 and conditional_count(dot, "IF") == 1

    pipe.launch()  # flag 0: 8 iterations, tracer silent
    assert loop.iterations() == 8
    assert _tracer_value(tracer) == 0
    flag[:] = 1
    pipe.launch()  # flag 1: 8 iterations, tracer fires every iteration
    assert loop.iterations() == 8
    assert _tracer_value(tracer) == 8
    del y


def test_repeat_while_inside_repeat_while_is_refused_at_build():
    guard = cp.ones(1, dtype=cp.uint32)
    inner = repeat_while(lambda: None, SkipGuard.nonzero(guard), 2)
    pipe = GraphPipeline().add(repeat_while(inner, SkipGuard.nonzero(guard), 2))
    with pytest.raises(ValueError, match="repeat_while inside repeat_while"):
        pipe.build()
    pipe2 = GraphPipeline().add(skippable(inner, SkipGuard.nonzero(guard)))
    with pytest.raises(ValueError, match="repeat_while inside skippable"):
        pipe2.build()


# --------------------------------------------------------------------------- #
# (g) enabled-mode refusal. RED: drop the RepeatWhile isinstance in
#     _prebuild_guarded (treat it as a plain step) -> the member is recorded
#     toggleable (or is_node_toggleable is called on the WHILE node).
# --------------------------------------------------------------------------- #
def test_repeat_while_member_is_not_toggleable_and_enabled_mode_refuses_it():
    y = cp.zeros(_N, dtype=cp.float32)
    guard = cp.ones(1, dtype=cp.uint32)
    loop = repeat_while(
        lambda: _bump(_GRID, _BLOCK, (y, np.int32(_N))), SkipGuard.nonzero(guard), 2
    )
    pipe = GraphPipeline().add(loop, name="loop")
    pipe.build()
    assert not pipe.is_member_toggleable("loop")
    with pytest.raises(NonToggleableMemberError):
        pipe.set_member_enabled("loop", False)

    composer = GraphComposer(None, mode="enabled")
    composer.register(loop)
    composer.register(lambda: _bump(_GRID, _BLOCK, (y, np.int32(_N))))
    with pytest.raises(NonToggleableMemberError):
        composer.build()


# --------------------------------------------------------------------------- #
# (h) zero-iteration entry. RED: make the head ignore the guard -> 64.
# --------------------------------------------------------------------------- #
def test_repeat_while_zero_iterations_when_guard_is_false_at_entry():
    y = cp.zeros(_N, dtype=cp.float32)
    guard = cp.zeros(1, dtype=cp.uint32)
    tracer = cp.zeros(1, dtype=cp.int32)

    def body():
        _bump(_GRID, _BLOCK, (y, np.int32(_N)))
        _tracer((1,), (1,), (tracer,))

    loop = repeat_while(body, SkipGuard.nonzero(guard), 64)
    pipe = GraphPipeline().add(loop)
    pipe.build()
    pipe.launch()
    assert loop.iterations() == 0
    assert _tracer_value(tracer) == 0
    assert not cp.asnumpy(y).any(), "state unchanged"


# --------------------------------------------------------------------------- #
# Beyond (a)-(h): the ran cell as the body's iteration index, the concurrent
# member form, the structural dot, host-arm parity, validation, retention.
# --------------------------------------------------------------------------- #
def test_repeat_while_iteration_index_is_visible_to_the_body():
    """y += ran each iteration, 8 iterations -> sum(0..7) == 28: the tail
    increments AFTER the body (RED: tail before the body -> 36)."""
    y = cp.zeros(_N, dtype=cp.float32)
    guard = cp.ones(1, dtype=cp.uint32)
    loop = repeat_while(
        lambda: _add_index(_GRID, _BLOCK, (y, np.int32(_N), loop.iteration_index)),
        SkipGuard.nonzero(guard), 8,
    )
    pipe = GraphPipeline().add(loop)
    pipe.build()
    pipe.launch()
    assert loop.iterations() == 8
    assert np.array_equal(cp.asnumpy(y), np.full(_N, 28.0, dtype=np.float32))


def test_repeat_while_as_add_concurrent_member():
    y = cp.zeros(_N, dtype=cp.float32)
    y2 = cp.zeros(_N, dtype=cp.float32)
    guard = cp.ones(1, dtype=cp.uint32)
    loop = repeat_while(
        lambda: _bump(_GRID, _BLOCK, (y, np.int32(_N))), SkipGuard.nonzero(guard), 8
    )
    pipe = GraphPipeline().add_concurrent(
        [loop, lambda: _bump(_GRID, _BLOCK, (y2, np.int32(_N)))]
    )
    pipe.build()
    pipe.launch()
    assert loop.iterations() == 8
    assert np.array_equal(cp.asnumpy(y), _eager_steps(8))
    assert np.array_equal(cp.asnumpy(y2), _eager_steps(1))


def test_repeat_while_graph_struct_one_while_node_head_and_tail_setters():
    y = cp.zeros(_N, dtype=cp.float32)
    guard = cp.ones(1, dtype=cp.uint32)
    loop = repeat_while(
        lambda: _bump(_GRID, _BLOCK, (y, np.int32(_N))), SkipGuard.nonzero(guard), 2
    )
    pipe = GraphPipeline().add(loop)
    pipe.build()
    dot = pipe._dot()
    if conditional_count(dot, "WHILE") is None:  # this driver labels no node types
        assert len(clusters(dot)) == 2, "outer graph + the WHILE body"
    else:
        assert conditional_count(dot, "WHILE") == 1
    assert "MEMSET" not in dot.upper() or dot.upper().count("MEMSET") == 0
    labels = pipe.node_labels()
    assert any("setLoopHeadKernel" in lab for lab in labels)
    assert any("setLoopTailKernel" in lab for lab in labels)
    assert any("rw_bump_kernel" in lab for lab in labels)


def test_repeat_while_host_arm_numpy_parity():
    live = np.array([5], dtype=np.uint32)
    calls = []

    def dec():
        calls.append(1)
        live[0] -= 1

    loop = repeat_while(dec, SkipGuard.nonzero(live), 64)
    assert loop.iterations() == 0
    loop()
    assert loop.iterations() == 5 and len(calls) == 5
    live[0] = 1000
    calls.clear()
    loop()
    assert loop.iterations() == 64 and len(calls) == 64
    live[0] = 0
    calls.clear()
    loop()
    assert loop.iterations() == 0 and calls == []
    assert not isinstance(loop, Skippable)


@pytest.mark.parametrize(
    "bad, exc",
    [(0, ValueError), (-3, ValueError), (True, TypeError), (2.0, TypeError)],
)
def test_repeat_while_max_iters_validation(bad, exc):
    guard = np.ones(1, dtype=np.uint32)
    with pytest.raises(exc):
        RepeatWhile(lambda: None, SkipGuard.nonzero(guard), bad)


def test_repeat_while_retains_guard_and_cell_references():
    import gc

    y = cp.zeros(_N, dtype=cp.float32)

    def make():
        guard = cp.ones(1, dtype=cp.uint32)  # no other reference kept
        return repeat_while(
            lambda: _bump(_GRID, _BLOCK, (y, np.int32(_N))), SkipGuard.nonzero(guard), 4
        )

    loop = make()
    gc.collect()
    pipe = GraphPipeline().add(loop)
    pipe.build()
    gc.collect()
    pipe.launch()
    assert loop.iterations() == 4
