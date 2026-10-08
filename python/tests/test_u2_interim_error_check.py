# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""The interim ``cudaGetLastError`` bracket around
``Graph.launcher()`` and ``Launcher.launch`` in ``eagle_core.cu``.

Release-build ``Launcher::instantiate_``/``Launcher::launch`` use the
debug-only ``EAGLE_CHECK`` (not ``EAGLE_CHECK_ALWAYS``), so a rejected CUDA
call is swallowed and the caller gets a silently INERT ``Launcher`` instead of
an error (the swallowed-failure class this names, and the exact mechanism a prior
probe's ``ConditionalGroupTest.MultiLauncherFanOut``
(``eagle/tests/test_ConditionalGroup.cu``)
demonstrated in C++: a second ``launcher()`` on a conditional-bearing graph
does not throw, but ``cudaGetLastError()`` reads back 801
(``cudaErrorNotSupported``) and the resulting exec is dead).

This bracket covers exactly two Python bindings -- ``graph.launcher()`` and
``Launcher.launch`` -- with a clear-before/check-after ``cudaGetLastError()``,
raising ``RuntimeError`` on nonzero. This is Python-binding-only: no change to
``Launcher.h``/``Graph.h`` or any shared C++ header, so downstream consumers see
zero change. Its REDdenable gate is the natural one: the SAME
MultiLauncherFanOut scenario must now RAISE (801) from Python instead of
silently returning something inert.
"""

from __future__ import annotations

import pytest

from eagle import SkipGuard, skippable
from eagle.pipeline import GraphPipeline

pytestmark = pytest.mark.gpu
cp = pytest.importorskip("cupy")

_TRACER_SRC = r"""
extern "C" __global__ void u2_tracer_kernel(int* counter) {
    atomicAdd(counter, 1);
}
"""
_tracer_kernel = cp.RawKernel(_TRACER_SRC, "u2_tracer_kernel")


def _fresh_tracer():
    t = cp.zeros(1, dtype=cp.int32)
    _tracer_kernel((1,), (1,), (t,))  # NVRTC warmup, outside capture
    cp.cuda.runtime.deviceSynchronize()
    t[:] = 0
    return t


def _build_conditional_pipeline():
    """A minimal conditional-bearing GraphPipeline -- the shape
    ``MultiLauncherFanOut`` needs: ``launcher()`` must succeed once, then a
    SECOND raw instantiate of the SAME underlying graph must be rejected by
    the driver (801) precisely because the graph carries a conditional (IF)
    node."""
    count = cp.ones(1, dtype=cp.uint32)
    tracer = _fresh_tracer()
    guard = SkipGuard.nonzero(count)
    step = skippable(lambda: _tracer_kernel((1,), (1,), (tracer,)), guard)

    pipe = GraphPipeline()
    pipe.add(step)
    pipe.build()  # first launcher() -- must succeed (pipe._launcher set inside)
    return pipe, tracer


def test_first_launcher_on_conditional_graph_succeeds_and_fires():
    """Sanity/positive control: the bracket must not disturb the normal,
    single-launcher path -- build() -> launch() -> tracer fires."""
    pipe, tracer = _build_conditional_pipeline()
    pipe.launch()
    assert int(tracer.get()[0]) == 1


def test_second_launcher_on_conditional_graph_raises_801():
    """The REDdenable gate: a MultiLauncherFanOut reproducer
    (a second ``launcher()`` on a conditional-bearing graph) must now RAISE
    from Python -- a loud ``RuntimeError`` naming the driver's 801
    (``cudaErrorNotSupported``) -- instead of returning an inert Launcher
    the caller has no way to detect went bad.

    Anchored on the STABLE IDENTIFIER, not the prose.
    ``match="801"`` alone silently described a message shape nothing produced --
    CUDA puts no digits in either ``cudaGetErrorString`` or ``cudaGetErrorName``
    (measured: 801 -> "operation not supported" / "cudaErrorNotSupported"), and
    the prior formatter uses the string alone, so this assertion only ever passed
    against a binary built from an uncommitted local edit. ``EAGLE_CHECK_ALWAYS``
    now formats the name and the code itself (eagle holds the ``cudaError_t`` at
    the raise site), so both anchors are real. Assert the NAME as well as the
    number: the English sentence is CUDA-version-dependent with no stability
    guarantee, whereas ``cudaErrorNotSupported`` is the stable spelling."""
    pipe, tracer = _build_conditional_pipeline()

    with pytest.raises(RuntimeError, match=r"cudaErrorNotSupported \(801\)"):
        pipe.graph.launcher()  # the SECOND instantiate of the same graph

    # The FIRST launcher must be entirely unaffected by the failed second
    # instantiate attempt (mirrors the C++ test's final assertion).
    pipe.launch()
    assert int(tracer.get()[0]) == 1, "the first launcher must still work"


def test_launch_binding_does_not_raise_on_healthy_replay():
    """Negative control for the OTHER half of the bracket (``Launcher.launch``
    itself): an ordinary, non-conditional pipeline's replay must not raise --
    the clear-before/check-after bracket must be silent on the happy path."""
    a = cp.zeros(4, dtype=cp.float64)
    kernel = cp.ElementwiseKernel("float64 x", "float64 y", "y = x + 1.0", "u2_add1")
    kernel(a, a)  # NVRTC warmup, outside capture
    cp.cuda.runtime.deviceSynchronize()
    a.fill(0.0)

    pipe = GraphPipeline()
    pipe.add(lambda: kernel(a, a))
    pipe.build()
    pipe.launch(3)  # must not raise
    assert (cp.asnumpy(a) == 3.0).all()
