# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Parity certification: the bound C++-native
``eagle._core.GraphComposer`` (``eagle/eagle/compose/GraphComposer.h``) vs
the pure-Python ``eagle.compose.GraphComposer`` (``eagle/python/eagle/
compose.py``, BYTE-FROZEN this package). The SAME scenario is built from
Python TWICE — once driving each engine — and outputs are asserted cupy
bitwise-identical.

Bridging note (the two engines are deliberately NOT the same surface —
see ``eagle_core.cu``'s ``GraphComposer`` binding doc and ``GraphComposer.h``'s
class doc): the C++ engine's raw member callable takes the target stream
EXPLICITLY (``step(stream: int)``, wrap with ``cp.cuda.ExternalStream``)
where the Python engine's callable is a plain zero-arg step; a built
``GraphPipeline`` is registered on the Python side via ``.register(pipe)``
but on the C++ side via ``.register_launcher(pipe._launch_handles().launcher)``
— which MOVES the underlying ``Launcher`` out from under the Python
``GraphPipeline`` object (mirrors ``Graph.from_captured``'s own move-only
handling elsewhere in this binding), so a ``GraphPipeline`` fed to the C++
engine must never be reused (built once, handed over, done). Each scenario
below therefore builds two INDEPENDENT sets of buffers/pipelines, one per
engine, and compares their final device contents.

Coverage:

1. ``mode="sequenced"`` parity (two stream-owning members).
2. The two-level nesting scenario (MANDATORY) — composer-of-composers,
   bitwise, across three routing combinations.
3. ``mode="enabled"`` parity (capture-once, toggle-without-recapture).
4. Rejection parity: the SAME error class (``ValueError``) for a
   non-sequenced-outer nesting attempt on both engines — nanobind's
   standard exception table maps ``std::invalid_argument`` to
   ``ValueError``, exactly the type ``compose.py`` itself raises for the
   same rejection.
"""

from __future__ import annotations

import numpy as np
import pytest

from eagle import _core
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


# ============================================================================
# 1. mode="sequenced" parity: two independent, stream-owning members.
# ============================================================================
def test_sequenced_parity_bitwise():
    n = 64
    reps = 5
    a = cp.arange(n, dtype=cp.float64)
    b = cp.arange(n, dtype=cp.float64) * 2.0 - 1.0
    kA = cp.ElementwiseKernel("float64 x", "float64 y", "y += x * 0.0 + 1.0",
        "compose_parity_seq_A")
    kB = cp.ElementwiseKernel("float64 x", "float64 y", "y += x * 0.0 + 1.0",
        "compose_parity_seq_B")
    _warm(kA, a, cp.empty_like(a))
    _warm(kB, b, cp.empty_like(b))

    # -- Python engine --------------------------------------------------
    py_outA = cp.zeros(n, dtype=cp.float64)
    py_outB = cp.zeros(n, dtype=cp.float64)
    py_pipeA = GraphPipeline()
    py_pipeA.add(lambda: kA(a, py_outA))
    py_pipeA.build()
    py_pipeB = GraphPipeline()
    py_pipeB.add(lambda: kB(b, py_outB))
    py_pipeB.build()

    py_composer = GraphComposer(None, mode="sequenced")
    py_composer.register(py_pipeA)
    py_composer.register(py_pipeB)
    py_composer.build()
    py_composer.launch(reps)
    cp.cuda.runtime.deviceSynchronize()

    # -- C++ engine -------------------------------------------------------
    cpp_outA = cp.zeros(n, dtype=cp.float64)
    cpp_outB = cp.zeros(n, dtype=cp.float64)
    cpp_pipeA = GraphPipeline()
    cpp_pipeA.add(lambda: kA(a, cpp_outA))
    cpp_pipeA.build()
    cpp_pipeB = GraphPipeline()
    cpp_pipeB.add(lambda: kB(b, cpp_outB))
    cpp_pipeB.build()

    cpp_composer = _core.GraphComposer(mode="sequenced")
    cpp_composer.register_launcher(cpp_pipeA._launch_handles().launcher)
    cpp_composer.register_launcher(cpp_pipeB._launch_handles().launcher)
    cpp_composer.build()
    cpp_composer.launch(reps)
    cp.cuda.runtime.deviceSynchronize()

    assert np.array_equal(cp.asnumpy(py_outA), cp.asnumpy(cpp_outA))
    assert np.array_equal(cp.asnumpy(py_outB), cp.asnumpy(cpp_outB))
    assert float(cp.asnumpy(cpp_outA)[0]) == float(reps)


# ============================================================================
# 2. MANDATORY: the two-level nesting scenario, bitwise, Python engine vs
#    C++ engine, across the same three routing combinations as
#    test_netsparse_compose.py::test_nested_sequenced_composer_two_level_bitwise.
# ============================================================================
def test_two_level_nesting_parity_bitwise():
    n = 64
    a = cp.arange(n, dtype=cp.float64)
    b = cp.arange(n, dtype=cp.float64) * 2.0 - 1.0
    kA = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 3.0 + 1.0", "compose_parity_nest_A"
    )
    kB = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x - 5.0", "compose_parity_nest_B"
    )
    _warm(kA, a, cp.empty_like(a))
    _warm(kB, b, cp.empty_like(b))

    def _build_python():
        outA = cp.full(n, _SENTINEL, dtype=cp.float64)
        outB = cp.full(n, _SENTINEL, dtype=cp.float64)
        pipeA = GraphPipeline()
        pipeA.add(lambda: kA(a, outA))
        pipeA.build()
        pipeB = GraphPipeline()
        pipeB.add(lambda: kB(b, outB))
        pipeB.build()

        inner = GraphComposer([1, 1], mode="sequenced")
        inner.register(pipeA)
        inner.register(pipeB)
        inner.build()

        outer = GraphComposer([1], mode="sequenced")
        outer.register(inner)
        outer.build()
        return outer, inner, outA, outB

    def _build_cpp():
        outA = cp.full(n, _SENTINEL, dtype=cp.float64)
        outB = cp.full(n, _SENTINEL, dtype=cp.float64)
        pipeA = GraphPipeline()
        pipeA.add(lambda: kA(a, outA))
        pipeA.build()
        pipeB = GraphPipeline()
        pipeB.add(lambda: kB(b, outB))
        pipeB.build()

        inner = _core.GraphComposer(mode="sequenced")
        inner.register_launcher(pipeA._launch_handles().launcher)
        inner.register_launcher(pipeB._launch_handles().launcher)
        inner.build()

        outer = _core.GraphComposer(mode="sequenced")
        outer.register_nested(inner)
        outer.build()
        # LIFETIME CONTRACT (register_launcher docstring): the raw binding
        # does NOT keep the capture source alive -- the pipelines must
        # outlive the composer, so they ride along in the return value.
        return outer, inner, outA, outB, pipeA, pipeB

    py_outer, py_inner, py_outA, py_outB = _build_python()
    cpp_outer, cpp_inner, cpp_outA, cpp_outB, _cpp_pipeA, _cpp_pipeB = _build_cpp()

    def _reset():
        py_outA.fill(_SENTINEL)
        py_outB.fill(_SENTINEL)
        cpp_outA.fill(_SENTINEL)
        cpp_outB.fill(_SENTINEL)

    # Combo 1: outer=ON, inner=(1,1) -- both members fire on both engines.
    _reset()
    py_outer.launch(1)
    cpp_outer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(py_outA), cp.asnumpy(cpp_outA)), "combo1: A"
    assert np.array_equal(cp.asnumpy(py_outB), cp.asnumpy(cpp_outB)), "combo1: B"
    assert not np.all(cp.asnumpy(py_outA) == _SENTINEL)

    # Combo 2: outer=ON, inner=(1,0) -- only member 0 fires on both engines.
    _reset()
    py_inner.set_routing([1, 0])
    cpp_inner.set_routing([True, False])
    py_outer.launch(1)
    cpp_outer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(py_outA), cp.asnumpy(cpp_outA)), "combo2: A"
    assert np.array_equal(cp.asnumpy(py_outB), cp.asnumpy(cpp_outB)), "combo2: B"
    assert np.all(cp.asnumpy(py_outB) == _SENTINEL), "combo2: B untouched (py)"
    assert np.all(cp.asnumpy(cpp_outB) == _SENTINEL), "combo2: B untouched (cpp)"

    # Combo 3: outer=OFF, inner=(1,1) -- the outer gate wins on both engines,
    # despite inner being all-on; neither inner member ever fires.
    _reset()
    py_inner.set_routing([1, 1])
    cpp_inner.set_routing([True, True])
    py_outer.set_routing([0])
    cpp_outer.set_routing([False])
    py_outer.launch(1)
    cpp_outer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.all(cp.asnumpy(py_outA) == _SENTINEL), "combo3: py outer gate wins (A)"
    assert np.all(cp.asnumpy(cpp_outA) == _SENTINEL), "combo3: cpp outer gate wins (A)"
    assert np.all(cp.asnumpy(py_outB) == _SENTINEL), "combo3: py outer gate wins (B)"
    assert np.all(cp.asnumpy(cpp_outB) == _SENTINEL), "combo3: cpp outer gate wins (B)"

    # Fired-history parity: same fired SETS across combos, modulo the
    # documented list-vs-frozenset representation difference (the binding's
    # own doc: eagle._core.GraphComposer.fired_history() returns
    # list[list[int]], never frozenset).
    py_fired = py_outer.fired_history()
    cpp_fired = [set(x) for x in cpp_outer.fired_history()]
    assert [set(f) for f in py_fired] == cpp_fired

    py_inner_fired = py_inner.fired_history()
    cpp_inner_fired = [set(x) for x in cpp_inner.fired_history()]
    assert [set(f) for f in py_inner_fired] == cpp_inner_fired
    # combo 3's outer gate never reached the inner composer on EITHER engine.
    assert len(cpp_inner_fired) == 2


# ============================================================================
# 3. mode="enabled" parity: capture-once, toggle-without-recapture.
# ============================================================================
def test_enabled_parity_bitwise():
    n = 32
    buf_init = 0.0

    tracer_src = r"""
extern "C" __global__ void compose_parity_enabled_inc(double* buf, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) buf[tid] += 1.0;
}
"""
    kernel = cp.RawKernel(tracer_src, "compose_parity_enabled_inc")

    def _fresh(engine_tag):
        bufA = cp.full(n, buf_init, dtype=cp.float64)
        bufB = cp.full(n, buf_init, dtype=cp.float64)
        scratch = cp.empty(n, dtype=cp.float64)
        kernel((1,), (n,), (scratch, n))  # warm-up NVRTC compile
        cp.cuda.runtime.deviceSynchronize()
        return bufA, bufB

    py_bufA, py_bufB = _fresh("py")
    py_composer = GraphComposer(None, mode="enabled")
    py_composer.register(lambda: kernel((1,), (n,), (py_bufA, n)), name="A")
    py_composer.register(lambda: kernel((1,), (n,), (py_bufB, n)), name="B")
    py_composer.build()

    cpp_bufA, cpp_bufB = _fresh("cpp")

    def cpp_stepA(stream):
        with cp.cuda.ExternalStream(stream):
            kernel((1,), (n,), (cpp_bufA, n))

    def cpp_stepB(stream):
        with cp.cuda.ExternalStream(stream):
            kernel((1,), (n,), (cpp_bufB, n))

    cpp_composer = _core.GraphComposer(mode="enabled")
    cpp_composer.register_callable(cpp_stepA, name="A")
    cpp_composer.register_callable(cpp_stepB, name="B")
    cpp_composer.build()

    py_composer.launch(1)
    cpp_composer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(py_bufA), cp.asnumpy(cpp_bufA))
    assert np.array_equal(cp.asnumpy(py_bufB), cp.asnumpy(cpp_bufB))

    py_composer.set_routing([True, False])
    cpp_composer.set_routing([True, False])
    py_composer.launch(1)
    cpp_composer.launch(1)
    cp.cuda.runtime.deviceSynchronize()
    assert np.array_equal(cp.asnumpy(py_bufA), cp.asnumpy(cpp_bufA))
    assert np.array_equal(cp.asnumpy(py_bufB), cp.asnumpy(cpp_bufB))
    assert float(cp.asnumpy(cpp_bufA)[0]) == 2.0
    assert float(cp.asnumpy(cpp_bufB)[0]) == 1.0, "member B disabled -- untouched"


# ============================================================================
# 4. Rejection parity: the SAME error class for a non-sequenced-outer
#    nesting attempt on both engines.
# ============================================================================
def test_nesting_under_non_sequenced_outer_raises_same_error_class():
    py_flat = GraphComposer(None, mode="enabled")
    py_inner = GraphComposer(None, mode="sequenced")
    py_inner.register(lambda: None)
    py_inner.build()
    with pytest.raises(ValueError):
        py_flat.register(py_inner)

    cpp_flat = _core.GraphComposer(mode="enabled")
    cpp_inner = _core.GraphComposer(mode="sequenced")
    cpp_inner.register_callable(lambda stream: None)
    cpp_inner.build()
    with pytest.raises(ValueError):
        cpp_flat.register_nested(cpp_inner)


def test_launch_before_build_raises_same_error_class():
    py_composer = GraphComposer(None, mode="sequenced")
    py_composer.register(lambda: None)
    with pytest.raises(RuntimeError):
        py_composer.launch()

    cpp_composer = _core.GraphComposer(mode="sequenced")
    cpp_composer.register_callable(lambda stream: None)
    with pytest.raises(RuntimeError):
        cpp_composer.launch()
