# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""eagle._counters + the three hooked call sites' certification,
reviewed and locked here.

RED-first: every row below was run against the UNHOOKED tree (``git stash``
over ``eagle/pipeline.py`` and ``eagle/plan.py``'s hook lines only —
``eagle/_counters.py`` and this file already existed) before the hooks
landed; each row's own docstring records what that run showed. The counters
are process-global and monotonic (``eagle._counters`` — no reset), so every
row reads a snapshot immediately before the event under test and asserts the
DELTA against a snapshot immediately after, never an absolute value — that is
what makes the rows order-independent within one process.

The captures/instantiations rows mirror ``test_engine_hardening_pipeline.py``'s smallest
fixture (a ``cp.ElementwiseKernel`` on a bare ``GraphPipeline``).
``Plan.bind``/``BoundPlan.rebind`` refuse anything but a real ``aether-abi/2``
artifact, so the binds/rebinds rows mirror ``test_plan_bind_launch.py``'s
bundle fixture, trimmed to the smallest bindable shape in that file: one
``Scalar`` in, one by-value uniform, one ``Scalar`` out.

THE DISCRIMINATING ROW ("recapture-without-reinstantiate: captures +1,
instantiations +0") HAS NO PYTHON-VISIBLE SEAM IN THIS TREE —
see ``test_the_recapture_without_reinstantiate_row_has_no_seam``'s own
docstring for the full grep trail. Reported as absent rather than invented
by widening the hook's scope.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from eagle import _counters
from eagle.pipeline import GraphPipeline

hawk_trace = pytest.importorskip("hawk.trace")
hawk_artifact = pytest.importorskip("hawk.artifact")

#: The GPU this process runs on, the same one test_plan_bind_launch.py
#: builds against.
DEVICE_ARCH = hawk_artifact.arch()

_ALL_ZERO = {"captures": 0, "instantiations": 0, "binds": 0, "rebinds": 0}


@hawk_trace.kernel
def _n20i_scale(x: hawk_trace.Scalar, a: hawk_trace.Param,
                y: hawk_trace.Mutable[hawk_trace.Scalar]):
    """The smallest v2 body this file needs: one ``Scalar`` in, one by-value
    uniform, one ``Scalar`` out — smaller than test_plan_bind_launch.py's own
    smallest subject (``e4_vec3_scale``, a ``Vector[3]``), because the
    binds/rebinds rows only need SOME real v2 artifact to bind, not a
    particular role shape."""
    y = a * x


@pytest.fixture(scope="module")
def _n20i_unit(tmp_path_factory):
    return hawk_artifact.build_bundle(
        [_n20i_scale], tmp_path_factory.mktemp("n20i_counters"),
        targets=("cuda",), device_arch=DEVICE_ARCH,
    )


@pytest.fixture(scope="module")
def _n20i_plugin(_n20i_unit):
    """The plan-able view of the unit's DEVICE entry — trimmed from
    test_plan_bind_launch.py's ``_device_subject``: this file needs the plan
    view only, not the layout self-check that file's own rows exist
    to exercise."""
    from eagle.registry import load_manifest

    reg = load_manifest(_n20i_unit.manifest_path)
    loaded = reg["_n20i_scale"]
    sidecar = json.loads(
        (_n20i_unit.directory / "_n20i_scale.json").read_text()
    )
    view = hawk_artifact.plan_view(sidecar)
    view.pop("host_entry", None)
    return SimpleNamespace(device_function=loaded.fn.kernel.ptr,
                           _keepalive=(reg, loaded), **view)


def _delta(before: dict, after: dict) -> dict:
    return {k: after[k] - before[k] for k in before}


# --------------------------------------------------------------------------- #
# binds / rebinds — need a real v2 HAWK artifact.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_bind_moves_binds_by_exactly_one_and_nothing_else(_n20i_plugin):
    """RED (measured against the unhooked tree, ``git stash`` over the two
    hook lines in ``eagle/plan.py``): ``AssertionError`` — the observed delta
    was ``{'captures': 0, 'instantiations': 0, 'binds': 0, 'rebinds': 0}``
    (``Plan.bind`` ran, correctly, but nothing counted it). GREEN after the
    hook: exactly ``binds`` moves, by 1."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    x = cp.asarray(np.array([2.0], dtype=np.float64))
    y = cp.zeros(1, dtype=np.float64)
    p = eplan.plan(_n20i_plugin, structure=eexec.DeviceKernel)

    before = _counters.snapshot()
    bound = p.bind(x=x, a=3.0, y=y)
    after = _counters.snapshot()

    assert _delta(before, after) == {**_ALL_ZERO, "binds": 1}
    bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    assert float(cp.asnumpy(y)[0]) == 6.0


@pytest.mark.gpu
def test_rebind_moves_rebinds_by_exactly_one_and_nothing_else(_n20i_plugin):
    """RED (measured against the unhooked tree): the delta was all-zero —
    ``BoundPlan.rebind`` ran and repacked the slot, but nothing counted it.
    GREEN after the hook: exactly ``rebinds`` moves, by 1 (``binds`` does
    NOT move again — the plan was bound once, above the measured window)."""
    import cupy as cp

    import eagle.exec as eexec
    from eagle import plan as eplan

    x = cp.asarray(np.array([2.0], dtype=np.float64))
    y = cp.zeros(1, dtype=np.float64)
    p = eplan.plan(_n20i_plugin, structure=eexec.DeviceKernel)
    bound = p.bind(x=x, a=3.0, y=y)

    x2 = cp.asarray(np.array([5.0], dtype=np.float64))
    before = _counters.snapshot()
    bound.rebind(x=x2)
    after = _counters.snapshot()

    assert _delta(before, after) == {**_ALL_ZERO, "rebinds": 1}
    bound.launch()
    cp.cuda.runtime.deviceSynchronize()
    assert float(cp.asnumpy(y)[0]) == 15.0


# --------------------------------------------------------------------------- #
# captures / instantiations — the plain cupy fixture, mirroring
# test_engine_hardening_pipeline.py.
# --------------------------------------------------------------------------- #
@pytest.mark.gpu
def test_build_moves_captures_and_instantiations_by_exactly_one():
    """RED (measured against the unhooked tree, ``git stash`` over the two
    hook lines in ``eagle/pipeline.py``'s ``build()``): the delta was
    all-zero — ``build()`` captured and instantiated a real graph (the
    replay below proves it ran), but nothing counted either half. GREEN
    after the hook: ``captures`` and ``instantiations`` each move by exactly
    1, ``binds``/``rebinds`` stay at 0 (this fixture never touches
    ``eagle.plan``)."""
    import cupy as cp

    n = 64
    a = cp.arange(n, dtype=cp.float64)
    k = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 2.0 + 1.0", "n20i_build"
    )
    out = cp.zeros(n, dtype=cp.float64)
    k(a, cp.empty_like(out))  # warm (compile) OUTSIDE capture
    cp.cuda.runtime.deviceSynchronize()

    pipe = GraphPipeline()
    pipe.add(lambda: k(a, out))

    before = _counters.snapshot()
    pipe.build()
    after = _counters.snapshot()

    assert _delta(before, after) == {**_ALL_ZERO, "captures": 1, "instantiations": 1}
    pipe.launch()
    cp.cuda.runtime.deviceSynchronize()
    np.testing.assert_array_equal(cp.asnumpy(out), cp.asnumpy(a) * 2.0 + 1.0)


@pytest.mark.gpu
def test_set_member_enabled_moves_nothing():
    """RED (measured against the unhooked tree): trivially all-zero already
    (no counter existed to move), so this row's own value is confirming the
    hooks do NOT touch ``set_member_enabled`` at all — the discriminating
    half of the "no recapture, no re-instantiate" claim. GREEN after the
    hook: still all-zero, across TWO toggles."""
    import cupy as cp

    n = 64
    a = cp.arange(n, dtype=cp.float64)
    k = cp.ElementwiseKernel(
        "float64 x", "float64 y", "y = x * 2.0 + 1.0", "n20i_toggle"
    )
    out = cp.zeros(n, dtype=cp.float64)
    k(a, cp.empty_like(out))
    cp.cuda.runtime.deviceSynchronize()

    pipe = GraphPipeline()
    pipe.add(lambda: k(a, out), name="only")
    pipe.build()

    before = _counters.snapshot()
    pipe.set_member_enabled("only", False)
    pipe.set_member_enabled("only", True)
    after = _counters.snapshot()

    assert _delta(before, after) == _ALL_ZERO


@pytest.mark.gpu
def test_a_second_build_on_a_new_pipe_is_plus_one_plus_one_again():
    """RED (measured against the unhooked tree): all-zero, same as every
    other row before the hook. GREEN after the hook: a SECOND, unrelated
    pipeline's ``build()`` moves ``captures``/``instantiations`` by another
    +1/+1 — the counters are monotonic and NEVER reset, so the absolute
    values after two builds are each >= 2."""
    import cupy as cp

    n = 32
    a = cp.arange(n, dtype=cp.float64)
    k = cp.ElementwiseKernel("float64 x", "float64 y", "y = x + 1.0", "n20i_mono")
    out1 = cp.zeros(n, dtype=cp.float64)
    out2 = cp.zeros(n, dtype=cp.float64)
    k(a, cp.empty_like(out1))
    cp.cuda.runtime.deviceSynchronize()

    pipe1 = GraphPipeline().add(lambda: k(a, out1))
    pipe1.build()  # unmeasured warm-up build

    before = _counters.snapshot()
    pipe2 = GraphPipeline().add(lambda: k(a, out2))
    pipe2.build()
    after = _counters.snapshot()

    assert _delta(before, after) == {**_ALL_ZERO, "captures": 1, "instantiations": 1}
    assert after["captures"] >= 2 and after["instantiations"] >= 2, (
        "monotonic, never reset: two builds this process must leave both "
        "counters at 2 or more"
    )


# --------------------------------------------------------------------------- #
# The discriminating row, and why it is not here.
# --------------------------------------------------------------------------- #
@pytest.mark.skip(
    reason=(
        "the recapture-without-reinstantiate path ('captures +1, "
        "instantiations +0, this is the DISCRIMINATING row') "
        "names no code path that exists in this tree. GraphPipeline.build() "
        "(eagle/pipeline.py:236-374) is the ONLY capture/instantiate call "
        "site and it always pairs capturer.begin()/capturer.end() (capture) "
        "with Graph.from_captured(...)/g.launcher() (instantiate) inside ONE "
        "method call -- there is no code path that captures without also "
        "instantiating, so this delta cannot occur against real behaviour. "
        "set_member_enabled's own docstring (pipeline.py ~438-440) reads "
        "'Toggling never recaptures, never re-instantiates' -- the OPPOSITE "
        "of a capture-only path; that is the 0/0 case covered by "
        "test_set_member_enabled_moves_nothing above, not this one. The only "
        "other 'recapture' language in eagle is "
        "eagle.compose.GraphComposer._rebuild_pipe (mode='rebuild'), which "
        "also pairs capture+instantiate in one _build_flat_pipe(...).build() "
        "call (so it is +1/+1, not the claimed +1/0), and compose.py's "
        "call site is not one of the three hooked here regardless. The C++ "
        "core (eagle/python/src/eagle_core.cu) has no cudaGraphExecUpdate or "
        "similar partial-recapture primitive either (grepped for "
        "ExecUpdate/exec_update/recapture: none). Reported as absent "
        "rather than inventing a new "
        "code path under a hook that is supposed to make NO behaviour "
        "change."
    )
)
def test_the_recapture_without_reinstantiate_row_has_no_seam():
    pass
