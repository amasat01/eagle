# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Bound-core GraphPipeline behavior — the durable half of the parity gate.

During earlier development these specs were run through BOTH the pre-uniformization
pure-cupy pipeline and the bound-core :class:`eagle.pipeline.GraphPipeline`, and
asserted bit-identical (rtol=0, atol=0) on replay numerics, ``num_nodes`` and
``node_labels``. That gate passed, the cupy implementation was retired, and these
frozen expectations remain as behavioral tests of the single (bound) graph
implementation — multi-node capture, introspection, and replay accumulation the
smaller ``test_graph_plugin`` cases don't cover.
"""

import pathlib

import numpy as np
import pytest

from eagle import GraphPipeline, LoadedPure, LoadedVector

FIX = pathlib.Path(__file__).resolve().parent / "fixtures"
N = 4096
MU = 3.986004418e5

pytestmark = pytest.mark.gpu


def _gravity_ref(pos, mu):
    r = np.linalg.norm(pos, axis=0)
    return -mu * pos / r**3


def _make_P(seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(3, N))
    d /= np.linalg.norm(d, axis=0)
    return np.ascontiguousarray((d * rng.uniform(7e3, 4.2e4, N)).astype(np.float64))


def _max_rel(g, ref):
    return float(np.max(np.linalg.norm(g - ref, axis=0) / np.linalg.norm(ref, axis=0)))


def _run(spec, replays):
    """Build+replay a spec through GraphPipeline; return (result, introspection)."""
    import cupy as cp

    state = spec()  # fresh device arrays + steps
    pipe = GraphPipeline()
    for step in state["steps"]:
        pipe.add(step)
    pipe.build()
    pipe.launch(n=replays)
    result = cp.asnumpy(state["read"]())
    intro = None
    if pipe.introspection_available():
        intro = (pipe.num_nodes(), tuple(pipe.node_labels()))
    return result, intro


# ---- specs (zero-arg factories returning fresh arrays + capture steps) -------


def _spec_gravity_zero():
    import cupy as cp

    plugin = LoadedVector(FIX / "gravity.ptx")
    P = _make_P(seed=1)
    pos = cp.asarray(P)
    out = cp.zeros((3, N))
    term = cp.zeros(N, dtype=cp.bool_)

    def _zero():
        return cp.cuda.runtime.memsetAsync(
            out.data.ptr, 0, out.nbytes, cp.cuda.get_current_stream().ptr
        )

    return {
        "steps": [
            _zero,
            lambda: plugin.launch(out=out, position=pos, mu=MU, terminated=term),
        ],
        "read": lambda: out,
        "golden": _gravity_ref(P, MU),
    }


def _spec_two_accel():
    import cupy as cp

    g1 = LoadedVector(FIX / "gravity.ptx")
    g2 = LoadedVector(FIX / "gravity.ptx")
    P = _make_P(seed=3)
    pos = cp.asarray(P)
    out = cp.zeros((3, N))
    term = cp.zeros(N, dtype=cp.bool_)

    def _zero():
        return cp.cuda.runtime.memsetAsync(
            out.data.ptr, 0, out.nbytes, cp.cuda.get_current_stream().ptr
        )

    return {
        "steps": [
            _zero,
            lambda: g1.launch(out=out, position=pos, mu=MU, terminated=term),
            lambda: g2.launch(out=out, position=pos, mu=MU, terminated=term),
        ],
        "read": lambda: out,
        "golden": 2.0 * _gravity_ref(P, MU),
    }


def _spec_pure_bump():
    import cupy as cp

    plugin = LoadedPure(FIX / "bump.ptx")
    rate = 2.5
    n = 1024
    counter0 = np.linspace(0.0, 10.0, n)
    counter = cp.asarray(counter0)
    term = cp.zeros(n, dtype=cp.bool_)

    return {
        "steps": [
            lambda: plugin.launch(counter=counter, rate=rate, terminated=term),
        ],
        "read": lambda: counter,
        "counter0": counter0,
        "rate": rate,
    }


# ---- behavioral assertions (frozen from the parity gate) ---------------------


def _assert_behavior(spec, replays, want_nodes, want_labels):
    res, intro = _run(spec, replays)
    assert intro is not None  # bound core always has introspection
    # node_labels includes non-kernel nodes (e.g. MEMSET); num_nodes counts only
    # the kernel octagons.
    assert intro[0] == want_nodes
    assert list(intro[1]) == want_labels
    return res


def test_gravity_zero():
    res = _assert_behavior(
        _spec_gravity_zero, replays=2, want_nodes=1,
        want_labels=["MEMSET", "raptor_kernel"],
    )
    assert _max_rel(res, _spec_gravity_zero()["golden"]) < 1e-12


def test_two_accel():
    res = _assert_behavior(
        _spec_two_accel, replays=2, want_nodes=2,
        want_labels=["MEMSET", "raptor_kernel", "raptor_kernel"],
    )
    assert _max_rel(res, _spec_two_accel()["golden"]) < 1e-12


def test_pure_bump():
    res = _assert_behavior(
        _spec_pure_bump, replays=3, want_nodes=1, want_labels=["raptor_kernel"]
    )
    st = _spec_pure_bump()
    assert np.allclose(res, st["counter0"] + 3 * st["rate"])
