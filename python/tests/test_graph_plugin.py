# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""A deployed (PTX) launchable is a capturable CUDA-graph node — via eagle only.

Rebased onto the committed fixtures (no producer import): a :class:`LoadedVector`
and a :class:`LoadedPure` loaded from a fixture ``.ptx`` are issued through the
capturable ``launch`` primitive inside a :class:`~eagle.pipeline.GraphPipeline`
capture, instantiated once, and replayed — proving each becomes a graph node with no
special plugin machinery, and that replay is correct/bit-stable.

Skipped here (need producer code to build the artifact): the in-process
``Compiled*`` graph-node case, the cubin/fatbin arch-locked artifacts, and the
lookup-table force. Those stay in the producer's suite.
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


def test_loaded_acceleration_standalone():
    # a precompiled plugin runs on its own (no graph), matching the numpy golden
    import cupy as cp

    plugin = LoadedVector(FIX / "gravity.ptx")
    P = _make_P(seed=2)
    out = cp.asnumpy(plugin(position=cp.asarray(P), mu=MU))
    assert _max_rel(out, _gravity_ref(P, MU)) < 1e-12


def test_loaded_acceleration_as_cuda_graph_node():
    import cupy as cp

    plugin = LoadedVector(FIX / "gravity.ptx")
    P = _make_P(seed=1)
    pos = cp.asarray(P)
    out = cp.zeros((3, N))
    term = cp.zeros(N, dtype=cp.bool_)

    def _zero():
        # reset the accumulator on the capture stream (a memset node, not a kernel
        # node) — gravity accumulates (+=), so it must start from zero each replay.
        return cp.cuda.runtime.memsetAsync(
            out.data.ptr, 0, out.nbytes, cp.cuda.get_current_stream().ptr
        )

    pipe = GraphPipeline()
    pipe.add(_zero)
    pipe.add(lambda: plugin.launch(out=out, position=pos, mu=MU, terminated=term))
    pipe.build()

    pipe.launch(n=2)  # replay
    assert _max_rel(cp.asnumpy(out), _gravity_ref(P, MU)) < 1e-12  # correctness first

    if pipe.introspection_available():  # bonus, cupy >= 14
        assert pipe.num_nodes() == 1  # one KERNEL node (the memset is excluded)
        assert "raptor_kernel" in pipe.node_labels()


def test_loaded_pure_as_cuda_graph_node():
    # the pure RMW launchable captured + replayed: counter += rate on each replay.
    import cupy as cp

    plugin = LoadedPure(FIX / "bump.ptx")
    rate = 2.5
    n = 1024
    counter0 = np.linspace(0.0, 10.0, n)
    counter = cp.asarray(counter0)
    term = cp.zeros(n, dtype=cp.bool_)

    pipe = GraphPipeline()
    pipe.add(lambda: plugin.launch(counter=counter, rate=rate, terminated=term))
    pipe.build()

    replays = 3
    pipe.launch(n=replays)
    assert np.allclose(cp.asnumpy(counter), counter0 + replays * rate)

    if pipe.introspection_available():
        assert pipe.num_nodes() == 1
        assert "raptor_kernel" in pipe.node_labels()
