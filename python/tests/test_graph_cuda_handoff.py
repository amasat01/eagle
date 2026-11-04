# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""``Graph.cuda_graph()``: a built GPU graph handed to a foreign CUDA graph as a child node. The foreign side is raw
driver calls (ctypes on libcuda), so nothing of eagle runs the child: bitwise equal to ``Graph.launch``, a bind is
carried by the next clone, and every refusal names its node and quotes the CUDA error and the driver."""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import eagle
from eagle.graph import CudaGraphHandle
from _reverse_fixtures import chain, plans

N = 64


def _noop(*_):
    pass


_NOOP = _noop


def _driver():
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        pytest.skip("no libcuda")
    return lib


def _ck(code, what):
    assert code == 0, f"{what}: CUresult {code}"


def run_as_child(handle: int):
    """cuGraphAddChildGraphNode into a fresh parent, instantiate, launch, wait: the consumer's side of the contract."""
    cu = _driver()
    parent, node, exec_, stream = (ctypes.c_void_p() for _ in range(4))
    _ck(cu.cuGraphCreate(ctypes.byref(parent), 0), "cuGraphCreate")
    _ck(cu.cuGraphAddChildGraphNode(ctypes.byref(node), parent, None, ctypes.c_size_t(0), ctypes.c_void_p(handle)),
        "cuGraphAddChildGraphNode")
    _ck(cu.cuGraphInstantiateWithFlags(ctypes.byref(exec_), parent, ctypes.c_ulonglong(0)), "instantiate")
    _ck(cu.cuStreamCreate(ctypes.byref(stream), 0), "cuStreamCreate")
    _ck(cu.cuGraphLaunch(exec_, stream), "cuGraphLaunch")
    _ck(cu.cuStreamSynchronize(stream), "cuStreamSynchronize")
    cu.cuGraphExecDestroy(exec_)
    cu.cuGraphDestroy(parent)
    cu.cuStreamDestroy(stream)


@pytest.mark.gpu
class TestOnTheGpu:
    @pytest.fixture(autouse=True)
    def _need(self):
        pytest.importorskip("cupy")
        pytest.importorskip("hawk")

    def _chain(self, tmp_path, xv):
        import cupy as cp

        ps = plans(tmp_path, "gpu")
        x = cp.asarray(xv)
        g, z = chain(ps, x, 0.75)
        return cp, g, z, x, ps

    def test_the_child_writes_what_launch_writes(self, tmp_path):
        cp, g, z, _x, _ps = self._chain(tmp_path, np.linspace(-1.5, 1.5, N))
        g.launch()
        want = cp.asnumpy(z).copy()
        z[...] = 0
        with g.cuda_graph() as h:
            assert isinstance(h, CudaGraphHandle) and h.handle != 0
            run_as_child(h.handle)
        np.testing.assert_array_equal(cp.asnumpy(z), want)
        assert np.any(want != 0)

    def test_the_handle_outlives_the_graph_and_closes_once(self, tmp_path):
        cp, g, z, _x, _ps = self._chain(tmp_path, np.linspace(-1.5, 1.5, N))
        h = g.cuda_graph()
        del g
        run_as_child(h.handle)
        assert "closed" not in repr(h)
        h.close()
        h.close()
        with pytest.raises(ValueError, match="closed"):
            h.handle  # noqa: B018

    def test_bind_then_a_new_clone_writes_the_new_arrays(self, tmp_path):
        cp, g, z, x, ps = self._chain(tmp_path, np.linspace(-1.5, 1.5, N))
        x2 = cp.asarray(np.linspace(0.5, 2.5, N))
        old = g.cuda_graph()                       # taken before the swap: still the old arrays
        g._rebind([(x, x2)])
        z[...] = 0
        run_as_child(old.handle)
        g.launch()
        z_new = cp.asnumpy(z).copy()
        z[...] = 0
        with g.cuda_graph() as new:
            run_as_child(new.handle)
        np.testing.assert_array_equal(cp.asnumpy(z), z_new)
        z[...] = 0
        run_as_child(old.handle)                  # a snapshot: the old clone still reads the old input
        assert not np.array_equal(cp.asnumpy(z), z_new)
        old.close()

    def test_a_graph_with_a_host_node_is_refused(self, tmp_path):
        import cupy as cp

        a, b = cp.arange(8, dtype=cp.float64), cp.zeros(8)

        def work():
            cp.multiply(a, 2.0, out=b)
            cp.cuda.runtime.launchHostFunc(cp.cuda.get_current_stream().ptr, _NOOP, 0)

        g = eagle.Graph().add_node(eagle.graph.opaque(work, reads=[a], writes=[b])).build()
        with pytest.raises(RuntimeError, match=r"host-function node") as e:
            g.cuda_graph()
        print(e.value)

    def test_a_segmented_graph_is_refused(self, tmp_path):
        import cupy as cp
        from test_graph_gpu_eager import _graph, _x

        g, _z = _graph(plans(tmp_path, "gpu"), cp, _x(cp))
        g.build()
        with pytest.raises(RuntimeError, match=r"segmented.*'test_gpu_eager' \(eager_host\)"):
            g.cuda_graph()

    def test_a_conditional_graph_is_refused_with_the_driver_error(self, tmp_path):
        import cupy as cp

        n = cp.asarray(np.array([3], np.uint32))
        one = cp.asarray(np.array([1], np.uint32))
        acc = cp.zeros(4)
        dec = eagle.graph.opaque(lambda: cp.subtract(n, one, out=n), reads=[n, one], writes=[n], name="dec")
        body = eagle.graph.opaque(lambda: cp.add(acc, 1.0, out=acc), reads=[acc], writes=[acc], name="add")
        loop = eagle.repeat_while((body, dec), eagle.SkipGuard.nonzero(n), 5)
        loop()
        n[...] = 3
        g = eagle.Graph().add_node(loop, name="loop").build()
        try:
            h = g.cuda_graph()
        except RuntimeError as e:
            msg = str(e)
            assert "conditional node" in msg and "cudaErrorNotSupported" in msg and "driver " in msg, msg
            print(msg)
        else:
            h.close()
            pytest.skip("this CUDA driver accepts a conditional graph as a child graph: the refusal row does not apply")


def test_a_cpu_graph_is_refused(tmp_path):
    ps = plans(tmp_path, "cpu")
    g, _z = chain(ps, np.linspace(-1.5, 1.5, N), 0.75)
    with pytest.raises(RuntimeError, match="built for the CPU"):
        g.cuda_graph()
