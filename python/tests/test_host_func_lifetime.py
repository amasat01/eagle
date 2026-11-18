# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A host node's function may reach back to the object that owns the node: the garbage
collector frees that cycle, and an interpreter that exits with one alive reports no leaked
binding objects."""

from __future__ import annotations

import gc
import os
import subprocess
import sys
import textwrap
import weakref

import numpy as np
import pytest

import eagle
from eagle import _transfer

try:
    import cupy as cp
    _devices = cp.cuda.runtime.getDeviceCount()
except Exception:                      # no cupy, or no device: the GPU row skips
    cp, _devices = None, 0


class _Owner:
    """Stands for a graph: it holds a host node and an eagle array."""


def _owner_in_a_cycle():
    owner = _Owner()
    owner.array = eagle.to_array(np.arange(8.0))
    owner.node = _transfer.HostNode(lambda: owner.array)   # the node's function reaches its owner
    return owner


def _exit_report(directory, code):
    """The stderr of a fresh interpreter that runs ``code`` from a file (hawk traces kernel
    source from disk) and exits."""
    script = directory / "child.py"
    script.write_text(textwrap.dedent(code))
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=300, env={**os.environ})
    assert done.returncode == 0, done.stderr
    return done.stderr


def test_a_host_node_whose_function_reaches_its_owner_is_collected():
    alive = weakref.ref(_owner_in_a_cycle())
    gc.collect()
    assert alive() is None


def test_an_exit_with_a_host_node_cycle_alive_reports_no_leaked_objects(tmp_path):
    stderr = _exit_report(tmp_path, """
        import numpy as np
        import eagle
        from eagle import _transfer

        class Owner:
            pass

        KEPT = Owner()
        KEPT.array = eagle.to_array(np.arange(8.0))
        KEPT.node = _transfer.HostNode(lambda: KEPT.array)
    """)
    assert "leaked" not in stderr, stderr


@pytest.mark.gpu
@pytest.mark.skipif(not _devices, reason="needs a GPU")
def test_an_exit_with_a_built_gpu_graph_with_a_host_node_alive_reports_no_leaked_objects(tmp_path):
    pytest.importorskip("hawk")
    stderr = _exit_report(tmp_path, f"""
        import cupy as cp
        import eagle
        import hawk

        @hawk.kernel
        def twice(x: hawk.Scalar, y: hawk.Mutable[hawk.Scalar]):
            y = 2.0 * x + 1.0

        deployed = eagle.deploy(twice, targets=("host", "cuda"), cache_dir={str(tmp_path)!r})

        @eagle.Graph.builder
        def across(d_in, d_mid, h_mid, h_out, d_out):
            g = eagle.Graph()
            deployed(d_in, d_mid)                  # a device node
            eagle.copy(d_mid, into=h_mid)
            deployed(h_mid, h_out)                 # the same kernel on host planes: a host node
            eagle.copy(h_out, into=d_out)
            return g

        D_OUT = cp.zeros(8)
        KEPT = across(cp.ones(8), cp.zeros(8), eagle.pinned(8), eagle.pinned(8), D_OUT)
        KEPT.launch(1)
        assert float(D_OUT[0].get()) == 7.0
    """)
    assert "leaked" not in stderr, stderr
