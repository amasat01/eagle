# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Memory a captured graph uses lives as long as the graph: drop the bound plan and the caller's
handles, collect, replay twice; the values stay right (an owned companion or keyed plane freed early
shows up as a wrong sum or an illegal address)."""

import gc

import numpy as np
import pytest

import eagle
from test_split_keyed import COL, R, _auto, _host, _want, _x, share

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("mode", ["legacy", "own_stream"])
def test_a_graph_outlives_its_dropped_bound_plan(tmp_path, mode):
    cp = pytest.importorskip("cupy")
    p = _auto(share, tmp_path, "float64", targets=("cuda",))
    own = cp.cuda.Stream(non_blocking=True)
    scope = own if mode == "own_stream" else cp.cuda.Stream.null
    with scope:
        x_host = _x("float64")
        x, z = cp.asarray(x_host), cp.zeros((R, COL))
        bound = p.bind(x=x, out=z)
        g = eagle.Graph().add(bound, name="share").build()
        del bound                       # the graph is the only holder of the plan's own planes now
        gc.collect()
        cp.cuda.runtime.deviceSynchronize()
        first = None
        for _ in range(2):
            z[...] = 0
            g.launch()
            cp.cuda.runtime.deviceSynchronize()
            got = _host(z).copy()
            if first is None:
                first = got
                np.testing.assert_allclose(first, _want(x_host), rtol=1e-12)
            np.testing.assert_array_equal(got, first)
        del g
        gc.collect()
        cp.cuda.runtime.deviceSynchronize()
