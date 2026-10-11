# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""A record=True graph binds each parameter as a one-cell plane (so update is
a copy); a plain graph keeps the by-value immediate. update's names and
refusals."""

import numpy as np
import pytest

import eagle
from _reverse_fixtures import chain, plans

N = 16


def test_recorded_parameters_are_cells_plain_ones_are_values(tmp_path):
    ps = plans(tmp_path, "cpu")
    x = np.linspace(-1, 1, N)
    plain, _ = chain(ps, x, 0.5)
    recorded, _ = chain(ps, x, 0.5, record=True)
    p, r = plain._nodes[0].payload, recorded._nodes[0].payload
    assert ("uniform", "a") in p._arg_spec and p._planes["a"] == 0.5
    assert ("lookup", "a") in r._arg_spec
    cell = r._planes["a"]
    assert cell.shape == (1,) and cell[0] == 0.5
    assert recorded._recorder.cells[("square", "a")] is cell
    assert recorded.node_labels() == plain.node_labels() == ["square", "sin"]


def test_update_changes_the_next_launch(tmp_path):
    ps = plans(tmp_path, "cpu")
    x = np.linspace(-1, 1, N)
    g, z = chain(ps, x, 0.5, record=True)
    g.update({"square.a": 2.0})
    g.launch()
    np.testing.assert_allclose(z, np.sin(2.0 * x * x), rtol=1e-15)


def test_update_refusals(tmp_path):
    ps = plans(tmp_path, "cpu")
    g, _ = chain(ps, np.linspace(-1, 1, N), 0.5, record=True)
    with pytest.raises(ValueError, match=r"'b' is not a parameter of this graph \(parameters: \['a', 'square.a'\]\)"):
        g.update(b=1.0)
    g.launch()
    with pytest.raises(RuntimeError, match="the Record of the last launch is still live"):
        g.update(a=1.0)
    g.record.detach()
    g.update(a=1.0)
    plain, _ = chain(ps, np.linspace(-1, 1, N), 0.5)
    with pytest.raises(ValueError, match="only a record=True graph holds its parameters in planes"):
        plain.update(a=1.0)
