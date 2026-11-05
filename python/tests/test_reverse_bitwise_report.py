# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""grads.report.bitwise: True for every gradient a sample-order sum makes,
False exactly for one fed by a cross-sample scatter-add (a table read by
index); strict=True refuses such a reverse."""

import numpy as np
import pytest

import eagle
import eagle.plan
from _reverse_fixtures import chain, plans

hawk = pytest.importorskip("hawk")
from hawk import Index, Mutable, Scalar, Table  # noqa: E402

def _bytes(a):
    return np.asarray(a.get() if hasattr(a, "get") else a).tobytes()


N = 64


@hawk.kernel
def br_gather(table: Table[Scalar], where: Index, x: Scalar, y: Mutable[Scalar]):
    y = x * table.at(where)


def _gather_graph(tmp_path):
    (p,) = eagle.plan.auto([br_gather], targets=("host",), cache_dir=str(tmp_path / "c"))
    table = np.linspace(0.5, 2.0, 5)
    where = (np.arange(N) % 5).astype(np.int32)
    x, y = np.linspace(-1, 1, N), np.zeros(N)
    g = eagle.Graph().add(p.bind(table=table, where=where, x=x, y=y), name="lookup")
    return g.build(record=True).launch(), table, where, x


def test_a_scatter_add_gradient_is_reported_not_bitwise(tmp_path):
    g, table, where, x = _gather_graph(tmp_path)
    grads = eagle.diff.reverse(g.record, {"y": np.ones(N)})
    assert grads.report.bitwise == {"table": False, "x": True}
    np.testing.assert_allclose(grads["x"], table[where])
    np.testing.assert_allclose(grads["table"], np.bincount(where, weights=x), rtol=1e-12)


def test_strict_refuses_a_scatter_add_gradient(tmp_path):
    g, *_ = _gather_graph(tmp_path)
    with pytest.raises(ValueError, match=(
            "strict=True, but the gradient of 'table' comes through a cross-sample "
            "scatter-add in node 'lookup'")):
        eagle.diff.reverse(g.record, {"y": np.ones(N)}, strict=True)


def test_a_gather_free_graph_is_bitwise_and_repeats_exactly(tmp_path):
    g, _z = chain(plans(tmp_path, "cpu"), np.linspace(-1, 1, N), 0.4, record=True)
    g.launch()
    rev = eagle.diff.Reverse(g, strict=True)
    first = rev(g.record, {"sin.y": np.linspace(1, 2, N)})
    again = rev(g.record, {"sin.y": np.linspace(1, 2, N)})
    assert first.report.bitwise == {"x": True, "a": True}
    assert first.report.generation == again.report.generation == 1
    assert _bytes(first["x"]) == _bytes(again["x"])
    assert _bytes(first["a"]) == _bytes(again["a"])
