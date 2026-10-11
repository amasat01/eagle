# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The node-kind registry: idempotent, never replaces, refuses unknown kinds."""

import pytest

import eagle
from eagle import graph as G


def _k(tag):
    return dict(reads=lambda n: (), writes=lambda n: (),
                lower_gpu=lambda n, c: (lambda: tag), lower_cpu=lambda n, c: (lambda: tag))


def test_identical_reregistration_is_idempotent_and_different_is_refused():
    kw = _k("a")
    first = G.register_node_kind("zz_test_kind_once", **kw)
    assert G.register_node_kind("zz_test_kind_once", **kw) is first
    with pytest.raises(ValueError, match="already registered with a different definition"):
        G.register_node_kind("zz_test_kind_once", **_k("b"))
    assert G.get_node_kind("zz_test_kind_once") is first


def test_builtins_cannot_be_replaced():
    for kind in ("launch", "opaque", "group", "skippable", "repeat_while"):
        with pytest.raises(ValueError, match="already registered"):
            G.register_node_kind(kind, **_k("x"))


def test_a_registered_kind_enters_a_graph_without_editing_graph_py():
    ran = []
    G.register_node_kind(
        "zz_test_kind_ran", reads=lambda n: (), writes=lambda n: (),
        lower_gpu=lambda n, c: (lambda: ran.append("gpu")),
        lower_cpu=lambda n, c: (lambda: ran.append("cpu")))
    g = eagle.Graph().add_node(G.Node("zz_test_kind_ran", None), name="mine")
    g.build().launch()
    assert ran == ["cpu"] and g.node_labels() == ["mine"]


def test_unregistered_kind_is_refused_naming_the_registry():
    with pytest.raises(ValueError, match="register_node_kind"):
        eagle.Graph().add_node(G.Node("zz_no_such_kind", None))
