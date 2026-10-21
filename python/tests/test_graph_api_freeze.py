# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""The irreversible public surface of eagle.Graph, pinned by inspection."""

import inspect

import eagle
from eagle import graph as G
from eagle._graph_edges import Edge


def _sig(f):
    return str(inspect.signature(f))


def test_graph_methods_and_signatures():
    C = eagle.Graph
    assert _sig(C.add) == "(self, bound, *, name=None, grad=True) -> 'Graph'"
    assert _sig(C.add_node) == "(self, node, *, name=None) -> 'Graph'"
    assert _sig(C.add_group) == "(self, branches, *, names=None) -> 'Graph'"
    assert _sig(C.build) == "(self, *, backend=None, record=False, dispatch=None) -> 'Graph'"
    assert _sig(C.launch) == "(self, n: 'int' = 1) -> 'Graph'"
    assert _sig(C.enqueue) == "(self, n: 'int' = 1) -> 'Graph'"
    assert _sig(C.synchronize) == "(self) -> 'Graph'"
    assert _sig(C.node_labels) == "(self) -> 'list'"
    assert _sig(C.num_nodes) == "(self) -> 'int'" or _sig(C.num_nodes) == "(self)"
    assert _sig(C.edges) == "(self) -> 'tuple'"
    assert _sig(C.show) == "(self, *, file=None) -> 'None'"
    assert _sig(C.plot) == ("(self, ax=None, *, edge_labels: 'bool' = True, "
                            "show_schedule: 'bool' = True)")
    assert _sig(C.to_mermaid) == "(self) -> 'str'"
    assert _sig(C.to_dot) == "(self) -> 'str'"
    for prop in ("schedule", "backend", "record", "dispatch"):
        assert isinstance(inspect.getattr_static(C, prop), property)
    assert not hasattr(C, "run") and not hasattr(C, "replay")


def test_module_functions_and_registry():
    assert _sig(G.register_node_kind) == (
        "(kind: 'str', *, reads, writes, lower_gpu, lower_cpu, transpose=None, "
        "record_planes=None, native_entry=None, tier='capture', check=None) -> 'NodeKind'")
    assert _sig(G.opaque) == ("(fn, *, reads=(), writes=(), eager=False, stream=None, "
                              "name=None, grad=True) -> 'Node'")
    assert _sig(G.child) == "(owner, *, reads=(), writes=(), name=None) -> 'Node'"
    # one home for conditionals: the existing eagle.skippable / eagle.repeat_while
    assert not hasattr(G, "skippable") and not hasattr(G, "repeat_while")
    assert not hasattr(eagle.Graph, "skippable")
    assert not hasattr(eagle.Graph, "repeat_while")
    assert [f.name for f in G.NodeKind.__dataclass_fields__.values()] == [
        "kind", "reads", "writes", "lower_gpu", "lower_cpu", "transpose",
        "record_planes", "native_entry", "tier", "check"]
    assert Edge._fields == ("src", "dst", "plane", "kind")


def test_exports_and_no_dag_name():
    assert eagle.Graph is G.Graph and eagle.Busy.__name__ == "Busy"
    assert "Graph" in eagle.__all__ and "Busy" in eagle.__all__
    assert not hasattr(eagle, "Dag")
    import eagle._counters as c
    assert "graph_builds" in c.snapshot()
