# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0
"""Seeing an eagle.Graph: show()/str(), repr, plot(), to_mermaid(), to_dot().
Pure host; the picture is the C++ layout (eagle/graph/*.h) drawn or printed."""

import io
import sys
import threading

import numpy as np
import pytest

import eagle
from eagle import SkipGuard, _graph_view, repeat_while, skippable
from eagle.graph import opaque

from _graph_fixtures import host_chain


def _fork_join():
    x, left, right, out = (np.zeros(4) for _ in range(4))
    nop = lambda: None  # noqa: E731
    g = eagle.Graph()
    g.add_group([opaque(nop, reads={"x": x}, writes={"a": left}),
                 opaque(nop, reads={"x": x}, writes={"b": right})], names=["l", "r"])
    g.add_node(opaque(nop, reads={"a": left, "b": right}, writes={"out": out}), name="join")
    return g


def test_fork_join_snapshot_before_and_after_build():
    g = _fork_join()
    assert repr(g) == "<eagle.Graph 3 nodes, 2 stages, unbuilt>"
    assert str(g) == (
        "eagle.Graph  3 nodes \u00b7 1 group \u00b7 backend: (not built)\n"
        "\n"
        "  stage 0 \u250c\u254c\u254c\u254c\u2510   \u250c\u254c\u254c\u254c\u2510\n"
        "          \u254e l \u254e   \u254e r \u254e\n"
        "          \u2514\u254c\u252c\u254c\u2518   \u2514\u254c\u252c\u254c\u2518\n"
        "            a       b\n"
        "            \u2514\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2518\n"
        "  stage 1   \u250c\u254c\u254c\u254c\u2534\u254c\u254c\u254c\u2510\n"
        "            \u254e join  \u254e\n"
        "            \u2514\u254c\u254c\u254c\u254c\u254c\u254c\u254c\u2518\n")
    g.build()
    assert str(g) == (
        "eagle.Graph  3 nodes \u00b7 1 group \u00b7 backend: cpu\n"
        "\n"
        "  stage 0 work_share \u250c\u254c\u254c\u254c\u2510   \u250c\u254c\u254c\u254c\u2510\n"
        "                     \u254e l \u254e   \u254e r \u254e\n"
        "                     \u2514\u254c\u252c\u254c\u2518   \u2514\u254c\u252c\u254c\u2518\n"
        "                       a       b\n"
        "                       \u2514\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2518\n"
        "  stage 1 work_share   \u250c\u254c\u254c\u254c\u2534\u254c\u254c\u254c\u2510\n"
        "                       \u254e join  \u254e\n"
        "                       \u2514\u254c\u254c\u254c\u254c\u254c\u254c\u254c\u2518\n")
    assert repr(g) == "<eagle.Graph 3 nodes, 2 stages, cpu>"


def test_show_prints_what_str_returns():
    g = _fork_join().build()
    buf = io.StringIO()
    g.show(file=buf)
    assert buf.getvalue() == str(g) + "\n"


def test_exact_chain_snapshot(tmp_path):
    b1, b2, *_ = host_chain(tmp_path)
    g = eagle.Graph().add(b1, name="scale").add(b2, name="addone")
    assert str(g) == (
        "eagle.Graph  2 nodes \u00b7 0 groups \u00b7 backend: (not built)\n"
        "\n"
        "  stage 0  \u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n"
        "           \u2502 scale \u2502\n"
        "           \u2514\u2500\u2500\u2500\u252c\u2500\u2500\u2500\u2518\n"
        "               y\n"
        "               \u2502\n"
        "  stage 1 \u250c\u2500\u2500\u2500\u2500\u2534\u2500\u2500\u2500\u2500\u2510\n"
        "          \u2502 addone  \u2502\n"
        "          \u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\n")


def test_conditionals_are_frames_with_their_labels():
    flag = np.ones(1, dtype=np.uint32)
    a = np.zeros(2)
    nop = lambda: None  # noqa: E731
    g = (eagle.Graph()
         .add_node(opaque(nop, writes={"a": a}), name="seed")
         .add_node(skippable(opaque(nop, reads={"a": a}), SkipGuard.nonzero(flag)), name="maybe")
         .add_node(repeat_while(opaque(nop, reads={"a": a}), SkipGuard.nonzero(flag), 7),
                   name="loop"))
    text = str(g)
    assert "skip if guard" in text and "repeat ≤ 7" in text
    assert "╭" in text and "╰" in text          # rounded frame corners
    assert text.count("╌") > 0                        # opaque nodes are dashed
    ascii_only = _graph_view.description(g).ascii(unicode=False)
    assert all(ord(c) < 128 for c in ascii_only) and "skip if guard" in ascii_only


def test_recorded_tag_is_in_the_header_when_recording():
    g = _fork_join().build(record=True)
    assert _graph_view.header(g).endswith("backend: cpu · recorded")


def test_empty_graph_shows_a_placeholder():
    assert "(empty)" in str(eagle.Graph())
    assert repr(eagle.Graph()) == "<eagle.Graph 0 nodes, 0 stages, unbuilt>"


def test_export_round_trip_lists_each_node_and_edge_once():
    g = _fork_join().build()
    n, e = g.num_nodes(), len(g.edges())
    mer, dot = g.to_mermaid(), g.to_dot()
    assert mer.startswith("flowchart TD\n") and dot.startswith("digraph eagle {")
    for i in range(n):
        assert mer.count(f"  n{i}[") + mer.count(f"    n{i}[") == 1
        assert dot.count(f"\n  n{i} [label") + dot.count(f"\n    n{i} [label") == 1
    assert mer.count("-->") + mer.count("-.->") + mer.count("==>") == e
    assert dot.count(" -> ") == e
    assert "n0 -->|\"x\"" not in mer      # the fork members share the READ plane: no edge


def test_plot_returns_a_figure_headless(tmp_path):
    pytest.importorskip("matplotlib")
    flag = np.ones(1, dtype=np.uint32)
    a = np.zeros(2)
    g = _fork_join()
    g.add_node(skippable(opaque(lambda: None, reads={"out": a}), SkipGuard.nonzero(flag)),
               name="maybe")
    g.build()
    fig = g.plot()
    from matplotlib.figure import Figure

    assert isinstance(fig, Figure)
    out = tmp_path / "graph.png"
    fig.savefig(out)
    assert out.stat().st_size > 1000
    assert g.plot(edge_labels=False, show_schedule=False) is not fig
    texts = {t.get_text() for t in fig.axes[0].texts}
    assert {"l", "r", "join", "maybe", "skip if guard"} <= texts


def test_plot_without_matplotlib_names_the_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.figure", None)
    with pytest.raises(ImportError, match=r"pip install raptor-eagle\[plot\]"):
        _fork_join().plot()


def test_show_during_a_concurrent_launch_neither_blocks_nor_raises_busy():
    entered, release = threading.Event(), threading.Event()

    def blocker():
        entered.set()
        assert release.wait(30)

    g = eagle.Graph().add_node(opaque(blocker, writes={"a": np.zeros(1)}), name="slow").build()
    t = threading.Thread(target=g.launch)
    t.start()
    try:
        assert entered.wait(30)
        shown = []
        for call in (lambda: shown.append(str(g)), g.to_mermaid, g.to_dot, lambda: repr(g),
                     lambda: g.plot()):
            try:
                call()
            except ImportError:
                pass                              # no matplotlib here: not a Busy
        assert shown and "slow" in shown[0]
        with pytest.raises(eagle.Busy):
            g.launch()                            # the guard is still held
    finally:
        release.set()
        t.join(30)
