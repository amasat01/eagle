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
        "  stage 0 ┌─────┐   ┌─────┐\n"
        "          │ <l> │   │ <r> │\n"
        "          └──┬──┘   └──┬──┘\n"
        "             a         b\n"
        "             └────┬────┘\n"
        "  stage 1    ┌────┴────┐\n"
        "             │ <join>  │\n"
        "             └─────────┘\n")
    g.build()
    assert str(g) == (
        "eagle.Graph  3 nodes \u00b7 1 group \u00b7 backend: cpu\n"
        "\n"
        "  stage 0 work_share ┌─────┐   ┌─────┐\n"
        "                     │ <l> │   │ <r> │\n"
        "                     └──┬──┘   └──┬──┘\n"
        "                        a         b\n"
        "                        └────┬────┘\n"
        "  stage 1 work_share    ┌────┴────┐\n"
        "                        │ <join>  │\n"
        "                        └─────────┘\n")
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
    assert "┌─ skip if guard" in text and "┌─ repeat" in text   # frames are light boxes
    assert "<inc>" not in text and "╌" not in text and "╭" not in text   # no glyphs a browser font may lack
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
        assert sum(ln.strip().startswith(f"n{i}[") for ln in mer.splitlines()) == 1
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

# --- the picture's rows are the add-stages; labels whole; hazard kind as line style ---

def _plain_nodes(*names):
    return [(n, "launch", "", f, False) for n, f in names]


def _staged(count, extra=()):
    return [("stage", f"stage {k}", -1) for k in range(count)] + list(extra)


def _nested_shapes():
    from eagle._core import GraphDescription

    raw = lambda *e: [(s, d, p, "RAW") for s, d, p in e]  # noqa: E731
    return {
        "fork_in_branch": GraphDescription(
            _plain_nodes(("reset", 0), ("inc_p", 1), ("inc_q", 1), ("add", 1), ("mul_s", 1), ("out", 2)),
            raw((0, 1, "p"), (0, 2, "q"), (0, 4, "s"), (1, 3, "p"), (2, 3, "q"), (3, 5, "r"), (4, 5, "s")),
            _staged(3)),
        "conditional_arm": GraphDescription(
            _plain_nodes(("reset", 0), ("inc", 2), ("mul", 2), ("inc10", 2)),
            raw((0, 1, "x"), (1, 2, "x"), (2, 3, "x")), _staged(2, [("if", "skip if guard", 1)])),
        "loop_in_loop": GraphDescription(
            _plain_nodes(("seed", 0), ("step", 3), ("fold", 2)), raw((0, 1, "a"), (1, 2, "a")),
            _staged(2, [("while", "repeat <= 3", 1), ("while", "repeat <= 2", 2)])),
    }


_NOP = lambda: None  # noqa: E731


def _two_in_sequence():
    return (eagle.Graph().add_node(opaque(_NOP, writes={"a": np.zeros(4)}), name="first")
            .add_node(opaque(_NOP, writes={"b": np.zeros(4)}), name="second"))


def _skips_a_stage():
    a, b, c = (np.zeros(4) for _ in range(3))
    return (eagle.Graph().add_node(opaque(_NOP, writes={"a": a}), name="s0")
            .add_node(opaque(_NOP, writes={"b": b}), name="s1")
            .add_node(opaque(_NOP, reads={"a": a, "b": b}, writes={"c": c}), name="s2"))


def _long_names():
    a, b = np.zeros(4), np.zeros(4)
    return (eagle.Graph().add_node(opaque(_NOP, writes={"state_position": a}), name="integrate_positions")
            .add_node(opaque(_NOP, reads={"state_position": a}, writes={"energy_total": b}),
                      name="measure_energy")
            .add_node(opaque(_NOP, writes={"state_position": a}), name="reset_positions"))


def _fork_conditional_loop():
    a, b, c, d = (np.zeros(4) for _ in range(4))
    flag = np.ones(1, dtype=np.uint32)
    return (eagle.Graph().add_node(opaque(_NOP, writes={"a": a}), name="seed")
            .add_group([opaque(_NOP, reads={"a": a}, writes={"b": b}),
                        skippable(opaque(_NOP, reads={"a": a}, writes={"c": c}), SkipGuard.nonzero(flag)),
                        opaque(_NOP, reads={"a": a}, writes={"d": d})], names=["l", "m", "r"])
            .add_node(repeat_while(opaque(_NOP, reads={"b": b, "c": c, "d": d}, writes={"a": a}),
                                   SkipGuard.nonzero(flag), 5), name="loop"))


_GOLD_A = r"""eagle.Graph  2 nodes · 0 groups · backend: (not built)

  stage 0  ┌─────────┐
           │ <first> │
           └─────────┘


  stage 1 ┌───────────┐
          │ <second>  │
          └───────────┘
"""
_GOLD_B = r"""eagle.Graph  3 nodes · 0 groups · backend: (not built)

  stage 0   ┌───────┐
            │ <s0>  │
            └───┬───┘
                a
                └─────┐
  stage 1 ┌───────┐   │
          │ <s1>  │   │
          └───┬───┘   │
              b       │
              └─┬─────┘
  stage 2   ┌───┴───┐
            │ <s2>  │
            └───────┘
"""
_GOLD_C = r"""eagle.Graph  3 nodes · 0 groups · backend: (not built)

  stage 0 ┌───────────────────────┐
          │ <integrate_positions> │
          └───────────┬───────────┘
                      state_position
                     ┌┴~~~~~~~~~~~~~┐
  stage 1  ┌─────────┴─────────┐    !
           │ <measure_energy>  │    !
           └─────────┬─────────┘    !
                     state_position !
                     └.┬~~~~~~~~~~~~┘
  stage 2    ┌─────────┴─────────┐
             │ <reset_positions> │
             └───────────────────┘
  edges:  ── RAW data flows  .. WAR must wait to overwrite  ~~ WAW same output
"""
_GOLD_D = r"""eagle.Graph  5 nodes · 1 group · backend: (not built)

  stage 0                 ┌─────────┐
                          │ <seed>  │
                          └────┬────┘
                               a
             ┌──────┬──────────┴─┬───────────────┐
             │      !   ┌─ skip if guard ─┐      │
  stage 1 ┌──┴──┐   !   │ ┌──────┴──────┐ │   ┌──┴──┐
          │ <l> │   !   │ │     <m>     │ │   │ <r> │
          └──┬──┘   !   │ └──────┬──────┘ │   └──┬──┘
             │      !   └────────┼────────┘      │
             b,a    !            c,a             d,a
             └──────┴──────────┬─┴───────────────┘
                       ┌─ repeat ≤ 5 ──┐
  stage 2              │ ┌─────┴─────┐ │
                       │ │  <loop>   │ │
                       │ └───────────┘ │
                       └───────────────┘
  edges:  ── RAW data flows  ~~ WAW same output
"""
_GOLD_NESTED = {
    "fork_in_branch": r"""  stage 0            ┌───────┐
                     │ reset │
                     └───┬───┘
                         p,q,s
              ┌──────────┴┬───────────┐
  stage 1 ┌───┴───┐   ┌───┴───┐   ┌───┴───┐
          │ inc_p │   │ inc_q │   │ mul_s │
          └───┬───┘   └───┬───┘   └───┬───┘
              p           q           s
              └─────────┬─┘    ┌──────┘
                     ┌──┴──┐   │
                     │ add │   │
                     └──┬──┘   │
                        r      │
                        └─┬────┘
  stage 2              ┌──┴──┐
                       │ out │
                       └─────┘
""",
    "conditional_arm": r"""  stage 0      ┌───────┐
               │ reset │
               └───┬───┘
                   x
                   │
          ┌─ skip if guard ─┐
  stage 1 │ ┌──────┴──────┐ │
          │ │     inc     │ │
          │ └──────┬──────┘ │
          │        x        │
          │        │        │
          │ ┌──────┴──────┐ │
          │ │     mul     │ │
          │ └──────┬──────┘ │
          │        x        │
          │        │        │
          │ ┌──────┴──────┐ │
          │ │    inc10    │ │
          │ └─────────────┘ │
          └─────────────────┘
""",
    "loop_in_loop": r"""  stage 0       ┌───────┐
                │ seed  │
                └───┬───┘
                    a
                    │
          ┌─ repeat <= 3 ─────┐
          │ ┌─ repeat <= 2 ─┐ │
  stage 1 │ │ ┌─────┴─────┐ │ │
          │ │ │   step    │ │ │
          │ │ └─────┬─────┘ │ │
          │ └───────┼───────┘ │
          │         a         │
          │         │         │
          │   ┌─────┴─────┐   │
          │   │   fold    │   │
          │   └───────────┘   │
          └───────────────────┘
""",
}


@pytest.mark.parametrize("build, gold", [(_two_in_sequence, _GOLD_A), (_skips_a_stage, _GOLD_B),
                                         (_long_names, _GOLD_C), (_fork_conditional_loop, _GOLD_D)])
def test_picture_snapshots(build, gold):
    assert str(build()) == gold


@pytest.mark.parametrize("name", ["fork_in_branch", "conditional_arm", "loop_in_loop"])
def test_nested_shapes_render(name):
    assert _nested_shapes()[name].ascii() == _GOLD_NESTED[name]


@pytest.mark.parametrize("build", [_two_in_sequence, _skips_a_stage, _long_names, _fork_conditional_loop])
def test_row_is_the_stage(build):
    g = build()
    layer, _order, _xy = _graph_view.description(g).layout()
    assert layer == _graph_view._stage_of_leaves(g)


def test_nothing_is_cut_when_there_is_room_and_each_edge_has_one_label():
    text = str(_long_names())
    assert "state_position" in text and "…" not in text
    from eagle._core import GraphDescription

    d = _nested_shapes()["fork_in_branch"]
    nodes, edges = [(f"n{i}", "launch", "", f, False) for i, f in enumerate([0, 1, 1, 1, 1, 2])], []
    for i, (s, t, _p, _k) in enumerate([(0, 1, 0, 0), (0, 2, 0, 0), (0, 4, 0, 0), (1, 3, 0, 0),
                                         (2, 3, 0, 0), (3, 5, 0, 0), (4, 5, 0, 0)]):
        edges.append((s, t, f"plane{i}", "RAW"))
    shown = GraphDescription(nodes, edges, _staged(3)).ascii()
    assert d.num_edges() == len(edges)
    for _s, _t, plane, _k in edges:
        assert shown.count(plane) == 1


def test_hazard_kind_is_line_style_with_one_legend_line():
    text = str(_long_names())
    assert ".." in text and "~~" in text and ":WAR" not in text and ":WAW" not in text
    assert text.count("edges:") == 1 and "edges:" not in str(_skips_a_stage())


def test_exports_do_not_show_the_stage_rows():
    g = _fork_conditional_loop()
    mer, dot = g.to_mermaid(), g.to_dot()
    assert "stage" not in mer and "stage" not in dot
    assert "subgraph f0[" in mer and "cluster_f0 " in dot
