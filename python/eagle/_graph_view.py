# Copyright 2026 Alessandro Masat
# SPDX-License-Identifier: Apache-2.0

"""Seeing an :class:`eagle.graph.Graph`: the rows the C++ graph view
(``eagle/graph``) draws from, and the matplotlib drawing of its layout. Reads a
graph as it stands; never builds, launches or takes the busy guard."""

from __future__ import annotations

_STYLES = {"RAW": "-", "WAR": "--", "WAW": ":"}
_COND_FRAMES = {"skippable": ("if", lambda c: "skip if guard"),
                "repeat_while": ("while", lambda c: f"repeat ≤ {c.wrapper.max_iters}")}


def _stage_of_leaves(g) -> list:
    """The top-level stage each executable node sits in."""
    out = []
    for stage, node in enumerate(g._nodes):
        out += [stage] * (sum(len(b) for b in node.payload) if node.kind == "group" else 1)
    return out


def _tier(node) -> str:
    """How a node of your own code runs: ``child``, ``stream`` or ``host``
    (``""`` for a captured node)."""
    if node.kind == "child":
        return "child"
    from .graph import get_node_kind  # graph imports this module first

    tier = get_node_kind(node.kind).tier
    return "" if tier == "capture" else tier


def rows(g, drawing=True):
    """``(nodes, edges, frames)`` as ``_core.GraphDescription`` takes them: a
    ``"stage"`` frame per top-level add-stage (the row the layout gives its
    nodes), the conditionals inside them. ``drawing=False`` is the reverse pass's copy: labels without tier tags and
    no node opaque (eagle.diff has checked each has a vjp)."""
    from ._graph_kinds import OWN_CODE_KINDS

    leaves, _groups = g._leaves()
    stage_of = _stage_of_leaves(g)
    schedule = g._schedule if g._built else ()
    nodes = []
    frames = [("stage", f"stage {k}", -1) for k in range(len(g._nodes))]   # one row each
    for i, leaf in enumerate(leaves):
        n = leaf.node
        shown, frame = n, stage_of[i]
        label = n.label
        if n.kind in _COND_FRAMES:
            kind, text = _COND_FRAMES[n.kind]
            frames.append((kind, text(n.payload), stage_of[i]))
            frame = len(frames) - 1
            shown = n.payload.child
            label = n.name or shown.label
        tier = _tier(shown) if drawing else ""
        label += f" [{tier}]" if tier else ""
        nodes.append((label, n.kind, schedule[stage_of[i]] if schedule else "", frame,
                      drawing and shown.kind in OWN_CODE_KINDS))
    return nodes, [(e.src, e.dst, e.plane, e.kind) for e in g.edges()], frames


def description(g, drawing=True):
    from ._core import GraphDescription

    return GraphDescription(*rows(g, drawing))


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def header(g) -> str:
    leaves, groups = g._leaves()
    backend = g._backend if g._built else "(not built)"
    recorded = " · recorded" if g._record_on else ""
    return (f"eagle.Graph  {_count(len(leaves), 'node')} · {_count(len(groups), 'group')} "
            f"· backend: {backend}{recorded}")


def text(g) -> str:
    body = description(g).ascii() if g._nodes else "  (empty)\n"
    if g._backend == "cpu":         # stream= is kept and shown, never silently dropped
        body += "".join(f"  {a.label} [stream: cpu runs eagerly]\n" for a in g._leaves()[0]
                        if _tier(a.node) == "stream")
    return header(g) + "\n\n" + body


def _draw_frames(ax, frames, nodes, pos, half):
    from matplotlib.patches import Rectangle

    for fi, (kind, label, _parent) in enumerate(frames):
        members = [i for i, n in enumerate(nodes) if n[3] == fi]
        if kind == "stage":         # a row, tagged in the gutter
            continue
        x0 = min(pos[i][0] - half[i][0] for i in members) - 0.3
        x1 = max(pos[i][0] + half[i][0] for i in members) + 0.3
        y0 = min(pos[i][1] - half[i][1] for i in members) - 0.3
        y1 = max(pos[i][1] + half[i][1] for i in members) + 0.4
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linestyle="--",
                               linewidth=1.0, edgecolor="0.45"))
        ax.text(x0 + 0.1, y1 - 0.05, label, fontsize=8, color="0.35", va="top")


def _draw_nodes(ax, nodes, pos, half):
    from matplotlib.patches import FancyBboxPatch

    for (label, _kind, _stage, _frame, opaque_node), (cx, cy), (hw, hh) in zip(nodes, pos, half):
        ax.add_patch(FancyBboxPatch((cx - hw, cy - hh), 2 * hw, 2 * hh,
                                    boxstyle="round,pad=0.02,rounding_size=0.08",
                                    facecolor="white", edgecolor="0.15",
                                    linestyle="--" if opaque_node else "-"))
        ax.text(cx, cy, label, ha="center", va="center", fontsize=9)


def _draw_edges(ax, edges, pos, half, labels):
    from matplotlib.patches import FancyArrowPatch

    pairs: dict = {}
    for src, dst, plane, kind in edges:
        pairs.setdefault((src, dst), []).append((plane, kind))
    for (src, dst), found in pairs.items():
        (x0, y0), (x1, y1) = pos[src], pos[dst]
        for kind in dict.fromkeys(k for _p, k in found):          # one arrow per hazard kind
            ax.add_patch(FancyArrowPatch((x0, y0 - half[src][1]), (x1, y1 + half[dst][1]),
                                         arrowstyle="-|>", mutation_scale=10,
                                         linestyle=_STYLES[kind], color="0.3"))
        names = ", ".join(dict.fromkeys(p if k == "RAW" else f"{p}:{k}" for p, k in found if p))
        if labels and names:
            ax.text(x0 + 0.38 * (x1 - x0), y0 + 0.38 * (y1 - y0), names, fontsize=7, color="0.2",
                    ha="center", va="bottom", bbox=dict(facecolor="white", edgecolor="none", pad=0.5))


def _draw_schedule(ax, layer, nodes, frames, left, step):
    """Tag the first row of each stage in the gutter (a stage may span rows)."""
    stage = [n[3] if frames[n[3]][0] == "stage" else frames[n[3]][2] for n in nodes]
    for lv in sorted(set(layer)):
        here = [i for i in range(len(nodes)) if layer[i] == lv]
        if lv and any(layer[i] == lv - 1 and stage[i] == stage[here[0]] for i in range(len(nodes))):
            continue
        tags = {nodes[i][2] for i in here} - {""}
        tag = f"  {tags.pop()}" if len(tags) == 1 else ""
        ax.text(left, -lv * step, f"stage {stage[here[0]]}{tag}", fontsize=8, color="0.4",
                ha="right", va="center")


def plot(g, ax, edge_labels: bool, show_schedule: bool):
    """Draw ``g`` at the C++ layout's coordinates; returns the Figure."""
    try:
        from matplotlib.figure import Figure
    except ImportError as err:
        raise ImportError("eagle.Graph.plot needs matplotlib; install it with "
                          "`pip install raptor-eagle[plot]`") from err
    nodes, edges, frames = rows(g)
    layer, _order, xy = description(g).layout()
    if ax is None:
        width = 1.6 * (2 + 2 * max((abs(p[0]) for p in xy), default=0))
        fig = Figure(figsize=(max(4.0, width), max(3.0, 1.3 * (1 + max(layer, default=0)))))
        ax = fig.subplots()
    else:
        fig = ax.figure
    step = 1.8
    pos = [(x * 2.6, -y * step) for x, y in xy]
    half = [(max(0.7, 0.085 * len(n[0]) + 0.35), 0.32) for n in nodes]
    _draw_frames(ax, frames, nodes, pos, half)
    _draw_nodes(ax, nodes, pos, half)
    _draw_edges(ax, edges, pos, half, edge_labels)
    left = min(p[0] - h[0] for p, h in zip(pos, half))
    right = max(p[0] + h[0] for p, h in zip(pos, half))
    if show_schedule:
        _draw_schedule(ax, layer, nodes, frames, left - 0.5, step)
    ys = [p[1] for p in pos]
    ax.set_xlim(left - (3.4 if show_schedule else 0.8), right + 0.8)
    ax.set_ylim(min(ys) - 1.1, max(ys) + 1.1)
    ax.axis("off")
    ax.set_title(header(g), fontsize=9, loc="left")
    return fig
