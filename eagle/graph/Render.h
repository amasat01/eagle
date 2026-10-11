// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file graph/Render.h
 * @brief Text renderings of a Description: an ASCII block diagram, Mermaid
 *        and Graphviz DOT. Each lists every node and every edge once.
 */

#include "eagle/graph/Ascii.h"
#include "eagle/graph/Description.h"
#include "eagle/graph/Layout.h"

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace eagle {
namespace graph {

namespace detail {

inline std::string escape(const std::string& s, char quote, const std::string& with)
{
    std::string out;
    for (char c : s) {
        if (c == quote)
            out += with;
        else if (c == '\n')
            out += ' ';
        else
            out += c;
    }
    return out;
}

inline std::string dotEscape(const std::string& s)
{
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\')
            out += '\\';
        out += (c == '\n') ? ' ' : c;
    }
    return out;
}

/** @brief Frame ids by parent (``-1`` first), and node ids by frame. */
struct FrameTree {
    std::vector<std::vector<gidx>> childFrames;   // index parent + 1
    std::vector<std::vector<gidx>> frameNodes;    // index frame + 1
    explicit FrameTree(const Description& d)
        : childFrames(d.frames.size() + 1), frameNodes(d.frames.size() + 1)
    {
        for (const Frame& f : d.frames)
            childFrames[f.parent + 1].push_back(f.id);
        for (const Node& n : d.nodes)
            frameNodes[n.frame + 1].push_back(n.id);
    }
};

}  // namespace detail

/** @brief The box block diagram: one box per node placed by the layout's
 *  layer (row) and order (column), edges routed between the boxes with the
 *  plane names under the source (``:WAR`` / ``:WAW`` appended when the hazard
 *  is not a read-after-write), an edge that skips layers carried down as a
 *  vertical line through the layers in between, a conditional as a frame
 *  around its nodes (its label in the top border), an opaque node with a
 *  dashed border, and a ``stage <k> <schedule>`` tag in the left gutter.
 *  ``unicode = false`` draws with ``+ - |`` only. A label longer than
 *  ``maxLabel`` cells is cut with an ellipsis. Deterministic: the same
 *  Description and Layout always give the same text.
 *
 *  A frame is drawn as the bounding box of its nodes; a node of another frame
 *  that falls inside that box is not moved out of it. */
inline std::string ascii(const Description& d, const Layout& lay, bool unicode = true,
    std::size_t maxLabel = 24)
{
    return detail::BoxDiagram(d, lay, unicode, maxLabel).render();
}

/** @brief The box diagram with the default layout. */
inline std::string ascii(const Description& d) { return ascii(d, layout(d)); }

/** @brief Mermaid ``flowchart TD``: frames as subgraphs, opaque nodes in the
 *  ``opaque`` class (dashed), edge arrow by kind (RAW ``-->``, WAR ``-.->``,
 *  WAW ``==>``) labelled with the plane. */
inline std::string mermaid(const Description& d)
{
    d.validate();
    const detail::FrameTree tree(d);
    std::string out = "flowchart TD\n";
    auto node = [&](const Node& n, const std::string& pad) {
        std::string text = detail::escape(n.label, '"', "#quot;");
        if (!n.stage.empty())
            text += "<br/>" + n.stage;
        out += pad + "n" + std::to_string(n.id) + "[\"" + text + "\"]\n";
    };
    auto emit = [&](auto&& self, gidx frame, const std::string& pad) -> void {
        for (gidx v : tree.frameNodes[frame + 1])
            node(d.nodes[v], pad);
        for (gidx f : tree.childFrames[frame + 1]) {
            out += pad + "subgraph f" + std::to_string(f) + "[\""
                + detail::escape(d.frames[f].text(), '"', "#quot;") + "\"]\n";
            self(self, f, pad + "  ");
            out += pad + "end\n";
        }
    };
    emit(emit, -1, "  ");
    for (const Edge& e : d.edges) {
        const char* arrow = e.kind == EdgeKind::RAW ? "-->"
            : e.kind == EdgeKind::WAR ? "-.->" : "==>";
        out += "  n" + std::to_string(e.src) + " " + arrow
            + (e.plane.empty() ? std::string(" ")
                               : "|\"" + detail::escape(e.plane, '"', "#quot;") + "\"| ")
            + "n" + std::to_string(e.dst) + "\n";
    }
    std::string opaque;
    for (const Node& n : d.nodes)
        if (n.opaque)
            opaque += (opaque.empty() ? "" : ",") + std::string("n") + std::to_string(n.id);
    if (!opaque.empty())
        out += "  classDef opaque stroke-dasharray: 4 2\n  class " + opaque + " opaque\n";
    return out;
}

/** @brief Graphviz DOT: frames as clusters, an opaque node dashed, edge style
 *  by kind (RAW solid, WAR dashed, WAW bold) labelled with the plane. */
inline std::string dot(const Description& d)
{
    d.validate();
    const detail::FrameTree tree(d);
    std::string out = "digraph eagle {\n  rankdir=TB;\n  node [shape=box];\n";
    auto node = [&](const Node& n, const std::string& pad) {
        std::string text = detail::dotEscape(n.label);
        if (!n.stage.empty())
            text += "\\n" + n.stage;
        out += pad + "n" + std::to_string(n.id) + " [label=\"" + text + "\""
            + (n.opaque ? ", style=dashed" : "") + "];\n";
    };
    auto emit = [&](auto&& self, gidx frame, const std::string& pad) -> void {
        for (gidx v : tree.frameNodes[frame + 1])
            node(d.nodes[v], pad);
        for (gidx f : tree.childFrames[frame + 1]) {
            out += pad + "subgraph cluster_f" + std::to_string(f) + " {\n" + pad
                + "  label=\"" + detail::dotEscape(d.frames[f].text()) + "\";\n";
            self(self, f, pad + "  ");
            out += pad + "}\n";
        }
    };
    emit(emit, -1, "  ");
    for (const Edge& e : d.edges) {
        const char* style = e.kind == EdgeKind::RAW ? "solid"
            : e.kind == EdgeKind::WAR ? "dashed" : "bold";
        out += "  n" + std::to_string(e.src) + " -> n" + std::to_string(e.dst) + " [";
        if (!e.plane.empty())
            out += "label=\"" + detail::dotEscape(e.plane) + "\", ";
        out += std::string("style=") + style + "];\n";
    }
    out += "}\n";
    return out;
}

}  // namespace graph
}  // namespace eagle
