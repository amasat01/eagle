// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file graph/Description.h
 * @brief The generic description every eagle graph can be rendered from:
 *        nodes, hazard edges and conditional frames. Plain data, no CUDA.
 */

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace eagle {
namespace graph {

/** @brief Signed index of nodes, frames and edges (``-1`` = none). */
using gidx = std::int64_t;

/** @brief The hazard an edge carries: read-after-write, write-after-read or
 *  write-after-write. Only ``RAW`` carries data (adjoint) flow. */
enum class EdgeKind { RAW, WAR, WAW };

/** @brief ``"RAW"``, ``"WAR"`` or ``"WAW"``. */
inline const char* kindName(EdgeKind k)
{
    return k == EdgeKind::RAW ? "RAW" : k == EdgeKind::WAR ? "WAR" : "WAW";
}

/** @brief What a hazard means, in plain words: RAW "data flows", WAR "must wait
 *  to overwrite", WAW "same output". */
inline const char* kindMeaning(EdgeKind k)
{
    return k == EdgeKind::RAW ? "data flows" : k == EdgeKind::WAR ? "must wait to overwrite" : "same output";
}

/** @brief The inverse of @c kindName.
 *  @throws std::invalid_argument for any other text. */
inline EdgeKind parseEdgeKind(const std::string& s)
{
    for (EdgeKind k : { EdgeKind::RAW, EdgeKind::WAR, EdgeKind::WAW })
        if (s == kindName(k))
            return k;
    throw std::invalid_argument("edge kind must be 'RAW', 'WAR' or 'WAW', got '" + s + "'");
}

/** @brief A region wrapped around nodes: a skippable (``"if"``), a loop
 *  (``"while"``) or a plain stage band (``"stage"``). ``parent`` is the
 *  enclosing frame's id, ``-1`` at top level. */
struct Frame {
    gidx id = 0;
    std::string kind;
    std::string label;
    gidx parent = -1;

    /** @brief The text drawn on the frame: its label, else its kind. */
    const std::string& text() const { return label.empty() ? kind : label; }
};

/** @brief One executable node. ``stage`` is the schedule it runs under
 *  (``"work_share"``, ``"branch_tasks"``, ``"capture"``) or empty before a
 *  build; ``frame`` the id of the frame it sits in or ``-1``; ``opaque``
 *  marks a node whose planes were declared by the caller. */
struct Node {
    gidx id = 0;
    std::string label;
    std::string kind;
    std::string stage;
    gidx frame = -1;
    bool opaque = false;
};

/** @brief ``src`` must finish before ``dst`` starts. */
struct Edge {
    gidx src = 0;
    gidx dst = 0;
    std::string plane;
    EdgeKind kind = EdgeKind::RAW;
};

/** @brief Nodes, edges and frames. Ids equal positions. */
struct Description {
    std::vector<Node> nodes;
    std::vector<Edge> edges;
    std::vector<Frame> frames;

    /** @brief Check the description is well formed.
     *  @throws std::invalid_argument naming the first defect: an id that is
     *          not its position, an edge or node frame pointing nowhere, a
     *          self edge, a frame cycle, or an edge cycle. */
    void validate() const
    {
        const gidx n = gidx(nodes.size());
        const gidx f = gidx(frames.size());
        for (gidx i = 0; i < n; ++i)
            if (nodes[i].id != i)
                bad("node at position " + std::to_string(i) + " has id "
                    + std::to_string(nodes[i].id));
        for (gidx i = 0; i < f; ++i) {
            if (frames[i].id != i)
                bad("frame at position " + std::to_string(i) + " has id "
                    + std::to_string(frames[i].id));
            const gidx p = frames[i].parent;
            if (p < -1 || p >= f)
                bad("frame " + std::to_string(i) + " has parent "
                    + std::to_string(p) + " which does not exist");
        }
        for (gidx i = 0; i < f; ++i) {          // frame parents must not cycle
            gidx cur = i;
            for (gidx hop = 0; cur != -1; ++hop) {
                if (hop > f)
                    bad("frame " + std::to_string(i) + " is in a parent cycle");
                cur = frames[cur].parent;
            }
        }
        for (const Node& nd : nodes)
            if (nd.frame < -1 || nd.frame >= f)
                bad("node " + std::to_string(nd.id) + " is in frame "
                    + std::to_string(nd.frame) + " which does not exist");
        std::vector<gidx> indeg(n, 0);
        std::vector<std::vector<gidx>> out(n);
        for (const Edge& e : edges) {
            if (e.src < 0 || e.src >= n || e.dst < 0 || e.dst >= n)
                bad("edge " + std::to_string(e.src) + " -> "
                    + std::to_string(e.dst) + " is dangling (" + std::to_string(n)
                    + " nodes)");
            if (e.src == e.dst)
                bad("edge " + std::to_string(e.src) + " -> "
                    + std::to_string(e.dst) + " is a self edge");
            out[e.src].push_back(e.dst);
            ++indeg[e.dst];
        }
        std::vector<gidx> ready;
        for (gidx i = 0; i < n; ++i)
            if (indeg[i] == 0)
                ready.push_back(i);
        gidx seen = 0;
        while (!ready.empty()) {
            const gidx v = ready.back();
            ready.pop_back();
            ++seen;
            for (gidx w : out[v])
                if (--indeg[w] == 0)
                    ready.push_back(w);
        }
        if (seen != n)
            bad("the edges contain a cycle");
    }

private:
    static void bad(const std::string& what)
    {
        throw std::invalid_argument("eagle::graph::Description: " + what);
    }
};

}  // namespace graph
}  // namespace eagle
