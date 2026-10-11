// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file graph/Describe.h
 * @brief Adapters from eagle's native graphs to a Description, so any of them
 *        prints and draws with the same code: ``describe(cpu::Graph)``,
 *        ``describe(cuda::Graph)`` (CUDA builds), ``describe(GraphComposer)``
 *        and ``describe(ConditionalGroup, body)``.
 *
 * A native node carries no plane names, so a native graph's nodes are labelled
 * ``native<i>`` and its edges are completion edges (``RAW``, no plane).
 */

#include "eagle/compose.h"
#include "eagle/conditional.h"
#include "eagle/cpu/Graph.h"
#include "eagle/cuda/Graph.h"
#include "eagle/graph/Description.h"

#include <string>
#include <vector>

namespace eagle {
namespace graph {

namespace detail {

inline Description fromDeps(const std::vector<std::vector<idx_t>>& deps)
{
    Description d;
    for (gidx i = 0; i < gidx(deps.size()); ++i) {
        d.nodes.push_back({ i, "native" + std::to_string(i), "native", "", -1, false });
        for (eagle::idx_t dep : deps[i])
            d.edges.push_back({ gidx(dep), i, "", EdgeKind::RAW });
    }
    return d;
}

}  // namespace detail

/** @brief The host graph: one node per native node, an edge per dependency. */
inline Description describe(const cpu::Graph& g)
{
    return detail::fromDeps(g.nativeDeps());
}

#ifndef EAGLE_CPU_ONLY
/** @brief The CUDA graph: one node per native node, an edge per dependency. */
inline Description describe(const cuda::Graph& g)
{
    return detail::fromDeps(g.nativeDeps());
}
#endif

/** @brief A composer: one node per member (its name, else ``member<i>``).
 *  Under ``mode="sequenced"`` members replay in registration order, so they
 *  form a chain; the flat modes (``enabled``, ``rebuild``) run members as
 *  siblings and carry no edges. */
inline Description describe(const compose::GraphComposer& c)
{
    Description d;
    const auto names = c.memberNames();
    for (gidx i = 0; i < gidx(names.size()); ++i) {
        d.nodes.push_back({ i, names[i].empty() ? "member" + std::to_string(i) : names[i],
            "member", "", -1, false });
        if (i > 0 && c.mode() == "sequenced")
            d.edges.push_back({ i - 1, i, "", EdgeKind::RAW });
    }
    return d;
}

/** @brief A conditional group around ``body`` (its description): every
 *  top-level node of ``body`` moves into one new ``"if"`` frame. */
inline Description describe(const conditional::ConditionalGroup&, Description body)
{
    const gidx frame = gidx(body.frames.size());
    for (Frame& f : body.frames)
        if (f.parent == -1)
            f.parent = frame;
    for (Node& n : body.nodes)
        if (n.frame == -1)
            n.frame = frame;
    // the new frame is appended last, so shift: ids must equal positions and
    // a child may precede its parent, which validate() allows
    body.frames.push_back({ frame, "if", "skip if guard", -1 });
    return body;
}

}  // namespace graph
}  // namespace eagle
