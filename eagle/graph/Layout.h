// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file graph/Layout.h
 * @brief A deterministic layered layout of a Description: layers are the
 *        longest path from the sources, or, when every node sits in a
 *        ``"stage"`` frame, one row per stage (a stage grows extra rows only
 *        for edges between its own nodes); the order inside a layer
 *        comes from barycenter sweeps with ties broken by node id, and the
 *        coordinates sit on a unit grid. The same input always gives the same
 *        output; nothing here builds, launches or touches a device.
 */

#include "eagle/graph/Description.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace eagle {
namespace graph {

/** @brief ``layer[v]`` and ``order[v]`` (position inside the layer) per node;
 *  ``xy[v] = {order - (width - 1) / 2, layer}``, y growing downwards. */
struct Layout {
    std::vector<gidx> layer;
    std::vector<gidx> order;
    std::vector<std::array<double, 2>> xy;
};

namespace detail {

/** @brief The stage of each node: the rank, among the ``"stage"`` frames, of
 *  the one enclosing it; empty unless every node sits in a stage. */
inline std::vector<gidx> stagesOf(const Description& d)
{
    std::vector<gidx> rank(d.frames.size(), -1);
    gidx count = 0;
    for (const Frame& f : d.frames)
        if (f.kind == "stage")
            rank[f.id] = count++;
    std::vector<gidx> out;
    for (const Node& nd : d.nodes) {
        gidx s = -1;
        for (gidx f = nd.frame; f != -1; f = d.frames[f].parent)
            s = rank[f] >= 0 ? rank[f] : s;
        if (s < 0)
            return {};
        out.push_back(s);
    }
    return out;
}

/** @brief Crossings between adjacent layers for the given positions. Edges
 *  that skip a layer carry no crossing here (no dummy nodes are inserted). */
inline gidx crossings(const std::vector<std::pair<gidx, gidx>>& edges,
    const std::vector<gidx>& layer, const std::vector<gidx>& order)
{
    gidx total = 0;
    for (std::size_t i = 0; i < edges.size(); ++i) {
        const auto [a, b] = edges[i];
        if (layer[b] != layer[a] + 1)
            continue;
        for (std::size_t j = i + 1; j < edges.size(); ++j) {
            const auto [c, d] = edges[j];
            if (layer[c] != layer[a] || layer[d] != layer[b])
                continue;
            if ((order[a] - order[c]) * (order[b] - order[d]) < 0)
                ++total;
        }
    }
    return total;
}

}  // namespace detail

/** @brief Lay out ``d``. ``sweeps`` is the number of down-and-up barycenter
 *  rounds; the ordering with the fewest crossings seen (the initial id order
 *  included) wins, the earliest on a tie.
 *  @throws std::invalid_argument if ``d`` fails ``validate()``. */
inline Layout layout(const Description& d, int sweeps = 4)
{
    d.validate();
    const gidx n = gidx(d.nodes.size());
    std::set<std::pair<gidx, gidx>> unique;
    for (const Edge& e : d.edges)
        unique.emplace(e.src, e.dst);
    const std::vector<std::pair<gidx, gidx>> edges(unique.begin(), unique.end());
    std::vector<std::vector<gidx>> preds(n), succs(n);
    for (auto [a, b] : edges) {
        succs[a].push_back(b);
        preds[b].push_back(a);
    }

    Layout out;
    out.layer.assign(n, 0);
    out.order.assign(n, 0);
    out.xy.assign(n, { 0.0, 0.0 });

    // layers: longest path, in a topological order that takes the lowest stage
    // then the smallest id; a stage starts below the last row of the one before
    std::vector<gidx> stage = detail::stagesOf(d);
    if (stage.empty())
        stage.assign(n, 0);
    for (auto [a, b] : edges)
        if (stage[a] > stage[b])
            throw std::invalid_argument("eagle::graph::layout: edge " + std::to_string(a)
                + " -> " + std::to_string(b) + " runs back to an earlier stage");
    std::vector<gidx> indeg(n, 0);
    for (gidx v = 0; v < n; ++v)
        indeg[v] = gidx(preds[v].size());
    std::set<std::pair<gidx, gidx>> ready;
    for (gidx v = 0; v < n; ++v)
        if (indeg[v] == 0)
            ready.emplace(stage[v], v);
    gidx current = 0, base = 0, deepest = -1;
    while (!ready.empty()) {
        const gidx v = ready.begin()->second;
        ready.erase(ready.begin());
        if (stage[v] != current) {
            current = stage[v];
            base = deepest + 1;
        }
        out.layer[v] = std::max(out.layer[v], base);
        deepest = std::max(deepest, out.layer[v]);
        for (gidx w : succs[v]) {
            out.layer[w] = std::max(out.layer[w], out.layer[v] + 1);
            if (--indeg[w] == 0)
                ready.emplace(stage[w], w);
        }
    }
    gidx layers = 0;
    for (gidx v = 0; v < n; ++v)
        layers = std::max(layers, out.layer[v] + 1);

    std::vector<std::vector<gidx>> rows(layers);
    for (gidx v = 0; v < n; ++v)
        rows[out.layer[v]].push_back(v);
    auto stamp = [&](std::vector<gidx>& order) {
        for (const auto& row : rows)
            for (gidx i = 0; i < gidx(row.size()); ++i)
                order[row[i]] = i;
    };
    stamp(out.order);

    std::vector<gidx> best = out.order;
    gidx bestCross = detail::crossings(edges, out.layer, out.order);

    auto reorder = [&](gidx l, const std::vector<std::vector<gidx>>& nbr) {
        auto& row = rows[l];
        std::vector<std::pair<double, gidx>> key;
        for (gidx v : row) {
            double bary = double(out.order[v]);
            double sum = 0;
            gidx cnt = 0;
            for (gidx w : nbr[v])
                if (out.layer[w] == (&nbr == &preds ? l - 1 : l + 1)) {
                    sum += double(out.order[w]);
                    ++cnt;
                }
            if (cnt > 0)
                bary = sum / double(cnt);
            key.emplace_back(bary, v);
        }
        std::stable_sort(key.begin(), key.end(), [&](const auto& x, const auto& y) {
            if (x.first != y.first)
                return x.first < y.first;
            return x.second < y.second;
        });
        for (gidx i = 0; i < gidx(key.size()); ++i) {
            row[i] = key[i].second;
            out.order[row[i]] = i;
        }
    };
    auto consider = [&]() {
        const gidx c = detail::crossings(edges, out.layer, out.order);
        if (c < bestCross) {
            bestCross = c;
            best = out.order;
        }
    };
    for (int s = 0; s < sweeps && bestCross > 0; ++s) {
        for (gidx l = 1; l < layers; ++l)
            reorder(l, preds);
        consider();
        for (gidx l = layers - 2; l >= 0; --l)
            reorder(l, succs);
        consider();
    }
    out.order = best;
    for (gidx v = 0; v < n; ++v) {
        const double width = double(rows[out.layer[v]].size());
        out.xy[v] = { double(out.order[v]) - (width - 1.0) / 2.0, double(out.layer[v]) };
    }
    return out;
}

}  // namespace graph
}  // namespace eagle
