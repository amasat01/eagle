// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file diff/Transpose.h
 * @brief The reverse of a static graph: which derivative step runs when, where
 *        each step's incoming adjoint comes from, which adjoints are summed,
 *        and which forward planes must be saved before they are overwritten.
 *        Plain data and host loops, no CUDA.
 *
 * The forward graph runs its nodes in id order. Its reverse runs one vjp step
 * per forward node in the opposite order. A plane read by several nodes
 * receives one adjoint from each of them; those are summed elementwise in a
 * FIXED order (the caller's seed first, then the readers in the order they
 * were added), never with atomics, so the sum is the same on every run and
 * every thread count.
 *
 * A vjp recomputes from the values its node READ. When a later node (or the
 * node itself, in place) overwrites such a plane, the forward value is gone by
 * the time the reverse runs, so the plane is copied once just before the
 * first node that overwrites it. Nothing else is kept: every other plane the
 * reverse reads is still the forward's own.
 */

#include "eagle/graph/Description.h"

#include <algorithm>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace eagle {
namespace diff {

using graph::gidx;

/** @brief What the reverse needs to know about one forward node: the planes
 *  its vjp reads (the node's inputs) and the planes it writes (whose adjoints
 *  flow into its vjp). Plane names are keys that are unique in the graph: two
 *  nodes touching the same memory use the same key. */
struct NodeVjp {
    gidx node = 0;
    std::vector<std::string> reads;
    std::vector<std::string> writes;
};

/** @brief Copy forward plane @c plane just before forward node @c node runs. */
struct Snapshot {
    gidx node = 0;
    std::string plane;
};

/** @brief Reverse node @c target sums the adjoints of @c plane produced by
 *  @c sources, elementwise, in exactly this order. */
struct Join {
    gidx target = 0;
    std::string plane;
    std::vector<gidx> sources;
};

/** @brief An adjoint source that is the caller's seed for the plane. */
inline constexpr gidx kSeed = -1;
/** @brief An adjoint source that is zero: nothing flows into the plane. */
inline constexpr gidx kZero = -2;
/** @brief A read that comes from the forward's own plane, not a snapshot. */
inline constexpr gidx kLive = -1;

/** @brief The reverse of a graph. Reverse nodes run in id order. */
struct ReversePlan {
    /** @brief vjp steps (kind "vjp", or "broadcast" for a reduce) and summing
     *  joins (kind "sum"); RAW edges carry adjoint planes. */
    graph::Description reversed;
    /** @brief Per reverse node, the forward node it reverses; -1 for a join. */
    std::vector<gidx> forward;
    std::vector<Join> joins;
    std::vector<Snapshot> snapshots;
    /** @brief Per reverse node, per @c NodeVjp::writes entry: the reverse node
     *  whose adjoint it consumes, @c kSeed or @c kZero. Empty for a join. */
    std::vector<std::vector<gidx>> seedOf;
    /** @brief Per reverse node, per @c NodeVjp::reads entry: the index of the
     *  snapshot holding the value the node read, or @c kLive. */
    std::vector<std::vector<gidx>> readOf;
    /** @brief Per graph input (a plane read before anything writes it), in
     *  first-use order: where its gradient comes from (a reverse node, or
     *  @c kZero). */
    std::vector<std::pair<std::string, gidx>> gradients;
};

namespace detail {

[[noreturn]] inline void refuse(const graph::Node& n, const std::string& why)
{
    throw std::invalid_argument("node '" + n.label + "' " + why);
}

inline bool contains(const std::vector<std::string>& v, const std::string& s)
{
    return std::find(v.begin(), v.end(), s) != v.end();
}

/** @brief The NodeVjp of every node, indexed by node id; refuses a node
 *  without one, or one listed twice. */
inline std::vector<const NodeVjp*> byNode(gidx n, const std::vector<NodeVjp>& vjps)
{
    std::vector<const NodeVjp*> out(static_cast<std::size_t>(n), nullptr);
    for (const NodeVjp& v : vjps) {
        if (v.node < 0 || v.node >= n)
            throw std::invalid_argument("a vjp names node " + std::to_string(v.node)
                                        + ", which is not in the graph");
        if (out[v.node])
            throw std::invalid_argument("node " + std::to_string(v.node) + " has two vjps");
        out[v.node] = &v;
    }
    return out;
}

}  // namespace detail

/** @brief The copies the reverse needs: for each plane a node reads, the first
 *  node at or after it that writes the plane (itself, when in place), deduped.
 *  @param vjps one entry per forward node, in node order. */
inline std::vector<Snapshot> snapshots(const std::vector<NodeVjp>& vjps)
{
    std::vector<Snapshot> out;
    for (std::size_t i = 0; i < vjps.size(); ++i)
        for (const std::string& p : vjps[i].reads)
            for (std::size_t m = i; m < vjps.size(); ++m)
                if (detail::contains(vjps[m].writes, p)) {
                    const Snapshot s { vjps[m].node, p };
                    if (std::none_of(out.begin(), out.end(), [&](const Snapshot& o) {
                            return o.node == s.node && o.plane == s.plane;
                        }))
                        out.push_back(s);
                    break;
                }
    return out;
}

/** @brief Reverse @p forward, given each node's vjp planes and the planes the
 *  caller seeds (the gradient of the objective with respect to their final
 *  value); an unseeded plane contributes nothing of its own.
 *  @throws std::invalid_argument for a graph that is already a reverse (second
 *          order), an opaque node, or a node with
 *          no entry in @p vjps; the message names the node. */
inline ReversePlan transpose(const graph::Description& forward, const std::vector<NodeVjp>& vjps,
                             const std::vector<std::string>& seeded)
{
    forward.validate();
    const gidx n = gidx(forward.nodes.size());
    for (const graph::Node& nd : forward.nodes) {
        if (nd.kind == "vjp" || nd.kind == "sum" || nd.kind == "broadcast")
            detail::refuse(nd, "is a step of a reverse pass; reversing a reverse "
                               "(second order) is not supported");
        if (nd.opaque)
            detail::refuse(nd, "is opaque: eagle does not know its derivative; move it "
                               "out of the recorded graph, or give its node kind a transpose");
    }
    const std::vector<const NodeVjp*> of = detail::byNode(n, vjps);
    std::vector<NodeVjp> ordered;
    for (gidx i = 0; i < n; ++i) {
        if (!of[i])
            detail::refuse(forward.nodes[i], "has no vjp");
        ordered.push_back(*of[i]);
    }

    ReversePlan plan;
    plan.snapshots = snapshots(ordered);
    auto snapshotOf = [&](gidx node, const std::string& p) {
        for (gidx m = node; m < n; ++m)
            if (detail::contains(ordered[m].writes, p))
                for (std::size_t s = 0; s < plan.snapshots.size(); ++s)
                    if (plan.snapshots[s].node == m && plan.snapshots[s].plane == p)
                        return gidx(s);
        return kLive;
    };

    // The adjoint each plane has accumulated so far, walking the forward nodes
    // backwards. A plane's final value is what the caller seeds.
    std::map<std::string, std::vector<gidx>> pending;
    for (const std::string& p : seeded) {
        if (std::none_of(ordered.begin(), ordered.end(),
                [&](const NodeVjp& v) { return detail::contains(v.writes, p); }))
            throw std::invalid_argument("seeded plane '" + p + "' is not written by the graph");
        pending[p] = { kSeed };
    }

    auto addNode = [&](std::string label, std::string kind, gidx fwd) {
        const gidx id = gidx(plan.reversed.nodes.size());
        plan.reversed.nodes.push_back({ id, std::move(label), std::move(kind), "", -1, false });
        plan.forward.push_back(fwd);
        plan.seedOf.emplace_back();
        plan.readOf.emplace_back();
        return id;
    };
    auto link = [&](gidx src, gidx dst, const std::string& p) {
        if (src >= 0)
            plan.reversed.edges.push_back({ src, dst, p, graph::EdgeKind::RAW });
    };
    // One source for plane p: the single contributor, a new join, or zero.
    auto collect = [&](const std::string& p) {
        std::vector<gidx> src = std::move(pending[p]);
        pending[p].clear();
        if (src.empty())
            return kZero;
        if (src.size() == 1)
            return src.front();
        // the seed first, then the readers in the order they were added
        std::sort(src.begin(), src.end(), [&](gidx a, gidx b) {
            const gidx fa = a < 0 ? -1 : plan.forward[a];
            const gidx fb = b < 0 ? -1 : plan.forward[b];
            return fa < fb;
        });
        const gidx j = addNode("sum(" + p + ")", "sum", -1);
        for (gidx s : src)
            link(s, j, p);
        plan.joins.push_back({ j, p, std::move(src) });
        return j;
    };

    for (gidx i = n - 1; i >= 0; --i) {
        const graph::Node& nd = forward.nodes[i];
        const NodeVjp& v = ordered[i];
        std::vector<gidx> seeds;
        for (const std::string& p : v.writes)
            seeds.push_back(collect(p));
        const bool reduce = nd.kind == "reduce";
        const gidx r = addNode((reduce ? "broadcast(" : "vjp(") + nd.label + ")",
                               reduce ? "broadcast" : "vjp", i);
        for (std::size_t k = 0; k < seeds.size(); ++k)
            link(seeds[k], r, v.writes[k]);
        plan.seedOf[r] = std::move(seeds);
        for (const std::string& p : v.reads) {
            plan.readOf[r].push_back(snapshotOf(i, p));
            pending[p].push_back(r);
        }
    }

    // Graph inputs: planes whose first use is a read.
    std::vector<std::string> seen;
    for (const NodeVjp& v : ordered) {
        for (const std::string& p : v.reads)
            if (!detail::contains(seen, p)) {
                seen.push_back(p);
                plan.gradients.emplace_back(p, collect(p));
            }
        for (const std::string& p : v.writes)
            if (!detail::contains(seen, p))
                seen.push_back(p);
    }
    plan.reversed.validate();
    return plan;
}

/** @brief @p dst = ((src[0] + src[1]) + src[2]) + ... elementwise, over @p n
 *  elements, in exactly the order of @p src: the same bits for every thread
 *  count. @p dst may alias @p src[0]. */
template <class T>
inline void orderedSum(T* dst, const std::vector<const T*>& src, std::ptrdiff_t n)
{
    if (src.empty())
        throw std::invalid_argument("orderedSum: no sources");
#pragma omp parallel for schedule(static)
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        T acc = src[0][i];
        for (std::size_t k = 1; k < src.size(); ++k)
            acc += src[k][i];
        dst[i] = acc;
    }
}

}  // namespace diff
}  // namespace eagle
