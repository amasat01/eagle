// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file graph/Edges.h
 * @brief Hazard edges derived from the arrays two nodes touch. Never
 *        declared by the caller. Stride-aware: an array is described by its
 *        base address, shape, byte strides and item size, and two arrays
 *        overlap when they share a byte. The test is exact for strided views
 *        under a work cap (so column views of one matrix do not overlap) and
 *        conservative past it: a doubt reports an overlap, which can only add
 *        an edge or a conflict, never miss one.
 */

#include "eagle/graph/Description.h"

#include <algorithm>
#include <numeric>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace eagle {
namespace graph {

/** @brief One plane a node touches. ``access``: 0 read, 1 write, 2 in place
 *  (read and write). ``strides`` are in bytes and may be negative. */
struct PlaneRef {
    std::string name;
    gidx node = 0;
    std::uint64_t base = 0;
    std::vector<gidx> shape;
    std::vector<gidx> strides;
    gidx itemsize = 0;
    int access = 0;
};

/** @brief The branches of one fork/join group: ``branches[b]`` lists the
 *  nodes of branch ``b`` (run in order inside the branch). Nodes of different
 *  branches are siblings; they may run concurrently. */
struct SiblingGroup {
    std::vector<std::vector<gidx>> branches;
};

/** @brief Two sibling nodes touching overlapping memory, at least one of them
 *  writing. ``kind`` is the hazard from ``a`` (earlier) to ``b``. */
struct Conflict {
    gidx a = 0;
    gidx b = 0;
    std::string planeA;
    std::string planeB;
    EdgeKind kind = EdgeKind::RAW;
};

/** @brief Work cap of the exact overlap search (search steps). */
inline constexpr std::int64_t kOverlapMaxWork = 100000;

namespace detail {

inline bool reads(const PlaneRef& r) { return r.access == 0 || r.access == 2; }
inline bool writes(const PlaneRef& r) { return r.access == 1 || r.access == 2; }

inline bool empty(const PlaneRef& r)
{
    for (gidx s : r.shape)
        if (s == 0)
            return true;
    return r.itemsize <= 0;
}

/** @brief ``[lo, hi)`` bytes an array can reach. */
inline std::pair<std::int64_t, std::int64_t> extent(const PlaneRef& r)
{
    std::int64_t lo = 0, hi = 0;
    for (std::size_t k = 0; k < r.shape.size(); ++k) {
        const std::int64_t reach = std::int64_t(r.shape[k] - 1) * r.strides[k];
        (reach < 0 ? lo : hi) += reach;
    }
    const auto base = std::int64_t(r.base);
    return { base + lo, base + hi + r.itemsize };
}

struct Term {
    std::int64_t coeff;
    std::int64_t count;     // t in [0, count - 1]
};

inline std::int64_t floorDiv(std::int64_t a, std::int64_t b)
{
    std::int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0)))
        --q;
    return q;
}

inline std::int64_t ceilDiv(std::int64_t a, std::int64_t b)
{
    return -floorDiv(-a, b);
}

/** @brief Is there t_k in [0,count-1] with sum coeff_k t_k in [lo, hi]? The
 *  term with the most steps is solved in closed form, the rest searched. */
inline bool solvable(std::vector<Term> terms, std::int64_t lo, std::int64_t hi,
    std::int64_t& work, bool& gaveUp)
{
    terms.erase(std::remove_if(terms.begin(), terms.end(),
                    [](const Term& t) { return t.count <= 1 || t.coeff == 0; }),
        terms.end());
    if (terms.empty())
        return lo <= 0 && 0 <= hi;
    // every reachable sum is a multiple of the gcd of the coefficients: no
    // multiple inside [lo, hi] means no overlap, exactly, whatever the counts
    std::int64_t g = 0;
    for (const Term& t : terms)
        g = std::gcd(g, t.coeff < 0 ? -t.coeff : t.coeff);
    if (ceilDiv(lo, g) * g > hi)
        return false;
    std::sort(terms.begin(), terms.end(),
        [](const Term& x, const Term& y) { return x.count < y.count; });
    const Term last = terms.back();
    terms.pop_back();
    // min/max of what the searched terms can still add, from each depth
    const std::size_t m = terms.size();
    std::vector<std::int64_t> mn(m + 1, 0), mx(m + 1, 0);
    for (std::size_t i = m; i-- > 0;) {
        const std::int64_t reach = terms[i].coeff * (terms[i].count - 1);
        mn[i] = mn[i + 1] + std::min<std::int64_t>(0, reach);
        mx[i] = mx[i + 1] + std::max<std::int64_t>(0, reach);
    }
    struct Search {
        const std::vector<Term>& terms;
        const std::vector<std::int64_t>& mn;
        const std::vector<std::int64_t>& mx;
        Term last;
        std::int64_t lo, hi;
        std::int64_t& work;
        bool& gaveUp;
        bool go(std::size_t depth, std::int64_t acc)
        {
            if (++work > kOverlapMaxWork) {
                gaveUp = true;
                return true;               // conservative: report an overlap
            }
            // can the rest (searched terms + last) still reach [lo, hi]?
            const std::int64_t lreach = last.coeff * (last.count - 1);
            const std::int64_t rmin = acc + mn[depth] + std::min<std::int64_t>(0, lreach);
            const std::int64_t rmax = acc + mx[depth] + std::max<std::int64_t>(0, lreach);
            if (rmax < lo || rmin > hi)
                return false;
            if (depth == terms.size()) {
                // acc + last.coeff * t in [lo, hi], t in [0, count-1]
                std::int64_t tlo, thi;
                if (last.coeff > 0) {
                    tlo = ceilDiv(lo - acc, last.coeff);
                    thi = floorDiv(hi - acc, last.coeff);
                } else {
                    tlo = ceilDiv(hi - acc, last.coeff);
                    thi = floorDiv(lo - acc, last.coeff);
                }
                return std::max<std::int64_t>(tlo, 0)
                    <= std::min<std::int64_t>(thi, last.count - 1);
            }
            const Term& t = terms[depth];
            for (std::int64_t i = 0; i < t.count; ++i)
                if (go(depth + 1, acc + t.coeff * i))
                    return true;
            return false;
        }
    } search { terms, mn, mx, last, lo, hi, work, gaveUp };
    return search.go(0, 0);
}

/** @brief Do the two arrays share a byte? */
inline bool overlap(const PlaneRef& a, const PlaneRef& b)
{
    if (empty(a) || empty(b))
        return false;
    const auto ea = extent(a), eb = extent(b);
    if (!(ea.first < eb.second && eb.first < ea.second))
        return false;
    // a byte range [x, x+ia) meets [y, y+ib) iff -ia < x - y < ib, with
    // x = baseA + sum(sA tA), y = baseB + sum(sB tB).
    const std::int64_t d = std::int64_t(b.base) - std::int64_t(a.base);
    std::vector<Term> terms;
    for (std::size_t k = 0; k < a.shape.size(); ++k)
        terms.push_back({ a.strides[k], a.shape[k] });
    for (std::size_t k = 0; k < b.shape.size(); ++k)
        terms.push_back({ -b.strides[k], b.shape[k] });
    std::int64_t work = 0;
    bool gaveUp = false;
    // sum(coeff t) = x - y - (baseA - baseB) ... shift: x - y in (-ia, ib)
    // <=> sum in (d - ia, d + ib) <=> [d - ia + 1, d + ib - 1]
    return solvable(terms, d - a.itemsize + 1, d + b.itemsize - 1, work, gaveUp);
}

inline std::vector<std::vector<const PlaneRef*>> byNode(
    const std::vector<PlaneRef>& refs, gidx& n)
{
    n = 0;
    for (const PlaneRef& r : refs)
        n = std::max<gidx>(n, r.node + 1);
    std::vector<std::vector<const PlaneRef*>> out(n);
    for (const PlaneRef& r : refs)
        out[r.node].push_back(&r);
    return out;
}

/** @brief The hazards from node ``a`` (earlier) to ``b``, in the fixed order:
 *  writes of a vs reads of b (RAW), writes vs writes (WAW), reads of a vs
 *  writes of b (WAR). Calls ``f(planeNameOfA, planeNameOfB, kind)``. */
template<typename F>
void hazards(const std::vector<const PlaneRef*>& a,
    const std::vector<const PlaneRef*>& b, F&& f)
{
    for (const PlaneRef* wa : a) {
        if (!writes(*wa))
            continue;
        for (const PlaneRef* rb : b)
            if (reads(*rb) && overlap(*wa, *rb))
                f(wa->name, rb->name, EdgeKind::RAW);
        for (const PlaneRef* wb : b)
            if (writes(*wb) && overlap(*wa, *wb))
                f(wa->name, wb->name, EdgeKind::WAW);
    }
    for (const PlaneRef* ra : a) {
        if (!reads(*ra))
            continue;
        for (const PlaneRef* wb : b)
            if (writes(*wb) && overlap(*ra, *wb))
                f(ra->name, wb->name, EdgeKind::WAR);
    }
}

/** @brief ``branchOf[node]``: for each group, the branch index of the node
 *  (``-1`` if the node is not in it). */
inline std::vector<std::vector<gidx>> branchTable(
    gidx n, const std::vector<SiblingGroup>& groups)
{
    std::vector<std::vector<gidx>> table(groups.size(), std::vector<gidx>(n, -1));
    for (std::size_t g = 0; g < groups.size(); ++g)
        for (std::size_t b = 0; b < groups[g].branches.size(); ++b)
            for (gidx node : groups[g].branches[b])
                if (node >= 0 && node < n)
                    table[g][node] = gidx(b);
    return table;
}

inline bool siblings(const std::vector<std::vector<gidx>>& table, gidx a, gidx b)
{
    for (const auto& row : table)
        if (row[a] != -1 && row[b] != -1 && row[a] != row[b])
            return true;
    return false;
}

}  // namespace detail

/** @brief Every hazard edge between nodes, in add order (ascending source,
 *  then destination), siblings of one group excluded (their overlap is a
 *  conflict, see @ref siblingConflicts). A hazard on the same plane name
 *  between the same two nodes collapses to one edge per kind. */
inline std::vector<Edge> deriveEdges(const std::vector<PlaneRef>& refs,
    const std::vector<SiblingGroup>& groups = {})
{
    gidx n = 0;
    const auto nodes = detail::byNode(refs, n);
    const auto table = detail::branchTable(n, groups);
    std::vector<Edge> out;
    std::set<std::tuple<gidx, gidx, std::string, int>> seen;
    for (gidx i = 0; i < n; ++i)
        for (gidx j = i + 1; j < n; ++j) {
            if (detail::siblings(table, i, j))
                continue;
            detail::hazards(nodes[i], nodes[j],
                [&](const std::string& plane, const std::string&, EdgeKind kind) {
                    if (seen.emplace(i, j, plane, int(kind)).second)
                        out.push_back({ i, j, plane, kind });
                });
        }
    return out;
}

/** @brief Overlaps between siblings of one group, at least one side writing:
 *  every one is a race, not an ordering. */
inline std::vector<Conflict> siblingConflicts(const std::vector<PlaneRef>& refs,
    const std::vector<SiblingGroup>& groups)
{
    gidx n = 0;
    const auto nodes = detail::byNode(refs, n);
    const auto table = detail::branchTable(n, groups);
    std::vector<Conflict> out;
    for (gidx i = 0; i < n; ++i)
        for (gidx j = i + 1; j < n; ++j) {
            if (!detail::siblings(table, i, j))
                continue;
            detail::hazards(nodes[i], nodes[j],
                [&](const std::string& pa, const std::string& pb, EdgeKind kind) {
                    out.push_back({ i, j, pa, pb, kind });
                });
        }
    return out;
}

/** @brief The flat form: each inner list is a set of nodes that each sit in
 *  a branch of their own. */
inline std::vector<Conflict> siblingConflicts(const std::vector<PlaneRef>& refs,
    const std::vector<std::vector<gidx>>& groups)
{
    std::vector<SiblingGroup> g;
    for (const auto& nodes : groups) {
        SiblingGroup sg;
        for (gidx node : nodes)
            sg.branches.push_back({ node });
        g.push_back(std::move(sg));
    }
    return siblingConflicts(refs, g);
}

}  // namespace graph
}  // namespace eagle
