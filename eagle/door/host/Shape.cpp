// SPDX-License-Identifier: Apache-2.0

// See host/Shape.h.
#include "eagle/door/host/Shape.h"

#include <algorithm>
#include <cmath>
#include <functional>

namespace eagle {
namespace door {

namespace {

/** @brief @p p as a hazard reference: its memory (base, shape, byte strides) and how it is touched; ``name`` carries @p use. */
graph::PlaneRef plane_ref(const eg_plane_s& p, int use, int access) {
    const DLTensor& t = p.tensor->dl_tensor;
    graph::PlaneRef r;
    r.name = std::to_string(use);
    r.base = reinterpret_cast<std::uintptr_t>(t.data) + t.byte_offset;
    r.itemsize = static_cast<graph::gidx>(t.dtype.bits / 8) * t.dtype.lanes;
    r.access = access;
    graph::gidx stride = r.itemsize;            // a plane without strides is compact, row-major
    r.shape.assign(t.shape, t.shape + t.ndim);
    r.strides.assign(static_cast<std::size_t>(t.ndim), 0);
    for (int k = t.ndim; k-- > 0;) {
        r.strides[static_cast<std::size_t>(k)] = t.strides != nullptr ? t.strides[k] * r.itemsize : stride;
        stride *= t.shape[k];
    }
    return r;
}

/** @brief The hazards from node @p a (earlier) to @p b: f(use of a, use of b, kind). */
template <class F>
void hazards(const std::vector<graph::PlaneRef>& a, const std::vector<graph::PlaneRef>& b, F&& f) {
    std::vector<const graph::PlaneRef*> pa, pb;
    for (const graph::PlaneRef& r : a) pa.push_back(&r);
    for (const graph::PlaneRef& r : b) pb.push_back(&r);
    graph::detail::hazards(pa, pb, [&](const std::string& ua, const std::string& ub, graph::EdgeKind kind) { f(std::stoi(ua), std::stoi(ub), kind); });
}

int edge_kind(graph::EdgeKind k) { return k == graph::EdgeKind::RAW ? EG_GEDGE_RAW : k == graph::EdgeKind::WAR ? EG_GEDGE_WAR : EG_GEDGE_WAW; }

}  // namespace

int GraphShape::push(eg_gnode_kind kind, int parent, int arm, std::string label, std::vector<PlaneUse> uses, std::string unit) {
    Node n;
    n.kind = kind;
    n.parent = parent;
    n.arm = arm;
    n.label = std::move(label);
    n.unit = std::move(unit);
    for (std::size_t u = 0; u < uses.size(); ++u) n.refs.push_back(plane_ref(*uses[u].plane, static_cast<int>(u), uses[u].access));
    n.uses = std::move(uses);
    nodes_.push_back(std::move(n));
    return static_cast<int>(nodes_.size()) - 1;
}

bool GraphShape::within(int node, int ancestor) const {
    for (int p = node; p >= 0; p = nodes_[static_cast<std::size_t>(p)].parent)
        if (p == ancestor) return true;
    return false;
}

/** @brief Whether @p a and @p b sit in different ``through`` children (BRANCH or ARM) of one container: they never order each other by a hazard. */
bool GraphShape::apart(int a, int b, eg_gnode_kind through) const {
    for (int x = a; x >= 0; x = nodes_[static_cast<std::size_t>(x)].parent) {
        if (nodes_[static_cast<std::size_t>(x)].kind != through) continue;
        for (int y = b; y >= 0; y = nodes_[static_cast<std::size_t>(y)].parent)
            if (nodes_[static_cast<std::size_t>(y)].kind == through && y != x && nodes_[static_cast<std::size_t>(y)].parent == nodes_[static_cast<std::size_t>(x)].parent) return true;
    }
    return false;
}

int GraphShape::add(eg_gnode_kind kind, std::string label, std::string unit, std::vector<PlaneUse> uses, NodeFacts facts) {
    const int ordinal = push(kind, container(), -1, std::move(label), std::move(uses), std::move(unit));
    Node& n = nodes_[static_cast<std::size_t>(ordinal)];
    n.facts = facts;
    if (kind == EG_GNODE_LAUNCH) n.launch = launches_++;
    else if (kind == EG_GNODE_RUN || kind == EG_GNODE_COPY || kind == EG_GNODE_REDUCE || kind == EG_GNODE_FFT || kind == EG_GNODE_GEMM) n.other = others_++;
    return ordinal;
}

GraphShape::Frame& GraphShape::innermost(bool fork, const char* call) {
    if (frames_.empty()) refuse(EG_BAD_REQUEST, std::string(call) + (fork ? ": no fork is open" : ": no conditional is open"));
    if (frames_.back().fork != fork)
        refuse(EG_BAD_REQUEST, std::string(call) + (fork ? ": a conditional is still open inside the fork branch" : ": a fork is still open inside the conditional's arm"));
    return frames_.back();
}

void GraphShape::fork_begin() {
    Frame f;
    f.fork = true;
    f.node = push(EG_GNODE_FORK, container(), -1, "a fork");
    f.branches.push_back(push(EG_GNODE_BRANCH, f.node, 0, "a branch"));
    frames_.push_back(std::move(f));
}

void GraphShape::fork_branch() {
    Frame& f = innermost(true, "eg_record_fork_branch");
    if (static_cast<int>(nodes_.size()) - 1 == f.branches.back()) refuse(EG_BAD_REQUEST, "eg_record_fork_branch: the branch is empty");
    const int arm = static_cast<int>(f.branches.size());
    f.branches.push_back(push(EG_GNODE_BRANCH, f.node, arm, "a branch"));
}

std::string GraphShape::fork_end() {
    Frame& f = innermost(true, "eg_record_fork_end");
    const int last = static_cast<int>(nodes_.size()) - 1;
    if (last == f.branches.back()) refuse(EG_BAD_REQUEST, "eg_record_fork_end: the branch is empty");
    std::vector<std::vector<int>> members(f.branches.size());   // the nodes with planes, by branch
    for (int i = f.node + 1; i <= last; ++i)
        for (std::size_t b = 0; b < f.branches.size(); ++b)
            if (!nodes_[static_cast<std::size_t>(i)].refs.empty() && within(i, f.branches[b])) members[b].push_back(i);
    for (std::size_t b = 1; b < members.size(); ++b)
        for (std::size_t a = 0; a < b; ++a)
            for (int i : members[a])
                for (int j : members[b]) {
                    const Node &ni = nodes_[static_cast<std::size_t>(i)], &nj = nodes_[static_cast<std::size_t>(j)];
                    const auto waited = [&](const Order& o) { return o.from == i && o.branch == f.branches[b] && j >= o.since; };
                    if (std::any_of(f.ordered.begin(), f.ordered.end(), waited)) continue;
                    std::string reason;
                    hazards(ni.refs, nj.refs, [&](int ua, int ub, graph::EdgeKind kind) {
                        if (!reason.empty()) return;
                        reason = "eg_record_fork_end: a " + std::string(graph::kindName(kind)) + " race between node " + std::to_string(i) + " (" +
                                 ni.label + ") slot '" + ni.uses[static_cast<std::size_t>(ua)].name + "' and node " + std::to_string(j) + " (" +
                                 nj.label + ") slot '" + nj.uses[static_cast<std::size_t>(ub)].name + "' of different branches: give the branches " +
                                 "disjoint planes or order them with eg_record_wait";
                    });
                    if (!reason.empty()) return reason;
                }
    const int fork = f.node;
    frames_.pop_back();
    nodes_[static_cast<std::size_t>(push(EG_GNODE_JOIN, nodes_[static_cast<std::size_t>(fork)].parent, -1, "a join"))].closes = fork;
    return "";
}

void GraphShape::wait(const std::vector<eg_plane_s*>& planes) {
    Frame& f = innermost(true, "eg_record_wait");
    const int mine = f.branches.back();
    std::vector<int> writers;                   // the nodes of earlier branches that write a named plane
    for (std::size_t k = 0; k < planes.size(); ++k) {
        const std::string at = "eg_record_wait: planes[" + std::to_string(k) + "]";
        const graph::PlaneRef named = plane_ref(*planes[k], 0, 0);
        bool found = false;
        for (int i = f.node + 1; i < static_cast<int>(nodes_.size()); ++i) {
            const Node& n = nodes_[static_cast<std::size_t>(i)];
            for (const graph::PlaneRef& r : n.refs) {
                if (r.access == 0 || !graph::detail::overlap(named, r)) continue;
                if (within(i, mine))
                    refuse(EG_BAD_REQUEST, at + " is written by this branch (node " + std::to_string(i) + " slot '" + n.uses[static_cast<std::size_t>(std::stoi(r.name))].name + "')");
                writers.push_back(i);
                found = true;
            }
        }
        if (!found) refuse(EG_BAD_REQUEST, at + " is written by no earlier sibling branch");
    }
    const int wait = push(EG_GNODE_WAIT, mine, -1, "a wait");
    std::sort(writers.begin(), writers.end());
    writers.erase(std::unique(writers.begin(), writers.end()), writers.end());
    for (int from : writers) f.ordered.push_back({ from, mine, wait });
    // the plane of each waiting edge: keep the first named plane on the wait node as a read, so a description can name it
    for (eg_plane_s* p : planes) nodes_[static_cast<std::size_t>(wait)].uses.push_back({ p, 0, -1, "wait" });
}

void GraphShape::cond_begin(int kind, std::size_t want) {
    const int test = static_cast<int>(nodes_.size()) - 1;
    nodes_[static_cast<std::size_t>(test)].arm = kind;
    Frame f;
    f.node = test;
    f.want = want;
    f.branches.push_back(push(EG_GNODE_ARM, test, 0, "an arm"));
    frames_.push_back(std::move(f));
}

void GraphShape::cond_arm() {
    Frame& f = innermost(false, "eg_record_cond_arm");
    if (f.branches.size() >= f.want)
        refuse(EG_BAD_REQUEST, "eg_record_cond_arm: this conditional takes " + std::to_string(f.want) + " arm(s) and all are open");
    const int arm = static_cast<int>(f.branches.size());
    f.branches.push_back(push(EG_GNODE_ARM, f.node, arm, "an arm"));
}

void GraphShape::cond_end(bool arms_need_work) {
    const Frame& f = innermost(false, "eg_record_cond_end");
    if (f.branches.size() != f.want)
        refuse(EG_BAD_REQUEST, "eg_record_cond_end: the conditional has " + std::to_string(f.branches.size()) + " arm(s) of the " + std::to_string(f.want) + " it takes");
    if (arms_need_work)    // a captured arm with no work never completes: refuse before anything is closed
        for (std::size_t arm = 0; arm < f.branches.size(); ++arm)
            if (std::none_of(nodes_.begin() + f.branches[arm] + 1, nodes_.end(), [&](const Node& n) { return n.parent == f.branches[arm]; }))
                refuse(EG_BAD_REQUEST, "eg_record_cond_end: arm " + std::to_string(arm) + " holds no node; every arm of a device recording must launch at least one kernel");
    frames_.pop_back();
}

ReversePlan GraphShape::reverse_plan() const {
    ReversePlan plan;
    std::vector<std::vector<int>> kids(nodes_.size() + 1);      // the nodes of each container, by parent + 1
    for (std::size_t i = 0; i < nodes_.size(); ++i) kids[static_cast<std::size_t>(nodes_[i].parent + 1)].push_back(static_cast<int>(i));
    const auto keyed = [](int node, const std::vector<int>& iters) {
        std::vector<int> key{ node };
        key.insert(key.end(), iters.begin(), iters.end());
        return key;
    };
    const auto add_test = [&](int cond, std::vector<int> key) {
        const Node& n = nodes_[static_cast<std::size_t>(cond)];
        plan.tests.push_back({ n.arm, (n.facts.cond_flags & EG_COND_NEGATE) != 0, key });
        plan.test_of[std::move(key)] = static_cast<int>(plan.tests.size()) - 1;
        return static_cast<int>(plan.tests.size()) - 1;
    };
    int loop = -1, iteration = -1;              // the windowed WHILE the walk is inside, and its iteration
    std::function<void(int, const std::vector<int>&, const std::vector<RGate>&)> walk =
        [&](int container, const std::vector<int>& iters, const std::vector<RGate>& gates) {
            for (int c : kids[static_cast<std::size_t>(container + 1)]) {
                const Node& n = nodes_[static_cast<std::size_t>(c)];
                if (n.kind == EG_GNODE_RUN && !iters.empty())
                    refuse(EG_NOT_WRITTEN, "eg_record_end: the run (node " + std::to_string(c) + ") sits in a WHILE: a phase count must be static: unroll the phases into the graph");
                if (n.kind == EG_GNODE_LAUNCH || n.other >= 0) {
                    plan.leaves.push_back({ n.launch >= 0 ? n.launch : launches_ + n.other, gates, keyed(c, iters), loop, iteration });
                    plan.leaf_of[plan.leaves.back().key] = static_cast<int>(plan.leaves.size()) - 1;
                } else if (n.kind == EG_GNODE_FORK) {
                    for (int b : kids[static_cast<std::size_t>(c + 1)]) walk(b, iters, gates);
                } else if (n.kind == EG_GNODE_COND) {
                    const std::vector<int>& arms = kids[static_cast<std::size_t>(c + 1)];
                    if (n.arm != EG_COND_WHILE) {
                        const int t = add_test(c, keyed(c, iters));
                        for (std::size_t a = 0; a < arms.size(); ++a) {
                            std::vector<RGate> inner = gates;
                            inner.push_back({ t, static_cast<int>(a) });
                            walk(arms[a], iters, inner);
                        }
                        continue;
                    }
                    if (n.facts.max_iters < 1)
                        refuse(EG_BAD_REQUEST, "eg_record_end: the WHILE of node " + std::to_string(c) + " has max_iters 0; EG_RECORD_DIFF needs a cap, which bounds the tape");
                    const bool outermost = iters.empty();   // an inner WHILE stays unrolled inside its outer iteration
                    const int cap = static_cast<int>(n.facts.max_iters);
                    const int w = std::min(cap, std::clamp(static_cast<int>(std::ceil(std::sqrt(static_cast<double>(cap)))), 16, 1024));
                    const std::size_t first = plan.leaves.size();
                    if (outermost) loop = static_cast<int>(plan.loops.size());
                    for (int k = 0; k < cap; ++k) {
                        std::vector<int> inner_iters = iters;
                        inner_iters.push_back(k);
                        std::vector<RGate> inner = gates;
                        inner.push_back({ add_test(c, keyed(c, inner_iters)), 0 });
                        if (outermost) iteration = k;
                        walk(arms[0], inner_iters, inner);
                    }
                    if (outermost) {
                        const int per = static_cast<int>(plan.leaves.size() - first) / cap;
                        if (per > 0) {          // a body with no launch has nothing to rebuild
                            plan.loop_of_cond[c] = loop;
                            plan.loops.push_back({ c, cap, w, (cap + w - 1) / w, static_cast<int>(first), per });
                        }
                        loop = -1;
                        iteration = -1;
                    }
                }
            }
        };
    walk(-1, {}, {});
    return plan;
}

void GraphShape::describe(DescribeStore& store, std::uint32_t flags, eg_graph_description& out) const {
    const int n = static_cast<int>(nodes_.size());
    const auto at = [](int i) { return static_cast<std::size_t>(i); };
    const auto executes = [](eg_gnode_kind k) { return k <= EG_GNODE_GEMM || k == EG_GNODE_COND; };
    store = DescribeStore();
    std::vector<eg_graph_edge>& edges = store.edges;
    const auto edge = [&](int from, int to, int kind, int from_slot = -1, int to_slot = -1, eg_plane* plane = nullptr) {
        edges.push_back({ from, to, kind, from_slot, to_slot, plane });
    };
    std::vector<int> last_in(at(n) + 1, -1);     // per container (index parent + 1): the last node recorded in it so far
    std::vector<int> latest_wait(at(n), -1);     // per BRANCH: its latest WAIT so far
    for (int i = 0; i < n; ++i) {
        const Node& nd = nodes_[at(i)];
        if (nd.parent >= 0) edge(nd.parent, i, EG_GEDGE_ORDER);
        int& last = last_in[at(nd.parent + 1)];
        if (last >= 0 && nodes_[at(last)].kind == EG_GNODE_JOIN) edge(last, i, EG_GEDGE_ORDER);   // the next node waits for every branch
        if (nd.parent >= 0 && latest_wait[at(nd.parent)] >= 0) edge(latest_wait[at(nd.parent)], i, EG_GEDGE_ORDER);
        if (nd.kind == EG_GNODE_WAIT) latest_wait[at(nd.parent)] = i;
        last = i;
        if (nd.kind == EG_GNODE_JOIN)
            for (int k = nd.closes + 1; k < i; ++k)
                if (executes(nodes_[at(k)].kind)) edge(k, i, EG_GEDGE_JOIN);
        if (nd.kind == EG_GNODE_WAIT)
            for (const PlaneUse& u : nd.uses) {
                const graph::PlaneRef named = plane_ref(*u.plane, 0, 0);
                for (int k = nodes_[at(nd.parent)].parent + 1; k < i; ++k)       // the nodes of the fork before this wait
                    if (apart(k, i, EG_GNODE_BRANCH))
                        for (const graph::PlaneRef& r : nodes_[at(k)].refs)
                            if (r.access != 0 && graph::detail::overlap(named, r)) edge(k, i, EG_GEDGE_WAIT, nodes_[at(k)].uses[at(std::stoi(r.name))].slot, -1, u.plane);
            }
    }
    for (int j = 0; j < n; ++j)                  // the hazards, between nodes that can follow one another
        for (int i = 0; i < j; ++i) {
            const Node &a = nodes_[at(i)], &b = nodes_[at(j)];
            if (a.refs.empty() || b.refs.empty() || apart(i, j, EG_GNODE_BRANCH) || apart(i, j, EG_GNODE_ARM)) continue;
            hazards(a.refs, b.refs, [&](int ua, int ub, graph::EdgeKind kind) {
                edge(i, j, edge_kind(kind), a.uses[at(ua)].slot, b.uses[at(ub)].slot, a.uses[at(ua)].plane);
            });
        }
    std::vector<int> level(at(n), 0);            // every edge goes to a later node: the longest path in one sorted pass
    std::stable_sort(edges.begin(), edges.end(), [](const eg_graph_edge& x, const eg_graph_edge& y) { return x.to < y.to; });
    for (const eg_graph_edge& e : edges) level[at(e.to)] = std::max(level[at(e.to)], level[at(e.from)] + 1);
    store.planes.resize(at(n));
    store.access.resize(at(n));
    store.slot.resize(at(n));
    for (int i = 0; i < n; ++i) {
        const Node& nd = nodes_[at(i)];
        for (const PlaneUse& u : nd.uses) {
            store.planes[at(i)].push_back(u.plane);
            store.access[at(i)].push_back(static_cast<std::uint8_t>(u.access + 1));
            store.slot[at(i)].push_back(u.slot);
        }
        eg_graph_node g{};
        g.struct_size = sizeof(g);
        g.kind = nd.kind;
        g.parent = nd.parent;
        g.level = level[at(i)];
        g.arm = nd.kind == EG_GNODE_ARM ? nd.arm : nd.kind == EG_GNODE_COND ? nd.arm : -1;
        g.n_planes = static_cast<std::int32_t>(nd.uses.size());
        g.unit = nd.unit.empty() ? nullptr : nd.unit.c_str();
        g.cond_flags = nd.facts.cond_flags;
        g.has_baseline = nd.facts.has_baseline;
        g.max_iters = nd.facts.max_iters;
        g.cell_index = nd.facts.cell_index;
        g.n_plans = nd.facts.n_plans;
        g.launch = nd.launch;
        g.planes = store.planes[at(i)].data();
        g.access = store.access[at(i)].data();
        g.slot = store.slot[at(i)].data();
        store.nodes.push_back(g);
    }
    out = eg_graph_description{};
    out.struct_size = sizeof(out);
    out.flags = flags;
    out.n_nodes = n;
    out.nodes = store.nodes.data();
    out.n_edges = static_cast<std::int32_t>(edges.size());
    out.edges = edges.data();
    out.n_levels = n == 0 ? 0 : *std::max_element(level.begin(), level.end()) + 1;
    out.node_size = sizeof(eg_graph_node);
    out.edge_size = sizeof(eg_graph_edge);
}

}  // namespace door
}  // namespace eagle
