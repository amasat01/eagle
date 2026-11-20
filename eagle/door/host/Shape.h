// SPDX-License-Identifier: Apache-2.0

// The shape of a recording: its nodes (every recorded call, a conditional's test and arms, a fork's branches, waits and join),
// who contains whom, and the planes each node touches. It is kept while recording and is the same for a host and a device
// recording: it validates the structure calls, refuses a race between the branches of a fork at its end, and later yields the
// description eg_graph_describe returns (edges and schedule levels derived from the planes).
#pragma once

#include "eagle/door/Context.h"
#include "eagle/graph/Edges.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace eagle {
namespace door {

/** @brief One plane a node touches: through unit slot ``slot`` (-1 for a node that is not a launch), read, written or both. */
struct PlaneUse {
    eg_plane_s* plane = nullptr;
    int access = 0;                             ///< graph::PlaneRef access: 0 read, 1 write, 2 in place
    int slot = -1;
    std::string name;                           ///< the slot's name, for a refusal
};

/** @brief The arrays an ``eg_graph_description`` points into. */
struct DescribeStore {
    std::vector<eg_graph_node> nodes;
    std::vector<std::vector<eg_plane*>> planes;
    std::vector<std::vector<std::uint8_t>> access;
    std::vector<std::vector<std::int32_t>> slot;
    std::vector<eg_graph_edge> edges;
};

/** @brief What a node reports beyond its planes: a conditional's test, a run's plan count. */
struct NodeFacts {
    std::uint32_t cond_flags = 0;
    std::int32_t has_baseline = 0, n_plans = 0;
    std::int64_t max_iters = 0, cell_index = 0;
    bool loop_controlled = false;               ///< WHILE: it holds a counter, a limit or a break plane
};

/** @brief A conditional's test as the reverse needs it: which kind, and whether it is negated. One per test the forward can evaluate. */
struct RTest {
    int kind = EG_COND_IF;
    bool negate = false;
    std::vector<int> key;                       ///< the COND node's ordinal, then the iteration of each enclosing WHILE; a WHILE's own test adds its iteration
};

/** @brief The condition a leaf's run hangs on: test ``test`` took arm ``arm`` (a WHILE: its iteration ran). */
struct RGate {
    int test = 0, arm = 0;
};

/** @brief A launch, copy, reduce, transform or product as the reverse sees it: one copy per unrolled WHILE iteration, with the gates of every conditional around it. */
struct RLeaf {
    int node = 0;                               ///< its entry in GraphDiff::nodes: the ordinal among the eg_launch calls, or the launch count plus its ordinal among the copy, reduce, fft and gemm calls
    std::vector<RGate> gates;                   ///< empty: it always runs
    std::vector<int> key;                       ///< the launch node's ordinal, then the iteration of each enclosing WHILE
    int loop = -1;                              ///< the windowed WHILE (index in ReversePlan::loops) it is a copy for, else -1
    int iter = -1;                              ///< its iteration in that loop
};

/** @brief An outermost WHILE whose reverse rebuilds states window by window: its iterations ``0 .. cap - 1`` are the leaves
 *  ``first_leaf + k * per_iter ...``, and the forward keeps the planes the body writes at the start of every ``w``-th iteration. */
struct RLoop {
    int cond = 0;                               ///< the COND node's ordinal
    int cap = 0, w = 0, groups = 0;             ///< max_iters, the window, ceil(cap / w)
    int first_leaf = 0, per_iter = 0;           ///< the first leaf of iteration 0 and the leaves of one iteration
};

/** @brief The leaves of a recording in the order the forward runs them, its WHILE bodies unrolled ``max_iters`` times. */
struct ReversePlan {
    std::vector<RLeaf> leaves;
    std::vector<RTest> tests;
    std::vector<RLoop> loops;
    std::map<int, int> loop_of_cond;            ///< COND node ordinal -> index in ``loops``
    std::map<std::vector<int>, int> leaf_of, test_of;   ///< by key
};

/** @brief What a conditional test saw at its last replay (the cell and baseline values), or that it was not evaluated. */
struct TestSave {
    bool evaluated = false;
    std::int32_t cell = 0, base = 0;
};

/** @brief Whether leaf's gate @p g was taken by the saved test @p s of @p t. */
inline bool took(const RTest& t, const TestSave& s, int arm) {
    if (!s.evaluated) return false;
    if (t.kind == EG_COND_SWITCH) return s.cell == arm;
    const bool truth = (s.cell != s.base) != t.negate;
    return t.kind == EG_COND_IF_ELSE ? arm == (truth ? 0 : 1) : (arm == 0 && truth);
}

class GraphShape {
public:
    /** @brief A call recorded (a launch, run, reduce, copy, fft, gemm, host call or conditional test) into the current container; returns its ordinal. */
    int add(eg_gnode_kind kind, std::string label, std::string unit, std::vector<PlaneUse> uses, NodeFacts facts = {});
    void fork_begin();
    void fork_branch();
    /** @brief Close the last branch. Returns "" when it joined, else the reason of the race between two branches (the fork stays open). */
    std::string fork_end();
    /** @brief Nodes recorded after this call in the current branch run after the earlier branches' nodes that write @p planes. */
    void wait(const std::vector<eg_plane_s*>& planes);
    /** @brief Open a conditional of @p kind (an eg_cond_kind) and @p want arms; its test node was just added. */
    void cond_begin(int kind, std::size_t want);
    void cond_arm();
    /** @brief Close the innermost conditional; with @p arms_need_work (a capture) an arm with no node is refused, nothing closed. */
    void cond_end(bool arms_need_work = false);
    bool open() const { return !frames_.empty(); }
    /** @brief The leaves of the recording for EG_RECORD_DIFF. Refuses a run (NOT_WRITTEN), and a WHILE without max_iters (BAD_REQUEST). */
    ReversePlan reverse_plan() const;
    /** @brief How many leaves the recording holds: its launches and its copies, reduces, transforms, products and host calls. */
    int leaves() const { return launches_ + others_; }
    /** @brief Fill @p store from the nodes and return its count of levels; the description points into @p store. */
    void describe(DescribeStore& store, std::uint32_t flags, eg_graph_description& out) const;

private:
    struct Node {
        eg_gnode_kind kind;
        int parent, arm;
        int closes = -1;                        ///< a JOIN: the FORK it closes
        std::string label, unit;
        std::vector<PlaneUse> uses;
        NodeFacts facts;
        int launch = -1;                        ///< a LAUNCH: its ordinal among the launches
        int other = -1;                         ///< a RUN, COPY, REDUCE, FFT, GEMM or HOST: its ordinal among those calls
        std::vector<graph::PlaneRef> refs;      ///< ``uses`` as hazard references; ``name`` is the index in ``uses``
    };
    struct Order {
        int from, branch, since;                ///< the nodes of ``branch`` from ordinal ``since`` run after node ``from``
    };
    struct Frame {
        bool fork = false;
        int node = -1;                          ///< the FORK or COND node
        std::vector<int> branches;              ///< a fork's BRANCH nodes; a conditional's ARM nodes
        std::size_t want = 0;                   ///< a conditional's arm count
        std::vector<Order> ordered;
    };
    std::vector<Node> nodes_;
    std::vector<Frame> frames_;

    int container() const { return frames_.empty() ? -1 : frames_.back().branches.back(); }
    int push(eg_gnode_kind kind, int parent, int arm, std::string label, std::vector<PlaneUse> uses = {}, std::string unit = "");
    int launches_ = 0, others_ = 0;
    Frame& innermost(bool fork, const char* call);
    bool within(int node, int ancestor) const;
    bool apart(int a, int b, eg_gnode_kind through) const;
};

}  // namespace door
}  // namespace eagle
