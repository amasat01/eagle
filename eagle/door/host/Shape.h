// SPDX-License-Identifier: Apache-2.0

// The shape of a recording: its nodes (every recorded call, a conditional's test and arms, a fork's branches, waits and join),
// who contains whom, and the planes each node touches. It is kept while recording and is the same for a host and a device
// recording: it validates the structure calls, refuses a race between the branches of a fork at its end, and later yields the
// description eg_graph_describe returns (edges and schedule levels derived from the planes).
#pragma once

#include "eagle/door/Context.h"
#include "eagle/graph/Edges.h"

#include <cstdint>
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
};

class GraphShape {
public:
    /** @brief A call recorded (a launch, run, reduce, copy, fft, gemm or conditional test) into the current container; returns its ordinal. */
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
    void cond_end();
    bool open() const { return !frames_.empty(); }
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
    int launches_ = 0;
    Frame& innermost(bool fork, const char* call);
    bool within(int node, int ancestor) const;
    bool apart(int a, int b, eg_gnode_kind through) const;
};

}  // namespace door
}  // namespace eagle
