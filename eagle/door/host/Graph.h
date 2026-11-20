// SPDX-License-Identifier: Apache-2.0

// What a host graph is: the recorded node tree (host/Record.cpp) and, for a graph ended with EG_RECORD_DIFF, the tape
// eg_graph_reverse reads (host/GraphReverse.cpp).
#pragma once

#include "eagle/door/Context.h"
#include "eagle/door/host/Ops.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace eagle {
namespace door {

/** @brief One recorded call, or one conditional with an arm list per arm. */
struct Node {
    enum Kind { Launch, Run, Reduce, Copy, Fft, Gemm, Cond } kind = Launch;
    eg_plan_s* plan = nullptr;
    eg_runner_s* runner = nullptr;
    host::ReduceOp reduce;
    eg_fft_s* fft = nullptr;
    eg_gemm_s* gemm = nullptr;
    eg_plane_s* rhs = nullptr;                  ///< a product's B (its A is ``src``, its C ``dst``)
    eg_plane_s* src = nullptr;                  ///< a copy's source, a transform's input
    eg_plane_s* dst = nullptr;                  ///< a copy's target, a transform's output
    eg_cond_kind cond = EG_COND_IF;
    eg_plane_s* cell = nullptr;
    std::size_t want = 0;                       ///< arms the conditional needs
    std::vector<std::vector<Node>> arms;
};

/** @brief How a slot of a launch's vjp reads the launch (one per vjp slot that is not the sample count). */
struct VjpLink {
    enum Kind { VALUE_PLANE, VALUE_UNIFORM, ADJOINT_IN, GRAD } kind = VALUE_PLANE;
    int slot = -1;                              ///< the vjp's slot
    int primal = -1;                            ///< the launch's slot it names
    int plane = -1;                             ///< index in GraphDiff::planes (not for a uniform)
    bool reduced = false;                       ///< GRAD: summed over samples into a one-cell plane
};

/** @brief One launch node's accesses and its vjp. */
struct DiffNode {
    std::vector<int> reads, writes;             ///< indices in GraphDiff::planes
    eg_unit_s* vjp = nullptr;                   ///< a reference is held; null: no gradient flows through the node
    std::vector<VjpLink> links;
};

/** @brief A plane the graph copied just before node ``node`` overwrites it (or reads it in place). */
struct Snapshot {
    std::size_t node = 0;
    int plane = -1;
    std::vector<std::uint64_t> bytes;
};

/** @brief The tape of a graph ended with EG_RECORD_DIFF. */
struct GraphDiff {
    std::vector<eg_plane_s*> planes;            ///< every plane a launch binds, once
    std::vector<DiffNode> nodes;                ///< per top-level node, which are all launches
    std::vector<Snapshot> snapshots;
    bool launched = false;
    /** @brief A plan the graph runs: its bind count at eg_record_end (a launch of the graph refuses a rebound plan) and its
     *  launch count just after the graph's last launch (the reverse refuses a plan launched outside the graph since). */
    struct PlanMark {
        eg_plan_s* plan = nullptr;
        std::uint64_t binds = 0, launches = 0;
    };
    std::vector<PlanMark> marks;
};


/** @brief The tape of a graph ended with EG_RECORD_DIFF over @p nodes (host/GraphReverse.cpp); refuses what the reverse does not cover. */
std::unique_ptr<GraphDiff> diff_plan(eg_ctx& ctx, const std::vector<Node>& nodes, const eg_record_end_desc& desc);
void diff_free(eg_ctx& ctx, GraphDiff& diff);

}  // namespace door
}  // namespace eagle

struct eg_graph_s {
    std::vector<eagle::door::Node> nodes;
    std::unique_ptr<eagle::door::GraphDiff> diff;   ///< set when ended with EG_RECORD_DIFF
};
