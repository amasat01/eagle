// SPDX-License-Identifier: Apache-2.0

// What a host graph is: the recorded node tree (host/Record.cpp) and, for a graph ended with EG_RECORD_DIFF, the tape
// eg_graph_reverse reads (host/GraphReverse.cpp).
#pragma once

#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"
#include "eagle/door/host/Ops.h"
#include "eagle/door/host/Shape.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace eagle {
namespace door {

/** @brief Whether a slot of role @p r is read / written by its unit (MUTABLE is a write here: the reverse's view). */
inline bool reads_role(std::int32_t r) {
    return r == EG_ROLE_PER_SAMPLE || r == EG_ROLE_LOOKUP || r == EG_ROLE_VEC_IN || r == EG_ROLE_MAT_IN || r == EG_ROLE_WIDE_IN;
}
inline bool writes_role(std::int32_t r) { return r == EG_ROLE_MUTABLE || r == EG_ROLE_WIDE_OUT || r == EG_ROLE_ACCUM_OUT || r == EG_ROLE_OUT; }

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
    int ordinal = -1;                           ///< the node's ordinal in the recording's shape
    eg_cond_kind cond = EG_COND_IF;
    eg_plane_s* cell = nullptr;
    std::int64_t cell_index = 0;                ///< the element of ``cell`` the test reads
    eg_plane_s* baseline = nullptr;             ///< the plane the test compares with (null: zero)
    std::int64_t baseline_index = 0;
    std::size_t want = 0;                       ///< arms the conditional needs
    bool negate = false;                        ///< the conditional tests its cell for zero (EG_COND_NEGATE)
    std::int64_t max_iters = 0;                 ///< WHILE: the most iterations a replay runs; 0: no cap
    std::vector<std::vector<Node>> arms;
};

/** @brief Zeroed bytes, 8-byte aligned: host memory, or (given a context) device memory freed in stream order. */
struct Bytes {
    std::vector<std::uint64_t> words;
    std::shared_ptr<char> dev;
    explicit Bytes(std::size_t n = 0, eg_ctx* on = nullptr) {
        if (on == nullptr) {
            words.assign((n + 7) / 8, 0);
            return;
        }
        dev.reset(static_cast<char*>(device_alloc(*on, std::max<std::size_t>(n, 8))), [on](char* p) { device_free(*on, p); });
    }
    char* data() { return dev ? dev.get() : reinterpret_cast<char*>(words.data()); }
    const char* data() const { return dev ? dev.get() : reinterpret_cast<const char*>(words.data()); }
};

/** @brief How a slot of a launch's vjp reads the launch (one per vjp slot that is not the sample count). */
struct VjpLink {
    enum Kind { VALUE_PLANE, VALUE_UNIFORM, ADJOINT_IN, GRAD } kind = VALUE_PLANE;
    int slot = -1;                              ///< the vjp's slot
    int primal = -1;                            ///< the launch's slot it names
    int plane = -1;                             ///< index in GraphDiff::planes (not for a uniform)
    bool reduced = false;                       ///< GRAD: summed over samples into a one-cell plane
    bool scatter = false;                       ///< GRAD: the vjp slot is an ACCUM_OUT, a cross-sample scatter-add
};

/** @brief One launch node's accesses and its vjp. */
struct DiffNode {
    eg_plan_s* plan = nullptr;
    std::vector<int> reads, writes;             ///< indices in GraphDiff::planes
    eg_unit_s* vjp = nullptr;                   ///< a reference is held; null: no gradient flows through the node
    std::vector<VjpLink> links;
};

/** @brief A plane the graph copied just before node ``node`` overwrites it (or reads it in place). */
struct Snapshot {
    std::size_t node = 0;
    int plane = -1;
    std::size_t size = 0;                       ///< bytes of the plane
    Bytes bytes;
};

/** @brief The tape of a graph ended with EG_RECORD_DIFF. */
struct GraphDiff {
    std::vector<eg_plane_s*> planes;            ///< every plane a launch binds, once
    eg_ctx* dev = nullptr;                      ///< the context of a DEVICE graph (its buffers and vjps live on the device), else null
    std::vector<DiffNode> nodes;                ///< per launch, by the ordinal among the eg_launch calls
    ReversePlan plan;                           ///< the launches in the order the forward runs them, WHILE bodies unrolled
    std::vector<TestSave> saved;                ///< per ``plan.tests``: what the last replay's test saw
    std::vector<Snapshot> snapshots;            ///< ``node`` is the index in ``plan.leaves``
    bool launched = false;
    /** @brief A plan the graph runs: its bind count at eg_record_end (a launch of the graph refuses a rebound plan) and its
     *  launch count just after the graph's last launch (the reverse refuses a plan launched outside the graph since). */
    struct PlanMark {
        eg_plan_s* plan = nullptr;
        std::uint64_t binds = 0, launches = 0;
    };
    std::vector<PlanMark> marks;
};


/** @brief The tape of a graph ended with EG_RECORD_DIFF (host/GraphReverse.cpp): @p plan from GraphShape::reverse_plan, @p launches by launch ordinal; refuses what the reverse does not cover. */
std::unique_ptr<GraphDiff> diff_plan(eg_ctx& ctx, const std::vector<eg_plan_s*>& launches, const ReversePlan& plan, const eg_record_end_desc& desc);
/** @brief Whether leaf @p leaf ran at the last replay, from the tests the replay saved. */
bool leaf_ran(const GraphDiff& diff, std::size_t leaf);
void diff_free(eg_ctx& ctx, GraphDiff& diff);
/** @brief Refuse a launch of the graph when a plan of @p diff was rebound since eg_record_end; after a launch, @p diff_launched marks it. */
void diff_check_launch(const GraphDiff& diff);
void diff_launched(GraphDiff& diff);

}  // namespace door
}  // namespace eagle

struct eg_graph_s {
    std::vector<eagle::door::Node> nodes;
    std::unique_ptr<eagle::door::GraphDiff> diff;   ///< set when ended with EG_RECORD_DIFF
    eagle::door::GraphShape shape;                  ///< what eg_graph_describe returns
    mutable std::unique_ptr<eagle::door::DescribeStore> described;   ///< the arrays of the last description
};
