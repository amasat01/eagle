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
#include <string>
#include <utility>
#include <vector>

namespace eagle {
namespace door {

/** @brief Whether a slot of role @p r is read / written by its unit (MUTABLE is a write here: the reverse's view). */
inline bool reads_role(std::int32_t r) {
    return r == EG_ROLE_PLANE_IN || r == EG_ROLE_LOOKUP || r == EG_ROLE_WIDE_IN;
}
inline bool writes_role(std::int32_t r) { return r == EG_ROLE_MUTABLE || r == EG_ROLE_PLANE_OUT || r == EG_ROLE_OUT; }

/** @brief One recorded call, or one conditional with an arm list per arm. */
struct Node {
    enum Kind { Launch, Run, Reduce, Copy, Fft, Gemm, Cond, Host, Scan, Compact } kind = Launch;
    eg_plan_s* plan = nullptr;
    eg_runner_s* runner = nullptr;
    host::ReduceOp reduce;
    host::ScanOp scan;                          ///< a scan
    host::CompactOp compact;                    ///< a compaction
    eg_fft_s* fft = nullptr;
    eg_gemm_s* gemm = nullptr;
    eg_host_node_s* host = nullptr;             ///< a host call's node
    std::vector<eg_plane_s*> hplanes;           ///< a host call's declared planes, in declaration order
    std::vector<std::uint32_t> haccess;         ///< and how each is declared (EG_ACCESS_READ, EG_ACCESS_WRITE or both)
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
    eg_plane_s* counter = nullptr;              ///< WHILE: the one-cell plane that receives the iteration index and the completed count
    eg_plane_s* limit = nullptr;                ///< WHILE: the plane whose element ``limit_index`` bounds the iteration count
    std::int64_t limit_index = 0;
    eg_plane_s* brk = nullptr;                  ///< WHILE: the plane whose element ``brk_index`` stops the loop once an iteration leaves it non-zero
    std::int64_t brk_index = 0;
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
    bool scatter = false;                       ///< GRAD: the vjp slot is a zeroed PLANE_OUT, a cross-sample scatter-add
};

/** @brief How a host call's declared plane is seen by its vjp: its index in GraphDiff::planes (-1: untracked, it has no value or adjoint the vjp needs) and its declared access. */
struct HostArg {
    int plane = -1;
    std::uint32_t access = 0;
    bool real = false;                          ///< a float plane: it has an adjoint
};

/** @brief One leaf's accesses and what reverses it: a launch has its vjp, the other calls the transposes the door makes at eg_record_end
 *  (a copy and a sum reduce need none). A call with no float plane has no adjoint and is left with no accesses. */
struct DiffNode {
    Node::Kind kind = Node::Launch;
    eg_plan_s* plan = nullptr;                  ///< a launch
    std::vector<int> reads, writes;             ///< indices in GraphDiff::planes
    std::vector<int> kept_reads, overwrites;    ///< planes with no adjoint a vjp reads the value of, and kept ones the leaf overwrites (indices in GraphDiff::planes)
    std::vector<int> valued;                    ///< the reads whose values the reverse needs: a launch's reads, a product's A and B
    eg_unit_s* vjp = nullptr;                   ///< a launch: a reference is held; null: no gradient flows through the node
    std::vector<VjpLink> links;
    host::ReduceOp reduce;                      ///< a reduce
    eg_fft_s *fft = nullptr, *adjoint = nullptr;    ///< a transform (a reference is held) and the transform that is its adjoint (made here, freed with the tape)
    eg_gemm_s gemm, gemm_a, gemm_b;             ///< a product, and the products that give the gradients of its A and of its B
    eg_runner_s* runner = nullptr;              ///< a run (a recorded runner: the graph's node holds its reference)
    eg_host_node_s* host = nullptr;             ///< a host call: its node (the recorded graph holds the reference)
    std::vector<HostArg> args;                  ///< a host call: one per declared plane
    std::vector<int> slots;                     ///< a run: the primal slot of each entry of ``reads``; the first ``writes.size()`` are the states it overwrites
};

/** @brief A plane the graph copied just before node ``node`` overwrites it (or reads it in place). */
struct Snapshot {
    std::size_t node = 0;
    int alias = -1;                             ///< a leaf copy in a windowed WHILE: the entry whose buffer it shares (same body leaf, iteration mod w); else -1
    int plane = -1;
    std::size_t size = 0;                       ///< bytes of the plane
    Bytes bytes;
};

/** @brief What the forward keeps of a windowed WHILE: the planes its body writes, copied at the start of every w-th iteration. */
struct LoopTape {
    std::vector<int> planes;                    ///< indices in GraphDiff::planes
    std::vector<std::vector<Bytes>> ckpt;       ///< per window, per plane of ``planes``
    int built = -1;                             ///< the window whose iterations the reverse has rebuilt into the shared buffers
    int counter = -1;                           ///< the index in GraphDiff::planes of the loop's counter plane, or -1
    int shown = -1;                             ///< the value the reverse last wrote into the counter plane, or -1
    int final_count = 0;                        ///< the counter plane's value when the reverse began: the forward's iteration count
};

/** @brief The tape of a graph ended with EG_RECORD_DIFF. */
struct GraphDiff {
    std::vector<eg_plane_s*> planes;            ///< every plane a launch binds, once
    eg_ctx* dev = nullptr;                      ///< the context of a DEVICE graph (its buffers and vjps live on the device), else null
    std::vector<DiffNode> nodes;                ///< the launches by their ordinal, then the other leaves by theirs
    ReversePlan plan;                           ///< the launches in the order the forward runs them, WHILE bodies unrolled
    std::vector<TestSave> saved;                ///< per ``plan.tests``: what the last replay's test saw
    std::vector<LoopTape> loops;                ///< per ``plan.loops``
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


/** @brief The words that name host call @p n in a description, a race and a failure. */
std::string host_node_label(const Node& n);
/** @brief The planes host call @p n touches, as the shape records them: a plane declared READ is read, WRITE is written, both is in place. */
std::vector<PlaneUse> host_node_uses(const Node& n);
/** @brief Check a host call's request (handles, the access of each plane, no plane twice) and make its node; nothing is held yet. */
Node host_node_check(eg_ctx& ctx, eg_host_node_s* node, std::int32_t n_planes, eg_plane_s* const* planes, const std::uint32_t* access);
/** @brief Run host call @p n now on the calling thread: on a device context the stream is synchronized first; the callback runs with the context marked busy.
 *  A nonzero return is EG_FAULT "host node '<name>' failed (<code>)". */
void host_node_run(eg_ctx& ctx, const Node& n);
/** @brief Call the vjp of host node @p node with the four arrays (host/HostNode.cpp), under the same rules; a failure is EG_FAULT naming the node. */
void host_vjp_run(eg_ctx& ctx, const eg_host_node_s& node, std::int32_t n, DLManagedTensorVersioned* const* primals, DLManagedTensorVersioned* const* adjoints_in, DLManagedTensorVersioned* const* adjoints_out);
/** @brief Take / drop the references a recorded node holds (its plans, runner, transform, product, planes and host node). */
void node_hold(const Node& n);
void node_drop(eg_ctx& ctx, const Node& n);
/** @brief eg_host_call while a device recording is open: the capture ends before the call and the call becomes a step of the graph that eg_graph_launch runs between
 *  that capture and the next; refused NOT_WRITTEN inside a conditional arm, a fork branch or a WHILE (device/Device.cpp). */
void device_record_host_node(eg_ctx& ctx, Node&& node);

/** @brief What one launch of a step graph left for its reverse: the tape's snapshots (the planes a later leaf overwrote), the windowed loops'
 *  checkpoints, the tests the launch saw and the value of every plane of the graph after it. Kept per step of a checkpoint window. */
struct StepTape {
    std::vector<Bytes> snapshots;               ///< by GraphDiff::snapshots (an aliased entry has none of its own)
    std::vector<std::vector<Bytes>> ckpt;       ///< by GraphDiff::loops: the buffers of its windows, plane by plane within a window
    std::vector<TestSave> saved;
    std::vector<Bytes> planes;                  ///< by GraphDiff::planes
};
/** @brief Keep what the last launch of step graph @p graph (a DIFF graph) left, into @p out; and put it back before that step's reverse. */
void step_tape_save(eg_ctx& ctx, eg_graph_s* graph, StepTape& out);
void step_tape_restore(eg_ctx& ctx, eg_graph_s* graph, const StepTape& tape);
/** @brief A plane of a step graph's tape: whether the graph reads it before any node writes it, and whether it writes it. */
struct StepPlane {
    eg_plane_s* plane = nullptr;
    bool input = false, written = false;
};
std::vector<StepPlane> step_graph_planes(eg_ctx& ctx, eg_graph_s* graph);
/** @brief A float plane with a running adjoint (@p adjoint, the plane's element count of floats of the plane's width, in the plane's own sample order). */
struct StepAdjoint {
    eg_plane_s* plane = nullptr;
    char* adjoint = nullptr;
};
/** @brief What a step graph's reverse reports: the memory passes it ran in (one per call) and whether every adjoint it wrote is bitwise reproducible. */
struct StepReverseInfo {
    std::int32_t passes = 0;
    bool bitwise = true;
};
/** @brief Reverse the last launch of step graph @p graph over @p states: a plane the graph writes is seeded with its adjoint and gets it REPLACED
 *  by what flows to it through the graph's reads (zero when the graph never reads it first); a plane it only reads gets that ADDED. Planes of
 *  the graph not listed have no adjoint: neither seeded nor written. */
void graph_reverse_step(eg_ctx& ctx, eg_graph_s* graph, const std::vector<StepAdjoint>& states, StepReverseInfo& info);

/** @brief The tape of a graph ended with EG_RECORD_DIFF (host/GraphReverse.cpp): @p plan from GraphShape::reverse_plan, @p launches by launch ordinal and @p others (copies, reduces, transforms and products) by their ordinal; refuses what the reverse does not cover. */
std::unique_ptr<GraphDiff> diff_plan(eg_ctx& ctx, const std::vector<eg_plan_s*>& launches, const std::vector<const Node*>& others, const ReversePlan& plan, const eg_record_end_desc& desc);
/** @brief Refuse what has no gradient to pull back among @p others (the copies, reduces, transforms, products and runs of the recording, in order): a reduce
 *  with an op other than sum (BAD_REQUEST), a 2-D or companion float sum, a run of a runner that is not EG_RUN_RECORD (NOT_WRITTEN), a runner run twice (BAD_REQUEST). */
void leaf_checks(const std::vector<const Node*>& others);
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
    int refs = 1;                                   ///< the caller's reference plus one per runner that runs it as its step graph
    bool caller_ref = true;
    mutable std::unique_ptr<eagle::door::DescribeStore> described;   ///< the arrays of the last description
};
