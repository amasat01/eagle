// SPDX-License-Identifier: Apache-2.0

// Host recording: between eg_record_begin and eg_record_end the recordable
// calls (eg_launch, eg_run, eg_reduce, eg_copy, eg_fft_run, eg_gemm_run) and conditionals are kept as
// an ordered tree of nodes instead of running; eg_graph_launch replays it in
// C. A node holds a reference to each plan, runner and plane it names (the
// plan, runner, transform plan and plane counts of slice 1), so a recorded graph outlives the
// caller's handles; eg_graph_free drops them. In a device recording the same nodes become host nodes of the capture
// (record_host_step): the graph holds them and their references until it is freed.
#include "eagle/door/Context.h"
#include "eagle/door/host/Ops.h"
#include "eagle/door/device/Seam.h"
#include "eagle/door/host/Team.h"

#include <memory>

namespace eagle {
namespace door {

namespace {

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

/** @brief The planes a node names directly. */
std::vector<eg_plane_s*> planes_of(const Node& n) {
    switch (n.kind) {
        case Node::Reduce: return { n.reduce.in, n.reduce.out, n.reduce.comp };
        case Node::Copy:
        case Node::Fft: return { n.src, n.dst };
        case Node::Gemm: return { n.src, n.rhs, n.dst };
        case Node::Cond: return { n.cell };
        default: return {};
    }
}

void retain(const Node& n) {
    if (n.plan != nullptr) ++n.plan->refs;
    if (n.runner != nullptr) ++n.runner->refs;
    if (n.fft != nullptr) ++n.fft->refs;
    if (n.gemm != nullptr) ++n.gemm->refs;
    for (eg_plane_s* p : planes_of(n))
        if (p != nullptr) ++p->refs;
}

void release(eg_ctx& ctx, const std::vector<Node>& nodes) {
    for (const Node& n : nodes) {
        if (n.plan != nullptr) plan_release(ctx, n.plan);
        if (n.runner != nullptr) runner_release(ctx, n.runner);
        if (n.fft != nullptr) fft_release(ctx, n.fft);
        if (n.gemm != nullptr) gemm_release(ctx, n.gemm);
        for (eg_plane_s* p : planes_of(n))
            if (p != nullptr) plane_release(ctx, p);
        for (const std::vector<Node>& arm : n.arms) release(ctx, arm);
    }
}

}  // namespace
}  // namespace door
}  // namespace eagle

struct eg_graph_s {
    std::vector<eagle::door::Node> nodes;
};

struct eg_recording_s {
    std::vector<eagle::door::Node> top;
    std::vector<eagle::door::Node*> open;       ///< the open conditionals, outermost first

    /** @brief The list the next call is recorded into. */
    std::vector<eagle::door::Node>& current() { return open.empty() ? top : open.back()->arms.back(); }
};

namespace eagle {
namespace door {

namespace {

void replay(eg_ctx& ctx, const std::vector<Node>& nodes);

/** @brief One host step of a device graph: the recorded node, what it is called in a failure, and its context. */
struct HostStep {
    eg_ctx* ctx = nullptr;
    std::vector<Node> node;                     ///< the one node it runs
    std::string name;
};

/** @brief The step's words in a failure: the call that was recorded. */
std::string describe(const Node& n) {
    switch (n.kind) {
        case Node::Launch: return "eg_launch of unit '" + n.plan->unit->name + "'";
        case Node::Run: return "eg_run";
        case Node::Reduce: return "eg_reduce";
        case Node::Copy: return "eg_copy";
        case Node::Fft: return "eg_fft_run";
        case Node::Gemm: return "eg_gemm_run";
        case Node::Cond: return "a conditional";
    }
    return "a host step";
}

/** @brief What the driver's host node runs at each replay, on a driver thread: the recorded step on the context's host
 *  team. It makes no CUDA call (the driver forbids it inside a host node; the host steps make none) and never throws;
 *  a failure is stored for the next eg_graph_launch or eg_sync. */
void run_host_step(void* user, StepFailure& failure) noexcept {
    const auto* step = static_cast<const HostStep*>(user);
    try {
        replay(*step->ctx, step->node);
    } catch (const std::exception& e) {
        failure.set(step->name + ": " + e.what());
    } catch (...) {
        failure.set(step->name + ": an unknown failure inside eagle");
    }
}

/** @brief The graph is freed: drop the references the step holds. */
void release_host_step(eg_ctx& ctx, void* user) {
    const std::unique_ptr<HostStep> step(static_cast<HostStep*>(user));
    release(ctx, step->node);
}

/** @brief Keep @p node as a host node of the device recording in scope, holding a reference to what it names. */
void record_host_step(eg_ctx& ctx, Node&& node) {
    auto step = std::make_unique<HostStep>();
    step->ctx = &ctx;
    step->name = describe(node);
    step->node.push_back(std::move(node));
    retain(step->node.front());
    try {
        device_record_host_step(ctx, run_host_step, step.get(), release_host_step);
    } catch (...) {
        release(ctx, step->node);
        throw;
    }
    step.release();
}

/** @brief Record @p node when @p ctx is recording; false when it is not (the caller runs it). A device recording keeps
 *  it as a host node of the capture. */
bool record(eg_ctx& ctx, Node&& node) {
    if (ctx.dev_rec != nullptr) {
        record_host_step(ctx, std::move(node));
        return true;
    }
    if (ctx.rec == nullptr) return false;
    std::vector<Node>& list = ctx.rec->current();
    list.push_back(std::move(node));
    retain(list.back());
    return true;
}

/** @brief Run @p nodes in order; each conditional reads its int32 cell now. */
void replay(eg_ctx& ctx, const std::vector<Node>& nodes) {
    for (const Node& n : nodes) {
        switch (n.kind) {
            case Node::Launch:
                check_bound(*n.plan);
                n.plan->finished = 0;    // a replayed launch counts from zero, as eg_launch does
                host::run_plan(*n.plan, host::team_threads(ctx));
                break;
            case Node::Run: run_loop(ctx, n.runner, nullptr); break;
            case Node::Reduce: host::reduce_exec(n.reduce, host::team_threads(ctx)); break;
            case Node::Copy: host::copy_exec(*n.src, *n.dst); break;
            case Node::Fft: host::fft_exec(*n.fft, *n.src, *n.dst, host::team_threads(ctx)); break;
            case Node::Gemm: host::gemm_exec(*n.gemm, *n.src, *n.rhs, *n.dst); break;
            case Node::Cond: {
                const DLTensor& t = n.cell->tensor->dl_tensor;
                const auto* cell = reinterpret_cast<const std::int32_t*>(static_cast<const char*>(t.data) + t.byte_offset);
                if (n.cond == EG_COND_IF) {
                    if (*cell != 0) replay(ctx, n.arms[0]);
                } else if (n.cond == EG_COND_IF_ELSE) {
                    replay(ctx, n.arms[*cell != 0 ? 0 : 1]);
                } else if (n.cond == EG_COND_WHILE) {
                    while (*cell != 0) replay(ctx, n.arms[0]);
                } else if (*cell >= 0 && static_cast<std::size_t>(*cell) < n.want) {
                    replay(ctx, n.arms[static_cast<std::size_t>(*cell)]);
                }
                break;
            }
        }
    }
}

eg_recording_s& recording(eg_ctx& ctx) {
    if (ctx.rec == nullptr) refuse(EG_BAD_REQUEST, "not recording: call eg_record_begin first");
    return *ctx.rec;
}

}  // namespace

/** @brief eg_launch while recording: keep the launch (the caller has checked the plan is bound). */
bool record_launch(eg_ctx& ctx, eg_plan_s* plan) {
    Node n;
    n.kind = Node::Launch;
    n.plan = plan;
    return record(ctx, std::move(n));
}

/** @brief eg_run while recording: keep the run. */
bool record_run(eg_ctx& ctx, eg_runner_s* runner) {
    Node n;
    n.kind = Node::Run;
    n.runner = runner;
    return record(ctx, std::move(n));
}

/** @brief eg_reduce: check, then keep it while recording, else run it. */
void reduce(eg_ctx& ctx, const eg_reduce_desc* desc) {
    const eg_reduce_desc d = read_input(desc, "eg_reduce_desc");
    Node n;
    n.kind = Node::Reduce;
    n.reduce = host::reduce_check(ctx, d);
    if (!record(ctx, std::move(n))) host::reduce_exec(n.reduce, host::team_threads(ctx));
}

/** @brief eg_copy: check, then keep it while recording, else run it. */
void copy(eg_ctx& ctx, eg_plane_s* src, eg_plane_s* dst) {
    host::copy_check(ctx, src, dst);
    Node n;
    n.kind = Node::Copy;
    n.src = src;
    n.dst = dst;
    if (!record(ctx, std::move(n))) host::copy_exec(*src, *dst);
}

/** @brief eg_fft_run while recording: keep the checked run. */
bool record_fft(eg_ctx& ctx, eg_fft_s* plan, eg_plane_s* in, eg_plane_s* out) {
    Node n;
    n.kind = Node::Fft;
    n.fft = plan;
    n.src = in;
    n.dst = out;
    return record(ctx, std::move(n));
}

/** @brief eg_gemm_run while recording: keep the checked run. */
bool record_gemm(eg_ctx& ctx, eg_gemm_s* gemm, eg_plane_s* a, eg_plane_s* b, eg_plane_s* c) {
    Node n;
    n.kind = Node::Gemm;
    n.gemm = gemm;
    n.src = a;
    n.rhs = b;
    n.dst = c;
    return record(ctx, std::move(n));
}

void record_begin(eg_ctx& ctx) {
    if (ctx.rec != nullptr) refuse(EG_BAD_REQUEST, "eg_record_begin: already recording");
    ctx.rec = new eg_recording_s;
}

void record_cond_begin(eg_ctx& ctx, const eg_cond_desc* desc) {
    eg_recording_s& rec = recording(ctx);
    const eg_cond_desc d = read_input(desc, "eg_cond_desc");
    if (d.flags != 0) refuse(EG_BAD_REQUEST, "eg_cond_desc.flags: unknown bits " + std::to_string(d.flags));
    std::size_t want = 1;
    switch (d.kind) {
        case EG_COND_IF:
        case EG_COND_WHILE:
            break;
        case EG_COND_IF_ELSE: want = 2; break;
        case EG_COND_SWITCH:
            if (d.n_arms < 1) refuse(EG_BAD_REQUEST, "eg_cond_desc.n_arms " + std::to_string(d.n_arms) + " < 1");
            want = static_cast<std::size_t>(d.n_arms);
            break;
        default: refuse(EG_BAD_REQUEST, "unknown conditional kind " + std::to_string(d.kind));
    }
    if (d.kind != EG_COND_SWITCH && d.n_arms != 0)
        refuse(EG_BAD_REQUEST, "eg_cond_desc.n_arms is for SWITCH only: pass 0");
    check_handle(ctx.planes, d.cell, "eg_cond_desc.cell");
    const DLTensor& t = d.cell->tensor->dl_tensor;
    if (t.dtype.code != kDLInt || t.dtype.bits != 32 || t.dtype.lanes != 1 || t.ndim < 1 || t.shape[0] < 1)
        refuse(EG_BAD_REQUEST, "eg_cond_desc.cell: an int32 plane of at least one element is expected");
    Node n;
    n.kind = Node::Cond;
    n.cond = d.kind;
    n.cell = d.cell;
    n.want = want;
    n.arms.emplace_back();                      // the first arm is being recorded
    record(ctx, std::move(n));
    rec.open.push_back(&rec.current().back());
}

void record_cond_arm(eg_ctx& ctx) {
    eg_recording_s& rec = recording(ctx);
    if (rec.open.empty()) refuse(EG_BAD_REQUEST, "eg_record_cond_arm: no conditional is open");
    Node& n = *rec.open.back();
    if (n.arms.size() >= n.want)
        refuse(EG_BAD_REQUEST, "eg_record_cond_arm: this conditional takes " + std::to_string(n.want) +
               " arm(s) and all are open");
    n.arms.emplace_back();
}

void record_cond_end(eg_ctx& ctx) {
    eg_recording_s& rec = recording(ctx);
    if (rec.open.empty()) refuse(EG_BAD_REQUEST, "eg_record_cond_end: no conditional is open");
    const Node& n = *rec.open.back();
    if (n.arms.size() != n.want)
        refuse(EG_BAD_REQUEST, "eg_record_cond_end: the conditional has " + std::to_string(n.arms.size()) +
               " arm(s) of the " + std::to_string(n.want) + " it takes");
    rec.open.pop_back();
}

void record_end(eg_ctx& ctx, eg_graph** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_record_end: out is NULL");
    *out = nullptr;
    eg_recording_s& rec = recording(ctx);
    if (!rec.open.empty()) refuse(EG_BAD_REQUEST, "eg_record_end: a conditional is still open");
    auto graph = std::make_unique<eg_graph_s>();
    graph->nodes = std::move(rec.top);
    ctx.graphs.insert(graph.get());
    delete ctx.rec;
    ctx.rec = nullptr;
    *out = graph.release();
}

void graph_launch(eg_ctx& ctx, eg_graph_s* graph) {
    check_handle(ctx.graphs, graph, "graph");
    replay(ctx, graph->nodes);
}

void graph_free(eg_ctx& ctx, eg_graph_s* graph) {
    check_handle(ctx.graphs, graph, "graph");
    ctx.graphs.erase(graph);
    release(ctx, graph->nodes);
    delete graph;
}

/** @brief Delete the open recording and every graph without releasing their references
 *  (the context frees every plan, runner and plane itself right after). */
void record_drop_all(eg_ctx& ctx) {
    delete ctx.rec;
    ctx.rec = nullptr;
    for (eg_graph_s* g : ctx.graphs) delete g;
    ctx.graphs.clear();
}

}  // namespace door
}  // namespace eagle
