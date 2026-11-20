// SPDX-License-Identifier: Apache-2.0

// Host recording: between eg_record_begin and eg_record_end the recordable
// calls (eg_launch, eg_run, eg_reduce, eg_copy, eg_fft_run, eg_gemm_run) and conditionals are kept as
// an ordered tree of nodes instead of running; eg_graph_launch replays it in
// C. A node holds a reference to each plan, runner and plane it names (the
// plan, runner, transform plan and plane counts of slice 1), so a recorded graph outlives the
// caller's handles; eg_graph_free drops them. In a device recording the same nodes become host nodes of the capture
// (record_host_step): the graph holds them and their references until it is freed.
#include "eagle/door/Context.h"
#include "eagle/door/host/Graph.h"
#include "eagle/door/host/Ops.h"
#include "eagle/door/device/Seam.h"
#include "eagle/door/host/Team.h"
#include "eagle/door/host/Shape.h"

#include <cstring>
#include <functional>
#include <memory>

namespace eagle {
namespace door {

namespace {

/** @brief The planes a node names directly. */
std::vector<eg_plane_s*> planes_of(const Node& n) {
    switch (n.kind) {
        case Node::Reduce: return { n.reduce.in, n.reduce.out, n.reduce.comp };
        case Node::Copy:
        case Node::Fft: return { n.src, n.dst };
        case Node::Gemm: return { n.src, n.rhs, n.dst };
        case Node::Cond: return { n.cell, n.baseline };
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

/** @brief What a recording keeps besides the node tree: the open conditionals (their arm lists take the next call) and the shape. */
struct eg_recording_s {
    std::vector<eagle::door::Node> top;
    std::vector<eagle::door::Node*> conds;      ///< the open conditionals, outermost first
    eagle::door::GraphShape shape;

    /** @brief The list the next call is recorded into. */
    std::vector<eagle::door::Node>& current() { return conds.empty() ? top : conds.back()->arms.back(); }
};

namespace eagle {
namespace door {

namespace {

/** @brief What a replay of an EG_RECORD_DIFF graph keeps for the reverse: the copies before each leaf and what each test saw. */
struct Tape {
    GraphDiff* diff;
    std::vector<int> iters;                     ///< the iteration of each enclosing WHILE, outermost first
};

void replay(eg_ctx& ctx, const std::vector<Node>& nodes, Tape* tape = nullptr);
void run_node(eg_ctx& ctx, const Node& n, Tape* tape);

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

/** @brief How a slot of role @p role touches its plane: 0 read, 1 write, 2 in place (graph::PlaneRef access); -1: not a plane. */
int slot_access(std::int32_t role) {
    if (role == EG_ROLE_MUTABLE || role == EG_ROLE_TERMINATED) return 2;
    if (reads_role(role)) return 0;
    return writes_role(role) ? 1 : -1;
}

/** @brief The planes @p n touches: one per bound plane of each launch, and the named planes of the other calls. */
std::vector<PlaneUse> uses_of(const Node& n) {
    std::vector<PlaneUse> uses;
    const auto add = [&](eg_plane_s* p, const char* name, int access, int slot = -1) {
        if (p != nullptr && access >= 0) uses.push_back({ p, access, slot, name });
    };
    const auto add_plan = [&](const eg_plan_s& plan) {
        for (std::size_t s = 0; s < plan.bindings.size(); ++s)
            add(plan.bindings[s].plane, plan.unit->slots[s].name.c_str(), slot_access(plan.unit->slots[s].role), static_cast<int>(s));
    };
    switch (n.kind) {
        case Node::Launch: add_plan(*n.plan); break;
        case Node::Run:
            for (const eg_plan_s* plan : n.runner->plans) add_plan(*plan);
            break;
        case Node::Reduce:
            add(n.reduce.in, "in", 0);
            add(n.reduce.out, "out", 1);
            add(n.reduce.comp, "comp", n.reduce.pair ? 1 : 0);
            break;
        case Node::Copy:
        case Node::Fft:
            add(n.src, "src", 0);
            add(n.dst, "dst", 1);
            break;
        case Node::Gemm:
            add(n.src, "a", 0);
            add(n.rhs, "b", 0);
            add(n.dst, "c", 1);
            break;
        case Node::Cond:
            add(n.cell, "cell", 0);
            add(n.baseline, "baseline", 0);
            break;
    }
    return uses;
}

/** @brief The unit a launch runs, or a run's first plan's; empty for the other calls. */
std::string unit_name(const Node& n) {
    if (n.plan != nullptr) return n.plan->unit->name;
    return n.runner != nullptr && !n.runner->plans.empty() ? n.runner->plans.front()->unit->name : std::string();
}

/** @brief What a conditional's test and a run report beyond their planes. */
NodeFacts facts_of(const Node& n) {
    NodeFacts f;
    if (n.kind == Node::Cond) {
        f.cond_flags = n.negate ? EG_COND_NEGATE : 0;
        f.has_baseline = n.baseline != nullptr;
        f.max_iters = n.max_iters;
        f.cell_index = n.cell_index;
    } else if (n.kind == Node::Run) {
        f.n_plans = static_cast<std::int32_t>(n.runner->plans.size());
    }
    return f;
}

/** @brief The description kind of a recorded node. */
eg_gnode_kind gnode_kind(const Node& n) {
    switch (n.kind) {
        case Node::Launch: return EG_GNODE_LAUNCH;
        case Node::Run: return EG_GNODE_RUN;
        case Node::Reduce: return EG_GNODE_REDUCE;
        case Node::Copy: return EG_GNODE_COPY;
        case Node::Fft: return EG_GNODE_FFT;
        case Node::Gemm: return EG_GNODE_GEMM;
        case Node::Cond: return EG_GNODE_COND;
    }
    return EG_GNODE_LAUNCH;
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
        device_shape(ctx).add(gnode_kind(node), describe(node), unit_name(node), uses_of(node), facts_of(node));
        record_host_step(ctx, std::move(node));
        return true;
    }
    if (ctx.rec == nullptr) return false;
    std::vector<Node>& list = ctx.rec->current();
    list.push_back(std::move(node));
    retain(list.back());
    list.back().ordinal = ctx.rec->shape.add(gnode_kind(list.back()), describe(list.back()), unit_name(list.back()), uses_of(list.back()), facts_of(list.back()));
    return true;
}

/** @brief Element @p index of the one-dimensional 32-bit plane @p p, read now. */
std::int32_t element(const eg_plane_s& p, std::int64_t index) {
    const DLTensor& t = p.tensor->dl_tensor;
    const std::int64_t stride = t.strides != nullptr ? t.strides[0] : 1;
    std::int32_t v;
    std::memcpy(&v, static_cast<const char*>(t.data) + t.byte_offset + index * stride * 4, sizeof(v));
    return v;
}

/** @brief Run @p nodes in order; each conditional reads its cell now. */
void replay(eg_ctx& ctx, const std::vector<Node>& nodes, Tape* tape) {
    for (const Node& n : nodes) run_node(ctx, n, tape);
}

/** @brief Iteration @p k of the WHILE at node @p cond starts: when it opens a window, copy the planes the body writes. */
void keep_window_start(GraphDiff& diff, int cond, int k) {
    const auto it = diff.plan.loop_of_cond.find(cond);
    if (it == diff.plan.loop_of_cond.end()) return;
    const int w = diff.plan.loops[static_cast<std::size_t>(it->second)].w;
    if (k % w != 0) return;
    LoopTape& tape = diff.loops[static_cast<std::size_t>(it->second)];
    std::vector<Bytes>& group = tape.ckpt[static_cast<std::size_t>(k / w)];
    for (std::size_t q = 0; q < tape.planes.size(); ++q) {
        const DLTensor& t = diff.planes[static_cast<std::size_t>(tape.planes[q])]->tensor->dl_tensor;
        std::memcpy(group[q].data(), static_cast<const char*>(t.data) + t.byte_offset, static_cast<std::size_t>(elements(t)) * (t.dtype.bits / 8));
    }
}

/** @brief The key of node @p n under the loops @p tape is in, then @p extra (a WHILE's own iteration) when given. */
std::vector<int> tape_key(const Tape& tape, const Node& n, int extra = -1) {
    std::vector<int> key{ n.ordinal };
    key.insert(key.end(), tape.iters.begin(), tape.iters.end());
    if (extra >= 0) key.push_back(extra);
    return key;
}

/** @brief Copy the planes leaf @p n overwrites, as they are before it, into the tape's snapshots of the leaf. */
void keep_leaf_copies(Tape& tape, const Node& n) {
    const int leaf = tape.diff->plan.leaf_of.at(tape_key(tape, n));
    if (tape.diff->plan.leaves[static_cast<std::size_t>(leaf)].loop >= 0) return;   // a windowed WHILE's copies are rebuilt by the reverse
    for (Snapshot& s : tape.diff->snapshots)
        if (s.node == static_cast<std::size_t>(leaf)) {
            const DLTensor& t = tape.diff->planes[static_cast<std::size_t>(s.plane)]->tensor->dl_tensor;
            std::memcpy(s.bytes.data(), static_cast<const char*>(t.data) + t.byte_offset, s.size);
        }
}

/** @brief Run one recorded node. */
void run_node(eg_ctx& ctx, const Node& n, Tape* tape) {
    if (tape != nullptr && n.kind != Node::Cond) keep_leaf_copies(*tape, n);
    switch (n.kind) {
        case Node::Launch:
            check_bound(*n.plan);
            n.plan->finished = 0;    // a replayed launch counts from zero, as eg_launch does
            host::run_plan(*n.plan, host::team_threads(ctx));
            break;
        case Node::Run: run_from_start(ctx, n.runner); break;
        case Node::Reduce: host::reduce_exec(n.reduce, host::team_threads(ctx)); break;
        case Node::Copy: host::copy_exec(*n.src, *n.dst); break;
        case Node::Fft: host::fft_exec(*n.fft, *n.src, *n.dst, host::team_threads(ctx)); break;
        case Node::Gemm: host::gemm_exec(*n.gemm, *n.src, *n.rhs, *n.dst); break;
        case Node::Cond: {
            // int32 or uint32: equal or not, or an arm index, reads the same bits
            const auto cell = [&] { return element(*n.cell, n.cell_index); };
            const auto base = [&] { return n.baseline != nullptr ? element(*n.baseline, n.baseline_index) : 0; };
            const auto truth = [&] { return (cell() != base()) != n.negate; };
            const auto save = [&](int extra) {      // what the test sees, for the reverse
                if (tape == nullptr) return;
                TestSave& t = tape->diff->saved[static_cast<std::size_t>(tape->diff->plan.test_of.at(tape_key(*tape, n, extra)))];
                t = { true, cell(), base() };
            };
            if (n.cond == EG_COND_IF) {
                save(-1);
                if (truth()) replay(ctx, n.arms[0], tape);
            } else if (n.cond == EG_COND_IF_ELSE) {
                save(-1);
                replay(ctx, n.arms[truth() ? 0 : 1], tape);
            } else if (n.cond == EG_COND_WHILE) {
                for (std::int64_t i = 0; n.max_iters == 0 || i < n.max_iters; ++i) {
                    save(static_cast<int>(i));
                    if (!truth()) break;
                    if (tape != nullptr && tape->iters.empty()) keep_window_start(*tape->diff, n.ordinal, static_cast<int>(i));
                    if (tape != nullptr) tape->iters.push_back(static_cast<int>(i));
                    replay(ctx, n.arms[0], tape);
                    if (tape != nullptr) tape->iters.pop_back();
                }
            } else {
                save(-1);
                const std::int32_t arm = cell();
                if (arm >= 0 && static_cast<std::size_t>(arm) < n.want) replay(ctx, n.arms[static_cast<std::size_t>(arm)], tape);
            }
            break;
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
    if ((d.flags & ~EG_COND_NEGATE) != 0)
        refuse(EG_BAD_REQUEST, "eg_cond_desc.flags: unknown bits " + std::to_string(d.flags & ~EG_COND_NEGATE));
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
    if (d.kind == EG_COND_SWITCH && (d.flags & EG_COND_NEGATE) != 0)
        refuse(EG_BAD_REQUEST, "eg_cond_desc.flags: EG_COND_NEGATE is not for SWITCH, whose cell is an arm index");
    if (d.max_iters < 0 || (d.kind != EG_COND_WHILE && d.max_iters != 0))
        refuse(EG_BAD_REQUEST, "eg_cond_desc.max_iters " + std::to_string(d.max_iters) + ": WHILE only, 0 or more");
    if (d.kind == EG_COND_SWITCH && d.baseline != nullptr)
        refuse(EG_BAD_REQUEST, "eg_cond_desc.baseline: not for SWITCH, whose cell is an arm index");
    check_handle(ctx.planes, d.cell, "eg_cond_desc.cell");
    const DLTensor& t = d.cell->tensor->dl_tensor;
    const auto check_element = [](const DLTensor& plane, std::int64_t index, const char* what) {
        if (index < 0 || index >= plane.shape[0])
            refuse(EG_BAD_REQUEST, std::string(what) + "_index " + std::to_string(index) + " is outside the plane's " + std::to_string(plane.shape[0]) + " element(s)");
    };
    if ((t.dtype.code != kDLInt && t.dtype.code != kDLUInt) || t.dtype.bits != 32 || t.dtype.lanes != 1 || t.ndim != 1)
        refuse(EG_BAD_REQUEST, "eg_cond_desc.cell: a one-dimensional int32 or uint32 plane is expected");
    check_element(t, d.cell_index, "eg_cond_desc.cell");
    if (d.baseline != nullptr) {
        check_handle(ctx.planes, d.baseline, "eg_cond_desc.baseline");
        const DLTensor& b = d.baseline->tensor->dl_tensor;
        if (b.dtype.code != t.dtype.code || b.dtype.bits != 32 || b.dtype.lanes != 1 || b.ndim != 1)
            refuse(EG_BAD_REQUEST, "eg_cond_desc.baseline: a one-dimensional plane of the dtype of the cell is expected");
        check_element(b, d.baseline_index, "eg_cond_desc.baseline");
    }
    Node n;
    n.kind = Node::Cond;
    n.cond = d.kind;
    n.cell = d.cell;
    n.cell_index = d.cell_index;
    n.baseline = d.baseline;
    n.baseline_index = d.baseline_index;
    n.want = want;
    n.negate = (d.flags & EG_COND_NEGATE) != 0;
    n.max_iters = d.max_iters;
    n.arms.emplace_back();                      // the first arm is being recorded
    record(ctx, std::move(n));
    Node* test = &rec.current().back();
    rec.shape.cond_begin(d.kind, want);
    rec.conds.push_back(test);
}

void record_cond_arm(eg_ctx& ctx) {
    eg_recording_s& rec = recording(ctx);
    rec.shape.cond_arm();
    rec.conds.back()->arms.emplace_back();
}

void record_cond_end(eg_ctx& ctx) {
    eg_recording_s& rec = recording(ctx);
    rec.shape.cond_end();
    rec.conds.pop_back();
}

void record_fork_begin(eg_ctx& ctx) { recording(ctx).shape.fork_begin(); }

void record_fork_branch(eg_ctx& ctx) { recording(ctx).shape.fork_branch(); }

void record_fork_end(eg_ctx& ctx) {
    eg_recording_s& rec = recording(ctx);
    const std::string race = rec.shape.fork_end();
    if (!race.empty()) refuse(EG_BAD_REQUEST, race);    // the fork stays open, as before the call
}

void record_abort(eg_ctx& ctx) {
    release(ctx, recording(ctx).top);
    delete ctx.rec;
    ctx.rec = nullptr;
}

void record_wait(eg_ctx& ctx, std::int32_t n_planes, eg_plane_s* const* planes) {
    eg_recording_s& rec = recording(ctx);
    if (n_planes < 1 || planes == nullptr) refuse(EG_BAD_REQUEST, "eg_record_wait: n_planes " + std::to_string(n_planes) + " without planes");
    for (std::int32_t k = 0; k < n_planes; ++k) check_handle(ctx.planes, planes[k], ("eg_record_wait: planes[" + std::to_string(k) + "]").c_str());
    rec.shape.wait(std::vector<eg_plane_s*>(planes, planes + n_planes));
}

void record_end(eg_ctx& ctx, const eg_record_end_desc* desc, eg_graph** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_record_end: out is NULL");
    *out = nullptr;
    eg_recording_s& rec = recording(ctx);
    if (rec.shape.open()) refuse(EG_BAD_REQUEST, "eg_record_end: a conditional or fork is still open");
    std::unique_ptr<GraphDiff> diff;
    if (desc != nullptr) {
        const eg_record_end_desc d = read_input(desc, "eg_record_end_desc");
        if ((d.flags & ~EG_RECORD_DIFF) != 0) refuse(EG_BAD_REQUEST, "eg_record_end_desc.flags: unknown bits " + std::to_string(d.flags & ~EG_RECORD_DIFF));
        if ((d.flags & EG_RECORD_DIFF) != 0) {      // a refusal leaves the recording open
            std::vector<eg_plan_s*> launches;       // by launch ordinal: the tree's launches in recording order
            std::vector<const Node*> others;        // the copies, reduces, transforms and products, in recording order
            const std::function<void(const std::vector<Node>&)> collect = [&](const std::vector<Node>& nodes) {
                for (const Node& n : nodes) {
                    if (n.kind == Node::Launch) launches.push_back(n.plan);
                    else if (n.kind != Node::Cond) others.push_back(&n);
                    for (const std::vector<Node>& arm : n.arms) collect(arm);
                }
            };
            collect(rec.top);
            diff = diff_plan(ctx, launches, others, rec.shape.reverse_plan(), d);
        }
        else if (d.n_vjps != 0) refuse(EG_BAD_REQUEST, "eg_record_end_desc.n_vjps: vjps without EG_RECORD_DIFF");
    }
    auto graph = std::make_unique<eg_graph_s>();
    graph->nodes = std::move(rec.top);
    graph->shape = std::move(rec.shape);
    graph->diff = std::move(diff);
    ctx.graphs.insert(graph.get());
    delete ctx.rec;
    ctx.rec = nullptr;
    *out = graph.release();
}

void graph_launch(eg_ctx& ctx, eg_graph_s* graph) {
    check_handle(ctx.graphs, graph, "graph");
    if (graph->diff == nullptr) {
        replay(ctx, graph->nodes);
        return;
    }
    GraphDiff& diff = *graph->diff;
    diff_check_launch(diff);
    diff.launched = false;
    for (TestSave& t : diff.saved) t = TestSave();
    Tape tape{ &diff, {} };
    replay(ctx, graph->nodes, &tape);
    diff_launched(diff);
}

void graph_describe(eg_ctx& ctx, const eg_graph_s* graph, eg_graph_description* out) {
    check_handle(ctx.graphs, const_cast<eg_graph_s*>(graph), "graph");
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_graph_describe: out is NULL");
    if (out->struct_size < sizeof(eg_graph_description))
        refuse(EG_BAD_REQUEST, "eg_graph_description: struct_size " + std::to_string(out->struct_size) + " is below the door's " + std::to_string(sizeof(eg_graph_description)) + "; the node and edge sizes it ends with are needed to step the arrays");
    if (graph->described == nullptr) graph->described = std::make_unique<DescribeStore>();
    eg_graph_description d;
    graph->shape.describe(*graph->described, graph->diff != nullptr ? EG_RECORD_DIFF : 0, d);
    write_output(out, d, "eg_graph_description");
}

void graph_free(eg_ctx& ctx, eg_graph_s* graph) {
    check_handle(ctx.graphs, graph, "graph");
    ctx.graphs.erase(graph);
    release(ctx, graph->nodes);
    if (graph->diff != nullptr) diff_free(ctx, *graph->diff);
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
