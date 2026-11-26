// SPDX-License-Identifier: Apache-2.0

// The device group: a context's device and its own stream, a unit's device
// module and function, and one kernel launch per partition of a DEVICE plan,
// queued on the context's stream. Every driver call goes through eagle's
// driver entry (exec/DeviceKernel.h), with the context's primary context
// current for the call only. The host-only build refuses EG_CTX_DEVICE and
// structure DEVICE.
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/door/host/Graph.h"
#include "eagle/door/host/Ops.h"
#include "eagle/door/host/Segment.h"

#include <atomic>

#ifndef EAGLE_CPU_ONLY
#include "eagle/door/device/Fft.h"
#include "eagle/door/device/Gemm.h"
#include "eagle/door/device/Loop.h"
#include "eagle/exec/DeviceKernel.h"
#include "eagle/graph/Edges.h"
#include "eagle/policy/Launch.h"
#include "eagle/policy/Residency.h"

#include <algorithm>
#include <map>
#include <set>
#include <chrono>
#include <limits>
#include <memory>
#include <cstring>

#include <dlfcn.h>
#endif

namespace eagle {
namespace door {

namespace {

/** @brief The streams the calling thread has in scope: one entry per context that thread gave a stream other than its default. */
std::unordered_map<std::uint64_t, std::uint64_t>& thread_streams() {
    thread_local std::unordered_map<std::uint64_t, std::uint64_t> streams;
    return streams;
}

std::atomic<std::uint64_t> streams_in_use_serial{ 0 };

}  // namespace

ThreadStream::ThreadStream() : serial_(++streams_in_use_serial) {}

ThreadStream::operator std::uint64_t() const {
    const auto& mine = thread_streams();
    const auto at = mine.find(serial_);
    return at != mine.end() ? at->second : default_;
}

ThreadStream& ThreadStream::operator=(std::uint64_t stream) {
    if (stream == default_)
        thread_streams().erase(serial_);
    else
        thread_streams()[serial_] = stream;
    return *this;
}

void ThreadStream::set_default(std::uint64_t stream) {
    default_ = stream;
    thread_streams().erase(serial_);
}

#ifdef EAGLE_CPU_ONLY

void device_open(eg_ctx&, std::int32_t device) {
    refuse(EG_NOT_WRITTEN, "eg_ctx_settings EG_CTX_DEVICE (device " + std::to_string(device) +
           "): the device group is not built");
}
void device_close(eg_ctx&) noexcept {}
std::int32_t device_count(eg_ctx&) { return 0; }
void device_props(std::int32_t, struct eg_device_props&) { refuse(EG_NOT_WRITTEN, "eg_device_props: the device group is not built"); }
void device_unit_load(eg_unit_s&, const eg_unit_meta&, void*) {}
void device_unit_free(eg_unit_s&) noexcept {}
void device_plan_check(const eg_ctx&, const eg_unit_s&) {
    refuse(EG_NOT_WRITTEN, "structure DEVICE: the device group is not built");
}
void device_gemm_ready(eg_ctx&) {}
void device_gemm_run(eg_ctx&, const eg_gemm_s&, eg_plane_s*, eg_plane_s*, eg_plane_s*) {}
void device_launch(eg_plan_s&) {}
void device_launch_call(eg_plan_s&) {}
void device_classify(eg_plan_s&) {}
void device_loop_drop(eg_runner_s&) {}
void device_sync(eg_ctx&) {}
void device_plan_cells(eg_plan_s&) {}
void device_plan_free(eg_plan_s&) noexcept {}
void* device_alloc(eg_ctx&, std::size_t, bool) { return nullptr; }
void device_free(eg_ctx&, void*) noexcept {}
void* pinned_alloc(eg_ctx&, std::size_t, bool) { return nullptr; }
void pinned_free(eg_ctx&, void*) noexcept {}
void device_cell_work(eg_plan_s&) {}
void device_read(eg_ctx&, const void*, void*, std::size_t) {}
void device_write(eg_ctx&, void*, const void*, std::size_t) {}
void device_copy_bytes(eg_ctx&, void*, const void*, std::size_t) {}
void device_zero(eg_ctx&, void*, std::size_t) {}
void device_runner_initial(eg_ctx&, eg_runner_s&) {}
void device_columns(eg_ctx&, int, void*, std::int64_t, const void*, std::int64_t, const std::int64_t*, std::int64_t, std::int64_t,
                    unsigned, std::int64_t) {}
void device_launch_prefix(eg_plan_s&, std::int64_t, std::int64_t) {}
void device_mask_count(eg_ctx&, const void*, std::int64_t, unsigned, std::uint64_t) {}
void device_pair_sum(eg_ctx&, bool, const void*, std::int64_t, void*, void*) {}
bool device_active_compact(eg_ctx&, eg_runner_s&, const void*, const std::vector<Owned>&, bool) { return false; }
void device_active_restore(eg_ctx&, eg_runner_s&, const void*, const std::vector<Owned>&) {}
std::int64_t device_free_bytes(eg_ctx&) { return 0; }
void device_event_op(eg_ctx&, unsigned, unsigned, const void*) {}
void device_reduce(eg_ctx&, const eg_reduce_desc&) {}
void device_scan(eg_ctx&, const eg_scan_desc&) { device_group_check("eg_scan"); }
void device_compact(eg_ctx&, const eg_compact_desc&) { device_group_check("eg_compact"); }
void device_order_by_key(eg_ctx&, const eg_order_desc&)
{
    device_group_check("eg_order_by_key");
}
void device_reorder(eg_ctx&, const eg_reorder_desc&)
{
    device_group_check("eg_reorder");
}
void device_reduce_leaf(eg_ctx&, const host::ReduceOp&) {}
void device_copy(eg_ctx&, eg_plane_s*, eg_plane_s*) {}
void device_fft_new(eg_ctx&, eg_fft_s&) {}
void device_fft_run(eg_ctx&, eg_fft_s&, eg_plane_s*, eg_plane_s*) {}
void device_fft_free(eg_ctx&, eg_fft_s&) noexcept {}
void device_runner_new(eg_ctx&, const eg_runner_desc*, eg_runner**) {
    refuse(EG_NOT_WRITTEN, "structure DEVICE: the device group is not built");
}
void device_run(eg_ctx&, eg_runner_s*, eg_run_report*) {}
DeviceRun::~DeviceRun() = default;
void device_record_begin(eg_ctx&) {}
void device_record_launch(eg_ctx&, eg_plan_s&) {}
void device_record_run(eg_ctx&, eg_runner_s*) {}
void device_record_host_node(eg_ctx&, Node&&) {}
void device_record_host_step(eg_ctx&, void (*)(void*, StepFailure&) noexcept, void*, void (*)(eg_ctx&, void*)) {}
void device_record_end(eg_ctx&, const eg_record_end_desc*, eg_graph**) {}
void device_record_abort(eg_ctx&) {}
void device_record_cond_begin(eg_ctx&, const eg_cond_desc*) {}
void device_record_cond_arm(eg_ctx&) {}
void device_record_cond_end(eg_ctx&) {}
void device_record_fork_begin(eg_ctx&) {}
void device_record_fork_branch(eg_ctx&) {}
void device_record_fork_end(eg_ctx&) {}
void device_record_wait(eg_ctx&, std::int32_t, eg_plane_s* const*) {}
bool device_cond_arms_supported() { return false; }
void device_diff_read_saved(eg_ctx&, GraphDiff&) {}
GraphShape& device_shape(eg_ctx&) { refuse(EG_NOT_WRITTEN, "the device group is not built"); }
void device_graph_describe(eg_ctx&, const eg_graph*, eg_graph_description*) {}
GraphDiff* device_graph_diff(eg_ctx&, eg_graph*) { return nullptr; }
bool device_graph(const eg_ctx&, const eg_graph*) { return false; }
void device_graph_launch(eg_ctx&, eg_graph*) {}
void device_graph_free(eg_ctx&, eg_graph*) {}
void device_graph_retain(eg_graph*) {}
void device_graph_release(eg_ctx&, eg_graph*) {}
void device_graphs_drop(eg_ctx&) noexcept {}
void device_stream_use(eg_ctx&, std::uint64_t) { device_group_check("eg_stream_use"); }
void device_stream_new(eg_ctx&, const std::string&, std::uint64_t*) { device_group_check("eg_stream_new"); }
void device_stream_free(eg_ctx&, std::uint64_t) { device_group_check("eg_stream_free"); }
void device_stream_sync(eg_ctx&, std::uint64_t) { device_group_check("eg_stream_sync"); }
void device_event_new(eg_ctx&, eg_event**) { device_group_check("eg_event_new"); }
void device_event_free(eg_ctx&, eg_event*) { device_group_check("eg_event_free"); }
void device_event_record(eg_ctx&, eg_event*) { device_group_check("eg_event_record"); }
void device_event_wait(eg_ctx&, eg_event*) { device_group_check("eg_event_wait"); }
void device_event_elapsed(eg_ctx&, const eg_event*, const eg_event*, double*) { device_group_check("eg_event_elapsed"); }
void device_group_check(const char* call) {
    refuse(EG_NOT_WRITTEN, std::string(call) + ": the device group is not built");
}
void device_quiet(const eg_ctx&, const char*, eg_status, const char*) {}

#else

namespace {

using cuda::detail::driver_entry;
using cuda::detail::driver_error_text;

#define EG_DRIVER(fn) static const auto fn##_ = driver_entry<decltype(&::fn)>(#fn)

/** @brief Refuse EG_FAULT naming @p what and the driver's text unless @p r is CUDA_SUCCESS. */
void check(CUresult r, const std::string& what) {
    if (r != CUDA_SUCCESS) refuse(EG_FAULT, what + " failed: " + driver_error_text(r));
}

/** @brief The primary context of @p dev current on this thread for one scope (pushed, then popped). */
struct Current {
    explicit Current(const DeviceState& dev) {
        EG_DRIVER(cuCtxPushCurrent);
        check(cuCtxPushCurrent_(reinterpret_cast<CUcontext>(dev.context)), "cuCtxPushCurrent");
    }
    ~Current() {
        EG_DRIVER(cuCtxPopCurrent);
        CUcontext popped = nullptr;
        cuCtxPopCurrent_(&popped);
    }
    Current(const Current&) = delete;
    Current& operator=(const Current&) = delete;
};

/** @brief One attribute of @p dev. */
int attribute(CUdevice dev, CUdevice_attribute a, const char* what) {
    EG_DRIVER(cuDeviceGetAttribute);
    int v = 0;
    check(cuDeviceGetAttribute_(&v, a, dev), std::string("cuDeviceGetAttribute (") + what + ")");
    return v;
}

/**
 * @brief The block of a launch over @p n samples when the meta states none: the old path's
 * ``eagle.launch._resolved_block`` with no declared block, no caller block and no sibling
 * launch, i.e. ``_launch_policy.resolve_block`` with ``siblings`` 1:
 * clamp(32 * ceil(n / (2 * SMs * 32)), 32, min(256, the function's thread limit)), narrowed to the
 * largest warp multiple one SM holds at the function's registers; 256 when a fact is missing.
 */
unsigned resolved_block(const DeviceState& d, const eg_unit_s& u, std::int64_t n) {
    constexpr int warp = 32, fallback = 256;
    if (n <= 0 || d.sms <= 0 || d.regs_per_sm <= 0 || d.smem_per_sm <= 0 || u.max_threads <= 0 || u.regs <= 0)
        return fallback;
    const int upper = std::min(fallback, u.max_threads / warp * warp);
    if (upper < warp) return fallback;
    const std::int64_t denom = 2LL * d.sms * warp;
    const std::int64_t ideal = warp * ((n + denom - 1) / denom);
    const policy::ResidencyDevice r{ d.cc_major, d.cc_minor, d.regs_per_sm, d.threads_per_sm };
    for (int bs = static_cast<int>(std::min<std::int64_t>(std::max<std::int64_t>(ideal, warp), upper)); bs >= warp; bs -= warp)
        if (policy::residentBlocks(r, u.regs, bs) >= 1) return static_cast<unsigned>(bs);
    return fallback;
}

/** @brief The block of a launch of @p u over @p n samples: the meta's, else the one resolved for @p n. */
unsigned launch_block(const DeviceState& d, const eg_unit_s& u, std::int64_t n) {
    return u.block > 0 ? static_cast<unsigned>(u.block) : resolved_block(d, u, n);
}

/** @brief The threads of the block that serves one group of @p u: the group's size bound rounded up to whole warps, at most 256 and what
 *  the kernel can launch with; 256 when the bound is not stated. A group larger than the block is strided over its threads. */
unsigned group_threads(const eg_unit_s& u) {
    constexpr std::int64_t warp = 32, widest = 256;
    const std::int64_t bound = u.group_record.max_size;
    const std::int64_t wanted = bound > 0 ? std::min(widest, (bound + warp - 1) / warp * warp) : widest;
    return static_cast<unsigned>(std::max(warp, std::min(wanted, static_cast<std::int64_t>(u.max_threads) / warp * warp)));
}

/** @brief The dynamic shared memory one group of @p u asks for: each scratch plane's bytes rounded up to eight, the carve the unit's text makes.
 *  Refuses by name a group whose scratch does not fit the card's shared memory per block. */
unsigned group_shared_bytes(const DeviceState& d, const eg_unit_s& u) {
    std::int64_t bytes = 0;
    for (const GroupScratchFacts& plane : u.group_record.scratch) bytes += (plane.bytes + 7) / 8 * 8;
    if (bytes > d.smem_per_block)
        refuse(EG_NOT_WRITTEN, "unit '" + u.name + "' on DEVICE: a group's scratch is " + std::to_string(bytes) +
               " bytes and the card's shared memory per block is " + std::to_string(d.smem_per_block) + "; scratch in a device plane is not written yet");
    return static_cast<unsigned>(bytes);
}

/** @brief Launch @p u over the groups of @p p, a block per group: the partition's base is its first group, its count the groups. */
void launch_groups(const DeviceState& d, const eg_unit_s& u, const std::vector<void*>& args, const exec::Partition& p, CUstream stream) {
    if (p.count <= 0) return;
    EG_DRIVER(cuLaunchKernel);
    std::int64_t base = p.base, count = p.count, n = p.nSamples;
    std::vector<void*> params = args;
    params.push_back(&base);
    params.push_back(&count);
    params.push_back(&n);
    check(cuLaunchKernel_(reinterpret_cast<CUfunction>(u.function), static_cast<unsigned>(p.count), 1, 1, group_threads(u), 1, 1,
                          group_shared_bytes(d, u), stream, params.data(), nullptr),
          "unit '" + u.name + "': the group launch");
}

/** @brief Keep the geometry of the launch @p plan's first partition gets, the figures eg_runner_figures reads. */
void note_launch(const DeviceState& d, const eg_plan_s& plan, DeviceRun& r) {
    if (plan.unit->function == 0 || plan.parts.empty()) return;    // a split unit launches per stage
    const std::int64_t count = plan.parts[0].count;
    if (plan.unit->group_record.present) {
        r.block = group_threads(*plan.unit);
        r.grid = static_cast<unsigned>(count);
        return;
    }
    r.block = launch_block(d, *plan.unit, count);
    r.grid = exec::DeviceKernel::grid(count, r.block);
}

/** @brief The loop iterations that cover @p max_steps at @p every steps each. */
std::int64_t loop_cap(std::int64_t max_steps, std::int64_t every) { return max_steps / every + (max_steps % every != 0); }

/** @brief @p owned as device addresses. */
std::vector<DevicePlane> planes_of(const std::vector<Owned>& owned) {
    std::vector<DevicePlane> out;
    for (const Owned& o : owned) out.push_back({ o.data, static_cast<unsigned>(o.bytes) });
    return out;
}

/** @brief The planes a reorder of @p runner moves, as device addresses (the mask first). */
std::vector<DevicePlane> reorder_table(const eg_runner_s& runner) { return planes_of(owned_planes(runner)); }

/** @brief Queue on @p s the compaction of @p mask into the runner's device map and count, with the reorder trigger of a reordering set. */
void queue_compact(const eg_runner_s& runner, std::uint64_t s, const void* mask) {
    const DeviceRun& r = runner.dev;
    const ActiveState& a = runner.act;
    auto* rc = reinterpret_cast<unsigned*>(r.rcells);
    compact_step(s, mask, a.n, reinterpret_cast<std::int32_t*>(r.map), reinterpret_cast<unsigned*>(r.cells) + 3,
                 reinterpret_cast<void*>(r.scratch), a.reorder ? rc : nullptr, a.reorder ? rc + 1 : nullptr,
                 a.reorder ? rc + 2 : nullptr, static_cast<float>(a.theta));
}

/** @brief Queue on @p s the reorder of @p planes by @p mask (staged in @p scratch). */
void queue_reorder(const eg_runner_s& runner, std::uint64_t s, const void* mask, const std::vector<DevicePlane>& planes, void* scratch) {
    const DeviceRun& r = runner.dev;
    auto* rc = reinterpret_cast<unsigned*>(r.rcells);
    reorder_step(s, planes, mask, reinterpret_cast<std::int32_t*>(r.perm), reinterpret_cast<std::int32_t*>(r.inv),
                 reinterpret_cast<std::int32_t*>(r.map), reinterpret_cast<unsigned*>(r.cells) + 3, rc + 1, rc + 2, scratch,
                 runner.act.n, rc + 3);
}

/** @brief Queue on @p s every plane of @p planes back in sample order, the span reset, the map recomputed from the restored mask. */
void queue_restore(const eg_runner_s& runner, std::uint64_t s, const void* mask, const std::vector<DevicePlane>& planes, void* scratch) {
    const DeviceRun& r = runner.dev;
    const std::int64_t n = runner.act.n;
    restore_step(s, planes, reinterpret_cast<std::int32_t*>(r.perm), reinterpret_cast<std::int32_t*>(r.inv), scratch, n);
    const unsigned span_fire[2] = { static_cast<unsigned>(n), 0 };
    cells_write(s, r.rcells + sizeof(unsigned), span_fire, 2);
    queue_compact(runner, s, mask);
    const unsigned zero = 0;
    cells_write(s, r.rcells + 2 * sizeof(unsigned), &zero, 1);
}

/** @brief A zeroed device array of @p count uint32 cells (its address) from device_alloc. */
std::uint64_t cells_new(eg_ctx& ctx, unsigned count) {
    return reinterpret_cast<std::uint64_t>(device_alloc(ctx, count * sizeof(unsigned), true));
}

/** @brief Free cells made by cells_new, on the stream they were made on (0-safe). */
void cells_free(eg_ctx& ctx, std::uint64_t cells) noexcept { device_free(ctx, reinterpret_cast<void*>(cells)); }

/** @brief The runner's reorder scratch, grown to stage @p planes (a recorded run moves eagle's own planes too). */
void* reorder_scratch(DeviceRun& r, std::int64_t n, const std::vector<DevicePlane>& planes) {
    const std::size_t need = reorder_scratch_bytes(n, planes);
    if (need > r.rscratch_bytes) {
        cells_free(*r.ctx, r.rscratch);
        r.rscratch = cells_new(*r.ctx, static_cast<unsigned>((need + sizeof(unsigned) - 1) / sizeof(unsigned)));
        r.rscratch_bytes = need;
    }
    return reinterpret_cast<void*>(r.rscratch);
}

}  // namespace

/** @brief A capture in progress: the stream it is on, the plans it launched (one reference each) with their
 *  argument bytes at that launch, and the device frees asked for meanwhile, run when it ends. */
/** @brief A host node of a device graph: the function the driver calls at each replay and what it runs. */
struct HostHook {
    void (*run)(void*, StepFailure&) noexcept;
    void* user;
    void (*release)(eg_ctx&, void*);
    std::shared_ptr<StepFailure> failure;
};

/** @brief An open conditional (``weave``) or fork (``fork``) of the recording; ``outer`` is the stream in scope when it opened. */
struct Frame {
    DeviceWeave* weave = nullptr;
    DeviceFork* fork = nullptr;
    unsigned arm = 0, arms = 0;                         ///< a conditional's open arm and arm count; a fork's branch count is ``arms``
    const unsigned* iteration = nullptr;                ///< a WHILE: its iteration counter word; its cap
    unsigned cap = 0;
    unsigned* counter = nullptr;                        ///< a WHILE: the caller's counter word, or null
    std::uint64_t outer = 0;
};

/** @brief The loop states a recording keeps for its WHILE conditionals: two device words each, made before the capture opens
 *  (nothing can be allocated inside it). */
constexpr unsigned kRecordedLoops = 64;

/** @brief The streams and events a recording's conditionals and forks can hold at once, made before the capture opens. */
constexpr unsigned kRecordStreams = 32, kRecordEvents = 128;

/** @brief One call a recording kept to capture again: a launch, a copy, reduce, transform or product (``leaf``), or a structure call (with its description). */
struct Op {
    enum Kind { Launch, Leaf, CondBegin, CondArm, CondEnd, ForkBegin, ForkBranch, ForkEnd, Wait } kind = Launch;
    eg_plan_s* plan = nullptr;
    eg_cond_desc cond{};
    std::vector<eg_plane_s*> planes;
    Op(Kind k = Launch, eg_plan_s* p = nullptr, eg_cond_desc c = {}, std::vector<eg_plane_s*> pl = {}) : kind(k), plan(p), cond(c), planes(std::move(pl)) {}
    Node leaf;                                          ///< a Leaf: the checked call; a product's ``gemm`` is left unset, ``product`` is its copy
    eg_gemm_s product;
};

/** @brief The tests of one conditional (its node ordinal ``cond``) that a replay saves: ``capacity`` entries (one, or one per iteration of the WHILE
 *  around or at it) of three words each: evaluated, the cell, the baseline. */
struct DiffSlot {
    int cond = -1;
    unsigned capacity = 1;
    unsigned* buffer = nullptr;
};

/** @brief A plane copied to ``dst`` just before a launch: directly, or into ``table[iteration]`` inside a WHILE body. */
struct LeafCopy {
    const void* plane = nullptr;
    std::size_t bytes = 0;
    void* direct = nullptr;
    void* const* table = nullptr;
    unsigned capacity = 0;
    unsigned window = 1;                                ///< a table copy happens at the iterations that are multiples of this, into ``table[iteration / window]``
};

/** @brief What the second capture of a EG_RECORD_DIFF recording adds: the saved tests by conditional ordinal and the copies in front of each launch. */
struct DiffPass {
    std::size_t launches = 0;                           ///< the launches of the recording: the other leaves follow them in the tape's nodes
    std::vector<DiffSlot> slots;
    std::vector<std::vector<LeafCopy>> before;          ///< per launch ordinal
    std::map<int, std::vector<LeafCopy>> checkpoints;   ///< per windowed WHILE (its node ordinal): the planes its body writes, kept at the start of each window
};

/** @brief One step of a device graph that holds a run or a host call: an instantiated capture, the run of a DEVICE runner or a host call that
 *  eg_graph_launch runs eagerly between two captures (a run reads back and synchronizes, a host call is the caller's code, which a captured graph
 *  cannot hold). Exactly one member is set. */
struct Piece {
    DeviceLoop* loop = nullptr;
    eg_runner_s* runner = nullptr;                      ///< one reference, held by the recording and then by the graph
    std::shared_ptr<Node> host;                         ///< a host call, holding its references (node_hold) until the graph drops the step
};

struct DeviceRecording {
    std::vector<Piece> pieces;                          ///< the steps closed so far: the capture in progress is the one after them
    std::vector<std::size_t> plan_piece;                ///< per recorded launch (as ``plans``): the index in ``pieces`` its capture will have
    std::vector<Frame> frames;                          ///< the open conditionals and forks, outermost first
    GraphShape shape;                                   ///< the nodes, forks and conditionals recorded: races, waits and the description
    void* loop_cells = nullptr;                         ///< kRecordedLoops pairs of device words
    DevicePool* pool = nullptr;                         ///< the arm and branch streams and the events of its forks
    unsigned loops = 0;                                 ///< the pairs used
    std::vector<Op> log;                                ///< the structure calls, launches, copies, reduces, transforms and products, kept to capture them again for EG_RECORD_DIFF
    unsigned others = 0;                                ///< the copies, reduces, transforms and products logged so far
    DiffPass* dp = nullptr;                             ///< the second capture of a EG_RECORD_DIFF recording: what it adds
    std::vector<std::vector<std::uint64_t>> kernels;    ///< per recorded launch (as ``plans``): the kernel nodes of its partitions
    std::vector<std::unique_ptr<HostHook>> hooks;       ///< the host nodes captured so far
    std::shared_ptr<StepFailure> failure;               ///< shared by the hooks and, once recorded, the graph
    DeviceCapture* capture = nullptr;
    std::uint64_t stream = 0;
    std::vector<eg_plan_s*> plans;
    std::vector<eg_fft_s*> ffts;    ///< the transform plans run while capturing, one reference each
    std::vector<std::vector<unsigned char>> bound;
    std::vector<std::function<void()>> later;
};

/** @brief A recorded device graph: the instantiated capture and the plans whose bindings it holds by value. */
struct DeviceGraph {
    int refs = 1;                                       ///< the caller's reference plus one per runner that runs it as its step graph
    bool caller_ref = true;
    std::vector<void*> buffers;                         ///< device memory the graph owns beside its tape (save slots, copy tables), freed with it
    std::vector<DiffSlot> slots;                        ///< a EG_RECORD_DIFF graph's saved tests
    unsigned loops = 0;                                 ///< its WHILE conditionals
    GraphShape shape;                                   ///< what eg_graph_describe returns
    mutable DescribeStore described;                    ///< the arrays of the last description
    void* loop_cells = nullptr;                         ///< the loop states of its WHILE conditionals, freed with it
    std::vector<std::vector<std::uint64_t>> kernels;    ///< per recorded launch (as ``plans``): its kernel nodes, to re-apply a rebound plan's arguments
    std::unique_ptr<GraphDiff> diff;                    ///< the tape of a graph ended with EG_RECORD_DIFF
    std::vector<std::unique_ptr<HostHook>> hooks;       ///< its host nodes, each holding the references its step runs
    std::shared_ptr<StepFailure> failure;               ///< the first failure of a host node, kept for the next launch or sync
    std::vector<Piece> pieces;                          ///< its steps in launch order: a graph without a run has one capture
    std::vector<std::size_t> plan_piece;                ///< per recorded launch (as ``plans``): the capture in ``pieces`` that holds its kernel nodes
    std::vector<eg_plan_s*> plans;
    std::vector<eg_fft_s*> ffts;    ///< the transform plans it runs, one reference each
    std::vector<std::vector<unsigned char>> bound;
};

namespace {

/** @brief Run @p free now, or once the capture in progress on @p ctx ends; a forked child makes no device call. */
void now_or_after_capture(eg_ctx& ctx, std::function<void()> free) noexcept {
    if (forked(ctx)) return;
    try {
        if (ctx.dev_rec != nullptr) ctx.dev_rec->later.push_back(std::move(free));
        else free();
    } catch (...) {
    }
}

/** @brief The bytes of @p plan's bound arguments (views, mirrors, uniforms): what a capture holds by value. */
std::vector<unsigned char> binding_bytes(const eg_plan_s& plan) {
    std::vector<unsigned char> bytes;
    for (const eg_plan_s::Binding& b : plan.bindings) {
        const auto* h = reinterpret_cast<const unsigned char*>(&b.handle);
        const auto* m = reinterpret_cast<const unsigned char*>(&b.mirror);
        bytes.insert(bytes.end(), h, h + sizeof(b.handle));
        bytes.insert(bytes.end(), m, m + sizeof(b.mirror));
        bytes.insert(bytes.end(), b.value, b.value + sizeof(b.value));
    }
    return bytes;
}

/** @brief Run the frees a capture held back, under the context's device. */
void run_later(eg_ctx& ctx, DeviceRecording& rec) noexcept {
    for (std::function<void()>& f : rec.later) {
        try {
            f();
        } catch (...) {
        }
    }
    rec.later.clear();
    (void)ctx;
}


/** @brief The driver's host-function call of a host node: run its step. */
void CUDA_CB host_hook_run(void* data) {
    auto* hook = static_cast<HostHook*>(data);
    hook->run(hook->user, *hook->failure);
}

/** @brief Drop what the host nodes @p hooks hold; the stream they ran on is idle or never ran them. */
void release_hooks(eg_ctx& ctx, std::vector<std::unique_ptr<HostHook>>& hooks) noexcept {
    for (std::unique_ptr<HostHook>& h : hooks) {
        try {
            h->release(ctx, h->user);
        } catch (...) {
        }
    }
    hooks.clear();
}

/** @brief Wait until the context's stream is idle, so no host node runs after its references are dropped. */
void drain_stream(eg_ctx& ctx) {
    EG_DRIVER(cuStreamSynchronize);
    Current current(ctx.dev);
    check(cuStreamSynchronize_(reinterpret_cast<CUstream>(static_cast<std::uint64_t>(ctx.dev.stream))), "cuStreamSynchronize");
}

/** @brief Refuse FAULT with the first failure a host node of @p failure kept, and clear it. */
void raise_step_failure(StepFailure& failure) {
    std::string why;
    if (failure.take(why)) refuse(EG_FAULT, "a host step of a device graph failed: " + why);
}

}  // namespace

/** @brief The devices the driver sees, or -1 with @p why (the runtime's words) when no driver can be opened. A driver
 *  whose cuInit reports no device sees 0; any other failure after the driver opened is a FAULT. */
std::int32_t driver_device_count(std::string& why) {
    try {
        EG_DRIVER(cuInit);
        EG_DRIVER(cuDeviceGetCount);
        const CUresult init = cuInit_(0);
        if (init == CUDA_ERROR_NO_DEVICE) return 0;
        check(init, "cuInit");
        int count = 0;
        check(cuDeviceGetCount_(&count), "cuDeviceGetCount");
        return count;
    } catch (const std::runtime_error& e) {
        if (dynamic_cast<const Refusal*>(&e) != nullptr) throw;
        why = e.what();
        return -1;
    }
}

/** @brief eg_device_count: the devices the driver sees, 0 when there is no usable driver. */
std::int32_t device_count(eg_ctx&) {
    std::string why;
    return std::max<std::int32_t>(0, driver_device_count(why));
}

/** @brief eg_device_props: the raw attributes of device @p device, then the derived fields by eagle::deriveFields (the
 *  formulas the dispatch rule reads). */
void device_props(std::int32_t device, struct eg_device_props& out) {
    std::string why;
    const std::int32_t count = driver_device_count(why);
    if (count < 0) refuse(EG_BAD_REQUEST, "eg_device_props: no CUDA driver could be opened (" + why + ")");
    if (device < 0 || device >= count)
        refuse(EG_BAD_REQUEST, "eg_device_props: device " + std::to_string(device) + ": the driver sees " + std::to_string(count) + " device(s)");
    EG_DRIVER(cuDeviceGet);
    EG_DRIVER(cuDeviceGetName);
    CUdevice dev = 0;
    check(cuDeviceGet_(&dev, device), "cuDeviceGet");
    eagle::DeviceProps p;
    p.cc_major = attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "compute capability");
    p.cc_minor = attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "compute capability");
    p.sm_count = attribute(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "SMs");
    p.clock_rate_khz = attribute(dev, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, "clock rate");
    p.memory_clock_rate_khz = attribute(dev, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, "memory clock rate");
    p.memory_bus_width_bits = attribute(dev, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, "memory bus width");
    p.shared_mem_per_block = static_cast<std::size_t>(attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, "shared memory per block"));
    p.shared_mem_per_sm = static_cast<std::size_t>(attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "shared memory per SM"));
    p.regs_per_block = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK, "registers per block");
    p.regs_per_sm = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, "registers per SM");
    p.warp_size = attribute(dev, CU_DEVICE_ATTRIBUTE_WARP_SIZE, "warp size");
    p = eagle::deriveFields(p);
    char name[sizeof(out.name)] = {};
    check(cuDeviceGetName_(name, static_cast<int>(sizeof(name) - 1), dev), "cuDeviceGetName");
    const std::uint32_t size = out.struct_size;
    std::memset(&out, 0, sizeof(out));
    out.struct_size = size;
    out.cc_major = p.cc_major;
    out.cc_minor = p.cc_minor;
    out.sm_count = p.sm_count;
    out.warp_size = p.warp_size;
    out.regs_per_sm = p.regs_per_sm;
    out.regs_per_block = p.regs_per_block;
    out.shared_mem_per_block = static_cast<std::int64_t>(p.shared_mem_per_block);
    out.shared_mem_per_sm = static_cast<std::int64_t>(p.shared_mem_per_sm);
    std::memcpy(out.name, name, sizeof(name));
    out.clock_rate_khz = p.clock_rate_khz;
    out.memory_clock_rate_khz = p.memory_clock_rate_khz;
    out.memory_bus_width_bits = p.memory_bus_width_bits;
    out.peak_bytes_per_s = p.peak_bytes_per_s;
    out.peak_flops_sp = p.peak_flops_sp;
    out.peak_flops_dp = p.peak_flops_dp;
    out.fp64_ratio = p.fp64_ratio;
    out.ridge_flops_per_byte_sp = p.ridgeFlopsPerByte("float32");
    out.ridge_flops_per_byte_dp = p.ridgeFlopsPerByte("float64");
}

/** @brief EG_CTX_DEVICE: retain the device's primary context, make the context's stream, read the launch facts. */
void device_open(eg_ctx& ctx, std::int32_t device) {
    std::string why;
    const std::int32_t count = driver_device_count(why);
    if (count < 0)
        refuse(EG_BAD_REQUEST, "eg_ctx_settings EG_CTX_DEVICE: no CUDA driver could be opened (" + why + ")");
    if (device >= count)
        refuse(EG_BAD_REQUEST, "eg_ctx_settings.device " + std::to_string(device) + ": the driver sees " +
               std::to_string(count) + " device(s)");
    EG_DRIVER(cuDeviceGet);
    EG_DRIVER(cuDevicePrimaryCtxRetain);
    EG_DRIVER(cuStreamCreate);
    DeviceState& d = ctx.dev;
    CUdevice dev = 0;
    check(cuDeviceGet_(&dev, device), "cuDeviceGet");
    CUcontext primary = nullptr;
    check(cuDevicePrimaryCtxRetain_(&primary, dev), "cuDevicePrimaryCtxRetain");
    d.ordinal = device;
    d.handle = dev;
    d.context = reinterpret_cast<std::uint64_t>(primary);
    Current current(d);
    CUstream stream = nullptr;
    check(cuStreamCreate_(&stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
    d.own_stream = reinterpret_cast<std::uint64_t>(stream);
    d.stream.set_default(d.own_stream);
    d.sms = attribute(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "SMs");
    d.regs_per_sm = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, "registers per SM");
    d.smem_per_sm = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "shared memory per SM");
    d.smem_per_block = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, "shared memory per block");
    d.threads_per_sm = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, "threads per SM");
    d.fp64_ratio = std::max(1, attribute(dev, CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, "FP64 ratio"));
    d.cc_major = attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "compute capability");
    d.cc_minor = attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "compute capability");
    d.clock_khz = attribute(dev, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, "clock rate");
    d.memory_clock_khz = attribute(dev, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE, "memory clock rate");
    d.bus_bits = attribute(dev, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH, "memory bus width");
    d.l2_bytes = attribute(dev, CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, "L2 cache size");
}

/** @brief Destroy the context's stream (queued work still finishes) and release the primary context. */
void device_close(eg_ctx& ctx) noexcept {
    DeviceState& d = ctx.dev;
    if (d.context == 0) return;
    try {
        if (!d.gemm_handles.empty()) {
            Current current(d);
            gemm_handles_free(ctx);
        }
        EG_DRIVER(cuStreamDestroy);
        EG_DRIVER(cuEventDestroy);
        Current current(d);
        for (const auto& e : d.events) cuEventDestroy_(reinterpret_cast<CUevent>(e.second));
        for (const auto& s : d.scan_state) scan_state_free(s.second.data);
        for (const auto& m : d.made_streams) cuStreamDestroy_(reinterpret_cast<CUstream>(m.second));    // never the caller's streams
        if (d.own_stream != 0) cuStreamDestroy_(reinterpret_cast<CUstream>(d.own_stream));
        EG_DRIVER(cuDevicePrimaryCtxRelease);
        cuDevicePrimaryCtxRelease_(d.handle);
    } catch (...) {
    }
    d = DeviceState{};
}

/**
 * @brief Load the unit's device image into the context's device and resolve its function. The image's
 * bytes are the symbol ``device_image`` of the library, its length the ``uint64_t`` the symbol
 * ``device_image_size`` holds (else ``device_image_bytes``); PTX, cubin and fatbin load alike. As the
 * old path's ``module_load``, a NUL is appended (PTX text must end in one; a cubin or fatbin ignores it).
 */
void device_unit_load(eg_unit_s& unit, const eg_unit_meta& m, void* library) {
    const std::string who = "unit '" + unit.name + "'";
    if (m.device_function == nullptr)
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + who + " names a device image and no device_function");
    const void* image = ::dlsym(library, m.device_image);
    if (image == nullptr) refuse(EG_FAULT, who + ": the library lacks its device image '" + m.device_image + "'");
    std::uint64_t bytes = m.device_image_bytes;
    if (m.device_image_size != nullptr) {
        const auto* size = static_cast<const std::uint64_t*>(::dlsym(library, m.device_image_size));
        if (size == nullptr) refuse(EG_FAULT, who + ": the library lacks its image size '" + m.device_image_size + "'");
        bytes = *size;
    }
    if (bytes == 0) refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + who + " has a device image of 0 bytes");
    const std::string copy(static_cast<const char*>(image), static_cast<std::size_t>(bytes));
    EG_DRIVER(cuModuleLoadData);
    EG_DRIVER(cuModuleGetFunction);
    EG_DRIVER(cuFuncGetAttribute);
    Current current(unit.ctx->dev);
    CUmodule module = nullptr;
    const CUresult loaded = cuModuleLoadData_(&module, copy.c_str());
    if (loaded == CUDA_ERROR_UNSUPPORTED_PTX_VERSION) {   // the driver's own answer: its JIT is older than the image's PTX
        EG_DRIVER(cuDriverGetVersion);
        int version = 0;
        cuDriverGetVersion_(&version);
        refuse(EG_BAD_REQUEST, who + ": image needs a newer driver (PTX " +
               (m.device_ptx_isa > 0 ? std::to_string(m.device_ptx_isa / 10) + "." + std::to_string(m.device_ptx_isa % 10) : std::string("unstated")) +
               ", compiler " + (m.device_compiler != nullptr ? m.device_compiler : "unstated") + ", driver " +
               std::to_string(version / 1000) + "." + std::to_string(version % 1000 / 10) + ")");
    }
    if (loaded == CUDA_ERROR_NO_BINARY_FOR_GPU)
        refuse(EG_BAD_REQUEST, who + ": no machine code for this device and no loadable PTX");
    check(loaded, who + ": loading the device image (cuModuleLoadData)");
    unit.module = reinterpret_cast<std::uint64_t>(module);
    CUfunction fn = nullptr;
    const CUresult r = cuModuleGetFunction_(&fn, module, m.device_function);
    if (r == CUDA_ERROR_NOT_FOUND)
        refuse(EG_FAULT, who + ": the device image lacks the function '" + m.device_function + "'");
    check(r, who + ": cuModuleGetFunction");
    unit.function = reinterpret_cast<std::uint64_t>(fn);
    check(cuFuncGetAttribute_(&unit.regs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn), who + ": cuFuncGetAttribute");
    check(cuFuncGetAttribute_(&unit.max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, fn), who + ": cuFuncGetAttribute");
    check(cuFuncGetAttribute_(&unit.local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn), who + ": cuFuncGetAttribute");
    check(cuFuncGetAttribute_(&unit.shared_bytes, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn), who + ": cuFuncGetAttribute");
    check(cuFuncGetAttribute_(&unit.const_bytes, CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES, fn), who + ": cuFuncGetAttribute");
    if (unit.finish.present && unit.finish.steps == EG_STEPS_AUTO) {    // the fast entries of an automatic unit
        const std::string base = m.device_function;
        for (const char* suffix : { "_persist", "_range" }) {
            CUfunction e = nullptr;
            const CUresult er = cuModuleGetFunction_(&e, module, (base + suffix).c_str());
            if (er == CUDA_ERROR_NOT_FOUND) continue;
            check(er, who + ": cuModuleGetFunction (" + base + suffix + ")");
            (suffix[1] == 'p' ? unit.persist_fn : unit.range_fn) = reinterpret_cast<std::uint64_t>(e);
            if (suffix[1] == 'p') {
                check(cuFuncGetAttribute_(&unit.persist_regs, CU_FUNC_ATTRIBUTE_NUM_REGS, e), who + ": cuFuncGetAttribute");
                check(cuFuncGetAttribute_(&unit.persist_max_threads, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, e),
                      who + ": cuFuncGetAttribute");
            }
        }
    }
    for (CompanionFacts& c : unit.companions) {
        if (c.device_fold_name.empty()) continue;
        CUfunction fold = nullptr;
        const CUresult f = cuModuleGetFunction_(&fold, module, c.device_fold_name.c_str());
        if (f == CUDA_ERROR_NOT_FOUND)
            refuse(EG_FAULT, who + ": the device image lacks the fold function '" + c.device_fold_name + "'");
        check(f, who + ": cuModuleGetFunction (fold)");
        c.device_fold = reinterpret_cast<std::uint64_t>(fold);
    }
}

/** @brief Unload the unit's device module, if any. */
void device_unit_free(eg_unit_s& unit) noexcept {
    if (unit.module == 0) return;
    const std::uint64_t module = unit.module;
    eg_ctx* ctx = unit.ctx;
    unit.module = unit.function = 0;
    now_or_after_capture(*ctx, [ctx, module] {
        EG_DRIVER(cuModuleUnload);
        Current current(ctx->dev);
        cuModuleUnload_(reinterpret_cast<CUmodule>(module));
    });
}

/** @brief Queue on @p stream the fold of cell @p i of @p plan: the rows [@p first, @p end) (and the compensation rows of a sum) into
 *  @p out, by the canonical tree; with no row the cell takes the op's identity. */
void fold_cell(const eg_plan_s& plan, std::size_t i, const DLTensor& rows, const eg_plane_s* comp, std::int32_t op, void* out, std::uint64_t stream,
               std::int64_t first, std::int64_t end) {
    const std::size_t bytes = rows.dtype.bits / 8;
    if (end <= first) {
        const double low = -std::numeric_limits<double>::infinity(), high = std::numeric_limits<double>::infinity();
        const double value = op == EG_REDUCE_SUM ? 0.0 : op == EG_REDUCE_MAX ? low : high;
        const float value32 = static_cast<float>(value);
        write_bytes(stream, out, bytes == sizeof(float) ? static_cast<const void*>(&value32) : static_cast<const void*>(&value), bytes);
        return;
    }
    const auto at = [&](const void* data) { return const_cast<char*>(static_cast<const char*>(data)) + static_cast<std::size_t>(first) * bytes; };
    cell_fold(stream, at(rows.data), comp != nullptr ? at(comp->tensor->dl_tensor.data) : nullptr, end - first, rows.dtype.bits,
              op == EG_REDUCE_SUM ? 0 : op == EG_REDUCE_MAX ? 1 : 2, out, plan.work[2 * i]->tensor->dl_tensor.data,
              plan.work[2 * i + 1]->tensor->dl_tensor.data);
}

/** @brief Queue one launch per partition of @p plan on @p stream: grid from the partition's count and the
 *  meta's block (else the resolved one), the bound views and uniforms by value, then the triple; then each
 *  companion's fold function over its target and each cell's rows folded into the cell the caller bound
 *  (as the host's ``fold_plan``). A gemm unit the door routes queues its cuBLAS products instead, unless @p products is false
 *  (a runner's loop launches the kernel). */
void launch_on(const eg_plan_s& plan, std::uint64_t stream, bool products = true, std::int64_t live = -1) {
    const DeviceState& d = plan.ctx->dev;
    const eg_unit_s& u = *plan.unit;
    if (products && gemm_routed(plan)) {    // the stated products instead of the kernel
        gemm_launch(plan, stream);
        return;
    }
    const auto s = reinterpret_cast<CUstream>(stream);
    std::size_t group = 0;
    for (const DispatchFacts& f : u.dispatches) {    // an AUTO group's counters eagle made: zeroed in the stream at every launch (nothing reads them)
        if (f.strategy != EG_DISPATCH_AUTO) continue;
        const eg_plane_s* mix = plan.mix[group++];
        if (plan.bindings[static_cast<std::size_t>(f.mix)].plane == nullptr)
            zero_bytes(stream, static_cast<char*>(mix->tensor->dl_tensor.data) + mix->tensor->dl_tensor.byte_offset, static_cast<std::size_t>(f.k) * sizeof(std::uint32_t));
    }
    for (const exec::Partition& p : plan.parts) {
        if (u.group_record.present) launch_groups(d, u, plan.params, p, s);
        else exec::DeviceKernel::run(reinterpret_cast<CUfunction>(u.function), plan.params, p, s, launch_block(d, u, p.count));
    }
    for (const CompanionFacts& c : u.companions) {
        const std::int64_t size = elements(plan.bindings[c.target].plane->tensor->dl_tensor);
        exec::DeviceKernel::run(reinterpret_cast<CUfunction>(c.device_fold), plan.params, exec::Partition{ 0, size, size }, s);
    }
    std::int64_t lo = plan.n_samples, hi = 0;    // the span of the partitions, cut at an index set's live count: the rows a launch wrote lie in it
    for (const exec::Partition& p : plan.parts)
        if (p.count > 0) lo = std::min(lo, p.base), hi = std::max(hi, p.base + p.count);
    if (live >= 0) hi = std::min(hi, live);
    const auto written = rows_written(plan, lo, hi);
    for (std::size_t i = 0; i < u.cells.size(); ++i) {
        const DLTensor& rows = plan.rows[i]->tensor->dl_tensor;
        const DLTensor& out = plan.bindings[u.cells[i].slot].plane->tensor->dl_tensor;
        fold_cell(plan, i, rows, plan.comps[i], u.cells[i].op, static_cast<char*>(out.data) + out.byte_offset, stream, written.first, written.second);
    }
}

/** @brief Refuse a DEVICE plan of @p unit on @p ctx: a host-only context, a device feature this
 *  build does not run (each by name), a unit with no device function. */
void device_plan_check(const eg_ctx& ctx, const eg_unit_s& unit) {
    if (ctx.dev.ordinal < 0)
        refuse(EG_BAD_REQUEST, "structure DEVICE on a host-only context (made without EG_CTX_DEVICE)");
    const std::string who = "unit '" + unit.name + "' on DEVICE: ";
    if (unit.stages.empty() && unit.function == 0) refuse(EG_NOT_WRITTEN, who + "the unit has no device image");
    for (const StageFacts& st : unit.stages)
        if (st.grid == EG_GRID_CHUNKS || st.grid == EG_GRID_TARGETS)
            refuse(EG_NOT_WRITTEN, who + "the stage '" + st.ref_name + "' is an ordered scatter (team grid " + (st.grid == EG_GRID_CHUNKS ? "CHUNKS" : "TARGETS") + "), which is not written on the device");
    for (const TeamPlaneFacts& t : unit.team_planes)
        if (t.unit == EG_UNIT_TARGET) refuse(EG_NOT_WRITTEN, who + "the team plane '" + t.name + "' has TARGET units (an ordered scatter's partial sums), which are not written on the device");
    for (const StageFacts& st : unit.stages)
        if (st.unit->function == 0) refuse(EG_NOT_WRITTEN, who + "the stage '" + st.ref_name + "' has no device image");
    for (const CompanionFacts& c : unit.companions)
        if (c.device_fold == 0)
            refuse(EG_NOT_WRITTEN, who + "the companion '" + unit.slots[c.slot].name + "' has no device fold function");
    for (const StageFacts& st : unit.stages)
        for (const CompanionFacts& c : st.unit->companions)
            if (c.device_fold == 0)
                refuse(EG_NOT_WRITTEN, who + "the companion '" + st.unit->slots[c.slot].name + "' of the stage '" + st.ref_name + "' has no device fold function");
}

/** @brief Make the cuBLAS handle of the stream in scope (see Seam.h). */
void device_gemm_ready(eg_ctx& ctx) {
    Current current(ctx.dev);
    gemm_handle_make(ctx, ctx.dev.stream);
}

namespace {

/** @brief The description kind of the copy, reduce, transform or product kind @p k of a leaf. */
eg_gnode_kind gnode_kind_of(Node::Kind k) {
    return k == Node::Copy ? EG_GNODE_COPY : k == Node::Reduce ? EG_GNODE_REDUCE : k == Node::Fft ? EG_GNODE_FFT : k == Node::Host ? EG_GNODE_HOST : EG_GNODE_GEMM;
}

/** @brief How a unit slot's role touches its plane (graph::PlaneRef access: 0 read, 1 write, 2 in place); -1: not a plane. */
int slot_access(std::int32_t role) {
    if (role == EG_ROLE_MUTABLE || role == EG_ROLE_TERMINATED) return 2;
    if (reads_role(role)) return 0;
    return writes_role(role) ? 1 : -1;
}

/** @brief Record the call of kind @p kind that touches @p uses in the shape of the recording on @p ctx (none: nothing). */
int note_node(eg_ctx& ctx, eg_gnode_kind kind, const std::string& label, std::vector<PlaneUse> uses, const std::string& unit = "", NodeFacts facts = {}) {
    if (ctx.dev_rec == nullptr) return -1;
    uses.erase(std::remove_if(uses.begin(), uses.end(), [](const PlaneUse& u) { return u.plane == nullptr || u.access < 0; }), uses.end());
    return ctx.dev_rec->shape.add(kind, label, unit, std::move(uses), facts);
}

/** @brief Keep the checked copy, reduce, transform or product @p leaf (a product's copy in @p product) in the recording on @p ctx, if any: its shape
 *  node, a log entry to capture it again, and the copies the tape keeps in front of it. Called before the call queues its work. */
void note_leaf(eg_ctx& ctx, const char* label, std::vector<PlaneUse> uses, Node leaf, const eg_gemm_s* product = nullptr);

}  // namespace

/** @brief eg_gemm_run of a device product: queued on the context's stream, in a capture or not. */
void device_gemm_run(eg_ctx& ctx, const eg_gemm_s& gemm, eg_plane_s* a, eg_plane_s* b, eg_plane_s* c) {
    const auto at = [](const eg_plane_s* p) { return static_cast<char*>(p->tensor->dl_tensor.data) + p->tensor->dl_tensor.byte_offset; };
    const auto stride = [](const eg_plane_s* p) { return static_cast<int>(p->tensor->dl_tensor.shape[1]); };
    Node leaf;
    leaf.kind = Node::Gemm;
    leaf.src = a;
    leaf.rhs = b;
    leaf.dst = c;
    note_leaf(ctx, "eg_gemm_run", { { a, 0, -1, "a" }, { b, 0, -1, "b" }, { c, 1, -1, "c" } }, std::move(leaf), &gemm);
    Current current(ctx.dev);
    gemm_product(ctx, ctx.dev.stream, gemm.call, gemm.f32, at(a), stride(a), at(b), stride(b), at(c), stride(c));
}

/** @brief A split unit on the device: the stages in order on the context's stream, each with its argument list (the source slot or team
 *  plane each stage slot joined at load). A team stage is `grid` blocks of `threads * warps` threads with the tail (0, grid, n); a plain
 *  stage launches like the unit would. Planes with a promised fill are refilled first, on the stream. */
void launch_split(const eg_plan_s& plan) {
    const DeviceState& d = plan.ctx->dev;
    const eg_unit_s& u = *plan.unit;
    const auto s = reinterpret_cast<CUstream>(static_cast<std::uint64_t>(d.stream));
    EG_DRIVER(cuMemsetD32Async);
    EG_DRIVER(cuLaunchKernel);
    for (std::size_t k = 0; k < u.team_planes.size(); ++k) {
        const TeamPlaneFacts& t = u.team_planes[k];
        if (!t.fill_promised || t.wide != EG_WIDE_NONE) continue;
        const DLTensor& tensor = plan.team[k]->tensor->dl_tensor;
        std::uint32_t word = 0;
        if (t.dtype == EG_DTYPE_INT32) word = static_cast<std::uint32_t>(static_cast<std::int32_t>(t.fill));
        else if (u.scalar_type == EG_SCALAR_FLOAT32) { const float f = static_cast<float>(t.fill); std::memcpy(&word, &f, sizeof(word)); }
        check(cuMemsetD32Async_(reinterpret_cast<CUdeviceptr>(tensor.data), word, static_cast<std::size_t>(elements(tensor)) * tensor.dtype.bits / 32, s),
              "unit '" + u.name + "': refilling team plane '" + t.name + "'");
    }
    for (const StageFacts& st : u.stages) {
        const eg_unit_s& su = *st.unit;
        const std::size_t n_slots = su.slots.size();
        std::vector<void*> params(n_slots);
        std::vector<plugin::ScalarHandle> handles(n_slots);
        std::vector<plugin::GRefMirror> mirrors(n_slots);
        std::uint32_t n32 = static_cast<std::uint32_t>(plan.n_samples);
        for (std::size_t j = 0; j < n_slots; ++j) {
            if (st.row_of[j] >= 0 || st.comp_of[j] >= 0) {    // the rows this stage writes into a cell: eagle's
                const std::size_t c = static_cast<std::size_t>(st.row_of[j] >= 0 ? st.row_of[j] : st.comp_of[j]);
                const DLTensor& t = (st.row_of[j] >= 0 ? plan.rows[c] : plan.comps[c])->tensor->dl_tensor;
                plugin::ScalarHandle& h = handles[j];
                h.data = static_cast<char*>(t.data) + t.byte_offset;
                h.samples = static_cast<std::uint64_t>(elements(t));    // the cell rows' own length
                h.stride = 1;
                h.deviceType = plugin::kEagleAbiDeviceCUDA;
                h.deviceId = 0;
                params[j] = &h;
            } else if (st.read_of[j] >= 0) {    // a cell an earlier stage folded: the caller's plane or eagle's, read as a table
                const std::size_t c = static_cast<std::size_t>(st.read_of[j]);
                const eg_plane_s* p = u.split_cells[c].source >= 0 ? plan.bindings[static_cast<std::size_t>(u.split_cells[c].source)].plane : plan.outs[c];
                const DLTensor& t = p->tensor->dl_tensor;
                plugin::ScalarHandle& h = handles[j];
                h.data = static_cast<char*>(t.data) + t.byte_offset;
                h.samples = static_cast<std::uint64_t>(elements(t));
                h.stride = 1;
                h.deviceType = plugin::kEagleAbiDeviceCUDA;
                h.deviceId = 0;
                params[j] = &h;
            } else if (st.source_of[j] >= 0) {
                params[j] = plan.params[static_cast<std::size_t>(st.source_of[j])];
            } else if (st.plane_of[j] >= 0 && su.slots[j].mirror) {    // a vector plane: the generated code reads the plane's own pitches
                mirrors[j] = team_mirror(plan.team[static_cast<std::size_t>(st.plane_of[j])]->tensor->dl_tensor,
                                         u.team_planes[static_cast<std::size_t>(st.plane_of[j])], plugin::kEagleAbiDeviceCUDA);
                params[j] = &mirrors[j];
            } else if (st.plane_of[j] >= 0) {
                const DLTensor& t = plan.team[static_cast<std::size_t>(st.plane_of[j])]->tensor->dl_tensor;
                const int role = su.slots[j].role;
                plugin::ScalarHandle& h = handles[j];
                h.data = static_cast<char*>(t.data) + t.byte_offset;
                h.samples = static_cast<std::uint64_t>(role == EG_ROLE_LOOKUP || zeroed_out(su.slots[j]) ? elements(t)
                                                       : role == EG_ROLE_WIDE_IN || kept_out(su.slots[j]) ? t.shape[t.ndim - 1] : plan.n_samples);
                h.stride = 1;
                h.deviceType = plugin::kEagleAbiDeviceCUDA;
                h.deviceId = 0;
                params[j] = &h;
            } else {
                params[j] = &n32;
            }
        }
        const auto fn = reinterpret_cast<CUfunction>(su.function);
        if (st.team) {
            const std::int64_t per_block = static_cast<std::int64_t>(st.lanes) * st.warps;
            const std::int64_t blocks = st.grid == EG_GRID_KEYS
                ? key_count(u, u.keyed[static_cast<std::size_t>(st.keyed)], plan.n_samples) * st.groups
                : (plan.n_samples + per_block - 1) / per_block;
            if (blocks <= 0) continue;
            std::int64_t base = 0, count = blocks, n = plan.n_samples;
            params.push_back(&base);
            params.push_back(&count);
            params.push_back(&n);
            check(cuLaunchKernel_(fn, static_cast<unsigned>(blocks), 1, 1, static_cast<unsigned>(st.threads * st.warps), 1, 1, 0, s, params.data(), nullptr),
                  "unit '" + su.name + "': the team stage launch");
            continue;
        }
        for (const exec::Partition& p : plan.parts)
            exec::DeviceKernel::run(fn, params, p, s, su.block > 0 ? static_cast<unsigned>(su.block) : resolved_block(d, su, p.count));
        for (const StageFacts::Fold& f : st.folds) {    // the stage's companions folded into their targets, with the stage's own arguments
            const DLTensor& target = (f.source >= 0 ? plan.bindings[static_cast<std::size_t>(f.source)].plane : plan.team[static_cast<std::size_t>(f.plane)])->tensor->dl_tensor;
            const std::int64_t size = elements(target);
            exec::DeviceKernel::run(reinterpret_cast<CUfunction>(su.companions[f.companion].device_fold), params, exec::Partition{ 0, size, size }, s);
        }
        if (plan.n_samples == 0) continue;
        for (std::size_t c = 0; c < u.split_cells.size(); ++c) {    // this stage's cell rows folded into their cells before the next stage reads them
            const SplitCell& sc = u.split_cells[c];
            if (sc.stage != static_cast<std::int32_t>(&st - u.stages.data())) continue;
            const eg_plane_s* o = sc.source >= 0 ? plan.bindings[static_cast<std::size_t>(sc.source)].plane : plan.outs[c];
            fold_cell(plan, c, plan.rows[c]->tensor->dl_tensor, plan.comps[c], sc.op, static_cast<char*>(o->tensor->dl_tensor.data) + o->tensor->dl_tensor.byte_offset, d.stream,
                      0, plan.rows[c]->tensor->dl_tensor.shape[0]);
        }
    }
}

namespace seg = host::seg;

/** @brief Queue the classify unit of a static selector over every sample (the old runner launches it at block 256). */
void classify_on_device(const eg_plan_s& plan) {
    const SegFacts& g = plan.unit->seg;
    const std::int64_t n = plan.n_samples;
    std::uint32_t n32 = static_cast<std::uint32_t>(n);
    std::vector<seg::Owned> none;
    std::vector<void*> p = seg::params_of(plan, static_cast<std::size_t>(g.k), *g.classify, none, &n32);
    exec::DeviceKernel::run(reinterpret_cast<CUfunction>(g.classify->function), p, exec::Partition{ 0, n, n }, reinterpret_cast<CUstream>(static_cast<std::uint64_t>(plan.ctx->dev.stream)), 256);
}

/** @brief One launch of a SEGMENTED plan on the device (a dynamic selector adds the guarded run's stale compaction and the sweep): the classify unit, eagle's stable K-class sort of copies of the owned planes
 *  (``reorderClassesDevice``, as the old device runner's class sort), each branch over its run (the old runner's block choice), the written
 *  copies restored and copied back to the caller's planes. The classes and the run offsets are read back to the host (a stream sync). */
void launch_segmented(const eg_plan_s& plan) {
    eg_ctx& ctx = *plan.ctx;
    const DeviceState& d = ctx.dev;
    const SegFacts& g = plan.unit->seg;
    SegState& S = plan.seg;
    const std::uint64_t stream = d.stream;
    const auto cs = reinterpret_cast<CUstream>(stream);
    const std::int64_t n = plan.n_samples;
    const unsigned K = static_cast<unsigned>(g.k);
    if (n == 0) return;
    const std::string at = "unit '" + plan.unit->name + "': dispatch group '" + g.name + "': ";
    std::uint32_t n32 = static_cast<std::uint32_t>(n);
    if (!g.dynamic) classify_on_device(plan);    // the classes of every sample, in sample order
    std::vector<seg::Owned> owned = seg::collect(plan, !g.dynamic, true);
    if (g.dynamic)    // the class and stale planes are slot-indexed scratch to the units: moved with the sort like the planes they sit beside, kept in sample order
        for (std::int32_t slot : { g.class_slot, g.stale_slot }) {
            seg::Owned o;
            const DLTensor& t = plan.segdev[slot == g.class_slot ? 1 : 2]->tensor->dl_tensor;
            o.slot = slot, o.data = static_cast<char*>(t.data) + t.byte_offset, o.elem = slot == g.class_slot ? 8 : 1, o.rows = 1, o.written = true;
            o.bind = plan.bindings[static_cast<std::size_t>(slot)];
            owned.push_back(std::move(o));
        }
    seg::Owned* mask = nullptr;
    for (seg::Owned& o : owned)
        if (o.slot == g.mask_slot) mask = &o;
    if (mask != nullptr && mask->elem != 1)
        refuse(EG_NOT_WRITTEN, at + "a TERMINATED mask wider than one byte is not written on the device");
    const std::size_t un = static_cast<std::size_t>(n);
    constexpr std::size_t kFixed = 9;    // perm, inv, map, cells, offsets, class bytes, zero mask, scratch, compaction scratch
    if (S.dev_bufs.size() != kFixed + owned.size()) {
        for (void* b : S.dev_bufs) device_free(ctx, b);
        S.dev_bufs.clear();
        std::vector<DevicePlane> planes{ { nullptr, 1 }, { nullptr, 1 } };
        for (const seg::Owned& o : owned)
            if (&o != mask) planes.push_back({ nullptr, static_cast<unsigned>(o.elem) });
        const std::size_t scratch = class_sort_scratch_bytes(n, planes, K);
        for (std::size_t bytes : { un * 4, un * 4, un * 4, std::size_t{ 16 }, (static_cast<std::size_t>(K) + 1) * 4, un, un, scratch, compact_scratch_bytes(n) })
            S.dev_bufs.push_back(device_alloc(ctx, bytes));
        for (const seg::Owned& o : owned) S.dev_bufs.push_back(device_alloc(ctx, o.rows * un * o.elem));
    }
    auto* perm = static_cast<std::int32_t*>(S.dev_bufs[0]);
    auto* inv = static_cast<std::int32_t*>(S.dev_bufs[1]);
    auto* imap = static_cast<std::int32_t*>(S.dev_bufs[2]);
    auto* cells = static_cast<unsigned*>(S.dev_bufs[3]);    // [count, span, class sorts]
    auto* off32 = static_cast<unsigned*>(S.dev_bufs[4]);
    void* cls8 = S.dev_bufs[5];
    void* zero_mask = S.dev_bufs[6];
    void* scratch = S.dev_bufs[7];
    if (g.dynamic) {    // the sort key: the classes the units wrote, taken again when the batch has gone stale (the first sort sees all zeros, as the old runner's)
        std::int64_t live = n;    // the live samples: every one without a mask, else the zero bytes of the caller's mask
        if (mask != nullptr) {
            mask_count(stream, mask->data, n, 1, cells + 3);
            unsigned dead = 0;
            cells_read(stream, reinterpret_cast<std::uint64_t>(cells + 3), &dead, 1);
            live = n - dead;
        }
        if (!S.sorted || static_cast<double>(S.stale_now) > host::kStaleMax * static_cast<double>(std::max<std::int64_t>(live, 1))) {
            std::vector<std::int64_t> cls(un);
            const DLTensor& ct = plan.segdev[1]->tensor->dl_tensor;
            device_read(ctx, static_cast<char*>(ct.data) + ct.byte_offset, cls.data(), un * sizeof(std::int64_t));
            for (std::size_t i = 0; i < un; ++i) S.sortkey[i] = static_cast<std::uint8_t>(std::clamp<std::int64_t>(cls[i], 0, g.k - 1));
            S.sorted = true;
        }
        device_write(ctx, cls8, S.sortkey.data(), un);
    } else {    // the class plane as bytes: read back, range-checked (the old class sort refuses a class outside [0, k)), written as uint8
        std::vector<std::int64_t> cls(un);
        const DLTensor& ct = plan.segdev[1]->tensor->dl_tensor;
        device_read(ctx, static_cast<char*>(ct.data) + ct.byte_offset, cls.data(), un * sizeof(std::int64_t));
        std::vector<std::uint8_t> bytes(un);
        for (std::size_t i = 0; i < un; ++i) {
            if (cls[i] < 0 || cls[i] >= g.k)
                refuse(EG_BAD_REQUEST, at + "the classify unit wrote class " + std::to_string(cls[i]) + " for sample " + std::to_string(i) + ", outside [0, " + std::to_string(g.k) + ")");
            bytes[i] = static_cast<std::uint8_t>(cls[i]);
        }
        device_write(ctx, cls8, bytes.data(), un);
    }
    for (std::size_t i = 0; i < owned.size(); ++i) {    // the copies the units read in sorted order
        seg::Owned& o = owned[i];
        void* copy = S.dev_bufs[kFixed + i];
        device_copy_bytes(ctx, copy, o.data, o.rows * un * o.elem);
        if (plan.unit->slots[static_cast<std::size_t>(o.slot)].mirror) o.bind.mirror.data_ = static_cast<double*>(copy);
        else o.bind.handle.data = copy;
    }
    std::vector<DevicePlane> planes;
    const void* mask_copy = zero_mask;
    if (mask != nullptr) mask_copy = S.dev_bufs[kFixed + static_cast<std::size_t>(mask - owned.data())];
    planes.push_back({ const_cast<void*>(mask_copy), 1 });
    planes.push_back({ cls8, 1 });
    const auto rows_of = [&](std::size_t i, std::vector<DevicePlane>& to) {    // a plane of several rows moves row by row
        for (std::size_t r = 0; r < owned[i].rows; ++r)
            to.push_back({ static_cast<char*>(S.dev_bufs[kFixed + i]) + r * un * owned[i].elem, static_cast<unsigned>(owned[i].elem) });
    };
    for (std::size_t i = 0; i < owned.size(); ++i)
        if (&owned[i] != mask) rows_of(i, planes);
    reset_step(stream, perm, inv, imap, cells, cells + 1, n);
    class_sort_step(stream, planes, mask_copy, K, cls8, perm, inv, imap, cells, cells + 1, off32, cells + 2, scratch, n);
    ++S.sorts;
    std::vector<unsigned> off(K + 1);
    cells_read(stream, reinterpret_cast<std::uint64_t>(off32), off.data(), K + 1);
    std::vector<std::int64_t> off64(off.begin(), off.end());
    device_write(ctx, static_cast<char*>(plan.segdev[0]->tensor->dl_tensor.data) + plan.segdev[0]->tensor->dl_tensor.byte_offset, off64.data(), off64.size() * sizeof(std::int64_t));
    // The argument list of a stage unit: each stage slot joined at load to a slot of the planned unit (the caller's plane, or the sorted copy of it) or to a
    // scratch plane eagle made. @p slots maps a stage slot to the planned unit's slot (-1: none).
    const auto plane_arg = [&](std::int32_t slot, std::vector<void*>& args, std::size_t j, const eg_plane_s* scratch, const SlotFacts& sf, const TeamPlaneFacts* joined, std::vector<plugin::ScalarHandle>& handles,
                              std::vector<plugin::GRefMirror>& mirrors) {
        if (scratch != nullptr) {
            const DLTensor& t = scratch->tensor->dl_tensor;
            if (sf.mirror) {    // a vector plane: the generated code reads the plane's own pitches
                mirrors[j] = team_mirror(t, *joined, plugin::kEagleAbiDeviceCUDA);
                args[j] = &mirrors[j];
                return;
            }
            const int role = sf.role;
            plugin::ScalarHandle& h = handles[j];
            h.data = static_cast<char*>(t.data) + t.byte_offset;
            h.samples = static_cast<std::uint64_t>(role == EG_ROLE_LOOKUP || zeroed_out(sf) ? elements(t) : role == EG_ROLE_WIDE_IN || kept_out(sf) ? t.shape[t.ndim - 1] : n);
            h.stride = 1;
            h.deviceType = plugin::kEagleAbiDeviceCUDA;
            h.deviceId = 0;
            args[j] = &h;
            return;
        }
        void* arg = plan.params[static_cast<std::size_t>(slot)];
        for (seg::Owned& o : owned)
            if (o.slot == slot) arg = plan.unit->slots[static_cast<std::size_t>(slot)].mirror ? static_cast<void*>(&o.bind.mirror) : static_cast<void*>(&o.bind.handle);
        args[j] = arg;
    };
    // A member that is one stage of itself runs that stage over its own slots (the member's argument list @p p) and its own scratch planes.
    const auto stage_args = [&](const eg_unit_s& u, std::size_t mi, const std::vector<void*>& p, std::vector<plugin::ScalarHandle>& handles,
                                std::vector<plugin::GRefMirror>& mirrors) {
        const StageFacts& st = u.stages[0];
        std::vector<void*> sp(u.slots.size());
        handles.resize(u.slots.size());
        mirrors.resize(u.slots.size());
        for (std::size_t s = 0; s < u.slots.size(); ++s) {
            if (st.source_of[s] >= 0) sp[s] = p[static_cast<std::size_t>(st.source_of[s])];
            else if (st.plane_of[s] >= 0) plane_arg(-1, sp, s, plan.seg_team(u, mi)[static_cast<std::size_t>(st.plane_of[s])], u.slots[s], &u.team_planes[static_cast<std::size_t>(st.plane_of[s])], handles, mirrors);
            else sp[s] = &n32;
        }
        return sp;
    };
    for (unsigned j = 0; j < K; ++j) {    // branch j over its run; the unit reads its own offset
        const std::int64_t run = static_cast<std::int64_t>(off[j + 1]) - static_cast<std::int64_t>(off[j]);
        if (run <= 0) continue;
        const eg_unit_s& u = *g.branches[j];
        std::vector<void*> p = seg::params_of(plan, j, u, owned, &n32);
        std::vector<plugin::ScalarHandle> handles;
        std::vector<plugin::GRefMirror> mirrors;
        if (!u.stages.empty()) p = stage_args(u, j, p, handles, mirrors);
        exec::DeviceKernel::run(reinterpret_cast<CUfunction>(u.function), p, exec::Partition{ 0, run, n }, cs, resolved_block(d, u, run));
    }
    if (g.dynamic && mask != nullptr)    // a stale sample that finished in this step is not swept: stale and live, as the old runner's
        stale_live_step(stream, S.dev_bufs[kFixed + static_cast<std::size_t>(owned.size()) - 1], mask_copy, n);
    if (g.dynamic) {    // the samples whose branch moved (stale), compacted into the sweep's active map; the fused unit completes them
        compact_keep_step(stream, S.dev_bufs[kFixed + static_cast<std::size_t>(owned.size()) - 1], n, static_cast<std::int32_t*>(plan.segdev[3]->tensor->dl_tensor.data),
                          static_cast<unsigned*>(plan.segdev[4]->tensor->dl_tensor.data), S.dev_bufs[8]);
        unsigned stale = 0;
        device_read(ctx, plan.segdev[4]->tensor->dl_tensor.data, &stale, sizeof(stale));
        S.stale_now = stale;
        if (stale > 0) {
            const eg_unit_s& u = *g.sweep;
            std::vector<void*> p = seg::params_of(plan, K, u, owned, &n32);
            if (!u.stages.empty()) {    // a split sweep: its one stage over the stage's own slots (a team stage is a grid over the batch, the kernel reads the active count)
                const StageFacts& st = u.stages[0];
                std::vector<plugin::ScalarHandle> handles;
                std::vector<plugin::GRefMirror> mirrors;
                std::vector<void*> sp = stage_args(u, K, p, handles, mirrors);
                if (st.team) {
                    EG_DRIVER(cuLaunchKernel);
                    const std::int64_t per_block = static_cast<std::int64_t>(st.lanes) * st.warps;
                    const std::int64_t blocks = (n + per_block - 1) / per_block;
                    std::int64_t base = 0, count = blocks, total = n;
                    sp.push_back(&base);
                    sp.push_back(&count);
                    sp.push_back(&total);
                    check(cuLaunchKernel_(reinterpret_cast<CUfunction>(u.function), static_cast<unsigned>(blocks), 1, 1, static_cast<unsigned>(st.threads * st.warps), 1, 1, 0, cs, sp.data(), nullptr),
                          "unit '" + u.name + "': the sweep's team stage launch");
                } else {
                    exec::DeviceKernel::run(reinterpret_cast<CUfunction>(u.function), sp, exec::Partition{ 0, static_cast<std::int64_t>(stale), n }, cs, resolved_block(d, u, stale));
                }
            } else {
                exec::DeviceKernel::run(reinterpret_cast<CUfunction>(u.function), p, exec::Partition{ 0, static_cast<std::int64_t>(stale), n }, cs, resolved_block(d, u, stale));
            }
        }
    }
    std::vector<DevicePlane> written;    // back to sample order: the planes the units wrote
    for (std::size_t i = 0; i < owned.size(); ++i)
        if (owned[i].written) rows_of(i, written);
    if (!written.empty()) restore_step(stream, written, perm, inv, scratch, n);
    for (std::size_t i = 0; i < owned.size(); ++i)
        if (owned[i].written) device_copy_bytes(ctx, owned[i].data, S.dev_bufs[kFixed + i], owned[i].rows * un * owned[i].elem);
}

/** @brief Queue the plan's launches (launch_on) on the context's stream. */
void device_launch(eg_plan_s& plan) {
    Current current(plan.ctx->dev);
    if (plan.unit->seg.present) { launch_segmented(plan); return; }
    if (!plan.unit->stages.empty()) { launch_split(plan); return; }
    if (set_count_slot(plan) >= 0) {    // an index set or a caller's active set: its live count is checked against the entries held (a device read), the kernel maps the lanes itself
        const DLTensor& c = plan.bindings[static_cast<std::size_t>(set_count_slot(plan))].plane->tensor->dl_tensor;
        unsigned live = 0;
        device_read(*plan.ctx, static_cast<char*>(c.data) + c.byte_offset, &live, sizeof(live));
        check_index_live(plan, live);
        launch_on(plan, plan.ctx->dev.stream, true, live);
        return;
    }
    launch_on(plan, plan.ctx->dev.stream);
}

namespace {

/** @brief Queue the zeroing of @p plan's eagle-made finished counter on @p stream (none: nothing). */
void finished_zero(const eg_plan_s& plan, std::uint64_t stream) {
    if (plan.device_counter != 0) zero_bytes(stream, reinterpret_cast<void*>(plan.device_counter), sizeof(unsigned));
}

}  // namespace

void device_launch_call(eg_plan_s& plan) {
    {
        Current current(plan.ctx->dev);
        finished_zero(plan, plan.ctx->dev.stream);
    }
    device_launch(plan);
}

/** @brief Queue the classify unit of the static SEGMENTED plan @p plan over every sample (a runner's look at the start of a run). */
void device_classify(eg_plan_s& plan) {
    Current current(plan.ctx->dev);
    classify_on_device(plan);
}

/** @brief Forget the captured loop of @p runner: it holds the launch cap it was captured with, which a run of fewer steps must not leave behind. */
void device_loop_drop(eg_runner_s& runner) {
    if (runner.dev.loop == nullptr) return;
    loop_free(runner.dev.loop);
    runner.dev.loop = nullptr;
}

/** @brief A DEVICE plan's eagle-made finished counter: one zeroed device cell. */
void device_plan_cells(eg_plan_s& plan) {
    Current current(plan.ctx->dev);
    plan.device_counter = cells_new(*plan.ctx, 1);
    if (plan.unit->finish.steps == EG_STEPS_AUTO) plan.device_steps = cells_new(*plan.ctx, 2);    // the int64 steps word, 0
}

/** @brief Free what device_plan_cells made. */
void device_plan_free(eg_plan_s& plan) noexcept {
    if (plan.device_counter == 0) return;
    eg_ctx* ctx = plan.ctx;
    const std::uint64_t counter = plan.device_counter, steps = plan.device_steps;
    plan.device_counter = plan.device_steps = 0;
    cells_free(*ctx, counter);
    cells_free(*ctx, steps);
}

/** @brief A device allocation of @p bytes on the context's device, zeroed when @p zero, stream-ordered on the context's
 *  stream; the block remembers that stream so its free is ordered after the work queued there. */
void* device_alloc(eg_ctx& ctx, std::size_t bytes, bool zero) {
    Current current(ctx.dev);
    void* data = alloc_async(ctx.dev.stream, bytes, zero);
    if (data != nullptr) ctx.dev.alloc_stream[data] = ctx.dev.stream;
    return data;
}

/** @brief Free a device_alloc allocation on the stream it was allocated on, ordered after the work queued on that stream and
 *  on the stream now in scope (a forked child makes no device call; a free while recording waits for eg_record_end). */
void device_free(eg_ctx& ctx, void* data) noexcept {
    if (data == nullptr || forked(ctx)) return;
    eg_ctx* c = &ctx;
    now_or_after_capture(ctx, [c, data] {
        Current current(c->dev);
        const auto at = c->dev.alloc_stream.find(data);
        free_async(at->second, c->dev.stream, data);
        c->dev.alloc_stream.erase(at);
    });
}

/** @brief A page-locked host allocation of @p bytes, zeroed when @p zero. */
void* pinned_alloc(eg_ctx& ctx, std::size_t bytes, bool zero) {
    Current current(ctx.dev);
    void* data = host_pinned_new(bytes);
    if (zero) std::memset(data, 0, bytes);
    return data;
}

/** @brief Free a pinned_alloc allocation (a forked child makes no driver call). */
void pinned_free(eg_ctx& ctx, void* data) noexcept {
    if (data == nullptr || forked(ctx)) return;
    eg_ctx* c = &ctx;
    now_or_after_capture(ctx, [c, data] {
        Current current(c->dev);
        host_pinned_free(data);
    });
}

/** @brief Read @p bytes (a multiple of 4) of device memory at @p device into @p host, ordered after the context's queued work. */
void device_read(eg_ctx& ctx, const void* device, void* host, std::size_t bytes) {
    Current current(ctx.dev);
    cells_read(ctx.dev.stream, reinterpret_cast<std::uint64_t>(device), static_cast<unsigned*>(host), static_cast<unsigned>(bytes / sizeof(unsigned)));
}

void device_write(eg_ctx& ctx, void* device, const void* host, std::size_t bytes) {
    if (bytes == 0) return;
    Current current(ctx.dev);
    write_bytes(ctx.dev.stream, device, host, bytes);
}

void device_copy_bytes(eg_ctx& ctx, void* dst, const void* src, std::size_t bytes) {
    if (bytes == 0 || dst == src) return;
    Current current(ctx.dev);
    copy_step(ctx.dev.stream, dst, src, bytes);
}

void device_zero(eg_ctx& ctx, void* data, std::size_t bytes) {
    if (bytes == 0) return;
    Current current(ctx.dev);
    zero_bytes(ctx.dev.stream, data, bytes);
}

void device_columns(eg_ctx& ctx, int op, void* dst, std::int64_t dst_ld, const void* src, std::int64_t src_ld,
                    const std::int64_t* index, std::int64_t count, std::int64_t rows, unsigned bytes, std::int64_t value) {
    Current current(ctx.dev);
    diff_columns(ctx.dev.stream, op, dst, dst_ld, src, src_ld, index, count, rows, bytes, value);
}

/** @brief One launch of @p plan over its first @p count samples (``Partition{0, count, of}``), the block the meta states or the
 *  one resolved at @p of samples (the old path's ``_resolved_block`` of the plan's width). */
void device_launch_prefix(eg_plan_s& plan, std::int64_t count, std::int64_t of) {
    if (count <= 0) return;
    if (plan.unit->group_record.present)
        refuse(EG_NOT_WRITTEN, "unit '" + plan.unit->name + "': a launch over a prefix of the lanes is not written for a unit with a group");
    const DeviceState& d = plan.ctx->dev;
    const eg_unit_s& u = *plan.unit;
    Current current(d);
    exec::DeviceKernel::run(reinterpret_cast<CUfunction>(u.function), plan.params, exec::Partition{ 0, count, of },
                            reinterpret_cast<CUstream>(static_cast<std::uint64_t>(d.stream)), u.block > 0 ? static_cast<unsigned>(u.block) : resolved_block(d, u, of));
    for (const CompanionFacts& c : u.companions) {    // a plan with a companion folds it into its target after the launch, as eg_launch does
        const std::int64_t size = elements(plan.bindings[c.target].plane->tensor->dl_tensor);
        exec::DeviceKernel::run(reinterpret_cast<CUfunction>(c.device_fold), plan.params, exec::Partition{ 0, size, size }, reinterpret_cast<CUstream>(static_cast<std::uint64_t>(d.stream)));
    }
}

void device_mask_count(eg_ctx& ctx, const void* mask, std::int64_t n, unsigned width, std::uint64_t counter) {
    Current current(ctx.dev);
    mask_count(ctx.dev.stream, mask, n, width, reinterpret_cast<unsigned*>(counter));
}

void device_pair_sum(eg_ctx& ctx, bool f64, const void* in, std::int64_t n, void* hi, void* comp) {
    Current current(ctx.dev);
    reduce_step(ctx.dev.stream, f64 ? 1 : 0, 0, in, n, 1, hi, comp, true);
}

void device_event_op(eg_ctx& ctx, unsigned bytes, unsigned op, const void* planes) {
    Current current(ctx.dev);
    event_op(ctx.dev.stream, bytes, op, planes);
}

std::int64_t device_free_bytes(eg_ctx& ctx) {
    Current current(ctx.dev);
    return free_bytes();
}

/** @brief eg_reduce over device planes: checked as the host's, then the canonical ordered tree on the context's
 *  stream (the order the host runs); returns with the result in place (in a capture, once the graph has replayed). */
void device_reduce(eg_ctx& ctx, const eg_reduce_desc& d) {
    if (d.index != nullptr) refuse(EG_NOT_WRITTEN, "eg_reduce: index (the position of a minimum or maximum) is not written for a device context");
    if (d.op == EG_REDUCE_ANY) refuse(EG_NOT_WRITTEN, "eg_reduce: op ANY is not written for a device context");
    check_handle(ctx.planes, d.in, "eg_reduce_desc.in");
    if (d.op == EG_REDUCE_ALL && d.out != nullptr && d.out->tensor->dl_tensor.dtype.code == kDLUInt)
        refuse(EG_NOT_WRITTEN, "eg_reduce: a uint32 result of op ALL is not written for a device context");
    if (d.in->tensor->dl_tensor.dtype.code == kDLUInt)
        refuse(EG_NOT_WRITTEN, "eg_reduce: a uint32 plane is not written for a device context");
    device_reduce_leaf(ctx, host::reduce_check(ctx, d));
}

/** @brief The working memory of the eager scans, compactions and sorts on @p stream: kept by the context, grown (never shrunk) when a larger
 *  one needs it, after the work queued on the stream has run. Its first @p state_bytes bytes are zero and the work leaves them zero;
 *  the @p temp_bytes after them (when asked) are any state. */
void* scan_state_for(eg_ctx& ctx, std::uint64_t stream, std::size_t state_bytes, std::size_t temp_bytes = 0) {
    DeviceState::ScanState& have = ctx.dev.scan_state[stream];
    const std::size_t bytes = (state_bytes + 15) / 16 * 16 + temp_bytes;
    if (have.bytes < bytes) {
        const std::size_t want = std::max(bytes, 2 * have.bytes);
        void* grown = scan_state_make(want);
        if (have.data != nullptr) {
            EG_DRIVER(cuStreamSynchronize);
            check(cuStreamSynchronize_(reinterpret_cast<CUstream>(stream)), "eg_scan: cuStreamSynchronize");
            scan_state_free(have.data);
        }
        have.data = grown;
        have.bytes = want;
        have.clean = want;
    }
    if (have.clean < state_bytes) {    // memory past the zero prefix was used as temporary memory: clear what this work needs
        zero_bytes(stream, static_cast<char*>(have.data) + have.clean, state_bytes - have.clean);
        have.clean = state_bytes;
    }
    if (temp_bytes != 0) have.clean = (state_bytes + 15) / 16 * 16;    // the temporary memory after the zero prefix is now in any state
    return have.data;
}

/** @brief eg_scan over device planes: the checks and the node are the host's; the scan is one pass with decoupled look-back
 *  (device/ScanKernels.cuh) on the stream in scope of the calling thread. In a recording its working memory is a pair of
 *  graph-owned stream-ordered allocation nodes, so a replay owns its own; eagerly it is the context's. */
void device_scan(eg_ctx& ctx, const eg_scan_desc& d) {
    const host::ScanOp s = host::scan_check(ctx, d);
    if (s.index != nullptr)
        refuse(EG_NOT_WRITTEN,
            "eg_scan: index (the running winner) is not written for a device "
            "context");
    const DLDataType& type = s.in->tensor->dl_tensor.dtype;
    const auto at = [](const eg_plane_s* p) -> void* {
        return static_cast<char*>(p->tensor->dl_tensor.data) + p->tensor->dl_tensor.byte_offset;
    };
    Node leaf;
    leaf.kind = Node::Scan;
    leaf.scan = s;
    note_leaf(ctx, "eg_scan", { { s.in, 0, -1, "in" }, { s.out, 1, -1, "out" } }, std::move(leaf));
    if (s.n == 0) return;
    const ScanElement element = type.code == kDLFloat ? (type.bits == 32 ? ScanElement::kFloat32 : ScanElement::kFloat64)
                              : type.code == kDLInt   ? (type.bits == 32 ? ScanElement::kInt32 : ScanElement::kInt64)
                                                      : (type.bits == 32 ? ScanElement::kUint32 : ScanElement::kUint64);
    const std::uint64_t stream = ctx.dev.stream;
    const std::size_t bytes = scan_state_bytes(type.bits, s.n);
    Current current(ctx.dev);
    if (ctx.dev_rec != nullptr) {
        void* state = alloc_async(stream, bytes, true);
        scan_step(stream, element, s.op, s.exclusive, at(s.in), at(s.out), s.n, state);
        free_async(stream, stream, state);
    } else {
        scan_step(stream, element, s.op, s.exclusive, at(s.in), at(s.out), s.n, scan_state_for(ctx, stream, bytes));
    }
}

/** @brief eg_compact over device planes: the checks and the node are the host's; the compaction is one pass over the mask
 *  (device/ScanKernels.cuh) on the stream in scope of the calling thread, with the working memory of a scan. */
void device_compact(eg_ctx& ctx, const eg_compact_desc& d) {
    const host::CompactOp c = host::compact_check(ctx, d);
    const auto at = [](const eg_plane_s* p) -> void* {
        return static_cast<char*>(p->tensor->dl_tensor.data) + p->tensor->dl_tensor.byte_offset;
    };
    Node leaf;
    leaf.kind = Node::Compact;
    leaf.compact = c;
    note_leaf(ctx, "eg_compact", { { c.keep, 0, -1, "keep" }, { c.kept, 1, -1, "kept" }, { c.count, 1, -1, "count" } }, std::move(leaf));
    const std::uint64_t stream = ctx.dev.stream;
    Current current(ctx.dev);
    if (c.n == 0) {
        zero_bytes(stream, at(c.count), sizeof(std::uint32_t));
        return;
    }
    const std::size_t bytes = compact_state_bytes(c.n);
    if (ctx.dev_rec != nullptr) {
        void* state = alloc_async(stream, bytes, true);
        compact_step(stream, at(c.keep), at(c.kept), at(c.count), c.n, state);
        free_async(stream, stream, state);
    } else {
        compact_step(stream, at(c.keep), at(c.kept), at(c.count), c.n, scan_state_for(ctx, stream, bytes));
    }
}

/** @brief What a recorded sort's host step reads after the replay's copy: the 32-bit count of keys outside the bound, in page-locked memory. */
struct OutsideKeys {
    std::uint32_t* count;
    std::int64_t bound;
};

/** @brief eg_order_by_key over device planes: the checks and the node are the host's; the sort is stable radix passes over the bits the
 *  bound needs (eagle/cuda/KeyedSort.cuh) on the stream in scope of the calling thread. Keys outside [0, bound) fail the call when
 *  it runs eagerly (it waits for the count) and the next ``eg_graph_launch`` or ``eg_sync`` after a replay (a host step reads the
 *  count the replay copied to page-locked memory). */
void device_order_by_key(eg_ctx& ctx, const eg_order_desc& d) {
    const host::OrderOp o = host::order_check(ctx, d);
    const auto at = [](const eg_plane_s* p) -> void* {
        return static_cast<char*>(p->tensor->dl_tensor.data) + p->tensor->dl_tensor.byte_offset;
    };
    Node leaf;
    leaf.kind = Node::Order;
    leaf.order = o;
    note_leaf(ctx, "eg_order_by_key", { { o.keys, 0, -1, "keys" }, { o.order, 1, -1, "order" }, { o.starts, 1, -1, "starts" } }, std::move(leaf));
    const std::uint64_t stream = ctx.dev.stream;
    Current current(ctx.dev);
    if (o.n == 0) {
        zero_bytes(stream, at(o.starts), static_cast<std::size_t>(o.bound + 1) * sizeof(std::uint32_t));
        return;
    }
    const DLDataType& type = o.keys->tensor->dl_tensor.dtype;
    const bool is_signed = type.code == kDLInt;
    const std::size_t state_bytes = order_state_bytes(o.n, o.bound), temp_bytes = order_temp_bytes(o.n, o.bound);
    const auto refusal = [&](std::uint32_t outside) {
        return "sort: " + std::to_string(outside) + " keys are outside [0, " + std::to_string(o.bound) + ")";
    };
    if (ctx.dev_rec == nullptr) {
        char* memory = static_cast<char*>(scan_state_for(ctx, stream, state_bytes, temp_bytes));
        void* temp = memory + (state_bytes + 15) / 16 * 16;
        order_step(stream, type.bits, is_signed, at(o.keys), o.n, o.bound, at(o.order), at(o.starts), memory, temp);
        std::uint32_t outside = 0;
        read_bytes(stream, &outside, temp, sizeof(outside));
        if (outside != 0) refuse(EG_BAD_REQUEST, refusal(outside));
        return;
    }
    void* state = nullptr;
    void* temp = nullptr;
    OutsideKeys* found = nullptr;
    const char* where = "the working memory";
    try {
        state = alloc_async(stream, state_bytes, true);
        temp = alloc_async(stream, temp_bytes, false);
        where = "the page-locked count";
        found = new OutsideKeys{ static_cast<std::uint32_t*>(host_pinned_new(sizeof(std::uint32_t))), o.bound };
        where = "the sort";
        order_step(stream, type.bits, is_signed, at(o.keys), o.n, o.bound, at(o.order), at(o.starts), state, temp);
        where = "the copy of the count";
        queue_read_bytes(stream, found->count, temp, sizeof(std::uint32_t));    // a copy node of the capture, read by the host step after it
    } catch (const std::exception& why) {
        refuse(EG_FAULT, std::string("eg_order_by_key while recording: ") + where + ": " + why.what());
    }
    free_async(stream, stream, temp);
    free_async(stream, stream, state);
    try {
        device_record_host_step(
            ctx,
            [](void* user, StepFailure& failure) noexcept {
                const auto* keys = static_cast<const OutsideKeys*>(user);
                if (*keys->count != 0)
                    failure.set("sort: " + std::to_string(*keys->count) + " keys are outside [0, " + std::to_string(keys->bound) + ")");
            },
            found,
            [](eg_ctx&, void* user) {
                auto* keys = static_cast<OutsideKeys*>(user);
                host_pinned_free(keys->count);
                delete keys;
            });
    } catch (...) {
        host_pinned_free(found->count);
        delete found;
        throw;
    }
}

/** @brief eg_reorder over device planes: not written on a device. */
void device_reorder(eg_ctx&, const eg_reorder_desc&)
{
    refuse(EG_NOT_WRITTEN,
        "eg_reorder: the reorder by an order is not written for a device "
        "context");
}

/** @brief The checked reduce @p r on the context's stream, kept in the recording if there is one. */
void device_reduce_leaf(eg_ctx& ctx, const host::ReduceOp& r) {
    const auto at = [](const eg_plane_s* p) -> void* {
        return p == nullptr ? nullptr : static_cast<char*>(p->tensor->dl_tensor.data) + p->tensor->dl_tensor.byte_offset;
    };
    Node leaf;
    leaf.kind = Node::Reduce;
    leaf.reduce = r;
    note_leaf(ctx, "eg_reduce", { { r.in, 0, -1, "in" }, { r.out, 1, -1, "out" }, { r.comp, r.pair ? 1 : 0, -1, "comp" } }, std::move(leaf));
    Current current(ctx.dev);
    reduce_step(ctx.dev.stream, r.code, r.op == EG_REDUCE_ALL ? 3 : r.op, at(r.in), r.n, r.cols, at(r.out), at(r.comp), r.pair);
}

/** @brief Whether @p p is page-locked host memory the device can DMA: a pinned plane the door made, or any range the driver knows as host memory (registered or allocated pinned). */
static bool host_pinned(const eg_plane_s& p) {
    if (p.pinned) return true;
    EG_DRIVER(cuPointerGetAttribute);
    const DLTensor& t = p.tensor->dl_tensor;
    unsigned int kind = 0;
    const CUdeviceptr at = reinterpret_cast<CUdeviceptr>(static_cast<char*>(t.data) + t.byte_offset);
    return cuPointerGetAttribute_(&kind, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, at) == CUDA_SUCCESS && kind == CU_MEMORYTYPE_HOST;
}

/** @brief eg_copy with at least one device plane: checked as the host's, then a copy on the context's stream. Two device
 *  planes copy device to device (planes of one range of memory copy nothing; partly overlapping ones are refused). One
 *  host plane, pageable or pinned, copies across and waits, so the host bytes are safe to reuse on return. */
void device_copy(eg_ctx& ctx, eg_plane_s* src, eg_plane_s* dst) {
    host::copy_check(ctx, src, dst);
    const DLTensor& s = src->tensor->dl_tensor;
    const DLTensor& t = dst->tensor->dl_tensor;
    char *ps = static_cast<char*>(s.data) + s.byte_offset, *pd = static_cast<char*>(t.data) + t.byte_offset;
    const std::size_t bytes = static_cast<std::size_t>(elements(s)) * (s.dtype.bits / 8);
    const bool crosses = (s.device.device_type == kDLCPU) != (t.device.device_type == kDLCPU);
    if (crosses && ctx.dev_rec != nullptr) {    // a captured memcpy node needs page-locked host memory; refuse before the recording changes
        const eg_plane_s* host_side = s.device.device_type == kDLCPU ? src : dst;
        if (!host_pinned(*host_side))
            refuse(EG_BAD_REQUEST, std::string("eg_copy while recording: the host plane (") + (host_side == src ? "src" : "dst") + ") is not page-locked, and a copy captured in a device graph "
                   "reads or writes it by DMA when the graph runs; make the host plane with eg_plane_new EG_MEM_PINNED, or copy before or after the recording");
    }
    Node leaf;
    leaf.kind = Node::Copy;
    leaf.src = src;
    leaf.dst = dst;
    note_leaf(ctx, crosses ? (s.device.device_type == kDLCPU ? "eg_copy (host to device)" : "eg_copy (device to host)") : "eg_copy", { { src, 0, -1, "src" }, { dst, 1, -1, "dst" } }, std::move(leaf));
    if (bytes == 0 || ps == pd) return;
    if (crosses && ctx.dev_rec != nullptr) {    // one memcpy node of the capture, in the direction of the planes; the caller keeps the host plane alive until the graph has run
        EG_DRIVER(cuMemcpyAsync);
        Current current(ctx.dev);
        check(cuMemcpyAsync_(reinterpret_cast<CUdeviceptr>(pd), reinterpret_cast<CUdeviceptr>(ps), bytes, reinterpret_cast<CUstream>(static_cast<std::uint64_t>(ctx.dev.stream))), "cuMemcpyAsync");
        return;
    }
    if (s.device.device_type == kDLCPU) {
        Current current(ctx.dev);
        write_bytes(ctx.dev.stream, pd, ps, bytes);
        return;
    }
    if (t.device.device_type == kDLCPU) {
        Current current(ctx.dev);
        read_bytes(ctx.dev.stream, pd, ps, bytes);
        return;
    }
    if (ps < pd + bytes && pd < ps + bytes) refuse(EG_BAD_REQUEST, "eg_copy: the device planes overlap");
    Current current(ctx.dev);
    copy_step(ctx.dev.stream, pd, ps, bytes);
}

/** @brief eg_fft_new on a device context: the cuFFT plans, and a C2R's copy of its input. */
void device_fft_new(eg_ctx& ctx, eg_fft_s& plan) {
    Current current(ctx.dev);
    plan.dev = fft_build(plan);
    const std::size_t bytes = fft_work_bytes(plan);
    if (bytes == 0) return;
    try {
        plan.work = device_alloc(ctx, bytes, false);
    } catch (...) {
        fft_drop(plan.dev);
        plan.dev = nullptr;
        throw;
    }
}

/** @brief eg_fft_run of a device plan: queued on the context's stream; a capture in progress keeps the plan until its graph is freed. */
void device_fft_run(eg_ctx& ctx, eg_fft_s& plan, eg_plane_s* in, eg_plane_s* out) {
    const auto at = [](const eg_plane_s* p) -> void* { return static_cast<char*>(p->tensor->dl_tensor.data) + p->tensor->dl_tensor.byte_offset; };
    Node leaf;
    leaf.kind = Node::Fft;
    leaf.fft = &plan;
    leaf.src = in;
    leaf.dst = out;
    note_leaf(ctx, "eg_fft_run", { { in, 0, -1, "src" }, { out, 1, -1, "dst" } }, std::move(leaf));
    Current current(ctx.dev);
    fft_enqueue(*plan.dev, plan, at(in), at(out), plan.work, ctx.dev.stream);
    if (ctx.dev_rec == nullptr) return;
    ctx.dev_rec->ffts.push_back(&plan);
    ++plan.refs;
}

/** @brief Free a device plan's cuFFT plans (after the capture in progress, if any) and its input copy. */
void device_fft_free(eg_ctx& ctx, eg_fft_s& plan) noexcept {
    eg_ctx* c = &ctx;
    DeviceFft* dev = plan.dev;
    now_or_after_capture(ctx, [c, dev] {
        Current current(c->dev);
        fft_drop(dev);
    });
    device_free(ctx, plan.work);
}

/** @brief The two reduction work buffers of each cell component of @p plan (made after its rows). */
void device_cell_work(eg_plan_s& plan) {
    for (const eg_plane_s* r : plan.rows) {
        const DLTensor& t = r->tensor->dl_tensor;
        for (int k = 0; k < 2; ++k) plan.work.push_back(make_plane(*plan.ctx, t.dtype, { cell_work_elems(t.shape[0]) }, true));
    }
}

/** @brief The loop and its cells go with the runner; a forked child makes no device call. */
DeviceRun::~DeviceRun() {
    if (ctx == nullptr || forked(*ctx)) return;
    eg_ctx* c = ctx;
    DeviceLoop* l = loop;
    const std::vector<std::uint64_t> all{ cells, map, scratch, perm, inv, rcells, rscratch, fast };
    now_or_after_capture(*c, [c, l, all] {
        Current current(c->dev);
        loop_free(l);
    });
    for (std::uint64_t cell : all) cells_free(*c, cell);
}

namespace {

/** @brief The old path's entry-pick constants (``_until_done``): PICK_FUSED_DIVISOR, PICK_PERSIST_MULTIPLE,
 *  ``_launch_policy.W_KNEE_DEFAULT`` and AUTO_STEP_OPS. */
constexpr std::int64_t kPickFusedDivisor = 4, kPickPersistMultiple = 8, kWKnee = 8, kAutoStepOps = 33;

policy::ResidencyDevice residency(const DeviceState& d) { return { d.cc_major, d.cc_minor, d.regs_per_sm, d.threads_per_sm }; }

policy::LaunchDevice launch_device(const DeviceState& d) {
    policy::LaunchDevice l;
    l.residency = residency(d);
    l.sms = d.sms;
    l.fp64Ratio = std::max(1, d.fp64_ratio);
    return l;
}

/** @brief ``_launch_policy.capacity``: SMs times the threads one SM keeps resident at @p regs, 0: unknown. */
std::int64_t capacity(const DeviceState& d, int regs) {
    const std::int64_t threads = d.regs_per_sm > 0 && regs > 0 ? policy::residentWarps(residency(d), regs) * 32LL : 0;
    return d.sms <= 0 || threads <= 0 ? 0 : d.sms * threads;
}

/** @brief ``_launch_policy.persistent_geometry``: grid SMs * @p blocks, the block narrowed from @p ideal to the
 *  largest warp multiple of which @p blocks fit one SM at the persist entry's registers. */
void persist_geometry(const DeviceState& d, const eg_unit_s& u, int blocks, int ideal, DeviceRun& r) {
    r.block = static_cast<unsigned>(ideal);
    r.grid = d.sms <= 0 ? 0u : static_cast<unsigned>(d.sms * blocks);
    if (d.sms <= 0 || d.regs_per_sm <= 0 || u.persist_max_threads <= 0 || u.persist_regs <= 0) return;
    for (int bs = std::min(ideal, u.persist_max_threads / 32 * 32); bs >= 32; bs -= 32)
        if (policy::residentBlocks(residency(d), u.persist_regs, bs) >= blocks) {
            r.block = static_cast<unsigned>(bs);
            return;
        }
}

/** @brief The no-graph entry of a runner over an automatic unit (``Runner._build_auto`` / ``_build_fast``, its
 *  first-run size rule, ``_fast_mode_for`` with ``_entry_regime``), the steps word set to the budget once and the
 *  persist launch sized at the first step fill; what needs the policy band loop, a priced entry pick or an
 *  entry this build does not launch is refused by name. */
void fast_runner_setup(eg_ctx& ctx, const eg_runner_desc& d, eg_runner_s& runner) {
    const eg_plan_s& plan = *d.plans[0];
    const eg_unit_s& u = *plan.unit;
    const DeviceState& dev = ctx.dev;
    const std::string who = "unit '" + u.name + "' (finish steps AUTO): ";
    if (d.every != 0)
        refuse(EG_BAD_REQUEST, "every " + std::to_string(d.every) + ": " + who + "the runner picks the steps of each launch");
    if (u.active_map_slot >= 0)
        refuse(EG_NOT_WRITTEN, who + "an active set with an automatic step count on a device is not written in this build "
               "of the door");
    if (plan.parts.size() != 1)
        refuse(EG_NOT_WRITTEN, who + "a plan of " + std::to_string(plan.parts.size()) + " partitions runs the automatic "
               "band loop (the per-launch step policy), which is not written in this build of the door");
    if (u.range_fn != 0)
        refuse(EG_NOT_WRITTEN, who + "the range entry '_range' (a unit built with the fast entries) is not written in this "
               "build of the door");
    const std::int64_t n = plan.n_samples;
    const std::int64_t wave = std::min(capacity(dev, u.regs), dev.sms * 32 * kWKnee);
    const std::int64_t persist_cap = u.persist_fn != 0 ? capacity(dev, u.persist_regs) : 0;
    int mode = 0;
    if (wave > 0 && n <= wave) mode = EG_MODE_FUSED_ONE;
    else if (wave > 0 && u.persist_fn != 0) mode = EG_MODE_PERSIST;
    if (persist_cap > 0 && n >= kPickPersistMultiple * persist_cap) {
        if (u.persist_fn != 0) mode = EG_MODE_PERSIST;
    } else if (wave > 0 && n * kPickFusedDivisor <= wave) {
        mode = EG_MODE_FUSED_ONE;
    }
    if (mode == 0)
        refuse(EG_NOT_WRITTEN, who + "a batch of " + std::to_string(n) + " samples that neither entry takes runs the automatic "
               "band loop (the per-launch step policy), which is not written in this build of the door");
    runner.dev.mode = mode;
    const unsigned word[2] = { static_cast<unsigned>(static_cast<std::uint64_t>(d.max_steps) & 0xFFFFFFFFu),
                               static_cast<unsigned>(static_cast<std::uint64_t>(d.max_steps) >> 32) };
    cells_write(dev.stream, reinterpret_cast<std::uint64_t>(steps_cell(plan)), word, 2);
    if (mode != EG_MODE_PERSIST) return;
    runner.dev.fast = cells_new(ctx, 2);
    const bool fp64 = u.scalar_type != EG_SCALAR_FLOAT32;
    const double ops = static_cast<double>(u.finish.ops > 0 ? u.finish.ops : kAutoStepOps);
    const policy::LaunchDevice l = launch_device(dev);
    runner.dev.fill = policy::PERSIST_FIRST_FILL;
    runner.dev.chunk = static_cast<unsigned>(policy::persistChunk(l, fp64, ops, 1e9));
    const int threads = policy::persistThreads(l, fp64, u.persist_regs, n, policy::PERSIST_FIRST_FILL,
                                               static_cast<int>(runner.dev.chunk), ops);
    int blocks = 1;
    while (threads > 0 && threads / blocks > 512) blocks *= 2;
    persist_geometry(dev, u, threads > 0 ? blocks : 1, threads > 0 ? threads / blocks / 32 * 32 : 256, runner.dev);
    // the persist entry's fetch counter is a uint32 the lanes keep fetching from until they pass n: n plus one
    // overshooting fetch per lane must not wrap it, or the launch never ends
    const std::int64_t lanes = static_cast<std::int64_t>(runner.dev.grid) * runner.dev.block;
    if (n + lanes > (std::int64_t{ 1 } << 32))
        refuse(EG_BAD_REQUEST, who + "the persist entry runs fewer than 2^32 samples per launch (" + std::to_string(n) +
               " samples plus " + std::to_string(lanes) + " lanes); split the batch");
}

/** @brief One no-graph run (``Runner._run_fast``): re-sync or zero the counter as the entry counts, one launch of
 *  the fused entry (word = the budget) or the persist entry, one readback; the report as the old fast path. */
void fast_run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report) {
    EG_DRIVER(cuLaunchKernel);
    eg_plan_s& first = *runner->plans[0];
    const FinishFacts& f = first.unit->finish;
    const std::int64_t n = first.n_samples;
    DeviceRun& r = runner->dev;
    const std::uint64_t stream = ctx.dev.stream;
    Current current(ctx.dev);
    const bool persist = r.mode == EG_MODE_PERSIST;    // the one exact entry here: it maxes each sample's steps
    const bool counts = (f.flags & EG_FINISH_ENTRY_COUNTS) != 0 && persist;
    if (persist) {
        const unsigned zero[2] = { 0, 0 };
        cells_write(stream, r.fast, zero, 2);
        cells_write(stream, first.device_counter, zero, 1);
    }
    const DLTensor& t = first.bindings[static_cast<std::size_t>(f.mask_slot)].plane->tensor->dl_tensor;
    if (!counts)
        mask_count(stream, static_cast<const char*>(t.data) + t.byte_offset, n, t.dtype.bits / 8u,
                   reinterpret_cast<unsigned*>(first.device_counter));
    if (persist) {
        long long base = 0, count = n, samples = n;
        void* next = reinterpret_cast<void*>(r.fast);
        void* util = nullptr;
        void* steps = reinterpret_cast<void*>(r.fast + sizeof(unsigned));
        void* stepsum = nullptr;
        unsigned chunk = r.chunk;
        std::vector<void*> args(first.params.begin(), first.params.end());
        for (void* a : { static_cast<void*>(&base), static_cast<void*>(&count), static_cast<void*>(&samples),
                         static_cast<void*>(&next), static_cast<void*>(&util), static_cast<void*>(&steps),
                         static_cast<void*>(&stepsum), static_cast<void*>(&chunk) })
            args.push_back(a);
        check(cuLaunchKernel_(reinterpret_cast<CUfunction>(first.unit->persist_fn), r.grid, 1, 1, r.block, 1, 1, 0,
                              reinterpret_cast<CUstream>(stream), args.data(), nullptr),
              "unit '" + first.unit->name + "': the persist launch (cuLaunchKernel)");
    } else {
        launch_on(first, stream, false);
    }
    unsigned steps_read = 0, finished = 0;
    if (persist) cells_read(stream, r.fast + sizeof(unsigned), &steps_read, 1);
    cells_read(stream, first.device_counter, &finished, 1);
    const bool done = static_cast<std::int64_t>(finished) == n;
    eg_run_report out{};
    out.flags = (done ? EG_REPORT_DONE : 0u) | (persist || !done ? EG_REPORT_EXACT_STEPS : 0u);
    out.mode = r.mode;
    out.n = n;
    out.finished = finished;
    out.launches = 1;
    out.steps = persist ? static_cast<std::int64_t>(steps_read) : runner->max_steps;
    out.mapped = n;
    note_launches_by_k(out, runner->max_steps, 1);
    write_report(report, out);
}

}  // namespace

/** @brief eg_runner_new over DEVICE plans: the band loop (``_until_done.Runner``) of a fixed-step finishing unit,
 *  with an active set when its units read one (launches over the map's live prefix, compaction every ``every``
 *  steps when a sample finished); what else a device runner would run is refused by name. */
void device_runner_new(eg_ctx& ctx, const eg_runner_desc* desc, eg_runner** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_runner_new: out is NULL");
    *out = nullptr;
    eg_runner_desc d = read_input(desc, "eg_runner_desc");
    for (std::int32_t i = 0; i < d.n_plans; ++i)
        if (ctx.plans.count(d.plans[i]) != 0) {
            const eg_unit_s& u = *d.plans[i]->unit;
            const char* inside = !u.companions.empty() || !u.cells.empty() ? "companion or cell folding"
                                 : u.index_count_slot >= 0                  ? "an index set"
                                 : u.dev_box_t > 0                          ? "a device thread box"
                                                                            : nullptr;
            if (inside != nullptr)
                refuse(EG_NOT_WRITTEN, "plans[" + std::to_string(i) + "]: " + inside + " inside a device runner loop "
                       "is not written in this build of the door");
        }
    KeptDerived kept;
    check_runner(ctx, d, true, kept);
    for (std::int32_t i = 0; i < d.n_step_graphs; ++i)
        if (!reinterpret_cast<const DeviceGraph*>(d.step_graphs[i].graph)->hooks.empty())
            refuse(EG_BAD_REQUEST, "step_graphs[" + std::to_string(i) + "]: a graph holding host nodes cannot run inside the captured loop (the driver runs a host node "
                   "outside the loop's device graph)");
    for (std::int32_t i = 0; i < d.n_plans; ++i)
        if (d.plans[i]->structure != EG_STRUCTURE_DEVICE)
            refuse(EG_BAD_REQUEST, "plans[" + std::to_string(i) + "] is a host plan in a runner over DEVICE plans: "
                   "one runner runs on one side");
    if (d.plans[0]->unit->seg.present) {    // a SEGMENTED group reads its classes back at each launch: no captured loop, the runner steps it
        if (d.every != 0 && d.every != d.plans[0]->unit->finish.steps)
            refuse(EG_BAD_REQUEST, "every " + std::to_string(d.every) + ": a runner over a SEGMENTED group advances "
                   "unit '" + d.plans[0]->unit->name + "''s " + std::to_string(d.plans[0]->unit->finish.steps) + " steps per launch");
        if (d.flags & EG_RUN_REORDER)
            refuse(EG_BAD_REQUEST, "EG_RUN_REORDER: a SEGMENTED group reads no active set, so there is nothing to reorder");
        auto* runner = new eg_runner_s;
        runner->plans.assign(d.plans, d.plans + d.n_plans);
        runner->max_steps = d.max_steps;
        runner->dev.ctx = &ctx;
        runner->dev.stepped = true;
        runner->step_graphs.assign(d.step_graphs, d.step_graphs + d.n_step_graphs);
        ctx.runners.insert(runner);
        for (const eg_step_graph& g : runner->step_graphs) step_graph_retain(ctx, g.graph);
        for (eg_plan_s* p : runner->plans) ++p->refs;
        *out = runner;
        return;
    }
    const std::int64_t k = launch_steps(d.plans[0]->unit->finish, static_cast<std::size_t>(d.n_plans));
    if (k == EG_STEPS_AUTO) {    // the no-graph entries: no loop, no loop cells
        if (d.events == EG_EVENTS_REFINE)
            refuse(EG_NOT_WRITTEN, "events REFINE of an automatic unit on a device is not written in this build of the door");
        if (d.flags & EG_RUN_REORDER)
            refuse(EG_NOT_WRITTEN, "EG_RUN_REORDER (reordering the planes by the active set) with automatic steps on a "
                   "device is not written in this build of the door");
        auto* runner = new eg_runner_s;
        try {
            runner->plans.assign(d.plans, d.plans + d.n_plans);
            runner->max_steps = d.max_steps;
            runner->dev.ctx = &ctx;
            Current current(ctx.dev);
            // A recorded run of an automatic unit runs its K-step twin (host/Reverse.cpp): no entry of its own to pick.
            if (d.flags & EG_RUN_RECORD) record_new(ctx, *runner, d);
            else fast_runner_setup(ctx, d, *runner);
            ctx.runners.insert(runner);
        } catch (...) {
            record_free(ctx, *runner);
            delete runner;
            throw;
        }
        for (eg_plan_s* p : runner->plans) ++p->refs;
        *out = runner;
        return;
    }
    const bool active = (d.plans[0]->unit->active_map_slot >= 0 || (d.flags & EG_RUN_REORDER)) && check_active(d);
    const std::int64_t group = d.plans[0]->unit->group;
    if (active && group != 1)
        refuse(EG_NOT_WRITTEN, "an active set over groups of " + std::to_string(group) + " lanes on a device is not "
               "written in this build of the door (the old device runner cannot capture its compaction either)");
    const bool reorder = active && (d.flags & EG_RUN_REORDER) && d.plans[0]->n_samples >= kMinReorderSpan;
    if (!active && d.every != 0 && d.every != k)
        refuse(EG_BAD_REQUEST, "every " + std::to_string(d.every) + ": a runner without an active set advances "
               "unit '" + d.plans[0]->unit->name + "''s " + std::to_string(k) + " steps per launch");
    const std::int64_t every = active ? active_every_steps(d) : k;
    const std::int64_t cap = loop_cap(d.max_steps, every);
    if (cap >= 0xFFFFFFFFll)
        refuse(EG_BAD_REQUEST, "max_steps " + std::to_string(d.max_steps) + ": the device loop counts launches in "
               "uint32");
    auto* runner = new eg_runner_s;
    try {
        runner->plans.assign(d.plans, d.plans + d.n_plans);
        runner->max_steps = d.max_steps;
        runner->step_graphs.assign(d.step_graphs, d.step_graphs + d.n_step_graphs);
        runner->dev.ctx = &ctx;
        Current current(ctx.dev);
        runner->dev.cells = cells_new(ctx, 6);    // [remaining, ran, total, live count, finished at the last compaction, compactions]
        const std::int64_t n = d.plans[0]->n_samples;
        if (active) {
            ActiveState& a = runner->act;
            a.n = n;
            a.every_steps = every;
            a.every_launches = every / k;
            a.reorder = reorder;
            a.theta = d.reorder_theta;
            runner->active = true;
            runner->dev.map = cells_new(ctx, static_cast<unsigned>(std::max<std::int64_t>(n, 1)));
            runner->dev.scratch = cells_new(ctx, static_cast<unsigned>((compact_scratch_bytes(n) + sizeof(unsigned) - 1) / sizeof(unsigned)));
            if (reorder) {
                DeviceRun& r = runner->dev;
                r.perm = cells_new(ctx, static_cast<unsigned>(n));
                r.inv = cells_new(ctx, static_cast<unsigned>(n));
                r.rcells = cells_new(ctx, 5);    // [live32, span, fire, reorders, zero]
                reorder_scratch(r, n, reorder_table(*runner));
            }
        }
        device_runner_initial(ctx, *runner);
        if ((d.flags & EG_RUN_RECORD) || d.events == EG_EVENTS_REFINE) record_new(ctx, *runner, d);    // the host's record and refine checks; the run is host/Reverse.cpp's on the device
        ctx.runners.insert(runner);
    } catch (...) {
        record_free(ctx, *runner);
        delete runner;
        throw;
    }
    for (const eg_step_graph& g : runner->step_graphs) step_graph_retain(ctx, g.graph);
    for (eg_plan_s* p : runner->plans) {
        ++p->refs;
        if (active) {
            p->active_owner = runner;
            bind_raw(*p, p->unit->active_map_slot, reinterpret_cast<void*>(runner->dev.map), static_cast<std::uint64_t>(runner->act.n));
            bind_raw(*p, p->unit->active_count_slot, reinterpret_cast<void*>(runner->dev.cells + 3 * sizeof(unsigned)), 1);
        }
    }
    *out = runner;
}

void device_runner_initial(eg_ctx& ctx, eg_runner_s& runner) {
    DeviceRun& r = runner.dev;
    if (r.cells == 0) return;    // a fast (steps AUTO) runner writes its cells at each run
    Current current(ctx.dev);
    const std::uint64_t stream = ctx.dev.stream;
    const std::int64_t n = runner.plans[0]->n_samples;
    const unsigned cells[6] = { 0, 0, static_cast<unsigned>(n), static_cast<unsigned>(n), 0, 0 };    // every sample live
    cells_write(stream, r.cells, cells, 6);
    if (!runner.active) return;
    std::vector<unsigned> identity(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) identity[static_cast<std::size_t>(i)] = static_cast<unsigned>(i);
    cells_write(stream, r.map, identity.data(), static_cast<unsigned>(n));
    if (!runner.act.reorder) return;
    const unsigned rcells[5] = { 0, static_cast<unsigned>(n), 0, 0, 0 };    // span = n
    cells_write(stream, r.perm, identity.data(), static_cast<unsigned>(n));
    cells_write(stream, r.inv, identity.data(), static_cast<unsigned>(n));
    cells_write(stream, r.rcells, rcells, 5);
}

bool device_active_compact(eg_ctx& ctx, eg_runner_s& runner, const void* mask, const std::vector<Owned>& owned, bool reorder) {
    Current current(ctx.dev);
    const std::uint64_t s = ctx.dev.stream;
    queue_compact(runner, s, mask);
    if (!reorder || !runner.act.reorder) return false;
    unsigned fire = 0;
    cells_read(s, runner.dev.rcells + 2 * sizeof(unsigned), &fire, 1);
    runner.act.fire = fire != 0;
    if (!runner.act.fire) return false;
    const std::vector<DevicePlane> planes = planes_of(owned);
    queue_reorder(runner, s, mask, planes, reorder_scratch(runner.dev, runner.act.n, planes));
    ++runner.act.reorders;
    runner.act.fire = false;
    return true;
}

void device_active_restore(eg_ctx& ctx, eg_runner_s& runner, const void* mask, const std::vector<Owned>& owned) {
    Current current(ctx.dev);
    const std::vector<DevicePlane> planes = planes_of(owned);
    queue_restore(runner, ctx.dev.stream, mask, planes, reorder_scratch(runner.dev, runner.act.n, planes));
    runner.act.fire = false;
    runner.act.restored_at = runner.act.reorders;
}

namespace {

/** @brief eg_run of a runner over a SEGMENTED group (``DispatchRunner._run``): re-sync the counter to the mask, then launch the
 *  group once per step (it syncs itself) and read the counter, until every sample finished or the steps are spent. */
void stepped_run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report) {
    eg_plan_s& first = *runner->plans[0];
    const FinishFacts& f = first.unit->finish;
    const std::int64_t n = first.n_samples;
    const std::uint64_t stream = ctx.dev.stream;
    Current current(ctx.dev);
    const DLTensor& t = first.bindings[static_cast<std::size_t>(f.mask_slot)].plane->tensor->dl_tensor;
    mask_count(stream, static_cast<const char*>(t.data) + t.byte_offset, n, t.dtype.bits / 8u, reinterpret_cast<unsigned*>(first.device_counter));
    const std::int64_t cap = loop_cap(runner->max_steps, launch_steps(f, runner->plans.size()));
    unsigned finished = 0;
    std::int64_t launches = 0;
    cells_read(stream, first.device_counter, &finished, 1);
    while (launches < cap && static_cast<std::int64_t>(finished) != n) {
        run_step(*runner, [&](eg_graph_s* g) { device_graph_launch(ctx, g); }, [&](eg_plan_s* p) { if (p == &first) device_launch(*p); });
        cells_read(stream, first.device_counter, &finished, 1);
        ++launches;
    }
    eg_run_report out{};
    out.flags = (static_cast<std::int64_t>(finished) == n ? EG_REPORT_DONE : 0u) | (launch_steps(f, runner->plans.size()) == 1 ? EG_REPORT_EXACT_STEPS : 0u);
    out.mode = EG_MODE_BAND;
    out.n = n;
    out.finished = finished;
    out.launches = launches;
    out.steps = launches * launch_steps(f, runner->plans.size());
    out.mapped = launches * n;
    note_launches_by_k(out, launch_steps(f, runner->plans.size()), launches);
    write_report(report, out);
}

}  // namespace

void reapply_bindings(DeviceGraph& graph);

/** @brief eg_run of a DEVICE runner (``_until_done.Runner._run``, band): re-sync the counter to the mask, capture
 *  the WHILE loop at the first run (again when a plan's bindings changed), launch it, read the iterations
 *  and the counter back, fill @p report as the old device path does. */
void device_run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report) {
    check_handle(ctx.runners, runner, "runner");
    if (ctx.rec != nullptr)
        refuse(EG_NOT_WRITTEN, "recording a DEVICE run in a host recording is not written: record it on a device context");
    for (const eg_plan_s* p : runner->plans) check_bound(*p);
    if (ctx.dev_rec != nullptr) return device_record_run(ctx, runner);    // a recorded run fills no report
    check_report(report);
    if (runner->record != nullptr) {    // EG_RUN_RECORD: the recording forward (host/Reverse.cpp) over device planes
        record_forward(ctx, *runner, report);
        return;
    }
    if (runner->dev.stepped) return stepped_run(ctx, runner, report);
    if (runner->dev.mode != EG_MODE_PERSIST) note_launch(ctx.dev, *runner->plans[0], runner->dev);    // the persist geometry is set at construction
    if (runner->dev.mode != EG_MODE_BAND) return fast_run(ctx, runner, report);
    eg_plan_s& first = *runner->plans[0];
    const FinishFacts& f = first.unit->finish;
    const std::int64_t n = first.n_samples, k = launch_steps(f, runner->plans.size());
    const std::int64_t every = runner->active ? runner->act.every_steps : k;    // steps per loop iteration
    DeviceRun& r = runner->dev;
    const std::uint64_t stream = ctx.dev.stream;
    Current current(ctx.dev);
    for (const eg_step_graph& g : runner->step_graphs) reapply_bindings(*reinterpret_cast<DeviceGraph*>(g.graph));    // the loop captures the graphs as they are now
    std::vector<unsigned char> bound;
    for (const eg_plan_s* p : runner->plans) {
        const std::vector<unsigned char> one = binding_bytes(*p);
        bound.insert(bound.end(), one.begin(), one.end());
    }
    for (const eg_step_graph& g : runner->step_graphs)    // a rebound plan of a step graph captures again, as a rebound plan of the runner does
        for (const eg_plan_s* p : reinterpret_cast<DeviceGraph*>(g.graph)->plans) {
            const std::vector<unsigned char> one = binding_bytes(*p);
            bound.insert(bound.end(), one.begin(), one.end());
        }
    if (r.loop != nullptr && bound != r.bound) {
        loop_free(r.loop);
        r.loop = nullptr;
    }
    auto* counter = reinterpret_cast<unsigned*>(first.device_counter);
    unsigned compactions0 = 0, reorders0 = 0;
    if (runner->active) cells_read(stream, r.cells + 5 * sizeof(unsigned), &compactions0, 1);
    if (runner->act.reorder) cells_read(stream, r.rcells + 3 * sizeof(unsigned), &reorders0, 1);
    const DLTensor& t = first.bindings[static_cast<std::size_t>(f.mask_slot)].plane->tensor->dl_tensor;
    if (runner->active && t.dtype.bits != 8)
        refuse(EG_BAD_REQUEST, "the mask of an active set on a device is a bool or uint8 plane, got " + std::to_string(t.dtype.bits) + " bits");
    if (f.steps == EG_STEPS_AUTO) {    // a several-plan step: the finisher takes one step per launch, the runner writes its steps word
        const unsigned one_step[2] = { 1, 0 };
        cells_write(stream, reinterpret_cast<std::uint64_t>(steps_cell(first)), one_step, 2);
    }
    mask_count(stream, static_cast<const char*>(t.data) + t.byte_offset, n, t.dtype.bits / 8u, counter);
    if (r.loop == nullptr) {
        const auto t0 = std::chrono::steady_clock::now();
        auto* cells = reinterpret_cast<unsigned*>(r.cells);
        const auto cap = static_cast<unsigned>(loop_cap(runner->max_steps, every));
        const auto body = [runner](std::uint64_t s) {
            for (std::int64_t l = 0; l < (runner->active ? runner->act.every_launches : 1); ++l)
                run_step(*runner,    // a step graph's capture joins the loop body as a child graph, between the plans' launches at its position
                         [s](eg_graph_s* g) { for (const Piece& piece : reinterpret_cast<DeviceGraph*>(g)->pieces) loop_embed(piece.loop, s); },
                         [s](eg_plan_s* p) { launch_on(*p, s, false); });
        };
        std::vector<LoopGate> gates;
        if (runner->active) {
            const ActiveState& a = runner->act;
            gates.push_back({ [runner, &t, cells, counter](std::uint64_t s) {
                                  queue_compact(*runner, s, static_cast<const char*>(t.data) + t.byte_offset);
                                  compact_mark(s, counter, cells + 4, cells + 5);
                              },
                              counter, cells + 4 });
            if (a.reorder)
                gates.push_back({ [runner, &r, &t](std::uint64_t s) {
                                      queue_reorder(*runner, s, static_cast<const char*>(t.data) + t.byte_offset, reorder_table(*runner),
                                                    reinterpret_cast<void*>(r.rscratch));
                                  },
                                  reinterpret_cast<unsigned*>(r.rcells) + 2, reinterpret_cast<unsigned*>(r.rcells) + 4 });
        }
        r.loop = loop_capture(stream, counter, cells + 2, cells, cap, body, gates);
        stream_wait(stream);
        r.build_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        r.bound = std::move(bound);
    }
    loop_launch(r.loop, stream);
    unsigned ran = 0, finished = 0;
    cells_read(stream, r.cells + sizeof(unsigned), &ran, 1);
    cells_read(stream, first.device_counter, &finished, 1);
    eg_run_report out{};
    unsigned compactions = 0;
    if (runner->active) cells_read(stream, r.cells + 5 * sizeof(unsigned), &compactions, 1);
    out.flags = (static_cast<std::int64_t>(finished) == n ? EG_REPORT_DONE : 0u) | (every == 1 ? EG_REPORT_EXACT_STEPS : 0u);
    out.mode = EG_MODE_BAND;
    out.n = n;
    out.finished = finished;
    out.launches = ran;
    out.steps = static_cast<std::int64_t>(ran) * every;
    out.compactions = compactions - compactions0;
    if (runner->act.reorder) {    // the end-of-run exit: every plane back in sample order, the map recomputed from the restored mask
        unsigned reorders = 0;
        cells_read(stream, r.rcells + 3 * sizeof(unsigned), &reorders, 1);
        out.reorders = reorders - reorders0;
        if (reorders != runner->act.restored_at) {
            queue_restore(*runner, stream, static_cast<const char*>(t.data) + t.byte_offset, reorder_table(*runner),
                          reinterpret_cast<void*>(r.rscratch));
            runner->act.restored_at = reorders;
        }
    }
    out.mapped = static_cast<std::int64_t>(ran) * n;
    note_launches_by_k(out, every, ran);
    out.build_s = r.build_s;
    write_report(report, out);
}

/** @brief Refuse @p call with @p status while @p ctx records by capture: @p why. */
void device_quiet(const eg_ctx& ctx, const char* call, eg_status status, const char* why) {
    if (ctx.dev_rec != nullptr)
        refuse(status, std::string(call) + " while recording on a device context: " + why);
}

/** @brief eg_record_begin on a device context: capture the stream in scope (the old device graph's capture,
 *  ``GraphPipeline.build`` over cuda/StreamCapturer.h); launches are recorded, not run, until eg_record_end. */
void device_record_begin(eg_ctx& ctx) {
    if (ctx.dev_rec != nullptr) refuse(EG_BAD_REQUEST, "eg_record_begin: already recording");
    auto rec = std::make_unique<DeviceRecording>();
    rec->stream = ctx.dev.stream;
    Current current(ctx.dev);
    rec->loop_cells = device_alloc(ctx, kRecordedLoops * 2 * sizeof(unsigned), true);
    try {
        rec->pool = pool_new(kRecordStreams, kRecordEvents);
        rec->capture = capture_begin(rec->stream);
    } catch (...) {
        pool_free(rec->pool);
        device_free(ctx, rec->loop_cells);
        throw;
    }
    ctx.dev_rec = rec.release();
}

namespace {

/** @brief The innermost open WHILE of @p rec, or null. */
const Frame* innermost_loop(const DeviceRecording& rec) {
    for (std::size_t i = rec.frames.size(); i-- > 0;)
        if (rec.frames[i].iteration != nullptr) return &rec.frames[i];
    return nullptr;
}

/** @brief The second capture of a EG_RECORD_DIFF recording: capture the copies the tape keeps in front of launch @p launch. */
void diff_copies(eg_ctx& ctx, const DeviceRecording& rec, std::size_t launch) {
    if (launch >= rec.dp->before.size()) return;
    for (const LeafCopy& c : rec.dp->before[launch]) {
        if (c.direct != nullptr) {
            copy_step(ctx.dev.stream, c.direct, c.plane, c.bytes);
        } else {
            const Frame* loop = innermost_loop(rec);
            table_copy_step(ctx.dev.stream, c.plane, c.bytes, c.table, loop->iteration, c.capacity, 1);
        }
    }
}

void note_leaf(eg_ctx& ctx, const char* label, std::vector<PlaneUse> uses, Node leaf, const eg_gemm_s* product) {
    if (ctx.dev_rec == nullptr) return;
    DeviceRecording& rec = *ctx.dev_rec;
    note_node(ctx, gnode_kind_of(leaf.kind), label, std::move(uses));
    if (rec.dp != nullptr) diff_copies(ctx, rec, rec.dp->launches + rec.others);
    Op op;
    op.kind = Op::Leaf;
    op.leaf = std::move(leaf);
    if (product != nullptr) op.product = *product;
    rec.log.push_back(std::move(op));
    ++rec.others;
}

/** @brief The second capture saves the test of the conditional node @p node into its slot, entry ``*iteration`` of the innermost WHILE (entry 0 outside one). */
void diff_save(eg_ctx& ctx, const DeviceRecording& rec, int node, const unsigned* cell, const unsigned* baseline) {
    for (const DiffSlot& slot : rec.dp->slots) {
        if (slot.cond != node) continue;
        const Frame* loop = innermost_loop(rec);
        save_test_step(ctx.dev.stream, cell, baseline, loop != nullptr ? loop->iteration : nullptr, slot.buffer, slot.capacity);
    }
}

/** @brief The second capture keeps the planes a windowed WHILE's body writes at the start of every window, as its forward does on the host. */
void diff_checkpoints(eg_ctx& ctx, const DeviceRecording& rec, int node) {
    const auto it = rec.dp->checkpoints.find(node);
    if (it == rec.dp->checkpoints.end()) return;
    const Frame* loop = innermost_loop(rec);
    for (const LeafCopy& c : it->second) table_copy_step(ctx.dev.stream, c.plane, c.bytes, c.table, loop->iteration, c.capacity, c.window);
}

}  // namespace

/** @brief eg_launch of a DEVICE plan while recording: its launches captured on the recording's stream. An index set's grid is
 *  its partition (the capacity) and its kernel reads the count plane at each replay, so a replay runs at the count last
 *  written; the plain launch's host read of the count (check_index_live) cannot be made inside a capture. */
void device_record_launch(eg_ctx& ctx, eg_plan_s& plan) {
    const eg_unit_s& u = *plan.unit;
    const std::string who = "recording a DEVICE launch of unit '" + u.name + "': ";
    if (!u.stages.empty()) refuse(EG_NOT_WRITTEN, who + "a split unit is not recorded");
    if (u.seg.present) refuse(EG_NOT_WRITTEN, who + "a SEGMENTED group reads its classes back to the host at each launch and is not recorded");
    if (!u.companions.empty() || !u.cells.empty())
        refuse(EG_NOT_WRITTEN, who + "companion and cell folds are not written in this build of the door");
    DeviceRecording& rec = *ctx.dev_rec;
    std::vector<PlaneUse> uses;
    for (std::size_t i = 0; i < plan.bindings.size(); ++i)
        uses.push_back({ plan.bindings[i].plane, slot_access(u.slots[i].role), static_cast<int>(i), u.slots[i].name });
    note_node(ctx, EG_GNODE_LAUNCH, "eg_launch of unit '" + u.name + "'", std::move(uses), u.name);
    Current current(ctx.dev);
    if (rec.dp != nullptr) diff_copies(ctx, rec, rec.plans.size());
    rec.log.push_back(Op{ Op::Launch, &plan, {}, {} });
    rec.bound.push_back(binding_bytes(plan));
    rec.plans.push_back(&plan);
    ++plan.refs;
    finished_zero(plan, ctx.dev.stream);    // each replay counts from zero, as eg_launch does
    launch_on(plan, ctx.dev.stream);
    // The kernel node of each partition, kept to re-apply the arguments of a plan rebound after the recording.
    const unsigned kernels = gemm_routed(plan) ? 0u : static_cast<unsigned>(std::count_if(plan.parts.begin(), plan.parts.end(), [](const exec::Partition& p) { return p.count > 0; }));
    rec.kernels.push_back(capture_kernels(ctx.dev.stream, kernels));
    rec.plan_piece.push_back(rec.pieces.size());
}

/** @brief eg_run of a DEVICE runner while recording: the capture in progress ends here and the run becomes a step of its own, which eg_graph_launch
 *  runs eagerly from the start of the runner (reset, then the run) between this capture and the next one. */
void device_record_run(eg_ctx& ctx, eg_runner_s* runner) {
    DeviceRecording& rec = *ctx.dev_rec;
    if (!rec.frames.empty())
        refuse(EG_NOT_WRITTEN, "recording the run of a DEVICE runner inside a conditional arm, a fork branch or a WHILE: a run is a step of the graph between two "
               "captures and cannot be a node inside one; record it outside them");
    std::vector<PlaneUse> uses;
    for (const eg_plan_s* plan : runner->plans)
        for (std::size_t i = 0; i < plan->bindings.size(); ++i)
            uses.push_back({ plan->bindings[i].plane, slot_access(plan->unit->slots[i].role), static_cast<int>(i), plan->unit->slots[i].name });
    NodeFacts facts;
    facts.n_plans = static_cast<std::int32_t>(runner->plans.size());
    note_node(ctx, EG_GNODE_RUN, "eg_run", std::move(uses), runner->plans.front()->unit->name, facts);
    Current current(ctx.dev);
    if (rec.dp != nullptr) diff_copies(ctx, rec, rec.dp->launches + rec.others);
    Op op;
    op.kind = Op::Leaf;
    op.leaf.kind = Node::Run;
    op.leaf.runner = runner;
    rec.log.push_back(std::move(op));
    ++rec.others;
    DeviceCapture* ended = rec.capture;
    rec.capture = nullptr;
    rec.pieces.push_back({ capture_end(ended), nullptr, nullptr });
    ++runner->refs;
    rec.pieces.push_back({ nullptr, runner, nullptr });
    rec.capture = capture_begin(rec.stream);
}

/** @brief eg_host_call of a device recording: the capture ends here and the call becomes a step of the graph, which eg_graph_launch runs on its own thread
 *  (after synchronizing the stream) between this capture and the next, as it does a run. */
void device_record_host_node(eg_ctx& ctx, Node&& node) {
    DeviceRecording& rec = *ctx.dev_rec;
    if (!rec.frames.empty())
        refuse(EG_NOT_WRITTEN, "recording " + host_node_label(node) + " inside a conditional arm, a fork branch or a WHILE of a device recording: a host node is a step of the graph between two "
               "captures and cannot be a node inside one; record it outside them");
    note_node(ctx, EG_GNODE_HOST, host_node_label(node), host_node_uses(node), node.host->name);
    Current current(ctx.dev);
    if (rec.dp != nullptr) diff_copies(ctx, rec, rec.dp->launches + rec.others);
    Op op;
    op.kind = Op::Leaf;
    op.leaf = node;
    rec.log.push_back(std::move(op));
    ++rec.others;
    auto step = std::make_shared<Node>(std::move(node));
    DeviceCapture* ended = rec.capture;
    rec.capture = nullptr;
    rec.pieces.push_back({ capture_end(ended), nullptr, nullptr });
    node_hold(*step);
    rec.pieces.push_back({ nullptr, nullptr, std::move(step) });
    rec.capture = capture_begin(rec.stream);
}

namespace {

DeviceRecording& recording_of(eg_ctx& ctx) {
    if (ctx.dev_rec == nullptr) refuse(EG_BAD_REQUEST, "not recording: call eg_record_begin first");
    return *ctx.dev_rec;
}

/** @brief The 32-bit device word at element @p index of the one-dimensional int32 or uint32 device plane @p plane, a conditional's
 *  test operand (@p what names it in a refusal). */
unsigned* test_word(eg_ctx& ctx, eg_plane_s* plane, std::int64_t index, const std::string& what) {
    check_handle(ctx.planes, plane, what.c_str());
    const DLTensor& t = plane->tensor->dl_tensor;
    if (t.device.device_type != kDLCUDA) refuse(EG_BAD_REQUEST, what + ": a device plane is expected in a device recording");
    if ((t.dtype.code != kDLInt && t.dtype.code != kDLUInt) || t.dtype.bits != 32 || t.dtype.lanes != 1 || t.ndim != 1)
        refuse(EG_BAD_REQUEST, what + ": a one-dimensional int32 or uint32 plane is expected");
    if (index < 0 || index >= t.shape[0])
        refuse(EG_BAD_REQUEST, what + ": index " + std::to_string(index) + " is outside the plane's " + std::to_string(t.shape[0]) + " elements");
    return reinterpret_cast<unsigned*>(static_cast<char*>(t.data) + t.byte_offset) + index;
}

/** @brief Free the captures among @p pieces and, when @p release, drop the runner references they hold and forget the steps. */
void drop_pieces(eg_ctx& ctx, std::vector<Piece>& pieces, bool release) noexcept {
    for (Piece& p : pieces) {
        loop_free(p.loop);
        p.loop = nullptr;
        if (release && p.runner != nullptr) runner_release(ctx, p.runner);
        if (release && p.host) node_drop(ctx, *p.host);
    }
    if (release) pieces.clear();
}

/** @brief Drop the recording in progress without a graph: the open weaves and forks, the capture, and what the recorded calls hold. */
void abort_recording(eg_ctx& ctx) noexcept {
    const std::unique_ptr<DeviceRecording> rec(ctx.dev_rec);
    ctx.dev_rec = nullptr;
    try {
        Current current(ctx.dev);
        for (std::size_t i = rec->frames.size(); i-- > 0;) {
            cond_drop(rec->frames[i].weave);
            fork_drop(rec->frames[i].fork);
        }
        ctx.dev.stream = rec->stream;
        capture_abort(rec->capture);
        pool_free(rec->pool);
        run_later(ctx, *rec);
        device_free(ctx, rec->loop_cells);
        drop_pieces(ctx, rec->pieces, true);
    } catch (...) {
    }
    release_hooks(ctx, rec->hooks);
    for (eg_plan_s* p : rec->plans) plan_release(ctx, p);
    for (eg_fft_s* f : rec->ffts) fft_release(ctx, f);
}

}  // namespace

/** @brief eg_record_cond_begin on a device context: a conditional node woven into the capture (device/Loop.cu); its first arm is being captured. */
void device_record_cond_begin(eg_ctx& ctx, const eg_cond_desc* desc) {
    DeviceRecording& rec = recording_of(ctx);
    const eg_cond_desc d = read_input(desc, "eg_cond_desc");
    if ((d.flags & ~EG_COND_NEGATE) != 0) refuse(EG_BAD_REQUEST, "eg_cond_desc.flags: unknown bits " + std::to_string(d.flags & ~EG_COND_NEGATE));
    unsigned arms = 1;
    switch (d.kind) {
        case EG_COND_IF:
        case EG_COND_WHILE: break;
        case EG_COND_IF_ELSE: arms = 2; break;
        case EG_COND_SWITCH:
            if (d.n_arms < 1) refuse(EG_BAD_REQUEST, "eg_cond_desc.n_arms " + std::to_string(d.n_arms) + " < 1");
            arms = static_cast<unsigned>(d.n_arms);
            break;
        default: refuse(EG_BAD_REQUEST, "unknown conditional kind " + std::to_string(d.kind));
    }
    if (d.kind != EG_COND_SWITCH && d.n_arms != 0) refuse(EG_BAD_REQUEST, "eg_cond_desc.n_arms is for SWITCH only: pass 0");
    const bool negate = (d.flags & EG_COND_NEGATE) != 0;
    if (negate && d.kind == EG_COND_SWITCH) refuse(EG_BAD_REQUEST, "eg_cond_desc.flags: EG_COND_NEGATE on a SWITCH");
    if (d.baseline != nullptr && d.kind == EG_COND_SWITCH) refuse(EG_BAD_REQUEST, "eg_cond_desc.baseline: a SWITCH reads its arm from the cell alone");
    if (d.max_iters < 0 || d.max_iters >= static_cast<std::int64_t>(0xFFFFFFFFu))
        refuse(EG_BAD_REQUEST, "eg_cond_desc.max_iters " + std::to_string(d.max_iters) + " is outside 0 .. 2^32 - 2");
    if (d.max_iters != 0 && d.kind != EG_COND_WHILE) refuse(EG_BAD_REQUEST, "eg_cond_desc.max_iters is for WHILE only: pass 0");
    if ((d.kind == EG_COND_IF_ELSE || d.kind == EG_COND_SWITCH) && !device_cond_arms_supported())
        refuse(EG_NOT_WRITTEN, "eg_record_cond_begin: IF_ELSE and SWITCH need a driver with conditional arms (12.8 or later); "
               "the capability bit EG_CAP_COND_IF_ELSE / EG_CAP_COND_SWITCH is not set on this context");
    if (d.kind == EG_COND_WHILE && rec.loops >= kRecordedLoops)
        refuse(EG_NOT_WRITTEN, "eg_record_cond_begin: a device recording holds at most " + std::to_string(kRecordedLoops) + " WHILE conditionals");
    if (!pool_has(rec.pool, arms, 0))
        refuse(EG_NOT_WRITTEN, "eg_record_cond_begin: a device recording holds at most " + std::to_string(kRecordStreams) + " open arms and branches at once");
    check_loop_control(ctx, d);
    const unsigned* cell = d.cell != nullptr || d.kind != EG_COND_WHILE ? test_word(ctx, d.cell, d.cell_index, "eg_cond_desc.cell") : nullptr;
    const unsigned* baseline = nullptr;
    if (d.baseline != nullptr) {
        baseline = test_word(ctx, d.baseline, d.baseline_index, "eg_cond_desc.baseline");
        if (d.baseline->tensor->dl_tensor.dtype.code != d.cell->tensor->dl_tensor.dtype.code)
            refuse(EG_BAD_REQUEST, "eg_cond_desc.baseline: the cell's dtype is expected");
    }
    NodeFacts facts;
    facts.cond_flags = negate ? EG_COND_NEGATE : 0;
    facts.has_baseline = d.baseline != nullptr;
    facts.max_iters = d.max_iters;
    facts.cell_index = d.cell_index;
    facts.has_brk = d.brk != nullptr;
    facts.counter = d.counter;
    const int node = note_node(ctx, EG_GNODE_COND, "a conditional", { { d.cell, 0, -1, "cell" }, { d.baseline, 0, -1, "baseline" },
                                                                   { d.counter, 1, -1, "counter" }, { d.limit, 0, -1, "limit" }, { d.brk, 2, -1, "brk" } }, "", facts);
    rec.shape.cond_begin(d.kind, arms);
    Current current(ctx.dev);
    if (rec.dp != nullptr && d.kind != EG_COND_WHILE) diff_save(ctx, rec, node, cell, baseline);    // outside the conditional: the test as it is when it is reached
    Op op;
    op.kind = Op::CondBegin;
    op.cond = d;
    rec.log.push_back(op);
    unsigned* loop = d.kind == EG_COND_WHILE ? static_cast<unsigned*>(rec.loop_cells) + 2 * rec.loops : nullptr;
    Frame frame;
    frame.arms = arms;
    frame.outer = ctx.dev.stream;
    frame.weave = cond_begin(rec.pool, ctx.dev.stream, static_cast<int>(d.kind), cell, baseline, negate, arms, static_cast<unsigned>(d.max_iters), loop,
                            d.limit != nullptr ? test_word(ctx, d.limit, d.limit_index, "eg_cond_desc.limit") : nullptr,
                            d.brk != nullptr ? test_word(ctx, d.brk, d.brk_index, "eg_cond_desc.brk") : nullptr);
    if (loop != nullptr) {
        frame.iteration = loop + 1;
        frame.cap = static_cast<unsigned>(d.max_iters);
        ++rec.loops;
    }
    ctx.dev.stream = cond_stream(frame.weave, 0);
    if (d.counter != nullptr) {      // the iteration index is visible to the body from its first node
        frame.counter = test_word(ctx, d.counter, 0, "eg_cond_desc.counter");
        counter_step(ctx.dev.stream, frame.iteration, frame.counter);
    }
    rec.frames.push_back(std::move(frame));
    if (rec.dp != nullptr && d.kind == EG_COND_WHILE) {    // at the start of each iteration: the test, and the window's checkpoint
        diff_checkpoints(ctx, rec, node);
        diff_save(ctx, rec, node, cell, baseline);
    }
}

/** @brief eg_record_abort on a device context: the open capture is ended and destroyed with everything the recording holds. */
void device_record_abort(eg_ctx& ctx) {
    recording_of(ctx);
    abort_recording(ctx);
}

/** @brief eg_record_cond_arm on a device context: the capture moves to the next arm's stream. */
void device_record_cond_arm(eg_ctx& ctx) {
    DeviceRecording& rec = recording_of(ctx);
    rec.shape.cond_arm();
    rec.log.push_back(Op{ Op::CondArm, nullptr, {}, {} });
    Frame& f = rec.frames.back();
    ++f.arm;
    ctx.dev.stream = cond_stream(f.weave, f.arm);
}

/** @brief eg_record_cond_end on a device context: every arm is closed (each must hold a node) and the capture goes on after the conditional. */
void device_record_cond_end(eg_ctx& ctx) {
    DeviceRecording& rec = recording_of(ctx);
    rec.shape.cond_end(true);
    rec.log.push_back(Op{ Op::CondEnd, nullptr, {}, {} });
    const Frame f = rec.frames.back();
    rec.frames.pop_back();
    ctx.dev.stream = f.outer;
    Current current(ctx.dev);
    try {
        cond_end(f.weave);
        if (f.counter != nullptr) counter_step(ctx.dev.stream, f.iteration, f.counter);    // the loop's completed iterations
    } catch (const std::invalid_argument& e) {
        refuse(EG_BAD_REQUEST, std::string("eg_record_cond_end: ") + e.what());
    }
}

/** @brief eg_record_fork_begin on a device context: branch 0 is a stream that starts after everything recorded so far. */
void device_record_fork_begin(eg_ctx& ctx) {
    DeviceRecording& rec = recording_of(ctx);
    if (!pool_has(rec.pool, 1, 2))
        refuse(EG_NOT_WRITTEN, "eg_record_fork_begin: a device recording holds at most " + std::to_string(kRecordStreams) + " open arms and branches at once");
    rec.shape.fork_begin();
    rec.log.push_back(Op{ Op::ForkBegin, nullptr, {}, {} });
    Current current(ctx.dev);
    Frame frame;
    frame.outer = ctx.dev.stream;
    frame.fork = fork_begin(rec.pool, ctx.dev.stream);
    frame.arms = 1;
    ctx.dev.stream = fork_stream(frame.fork, 0);
    rec.frames.push_back(std::move(frame));
}

/** @brief eg_record_fork_branch on a device context: close the current branch (it must hold a node) and open the next. */
void device_record_fork_branch(eg_ctx& ctx) {
    DeviceRecording& rec = recording_of(ctx);
    if (!pool_has(rec.pool, 1, 1))
        refuse(EG_NOT_WRITTEN, "eg_record_fork_branch: a device recording holds at most " + std::to_string(kRecordStreams) + " open arms and branches at once");
    rec.shape.fork_branch();
    rec.log.push_back(Op{ Op::ForkBranch, nullptr, {}, {} });
    Frame& f = rec.frames.back();
    Current current(ctx.dev);
    ctx.dev.stream = fork_branch(f.fork);
    ++f.arms;
}

/** @brief eg_record_fork_end on a device context: a race between the branches ends the recording (as the host's does), else they join. */
void device_record_fork_end(eg_ctx& ctx) {
    DeviceRecording& rec = recording_of(ctx);
    const std::string race = rec.shape.fork_end();
    if (!race.empty()) refuse(EG_BAD_REQUEST, race);    // the fork stays open, as before the call
    rec.log.push_back(Op{ Op::ForkEnd, nullptr, {}, {} });
    const Frame f = rec.frames.back();
    rec.frames.pop_back();
    ctx.dev.stream = f.outer;
    Current current(ctx.dev);
    fork_end(f.fork);
}

/** @brief eg_record_wait on a device context: the current branch's stream waits for the earlier branches of its fork (the shape refuses a
 *  plane no earlier branch writes), which orders it after the writers and a little more. */
void device_record_wait(eg_ctx& ctx, std::int32_t n_planes, eg_plane_s* const* planes) {
    DeviceRecording& rec = recording_of(ctx);
    if (n_planes < 1 || planes == nullptr) refuse(EG_BAD_REQUEST, "eg_record_wait: n_planes " + std::to_string(n_planes) + " without planes");
    for (std::int32_t k = 0; k < n_planes; ++k) check_handle(ctx.planes, planes[k], ("eg_record_wait: planes[" + std::to_string(k) + "]").c_str());
    if (rec.frames.empty() || rec.frames.back().fork == nullptr) rec.shape.wait({});    // the shape names what is wrong
    Frame& f = rec.frames.back();
    const unsigned self = f.arms - 1;
    if (!pool_has(rec.pool, 0, self))
        refuse(EG_NOT_WRITTEN, "eg_record_wait: a device recording holds at most " + std::to_string(kRecordEvents) + " events for its forks and waits");
    rec.shape.wait(std::vector<eg_plane_s*>(planes, planes + n_planes));
    rec.log.push_back(Op{ Op::Wait, nullptr, {}, std::vector<eg_plane_s*>(planes, planes + n_planes) });
    Current current(ctx.dev);
    for (unsigned j = 0; j < self; ++j) fork_wait(f.fork, self, j);
}

/** @brief Whether this context's driver has conditional arms (IF_ELSE, SWITCH): the capability bits' truth on a device context. */
bool device_cond_arms_supported() { return cond_arms_supported(); }

/** @brief The shape of the device recording in progress (device/Seam.h): the host calls a device recording keeps as nodes add themselves to it. */
GraphShape& device_shape(eg_ctx& ctx) { return recording_of(ctx).shape; }

/** @brief eg_graph_describe of a device graph: the description built at record time. */
void device_graph_describe(eg_ctx&, const eg_graph* handle, eg_graph_description* out) {
    const auto* graph = reinterpret_cast<const DeviceGraph*>(handle);
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_graph_describe: out is NULL");
    if (out->struct_size < 2 * sizeof(std::uint32_t)) refuse(EG_BAD_REQUEST, "eg_graph_description: struct_size " + std::to_string(out->struct_size) + " is too small");
    eg_graph_description d;
    graph->shape.describe(graph->described, graph->diff != nullptr ? EG_RECORD_DIFF : 0, d);
    write_output(out, d, "eg_graph_description");
}

/** @brief A host node on the capture in scope (device/Seam.h). Nothing in it makes a CUDA call. */
void device_record_host_step(eg_ctx& ctx, void (*run)(void*, StepFailure&) noexcept, void* user, void (*release)(eg_ctx&, void*)) {
    DeviceRecording& rec = *ctx.dev_rec;
    EG_DRIVER(cuLaunchHostFunc);
    if (!rec.failure) rec.failure = std::make_shared<StepFailure>();
    rec.hooks.push_back(std::make_unique<HostHook>(HostHook{ run, user, release, rec.failure }));
    Current current(ctx.dev);
    const CUresult r = cuLaunchHostFunc_(reinterpret_cast<CUstream>(static_cast<std::uint64_t>(ctx.dev.stream)), &host_hook_run, rec.hooks.back().get());
    if (r != CUDA_SUCCESS) {
        rec.hooks.pop_back();    // the caller still owns the step it passed
        check(r, "cuLaunchHostFunc");
    }
}

namespace {

/** @brief Free what @p rec holds: its plan and transform references, its steps, hooks and loop states. Run after the capture ended or was dropped. */
void release_recording(eg_ctx& ctx, DeviceRecording& rec) {
    drop_pieces(ctx, rec.pieces, true);
    release_hooks(ctx, rec.hooks);
    for (eg_plan_s* p : rec.plans) plan_release(ctx, p);
    for (eg_fft_s* f : rec.ffts) fft_release(ctx, f);
    device_free(ctx, rec.loop_cells);
    rec.loop_cells = nullptr;
}

/** @brief What the second capture of a EG_RECORD_DIFF recording adds, built from the tape @p diff (device memory made here, in @p owned, outside any
 *  capture): a save slot for every conditional with tests, and the copies the tape keeps in front of each launch (one table per plane of a WHILE body). */
void plan_diff_pass(eg_ctx& ctx, const GraphDiff& diff, std::size_t n_launches, DiffPass& pass, std::vector<void*>& owned) {
    const ReversePlan& plan = diff.plan;
    const auto grab = [&](std::size_t bytes) {
        void* data = device_alloc(ctx, std::max<std::size_t>(bytes, 8), true);
        owned.push_back(data);
        return data;
    };
    std::map<int, unsigned> entries;                                            // per conditional node: its tests (one per iteration inside a loop)
    for (const RTest& t : plan.tests) {
        if (t.key.size() > 2) refuse(EG_NOT_WRITTEN, "eg_record_end: a test inside two nested WHILE loops is not written on a device");
        ++entries[t.key.front()];
    }
    for (const auto& e : entries) pass.slots.push_back({ e.first, e.second, static_cast<unsigned*>(grab(static_cast<std::size_t>(e.second) * 3 * sizeof(unsigned))) });
    pass.launches = n_launches;
    pass.before.assign(diff.nodes.size(), {});
    std::map<std::pair<int, int>, std::vector<void*>> tables;                   // (launch, plane) in a loop: the copy of each iteration
    std::map<int, unsigned> iterations;                                         // per launch in a loop: its iteration count
    for (const RLeaf& l : plan.leaves) {
        if (l.key.size() > 2) refuse(EG_NOT_WRITTEN, "eg_record_end: a launch inside two nested WHILE loops is not written on a device");
        if (l.key.size() == 2) iterations[l.node] = std::max<unsigned>(iterations[l.node], static_cast<unsigned>(l.key[1]) + 1);
    }
    for (const Snapshot& snap : diff.snapshots) {
        const RLeaf& leaf = plan.leaves[snap.node];
        if (diff.planes[static_cast<std::size_t>(snap.plane)]->tensor->dl_tensor.device.device_type == kDLCPU)
            refuse(EG_NOT_WRITTEN, "eg_record_end: a later leaf overwrites a host plane an earlier leaf of the EG_RECORD_DIFF recording reads; the tape's copy of a host plane is not written on a device");
        if (leaf.loop >= 0) continue;                                           // a windowed WHILE's states are rebuilt by the reverse from its checkpoints
        const DLTensor& t = diff.planes[static_cast<std::size_t>(snap.plane)]->tensor->dl_tensor;
        const char* plane = static_cast<const char*>(t.data) + t.byte_offset;
        auto& copies = pass.before[static_cast<std::size_t>(leaf.node)];
        if (leaf.key.size() == 1) {
            copies.push_back({ plane, snap.size, const_cast<char*>(snap.bytes.data()), nullptr, 0 });
            continue;
        }
        std::vector<void*>& table = tables[{ leaf.node, snap.plane }];
        if (table.empty()) {
            table.assign(iterations[leaf.node], grab(snap.size));              // an iteration with no copy of its own writes a spare
            copies.push_back({ plane, snap.size, nullptr, nullptr, iterations[leaf.node] });
        }
        table[static_cast<std::size_t>(leaf.key[1])] = const_cast<char*>(snap.bytes.data());
    }
    for (std::size_t L = 0; L < plan.loops.size(); ++L) {
        const RLoop& loop = plan.loops[L];
        const LoopTape& tape = diff.loops[L];
        for (std::size_t q = 0; q < tape.planes.size(); ++q) {
            const DLTensor& t = diff.planes[static_cast<std::size_t>(tape.planes[q])]->tensor->dl_tensor;
            std::vector<void*> table;
            for (const std::vector<Bytes>& group : tape.ckpt) table.push_back(const_cast<char*>(group[q].data()));
            void** device_table = static_cast<void**>(grab(table.size() * sizeof(void*)));
            device_write(ctx, device_table, table.data(), table.size() * sizeof(void*));
            pass.checkpoints[loop.cond].push_back({ static_cast<const char*>(t.data) + t.byte_offset, static_cast<std::size_t>(elements(t)) * (t.dtype.bits / 8),
                                                    nullptr, device_table, static_cast<unsigned>(loop.groups), static_cast<unsigned>(loop.w) });
        }
    }
    for (auto& entry : tables) {
        void** device_table = static_cast<void**>(grab(entry.second.size() * sizeof(void*)));
        device_write(ctx, device_table, entry.second.data(), entry.second.size() * sizeof(void*));
        for (LeafCopy& c : pass.before[static_cast<std::size_t>(entry.first.first)])
            if (c.direct == nullptr && c.table == nullptr && c.plane == static_cast<const char*>(diff.planes[static_cast<std::size_t>(entry.first.second)]->tensor->dl_tensor.data) +
                                                              diff.planes[static_cast<std::size_t>(entry.first.second)]->tensor->dl_tensor.byte_offset)
                c.table = device_table;
    }
}

/** @brief Capture the copy, reduce, transform or product kept in @p op again. */
void replay_leaf(eg_ctx& ctx, const Op& op) {
    const Node& n = op.leaf;
    switch (n.kind) {
        case Node::Copy: device_copy(ctx, n.src, n.dst); break;
        case Node::Reduce: device_reduce_leaf(ctx, n.reduce); break;
        case Node::Fft: device_fft_run(ctx, *n.fft, n.src, n.dst); break;
        case Node::Run: device_record_run(ctx, n.runner); break;
        case Node::Host: device_record_host_node(ctx, Node(n)); break;
        default: device_gemm_run(ctx, op.product, n.src, n.rhs, n.dst);
    }
}

/** @brief Capture @p rec's calls again with @p pass's copies and saves in them: the capture of a EG_RECORD_DIFF recording. @p ctx.dev_rec is the new recording. */
void capture_again(eg_ctx& ctx, const DeviceRecording& first, DeviceRecording& second) {
    for (const Op& op : first.log) {
        switch (op.kind) {
            case Op::Launch: device_record_launch(ctx, *op.plan); break;
            case Op::Leaf: replay_leaf(ctx, op); break;
            case Op::CondBegin: device_record_cond_begin(ctx, &op.cond); break;
            case Op::CondArm: device_record_cond_arm(ctx); break;
            case Op::CondEnd: device_record_cond_end(ctx); break;
            case Op::ForkBegin: device_record_fork_begin(ctx); break;
            case Op::ForkBranch: device_record_fork_branch(ctx); break;
            case Op::ForkEnd: device_record_fork_end(ctx); break;
            case Op::Wait: device_record_wait(ctx, static_cast<std::int32_t>(op.planes.size()), op.planes.data()); break;
        }
    }
    (void)second;
}

}  // namespace

/** @brief eg_record_end on a device context: end the capture and instantiate it (FAULT when the capture failed). With EG_RECORD_DIFF the shape's
 *  reverse plan and the tape are made first, and the recording is captured again with the tape's copies and the tests' saves in it. */
void device_record_end(eg_ctx& ctx, const eg_record_end_desc* desc, eg_graph** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_record_end: out is NULL");
    *out = nullptr;
    if (ctx.dev_rec->shape.open()) refuse(EG_BAD_REQUEST, "eg_record_end: a conditional or fork is still open");
    bool diff_wanted = false;
    ReversePlan reverse;
    if (desc != nullptr) {
        const eg_record_end_desc d = read_input(desc, "eg_record_end_desc");
        if ((d.flags & ~EG_RECORD_DIFF) != 0) refuse(EG_BAD_REQUEST, "eg_record_end_desc.flags: unknown bits " + std::to_string(d.flags & ~EG_RECORD_DIFF));
        diff_wanted = (d.flags & EG_RECORD_DIFF) != 0;
        if (!diff_wanted && d.n_vjps != 0) refuse(EG_BAD_REQUEST, "eg_record_end_desc.n_vjps: vjps without EG_RECORD_DIFF");
        if (diff_wanted) {                       // a refusal leaves the recording open, as the host's does
            reverse = ctx.dev_rec->shape.reverse_plan();
            if (ctx.dev_rec->shape.leaves() != static_cast<int>(ctx.dev_rec->plans.size() + ctx.dev_rec->others))
                refuse(EG_NOT_WRITTEN, "eg_record_end: a call that runs on the host in a device recording (a host plan, the copy or reduce of host planes) is not written in EG_RECORD_DIFF");
            std::vector<const Node*> leaves;
            for (const Op& op : ctx.dev_rec->log)
                if (op.kind == Op::Leaf) leaves.push_back(&op.leaf);
            leaf_checks(leaves);
        }
    }
    std::unique_ptr<DeviceRecording> rec(ctx.dev_rec);
    ctx.dev_rec = nullptr;
    Current current(ctx.dev);
    DeviceLoop* loop = nullptr;
    std::string failed;
    try {
        if (rec->capture == nullptr) throw std::runtime_error("a run of a DEVICE runner could not start the capture after it");
        loop = capture_end(rec->capture);
    } catch (const std::exception& e) {
        failed = e.what();
    } catch (...) {
        failed = "unknown error";
    }
    rec->capture = nullptr;
    pool_free(rec->pool);
    rec->pool = nullptr;
    run_later(ctx, *rec);
    if (loop == nullptr) {
        release_recording(ctx, *rec);
        refuse(EG_FAULT, "eg_record_end: the device capture failed: " + failed);
    }
    rec->pieces.push_back({ loop, nullptr, nullptr });
    loop = nullptr;
    auto graph = std::make_unique<DeviceGraph>();
    if (diff_wanted) {
        // The tape allocates (not allowed inside a capture), so it is planned now, and the calls are captured again with its copies and saves in them.
        DiffPass pass;
        std::unique_ptr<DeviceRecording> second;
        try {
            std::vector<const Node*> others;
            for (Op& op : rec->log)
                if (op.kind == Op::Leaf) {
                    if (op.leaf.kind == Node::Gemm) op.leaf.gemm = &op.product;
                    others.push_back(&op.leaf);
                }
            graph->diff = diff_plan(ctx, rec->plans, others, reverse, *desc);
            drop_pieces(ctx, rec->pieces, false);    // the first capture served the plan; the second is the graph
            plan_diff_pass(ctx, *graph->diff, rec->plans.size(), pass, graph->buffers);
            second = std::make_unique<DeviceRecording>();
            second->stream = rec->stream;
            second->loop_cells = device_alloc(ctx, kRecordedLoops * 2 * sizeof(unsigned), true);
            second->pool = pool_new(kRecordStreams, kRecordEvents);
            second->capture = capture_begin(second->stream);
            second->dp = &pass;
            ctx.dev_rec = second.get();
            capture_again(ctx, *rec, *second);
            ctx.dev_rec = nullptr;
            DeviceCapture* ended = second->capture;
            second->capture = nullptr;
            loop = capture_end(ended);
            second->pieces.push_back({ loop, nullptr, nullptr });
            loop = nullptr;
        } catch (...) {
            const bool opened = second != nullptr && second->capture != nullptr;
            ctx.dev_rec = nullptr;
            ctx.dev.stream = rec->stream;
            if (second != nullptr) {
                for (Frame& f : second->frames) {
                    cond_drop(f.weave);
                    fork_drop(f.fork);
                }
                if (opened) capture_abort(second->capture);
                pool_free(second->pool);
                run_later(ctx, *second);
                release_recording(ctx, *second);
            }
            if (loop != nullptr) loop_free(loop);
            if (graph->diff != nullptr) diff_free(ctx, *graph->diff);
            for (void* p : graph->buffers) device_free(ctx, p);
            release_recording(ctx, *rec);
            throw;
        }
        pool_free(second->pool);
        run_later(ctx, *second);
        for (eg_plan_s* p : rec->plans) plan_release(ctx, p);    // the second capture holds its own references
        rec->plans.clear();
        for (eg_fft_s* f : rec->ffts) fft_release(ctx, f);
        rec->ffts.clear();
        drop_pieces(ctx, rec->pieces, true);
        device_free(ctx, rec->loop_cells);
        rec->loop_cells = nullptr;
        graph->slots = pass.slots;
        rec = std::move(second);
    }
    graph->pieces = std::move(rec->pieces);
    graph->plan_piece = std::move(rec->plan_piece);
    graph->shape = std::move(rec->shape);
    graph->kernels = std::move(rec->kernels);
    graph->loops = rec->loops;
    if (rec->loops != 0) graph->loop_cells = rec->loop_cells;
    else device_free(ctx, rec->loop_cells);
    graph->plans = std::move(rec->plans);
    graph->ffts = std::move(rec->ffts);
    graph->bound = std::move(rec->bound);
    graph->hooks = std::move(rec->hooks);
    graph->failure = std::move(rec->failure);
    ctx.dev_graphs.insert(graph.get());
    *out = reinterpret_cast<eg_graph*>(graph.release());
}

/** @brief The tests the last replay of a device graph saved, into ``diff.saved``: every slot read in one sync, each test by its key (the conditional's
 *  node ordinal, then the iteration of the WHILE around it or at it). */
void device_diff_read_saved(eg_ctx& ctx, GraphDiff& diff) {
    const DeviceGraph* graph = nullptr;
    for (const DeviceGraph* g : ctx.dev_graphs)
        if (g->diff.get() == &diff) graph = g;
    if (graph == nullptr) return;
    std::vector<std::vector<std::uint32_t>> raw(graph->slots.size());
    std::vector<ReadItem> items;
    for (std::size_t i = 0; i < graph->slots.size(); ++i) {
        raw[i].assign(static_cast<std::size_t>(graph->slots[i].capacity) * 3, 0u);
        items.push_back({ raw[i].data(), graph->slots[i].buffer, raw[i].size() * sizeof(std::uint32_t) });
    }
    Current current(ctx.dev);
    read_many(ctx.dev.stream, items);
    for (std::size_t t = 0; t < diff.plan.tests.size(); ++t) {
        const std::vector<int>& key = diff.plan.tests[t].key;
        for (std::size_t i = 0; i < graph->slots.size(); ++i) {
            if (graph->slots[i].cond != key.front()) continue;
            const std::size_t entry = key.size() > 1 ? static_cast<std::size_t>(key.back()) : 0;
            TestSave& save = diff.saved[t];
            save.evaluated = raw[i][3 * entry] != 0;
            save.cell = static_cast<std::int32_t>(raw[i][3 * entry + 1]);
            save.base = static_cast<std::int32_t>(raw[i][3 * entry + 2]);
        }
    }
}

/** @brief Whether @p graph is a device graph of @p ctx. */
bool device_graph(const eg_ctx& ctx, const eg_graph* graph) {
    return ctx.dev_graphs.count(reinterpret_cast<DeviceGraph*>(const_cast<eg_graph*>(graph))) != 0;
}

/** @brief Re-apply to the instantiated graph the arguments of every recorded launch whose plan was rebound since the last launch,
 *  and make the instance again when the driver refuses to update a node of it: a replay never runs on stale arguments. A plan the
 *  graph holds as cuBLAS products (no kernel node) or in the tape of EG_RECORD_DIFF cannot be re-applied and is refused. */
/** @brief Give kernel node @p node of @p loop the arguments @p args followed by the triple (@p base, @p count, @p n), in the template
 *  and in the instance; false when the driver refused the update of the instance (the template is updated either way). */
bool set_kernel_args(const DeviceLoop* loop, std::uint64_t node, const std::vector<void*>& args, std::int64_t base, std::int64_t count, std::int64_t n) {
    static const auto get = cuda::detail::driver_entry<decltype(&::cuGraphKernelNodeGetParams)>("cuGraphKernelNodeGetParams");
    static const auto set = cuda::detail::driver_entry<decltype(&::cuGraphKernelNodeSetParams)>("cuGraphKernelNodeSetParams");
    static const auto set_exec = cuda::detail::driver_entry<decltype(&::cuGraphExecKernelNodeSetParams)>("cuGraphExecKernelNodeSetParams");
    const auto at = reinterpret_cast<CUgraphNode>(node);
    CUDA_KERNEL_NODE_PARAMS p = {};
    check(get(at, &p), "cuGraphKernelNodeGetParams");
    std::vector<void*> params = args;
    params.push_back(&base);
    params.push_back(&count);
    params.push_back(&n);
    p.kernelParams = params.data();
    p.extra = nullptr;
    check(set(at, &p), "cuGraphKernelNodeSetParams");
    return set_exec(reinterpret_cast<CUgraphExec>(loop_exec(loop)), at, &p) == CUDA_SUCCESS;
}

void reapply_bindings(DeviceGraph& graph) {
    std::set<std::size_t> stale;    // the captures whose instance the driver refused to update
    for (std::size_t i = 0; i < graph.plans.size(); ++i) {
        const eg_plan_s& plan = *graph.plans[i];
        if (binding_bytes(plan) == graph.bound[i]) continue;
        if (graph.diff != nullptr || gemm_routed(plan))
            refuse(EG_BAD_REQUEST, "eg_graph_launch: a plan of unit '" + plan.unit->name + "' was rebound since the recording; " +
                   (graph.diff != nullptr ? "a graph recorded with EG_RECORD_DIFF replays the planes it recorded" : "a product plan's bindings are fixed in the recording") +
                   ": record it again");
        std::size_t k = 0;
        for (const exec::Partition& p : plan.parts) {
            if (p.count <= 0) continue;
            if (!set_kernel_args(graph.pieces[graph.plan_piece[i]].loop, graph.kernels[i][k++], plan.params, p.base, p.count, p.nSamples)) stale.insert(graph.plan_piece[i]);
        }
        graph.bound[i] = binding_bytes(plan);
    }
    for (std::size_t piece : stale) loop_reinstantiate(graph.pieces[piece].loop);
}

/** @brief eg_graph_launch of a device graph: replay it once on the stream in scope (``GraphPipeline.launch``), on the bindings the
 *  plans hold now (a rebound plan's arguments are re-applied first). */
void device_graph_launch(eg_ctx& ctx, eg_graph* handle) {
    auto* graph = reinterpret_cast<DeviceGraph*>(handle);
    device_quiet(ctx, "eg_graph_launch", EG_NOT_WRITTEN, "a graph inside a device graph is not written in this build of the door");
    if (graph->failure) raise_step_failure(*graph->failure);    // a host node of an earlier replay failed
    Current current(ctx.dev);
    reapply_bindings(*graph);
    if (graph->diff != nullptr) diff_check_launch(*graph->diff);
    for (const eg_plan_s* p : graph->plans) fault_reset(ctx, *p);    // a checked unit's cell starts clear at each replay
    for (const DiffSlot& slot : graph->slots)    // the tests this replay will save start as not evaluated
        zero_bytes(ctx.dev.stream, slot.buffer, static_cast<std::size_t>(slot.capacity) * 3 * sizeof(unsigned));
    for (const Piece& piece : graph->pieces) {    // the captures and the runs between them, in the order recorded
        if (piece.loop != nullptr) loop_launch(piece.loop, ctx.dev.stream);
        else if (piece.host) host_node_run(ctx, *piece.host);
        else {
            fault_reset(ctx, *piece.runner);
            run_from_start(ctx, piece.runner);
            fault_check(ctx, *piece.runner);
        }
    }
    for (const eg_plan_s* p : graph->plans) fault_check(ctx, *p);
    if (graph->diff != nullptr) diff_launched(*graph->diff);
}

GraphDiff* device_graph_diff(eg_ctx&, eg_graph* graph) { return reinterpret_cast<DeviceGraph*>(graph)->diff.get(); }

/** @brief eg_graph_free of a device graph: drop its plan references and its instantiated capture. */
void device_graph_free(eg_ctx& ctx, eg_graph* handle) {
    auto* graph = reinterpret_cast<DeviceGraph*>(handle);
    if (!graph->caller_ref) refuse(EG_BAD_REQUEST, "graph: already freed by the caller");
    graph->caller_ref = false;
    device_graph_release(ctx, handle);
}

void device_graph_retain(eg_graph* handle) { ++reinterpret_cast<DeviceGraph*>(handle)->refs; }

void device_graph_release(eg_ctx& ctx, eg_graph* handle) {
    auto* graph = reinterpret_cast<DeviceGraph*>(handle);
    if (--graph->refs > 0) return;
    ctx.dev_graphs.erase(graph);
    const std::unique_ptr<DeviceGraph> owned(graph);
    if (!graph->hooks.empty()) {    // replays may still be queued: their host nodes run before the references go
        try {
            drain_stream(ctx);
        } catch (...) {
        }
        release_hooks(ctx, graph->hooks);
    }
    for (eg_plan_s* p : graph->plans) plan_release(ctx, p);
    for (eg_fft_s* f : graph->ffts) fft_release(ctx, f);
    if (graph->diff != nullptr) diff_free(ctx, *graph->diff);
    eg_ctx* c = &ctx;
    std::vector<Piece> pieces = std::move(graph->pieces);
    for (Piece& p : pieces) {
        if (p.runner != nullptr) runner_release(ctx, p.runner);
        if (p.host) node_drop(ctx, *p.host);
    }
    now_or_after_capture(ctx, [c, pieces] {
        Current current(c->dev);
        for (const Piece& p : pieces) loop_free(p.loop);
    });
    device_free(ctx, graph->loop_cells);
    for (void* p : graph->buffers) device_free(ctx, p);
}

/** @brief At context free: end a capture in progress (dropping it) and free every device graph, without releasing
 *  their references (the context frees every plan itself right after); a forked child makes no device call. */
void device_graphs_drop(eg_ctx& ctx) noexcept {
    const bool child = forked(ctx);
    if (ctx.dev_rec != nullptr) {
        const std::unique_ptr<DeviceRecording> rec(ctx.dev_rec);
        ctx.dev_rec = nullptr;
        if (!child) {
            try {
                Current current(ctx.dev);
                for (std::size_t i = rec->frames.size(); i-- > 0;) {    // the open weaves and forks hold streams of the capture: they go first
                    cond_drop(rec->frames[i].weave);
                    fork_drop(rec->frames[i].fork);
                }
                ctx.dev.stream = rec->stream;
                capture_abort(rec->capture);
                pool_free(rec->pool);
                run_later(ctx, *rec);
            } catch (...) {
            }
            release_hooks(ctx, rec->hooks);
        }
    }
    for (DeviceGraph* g : ctx.dev_graphs) {
        if (!child) {
            try {
                Current current(ctx.dev);
                drop_pieces(ctx, g->pieces, false);
            } catch (...) {
            }
            if (!g->hooks.empty()) {
                try {
                    drain_stream(ctx);
                } catch (...) {
                }
                release_hooks(ctx, g->hooks);
            }
        }
        delete g;
    }
    ctx.dev_graphs.clear();
}

/** @brief The marker bit of a handle eg_stream_new makes: a driver stream is a pointer and never has it, so a handle and a
 *  caller's raw stream cannot be mistaken for each other. */
constexpr std::uint64_t kMadeStreamTag = std::uint64_t{ 1 } << 63;

/** @brief Refuse BAD_REQUEST naming @p call while a recording is open: the capture is on the stream in scope. */
void refuse_stream_call_in_recording(const eg_ctx& ctx, const char* call) {
    if (ctx.dev_rec != nullptr)
        refuse(EG_BAD_REQUEST, std::string(call) + " while recording: the capture is on the stream in scope; call it before "
               "eg_record_begin or after eg_record_end");
}

/** @brief The CUstream of the handle @p stream of eg_stream_new, refusing one the context did not make or already freed. */
std::uint64_t made_stream(const eg_ctx& ctx, std::uint64_t stream, const char* call) {
    const auto at = ctx.dev.made_streams.find(stream);
    if (at == ctx.dev.made_streams.end())
        refuse(EG_BAD_REQUEST, std::string(call) + ": stream handle " + std::to_string(stream) +
               " was not made by eg_stream_new on this context, or it was already freed");
    return at->second;
}

/** @brief Refuse BAD_REQUEST unless the raw stream @p stream belongs to the context's device. */
void check_raw_stream(const eg_ctx& ctx, std::uint64_t stream, const char* call) {
    EG_DRIVER(cuStreamGetCtx);
    Current current(ctx.dev);
    CUcontext owner = nullptr;
    check(cuStreamGetCtx_(reinterpret_cast<CUstream>(stream), &owner), std::string(call) + ": cuStreamGetCtx");
    if (owner != reinterpret_cast<CUcontext>(ctx.dev.context))
        refuse(EG_BAD_REQUEST, std::string(call) + ": the stream belongs to another context than device " +
               std::to_string(ctx.dev.ordinal) + "'s primary context");
}

/** @brief eg_stream_use: make @p stream (0: eagle's own; a handle of eg_stream_new; or a raw stream of the context's device)
 *  the stream in scope of the calling thread; the launches, runs and replays that thread makes after it are ordered on it. */
void device_stream_use(eg_ctx& ctx, std::uint64_t stream) {
    if (ctx.dev.ordinal < 0)
        refuse(EG_BAD_REQUEST, "eg_stream_use on a host-only context (made without EG_CTX_DEVICE): it has no stream");
    refuse_stream_call_in_recording(ctx, "eg_stream_use");
    std::uint64_t scope = ctx.dev.own_stream;
    if ((stream & kMadeStreamTag) != 0) {
        scope = made_stream(ctx, stream, "eg_stream_use");
    } else if (stream != 0) {
        check_raw_stream(ctx, stream, "eg_stream_use");
        ctx.dev.adopted_streams.insert(stream);
        scope = stream;
    }
    if (!ctx.dev.gemm_handles.empty()) {    // a gemm unit was planned: the stream gets its handle now, outside any capture
        Current current(ctx.dev);
        gemm_handle_make(ctx, scope);
    }
    ctx.dev.stream = scope;
}

/** @brief The ordinal the text @p device of eg_stream_new names: "" the context's own device, else "cuda:N" or "N". */
std::int32_t stream_device(const eg_ctx& ctx, const std::string& device) {
    if (device.empty()) return ctx.dev.ordinal;
    const std::string digits = device.rfind("cuda:", 0) == 0 ? device.substr(5) : device;
    const bool numeric = !digits.empty() && digits.size() < 9 && digits.find_first_not_of("0123456789") == std::string::npos;
    std::string why;
    if (!numeric || std::stoi(digits) >= driver_device_count(why))
        refuse(EG_BAD_REQUEST, "eg_stream_new: '" + device + "' names no device of this backend (write cuda:N, N below the device count)");
    return std::stoi(digits);
}

/** @brief eg_stream_new: a queue the library owns on the context's device; its handle carries kMadeStreamTag. */
void device_stream_new(eg_ctx& ctx, const std::string& device, std::uint64_t* stream) {
    refuse_stream_call_in_recording(ctx, "eg_stream_new");
    const std::int32_t ordinal = stream_device(ctx, device);
    if (ordinal != ctx.dev.ordinal)
        refuse(EG_NOT_WRITTEN, "eg_stream_new: device " + std::to_string(ordinal) + " is not the context's device " +
               std::to_string(ctx.dev.ordinal) + "; a stream of another device needs a context of that device (several devices are not written)");
    EG_DRIVER(cuStreamCreate);
    Current current(ctx.dev);
    CUstream made = nullptr;
    check(cuStreamCreate_(&made, CU_STREAM_NON_BLOCKING), "eg_stream_new: cuStreamCreate");
    const std::uint64_t handle = kMadeStreamTag | ++ctx.dev.next_made_stream;
    ctx.dev.made_streams[handle] = reinterpret_cast<std::uint64_t>(made);
    *stream = handle;
}

/** @brief eg_stream_free: wait for the work queued on the handle @p stream, hand the planes made on it to the context's own
 *  stream (their release must not name a dead stream) and destroy it; the calling thread's scope falls back to the own stream. */
void device_stream_free(eg_ctx& ctx, std::uint64_t stream) {
    refuse_stream_call_in_recording(ctx, "eg_stream_free");
    const std::uint64_t queue = made_stream(ctx, stream, "eg_stream_free");
    EG_DRIVER(cuStreamSynchronize);
    EG_DRIVER(cuStreamDestroy);
    Current current(ctx.dev);
    check(cuStreamSynchronize_(reinterpret_cast<CUstream>(queue)), "eg_stream_free: cuStreamSynchronize");
    for (auto& block : ctx.dev.alloc_stream)
        if (block.second == queue) block.second = ctx.dev.own_stream;
    gemm_handle_drop(ctx, queue);
    if (const auto at = ctx.dev.scan_state.find(queue); at != ctx.dev.scan_state.end()) {
        scan_state_free(at->second.data);
        ctx.dev.scan_state.erase(at);
    }
    if (static_cast<std::uint64_t>(ctx.dev.stream) == queue) ctx.dev.stream = ctx.dev.own_stream;
    check(cuStreamDestroy_(reinterpret_cast<CUstream>(queue)), "eg_stream_free: cuStreamDestroy");
    ctx.dev.made_streams.erase(stream);
}

/** @brief eg_stream_sync: wait for the work queued on @p stream only (0: eagle's own, a handle of eg_stream_new, or a raw
 *  stream adopted by eg_stream_use); a host node of a device graph that failed meanwhile is reported as eg_sync does. */
void device_stream_sync(eg_ctx& ctx, std::uint64_t stream) {
    refuse_stream_call_in_recording(ctx, "eg_stream_sync");
    std::uint64_t queue = ctx.dev.own_stream;
    if ((stream & kMadeStreamTag) != 0) {
        queue = made_stream(ctx, stream, "eg_stream_sync");
    } else if (stream != 0) {
        if (ctx.dev.adopted_streams.count(stream) == 0)
            refuse(EG_BAD_REQUEST, "eg_stream_sync: stream " + std::to_string(stream) +
                   " is neither a handle of eg_stream_new nor a raw stream given to eg_stream_use on this context");
        queue = stream;
    }
    EG_DRIVER(cuStreamSynchronize);
    Current current(ctx.dev);
    check(cuStreamSynchronize_(reinterpret_cast<CUstream>(queue)), "eg_stream_sync: cuStreamSynchronize");
    for (DeviceGraph* g : ctx.dev_graphs)
        if (g->failure) raise_step_failure(*g->failure);
}

/** @brief Refuse BAD_REQUEST naming @p call while a device recording is open: an event is a mark of the eager level. */
void refuse_event_in_recording(const eg_ctx& ctx, const char* call) {
    if (ctx.dev_rec != nullptr)
        refuse(EG_BAD_REQUEST, std::string(call) + ": an event is a mark of the eager level, not a node of a recording; "
               "order inside one is eg_record_wait");
}

/** @brief The CUevent of the door's event handle @p event, refusing one this context's device did not make. */
CUevent driver_event(const eg_ctx& ctx, const void* event, const char* call) {
    const auto at = ctx.dev.events.find(event);
    if (at == ctx.dev.events.end()) refuse(EG_BAD_REQUEST, std::string(call) + ": the event was not made on this context, or it was already freed");
    return reinterpret_cast<CUevent>(at->second);
}

/** @brief eg_event_new: the door's handle (its registry and its refusals) plus a driver event of the context's device. */
void device_event_new(eg_ctx& ctx, eg_event** out) {
    event_new(ctx, out);
    try {
        EG_DRIVER(cuEventCreate);
        Current current(ctx.dev);
        CUevent made = nullptr;
        check(cuEventCreate_(&made, CU_EVENT_DEFAULT), "eg_event_new: cuEventCreate");
        ctx.dev.events[*out] = reinterpret_cast<std::uint64_t>(made);
    } catch (...) {
        event_free(ctx, *out);
        *out = nullptr;
        throw;
    }
}

/** @brief eg_event_free: the driver event first (the driver releases it when the mark it holds is reached), then the handle. */
void device_event_free(eg_ctx& ctx, eg_event* event) {
    const auto at = ctx.dev.events.find(event);
    if (at != ctx.dev.events.end()) {
        EG_DRIVER(cuEventDestroy);
        Current current(ctx.dev);
        cuEventDestroy_(reinterpret_cast<CUevent>(at->second));
        ctx.dev.events.erase(at);
    }
    event_free(ctx, event);
}

/** @brief eg_event_record: mark the stream in scope of the calling thread. */
void device_event_record(eg_ctx& ctx, eg_event* event) {
    refuse_event_in_recording(ctx, "eg_event_record");
    event_record(ctx, event);    // the refusals of a foreign handle or a recording, and the "recorded" fact
    EG_DRIVER(cuEventRecord);
    Current current(ctx.dev);
    check(cuEventRecord_(driver_event(ctx, event, "eg_event_record"), reinterpret_cast<CUstream>(static_cast<std::uint64_t>(ctx.dev.stream))),
          "eg_event_record: cuEventRecord");
}

/** @brief eg_event_wait: the stream in scope of the calling thread waits for the mark; the host thread does not. */
void device_event_wait(eg_ctx& ctx, eg_event* event) {
    refuse_event_in_recording(ctx, "eg_event_wait");
    event_wait(ctx, event);    // the refusals of a foreign handle, a recording and an event never recorded
    EG_DRIVER(cuStreamWaitEvent);
    Current current(ctx.dev);
    check(cuStreamWaitEvent_(reinterpret_cast<CUstream>(static_cast<std::uint64_t>(ctx.dev.stream)), driver_event(ctx, event, "eg_event_wait"), 0),
          "eg_event_wait: cuStreamWaitEvent");
}

/** @brief eg_event_elapsed: wait for both marks, then the driver's milliseconds from @p start to @p stop. */
void device_event_elapsed(eg_ctx& ctx, const eg_event* start, const eg_event* stop, double* ms) {
    refuse_event_in_recording(ctx, "eg_event_elapsed");
    event_elapsed(ctx, start, stop, ms);    // the refusals; the host's figure is replaced below
    EG_DRIVER(cuEventSynchronize);
    EG_DRIVER(cuEventElapsedTime);
    Current current(ctx.dev);
    const CUevent first = driver_event(ctx, start, "eg_event_elapsed");
    const CUevent second = driver_event(ctx, stop, "eg_event_elapsed");
    check(cuEventSynchronize_(first), "eg_event_elapsed: cuEventSynchronize");
    check(cuEventSynchronize_(second), "eg_event_elapsed: cuEventSynchronize");
    float elapsed = 0.0f;
    check(cuEventElapsedTime_(&elapsed, first, second), "eg_event_elapsed: cuEventElapsedTime");
    *ms = static_cast<double>(elapsed);
}

/** @brief eg_sync: wait until every stream of the context is idle: eagle's own, every queue of eg_stream_new and the caller's
 *  raw stream when it is in scope (a host-only context has nothing queued). */
void device_sync(eg_ctx& ctx) {
    if (ctx.dev.ordinal < 0) return;
    EG_DRIVER(cuStreamSynchronize);
    Current current(ctx.dev);
    check(cuStreamSynchronize_(reinterpret_cast<CUstream>(ctx.dev.own_stream)), "eg_sync: cuStreamSynchronize");
    for (const auto& made : ctx.dev.made_streams)
        check(cuStreamSynchronize_(reinterpret_cast<CUstream>(made.second)), "eg_sync: cuStreamSynchronize");
    const std::uint64_t scope = ctx.dev.stream;
    if (ctx.dev.adopted_streams.count(scope) != 0)    // a caller's stream is waited for while it is in scope; the caller owns the others
        check(cuStreamSynchronize_(reinterpret_cast<CUstream>(scope)), "eg_sync: cuStreamSynchronize");
    for (DeviceGraph* g : ctx.dev_graphs)    // a host node of a replay that has run now reports its failure
        if (g->failure) raise_step_failure(*g->failure);
}

#undef EG_DRIVER

#endif  // EAGLE_CPU_ONLY

}  // namespace door
}  // namespace eagle
