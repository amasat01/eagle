// SPDX-License-Identifier: Apache-2.0

// The device group: a context's device and its own stream, a unit's device
// module and function, and one kernel launch per partition of a DEVICE plan,
// queued on the context's stream. Every driver call goes through eagle's
// driver entry (exec/DeviceKernel.h), with the context's primary context
// current for the call only. The host-only build refuses EG_CTX_DEVICE and
// structure DEVICE.
#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/door/host/Ops.h"
#include "eagle/door/host/Segment.h"

#ifndef EAGLE_CPU_ONLY
#include "eagle/door/device/Loop.h"
#include "eagle/exec/DeviceKernel.h"
#include "eagle/policy/Launch.h"
#include "eagle/policy/Residency.h"

#include <algorithm>
#include <chrono>
#include <cstring>

#include <dlfcn.h>
#endif

namespace eagle {
namespace door {

#ifdef EAGLE_CPU_ONLY

void device_open(eg_ctx&, std::int32_t device) {
    refuse(EG_NOT_WRITTEN, "eg_ctx_settings EG_CTX_DEVICE (device " + std::to_string(device) +
           "): the device group is not built");
}
void device_close(eg_ctx&) noexcept {}
void device_unit_load(eg_unit_s&, const eg_unit_meta&, void*) {}
void device_unit_free(eg_unit_s&) noexcept {}
void device_plan_check(const eg_ctx&, const eg_unit_s&) {
    refuse(EG_NOT_WRITTEN, "structure DEVICE: the device group is not built");
}
void device_launch(eg_plan_s&) {}
void device_sync(eg_ctx&) {}
void device_plan_cells(eg_plan_s&) {}
void device_plan_free(eg_plan_s&) noexcept {}
void* device_alloc(eg_ctx&, std::size_t) { return nullptr; }
void device_free(eg_ctx&, void*) noexcept {}
void device_cell_work(eg_plan_s&) {}
void device_read(eg_ctx&, const void*, void*, std::size_t) {}
void device_write(eg_ctx&, void*, const void*, std::size_t) {}
void device_copy_bytes(eg_ctx&, void*, const void*, std::size_t) {}
void device_zero(eg_ctx&, void*, std::size_t) {}
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
void device_copy(eg_ctx&, eg_plane_s*, eg_plane_s*) {}
void device_runner_new(eg_ctx&, const eg_runner_desc*, eg_runner**) {
    refuse(EG_NOT_WRITTEN, "structure DEVICE: the device group is not built");
}
void device_run(eg_ctx&, eg_runner_s*, eg_run_report*) {}
DeviceRun::~DeviceRun() = default;
void device_record_begin(eg_ctx&) {}
void device_record_launch(eg_ctx&, eg_plan_s&) {}
void device_record_end(eg_ctx&, eg_graph**) {}
bool device_graph(const eg_ctx&, const eg_graph*) { return false; }
void device_graph_launch(eg_ctx&, eg_graph*) {}
void device_graph_free(eg_ctx&, eg_graph*) {}
void device_graphs_drop(eg_ctx&) noexcept {}
void device_stream_use(eg_ctx&, std::uint64_t) {
    refuse(EG_BAD_REQUEST, "eg_stream_use on a host-only context (made without EG_CTX_DEVICE): it has no stream");
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

/** @brief The runner's reorder scratch, grown to stage @p planes (a recorded run moves eagle's own planes too). */
void* reorder_scratch(DeviceRun& r, std::int64_t n, const std::vector<DevicePlane>& planes) {
    const std::size_t need = reorder_scratch_bytes(n, planes);
    if (need > r.rscratch_bytes) {
        cells_free(r.rscratch);
        r.rscratch = cells_new(static_cast<unsigned>((need + sizeof(unsigned) - 1) / sizeof(unsigned)));
        r.rscratch_bytes = need;
    }
    return reinterpret_cast<void*>(r.rscratch);
}

}  // namespace

/** @brief A capture in progress: the stream it is on, the plans it launched (one reference each) with their
 *  argument bytes at that launch, and the device frees asked for meanwhile, run when it ends. */
struct DeviceRecording {
    DeviceCapture* capture = nullptr;
    std::uint64_t stream = 0;
    std::vector<eg_plan_s*> plans;
    std::vector<std::vector<unsigned char>> bound;
    std::vector<std::function<void()>> later;
};

/** @brief A recorded device graph: the instantiated capture and the plans whose bindings it holds by value. */
struct DeviceGraph {
    DeviceLoop* loop = nullptr;
    std::vector<eg_plan_s*> plans;
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

}  // namespace

/** @brief EG_CTX_DEVICE: retain the device's primary context, make the context's stream, read the launch facts. */
void device_open(eg_ctx& ctx, std::int32_t device) {
    EG_DRIVER(cuInit);
    EG_DRIVER(cuDeviceGetCount);
    EG_DRIVER(cuDeviceGet);
    EG_DRIVER(cuDevicePrimaryCtxRetain);
    EG_DRIVER(cuStreamCreate);
    check(cuInit_(0), "cuInit");
    int count = 0;
    check(cuDeviceGetCount_(&count), "cuDeviceGetCount");
    if (device >= count)
        refuse(EG_BAD_REQUEST, "eg_ctx_settings.device " + std::to_string(device) + ": the driver sees " +
               std::to_string(count) + " device(s)");
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
    d.stream = d.own_stream = reinterpret_cast<std::uint64_t>(stream);
    d.sms = attribute(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "SMs");
    d.regs_per_sm = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, "registers per SM");
    d.smem_per_sm = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "shared memory per SM");
    d.threads_per_sm = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, "threads per SM");
    d.fp64_ratio = std::max(1, attribute(dev, CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO, "FP64 ratio"));
    d.cc_major = attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "compute capability");
    d.cc_minor = attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "compute capability");
}

/** @brief Destroy the context's stream (queued work still finishes) and release the primary context. */
void device_close(eg_ctx& ctx) noexcept {
    DeviceState& d = ctx.dev;
    if (d.context == 0) return;
    try {
        if (d.own_stream != 0) {    // never the caller's stream
            EG_DRIVER(cuStreamDestroy);
            Current current(d);
            cuStreamDestroy_(reinterpret_cast<CUstream>(d.own_stream));
        }
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

/** @brief Queue on @p stream the fold of cell @p i of @p plan: its rows (and the compensation rows of a sum) into @p out, by the canonical tree. */
void fold_cell(const eg_plan_s& plan, std::size_t i, const DLTensor& rows, const eg_plane_s* comp, std::int32_t op, void* out, std::uint64_t stream) {
    cell_fold(stream, rows.data, comp != nullptr ? comp->tensor->dl_tensor.data : nullptr, rows.shape[0], rows.dtype.bits,
              op == EG_REDUCE_SUM ? 0 : op == EG_REDUCE_MAX ? 1 : 2, out, plan.work[2 * i]->tensor->dl_tensor.data,
              plan.work[2 * i + 1]->tensor->dl_tensor.data);
}

/** @brief Queue one launch per partition of @p plan on @p stream: grid from the partition's count and the
 *  meta's block (else the resolved one), the bound views and uniforms by value, then the triple; then each
 *  companion's fold function over its target and each cell's rows folded into the cell the caller bound
 *  (as the host's ``fold_plan``). */
void launch_on(const eg_plan_s& plan, std::uint64_t stream) {
    const DeviceState& d = plan.ctx->dev;
    const eg_unit_s& u = *plan.unit;
    const auto s = reinterpret_cast<CUstream>(stream);
    std::size_t group = 0;
    for (const DispatchFacts& f : u.dispatches) {    // an AUTO group's counters eagle made: zeroed in the stream at every launch (nothing reads them)
        if (f.strategy != EG_DISPATCH_AUTO) continue;
        const eg_plane_s* mix = plan.mix[group++];
        if (plan.bindings[static_cast<std::size_t>(f.mix)].plane == nullptr)
            zero_bytes(stream, static_cast<char*>(mix->tensor->dl_tensor.data) + mix->tensor->dl_tensor.byte_offset, static_cast<std::size_t>(f.k) * sizeof(std::uint32_t));
    }
    for (const exec::Partition& p : plan.parts)
        exec::DeviceKernel::run(reinterpret_cast<CUfunction>(u.function), plan.params, p, s,
                                u.block > 0 ? static_cast<unsigned>(u.block) : resolved_block(d, u, p.count));
    for (const CompanionFacts& c : u.companions) {
        const std::int64_t size = elements(plan.bindings[c.target].plane->tensor->dl_tensor);
        exec::DeviceKernel::run(reinterpret_cast<CUfunction>(c.device_fold), plan.params, exec::Partition{ 0, size, size }, s);
    }
    if (plan.n_samples == 0) return;
    for (std::size_t i = 0; i < u.cells.size(); ++i) {
        const DLTensor& rows = plan.rows[i]->tensor->dl_tensor;
        const DLTensor& out = plan.bindings[u.cells[i].slot].plane->tensor->dl_tensor;
        fold_cell(plan, i, rows, plan.comps[i], u.cells[i].op, static_cast<char*>(out.data) + out.byte_offset, stream);
    }
}

/** @brief Refuse a DEVICE plan of @p unit on @p ctx: a host-only context, a device feature this
 *  build does not run (each by name), a unit with no device function. */
void device_plan_check(const eg_ctx& ctx, const eg_unit_s& unit) {
    if (ctx.dev.ordinal < 0)
        refuse(EG_BAD_REQUEST, "structure DEVICE on a host-only context (made without EG_CTX_DEVICE)");
    const std::string who = "unit '" + unit.name + "' on DEVICE: ";
    if (unit.seg.present && unit.seg.dynamic && unit.seg.mask_slot >= 0)
        refuse(EG_NOT_WRITTEN, who + "the SEGMENTED dispatch group '" + unit.seg.name + "' has a dynamic selector and a TERMINATED mask, which is not written on the device (no mask: every sample is live)");
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

/** @brief A split unit on the device: the stages in order on the context's stream, each with its argument list (the source slot or team
 *  plane each stage slot joined at load). A team stage is `grid` blocks of `threads * warps` threads with the tail (0, grid, n); a plain
 *  stage launches like the unit would. Planes with a promised fill are refilled first, on the stream. */
void launch_split(const eg_plan_s& plan) {
    const DeviceState& d = plan.ctx->dev;
    const eg_unit_s& u = *plan.unit;
    const auto s = reinterpret_cast<CUstream>(d.stream);
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
        std::uint32_t n32 = static_cast<std::uint32_t>(plan.n_samples);
        for (std::size_t j = 0; j < n_slots; ++j) {
            if (st.row_of[j] >= 0 || st.comp_of[j] >= 0) {    // the rows this stage writes into a cell: eagle's
                const std::size_t c = static_cast<std::size_t>(st.row_of[j] >= 0 ? st.row_of[j] : st.comp_of[j]);
                const DLTensor& t = (st.row_of[j] >= 0 ? plan.rows[c] : plan.comps[c])->tensor->dl_tensor;
                plugin::ScalarHandle& h = handles[j];
                h.data = static_cast<char*>(t.data) + t.byte_offset;
                h.samples = static_cast<std::uint64_t>(plan.n_samples);
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
            } else if (st.plane_of[j] >= 0) {
                const DLTensor& t = plan.team[static_cast<std::size_t>(st.plane_of[j])]->tensor->dl_tensor;
                const int role = su.slots[j].role;
                plugin::ScalarHandle& h = handles[j];
                h.data = static_cast<char*>(t.data) + t.byte_offset;
                h.samples = static_cast<std::uint64_t>(role == EG_ROLE_LOOKUP ? elements(t)
                                                       : role == EG_ROLE_WIDE_IN || role == EG_ROLE_WIDE_OUT ? t.shape[t.ndim - 1] : plan.n_samples);
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
            fold_cell(plan, c, plan.rows[c]->tensor->dl_tensor, plan.comps[c], sc.op, static_cast<char*>(o->tensor->dl_tensor.data) + o->tensor->dl_tensor.byte_offset, d.stream);
        }
    }
}

namespace seg = host::seg;

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
    if (!g.dynamic) {    // the classes of every sample, in sample order (the old runner launches it at block 256)
        std::vector<seg::Owned> none;
        std::vector<void*> p = seg::params_of(plan, K, *g.classify, none, &n32);
        exec::DeviceKernel::run(reinterpret_cast<CUfunction>(g.classify->function), p, exec::Partition{ 0, n, n }, cs, 256);
    }
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
        if (!S.sorted || static_cast<double>(S.stale_now) > host::kStaleMax * static_cast<double>(std::max<std::int64_t>(n, 1))) {
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
    std::vector<unsigned> off(K + 1);
    cells_read(stream, reinterpret_cast<std::uint64_t>(off32), off.data(), K + 1);
    std::vector<std::int64_t> off64(off.begin(), off.end());
    device_write(ctx, static_cast<char*>(plan.segdev[0]->tensor->dl_tensor.data) + plan.segdev[0]->tensor->dl_tensor.byte_offset, off64.data(), off64.size() * sizeof(std::int64_t));
    for (unsigned j = 0; j < K; ++j) {    // branch j over its run; the unit reads its own offset
        const std::int64_t run = static_cast<std::int64_t>(off[j + 1]) - static_cast<std::int64_t>(off[j]);
        if (run <= 0) continue;
        const eg_unit_s& u = *g.branches[j];
        std::vector<void*> p = seg::params_of(plan, j, u, owned, &n32);
        exec::DeviceKernel::run(reinterpret_cast<CUfunction>(u.function), p, exec::Partition{ 0, run, n }, cs, resolved_block(d, u, run));
    }
    if (g.dynamic) {    // the samples whose branch moved (stale), compacted into the sweep's active map; the fused unit completes them
        compact_keep_step(stream, S.dev_bufs[kFixed + static_cast<std::size_t>(owned.size()) - 1], n, static_cast<std::int32_t*>(plan.segdev[3]->tensor->dl_tensor.data),
                          static_cast<unsigned*>(plan.segdev[4]->tensor->dl_tensor.data), S.dev_bufs[8]);
        unsigned stale = 0;
        device_read(ctx, plan.segdev[4]->tensor->dl_tensor.data, &stale, sizeof(stale));
        S.stale_now = stale;
        if (stale > 0) {
            const eg_unit_s& u = *g.sweep;
            std::vector<void*> p = seg::params_of(plan, K, u, owned, &n32);
            exec::DeviceKernel::run(reinterpret_cast<CUfunction>(u.function), p, exec::Partition{ 0, static_cast<std::int64_t>(stale), n }, cs, resolved_block(d, u, stale));
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
    if (plan.unit->index_count_slot >= 0) {    // an index set: its live count is checked against the entries held (a device read), the kernel maps the lanes itself
        const DLTensor& c = plan.bindings[static_cast<std::size_t>(plan.unit->index_count_slot)].plane->tensor->dl_tensor;
        unsigned live = 0;
        device_read(*plan.ctx, static_cast<char*>(c.data) + c.byte_offset, &live, sizeof(live));
        check_index_live(plan, live);
    }
    launch_on(plan, plan.ctx->dev.stream);
}

/** @brief A DEVICE plan's eagle-made finished counter: one zeroed device cell. */
void device_plan_cells(eg_plan_s& plan) {
    Current current(plan.ctx->dev);
    plan.device_counter = cells_new(1);
    if (plan.unit->finish.steps == EG_STEPS_AUTO) plan.device_steps = cells_new(2);    // the int64 steps word, 0
}

/** @brief Free what device_plan_cells made. */
void device_plan_free(eg_plan_s& plan) noexcept {
    if (plan.device_counter == 0) return;
    eg_ctx* ctx = plan.ctx;
    const std::uint64_t counter = plan.device_counter, steps = plan.device_steps;
    plan.device_counter = plan.device_steps = 0;
    now_or_after_capture(*ctx, [ctx, counter, steps] {
        Current current(ctx->dev);
        cells_free(counter);
        cells_free(steps);
    });
}

/** @brief A zeroed device allocation of @p bytes on the context's device. */
void* device_alloc(eg_ctx& ctx, std::size_t bytes) {
    Current current(ctx.dev);
    return reinterpret_cast<void*>(cells_new(static_cast<unsigned>((bytes + sizeof(unsigned) - 1) / sizeof(unsigned))));
}

/** @brief Free a device_alloc allocation (a forked child makes no device call). */
void device_free(eg_ctx& ctx, void* data) noexcept {
    if (data == nullptr || forked(ctx)) return;
    eg_ctx* c = &ctx;
    now_or_after_capture(ctx, [c, data] {
        Current current(c->dev);
        cells_free(reinterpret_cast<std::uint64_t>(data));
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
    const DeviceState& d = plan.ctx->dev;
    const eg_unit_s& u = *plan.unit;
    Current current(d);
    exec::DeviceKernel::run(reinterpret_cast<CUfunction>(u.function), plan.params, exec::Partition{ 0, count, of },
                            reinterpret_cast<CUstream>(d.stream), u.block > 0 ? static_cast<unsigned>(u.block) : resolved_block(d, u, of));
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
 *  stream (the order the host runs); returns with the result in place. */
void device_reduce(eg_ctx& ctx, const eg_reduce_desc& d) {
    const host::ReduceOp r = host::reduce_check(ctx, d);
    const auto at = [](const eg_plane_s* p) -> void* {
        return p == nullptr ? nullptr : static_cast<char*>(p->tensor->dl_tensor.data) + p->tensor->dl_tensor.byte_offset;
    };
    Current current(ctx.dev);
    reduce_step(ctx.dev.stream, r.code, r.op == EG_REDUCE_ALL ? 3 : r.op, at(r.in), r.n, r.cols, at(r.out), at(r.comp), r.pair);
}

/** @brief eg_copy over device planes: checked as the host's, then a device-to-device copy on the context's stream
 *  (planes of one range of memory copy nothing; partly overlapping ones are refused). */
void device_copy(eg_ctx& ctx, eg_plane_s* src, eg_plane_s* dst) {
    host::copy_check(ctx, src, dst);
    const DLTensor& s = src->tensor->dl_tensor;
    const DLTensor& t = dst->tensor->dl_tensor;
    const char *ps = static_cast<const char*>(s.data) + s.byte_offset, *pd = static_cast<const char*>(t.data) + t.byte_offset;
    const std::size_t bytes = static_cast<std::size_t>(elements(s)) * (s.dtype.bits / 8);
    if (bytes == 0 || ps == pd) return;
    if (ps < pd + bytes && pd < ps + bytes) refuse(EG_BAD_REQUEST, "eg_copy: the device planes overlap");
    Current current(ctx.dev);
    copy_step(ctx.dev.stream, const_cast<char*>(pd), ps, bytes);
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
        for (std::uint64_t cell : all) cells_free(cell);
    });
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
    cells_write(dev.stream, plan.device_steps, word, 2);
    if (mode != EG_MODE_PERSIST) return;
    runner.dev.fast = cells_new(2);
    const bool fp64 = u.scalar_type != EG_SCALAR_FLOAT32;
    const double ops = static_cast<double>(u.finish.ops > 0 ? u.finish.ops : kAutoStepOps);
    const policy::LaunchDevice l = launch_device(dev);
    runner.dev.chunk = static_cast<unsigned>(policy::persistChunk(l, fp64, ops, 1e9));
    const int threads = policy::persistThreads(l, fp64, u.persist_regs, n, policy::PERSIST_FIRST_FILL,
                                               static_cast<int>(runner.dev.chunk), ops);
    int blocks = 1;
    while (threads > 0 && threads / blocks > 512) blocks *= 2;
    persist_geometry(dev, u, threads > 0 ? blocks : 1, threads > 0 ? threads / blocks / 32 * 32 : 256, runner.dev);
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
        launch_on(first, stream);
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
    write_report(report, out);
}

}  // namespace

/** @brief eg_runner_new over DEVICE plans: the band loop (``_until_done.Runner``) of a fixed-step finishing unit,
 *  with an active set when its units read one (launches over the map's live prefix, compaction every ``every``
 *  steps when a sample finished); what else a device runner would run is refused by name. */
void device_runner_new(eg_ctx& ctx, const eg_runner_desc* desc, eg_runner** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_runner_new: out is NULL");
    *out = nullptr;
    const eg_runner_desc d = read_input(desc, "eg_runner_desc");
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
    check_runner(ctx, d, true);
    for (std::int32_t i = 0; i < d.n_plans; ++i)
        if (d.plans[i]->structure != EG_STRUCTURE_DEVICE)
            refuse(EG_BAD_REQUEST, "plans[" + std::to_string(i) + "] is a host plan in a runner over DEVICE plans: "
                   "one runner runs on one side");
    const std::int64_t k = d.plans[0]->unit->finish.steps;
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
        runner->dev.ctx = &ctx;
        Current current(ctx.dev);
        runner->dev.cells = cells_new(6);    // [remaining, ran, total, live count, finished at the last compaction, compactions]
        const std::int64_t n = d.plans[0]->n_samples;
        const unsigned total[2] = { static_cast<unsigned>(n), static_cast<unsigned>(n) };
        cells_write(ctx.dev.stream, runner->dev.cells + 2 * sizeof(unsigned), total, 2);    // every sample live
        if (active) {
            ActiveState& a = runner->act;
            a.n = n;
            a.every_steps = every;
            a.every_launches = every / k;
            a.reorder = reorder;
            a.theta = d.reorder_theta;
            runner->active = true;
            runner->dev.map = cells_new(static_cast<unsigned>(std::max<std::int64_t>(n, 1)));
            runner->dev.scratch = cells_new(static_cast<unsigned>((compact_scratch_bytes(n) + sizeof(unsigned) - 1) / sizeof(unsigned)));
            std::vector<unsigned> identity(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i) identity[static_cast<std::size_t>(i)] = static_cast<unsigned>(i);
            cells_write(ctx.dev.stream, runner->dev.map, identity.data(), static_cast<unsigned>(n));
            if (reorder) {
                DeviceRun& r = runner->dev;
                r.perm = cells_new(static_cast<unsigned>(n));
                r.inv = cells_new(static_cast<unsigned>(n));
                r.rcells = cells_new(5);    // [live32, span, fire, reorders, zero]
                cells_write(ctx.dev.stream, r.perm, identity.data(), static_cast<unsigned>(n));
                cells_write(ctx.dev.stream, r.inv, identity.data(), static_cast<unsigned>(n));
                cells_write(ctx.dev.stream, r.rcells + sizeof(unsigned), total, 1);    // span = n
                reorder_scratch(r, n, reorder_table(*runner));
            }
        }
        if ((d.flags & EG_RUN_RECORD) || d.events == EG_EVENTS_REFINE) record_new(ctx, *runner, d);    // the host's record and refine checks; the run is host/Reverse.cpp's on the device
        ctx.runners.insert(runner);
    } catch (...) {
        record_free(ctx, *runner);
        delete runner;
        throw;
    }
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

/** @brief eg_run of a DEVICE runner (``_until_done.Runner._run``, band): re-sync the counter to the mask, capture
 *  the WHILE loop at the first run (again when a plan's bindings changed), launch it, read the iterations
 *  and the counter back, fill @p report as the old device path does. */
void device_run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report) {
    check_handle(ctx.runners, runner, "runner");
    if (ctx.rec != nullptr || ctx.dev_rec != nullptr)
        refuse(EG_NOT_WRITTEN, "recording a DEVICE run (eg_run while recording) is not written in this build of "
               "the door");
    check_report(report);
    for (const eg_plan_s* p : runner->plans) check_bound(*p);
    if (runner->record != nullptr) {    // EG_RUN_RECORD: the recording forward (host/Reverse.cpp) over device planes
        record_forward(ctx, *runner, report);
        return;
    }
    if (runner->dev.mode != EG_MODE_BAND) return fast_run(ctx, runner, report);
    eg_plan_s& first = *runner->plans[0];
    const FinishFacts& f = first.unit->finish;
    const std::int64_t n = first.n_samples, k = f.steps;
    const std::int64_t every = runner->active ? runner->act.every_steps : k;    // steps per loop iteration
    DeviceRun& r = runner->dev;
    const std::uint64_t stream = ctx.dev.stream;
    Current current(ctx.dev);
    std::vector<unsigned char> bound;
    for (const eg_plan_s* p : runner->plans) {
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
    mask_count(stream, static_cast<const char*>(t.data) + t.byte_offset, n, t.dtype.bits / 8u, counter);
    if (r.loop == nullptr) {
        const auto t0 = std::chrono::steady_clock::now();
        auto* cells = reinterpret_cast<unsigned*>(r.cells);
        const auto cap = static_cast<unsigned>(loop_cap(runner->max_steps, every));
        const auto body = [runner](std::uint64_t s) {
            for (std::int64_t l = 0; l < (runner->active ? runner->act.every_launches : 1); ++l)
                for (eg_plan_s* p : runner->plans) launch_on(*p, s);
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
    rec->capture = capture_begin(rec->stream);
    ctx.dev_rec = rec.release();
}

/** @brief eg_launch of a DEVICE plan while recording: its launches captured on the recording's stream. */
void device_record_launch(eg_ctx& ctx, eg_plan_s& plan) {
    const eg_unit_s& u = *plan.unit;
    const std::string who = "recording a DEVICE launch of unit '" + u.name + "': ";
    if (!u.stages.empty()) refuse(EG_NOT_WRITTEN, who + "a split unit is not recorded");
    if (u.seg.present) refuse(EG_NOT_WRITTEN, who + "a SEGMENTED group reads its classes back to the host at each launch and is not recorded");
    if (u.index_count_slot >= 0)
        refuse(EG_NOT_WRITTEN, who + "an index set (its live count is read on the host at each launch) is not written in "
               "this build of the door");
    if (!u.companions.empty() || !u.cells.empty())
        refuse(EG_NOT_WRITTEN, who + "companion and cell folds are not written in this build of the door");
    DeviceRecording& rec = *ctx.dev_rec;
    rec.bound.push_back(binding_bytes(plan));
    rec.plans.push_back(&plan);
    ++plan.refs;
    Current current(ctx.dev);
    launch_on(plan, rec.stream);
}

/** @brief eg_record_end on a device context: end the capture and instantiate it (FAULT when the capture failed). */
void device_record_end(eg_ctx& ctx, eg_graph** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_record_end: out is NULL");
    *out = nullptr;
    const std::unique_ptr<DeviceRecording> rec(ctx.dev_rec);
    ctx.dev_rec = nullptr;
    Current current(ctx.dev);
    DeviceLoop* loop = nullptr;
    std::string failed;
    try {
        loop = capture_end(rec->capture);
    } catch (const std::exception& e) {
        failed = e.what();
    } catch (...) {
        failed = "unknown error";
    }
    rec->capture = nullptr;
    run_later(ctx, *rec);
    if (loop == nullptr) {
        for (eg_plan_s* p : rec->plans) plan_release(ctx, p);
        refuse(EG_FAULT, "eg_record_end: the device capture failed: " + failed);
    }
    auto graph = std::make_unique<DeviceGraph>();
    graph->loop = loop;
    graph->plans = std::move(rec->plans);
    graph->bound = std::move(rec->bound);
    ctx.dev_graphs.insert(graph.get());
    *out = reinterpret_cast<eg_graph*>(graph.release());
}

/** @brief Whether @p graph is a device graph of @p ctx. */
bool device_graph(const eg_ctx& ctx, const eg_graph* graph) {
    return ctx.dev_graphs.count(reinterpret_cast<DeviceGraph*>(const_cast<eg_graph*>(graph))) != 0;
}

/** @brief eg_graph_launch of a device graph: replay it once on the stream in scope (``GraphPipeline.launch``); a plan
 *  rebound since the recording is refused (the graph holds the bindings it captured). */
void device_graph_launch(eg_ctx& ctx, eg_graph* handle) {
    auto* graph = reinterpret_cast<DeviceGraph*>(handle);
    device_quiet(ctx, "eg_graph_launch", EG_NOT_WRITTEN, "a graph inside a device graph is not written in this build of the door");
    for (std::size_t i = 0; i < graph->plans.size(); ++i)
        if (binding_bytes(*graph->plans[i]) != graph->bound[i])
            refuse(EG_BAD_REQUEST, "eg_graph_launch: a plan of unit '" + graph->plans[i]->unit->name + "' was rebound since "
                   "the recording; a device graph replays the bindings it captured: record it again");
    Current current(ctx.dev);
    loop_launch(graph->loop, ctx.dev.stream);
}

/** @brief eg_graph_free of a device graph: drop its plan references and its instantiated capture. */
void device_graph_free(eg_ctx& ctx, eg_graph* handle) {
    auto* graph = reinterpret_cast<DeviceGraph*>(handle);
    ctx.dev_graphs.erase(graph);
    const std::unique_ptr<DeviceGraph> owned(graph);
    for (eg_plan_s* p : graph->plans) plan_release(ctx, p);
    eg_ctx* c = &ctx;
    DeviceLoop* loop = graph->loop;
    now_or_after_capture(ctx, [c, loop] {
        Current current(c->dev);
        loop_free(loop);
    });
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
                capture_abort(rec->capture);
                run_later(ctx, *rec);
            } catch (...) {
            }
        }
    }
    for (DeviceGraph* g : ctx.dev_graphs) {
        if (!child) {
            try {
                Current current(ctx.dev);
                loop_free(g->loop);
            } catch (...) {
            }
        }
        delete g;
    }
    ctx.dev_graphs.clear();
}

/** @brief eg_stream_use: make @p stream (a stream of the context's device; 0: eagle's own) the stream in scope; the
 *  launches, runs and replays after it are ordered on it (the old path's ``stream_create`` over an external stream). */
void device_stream_use(eg_ctx& ctx, std::uint64_t stream) {
    if (ctx.dev.ordinal < 0)
        refuse(EG_BAD_REQUEST, "eg_stream_use on a host-only context (made without EG_CTX_DEVICE): it has no stream");
    if (ctx.dev_rec != nullptr)
        refuse(EG_BAD_REQUEST, "eg_stream_use while recording: the capture is on the stream in scope; call it before "
               "eg_record_begin or after eg_record_end");
    if (stream == 0) {
        ctx.dev.stream = ctx.dev.own_stream;
        return;
    }
    EG_DRIVER(cuStreamGetCtx);
    Current current(ctx.dev);
    CUcontext owner = nullptr;
    check(cuStreamGetCtx_(reinterpret_cast<CUstream>(stream), &owner), "eg_stream_use: cuStreamGetCtx");
    if (owner != reinterpret_cast<CUcontext>(ctx.dev.context))
        refuse(EG_BAD_REQUEST, "eg_stream_use: the stream belongs to another context than device " +
               std::to_string(ctx.dev.ordinal) + "'s primary context");
    ctx.dev.stream = stream;
}

/** @brief eg_sync: wait until the context's stream is idle (a host-only context has nothing queued). */
void device_sync(eg_ctx& ctx) {
    if (ctx.dev.ordinal < 0) return;
    EG_DRIVER(cuStreamSynchronize);
    Current current(ctx.dev);
    check(cuStreamSynchronize_(reinterpret_cast<CUstream>(ctx.dev.stream)), "eg_sync: cuStreamSynchronize");
}

#undef EG_DRIVER

#endif  // EAGLE_CPU_ONLY

}  // namespace door
}  // namespace eagle
