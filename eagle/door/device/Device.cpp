// SPDX-License-Identifier: Apache-2.0

// The device group: a context's device and its own stream, a unit's device
// module and function, and one kernel launch per partition of a DEVICE plan,
// queued on the context's stream. Every driver call goes through eagle's
// driver entry (exec/DeviceKernel.h), with the context's primary context
// current for the call only. The host-only build refuses EG_CTX_DEVICE and
// structure DEVICE.
#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"

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

/** @brief The planes a reorder of @p runner moves, as device addresses (the mask first). */
std::vector<DevicePlane> reorder_table(const eg_runner_s& runner) {
    std::vector<DevicePlane> out;
    for (const Owned& o : owned_planes(runner)) out.push_back({ o.data, static_cast<unsigned>(o.bytes) });
    return out;
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
    check(cuModuleLoadData_(&module, copy.c_str()), who + ": loading the device image (cuModuleLoadData)");
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

/** @brief Queue one launch per partition of @p plan on @p stream: grid from the partition's count and the
 *  meta's block (else the resolved one), the bound views and uniforms by value, then the triple; then each
 *  companion's fold function over its target and each cell's rows folded into the cell the caller bound
 *  (as the host's ``fold_plan``). */
void launch_on(const eg_plan_s& plan, std::uint64_t stream) {
    const DeviceState& d = plan.ctx->dev;
    const eg_unit_s& u = *plan.unit;
    const auto s = reinterpret_cast<CUstream>(stream);
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
        cell_fold(stream, rows.data, rows.shape[0], rows.dtype.bits, u.cells[i].op == EG_REDUCE_MAX,
                  static_cast<char*>(out.data) + out.byte_offset, plan.work[2 * i]->tensor->dl_tensor.data,
                  plan.work[2 * i + 1]->tensor->dl_tensor.data);
    }
}

/** @brief Refuse a DEVICE plan of @p unit on @p ctx: a host-only context, a device feature this
 *  build does not run (each by name), a unit with no device function. */
void device_plan_check(const eg_ctx& ctx, const eg_unit_s& unit) {
    if (ctx.dev.ordinal < 0)
        refuse(EG_BAD_REQUEST, "structure DEVICE on a host-only context (made without EG_CTX_DEVICE)");
    const std::string who = "unit '" + unit.name + "' on DEVICE: ";
    if (unit.function == 0) refuse(EG_NOT_WRITTEN, who + "the unit has no device image");
    for (const CompanionFacts& c : unit.companions)
        if (c.device_fold == 0)
            refuse(EG_NOT_WRITTEN, who + "the companion '" + unit.slots[c.slot].name + "' has no device fold function");
}

/** @brief Queue the plan's launches (launch_on) on the context's stream. */
void device_launch(eg_plan_s& plan) {
    Current current(plan.ctx->dev);
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
                                 : u.scalar_type == EG_SCALAR_BANDED        ? "a BANDED scalar type"
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
        if (d.flags & EG_RUN_REORDER)
            refuse(EG_NOT_WRITTEN, "EG_RUN_REORDER (reordering the planes by the active set) with automatic steps on a "
                   "device is not written in this build of the door");
        auto* runner = new eg_runner_s;
        try {
            runner->plans.assign(d.plans, d.plans + d.n_plans);
            runner->max_steps = d.max_steps;
            runner->dev.ctx = &ctx;
            Current current(ctx.dev);
            fast_runner_setup(ctx, d, *runner);
            ctx.runners.insert(runner);
        } catch (...) {
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
                r.rscratch = cells_new(static_cast<unsigned>((reorder_scratch_bytes(n, reorder_table(*runner)) + sizeof(unsigned) - 1) / sizeof(unsigned)));
            }
        }
        ctx.runners.insert(runner);
    } catch (...) {
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
            gates.push_back({ [&t, n, cells, &r, counter, &a](std::uint64_t s) {
                                  auto* rc = reinterpret_cast<unsigned*>(r.rcells);
                                  compact_step(s, static_cast<const char*>(t.data) + t.byte_offset, n,
                                               reinterpret_cast<std::int32_t*>(r.map), cells + 3, reinterpret_cast<void*>(r.scratch),
                                               a.reorder ? rc : nullptr, a.reorder ? rc + 1 : nullptr,
                                               a.reorder ? rc + 2 : nullptr, static_cast<float>(a.theta));
                                  compact_mark(s, counter, cells + 4, cells + 5);
                              },
                              counter, cells + 4 });
            if (a.reorder)
                gates.push_back({ [runner, n, cells, &r, &t](std::uint64_t s) {
                                      auto* rc = reinterpret_cast<unsigned*>(r.rcells);
                                      reorder_step(s, reorder_table(*runner), static_cast<const char*>(t.data) + t.byte_offset,
                                                   reinterpret_cast<std::int32_t*>(r.perm), reinterpret_cast<std::int32_t*>(r.inv),
                                                   reinterpret_cast<std::int32_t*>(r.map), cells + 3, rc + 1, rc + 2,
                                                   reinterpret_cast<void*>(r.rscratch), n, rc + 3);
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
            auto* rc = reinterpret_cast<unsigned*>(r.rcells);
            restore_step(stream, reorder_table(*runner), reinterpret_cast<std::int32_t*>(r.perm), reinterpret_cast<std::int32_t*>(r.inv),
                         reinterpret_cast<void*>(r.rscratch), n);
            const unsigned span_fire[2] = { static_cast<unsigned>(n), 0 };
            cells_write(stream, r.rcells + sizeof(unsigned), span_fire, 2);
            compact_step(stream, static_cast<const char*>(t.data) + t.byte_offset, n, reinterpret_cast<std::int32_t*>(r.map),
                         reinterpret_cast<unsigned*>(r.cells) + 3, reinterpret_cast<void*>(r.scratch), rc, rc + 1, rc + 2,
                         static_cast<float>(runner->act.theta));
            const unsigned zero = 0;
            cells_write(stream, r.rcells + 2 * sizeof(unsigned), &zero, 1);
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
