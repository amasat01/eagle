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
void device_runner_new(eg_ctx&, const eg_runner_desc*, eg_runner**) {
    refuse(EG_NOT_WRITTEN, "structure DEVICE: the device group is not built");
}
void device_run(eg_ctx&, eg_runner_s*, eg_run_report*) {}
DeviceRun::~DeviceRun() = default;

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
    d.stream = reinterpret_cast<std::uint64_t>(stream);
    d.sms = attribute(dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, "SMs");
    d.regs_per_sm = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR, "registers per SM");
    d.smem_per_sm = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR, "shared memory per SM");
    d.threads_per_sm = attribute(dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR, "threads per SM");
    d.cc_major = attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, "compute capability");
    d.cc_minor = attribute(dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, "compute capability");
}

/** @brief Destroy the context's stream (queued work still finishes) and release the primary context. */
void device_close(eg_ctx& ctx) noexcept {
    DeviceState& d = ctx.dev;
    if (d.context == 0) return;
    try {
        if (d.stream != 0) {
            EG_DRIVER(cuStreamDestroy);
            Current current(d);
            cuStreamDestroy_(reinterpret_cast<CUstream>(d.stream));
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
    try {
        EG_DRIVER(cuModuleUnload);
        Current current(unit.ctx->dev);
        cuModuleUnload_(reinterpret_cast<CUmodule>(unit.module));
    } catch (...) {
    }
    unit.module = unit.function = 0;
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
    const char* feature = unit.finish.present && unit.finish.steps == EG_STEPS_AUTO
                              ? "an automatic step count (finish steps AUTO: the fused, persistent and wave entries)"
                          : unit.index_count_slot >= 0           ? "an index set"
                          : unit.device_box                      ? "a device thread box (team grid)"
                          : unit.scalar_type == EG_SCALAR_BANDED ? "a BANDED scalar type"
                                                                 : nullptr;
    if (feature != nullptr) refuse(EG_NOT_WRITTEN, who + feature + " is not written in this build of the door");
    if (unit.function == 0) refuse(EG_NOT_WRITTEN, who + "the unit has no device image");
    for (const CompanionFacts& c : unit.companions)
        if (c.device_fold == 0)
            refuse(EG_NOT_WRITTEN, who + "the companion '" + unit.slots[c.slot].name + "' has no device fold function");
}

/** @brief Queue the plan's launches (launch_on) on the context's stream. */
void device_launch(eg_plan_s& plan) {
    Current current(plan.ctx->dev);
    launch_on(plan, plan.ctx->dev.stream);
}

/** @brief A DEVICE plan's eagle-made finished counter: one zeroed device cell. */
void device_plan_cells(eg_plan_s& plan) {
    Current current(plan.ctx->dev);
    plan.device_counter = cells_new(1);
}

/** @brief Free what device_plan_cells made. */
void device_plan_free(eg_plan_s& plan) noexcept {
    if (plan.device_counter == 0) return;
    Current current(plan.ctx->dev);
    cells_free(plan.device_counter);
    plan.device_counter = 0;
}

/** @brief A zeroed device allocation of @p bytes on the context's device. */
void* device_alloc(eg_ctx& ctx, std::size_t bytes) {
    Current current(ctx.dev);
    return reinterpret_cast<void*>(cells_new(static_cast<unsigned>((bytes + sizeof(unsigned) - 1) / sizeof(unsigned))));
}

/** @brief Free a device_alloc allocation (a forked child makes no device call). */
void device_free(eg_ctx& ctx, void* data) noexcept {
    if (data == nullptr || forked(ctx)) return;
    try {
        Current current(ctx.dev);
        cells_free(reinterpret_cast<std::uint64_t>(data));
    } catch (...) {
    }
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
    Current current(ctx->dev);
    loop_free(loop);
    cells_free(cells);
    cells_free(map);
    cells_free(scratch);
}

/** @brief eg_runner_new over DEVICE plans: the band loop (``_until_done.Runner``) of a fixed-step finishing unit,
 *  with an active set when its units read one (launches over the map's live prefix, compaction every ``every``
 *  steps when a sample finished); what else a device runner would run is refused by name. */
void device_runner_new(eg_ctx& ctx, const eg_runner_desc* desc, eg_runner** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_runner_new: out is NULL");
    *out = nullptr;
    const eg_runner_desc d = read_input(desc, "eg_runner_desc");
    for (std::int32_t i = 0; i < d.n_plans; ++i)
        if (ctx.plans.count(d.plans[i]) != 0 && (!d.plans[i]->unit->companions.empty() || !d.plans[i]->unit->cells.empty()))
            refuse(EG_NOT_WRITTEN, "plans[" + std::to_string(i) + "]: companion or cell folding inside a device runner loop "
                   "is not written in this build of the door");
    check_runner(ctx, d, true);
    for (std::int32_t i = 0; i < d.n_plans; ++i)
        if (d.plans[i]->structure != EG_STRUCTURE_DEVICE)
            refuse(EG_BAD_REQUEST, "plans[" + std::to_string(i) + "] is a host plan in a runner over DEVICE plans: "
                   "one runner runs on one side");
    if (d.flags & EG_RUN_REORDER)
        refuse(EG_NOT_WRITTEN, "EG_RUN_REORDER (reordering the planes by the active set) on a device is not written "
               "in this build of the door");
    const std::int64_t k = d.plans[0]->unit->finish.steps;
    const bool active = d.plans[0]->unit->active_map_slot >= 0 && check_active(d);
    if (active && d.plans[0]->unit->group != 1)
        refuse(EG_NOT_WRITTEN, "an active set over groups of " + std::to_string(d.plans[0]->unit->group) +
               " lanes on a device is not written in this build of the door");
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
            runner->active = true;
            runner->dev.map = cells_new(static_cast<unsigned>(std::max<std::int64_t>(n, 1)));
            runner->dev.scratch = cells_new(static_cast<unsigned>((compact_scratch_bytes(n) + sizeof(unsigned) - 1) / sizeof(unsigned)));
            std::vector<unsigned> identity(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i) identity[static_cast<std::size_t>(i)] = static_cast<unsigned>(i);
            cells_write(ctx.dev.stream, runner->dev.map, identity.data(), static_cast<unsigned>(n));
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
    if (ctx.rec != nullptr)
        refuse(EG_NOT_WRITTEN, "recording a DEVICE run (eg_run while recording) is not written in this build of "
               "the door");
    check_report(report);
    for (const eg_plan_s* p : runner->plans) check_bound(*p);
    eg_plan_s& first = *runner->plans[0];
    const FinishFacts& f = first.unit->finish;
    const std::int64_t n = first.n_samples, k = f.steps;
    const std::int64_t every = runner->active ? runner->act.every_steps : k;    // steps per loop iteration
    DeviceRun& r = runner->dev;
    const std::uint64_t stream = ctx.dev.stream;
    Current current(ctx.dev);
    std::vector<unsigned char> bound;
    for (const eg_plan_s* p : runner->plans)
        for (const eg_plan_s::Binding& b : p->bindings) {
            const auto* h = reinterpret_cast<const unsigned char*>(&b.handle);
            const auto* m = reinterpret_cast<const unsigned char*>(&b.mirror);
            bound.insert(bound.end(), h, h + sizeof(b.handle));
            bound.insert(bound.end(), m, m + sizeof(b.mirror));
            bound.insert(bound.end(), b.value, b.value + sizeof(b.value));
        }
    if (r.loop != nullptr && bound != r.bound) {
        loop_free(r.loop);
        r.loop = nullptr;
    }
    auto* counter = reinterpret_cast<unsigned*>(first.device_counter);
    unsigned compactions0 = 0;
    if (runner->active) cells_read(stream, r.cells + 5 * sizeof(unsigned), &compactions0, 1);
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
        std::function<void(std::uint64_t)> compact;
        if (runner->active)
            compact = [&t, n, cells, &r, counter](std::uint64_t s) {
                compact_step(s, static_cast<const char*>(t.data) + t.byte_offset, n, reinterpret_cast<std::int32_t*>(r.map),
                             cells + 3, reinterpret_cast<void*>(r.scratch), counter, cells + 4, cells + 5);
            };
        r.loop = loop_capture(stream, counter, cells + 2, cells, cap, body, compact, cells + 4);
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
    out.mapped = static_cast<std::int64_t>(ran) * n;
    out.build_s = r.build_s;
    write_report(report, out);
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
