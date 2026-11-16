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
#include "eagle/exec/DeviceKernel.h"
#include "eagle/policy/Residency.h"

#include <algorithm>

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

/** @brief Refuse a DEVICE plan of @p unit on @p ctx: a host-only context, a device feature this
 *  build does not run (each by name), a unit with no device function. */
void device_plan_check(const eg_ctx& ctx, const eg_unit_s& unit) {
    if (ctx.dev.ordinal < 0)
        refuse(EG_BAD_REQUEST, "structure DEVICE on a host-only context (made without EG_CTX_DEVICE)");
    const std::string who = "unit '" + unit.name + "' on DEVICE: ";
    const char* feature = unit.active_map_slot >= 0              ? "an active set (compaction)"
                          : unit.finish.present                  ? "a finish record (runners on a device)"
                          : !unit.companions.empty()             ? "a companion plane"
                          : !unit.cells.empty()                  ? "cells"
                          : unit.index_count_slot >= 0           ? "an index set"
                          : unit.device_box                      ? "a device thread box (team grid)"
                          : unit.scalar_type == EG_SCALAR_BANDED ? "a BANDED scalar type"
                                                                 : nullptr;
    if (feature != nullptr) refuse(EG_NOT_WRITTEN, who + feature + " is not written in this build of the door");
    if (unit.function == 0) refuse(EG_NOT_WRITTEN, who + "the unit has no device image");
}

/** @brief Queue one launch per partition on the context's stream: grid from the partition's count and the
 *  meta's block (else the resolved one), the bound views and uniforms by value, then the triple. */
void device_launch(eg_plan_s& plan) {
    const DeviceState& d = plan.ctx->dev;
    const eg_unit_s& u = *plan.unit;
    Current current(d);
    for (const exec::Partition& p : plan.parts)
        exec::DeviceKernel::run(reinterpret_cast<CUfunction>(u.function), plan.params, p, reinterpret_cast<CUstream>(d.stream),
                                u.block > 0 ? static_cast<unsigned>(u.block) : resolved_block(d, u, p.count));
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
