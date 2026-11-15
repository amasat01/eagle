// SPDX-License-Identifier: Apache-2.0

// The context: creation, the reason text, what it is, and freeing it with
// every handle still made from it.
#include "eagle/door/Context.h"

#include "eagle/door/host/Team.h"

#include <algorithm>

#include <dlfcn.h>

namespace eagle {
namespace door {

namespace {

/** @brief Check the settings of a new context; refuses as eg_ctx_new states. */
void check_settings(const eg_ctx_settings* in) {
    const eg_ctx_settings s = read_input(in, "eg_ctx_settings");
    if (s.flags & ~EG_CTX_DEVICE)
        refuse(EG_BAD_REQUEST, "eg_ctx_settings.flags: unknown bits " +
               std::to_string(s.flags & ~EG_CTX_DEVICE));
    if (s.abi_major != EG_DOOR_VERSION_MAJOR)
        refuse(EG_BAD_REQUEST, "eg_ctx_settings.abi_major " + std::to_string(s.abi_major) +
               ": this eagle speaks eagle-door/" + std::to_string(EG_DOOR_VERSION_MAJOR));
    if (!(s.flags & EG_CTX_DEVICE)) {
        if (s.device != 0)
            refuse(EG_BAD_REQUEST, "eg_ctx_settings.device " + std::to_string(s.device) +
                   " without EG_CTX_DEVICE: a host-only context takes device 0");
        return;
    }
    if (s.device < 0)
        refuse(EG_BAD_REQUEST, "eg_ctx_settings.device " + std::to_string(s.device) +
               ": a device ordinal is >= 0");
    refuse(EG_NOT_WRITTEN, "eg_ctx_settings EG_CTX_DEVICE (device " + std::to_string(s.device) +
           "): the device group is not built");
}

}  // namespace

/** @brief Make a context; a refused one keeps its status and reason (eg_ctx_new). */
eg_status ctx_new(const eg_ctx_settings* settings, eg_ctx** out) noexcept {
    if (out == nullptr) return EG_BAD_REQUEST;
    *out = nullptr;
    eg_ctx* ctx = new (std::nothrow) eg_ctx;
    if (ctx == nullptr) return EG_FAULT;
    ctx->pid = static_cast<long>(::getpid());
    const eg_status status = call(ctx, [&](eg_ctx&) { check_settings(settings); });
    ctx->refused = status;
    *out = ctx;
    return status;
}

/**
 * @brief Free every handle still made from @p ctx: graphs and any open recording first, then runners (they hold
 * plan references), then plans (they hold plane and unit references), then
 * planes, then units. In a forked child no deleter runs and no library is
 * closed: only eagle's own memory is freed.
 */
void ctx_free_all(eg_ctx& ctx) {
    const bool child = forked(ctx);
    record_drop_all(ctx);
    for (eg_runner_s* runner : ctx.runners) delete runner;
    ctx.runners.clear();
    for (eg_plan_s* plan : std::vector<eg_plan_s*>(ctx.plans.begin(), ctx.plans.end())) {
        if (child) delete plan;
        else plan_free(ctx, plan);
    }
    ctx.plans.clear();
    for (eg_plane_s* plane : ctx.planes) {
        if (!child && plane->tensor != nullptr && plane->tensor->deleter != nullptr)
            plane->tensor->deleter(plane->tensor);
        delete plane;
    }
    ctx.planes.clear();
    for (eg_unit_s* unit : ctx.units) {
        if (!child && unit->library != nullptr) ::dlclose(unit->library);
        delete unit;
    }
    ctx.units.clear();
}

/** @brief Fill @p out per the OUTPUT struct rule (eg_ctx_info). */
void ctx_info(eg_ctx& ctx, struct eg_ctx_info* out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_ctx_info: out is NULL");
    if (out->struct_size < sizeof(std::uint32_t))
        refuse(EG_BAD_REQUEST, "eg_ctx_info.struct_size " + std::to_string(out->struct_size) +
               " is too small");
    struct eg_ctx_info info{};
    info.abi_major = EG_DOOR_VERSION_MAJOR;
    info.abi_minor = EG_DOOR_VERSION_MINOR;
    info.device = -1;
    info.capabilities = EG_CAP_HOST | EG_CAP_COND_IF_ELSE | EG_CAP_COND_SWITCH;
    info.host_threads = host::team_threads(ctx);
    info.device_type = 0;
    info.stream = -1;
    const std::uint32_t n = std::min<std::uint32_t>(out->struct_size, sizeof(info));
    info.struct_size = n;
    std::memcpy(static_cast<void*>(out), static_cast<const void*>(&info), n);
}

/** @brief The reason text, two-call protocol (eg_error); never changes the reason. */
eg_status error_read(eg_ctx* ctx, char* buf, std::size_t cap, std::size_t* len) noexcept {
    if (ctx == nullptr || len == nullptr) return EG_BAD_REQUEST;
    bool idle = false;
    if (!ctx->busy.compare_exchange_strong(idle, true, std::memory_order_acquire))
        return EG_BAD_REQUEST;
    const char* text = ctx->reason_lost ? "out of memory" : ctx->reason.c_str();
    const std::size_t n = std::strlen(text);
    *len = n;
    eg_status status = EG_OK;
    if (cap > 0) {
        if (buf == nullptr || cap <= n) status = EG_BAD_REQUEST;
        else std::memcpy(buf, text, n + 1);
    }
    ctx->busy.store(false, std::memory_order_release);
    return status;
}

}  // namespace door
}  // namespace eagle
