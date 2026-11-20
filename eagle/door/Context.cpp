// SPDX-License-Identifier: Apache-2.0

// The context: creation, the reason text, what it is, and freeing it with
// every handle still made from it.
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"

#include "eagle/door/Machine.h"
#include "eagle/door/host/Team.h"

#include <algorithm>


namespace eagle {
namespace door {

namespace {

/** @brief Check the settings of a new context and open its device; refuses as eg_ctx_new states. */
void check_settings(eg_ctx& ctx, const eg_ctx_settings* in) {
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
    device_open(ctx, s.device);
}

// The row tables come from the header the build force-includes (DOOR_ROWS in the Makefile, EAGLE_DOOR_ROWS in CMake).
#ifndef EG_WEIGHT_ROWS
#error "the door's row header is not included: set DOOR_ROWS (Makefile) or EAGLE_DOOR_ROWS (CMake)"
#endif
#define EG_WEIGHT_ROW_INIT(...) { __VA_ARGS__ },
const eagle::policy::WeightRow kWeightRows[] = { EG_WEIGHT_ROWS(EG_WEIGHT_ROW_INIT) };
#undef EG_WEIGHT_ROW_INIT
constexpr int kWeightRowCount = static_cast<int>(sizeof(kWeightRows) / sizeof(kWeightRows[0]));

#define EG_GEMM_ROUTE_ROW_INIT(...) { __VA_ARGS__ },
// A trailing zero row keeps the array non-empty when a host-only door's header gives no rows; it is not counted.
const eagle::policy::GemmRouteRow kGemmRouteRows[] = { EG_GEMM_ROUTE_ROWS(EG_GEMM_ROUTE_ROW_INIT) { 0, 0, 0, 0, 0 } };
#undef EG_GEMM_ROUTE_ROW_INIT
constexpr int kGemmRouteRowCount = static_cast<int>(sizeof(kGemmRouteRows) / sizeof(kGemmRouteRows[0])) - 1;

/** @brief The built-in row of @p target (the row whose class is all zero); a table without one is a defect of the row header. */
const eagle::policy::WeightRow& builtin_weight_row(eagle::policy::Target target) {
    for (const eagle::policy::WeightRow& r : kWeightRows)
        if (r.target == static_cast<int>(target) && !r.vendor && !r.family && !r.model) return r;
    refuse(EG_FAULT, "eagle_door_rows.h: a target has no built-in weights row");
}

/** @brief Look the dispatch weights up once, by the class of the machine this context runs on (a host context: the host's CPUID class; a device
 *  context: the device's compute capability). */
void resolve_weights(eg_ctx& ctx) {
    namespace pol = eagle::policy;
    const bool device = ctx.dev.ordinal >= 0;
    const HostClass host = device ? HostClass{} : host_class();
    ctx.weight_vendor = device ? 0 : host.vendor;
    ctx.weight_family = device ? ctx.dev.cc_major : host.family;
    ctx.weight_model = device ? ctx.dev.cc_minor : host.model;
    const pol::Target target = device ? pol::Target::Device : pol::Target::Host;
    ctx.weight_row = pol::weightRowIndex(kWeightRows, kWeightRowCount, target, ctx.weight_vendor, ctx.weight_family, ctx.weight_model);
    const pol::WeightRow& row = ctx.weight_row >= 0 ? kWeightRows[ctx.weight_row] : builtin_weight_row(target);
    if (device) {
        ctx.props.cc_major = ctx.dev.cc_major;
        ctx.props.cc_minor = ctx.dev.cc_minor;
        ctx.props.sm_count = ctx.dev.sms;
        ctx.props.clock_rate_khz = ctx.dev.clock_khz;
        ctx.props.memory_clock_rate_khz = ctx.dev.memory_clock_khz;
        ctx.props.memory_bus_width_bits = ctx.dev.bus_bits;
        ctx.props.regs_per_sm = ctx.dev.regs_per_sm;
        ctx.props.warp_size = 32;
        ctx.props = eagle::deriveFields(ctx.props);
        ctx.weights = pol::weightsOfRow(row, pol::Machine{ 1, static_cast<double>(ctx.dev.l2_bytes) });
    } else {
        const int cores = physical_cores();
        ctx.props = eagle::cpu::deviceProps();
        ctx.weights = pol::weightsOfRow(row, pol::Machine{ static_cast<double>(cores), host_cache_bytes(cores) });
    }
}

}  // namespace

void resolve_gemm_routes(eg_ctx& ctx, const eagle::policy::GemmRouteRow* rows, int count) {
    if (ctx.dev.ordinal < 0) return;
    if (const char* fault = eagle::policy::gemmRouteRowsFault(rows, count)) refuse(EG_FAULT, fault);
    for (int w = 0; w < 2; ++w)
        ctx.gemm_by_cublas[w] = eagle::policy::gemmRoute(rows, count, ctx.dev.cc_major, ctx.dev.cc_minor, w ? 64 : 32) == 1;
}

/** @brief Make a context; a refused one keeps its status and reason (eg_ctx_new). */
eg_status ctx_new(const eg_ctx_settings* settings, eg_ctx** out) noexcept {
    if (out == nullptr) return EG_BAD_REQUEST;
    *out = nullptr;
    eg_ctx* ctx = new (std::nothrow) eg_ctx;
    if (ctx == nullptr) return EG_FAULT;
    ctx->pid = static_cast<long>(::getpid());
    const eg_status status = call(ctx, [&](eg_ctx& c) { check_settings(c, settings); resolve_weights(c); resolve_gemm_routes(c, kGemmRouteRows, kGemmRouteRowCount); });
    ctx->refused = status;
    *out = ctx;
    return status;
}

/**
 * @brief Free every handle still made from @p ctx: graphs and any open recording first, then runners (they hold
 * plan references), then plans (they hold plane and unit references), then
 * planes, then units, then the device. In a forked child no deleter runs, no
 * library is closed and no device call is made: only eagle's own memory is freed.
 */
void ctx_free_all(eg_ctx& ctx) {
    const bool child = forked(ctx);
    record_drop_all(ctx);
    device_graphs_drop(ctx);
    for (eg_runner_s* runner : ctx.runners) delete runner;
    ctx.runners.clear();
    for (eg_plan_s* plan : std::vector<eg_plan_s*>(ctx.plans.begin(), ctx.plans.end())) {
        if (child) delete plan;
        else plan_free(ctx, plan);
    }
    ctx.plans.clear();
    fft_free_all(ctx);
    for (eg_gemm_s* gemm : ctx.gemms) delete gemm;
    ctx.gemms.clear();
    for (eg_plane_s* plane : ctx.planes) {
        if (!child && plane->tensor != nullptr && plane->tensor->deleter != nullptr)
            plane->tensor->deleter(plane->tensor);
        delete plane;
    }
    ctx.planes.clear();
    for (eg_unit_s* unit : ctx.units) {
        if (!child) device_unit_free(*unit);
        if (!child && unit->library != nullptr) close_library(unit->library, unit->library_fd);
        delete unit;
    }
    ctx.units.clear();
    if (!child) device_close(ctx);
}

/** @brief eg_ctx_free's body: with exports outstanding mark @p ctx closed and return true (the last export's release
 *  frees it); otherwise free everything now and return false. */
bool ctx_close(eg_ctx& ctx) {
    if (!forked(ctx) && ctx.exports > 0) {
        ctx.closed = true;
        return true;
    }
    ctx_free_all(ctx);
    return false;
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
    const bool device = ctx.dev.ordinal >= 0;
    info.device = ctx.dev.ordinal;
    info.capabilities = EG_CAP_HOST | EG_CAP_COND_IF_ELSE | EG_CAP_COND_SWITCH | (device ? EG_CAP_DEVICE : 0);
    info.host_threads = host::team_threads(ctx);
    info.device_type = device ? kDLCUDA : 0;
    info.stream = device ? static_cast<std::int64_t>(ctx.dev.stream) : -1;
    info.weight_vendor = ctx.weight_vendor;
    info.weight_family = ctx.weight_family;
    info.weight_model = ctx.weight_model;
    info.weight_row = ctx.weight_row;
    const std::uint32_t n = std::min<std::uint32_t>(out->struct_size, sizeof(info));
    info.struct_size = n;
    std::memcpy(static_cast<void*>(out), static_cast<const void*>(&info), n);
}

/** @brief The calling thread's reason text, two-call protocol (eg_error); never changes a reason. */
eg_status error_read(eg_ctx* ctx, char* buf, std::size_t cap, std::size_t* len) noexcept {
    if (ctx == nullptr || len == nullptr) return EG_BAD_REQUEST;
    Hold hold(*ctx);
    const auto found = ctx->reasons.find(std::this_thread::get_id());
    const bool lost = found == ctx->reasons.end() ? ctx->reason_lost : found->second.lost;
    const char* text = lost ? "out of memory" : found == ctx->reasons.end() ? "" : found->second.text.c_str();
    const std::size_t n = std::strlen(text);
    *len = n;
    eg_status status = EG_OK;
    if (cap > 0) {
        if (buf == nullptr || cap <= n) status = EG_BAD_REQUEST;
        else std::memcpy(buf, text, n + 1);
    }
    return status;
}

}  // namespace door
}  // namespace eagle
