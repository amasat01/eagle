// SPDX-License-Identifier: Apache-2.0

// The door's internal state: the context and the handles made from it.
//
// Every exported function of eg_door.h is a shim in abi.cpp that enters the
// context through `call` below. The shim refuses a second thread, a forked
// child and a context refused at creation BEFORE the body runs; the body
// refuses by throwing `Refusal`, and `call` turns every exception into a
// status plus the reason text in the context. No exception leaves `call`.
#pragma once

#include "eagle/door/eg_door.h"
#include "eagle/exec/Partition.h"
#include "plugin/dlpack.h"
#include "plugin/gref_layout.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include <unistd.h>

namespace eagle {
namespace door {

/** @brief A refusal of the request: the status kind and the reason text. */
struct Refusal : std::runtime_error {
    eg_status status;
    Refusal(eg_status s, const std::string& reason) : std::runtime_error(reason), status(s) {}
};

/** @brief Refuse with @p status and @p reason (never returns). */
[[noreturn]] inline void refuse(eg_status status, const std::string& reason) {
    throw Refusal(status, reason);
}

/**
 * @brief Read a versioned INPUT struct: zero-fill, copy the bytes that fit.
 *
 * A struct whose ``struct_size`` cannot hold ``struct_size`` and ``flags`` is
 * refused; the rest of a short struct reads as zero.
 */
template <class T>
T read_input(const T* in, const char* what) {
    if (in == nullptr) refuse(EG_BAD_REQUEST, std::string(what) + " is NULL");
    if (in->struct_size < 2 * sizeof(std::uint32_t))
        refuse(EG_BAD_REQUEST, std::string(what) + ": struct_size " +
               std::to_string(in->struct_size) + " is too small");
    T out{};
    const std::size_t n = in->struct_size < sizeof(T) ? in->struct_size : sizeof(T);
    std::memcpy(static_cast<void*>(&out), static_cast<const void*>(in), n);
    return out;
}

/** @brief One argument slot of a loaded unit, as eg_bind and eg_launch read it. */
struct SlotFacts {
    std::int32_t role = 0;
    std::int32_t dtype = 0;
    std::string name;
};

}  // namespace door
}  // namespace eagle

/** @brief One loaded unit: its library, its entry and the slot facts copied from its meta. */
struct eg_unit_s {
    eg_ctx* ctx = nullptr;
    void* library = nullptr;                  ///< dlopen handle, closed when ``refs`` drops to 0
    eagle::exec::HostEntryV2 host_entry = nullptr;
    std::string name;
    std::int32_t scalar_type = 0;
    std::int32_t access = 0;
    std::vector<eagle::door::SlotFacts> slots;
    int refs = 1;                             ///< the caller's reference plus one per plan
    bool caller_ref = true;
};

/** @brief One DLPack tensor taken by eagle: reference-counted, deleter called once. */
struct eg_plane_s {
    eg_ctx* ctx = nullptr;
    DLManagedTensorVersioned* tensor = nullptr;
    int refs = 1;                             ///< the caller's reference plus one per binding
    bool caller_ref = true;
};

/** @brief A unit under one structure over fixed partitions, plus its bindings. */
struct eg_plan_s {
    /** @brief One slot's binding: a plane (by handle) or a uniform (by value). */
    struct Binding {
        eg_plane_s* plane = nullptr;
        bool bound = false;
        eagle::plugin::ScalarHandle handle{};
        alignas(8) unsigned char value[16] = {};
    };
    eg_ctx* ctx = nullptr;
    eg_unit_s* unit = nullptr;
    eg_structure structure = EG_STRUCTURE_HOST_TEAM;
    std::int64_t n_samples = 0;
    std::vector<eagle::exec::Partition> parts;
    std::vector<Binding> bindings;            ///< one per slot; never resized after eg_plan_new
    std::vector<void*> params;                ///< the entry's argument list, into ``bindings``
};

/** @brief All door state of one caller. */
struct eg_ctx_s {
    long pid = 0;                             ///< the process that made the context
    std::atomic<bool> busy{ false };          ///< a call is running on the context
    eg_status refused = EG_OK;                ///< a refused eg_ctx_new: every call returns it
    std::int32_t host_threads = 0;            ///< 0: the default budget
    std::string reason;                       ///< the last non-OK status's reason
    bool reason_lost = false;                 ///< the reason could not be stored (out of memory)
    std::unordered_set<eg_unit_s*> units;
    std::unordered_set<eg_plane_s*> planes;
    std::unordered_set<eg_plan_s*> plans;

    /** @brief Store @p text as the reason, never throwing. */
    void set_reason(const char* text) noexcept {
        try {
            reason = text;
            reason_lost = false;
        } catch (...) {
            reason_lost = true;
        }
    }
};

namespace eagle {
namespace door {

/** @brief True in a process other than the one that made @p ctx. */
inline bool forked(const eg_ctx& ctx) { return ctx.pid != static_cast<long>(::getpid()); }

/**
 * @brief Run @p body on @p ctx under the door's rules and return its status.
 *
 * Refused before @p body runs: a NULL context, a second thread (no reason is
 * written: the reason belongs to the thread inside), a forked child unless
 * @p in_child, and a context refused at creation unless @p in_refused.
 */
template <class F>
eg_status call(eg_ctx* ctx, F&& body, bool in_child = false, bool in_refused = false) noexcept {
    if (ctx == nullptr) return EG_BAD_REQUEST;
    bool idle = false;
    if (!ctx->busy.compare_exchange_strong(idle, true, std::memory_order_acquire))
        return EG_BAD_REQUEST;
    eg_status status = EG_OK;
    if (!in_child && forked(*ctx)) {
        status = EG_BAD_REQUEST;
        ctx->set_reason("this context belongs to another process: a forked child may only call "
                        "eg_error and eg_ctx_free");
    } else if (!in_refused && ctx->refused != EG_OK) {
        status = ctx->refused;
    } else {
        try {
            body(*ctx);
        } catch (const Refusal& r) {
            status = r.status;
            ctx->set_reason(r.what());
        } catch (const std::bad_alloc&) {
            status = EG_FAULT;
            ctx->set_reason("out of memory");
        } catch (const std::exception& e) {
            status = EG_FAULT;
            ctx->set_reason(e.what());
        } catch (...) {
            status = EG_FAULT;
            ctx->set_reason("an unknown failure inside eagle");
        }
    }
    ctx->busy.store(false, std::memory_order_release);
    return status;
}

/** @brief Refuse @p handle unless @p set (a context's registry) holds it. */
template <class H>
void check_handle(const std::unordered_set<H*>& set, H* handle, const char* what) {
    if (handle == nullptr) refuse(EG_BAD_REQUEST, std::string(what) + " is NULL");
    if (set.count(handle) == 0)
        refuse(EG_BAD_REQUEST, std::string(what) +
               " is not a live handle of this context (freed, or made by another context)");
}

// Bodies of the exported functions (Context.cpp, Unit.cpp, Plane.cpp, Plan.cpp).
eg_status ctx_new(const eg_ctx_settings* settings, eg_ctx** out) noexcept;
eg_status error_read(eg_ctx* ctx, char* buf, std::size_t cap, std::size_t* len) noexcept;
void ctx_free_all(eg_ctx& ctx);
void ctx_info(eg_ctx& ctx, struct eg_ctx_info* out);
void unit_load(eg_ctx& ctx, const eg_unit_load_desc* desc, eg_unit** out);
void unit_release(eg_ctx& ctx, eg_unit_s* unit);
void plane_wrap(eg_ctx& ctx, DLManagedTensorVersioned* tensor, eg_plane** out);
void plane_release(eg_ctx& ctx, eg_plane_s* plane);
void plan_new(eg_ctx& ctx, const eg_plan_desc* desc, eg_plan** out);
void plan_free(eg_ctx& ctx, eg_plan_s* plan);
void bind(eg_ctx& ctx, eg_plan_s* plan, const eg_bind_desc* desc);
void launch(eg_ctx& ctx, eg_plan_s* plan, std::uint64_t* tiles);

}  // namespace door
}  // namespace eagle
