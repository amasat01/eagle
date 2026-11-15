// SPDX-License-Identifier: Apache-2.0

// The extern "C" shims of eagle-door/1: each enters the context through
// `door::call` (thread, process and refusal checks; no exception crosses)
// and runs its body. The functions this build does not have yet are defined
// too, refusing EG_NOT_WRITTEN with a reason naming them, so the library
// links whole.
#include "eagle/door/Context.h"

#define EG_EXPORT extern "C" __attribute__((visibility("default")))

namespace door = eagle::door;

namespace {

/** @brief The body of a function not written in this build. */
eg_status not_written(eg_ctx* ctx, const char* name) {
    return door::call(ctx, [name](eg_ctx&) {
        door::refuse(EG_NOT_WRITTEN, std::string(name) + " is not written in this build of the door");
    });
}

/** @brief A ``*_free`` of a handle kind this build cannot make: NULL is OK. */
eg_status not_written_free(eg_ctx* ctx, const void* handle, const char* name) {
    if (handle == nullptr) return EG_OK;
    return not_written(ctx, name);
}

}  // namespace

// ---- context ---------------------------------------------------------------

EG_EXPORT eg_status eg_ctx_new(const struct eg_ctx_settings* settings, eg_ctx** out) {
    return door::ctx_new(settings, out);
}

EG_EXPORT eg_status eg_ctx_free(eg_ctx* ctx) {
    if (ctx == nullptr) return EG_OK;
    const eg_status status = door::call(ctx, door::ctx_free_all, true, true);
    if (status != EG_OK && !door::forked(*ctx)) return status;
    delete ctx;
    return EG_OK;
}

EG_EXPORT eg_status eg_error(eg_ctx* ctx, char* buf, size_t cap, size_t* len) {
    return door::error_read(ctx, buf, cap, len);
}

EG_EXPORT eg_status eg_ctx_info(eg_ctx* ctx, struct eg_ctx_info* out) {
    return door::call(ctx, [out](eg_ctx& c) { door::ctx_info(c, out); });
}

EG_EXPORT eg_status eg_host_threads(eg_ctx* ctx, int32_t threads) {
    return door::call(ctx, [threads](eg_ctx& c) {
        if (threads < 0)
            door::refuse(EG_BAD_REQUEST, "eg_host_threads: threads " + std::to_string(threads) + " < 0");
        c.host_threads = threads;
    });
}

// ---- units, planes, plans --------------------------------------------------

EG_EXPORT eg_status eg_unit_load(eg_ctx* ctx, const struct eg_unit_load_desc* desc, eg_unit** out) {
    return door::call(ctx, [&](eg_ctx& c) { door::unit_load(c, desc, out); });
}

EG_EXPORT eg_status eg_unit_free(eg_ctx* ctx, eg_unit* unit) {
    if (unit == nullptr) return EG_OK;
    return door::call(ctx, [unit](eg_ctx& c) {
        door::check_handle(c.units, unit, "unit");
        if (!unit->caller_ref) door::refuse(EG_BAD_REQUEST, "unit: already freed by the caller");
        unit->caller_ref = false;
        door::unit_release(c, unit);
    });
}

EG_EXPORT eg_status eg_plane_wrap(eg_ctx* ctx, struct DLManagedTensorVersioned* tensor, eg_plane** out) {
    return door::call(ctx, [&](eg_ctx& c) { door::plane_wrap(c, tensor, out); });
}

EG_EXPORT eg_status eg_plane_free(eg_ctx* ctx, eg_plane* plane) {
    if (plane == nullptr) return EG_OK;
    return door::call(ctx, [plane](eg_ctx& c) {
        door::check_handle(c.planes, plane, "plane");
        if (!plane->caller_ref) door::refuse(EG_BAD_REQUEST, "plane: already freed by the caller");
        plane->caller_ref = false;
        door::plane_release(c, plane);
    });
}

EG_EXPORT eg_status eg_plan_new(eg_ctx* ctx, const struct eg_plan_desc* desc, eg_plan** out) {
    return door::call(ctx, [&](eg_ctx& c) { door::plan_new(c, desc, out); });
}

EG_EXPORT eg_status eg_plan_free(eg_ctx* ctx, eg_plan* plan) {
    if (plan == nullptr) return EG_OK;
    return door::call(ctx, [plan](eg_ctx& c) {
        door::check_handle(c.plans, plan, "plan");
        door::plan_free(c, plan);
    });
}

EG_EXPORT eg_status eg_bind(eg_ctx* ctx, eg_plan* plan, const struct eg_bind_desc* desc) {
    return door::call(ctx, [&](eg_ctx& c) { door::bind(c, plan, desc); });
}

EG_EXPORT eg_status eg_launch(eg_ctx* ctx, eg_plan* plan, uint64_t* tiles) {
    return door::call(ctx, [&](eg_ctx& c) { door::launch(c, plan, tiles); });
}

// ---- not written in this build ---------------------------------------------

EG_EXPORT eg_status eg_runner_new(eg_ctx* ctx, const struct eg_runner_desc*, eg_runner**) {
    return not_written(ctx, "eg_runner_new");
}
EG_EXPORT eg_status eg_runner_free(eg_ctx* ctx, eg_runner* runner) {
    return not_written_free(ctx, runner, "eg_runner_free");
}
EG_EXPORT eg_status eg_run(eg_ctx* ctx, eg_runner*, struct eg_run_report*) {
    return not_written(ctx, "eg_run");
}
EG_EXPORT eg_status eg_run_event_times(eg_ctx* ctx, eg_runner*, eg_plane*) {
    return not_written(ctx, "eg_run_event_times");
}
EG_EXPORT eg_status eg_reverse(eg_ctx* ctx, eg_runner*, const struct eg_bind_desc*,
                               const struct eg_bind_desc*) {
    return not_written(ctx, "eg_reverse");
}
EG_EXPORT eg_status eg_reduce(eg_ctx* ctx, const struct eg_reduce_desc*) {
    return not_written(ctx, "eg_reduce");
}
EG_EXPORT eg_status eg_copy(eg_ctx* ctx, eg_plane*, eg_plane*) { return not_written(ctx, "eg_copy"); }
EG_EXPORT eg_status eg_record_begin(eg_ctx* ctx) { return not_written(ctx, "eg_record_begin"); }
EG_EXPORT eg_status eg_record_cond_begin(eg_ctx* ctx, const struct eg_cond_desc*) {
    return not_written(ctx, "eg_record_cond_begin");
}
EG_EXPORT eg_status eg_record_cond_arm(eg_ctx* ctx) { return not_written(ctx, "eg_record_cond_arm"); }
EG_EXPORT eg_status eg_record_cond_end(eg_ctx* ctx) { return not_written(ctx, "eg_record_cond_end"); }
EG_EXPORT eg_status eg_record_end(eg_ctx* ctx, eg_graph**) { return not_written(ctx, "eg_record_end"); }
EG_EXPORT eg_status eg_graph_launch(eg_ctx* ctx, eg_graph*) { return not_written(ctx, "eg_graph_launch"); }
EG_EXPORT eg_status eg_graph_free(eg_ctx* ctx, eg_graph* graph) {
    return not_written_free(ctx, graph, "eg_graph_free");
}
EG_EXPORT eg_status eg_device_count(eg_ctx* ctx, int32_t*) { return not_written(ctx, "eg_device_count"); }
EG_EXPORT eg_status eg_device_props(eg_ctx* ctx, int32_t, struct eg_device_props*) {
    return not_written(ctx, "eg_device_props");
}
EG_EXPORT eg_status eg_stream_use(eg_ctx* ctx, uint64_t) { return not_written(ctx, "eg_stream_use"); }
EG_EXPORT eg_status eg_sync(eg_ctx* ctx) { return not_written(ctx, "eg_sync"); }
EG_EXPORT eg_status eg_record_child(eg_ctx* ctx, uint64_t) { return not_written(ctx, "eg_record_child"); }
