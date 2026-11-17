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

EG_EXPORT eg_status eg_plane_new(eg_ctx* ctx, const struct eg_plane_desc* desc, eg_plane** out) {
    return door::call(ctx, [&](eg_ctx& c) { door::plane_new(c, desc, out); });
}

EG_EXPORT eg_status eg_plane_dlpack(eg_ctx* ctx, eg_plane* plane, struct DLManagedTensorVersioned** out) {
    return door::call(ctx, [&](eg_ctx& c) { door::plane_dlpack(c, plane, out); });
}

EG_EXPORT eg_status eg_plan_new(eg_ctx* ctx, const struct eg_plan_desc* desc, eg_plan** out) {
    return door::call(ctx, [&](eg_ctx& c) {
        door::device_quiet(c, "eg_plan_new", EG_BAD_REQUEST, "a plan's cells are made outside the capture; make it first");
        door::plan_new(c, desc, out);
    });
}

EG_EXPORT eg_status eg_plan_free(eg_ctx* ctx, eg_plan* plan) {
    if (plan == nullptr) return EG_OK;
    return door::call(ctx, [plan](eg_ctx& c) {
        door::check_handle(c.plans, plan, "plan");
        if (!plan->caller_ref) door::refuse(EG_BAD_REQUEST, "plan: already freed by the caller");
        plan->caller_ref = false;
        door::plan_release(c, plan);
    });
}

EG_EXPORT eg_status eg_bind(eg_ctx* ctx, eg_plan* plan, const struct eg_bind_desc* desc) {
    return door::call(ctx, [&](eg_ctx& c) { door::bind(c, plan, desc); });
}

EG_EXPORT eg_status eg_clear(eg_ctx* ctx, eg_plan* plan, int32_t n_slots, const int32_t* slots) {
    return door::call(ctx, [&](eg_ctx& c) { door::clear(c, plan, n_slots, slots); });
}

EG_EXPORT eg_status eg_set_count(eg_ctx* ctx, eg_plan* plan, uint32_t m) {
    return door::call(ctx, [&](eg_ctx& c) { door::set_count(c, plan, m); });
}

EG_EXPORT eg_status eg_banded_pack(eg_ctx* ctx, const double* in, uint64_t* out, uint64_t n) {
    return door::call(ctx, [&](eg_ctx& c) { door::banded_pack(c, in, out, n); });
}

EG_EXPORT eg_status eg_banded_unpack(eg_ctx* ctx, const uint64_t* in, double* out, uint64_t n) {
    return door::call(ctx, [&](eg_ctx& c) { door::banded_unpack(c, in, out, n); });
}

EG_EXPORT eg_status eg_banded_pack_scalar(eg_ctx* ctx, double x, uint64_t* out) {
    return door::call(ctx, [&](eg_ctx& c) { door::banded_pack_scalar(c, x, out); });
}

EG_EXPORT eg_status eg_plan_entry_figures(eg_ctx* ctx, const eg_plan* plan, struct eg_entry_figures* out) {
    return door::call(ctx, [&](eg_ctx& c) { door::plan_entry_figures(c, plan, out); });
}

EG_EXPORT eg_status eg_launch(eg_ctx* ctx, eg_plan* plan, uint64_t* tiles) {
    return door::call(ctx, [&](eg_ctx& c) { door::launch(c, plan, tiles); });
}

// ---- graph operations and recording ----------------------------------------

EG_EXPORT eg_status eg_reduce(eg_ctx* ctx, const struct eg_reduce_desc* desc) {
    return door::call(ctx, [&](eg_ctx& c) {
        const eg_reduce_desc d = door::read_input(desc, "eg_reduce_desc");
        door::device_quiet(c, "eg_reduce", EG_NOT_WRITTEN, door::kHostWorkInGraph);
        if (door::device_planes(c, "eg_reduce", { d.in, d.out, d.comp })) door::device_reduce(c, d);
        else door::reduce(c, desc);
    });
}
EG_EXPORT eg_status eg_copy(eg_ctx* ctx, eg_plane* src, eg_plane* dst) {
    return door::call(ctx, [&](eg_ctx& c) {
        door::device_quiet(c, "eg_copy", EG_NOT_WRITTEN, door::kHostWorkInGraph);
        if (door::device_planes(c, "eg_copy", { src, dst })) door::device_copy(c, src, dst);
        else door::copy(c, src, dst);
    });
}
EG_EXPORT eg_status eg_record_begin(eg_ctx* ctx) {
    return door::call(ctx, [](eg_ctx& c) {
        if (c.dev.ordinal >= 0) door::device_record_begin(c);    // a device context records by capture
        else door::record_begin(c);
    });
}
EG_EXPORT eg_status eg_record_cond_begin(eg_ctx* ctx, const struct eg_cond_desc* desc) {
    return door::call(ctx, [desc](eg_ctx& c) {
        door::device_quiet(c, "eg_record_cond_begin", EG_NOT_WRITTEN, "conditionals in a device recording are not written in this build of the door");
        door::record_cond_begin(c, desc);
    });
}
EG_EXPORT eg_status eg_record_cond_arm(eg_ctx* ctx) { return door::call(ctx, door::record_cond_arm); }
EG_EXPORT eg_status eg_record_cond_end(eg_ctx* ctx) { return door::call(ctx, door::record_cond_end); }
EG_EXPORT eg_status eg_record_end(eg_ctx* ctx, eg_graph** out) {
    return door::call(ctx, [out](eg_ctx& c) {
        if (c.dev_rec != nullptr) door::device_record_end(c, out);
        else door::record_end(c, out);
    });
}
EG_EXPORT eg_status eg_graph_launch(eg_ctx* ctx, eg_graph* graph) {
    return door::call(ctx, [graph](eg_ctx& c) {
        if (door::device_graph(c, graph)) door::device_graph_launch(c, graph);
        else door::graph_launch(c, graph);
    });
}
EG_EXPORT eg_status eg_graph_free(eg_ctx* ctx, eg_graph* graph) {
    if (graph == nullptr) return EG_OK;
    return door::call(ctx, [graph](eg_ctx& c) {
        if (door::device_graph(c, graph)) door::device_graph_free(c, graph);
        else door::graph_free(c, graph);
    });
}

// ---- runs ------------------------------------------------------------------

EG_EXPORT eg_status eg_runner_new(eg_ctx* ctx, const struct eg_runner_desc* desc, eg_runner** out) {
    return door::call(ctx, [&](eg_ctx& c) {
        door::device_quiet(c, "eg_runner_new", EG_BAD_REQUEST, "a runner's cells are made outside the capture; make it first");
        if (door::device_plans(c, desc)) door::device_runner_new(c, desc, out);
        else door::runner_new(c, desc, out);
    });
}

EG_EXPORT eg_status eg_runner_free(eg_ctx* ctx, eg_runner* runner) {
    if (runner == nullptr) return EG_OK;
    return door::call(ctx, [runner](eg_ctx& c) {
        door::check_handle(c.runners, runner, "runner");
        if (!runner->caller_ref) door::refuse(EG_BAD_REQUEST, "runner: already freed by the caller");
        runner->caller_ref = false;
        door::runner_release(c, runner);
    });
}

EG_EXPORT eg_status eg_runner_reset(eg_ctx* ctx, eg_runner* runner) {
    return door::call(ctx, [&](eg_ctx& c) { door::runner_reset(c, runner); });
}

EG_EXPORT eg_status eg_run(eg_ctx* ctx, eg_runner* runner, struct eg_run_report* report) {
    return door::call(ctx, [&](eg_ctx& c) {
        door::check_handle(c.runners, runner, "runner");
        if (runner->dev.ctx != nullptr) door::device_run(c, runner, report);
        else {
            door::device_quiet(c, "eg_run of a host runner", EG_NOT_WRITTEN, door::kHostWorkInGraph);
            door::run(c, runner, report);
        }
    });
}

EG_EXPORT eg_status eg_runner_figures(eg_ctx* ctx, const eg_runner* runner, struct eg_runner_figures* out) {
    return door::call(ctx, [&](eg_ctx& c) { door::runner_figures(c, runner, out); });
}

EG_EXPORT eg_status eg_reverse(eg_ctx* ctx, eg_runner* runner, const struct eg_bind_desc* seeds,
                               const struct eg_bind_desc* grads) {
    return door::call(ctx, [&](eg_ctx& c) {
        door::device_quiet(c, "eg_reverse", EG_NOT_WRITTEN, door::kHostWorkInGraph);
        door::reverse(c, runner, seeds, grads);
    });
}

EG_EXPORT eg_status eg_sync(eg_ctx* ctx) {
    return door::call(ctx, [](eg_ctx& c) {
        door::device_quiet(c, "eg_sync", EG_BAD_REQUEST, "a capture cannot wait; call it after eg_record_end");
        door::device_sync(c);
    });
}
EG_EXPORT eg_status eg_stream_use(eg_ctx* ctx, uint64_t stream) {
    return door::call(ctx, [stream](eg_ctx& c) { door::device_stream_use(c, stream); });
}

// ---- not written in this build ---------------------------------------------

EG_EXPORT eg_status eg_run_events(eg_ctx* ctx, eg_runner* runner, eg_plane* t_stop, eg_plane* status) {
    return door::call(ctx, [=](eg_ctx& c) { door::run_events(c, runner, t_stop, status); });
}
EG_EXPORT eg_status eg_device_count(eg_ctx* ctx, int32_t*) { return not_written(ctx, "eg_device_count"); }
EG_EXPORT eg_status eg_device_props(eg_ctx* ctx, int32_t, struct eg_device_props*) {
    return not_written(ctx, "eg_device_props");
}
EG_EXPORT eg_status eg_record_child(eg_ctx* ctx, uint64_t) { return not_written(ctx, "eg_record_child"); }
