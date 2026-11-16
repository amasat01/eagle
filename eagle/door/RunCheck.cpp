// SPDX-License-Identifier: Apache-2.0

// What a host and a DEVICE runner check alike: the runner description, the
// step's plans, and the report eg_run fills.
#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"

#include <algorithm>
#include <limits>

namespace eagle {
namespace door {

namespace {

/** @brief Refuse what this build of the runner does not run, by name. */
void check_desc(const eg_runner_desc& d, bool device) {
    if (d.flags & ~(EG_RUN_RECORD | EG_RUN_REORDER))
        refuse(EG_BAD_REQUEST, "eg_runner_desc.flags: unknown bits " + std::to_string(d.flags));
    if (d.flags & EG_RUN_RECORD)
        refuse(EG_NOT_WRITTEN, "EG_RUN_RECORD (recording a run for eg_reverse) is not written in this "
               "build of the door");
    if (d.events == EG_EVENTS_REFINE)
        refuse(EG_NOT_WRITTEN, "events REFINE (locating each stop inside its step) is not written in "
               "this build of the door");
    if (d.events != EG_EVENTS_STEP) refuse(EG_BAD_REQUEST, "unknown events " + std::to_string(d.events));
    if (d.on_no_event != EG_NO_EVENT_REFUSE && d.on_no_event != EG_NO_EVENT_ZERO)
        refuse(EG_BAD_REQUEST, "unknown on_no_event " + std::to_string(d.on_no_event));
    if (d.steps_reverse == EG_STEPS_REVERSE_FUSED || d.steps_reverse == EG_STEPS_REVERSE_AUTO)
        refuse(EG_NOT_WRITTEN, "steps_reverse FUSED and AUTO (the reverse of a recorded run) are not "
               "written in this build of the door");
    if (d.steps_reverse != EG_STEPS_REVERSE_PER_STEP)
        refuse(EG_BAD_REQUEST, "unknown steps_reverse " + std::to_string(d.steps_reverse));
    if (d.n_derived < 0) refuse(EG_BAD_REQUEST, "n_derived " + std::to_string(d.n_derived) + " < 0");
    if (d.n_derived > 0)
        refuse(EG_NOT_WRITTEN, "derived units (n_derived > 0, the record and refine roles) are not "
               "written in this build of the door");
    if (d.step_graph != nullptr && device)
        refuse(EG_NOT_WRITTEN, "step_graph (recorded device work run once per step) is not written in this build "
               "of the door");
    if (d.step_graph != nullptr)
        refuse(EG_BAD_REQUEST, "step_graph: a host runner runs no recorded device work per step; pass NULL");
    if (d.window != 0 || d.memory_budget != 0)
        refuse(EG_BAD_REQUEST, "window and memory_budget set how a recorded run is reversed: without "
               "EG_RUN_RECORD both are 0");
    if (d.max_steps < 1) refuse(EG_BAD_REQUEST, "max_steps " + std::to_string(d.max_steps) + " < 1");
    if (d.n_plans < 1 || d.plans == nullptr)
        refuse(EG_BAD_REQUEST, "eg_runner_desc: n_plans " + std::to_string(d.n_plans) + " needs plans");
}

/** @brief Check the step's plans: the first finishes, no other does, one sample count. */
void check_plans(const eg_ctx& ctx, const eg_runner_desc& d) {
    for (int i = 0; i < d.n_plans; ++i) check_handle(ctx.plans, d.plans[i], "eg_runner_desc.plans[i]");
    const eg_plan_s& first = *d.plans[0];
    const FinishFacts& f = first.unit->finish;
    if (!f.present)
        refuse(EG_BAD_REQUEST, "plans[0] (unit '" + first.unit->name + "') has no finish: the first plan "
               "of a runner carries the finish mask");
    if (first.n_samples > std::numeric_limits<std::uint32_t>::max())
        refuse(EG_BAD_REQUEST, "n_samples " + std::to_string(first.n_samples) + ": the finished counter "
               "is uint32");
    if (d.n_plans > 1 && f.steps == EG_STEPS_AUTO)
        refuse(EG_NOT_WRITTEN, "unit '" + first.unit->name + "': finish steps AUTO in a several-plan step is "
               "not written in this build of the door");
    for (int i = 1; i < d.n_plans; ++i) {
        const eg_plan_s& p = *d.plans[i];
        if (p.n_samples != first.n_samples)
            refuse(EG_BAD_REQUEST, "plans[" + std::to_string(i) + "] runs " + std::to_string(p.n_samples) +
                   " samples, plans[0] runs " + std::to_string(first.n_samples));
        if (p.unit->finish.present)
            refuse(EG_NOT_WRITTEN, "plans[" + std::to_string(i) + "] (unit '" + p.unit->name + "'): a second "
                   "finishing plan in one step is not written in this build of the door");
    }
    if (d.n_plans > 1 && f.steps != 1)
        refuse(EG_BAD_REQUEST, "unit '" + first.unit->name + "' advances " + std::to_string(f.steps) +
               " steps per launch, but each plan of a several-plan step takes one step per launch");
}

}  // namespace

/** @brief eg_runner_new's checks of @p d both runners share (@p device: a DEVICE runner). */
void check_runner(const eg_ctx& ctx, const eg_runner_desc& d, bool device) {
    check_desc(d, device);
    check_plans(ctx, d);
}

/** @brief Refuse a NULL report or one too small to hold its struct_size. */
void check_report(const eg_run_report* report) {
    if (report == nullptr) refuse(EG_BAD_REQUEST, "eg_run: report is NULL");
    if (report->struct_size < sizeof(std::uint32_t))
        refuse(EG_BAD_REQUEST, "eg_run_report.struct_size " + std::to_string(report->struct_size) + " is too small");
}

/** @brief Write @p r into @p report per the OUTPUT struct rule (the bytes its struct_size covers). */
void write_report(eg_run_report* report, eg_run_report r) {
    const std::uint32_t size = std::min<std::uint32_t>(report->struct_size, sizeof(r));
    r.struct_size = size;
    std::memcpy(static_cast<void*>(report), static_cast<const void*>(&r), size);
}

}  // namespace door
}  // namespace eagle
