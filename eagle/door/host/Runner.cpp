// SPDX-License-Identifier: Apache-2.0

// Runners on the host: the run-until-done loop over bound host plans. Each
// loop iteration launches the step's plans in order on the context's team;
// the loop stops when the first plan's finished counter reaches the sample
// count or after ceil(max_steps / K) iterations. Recording, reordering, event
// refine, compaction and automatic steps are refused by name.
#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/door/host/Team.h"

#include <algorithm>
#include <limits>

namespace eagle {
namespace door {

namespace {

/** @brief Refuse what this build of the runner does not run, by name. */
void check_desc(const eg_runner_desc& d) {
    if (d.flags & ~(EG_RUN_RECORD | EG_RUN_REORDER))
        refuse(EG_BAD_REQUEST, "eg_runner_desc.flags: unknown bits " + std::to_string(d.flags));
    if (d.flags & EG_RUN_RECORD)
        refuse(EG_NOT_WRITTEN, "EG_RUN_RECORD (recording a run for eg_reverse) is not written in this "
               "build of the door");
    if (d.flags & EG_RUN_REORDER)
        refuse(EG_NOT_WRITTEN, "EG_RUN_REORDER (reordering the bound planes) is not written in this "
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
    if (d.every != 0)
        refuse(EG_BAD_REQUEST, "every " + std::to_string(d.every) + ": the plans read no active set, so "
               "compaction does not apply; pass 0");
}

}  // namespace

/** @brief eg_runner_new: check the request and keep a reference to each plan. */
void runner_new(eg_ctx& ctx, const eg_runner_desc* desc, eg_runner** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_runner_new: out is NULL");
    *out = nullptr;
    const eg_runner_desc d = read_input(desc, "eg_runner_desc");
    check_desc(d);
    check_plans(ctx, d);
    auto* runner = new eg_runner_s;
    try {
        runner->plans.assign(d.plans, d.plans + d.n_plans);
        runner->max_steps = d.max_steps;
        ctx.runners.insert(runner);
    } catch (...) {
        delete runner;
        throw;
    }
    for (eg_plan_s* p : runner->plans) ++p->refs;
    *out = runner;
}

/** @brief eg_runner_free: drop the runner's plan references. */
void runner_free(eg_ctx& ctx, eg_runner_s* runner) {
    ctx.runners.erase(runner);
    for (eg_plan_s* p : runner->plans) plan_release(ctx, p);
    delete runner;
}

/** @brief eg_run: re-sync the counter to the mask, launch the step until done or the cap,
 *  and report (``_until_done.Runner._run`` and ``_host_loop._run_host_loop``, plain loop). */
void run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report) {
    check_handle(ctx.runners, runner, "runner");
    if (report == nullptr) refuse(EG_BAD_REQUEST, "eg_run: report is NULL");
    if (report->struct_size < sizeof(std::uint32_t))
        refuse(EG_BAD_REQUEST, "eg_run_report.struct_size " + std::to_string(report->struct_size) +
               " is too small");
    for (const eg_plan_s* p : runner->plans) check_bound(*p);
    eg_plan_s& first = *runner->plans[0];
    const FinishFacts& f = first.unit->finish;
    const std::int64_t n = first.n_samples;
    const DLTensor& t = first.bindings[f.mask_slot].plane->tensor->dl_tensor;
    const auto* mask = static_cast<const unsigned char*>(t.data) + t.byte_offset;
    first.finished = static_cast<std::uint32_t>(std::count_if(mask, mask + n, [](unsigned char m) { return m != 0; }));
    const std::int64_t cap = runner->max_steps / f.steps + (runner->max_steps % f.steps != 0);
    const int threads = host::team_threads(ctx);
    std::int64_t launches = 0;
    while (launches < cap && static_cast<std::int64_t>(first.finished) != n) {
        for (const eg_plan_s* p : runner->plans) host::run_plan(*p, threads);
        ++launches;
    }
    eg_run_report r{};
    r.flags = (static_cast<std::int64_t>(first.finished) == n ? EG_REPORT_DONE : 0u) |
              (f.steps == 1 ? EG_REPORT_EXACT_STEPS : 0u);
    r.mode = EG_MODE_HOST;
    r.n = n;
    r.finished = first.finished;
    r.launches = launches;
    r.steps = launches * f.steps;
    r.mapped = launches * n;
    const std::uint32_t size = std::min<std::uint32_t>(report->struct_size, sizeof(r));
    r.struct_size = size;
    std::memcpy(static_cast<void*>(report), static_cast<const void*>(&r), size);
}

}  // namespace door
}  // namespace eagle
