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
    const bool record = (d.flags & EG_RUN_RECORD) != 0;
    if (record && device)
        refuse(EG_NOT_WRITTEN, "EG_RUN_RECORD on a DEVICE runner (recording on the device) is not written in this "
               "build of the door");
    const bool refine = d.events == EG_EVENTS_REFINE;
    if (refine && device)
        refuse(EG_NOT_WRITTEN, "events REFINE on a DEVICE runner is not written in this build of the door");
    if (d.events != EG_EVENTS_STEP && !refine) refuse(EG_BAD_REQUEST, "unknown events " + std::to_string(d.events));
    if (d.on_no_event != EG_NO_EVENT_REFUSE && d.on_no_event != EG_NO_EVENT_ZERO)
        refuse(EG_BAD_REQUEST, "unknown on_no_event " + std::to_string(d.on_no_event));
    if (d.steps_reverse == EG_STEPS_REVERSE_AUTO)
        refuse(EG_NOT_WRITTEN, "steps_reverse AUTO (the old path's measured-table choice between PER_STEP and FUSED) is not "
               "written in this build of the door");
    if (d.steps_reverse == EG_STEPS_REVERSE_FUSED && !record)
        refuse(EG_BAD_REQUEST, "steps_reverse FUSED needs EG_RUN_RECORD");
    if (d.steps_reverse != EG_STEPS_REVERSE_PER_STEP && d.steps_reverse != EG_STEPS_REVERSE_FUSED)
        refuse(EG_BAD_REQUEST, "unknown steps_reverse " + std::to_string(d.steps_reverse));
    if (d.n_derived < 0) refuse(EG_BAD_REQUEST, "n_derived " + std::to_string(d.n_derived) + " < 0");
    if (d.n_derived > 0 && !record && !refine)
        refuse(EG_NOT_WRITTEN, "derived units without EG_RUN_RECORD (the refine roles) are not "
               "written in this build of the door");
    if (record && (d.n_derived == 0 || d.derived_roles == nullptr || d.derived_units == nullptr))
        refuse(EG_BAD_REQUEST, "EG_RUN_RECORD needs its derived units: n_derived, derived_roles and derived_units");
    if (d.step_graph != nullptr) {
        if (device)
            refuse(EG_NOT_WRITTEN, "step_graph (recorded device work run once per step) is not written in this build "
                   "of the door");
        refuse(EG_BAD_REQUEST, "step_graph: a host runner runs no recorded device work per step; pass NULL");
    }
    if (!record && (d.window != 0 || d.memory_budget != 0))
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

constexpr std::int64_t kMinEvery = 4;            ///< the fewest steps between two compactions
constexpr std::int64_t kDefaultEvery = 16;       ///< the cadence when ``every`` is 0, rounded up to whole launches
constexpr double kThetaLow = 0.25, kThetaHigh = 0.75;

}  // namespace

/** @brief eg_runner_new's checks of @p d both runners share (@p device: a DEVICE runner). */
void check_runner(const eg_ctx& ctx, const eg_runner_desc& d, bool device) {
    check_desc(d, device);
    check_plans(ctx, d);
}

/** @brief The active set the plans read, or none: all of them or none must read one. */
bool check_active(const eg_runner_desc& d) {
    int reading = 0;
    for (int i = 0; i < d.n_plans; ++i) reading += d.plans[i]->unit->active_map_slot >= 0;
    if (reading == 0) {
        if (d.every != 0)
            refuse(EG_BAD_REQUEST, "every " + std::to_string(d.every) + ": the plans read no active set, so "
                   "compaction does not apply; pass 0");
        if (d.flags & EG_RUN_REORDER)
            refuse(EG_BAD_REQUEST, "EG_RUN_REORDER: the plans read no active set, so there is nothing to "
                   "reorder; build the unit with one");
        return false;
    }
    if (reading != d.n_plans)
        refuse(EG_BAD_REQUEST, "the plans of one step must all read an active set or none: " +
               std::to_string(reading) + " of " + std::to_string(d.n_plans) + " do");
    for (int i = 0; i < d.n_plans; ++i) {
        const eg_plan_s& p = *d.plans[i];
        if (p.parts.size() != 1 || p.parts[0].base != 0 || p.parts[0].count != p.n_samples)
            refuse(EG_BAD_REQUEST, "plans[" + std::to_string(i) + "]: a plan that reads an active set runs "
                   "one whole partition");
        if (p.active_owner != nullptr)
            refuse(EG_BAD_REQUEST, "plans[" + std::to_string(i) + "] already reads the active set of another "
                   "runner");
    }
    const std::int64_t group = d.plans[0]->unit->group;
    for (int i = 1; i < d.n_plans; ++i)
        if (d.plans[i]->unit->group != group)
            refuse(EG_BAD_REQUEST, "plans[" + std::to_string(i) + "] compacts groups of " +
                   std::to_string(d.plans[i]->unit->group) + " lanes, plans[0] groups of " + std::to_string(group));
    if (group > 1 && (d.flags & EG_RUN_REORDER))
        refuse(EG_BAD_REQUEST, "EG_RUN_REORDER: an active set over groups of " + std::to_string(group) +
               " lanes compacts whole groups, which a reorder would split");
    const std::int64_t k = d.plans[0]->unit->finish.steps;
    if (k == EG_STEPS_AUTO) {
        if (d.every != 0)
            refuse(EG_BAD_REQUEST, "every " + std::to_string(d.every) + ": a unit with steps AUTO compacts after "
                   "every launch in which a sample finished; pass 0");
    } else {
        if (d.every < 0) refuse(EG_BAD_REQUEST, "every " + std::to_string(d.every) + " < 0");
        if (d.every % k != 0)
            refuse(EG_BAD_REQUEST, "every " + std::to_string(d.every) + " is not a whole number of launches of " +
                   std::to_string(k) + " step(s)");
        if (d.every != 0 && d.every < kMinEvery)
            refuse(EG_BAD_REQUEST, "every " + std::to_string(d.every) + " is below " + std::to_string(kMinEvery) +
                   " steps: a compaction scans every sample, so a shorter cadence costs more than it saves");
    }
    if (d.flags & EG_RUN_REORDER)
        if (!(d.reorder_theta >= kThetaLow && d.reorder_theta <= kThetaHigh))
            refuse(EG_BAD_REQUEST, "reorder_theta " + std::to_string(d.reorder_theta) + " is outside [" +
                   std::to_string(kThetaLow) + ", " + std::to_string(kThetaHigh) + "]");
    return true;
}

std::vector<Owned> owned_planes(const eg_runner_s& runner) {
    std::vector<Owned> out;
    const auto add = [&](const eg_plane_s* plane) {
        const DLTensor& t = plane->tensor->dl_tensor;
        char* data = static_cast<char*>(t.data) + t.byte_offset;
        for (const Owned& o : out)
            if (o.data == data) return;
        const std::size_t bytes = static_cast<std::size_t>(t.dtype.bits / 8);
        std::int64_t rows = 1;               // a (components, n) plane moves one row of n samples at a time
        for (int i = 0; i + 1 < t.ndim; ++i) rows *= t.shape[i];
        for (std::int64_t r = 0; r < rows; ++r)
            out.push_back({ data + static_cast<std::size_t>(r * t.shape[t.ndim - 1]) * bytes, bytes });
    };
    const eg_plan_s& first = *runner.plans[0];
    add(first.bindings[first.unit->finish.mask_slot].plane);
    for (const eg_plan_s* p : runner.plans)
        for (std::size_t s = 0; s < p->bindings.size(); ++s) {
            const std::int32_t role = p->unit->slots[s].role;
            if (p->bindings[s].plane != nullptr && !p->unit->slots[s].gathered &&    // a gathered matrix's columns are not lanes
                (role == EG_ROLE_MUTABLE || role == EG_ROLE_PER_SAMPLE || role == EG_ROLE_VEC_IN ||
                 role == EG_ROLE_MAT_IN || role == EG_ROLE_TERMINATED || role == EG_ROLE_OUT ||
                 role == EG_ROLE_WIDE_IN || role == EG_ROLE_WIDE_OUT))
                add(p->bindings[s].plane);
        }
    return out;
}

/** @brief Refuse an index set's live count above the entries its plan holds: the map's, else the sample count's. */
void check_index_live(const eg_plan_s& plan, std::uint64_t live) {
    std::uint64_t limit = static_cast<std::uint64_t>(plan.n_samples);   // a prefix set: up to the sample count
    if (plan.unit->index_map_slot >= 0) {
        const DLTensor& m = plan.bindings[plan.unit->index_map_slot].plane->tensor->dl_tensor;
        limit = static_cast<std::uint64_t>(m.shape[0]);
    }
    if (live > limit)
        refuse(EG_BAD_REQUEST, "the index count " + std::to_string(live) + " is above the " +
               (plan.unit->index_map_slot >= 0 ? "map's " : "sample count's ") + std::to_string(limit) + " entries");
}

/** @brief The steps between two compactions of a fixed-step active set: @p d's ``every``, else the smallest
 *  whole number of launches of at least kDefaultEvery steps. */
std::int64_t active_every_steps(const eg_runner_desc& d) {
    const std::int64_t k = d.plans[0]->unit->finish.steps;
    return d.every != 0 ? d.every : k * ((kDefaultEvery + k - 1) / k);
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
