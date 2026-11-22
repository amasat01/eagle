// SPDX-License-Identifier: Apache-2.0

// The look of an AUTO dispatch group that names its segmented set. At the start of a run the classify unit gives the exact mix of the live
// samples, and the rule (eagle/policy/Dispatch.h, priced by the context's weights row) says fused or segmented. A segmented run is stepped by
// the runner, one launch of the segmented plan per step, and looks at the live mix again every DECISION_EVERY steps; the first look that leaves the
// band hands the rest of the run to the fused path for good (the fused plan the caller bound, run by the runner's own loop). A run that starts
// fused never looks again. Only a static selector is decided: its classes are known before the first step.
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"
#include "eagle/door/host/Segment.h"
#include "eagle/door/host/Team.h"

namespace eagle {
namespace door {

namespace {

namespace pol = eagle::policy;

/** @brief The cost vector the rule reads from an AUTO record: the branch costs as stated; a branch whose vectorisation is not stated takes the fused loop's. */
pol::CostVector cost_of(const DispatchFacts& g) {
    pol::CostVector c;
    c.count = g.k;
    c.fusedVectorised = (g.cost_flags & EG_DISPATCH_FUSED_VECTORISED) != 0;
    bool all_stated = true;
    for (const eg_branch_cost& b : g.costs) all_stated = all_stated && (b.flags & EG_BRANCH_VECTOR_STATED) != 0;
    for (std::int32_t j = 0; j < g.k; ++j) {
        const eg_branch_cost& b = g.costs[static_cast<std::size_t>(j)];
        pol::BranchCost& out = c.branch[static_cast<std::size_t>(j)];
        out.arith32 = b.arith32, out.arith64 = b.arith64, out.special32 = b.special32, out.special64 = b.special64;
        out.loads = b.loads, out.gathers = b.gathers;
        out.chain32 = b.chain32, out.chain64 = b.chain64, out.chainSpecial32 = b.chain_special32, out.chainSpecial64 = b.chain_special64;
        out.vectorised = all_stated ? (b.flags & EG_BRANCH_VECTORISED) != 0 : c.fusedVectorised;
        out.stores = (b.flags & EG_BRANCH_HAS_STORES) != 0 ? b.stores : -1.0;
    }
    return c;
}

const DispatchFacts& group_of(const eg_unit_s& unit) {
    for (const DispatchFacts& f : unit.dispatches)
        if (f.strategy == EG_DISPATCH_AUTO && f.names_set) return f;
    throw std::logic_error("look: a unit with an AUTO segmented set has no such group");
}

/** @brief The samples that are live (mask element zero) in each class: @p classes holds the class of every sample, @p mask the mask plane's bytes. */
void count_live(const std::vector<std::int64_t>& classes, const unsigned char* mask, std::size_t width, std::int64_t n, pol::Mix& mix) {
    mix.counts.fill(0);
    for (std::int64_t i = 0; i < n; ++i) {
        bool dead = false;
        for (std::size_t b = 0; b < width; ++b) dead = dead || mask[static_cast<std::size_t>(i) * width + b] != 0;
        if (!dead) ++mix.counts[static_cast<std::size_t>(classes[static_cast<std::size_t>(i)])];
    }
}

/** @brief The live mix now: the classes the segmented plan holds (@p classify: run the classify unit first) and the mask as it stands. */
void live_mix(eg_ctx& ctx, eg_plan_s& first, eg_plan_s& seg, bool device, bool classify, pol::Mix& mix) {
    const std::int64_t n = first.n_samples;
    const DLTensor& t = first.bindings[static_cast<std::size_t>(first.unit->finish.mask_slot)].plane->tensor->dl_tensor;
    const std::size_t width = t.dtype.bits / 8;
    std::vector<std::int64_t> classes(static_cast<std::size_t>(n));
    std::vector<unsigned char> mask_bytes;
    const unsigned char* mask = static_cast<const unsigned char*>(t.data) + t.byte_offset;
    if (device) {
        if (classify) device_classify(seg);
        const DLTensor& ct = seg.segdev[1]->tensor->dl_tensor;
        device_read(ctx, static_cast<const char*>(ct.data) + ct.byte_offset, classes.data(), classes.size() * sizeof(std::int64_t));
        mask_bytes.resize(static_cast<std::size_t>(n) * width);
        device_read(ctx, mask, mask_bytes.data(), mask_bytes.size());
        mask = mask_bytes.data();
    } else {
        if (classify) host::classify_host(seg, host::team_threads(ctx));
        classes = seg.seg.cls;
    }
    count_live(classes, mask, width, n, mix);
}

/** @brief Bind the planes and uniforms the runner's plan holds to the segmented plan's slots of the same name (eagle's own slots of the group are made by the plan). */
void sync_bindings(eg_ctx& ctx, const eg_plan_s& first, eg_plan_s& seg) {
    std::vector<std::int32_t> plane_slots, uniform_slots;
    std::vector<eg_plane*> planes;
    std::vector<const void*> values;
    const eg_unit_s& su = *seg.unit;
    for (std::size_t i = 0; i < su.slots.size(); ++i) {
        const SlotFacts& sl = su.slots[i];
        if (sl.made || sl.role == EG_ROLE_NSAMPLES || sl.role == EG_ROLE_EXTENT
            || static_cast<std::int32_t>(i) == su.finish.counter_slot)
            continue;
        const eg_plan_s::Binding* from = nullptr;
        std::size_t at = 0;
        for (const SlotFacts& f : first.unit->slots) {
            if (f.name == sl.name && first.bindings[at].bound) from = &first.bindings[at];
            ++at;
        }
        if (from == nullptr)
            refuse(EG_BAD_REQUEST, "unit '" + first.unit->name + "': the segmented set's slot '" + sl.name + "' is not a bound slot of the AUTO unit");
        if (sl.role == EG_ROLE_UNIFORM) {
            uniform_slots.push_back(static_cast<std::int32_t>(i));
            values.push_back(from->value);
        } else if (from->plane != seg.bindings[i].plane) {
            plane_slots.push_back(static_cast<std::int32_t>(i));
            planes.push_back(from->plane);
        }
    }
    eg_bind_desc d{};
    d.struct_size = sizeof(d);
    d.n_planes = static_cast<std::int32_t>(planes.size());
    d.n_uniforms = static_cast<std::int32_t>(uniform_slots.size());
    d.plane_slots = plane_slots.data();
    d.planes = planes.data();
    d.uniform_slots = uniform_slots.data();
    d.uniform_values = values.data();
    bind(ctx, &seg, &d);
}

}  // namespace

void look_check(const eg_ctx& ctx, const eg_runner_desc& desc) {
    if (desc.plans == nullptr) return;
    for (int i = 0; i < desc.n_plans; ++i) {
        if (ctx.plans.count(desc.plans[i]) == 0 || desc.plans[i]->unit->auto_seg == nullptr) continue;
        const eg_unit_s& u = *desc.plans[i]->unit;
        const std::string who = "unit '" + u.name + "': an AUTO dispatch group with a segmented set ";
        if (desc.n_plans > 1) refuse(EG_NOT_WRITTEN, who + "is the whole step of its runner, not one plan of several");
        if ((desc.flags & (EG_RUN_RECORD | EG_RUN_REORDER)) || desc.events == EG_EVENTS_REFINE)
            refuse(EG_NOT_WRITTEN, who + "is not recorded, refined or reordered");
        if (u.finish.steps == EG_STEPS_AUTO || u.finish.steps != u.auto_seg->finish.steps)
            refuse(EG_NOT_WRITTEN, who + "needs a fixed step count per launch, the same as its segmented set's");
        if (desc.plans[i]->parts.size() != 1 || desc.plans[i]->parts[0].base != 0 || desc.plans[i]->parts[0].count != desc.plans[i]->n_samples)
            refuse(EG_NOT_WRITTEN, who + "runs one whole partition");
    }
}

void look_attach(eg_ctx& ctx, eg_runner_s& runner) {
    eg_plan_s& first = *runner.plans[0];
    if (first.unit->auto_seg == nullptr) return;
    eg_plan_desc d{};
    d.struct_size = sizeof(d);
    d.unit = first.unit->auto_seg;
    d.structure = first.structure;
    d.n_samples = first.n_samples;
    eg_plan* seg = nullptr;
    plan_new(ctx, &d, &seg);
    auto* look = new AutoLook;
    look->seg = seg;
    look->group = &group_of(*first.unit);
    look->cost = cost_of(*look->group);
    runner.look = look;
}

void look_reset(eg_ctx& ctx, eg_runner_s& runner) {
    AutoLook& L = *runner.look;
    L.state = pol::HysteresisState{};
    L.gain = 1.0;
    L.looks = 0;
    seg_reset(ctx, *L.seg);
}

void look_free(eg_ctx& ctx, eg_runner_s& runner) {
    if (runner.look == nullptr) return;
    plan_release(ctx, runner.look->seg);
    delete runner.look;
    runner.look = nullptr;
}

void look_run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report) {
    check_report(report);
    for (const eg_plan_s* p : runner->plans) check_bound(*p);
    AutoLook& L = *runner->look;
    eg_plan_s& first = *runner->plans[0];
    eg_plan_s& seg = *L.seg;
    const bool device = first.structure == EG_STRUCTURE_DEVICE;
    const FinishFacts& f = first.unit->finish;
    const std::int64_t n = first.n_samples, K = f.steps, total = runner->max_steps;
    sync_bindings(ctx, first, seg);
    pol::Mix mix;
    mix.fusedSteps = K;
    L.state = pol::HysteresisState{};
    L.looks = 0;
    const auto look = [&](std::int64_t steps_done, bool classify) {
        live_mix(ctx, first, seg, device, classify, mix);
        mix.remainingSteps = total - steps_done;
        L.gain = pol::predictedGain(L.cost, mix, ctx.props, ctx.weights);
        L.state.current = pol::decide(L.gain, L.state.current);
        ++L.looks;
    };
    std::int64_t steps = 0, launches = 0;
    const std::int64_t sorts0 = seg.seg.sorts;
    const DLTensor& mt = first.bindings[static_cast<std::size_t>(f.mask_slot)].plane->tensor->dl_tensor;
    const auto finished_now = [&]() -> std::int64_t {    // the samples whose mask is set, counted from the plane
        std::vector<unsigned char> bytes;
        const unsigned char* m = static_cast<const unsigned char*>(mt.data) + mt.byte_offset;
        const std::size_t width = mt.dtype.bits / 8;
        if (device) {
            bytes.resize(static_cast<std::size_t>(n) * width);
            device_read(ctx, m, bytes.data(), bytes.size());
            m = bytes.data();
        }
        std::int64_t set = 0;
        for (std::int64_t i = 0; i < n; ++i)
            for (std::size_t b = 0; b < width; ++b)
                if (m[static_cast<std::size_t>(i) * width + b] != 0) { ++set; break; }
        return set;
    };
    std::int64_t finished = finished_now();
    if (finished != n && total > 0) {
        look(0, true);
        if (L.state.current == pol::Policy::Segmented) {
            seg.finished = static_cast<std::uint32_t>(finished);
            if (device) device_mask_count(ctx, static_cast<const char*>(mt.data) + mt.byte_offset, n, mt.dtype.bits / 8u, seg.device_counter);
            const int threads = host::team_threads(ctx);
            while (L.state.current == pol::Policy::Segmented && steps < total && finished != n) {
                if (device) device_launch(seg);
                else host::run_plan(seg, threads);
                steps += K;
                ++launches;
                if (device) {
                    unsigned c = 0;
                    device_read(ctx, reinterpret_cast<const void*>(seg.device_counter), &c, sizeof(c));
                    finished = c;
                } else {
                    finished = seg.finished;
                }
                if (pol::due(L.state, static_cast<int>(K)) && finished != n && steps < total) look(steps, false);
            }
        }
    }
    eg_run_report rep{};
    rep.struct_size = sizeof(rep);
    const std::int64_t left = total - steps;
    if (finished != n && left > 0) {    // the rest of the run is the fused path, as for any runner
        runner->max_steps = left;
        try {
            if (device) device_run(ctx, runner, &rep);
            else run_loop(ctx, runner, &rep);
        } catch (...) {
            runner->max_steps = total;
            throw;
        }
        runner->max_steps = total;
        if (device && launches > 0) device_loop_drop(*runner);
    } else {
        rep.mode = device ? EG_MODE_BAND : EG_MODE_HOST;
        rep.n = n;
        rep.finished = finished;
        rep.flags = (finished == n ? EG_REPORT_DONE : 0u) | (K == 1 ? EG_REPORT_EXACT_STEPS : 0u);
    }
    rep.launches += launches;
    rep.steps += steps;
    rep.mapped += launches * n;
    note_launches_by_k(rep, K, launches);
    if (L.looks > 0) {    // the run decided the group: the policy in force at the end and the gain of the last look
        rep.flags |= EG_REPORT_DISPATCH;
        rep.dispatch_policy = L.state.current == pol::Policy::Segmented ? EG_DISPATCH_POLICY_SEGMENTED : EG_DISPATCH_POLICY_SWITCH;
        rep.dispatch_gain = L.gain;
        rep.class_sorts = seg.seg.sorts - sorts0;
    }
    if (report != nullptr) write_report(report, rep);
}

}  // namespace door
}  // namespace eagle
