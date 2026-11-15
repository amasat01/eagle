// SPDX-License-Identifier: Apache-2.0

// Runners on the host: the run-until-done loop over bound host plans. Each
// loop iteration launches the step's plans in order on the context's team;
// the loop stops when the first plan's finished counter reaches the sample
// count or after ceil(max_steps / K) iterations. Units that read an active set
// compact it every ``every`` steps and may reorder the bound planes
// (EG_RUN_REORDER). Recording, event refine and automatic steps are refused by name.
#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/door/host/Team.h"
#include "eagle/policy/Launch.h"

#include <algorithm>
#include <limits>

namespace eagle {
namespace door {

namespace {

constexpr std::int64_t kMinEvery = 4;            ///< the fewest steps between two compactions
constexpr std::int64_t kDefaultEvery = 16;       ///< the cadence when ``every`` is 0, rounded up to whole launches
constexpr std::int64_t kMinReorderSpan = 4096;   ///< the fewest samples a reorder is built for and fires over
constexpr std::int64_t kAutoKMin = 8;            ///< AUTO: the fewest steps of a launch but the last, and the compaction cadence
constexpr std::int64_t kAutoK0 = 16;             ///< AUTO: the steps of the first launch
constexpr double kAutoRate = 8.0;                ///< AUTO: a compaction's cost in full-live steps on the host
constexpr double kThetaLow = 0.25, kThetaHigh = 0.75;

/** @brief Refuse what this build of the runner does not run, by name. */
void check_desc(const eg_runner_desc& d) {
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


/** @brief Recompute the map and count from the mask; for a reordering set also the trigger:
 *  fire when the live count is below theta * 32 * (32-slot groups holding a live sample),
 *  below the span, and the span is at least the floor (the old host rule, bit for bit). */
void compact(ActiveState& a, const unsigned char* mask) {
    std::uint32_t live = 0;
    for (std::int64_t i = 0; i < a.n; ++i)
        if (mask[i] == 0) a.map[live++] = static_cast<std::int32_t>(i);
    a.count = live;
    if (!a.reorder) return;
    std::int64_t live32 = 0;
    for (std::int64_t g = 0; g < a.n; g += 32) {
        const std::int64_t end = std::min<std::int64_t>(g + 32, a.n);
        live32 += std::any_of(mask + g, mask + end, [](unsigned char m) { return m == 0; });
    }
    const double limit = static_cast<double>(static_cast<float>(a.theta)) * 32.0 * static_cast<double>(live32);
    a.fire = static_cast<double>(live) < limit && static_cast<std::int64_t>(live) < a.span &&
             a.span >= kMinReorderSpan;
}

/** @brief A plane the reorder moves: its first byte and element size. */
struct Owned {
    char* data;
    std::size_t bytes;
};

/** @brief The per-sample planes bound to the runner's plans (the mask first), each once. */
std::vector<Owned> owned_planes(const eg_runner_s& runner) {
    std::vector<Owned> out;
    const auto add = [&](const eg_plane_s* plane) {
        const DLTensor& t = plane->tensor->dl_tensor;
        char* data = static_cast<char*>(t.data) + t.byte_offset;
        for (const Owned& o : out)
            if (o.data == data) return;
        out.push_back({ data, static_cast<std::size_t>(t.dtype.bits / 8) });
    };
    const eg_plan_s& first = *runner.plans[0];
    add(first.bindings[first.unit->finish.mask_slot].plane);
    for (const eg_plan_s* p : runner.plans)
        for (std::size_t s = 0; s < p->bindings.size(); ++s) {
            const std::int32_t role = p->unit->slots[s].role;
            if (p->bindings[s].plane != nullptr &&
                (role == EG_ROLE_MUTABLE || role == EG_ROLE_PER_SAMPLE || role == EG_ROLE_VEC_IN ||
                 role == EG_ROLE_MAT_IN || role == EG_ROLE_TERMINATED || role == EG_ROLE_OUT))
                add(p->bindings[s].plane);
        }
    return out;
}

/** @brief Move the live slots of [0, span) to the front (stable), every owned plane and ``perm``
 *  with them; the map becomes the identity over the live prefix. */
void reorder(ActiveState& a, const std::vector<Owned>& owned) {
    const std::size_t sp = static_cast<std::size_t>(a.span);
    const unsigned char* mask = reinterpret_cast<const unsigned char*>(owned[0].data);
    std::vector<std::size_t> dst(sp);
    std::size_t live = 0;
    for (std::size_t i = 0; i < sp; ++i) live += mask[i] == 0;
    std::size_t kept = 0, dropped = 0;
    for (std::size_t i = 0; i < sp; ++i) dst[i] = mask[i] == 0 ? kept++ : live + dropped++;
    std::vector<char> tmp;
    for (const Owned& o : owned) {
        tmp.assign(o.data, o.data + sp * o.bytes);
        for (std::size_t i = 0; i < sp; ++i) std::memcpy(o.data + dst[i] * o.bytes, tmp.data() + i * o.bytes, o.bytes);
    }
    const std::vector<std::int32_t> old(a.perm.begin(), a.perm.begin() + static_cast<std::ptrdiff_t>(sp));
    for (std::size_t i = 0; i < sp; ++i) a.perm[dst[i]] = old[i];
    for (std::size_t s = 0; s < sp; ++s) a.inv[static_cast<std::size_t>(a.perm[s])] = static_cast<std::int32_t>(s);
    for (std::size_t i = 0; i < live; ++i) a.map[i] = static_cast<std::int32_t>(i);
    a.count = static_cast<std::uint32_t>(live);
    a.span = static_cast<std::int64_t>(live);
    a.fire = false;
    ++a.reorders;
}

/** @brief Put every owned plane back in sample order, ``perm``/``inv`` back to the identity, and recompute the map. */
void restore(ActiveState& a, const std::vector<Owned>& owned) {
    const std::size_t n = static_cast<std::size_t>(a.n);
    std::vector<char> tmp;
    for (const Owned& o : owned) {
        tmp.assign(o.data, o.data + n * o.bytes);
        for (std::size_t s = 0; s < n; ++s)
            std::memcpy(o.data + static_cast<std::size_t>(a.perm[s]) * o.bytes, tmp.data() + s * o.bytes, o.bytes);
    }
    for (std::size_t i = 0; i < n; ++i) a.perm[i] = a.inv[i] = static_cast<std::int32_t>(i);
    a.span = a.n;
    compact(a, reinterpret_cast<const unsigned char*>(owned[0].data));
    a.fire = false;
    a.restored_at = a.reorders;
}

/** @brief The loop of an active set (``_host_loop.py``, the fixed-steps branch): per iteration up to
 *  ``every_launches`` launches over the live prefix, then a compaction when the finished count moved
 *  since the last one, then the reorder when the compaction fired it; the planes land back in sample
 *  order at the end. Counts iterations, compactions and reorders of this run. */
void run_active(eg_runner_s& runner, int threads, std::int64_t& iterations, std::int64_t& compactions,
                std::int64_t& reorders) {
    ActiveState& a = runner.act;
    eg_plan_s& first = *runner.plans[0];
    const std::int64_t n = a.n;
    const std::vector<Owned> owned = owned_planes(runner);
    const auto* mask = reinterpret_cast<const unsigned char*>(owned[0].data);
    const std::int64_t cap = (runner.max_steps + a.every_steps - 1) / a.every_steps;
    const std::int64_t c0 = a.compactions, r0 = a.reorders;
    while (iterations < cap && static_cast<std::int64_t>(first.finished) != n) {
        for (std::int64_t l = 0; l < a.every_launches; ++l) {
            const std::int64_t live = std::min<std::int64_t>(a.count, n);
            if (live <= 0 || static_cast<std::int64_t>(first.finished) == n) break;
            for (const eg_plan_s* p : runner.plans) host::run_prefix(*p, threads, live);
        }
        if (first.finished != a.done_at_last) {
            compact(a, mask);
            ++a.compactions;
            a.done_at_last = first.finished;
        }
        if (a.reorder && a.fire) reorder(a, owned);
        ++iterations;
    }
    if (a.reorder && a.reorders != a.restored_at) restore(a, owned);
    compactions = a.compactions - c0;
    reorders = a.reorders - r0;
}


/** @brief What an AUTO run reports beyond the iteration count. */
struct AutoOut {
    std::int64_t steps = 0, mapped = 0, word = 0;
};

/** @brief The band pacing of a unit with steps AUTO (``_host_loop.py``'s non-measured branch and
 *  ``_until_done._host_policy``): the first launch takes kAutoK0 steps, each next one the power-of-two
 *  band of the finishing rate, the last cut to the steps left; with an active set each launch is
 *  followed by its compaction (and reorder). The planes equal every other pacing's. */
void run_auto(eg_runner_s& runner, int threads, std::int64_t& iterations, std::int64_t& compactions,
              std::int64_t& reorders, AutoOut& out) {
    eg_plan_s& first = *runner.plans[0];
    const FinishFacts& f = first.unit->finish;
    const std::int64_t n = first.n_samples, cap = runner.max_steps;
    const std::int64_t cap_iterations = (cap + kAutoKMin - 1) / kAutoKMin + 1;
    std::int64_t k = kAutoK0, fin_at = first.finished, done = 0;
    std::int64_t& word = first.steps_word;
    word = std::min(kAutoK0, cap);
    bool go = static_cast<std::int64_t>(first.finished) != n && cap > 0;
    ActiveState& a = runner.act;
    std::vector<Owned> owned;
    const unsigned char* mask = nullptr;
    std::int64_t c0 = 0, r0 = 0;
    if (runner.active) {
        owned = owned_planes(runner);
        mask = reinterpret_cast<const unsigned char*>(owned[0].data);
        c0 = a.compactions;
        r0 = a.reorders;
    }
    while (iterations < cap_iterations && go) {
        for (std::size_t i = 0; i < runner.plans.size(); ++i) {
            eg_plan_s& p = *runner.plans[i];
            p.steps_word = word;
            if (!runner.active) {
                host::run_plan(p, threads);
            } else if (const std::int64_t live = std::min<std::int64_t>(a.count, n); live > 0) {
                host::run_prefix(p, threads, live);
            }
        }
        if (runner.active) {
            if (first.finished != a.done_at_last) {
                compact(a, mask);
                ++a.compactions;
                a.done_at_last = first.finished;
            }
            if (a.reorder && a.fire) reorder(a, owned);
        }
        const std::int64_t fin = first.finished;
        done += word;
        out.mapped += n - fin_at;
        const std::int64_t next = eagle::policy::bandK(kAutoRate, fin - fin_at, n - fin, k, f.steps_max, kAutoKMin);
        fin_at = fin;
        k = next;
        const std::int64_t left = cap - done;
        go = fin != n && left > 0;
        if (go) word = std::min(next, left);
        ++iterations;
    }
    if (runner.active && a.reorder && a.reorders != a.restored_at) restore(a, owned);
    if (runner.active) {
        compactions = a.compactions - c0;
        reorders = a.reorders - r0;
    }
    out.steps = done;
    out.word = word;
}

}  // namespace

/** @brief eg_runner_new: check the request and keep a reference to each plan. */
void runner_new(eg_ctx& ctx, const eg_runner_desc* desc, eg_runner** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_runner_new: out is NULL");
    *out = nullptr;
    const eg_runner_desc d = read_input(desc, "eg_runner_desc");
    check_desc(d);
    check_plans(ctx, d);
    const bool active = check_active(d);
    auto* runner = new eg_runner_s;
    try {
        runner->plans.assign(d.plans, d.plans + d.n_plans);
        runner->max_steps = d.max_steps;
        if (active) {
            const std::int64_t k = d.plans[0]->unit->finish.steps, n = d.plans[0]->n_samples;
            ActiveState& a = runner->act;
            a.n = n;
            a.every_steps = k == EG_STEPS_AUTO ? kAutoKMin
                          : d.every != 0 ? d.every : k * ((kDefaultEvery + k - 1) / k);
            a.every_launches = k == EG_STEPS_AUTO ? 1 : a.every_steps / k;
            a.map.resize(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i) a.map[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
            a.count = static_cast<std::uint32_t>(n);
            a.reorder = (d.flags & EG_RUN_REORDER) != 0 && n >= kMinReorderSpan;
            a.theta = d.reorder_theta;
            if (a.reorder) {
                a.perm.resize(static_cast<std::size_t>(n));
                a.inv.resize(static_cast<std::size_t>(n));
                for (std::int64_t i = 0; i < n; ++i) a.perm[static_cast<std::size_t>(i)] = a.inv[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
                a.span = n;
            }
            runner->active = true;
        }
        ctx.runners.insert(runner);
    } catch (...) {
        delete runner;
        throw;
    }
    for (eg_plan_s* p : runner->plans) {
        ++p->refs;
        if (active) {
            p->active_owner = runner;
            bind_raw(*p, p->unit->active_map_slot, runner->act.map.data(), static_cast<std::uint64_t>(runner->act.n));
            bind_raw(*p, p->unit->active_count_slot, &runner->act.count, 1);
        }
    }
    *out = runner;
}

/** @brief eg_runner_free: drop the runner's plan references. */
void runner_free(eg_ctx& ctx, eg_runner_s* runner) {
    ctx.runners.erase(runner);
    for (eg_plan_s* p : runner->plans) {
        if (p->active_owner == runner) p->active_owner = nullptr;
        plan_release(ctx, p);
    }
    delete runner;
}

/** @brief Drop one reference to @p runner (the caller's or a graph's); the last frees it. */
void runner_release(eg_ctx& ctx, eg_runner_s* runner) {
    if (--runner->refs == 0) runner_free(ctx, runner);
}

/** @brief eg_run: check the report, then run the loop and fill it. */
void run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report) {
    check_handle(ctx.runners, runner, "runner");
    if (ctx.rec == nullptr) {                   // a recorded run fills no report
        if (report == nullptr) refuse(EG_BAD_REQUEST, "eg_run: report is NULL");
        if (report->struct_size < sizeof(std::uint32_t))
            refuse(EG_BAD_REQUEST, "eg_run_report.struct_size " + std::to_string(report->struct_size) +
                   " is too small");
    }
    for (const eg_plan_s* p : runner->plans) check_bound(*p);
    if (!record_run(ctx, runner)) run_loop(ctx, runner, report);
}

/** @brief Re-sync the counter to the mask, launch the step until done or the cap, and fill
 *  @p report when given (``_until_done.Runner._run`` and ``_host_loop._run_host_loop``, plain loop). */
void run_loop(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report) {
    eg_plan_s& first = *runner->plans[0];
    const FinishFacts& f = first.unit->finish;
    const std::int64_t n = first.n_samples;
    const DLTensor& t = first.bindings[f.mask_slot].plane->tensor->dl_tensor;
    const auto* mask = static_cast<const unsigned char*>(t.data) + t.byte_offset;
    first.finished = static_cast<std::uint32_t>(std::count_if(mask, mask + n, [](unsigned char m) { return m != 0; }));
    const int threads = host::team_threads(ctx);
    std::int64_t launches = 0, compactions = 0, reorders = 0;
    const std::int64_t per_iteration = runner->active ? runner->act.every_steps : f.steps;   // not read with AUTO
    AutoOut aut;
    const bool automatic = f.steps == EG_STEPS_AUTO;
    if (automatic) {
        run_auto(*runner, threads, launches, compactions, reorders, aut);
    } else if (runner->active) {
        run_active(*runner, threads, launches, compactions, reorders);
    } else {
        const std::int64_t cap = runner->max_steps / f.steps + (runner->max_steps % f.steps != 0);
        while (launches < cap && static_cast<std::int64_t>(first.finished) != n) {
            for (const eg_plan_s* p : runner->plans) host::run_plan(*p, threads);
            ++launches;
        }
    }
    if (report == nullptr) return;
    eg_run_report r{};
    r.flags = (static_cast<std::int64_t>(first.finished) == n ? EG_REPORT_DONE : 0u) |
              ((automatic ? launches == 0 || static_cast<std::int64_t>(first.finished) != n || aut.word == 1
                          : per_iteration == 1) ? EG_REPORT_EXACT_STEPS : 0u);
    r.mode = EG_MODE_HOST;
    r.n = n;
    r.finished = first.finished;
    r.launches = launches;
    r.steps = automatic ? aut.steps : launches * per_iteration;
    r.compactions = compactions;
    r.reorders = reorders;
    r.mapped = automatic && runner->active ? aut.mapped : launches * n;
    const std::uint32_t size = std::min<std::uint32_t>(report->struct_size, sizeof(r));
    r.struct_size = size;
    std::memcpy(static_cast<void*>(report), static_cast<const void*>(&r), size);
}

}  // namespace door
}  // namespace eagle
