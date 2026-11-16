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


/** @brief The mask plane: elements of any width, set when any byte is nonzero. */
struct Mask {
    const char* data;
    std::size_t bytes;
    bool set(std::int64_t i) const {
        const char* e = data + static_cast<std::size_t>(i) * bytes;
        for (std::size_t b = 0; b < bytes; ++b)
            if (e[b] != 0) return true;
        return false;
    }
};

/** @brief Recompute the map and count from the mask; for a reordering set also the trigger:
 *  fire when the live count is below theta * 32 * (32-slot groups holding a live sample),
 *  below the span, and the span is at least the floor (the old host rule, bit for bit). */
void compact(ActiveState& a, const Mask& mask) {
    if (a.group > 1) {      // a group stays live while any of its lanes is: the map holds group indices, the count the lanes launched
        std::uint32_t groups = 0;
        for (std::int64_t g = 0; g < a.n; g += a.group) {
            const std::int64_t end = std::min<std::int64_t>(g + a.group, a.n);
            for (std::int64_t i = g; i < end; ++i)
                if (!mask.set(i)) {
                    a.map[groups++] = static_cast<std::int32_t>(g / a.group);
                    break;
                }
        }
        a.count = static_cast<std::uint32_t>(groups * a.group);
        return;
    }
    std::uint32_t live = 0;
    for (std::int64_t i = 0; i < a.n; ++i)
        if (!mask.set(i)) a.map[live++] = static_cast<std::int32_t>(i);
    a.count = live;
    if (!a.reorder) return;
    std::int64_t live32 = 0;
    for (std::int64_t g = 0; g < a.n; g += 32) {
        const std::int64_t end = std::min<std::int64_t>(g + 32, a.n);
        for (std::int64_t i = g; i < end; ++i)
            if (!mask.set(i)) {
                ++live32;
                break;
            }
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

/** @brief Move the live slots of [0, span) to the front (stable), every owned plane and ``perm``
 *  with them; the map becomes the identity over the live prefix. */
void reorder(ActiveState& a, const std::vector<Owned>& owned) {
    const std::size_t sp = static_cast<std::size_t>(a.span);
    const Mask mask{ owned[0].data, owned[0].bytes };
    std::vector<std::size_t> dst(sp);
    std::size_t live = 0;
    for (std::size_t i = 0; i < sp; ++i) live += !mask.set(static_cast<std::int64_t>(i));
    std::size_t kept = 0, dropped = 0;
    for (std::size_t i = 0; i < sp; ++i) dst[i] = !mask.set(static_cast<std::int64_t>(i)) ? kept++ : live + dropped++;
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
    compact(a, Mask{ owned[0].data, owned[0].bytes });
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
    const Mask mask{ owned[0].data, owned[0].bytes };
    const std::int64_t cap = (runner.max_steps + a.every_steps - 1) / a.every_steps;
    const std::int64_t c0 = a.compactions, r0 = a.reorders;
    while (iterations < cap && static_cast<std::int64_t>(first.finished) != n) {
        for (std::int64_t l = 0; l < a.every_launches; ++l) {
            const std::int64_t live = a.count;
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
    Mask mask{ nullptr, 1 };
    std::int64_t c0 = 0, r0 = 0;
    if (runner.active) {
        owned = owned_planes(runner);
        mask = Mask{ owned[0].data, owned[0].bytes };
        c0 = a.compactions;
        r0 = a.reorders;
    }
    while (iterations < cap_iterations && go) {
        for (std::size_t i = 0; i < runner.plans.size(); ++i) {
            eg_plan_s& p = *runner.plans[i];
            p.steps_word = word;
            if (!runner.active) {
                host::run_plan(p, threads);
            } else if (a.count > 0) {
                host::run_prefix(p, threads, a.count);
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
    check_runner(ctx, d, false);
    const bool active = check_active(d);
    auto* runner = new eg_runner_s;
    try {
        runner->plans.assign(d.plans, d.plans + d.n_plans);
        runner->max_steps = d.max_steps;
        if (active) {
            const std::int64_t k = d.plans[0]->unit->finish.steps, n = d.plans[0]->n_samples;
            ActiveState& a = runner->act;
            a.n = n;
            a.group = d.plans[0]->unit->group;
            a.every_steps = k == EG_STEPS_AUTO ? kAutoKMin
                          : d.every != 0 ? d.every : k * ((kDefaultEvery + k - 1) / k);
            a.every_launches = k == EG_STEPS_AUTO ? 1 : a.every_steps / k;
            a.map.resize(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i) a.map[static_cast<std::size_t>(i)] = static_cast<std::int32_t>(i);
            a.count = static_cast<std::uint32_t>((n + a.group - 1) / a.group * a.group);
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
    if (ctx.rec == nullptr) check_report(report);   // a recorded run fills no report
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
    const Mask mask{ static_cast<const char*>(t.data) + t.byte_offset, static_cast<std::size_t>(t.dtype.bits / 8) };
    std::int64_t set = 0;
    for (std::int64_t i = 0; i < n; ++i) set += mask.set(i);
    first.finished = static_cast<std::uint32_t>(set);
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
    write_report(report, r);
}

}  // namespace door
}  // namespace eagle
