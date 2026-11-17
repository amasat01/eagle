// SPDX-License-Identifier: Apache-2.0

// Runners on the host: the run-until-done loop over bound host plans. Each
// loop iteration launches the step's plans in order on the context's team;
// the loop stops when the first plan's finished counter reaches the sample
// count or after ceil(max_steps / K) iterations. Units that read an active set
// compact it every ``every`` steps and may reorder the bound planes
// (EG_RUN_REORDER). An EG_RUN_RECORD runner runs host/Reverse.cpp's recording forward instead.
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/door/host/Team.h"
#include "eagle/policy/Launch.h"

#include <algorithm>
#include <limits>

namespace eagle {
namespace door {

namespace {

constexpr std::int64_t kAutoKMin = 8;            ///< AUTO: the fewest steps of a launch but the last, and the compaction cadence
constexpr std::int64_t kAutoK0 = 16;             ///< AUTO: the steps of the first launch
constexpr double kAutoRate = 8.0;                ///< AUTO: a compaction's cost in full-live steps on the host

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

void active_compact(ActiveState& a, const char* mask, std::size_t bytes) { compact(a, Mask{ mask, bytes }); }
void active_reorder(ActiveState& a, const std::vector<Owned>& owned) { reorder(a, owned); }
void active_restore(ActiveState& a, const std::vector<Owned>& owned) { restore(a, owned); }

/** @brief The counters of an active set back to a runner that has not run. */
void active_counters(ActiveState& a) {
    a.done_at_last = 0;
    a.compactions = a.reorders = a.restored_at = 0;
    a.fire = false;
}

/** @brief The host active set of ``a.n`` samples as a new runner holds it: every sample live, the map and (reordering) the
 *  permutation the identity, the counters zero. */
void active_identity(ActiveState& a) {
    const std::size_t n = static_cast<std::size_t>(a.n);
    a.map.resize(n);
    for (std::size_t i = 0; i < n; ++i) a.map[i] = static_cast<std::int32_t>(i);
    a.count = static_cast<std::uint32_t>((a.n + a.group - 1) / a.group * a.group);
    if (a.reorder) {
        a.perm.resize(n);
        a.inv.resize(n);
        for (std::size_t i = 0; i < n; ++i) a.perm[i] = a.inv[i] = static_cast<std::int32_t>(i);
        a.span = a.n;
    }
    active_counters(a);
}

/** @brief Zero @p bytes at @p data, on the context's stream for device memory. */
void zero_memory(eg_ctx& ctx, bool device, void* data, std::size_t bytes) {
    if (device) device_zero(ctx, data, bytes);
    else std::memset(data, 0, bytes);
}

/** @brief eg_runner_new: check the request and keep a reference to each plan. */
void runner_new(eg_ctx& ctx, const eg_runner_desc* desc, eg_runner** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_runner_new: out is NULL");
    *out = nullptr;
    eg_runner_desc d = read_input(desc, "eg_runner_desc");
    KeptDerived kept;
    check_runner(ctx, d, false, kept);
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
            a.every_steps = k == EG_STEPS_AUTO ? kAutoKMin : active_every_steps(d);
            a.every_launches = k == EG_STEPS_AUTO ? 1 : a.every_steps / k;
            a.reorder = (d.flags & EG_RUN_REORDER) != 0 && n >= kMinReorderSpan;
            a.theta = d.reorder_theta;
            active_identity(a);
            runner->active = true;
        }
        if ((d.flags & EG_RUN_RECORD) || d.events == EG_EVENTS_REFINE) record_new(ctx, *runner, d);
        ctx.runners.insert(runner);
    } catch (...) {
        record_free(ctx, *runner);
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

/** @brief A plan's SEGMENTED state back to a new plan's: sorts again at its next launch, from zero classes and stale marks, with no kept arrangement. */
void seg_reset(eg_ctx& ctx, eg_plan_s& p) {
    p.finished = 0;
    p.seg.sorted = false;
    p.seg.stale_now = 0;
    std::fill(p.seg.run_len.begin(), p.seg.run_len.end(), 0);
    p.seg.perm.clear();    // the planes are in sample order between launches: the arrangement starts as the identity
    std::fill(p.seg.cls.begin(), p.seg.cls.end(), 0);    // the classes and stale marks a first sort reads are zero, as in a new plan
    std::fill(p.seg.stale.begin(), p.seg.stale.end(), 0);
    std::fill(p.seg.sortkey.begin(), p.seg.sortkey.end(), 0);
    for (std::size_t i = 1; i < p.segdev.size() && i <= 2; ++i) {    // DEVICE: the class plane and, for a dynamic selector, the stale mask
        const DLTensor& st = p.segdev[i]->tensor->dl_tensor;
        zero_memory(ctx, true, static_cast<char*>(st.data) + st.byte_offset, static_cast<std::size_t>(elements(st)) * (st.dtype.bits / 8));
    }
}

/** @brief eg_runner_reset: the state eg_runner_new left, so the next eg_run starts the batch over. */
void runner_reset(eg_ctx& ctx, eg_runner_s* runner) {
    check_handle(ctx.runners, runner, "runner");
    if (ctx.rec != nullptr || ctx.dev_rec != nullptr)
        refuse(EG_BAD_REQUEST, "eg_runner_reset while recording (eg_record_begin .. eg_record_end): reset between runs");
    for (const eg_plan_s* p : runner->plans) check_bound(*p);
    eg_plan_s& first = *runner->plans[0];
    const bool device = first.structure == EG_STRUCTURE_DEVICE;
    ActiveState& a = runner->act;
    const DLTensor& t = first.bindings[static_cast<std::size_t>(first.unit->finish.mask_slot)].plane->tensor->dl_tensor;
    if (runner->active && a.reorder && a.reorders != a.restored_at) {    // planes of a run that left them permuted: sample order first
        const std::vector<Owned> owned = owned_planes(*runner);
        if (runner->dev.ctx != nullptr) device_active_restore(ctx, *runner, owned[0].data, owned);
        else restore(a, owned);
    }
    zero_memory(ctx, device, static_cast<char*>(t.data) + t.byte_offset, static_cast<std::size_t>(elements(t)) * (t.dtype.bits / 8));
    for (eg_plan_s* p : runner->plans) {
        p->finished = 0;
        p->steps_word = 0;
        if (p->device_counter != 0) device_zero(ctx, reinterpret_cast<void*>(p->device_counter), sizeof(unsigned));
        if (p->device_steps != 0) device_zero(ctx, reinterpret_cast<void*>(p->device_steps), sizeof(std::int64_t));
        for (const eg_plane_s* m : p->mix) {    // an AUTO group's mix counters
            const DLTensor& mt = m->tensor->dl_tensor;
            zero_memory(ctx, device, static_cast<char*>(mt.data) + mt.byte_offset, static_cast<std::size_t>(elements(mt)) * (mt.dtype.bits / 8));
        }
        seg_reset(ctx, *p);
    }
    if (runner->active) {
        if (runner->dev.ctx != nullptr) active_counters(a);
        else active_identity(a);
    }
    if (runner->dev.ctx != nullptr) device_runner_initial(ctx, *runner);
    record_reset(*runner);
}

/** @brief eg_runner_free: drop the runner's plan references. */
void runner_free(eg_ctx& ctx, eg_runner_s* runner) {
    ctx.runners.erase(runner);
    look_free(ctx, *runner);
    record_free(ctx, *runner);
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
    if (runner->record != nullptr) {
        if (ctx.rec != nullptr)
            refuse(EG_NOT_WRITTEN, "eg_run of an EG_RUN_RECORD runner inside a recording is not written in this build of the door");
        record_forward(ctx, *runner, report);
        return;
    }
    if (!record_run(ctx, runner)) run_loop(ctx, runner, report);
}

void runner_figures(eg_ctx& ctx, const eg_runner_s* runner, struct eg_runner_figures* out) {
    check_handle(ctx.runners, runner, "runner");
    const DeviceRun& r = runner->dev;
    struct eg_runner_figures f{};
    f.mode = r.ctx != nullptr ? r.mode : EG_MODE_HOST;
    f.steps_reverse = runner->steps_reverse;
    f.block = r.block > 0 ? static_cast<std::int32_t>(r.block) : -1;
    f.grid = r.block > 0 ? static_cast<std::int64_t>(r.grid) : -1;
    f.chunk = f.mode == EG_MODE_PERSIST ? static_cast<std::int32_t>(r.chunk) : -1;
    write_output(out, f, "eg_runner_figures");
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
