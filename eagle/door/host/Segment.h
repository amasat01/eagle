// SPDX-License-Identifier: Apache-2.0

// A SEGMENTED dispatch group on the host: the samples are sorted by class (a stable counting sort by (live, class)), each branch unit runs over
// the first run_len[j] threads of the sorted set and reads its own run from the offsets plane, and a dynamic selector's stale samples are then
// swept by the fused unit. The caller's per-sample planes are never moved: the units run over sorted copies, and the planes a unit writes are
// scattered back, so the caller sees sample order after every launch. What the sort and the stale count keep between launches lives in the plan.
#pragma once

#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/exec/HostTeam.h"

#include <algorithm>
#include <numeric>

namespace eagle {
namespace door {
namespace host {

/** @brief The stale samples a dynamic selector may leave before the batch is sorted again: this fraction of the live ones. */
constexpr double kStaleMax = 0.05;

namespace seg {

/** @brief One caller plane the branch units read in sorted order: a copy of its rows in the order of the sort, and a binding repointed at it. */
struct Owned {
    std::int32_t slot = -1;
    char* data = nullptr;                     ///< the caller's plane
    std::size_t elem = 0, rows = 0;
    bool written = false;
    std::vector<char> buf;                    ///< rows * n elements, in the order of the sort
    eg_plan_s::Binding bind;
};

inline bool live_at(const Owned* mask, std::size_t slot, std::size_t n) {
    if (mask == nullptr) return true;
    const char* p = mask->buf.data() + slot * mask->elem;
    for (std::size_t b = 0; b < mask->elem; ++b)
        if (p[b] != 0) return false;
    (void)n;
    return true;
}

/** @brief Move every row of @p o so that slot s holds what slot order[s] held. */
inline void reorder(Owned& o, const std::vector<std::int32_t>& order, std::size_t n) {
    std::vector<char> old = o.buf;
    for (std::size_t r = 0; r < o.rows; ++r)
        for (std::size_t s = 0; s < n; ++s)
            std::memcpy(o.buf.data() + (r * n + s) * o.elem, old.data() + (r * n + static_cast<std::size_t>(order[s])) * o.elem, o.elem);
}

/** @brief The planes the units read in sorted order: every per-sample plane of the branch and classify units of the group (the class and stale planes
 *  and the sweep's own are slot-indexed scratch, not moved), and the mask. The binding of each is a copy repointed at its sorted rows. */
inline std::vector<Owned> collect(const eg_plan_s& plan, bool with_classify, bool device = false) {
    const SegFacts& g = plan.unit->seg;
    const std::size_t n = static_cast<std::size_t>(plan.n_samples);
    std::vector<Owned> owned;
    std::vector<std::int32_t> seen;
    const auto add = [&](std::int32_t s, bool written) {
        if (s < 0 || std::find(seen.begin(), seen.end(), s) != seen.end()) return;
        const eg_plane_s* p = plan.bindings[static_cast<std::size_t>(s)].plane;
        if (p == nullptr) return;
        const DLTensor& t = p->tensor->dl_tensor;
        if (t.ndim < 1 || static_cast<std::size_t>(t.shape[t.ndim - 1]) != n) return;
        seen.push_back(s);
        Owned o;
        o.slot = s;
        o.data = static_cast<char*>(t.data) + t.byte_offset;
        o.elem = t.dtype.bits / 8;
        o.rows = n == 0 ? 0 : static_cast<std::size_t>(elements(t)) / n;
        o.written = written;
        if (!device) o.buf.resize(o.rows * n * o.elem);    // a device plan's copies are the caller's to make (device memory)
        o.bind = plan.bindings[static_cast<std::size_t>(s)];
        owned.push_back(std::move(o));
    };
    add(g.mask_slot, true);
    const std::size_t readers = g.branches.size() + (with_classify ? 1 : 0);
    for (std::size_t mi = 0; mi < readers; ++mi) {
        const eg_unit_s& m = mi < g.branches.size() ? *g.branches[mi] : *g.classify;
        for (std::size_t j = 0; j < m.slots.size(); ++j) {
            const SlotFacts& sl = m.slots[j];
            const std::int32_t s = g.join[mi][j];
            if (s < 0 || s == g.class_slot || s == g.stale_slot || sl.gathered) continue;
            const bool per_sample = sl.role == EG_ROLE_PER_SAMPLE || sl.role == EG_ROLE_VEC_IN || sl.role == EG_ROLE_MAT_IN ||
                                    sl.role == EG_ROLE_OUT || sl.role == EG_ROLE_MUTABLE;
            if (per_sample) add(s, sl.role == EG_ROLE_OUT || sl.role == EG_ROLE_MUTABLE);
        }
    }
    for (Owned& o : owned) {    // the sorted copies exist before any pointer into them is taken
        if (device) break;
        const SlotFacts& sf = plan.unit->slots[static_cast<std::size_t>(o.slot)];
        if (sf.mirror) o.bind.mirror.data_ = reinterpret_cast<double*>(o.buf.data());
        else o.bind.handle.data = o.buf.data();
    }
    return owned;
}

/** @brief Gather the rows of @p o into the order of @p perm (slot -> sample; empty: the identity). */
inline void gather(Owned& o, const std::vector<std::int32_t>& perm, std::size_t n) {
    for (std::size_t r = 0; r < o.rows; ++r)
        for (std::size_t s = 0; s < n; ++s) {
            const std::size_t from = perm.empty() ? s : static_cast<std::size_t>(perm[s]);
            std::memcpy(o.buf.data() + (r * n + s) * o.elem, o.data + (r * n + from) * o.elem, o.elem);
        }
}

/** @brief The argument list of unit @p m (member @p mi): its slots joined to the plan's, a sorted copy where the plane is owned. */
inline std::vector<void*> params_of(const eg_plan_s& plan, std::size_t mi, const eg_unit_s& m, std::vector<Owned>& owned, std::uint32_t* n32) {
    const SegFacts& g = plan.unit->seg;
    const std::size_t arity = &m == plan.unit ? g.own_slots : m.slots.size();    // the planned unit has grown the group's extra slots after its own
    std::vector<void*> params(arity);
    for (std::size_t j = 0; j < arity; ++j) {
        const std::int32_t s = g.join[mi][j];
        void* arg = n32;
        if (s == SegFacts::kMap) arg = &plan.seg.map_h;
        else if (s == SegFacts::kCount) arg = &plan.seg.count_h;
        else if (s >= 0) {
            arg = plan.params[static_cast<std::size_t>(s)];
            for (Owned& o : owned)
                if (o.slot == s) arg = plan.unit->slots[static_cast<std::size_t>(s)].mirror ? static_cast<void*>(&o.bind.mirror) : static_cast<void*>(&o.bind.handle);
        }
        params[j] = arg;
    }
    return params;
}

/** @brief Run unit @p m (member @p mi) over the first @p count threads. A member that is one plain stage of itself runs that stage over its own
 *  scratch planes: the stage slots joined at load to the member's source slots (@p params) and to the planes the plan made for the member. */
inline std::uint64_t run_unit(const eg_plan_s& plan, std::size_t mi, const eg_unit_s& m, std::vector<void*>& params, std::int64_t count, int threads) {
    if (count <= 0) return 0;
    const exec::Partition part{ 0, count, plan.n_samples };
    std::vector<void*> stage_params;
    std::vector<plugin::ScalarHandle> handles;
    std::uint32_t n32 = static_cast<std::uint32_t>(plan.n_samples);
    if (!m.stages.empty()) {
        const StageFacts& st = m.stages[0];
        const std::size_t n_slots = st.unit->slots.size();
        stage_params.resize(n_slots);
        handles.resize(n_slots);
        for (std::size_t j = 0; j < n_slots; ++j) {
            if (st.source_of[j] >= 0) {
                stage_params[j] = params[static_cast<std::size_t>(st.source_of[j])];
            } else if (st.plane_of[j] >= 0) {
                const DLTensor& d = plan.segteam[mi][static_cast<std::size_t>(st.plane_of[j])]->tensor->dl_tensor;
                const int role = st.unit->slots[j].role;
                plugin::ScalarHandle& h = handles[j];
                h.data = static_cast<char*>(d.data) + d.byte_offset;
                h.samples = static_cast<std::uint64_t>(role == EG_ROLE_LOOKUP ? elements(d) : role == EG_ROLE_WIDE_IN || role == EG_ROLE_WIDE_OUT ? d.shape[d.ndim - 1] : plan.n_samples);
                h.stride = 1;
                h.deviceType = plugin::kEagleAbiDeviceCPU;
                h.deviceId = 0;
                stage_params[j] = &h;
            } else {
                stage_params[j] = &n32;
            }
        }
    }
    void** args = m.stages.empty() ? params.data() : stage_params.data();
    if (plan.structure == EG_STRUCTURE_HOST_SERIAL) {
        exec::HostTeam::run_serial(m.host_entry, args, part);
        return 1;
    }
    return exec::HostTeam::run_team(m.host_entry, args, part, threads);
}

}  // namespace seg

/** @brief One launch of a SEGMENTED plan (see the header of this file). Returns the tiles run. */
inline std::uint64_t run_segmented(const eg_plan_s& plan, int threads) {
    const SegFacts& g = plan.unit->seg;
    SegState& S = plan.seg;
    const std::size_t n = static_cast<std::size_t>(plan.n_samples), k = static_cast<std::size_t>(g.k);
    std::uint64_t tiles = 0;
    if (n == 0) return 0;
    std::uint32_t n32 = static_cast<std::uint32_t>(plan.n_samples);
    const std::string at = "unit '" + plan.unit->name + "': dispatch group '" + g.name + "': ";
    if (!g.dynamic) {    // the classes of every sample, in sample order, from the classify unit
        std::vector<seg::Owned> none;
        std::vector<void*> p = seg::params_of(plan, k, *g.classify, none, &n32);
        tiles += seg::run_unit(plan, k, *g.classify, p, plan.n_samples, threads);
        for (std::size_t i = 0; i < n; ++i)
            if (S.cls[i] < 0 || S.cls[i] >= g.k)
                refuse(EG_BAD_REQUEST, at + "the classify unit wrote class " + std::to_string(S.cls[i]) + " for sample " + std::to_string(i) + ", outside [0, " + std::to_string(g.k) + ")");
        S.perm.clear();
        S.sorted = false;
    }
    std::vector<seg::Owned> owned = seg::collect(plan, !g.dynamic);
    for (seg::Owned& o : owned) seg::gather(o, S.perm, n);    // the arrangement of the last sort (the identity before a sort)
    const seg::Owned* mask = nullptr;
    for (const seg::Owned& o : owned)
        if (o.slot == g.mask_slot) mask = &o;
    const auto live = [&](std::size_t slot) { return seg::live_at(mask, slot, n); };
    std::size_t live_count = 0;
    for (std::size_t s = 0; s < n; ++s) live_count += live(s);
    if (!S.sorted || (g.dynamic && static_cast<double>(S.stale_now) > kStaleMax * static_cast<double>(std::max<std::size_t>(live_count, 1)))) {
        std::vector<std::size_t> start(k + 2, 0);    // the live samples by class, then the dead; stable
        std::vector<std::uint8_t> key(n);
        for (std::size_t s = 0; s < n; ++s) {
            const std::int64_t c = std::clamp<std::int64_t>(S.cls[s], 0, g.k - 1);    // the class plane of the arrangement (a static class is in range already)
            key[s] = live(s) ? static_cast<std::uint8_t>(c) : static_cast<std::uint8_t>(k);
            ++start[static_cast<std::size_t>(key[s]) + 1];
        }
        for (std::size_t c = 1; c < k + 2; ++c) start[c] += start[c - 1];
        for (std::size_t c = 0; c < k; ++c) S.run_len[c] = static_cast<std::int64_t>(start[c + 1] - start[c]);
        for (std::size_t c = 0; c <= k; ++c) S.offsets[c] = static_cast<std::int64_t>(start[c]);
        std::vector<std::int32_t> order(n);
        for (std::size_t s = 0; s < n; ++s) order[start[key[s]]++] = static_cast<std::int32_t>(s);
        for (seg::Owned& o : owned) seg::reorder(o, order, n);
        std::vector<std::int32_t> next(n);
        for (std::size_t s = 0; s < n; ++s) next[s] = S.perm.empty() ? order[s] : S.perm[static_cast<std::size_t>(order[s])];
        S.perm = std::move(next);
        S.sorted = true;
    }
    for (std::size_t j = 0; j < k; ++j) {
        std::vector<void*> p = seg::params_of(plan, j, *g.branches[j], owned, &n32);
        tiles += seg::run_unit(plan, j, *g.branches[j], p, S.run_len[j], threads);
    }
    if (g.dynamic) {    // the samples whose branch moved: the fused unit completes them (a live sample the guarded unit kept, not committed)
        std::size_t stale = 0;
        for (std::size_t s = 0; s < n; ++s)
            if (S.stale[s] != 0 && live(s)) S.map[stale++] = static_cast<std::int32_t>(s);
        S.stale_now = static_cast<std::int64_t>(stale);
        if (stale > 0) {
            S.count = static_cast<std::uint32_t>(stale);
            std::vector<void*> p = seg::params_of(plan, k, *g.sweep, owned, &n32);
            tiles += seg::run_unit(plan, k, *g.sweep, p, static_cast<std::int64_t>(stale), threads);
        }
    }
    for (const seg::Owned& o : owned) {    // back to sample order, the planes the units wrote
        if (!o.written) continue;
        for (std::size_t r = 0; r < o.rows; ++r)
            for (std::size_t s = 0; s < n; ++s)
                std::memcpy(o.data + (r * n + static_cast<std::size_t>(S.perm[s])) * o.elem, o.buf.data() + (r * n + s) * o.elem, o.elem);
    }
    return tiles;
}

}  // namespace host
}  // namespace door
}  // namespace eagle
