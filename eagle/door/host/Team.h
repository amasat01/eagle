// SPDX-License-Identifier: Apache-2.0

// The host team of a door launch: eagle's one process pool with the thread
// count the context states, passed per launch. The door never reads or
// writes the process-wide host thread setting or a thread's own team.
#pragma once

#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/door/host/Ops.h"
#include "eagle/door/host/Segment.h"
#include "eagle/exec/HostTeam.h"

#include <algorithm>

namespace eagle {
namespace door {
namespace host {

/** @brief The threads a launch of @p ctx uses: the context's cap (eg_host_threads), else
 *  the default budget (``OMP_NUM_THREADS`` read once, else the cores, capped by the cgroup). */
inline int team_threads(const eg_ctx& ctx) {
    if (ctx.host_threads > 0) return ctx.host_threads;
    const int env = exec::detail::omp_num_threads_env();
    return env > 0 ? env : exec::detail::host_default_threads();
}

/** @brief Run @p plan once over the prefix [0, @p live) of its samples (an active set's live slots). */
inline void run_prefix(const eg_plan_s& plan, int threads, std::int64_t live) {
    const exec::Partition part{ 0, live, plan.n_samples };
    if (plan.structure == EG_STRUCTURE_HOST_SERIAL)
        exec::HostTeam::run_serial(plan.unit->host_entry, plan.params.data(), part);
    else
        exec::HostTeam::run_team(plan.unit->host_entry, plan.params.data(), part, threads);
    fold_plan(plan, threads);
}

/** @brief The cells of the plane an ordered scatter stage @p st adds into: the elements of the caller's plane bound to its target slot. */
inline std::int64_t scatter_cells(const eg_plan_s& plan, const StageFacts& st) {
    return elements(plan.bindings[static_cast<std::size_t>(st.of_slot)].plane->tensor->dl_tensor);
}

/** @brief After a launch: an ordered scatter whose status plane holds a flag past element 0 met a key outside its cells that was not the skip key
 *  -1: the adds were dropped, the rest are in the target. Refused by name (the flags are the stage's own, the cells the target's). */
inline void check_scatter_status(const eg_plan_s& plan) {
    for (const StageFacts& st : plan.unit->stages) {
        if (st.grid != EG_GRID_CHUNKS || st.status_plane < 0) continue;
        const DLTensor& d = plan.team[static_cast<std::size_t>(st.status_plane)]->tensor->dl_tensor;
        const std::int32_t* flags = static_cast<const std::int32_t*>(d.data);
        bool outside = false;
        for (std::int64_t i = 1; i < elements(d); ++i) outside = outside || flags[i] != 0;
        if (!outside) continue;
        const std::string& of = plan.unit->slots[static_cast<std::size_t>(st.of_slot)].name;
        refuse(EG_FAULT, plan.unit->name + ": hawk.scatter(" + of + ", contract='ordered') got a key outside [0, " + std::to_string(scatter_cells(plan, st)) +
                         ") that is not the skip key -1; those adds were dropped, the rest are in " + of);
    }
}

/** @brief A split unit: its stages in order over the plan's bindings, each with its own argument list (the source slot or
 *  the team plane each stage slot joined at load). A team stage runs ONE serial call over (0, blocks, n), never tiled;
 *  a plain stage runs as the unit would over the plan's partitions. Planes with a promised fill are refilled first. */
inline std::uint64_t run_split(const eg_plan_s& plan, int threads) {
    for (std::size_t k = 0; k < plan.unit->team_planes.size(); ++k) {
        const TeamPlaneFacts& t = plan.unit->team_planes[k];
        if (!t.fill_promised || t.wide != EG_WIDE_NONE) continue;
        const DLTensor& d = plan.team[k]->tensor->dl_tensor;
        const std::int64_t count = elements(d);
        if (t.dtype == EG_DTYPE_INT32)
            std::fill_n(static_cast<std::int32_t*>(d.data), count, static_cast<std::int32_t>(t.fill));
        else if (plan.unit->scalar_type == EG_SCALAR_FLOAT32)
            std::fill_n(static_cast<float*>(d.data), count, static_cast<float>(t.fill));
        else
            std::fill_n(static_cast<double*>(d.data), count, static_cast<double>(t.fill));
    }
    std::uint64_t tiles = 0;
    for (const StageFacts& st : plan.unit->stages) {
        const std::size_t n_slots = st.unit->slots.size();
        std::vector<void*> params(n_slots);
        std::vector<plugin::ScalarHandle> handles(n_slots);
        std::uint32_t n32 = static_cast<std::uint32_t>(plan.n_samples);
        for (std::size_t j = 0; j < n_slots; ++j) {
            if (st.row_of[j] >= 0 || st.comp_of[j] >= 0) {    // the rows this stage writes into a cell: eagle's
                const std::size_t c = static_cast<std::size_t>(st.row_of[j] >= 0 ? st.row_of[j] : st.comp_of[j]);
                const DLTensor& d = (st.row_of[j] >= 0 ? plan.rows[c] : plan.comps[c])->tensor->dl_tensor;
                plugin::ScalarHandle& h = handles[j];
                h.data = static_cast<char*>(d.data) + d.byte_offset;
                h.samples = static_cast<std::uint64_t>(plan.n_samples);
                h.stride = 1;
                h.deviceType = plugin::kEagleAbiDeviceCPU;
                h.deviceId = 0;
                params[j] = &h;
            } else if (st.read_of[j] >= 0) {    // a cell an earlier stage folded: the caller's plane or eagle's, read as a table
                const std::size_t c = static_cast<std::size_t>(st.read_of[j]);
                const eg_plane_s* p = plan.unit->split_cells[c].source >= 0 ? plan.bindings[static_cast<std::size_t>(plan.unit->split_cells[c].source)].plane : plan.outs[c];
                const DLTensor& d = p->tensor->dl_tensor;
                plugin::ScalarHandle& h = handles[j];
                h.data = static_cast<char*>(d.data) + d.byte_offset;
                h.samples = static_cast<std::uint64_t>(elements(d));
                h.stride = 1;
                h.deviceType = plugin::kEagleAbiDeviceCPU;
                h.deviceId = 0;
                params[j] = &h;
            } else if (st.source_of[j] >= 0) {
                params[j] = plan.params[static_cast<std::size_t>(st.source_of[j])];
            } else if (st.plane_of[j] >= 0) {
                const DLTensor& d = plan.team[static_cast<std::size_t>(st.plane_of[j])]->tensor->dl_tensor;
                const int role = st.unit->slots[j].role;
                plugin::ScalarHandle& h = handles[j];
                h.data = static_cast<char*>(d.data) + d.byte_offset;
                h.samples = static_cast<std::uint64_t>(role == EG_ROLE_LOOKUP ? elements(d)
                                                       : role == EG_ROLE_WIDE_IN || role == EG_ROLE_WIDE_OUT ? d.shape[d.ndim - 1] : plan.n_samples);
                h.stride = 1;
                h.deviceType = plugin::kEagleAbiDeviceCPU;
                h.deviceId = 0;
                params[j] = &h;
            } else {
                params[j] = &n32;
            }
        }
        if (st.team) {
            const std::int64_t per_block = static_cast<std::int64_t>(st.lanes) * st.warps;
            const std::int64_t blocks = st.grid == EG_GRID_KEYS
                ? key_count(*plan.unit, plan.unit->keyed[static_cast<std::size_t>(st.keyed)], plan.n_samples) * st.groups
                : st.grid == EG_GRID_TARGETS ? scatter_cells(plan, st)
                : st.grid == EG_GRID_CHUNKS ? chunks_of(plan.n_samples, scatter_cells(plan, st), plan.unit->scatter_chunks)
                : (plan.n_samples + per_block - 1) / per_block;
            if (blocks > 0) {
                exec::HostTeam::run_serial(st.unit->host_entry, params.data(), exec::Partition{ 0, blocks, plan.n_samples });
                ++tiles;
            }
            continue;
        }
        for (const exec::Partition& part : plan.parts) {
            if (plan.structure == EG_STRUCTURE_HOST_SERIAL) {
                exec::HostTeam::run_serial(st.unit->host_entry, params.data(), part);
                tiles += part.count > 0 ? 1 : 0;
            } else {
                tiles += exec::HostTeam::run_team(st.unit->host_entry, params.data(), part, threads);
            }
        }
        if (plan.n_samples == 0) continue;
        for (std::size_t c = 0; c < plan.unit->split_cells.size(); ++c) {    // this stage's cell rows folded into their cells before the next stage reads them
            const SplitCell& sc = plan.unit->split_cells[c];
            if (sc.stage != static_cast<std::int32_t>(&st - plan.unit->stages.data())) continue;
            eg_reduce_desc rd{};
            rd.struct_size = sizeof(rd);
            rd.in = plan.rows[c];
            rd.out = sc.source >= 0 ? plan.bindings[static_cast<std::size_t>(sc.source)].plane : plan.outs[c];
            rd.comp = plan.comps[c];
            rd.op = sc.op;
            reduce_exec(reduce_check(*plan.ctx, rd), threads);
        }
    }
    check_scatter_status(plan);
    return tiles;
}

/** @brief Run @p plan once over each of its partitions on at most @p threads threads
 *  (HOST_SERIAL: one serial call per partition); returns the tiles run. */
inline std::uint64_t run_plan(const eg_plan_s& plan, int threads) {
    std::size_t group = 0;
    for (const DispatchFacts& f : plan.unit->dispatches) {    // an AUTO group's counters eagle made: zero at every launch (nothing reads them)
        if (f.strategy != EG_DISPATCH_AUTO) continue;
        const eg_plane_s* mix = plan.mix[group++];
        if (plan.bindings[static_cast<std::size_t>(f.mix)].plane == nullptr)
            std::memset(static_cast<char*>(mix->tensor->dl_tensor.data) + mix->tensor->dl_tensor.byte_offset, 0, static_cast<std::size_t>(f.k) * sizeof(std::uint32_t));
    }
    if (plan.unit->seg.present) return run_segmented(plan, threads);
    if (!plan.unit->stages.empty()) return run_split(plan, threads);
    std::uint64_t tiles = 0;
    if (plan.unit->index_count_slot >= 0) {    // an index set: lanes [0, count), read now; the entry maps them itself
        const eg_plane_s& c = *plan.bindings[plan.unit->index_count_slot].plane;
        const std::uint64_t live = *reinterpret_cast<const std::uint32_t*>(static_cast<const char*>(c.tensor->dl_tensor.data) +
                                                                          c.tensor->dl_tensor.byte_offset);
        check_index_live(plan, live);
        if (live > 0) run_prefix(plan, threads, static_cast<std::int64_t>(live));
        return live > 0 ? 1 : 0;
    }
    for (const exec::Partition& part : plan.parts) {
        if (plan.structure == EG_STRUCTURE_HOST_SERIAL) {
            exec::HostTeam::run_serial(plan.unit->host_entry, plan.params.data(), part);
            tiles += part.count > 0 ? 1 : 0;
        } else {
            tiles += exec::HostTeam::run_team(plan.unit->host_entry, plan.params.data(), part, threads);
        }
    }
    fold_plan(plan, threads);
    return tiles;
}

}  // namespace host
}  // namespace door
}  // namespace eagle
