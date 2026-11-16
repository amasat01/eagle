// SPDX-License-Identifier: Apache-2.0

// The host team of a door launch: eagle's one process pool with the thread
// count the context states, passed per launch. The door never reads or
// writes the process-wide host thread setting or a thread's own team.
#pragma once

#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/door/host/Ops.h"
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
            if (st.source_of[j] >= 0) {
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
            const std::int64_t blocks = (plan.n_samples + per_block - 1) / per_block;
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
    }
    return tiles;
}

/** @brief Run @p plan once over each of its partitions on at most @p threads threads
 *  (HOST_SERIAL: one serial call per partition); returns the tiles run. */
inline std::uint64_t run_plan(const eg_plan_s& plan, int threads) {
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
