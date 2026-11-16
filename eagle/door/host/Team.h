// SPDX-License-Identifier: Apache-2.0

// The host team of a door launch: eagle's one process pool with the thread
// count the context states, passed per launch. The door never reads or
// writes the process-wide host thread setting or a thread's own team.
#pragma once

#include "eagle/door/Context.h"
#include "eagle/door/host/Ops.h"
#include "eagle/exec/HostTeam.h"

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

/** @brief Run @p plan once over each of its partitions on at most @p threads threads
 *  (HOST_SERIAL: one serial call per partition); returns the tiles run. */
inline std::uint64_t run_plan(const eg_plan_s& plan, int threads) {
    std::uint64_t tiles = 0;
    if (plan.unit->index_count_slot >= 0) {    // an index set: lanes [0, count), read now; the entry maps them itself
        const eg_plane_s& c = *plan.bindings[plan.unit->index_count_slot].plane;
        const std::uint64_t live = *reinterpret_cast<const std::uint32_t*>(static_cast<const char*>(c.tensor->dl_tensor.data) +
                                                                          c.tensor->dl_tensor.byte_offset);
        std::uint64_t limit = static_cast<std::uint64_t>(plan.n_samples);   // a prefix set: up to the sample count
        if (plan.unit->index_map_slot >= 0) {
            const DLTensor& m = plan.bindings[plan.unit->index_map_slot].plane->tensor->dl_tensor;
            limit = static_cast<std::uint64_t>(m.shape[0]);
        }
        if (live > limit)
            refuse(EG_BAD_REQUEST, "the index count " + std::to_string(live) + " is above the " +
                   (plan.unit->index_map_slot >= 0 ? "map's " : "sample count's ") + std::to_string(limit) + " entries");
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
