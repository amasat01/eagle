// SPDX-License-Identifier: Apache-2.0

// The host team of a door launch: eagle's one process pool with the thread
// count the context states, passed per launch. The door never reads or
// writes the process-wide host thread setting or a thread's own team.
#pragma once

#include "eagle/door/Context.h"
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

/** @brief Run @p plan once over each of its partitions on at most @p threads threads
 *  (HOST_SERIAL: one serial call per partition); returns the tiles run. */
inline std::uint64_t run_plan(const eg_plan_s& plan, int threads) {
    std::uint64_t tiles = 0;
    for (const exec::Partition& part : plan.parts) {
        if (plan.structure == EG_STRUCTURE_HOST_SERIAL) {
            exec::HostTeam::run_serial(plan.unit->host_entry, plan.params.data(), part);
            tiles += part.count > 0 ? 1 : 0;
        } else {
            tiles += exec::HostTeam::run_team(plan.unit->host_entry, plan.params.data(), part, threads);
        }
    }
    return tiles;
}

}  // namespace host
}  // namespace door
}  // namespace eagle
