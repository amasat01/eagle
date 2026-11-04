// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The host table of a graph: its CPU stages frozen as plain data, run from
// inside another runtime's OpenMP region (`plugin/graph_handoff.h`).
#pragma once

#include "eagle/exec/HostSchedule.h"
#include "plugin/graph_handoff.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace eagle {
namespace exec {

/** @brief Count of binds and rebuilds of a graph; a table made at one value is stale at another. */
using HostVersion = std::shared_ptr<std::atomic<std::int64_t>>;

/**
 * @brief Run the nodes in order over the CALLER's team: the same loop as
 *        `runHostStage`, with the tile loop left orphaned so it binds to the
 *        region the caller is already in. Every thread of that region must call.
 *        Outside any region this is `runHostStage`.
 */
inline void runHostStageShared(const std::vector<HostBranch>& branches)
{
#ifdef _OPENMP
    if (omp_get_level() > 0) {
        const std::int64_t tile = static_cast<std::int64_t>(HostTeam::tile_size(0));
        for (const HostBranch& branch : branches)
            for (const HostBranchNode& node : branch) {
                if (node.entry == nullptr)
                    throw std::runtime_error("runHostStageShared: null host entry");
                for (const Partition& part : node.parts) {
                    if (part.count <= 0) continue;
                    const std::int64_t ntiles = (part.count + tile - 1) / tile;
#pragma omp for schedule(dynamic)
                    for (std::int64_t t = 0; t < ntiles; ++t) {
                        const std::int64_t base = part.base + t * tile;
                        const std::int64_t count = (t + 1 == ntiles)
                            ? (part.base + part.count - base) : tile;
                        node.entry(node.params.data(), base, count, part.nSamples);
                    }
                }
            }
        return;
    }
#endif
    runHostStage(branches);
}

/** @brief The table behind an `eagle_host_table*`: stages in order, and the version it was made at. */
struct HostTable {
    eagle_host_table head{ &HostTable::runEntry };   // first member: the handle is the table's address
    std::vector<std::vector<HostBranch>> stages;
    HostVersion version;
    std::int64_t madeAt = 0;

    HostTable(std::vector<std::vector<HostBranch>> s, HostVersion v)
        : stages(std::move(s)), version(std::move(v)), madeAt(version->load()) {}

    bool stale() const { return version->load() != madeAt; }

    static int runEntry(const eagle_host_table* t) noexcept
    {
        const auto* self = reinterpret_cast<const HostTable*>(t);
        if (self->stale()) return EAGLE_HOST_TABLE_STALE;
        try {
            for (const auto& stage : self->stages) runHostStageShared(stage);
        } catch (...) {
            return EAGLE_HOST_TABLE_FAILED;
        }
        return EAGLE_HOST_TABLE_OK;
    }
};

}  // namespace exec
}  // namespace eagle
