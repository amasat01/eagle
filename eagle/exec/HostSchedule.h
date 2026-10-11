// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// How the host runs one stage of a graph: the two CPU schedules and the pure
// rule that picks between them.
//
//   work_share    nodes run one after another, each spread over the whole team
//                 (`HostTeam::run`). Right when a node has enough samples to
//                 keep every core busy.
//   branch_tasks  the stage's independent branches run at once, one per thread,
//                 each branch's nodes serially on one thread. Right for small
//                 batches, where a per-node team barrier costs more than the
//                 node itself.
//
// Both give bit-identical results: a node is a per-sample kernel, so how its
// samples are cut across threads changes nothing, and a branch's nodes always
// run in order.
//
// One team region per run, never nested: inside a caller's parallel region the
// branches run in place on the calling thread.
#pragma once

#include "eagle/exec/Partition.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eagle {
namespace exec {

/** @brief Largest sample count for which `branch_tasks` beats `work_share`
 *  (measured: faster by 5-20x at 64-1024 samples, break-even near 16k). */
inline constexpr std::int64_t BRANCH_TASKS_MAX_SAMPLES = 4096;

enum class Schedule { WorkShare, BranchTasks };

/**
 * @brief Pick the schedule of one stage.
 *
 * `BranchTasks` iff the stage has at least two independent branches, a team of
 * at least two threads, at most `BRANCH_TASKS_MAX_SAMPLES` samples, and every
 * node is a native host entry (`nativeOnly`; a node with a Python host arm
 * cannot run inside a task). Pure: depends on nothing but its arguments.
 */
inline Schedule resolve(std::int64_t n, std::int64_t branches, std::int64_t threads,
                        bool nativeOnly) {
    const bool tasks = nativeOnly && branches >= 2 && threads >= 2 &&
                       n <= BRANCH_TASKS_MAX_SAMPLES;
    return tasks ? Schedule::BranchTasks : Schedule::WorkShare;
}

/** @brief One host node of a branch: the entry, its packed args and the
 *  partitions it covers. */
struct HostBranchNode {
    HostEntryV2 entry = nullptr;
    std::vector<void*> params;
    std::vector<Partition> parts;
};

using HostBranch = std::vector<HostBranchNode>;

namespace detail {
inline void runBranch(const HostBranch& branch) {
    for (const HostBranchNode& node : branch)
        for (const Partition& part : node.parts)
            if (part.count > 0)
                node.entry(node.params.data(), part.base, part.count, part.nSamples);
}
}  // namespace detail

/**
 * @brief Run the independent @p branches at once, one per thread; each
 *        branch's nodes run in order as plain serial calls over their whole
 *        partitions.
 *
 * A dynamic parallel-for over the branches rather than OpenMP tasks: the same
 * schedule for independent work, without the task queues that deadlock or
 * abort when several caller threads run a stage at the same time.
 */
inline void runHostBranches(const std::vector<HostBranch>& branches) {
    for (const HostBranch& branch : branches)
        for (const HostBranchNode& node : branch)
            if (node.entry == nullptr)
                throw std::runtime_error("runHostBranches: null host entry");
    const auto count = static_cast<std::int64_t>(branches.size());
#ifdef _OPENMP
    if (omp_get_level() == 0 && count > 1) {
        const int team = static_cast<int>(std::min<std::int64_t>(count, omp_get_max_threads()));
#pragma omp parallel for schedule(dynamic, 1) num_threads(team)
        for (std::int64_t b = 0; b < count; ++b) detail::runBranch(branches[b]);
        return;
    }
#endif
    for (const HostBranch& branch : branches) detail::runBranch(branch);
}

}  // namespace exec
}  // namespace eagle
