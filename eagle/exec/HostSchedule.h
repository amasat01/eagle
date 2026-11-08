// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// How the host runs one stage of a graph: the two CPU schedules, the pure rule
// that picks between them, and how many threads a stage is worth.
//
//   work_share    nodes run one after another, each cut into tiles spread over
//                 the team (`HostTeam::run`). Right when a node has enough
//                 samples to keep every core busy.
//   branch_tasks  the stage's independent branches run at once, one per thread,
//                 each branch's nodes serially on one thread. Right for small
//                 batches, where a per-node barrier costs more than the node.
//
// Both run on eagle's host pool (`HostPool.h`) as ONE launch per stage: a phase
// per node (its tiles are the lanes) for `work_share`, one phase with a lane per
// branch for `branch_tasks`. Both give bit-identical results: a node is a
// per-sample kernel, so how its samples are cut across threads changes nothing,
// and a branch's nodes always run in order.
//
// `detail::HostStage` (internal) adds the sizing: its first launch runs in place untimed (it is cold), its
// second is timed, and the stage then takes its whole budget (at most the branches or the
// packets) if that serial time is worth a pool wake (2 * `HOST_FLOOR_US`). A stage below
// that keeps running in place: no thread is woken for work smaller than the wake-up.
#pragma once

#include "eagle/exec/HostPool.h"
#include "eagle/exec/HostTeam.h"
#include "eagle/exec/Partition.h"
#include "eagle/util/Busy.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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

inline void checkEntries(const std::vector<HostBranch>& branches, const char* who) {
    for (const HostBranch& branch : branches)
        for (const HostBranchNode& node : branch)
            if (node.entry == nullptr)
                throw std::runtime_error(std::string(who) + ": null host entry");
}

/** @brief Samples a lane covers at least: one cache line of doubles. */
inline constexpr std::int64_t LANE_PACKET = 8;

/** @brief The tile of a @p count-sample partition cut for @p lanes threads: an even
 *  share rounded up to a packet, capped by the L2 rule (`HostTeam::tile_size`);
 *  `lanes <= 0` is the L2 rule alone. */
inline std::int64_t tileFor(std::int64_t count, std::int64_t lanes) {
    const auto l2 = static_cast<std::int64_t>(HostTeam::tile_size(0));
    if (lanes <= 0) return l2;
    const std::int64_t share = (count + lanes - 1) / lanes;
    return std::max<std::int64_t>(1, std::min(l2, (share + LANE_PACKET - 1) / LANE_PACKET * LANE_PACKET));
}

/** @brief A stage lowered to pool lanes: lane `i` is `calls[start[i], start[i+1])`,
 *  run in order on one thread; `gate[i]` is the first lane of its phase. */
struct HostPlan {
    struct Call { HostEntryV2 entry; void* const* params; std::int64_t base, count, nSamples; };
    std::vector<Call> calls;
    std::vector<std::uint32_t> start{ 0 }, gate;
    std::uint32_t phase = 0, width = 0;   // first lane of the open phase; the widest phase

    std::uint32_t lanes() const { return static_cast<std::uint32_t>(start.size() - 1); }
    void endLane() {
        start.push_back(static_cast<std::uint32_t>(calls.size()));
        gate.push_back(phase);
        width = std::max(width, lanes() - phase);
    }
    void endPhase() { phase = lanes(); }
    HostPool::Job job() const { return { lanes(), gate.data(), &runLane, this }; }

    static void runLane(const void* ctx, std::uint32_t lane) {
        const auto& plan = *static_cast<const HostPlan*>(ctx);
        for (std::uint32_t c = plan.start[lane]; c < plan.start[lane + 1]; ++c) {
            const Call& k = plan.calls[c];
            k.entry(k.params, k.base, k.count, k.nSamples);
        }
    }
};

/** @brief `branch_tasks`: one phase, one lane per branch, its nodes in order. */
inline HostPlan lowerBranches(const std::vector<HostBranch>& branches) {
    HostPlan plan;
    for (const HostBranch& branch : branches) {
        for (const HostBranchNode& node : branch)
            for (const Partition& p : node.parts)
                if (p.count > 0)
                    plan.calls.push_back({ node.entry, node.params.data(), p.base, p.count, p.nSamples });
        plan.endLane();
    }
    return plan;
}

/** @brief `work_share`: phase `j` holds node `j` of every branch, cut into tiles for
 *  @p lanes threads (`tileFor`), one tile per lane. */
inline HostPlan lowerTiles(const std::vector<HostBranch>& branches, std::int64_t lanes) {
    HostPlan plan;
    std::size_t depth = 0;
    for (const HostBranch& branch : branches) depth = std::max(depth, branch.size());
    for (std::size_t j = 0; j < depth; ++j) {
        for (const HostBranch& branch : branches) {
            if (j >= branch.size()) continue;
            const HostBranchNode& node = branch[j];
            for (const Partition& p : node.parts) {
                const std::int64_t tile = tileFor(p.count, lanes);
                for (std::int64_t base = p.base; base < p.base + p.count; base += tile) {
                    plan.calls.push_back({ node.entry, node.params.data(), base,
                                           std::min(tile, p.base + p.count - base), p.nSamples });
                    plan.endLane();
                }
            }
        }
        plan.endPhase();
    }
    return plan;
}

inline void runPlan(const HostPlan& plan, std::int64_t team) {
    const HostPool::Job job = plan.job();
    team = std::min<std::int64_t>(team, plan.width);
    if (team <= 1) HostPool::run_in_place(job);
    else HostPool::instance().run(job, static_cast<int>(team));
}
}  // namespace detail

/**
 * @brief Run the independent @p branches at once, one per thread; each
 *        branch's nodes run in order as plain serial calls over their whole
 *        partitions. One pool launch, a lane per branch.
 */
inline void runHostBranches(const std::vector<HostBranch>& branches) {
    detail::checkEntries(branches, "runHostBranches");
    detail::runPlan(detail::lowerBranches(branches), host_max_threads());
}

/**
 * @brief Run a `work_share` stage: the nodes one after another, each over the whole
 *        team, as ONE pool launch (a phase per node, its tiles the lanes).
 *
 * The same tiles as a `HostTeam::run` per node -- so the values are bit-identical --
 * but the team is woken once per stage, not once per node. A stage whose nodes have
 * one tile each (or a team of one) runs in place.
 */
inline void runHostStage(const std::vector<HostBranch>& branches) {
    detail::checkEntries(branches, "runHostStage");
    detail::runPlan(detail::lowerTiles(branches, 0), host_max_threads());
}

/**
 * @brief One stage of a built graph, sized by its own second launch.
 *
 * Launch 1 runs in place untimed, launch 2 in place and timed; from then on the stage uses
 * `min(budget, items)` threads if the serial time is at least 2 * `HOST_FLOOR_US`, else one. `items` is the
 * branches (`branch_tasks`) or the packets of the widest node (`work_share`).
 * One thread means in place. A new stage (a rebind) measures again. Safe to run
 * from several threads: a launch that finds the pool busy runs in place.
 */
namespace detail {

class HostStage {
public:
    HostStage(std::vector<HostBranch> branches, bool tasks)
        : branches_(std::move(branches)), tasks_(tasks) {
        detail::checkEntries(branches_, "HostStage");
        if (tasks_) items_ = static_cast<std::int64_t>(branches_.size());
        else
            for (const HostBranch& branch : branches_)
                for (const HostBranchNode& node : branch)
                    for (const Partition& p : node.parts)
                        items_ = std::max(items_, (p.count + detail::LANE_PACKET - 1) / detail::LANE_PACKET);
    }

    void run() {
        int state = state_.load(std::memory_order_acquire);
        if (state == READY) {
            if (cap_ <= 1) { runInPlace(); return; }    // one lane: the team setting cannot matter
            const int team = std::min(cap_, host_max_threads());
            if (team > 1) detail::runPlan(plan_, team);
            else runInPlace();
            return;
        }
        if (state == UNMEASURED && state_.compare_exchange_strong(state, WARMED, std::memory_order_acq_rel)) {
            runInPlace();    // launch 1 is cold: not timed
            return;
        }
        if (state == WARMED && state_.compare_exchange_strong(state, MEASURING, std::memory_order_acquire)) {
            const auto t0 = std::chrono::steady_clock::now();
            runInPlace();
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
            cap_ = us >= 2 * HOST_FLOOR_US ? static_cast<int>(std::max<std::int64_t>(items_, 1)) : 1;
            // tiles are cut for at most one lane per physical core: cut for cap_ (up to a packet per lane)
            // they shrink to one packet, and a heavy node pays a call per 8 samples
            if (cap_ > 1)
                plan_ = tasks_ ? detail::lowerBranches(branches_)
                               : detail::lowerTiles(branches_, std::min<std::int64_t>(cap_, physical_cores()));
            state_.store(READY, std::memory_order_release);
            return;
        }
        runInPlace();
    }

    /** @brief The threads the next launch uses; 1 = in place (also before launch 2). */
    int lanes() const {
        if (state_.load(std::memory_order_acquire) != READY) return 1;
        return std::max(1, std::min(cap_, host_max_threads()));
    }

private:
    enum : int { UNMEASURED, WARMED, MEASURING, READY };

    void runInPlace() const {
        for (const HostBranch& branch : branches_) detail::runBranch(branch);
    }

    std::vector<HostBranch> branches_;
    bool tasks_;
    std::int64_t items_ = 0;
    std::atomic<int> state_{ UNMEASURED };
    int cap_ = 1;              // written once before READY
    detail::HostPlan plan_;    // written once before READY
};

/** @brief A graph's host stages run back to back under its busy guard: the stages and the guard
 *  are borrowed (the binding keeps their owners alive). */
class HostSequence {
public:
    HostSequence(util::BusyGuard* guard, std::vector<HostStage*> stages)
        : guard_(guard), stages_(std::move(stages)) {}

    util::BusyGuard& guard() const { return *guard_; }

    void run(std::int64_t n) {
        for (std::int64_t i = 0; i < n; ++i)
            for (HostStage* stage : stages_) stage->run();
    }

private:
    util::BusyGuard* guard_;
    std::vector<HostStage*> stages_;
};

}  // namespace detail

}  // namespace exec
}  // namespace eagle
