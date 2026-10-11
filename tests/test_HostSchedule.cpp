// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* The host schedules: the pure rule that picks one, the branch runner, and the
 * one-team-region-never-nested guarantee. CUDA-free; compiled by CXX in BOTH
 * build modes with EAGLE_CPU_ONLY forced on this one source. */
#include <gtest/gtest.h>

#include "eagle/exec/HostSchedule.h"
#include "eagle/exec/HostTeam.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eagle_tests {
namespace HostScheduleTest {

using namespace eagle::exec;

namespace {

// One node: out[i] = in[i] * 2 + 1 over [base, base+count), serial.
struct Planes { const double* in; double* out; };

void step(void* const* params, std::int64_t base, std::int64_t count, std::int64_t)
{
    const auto* p = static_cast<const Planes*>(params[0]);
    for (std::int64_t i = base; i < base + count; ++i) p->out[i] = p->in[i] * 2.0 + 1.0;
}

std::atomic<int> g_deepest_level{ 0 };

void noteLevel(void* const*, std::int64_t, std::int64_t, std::int64_t)
{
#ifdef _OPENMP
    int seen = omp_get_level(), prev = g_deepest_level.load();
    while (seen > prev && !g_deepest_level.compare_exchange_weak(prev, seen)) {}
#endif
}

// `branches` independent chains of `depth` nodes ping-ponging between two planes.
struct Fork {
    std::vector<std::vector<double>> a, b;
    std::vector<Planes> ab, ba;
    Fork(int branches, int n) : a(branches), b(branches), ab(branches), ba(branches)
    {
        for (int k = 0; k < branches; ++k) {
            a[k].resize(n); b[k].assign(n, 0.0);
            for (int i = 0; i < n; ++i) a[k][i] = 0.001 * i + k;
            ab[k] = { a[k].data(), b[k].data() };
            ba[k] = { b[k].data(), a[k].data() };
        }
    }
    std::vector<HostBranch> branches(int depth, std::int64_t n) const
    {
        std::vector<HostBranch> out;
        for (std::size_t k = 0; k < a.size(); ++k) {
            HostBranch br;
            for (int d = 0; d < 2 * depth; ++d)
                br.push_back({ &step, { const_cast<Planes*>(d % 2 ? &ba[k] : &ab[k]) },
                               { Partition::whole(n) } });
            out.push_back(std::move(br));
        }
        return out;
    }
};

}  // namespace

TEST(HostScheduleTest, ResolveIsPinnedAtItsBoundaries)
{
    EXPECT_EQ(resolve(BRANCH_TASKS_MAX_SAMPLES, 2, 2, true), Schedule::BranchTasks);
    EXPECT_EQ(resolve(BRANCH_TASKS_MAX_SAMPLES + 1, 2, 2, true), Schedule::WorkShare);
    EXPECT_EQ(resolve(1024, 1, 8, true), Schedule::WorkShare);   // one branch
    EXPECT_EQ(resolve(1024, 2, 1, true), Schedule::WorkShare);   // one thread
    EXPECT_EQ(resolve(1024, 2, 2, false), Schedule::WorkShare);  // a non-native node
    EXPECT_EQ(resolve(1024, 8, 8, true), Schedule::BranchTasks);
}

TEST(HostScheduleTest, BranchesEqualASerialRunBitwise)
{
    const int n = 1000, depth = 4;
    Fork serial(6, n), tasked(6, n);
    for (const HostBranch& branch : serial.branches(depth, n))
        for (const HostBranchNode& node : branch)
            HostTeam::run_serial(node.entry, node.params.data(), node.parts[0]);
    runHostBranches(tasked.branches(depth, n));
    for (std::size_t k = 0; k < 6; ++k)
        EXPECT_EQ(serial.a[k], tasked.a[k]);
}

TEST(HostScheduleTest, EachBranchRunsItsNodesInOrder)
{
    Fork f(4, 64);
    runHostBranches(f.branches(1, 64));   // a -> b -> a: a' = 2*(2a+1)+1
    for (int k = 0; k < 4; ++k)
        for (int i = 0; i < 64; ++i)
            ASSERT_EQ(f.a[k][i], 2.0 * (2.0 * (0.001 * i + k) + 1.0) + 1.0);
}

TEST(HostScheduleTest, NoNestedRegionIsOpened)
{
    std::vector<HostBranch> branches(4, HostBranch{ { &noteLevel, {}, { Partition::whole(8) } } });
    g_deepest_level = 0;
    runHostBranches(branches);
    EXPECT_LE(g_deepest_level.load(), 1);

    // inside a caller's own parallel region everything runs in place
    g_deepest_level = 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(2)
#endif
    {
        runHostBranches(branches);
        HostTeam::run(&noteLevel, static_cast<void* const*>(nullptr), Partition::whole(8));
    }
    EXPECT_LE(g_deepest_level.load(), 1);
}

TEST(HostScheduleTest, NullEntryIsRefused)
{
    std::vector<HostBranch> branches(2, HostBranch{ { nullptr, {}, { Partition::whole(4) } } });
    EXPECT_THROW(runHostBranches(branches), std::runtime_error);
}

}  // namespace HostScheduleTest
}  // namespace eagle_tests
