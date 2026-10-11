// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* The windowed reverse of a stepped run: window plan and default, the record
 * slot, the recording loop, the column gather, the stop order, live counts
 * and the window driver's call order. CUDA-free; compiled by CXX in BOTH
 * build modes. */
#include <gtest/gtest.h>

#include "eagle/diff/Record.h"
#include "eagle/diff/Reverse.h"
#include "eagle/diff/Window.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace eagle_tests {
namespace DiffWindowTest {

using namespace eagle::diff;

TEST(DiffWindow, DefaultIsClampedCeilSqrt) {
    EXPECT_EQ(defaultWindow(1), 16);
    EXPECT_EQ(defaultWindow(256), 16);
    EXPECT_EQ(defaultWindow(257), 17);
    EXPECT_EQ(defaultWindow(10000), 100);
    EXPECT_EQ(defaultWindow(10001), 101);
    EXPECT_EQ(defaultWindow(std::int64_t{ 1 } << 40), 1024);
    EXPECT_EQ(ceilSqrt(0), 0);
    EXPECT_EQ(ceilSqrt(1), 1);
    EXPECT_EQ(ceilSqrt(99), 10);
    EXPECT_EQ(ceilSqrt(100), 10);
}

TEST(DiffWindow, PlanCutsAlignedWindows) {
    const WindowPlan p = planWindows(100, 37);
    EXPECT_EQ(p.count(), 3);
    EXPECT_EQ(p.start(2), 74);
    EXPECT_EQ(p.length(2), 26);
    EXPECT_EQ(p.length(0), 37);
    EXPECT_EQ(p.windowOf(73), 1);
    EXPECT_EQ(p.row(74), 0);
    EXPECT_EQ(planWindows(0, 5).count(), 0);
    EXPECT_THROW(planWindows(10, 0), std::invalid_argument);
    EXPECT_THROW(planWindows(-1, 4), std::invalid_argument);
}

TEST(DiffWindow, SlotRefusesLiveAndHeldAndStale) {
    RecordSlot slot;
    ASSERT_EQ(slot.begin(), RecordSlot::Begin::Ok);
    EXPECT_EQ(slot.generation(), 1u);
    EXPECT_EQ(slot.begin(), RecordSlot::Begin::Live);
    ASSERT_TRUE(slot.hold(1));
    EXPECT_TRUE(slot.held());
    EXPECT_EQ(slot.begin(), RecordSlot::Begin::Held);
    EXPECT_TRUE(slot.release(1, true));
    EXPECT_FALSE(slot.live());
    ASSERT_EQ(slot.begin(), RecordSlot::Begin::Ok);
    EXPECT_FALSE(slot.hold(1));   // generation 1 is stale now
    slot.abandon();
    EXPECT_FALSE(slot.live());
}

TEST(DiffWindow, SlotBeginsOnceUnderContention) {
    RecordSlot slot;
    std::vector<std::thread> threads;
    std::vector<int> ok(8, 0);
    for (int t = 0; t < 8; ++t)
        threads.emplace_back([&, t] { ok[t] = slot.begin() == RecordSlot::Begin::Ok; });
    for (auto& t : threads) t.join();
    int total = 0;
    for (int v : ok) total += v;
    EXPECT_EQ(total, 1);
    EXPECT_EQ(slot.generation(), 1u);
}

TEST(DiffWindow, RecordForwardCheckpointsWindowStarts) {
    std::vector<idx_t> checkpoints, steps;
    const idx_t ran = recordForward(
        10, 4, 1, [&] { return steps.size() == 9; }, [&](idx_t k) { checkpoints.push_back(k); },
        [&](idx_t j, idx_t k) { EXPECT_EQ(k, 1); steps.push_back(j); });
    EXPECT_EQ(ran, 9);
    EXPECT_EQ(checkpoints, (std::vector<idx_t>{ 0, 1, 2 }));
    EXPECT_EQ(recordForward(10, 4, 1, [] { return true; }, [](idx_t) { FAIL(); },
                            [](idx_t, idx_t) { FAIL(); }), 0);
}

TEST(DiffWindow, FusedLaunchesNeverCrossAWindowBoundary) {
    std::vector<std::pair<idx_t, idx_t>> launches;
    std::vector<idx_t> checkpoints;
    const idx_t ran = recordForward(
        23, 10, 4, [] { return false; }, [&](idx_t k) { checkpoints.push_back(k); },
        [&](idx_t j, idx_t k) { launches.emplace_back(j, k); });
    EXPECT_EQ(ran, 23);
    EXPECT_EQ(checkpoints, (std::vector<idx_t>{ 0, 1, 2 }));
    const std::vector<std::pair<idx_t, idx_t>> want{ { 0, 4 }, { 4, 4 }, { 8, 2 }, { 10, 4 }, { 14, 4 },
                                                     { 18, 2 }, { 20, 3 } };
    EXPECT_EQ(launches, want);
    EXPECT_THROW(recordForward(5, 2, 0, [] { return false; }, [](idx_t) {}, [](idx_t, idx_t) {}),
                 std::invalid_argument);
}

TEST(DiffWindow, ChunkSamplesIsALeafPowerOfTwo) {
    EXPECT_EQ(chunkSamples(1000, 10, 50, 256), 50);        // all fit
    EXPECT_EQ(chunkSamples(256 * 10, 10, 1000, 256), 256);
    EXPECT_EQ(chunkSamples(767 * 10, 10, 1000, 256), 512);  // rounded DOWN, never 3 * 256
    EXPECT_EQ(chunkSamples(1024 * 10, 10, 5000, 256), 1024);
    EXPECT_EQ(chunkSamples(255 * 10, 10, 1000, 256), 0);    // not one leaf block
    EXPECT_EQ(chunkSamples(5, 10, 1, 256), 0);              // not one sample
    EXPECT_THROW(chunkSamples(10, 10, 10, 3), std::invalid_argument);
}

TEST(DiffWindow, GatherColumnsReadsThroughTheOrder) {
    const idx_t n = 4;
    const std::vector<double> src{ 0, 1, 2, 3, 10, 11, 12, 13 };
    std::vector<double> dst(8, -1);
    const std::vector<idx_t> perm{ 3, 1, 0, 2 };
    gatherColumns(dst.data(), n, src.data(), n, perm.data(), 3, 2, sizeof(double));
    EXPECT_EQ(dst, (std::vector<double>{ 3, 1, 0, -1, 13, 11, 10, -1 }));
    std::vector<double> narrow(4, -1);                      // a 2-wide pass of a 4-wide plane
    gatherColumns(narrow.data(), 2, src.data(), n, perm.data(), 2, 2, sizeof(double));
    EXPECT_EQ(narrow, (std::vector<double>{ 3, 1, 13, 11 }));
    std::vector<std::uint8_t> b{ 1, 0, 1, 1 }, bd(4, 9);
    gatherColumns(bd.data(), n, b.data(), n, perm.data(), 4, 1, 1);
    EXPECT_EQ(bd, (std::vector<std::uint8_t>{ 1, 0, 1, 1 }));
    EXPECT_THROW(gatherColumns(bd.data(), n, b.data(), n, perm.data(), 4, 1, 3), std::invalid_argument);
}

TEST(DiffWindow, StopOrderAndLiveCounts) {
    const std::vector<idx_t> steps{ 3, 7, 0, 7, 5 };
    EXPECT_EQ(stopOrder(steps.data(), 5), (std::vector<idx_t>{ 1, 3, 4, 0, 2 }));
    EXPECT_EQ(liveCounts(steps.data(), 5), (std::vector<idx_t>{ 4, 4, 4, 3, 3, 2, 2 }));
    const std::vector<idx_t> none{ 0, 0 };
    EXPECT_TRUE(liveCounts(none.data(), 2).empty());
    const std::vector<idx_t> bad{ -1 };
    EXPECT_THROW(liveCounts(bad.data(), 1), std::invalid_argument);
}

TEST(DiffWindow, DriverRunsWindowsLastFirstOverTheLivePrefix) {
    const std::vector<idx_t> steps{ 5, 2 };
    const std::vector<idx_t> live = liveCounts(steps.data(), 2);
    std::vector<std::string> log;
    reverseWindows(
        2, live, [&](idx_t k, idx_t c) { log.push_back("g" + std::to_string(k) + ":" + std::to_string(c)); },
        [&](idx_t c, idx_t len) { log.push_back("f" + std::to_string(c) + "+" + std::to_string(len)); },
        [&](idx_t j, idx_t c) { log.push_back("b" + std::to_string(j) + ":" + std::to_string(c)); });
    const std::vector<std::string> want{ "g2:1", "f4+1", "b4:1", "g1:1", "f2+2", "b3:1", "b2:1",
                                         "g0:2", "f0+2", "b1:2", "b0:2" };
    EXPECT_EQ(log, want);
}

TEST(DiffWindow, FillPrefixWritesTheStepIndex) {
    std::vector<std::int32_t> k{ 0, 0, 0 };
    fillPrefix(k.data(), 7, 2);
    EXPECT_EQ(k, (std::vector<std::int32_t>{ 7, 7, 0 }));
}

TEST(DiffWindow, AccumulatePrefixLeavesTheTail) {
    std::vector<float> acc{ 1, 1, 1 };
    const std::vector<float> g{ 2, 3, 4 };
    accumulatePrefix(acc.data(), g.data(), 2);
    EXPECT_EQ(acc, (std::vector<float>{ 3, 4, 1 }));
}

}  // namespace DiffWindowTest
}  // namespace eagle_tests
