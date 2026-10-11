// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Residency: blocks per SM at a register count, against the CUDA occupancy
 * calculator's answers (cuda_occupancy.h, CUDA 12.6) for the four architecture
 * classes; the register edges; and the rule that keeps a bounded recompile.
 * CUDA-free; compiled by CXX in BOTH build modes. */
#include <gtest/gtest.h>

#include "eagle/policy/Residency.h"

#include <array>
#include <utility>

namespace eagle_tests {
namespace ResidencyTest {

using namespace eagle::policy;

namespace {

ResidencyDevice device(int major, int minor)
{
    // threads per SM: 2048 everywhere but the cc 8.9 part (1536)
    return { major, minor, 65536, (major == 8 && minor == 9) ? 1536 : 2048 };
}

struct Row { int major, minor, regs; std::array<int, 3> blocks; };  // block 256, 320, 384

// Active blocks per SM from cudaOccMaxActiveBlocksPerMultiprocessor.
const Row CALCULATOR[] = {
    {6, 1, 32, {8, 6, 5}},
    {6, 1, 40, {6, 4, 4}},
    {6, 1, 44, {5, 4, 3}},
    {6, 1, 46, {5, 4, 3}},
    {6, 1, 128, {2, 1, 1}},
    {6, 1, 160, {1, 1, 1}},
    {6, 1, 168, {1, 1, 1}},
    {6, 1, 172, {1, 0, 0}},
    {6, 1, 174, {1, 0, 0}},
    {6, 1, 186, {1, 0, 0}},
    {6, 1, 188, {1, 0, 0}},
    {6, 1, 210, {1, 0, 0}},
    {8, 0, 32, {8, 6, 5}},
    {8, 0, 40, {6, 4, 4}},
    {8, 0, 44, {5, 4, 3}},
    {8, 0, 46, {5, 4, 3}},
    {8, 0, 128, {2, 1, 1}},
    {8, 0, 160, {1, 1, 1}},
    {8, 0, 168, {1, 1, 1}},
    {8, 0, 172, {1, 0, 0}},
    {8, 0, 174, {1, 0, 0}},
    {8, 0, 186, {1, 0, 0}},
    {8, 0, 188, {1, 0, 0}},
    {8, 0, 210, {1, 0, 0}},
    {8, 9, 32, {6, 4, 4}},
    {8, 9, 40, {6, 4, 4}},
    {8, 9, 44, {5, 4, 3}},
    {8, 9, 46, {5, 4, 3}},
    {8, 9, 128, {2, 1, 1}},
    {8, 9, 160, {1, 1, 1}},
    {8, 9, 168, {1, 1, 1}},
    {8, 9, 172, {1, 0, 0}},
    {8, 9, 174, {1, 0, 0}},
    {8, 9, 186, {1, 0, 0}},
    {8, 9, 188, {1, 0, 0}},
    {8, 9, 210, {1, 0, 0}},
    {9, 0, 32, {8, 6, 5}},
    {9, 0, 40, {6, 4, 4}},
    {9, 0, 44, {5, 4, 3}},
    {9, 0, 46, {5, 4, 3}},
    {9, 0, 128, {2, 1, 1}},
    {9, 0, 160, {1, 1, 1}},
    {9, 0, 168, {1, 1, 1}},
    {9, 0, 172, {1, 0, 0}},
    {9, 0, 174, {1, 0, 0}},
    {9, 0, 186, {1, 0, 0}},
    {9, 0, 188, {1, 0, 0}},
    {9, 0, 210, {1, 0, 0}},
};

}  // namespace

TEST(Residency, MatchesTheOccupancyCalculator)
{
    constexpr int BLOCKS[3] = { 256, 320, 384 };
    for (const Row& row : CALCULATOR)
        for (int b = 0; b < 3; ++b)
            EXPECT_EQ(residentBlocks(device(row.major, row.minor), row.regs, BLOCKS[b]), row.blocks[b])
                << "cc " << row.major << "." << row.minor << " regs " << row.regs << " block " << BLOCKS[b];
}

TEST(Residency, ThreadsStepDownAtTheEdgesNotSmoothly)
{
    const ResidencyDevice a100 = device(8, 0);
    EXPECT_EQ(residentThreads(a100, 32), 2048);
    EXPECT_EQ(residentThreads(a100, 40), 1536);
    EXPECT_EQ(residentThreads(a100, 41), 1280);
    EXPECT_EQ(residentThreads(a100, 46), 1280);
    EXPECT_EQ(residentThreads(a100, 168), 384);
    EXPECT_EQ(residentThreads(a100, 172), 256);
    EXPECT_EQ(residentThreads(a100, 188), 256);
    EXPECT_EQ(residentThreads(device(8, 9), 40), 1536);  // the thread limit clamps the register file
    EXPECT_EQ(residentThreads(device(8, 9), 32), 1536);
}

TEST(Residency, NoBlockOverTheWarpBudgetIsLaunched)
{
    // 172+ registers leave 8 warps: the 10- and 11-warp blocks fit nowhere, 256 does once
    for (const auto& cc : { std::pair{ 6, 1 }, std::pair{ 8, 0 }, std::pair{ 8, 9 }, std::pair{ 9, 0 } })
        for (int regs : { 172, 186, 188, 210 }) {
            const ResidencyDevice dev = device(cc.first, cc.second);
            EXPECT_EQ(residentBlocks(dev, regs, 320), 0);
            EXPECT_EQ(residentBlocks(dev, regs, 352), 0);
            EXPECT_EQ(residentBlocks(dev, regs, 256), 1);
        }
}

TEST(Residency, CcSixZeroHasTwoSubPartitions)
{
    EXPECT_EQ(subPartitions(6, 0), 2);
    EXPECT_EQ(subPartitions(6, 1), 4);
    EXPECT_EQ(subPartitions(8, 0), 4);
    // 128 warp-units per sub-partition: 32 regs -> 4 units -> 32 warps each
    EXPECT_EQ(residentWarps({ 6, 0, 65536, 0 }, 32), 64);
    EXPECT_EQ(residentWarps({ 6, 0, 65536, 0 }, 40), 2 * (128 / 5));
}

TEST(Residency, UnknownDevicesReportZero)
{
    EXPECT_EQ(residentWarps({}, 32), 0);
    EXPECT_EQ(residentWarps(device(8, 0), 0), 0);
    EXPECT_EQ(residentBlocks(device(8, 0), 32, 0), 0);
    EXPECT_EQ(residentBlocks(device(8, 0), 32, 2048), 0);
}

TEST(Residency, EdgeIsTheTopOfThePreviousUnitWhenItHoldsMore)
{
    const ResidencyDevice a100 = device(8, 0);
    for (int regs = 41; regs <= 48; ++regs) EXPECT_EQ(registerEdge(a100, regs), 40) << regs;
    for (int regs = 169; regs <= 176; ++regs) EXPECT_EQ(registerEdge(a100, regs), 168) << regs;
    EXPECT_EQ(registerEdge(a100, 46), 40);
    EXPECT_EQ(registerEdge(a100, 174), 168);
    EXPECT_EQ(registerEdge(a100, 186), 0);  // 18 above the edge: out of reach
    EXPECT_EQ(registerEdge(a100, 188), 0);
    EXPECT_EQ(registerEdge(a100, 0), 0);
    EXPECT_EQ(registerEdge(a100, 300), 0);
}

TEST(Residency, EdgeNeedsAGainAfterTheThreadLimit)
{
    // cc 8.9 holds 1536 threads: 40 regs already reach it, so 33..40 have no edge to drop to
    EXPECT_EQ(registerEdge(device(8, 9), 36), 0);
    EXPECT_EQ(registerEdge(device(8, 9), 46), 40);
    EXPECT_EQ(registerEdge(device(8, 0), 36), 32);
    // 49..56 sits one step under the 48-register edge
    EXPECT_EQ(registerEdge(device(8, 0), 52), 48);
}

TEST(Residency, BoundedRecompileKeptOnlyWhenNothingSpillsAndItFits)
{
    const KernelResources free{ 46, 0 };
    EXPECT_TRUE(keepBounded(free, { 40, 0 }, 40));    // keep: fits, no local memory
    EXPECT_FALSE(keepBounded(free, { 40, 8 }, 40));   // a spilled double is refused
    EXPECT_FALSE(keepBounded(free, { 44, 0 }, 40));   // the compiler could not reach the edge
    EXPECT_FALSE(keepBounded(free, { 40, 0 }, 0));    // no edge in reach
    const KernelResources stacked{ 174, 24 };
    EXPECT_TRUE(keepBounded(stacked, { 168, 24 }, 168));   // local size unchanged
    EXPECT_FALSE(keepBounded(stacked, { 168, 40 }, 168));
    EXPECT_FALSE(keepBounded(stacked, { 168, 16 }, 168));
}

}  // namespace ResidencyTest
}  // namespace eagle_tests
