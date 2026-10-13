// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Launch policy: the threads a persistent launch puts on an SM per device
 * class, the chunk, the compaction cadence and the bound-on-demand rules.
 * CUDA-free; compiled by CXX in BOTH build modes. */
#include <gtest/gtest.h>

#include "eagle/policy/Launch.h"

namespace eagle_tests {
namespace LaunchTest {

using namespace eagle::policy;

namespace {

const LaunchDevice A100 = { { 8, 0, 65536, 2048 }, 108, 2, 1410000 };
const LaunchDevice P2000 = { { 6, 1, 65536, 2048 }, 8, 32, 1480000 };
const LaunchDevice L4 = { { 8, 9, 65536, 1536 }, 58, 64, 2040000 };
const LaunchDevice H100 = { { 9, 0, 65536, 2048 }, 132, 2, 1755000 };
const LaunchDevice T4 = { { 7, 5, 65536, 1024 }, 40, 32, 1590000 };

}  // namespace

TEST(Launch, SaturationFollowsTheFp64Class)
{
    EXPECT_EQ(persistSaturation(A100, true), 384);
    EXPECT_EQ(persistSaturation(P2000, true), 256);
    EXPECT_EQ(persistSaturation(L4, true), 256);
}

TEST(Launch, A100OscillatorRows)
{
    EXPECT_EQ(persistThreads(A100, true, 36, 1000000, 0.23), 384);
    EXPECT_EQ(persistThreads(A100, true, 36, 1000000, 1.0), 384);
    EXPECT_EQ(persistThreads(A100, true, 36, 100000, 0.23), 256);
    EXPECT_EQ(persistThreads(A100, true, 36, 270000, 0.23), 320);
}

TEST(Launch, ACeilingUnderTheSaturationIsTakenWhole)
{
    EXPECT_EQ(persistThreads(A100, true, 166, 1000000, 0.23), 384);
    EXPECT_EQ(persistThreads(A100, true, 166, 1000000, 1.0), 384);
    EXPECT_EQ(persistThreads(L4, true, 174, 1000000, 0.23), 256);
}

TEST(Launch, SlowFp64PartsKeepTheirTwoWarps)
{
    EXPECT_EQ(persistThreads(P2000, true, 43, 1000000, 0.23), 256);
    EXPECT_EQ(persistThreads(P2000, true, 43, 1000000, 1.0), 256);
    EXPECT_EQ(persistThreads(L4, true, 36, 1000000, 0.23), 256);
}

TEST(Launch, TheDrainLimitShrinksASmallBatch)
{
    EXPECT_LT(persistThreads(A100, true, 36, 270000, 0.23), persistThreads(A100, true, 36, 1000000, 0.23));
    EXPECT_EQ(persistThreads(A100, true, 36, 1, 1.0), 256);
}

TEST(Launch, DemandIsTheSaturationPlusTheFetchWarp)
{
    EXPECT_EQ(persistDemand(A100, true), 512);
    EXPECT_EQ(persistDemand(P2000, true), 384);
}

TEST(Launch, UnknownDevicesSizeNothing)
{
    EXPECT_EQ(persistThreads({}, true, 36, 1000000, 0.5), 0);
}

TEST(Launch, ThreadsNeverExceedTheResidencyCeiling)
{
    for (const LaunchDevice* dev : { &A100, &P2000, &L4 })
        for (int regs = 16; regs <= 255; regs += 5)
            for (double f : { 0.05, 0.5, 1.0 }) {
                const int t = persistThreads(*dev, true, regs, 10000000, f);
                EXPECT_LE(t, residentThreads(dev->residency, regs)) << regs;
                EXPECT_GT(t, 0);
            }
}

TEST(Launch, ChunkRuleRows)
{
    // (device: cc, SMs; precision, ops, mean steps) -> chunk; a claim's cost is a device-class fact, never the clock
    EXPECT_EQ(persistChunk(P2000, true, 41, 22), 4);
    EXPECT_EQ(persistChunk(P2000, true, 44, 22), 4);
    EXPECT_EQ(persistChunk(P2000, true, 44, 215), 8);
    EXPECT_EQ(persistChunk(P2000, false, 44, 22), 16);
    EXPECT_EQ(persistChunk(P2000, false, 44, 215), 32);
    EXPECT_EQ(persistChunk(A100, false, 41, 22), 16);
    EXPECT_EQ(persistChunk(A100, true, 44, 22), 8);
    EXPECT_EQ(persistChunk(A100, true, 44, 215), 32);
    EXPECT_EQ(persistChunk(A100, true, 1106, 498), 8);
    EXPECT_EQ(persistChunk(H100, false, 41, 22), 32);
    EXPECT_EQ(persistChunk(H100, true, 41, 22), 16);
    EXPECT_EQ(persistChunk(T4, true, 41, 22), 2);
    LaunchDevice slow = P2000;
    slow.clockKhz = 0;
    EXPECT_EQ(persistChunk(slow, false, 44, 22), persistChunk(P2000, false, 44, 22));
}

TEST(Launch, ChunkIgnoresTheClock)
{
    for (const LaunchDevice* dev : { &A100, &P2000, &H100, &T4 })
        for (bool fp64 : { true, false }) {
            LaunchDevice other = *dev;
            other.clockKhz = dev->clockKhz / 2;
            EXPECT_EQ(persistChunk(other, fp64, 41, 22), persistChunk(*dev, fp64, 41, 22));
        }
}

TEST(Launch, ChunkGrowsWithTheSmCount)
{
    LaunchDevice few = H100, many = H100;
    few.sms = 8;
    EXPECT_EQ(persistChunk(few, false, 44, 21.5), 4);
    EXPECT_EQ(persistChunk(many, false, 44, 21.5), 32);
}

TEST(Launch, APascalClaimCostsTwelveCycles)
{
    EXPECT_EQ(claimCycles(6), 12);
    EXPECT_EQ(claimCycles(7), 1);
    EXPECT_EQ(claimCycles(9), 1);
    LaunchDevice volta = P2000;
    volta.residency.ccMajor = 7;
    volta.residency.ccMinor = 5;
    EXPECT_EQ(persistChunk(P2000, true, 41, 22), 4);
    EXPECT_EQ(persistChunk(volta, true, 41, 22), 2);
}

TEST(Launch, ChunkIsAPowerOfTwoInItsBounds)
{
    EXPECT_EQ(persistChunk(P2000, true, 44, 1.0), PERSIST_CHUNK_MIN);
    EXPECT_EQ(persistChunk(P2000, true, 44, 1e9), PERSIST_CHUNK_MAX);
    EXPECT_EQ(persistChunk(P2000, true, 0, 22), PERSIST_CHUNK_MAX);
    for (const LaunchDevice* dev : { &A100, &P2000, &L4 })
        for (double m : { 1.0, 7.0, 23.0, 100.0, 5000.0 })
            for (double ops : { 10.0, 44.0, 1106.0 }) {
                const int c = persistChunk(*dev, true, ops, m);
                EXPECT_EQ(c & (c - 1), 0);
                EXPECT_GE(c, PERSIST_CHUNK_MIN);
                EXPECT_LE(c, PERSIST_CHUNK_MAX);
            }
}

TEST(Launch, ChunkNeverShrinksAsStepsGrow)
{
    int last = 0;
    for (double m = 1; m < 1e5; m *= 1.3) {
        const int c = persistChunk(A100, true, 44, m);
        EXPECT_GE(c, last);
        last = c;
    }
}

TEST(Launch, TheFetchTermAddsOneWarpOnceTheChunkIsKnown)
{
    EXPECT_EQ(persistThreads(P2000, true, 43, 1000000, 0.23, 4, 44), 384);
    EXPECT_EQ(persistThreads(A100, true, 36, 1000000, 0.23, 8, 44), 640);   // two fetch warps
    EXPECT_EQ(persistThreads(A100, true, 36, 1000000, 0.23, 32, 44), 512);  // one
    EXPECT_EQ(persistThreads(A100, true, 36, 1000000, 0.23, 1, 44), 768);   // at most three
    EXPECT_EQ(persistThreads(A100, true, 36, 1000000, 0.23, 8, 0), 384);    // ops unknown: none
    EXPECT_EQ(persistThreads(A100, true, 36, 1000000, 0.23, 0, 44), 384);   // chunk unknown: none
    EXPECT_EQ(persistThreads(A100, true, 166, 1000000, 0.23, 8, 44), 384); // a ceiling is still the ceiling
}

TEST(Launch, BandCadenceRows)
{
    // h = newly / (live * k) = 0.0022 at k = 100
    EXPECT_EQ(bandK(1.3, 220000, 1000000, 100, 1000, 8), 32);
    const long long mid = bandK(8.4, 220000, 1000000, 100, 1000, 8);
    EXPECT_TRUE(mid == 64 || mid == 128) << mid;
    EXPECT_EQ(bandK(44.0, 220000, 1000000, 100, 1000, 8), 256);
}

TEST(Launch, BandCadenceStaysInsideItsBounds)
{
    EXPECT_EQ(bandK(44.0, 0, 1000, 16, 100, 8), 100);        // nothing finishing: the longest launch
    EXPECT_EQ(bandK(44.0, 220000, 1000000, 100, 100, 8), 100);
    EXPECT_EQ(bandK(0.001, 220000, 1000000, 100, 1000, 8), 8);
    EXPECT_EQ(bandK(44.0, 5, 0, 16, 1000, 8), 8);            // all finished
}

TEST(Launch, BandRateFromTheCostModel)
{
    const double r = bandRate(A100, true, 1000000, 33.0);
    EXPECT_GT(r, 5.0);
    EXPECT_LT(r, 20.0);
    EXPECT_EQ(bandRate({}, true, 1000000, 33.0), 0.0);
}

TEST(Launch, ABoundIsDemandedOnlyAboveTheUnboundedResidency)
{
    EXPECT_FALSE(boundDemanded(640, 1536));
    EXPECT_TRUE(boundDemanded(1536, 1280));
    EXPECT_FALSE(boundDemanded(1536, 0));
    EXPECT_FALSE(boundDemanded(persistSaturation(A100, true), residentThreads(A100.residency, 36)));
}

TEST(Launch, ABoundIsRefusedOnTextGrowthOverFourPercent)
{
    const KernelResources free = { 46, 0, 10000 };
    EXPECT_TRUE(keepBounded(free, { 40, 0, 10000 }, 40));
    EXPECT_TRUE(keepBounded(free, { 40, 0, 10400 }, 40));
    EXPECT_FALSE(keepBounded(free, { 40, 0, 10401 }, 40));
    EXPECT_FALSE(keepBounded(free, { 32, 0, 12000 }, 32));
    EXPECT_TRUE(keepBounded({ 46, 0, 0 }, { 40, 0, 99999 }, 40));      // size unknown: not judged
}

TEST(Launch, ATimedBoundThatIsSlowerIsRefused)
{
    const KernelResources free = { 46, 0, 1000 };
    EXPECT_TRUE(keepBounded(free, { 40, 0, 1000 }, 40, 6.8, 6.7));
    EXPECT_FALSE(keepBounded(free, { 40, 0, 1000 }, 40, 6.8, 7.9));
    EXPECT_TRUE(keepBounded(free, { 40, 0, 1000 }, 40, 0.0, 7.9));   // untimed: static clauses only
}

}  // namespace LaunchTest
}  // namespace eagle_tests
