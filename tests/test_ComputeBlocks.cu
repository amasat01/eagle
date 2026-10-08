// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Tests for ``eagle::cuda::computeBlocks``. Verifies the policy
 * invariants (WARP lower bound, idealBlockSize upper bound,
 * multiple-of-WARP rounding, nBlocks formula) on nSMs-agnostic
 * boundary regimes. */
#include "TestBase.h"

#include "eagle/cuda/ComputeBlocks.h"
#include "eagle/launch/Traits.h"

#ifndef EAGLE_CPU_ONLY

namespace eagle_tests {
namespace ComputeBlocksTest {

using namespace eagle;
using eagle::cuda::computeBlocks;
using eagle::cuda::currentSMCount;

/* WARP lower bound for tiny N: gridSize × WARP dwarfs N → clamp to 32. */
TEST(ComputeBlocksTest, WarpLowerBoundForTinyN)
{
    idx_t blockSize = 0, nBlocks = 0;
    computeBlocks(/*numStates=*/1, nBlocks, blockSize,
        /*idealBlockSize=*/256);
    EXPECT_EQ(blockSize, 32);
    EXPECT_EQ(nBlocks, 1);

    computeBlocks(/*numStates=*/31, nBlocks, blockSize,
        /*idealBlockSize=*/256);
    EXPECT_EQ(blockSize, 32);
    EXPECT_EQ(nBlocks, 1);

    computeBlocks(/*numStates=*/32, nBlocks, blockSize,
        /*idealBlockSize=*/256);
    EXPECT_EQ(blockSize, 32);
    EXPECT_EQ(nBlocks, 1);

    computeBlocks(/*numStates=*/33, nBlocks, blockSize,
        /*idealBlockSize=*/256);
    EXPECT_EQ(blockSize, 32);
    EXPECT_EQ(nBlocks, 2);
}

/* For huge N the policy clamps to idealBlockSize. */
TEST(ComputeBlocksTest, IdealBlockSizeUpperBoundForHugeN)
{
    constexpr idx_t huge = idx_t{ 1 } << 24; /* ~16M */
    idx_t blockSize = 0, nBlocks = 0;

    computeBlocks(huge, nBlocks, blockSize, /*idealBlockSize=*/64);
    EXPECT_EQ(blockSize, 64);
    EXPECT_EQ(nBlocks, (huge + 63) / 64);

    computeBlocks(huge, nBlocks, blockSize, /*idealBlockSize=*/128);
    EXPECT_EQ(blockSize, 128);
    EXPECT_EQ(nBlocks, (huge + 127) / 128);

    computeBlocks(huge, nBlocks, blockSize, /*idealBlockSize=*/256);
    EXPECT_EQ(blockSize, 256);
    EXPECT_EQ(nBlocks, (huge + 255) / 256);
}

/* blockSize is always a multiple of WARP=32, regardless of cap. */
TEST(ComputeBlocksTest, BlockSizeIsMultipleOfWarp)
{
    constexpr idx_t WARP = 32;
    const std::vector<idx_t> Ns
        = { 1, 32, 33, 64, 100, 256, 257, 1024, 100'000, 1'000'000 };
    const std::vector<idx_t> caps = { 32, 64, 128, 256, 512, 1024 };
    for (idx_t N : Ns) {
        for (idx_t cap : caps) {
            idx_t bs = 0, nb = 0;
            computeBlocks(N, nb, bs, cap);
            EXPECT_EQ(bs % WARP, 0u)
                << "N=" << N << " cap=" << cap << " bs=" << bs;
            EXPECT_GE(bs, 32u);
            EXPECT_LE(bs, cap);
            /* nBlocks formula */
            EXPECT_EQ(nb, (N + bs - 1) / bs);
        }
    }
}

/* blockSize is non-decreasing in N (nSMs-agnostic check). */
TEST(ComputeBlocksTest, BlockSizeMonotoneInN)
{
    constexpr idx_t cap = 256;
    idx_t prevBs = 0;
    for (idx_t N : { idx_t(1), idx_t(100), idx_t(1000), idx_t(10000),
             idx_t(100'000), idx_t(1'000'000) }) {
        idx_t bs = 0, nb = 0;
        computeBlocks(N, nb, bs, cap);
        EXPECT_GE(bs, prevBs) << "blockSize must be non-decreasing in N (N="
                              << N << ")";
        prevBs = bs;
    }
}

/* currentSMCount returns a positive value, consistent across calls. */
TEST(ComputeBlocksTest, SMCountIsPositiveAndCached)
{
    int n1 = currentSMCount();
    int n2 = currentSMCount();
    EXPECT_GT(n1, 0);
    EXPECT_EQ(n1, n2);
}

/* The Traits-taking overload is pure forwarding: <maxBlockSize,
 * minBlocksPerSM> → (idealBlockSize, blocksPerSM). Pinned by equivalence
 * against explicit-argument calls, with the REAL eagle::launch::Traits and
 * a local trait-shaped struct (the overload is duck-typed on the shape). */
/* Local classes may not declare static data members — namespace scope. */
struct Local128x8 {
    static constexpr idx_t maxBlockSize   = 128;
    static constexpr idx_t minBlocksPerSM = 8;
};

TEST(ComputeBlocksTest, TraitsOverloadMatchesExplicitArguments)
{
    const std::vector<idx_t> Ns
        = { 1, 33, 4096, 100'000, idx_t{ 1 } << 24 };
    for (idx_t N : Ns) {
        idx_t bsT = 0, nbT = 0, bsV = 0, nbV = 0;

        computeBlocks<eagle::launch::Traits<256, 4>>(N, nbT, bsT);
        computeBlocks(N, nbV, bsV, /*idealBlockSize=*/256, /*blocksPerSM=*/4);
        EXPECT_EQ(bsT, bsV) << "Traits<256,4> N=" << N;
        EXPECT_EQ(nbT, nbV) << "Traits<256,4> N=" << N;

        computeBlocks<Local128x8>(N, nbT, bsT);
        computeBlocks(N, nbV, bsV, /*idealBlockSize=*/128, /*blocksPerSM=*/8);
        EXPECT_EQ(bsT, bsV) << "Local128x8 N=" << N;
        EXPECT_EQ(nbT, nbV) << "Local128x8 N=" << N;
    }
}

TEST(ComputeBlocksTest, TraitsOverloadZeroStates)
{
    idx_t blockSize = 7, nBlocks = 7;
    computeBlocks<eagle::launch::Traits<256, 4>>(0, nBlocks, blockSize);
    EXPECT_EQ(nBlocks, 0);
    EXPECT_EQ(blockSize, 32);
}

TEST(ComputeBlocksTest, Fp64PerfRatioIsSaneAndCached)
{
    const int r1 = eagle::cuda::currentFp64PerfRatio();
    const int r2 = eagle::cuda::currentFp64PerfRatio();
    // Known parts range from 2 (data-center) to 64 (consumer Ampere/Ada);
    // sane-window assertion, not a value pin — this must pass on any GPU.
    EXPECT_GE(r1, 1);
    EXPECT_LE(r1, 128);
    EXPECT_EQ(r1, r2);
    printf("[ INFO ] device FP32:FP64 throughput ratio = %d:1\n", r1);
}

} // namespace ComputeBlocksTest
} // namespace eagle_tests

#endif
