// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_Compact.cu
 * @brief ``eagle::filtering::compactDevice`` / ``compactHost``: the active-set
 *        index map over raw caller buffers.
 *
 * The map is the ascending indices of the kept samples and the count their
 * number, against a host reference, for the sizes that hit the scan's edges
 * (empty, one sample, one block, one past a block, multi-level), with a drop
 * mask and a keep mask; slots past the count are untouched. The device face is
 * also CAPTURED into a graph and replayed over a mask changed between replays
 * (RED: a face that read the mask on the host, or baked the first answer into
 * the graph, returns the first map on the second replay).
 */

#include <cuda_runtime_api.h>

#include <cstdint>
#include <random>
#include <vector>

#include "eagle/eagle.h"

#include "TestBase.h"

namespace eagle_tests {
namespace CompactTest {

using eagle::filtering::compactDevice;
using eagle::filtering::compactHost;
using eagle::filtering::compactScratchBytes;
using eagle::filtering::kCompactMaskIsDrop;

std::vector<std::uint8_t> randomMask(std::int64_t n, unsigned seed)
{
    std::mt19937 rng(seed);
    std::bernoulli_distribution keep(0.6);
    std::vector<std::uint8_t> m(static_cast<std::size_t>(n));
    for (auto& b : m)
        b = keep(rng) ? 1 : 0;
    return m;
}

std::vector<std::int32_t> reference(const std::vector<std::uint8_t>& m, bool drop)
{
    std::vector<std::int32_t> out;
    for (std::size_t i = 0; i < m.size(); ++i)
        if ((m[i] != 0) != drop)
            out.push_back(static_cast<std::int32_t>(i));
    return out;
}

/** @brief Device buffers for one compaction of @p n samples. */
struct DeviceBufs {
    std::int64_t n;
    std::uint8_t* mask = nullptr;
    std::int32_t* map = nullptr;
    std::uint32_t* count = nullptr;
    void* scratch = nullptr;

    explicit DeviceBufs(std::int64_t n_)
        : n(n_)
    {
        const std::size_t m = static_cast<std::size_t>(n > 0 ? n : 1);
        EXPECT_EQ(cudaMalloc(&mask, m), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&map, m * sizeof(std::int32_t)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&count, sizeof(std::uint32_t)), cudaSuccess);
        EXPECT_EQ(cudaMalloc(&scratch, compactScratchBytes(n)), cudaSuccess);
        EXPECT_EQ(cudaMemset(map, 0xff, m * sizeof(std::int32_t)), cudaSuccess);
    }
    ~DeviceBufs()
    {
        cudaFree(mask);
        cudaFree(map);
        cudaFree(count);
        cudaFree(scratch);
    }
    void setMask(const std::vector<std::uint8_t>& m)
    {
        if (n > 0)
            EXPECT_EQ(cudaMemcpy(mask, m.data(), m.size(), cudaMemcpyHostToDevice), cudaSuccess);
        // The copy (and the constructor's memset) ride the per-thread default
        // stream, which has no ordering with a replay on another stream.
        EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    }
    std::uint32_t readCount() const
    {
        std::uint32_t c = 0;
        EXPECT_EQ(cudaMemcpy(&c, count, sizeof c, cudaMemcpyDeviceToHost), cudaSuccess);
        return c;
    }
    std::vector<std::int32_t> readMap() const
    {
        std::vector<std::int32_t> out(static_cast<std::size_t>(n));
        if (n > 0)
            EXPECT_EQ(cudaMemcpy(out.data(), map, out.size() * sizeof(std::int32_t),
                          cudaMemcpyDeviceToHost),
                cudaSuccess);
        return out;
    }
};

class CompactSizes : public Test, public ::testing::WithParamInterface<std::int64_t> { };

TEST_P(CompactSizes, DeviceMatchesReference)
{
    const std::int64_t n = GetParam();
    for (const bool drop : { true, false }) {
        const auto m = randomMask(n, 7u + unsigned(n));
        DeviceBufs b(n);
        b.setMask(m);
        compactDevice(b.mask, drop ? kCompactMaskIsDrop : 0u, b.map, b.count, b.scratch, n, 0);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        const auto want = reference(m, drop);
        ASSERT_EQ(b.readCount(), want.size());
        const auto got = b.readMap();
        for (std::size_t i = 0; i < want.size(); ++i)
            ASSERT_EQ(got[i], want[i]) << "slot " << i;
        for (std::size_t i = want.size(); i < got.size(); ++i)
            ASSERT_EQ(got[i], -1) << "slot " << i << " past the count was written";
    }
}

TEST_P(CompactSizes, HostMatchesReference)
{
    const std::int64_t n = GetParam();
    for (const bool drop : { true, false }) {
        const auto m = randomMask(n, 11u + unsigned(n));
        std::vector<std::int32_t> map(static_cast<std::size_t>(n), -1);
        std::vector<eagle::idx_t> scratch(compactScratchBytes(n) / sizeof(eagle::idx_t));
        void* s = scratch.data();
        std::uint32_t count = 99;
        compactHost(m.data(), drop ? kCompactMaskIsDrop : 0u, map.data(), &count, s, n);
        const auto want = reference(m, drop);
        ASSERT_EQ(count, want.size());
        for (std::size_t i = 0; i < want.size(); ++i)
            ASSERT_EQ(map[i], want[i]) << "slot " << i;
        for (std::size_t i = want.size(); i < map.size(); ++i)
            ASSERT_EQ(map[i], -1);
    }
}

INSTANTIATE_TEST_SUITE_P(Edges, CompactSizes,
    ::testing::Values(std::int64_t { 0 }, std::int64_t { 1 }, std::int64_t { 7 },
        std::int64_t { 256 }, std::int64_t { 257 }, std::int64_t { 100003 }));

class CompactGraph : public Test { };

TEST_F(CompactGraph, CapturedReplayFollowsTheMask)
{
    const std::int64_t n = 5000;
    DeviceBufs b(n);
    cudaStream_t s = nullptr;
    ASSERT_EQ(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
    ASSERT_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    compactDevice(b.mask, kCompactMaskIsDrop, b.map, b.count, b.scratch, n, s);
    cudaGraph_t g = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(s, &g), cudaSuccess);
    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&exec, g, 0), cudaSuccess);
    for (unsigned seed : { 1u, 2u, 3u }) {
        const auto m = randomMask(n, seed);
        b.setMask(m);
        ASSERT_EQ(cudaGraphLaunch(exec, s), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        const auto want = reference(m, true);
        ASSERT_EQ(b.readCount(), want.size()) << "replay with seed " << seed;
        const auto got = b.readMap();
        for (std::size_t i = 0; i < want.size(); ++i)
            ASSERT_EQ(got[i], want[i]);
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
}

TEST_F(CompactGraph, RefusesMoreThanInt32Samples)
{
    EXPECT_THROW(compactHost(nullptr, 0u, nullptr, nullptr, nullptr, std::int64_t { 1 } << 31),
        std::invalid_argument);
    EXPECT_THROW(compactHost(nullptr, 0u, nullptr, nullptr, nullptr, -1), std::invalid_argument);
}

} // namespace CompactTest
} // namespace eagle_tests
