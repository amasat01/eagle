// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_Compact.cpp
 * @brief ``eagle::filtering::compactDevice`` / ``compactHost``: the active-set
 *        index map over raw caller buffers.
 *
 * The map is the ascending indices of the kept samples and the count their
 * number, against a reference, for the sizes that hit the scan's edges
 * (empty, one sample, one block, one past a block, multi-level), with a drop
 * mask and a keep mask; slots past the count are untouched. The host face
 * (C++-only mode; the CUDA build runs it in test_Compact.cu beside the device
 * face).
 */

#include <cstdint>
#include <random>
#include <vector>

#include "eagle/eagle.h"

#include "TestBase.h"

namespace eagle_tests {
namespace CompactTest {

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

class CompactSizes : public Test, public ::testing::WithParamInterface<std::int64_t> { };

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

TEST_F(CompactGraph, RefusesMoreThanInt32Samples)
{
    EXPECT_THROW(compactHost(nullptr, 0u, nullptr, nullptr, nullptr, std::int64_t { 1 } << 31),
        std::invalid_argument);
    EXPECT_THROW(compactHost(nullptr, 0u, nullptr, nullptr, nullptr, -1), std::invalid_argument);
}

} // namespace CompactTest
} // namespace eagle_tests
