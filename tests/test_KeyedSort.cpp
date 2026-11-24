// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_KeyedSort.cpp
 * @brief ``eagle::filtering::orderByKeyHost``: the stable permutation by an
 *        integer key and the first place of every key.
 *
 * The reference is ``std::stable_sort`` of the lanes by key, with the starts
 * from a count of the keys. Every integer key type of 8 to 64 bits is covered
 * (the door's planes carry only some of them), across thread counts, and a key
 * outside the bound is counted.
 */

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "eagle/filtering/KeyedSort.h"

#include "TestBase.h"

namespace eagle_tests {
namespace KeyedSortTest {

using eagle::filtering::orderByKeyHost;

/** @brief Check one sort of @p n lanes with keys in ``[0, usable)`` and bound
 * @p bound against the reference. */
template<typename Key>
void expectReference(
    std::int64_t n, std::int64_t usable, std::int64_t bound, unsigned seed)
{
    std::mt19937_64 rng(seed);
    std::vector<Key> keys(static_cast<std::size_t>(n));
    for (auto& k : keys)
        k = static_cast<Key>(rng() % static_cast<std::uint64_t>(usable));
    std::vector<std::int32_t> want(keys.size());
    std::iota(want.begin(), want.end(), 0);
    std::stable_sort(want.begin(), want.end(),
        [&](std::int32_t a, std::int32_t b) { return keys[a] < keys[b]; });
    std::vector<std::uint32_t> wantStarts(
        static_cast<std::size_t>(bound) + 1, 0u);
    for (Key k : keys)
        ++wantStarts[static_cast<std::size_t>(k) + 1];
    for (std::int64_t k = 0; k < bound; ++k)
        wantStarts[k + 1] += wantStarts[k];
    for (int team : { 1, 3, 10 }) {
        std::vector<std::int32_t> order(keys.size(), -1);
        std::vector<std::uint32_t> starts(wantStarts.size(), 7u);
        ASSERT_EQ(orderByKeyHost(
                      keys.data(), n, bound, order.data(), starts.data(), team),
            0);
        EXPECT_EQ(order, want)
            << "n=" << n << " bound=" << bound << " team=" << team;
        EXPECT_EQ(starts, wantStarts)
            << "n=" << n << " bound=" << bound << " team=" << team;
    }
}

TEST(KeyedSort, EveryIntegerKeyTypeEqualsAStableSortAtEveryTeam)
{
    expectReference<std::int8_t>(100000, 100, 100, 1);
    expectReference<std::int8_t>(100000, 128, 1000, 2);
    expectReference<std::uint8_t>(100000, 256, 256, 3);
    expectReference<std::int16_t>(100000, 30000, 30000, 4);
    expectReference<std::uint16_t>(100000, 65536, 70000, 5);
    expectReference<std::int32_t>(100000, 70000, 70000, 6);
    expectReference<std::uint32_t>(100000, 1 << 20, 1 << 20, 7);
    expectReference<std::int64_t>(100000, 5000, 5000, 8);
    expectReference<std::uint64_t>(100000, 70000, 70000, 9);
}

TEST(KeyedSort, SizesAroundAPassAndAChunkEqualAStableSort)
{
    for (std::int64_t n : { 0, 1, 255, 256, 257, 16383, 16384, 16385, 40000 }) {
        expectReference<std::int32_t>(n, 300, 300, 11);
        expectReference<std::int32_t>(n, 1, 1, 12);
    }
}

TEST(KeyedSort, KeysOutsideTheBoundAreCounted)
{
    std::vector<std::int32_t> keys{ 0, 4, 1, -1, 2, 5, 3 };
    std::vector<std::int32_t> order(keys.size());
    std::vector<std::uint32_t> starts(5);
    EXPECT_EQ(
        orderByKeyHost(keys.data(), 7, 4, order.data(), starts.data(), 1), 3);
    std::vector<std::uint64_t> wide{ 3, std::uint64_t{ 1 } << 40 };
    EXPECT_EQ(
        orderByKeyHost(wide.data(), 2, 4, order.data(), starts.data(), 1), 1);
}

} // namespace KeyedSortTest
} // namespace eagle_tests
