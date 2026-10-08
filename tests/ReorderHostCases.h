// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file ReorderHostCases.h
 * @brief The host-face rows of ``eagle::filtering::reorderHost`` /
 *        ``restoreHost`` and the compaction's reorder trigger, shared by the
 *        C++-only build (test_Reorder.cpp) and the CUDA build (test_Reorder.cu,
 *        beside the device face).
 *
 * Rows: after a reorder every plane holds sample ``i`` at ``inv[i]`` (planes of
 * every supported element size, the mask among them), the kept samples fill
 * ``[0, count)`` in ascending sample order, ``perm`` and ``inv`` are inverse,
 * the map is the identity and ``span == count``; two reorders over a thinning
 * mask, then a restore, give back the original planes and identity indices; a
 * ``fire`` word of 0 leaves everything untouched; the trigger fires on a
 * random live set and stays silent on a grouped one; the refusals.
 */

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "eagle/eagle.h"

#include "TestBase.h"

namespace eagle_tests {
namespace ReorderTest {

using eagle::filtering::compactHost;
using eagle::filtering::compactScratchBytes;
using eagle::filtering::kCompactMaskIsDrop;
using eagle::filtering::ReorderPlane;
using eagle::filtering::ReorderTrigger;
using eagle::filtering::reorderHost;
using eagle::filtering::reorderScratchBytes;
using eagle::filtering::restoreHost;

/** @brief A 16-byte element with a value per sample. */
struct alignas(16) Elem16 {
    std::uint64_t a, b;
    bool operator==(const Elem16& o) const { return a == o.a && b == o.b; }
};

/** @brief One batch: a drop mask and one plane per element size, each
 *  filled with a value that names its sample. */
struct HostBatch {
    std::int64_t n;
    std::vector<std::uint8_t> mask;
    std::vector<std::uint16_t> p2;
    std::vector<std::uint32_t> p4;
    std::vector<double> p8;
    std::vector<Elem16> p16;
    std::vector<std::int32_t> perm, inv, map;
    std::uint32_t count = 0, span = 0, fire = 1;
    std::vector<std::uint8_t> scratchBytes;
    void* scratch = nullptr;

    explicit HostBatch(std::int64_t n_)
        : n(n_)
        , mask(std::size_t(n), 0)
        , p2(std::size_t(n))
        , p4(std::size_t(n))
        , p8(std::size_t(n))
        , p16(std::size_t(n))
        , perm(std::size_t(n))
        , inv(std::size_t(n))
        , map(std::size_t(n))
        , span(std::uint32_t(n))
        , scratchBytes(reorderScratchBytes(n, 16) + 16)
    {
        for (std::int64_t i = 0; i < n; ++i) {
            p2[std::size_t(i)]  = std::uint16_t(i * 7 + 1);
            p4[std::size_t(i)]  = std::uint32_t(i * 13 + 5);
            p8[std::size_t(i)]  = 0.5 + double(i);
            p16[std::size_t(i)] = Elem16 { std::uint64_t(i), ~std::uint64_t(i) };
        }
        std::iota(perm.begin(), perm.end(), 0);
        std::iota(inv.begin(), inv.end(), 0);
        std::iota(map.begin(), map.end(), 0);
        count = std::uint32_t(n);
        auto addr = reinterpret_cast<std::uintptr_t>(scratchBytes.data());
        scratch   = scratchBytes.data() + ((16 - addr % 16) % 16);
    }
    std::array<ReorderPlane, 5> planes()
    {
        return { ReorderPlane { p2.data(), 2 }, ReorderPlane { mask.data(), 1 },
            ReorderPlane { p4.data(), 4 }, ReorderPlane { p8.data(), 8 },
            ReorderPlane { p16.data(), 16 } };
    }
    void reorder()
    {
        auto pl = planes();
        reorderHost(pl.data(), std::uint32_t(pl.size()), mask.data(), kCompactMaskIsDrop,
            perm.data(), inv.data(), map.data(), &count, &span, &fire, scratch, n);
    }
    void restore()
    {
        auto pl = planes();
        restoreHost(pl.data(), std::uint32_t(pl.size()), perm.data(), inv.data(), scratch, n);
    }
    /** Drop (in sample order) every sample @p drop marks, wherever it sits. */
    void finish(const std::vector<std::uint8_t>& drop)
    {
        for (std::int64_t i = 0; i < n; ++i)
            if (drop[std::size_t(i)])
                mask[std::size_t(inv[std::size_t(i)])] = 1;
    }
};

inline std::vector<std::uint8_t> randomDrop(std::int64_t n, double p, unsigned seed)
{
    std::mt19937 rng(seed);
    std::bernoulli_distribution d(p);
    std::vector<std::uint8_t> m = std::vector<std::uint8_t>(std::size_t(n));
    for (auto& b : m)
        b = d(rng) ? 1 : 0;
    return m;
}

/** @brief Every structural property of a batch right after a reorder, against
 *  the sample-order drop set @p dropped. */
inline void expectReordered(const HostBatch& b, const std::vector<std::uint8_t>& dropped)
{
    const std::int64_t n = b.n;
    std::int64_t kept    = 0;
    for (auto d : dropped)
        kept += d ? 0 : 1;
    ASSERT_EQ(b.count, std::uint32_t(kept));
    ASSERT_EQ(b.span, b.count);
    ASSERT_EQ(b.fire, 0u);
    for (std::int64_t i = 0; i < n; ++i) {
        const auto s = std::size_t(b.inv[std::size_t(i)]);
        ASSERT_EQ(std::size_t(b.perm[s]), std::size_t(i)) << "perm[inv[" << i << "]]";
        ASSERT_EQ(b.p2[s], std::uint16_t(i * 7 + 1)) << "sample " << i;
        ASSERT_EQ(b.p4[s], std::uint32_t(i * 13 + 5)) << "sample " << i;
        ASSERT_EQ(b.p8[s], 0.5 + double(i)) << "sample " << i;
        ASSERT_TRUE((b.p16[s] == Elem16 { std::uint64_t(i), ~std::uint64_t(i) })) << "sample " << i;
        ASSERT_EQ(b.mask[s], dropped[std::size_t(i)]) << "sample " << i;
    }
    for (std::int64_t t = 0; t < kept; ++t) {
        ASSERT_EQ(b.map[std::size_t(t)], std::int32_t(t));
        ASSERT_EQ(b.mask[std::size_t(t)], 0) << "slot " << t << " under the count is dropped";
        if (t > 0) {
            ASSERT_LT(b.perm[std::size_t(t - 1)], b.perm[std::size_t(t)]) << "kept order";
        }
    }
}

class ReorderHostSizes : public Test, public ::testing::WithParamInterface<std::int64_t> { };

TEST_P(ReorderHostSizes, KeptFirstAndEveryPlaneFollowsTheIndex)
{
    const std::int64_t n = GetParam();
    HostBatch b(n);
    const auto drop1 = randomDrop(n, 0.5, 3u + unsigned(n));
    b.finish(drop1);
    b.reorder();
    expectReordered(b, drop1);
    // thin further (monotone) and reorder again: only [0, span) moves
    auto drop2 = randomDrop(n, 0.5, 9u + unsigned(n));
    for (std::size_t i = 0; i < drop2.size(); ++i)
        drop2[i] = std::uint8_t(drop2[i] | drop1[i]);
    b.finish(drop2);
    b.fire = 1;
    b.reorder();
    expectReordered(b, drop2);
    b.restore();
    HostBatch ref(n);
    for (std::int64_t i = 0; i < n; ++i) {
        const auto s = std::size_t(i);
        ASSERT_EQ(b.perm[s], std::int32_t(i));
        ASSERT_EQ(b.inv[s], std::int32_t(i));
        ASSERT_EQ(b.p2[s], ref.p2[s]);
        ASSERT_EQ(b.p4[s], ref.p4[s]);
        ASSERT_EQ(b.p8[s], ref.p8[s]);
        ASSERT_TRUE(b.p16[s] == ref.p16[s]);
        ASSERT_EQ(b.mask[s], drop2[s]);
    }
}

INSTANTIATE_TEST_SUITE_P(Edges, ReorderHostSizes,
    ::testing::Values(std::int64_t { 0 }, std::int64_t { 1 }, std::int64_t { 7 },
        std::int64_t { 256 }, std::int64_t { 257 }, std::int64_t { 100003 }));

class ReorderHost : public Test { };

TEST_F(ReorderHost, ClearFireIsANoOp)
{
    HostBatch b(1000);
    b.finish(randomDrop(1000, 0.5, 1u));
    b.fire              = 0;
    const auto before   = b.p8;
    const auto maskWas  = b.mask;
    b.reorder();
    EXPECT_EQ(b.p8, before);
    EXPECT_EQ(b.mask, maskWas);
    EXPECT_EQ(b.span, 1000u);
    EXPECT_EQ(b.count, 1000u);
}

/** @brief The trigger a host compaction of @p mask computes (drop mask). */
inline std::uint32_t hostFire(const std::vector<std::uint8_t>& mask, std::uint32_t span, float theta,
    std::uint32_t* live32Out = nullptr)
{
    const auto n = std::int64_t(mask.size());
    std::vector<std::int32_t> map = std::vector<std::int32_t>(std::size_t(n));
    std::vector<std::uint8_t> scratch(compactScratchBytes(n));
    std::uint32_t count = 0, live32 = 77, fire = 77;
    const ReorderTrigger trig { &live32, &span, &fire, theta };
    compactHost(mask.data(), kCompactMaskIsDrop, map.data(), &count, scratch.data(), n, &trig);
    if (live32Out != nullptr)
        *live32Out = live32;
    return fire;
}

TEST_F(ReorderHost, TriggerFiresOnRandomOrderAndNotOnGroupedOrder)
{
    const std::int64_t n = 100000;
    const auto random    = randomDrop(n, 0.8, 5u); // ~20 % live, scattered
    std::vector<std::uint8_t> grouped(std::size_t(n), 1);
    for (std::int64_t i = 0; i < n / 5; ++i)
        grouped[std::size_t(i)] = 0; // the same live fraction, contiguous
    std::uint32_t live32 = 0;
    EXPECT_EQ(hostFire(random, std::uint32_t(n), 0.5f, &live32), 1u);
    EXPECT_GT(live32, std::uint32_t((n + 31) / 32 * 99 / 100)); // nearly every group is occupied
    EXPECT_EQ(hostFire(grouped, std::uint32_t(n), 0.5f, &live32), 0u);
    EXPECT_EQ(live32, std::uint32_t((n / 5 + 31) / 32));
    EXPECT_EQ(hostFire(std::vector<std::uint8_t>(std::size_t(n), 0), std::uint32_t(n), 0.75f), 0u);
    // nothing to gain once the span is down to the live count
    std::uint32_t live = 0;
    for (auto d : random)
        live += d ? 0u : 1u;
    EXPECT_EQ(hostFire(random, live, 0.5f), 0u);
    EXPECT_EQ(hostFire(random, live + 1, 0.5f), 1u);
    // nor under the span floor
    const std::vector<std::uint8_t> small(random.begin(), random.begin() + 4095);
    EXPECT_EQ(hostFire(small, 4095u, 0.5f), 0u);
    const std::vector<std::uint8_t> floor(random.begin(), random.begin() + 4096);
    EXPECT_EQ(hostFire(floor, 4096u, 0.5f), 1u);
}

TEST_F(ReorderHost, Refusals)
{
    HostBatch b(64);
    std::uint32_t s = 64, f = 1;
    void* sc        = b.scratch;
    auto call       = [&](ReorderPlane* p, std::uint32_t np) {
        reorderHost(p, np, b.mask.data(), kCompactMaskIsDrop, b.perm.data(), b.inv.data(),
            b.map.data(), &b.count, &s, &f, sc, 64);
    };
    ReorderPlane bad[2] = { { b.mask.data(), 1 }, { b.p8.data(), 3 } };
    EXPECT_THROW(call(bad, 2), std::invalid_argument); // element size
    ReorderPlane noMask[1] = { { b.p8.data(), 8 } };
    EXPECT_THROW(call(noMask, 1), std::invalid_argument); // the mask is not a plane
    ReorderPlane alias[3] = { { b.mask.data(), 1 }, { b.p8.data(), 8 },
        { reinterpret_cast<char*>(b.p8.data()) + 256, 8 } };
    EXPECT_THROW(call(alias, 3), std::invalid_argument); // two planes overlap
    ReorderPlane unaligned[2] = { { b.mask.data(), 1 },
        { reinterpret_cast<char*>(b.p8.data()) + 4, 8 } };
    EXPECT_THROW(call(unaligned, 2), std::invalid_argument);
    std::uint32_t live32 = 0, fire = 0, span = 64;
    std::vector<std::int32_t> map(64);
    std::vector<std::uint8_t> scratch(compactScratchBytes(64));
    const ReorderTrigger noWords { nullptr, &span, &fire, 0.5f };
    EXPECT_THROW(compactHost(b.mask.data(), 1u, map.data(), &b.count, scratch.data(), 64, &noWords),
        std::invalid_argument);
    const ReorderTrigger badTheta { &live32, &span, &fire, 1.5f };
    EXPECT_THROW(compactHost(b.mask.data(), 1u, map.data(), &b.count, scratch.data(), 64, &badTheta),
        std::invalid_argument);
}

} // namespace ReorderTest
} // namespace eagle_tests
