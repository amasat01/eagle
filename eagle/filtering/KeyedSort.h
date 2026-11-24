// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/exec/HostPool.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace eagle {
namespace filtering {

/**
 * @file KeyedSort.h
 * @brief Host stable sort of the lanes of an integer key plane: the
 *        permutation and the first place of every key.
 *
 * A chunked LSD radix sort, 8 bits of the key per pass and only the passes
 * the key bound needs. Every pass cuts the lanes, in their current order,
 * into contiguous chunks; each chunk counts its digits, the counters are
 * prefixed digit-major then chunk (which keeps equal digits in lane order),
 * and each chunk places its own lanes. No atomics. The stable permutation is
 * one value, so it is the same for every chunk count and thread count.
 * Host only: nothing here needs the launch machinery.
 */

namespace detail {

/** @brief The fewest lanes a chunk of a pass takes: below it a chunk costs more
 * in counters than it saves. */
inline constexpr std::int64_t kKeyedSortChunkLanes = 1 << 14;
/** @brief Bits of the key one pass sorts. */
inline constexpr unsigned kKeyedSortDigitBits = 8;
/** @brief The buckets of one pass. */
inline constexpr unsigned kKeyedSortBuckets = 1u << kKeyedSortDigitBits;

/** @brief Whether @p key lies in ``[0, bound)``. */
template<typename Key>
inline bool keyedSortInRange(Key key, std::uint64_t bound)
{
    if constexpr (std::is_signed_v<Key>) {
        if (key < 0)
            return false;
    }
    return std::uint64_t(key) < bound;
}

/** @brief The passes a key of type @p Key below @p bound needs: one per 8 bits
 * of the largest valid key, and at least one. */
template<typename Key>
inline unsigned keyedSortPasses(std::int64_t bound)
{
    const std::uint64_t top = std::min<std::uint64_t>(std::uint64_t(bound - 1),
        std::uint64_t(std::numeric_limits<Key>::max()));
    unsigned bits           = 0;
    for (std::uint64_t v = top; v != 0; v >>= 1)
        ++bits;
    return std::max(
        1u, (bits + kKeyedSortDigitBits - 1u) / kKeyedSortDigitBits);
}

/** @brief The digit of @p key that the pass with bit offset @p shift sorts. */
template<typename Key>
inline std::size_t keyedSortDigit(Key key, unsigned shift)
{
    return std::size_t(
        (std::uint64_t(key) >> shift) & (kKeyedSortBuckets - 1u));
}

/**
 * @brief One stable pass over the keys ``keyIn[0, n)``, which are in the order
 *        the previous pass left them; ``orderIn[r]`` is the lane of the r-th
 *        (the lanes themselves in order when @p orderIn is null).
 *
 * The lanes are placed into @p orderOut by the digit at @p shift, and the keys
 * into @p keyOut too when it is not null, so the next pass reads both in
 * sequence. @p count holds ``chunks * 256`` counters. The first pass also
 * counts the keys outside ``[0, bound)`` (@p checkBound) and returns their
 * number, leaving the outputs untouched; when @p starts is not null (a pass
 * that covers all of ``bound``) it receives the first place of each key.
 */
template<typename Key>
inline std::int64_t keyedSortPass(const Key* keyIn, const std::int32_t* orderIn,
    Key* keyOut, std::int32_t* orderOut, std::int64_t n, std::int64_t chunks,
    unsigned shift, std::vector<std::uint32_t>& count, int team,
    bool checkBound, std::int64_t bound, std::uint32_t* starts)
{
    std::vector<std::int64_t> bad(std::size_t(chunks), 0);
    std::fill(count.begin(), count.end(), 0u);
    exec::hostFor(chunks, team, [&](std::int64_t c) {
        std::uint32_t* mine  = count.data() + c * kKeyedSortBuckets;
        std::int64_t outside = 0;
        for (std::int64_t r = n * c / chunks; r < n * (c + 1) / chunks; ++r) {
            const Key key = keyIn[r];
            outside
                += checkBound && !keyedSortInRange(key, std::uint64_t(bound));
            ++mine[keyedSortDigit(key, shift)];
        }
        bad[std::size_t(c)] = outside;
    });
    std::int64_t outside = 0;
    for (std::int64_t b : bad)
        outside += b;
    if (outside != 0)
        return outside;
    // Digit-major, then chunk: equal digits land in lane order.
    std::uint32_t start = 0;
    for (std::size_t digit = 0; digit < kKeyedSortBuckets; ++digit) {
        for (std::int64_t c = 0; c < chunks; ++c) {
            std::uint32_t& slot
                = count[std::size_t(c) * kKeyedSortBuckets + digit];
            const std::uint32_t here = slot;
            slot                     = start;
            start += here;
        }
    }
    if (starts != nullptr) {
        // The first chunk's start of a digit is the digit's first place; a
        // digit no key reaches starts where the sorted lanes end.
        for (std::int64_t k = 0; k <= bound; ++k)
            starts[k] = k < std::int64_t(kKeyedSortBuckets)
                ? count[std::size_t(k)]
                : std::uint32_t(n);
    }
    exec::hostFor(chunks, team, [&](std::int64_t c) {
        std::uint32_t* mine = count.data() + c * kKeyedSortBuckets;
        for (std::int64_t r = n * c / chunks; r < n * (c + 1) / chunks; ++r) {
            const Key key          = keyIn[r];
            const std::uint32_t at = mine[keyedSortDigit(key, shift)]++;
            orderOut[at] = orderIn != nullptr ? orderIn[r] : std::int32_t(r);
            if (keyOut != nullptr)
                keyOut[at] = key;
        }
    });
    return 0;
}

} // namespace detail

/**
 * @brief Stable sort of ``n`` lanes by ``keys``, every key in ``[0, bound)``,
 *        on at most @p team host threads.
 *
 * ``order[j]`` is the lane that lands at sorted place j (equal keys keep lane
 * order); ``starts[k]`` is the first sorted place of key k for k in
 * ``[0, bound]``, so ``starts[bound] == n``. Needs ``bound`` of 1 or more and
 * ``n``, ``bound`` of at most 2^31 - 1. Returns 0, or the number of keys
 * outside ``[0, bound)``, in which case ``order`` and ``starts`` are left
 * unspecified. Working memory is 256 counters per chunk, and when more than
 * one pass is needed one lane-index buffer and two key buffers.
 */
template<typename Key>
inline std::int64_t orderByKeyHost(const Key* keys, std::int64_t n,
    std::int64_t bound, std::int32_t* order, std::uint32_t* starts,
    int team = exec::host_max_threads())
{
    static_assert(std::is_integral_v<Key>, "keys are integers");
    if (n == 0) {
        std::fill(starts, starts + bound + 1, 0u);
        return 0;
    }
    const std::int64_t chunks = std::clamp<std::int64_t>(
        n / detail::kKeyedSortChunkLanes, 1, std::max(team, 1));
    const unsigned passes = detail::keyedSortPasses<Key>(bound);
    std::vector<std::uint32_t> count(
        std::size_t(chunks) * detail::kKeyedSortBuckets);
    // One pass covers all of ``bound`` and gives ``starts`` from its counts.
    // More passes carry the keys with the lanes, so that every pass and the
    // walk for ``starts`` read them in sequence instead of gathering.
    std::vector<std::int32_t> spare(passes > 1 ? std::size_t(n) : 0);
    std::vector<Key> keyBuffer[2];
    if (passes > 1)
        for (auto& buffer : keyBuffer)
            buffer.resize(std::size_t(n));
    // The last pass writes ``order``, the one before it the spare buffer, and
    // so on back; the keys alternate between their two buffers.
    const Key* keyIn         = keys;
    const std::int32_t* from = nullptr;
    for (unsigned pass = 0; pass < passes; ++pass) {
        std::int32_t* to
            = (passes - 1u - pass) % 2u == 0u ? order : spare.data();
        Key* keyOut = passes > 1 ? keyBuffer[pass % 2u].data() : nullptr;
        const std::int64_t outside = detail::keyedSortPass(keyIn, from, keyOut,
            to, n, chunks, pass * detail::kKeyedSortDigitBits, count, team,
            pass == 0, bound, passes == 1 ? starts : nullptr);
        if (outside != 0)
            return outside;
        from  = to;
        keyIn = keyOut;
    }
    if (passes == 1)
        return 0;
    // ``keyIn`` now holds the keys in sorted order. Each key k below the last
    // gets its place from the one lane where the sorted keys step past k.
    exec::hostFor(chunks, team, [&](std::int64_t c) {
        const std::int64_t first = n * c / chunks, end = n * (c + 1) / chunks;
        std::int64_t previous
            = first == 0 ? -1 : std::int64_t(keyIn[first - 1]);
        for (std::int64_t j = first; j < end; ++j) {
            const std::int64_t key = std::int64_t(keyIn[j]);
            for (std::int64_t k = previous + 1; k <= key; ++k)
                starts[k] = std::uint32_t(j);
            previous = key;
        }
        if (c == chunks - 1)
            for (std::int64_t k = previous + 1; k <= bound; ++k)
                starts[k] = std::uint32_t(n);
    });
    return 0;
}

} // namespace filtering
} // namespace eagle
