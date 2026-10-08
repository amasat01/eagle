// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cpu/Scan.h"
#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/throw.h"

#include <stdexcept>

#ifndef EAGLE_CPU_ONLY
#include "eagle/cuda/Scan.h"
#endif

#include <cstddef>
#include <cstdint>

namespace eagle {
namespace filtering {

/**
 * @file Compact.h
 * @brief Active-set compaction over caller memory: a per-sample u8 mask in,
 *        the ascending index map of the kept samples and their count out.
 *
 * The same predicate -> scan -> scatter as ``FilteringSliceNode``, but over raw
 * caller buffers rather than a sliced object, so a seam entry (or any caller
 * holding only addresses) can run it:
 *
 * - ``mask``      ``n`` bytes, one per sample. With ``kCompactMaskIsDrop`` a set
 *                 byte DROPS the sample (a ``terminated`` mask); without it a
 *                 set byte KEEPS it.
 * - ``indexMap``  ``n`` ``int32`` slots; on return ``indexMap[0, count)`` holds
 *                 the kept sample indices in ASCENDING order (the order keeps a
 *                 gather through the map as coalesced as the live set allows).
 *                 Slots past ``count`` are left as they were.
 * - ``count``     one ``uint32`` word, the number of kept samples.
 * - ``scratch``   ``compactScratchBytes(n)`` bytes of working memory.
 *
 * The device face only ENQUEUES on its stream: nothing is allocated, nothing is
 * synchronised and nothing is read back to the host, so it is legal inside a
 * stream capture. The scan is the inclusive ``cuda::Scan`` (kernels only — no
 * copy node). The host face runs the OpenMP twins (``cpu::Scan``).
 *
 * ``n`` must be at most ``kCompactMaxSamples`` (the map is ``int32``).
 */

/** @brief Mask flag: a set byte drops the sample (otherwise it keeps it). */
inline constexpr std::uint32_t kCompactMaskIsDrop = 1u;

/** @brief The largest sample count the ``int32`` index map can address. */
inline constexpr std::int64_t kCompactMaxSamples = INT32_MAX;

/** @brief The smallest span a reorder is worth firing for: below it the
 *  fixed launch cost of the moves exceeds what the restored coalescing saves. */
inline constexpr std::uint32_t kReorderMinSpan = 4096u;

/**
 * @brief The locality trigger a compaction can compute on the way (the input of
 *        the occasional physical reorder, ``Reorder.h``).
 *
 * The predicate counts ``live32``, the number of aligned 32-sample groups that
 * hold at least one kept sample (one warp ballot per group, one atomic per
 * block), and the lane that writes the count writes
 * ``fire = count < theta * 32 * live32 && count < span && span >= 4096``: the
 * kept samples fill less than ``theta`` of the warps they occupy, a reorder
 * would shrink the span, and the span is large enough for the moves to pay
 * (``kReorderMinSpan``). Every pointer is a one-word buffer in the compaction's memory space;
 * ``live32`` is overwritten (it needs no initial value), ``span`` is read.
 */
struct ReorderTrigger {
    std::uint32_t* live32;     ///< OUT: groups of 32 holding a kept sample
    const std::uint32_t* span; ///< samples ``[0, span)`` may still be kept
    std::uint32_t* fire;       ///< OUT: 1 when a reorder pays, else 0
    float theta;               ///< the locality threshold, in (0, 1)
};

/** @brief Bytes of scratch the compaction of @p n samples needs: three
 *  ``idx_t`` planes of ``n`` (predicate, inclusive scan, scan block sums). */
inline std::size_t compactScratchBytes(std::int64_t n)
{
    const std::int64_t m = n > 0 ? n : 1;
    return std::size_t(3) * std::size_t(m) * sizeof(idx_t);
}

namespace detail {

/** @brief Validate a compaction request; throws ``std::invalid_argument``. */
inline void checkCompact(std::int64_t n)
{
    if (n < 0)
        EAGLE_THROW(std::invalid_argument,
            "compact: the sample count must be non-negative");
    if (n > kCompactMaxSamples)
        EAGLE_THROW(std::invalid_argument,
            "compact: the int32 index map addresses at most 2^31-1 samples");
}

/** @brief Validate a trigger; throws ``std::invalid_argument``. */
inline void checkTrigger(const ReorderTrigger& t)
{
    if (t.live32 == nullptr || t.span == nullptr || t.fire == nullptr)
        EAGLE_THROW(std::invalid_argument,
            "compact: a reorder trigger needs its live32, span and fire words");
    if (!(t.theta > 0.0f && t.theta < 1.0f))
        EAGLE_THROW(std::invalid_argument,
            "compact: the reorder trigger's theta must lie in (0, 1)");
}

/** @brief The trigger rule, shared by both faces so they agree to the bit. */
AETHER_DEVICEANDHOST() inline std::uint32_t triggerFires(std::uint32_t count,
    std::uint32_t live32, std::uint32_t span, float theta)
{
    const double limit = double(theta) * 32.0 * double(live32);
    return (double(count) < limit && count < span && span >= kReorderMinSpan) ? 1u : 0u;
}

#ifndef EAGLE_CPU_ONLY

/** @brief ``keep[i] = Drop ? !mask[i] : mask[i] != 0`` (as ``idx_t`` 0/1). */
template<bool Drop>
AETHER_KERNEL()
__launch_bounds__(256, 4) void compactPredicate(
    const std::uint8_t* mask, GRefArrT<idx_t> keep)
{
    const SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    if (i.global() >= keep.samples())
        return;
    const bool set = mask[i.global()] != 0;
    keep.eval(i)   = idx_t(Drop ? !set : set);
}

/** @brief Scatter every kept sample to ``indexMap[inclusive[i] - 1]`` and write
 *  the count (``inclusive[n - 1]``) from the last lane. (A template, like every
 *  kernel defined in an eagle header, so each translation unit's copy is the
 *  same inline definition.) */
template<typename MapT>
AETHER_KERNEL()
__launch_bounds__(256, 4) void compactScatter(CRefArrT<idx_t> keep,
    CRefArrT<idx_t> inclusive, MapT* indexMap, std::uint32_t* count)
{
    const SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    const idx_t n = idx_t(keep.samples());
    if (i.global() >= n)
        return;
    const idx_t pos = inclusive(i.global());
    if (i.global() == n - 1)
        *count = std::uint32_t(pos);
    if (keep(i.global()) != 0)
        indexMap[pos - 1] = MapT(i.global());
}

/** @brief ``compactPredicate`` that also counts the 32-sample groups holding a
 *  kept sample into ``*live32`` (one ballot per warp, one atomic per block;
 *  the launch block is a multiple of 32, so a warp IS an aligned group). */
template<bool Drop>
AETHER_KERNEL()
__launch_bounds__(256, 4) void compactPredicateLive32(
    const std::uint8_t* mask, GRefArrT<idx_t> keep, std::uint32_t* live32)
{
    __shared__ std::uint32_t groups;
    if (threadIdx.x == 0)
        groups = 0u;
    __syncthreads();
    const SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    bool kept = false;
    if (i.global() < keep.samples()) {
        const bool set = mask[i.global()] != 0;
        kept           = Drop ? !set : set;
        keep.eval(i)   = idx_t(kept);
    }
    const unsigned ballot = __ballot_sync(0xffffffffu, kept);
    if ((threadIdx.x & 31u) == 0u && ballot != 0u)
        atomicAdd(&groups, 1u);
    __syncthreads();
    if (threadIdx.x == 0 && groups != 0u)
        atomicAdd(live32, groups);
}

/** @brief ``compactScatter`` whose last lane also writes the trigger. */
template<typename MapT>
AETHER_KERNEL()
__launch_bounds__(256, 4) void compactScatterTrigger(CRefArrT<idx_t> keep,
    CRefArrT<idx_t> inclusive, MapT* indexMap, std::uint32_t* count,
    const std::uint32_t* live32, const std::uint32_t* span, std::uint32_t* fire,
    float theta)
{
    const SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    const idx_t n = idx_t(keep.samples());
    if (i.global() >= n)
        return;
    const idx_t pos = inclusive(i.global());
    if (i.global() == n - 1) {
        *count = std::uint32_t(pos);
        *fire  = triggerFires(std::uint32_t(pos), *live32, *span, theta);
    }
    if (keep(i.global()) != 0)
        indexMap[pos - 1] = MapT(i.global());
}

/** @brief ``*word = 0`` (the trigger's group count before the predicate). */
template<typename WordT>
AETHER_KERNEL() void compactZero(WordT* word) { *word = WordT(0); }

/** @brief ``*count = 0`` — the whole compaction of an empty batch. */
template<typename CountT>
AETHER_KERNEL() void compactEmpty(CountT* count) { *count = CountT(0); }

#endif // EAGLE_CPU_ONLY

} // namespace detail

#ifndef EAGLE_CPU_ONLY

/**
 * @brief Enqueue the compaction of @p n samples on @p stream (device memory).
 *
 * Kernels only, issued in order on ``stream``: predicate, inclusive scan,
 * scatter. Legal while ``stream`` is being captured.
 */
inline void compactDevice(const std::uint8_t* mask, std::uint32_t flags,
    std::int32_t* indexMap, std::uint32_t* count, void* scratch, std::int64_t n,
    cudaStream_t stream)
{
    detail::checkCompact(n);
    err::detail::clearStaleLastError();  // check only our own launches below
    if (n == 0) {
        detail::compactEmpty<<<1, 1, 0, stream>>>(count);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
        return;
    }
    const idx_t N   = idx_t(n);
    idx_t* base     = static_cast<idx_t*>(scratch);
    auto keep       = spanView<idx_t>(base, N, deviceDevice());
    auto inclusive  = spanView<idx_t>(base + N, N, deviceDevice());
    auto blockSums  = spanView<idx_t>(base + 2 * std::size_t(N), N, deviceDevice());
    const idx_t blk = 256;
    const idx_t nb  = (N + blk - 1) / blk;
    if (flags & kCompactMaskIsDrop)
        detail::compactPredicate<true><<<nb, blk, 0, stream>>>(mask, keep);
    else
        detail::compactPredicate<false><<<nb, blk, 0, stream>>>(mask, keep);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    cuda::Scan::enqueue<idx_t, aether::SumOp<idx_t>, true>(
        keep.as_const(), inclusive, blockSums, idx_t(0), stream);
    detail::compactScatter<<<nb, blk, 0, stream>>>(
        keep.as_const(), inclusive.as_const(), indexMap, count);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

/**
 * @brief ``compactDevice`` that also computes the reorder trigger @p trig
 *        (device words). ``trig == nullptr`` is the call above, unchanged.
 *
 * One extra kernel (zeroing ``live32``); the predicate ballots and the scatter
 * writes ``fire`` from the lane that writes the count. Capturable likewise.
 */
inline void compactDevice(const std::uint8_t* mask, std::uint32_t flags,
    std::int32_t* indexMap, std::uint32_t* count, void* scratch, std::int64_t n,
    cudaStream_t stream, const ReorderTrigger* trig)
{
    if (trig == nullptr) {
        compactDevice(mask, flags, indexMap, count, scratch, n, stream);
        return;
    }
    detail::checkCompact(n);
    err::detail::clearStaleLastError();  // check only our own launches below
    detail::checkTrigger(*trig);
    if (n == 0) {
        detail::compactEmpty<<<1, 1, 0, stream>>>(count);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
        detail::compactZero<<<1, 1, 0, stream>>>(trig->live32);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
        detail::compactZero<<<1, 1, 0, stream>>>(trig->fire);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
        return;
    }
    const idx_t N   = idx_t(n);
    idx_t* base     = static_cast<idx_t*>(scratch);
    auto keep       = spanView<idx_t>(base, N, deviceDevice());
    auto inclusive  = spanView<idx_t>(base + N, N, deviceDevice());
    auto blockSums  = spanView<idx_t>(base + 2 * std::size_t(N), N, deviceDevice());
    const idx_t blk = 256;
    const idx_t nb  = (N + blk - 1) / blk;
    detail::compactZero<<<1, 1, 0, stream>>>(trig->live32);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    if (flags & kCompactMaskIsDrop)
        detail::compactPredicateLive32<true><<<nb, blk, 0, stream>>>(mask, keep, trig->live32);
    else
        detail::compactPredicateLive32<false><<<nb, blk, 0, stream>>>(mask, keep, trig->live32);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    cuda::Scan::enqueue<idx_t, aether::SumOp<idx_t>, true>(
        keep.as_const(), inclusive, blockSums, idx_t(0), stream);
    detail::compactScatterTrigger<<<nb, blk, 0, stream>>>(keep.as_const(),
        inclusive.as_const(), indexMap, count, trig->live32, trig->span,
        trig->fire, trig->theta);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

#endif // EAGLE_CPU_ONLY

/**
 * @brief The host face of ``compactDevice`` over host memory (OpenMP twins).
 */
inline void compactHost(const std::uint8_t* mask, std::uint32_t flags,
    std::int32_t* indexMap, std::uint32_t* count, void* scratch, std::int64_t n)
{
    detail::checkCompact(n);
    if (n == 0) {
        *count = 0u;
        return;
    }
    const idx_t N   = idx_t(n);
    idx_t* base     = static_cast<idx_t*>(scratch);
    auto keep       = spanView<idx_t>(base, N, hostDevice());
    auto inclusive  = spanView<idx_t>(base + N, N, hostDevice());
    auto blockSums  = spanView<idx_t>(base + 2 * std::size_t(N), N, hostDevice());
    const bool drop = (flags & kCompactMaskIsDrop) != 0;
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < n; ++i) {
        const bool set = mask[i] != 0;
        base[i]        = idx_t(drop ? !set : set);
    }
    cpu::Scan::scan<idx_t, aether::SumOp<idx_t>, true>(
        keep.as_const(), inclusive, blockSums);
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < n; ++i) {
        if (base[i] != 0)
            indexMap[inclusive(idx_t(i)) - 1] = std::int32_t(i);
    }
    *count = std::uint32_t(inclusive(N - 1));
}

/**
 * @brief ``compactHost`` that also computes the reorder trigger @p trig (host
 *        words), by the same rule as the device face. ``trig == nullptr`` is
 *        the call above, unchanged.
 */
inline void compactHost(const std::uint8_t* mask, std::uint32_t flags,
    std::int32_t* indexMap, std::uint32_t* count, void* scratch, std::int64_t n,
    const ReorderTrigger* trig)
{
    if (trig == nullptr) {
        compactHost(mask, flags, indexMap, count, scratch, n);
        return;
    }
    detail::checkTrigger(*trig);
    compactHost(mask, flags, indexMap, count, scratch, n);
    // The predicate plane (scratch[0, n)) still holds the keep flags.
    const idx_t* keep          = static_cast<const idx_t*>(scratch);
    const std::int64_t ngroups = (n + 31) / 32;
    std::int64_t live32        = 0;
#pragma omp parallel for schedule(static) reduction(+ : live32)
    for (std::int64_t g = 0; g < ngroups; ++g) {
        const std::int64_t end = (g + 1) * 32 < n ? (g + 1) * 32 : n;
        bool any               = false;
        for (std::int64_t i = g * 32; i < end && !any; ++i)
            any = keep[i] != 0;
        live32 += any ? 1 : 0;
    }
    *trig->live32 = std::uint32_t(live32);
    *trig->fire   = detail::triggerFires(*count, *trig->live32, *trig->span, trig->theta);
}

} // namespace filtering
} // namespace eagle
