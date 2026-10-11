// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/filtering/Reorder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace eagle {
namespace filtering {

/**
 * @file ReorderClasses.h
 * @brief K-class active-set sort: one stable permutation of the slots by
 *        ``(live, class)`` -- the liveness reorder generalised to ``K`` classes
 *        -- and the trigger that decides when it pays.
 *
 * - ``classPlane``  ``uint8[n]``, each sample's class in ``[0, K)`` (K <= 255);
 *                   one of the moved planes. A value ``>= K`` is a precondition
 *                   violation (the device clamps it to ``K - 1``, never reads out of bounds).
 * - ``offsets``     ``uint32[K + 1]`` OUT, the exclusive prefix over the LIVE
 *                   samples: class ``c`` occupies slots ``[offsets[c], offsets[c+1])``,
 *                   ``offsets[K] == live count``.
 * - full sort       sorts ``[0, *span)`` by ``(live, class)`` (dead last, stable),
 *                   then ``count = span = live``.
 * - WITHIN_LIVE     sorts the slots ``[0, *count)`` among themselves by class only
 *                   (the reverse's form: liveness is the prefix); ``count`` and
 *                   ``span`` are unchanged.
 * - ``fire``        as for ``reorderDevice``: ``nullptr`` unconditional, else the
 *                   moving kernels exit while ``*fire == 0`` and the call clears it.
 * - ``classSorts``  incremented once when the permutation ran.
 *
 * The trigger (``classTrigger*``): ``mixed32`` = 32-slot groups holding two or
 * more live classes; ``class_fire = (mixed32 > 0 && mixed32 >= classTheta * live32
 * && span >= kReorderMinSpan) || stale > kClassStaleMax * count || fire`` (the
 * last term: the liveness trigger of the same compaction, so a thinning batch is
 * not lost when the classes happen to be grouped).
 *
 * Device: enqueue only (capturable). Host: the same moves in serial order, so
 * the two faces agree to the bit.
 */

/** @brief The reorder desc flag: sort inside the live prefix only. */
inline constexpr std::uint32_t kClassesWithinLive = 1u << 1;
/** @brief The most classes (the class plane is one byte; 255 is a sentinel). */
inline constexpr std::uint32_t kMaxClasses = 255u;
/** @brief The stale-fraction trigger: re-sort when more than this fraction of live samples changed class. */
inline constexpr double kClassStaleMax = 0.05;

/** @brief The inputs/outputs of the K-class trigger beside the base compaction trigger. */
struct ClassTrigger {
    const std::uint8_t* classPlane; ///< uint8[n]
    std::uint32_t classes;          ///< K
    float classTheta;               ///< in (0, 1)
    std::uint32_t* mixed32;         ///< OUT
    const std::uint32_t* stale;     ///< IN, may be nullptr (= 0)
    std::uint32_t* classFire;       ///< OUT
};

/** @brief Bytes of scratch a class sort of @p n samples needs (``reorderScratchBytes`` + dst + offsets). */
inline std::size_t classSortScratchBytes(std::int64_t n, std::uint32_t maxElemBytes, std::uint32_t classes)
{
    const std::size_t base = reorderScratchBytes(n, maxElemBytes);
    const std::size_t m    = std::size_t(n > 0 ? n : 1);
    return (base + 15u) / 16u * 16u + (m * 4u + 15u) / 16u * 16u + (std::size_t(classes) + 2u) * 4u;
}

namespace detail {

inline void checkClasses(std::uint32_t classes, const std::uint8_t* classPlane, std::int64_t n)
{
    if (classes < 2u || classes > kMaxClasses)
        EAGLE_THROW(std::invalid_argument, "classes: K must lie in [2, 255]");
    if (n > 0 && classPlane == nullptr)
        EAGLE_THROW(std::invalid_argument, "classes: a null class plane");
}

inline void checkClassTrigger(const ClassTrigger& t, std::int64_t n)
{
    checkClasses(t.classes, t.classPlane, n);
    if (t.mixed32 == nullptr || t.classFire == nullptr)
        EAGLE_THROW(std::invalid_argument, "classes: the trigger needs its mixed32 and class_fire words");
    if (!(t.classTheta > 0.0f && t.classTheta < 1.0f))
        EAGLE_THROW(std::invalid_argument, "classes: class_theta must lie in (0, 1)");
}

/** @brief The class trigger rule over its integer inputs. */
AETHER_DEVICEANDHOST() inline std::uint32_t classTriggerFires(std::uint32_t mixed, std::uint32_t live32,
    std::uint32_t span, float classTheta, std::uint32_t stale, std::uint32_t count, std::uint32_t liveFire)
{
    const bool warpMix = mixed > 0u && double(mixed) >= double(classTheta) * double(live32) && span >= kReorderMinSpan;
    const bool drift   = double(stale) > kClassStaleMax * double(count);
    return (warpMix || drift || liveFire != 0u) ? 1u : 0u;
}

/** @brief The sort key of slot with mask byte @p m and class @p c. */
AETHER_DEVICEANDHOST() inline std::uint32_t classKey(
    std::uint8_t m, std::uint8_t c, bool drop, bool withinLive, std::uint32_t classes)
{
    const std::uint32_t cc = c < classes ? std::uint32_t(c) : classes - 1u;
    if (withinLive)
        return cc;
    const bool set  = m != 0;
    const bool live = drop ? !set : set;
    return live ? cc : classes;
}

#ifndef EAGLE_CPU_ONLY

AETHER_KERNEL() void classInit(std::uint32_t* off) { off[0] = 0u; }

AETHER_KERNEL()
__launch_bounds__(256, 4) void classKeep(const std::uint8_t* mask, const std::uint8_t* cls, idx_t* keep,
    const std::uint32_t* lim, std::uint32_t c, std::uint32_t classes, bool drop, bool within, std::int64_t n)
{
    const std::int64_t i = reorderThread();
    if (i >= n)
        return;
    const bool in = i < std::int64_t(*lim) && classKey(mask[i], cls[i], drop, within, classes) == c;
    keep[i]       = idx_t(in);
}

AETHER_KERNEL()
__launch_bounds__(256, 4) void classAssign(const idx_t* keep, const idx_t* incl, const std::uint32_t* off,
    std::int32_t* dst, std::uint32_t c, std::int64_t n)
{
    const std::int64_t i = reorderThread();
    if (i < n && keep[i] != 0)
        dst[i] = std::int32_t(off[c] + std::uint32_t(incl[i]) - 1u);
}

AETHER_KERNEL() void classAdvance(const idx_t* incl, const std::uint32_t* lim, std::uint32_t* off, std::uint32_t c,
    std::int64_t n)
{
    std::int64_t l = std::int64_t(*lim);
    l              = l < n ? l : n;
    off[c + 1]     = off[c] + (l == 0 ? 0u : std::uint32_t(incl[l - 1]));
}

template<typename T>
AETHER_KERNEL()
__launch_bounds__(256, 4) void classPlace(const T* stage, T* plane, const std::int32_t* dst,
    const std::uint32_t* lim, const std::uint32_t* fire)
{
    if (fire != nullptr && *fire == 0u)
        return;
    const std::int64_t i = reorderThread();
    if (i < std::int64_t(*lim))
        plane[dst[i]] = stage[i];
}

AETHER_KERNEL()
__launch_bounds__(256, 4) void classIdentity(std::int32_t* indexMap, const std::uint32_t* live,
    const std::uint32_t* fire)
{
    if (fire != nullptr && *fire == 0u)
        return;
    const std::int64_t t = reorderThread();
    if (t < std::int64_t(*live))
        indexMap[t] = std::int32_t(t);
}

AETHER_KERNEL() void classFin(const std::uint32_t* off, std::uint32_t classes, bool within, std::uint32_t* offsets,
    std::uint32_t* count, std::uint32_t* span, std::uint32_t* fire, std::uint32_t* classSorts)
{
    if (fire != nullptr && *fire == 0u)
        return;
    for (std::uint32_t c = 0; c <= classes; ++c)
        offsets[c] = off[c];
    if (!within) {
        *count = off[classes];
        *span  = off[classes];
    }
    if (fire != nullptr)
        *fire = 0u;
    *classSorts += 1u;
}

/** @brief Per 32-slot group: any two live classes? (one warp per group) */
AETHER_KERNEL()
__launch_bounds__(256, 4) void classMixed32(const std::uint8_t* mask, const std::uint8_t* cls, std::uint32_t* mixed32,
    bool drop, std::int64_t n)
{
    const std::int64_t i = reorderThread();
    bool live            = false;
    unsigned c           = 0u;
    if (i < n) {
        const bool set = mask[i] != 0;
        live           = drop ? !set : set;
        c              = cls[i];
    }
    unsigned lo = live ? c : 255u;
    unsigned hi = live ? c : 0u;
    for (int d = 16; d > 0; d >>= 1) {
        lo = min(lo, __shfl_xor_sync(0xffffffffu, lo, d));
        hi = max(hi, __shfl_xor_sync(0xffffffffu, hi, d));
    }
    if ((threadIdx.x & 31u) == 0u && lo < hi)
        atomicAdd(mixed32, 1u);
}

AETHER_KERNEL() void classFireKernel(const std::uint32_t* count, const std::uint32_t* live32, const std::uint32_t* span,
    const std::uint32_t* mixed32, const std::uint32_t* stale, const std::uint32_t* liveFire, float classTheta,
    std::uint32_t* classFire)
{
    *classFire = classTriggerFires(*mixed32, *live32, *span, classTheta, stale != nullptr ? *stale : 0u, *count,
        liveFire != nullptr ? *liveFire : 0u);
}

template<typename T>
inline void classPlaneDevice(void* data, void* stage, const std::int32_t* dst, const std::uint32_t* lim,
    const std::uint32_t* fire, std::int64_t n, unsigned nb, cudaStream_t stream)
{
    T* plane = static_cast<T*>(data);
    T* st    = static_cast<T*>(stage);
    reorderStage<T><<<nb, 256, 0, stream>>>(plane, st, lim, nullptr, n);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    classPlace<T><<<nb, 256, 0, stream>>>(st, plane, dst, lim, fire);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

#endif // EAGLE_CPU_ONLY

} // namespace detail

#ifndef EAGLE_CPU_ONLY

/**
 * @brief Enqueue the class trigger on @p stream after a base ``compactDevice``
 *        with a trigger (it reads that call's ``count``, ``live32``, ``span``, ``fire``).
 */
inline void classTriggerDevice(const std::uint8_t* mask, std::uint32_t flags, const ReorderTrigger& base,
    const std::uint32_t* count, const ClassTrigger& t, std::int64_t n, cudaStream_t stream)
{
    detail::checkCompact(n);
    detail::checkClassTrigger(t, n);
    err::detail::clearStaleLastError();
    detail::compactZero<<<1, 1, 0, stream>>>(t.mixed32);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    if (n > 0) {
        detail::classMixed32<<<unsigned((n + 255) / 256), 256, 0, stream>>>(
            mask, t.classPlane, t.mixed32, (flags & kCompactMaskIsDrop) != 0, n);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
    }
    detail::classFireKernel<<<1, 1, 0, stream>>>(
        count, base.live32, base.span, t.mixed32, t.stale, base.fire, t.classTheta, t.classFire);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

/**
 * @brief Enqueue the K-class stable sort of @p n samples on @p stream.
 *
 * Needs ``classSortScratchBytes`` of scratch. Capturable: kernels only. Cost is
 * ``K`` (``K + 1`` for a full sort) inclusive scans.
 */
inline void reorderClassesDevice(const ReorderPlane* planes, std::uint32_t nPlanes, const std::uint8_t* mask,
    std::uint32_t flags, std::uint32_t classes, const std::uint8_t* classPlane, std::int32_t* perm, std::int32_t* inv,
    std::int32_t* indexMap, std::uint32_t* count, std::uint32_t* span, std::uint32_t* fire, std::uint32_t* offsets,
    std::uint32_t* classSorts, void* scratch, std::int64_t n, cudaStream_t stream)
{
    detail::checkReorder(planes, nPlanes, mask, n, true);
    detail::checkClasses(classes, classPlane, n);
    if (offsets == nullptr || classSorts == nullptr)
        EAGLE_THROW(std::invalid_argument, "classes: offsets and class_sorts must be device addresses");
    bool classSeen = false;
    for (std::uint32_t p = 0; p < nPlanes; ++p)
        classSeen = classSeen || (planes[p].data == static_cast<const void*>(classPlane) && planes[p].elemBytes == 1u);
    if (n > 0 && !classSeen)
        EAGLE_THROW(std::invalid_argument, "classes: the class plane must be one of the planes (it moves with them)");
    err::detail::clearStaleLastError();
    const bool within = (flags & kClassesWithinLive) != 0;
    const bool drop   = (flags & kCompactMaskIsDrop) != 0;
    if (n == 0) {
        detail::reorderEmpty<std::uint32_t><<<1, 1, 0, stream>>>(count, span, fire);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
        return;
    }
    const idx_t N   = idx_t(n);
    idx_t* base     = static_cast<idx_t*>(scratch);
    auto keep       = spanView<idx_t>(base, N, deviceDevice());
    auto inclusive  = spanView<idx_t>(base + N, N, deviceDevice());
    auto blockSums  = spanView<idx_t>(base + 2 * std::size_t(N), N, deviceDevice());
    char* raw       = static_cast<char*>(scratch);
    void* stage     = raw + detail::reorderStageOffset(n);
    std::uint32_t maxElem = 4u;
    for (std::uint32_t p = 0; p < nPlanes; ++p)
        maxElem = std::max(maxElem, planes[p].elemBytes);
    char* tail           = raw + (reorderScratchBytes(n, maxElem) + 15u) / 16u * 16u;
    std::int32_t* dst    = reinterpret_cast<std::int32_t*>(tail);
    std::uint32_t* off   = reinterpret_cast<std::uint32_t*>(tail + (std::size_t(n) * 4u + 15u) / 16u * 16u);
    const std::uint32_t* lim = within ? count : span;
    const unsigned nb = unsigned((n + 255) / 256);
    detail::classInit<<<1, 1, 0, stream>>>(off);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    const std::uint32_t nKeys = within ? classes : classes + 1u;  // the dead are one more key
    for (std::uint32_t c = 0; c < nKeys; ++c) {
        detail::classKeep<<<nb, 256, 0, stream>>>(mask, classPlane, base, lim, c, classes, drop, within, n);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
        cuda::Scan::enqueue<idx_t, aether::SumOp<idx_t>, true>(
            keep.as_const(), inclusive, blockSums, idx_t(0), stream);
        detail::classAssign<<<nb, 256, 0, stream>>>(base, base + N, off, dst, c, n);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
        detail::classAdvance<<<1, 1, 0, stream>>>(base + N, lim, off, c, n);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
    }
    for (std::uint32_t p = 0; p < nPlanes; ++p) {
        detail::withElem(planes[p].elemBytes, [&]<typename T>() {
            detail::classPlaneDevice<T>(planes[p].data, stage, dst, lim, fire, n, nb, stream);
        });
    }
    detail::classPlaneDevice<std::int32_t>(perm, stage, dst, lim, fire, n, nb, stream);
    detail::reorderPermInv<std::int32_t><<<nb, 256, 0, stream>>>(perm, inv, lim, fire);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    detail::classIdentity<<<nb, 256, 0, stream>>>(indexMap, within ? count : off + classes, fire);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    detail::classFin<<<1, 1, 0, stream>>>(off, classes, within, offsets, count, span, fire, classSorts);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

#endif // EAGLE_CPU_ONLY

/** @brief The host face of ``classTriggerDevice`` (serial; the oracle). */
inline void classTriggerHost(const std::uint8_t* mask, std::uint32_t flags, std::uint32_t live32,
    const std::uint32_t* span, std::uint32_t liveFire, const std::uint32_t* count, const ClassTrigger& t,
    std::int64_t n)
{
    detail::checkCompact(n);
    detail::checkClassTrigger(t, n);
    const bool drop = (flags & kCompactMaskIsDrop) != 0;
    std::uint32_t mixed = 0;
    for (std::int64_t g = 0; g < n; g += 32) {
        unsigned lo = 255u, hi = 0u;
        for (std::int64_t i = g; i < std::min<std::int64_t>(g + 32, n); ++i) {
            const bool set = mask[i] != 0;
            if (drop ? !set : set) {
                lo = std::min<unsigned>(lo, t.classPlane[i]);
                hi = std::max<unsigned>(hi, t.classPlane[i]);
            }
        }
        mixed += lo < hi ? 1u : 0u;
    }
    *t.mixed32   = mixed;
    *t.classFire = detail::classTriggerFires(
        mixed, live32, *span, t.classTheta, t.stale != nullptr ? *t.stale : 0u, *count, liveFire);
}

/** @brief The host face of ``reorderClassesDevice`` (serial stable counting sort). */
inline void reorderClassesHost(const ReorderPlane* planes, std::uint32_t nPlanes, const std::uint8_t* mask,
    std::uint32_t flags, std::uint32_t classes, const std::uint8_t* classPlane, std::int32_t* perm, std::int32_t* inv,
    std::int32_t* indexMap, std::uint32_t* count, std::uint32_t* span, std::uint32_t* fire, std::uint32_t* offsets,
    std::uint32_t* classSorts, std::int64_t n)
{
    detail::checkReorder(planes, nPlanes, mask, n, true);
    detail::checkClasses(classes, classPlane, n);
    if (fire != nullptr && *fire == 0u)
        return;
    const bool within = (flags & kClassesWithinLive) != 0;
    const bool drop   = (flags & kCompactMaskIsDrop) != 0;
    std::int64_t lim  = std::int64_t(within ? *count : *span);
    lim               = std::min<std::int64_t>(lim, n);
    std::vector<std::uint32_t> off(classes + 2u, 0u);
    std::vector<std::int32_t> dst(std::size_t(lim > 0 ? lim : 0));
    std::vector<std::uint32_t> key(std::size_t(lim > 0 ? lim : 0));
    for (std::int64_t i = 0; i < lim; ++i) {
        key[i] = detail::classKey(mask[i], classPlane[i], drop, within, classes);
        ++off[key[i] + 1u];
    }
    for (std::uint32_t c = 0; c <= classes; ++c)
        off[c + 1u] += off[c];
    std::vector<std::uint32_t> next(off.begin(), off.end());
    for (std::int64_t i = 0; i < lim; ++i)
        dst[i] = std::int32_t(next[key[i]]++);
    const auto move = [&]<typename T>(void* data) {
        T* p = static_cast<T*>(data);
        std::vector<T> tmp(p, p + lim);
        for (std::int64_t i = 0; i < lim; ++i)
            p[dst[i]] = tmp[i];
    };
    for (std::uint32_t q = 0; q < nPlanes; ++q)
        detail::withElem(planes[q].elemBytes, [&]<typename T>() { move.template operator()<T>(planes[q].data); });
    move.template operator()<std::int32_t>(perm);
    for (std::int64_t s = 0; s < lim; ++s)
        inv[perm[s]] = std::int32_t(s);
    const std::uint32_t live = off[classes];
    for (std::uint32_t t = 0; t < live; ++t)
        indexMap[t] = std::int32_t(t);
    for (std::uint32_t c = 0; c <= classes; ++c)
        offsets[c] = off[c];
    if (!within) {
        *count = live;
        *span  = live;
    }
    if (fire != nullptr)
        *fire = 0u;
    *classSorts += 1u;
}

} // namespace filtering
} // namespace eagle
