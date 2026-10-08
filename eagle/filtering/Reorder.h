// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/filtering/Compact.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace eagle {
namespace filtering {

/**
 * @file Reorder.h
 * @brief The occasional physical reorder behind an active-set index map: move
 *        the live samples of caller-owned planes to the front, in place, and
 *        keep the permutation that undoes it.
 *
 * An index map (``Compact.h``) keeps the live samples' warps full, but its
 * gathers lose coalescing as the live set thins out. A reorder restores it:
 * every registered plane is permuted IN PLACE over ``[0, span)`` so the kept
 * samples occupy ``[0, count)`` in ascending order and the dropped ones follow
 * them, also in ascending order. Addresses never change (each plane goes
 * through ONE scratch staging plane and back), so pointers baked into a
 * captured graph stay valid.
 *
 * - ``planes``    the per-sample planes to move, each ``n`` elements of
 *                 ``elemBytes`` in {1, 2, 4, 8, 16}, aligned to it. The mask is
 *                 one of them (it moves with the rest); a ``(D, n)`` C-contiguous
 *                 array is ``D`` planes.
 * - ``perm``      ``int32[n]``, slot -> sample id; permuted like a plane.
 * - ``inv``       ``int32[n]``, sample id -> slot; rebuilt from ``perm``.
 *                 ``plane[inv[i]]`` is sample ``i`` at every step boundary.
 * - ``indexMap``  the compaction's map; after a reorder it is the identity over
 *                 ``[0, count)`` (it always holds PHYSICAL slots).
 * - ``count``     written: the kept samples (the same value a compaction
 *                 writes).
 * - ``span``      read, then set to ``count``: samples at or past ``span`` are
 *                 all dropped and never move again. Start it at ``n``.
 * - ``fire``      ``nullptr``: unconditional. Otherwise every kernel that
 *                 writes exits while ``*fire == 0``, and the call clears it.
 * - ``scratch``   ``reorderScratchBytes(n, maxElemBytes)`` bytes.
 *
 * The keep flags and their inclusive scan are recomputed from the mask (the
 * call is self-contained), so a reorder may follow a compaction or stand
 * alone. Termination must be monotone, as for the map.
 *
 * The device face only ENQUEUES on its stream (two kernels per plane plus a
 * few fixed ones), so it is legal inside a stream capture; the host face runs
 * the same moves with OpenMP. ``restoreDevice`` / ``restoreHost`` undo every
 * reorder since the last restore (a full gather per plane) and reset
 * ``perm``/``inv`` to the identity; the caller recomputes the map afterwards.
 */

/** @brief A per-sample plane to move: its address and element size. */
struct ReorderPlane {
    void* data;              ///< ``n`` elements
    std::uint32_t elemBytes; ///< 1, 2, 4, 8 or 16
};

/** @brief The largest element a plane may carry, in bytes. */
inline constexpr std::uint32_t kReorderMaxElemBytes = 16;

namespace detail {

/** @brief Byte offset of the staging plane inside the reorder scratch. */
inline std::size_t reorderStageOffset(std::int64_t n)
{
    const std::size_t scan = compactScratchBytes(n);
    return (scan + 15u) / 16u * 16u;
}

/** @brief A 16-byte element, moved as one value. */
struct alignas(16) Bytes16 {
    std::uint64_t lo, hi;
};

inline bool supportedElem(std::uint32_t b)
{
    return b == 1u || b == 2u || b == 4u || b == 8u || b == 16u;
}

/** @brief Validate a reorder request; throws ``std::invalid_argument``. */
inline void checkReorder(const ReorderPlane* planes, std::uint32_t nPlanes,
    const std::uint8_t* mask, std::int64_t n, bool needMask)
{
    checkCompact(n);
    if (nPlanes > 0 && planes == nullptr)
        EAGLE_THROW(std::invalid_argument, "reorder: a null plane table");
    bool maskSeen = false;
    for (std::uint32_t p = 0; p < nPlanes; ++p) {
        const ReorderPlane& a = planes[p];
        if (!supportedElem(a.elemBytes))
            EAGLE_THROW(std::invalid_argument,
                "reorder: plane " + std::to_string(p) + " has " + std::to_string(a.elemBytes)
                    + "-byte elements; supported are 1, 2, 4, 8 and 16");
        if (n > 0 && a.data == nullptr)
            EAGLE_THROW(std::invalid_argument, "reorder: plane " + std::to_string(p) + " is null");
        if (reinterpret_cast<std::uintptr_t>(a.data) % a.elemBytes != 0)
            EAGLE_THROW(std::invalid_argument,
                "reorder: plane " + std::to_string(p) + " is not aligned to its element size");
        const auto lo = reinterpret_cast<std::uintptr_t>(a.data);
        const auto hi = lo + std::uintptr_t(n) * a.elemBytes;
        for (std::uint32_t q = 0; q < p; ++q) {
            const auto qlo = reinterpret_cast<std::uintptr_t>(planes[q].data);
            const auto qhi = qlo + std::uintptr_t(n) * planes[q].elemBytes;
            if (n > 0 && lo < qhi && qlo < hi)
                EAGLE_THROW(std::invalid_argument,
                    "reorder: planes " + std::to_string(q) + " and " + std::to_string(p) + " overlap");
        }
        if (a.data == static_cast<const void*>(mask) && a.elemBytes == 1u)
            maskSeen = true;
    }
    if (needMask && n > 0 && !maskSeen)
        EAGLE_THROW(std::invalid_argument,
            "reorder: the mask must be one of the planes (it moves with them)");
}

/** @brief Where slot @p i goes: kept -> ``incl - 1``, dropped -> after the
 *  kept ones, in order. ``cnt`` is the kept count over ``[0, span)``. */
AETHER_DEVICEANDHOST() inline std::int64_t reorderDst(
    bool kept, std::int64_t i, std::int64_t incl, std::int64_t cnt)
{
    return kept ? incl - 1 : cnt + (i - incl);
}

#ifndef EAGLE_CPU_ONLY

AETHER_DEVICE() inline std::int64_t reorderThread()
{
    return std::int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
}

/** @brief ``stage[i] = plane[i]`` over ``[0, *span)`` (``[0, n)`` without a span). */
template<typename T>
AETHER_KERNEL()
__launch_bounds__(256, 4) void reorderStage(const T* plane, T* stage,
    const std::uint32_t* span, const std::uint32_t* fire, std::int64_t n)
{
    if (fire != nullptr && *fire == 0u)
        return;
    const std::int64_t i   = reorderThread();
    const std::int64_t end = span != nullptr ? std::int64_t(*span) : n;
    if (i < end)
        stage[i] = plane[i];
}

/** @brief Scatter the staged ``[0, *span)`` to their reordered slots. */
template<typename T>
AETHER_KERNEL()
__launch_bounds__(256, 4) void reorderPlace(const T* stage, T* plane,
    const idx_t* keep, const idx_t* incl, const std::uint32_t* span,
    const std::uint32_t* fire)
{
    if (fire != nullptr && *fire == 0u)
        return;
    const std::int64_t i  = reorderThread();
    const std::int64_t sp = std::int64_t(*span);
    if (i >= sp)
        return;
    const std::int64_t cnt = std::int64_t(incl[sp - 1]);
    plane[reorderDst(keep[i] != 0, i, std::int64_t(incl[i]), cnt)] = stage[i];
}

/** @brief ``inv[perm[s]] = s`` over ``[0, *span)`` (the moved slots). */
template<typename IndexT>
AETHER_KERNEL()
__launch_bounds__(256, 4) void reorderPermInv(const IndexT* perm, IndexT* inv,
    const std::uint32_t* span, const std::uint32_t* fire)
{
    if (fire != nullptr && *fire == 0u)
        return;
    const std::int64_t s = reorderThread();
    if (s < std::int64_t(*span))
        inv[perm[s]] = IndexT(s);
}

/** @brief ``map[t] = t`` over ``[0, kept)``, the kept count over ``[0, *span)``. */
template<typename IndexT>
AETHER_KERNEL()
__launch_bounds__(256, 4) void reorderIdentity(IndexT* indexMap, const idx_t* incl,
    const std::uint32_t* span, const std::uint32_t* fire)
{
    if (fire != nullptr && *fire == 0u)
        return;
    const std::int64_t sp = std::int64_t(*span);
    if (sp == 0)
        return;
    const std::int64_t t = reorderThread();
    if (t < std::int64_t(incl[sp - 1]))
        indexMap[t] = IndexT(t);
}

/** @brief The last kernel: ``count = span = kept``, ``fire = 0``. */
template<typename WordT>
AETHER_KERNEL() void reorderFin(const idx_t* incl, WordT* count, WordT* span, WordT* fire)
{
    if (fire != nullptr && *fire == 0u)
        return;
    const WordT sp  = *span;
    const WordT cnt = sp == 0u ? WordT(0) : WordT(incl[sp - 1]);
    *count          = cnt;
    *span           = cnt;
    if (fire != nullptr)
        *fire = WordT(0);
}

/** @brief The whole reorder of an empty batch: ``count = span = 0``, ``fire = 0``. */
template<typename WordT>
AETHER_KERNEL() void reorderEmpty(WordT* count, WordT* span, WordT* fire)
{
    if (fire != nullptr && *fire == 0u)
        return;
    *count = WordT(0);
    *span  = WordT(0);
    if (fire != nullptr)
        *fire = WordT(0);
}

/** @brief ``plane[perm[s]] = stage[s]`` over ``[0, n)``: back to sample order. */
template<typename T, typename IndexT>
AETHER_KERNEL()
__launch_bounds__(256, 4) void restoreScatter(const T* stage, T* plane,
    const IndexT* perm, std::int64_t n)
{
    const std::int64_t s = reorderThread();
    if (s < n)
        plane[perm[s]] = stage[s];
}

/** @brief ``perm[i] = inv[i] = i`` over ``[0, n)``. */
template<typename IndexT>
AETHER_KERNEL()
__launch_bounds__(256, 4) void restoreIdentity(IndexT* perm, IndexT* inv, std::int64_t n)
{
    const std::int64_t i = reorderThread();
    if (i < n) {
        perm[i] = IndexT(i);
        inv[i]  = IndexT(i);
    }
}

/** @brief Stage + place one plane of element type @p T. */
template<typename T>
inline void reorderPlaneDevice(void* data, void* stage, const idx_t* keep,
    const idx_t* incl, const std::uint32_t* span, const std::uint32_t* fire,
    std::int64_t n, unsigned nb, cudaStream_t stream)
{
    T* plane = static_cast<T*>(data);
    T* st    = static_cast<T*>(stage);
    reorderStage<T><<<nb, 256, 0, stream>>>(plane, st, span, fire, n);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    reorderPlace<T><<<nb, 256, 0, stream>>>(st, plane, keep, incl, span, fire);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

/** @brief Stage + scatter-back one plane of element type @p T. */
template<typename T>
inline void restorePlaneDevice(void* data, void* stage, const std::int32_t* perm,
    std::int64_t n, unsigned nb, cudaStream_t stream)
{
    T* plane = static_cast<T*>(data);
    T* st    = static_cast<T*>(stage);
    reorderStage<T><<<nb, 256, 0, stream>>>(plane, st, nullptr, nullptr, n);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    restoreScatter<T, std::int32_t><<<nb, 256, 0, stream>>>(st, plane, perm, n);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

#endif // EAGLE_CPU_ONLY

/** @brief Call ``f.template operator()<T>()`` with the element type of @p bytes. */
template<typename F>
inline void withElem(std::uint32_t bytes, F&& f)
{
    switch (bytes) {
    case 1u: f.template operator()<std::uint8_t>(); break;
    case 2u: f.template operator()<std::uint16_t>(); break;
    case 4u: f.template operator()<std::uint32_t>(); break;
    case 8u: f.template operator()<std::uint64_t>(); break;
    default: f.template operator()<Bytes16>(); break;
    }
}

} // namespace detail

/** @brief Bytes of scratch a reorder (or restore) of @p n samples needs: the
 *  compaction's scan scratch, then one staging plane of the widest element
 *  (``perm`` included, so at least 4 bytes). */
inline std::size_t reorderScratchBytes(std::int64_t n, std::uint32_t maxElemBytes)
{
    const std::size_t m    = std::size_t(n > 0 ? n : 1);
    const std::size_t elem = maxElemBytes > 4u ? maxElemBytes : 4u;
    return detail::reorderStageOffset(n) + m * elem;
}

#ifndef EAGLE_CPU_ONLY

/**
 * @brief Enqueue the reorder of @p n samples on @p stream (device memory).
 *
 * Kernels only: predicate + inclusive scan of the mask, then stage + place for
 * every plane and for ``perm``, ``inv`` from ``perm``, the identity map, and
 * last the words (``count = span = kept``, ``fire = 0``). Legal while
 * ``stream`` is being captured; with a ``fire`` word it is a no-op on the
 * device while ``*fire == 0`` (the scan still runs, into scratch).
 */
inline void reorderDevice(const ReorderPlane* planes, std::uint32_t nPlanes,
    const std::uint8_t* mask, std::uint32_t flags, std::int32_t* perm,
    std::int32_t* inv, std::int32_t* indexMap, std::uint32_t* count,
    std::uint32_t* span, std::uint32_t* fire, void* scratch, std::int64_t n,
    cudaStream_t stream)
{
    detail::checkReorder(planes, nPlanes, mask, n, true);
    err::detail::clearStaleLastError();  // check only our own launches below
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
    void* stage     = static_cast<char*>(scratch) + detail::reorderStageOffset(n);
    const unsigned nb = unsigned((n + 255) / 256);
    if (flags & kCompactMaskIsDrop)
        detail::compactPredicate<true><<<nb, 256, 0, stream>>>(mask, keep);
    else
        detail::compactPredicate<false><<<nb, 256, 0, stream>>>(mask, keep);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    cuda::Scan::enqueue<idx_t, aether::SumOp<idx_t>, true>(
        keep.as_const(), inclusive, blockSums, idx_t(0), stream);
    const idx_t* k = base;
    const idx_t* c = base + N;
    for (std::uint32_t p = 0; p < nPlanes; ++p) {
        detail::withElem(planes[p].elemBytes, [&]<typename T>() {
            detail::reorderPlaneDevice<T>(planes[p].data, stage, k, c, span, fire, n, nb, stream);
        });
    }
    detail::reorderPlaneDevice<std::int32_t>(perm, stage, k, c, span, fire, n, nb, stream);
    detail::reorderPermInv<std::int32_t><<<nb, 256, 0, stream>>>(perm, inv, span, fire);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    detail::reorderIdentity<std::int32_t><<<nb, 256, 0, stream>>>(indexMap, c, span, fire);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    detail::reorderFin<std::uint32_t><<<1, 1, 0, stream>>>(c, count, span, fire);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

/**
 * @brief Enqueue the un-permutation of every plane on @p stream: sample ``i``
 *        back at slot ``i``; ``perm`` and ``inv`` := identity.
 *
 * A gather per plane over all ``n`` slots. The index map, count and span are
 * NOT touched (they describe the permuted layout): recompute the map and set
 * the span back to ``n`` afterwards.
 */
inline void restoreDevice(const ReorderPlane* planes, std::uint32_t nPlanes,
    std::int32_t* perm, std::int32_t* inv, void* scratch, std::int64_t n,
    cudaStream_t stream)
{
    detail::checkReorder(planes, nPlanes, nullptr, n, false);
    err::detail::clearStaleLastError();  // check only our own launches below
    if (n == 0)
        return;
    void* stage       = static_cast<char*>(scratch) + detail::reorderStageOffset(n);
    const unsigned nb = unsigned((n + 255) / 256);
    for (std::uint32_t p = 0; p < nPlanes; ++p) {
        detail::withElem(planes[p].elemBytes, [&]<typename T>() {
            detail::restorePlaneDevice<T>(planes[p].data, stage, perm, n, nb, stream);
        });
    }
    detail::restoreIdentity<std::int32_t><<<nb, 256, 0, stream>>>(perm, inv, n);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

#endif // EAGLE_CPU_ONLY

namespace detail {

template<typename T>
inline void reorderPlaneHost(void* data, void* stage, const idx_t* keep,
    const idx_t* incl, std::int64_t sp)
{
    T* plane = static_cast<T*>(data);
    T* st    = static_cast<T*>(stage);
    if (sp == 0)
        return;
    const std::int64_t cnt = std::int64_t(incl[sp - 1]);
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < sp; ++i)
        st[i] = plane[i];
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < sp; ++i)
        plane[reorderDst(keep[i] != 0, i, std::int64_t(incl[i]), cnt)] = st[i];
}

template<typename T>
inline void restorePlaneHost(void* data, void* stage, const std::int32_t* perm, std::int64_t n)
{
    T* plane = static_cast<T*>(data);
    T* st    = static_cast<T*>(stage);
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < n; ++i)
        st[i] = plane[i];
#pragma omp parallel for schedule(static)
    for (std::int64_t s = 0; s < n; ++s)
        plane[perm[s]] = st[s];
}

} // namespace detail

/** @brief The host face of ``reorderDevice`` over host memory (OpenMP);
 *  the same moves, so the two faces agree to the bit. */
inline void reorderHost(const ReorderPlane* planes, std::uint32_t nPlanes,
    const std::uint8_t* mask, std::uint32_t flags, std::int32_t* perm,
    std::int32_t* inv, std::int32_t* indexMap, std::uint32_t* count,
    std::uint32_t* span, std::uint32_t* fire, void* scratch, std::int64_t n)
{
    detail::checkReorder(planes, nPlanes, mask, n, true);
    if (fire != nullptr && *fire == 0u)
        return;
    if (n == 0) {
        *count = 0u;
        *span  = 0u;
        if (fire != nullptr)
            *fire = 0u;
        return;
    }
    const idx_t N   = idx_t(n);
    idx_t* base     = static_cast<idx_t*>(scratch);
    auto keep       = spanView<idx_t>(base, N, hostDevice());
    auto inclusive  = spanView<idx_t>(base + N, N, hostDevice());
    auto blockSums  = spanView<idx_t>(base + 2 * std::size_t(N), N, hostDevice());
    void* stage     = static_cast<char*>(scratch) + detail::reorderStageOffset(n);
    const bool drop = (flags & kCompactMaskIsDrop) != 0;
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < n; ++i) {
        const bool set = mask[i] != 0;
        base[i]        = idx_t(drop ? !set : set);
    }
    cpu::Scan::scan<idx_t, aether::SumOp<idx_t>, true>(keep.as_const(), inclusive, blockSums);
    const idx_t* k        = base;
    const idx_t* c        = base + N;
    const std::int64_t sp = std::int64_t(*span) < n ? std::int64_t(*span) : n;
    for (std::uint32_t p = 0; p < nPlanes; ++p) {
        detail::withElem(planes[p].elemBytes, [&]<typename T>() {
            detail::reorderPlaneHost<T>(planes[p].data, stage, k, c, sp);
        });
    }
    detail::reorderPlaneHost<std::int32_t>(perm, stage, k, c, sp);
#pragma omp parallel for schedule(static)
    for (std::int64_t s = 0; s < sp; ++s)
        inv[perm[s]] = std::int32_t(s);
    const std::int64_t cnt = sp == 0 ? 0 : std::int64_t(c[sp - 1]);
#pragma omp parallel for schedule(static)
    for (std::int64_t t = 0; t < cnt; ++t)
        indexMap[t] = std::int32_t(t);
    *count = std::uint32_t(cnt);
    *span  = std::uint32_t(cnt);
    if (fire != nullptr)
        *fire = 0u;
}

/** @brief The host face of ``restoreDevice`` over host memory (OpenMP). */
inline void restoreHost(const ReorderPlane* planes, std::uint32_t nPlanes,
    std::int32_t* perm, std::int32_t* inv, void* scratch, std::int64_t n)
{
    detail::checkReorder(planes, nPlanes, nullptr, n, false);
    if (n == 0)
        return;
    void* stage = static_cast<char*>(scratch) + detail::reorderStageOffset(n);
    for (std::uint32_t p = 0; p < nPlanes; ++p) {
        detail::withElem(planes[p].elemBytes, [&]<typename T>() {
            detail::restorePlaneHost<T>(planes[p].data, stage, perm, n);
        });
    }
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < n; ++i) {
        perm[i] = std::int32_t(i);
        inv[i]  = std::int32_t(i);
    }
}

} // namespace filtering
} // namespace eagle
