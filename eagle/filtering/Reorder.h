// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/filtering/Compact.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
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
 * - ``scratch``   ``reorderScratchBytes(n, maxElemBytes)`` bytes, or the
 *                 plane-table overload when a larger staging area is wanted.
 *
 * The keep flags and their inclusive scan are recomputed from the mask (the
 * call is self-contained), so a reorder may follow a compaction or stand
 * alone. Termination must be monotone, as for the map.
 *
 * Planes move in GROUPS: as many consecutive planes as fit the staging area
 * (at most ``kReorderMaxGroupPlanes``) are staged by one kernel and placed by a
 * second, so a reorder costs two kernels per group rather than two per plane.
 * The staging area defaults to one plane of the widest element (what
 * ``reorderScratchBytes(n, maxElemBytes)`` provides); passing
 * ``stageBytesPerSample`` to the reorder and sizing the scratch with the
 * plane-table overload of ``reorderScratchBytes`` makes groups larger. Each
 * kernel walks ``[0, *span)`` with a grid of at most four blocks per
 * multiprocessor.
 *
 * The device face only ENQUEUES on its stream (two kernels per group plus a
 * few fixed ones), so it is legal inside a stream capture; the host face runs
 * the same moves, in the same groups, with OpenMP. ``restoreDevice`` / ``restoreHost`` undo every
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

/** @brief The most planes one stage/place kernel pair moves. */
inline constexpr std::uint32_t kReorderMaxGroupPlanes = 64;

/** @brief The planes of one group, passed to a kernel by value. */
struct ReorderGroup {
    void* data[kReorderMaxGroupPlanes];            ///< plane addresses
    std::uint32_t elemBytes[kReorderMaxGroupPlanes]; ///< their element sizes
    std::uint64_t stageOffset[kReorderMaxGroupPlanes]; ///< byte offset of each plane's staging slice
    std::uint32_t count;                           ///< planes in the group
    std::int32_t permSlot;                         ///< index of the ``perm`` plane in the group, or -1
};

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

/** @brief The staging bytes per sample a reorder uses: @p requested, raised to
 *  the widest plane (``perm`` counts as 4 bytes) so every plane fits a group.
 *  Zero asks for exactly that widest plane. */
inline std::uint32_t effectiveStageBytes(const ReorderPlane* planes, std::uint32_t nPlanes,
    std::uint32_t requested)
{
    std::uint32_t widest = 4u;
    for (std::uint32_t p = 0; p < nPlanes; ++p)
        widest = planes[p].elemBytes > widest ? planes[p].elemBytes : widest;
    return requested > widest ? requested : widest;
}

/** @brief Fill @p g with the planes from index @p first on (the plane table,
 *  then ``perm`` as the last plane) that fit @p budgetBytes of staging and
 *  return the index one past the group. Each plane's slice is aligned to its
 *  element size. */
inline std::uint32_t nextReorderGroup(const ReorderPlane* planes, std::uint32_t nPlanes,
    void* perm, std::uint32_t first, std::int64_t n, std::uint64_t budgetBytes,
    ReorderGroup& g)
{
    const std::uint32_t total = nPlanes + 1u;
    std::uint64_t used        = 0;
    g.count                   = 0;
    g.permSlot                = -1;
    std::uint32_t k           = first;
    for (; k < total && g.count < kReorderMaxGroupPlanes; ++k) {
        const bool isPerm   = k == nPlanes;
        const std::uint32_t e = isPerm ? 4u : planes[k].elemBytes;
        const std::uint64_t at = (used + e - 1u) / e * e;
        if (g.count > 0 && at + std::uint64_t(n) * e > budgetBytes)
            break;
        g.data[g.count]        = isPerm ? perm : planes[k].data;
        g.elemBytes[g.count]   = e;
        g.stageOffset[g.count] = at;
        if (isPerm)
            g.permSlot = std::int32_t(g.count);
        used = at + std::uint64_t(n) * e;
        ++g.count;
    }
    return k;
}

}  // namespace detail

/** @brief Where a pointer lives, as seen from one device. */
enum class Residency : std::int32_t {
    Host = 0,         ///< ordinary host memory (CPU mode: everything)
    Device = 1,       ///< device memory of the queried device
    Managed = 2,      ///< managed (unified) memory
    MappedHost = 3,   ///< pinned host memory the device can address (allocated or registered mapped)
    Pageable = 4,     ///< host memory the device cannot address (pageable, or pinned but not mapped)
    OtherDevice = 5,  ///< device memory of a different device
};

/** @brief Name of a residency class, for messages. */
inline const char* residencyName(Residency r)
{
    switch (r) {
    case Residency::Host: return "host";
    case Residency::Device: return "device";
    case Residency::Managed: return "managed";
    case Residency::MappedHost: return "mapped pinned host";
    case Residency::Pageable: return "pageable host (not device-accessible)";
    case Residency::OtherDevice: return "another device's memory";
    }
    return "unknown";
}

/** @brief Classify @p ptr for @p device (a negative device = the current one).
 *
 * Reads the pointer's attributes; the error code is not the signal (a pageable
 * pointer reports success as "unregistered"). In CPU mode every pointer is
 * ``Host``. A setup-time query: call it once per buffer, not per reorder.
 */
inline Residency classifyPointer(const void* ptr, int device = -1)
{
#ifdef EAGLE_CPU_ONLY
    (void)ptr;
    (void)device;
    return Residency::Host;
#else
    if (device < 0 && cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return Residency::Pageable;
    }
    cudaPointerAttributes a{};
    if (cudaPointerGetAttributes(&a, ptr) != cudaSuccess) {
        (void)cudaGetLastError();  // sticky-free: an unknown pointer is simply not device-accessible
        return Residency::Pageable;
    }
    switch (a.type) {
    case cudaMemoryTypeDevice:
        return a.device == device ? Residency::Device : Residency::OtherDevice;
    case cudaMemoryTypeManaged:
        return Residency::Managed;
    case cudaMemoryTypeHost:
        return a.devicePointer != nullptr ? Residency::MappedHost : Residency::Pageable;
    default:
        return Residency::Pageable;
    }
#endif
}

/** @brief Setup-time check that every plane is addressable by @p device.
 *
 * Throws ``std::invalid_argument`` naming the plane index and its class when a
 * plane is pageable or lives on another device. Device, managed and mapped
 * pinned host planes pass. Not for the per-reorder hot path.
 */
inline void checkPlanesResident(const ReorderPlane* planes, std::uint32_t nPlanes, int device = -1)
{
    for (std::uint32_t p = 0; p < nPlanes; ++p) {
        if (planes[p].data == nullptr)
            continue;
        const Residency r = classifyPointer(planes[p].data, device);
        if (r == Residency::Pageable || r == Residency::OtherDevice)
            EAGLE_THROW(std::invalid_argument,
                "reorder: plane " + std::to_string(p) + " is " + residencyName(r)
                    + "; a device reorder needs device, managed or mapped pinned host memory");
    }
}

namespace detail {

/** @brief Validate a reorder request; throws ``std::invalid_argument``.
 *  @p onDevice: the planes are read by device kernels (residency is then checked
 *  under ``EAGLE_DEBUG_MODE`` only). */
inline void checkReorder(const ReorderPlane* planes, std::uint32_t nPlanes,
    const std::uint8_t* mask, std::int64_t n, bool needMask, bool onDevice = false)
{
    checkCompact(n);
#if defined(EAGLE_DEBUG_MODE) && !defined(EAGLE_CPU_ONLY)
    if (onDevice && n > 0 && nPlanes > 0 && planes != nullptr)
        checkPlanesResident(planes, nPlanes);
#else
    (void)onDevice;
#endif
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

/** @brief Grid-stride size of a kernel: the total number of threads. */
AETHER_DEVICE() inline std::int64_t reorderStride()
{
    return std::int64_t(gridDim.x) * blockDim.x;
}

/** @brief ``stage[i] = plane[i]`` for every ``i`` in ``[first, end)``, strided. */
template<typename T>
AETHER_DEVICE() inline void stageSlice(const void* plane, void* stage, std::int64_t end)
{
    const T* src = static_cast<const T*>(plane);
    T* dst       = static_cast<T*>(stage);
    for (std::int64_t i = reorderThread(); i < end; i += reorderStride())
        dst[i] = src[i];
}

/** @brief Move the staged slice of one plane to the reordered slots. */
template<typename T>
AETHER_DEVICE() inline void placeSlice(const void* stage, void* plane, const idx_t* keep,
    const idx_t* incl, std::int64_t sp, std::int64_t cnt)
{
    const T* src = static_cast<const T*>(stage);
    T* dst       = static_cast<T*>(plane);
    for (std::int64_t i = reorderThread(); i < sp; i += reorderStride())
        dst[reorderDst(keep[i] != 0, i, std::int64_t(incl[i]), cnt)] = src[i];
}

/** @brief Place ``perm`` and, from the values it held, rebuild ``inv`` and the
 *  identity map: sample ``perm[i]`` now sits at slot ``dst(i)``, and slots
 *  ``[0, cnt)`` map to themselves. */
AETHER_DEVICE() inline void placePermSlice(const void* stage, void* plane, std::int32_t* inv,
    std::int32_t* indexMap, const idx_t* keep, const idx_t* incl, std::int64_t sp, std::int64_t cnt)
{
    const auto* src = static_cast<const std::int32_t*>(stage);
    auto* dst       = static_cast<std::int32_t*>(plane);
    for (std::int64_t i = reorderThread(); i < sp; i += reorderStride()) {
        const std::int64_t d = reorderDst(keep[i] != 0, i, std::int64_t(incl[i]), cnt);
        const std::int32_t v = src[i];
        dst[d]               = v;
        inv[v]               = std::int32_t(d);
        if (i < cnt)
            indexMap[i] = std::int32_t(i);
    }
}

/** @brief Stage every plane of a group over ``[0, *span)``: one kernel for the
 *  whole group, planes one after another, each grid-strided. */
AETHER_KERNEL()
__launch_bounds__(256, 4) inline void reorderStageGroup(ReorderGroup group, void* stage,
    const std::uint32_t* span, const std::uint32_t* fire)
{
    if (fire != nullptr && *fire == 0u)
        return;
    const std::int64_t end = std::int64_t(*span);
    for (std::uint32_t j = 0; j < group.count; ++j) {
        void* at = static_cast<char*>(stage) + group.stageOffset[j];
        switch (group.elemBytes[j]) {
        case 1u: stageSlice<std::uint8_t>(group.data[j], at, end); break;
        case 2u: stageSlice<std::uint16_t>(group.data[j], at, end); break;
        case 4u: stageSlice<std::uint32_t>(group.data[j], at, end); break;
        case 8u: stageSlice<std::uint64_t>(group.data[j], at, end); break;
        default: stageSlice<Bytes16>(group.data[j], at, end); break;
        }
    }
}

/** @brief Place every staged plane of a group; the group holding ``perm`` also
 *  rebuilds ``inv`` and ``indexMap`` (so the words are the only kernel after). */
AETHER_KERNEL()
__launch_bounds__(256, 4) inline void reorderPlaceGroup(ReorderGroup group, const void* stage,
    const idx_t* keep, const idx_t* incl, const std::uint32_t* span,
    const std::uint32_t* fire, std::int32_t* inv, std::int32_t* indexMap)
{
    if (fire != nullptr && *fire == 0u)
        return;
    const std::int64_t sp = std::int64_t(*span);
    if (sp == 0)
        return;
    const std::int64_t cnt = std::int64_t(incl[sp - 1]);
    for (std::uint32_t j = 0; j < group.count; ++j) {
        const void* at = static_cast<const char*>(stage) + group.stageOffset[j];
        if (std::int32_t(j) == group.permSlot) {
            placePermSlice(at, group.data[j], inv, indexMap, keep, incl, sp, cnt);
            continue;
        }
        switch (group.elemBytes[j]) {
        case 1u: placeSlice<std::uint8_t>(at, group.data[j], keep, incl, sp, cnt); break;
        case 2u: placeSlice<std::uint16_t>(at, group.data[j], keep, incl, sp, cnt); break;
        case 4u: placeSlice<std::uint32_t>(at, group.data[j], keep, incl, sp, cnt); break;
        case 8u: placeSlice<std::uint64_t>(at, group.data[j], keep, incl, sp, cnt); break;
        default: placeSlice<Bytes16>(at, group.data[j], keep, incl, sp, cnt); break;
        }
    }
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

/** @brief ``perm[i] = inv[i] = i`` (and ``indexMap[i] = i`` when given) over
 *  ``[0, n)``; thread 0 also sets ``*count = *span = n``. */
template<typename IndexT, typename WordT>
AETHER_KERNEL()
__launch_bounds__(256, 4) void resetIdentity(IndexT* perm, IndexT* inv,
    IndexT* indexMap, WordT* count, WordT* span, std::int64_t n)
{
    const std::int64_t i = reorderThread();
    if (i < n) {
        perm[i] = IndexT(i);
        inv[i]  = IndexT(i);
        if (indexMap != nullptr)
            indexMap[i] = IndexT(i);
    }
    if (i == 0) {
        *count = WordT(n);
        *span  = WordT(n);
    }
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

/** @brief Bytes of scratch a reorder of @p n samples needs when it stages
 *  @p stageBytesPerSample bytes per sample (raised to the widest of @p planes
 *  and ``perm``): the scan scratch, then ``stageBytesPerSample * n`` of staging.
 *  Pass the same @p stageBytesPerSample to ``reorderDevice`` / ``reorderHost``;
 *  a larger value moves more planes per kernel pair. With 0 it equals
 *  ``reorderScratchBytes(n, widest element)``. */
inline std::size_t reorderScratchBytes(std::int64_t n, const ReorderPlane* planes,
    std::uint32_t nPlanes, std::uint32_t stageBytesPerSample)
{
    const std::size_t m = std::size_t(n > 0 ? n : 1);
    return detail::reorderStageOffset(n)
        + m * detail::effectiveStageBytes(planes, nPlanes, stageBytesPerSample);
}

#ifndef EAGLE_CPU_ONLY

/**
 * @brief Enqueue the reorder of @p n samples on @p stream (device memory).
 *
 * Kernels only: predicate + inclusive scan of the mask, then one stage kernel
 * and one place kernel per group of planes (``perm`` rides in the last group,
 * whose place kernel also rebuilds ``inv`` and the identity map), and last the
 * words (``count = span = kept``, ``fire = 0``). Each kernel's grid is
 * ``min(ceil(n / 256), 4 * multiprocessors)`` blocks striding over
 * ``[0, *span)``. Legal while ``stream`` is being captured; with a ``fire``
 * word it is a no-op on the device while ``*fire == 0`` (the scan still runs,
 * into scratch).
 *
 * @p stageBytesPerSample is the staging the scratch holds per sample (see the
 * plane-table ``reorderScratchBytes``); 0 means the widest plane's size, which
 * is what ``reorderScratchBytes(n, maxElemBytes)`` provides.
 */
inline void reorderDevice(const ReorderPlane* planes, std::uint32_t nPlanes,
    const std::uint8_t* mask, std::uint32_t flags, std::int32_t* perm,
    std::int32_t* inv, std::int32_t* indexMap, std::uint32_t* count,
    std::uint32_t* span, std::uint32_t* fire, void* scratch, std::int64_t n,
    cudaStream_t stream, std::uint32_t stageBytesPerSample = 0)
{
    detail::checkReorder(planes, nPlanes, mask, n, true, true);
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
    const idx_t* kp = base;
    const idx_t* c  = base + N;
    int device = 0, sms = 0;
    EAGLE_CHECK_ALWAYS(cudaGetDevice(&device));
    EAGLE_CHECK_ALWAYS(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    const unsigned gridBlocks = std::min<unsigned>(nb, 4u * unsigned(sms > 0 ? sms : 1));
    const std::uint64_t budget = std::uint64_t(n)
        * detail::effectiveStageBytes(planes, nPlanes, stageBytesPerSample);
    ReorderGroup group;
    for (std::uint32_t k = 0; k < nPlanes + 1u;) {
        k = detail::nextReorderGroup(planes, nPlanes, perm, k, n, budget, group);
        detail::reorderStageGroup<<<gridBlocks, 256, 0, stream>>>(group, stage, span, fire);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
        detail::reorderPlaceGroup<<<gridBlocks, 256, 0, stream>>>(group, stage, kp, c, span, fire, inv, indexMap);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
    }
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
    detail::checkReorder(planes, nPlanes, nullptr, n, false, true);
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

/**
 * @brief Enqueue the reset of @p n samples on @p stream (device memory): one
 *        kernel gives ``perm``, ``inv`` and (when non-null) ``indexMap`` the
 *        identity ``0..n-1`` and sets ``*count = *span = n``.
 *
 * Kernels only, legal while ``stream`` is being captured; no ``fire`` word is
 * read or written. Argument order as for ``reorderDevice``.
 */
inline void resetDevice(std::int32_t* perm, std::int32_t* inv, std::int32_t* indexMap,
    std::uint32_t* count, std::uint32_t* span, std::int64_t n, cudaStream_t stream)
{
    if (n < 0 || n > kCompactMaxSamples)
        EAGLE_THROW(std::invalid_argument, "reset: n must be in [0, 2^31 - 1]");
    err::detail::clearStaleLastError();  // check only our own launch below
    const unsigned nb = n == 0 ? 1u : unsigned((n + 255) / 256);
    detail::resetIdentity<std::int32_t, std::uint32_t><<<nb, 256, 0, stream>>>(perm, inv, indexMap, count, span, n);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

#endif // EAGLE_CPU_ONLY

namespace detail {

/** @brief Stage then place the planes of one group, the host face of the
 *  device kernel pair: every plane is copied to its staging slice first, then
 *  scattered back to its reordered slots. */
inline void reorderGroupHost(const ReorderGroup& g, void* stage, const idx_t* keep,
    const idx_t* incl, std::int64_t sp)
{
    if (sp == 0)
        return;
    const std::int64_t cnt = std::int64_t(incl[sp - 1]);
    for (std::uint32_t j = 0; j < g.count; ++j) {
        const std::size_t bytes = std::size_t(sp) * g.elemBytes[j];
        std::memcpy(static_cast<char*>(stage) + g.stageOffset[j], g.data[j], bytes);
    }
    for (std::uint32_t j = 0; j < g.count; ++j) {
        const char* src = static_cast<const char*>(stage) + g.stageOffset[j];
        withElem(g.elemBytes[j], [&]<typename T>() {
            const T* in = reinterpret_cast<const T*>(src);
            T* out      = static_cast<T*>(g.data[j]);
#pragma omp parallel for schedule(static)
            for (std::int64_t i = 0; i < sp; ++i)
                out[reorderDst(keep[i] != 0, i, std::int64_t(incl[i]), cnt)] = in[i];
        });
    }
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
 *  the same moves in the same groups, so the two faces agree to the bit. */
inline void reorderHost(const ReorderPlane* planes, std::uint32_t nPlanes,
    const std::uint8_t* mask, std::uint32_t flags, std::int32_t* perm,
    std::int32_t* inv, std::int32_t* indexMap, std::uint32_t* count,
    std::uint32_t* span, std::uint32_t* fire, void* scratch, std::int64_t n,
    std::uint32_t stageBytesPerSample = 0)
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
    const std::uint64_t budget = std::uint64_t(n)
        * detail::effectiveStageBytes(planes, nPlanes, stageBytesPerSample);
    ReorderGroup group;
    for (std::uint32_t q = 0; q < nPlanes + 1u;) {
        q = detail::nextReorderGroup(planes, nPlanes, perm, q, n, budget, group);
        detail::reorderGroupHost(group, stage, k, c, sp);
    }
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

/** @brief The host face of ``resetDevice`` over host memory (OpenMP). */
inline void resetHost(std::int32_t* perm, std::int32_t* inv, std::int32_t* indexMap,
    std::uint32_t* count, std::uint32_t* span, std::int64_t n)
{
    if (n < 0 || n > kCompactMaxSamples)
        EAGLE_THROW(std::invalid_argument, "reset: n must be in [0, 2^31 - 1]");
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 0; i < n; ++i) {
        perm[i] = std::int32_t(i);
        inv[i]  = std::int32_t(i);
        if (indexMap != nullptr)
            indexMap[i] = std::int32_t(i);
    }
    *count = std::uint32_t(n);
    *span  = std::uint32_t(n);
}

} // namespace filtering
} // namespace eagle
