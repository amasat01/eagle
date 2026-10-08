// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include <type_traits>

namespace eagle {
namespace cpu {

struct Host {

    /** @brief Safe scalar packet load — uses masked load for tail packets. */
    template<typename DataT, std::size_t W>
    static inline aether::simd::Packet<DataT, W> packetLoad(
        const DataT* ptr, const aether::PacketIndex<W>& pi)
    {
        using PacketT = aether::simd::Packet<DataT, W>;
        using MaskT   = aether::simd::PacketMask<DataT, W>;
        if (pi.full())
            return PacketT::load(ptr + pi.base_);
        else
            return PacketT::maskLoad(
                ptr + pi.base_, MaskT::firstN(pi.active_));
    }

    /** @brief View overload — forwards to the raw-pointer version.
     *  Accepts any object exposing ``.data()`` (an ``aether::View`` over a
     *  scalar array, mutable or const) so callers don't need a
     *  ``T* xPtr = view.data();`` local. */
    template<typename DataT, std::size_t W, typename ArrayLikeT>
        requires (!std::is_pointer_v<std::remove_cvref_t<ArrayLikeT>>)
              && requires(const ArrayLikeT& a) { a.data(); }
    static inline aether::simd::Packet<DataT, W> packetLoad(
        const ArrayLikeT& arr, const aether::PacketIndex<W>& pi)
    {
        return packetLoad<DataT, W>(arr.data(), pi);
    }

    /** @brief Safe scalar packet store — uses masked store for tail packets. */
    template<typename DataT, std::size_t W>
    static inline void packetStore(
        DataT* ptr, const aether::PacketIndex<W>& pi,
        const aether::simd::Packet<DataT, W>& val)
    {
        using PacketT = aether::simd::Packet<DataT, W>;
        using MaskT   = aether::simd::PacketMask<DataT, W>;
        if (pi.full())
            PacketT::store(ptr + pi.base_, val);
        else
            PacketT::maskStore(
                ptr + pi.base_, MaskT::firstN(pi.active_), val);
    }

    /** @brief View overload — forwards to the raw-pointer version. */
    template<typename DataT, std::size_t W, typename ArrayLikeT>
        requires (!std::is_pointer_v<std::remove_cvref_t<ArrayLikeT>>)
              && requires(const ArrayLikeT& a) { a.data(); }
    static inline void packetStore(
        const ArrayLikeT& arr, const aether::PacketIndex<W>& pi,
        const aether::simd::Packet<DataT, W>& val)
    {
        packetStore<DataT, W>(arr.data(), pi, val);
    }

    /** @brief Build a PacketMask from a contiguous bool array at the given
     *  packet index.  Each lane whose bool is `true` becomes active. */
    template<typename DataT, std::size_t W>
    static inline aether::simd::PacketMask<DataT, W> loadMask(
        const bool* flags, const aether::PacketIndex<W>& pi)
    {
        using MaskT = aether::simd::PacketMask<DataT, W>;
        unsigned bits = 0;
        const idx_t n = pi.active_;
        for (idx_t k = 0; k < n; ++k)
            bits |= (static_cast<unsigned>(flags[pi.base_ + k]) << k);
        // W==1 -- every non-double/float DataT via PreferredWidth (e.g. an
        // emulated scalar carrier), and double itself at an explicit W=1 --
        // resolves to aether::simd::PacketMask<DataT,1>, which has no
        // `StorageT` member, only a bare `bool mask_`
        // (`aether::simd::PacketMask<DataT, 1>`).
        // The wide-register primary template (Width>1) is the only one
        // exposing `StorageT`, so its bitmask construction below cannot
        // be used at W=1; branch at compile time on the (fixed,
        // caller-chosen) width rather than assuming either surface.
        if constexpr (W == 1)
            return MaskT{bits != 0};
        else
            return MaskT{static_cast<typename MaskT::StorageT>(bits)};
    }

    /** @brief View overload — forwards to the raw-pointer version. */
    template<typename DataT, std::size_t W, typename ArrayLikeT>
        requires (!std::is_pointer_v<std::remove_cvref_t<ArrayLikeT>>)
              && requires(const ArrayLikeT& a) { a.data(); }
    static inline aether::simd::PacketMask<DataT, W> loadMask(
        const ArrayLikeT& flags, const aether::PacketIndex<W>& pi)
    {
        return loadMask<DataT, W>(flags.data(), pi);
    }

    /** @brief Dispatch active mask lanes to a scalar kernel. */
    template<std::size_t W, typename MaskT, typename KernelFunc>
    static inline void dispatchMasked(
        const aether::PacketIndex<W>& pi, const MaskT& active,
        KernelFunc&& kernel)
    {
        if (!active.anyTrue()) return;
        for (idx_t k = 0; k < pi.active_; ++k) {
            if (active.lane(k))
                kernel(pi.scalar(k));
        }
    }

    /** @brief Combine a mask with the tail mask for partial packets.
     *
     *  Templated on `DataT` like `packetLoad`/`packetStore`/`loadMask` --
     *  this was previously the one function in the file hard-typed to
     *  `Real` (double), so it could not accept a mask built over any other
     *  `DataT` (e.g. SoftDouble). `DataT` is deduced from `mask` rather
     *  than listed before `W` so every existing call site
     *  (`applyTail<W>(...)` in `launchIfNot`/`launchIf`/
     *  `packetLaunchIfNot` below, and `eagle/util/Slice.h`'s
     *  `applyTail<W>(PacketMask<Real,W>::allTrue(), pi)`) keeps compiling
     *  unchanged -- their single explicit template argument still binds to
     *  `W`, exactly as before. Zero behavior change for the wide-`double`
     *  production path. */
    template<std::size_t W, typename DataT>
    static inline aether::simd::PacketMask<DataT, W> applyTail(
        aether::simd::PacketMask<DataT, W> mask,
        const aether::PacketIndex<W>& pi)
    {
        if (!pi.full())
            mask = mask & aether::simd::PacketMask<DataT, W>::firstN(pi.active_);
        return mask;
    }

    /** @brief Launcher wrapper — SIMD packet-based, processes W samples per
     *  iteration.  Uses context-aware ``packetBatchedFor``, which always
     *  passes ``parallel=false`` here — this call **never opens its own**
     *  ``#pragma omp parallel`` **region**:
     *   - If the caller is already inside an existing ``#pragma omp
     *     parallel`` region, work is shared across that region's threads
     *     via ``#pragma omp for`` (aether
     *     ``backend/cpu/Tiled.h``'s ``packetBatchedFor``).
     *   - Otherwise it runs **serially** (SIMD-vectorised, single-threaded)
     *     — a caller expecting multithreading here without an enclosing
     *     region gets a silent full serialization: it compiles, produces
     *     correct values, and loses all threading.
     *
     *  Region-scoping is the caller's responsibility. Consecutive
     *  ``launch``/``launchIf``/``launchIfNot``/``packetLaunch*`` calls made
     *  inside ONE such region have **no implicit barrier between them** —
     *  the work-share arm uses ``nowait`` (same aether citation above), so a
     *  later call's tiles may start before an earlier call's tiles have all
     *  finished on every thread; insert an explicit ``#pragma omp barrier``
     *  between them if that ordering matters.
     *
     *  Production exemplar of correct use: a caller that opens a
     *  ``#pragma omp parallel`` region and calls ``Host::launch`` from
     *  inside it — that is what makes the call actually
     *  ``Host::launch`` call actually multithread. Do NOT copy
     *  ``docs/examples/02_host_dispatch.cpp`` (a standalone, region-free
     *  correctness demo) as a threading example.
     *
     *  @param bytesPerSample  Caller's per-sample working-set estimate.
     *                         ``0`` (default) falls back to aether's
     *                         compile-time ``DEFAULT_TILE_SIZE``.  A
     *                         non-zero value drives runtime L2-derived
     *                         tile sizing — pass the same value across
     *                         every host kernel call in a step so each
     *                         thread's tile slice stays L2-resident
     *                         across kernels. */
    template<typename KernelFunc>
    static inline void launch(const idx_t& size, KernelFunc&& kernel,
        idx_t bytesPerSample = 0)
    {
        aether::packetBatchedFor<Real>(
            idx_t{0}, size, [&](const auto& pi) {
                for (idx_t k = 0; k < pi.active_; ++k)
                    kernel(pi.scalar(k));
            }, bytesPerSample, /*parallel=*/false);
    }

    /** @brief Launcher wrapper, excluding items where the flag is true.
     *
     *  Loads W boolean flags per packet into a PacketMask and skips
     *  the entire packet when no lanes are active, avoiding per-sample
     *  branch mispredictions.  ``bytesPerSample`` follows the same
     *  semantics as ``launch``. */
    template<typename KernelFunc>
    static inline void launchIfNot(const idx_t& size,
        CRefArrT<bool> terminated,
        KernelFunc&& kernel, idx_t bytesPerSample = 0)
    {
        const bool* flags = terminated.data();
        aether::packetBatchedFor<Real>(
            idx_t{0}, size, [&](const auto& pi) {
                constexpr idx_t W =
                    std::remove_cvref_t<decltype(pi)>::width;
                auto active = applyTail<W>(
                    ~loadMask<Real, W>(flags, pi), pi);
                dispatchMasked(pi, active, kernel);
            }, bytesPerSample, /*parallel=*/false);
    }

    /** @brief Launcher wrapper, including only items where the flag is true.
     *
     *  Loads W boolean flags per packet into a PacketMask and skips
     *  the entire packet when no lanes are active.  ``bytesPerSample``
     *  follows the same semantics as ``launch``. */
    template<typename KernelFunc>
    static inline void launchIf(const idx_t& size,
        const CRefArrT<bool> terminated,
        KernelFunc&& kernel, idx_t bytesPerSample = 0)
    {
        const bool* flags = terminated.data();
        aether::packetBatchedFor<Real>(
            idx_t{0}, size, [&](const auto& pi) {
                constexpr idx_t W =
                    std::remove_cvref_t<decltype(pi)>::width;
                auto active = applyTail<W>(
                    loadMask<Real, W>(flags, pi), pi);
                dispatchMasked(pi, active, kernel);
            }, bytesPerSample, /*parallel=*/false);
    }
    /** @brief Packet-aware launch — passes PacketIndex directly to kernel.
     *  The kernel receives a PacketIndex<W> and operates on W samples at once.
     *  ``bytesPerSample`` follows the same semantics as ``launch``. */
    template<typename KernelFunc>
    static inline void packetLaunch(const idx_t& size, KernelFunc&& kernel,
        idx_t bytesPerSample = 0)
    {
        aether::packetBatchedFor<Real>(
            idx_t{0}, size, kernel, bytesPerSample, /*parallel=*/false);
    }

    /** @brief Packet-aware launch excluding terminated samples.
     *  Skips entire packets where all lanes are terminated.
     *  For mixed packets, the kernel receives the full PacketIndex
     *  and must handle masking internally (or use the provided mask).
     *  ``bytesPerSample`` follows the same semantics as ``launch``. */
    template<typename KernelFunc>
    static inline void packetLaunchIfNot(const idx_t& size,
        CRefArrT<bool> terminated,
        KernelFunc&& kernel, idx_t bytesPerSample = 0)
    {
        const bool* flags = terminated.data();
        aether::packetBatchedFor<Real>(
            idx_t{0}, size, [&](const auto& pi) {
                constexpr idx_t W =
                    std::remove_cvref_t<decltype(pi)>::width;
                auto active = applyTail<W>(
                    ~loadMask<Real, W>(flags, pi), pi);
                if (active.anyTrue())
                    kernel(pi);
            }, bytesPerSample, /*parallel=*/false);
    }
};

} // namespace cpu
} // namespace eagle
