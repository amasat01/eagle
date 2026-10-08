// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/filtering/scan_detail.h"
#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"

#include <omp.h>
#include <algorithm>
#include <vector>

namespace eagle {
namespace cpu {

/** @brief OpenMP parallelized prefix sum (scan) */
struct Scan {
    using Self = Scan;

    /** @brief Run the overall scan.
     *
     *  Context-aware: nested in a parallel region, ``scanBlock_`` and
     *  ``applyOffsets_`` work-share via ``omp for`` and the sequential
     *  prefix is wrapped in ``omp single``. Outside a parallel region,
     *  each step opens its own ``parallel for``. */
    template<typename T, typename OP, bool Inclusive,
        unsigned int blockSize_ = 1024>
    static void scan(const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0)
    {
        const idx_t N = arr.samples();
        auto blockSize
            = [](const idx_t& size) { return std::min(blockSize_, size); };

        std::vector<idx_t> blockSizes;
        std::vector<idx_t> blocks;
        blockSizes.push_back(blockSize(N));
        blocks.push_back(filtering::detail::ScanCore::countBlocks<blockSize_>(
            N, blockSizes.back()));

        Self::scanBlock_<T, OP, Inclusive>(
            arr, results, blockSums, blockSizes[0], blocks[0], init);

        /* ``packetFlatFor`` uses ``omp for nowait`` when nested — explicit
         * barrier ensures the prefix below sees fully-written blockSums. */
        if (omp_in_parallel()) {
#pragma omp barrier
        }

        if (omp_in_parallel()) {
#pragma omp single
            {
                T acc = init;
                for (idx_t b = 0; b < blocks[0]; ++b) {
                    T temp       = blockSums(b);
                    blockSums(b) = acc;
                    acc          = OP{}(acc, temp);
                }
            }
        } else {
            T acc = init;
            for (idx_t b = 0; b < blocks[0]; ++b) {
                T temp       = blockSums(b);
                blockSums(b) = acc;
                acc          = OP{}(acc, temp);
            }
        }

        Self::applyOffsets_<T, OP>(
            results, blockSums.as_const(), blockSizes[0], blocks[0]);

        /* Trailing barrier (same nowait reason as above) so consumers
         * see fully-written ``results``. */
        if (omp_in_parallel()) {
#pragma omp barrier
        }
    }

    /** @brief Run the inclusive scan */
    template<typename T, typename OP, unsigned int blockSize_ = 1024>
    static void inclusive(const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0)
    {
        Scan::scan<T, OP, true, blockSize_>(arr, results, blockSums, init);
    }

    /** @brief Run the exclusive scan */
    template<typename T, typename OP, unsigned int blockSize_ = 1024>
    static void exclusive(const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0)
    {
        Scan::scan<T, OP, false, blockSize_>(arr, results, blockSums, init);
    }

protected:
    /** @brief Parallel inclusive scan, context-aware via
     *  ``aether::packetFor`` (``DataT=bool``, W=1; per-sample
     *  loop carries its own ``omp simd``). The calling graph inserts the
     *  necessary barriers when nested. */
    template<typename T, typename OP, bool Inclusive>
    static void scanBlock_(const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const idx_t blockSize,
        const idx_t numBlocks, T init)
    {
        aether::packetFor<bool>(
            idx_t{ 0 }, numBlocks,
            [&](const auto& pi) {
                const idx_t b     = pi.scalar(0).global();
                const idx_t start = b * blockSize;
                const idx_t end = std::min(start + blockSize, idx_t(arr.samples()));
                T acc             = init;
#pragma omp simd
                for (idx_t i = start; i < end; ++i) {
                    if constexpr (Inclusive) {
                        acc        = OP{}(acc, arr(i));
                        results(i) = acc;
                    } else {
                        results(i) = acc;
                        acc        = OP{}(acc, arr(i));
                    }
                }
                blockSums(b) = acc;
            },
            /*parallel=*/true);
    }

    /** @brief Apply offsets.  Mirrors ``scanBlock_`` shape. */
    template<typename T, typename OP>
    static void applyOffsets_(GRefArrT<T> results,
        const CRefArrT<T> blockSums, const idx_t blockSize,
        const idx_t numBlocks)
    {
        aether::packetFor<bool>(
            idx_t{ 0 }, numBlocks,
            [&](const auto& pi) {
                const idx_t b     = pi.scalar(0).global();
                const idx_t start = b * blockSize;
                const idx_t end = std::min(start + blockSize, idx_t(results.samples()));
                const T offset    = blockSums(b);
#pragma omp simd
                for (idx_t i = start; i < end; ++i) {
                    results(i) = OP{}(results(i), offset);
                }
            },
            /*parallel=*/true);
    }
};

} // namespace cpu
} // namespace eagle
