// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/exec/HostPool.h"
#include "eagle/filtering/scan_detail.h"
#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"

#include <algorithm>
#include <vector>

namespace eagle {
namespace cpu {

/** @brief Prefix sum (scan) on eagle's host pool */
struct Scan {
    using Self = Scan;

    /** @brief Run the overall scan on at most ``team`` host threads: every block is scanned on its own, the block
     *  totals are prefixed in order, and each block adds its offset. The blocks are fixed by ``blockSize_`` and the
     *  prefix runs in block order, so the results do not depend on ``team``. */
    template<typename T, typename OP, bool Inclusive,
        unsigned int blockSize_ = 1024>
    static void scan(const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        int team = exec::host_max_threads())
    {
        const idx_t N         = arr.samples();
        const idx_t blockSize = std::min(blockSize_, N);
        const idx_t numBlocks
            = filtering::detail::ScanCore::countBlocks<blockSize_>(N, blockSize);

        exec::hostFor(numBlocks, team, [&](std::int64_t b) {
            const idx_t start = idx_t(b) * blockSize;
            const idx_t end   = std::min(start + blockSize, N);
            T acc             = init;
            // Serial within a block: each result depends on the previous
            // one; the blocks are the parallel grain.
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
        });

        T acc = init;
        for (idx_t b = 0; b < numBlocks; ++b) {
            T temp       = blockSums(b);
            blockSums(b) = acc;
            acc          = OP{}(acc, temp);
        }

        const CRefArrT<T> offsets = blockSums.as_const();
        exec::hostFor(numBlocks, team, [&](std::int64_t b) {
            const idx_t start = idx_t(b) * blockSize;
            const idx_t end   = std::min(start + blockSize, N);
            const T offset    = offsets(b);
            for (idx_t i = start; i < end; ++i)
                results(i) = OP{}(results(i), offset);
        });
    }

    /** @brief Run the inclusive scan */
    template<typename T, typename OP, unsigned int blockSize_ = 1024>
    static void inclusive(const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        int team = exec::host_max_threads())
    {
        Scan::scan<T, OP, true, blockSize_>(arr, results, blockSums, init, team);
    }

    /** @brief Run the exclusive scan */
    template<typename T, typename OP, unsigned int blockSize_ = 1024>
    static void exclusive(const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        int team = exec::host_max_threads())
    {
        Scan::scan<T, OP, false, blockSize_>(arr, results, blockSums, init, team);
    }
};

} // namespace cpu
} // namespace eagle
