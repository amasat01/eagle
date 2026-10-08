// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include "eagle/cuda.h"
#ifndef EAGLE_CPU_ONLY
#include "eagle/cuda/WarpShuffle.h"
#endif
#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"

namespace eagle {
namespace filtering {
namespace detail {

/* Scan core routines — shared by the device (``cuda::Scan``) and host
 * (``cpu::Scan``) scanners. */
struct ScanCore {
    using Self = ScanCore;

    static constexpr unsigned int warpSize = 32;

#ifndef EAGLE_CPU_ONLY

    /** @brief Warp-level scanning */
    template<typename T, typename OP>
    AETHER_DEVICE()
    static inline T warpInclusive(T val, const idx_t& lane)
    {
#pragma unroll
        for (idx_t offset = 1; offset < warpSize; offset *= 2) {
            T n = cuda::shflUp(__activemask(), val,
                static_cast<unsigned int>(offset));
            if (lane >= offset)
                val = OP{}(n, val);
        }
        return val;
    }

#endif

    static constexpr unsigned int blockSize = 256;

    template<unsigned int blockSize_ = blockSize>
    AETHER_DEVICEHOST()
    static idx_t countBlocks(const idx_t N, const idx_t bsize = blockSize_)
    {
        return (N + bsize - 1) / bsize;
    }
};

#ifndef EAGLE_CPU_ONLY

/** @brief Run a per-block inclusive scan.
 *
 * SASS register footprint: ~17 regs (sm_61). ``(256, 4)`` documents
 * the cap consumed by ``Launcher::setLogicalSize`` re-tuning. */
template<typename T, typename OP>
AETHER_KERNEL()
__launch_bounds__(256, 4) void inclusiveScanBlock(
    CRefArrT<T> arr, GRefArrT<T> out,
    [[maybe_unused]] GRefArrT<T> blockSums, const T init)
{
    constexpr idx_t warpSize = 32;
    const idx_t numWarps     = blockDim.x / warpSize;

    extern __shared__ T shmem[];

    /* aether has no WRef: `make_work_view` IS the shared-memory View
     * (device-legal, block-local sample mode indexed by `SampleIndex::work()`
     * — `aether/view/WorkView.h`). The `reinterpret_cast` is applied AFTER the
     * `T*` pointer arithmetic so the byte offset of the second carve is
     * unchanged. */
    auto warpTotals = aether::make_work_view<idx_t>(
        reinterpret_cast<idx_t*>(shmem), numWarps); // One per warp
    auto blockOffsets = aether::make_work_view<idx_t>(
        reinterpret_cast<idx_t*>(shmem + numWarps),
        numWarps); // Final offsets per warp
    const SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);
    const idx_t lane   = threadIdx.x % warpSize;
    const idx_t warpId = threadIdx.x / warpSize;

    // Phase 1: Intra-warp scan using shuffle
    T val = ScanCore::warpInclusive<T, OP>(
        (i.global() < arr.samples()) ? arr.eval(i) : init, lane);

    // Store warp totals
    if (lane == 31)
        warpTotals(warpId) = val;
    __syncthreads();

    // Phase 2: Scan warp totals (only first warp does this)
    if (warpId == 0 && lane < numWarps) {
        T warpVal = ScanCore::warpInclusive<T, OP>(warpTotals(lane), lane);
        blockOffsets(lane) = OP{}(init, warpVal);
    }
    __syncthreads();

    // Phase 3: Add warp offset
    if (warpId > 0)
        val = OP{}(blockOffsets(warpId - 1), val);

    // Write result
    if (i.global() < arr.samples())
        out.eval(i) = val;

    // Write block sum
    if (threadIdx.x == blockDim.x - 1)
        blockSums(blockIdx.x) = val;
}

/** @brief Kernel to add the block offsets to the scanned indexes.
 *
 * SASS register footprint: ~6 regs (sm_61). */
template<typename T, typename OP>
AETHER_KERNEL()
__launch_bounds__(256, 4) void applyOffsets(
    GRefArrT<T> out, CRefArrT<T> blockSums)
{
    const SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= out.samples())
        return;

    if (blockIdx.x > 0) {
        const T offset = blockSums(blockIdx.x - 1);
        out.eval(i)    = OP{}(out.eval(i), offset);
    }
}

/** @brief Kernel to apply right-shifting to retrieve exclusive scan results.
 *
 * SASS register footprint: ~6 regs (sm_61). */
template<typename T, typename OP>
AETHER_KERNEL()
__launch_bounds__(256, 4) void rightShift(
    GRefArrT<T> to, CRefArrT<T> from, const T init)
{
    const SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= to.samples())
        return;

    if (i.global() == 0)
        to(0) = init;
    else {
        to.eval(i) = from(i.global() - 1);
    }
}

#endif

} // namespace detail
} // namespace filtering
} // namespace eagle
