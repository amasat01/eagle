// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"

#ifndef EAGLE_CPU_ONLY

#include <algorithm>

#include <cuda_runtime.h>

namespace eagle {
namespace cuda {

/** @brief Cached single-device SM count. */
inline int currentSMCount()
{
    static const int nSMs = []() {
        int dev = 0;
        int n   = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        return n;
    }();
    return nSMs;
}

/** @brief Cached FP32:FP64 throughput ratio of the current device.
 *
 *  ``cudaDevAttrSingleToDoublePrecisionPerfRatio`` — e.g. 2 on data-center
 *  parts (GV100/A100/H100), 32 on FP64-throttled workstation parts (sm_61
 *  Pascal, Turing), 64 on consumer Ampere/Ada. Consumers use it to select
 *  between native-FP64 and emulated (SoftDouble) kernel sets at graph
 *  construction time: emulation pays off when the ratio is large. */
inline int currentFp64PerfRatio()
{
    static const int ratio = []() {
        int dev = 0;
        int r   = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(
            &r, cudaDevAttrSingleToDoublePrecisionPerfRatio, dev);
        return r;
    }();
    return ratio;
}

/** @brief Block-size policy for an N-sample kernel.
 *
 *  Computes ``(blockSize, nBlocks)`` over ``numStates`` work items
 *  given the kernel's ``__launch_bounds__``-derived
 *  ``idealBlockSize`` cap and a target ``blocksPerSM``. Targets
 *  ``nSMs * blocksPerSM`` blocks, rounds up to WARP=32, clamps to
 *  ``[WARP, idealBlockSize]``.
 *
 *  The caller supplies ``blocksPerSM`` rather than calling
 *  ``cudaOccupancyMaxActiveBlocksPerMultiprocessor`` because that
 *  fails with ``cudaErrorInvalidResourceHandle`` when the consumer
 *  DSO links a statically-linked cudart distinct from the kernel
 *  stub's. */
inline void computeBlocks(idx_t numStates, idx_t& nBlocks, idx_t& blockSize,
    idx_t idealBlockSize = 256, idx_t blocksPerSM = 4)
{
    constexpr idx_t WARP = 32;

    /* Handle the 0 numStates case. Defaults to warp-sized blocksize but 0 num
     * blocks */
    if (numStates == 0) {
        nBlocks   = 0;
        blockSize = WARP;
        return;
    }

    const idx_t gridSize = static_cast<idx_t>(currentSMCount()) * blocksPerSM;

    idx_t bs  = (numStates + gridSize - 1) / gridSize;
    bs        = ((bs + WARP - 1) / WARP) * WARP;
    bs        = std::max<idx_t>(WARP, std::min<idx_t>(idealBlockSize, bs));
    blockSize = bs;
    nBlocks   = (numStates + bs - 1) / bs;
}

/** @brief ``Traits``-taking overload: block-size policy from a kernel's
 *  compile-time launch traits — ``eagle::launch::Traits`` or any type
 *  exposing ``maxBlockSize``/``minBlocksPerSM`` constants (duck-typed so
 *  this header carries no dependency on ``eagle/launch/Traits.h``).
 *
 *  Explicit opt-in by design: pre-existing call sites keep passing values,
 *  because auto-threading a trait's ``minBlocksPerSM`` into a site that
 *  defaulted ``blocksPerSM=4`` would change its runtime grid silently. */
template <typename LaunchTraits>
inline void computeBlocks(idx_t numStates, idx_t& nBlocks, idx_t& blockSize)
{
    computeBlocks(numStates, nBlocks, blockSize, LaunchTraits::maxBlockSize,
        LaunchTraits::minBlocksPerSM);
}

} // namespace cuda
} // namespace eagle

#endif // EAGLE_CPU_ONLY
