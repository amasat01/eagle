// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"

#ifndef EAGLE_CPU_ONLY
#include <cuda_runtime_api.h>

#include "eagle/cuda/WarpShuffle.h"
#endif

namespace eagle {
namespace reduce {
namespace detail {

/** @brief Reductions */
struct Core {
    using Self = Core;

#ifndef EAGLE_CPU_ONLY

    /** @brief Warp reduction using shuffle intrinsics */
    template<typename T, typename OP>
    AETHER_DEVICE()
    static inline T warpShuffleReduce(T val, unsigned int mask = 0xffffffffu)
    {
#pragma unroll
        for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
            T other = cuda::shflDown(mask, val, offset);
            val     = OP{}(val, other);
        }
        return val;
    }

#endif

    static constexpr unsigned int blockSize = 256;

    template<unsigned int blockSize_ = blockSize>
    static idx_t countBlocks(const idx_t N, const idx_t bsize = blockSize_)
    {
        int nBlocks = static_cast<int>(
            aether::math::floor(double(N) / (aether::math::log(double(N)) * bsize)));
        return std::max(1, nBlocks);
    }
};

#ifndef EAGLE_CPU_ONLY

/** @brief Run a single reduction.
 *
 * Block size is FIXED by the warp-shuffle reduction layout: the
 * recursive ``sdata[threadIdx.x + offset]`` pattern is only valid when
 * blockDim equals the templated ``blockSize_`` exactly. NEVER feed this
 * kernel through ``Launcher::setLogicalSize`` (which mutates blockDim).
 * SASS register footprint: ~8 regs (sm_61). */
template<typename T, typename OP, unsigned int blockSize_>
AETHER_KERNEL()
__launch_bounds__(blockSize_, 1) void reduceOnce(
    CRefArrT<T> g_idata, GRefArrT<T> g_odata, unsigned int n,
    const T init)
{
    static_assert(!(blockSize_ == 0) && !(blockSize_ & (blockSize_ - 1)),
        "Block size must be a power of two!");
    static_assert(blockSize_ <= 1024, "Maximum blockSize is 1024!");

    unsigned int in             = threadIdx.x + blockIdx.x * blockDim.x;
    const unsigned int gridSize = blockSize_ * gridDim.x;
    T val                       = init;

    while (in < n) {
        val = OP{}(val, g_idata(in));
        in += gridSize;
    }

    /* load data in shared memory */
    __shared__ T sdata[blockSize_];
    sdata[threadIdx.x] = val;
    __syncthreads();

    /* Do block-based reduction */
    if constexpr (blockSize_ >= 1024) {
        if (threadIdx.x < 512)
            sdata[threadIdx.x]
                = OP{}(sdata[threadIdx.x], sdata[threadIdx.x + 512]);
        __syncthreads();
    }
    if constexpr (blockSize_ >= 512) {
        if (threadIdx.x < 256)
            sdata[threadIdx.x]
                = OP{}(sdata[threadIdx.x], sdata[threadIdx.x + 256]);
        __syncthreads();
    }
    if constexpr (blockSize_ >= 256) {
        if (threadIdx.x < 128)
            sdata[threadIdx.x]
                = OP{}(sdata[threadIdx.x], sdata[threadIdx.x + 128]);
        __syncthreads();
    }
    if constexpr (blockSize_ >= 128) {
        if (threadIdx.x < 64)
            sdata[threadIdx.x]
                = OP{}(sdata[threadIdx.x], sdata[threadIdx.x + 64]);
        __syncthreads();
    }

    /* Final warp-based reduction*/
    if (threadIdx.x < 32) {
        if constexpr (blockSize_ >= 64) {
            sdata[threadIdx.x]
                = OP{}(sdata[threadIdx.x], sdata[threadIdx.x + 32]);
        }
        T out = Core::warpShuffleReduce<T, OP>(sdata[threadIdx.x]);
        if (threadIdx.x == 0)
            g_odata(blockIdx.x) = out;
    }
}

#endif

} // namespace detail
} // namespace reduce
} // namespace eagle
