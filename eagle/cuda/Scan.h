// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <type_traits>

#include "eagle/cuda/ScanKernels.cuh"
#include "eagle/cuda/StreamCapturer.h"
#include "eagle/typedefs.h"
#include "eagle/cuda.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"

namespace eagle {
namespace cuda {

#ifndef EAGLE_CPU_ONLY

/** @brief The device prefix sum: the single-pass scan of ``cuda/ScanKernels.cuh`` over a one-dimensional array. */
struct Scan {

    /** @brief Enqueue the scan on @p stream, without capturing it.
     *
     *  The launches ``graph`` captures, issued straight onto ``stream``: nothing is read back to the host and nothing is
     *  synchronised, so a caller that is ALREADY capturing ``stream`` records it into its own graph, and an uncaptured caller
     *  runs it asynchronously. Only the sum is written: ``OP`` is ``aether::SumOp<T>``, started from zero (@p init must
     *  be zero). Its working memory is @p blockSums, which is as long as @p arr.
     *
     *  A real sum is the same bits on every run: its grouping depends on the extent alone. */
    template<typename T, typename OP, bool Inclusive>
    static inline void enqueue(
        const CRefArrT<T> arr, GRefArrT<T> results,
        [[maybe_unused]] GRefArrT<T> blockSums, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        static_assert(std::is_same_v<OP, aether::SumOp<T>>,
            "cuda::Scan: only the sum is written on the device");
        const idx_t N = idx_t(arr.samples());
        EAGLE_TRACE("Starting scan over %i elements", N);
        EAGLE_ASSERT(arr.data() != results.data(),
            "Reduction array and work buffer must be distinct!");
        EAGLE_ASSERT(idx_t(results.samples()) == N,
            "Output array `results` must be the same size as `arr`!");
        EAGLE_ASSERT(init == T(0), "cuda::Scan starts from zero");
        if (N == 0)
            return;
        // One tile needs no working memory. More need ``state_bytes`` (a few percent of the array), which @p blockSums
        // (as long as the array) holds, cleared here because the scan expects it zero.
        const std::size_t bytes = scan::state_bytes<T>(N);
        void* state             = nullptr;
        if (scan::tiles_of<T>(N) > 1) {
            EAGLE_ASSERT(std::size_t(blockSums.samples()) * sizeof(T) >= bytes + 16,
                "cuda::Scan: `blockSums` must be as long as `arr`!");
            // the working memory is read and written in 16-byte steps: start it on a 16-byte boundary
            const auto raw = reinterpret_cast<std::uintptr_t>(blockSums.data());
            state          = reinterpret_cast<void*>((raw + 15) & ~std::uintptr_t(15));
            EAGLE_CHECK_ALWAYS(cudaMemsetAsync(state, 0, bytes, stream));
        }
        scan::scan_queue<T, scan::kSum>(
            stream, arr.data(), results.data(), N, !Inclusive, state);
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
    }

    /** @brief Capture the scan (``enqueue``) into a graph */
    template<typename T, typename OP, bool Inclusive>
    [[nodiscard]] static inline cudaGraph_t graph(
        const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        cuda::StreamCapturer capturer(stream);
        capturer.begin();
        enqueue<T, OP, Inclusive>(arr, results, blockSums, init, stream);
        return capturer.end();
    }
};

#endif // EAGLE_CPU_ONLY

} // namespace cuda
} // namespace eagle
