// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/filtering/scan_detail.h"
#include "eagle/typedefs.h"
#include "eagle/cuda.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"

namespace eagle {
namespace cuda {

#ifndef EAGLE_CPU_ONLY

/** @brief Perform the Prefix sum (scan) algorithm */
struct Scan {

    /** @brief Enqueue the full scan on @p stream, without capturing it.
     *
     *  The launch sequence ``graph`` captures, issued straight onto
     *  ``stream``: nothing is allocated, nothing is synchronised and nothing
     *  is read back to the host, so a caller that is ALREADY capturing
     *  ``stream`` (a graph-building consumer) records it into its own graph,
     *  and an uncaptured caller simply runs it asynchronously. ``blockSums``
     *  must hold at least ``arr.samples()`` elements. The exclusive variant
     *  ends with a device-to-device copy; the inclusive one is kernels only. */
    template<typename T, typename OP, bool Inclusive,
        unsigned int blockSize_ = filtering::detail::ScanCore::blockSize>
    static inline void enqueue(
        const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        constexpr idx_t warpSize = 32;
        const idx_t N            = idx_t(arr.samples());
        auto blockSize           = [](const idx_t& size) {
            return std::min(blockSize_,
                          (idx_t)(((size + warpSize - 1) / warpSize) * warpSize));
        };

        EAGLE_TRACE("Starting scan over %i elements", N);
        EAGLE_ASSERT(arr.data() != results.data(),
            "Reduction array and work buffer must be distinct!");
        EAGLE_ASSERT(idx_t(results.samples()) == N,
            "Output array `results` must be the same size as `arr`!");

        /* Pre-compute the blocks and, therefore, the number of recursion steps
         */
        std::vector<idx_t> blockSizes;
        std::vector<idx_t> blocks;
        std::vector<idx_t> shmemSizes;
        blockSizes.push_back(blockSize(N));
        blocks.push_back(
            filtering::detail::ScanCore::countBlocks<blockSize_>(
                N, blockSizes.back()));
        shmemSizes.push_back(
            2 * sizeof(idx_t) * (blockSizes.back() / warpSize));
        while (blocks.back() > 1) {
            blockSizes.push_back(blockSize(blocks.back()));
            blocks.push_back(filtering::detail::ScanCore::countBlocks<blockSize_>(
                blocks.back(), blockSizes.back()));
            shmemSizes.push_back(
                2 * sizeof(idx_t) * (blockSizes.back() / warpSize));
        }

        using RefT = GRefArrT<T>;
        auto refOver = [](T* p, idx_t n) { return spanView<T>(p, n, deviceDevice()); };
        /* We can safely use the bottom half of blockSums because blockSize_ >=
         * 32 by definition */
        const idx_t nSteps = blocks.size() - 1;
        RefT buffer = refOver(blockSums.data() + N / 2, N / 2);
        std::vector<RefT> res;
        std::vector<RefT> bsums;
        /* reserve the space for res and bsums */
        res.reserve(nSteps);
        bsums.reserve(nSteps);
        idx_t start = 0;
        for (idx_t i = 1; i < blocks.size(); i++) {
            res.push_back(refOver(buffer.data() + start, blocks[i - 1]));
            start += blocks[i - 1];
            bsums.push_back(refOver(blockSums.data() + start, blocks[i]));
        }

        /* Step 1: first level scan (always run, even for single-block cases) */
        // numBlocks is the number of blocks (blockSums size)
        filtering::detail::inclusiveScanBlock<T, OP>
            <<<blocks[0], blockSizes[0], shmemSizes[0], stream>>>(
                arr, results, blockSums, init);

        /* Multi-level recursion is necessary only when more than one block */
        if (blocks.size() > 1) {

            /* More scan launches for multi-block case */
            idx_t offset = 0;
            for (idx_t i = 0; i < res.size(); i++) {
                /* Force inclusivity for the blocks */
                filtering::detail::inclusiveScanBlock<T, OP>
                    <<<blocks[i + 1], blockSizes[i + 1], shmemSizes[i + 1],
                        stream>>>(
                        refOver(blockSums.data() + offset, blocks[i])
                            .as_const(),
                        res[i], bsums[i], init);
                offset += blocks[i];
            }

            /* apply the offsets in the opposite direction */
            for (idx_t i = res.size(); i > 0; i--) {
                filtering::detail::applyOffsets<T, OP>
                    <<<blocks[i], blockSizes[i], 0, stream>>>(
                        res[i - 1],
                        ((i == res.size()) ? bsums[i - 1] : res[i]).as_const());
            }
        }

        // Step 3: apply the final offsets
        filtering::detail::applyOffsets<T, OP>
            <<<blocks[0], blockSizes[0], 0, stream>>>(
                results, buffer.as_const());

        /* final optional step: do the right shift for exclusive case */
        if constexpr (!Inclusive) {
            filtering::detail::rightShift<T, OP>
                <<<blocks[0], blockSizes[0], 0, stream>>>(
                    blockSums, results.as_const(), init);
            /* The exclusive shift's D2D move routes through
             * `aether::copyAsync(View, View, Stream)` (aether/view/Copy.h)
             * instead of a raw `cudaMemcpyAsync`: aether owns residency and
             * the legal device-pair matrix, eagle owns only WHEN the move
             * happens (here: inside this capture region, so it lands as a
             * memcpy node in the captured graph). `blockSums` is a scratch
             * span at least N long; the copy is over its FIRST N elements, so
             * it is re-viewed at exactly N to match `results` — copyAsync
             * refuses an extents/strides mismatch rather than moving the
             * wrong bytes. */
            aether::copyAsync(results,
                refOver(blockSums.data(), N).as_const(), stream);
        }
    }

    /** @brief Capture the full scan algorithm (``enqueue``) into a graph */
    template<typename T, typename OP, bool Inclusive,
        unsigned int blockSize_ = filtering::detail::ScanCore::blockSize>
    static inline cudaGraph_t graph(
        const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        cuda::StreamCapturer capturer(stream);
        capturer.begin();
        enqueue<T, OP, Inclusive, blockSize_>(
            arr, results, blockSums, init, stream);
        return capturer.end();
    }

    /** @brief Instantiate and launch*/
    template<typename T, typename OP, bool Inclusive,
        unsigned int blockSize_ = filtering::detail::ScanCore::blockSize>
    static inline void launch(const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        cuda::Graph graph;
        graph.addNode(Scan::graph<T, OP, blockSize_, Inclusive>(
            arr, results, blockSums, init, stream));
        cuda::Launcher launcher = graph.launcher();
        launcher.launch();
        launcher.synchronize();
    }

    /** @brief Run a full inclusive scan algorithm */
    template<typename T, typename OP,
        unsigned int blockSize_ = filtering::detail::ScanCore::blockSize>
    static cudaGraph_t inclusiveGraph(
        const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        return Scan::graph<T, OP, true, blockSize_>(
            arr, results, blockSums, init, stream);
    }

    /** @brief Run a full exclusive scan algorithm */
    template<typename T, typename OP,
        unsigned int blockSize_ = filtering::detail::ScanCore::blockSize>
    static cudaGraph_t exclusiveGraph(
        const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        return Scan::graph<T, OP, false, blockSize_>(
            arr, results, blockSums, init, stream);
    }

    /** @brief Instantiate and launch an inclusive scan */
    template<typename T, typename OP,
        unsigned int blockSize_ = filtering::detail::ScanCore::blockSize>
    static void inclusive(const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        Scan::launch<T, OP, true, blockSize_>(
            arr, results, blockSums, init, stream);
    }

    /** @brief Instantiate and launch an exclusive scan*/
    template<typename T, typename OP,
        unsigned int blockSize_ = filtering::detail::ScanCore::blockSize>
    static void exclusive(const CRefArrT<T> arr, GRefArrT<T> results,
        GRefArrT<T> blockSums, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        Scan::launch<T, OP, false, blockSize_>(
            arr, results, blockSums, init, stream);
    }
};

#endif

} // namespace cuda
} // namespace eagle
