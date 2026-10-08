// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cuda.h"
#include "eagle/cpu/Graph.h"
#include "eagle/reduce/detail.h"
#include "eagle/cuda/Reduction.h"
#include "eagle/cpu/Reduction.h"
#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"

#include <cstddef>
#include <vector>

namespace eagle {
namespace reduce {

/**
 * @brief A reduction as a native node — the arena-backed twin of
 *        ``cuda::Reduction::reduceBlocking``, with both a CUDA and a host face.
 *
 * Reduces the input array into a single result. The CUDA face (``buildInto``,
 * ``EAGLE_CPU_ONLY`` undefined) captures the same multi-level
 * ``detail::reduceOnce`` chain the free builder uses; the host face
 * (``runHost``, always compiled) runs the OpenMP twin ``cpu::Reduction``. Both
 * share ``reserveScratch``.
 *
 * Contributed transparently with ``graph.addNative(ReductionNode{...}, deps)``
 * (CUDA ``Graph``) or ``hostGraph.addNative(ReductionNode{...}, deps)``
 * (``cpu::Graph``). The node holds only PODs (result pointer, input handle,
 * size, init, stream, scratch handle), so it is copyable — as ``addNative``
 * requires.
 *
 * @tparam T  element type. @tparam OP  reduction operator (as ``reduceOnce`` /
 *            ``cpu::Reduction``).
 */
template<typename T, typename OP>
class ReductionNode {
public:
    /** @brief The one non-owning reference tier (aether's `View`). */
    using HandleT = CRefArrT<T>;
    /** @brief Mode-agnostic stream type (``cudaStream_t`` under CUDA,
     *  ``int`` in ``EAGLE_CPU_ONLY``). */
    using StreamT = nativeStream_t;

    /**
     * @param result Host pointer the single reduced value is copied into.
     * @param input  Read-only device view over the array to reduce (e.g.
     *               ``arr.deviceRef().as_const()``); must outlive the launch.
     * @param n      Element count of @p input.
     * @param init   Initial accumulator value.
     * @param stream Stream the reduce chain is captured on.
     */
    ReductionNode(T* result, const HandleT& input, idx_t n, T init = 0,
        const StreamT& stream = 0)
        : result_(result)
        , input_(input)
        , n_(n)
        , init_(init)
        , stream_(stream)
    {
    }

    /** @brief Phase 1: reserve the reduction work buffer (sized like
     *  ``cuda::Reduction::makeBuffer`` — at least ``nBlocks`` elements). */
    void reserveScratch(ScratchArena& arena)
    {
        const idx_t nBlocks  = detail::Core::countBlocks<>(n_);
        const idx_t bufElems = (nBlocks > 1) ? nBlocks : n_;
        scratch_ = arena.reserve(std::size_t(bufElems) * sizeof(T));
    }

#ifndef EAGLE_CPU_ONLY

    /** @brief Phase 2 (CUDA): capture the multi-level reduce chain (first node
     *  depends on @p deps, the rest auto-chain) plus the D2H result copy, using
     *  the committed arena's pointer as the work buffer. Returns the last node. */
    idx_t buildInto(cuda::Graph& g, const std::vector<idx_t>& deps)
    {
        T* work = static_cast<T*>(g.scratchArena().resolve(scratch_));
        const idx_t nBlocks0  = detail::Core::countBlocks<>(n_);
        const idx_t bufElems  = (nBlocks0 > 1) ? nBlocks0 : n_;
        GRefArrT<T> ohandle = spanView<T>(work, bufElems, deviceDevice());

        constexpr unsigned int blockSize = detail::Core::blockSize;
        idx_t prevNBlocks = n_;
        idx_t nBlocks     = detail::Core::countBlocks<>(n_);

        g.stream(stream_);
        cuda::StreamCapturer capturer(stream_);

        bool first     = true;  // switch the read source to the work buffer
        bool firstNode = true;  // only the first captured node takes `deps`
        HandleT from   = input_;

        // Add the captured level as a child node, then destroy the source graph
        // (addNode clones it — the borrowed-input contract, so the raw handle from
        // end() is ours to release). reduceOnce is layout-locked (warp-shuffle
        // block size), so cap it kFixedSize: a later setLogicalSize must never
        // re-tune its grid.
        auto append = [&] {
            cudaGraph_t cg = capturer.end();
            if (firstNode) {
                g.addNode(cg, deps, cuda::kFixedSize);
                firstNode = false;
            } else {
                g.addNode(cg, {}, cuda::kFixedSize);  // auto-chain to previous
            }
            EAGLE_CHECK_ALWAYS(cudaGraphDestroy(cg));
        };

        while (nBlocks > 1) {
            capturer.begin();
            detail::reduceOnce<T, OP, blockSize>
                <<<nBlocks, blockSize, 0, stream_>>>(
                    from, ohandle, (unsigned int)prevNBlocks, init_);
            append();
            if (first) {
                from  = ohandle.as_const();
                first = false;
            }
            prevNBlocks = nBlocks;
            nBlocks     = detail::Core::countBlocks<>(nBlocks);
        }

        capturer.begin();
        detail::reduceOnce<T, OP, blockSize><<<1, blockSize, 0, stream_>>>(
            from, ohandle, (unsigned int)prevNBlocks, init_);
        append();

        /* (ScratchArena site): `work` is a raw slot resolved out
         * of eagle's own `ScratchArena`, never an aether array, and `result_`
         * is a caller-supplied raw host pointer. Both ends are outside
         * aether's residency model, so the rule ("eagle never issues raw
         * cudaMemcpy* on AETHER data") does not reach here. LISTED. */
        capturer.begin();
        EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(
            result_, work, sizeof(T), cudaMemcpyDeviceToHost, stream_));
        cudaGraph_t mg = capturer.end();
        g.addNode(mg);  // memcpy node (auto-chain); not a kernel, so no cap
        EAGLE_CHECK_ALWAYS(cudaGraphDestroy(mg));

        return g.lastNode();
    }

#endif  // EAGLE_CPU_ONLY

    /** @brief Host face: run the OpenMP reduction twin over ``input_`` and store
     *  the result. ``cpu::Reduction`` owns its own per-thread padded accumulators
     *  (outer thread loop over samples), so no arena scratch is touched — the
     *  reserved slot simply stays unused on the host path. */
    void runHost(cpu::Graph& /*g*/)
    {
        *result_ = cpu::Reduction<T, OP>::reduce(input_, init_);
    }

private:
    T* result_;
    HandleT input_;
    idx_t n_;
    T init_;
    StreamT stream_;
    ScratchHandle scratch_ {};
};

}  // namespace reduce
}  // namespace eagle
