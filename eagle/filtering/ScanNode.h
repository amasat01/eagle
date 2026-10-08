// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cuda/Scan.h"
#include "eagle/cpu/Scan.h"
#include "eagle/cuda.h"
#include "eagle/cpu/Graph.h"
#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"

#include <cstddef>
#include <vector>

namespace eagle {
namespace filtering {

/**
 * @brief A parallel prefix scan as a native node — the arena-backed twin of
 *        ``cuda::Scan::graph``, with both a CUDA and a host face.
 *
 * Scans ``input`` into ``output`` (inclusive or exclusive prefix), sourcing only
 * its BLOCKSUMS working buffer from the graph's ``ScratchArena``. ``input`` and
 * ``output`` are caller-owned views (GRefs) — the scan's I/O, not scratch — so a
 * chain of scans shares a single BLOCKSUMS slot (peak, not sum). The CUDA face
 * (``buildInto``) captures the ``cuda::Scan`` tree; the host face (``runHost``)
 * runs the OpenMP twin ``cpu::Scan::scan`` over the same BLOCKSUMS slot (host
 * pointer). Both share ``reserveScratch``.
 *
 * Contributed transparently with ``graph.addNative(ScanNode{...}, deps)`` (CUDA)
 * or ``hostGraph.addNative(ScanNode{...}, deps)`` (host). Holds only a pair of
 * GRefs + a size + stream + the scratch handle, so it is copyable — as
 * ``addNative`` requires.
 *
 * @tparam DataT     element type. @tparam OP scan operator (as ``cuda::Scan`` /
 *                   ``cpu::Scan``).
 * @tparam Inclusive ``true`` inclusive, ``false`` exclusive.
 */
template<typename DataT, typename OP, bool Inclusive>
class ScanNode {
public:
    using GRefT  = GRefArrT<DataT>;
    using CRefT  = CRefArrT<DataT>;
    /** @brief Mode-agnostic stream type (``cudaStream_t`` under CUDA,
     *  ``int`` in ``EAGLE_CPU_ONLY``). */
    using StreamT = nativeStream_t;

    /**
     * @param input  Read-only view of the array to scan (e.g.
     *               ``arr.deviceView().as_const()``).
     * @param output View the prefix result is written to (size == input).
     * @param stream Stream the scan tree is captured on (CUDA face).
     */
    ScanNode(const CRefT& input, const GRefT& output,
        const StreamT& stream = 0)
        : input_(input)
        , output_(output)
        , n_(idx_t(input.samples()))
        , stream_(stream)
    {
    }

    /** @brief Phase 1: reserve the BLOCKSUMS working buffer (N elements, as the
     *  Scanner's BLOCKSUMS component). */
    void reserveScratch(ScratchArena& arena)
    {
        scratch_ = arena.reserve(std::size_t(n_) * sizeof(DataT));
    }

#ifndef EAGLE_CPU_ONLY

    /** @brief Phase 2 (CUDA): capture the scan tree over ``input``/``output``
     *  with the committed arena's BLOCKSUMS, wiring @p deps. The scan tree is
     *  layout-locked (re-deriving the grid would overrun the recursion-level
     *  blockSums), so the child is added with a uniform ``kFixedSize`` cap.
     *  Returns the added node. */
    idx_t buildInto(cuda::Graph& g, const std::vector<idx_t>& deps)
    {
        DataT* bs = static_cast<DataT*>(g.scratchArena().resolve(scratch_));
        GRefT blockSums = spanView<DataT>(bs, n_, deviceDevice());

        g.stream(stream_);
        // cuda::Scan::graph returns an OWNING raw cudaGraph_t; addNode clones it
        // (borrowed) with the uniform layout-lock cap, then we release the source.
        cudaGraph_t sg = cuda::Scan::graph<DataT, OP, Inclusive>(
            input_, output_, blockSums, DataT(0), stream_);
        g.addNode(sg, deps, cuda::kFixedSize);
        EAGLE_CHECK_ALWAYS(cudaGraphDestroy(sg));

        return g.lastNode();
    }

#endif  // EAGLE_CPU_ONLY

    /** @brief Host face: run the OpenMP prefix-scan twin over ``input_`` into
     *  ``output_``, using the committed arena's BLOCKSUMS slot (host pointer)
     *  as the scan's per-block workspace — the exact ``Scanner::hostRun`` path.
     *  Outer OpenMP over sample tiles, SIMD within. */
    void runHost(cpu::Graph& g)
    {
        DataT* bs = static_cast<DataT*>(g.scratchArena().resolve(scratch_));
        GRefT blockSums = spanView<DataT>(bs, n_, hostDevice());
        cpu::Scan::scan<DataT, OP, Inclusive>(input_, output_, blockSums);
    }

private:
    CRefT input_;
    GRefT output_;
    idx_t n_;
    StreamT stream_;
    ScratchHandle scratch_ {};
};

}  // namespace filtering
}  // namespace eagle
