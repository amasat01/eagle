// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/filtering/Scanner.h"
#include "eagle/cuda/Scan.h"
#include "eagle/cpu/Scan.h"
#include "eagle/cuda.h"
#include "eagle/cpu/Graph.h"
#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/Slice.h"

#include <cstddef>
#include <vector>

namespace eagle {
namespace filtering {

/**
 * @brief Stream compaction (predicate -> scan -> scatter) as a native node —
 *        the arena-backed twin of ``FilteringSlice::deviceUpdate`` /
 *        ``hostUpdate``, with both a CUDA and a host face.
 *
 * Builds the index map of the samples a predicate keeps, sourcing the scanner's
 * IN/OUT/BLOCKSUMS (3N) working buffer from the graph's ``ScratchArena``. That
 * whole 3N is INTERNAL to the node — IN is a copy of the predicate, OUT is
 * consumed by the scatter, BLOCKSUMS is scan-internal — so a chain of interleaved
 * compaction nodes shares one 3N slot (peak, not sum). The sliced object's index
 * map + active count (the compaction OUTPUT) are caller-owned payload, never
 * arena scratch. The CUDA face (``buildInto``) captures predicate-copy ->
 * exclusive scan (kFixedSize) -> ``CUDAscatter``; the host face (``runHost``)
 * runs the OpenMP twins over the same 3N slot: copy -> ``cpu::Scan`` ->
 * ``OMPscatter`` (the exact ``FilteringSlice::hostUpdate`` path).
 *
 * @tparam MyClass the sliced object type (as ``FilteringSlice<MyClass>``).
 */
template<typename MyClass>
class FilteringSliceNode {
public:
    using SliceT    = util::Slice<MyClass>;
    using ScannerT  = Scanner<idx_t, aether::SumOp<idx_t>, false>;
    using SliceGRef = typename SliceT::GRef;
    /** @brief Mode-agnostic stream type (``cudaStream_t`` under CUDA,
     *  ``int`` in ``EAGLE_CPU_ONLY``). */
    using StreamT = nativeStream_t;

    /**
     * @param slice     View of the sliced object (``slice.deviceRef()`` /
     *                  ``slice.hostRef()``).
     * @param predicate Pointer to the N-element 0/1 keep-mask (device memory for
     *                  the CUDA face, host memory for the host face).
     * @param n         Sample count.
     * @param stream    Capture stream (CUDA face).
     */
    FilteringSliceNode(const SliceGRef& slice, const idx_t* predicate, idx_t n,
        const StreamT& stream = 0)
        : slice_(slice)
        , predicate_(predicate)
        , n_(n)
        , stream_(stream)
    {
    }

    /** @brief Phase 1: reserve the scanner's contiguous IN+OUT+BLOCKSUMS (3N). */
    void reserveScratch(ScratchArena& arena)
    {
        scratch_
            = arena.reserve(std::size_t(HANDLESIZE) * n_ * sizeof(idx_t));
    }

#ifndef EAGLE_CPU_ONLY

    /** @brief Phase 2 (CUDA): capture predicate-copy -> scan -> scatter over the
     *  arena 3N buffer, wiring @p deps to the first node. Returns the scatter
     *  node. */
    idx_t buildInto(cuda::Graph& g, const std::vector<idx_t>& deps)
    {
        idx_t* base = static_cast<idx_t*>(g.scratchArena().resolve(scratch_));
        // Contiguous SoA layout (component pitch = N): IN [base, +N),
        // OUT [+N, +2N), BLOCKSUMS [+2N, +3N). `packedSpanView` pitches at
        // exactly N — an owning `aether::Array` would pitch at its QUANTISED
        // capacity() instead, which is why the scanner view is built by hand
        // here rather than from an Array.
        auto vref = packedSpanView<HANDLESIZE, idx_t>(base, n_, deviceDevice());
        auto in   = vref.template component<IN>();
        auto out  = vref.template component<OUT>();
        auto bs   = vref.template component<BLOCKSUMS>();
        typename ScannerT::GRef sref = ScannerT::GRef::make(vref);

        g.stream(stream_);
        cuda::StreamCapturer capturer(stream_);

        // Node 1: copy the predicate into the scanner's IN component.
        // (ScratchArena site): `base` is a raw slot resolved out of
        // eagle's own `ScratchArena` and `predicate_` a caller-supplied raw
        // device pointer — neither end is an aether array, so the no-raw-memcpy
        // rule does not reach here. LISTED.
        capturer.begin();
        EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(base, predicate_,
            std::size_t(n_) * sizeof(idx_t), cudaMemcpyDeviceToDevice, stream_));
        cudaGraph_t pg = capturer.end();
        g.addNode(pg, deps);
        EAGLE_CHECK_ALWAYS(cudaGraphDestroy(pg));

        // Node 2: exclusive prefix scan (layout-locked -> uniform kFixedSize).
        cudaGraph_t sg = cuda::Scan::graph<idx_t, aether::SumOp<idx_t>, false>(
            in.as_const(), out, bs, 0, stream_);
        g.addNode(sg, {}, cuda::kFixedSize);
        EAGLE_CHECK_ALWAYS(cudaGraphDestroy(sg));

        // Node 3: scatter kept indices + write the active count (re-tunable).
        const idx_t nblocks = (n_ + EAGLE_BLOCKSIZE - 1) / EAGLE_BLOCKSIZE;
        capturer.begin();
        util::slice::CUDAscatter<SliceT, ScannerT>
            <<<nblocks, EAGLE_BLOCKSIZE, 0, stream_>>>(slice_, sref);
        cudaGraph_t cg = capturer.end();
        g.addNode(cg, {}, EAGLE_BLOCKSIZE);
        EAGLE_CHECK_ALWAYS(cudaGraphDestroy(cg));

        return g.lastNode();
    }

#endif  // EAGLE_CPU_ONLY

    /** @brief Host face: run the OpenMP compaction twins over the committed
     *  arena's 3N slot (host pointers), reconstructing the same SoA scanner view
     *  the CUDA face builds — copy predicate -> exclusive ``cpu::Scan`` ->
     *  ``OMPscatter`` (the ``FilteringSlice::hostUpdate`` path). Each stage is
     *  outer-OpenMP over sample tiles, SIMD-shaped within. */
    void runHost(cpu::Graph& g)
    {
        idx_t* base = static_cast<idx_t*>(g.scratchArena().resolve(scratch_));
        // Same contiguous SoA layout as buildInto: IN [base,+N), OUT [+N,+2N),
        // BLOCKSUMS [+2N,+3N).
        auto vref = packedSpanView<HANDLESIZE, idx_t>(base, n_, hostDevice());
        auto in   = vref.template component<IN>();
        auto out  = vref.template component<OUT>();
        auto bs   = vref.template component<BLOCKSUMS>();
        typename ScannerT::GRef sref = ScannerT::GRef::make(vref);

        // Stage 1: copy the predicate into the scanner's IN component (host).
        for (idx_t i = 0; i < n_; ++i)
            base[i] = predicate_[i];

        // Stage 2: exclusive prefix scan (OpenMP twin of cuda::Scan).
        cpu::Scan::scan<idx_t, aether::SumOp<idx_t>, false>(
            in.as_const(), out, bs);

        // Stage 3: scatter kept indices + write the active count (OpenMP twin
        // of CUDAscatter).
        util::slice::OMPscatter<SliceT, ScannerT>(slice_, sref);
    }

private:
    SliceGRef slice_;
    const idx_t* predicate_;
    idx_t n_;
    StreamT stream_;
    ScratchHandle scratch_ {};
};

}  // namespace filtering
}  // namespace eagle
