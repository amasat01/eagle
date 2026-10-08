// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <omp.h>

#include "eagle/filtering/Scanner.h"
#include "eagle/cuda/CapturedGraph.h"
#include "eagle/util/Slice.h"

namespace eagle {
namespace filtering {

/** @brief Reference to the slice.
 *
 *  There is no separate `work`/`MaybeVolatile` pair — see
 *  `RefScanner`. The ref-level `upload`/`download`/`fetch` forwarders went
 *  with the ones they forwarded to (`util::RefSlice`); a view-to-view async
 *  copy is transport work. */
template<typename SliceT>
class RefSlice : public util::RefSlice<typename SliceT::ParentT> {
    using ParentT = util::RefSlice<typename SliceT::ParentT>;
    using Self    = RefSlice;

public:
    using StreamT = typename ParentT::StreamT;
    /** @brief Expose element access operator */
    using ParentT::operator[];
    /** @brief Expose the scanner reference type */
    using ScannerT = typename SliceT::ScannerT::GRef;
    /** @brief Expose inherently sliced type */
    using T = typename SliceT::T::GRef;

    /** @brief Factory method to construct from parent type and scanner */
    AETHER_DEVICEHOST()
    static RefSlice make(const ParentT& parent, const ScannerT& scanner)
    {
        RefSlice s;
        static_cast<ParentT&>(s) = parent;
        s.scanner_               = scanner;
        return s;
    }

    /** @brief Expose the scanner */
    AETHER_DEVICEHOST() ScannerT& scanner() { return scanner_; }
    AETHER_DEVICEHOST() const ScannerT& scanner() const { return scanner_; }

    /** @brief Clone this reference object */
    AETHER_DEVICEHOST() RefSlice clone() const { return *this; }

protected:
    ScannerT scanner_;
};

/** @brief More flexible slice version that incorporates a scanner */
template<typename MyClass>
class FilteringSlice : public util::Slice<MyClass> {
    /* scanner type narrowed down */
    using Self = FilteringSlice;

public:
    using ParentT = util::Slice<MyClass>;
    using StreamT = typename ParentT::StreamT;
    /** @brief Expose the scanner type */
    using ScannerT = Scanner<idx_t, aether::SumOp<idx_t>, false>;
    /** @brief Expose inherently sliced type */
    using T = MyClass;
    /** @brief Expose the one non-owning reference tier. */
    using GRef = RefSlice<Self>;

    /** @brief Default constructor is forbidden */
    FilteringSlice() = delete;

    /** @brief Construct from the given object pointer */
    FilteringSlice(T* myobj = nullptr)
        : ParentT{ myobj }
        , scanner_{ ParentT::getSize(myobj) }
    {
    }

    /** @brief Construct from the given object */
    FilteringSlice(T& myobj)
        : ParentT{ myobj }
        , scanner_{ myobj.size() }
    {
    }

    /** @brief Move construct from data types */
    FilteringSlice(ParentT&& parent, ScannerT&& scanner)
        : ParentT{ std::move(parent) }
        , scanner_{ std::move(scanner) }
    {
    }

    /** @brief Copy constructor is forbidden */
    FilteringSlice(FilteringSlice& other)       = delete;
    FilteringSlice(const FilteringSlice& other) = delete;

    /** @brief Move constructor */
    FilteringSlice(FilteringSlice&& other)
        : ParentT{ std::move(other) }
        , scanner_{ std::move(other.scanner_) }
    {
    }

    /** @brief Copy assignment is forbidden */
    FilteringSlice& operator=(FilteringSlice& other)       = delete;
    FilteringSlice& operator=(const FilteringSlice& other) = delete;

    /** @brief Move assignment operator */
    FilteringSlice& operator=(FilteringSlice&& other)
    {
        ParentT::operator=(std::move(other));
        scanner_ = std::move(other.scanner_);
        return *this;
    }

    /** @brief Expose the scanner */
    ScannerT& scanner() { return scanner_; }
    const ScannerT& scanner() const { return scanner_; }

    /** @brief Run the scan and scatter on host */
    void hostUpdate()
    {
        ensureExtractedRefs_();
        scanner_.hostRun();
        util::slice::OMPscatter<ParentT, ScannerT>(ref_, scannerRef_);
    }

    /** @brief Return a host reference */
    GRef hostRef() const
    {
        return GRef::make(ParentT::hostRef(), scanner_.hostRef());
    }
    GRef ref() const { return hostRef(); }

#ifndef EAGLE_CPU_ONLY

    /** @brief Return a device Reference */
    GRef deviceRef() const
    {
        return GRef::make(ParentT::deviceRef(), scanner_.deviceRef());
    }

    /** @brief Capture the scan algorithm into a ``CapturedGraph``.
     *
     *  Tags every kernel with ``idealBlockSize = 0`` (grid-only
     *  mutation) — the warp-shuffle reduction layout depends on the
     *  captured ``blockSize`` and would be corrupted by re-tuning. */
    cuda::CapturedGraph scanGraph()
    {
        cudaGraph_t scan = scanner_.graph();
        return cuda::CapturedGraph { scan, countKernelsAsFixedCaps_(scan) };
    }

    /** @brief Capture the scatter kernel into a ``CapturedGraph``
     *  with cap ``EAGLE_BLOCKSIZE`` (element-wise; full re-tune OK). */
    cuda::CapturedGraph scatterGraph()
    {
        const idx_t N       = scanner_.size();
        const idx_t NBlocks = (N + EAGLE_BLOCKSIZE - 1) / EAGLE_BLOCKSIZE;
        cuda::StreamCapturer capturer(scanner_.stream());
        typename ParentT::GRef pref = ParentT::deviceRef();
        const typename ScannerT::GRef sref = scanner_.deviceRef();
        capturer.begin();
        util::slice::CUDAscatter<ParentT, ScannerT>
            <<<NBlocks, EAGLE_BLOCKSIZE, 0, scanner_.stream()>>>(pref, sref);
        return cuda::CapturedGraph { capturer.end(),
            std::vector<idx_t> { EAGLE_BLOCKSIZE } };
    }

    /** @brief Run a full device scan */
    void deviceUpdate(const bool& includeRetrieval = false)
    {
        cuda::Graph g;
        g.addNode(scanGraph());
        g.addNode(scatterGraph());
        /* Add data retrieval to node if required */
        if (includeRetrieval) {
            cuda::StreamCapturer capturer(stream());
            capturer.begin();
            ParentT::download(stream());
            g.addNode(capturer.end());
        }
        cuda::Launcher launcher = g.launcher();
        launcher.launch();
        launcher.synchronize();
    }

    /** @brief Return a const reference to the CUDA stream associated to this
     * slice */
    const StreamT& stream() const { return scanner_.stream(); }

    /** @brief Set the cuda stream */
    void stream(const StreamT& stream) { scanner_.stream(stream); }

    /** @brief Async memcpy from host to device - indexes */
    void upload(const StreamT& stream = 0, const bool& includeSliceData = true)
    {
        ParentT::upload(stream, includeSliceData);
        scanner_.upload(stream);
    }

    /** @brief Async memcpy from device to host */
    void download(
        const StreamT& stream = 0, const bool& includeSliceData = true)
    {
        ParentT::download(stream, includeSliceData);
        scanner_.download(stream);
    }

    /** @brief Mark the device copies stale (see `util::Slice::clearDevice`
     *  for why this no longer frees anything). */
    void clearDevice()
    {
        ParentT::clearDevice();
        scanner_.clearDevice();
    }

#endif

    /** @brief Clone this slice */
    FilteringSlice clone() const
    {
        return FilteringSlice(ParentT::clone(), scanner_.clone());
    }

protected:
#ifndef EAGLE_CPU_ONLY
    /** @brief Count kernel-type nodes inside ``g``; return a vector
     *  of ``cuda::kFixedSize`` of matching size — the scan
     *  tree's recursion-level kernels are layout-locked and must
     *  NOT be re-tuned by ``Launcher::setLogicalSize``: their grid
     *  is sized at capture for small fixed-size internal buffers
     *  and re-deriving gridDim from the launcher's logicalSize
     *  causes per-block sum writes to over-run the recursion-level
     *  ``blockSums`` array (an out-of-range scalar write). */
    static std::vector<idx_t> countKernelsAsFixedCaps_(cudaGraph_t g)
    {
        size_t numNodes = 0;
        EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(g, nullptr, &numNodes));
        if (numNodes == 0)
            return {};
        std::vector<cudaGraphNode_t> nodes(numNodes);
        EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(g, nodes.data(), &numNodes));
        idx_t kernelCount = 0;
        for (cudaGraphNode_t n : nodes) {
            cudaGraphNodeType type;
            EAGLE_CHECK_ALWAYS(cudaGraphNodeGetType(n, &type));
            if (type == cudaGraphNodeTypeKernel)
                ++kernelCount;
        }
        return std::vector<idx_t>(kernelCount, cuda::kFixedSize);
    }
#endif

    /** @brief Ensure that the references are extracted.
     *
     *  Context-aware: nested in a parallel region, exactly one thread
     *  runs the extraction inside an ``omp single`` block. */
    void ensureExtractedRefs_() const
    {
        if (omp_in_parallel()) {
#pragma omp single
            {
                if (!extractedRefs_) {
                    ref_           = ParentT::hostRef();
                    scannerRef_    = scanner_.hostRef();
                    extractedRefs_ = true;
                }
            }
        } else {
            if (!extractedRefs_) {
                ref_           = ParentT::hostRef();
                scannerRef_    = scanner_.hostRef();
                extractedRefs_ = true;
            }
        }
    }

    mutable typename ParentT::GRef ref_;
    mutable typename ScannerT::GRef scannerRef_;
    mutable bool extractedRefs_ = false;
    ScannerT scanner_;
};

} // namespace filtering
} // namespace eagle
