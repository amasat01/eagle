// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cuda/Scan.h"
#include "eagle/cpu/Scan.h"

namespace eagle {
namespace filtering {

/** @brief Handle elements */
enum Handle {
    IN,
    OUT,
    BLOCKSUMS,
    /* Leave the following item as last - it only acts as enum size */
    HANDLESIZE
};

/** @brief Reference scanner to simplify the access to the scan products.
 *
 *  There is no separate `work`/`MaybeVolatile` pair: `aether::View`
 *  (`aether/view/View.h`) is already the lightweight descriptor, so which
 *  storage it points at is just which `Chunk` the `View` was built over.
 *  Shared-memory (work) references are `aether::make_work_view` at the
 *  point of use, not a flavour of this class, so eagle carries no
 *  `WRef`/`VolatileRef` alias. */
template<typename ScannerT>
class RefScanner {
    using DataT = typename ScannerT::DataT;

public:
    /** @brief The packed (component, sample) view this reference wraps. */
    using ParentT = typename aether::Array<DataT, HANDLESIZE>::ViewT;
    /** @brief A single scan component, as a scalar view. */
    using componentT = GRefArrT<DataT>;

    static constexpr bool Inclusive = ScannerT::Inclusive;

    /** @brief Factory method to construct from the packed view */
    AETHER_DEVICEHOST()
    static RefScanner make(const ParentT& parent)
    {
        RefScanner s;
        s.data_ = parent;
        return s;
    }

    /** @brief The packed view underlying every component accessor. */
    AETHER_DEVICEHOST() const ParentT& packed() const { return data_; }

    /* `aether::View::component<I>` (aether/view/View.h) is first-class, so
     * eagle needs no `componentView<I>` workaround. The local COPY is
     * deliberate: `component<I>` has
     * a const overload returning a `View<const T, ...>`, and these accessors are
     * const members handing back a WRITABLE component (the scan writes through
     * `results()`/`blockSums()`), which is what the by-value `View` — a
     * non-owning descriptor — costs nothing to take. */

    /** @brief Return a component view used to manipulate the scan inputs */
    AETHER_DEVICEHOST() componentT inputs() const
    {
        ParentT v = data_;
        return v.template component<IN>();
    }

    /** @brief Return a component view exposing the scan results */
    AETHER_DEVICEHOST() componentT results() const
    {
        ParentT v = data_;
        return v.template component<OUT>();
    }

    /** @brief Return the per-block partial sums component view */
    AETHER_DEVICEHOST() componentT blockSums() const
    {
        ParentT v = data_;
        return v.template component<BLOCKSUMS>();
    }

    /** @brief Number of samples in each component. */
    AETHER_DEVICEHOST() idx_t size() const { return idx_t(data_.samples()); }

    /** @brief Return the number of independent elements in the scan results.
     *
     *  aether's `View` has no `back()`; the last sample is `v(v.samples()-1)`. */
    AETHER_DEVICEHOST() idx_t numElements() const
    {
        const componentT res = results();
        const idx_t last     = idx_t(res.samples()) - 1;
        if constexpr (Inclusive) {
            return res(last);
        } else {
            return res(last) + inputs()(last);
        }
    }

    /** @brief Data member made public for PODification */
    ParentT data_;
};

/**
 * @brief Parallel prefix-scan (prefix-sum) owning container.
 *
 * Computes an inclusive or exclusive prefix scan over the ``inputs()``
 * component and stores the result in ``results()``.  Supports both
 * CPU (``hostRun()``) and GPU (``deviceRun()``) execution.  A CUDA
 * graph node is available via ``graph()`` for integration into larger
 * CUDA graphs.
 *
 * @tparam DataT_     Element type of the scan (e.g. ``int``).
 * @tparam OP         Scan operator; typically ``aether::SumOp<DataT_>``.
 * @tparam Inclusive_ ``true`` for an inclusive scan; ``false`` for exclusive.
 */
template<typename DataT_, typename OP, bool Inclusive_>
class Scanner : public aether::Array<DataT_, HANDLESIZE> {
    using ParentT = aether::Array<DataT_, HANDLESIZE>;
    using Self    = Scanner;

public:
    static constexpr bool Inclusive = Inclusive_;
    using DataT                     = DataT_;
    /** @brief Expose the reference type (one tier; see `RefScanner`). */
    using GRef = RefScanner<Self>;

    /** @brief Expose the stream type */
    using StreamT = nativeStream_t;

    /** @brief Construct a scanner over @p n samples per component.
     *
     *  Zero-fills: `aether::Array(n)` deliberately leaves storage
     *  uninitialised, so this constructor zero-fills explicitly through
     *  `eagle::makeArray`. */
    explicit Scanner(const idx_t& n)
        : ParentT{ makeArray<DataT, HANDLESIZE>(n, DataT(0)) }
    {
    }

    /** @brief Move constructor is allowed */
    Scanner(Scanner&& other)
        : ParentT{ std::move(other) }
        , stream_{ other.stream_ }
        , deviceReady_{ std::exchange(other.deviceReady_, false) }
    {
    }

    /** @brief Move assignment is allowed */
    Scanner& operator=(Scanner&& other)
    {
        ParentT::operator=(std::move(other));
        stream_      = other.stream_;
        deviceReady_ = std::exchange(other.deviceReady_, false);
        return *this;
    }

    /** @brief Deep copy — see `eagle::cloneArray`. */
    Scanner clone() const
    {
        Scanner out(idx_t(ParentT::samples()));
        static_cast<ParentT&>(out) = cloneArray(static_cast<const ParentT&>(*this));
        out.stream_ = stream_;
        return out;
    }

    /** @brief Samples per component — NOT `aether::Array::size()`, which is
     *  the TOTAL element count (`HANDLESIZE * samples()`). */
    idx_t size() const { return idx_t(ParentT::samples()); }

    /** @brief Run the scan on host */
    void hostRun()
    {
        const GRef ref = this->hostRef();
        cpu::Scan::scan<DataT, OP, Inclusive>(ref.inputs().as_const(),
            ref.results(), ref.blockSums());
    }

#ifndef EAGLE_CPU_ONLY

    /** @brief Create a cuda graph that can be used to launch the execution of
     * the scan algorithm */
    cudaGraph_t graph()
    {
        ensureInitDevice_();
        const GRef ref = this->deviceRef();
        return cuda::Scan::graph<DataT, OP, Inclusive>(
            ref.inputs().as_const(), ref.results(), ref.blockSums(), 0,
            stream_);
    }

    /** @brief Run a device scan */
    void deviceRun(const bool& includeRetrieval = false)
    {
        cuda::Graph g;
        g.addNode(graph());
        /* Add data retrieval to node if required */
        if (includeRetrieval) {
            cuda::StreamCapturer capturer(stream_);
            capturer.begin();
            retrieveResults();
            g.addNode(capturer.end());
        }
        cuda::Launcher launcher = g.launcher();
        launcher.launch();
        launcher.synchronize();
    }

    /** @brief Return a const reference to the CUDA stream associated to this
     * scanner */
    const StreamT& stream() const { return stream_; }

    /** @brief Set the cuda stream */
    void stream(const StreamT& stream) { stream_ = stream; }

    /** @brief Retrieve the results (transfer only the part relative to the
     *  outputs).
     *
     *  This copy touches an aether array on BOTH ends, so it routes
     *  through `aether::copyAsync(View, View, Stream)` (aether/view/Copy.h)
     *  rather than a raw `cudaMemcpyAsync` — aether owns residency and the
     *  legal device-pair matrix (CUDA -> CUDAHost is async-capable), eagle
     *  owns only WHEN the move happens.
     *
     *  `component<OUT>` is what makes the site expressible at all: it
     *  offsets by the SOURCE mapping's own component pitch — `capacity()`,
     *  which aether quantises, never `samples()` — and extends over exactly
     *  `samples()` elements, so the hand-computed `data() + OUT * pitch` and
     *  `sz * sizeof(DataT)` of the raw form are both read off the view
     *  instead of re-derived. A host/device pitch disagreement now throws
     *  (extents/strides mismatch) instead of transferring the wrong bytes. */
    void retrieveResults()
    {
        auto hostV   = ParentT::hostView();
        auto deviceV = ParentT::deviceView();
        aether::copyAsync(hostV.template component<OUT>(),
            deviceV.template component<OUT>().as_const(), stream_);
    }

#endif

    /** @brief Return a host reference */
    GRef hostRef() const
    {
        return GRef::make(const_cast<Scanner&>(*this).ParentT::hostView());
    }
    GRef ref() const { return hostRef(); }

#ifndef EAGLE_CPU_ONLY
    /** @brief Return a device reference */
    GRef deviceRef() const
    {
        return GRef::make(const_cast<Scanner&>(*this).ParentT::deviceView());
    }

    /** @brief Upload the data marking device ready */
    void upload(const StreamT& stream)
    {
        ParentT::upload(stream);
        deviceReady_ = true;
    }

    /** @brief Mark the device copy stale.
     *
     *  Unlike an implementation that releases the device allocation here,
     *  `aether::Array` owns both chunks for its whole
     *  lifetime and exposes no way to drop one, so this now only invalidates
     *  the ready flag — the next `ensureInitDevice_()` re-uploads exactly as
     *  before, but the device memory is not returned in between. */
    void clearDevice() { deviceReady_ = false; }

#endif
protected:
#ifndef EAGLE_CPU_ONLY
    /** @brief Ensure that the device copy is current */
    void ensureInitDevice_()
    {
        if (!deviceReady_) {
            this->upload(stream_);
            deviceReady_ = true;
        }
    }
#endif

    StreamT stream_   = 0;
    bool deviceReady_ = false;
};

} // namespace filtering
} // namespace eagle
