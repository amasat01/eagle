// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/** @brief Templated slice - class to act as a slice of a given reference object
 */
#pragma once

#include <omp.h>

#include "eagle/cpu/Host.h"
#include "eagle/util/Observer.h"

namespace eagle {
namespace util {

/** @brief Reference Slice.
 *
 *  There is no separate `work`/`MaybeVolatile` pair — see
 *  `eagle::filtering::RefScanner` for the reasoning. The index array is now a
 *  HELD `aether::View` rather than a base class: aether's `Array` has no
 *  `Ref<work, MaybeVolatile>` family to inherit from, and a `View` is a value
 *  descriptor, so composition is the direct translation. */
template<typename SliceT>
class RefSlice {
    using Self = RefSlice;

public:
    /** @brief The index view this slice reads its true indices from. */
    using ParentT = GRefArrT<idx_t>;
    /** @brief Expose inherent sliced type — the sliced object's own view. */
    using T = typename SliceT::T::GRef;
    /** @brief Expose the accessed element reference types */
    using ViewT      = typename T::reference;
    using ConstViewT = typename T::reference;
    using StreamT    = nativeStream_t;

    /** @brief Factory method to construct from the index view and the view
     * being sliced */
    AETHER_DEVICEHOST()
    static RefSlice make(const ParentT& idxs, const T& ref, const ParentT& sz)
    {
        RefSlice s;
        s.indexes_   = idxs;
        s.coredata_  = ref;
        s.sliceSize_ = sz;
        return s;
    }

    /** @brief Return the view that points to the indexes */
    AETHER_DEVICEHOST() const ParentT& indexes() const { return indexes_; }

    /** @brief Return the view that points to the indexes (by value — a
     *  `View` copy is a ref-level `clone()`). */
    AETHER_DEVICEHOST() ParentT rawIndexes() const { return indexes_; }

    /** @brief Access the slice elements */
    AETHER_DEVICEHOST() ViewT operator[](const SampleIndex& i)
    {
        return coredata_.eval(truei(i));
    }
    AETHER_DEVICEHOST() decltype(auto) operator[](const SampleIndex& i) const
    {
        return coredata_.eval(truei(i));
    }

    /** @brief Alias the core size as slice offset (for vector elements) */
    AETHER_DEVICEHOST() inline idx_t vectorOffset() const { return coreSize(); }

    /** @brief Expose the size of the inner object */
    AETHER_DEVICEHOST() inline idx_t coreSize() const
    {
        return idx_t(coredata_.samples());
    }

    /** @brief return the size of the slice */
    AETHER_DEVICEHOST() idx_t size() const { return sliceSize_(0); }

    /** @brief Clone this reference object */
    AETHER_DEVICEHOST() RefSlice clone() const { return *this; }

    /** @brief Update the number of elements in this slice */
    AETHER_DEVICEHOST() void updateNumElements(const idx_t newn)
    {
        sliceSize_(0) = newn;
    }

    /* An earlier implementation carried REF-LEVEL
     * `upload`/`download`/`fetch` (a copy issued from one non-owning ref to
     * another). `aether::View` has no transfer surface at all — copies live on
     * `Array` (`upload()`/`download()`) or on `aether::copyAsync` over
     * `Chunk`s, which a View does not expose. The three methods that lived
     * here had no caller outside `filtering::RefSlice`'s identically-shaped
     * forwarders and are dropped rather than reimplemented as raw memcpy;
     * a view-to-view async copy is transport work. */

    /** @brief Data members made public for PODification */
    T coredata_;
    ParentT indexes_;
    ParentT sliceSize_;

protected:
    /** @brief Return the true index */
    AETHER_DEVICEHOST() inline SampleIndex truei(const SampleIndex& i) const
    {
        return SampleIndex::make(indexes_(i.global()));
    }
    AETHER_DEVICEHOST() inline SampleIndex truei(const idx_t& i) const
    {
        return SampleIndex::make(indexes_(i));
    }
};

/** @brief Slice */
template<typename MyClass>
class Slice : public aether::Array<idx_t> {
    using ParentT = aether::Array<idx_t>;
    using CoreT   = Observer<MyClass>;
    using Self    = Slice;

public:
    /** @brief Expose inherently sliced type */
    using T = MyClass;

    /** @brief Expose the one non-owning reference tier. */
    using GRef    = RefSlice<Self>;
    using StreamT = nativeStream_t;

    /* Slice determiner based on pointer value (For default construction) */
    static idx_t getSize(T* ptr)
    {
        if (ptr == nullptr)
            return 0;
        else
            return ptr->size();
    }

    /** @brief Default constructor is forbidden */
    Slice() = delete;

    /** @brief Construct from the given object pointer.
     *
     *  Every index starts at the object size — deliberately out of range, the
     *  "not yet scattered" sentinel `Array{sz, sz}` sets. */
    Slice(T* myobj = nullptr)
        : ParentT{ makeArray<idx_t>(Self::getSize(myobj), Self::getSize(myobj)) }
        , coredata_{ myobj }
        , sliceSize_{ makeArray<idx_t>(1, 0) }
    {
    }

    /** @brief Construct from the given object */
    Slice(T& myobj)
        : ParentT{ makeArray<idx_t>(myobj.size(), myobj.size()) }
        , coredata_{ myobj }
        , sliceSize_{ makeArray<idx_t>(1, 0) }
    {
    }

    /** @brief Construct from the given object and indexes and size */
    Slice(T& myobj, ParentT&& idxs, const idx_t sz)
        : ParentT{ std::move(idxs) }
        , coredata_{ myobj }
        , sliceSize_{ makeArray<idx_t>(1, sz) }
    {
    }

    /** @brief Move construct from data types */
    Slice(ParentT&& idxs, CoreT&& coredata, ParentT&& sz)
        : ParentT{ std::move(idxs) }
        , coredata_{ std::move(coredata) }
        , sliceSize_{ std::move(sz) }
    {
    }

    /** @brief Copy constructor is forbidden */
    Slice(Slice& other)       = delete;
    Slice(const Slice& other) = delete;

    /** @brief Move constructor */
    Slice(Slice&& other)
        : ParentT{ std::move(other) }
        , coredata_{ std::move(other.coredata_) }
        , sliceSize_{ std::move(other.sliceSize_) }
    {
    }

    /** @brief Copy assignment is forbidden */
    Slice& operator=(Slice& other)       = delete;
    Slice& operator=(const Slice& other) = delete;

    /** @brief Move assignment operator */
    Slice& operator=(Slice&& other)
    {
        ParentT::operator=(std::move(other));
        coredata_  = std::move(other.coredata_);
        sliceSize_ = std::move(other.sliceSize_);
        return *this;
    }

    /** @brief Expose the size of the inner object */
    inline idx_t coreSize() const { return coredata_->size(); }

    /** @brief Expose the size of the slice */
    inline idx_t size() const
    {
        return const_cast<Slice&>(*this).sliceSize_.hostView()(0);
    }

    /** @brief Number of index slots this slice owns. */
    inline idx_t numIndexes() const { return idx_t(ParentT::samples()); }

    /** @brief Return a host reference */
    GRef hostRef() const
    {
        Slice& self = const_cast<Slice&>(*this);
        return GRef::make(self.ParentT::hostView(), coredata_->hostRef(),
            self.sliceSize_.hostView());
    }
    GRef ref() const { return hostRef(); }

#ifndef EAGLE_CPU_ONLY

    /** @brief Return a device Reference */
    GRef deviceRef() const
    {
        Slice& self = const_cast<Slice&>(*this);
        return GRef::make(self.ParentT::deviceView(), coredata_->deviceRef(),
            self.sliceSize_.deviceView());
    }

    /** @brief Async memcpy from host to device - indexes */
    void upload(const StreamT& stream = 0, const bool& includeSliceData = true)
    {
        ParentT::upload(stream);
        sliceSize_.upload(stream);
        if (includeSliceData)
            coredata_->upload(stream);
    }

    /** @brief Async memcpy from device to host - Indexes */
    void download(
        const StreamT& stream = 0, const bool& includeSliceData = true)
    {
        ParentT::download(stream);
        sliceSize_.download(stream);
        if (includeSliceData)
            coredata_->download(stream);
    }

    /** @brief Focused async D2H of just the slice-size scalar.
     *
     *  Used by callers that need ``size()`` cheaply on host without
     *  pulling the full index array. After issuing this download and
     *  syncing the stream, ``size()`` returns the up-to-date active
     *  count. */
    void downloadSize(const StreamT& stream = 0)
    {
        sliceSize_.download(stream);
    }

    /** @brief Mark the device copy stale.
     *
     *  DEVIATION (see the port report): `aether::Array` owns its device chunk
     *  for its whole lifetime and exposes no release, so this no longer frees
     *  device memory — it is a no-op kept for source compatibility. */
    void clearDevice() { }

#endif

    /** @brief Clone this slice */
    Slice clone() const
    {
        return Slice(cloneArray(static_cast<const ParentT&>(*this)),
            std::move(coredata_.clone()), cloneArray(sliceSize_));
    }

    /** @brief update the core data */
    void updateCoreData(T& obj) { coredata_ = std::move(CoreT(obj)); }
    void updateCoreData(T* obj) { coredata_ = std::move(CoreT(obj)); }

    /** @brief Expose the core data */
    const CoreT& core() const { return coredata_; }

protected:
    CoreT coredata_;
    ParentT sliceSize_;
};

/** @brief Slice edit kernel */
namespace slice {

#ifndef EAGLE_CPU_ONLY

/** @brief Edit the slice from the given scanner type - CUDA kernel.
 *
 * SASS register footprint: ~11 regs (sm_61). ``__launch_bounds__(EAGLE_BLOCKSIZE,
 * 4)`` documents the cap for ``Launcher::setLogicalSize`` re-tuning. */
template<typename SliceT, typename ScannerT>
AETHER_KERNEL()
__launch_bounds__(EAGLE_BLOCKSIZE, 4) void CUDAscatter(
    typename SliceT::GRef slice, const typename ScannerT::GRef scanner)
{
    const SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() >= scanner.inputs().samples())
        return;

    /* Edit the size */
    if (i.global() == 0) {
        slice.updateNumElements(scanner.numElements());
    }

    /* Edit the slice indexes */
    if (scanner.inputs().eval(i)) {
        if constexpr (ScannerT::Inclusive)
            slice.rawIndexes()(scanner.results().eval(i) - 1) = i.global();
        else
            slice.rawIndexes()(scanner.results().eval(i)) = i.global();
    }
}

#endif

/** @brief Edit the slice from the given scanner type - OpenMP.
 *
 *  Context-aware via ``aether::packetFor``: inside an existing
 *  parallel region the per-sample loop is ``omp for`` work-shared,
 *  otherwise the primitive opens its own parallel region.  The body
 *  is dispatched scalar-per-lane today via ``dispatchMasked``; the
 *  packet shape is preserved so a future SIMD port can replace the
 *  scalar dispatch with packet-shaped writes without touching the
 *  outer loop structure. */
template<typename SliceT, typename ScannerT>
inline void OMPscatter(
    typename SliceT::GRef slice, const typename ScannerT::GRef scanner)
{
    aether::packetFor<Real>(
        std::size_t{ 0 }, std::size_t(scanner.inputs().samples()),
        [&](const auto& pi) {
            constexpr std::size_t W = std::remove_cvref_t<decltype(pi)>::width;
            auto active             = eagle::cpu::Host::applyTail<W>(
                aether::simd::PacketMask<Real, W>::allTrue(), pi);
            eagle::cpu::Host::dispatchMasked(
                pi, active, [&](const SampleIndex& i) {
                    if (i.global() == 0) {
                        slice.updateNumElements(scanner.numElements());
                    }
                    if (scanner.inputs().eval(i)) {
                        if constexpr (ScannerT::Inclusive)
                            slice.rawIndexes()(scanner.results().eval(i) - 1)
                                = i.global();
                        else
                            slice.rawIndexes()(scanner.results().eval(i))
                                = i.global();
                    }
                });
        },
        /*parallel=*/true);
}


} // namespace slice
} // namespace util
} // namespace eagle
