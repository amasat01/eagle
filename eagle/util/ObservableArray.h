// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/util/Observable.h"

namespace eagle {
namespace util {
namespace observable {

/** @brief aether scalar array wrapped with observer-pattern notifications on
 *         mutation.
 *
 *  The public base is `aether::Array<DataT>`.
 *  aether merges what used to be a `scalar::Array`/`vector::Array` two-tier into ONE
 *  `Array<T, Es...>`, and its non-owning face is `aether::View` (`hostView()`/
 *  `deviceView()`), not a `Ref<work, MaybeVolatile>` family — so this class
 *  exposes `hostRef()`/`deviceRef()` returning `GRefArrT<DataT>` and drops
 *  `WRef`/`VolatileRef` entirely (neither was ever instantiated in-tree). */
template<typename DataT>
class Array : public Observable<Array<DataT>>, public aether::Array<DataT> {
    using ParentT = aether::Array<DataT>;

public:
    using ObserverT = typename Observable<Array<DataT>>::ObserverT;
    /** @brief The one non-owning reference tier (aether's `View`). */
    using GRef    = GRefArrT<DataT>;
    using CRef    = CRefArrT<DataT>;
    using StreamT = nativeStream_t;

    /** @brief Default construction is forbidden */
    Array() = delete;

    /** @brief Construct from size and initial value.
     *
     *  aether's `Array(n)` leaves the storage UNINITIALISED;
     *  `eagle::makeArray` restores a zero fill explicitly. */
    Array(const idx_t& sz, const DataT& initVal = 0)
        : ParentT{ makeArray<DataT>(sz, initVal) }
    {
    }

    /** @brief Copy construction is forbidden */
    Array(Array& other)       = delete;
    Array(const Array& other) = delete;

    /** @brief Move construction is allowed */
    Array(ParentT&& other)
        : ParentT{ std::move(other) }
    {
    }
    Array(Array&& other)
        : ParentT{ std::move(other) }
    {
        this->finalizeMove(std::move(other));
    }

    /** @brief Copy assignment is forbidden */
    Array& operator=(Array& other)       = delete;
    Array& operator=(const Array& other) = delete;

    /** @brief Move assignment is allowed */
    Array& operator=(ParentT&& other)
    {
        ParentT::operator=(std::move(other));
        return *this;
    }
    Array& operator=(Array&& other)
    {
        ParentT::operator=(std::move(other));
        this->finalizeMove(std::move(other));
        return *this;
    }

    /** @brief Number of samples (not the total element count). For a scalar
     *  array `aether::Array::size()` agrees, but the name is pinned here so a
     *  later vector-valued sibling cannot silently change it. */
    idx_t size() const { return idx_t(ParentT::samples()); }

    /** @brief Non-owning view over the host copy. */
    GRef hostRef() const
    {
        return const_cast<Array&>(*this).ParentT::hostView();
    }
    GRef ref() const { return hostRef(); }

#ifndef EAGLE_CPU_ONLY
    /** @brief Non-owning view over the device copy. */
    GRef deviceRef() const
    {
        return const_cast<Array&>(*this).ParentT::deviceView();
    }
#else
    GRef deviceRef() const { return hostRef(); }
#endif

    /** @brief Create a clone of this observable array */
    Array clone() const
    {
        return Array(cloneArray(static_cast<const ParentT&>(*this)),
            this->observers());
    }

protected:
    /** @brief Protected constructor to clone */
    Array(ParentT&& other, const std::vector<ObserverT*>& obs)
        : ParentT{ std::move(other) }
    {
        this->observers() = obs;
    }
};

} // namespace observable
} // namespace util
} // namespace eagle
