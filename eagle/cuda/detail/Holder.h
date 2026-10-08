// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cuda/traits.h"

#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace cuda {
namespace detail {

/** @brief Holder class to manage labelled collections of items */
template<typename T, bool PtrType = false>
class Holder {
public:
    using DepsT = std::vector<T>;

public:
    /** @brief Default constructor */
    Holder() = default;

    /** @brief Copy constructor is forbidden */
    Holder(Holder& other)       = delete;
    Holder(const Holder& other) = delete;

    /** @brief Move constructor */
    Holder(Holder&& other)
        : nodes_{ std::move(other.nodes_) }
    {
    }

    /** @brief Copy assignment is forbidden */
    Holder& operator=(Holder& other)       = delete;
    Holder& operator=(const Holder& other) = delete;

    /** @brief Move assignment */
    Holder& operator=(Holder&& other)
    {
        nodes_ = std::move(other.nodes_);
        return *this;
    }

    /** @brief Retrieve the node */
    T& operator[](const idx_t& id) { return nodes_.at(id); }

    /** @brief Retrieve the node */
    const T& operator[](const idx_t& id) const { return nodes_.at(id); }

    /** @brief Create a new slot */
    T& createSlot()
    {
        if constexpr (PtrType)
            nodes_.emplace_back(nullptr);
        else
            nodes_.emplace_back(T());
        return nodes_.back();
    }

    /** @brief Make the dependencies for the given node */
    template<typename IdxT = std::initializer_list<idx_t>>
    DepsT findDependencies(const IdxT& ids = {}) const
    {
        DepsT out;
        if (!empty()) {
            if constexpr (IsMultipleDependency<IdxT>::value) {
                /* No dependency. Defaults to the last node */
                if (ids.size() == 0) {
                    out.push_back(nodes_.back());
                } else {
                    for (const auto& id : ids) {
                        /* make sure that the same node id is not added twice */
                        if (std::find(out.begin(), out.end(), (*this)[id])
                            == out.end()) {
                            out.push_back((*this)[id]);
                        }
                    }
                }
            } else {
                out.push_back((*this)[ids]);
            }
        }
        return out;
    }

    /** @brief return whether this holder is empty */
    bool empty() const { return nodes_.empty(); }

    /** @brief Return the number of nodes in this holder */
    idx_t size() const { return nodes_.size(); }

protected:
    DepsT nodes_;
};

} // namespace detail
} // namespace cuda
} // namespace eagle

#endif
