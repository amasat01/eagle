// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/util/Observer.h"

namespace eagle {
namespace util {

/** @brief CRTP class for Observability traits */
template<typename ChildT>
struct Observable {
    /** @brief Declare observer type */
    using Self      = ChildT;
    using ObserverT = Observer<Self>;

    /** @brief Register a new observer */
    void registerObserver(ObserverT* obs)
    {
        if (obs != nullptr) {
            observers_.push_back(obs);
            obs->onNotify(static_cast<Self*>(this));
        }
    }

    /** @brief Deregister the given observer */
    void deregisterObserver(ObserverT* obs)
    {
        if (obs != nullptr) {
            observers_.erase(
                std::remove(observers_.begin(), observers_.end(), obs),
                observers_.end());
        }
    }

    /** @brief Notify our observers of a move */
    void notifyObservers()
    {
        for (ObserverT* obs : observers_) {
            obs->onNotify(static_cast<Self*>(this));
        }
    }

    /** @brief Expose the observers */
    std::vector<ObserverT*>& observers() { return observers_; }
    const std::vector<ObserverT*>& observers() const { return observers_; }

    /** @brief Handle to perform a full move */
    void finalizeMove(Self&& other)
    {
        observers_ = std::move(other.observers_);
        notifyObservers();
    }

protected:
    /* Observers for move semantics tracing */
    std::vector<ObserverT*> observers_;
};
} // namespace util
} // namespace eagle
