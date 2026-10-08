// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/** @brief Templated observer for observer pattern - used to track move
 * semantics with custom references */
#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/throw.h"

namespace eagle {
namespace util {

/** @brief Move semantics observer */
template<typename T>
class Observer {
public:
    /** @brief Default construction */
    Observer() = default;

    /** @brief Construct by directly observing an object */
    Observer(T& obj) { observe(obj); }

    /** @brief Construct with the given pointer to object (if not nullptr) */
    Observer(T* obj)
    {
        if (obj != nullptr)
            observe(*obj);
    }

    /** @brief Copy construction is forbidden */
    Observer(Observer& other)       = delete;
    Observer(const Observer& other) = delete;

    /** @brief Move construction is allowed */
    Observer(Observer&& other) { *this = std::move(other); }

    /** @brief Copy assignment is forbidden */
    Observer& operator=(Observer& other)       = delete;
    Observer& operator=(const Observer& other) = delete;

    /** @brief Move assignment is allowed */
    Observer& operator=(Observer&& other)
    {
        if (ref_ == other.ref_) {
            /* Same referent (or both null): this observer's own registration
             * already stands — only the source's must go, or the referent
             * keeps a pointer into whatever storage `other` occupied. */
            if (ref_ != nullptr)
                ref_->deregisterObserver(&other);
        } else {
            /* Leave the current referent BEFORE adopting the new one. The old
             * observable must never retain a pointer to an observer that no
             * longer references it: its next notify (any move of the
             * observable) writes through that pointer, and this observer's
             * destructor — keyed on ref_ — would never clean it up. This is
             * REQUIRED even when `other` is empty (release-by-assignment). */
            if (ref_ != nullptr)
                ref_->deregisterObserver(this);
            ref_ = other.ref_;
            updateRegistration_(std::move(other));
        }
        other.ref_ = nullptr;
        return *this;
    }

    /** @brief Observe */
    void observe(T& obj)
    {
        if (ref_ == &obj)
            return;
        /* Same invariant as move assignment: leave the old referent first. */
        if (ref_ != nullptr)
            ref_->deregisterObserver(this);
        obj.registerObserver(this);
    }

    /** @brief Update internal refernece on the move */
    void onNotify(T* newobj) { ref_ = newobj; }

    /** @brief Expose the log */
    T* operator->()
    {
        assertValidObj_();
        return ref_;
    }
    const T* operator->() const
    {
        assertValidObj_();
        return ref_;
    }

    /** @brief Destructor calls de-registration */
    ~Observer()
    {
        if (ref_ != nullptr)
            ref_->deregisterObserver(this);
    }

    /** @brief Create a clone of this observer */
    Observer clone() const { return Observer(const_cast<T&>(ref_)); }

private:
    /** @brief run registration and de-registration on move */
    void updateRegistration_(Observer&& other)
    {
        if (other.ref_ != nullptr) {
            ref_->deregisterObserver(&other);
            ref_->registerObserver(this);
        }
    }

    /** @brief Assert the obj validity */
    void assertValidObj_() const
    {
        EAGLE_ASSERT(ref_ != nullptr, "Invalid object reference in Observer");
    }

    T* ref_ = nullptr;
};

} // namespace util
} // namespace eagle
