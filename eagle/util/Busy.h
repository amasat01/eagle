// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file util/Busy.h
 * @brief The per-object busy guard: one object shared across threads is
 *        memory-safe and caller-ordered, and every method that can run for a
 *        while (build, launch, enqueue) enters a non-blocking guard so a second
 *        thread entering the SAME object is told, never queued behind it and
 *        never handed a half-built state. Distinct objects never meet.
 *
 * Header-only, no CUDA, no allocation on the enter path beyond naming the
 * holder.
 */

#include <atomic>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace eagle {
namespace util {

/** @brief Thrown when a guarded method is entered while another is in flight
 *  on the same object. The text names the owner, the method refused and the
 *  method in flight. */
struct Busy : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class BusyGuard {
public:
    /** @brief RAII ownership of the guard; leaves it on destruction. */
    class Scope {
    public:
        Scope() = default;
        Scope(Scope&& o) noexcept : guard_(std::exchange(o.guard_, nullptr)) {}
        Scope& operator=(Scope&& o) noexcept
        {
            if (this != &o) {
                release_();
                guard_ = std::exchange(o.guard_, nullptr);
            }
            return *this;
        }
        Scope(const Scope&)            = delete;
        Scope& operator=(const Scope&) = delete;
        ~Scope() { release_(); }

    private:
        friend class BusyGuard;
        explicit Scope(BusyGuard* g) : guard_(g) {}
        void release_()
        {
            if (guard_ != nullptr)
                std::exchange(guard_, nullptr)->leave();
        }
        BusyGuard* guard_ = nullptr;
    };

    /** @param owner the text before the method name in a refusal, e.g.
     *  ``"eagle.Graph"``. */
    explicit BusyGuard(std::string owner) : owner_(std::move(owner)) {}

    BusyGuard(const BusyGuard&)            = delete;
    BusyGuard& operator=(const BusyGuard&) = delete;

    /** @brief Take the guard for ``method`` or return false (never blocks). */
    bool tryEnter(const std::string& method)
    {
        if (flag_.test_and_set(std::memory_order_acquire))
            return false;
        std::lock_guard<std::mutex> lock(nameMutex_);
        current_ = method;
        return true;
    }

    /** @brief Take the guard for ``method``.
     *  @throws Busy ``"<owner>.<method>: another call is still running on this
     *          <Type>; give each thread its own <Type>, or wait for it"`` when
     *          another call holds it; ``<Type>`` is the last dotted part of the owner. */
    void enter(const std::string& method)
    {
        if (tryEnter(method))
            return;
        const std::string type = owner_.substr(owner_.rfind('.') == std::string::npos ? 0 : owner_.rfind('.') + 1);
        throw Busy(owner_ + "." + method + ": another call is still running on this " + type
            + "; give each thread its own " + type + ", or wait for it");
    }

    /** @brief Scoped form of @ref enter. */
    Scope scope(const std::string& method)
    {
        enter(method);
        return Scope(this);
    }

    /** @brief Release a guard taken with @ref enter / @ref tryEnter. */
    void leave()
    {
        {
            std::lock_guard<std::mutex> lock(nameMutex_);
            current_.clear();
        }
        flag_.clear(std::memory_order_release);
    }

    /** @brief The method in flight (empty when idle). */
    std::string current() const
    {
        std::lock_guard<std::mutex> lock(nameMutex_);
        return current_;
    }

    const std::string& owner() const { return owner_; }

private:
    std::string owner_;
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
    mutable std::mutex nameMutex_;
    std::string current_;
};

}  // namespace util
}  // namespace eagle
