// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <vector>

namespace eagle {
namespace util {

/**
 * @brief Process-wide capture-depth counter + deferred-destroy queue
 *        (STOP-THE-LINE incident fix).
 *
 * @par The defect this closes
 * A CUDA-touching teardown (``cudaGraphExecDestroy``, ``cudaStreamDestroy``,
 * ``cudaGraphDestroy``, ``cudaFree``, ...) that fires WHILE any stream on the
 * process is actively being captured INVALIDATES that capture
 * (``cudaErrorStreamCaptureInvalidated``) even when the destroyed handle has
 * nothing to do with the capturing stream. Every such call site in this
 * codebase already swallows its OWN error (``EAGLE_CHECK_NOTHROW`` / a bare
 * unchecked call) so the destroy itself never throws -- correct, and by
 * design (a throwing destructor unwinding during teardown would
 * abort the process). But swallowing the ERROR does not undo the DAMAGE: the
 * capture is poisoned, the NEXT operation on it fails, and the native state
 * left behind can crash much later, at an unrelated GC or interpreter exit
 * (the exact shape root-caused in the incident this fixes: a dead
 * ``Launcher``'s ``cudaGraphExecDestroy``, run by Python's garbage collector
 * DURING a later, unrelated active capture, invalidated it).
 *
 * The fix is never making the illegal call while a capture is in flight, not
 * unswallowing its error. Every capture-opening call site (``StreamCapturer``,
 * ``CaptureConditional``) brackets itself with a :class:`CaptureScope`, which
 * increments/decrements ONE process-wide depth counter here. Every
 * CUDA-touching teardown site calls :meth:`CaptureGuardState::destroyOrDefer`
 * instead of calling the driver directly: if the depth is 0 (no capture
 * anywhere in the process), it runs immediately -- byte-identical to today's
 * behavior. If the depth is > 0, the destroy is queued and runs the moment
 * the OUTERMOST capture scope exits (``CaptureScope``'s destructor / explicit
 * ``release()``), plus a process-exit backstop so a queue that never drains
 * naturally (e.g. an aborted capture) does not leak forever.
 *
 * @par Thread-safety
 * This codebase's own GPU-serial law means captures are never concurrent
 * (one capture session in flight at a time, one thread driving it) --
 * documented ASSUMPTION, not separately enforced here. The mutex below is
 * cheap insurance against a destructor firing from a DIFFERENT thread than
 * the one driving capture (e.g. a refcount drop from an unrelated Python
 * thread), which the GPU-serial law does not by itself rule out.
 *
 * @note Pure host-side bookkeeping -- no CUDA type appears in this header,
 *       so it compiles identically in ``EAGLE_CPU_ONLY`` mode (there is no
 *       capture concept there; nothing calls into it).
 */
class CaptureGuardState {
public:
    /** @brief The single process-wide instance (Meyers singleton -- safe
     *  init order across translation units, destroyed at process exit,
     *  which is exactly when the backstop drain below needs to run). */
    static CaptureGuardState& instance()
    {
        static CaptureGuardState state;
        return state;
    }

    /** @brief Enter one capture scope. Call ONLY after the underlying
     *  ``cudaStream*Capture*`` call has SUCCEEDED (a failed begin never
     *  opened a capture, so it must not raise the depth). */
    void enter()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++depth_;
    }

    /** @brief Exit one capture scope. Symmetric with :meth:`enter` --
     *  callers must not call this without a matching prior ``enter()``
     *  (:class:`CaptureScope` enforces that pairing). Drains the deferred
     *  queue once the depth returns to 0. */
    void exit()
    {
        std::vector<std::function<void()>> drained;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (depth_ > 0)
                --depth_;
            if (depth_ == 0)
                drained.swap(pending_);
        }
        // Run outside the lock: a queued destroy must never re-enter this
        // class while mutex_ is held (none currently do; this just keeps
        // that true even if a future destroy op ever touched capture state).
        for (auto& op : drained)
            op();
    }

    /** @brief Run @p op now if no capture is active anywhere in the
     *  process (depth 0) -- identical timing to today's inline destroy.
     *  Otherwise enqueue it to run once the outermost capture scope exits
     *  (or at process exit, as a backstop). @p op must be the FULLY
     *  self-contained destroy call (its own error policy already decided
     *  at the call site, e.g. wrapped in ``EAGLE_CHECK_NOTHROW`` there) --
     *  this method does not add or remove error handling. */
    void destroyOrDefer(std::function<void()> op)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (depth_ > 0) {
                pending_.push_back(std::move(op));
                return;
            }
        }
        op();
    }

    /** @brief Current capture depth (tests / diagnostics). */
    int depth() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return depth_;
    }

    /** @brief Number of destroys queued, not yet run (tests / diagnostics). */
    std::size_t pendingCount() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_.size();
    }

    /** @brief Process-exit backstop: run any still-pending destroys
     *  regardless of depth, so an aborted/never-closed capture cannot leak
     *  them forever. Never throws -- teardown-adjacent, same convention as
     *  ``EAGLE_CHECK_NOTHROW`` call sites (a throwing destructor unwinding
     *  during process teardown would abort rather than report). */
    ~CaptureGuardState()
    {
        std::vector<std::function<void()>> drained;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            drained.swap(pending_);
        }
        for (auto& op : drained) {
            try {
                op();
            } catch (...) {
                // Swallowed deliberately: see the note above.
            }
        }
    }

    CaptureGuardState(const CaptureGuardState&)            = delete;
    CaptureGuardState& operator=(const CaptureGuardState&) = delete;

private:
    CaptureGuardState() = default;

    mutable std::mutex mutex_;
    int depth_ = 0;
    std::vector<std::function<void()>> pending_;
};

/**
 * @brief RAII capture-scope token: ``enter()``s :class:`CaptureGuardState`
 *        on construction, ``exit()``s on destruction or an explicit
 *        :meth:`release`.
 *
 * Intended as an ``std::optional<CaptureScope>`` MEMBER of a capture-opening
 * class (``StreamCapturer``, ``CaptureConditional``): ``.emplace()`` it right
 * after the underlying begin-capture call succeeds, ``.reset()`` it (calling
 * :meth:`release`) right after the underlying end-capture call runs --
 * regardless of whether that call reports success, since the capture SESSION
 * is over either way. Exception-safe even if a caller never reaches an
 * explicit release: the owning object's OWN destructor still destroys this
 * member (whenever that object itself is destroyed), so the depth is always
 * eventually decremented.
 */
class CaptureScope {
public:
    CaptureScope()
    {
        CaptureGuardState::instance().enter();
        active_ = true;
    }
    ~CaptureScope() { release(); }

    CaptureScope(const CaptureScope&)            = delete;
    CaptureScope& operator=(const CaptureScope&) = delete;
    CaptureScope(CaptureScope&&)                 = delete;
    CaptureScope& operator=(CaptureScope&&)      = delete;

    /** @brief Exit the scope early. Idempotent -- calling it again (or on a
     *  default-constructed-then-never-entered token, which never happens
     *  here since the constructor always enters) is a no-op. */
    /* GCC 14's -Wmaybe-uninitialized invents a path into the read below when
     * this scope lives inside a std::optional payload and the whole chain
     * inlines into the owning object's destructor at -O3 (first seen from
     * a downstream consumer's test under gcc-toolset-14). The
     * path does not exist: optional's reset/destroy only runs ENGAGED
     * (_M_engaged-guarded), and active_ carries a default initializer besides.
     * Suppressed at exactly this read. */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
    void release()
    {
        if (active_) {
            CaptureGuardState::instance().exit();
            active_ = false;
        }
    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

private:
    bool active_ = false;
};

} // namespace util
} // namespace eagle
