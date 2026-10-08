// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cuda/stream/Stream.h"
#include "eagle/util/CaptureGuard.h"

#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace cuda {

/** @brief Event manager for non-default streams */
class Event {
public:
    /** @brief Default constructor */
    Event(const unsigned int& flags = cudaEventDefault) { create_(flags); }

    /** @brief Copy constructor */
    Event(Event& other)       = delete;
    Event(const Event& other) = delete;

    /** @brief Move constructor */
    Event(Event&& other)
        : created_{ std::exchange(other.created_, false) }
        , cuda_{ std::exchange(other.cuda_, nullptr) }
    {
    }

    /** @brief Copy constructor is forbidden */
    Event& operator=(Event& other)       = delete;
    Event& operator=(const Event& other) = delete;

    /** @brief Move assignment operator*/
    Event& operator=(Event&& other) noexcept
    {
        destroy_();
        created_ = std::exchange(other.created_, false);
        cuda_    = std::exchange(other.cuda_, nullptr);
        return *this;
    }

    /** @brief record the event on the current stream */
    void record(const Stream& stream)
    {
        EAGLE_CHECK_ALWAYS(cudaEventRecord(cuda_, stream.cuda()));
    }

    /** @brief Synchronize the host wrt this event */
    void synchronize() const { EAGLE_CHECK_ALWAYS(cudaEventSynchronize(cuda_)); }

    /** @brief Get the low-level stream */
    const cudaEvent_t& cuda() const { return cuda_; }

    /** @brief Wait event query */
    inline void waitQuery() const
    {
        while (cudaEventQuery(cuda_) != cudaSuccess) {
            /* do nothing */
        }
    }

    /** @brief Default constructor */
    ~Event() { destroy_(); }

protected:
    void create_(const unsigned int& flags = cudaEventDefault)
    {
        if (!created_) {
            EAGLE_CHECK_ALWAYS(cudaEventCreateWithFlags(&cuda_, flags));
            created_ = true;
        }
    }

    void destroy_()
    {
        if (created_ && cuda_ != 0) {
            // STOP-THE-LINE fix: see Stream::destroy_()'s comment for the
            // full rationale -- this object's own state retires immediately
            // below; only the driver call is deferred if a capture is
            // active anywhere in the process.
            const cudaEvent_t handle = cuda_;
            created_                 = false;
            cuda_                    = nullptr;
            util::CaptureGuardState::instance().destroyOrDefer(
                [handle]() { EAGLE_CHECK_NOTHROW(cudaEventDestroy(handle)); });
        }
    }

    bool created_     = false;
    cudaEvent_t cuda_ = nullptr;
};

/* finalize the implementation of wait for */
inline void Stream::waitFor(const Event& event)
{
    EAGLE_CHECK_ALWAYS(cudaStreamWaitEvent(cuda_, event.cuda()));
}

} // namespace cuda
} // namespace eagle

#endif
