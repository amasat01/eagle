// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/CaptureGuard.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"

#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace cuda {

/* Forward declare event*/
class Event;

/** @brief Stream manager for non-default streams */
class Stream {
public:
    /** @brief Construct a stream.
     *
     *  @param nonBlocking when true the stream is created with
     *  ``cudaStreamNonBlocking`` (no implicit dependency on the legacy
     *  default stream) — the correct choice for CUDA-graph capture that
     *  must not entangle the default stream other libraries (cupy, torch)
     *  launch on. Default ``false`` reproduces ``cudaStreamCreate``
     *  (``cudaStreamDefault``) exactly — but note precisely what that
     *  buys: implicit ordering against the LEGACY default stream ONLY.
     *  Under ``-DCUDA_API_PER_THREAD_DEFAULT_STREAM=1`` (or the
     *  ``--default-stream per-thread`` compile flag) stream-0 work runs
     *  on the per-thread default stream and has NO implicit ordering
     *  edge, in either direction, with a blocking eagle Stream — a
     *  default-stream launch can race work on this stream with every
     *  CUDA call returning ``cudaSuccess``. Consumers mixing
     *  default-stream launches with eagle-Stream work under per-thread
     *  default streams must order explicitly (an Event or a sync).
     *  "Existing consumers are unchanged" therefore holds only for
     *  legacy-default-stream builds. */
    explicit Stream(bool nonBlocking = false) { create_(nonBlocking); }

    /** @brief Copy constructor */
    Stream(Stream& other)       = delete;
    Stream(const Stream& other) = delete;

    /** @brief Move constructor */
    Stream(Stream&& other)
        : created_{ std::exchange(other.created_, false) }
        , cuda_{ std::exchange(other.cuda_, nullptr) }
    {
    }

    /** @brief Copy constructor is forbidden */
    Stream& operator=(Stream& other)       = delete;
    Stream& operator=(const Stream& other) = delete;

    /** @brief Move assignment operator*/
    Stream& operator=(Stream&& other) noexcept
    {
        destroy_();
        created_ = std::exchange(other.created_, false);
        cuda_    = std::exchange(other.cuda_, nullptr);
        return *this;
    }

    /** @brief Wait for the given event */
    inline void waitFor(const Event& event);

    /** @brief Synchronize the host wrt this stream */
    void synchronize() const { EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(cuda_)); }

    /** @brief Get the low-level stream */
    const cudaStream_t& cuda() const { return cuda_; }

    /** @brief Default constructor */
    ~Stream() { destroy_(); }

protected:
    void create_(bool nonBlocking = false)
    {
        if (!created_) {
            EAGLE_CHECK_ALWAYS(cudaStreamCreateWithFlags(&cuda_,
                nonBlocking ? cudaStreamNonBlocking : cudaStreamDefault));
            created_ = true;
        }
    }

    void destroy_()
    {
        if (created_ && cuda_ != nullptr) {
            // NEVER double-destroy a cudaStream_t to "test" this line by hand:
            // verified (CUDA 12.6, P2000) to SEGFAULT inside the CUDA runtime
            // itself, unlike cudaEvent_t/cudaGraphExec_t double-destroy, which
            // return a clean error. See helper_stream_nothrow_reset.cu for the
            // safe (process-isolated, post-cudaDeviceReset) reproduction.
            //
            // STOP-THE-LINE fix: this object's own bookkeeping (created_/
            // cuda_) is retired IMMEDIATELY below (so a second destroy_()
            // call is still the safe no-op the guard above already made it),
            // but the ACTUAL driver call is handed to
            // eagle::util::CaptureGuardState::destroyOrDefer, which runs it
            // now if no capture is active anywhere in the process (today's
            // exact timing) or queues it to run once the outermost capture
            // scope exits (a destructor firing mid-capture, e.g. from
            // Python GC, must never invalidate that capture). The queued
            // closure still owns exactly one destroy of `handle` -- no
            // double-destroy risk, since `cuda_` is already cleared here.
            const cudaStream_t handle = cuda_;
            created_                  = false;
            cuda_                     = nullptr;
            util::CaptureGuardState::instance().destroyOrDefer(
                [handle]() { EAGLE_CHECK_NOTHROW(cudaStreamDestroy(handle)); });
        }
    }

    bool created_      = false;
    cudaStream_t cuda_ = nullptr;
};

} // namespace cuda
} // namespace eagle

#endif
