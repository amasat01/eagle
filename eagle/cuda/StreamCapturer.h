// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cuda/Stream.h"
#include "eagle/util/CaptureGuard.h"

#include <optional>

#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace cuda {

/**
 * @brief RAII helper for recording kernel launches into a CUDA graph via stream capture.
 *
 * Call ``begin()`` before launching kernels on the associated stream, and
 * ``end()`` afterwards to obtain a ``cudaGraph_t`` that can be added to a
 * ``Graph`` with ``Graph::addNode()``.
 *
 * @note Capturing the default stream (``stream = 0``, the constructor's
 *       default) only works because eagle exports
 *       ``CUDA_API_PER_THREAD_DEFAULT_STREAM=1`` as an INTERFACE compile
 *       definition on their CMake targets; a consumer compiling bare nvcc
 *       without that target's flags hits ``cudaErrorStreamCaptureUnsupported``
 *       (900) at ``begin()``.
 * @note Available only when ``EAGLE_CPU_ONLY`` is not defined.
 * @see Graph::addNode() — consumes the captured graph.
 * @see eagle::util::CaptureGuardState — ``begin()``/``end()`` are a capture
 *      ENTRY/EXIT point (STOP-THE-LINE incident fix): ``captureScope_``
 *      brackets the process-wide capture-depth counter so a CUDA-touching
 *      teardown elsewhere (e.g. a GC'd ``Launcher``) defers itself instead
 *      of invalidating this capture.
 */
class StreamCapturer {
    using Self = StreamCapturer;

public:
    /** @brief Construct by assigning the stream */
    StreamCapturer(const cudaStream_t& stream = 0)
        : stream_{ stream }
    {
    }

    /** @brief Copy constructor. NOTE: ``captureScope_`` is deliberately NOT
     *  copied (default-constructed empty on the copy) -- each
     *  ``StreamCapturer`` object tracks its OWN begin/end pairing, and
     *  copying/moving one mid-capture was already fragile before this
     *  change (two objects aliasing the same ``stream_``'s capture
     *  session); this is not exercised anywhere in this codebase's actual
     *  usage (construct, begin, end, done). */
    StreamCapturer(const StreamCapturer& other)
        : stream_{ other.stream_ }
    {
    }

    /** @brief Move constructor. See the copy constructor's note --
     *  ``captureScope_`` is not transferred. */
    StreamCapturer(StreamCapturer&& other)
        : stream_{ other.stream_ }
    {
        other.stream_ = 0;
    }

    /** @brief Assignment from stream */
    Self& operator=(const cudaStream_t& str)
    {
        stream_ = str;
        return *this;
    }

    /** @brief Copy assignment */
    Self& operator=(const StreamCapturer& other)
    {
        stream_ = other.stream_;
        return *this;
    }

    /** @brief Move assignment */
    Self& operator=(StreamCapturer&& other)
    {
        stream_       = other.stream_;
        other.stream_ = 0;
        return *this;
    }

    /** @brief Update the stream */
    Self& stream(const cudaStream_t& stream)
    {
        stream_ = stream;
        return *this;
    }

    /** @brief Expose the stream */
    const cudaStream_t& stream() const { return stream_; }

    /** @brief Begin the capture
     *
     *  Checked with ``EAGLE_CHECK_ALWAYS`` rather than ``EAGLE_CHECK``: this
     *  is a cold, once-per-build control-plane call, and a swallowed failure
     *  here leaves the stream not capturing while the caller believes it is.
     *
     *  Enters the process-wide capture-depth scope (STOP-THE-LINE fix) only
     *  AFTER the begin call has actually succeeded -- a failed begin never
     *  opened a capture, so it must not raise the depth.
     */
    void begin()
    {
        util::CaptureGuardState::instance().requireIdleThread();
        EAGLE_CHECK_ALWAYS(
            cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        captureScope_.emplace();
    }

    /** @brief End the capture and return the created graph
     *
     *  Checked with ``EAGLE_CHECK_ALWAYS``. ``cudaStreamEndCapture`` reports
     *  structural faults in the captured DAG — most importantly
     *  ``cudaErrorStreamCaptureUnjoined``, raised when a forked branch stream
     *  was never joined back before the capture ended — through its return
     *  code, *and* writes a null ``cudaGraph_t``. Under the debug-only
     *  ``EAGLE_CHECK`` that error was discarded and the null handle returned
     *  to the caller, so the fault only surfaced later during adoption or
     *  launch. It now throws ``aether::Error`` at the point of failure.
     *
     *  Exits the process-wide capture-depth scope (STOP-THE-LINE fix)
     *  UNCONDITIONALLY, before the error check below -- the capture SESSION
     *  is over the moment ``cudaStreamEndCapture`` returns, whether or not
     *  its result reports success, so any deferred teardown queued during
     *  this capture must be free to drain even on this failure path.
     *
     *  @throws aether::Error  If the capture did not end cleanly.
     */
    [[nodiscard]] cudaGraph_t end()
    {
        cudaGraph_t graph            = nullptr;
        const cudaError_t captureErr = cudaStreamEndCapture(stream_, &graph);
        captureScope_.reset();
        EAGLE_CHECK_ALWAYS(captureErr);
        return graph;
    }

protected:
    cudaStream_t stream_;

    /** @brief Engaged between a successful ``begin()`` and the matching
     *  ``end()`` -- see ``eagle::util::CaptureGuardState``. Deliberately not
     *  copied/moved by the special members above (see their notes). */
    std::optional<util::CaptureScope> captureScope_;
};

} // namespace cuda
} // namespace eagle

#endif
