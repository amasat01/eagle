// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/conditional/ConditionalGroup.h"
#include "eagle/cuda/Stream.h"
#include "eagle/cuda/detail/RuntimeCompat.h"
#include "eagle/util/CaptureGuard.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/throw.h"

#include <optional>

#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace cuda {

/**
 * @brief Weave a device-evaluated conditional node -- an IF (skippable
 *        region) or a WHILE (device-side loop) -- into an in-flight stream
 *        capture: the ``CaptureFork`` sibling for gated regions.
 *
 * ``GraphPipeline``-style capture records ordinary launches as a linear
 * chain; there is no capture-time entry point for a ``cudaGraphAddNode``
 * conditional today. ``CaptureConditional`` supplies exactly that, using the
 * capture-interop mechanism proven under
 * ``cudaStreamCaptureModeGlobal`` (eagle's real capture mode,
 * ``StreamCapturer::begin``): query the in-flight graph and dependency set via
 * ``cudaStreamGetCaptureInfo_v3``, create a conditional handle bound to it,
 * add the setter kernel node (``setCond`` for an IF -- the same ``CountGuard``
 * evaluation ``ConditionalGroup`` uses -- or the loop head for a WHILE) and
 * the conditional node explicitly, splice them back into the capture
 * stream's dependency set via ``cudaStreamUpdateCaptureDependencies``, then
 * open a NESTED capture of the body on a separate, pre-created stream.
 *
 * @par The two kinds
 * - IF (the two-argument constructor): the body runs at most once per
 *   replay, iff the guard holds when the setter node runs.
 * - WHILE (the four-argument constructor): the body runs up to @p loopCap
 *   times per replay, while the guard holds. The loop state is ONE
 *   caller-owned device ``unsigned int[2]`` cell ``[remaining, ran]``:
 *   the head setter (the node BEFORE the WHILE node) resets it every replay
 *   (``remaining = cap, ran = 0``) and sets ``handle = guard && cap > 0``;
 *   the tail setter, launched by ``end()`` on the body stream as the body's
 *   LAST node, sets ``handle = guard && --remaining > 0`` and ``++ran``. No
 *   memset node, no ``cudaGraphCondAssignDefault``, no host trip. ``ran``
 *   is the device-visible iteration index (0 during the first iteration),
 *   readable by body kernels through the same cell. An empty loop body is
 *   refused like an empty IF body -- it would be an infinite loop.
 *
 * @par Lifecycle -- mirrors ``CaptureFork`` exactly
 * Construction is the only phase that acquires resources (the body stream):
 * resource creation is illegal mid-capture under Global mode, so it must
 * happen BEFORE ``StreamCapturer::begin()``. ``begin()`` performs the
 * mid-capture weave and returns the body stream to launch the guarded
 * region's work on; ``end()`` closes the body capture. ``end()`` is
 * idempotent and the destructor calls it automatically (never throws from
 * the destructor -- same rationale as ``CaptureFork``: a failure here is
 * already being reported by the capture that is about to fail).
 *
 * @par Origin
 * @p origin may be the pipeline's main capturing stream, a ``CaptureFork``
 * branch stream, OR another ``CaptureConditional``'s body stream
 * (``bodyStream()``): weaving on any actively capturing stream is the same
 * mechanism (an eagle-owned kernel node followed by an explicitly-added
 * conditional node spliced back into whichever stream is capturing). A
 * skippable region inside a loop body is exactly an IF weave whose origin
 * is the WHILE weave's body stream.
 *
 * @note Available only when ``EAGLE_CPU_ONLY`` is not defined.
 * @see ConditionalGroup -- the explicit-builder sibling (IF only in v1:
 *      same predicate, same node shape, non-capture assembly).
 * @see CaptureFork -- the sibling this class's lifecycle mirrors.
 */
class CaptureConditional {
public:
    /** @brief Create the body stream for a later IF weave.
     *
     *  Must be called BEFORE ``StreamCapturer::begin()`` -- stream creation
     *  is not permitted while a Global-mode capture is in flight.
     *
     *  @param[in] origin  Stream the skippable region would have been
     *                     captured on directly (the pipeline's main stream,
     *                     a ``CaptureFork`` branch, or another weave's body
     *                     stream) -- the stream ``begin()`` will weave the
     *                     conditional into.
     *  @param[in] guard   The declarative predicate; retained by value
     *                     for the lifetime of this object.
     */
    CaptureConditional(const cudaStream_t& origin, CountGuard guard)
        : origin_{ origin }
        , guard_{ guard }
        , bodyStream_{ /*nonBlocking=*/true }
    {
    }

    /** @brief Create the body stream for a later WHILE weave (device-side
     *  loop). Same pre-capture rule as the IF constructor.
     *
     *  @param[in] loopCap      Maximum iterations per replay, >= 1; baked
     *                          into the head setter as a kernel argument
     *                          (a new cap is a new build). 0 is refused:
     *                          a loop that can never run is a caller error,
     *                          not a valid degenerate loop.
     *  @param[in] loopCounter  Device ``unsigned int[2]`` cell
     *                          ``[remaining, ran]`` owned by the caller for
     *                          the graph's whole lifetime (never nullptr).
     *                          eagle writes both words every replay. */
    CaptureConditional(const cudaStream_t& origin, CountGuard guard,
        unsigned int loopCap, unsigned int* loopCounter)
        : origin_{ origin }
        , guard_{ guard }
        , bodyStream_{ /*nonBlocking=*/true }
        , loopCap_{ loopCap }
        , loopCounter_{ loopCounter }
    {
        EAGLE_ASSERT(loopCounter != nullptr,
            "CaptureConditional: a WHILE weave needs a device [remaining, "
            "ran] counter cell (nullptr given)");
        if (loopCap == 0u)
            EAGLE_THROW(std::invalid_argument, "CaptureConditional: loopCap "
                "must be >= 1 (a loop capped at 0 iterations can never run)");
    }

    /** @brief Not copyable -- it owns the body stream. */
    CaptureConditional(const CaptureConditional&)            = delete;
    /** @brief Not copyable -- it owns the body stream. */
    CaptureConditional& operator=(const CaptureConditional&) = delete;
    /** @brief Not movable -- its lifetime is the begin/end scope. */
    CaptureConditional(CaptureConditional&&)                 = delete;
    /** @brief Not movable -- its lifetime is the begin/end scope. */
    CaptureConditional& operator=(CaptureConditional&&)      = delete;

    /** @brief True for a WHILE weave, false for an IF weave. */
    bool isLoop() const { return loopCounter_ != nullptr; }

    /** @brief The pre-created body stream, valid from construction: the
     *  origin a NESTED weave (a skippable region inside this loop's body)
     *  must be constructed against, BEFORE capture begins -- ``begin()``
     *  returns this same handle later, mid-capture, but a nested weave's
     *  own body stream cannot be created by then. */
    cudaStream_t bodyStream() const { return bodyStream_.cuda(); }

    /**
     * @brief Weave the setter kernel + conditional node at the origin's
     *        current capture tip and open capture of the body on the
     *        pre-created body stream.
     *
     *  Call with a capture already in flight on @p origin (the stream
     *  passed to the constructor). Every CUDA API call here is a cold,
     *  once-per-build control-plane call, checked with
     *  ``EAGLE_CHECK_ALWAYS`` (never the debug-only ``EAGLE_CHECK``) --
     *  same convention as ``StreamCapturer`` / ``CaptureFork``.
     *
     *  @return The body stream: launch the guarded region's work on it
     *          (e.g. via ``cp.cuda.ExternalStream`` on the Python side).
     */
    cudaStream_t begin()
    {
        cudaStreamCaptureStatus status;
        unsigned long long capId          = 0;
        cudaGraph_t capturedGraph         = nullptr;
        const cudaGraphNode_t* deps       = nullptr;
        const cudaGraphEdgeData* edgeData = nullptr;
        std::size_t numDeps               = 0;
        EAGLE_CHECK_ALWAYS(cuda::detail::streamGetCaptureInfo(origin_, &status,
            &capId, &capturedGraph, &deps, &edgeData, &numDeps));
        EAGLE_ASSERT(status == cudaStreamCaptureStatusActive,
            "CaptureConditional::begin: origin stream is not actively "
            "capturing");

        // No cudaGraphCondAssignDefault: the setter node below writes the
        // handle on every replay before the conditional node reads it (and
        // the loop tail rewrites it after every iteration), so a per-launch
        // reset to the default would only be overwritten -- and on SM 6.1 it
        // costs ~8 us of GPU time per replay.
        EAGLE_CHECK_ALWAYS(cudaGraphConditionalHandleCreate(
            &handle_, capturedGraph, 0, 0));

        cudaGraphNode_t nSet = isLoop()
            ? conditional::detail::addSetLoopHeadKernelNode(capturedGraph,
                handle_, guard_, loopCap_, loopCounter_, deps, numDeps)
            : conditional::detail::addSetCondKernelNode(
                capturedGraph, handle_, guard_, deps, numDeps);

        cudaGraphNodeParams cp = {};
        cp.type                = cudaGraphNodeTypeConditional;
        cp.conditional.handle  = handle_;
        cp.conditional.type    = isLoop() ? cudaGraphCondTypeWhile
                                          : cudaGraphCondTypeIf;
        cp.conditional.size    = 1;
        cudaGraphNode_t nCond;
        EAGLE_CHECK_ALWAYS(
            cuda::detail::graphAddNode(&nCond, capturedGraph, &nSet, 1, &cp));
        EAGLE_CHECK_ALWAYS(cuda::detail::streamUpdateCaptureDependencies(
            origin_, &nCond, 1, cudaStreamSetCaptureDependencies));

        bodyGraph_ = cp.conditional.phGraph_out[0];
        EAGLE_CHECK_ALWAYS(cudaStreamBeginCaptureToGraph(bodyStream_.cuda(),
            bodyGraph_, nullptr, nullptr, 0, cudaStreamCaptureModeGlobal));
        open_ = true;
        // STOP-THE-LINE fix: this opens a NESTED capture scope (the body
        // stream) -- enter the process-wide depth counter only after the
        // begin-capture call above actually succeeded. See
        // eagle::util::CaptureGuardState's doc for the incident this closes.
        captureScope_.emplace();
        return bodyStream_.cuda();
    }

    /** @brief Close the body capture. Idempotent -- calling it again (or
     *  without a preceding ``begin()``) is a no-op, not an error.
     *
     *  For a WHILE weave this first checks the body is non-empty (counted
     *  MID-capture, before anything eagle adds itself) and then launches
     *  the loop tail setter on the body stream, so it is captured as the
     *  body's last node -- after every node the caller launched, including
     *  a nested weave's conditional node -- before the body capture ends.
     *
     *  Exits the process-wide capture-depth scope (STOP-THE-LINE fix)
     *  UNCONDITIONALLY before the error check -- the nested capture session
     *  is over the moment ``cudaStreamEndCapture`` returns, whether or not
     *  it reports success. */
    void end()
    {
        if (!open_)
            return;

        // A conditional node whose body is empty never completes when it
        // fires (an IF hangs the stream; a WHILE is an infinite loop by
        // construction), so refuse a region that launched nothing. Counted
        // here, BEFORE the loop tail below is launched: the tail would
        // otherwise make every empty loop body look non-empty.
        // cudaGraphGetNodes on the body graph is valid mid-capture (the
        // capture adds nodes to the graph as launches are recorded).
        std::size_t bodyNodes = 0;
        const cudaError_t countErr = cudaGraphGetNodes(bodyGraph_, nullptr, &bodyNodes);
        if (countErr == cudaSuccess && bodyNodes > 0 && isLoop()) {
            // Captured on the body stream after everything the caller
            // launched there, so it depends on the whole body: the loop's
            // continue/stop decision is made once the iteration is done.
            conditional::detail::setLoopTailKernel<>
                <<<1, 1, 0, bodyStream_.cuda()>>>(
                    handle_, guard_.count, guard_.baseline, loopCounter_);
        }

        cudaGraph_t bodyOut           = nullptr;
        const cudaError_t captureErr  = cudaStreamEndCapture(bodyStream_.cuda(), &bodyOut);
        open_ = false;
        captureScope_.reset();
        EAGLE_CHECK_ALWAYS(countErr);
        EAGLE_CHECK_ALWAYS(captureErr);
        if (bodyNodes == 0)
            EAGLE_THROW(std::invalid_argument, isLoop()
                ? "CaptureConditional::end: the loop body launched no work; "
                  "a WHILE node with an empty body is an infinite loop -- the "
                  "step must launch at least one kernel"
                : "CaptureConditional::end: the guarded region launched no "
                  "work; an IF node with an empty body never completes when "
                  "it fires -- the step must launch at least one kernel");
    }

    /** @brief Close the body capture automatically if the caller did not.
     *  Never throws -- a destructor that escaped during unwinding would
     *  terminate, and a failure here is already being reported by the
     *  capture that is about to fail (identical rationale to
     *  ``CaptureFork::~CaptureFork``). */
    ~CaptureConditional()
    {
        try {
            end();
        } catch (...) {
            /* swallowed deliberately: see the note above */
        }
    }

private:
    cudaStream_t origin_;
    CountGuard guard_;
    Stream bodyStream_;
    unsigned int loopCap_          = 0;        // WHILE only; 0 for an IF weave
    unsigned int* loopCounter_     = nullptr;  // WHILE only; nullptr == IF weave
    cudaGraphConditionalHandle handle_ = 0;    // set by begin(); read by the tail launch
    cudaGraph_t bodyGraph_ = nullptr;
    bool open_             = false;

    /** @brief Engaged between a successful ``begin()`` and the matching
     *  ``end()`` -- see ``eagle::util::CaptureGuardState`` (STOP-THE-LINE
     *  fix). Tracks the SAME open/closed transitions as ``open_`` above;
     *  kept as a separate member rather than replacing ``open_`` to keep
     *  this change minimal against the pre-existing idempotency contract. */
    std::optional<util::CaptureScope> captureScope_;
};

} // namespace cuda
} // namespace eagle

#endif
