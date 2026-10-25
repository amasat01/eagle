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
#include <vector>

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
 * ``cudaStreamCaptureModeThreadLocal`` (eagle's real capture mode,
 * ``StreamCapturer::begin``): query the in-flight graph and dependency set via
 * ``cudaStreamGetCaptureInfo_v3``, create a conditional handle bound to it,
 * add the setter kernel node (``setCond`` for an IF -- the same ``CountGuard``
 * evaluation ``ConditionalGroup`` uses -- or the loop head for a WHILE) and
 * the conditional node explicitly, splice them back into the capture
 * stream's dependency set via ``cudaStreamUpdateCaptureDependencies``, then
 * open a NESTED capture of the body on a separate, pre-created stream.
 *
 * @par The kinds
 * The guard is ``(*count != base) != negate`` (a null ``count`` always
 * holds), evaluated by an eagle setter kernel at the node's position.
 * - IF: the body runs at most once per replay, iff the guard holds.
 * - IF/ELSE (two arms): arm 0 runs iff the guard holds, arm 1 otherwise.
 * - WHILE: the body runs while the guard holds, within its
 *   ``conditional::LoopTerms`` (cap, runtime trip count, break cell). The
 *   loop state is ONE caller-owned device ``unsigned int[2]`` cell
 *   ``[remaining, ran]``: the head setter (the node BEFORE the WHILE node)
 *   resets it every replay; the tail setter, launched by ``end()`` on the
 *   body stream as the body's LAST node, decrements ``remaining``,
 *   increments ``ran`` and decides the next iteration. No memset node, no
 *   ``cudaGraphCondAssignDefault``, no host trip. ``ran`` is the
 *   device-visible iteration index (0 during the first iteration),
 *   readable by body kernels through the same cell.
 * - SWITCH (n arms): arm ``*count`` runs (none when out of range).
 * IF/ELSE and SWITCH need an NVIDIA driver >= 12.8 and are refused at
 * construction below it (``conditionalArmsSupported``). An empty arm is
 * refused -- an empty conditional body never completes when it fires.
 *
 * @par Lifecycle -- mirrors ``CaptureFork`` exactly
 * Construction is the only phase that acquires resources (the body stream):
 * resource creation is illegal mid-capture on the capturing thread, so it must
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
/** @brief The node form a ``CaptureConditional`` weaves. */
enum class CondKind : unsigned char { If, IfElse, While, Switch };

/** @brief The driver version from which IF/ELSE (two arms) and SWITCH
 *  conditional nodes exist (CUDA 12.8). */
inline constexpr int kConditionalArmsDriver = 12080;

/** @brief Whether the installed driver provides IF/ELSE and SWITCH nodes.
 *  The value reaches the driver as a plain enum, so the toolkit the caller
 *  was built with does not matter. */
inline bool conditionalArmsSupported()
{
    int version = 0;
    return cudaDriverGetVersion(&version) == cudaSuccess
        && version >= kConditionalArmsDriver;
}

class CaptureConditional {
public:
    /** @brief Create the arm streams for a later weave of @p kind.
     *
     *  Must be called BEFORE ``StreamCapturer::begin()`` -- stream creation
     *  is not permitted while a capture is in flight.
     *
     *  @param[in] origin  Stream the region would have been captured on
     *                     directly (the pipeline's main stream, a
     *                     ``CaptureFork`` branch, or another weave's arm
     *                     stream) -- the stream ``begin()`` weaves into.
     *  @param[in] kind    The node form.
     *  @param[in] guard   The declarative predicate (SWITCH: ``count`` is
     *                     the arm index); retained by value.
     *  @param[in] arms    0 = the kind's own (IF/WHILE 1, IF/ELSE 2);
     *                     SWITCH needs >= 1.
     *  @param[in] negate  Nonzero inverts the guard (``not cell``, ``else``).
     *  @param[in] loop    WHILE only: ``cap`` >= 1 (``kLoopUncapped`` for
     *                     none) and a non-null ``counter``.
     *  @throws std::invalid_argument on an inconsistent form;
     *          std::runtime_error for IF/ELSE or SWITCH below driver 12.8.
     */
    CaptureConditional(const cudaStream_t& origin, CondKind kind,
        CountGuard guard, unsigned int arms = 0, unsigned int negate = 0,
        conditional::LoopTerms loop = {})
        : origin_{ origin }
        , kind_{ kind }
        , guard_{ guard }
        , negate_{ negate }
        , loop_{ loop }
    {
        const unsigned int own = kind == CondKind::IfElse ? 2u : 1u;
        if (arms == 0u)
            arms = own;
        if (kind != CondKind::Switch && arms != own)
            EAGLE_THROW(std::invalid_argument, "CaptureConditional: IF and "
                "WHILE take one arm, IF/ELSE two; only SWITCH takes n");
        if (kind == CondKind::Switch && guard.count == nullptr)
            EAGLE_THROW(std::invalid_argument, "CaptureConditional: a SWITCH "
                "needs the device arm-index cell (count)");
        if (kind == CondKind::While) {
            EAGLE_ASSERT(loop.counter != nullptr,
                "CaptureConditional: a WHILE weave needs a device [remaining, "
                "ran] counter cell (nullptr given)");
            if (loop.cap == 0u)
                EAGLE_THROW(std::invalid_argument, "CaptureConditional: loopCap "
                    "must be >= 1 (a loop capped at 0 iterations can never run)");
        }
        if ((kind == CondKind::IfElse || kind == CondKind::Switch)
            && !conditionalArmsSupported())
            EAGLE_THROW(std::runtime_error, "CaptureConditional: IF/ELSE and "
                "SWITCH nodes need an NVIDIA driver >= 12.8; this driver is "
                "older (lower the chain to negated IFs)");
        armStreams_.reserve(arms);
        for (unsigned int k = 0; k < arms; ++k)
            armStreams_.emplace_back(/*nonBlocking=*/true);
        armGraphs_.assign(arms, nullptr);
    }

    /** @brief An IF weave (the body runs iff @p guard holds). */
    CaptureConditional(const cudaStream_t& origin, CountGuard guard)
        : CaptureConditional(origin, CondKind::If, guard)
    {
    }

    /** @brief A WHILE weave capped at @p loopCap iterations per replay over
     *  the caller-owned ``[remaining, ran]`` cell @p loopCounter. */
    CaptureConditional(const cudaStream_t& origin, CountGuard guard,
        unsigned int loopCap, unsigned int* loopCounter)
        : CaptureConditional(origin, CondKind::While, guard, 1, 0,
              conditional::LoopTerms{ loopCap, loopCounter })
    {
    }

    /** @brief Not copyable -- it owns the arm streams. */
    CaptureConditional(const CaptureConditional&)            = delete;
    /** @brief Not copyable -- it owns the arm streams. */
    CaptureConditional& operator=(const CaptureConditional&) = delete;
    /** @brief Not movable -- its lifetime is the begin/end scope. */
    CaptureConditional(CaptureConditional&&)                 = delete;
    /** @brief Not movable -- its lifetime is the begin/end scope. */
    CaptureConditional& operator=(CaptureConditional&&)      = delete;

    /** @brief True for a WHILE weave. */
    bool isLoop() const { return kind_ == CondKind::While; }

    /** @brief The node form. */
    CondKind kind() const { return kind_; }

    /** @brief Number of arms (bodies). */
    unsigned int arms() const { return static_cast<unsigned int>(armStreams_.size()); }

    /** @brief Arm @p k's pre-created stream, valid from construction: the
     *  origin a NESTED weave inside that arm must be constructed against,
     *  BEFORE capture begins. */
    cudaStream_t armStream(unsigned int k) const { return armStreams_.at(k).cuda(); }

    /** @brief Arm 0's stream (the body of an IF or WHILE). */
    cudaStream_t bodyStream() const { return armStream(0); }

    /**
     * @brief Weave the setter kernel + conditional node at the origin's
     *        current capture tip and open capture of every arm on its
     *        pre-created stream.
     *
     *  Call with a capture already in flight on @p origin (the stream
     *  passed to the constructor). Every CUDA API call here is a cold,
     *  once-per-build control-plane call, checked with
     *  ``EAGLE_CHECK_ALWAYS`` (never the debug-only ``EAGLE_CHECK``) --
     *  same convention as ``StreamCapturer`` / ``CaptureFork``.
     *
     *  @return Arm 0's stream: launch the region's work on it (e.g. via
     *          ``cp.cuda.ExternalStream`` on the Python side).
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

        cudaGraphConditionalNodeType type = cudaGraphCondTypeIf;
        if (kind_ == CondKind::While) {
            void* args[] = { &handle_, &guard_, &negate_, &loop_ };
            setterNode_ = conditional::detail::addSetterNode(capturedGraph,
                (void*)conditional::detail::setLoopHeadKernel<>, args, deps, numDeps);
            type = cudaGraphCondTypeWhile;
        } else if (kind_ == CondKind::Switch) {
            void* args[] = { &handle_, &guard_.count };
            setterNode_ = conditional::detail::addSetterNode(capturedGraph,
                (void*)conditional::detail::setSwitchKernel<>, args, deps, numDeps);
            // cudaGraphCondTypeSwitch (12.8 headers): the driver takes the
            // raw value from older headers too.
            type = static_cast<cudaGraphConditionalNodeType>(2);
        } else {
            setterNode_ = conditional::detail::addSetCondKernelNode(
                capturedGraph, handle_, guard_, deps, numDeps, negate_);
        }

        cudaGraphNodeParams cp = {};
        cp.type                = cudaGraphNodeTypeConditional;
        cp.conditional.handle  = handle_;
        cp.conditional.type    = type;
        cp.conditional.size    = arms();
        cudaGraphNode_t nCond;
        EAGLE_CHECK_ALWAYS(cuda::detail::graphAddNode(
            &nCond, capturedGraph, &setterNode_, 1, &cp));
        EAGLE_CHECK_ALWAYS(cuda::detail::streamUpdateCaptureDependencies(
            origin_, &nCond, 1, cudaStreamSetCaptureDependencies));

        for (unsigned int k = 0; k < arms(); ++k)
            armGraphs_[k] = cp.conditional.phGraph_out[k];
        // STOP-THE-LINE fix: the arm captures are NESTED capture scopes --
        // enter the process-wide depth counter before the first one opens,
        // so a failure part-way still leaves it balanced by end(). See
        // eagle::util::CaptureGuardState's doc for the incident this closes.
        captureScope_.emplace();
        open_ = true;
        for (unsigned int k = 0; k < arms(); ++k) {
            EAGLE_CHECK_ALWAYS(cudaStreamBeginCaptureToGraph(armStreams_[k].cuda(),
                armGraphs_[k], nullptr, nullptr, 0, cudaStreamCaptureModeThreadLocal));
            ++opened_;
        }
        return bodyStream();
    }

    /** @brief Close every arm capture. Idempotent -- calling it again (or
     *  without a preceding ``begin()``) is a no-op, not an error.
     *
     *  Every arm must be non-empty (counted MID-capture, before anything
     *  eagle adds itself). A WHILE then launches the loop tail setter on
     *  the body stream, so it is captured as the body's last node -- after
     *  every node the caller launched, including a nested weave's
     *  conditional node -- before the body capture ends.
     *
     *  Exits the process-wide capture-depth scope (STOP-THE-LINE fix)
     *  UNCONDITIONALLY before the error check -- the nested capture
     *  sessions are over the moment ``cudaStreamEndCapture`` returns,
     *  whether or not it reports success. */
    void end()
    {
        if (!open_)
            return;

        // A conditional node whose body is empty never completes when it
        // fires (an IF hangs the stream; a WHILE is an infinite loop by
        // construction), so refuse an arm that launched nothing. Counted
        // here, BEFORE the loop tail below is launched: the tail would
        // otherwise make every empty loop body look non-empty.
        // cudaGraphGetNodes on an arm graph is valid mid-capture (the
        // capture adds nodes to the graph as launches are recorded).
        cudaError_t err   = cudaSuccess;
        bool emptyArm     = false;
        for (unsigned int k = 0; k < opened_ && err == cudaSuccess; ++k) {
            std::size_t nodes = 0;
            err      = cudaGraphGetNodes(armGraphs_[k], nullptr, &nodes);
            emptyArm = emptyArm || nodes == 0;
        }
        if (err == cudaSuccess && !emptyArm && isLoop() && opened_ == 1) {
            // Captured on the body stream after everything the caller
            // launched there, so it depends on the whole body: the loop's
            // continue/stop decision is made once the iteration is done.
            conditional::detail::setLoopTailKernel<>
                <<<1, 1, 0, bodyStream()>>>(handle_, guard_, negate_, loop_);
            err = tailNode_();
        }

        for (unsigned int k = 0; k < opened_; ++k) {
            cudaGraph_t out              = nullptr;
            const cudaError_t captureErr = cudaStreamEndCapture(armStreams_[k].cuda(), &out);
            if (err == cudaSuccess)
                err = captureErr;
        }
        open_   = false;
        opened_ = 0;
        captureScope_.reset();
        EAGLE_CHECK_ALWAYS(err);
        if (emptyArm)
            EAGLE_THROW(std::invalid_argument, isLoop()
                ? "CaptureConditional::end: the loop body launched no work; "
                  "a WHILE node with an empty body is an infinite loop -- the "
                  "step must launch at least one kernel"
                : "CaptureConditional::end: a guarded region launched no "
                  "work; a conditional arm with an empty body never completes "
                  "when it fires -- every arm must launch at least one kernel");
    }

    /** @brief The node handles of arm @p k's body graph (valid from
     *  ``begin()``; after ``end()`` a WHILE body ends with eagle's tail
     *  setter) -- the per-region node set a graph patch attributes body
     *  launches by. */
    std::vector<cudaGraphNode_t> bodyNodes(unsigned int k = 0) const
    {
        const cudaGraph_t g = armGraphs_.at(k);
        if (g == nullptr)
            return {};
        std::size_t n = 0;
        EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(g, nullptr, &n));
        std::vector<cudaGraphNode_t> nodes(n);
        if (n != 0)
            EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(g, nodes.data(), &n));
        return nodes;
    }

    /** @brief eagle's setter nodes: the head (from ``begin()``) and, for a
     *  WHILE, the tail (from ``end()``) -- the nodes that read the guard
     *  and loop cells. */
    std::vector<cudaGraphNode_t> setterNodes() const
    {
        std::vector<cudaGraphNode_t> nodes;
        if (setterNode_ != nullptr)
            nodes.push_back(setterNode_);
        if (tail_ != nullptr)
            nodes.push_back(tail_);
        return nodes;
    }

    /** @brief Close the arm captures automatically if the caller did not.
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
    /** @brief Record the tail setter just captured on the body stream (the
     *  stream's sole capture dependency). */
    cudaError_t tailNode_()
    {
        cudaStreamCaptureStatus status;
        unsigned long long id             = 0;
        cudaGraph_t graph                 = nullptr;
        const cudaGraphNode_t* deps       = nullptr;
        const cudaGraphEdgeData* edgeData = nullptr;
        std::size_t numDeps               = 0;
        const cudaError_t err = cuda::detail::streamGetCaptureInfo(bodyStream(),
            &status, &id, &graph, &deps, &edgeData, &numDeps);
        if (err == cudaSuccess && numDeps == 1)
            tail_ = deps[0];
        return err;
    }

    cudaStream_t origin_;
    CondKind kind_;
    CountGuard guard_;
    unsigned int negate_;
    conditional::LoopTerms loop_;
    std::vector<Stream> armStreams_;
    std::vector<cudaGraph_t> armGraphs_;
    cudaGraphConditionalHandle handle_ = 0;    // set by begin(); read by the tail launch
    cudaGraphNode_t setterNode_        = nullptr;
    cudaGraphNode_t tail_              = nullptr;
    unsigned int opened_               = 0;    // arm captures begun
    bool open_                         = false;

    /** @brief Engaged between ``begin()`` and the matching ``end()`` -- see
     *  ``eagle::util::CaptureGuardState`` (STOP-THE-LINE fix). */
    std::optional<util::CaptureScope> captureScope_;
};

} // namespace cuda
} // namespace eagle

#endif
