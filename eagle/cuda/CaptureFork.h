// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cuda/Stream.h"

#ifndef EAGLE_CPU_ONLY

#include <cstddef>
#include <vector>

namespace eagle {
namespace cuda {

/**
 * @brief Fork an in-flight stream capture into mutually independent sibling
 *        branches, then join them back.
 *
 * A single ``StreamCapturer`` session records everything launched on its
 * origin stream as one **linear chain** of graph nodes: each node depends on
 * the one before it, because that is what stream order means. Work that is
 * genuinely independent is therefore recorded as if it were sequential, and
 * the resulting graph forbids the scheduler from ever overlapping it.
 *
 * ``CaptureFork`` restores the missing structure. Between ``fork()`` and
 * ``join()`` each branch stream is its own capture-order chain, so the
 * captured DAG contains sibling nodes with **no edge between them**:
 *
 * @code
 *   Stream origin(true);
 *   CaptureFork fork(origin.cuda(), 2);   // pre-capture: streams + events
 *
 *   StreamCapturer capturer(origin.cuda());
 *   capturer.begin();
 *   stepA<<<..., origin.cuda()>>>(...);   // origin chain
 *
 *   fork.fork();
 *   stepB1<<<..., fork.branch(0)>>>(...); // sibling of stepB2
 *   stepB2<<<..., fork.branch(1)>>>(...); // sibling of stepB1
 *   fork.join();
 *
 *   stepC<<<..., origin.cuda()>>>(...);   // depends on BOTH branches
 *   cudaGraph_t g = capturer.end();
 * @endcode
 *
 * @par THE CONTRACT — what a fork does and does not promise
 *
 * Forking grants the scheduler **PERMISSION** to co-execute the branches. It
 * is never an **obligation**. Running every branch one after another, in any
 * order, is always a conforming schedule — a device with one free SM, a
 * device already saturated by another branch, and a future driver that simply
 * chooses not to overlap are all behaving correctly. Code must never depend on
 * co-execution actually happening, and no caller, test, or benchmark may treat
 * serialized execution as a defect.
 *
 * Consequently the **relative order among branches is unobservable**. There is
 * no ordering between sibling branches, so nothing may rely on one branch's
 * effects being visible to another, and nothing may rely on a particular
 * interleaving. The only ordering guarantees are: everything before ``fork()``
 * happens-before every branch, and every branch happens-before everything
 * after ``join()``.
 *
 * **The caller asserts mutual independence.** Branches must not race: no
 * branch may write memory another branch reads or writes. This class cannot
 * check that and does not try. Violating it produces a graph whose results
 * depend on the schedule — which, per the paragraph above, is free to change.
 *
 * Any benefit is bounded by how much of the device a single branch already
 * occupies. State that boundary **parametrically** — in terms of the device's
 * SM count, the per-branch occupancy, and the branch count (e.g. a branch that
 * already fills every SM leaves nothing for a sibling; roughly
 * ``SM_count / branch_count`` SMs are available per branch when all branches
 * are resident) — never as a constant measured on one GPU. Numbers from one
 * card are evidence about that card, not a property of this primitive.
 *
 * The **mechanism is deliberately private and swappable**. Events are how the
 * fork is expressed today; that is an implementation detail, not part of the
 * contract. The contract above ships forever; the events may not.
 *
 * @par Lifecycle
 *
 * Construction is the only phase that acquires resources: all branch streams
 * and all events are created up front, **before** the capture begins, because
 * resource creation is illegal mid-capture. No phase ever allocates device
 * memory. ``fork()`` and ``join()`` are pure capture-time signalling.
 *
 * ``join()`` is **idempotent** — calling it again is a no-op, not an error —
 * and the destructor joins automatically if the caller did not, so a scoped
 * ``CaptureFork`` is correct by construction. An unjoined branch is not a
 * benign leak: it makes ``cudaStreamEndCapture`` fail with
 * ``cudaErrorStreamCaptureUnjoined`` and discard the whole graph.
 *
 * @note Available only when ``EAGLE_CPU_ONLY`` is not defined.
 * @see StreamCapturer — owns the capture session this class forks within.
 */
class CaptureFork {
public:
    /** @brief Create the branch streams and events for a later fork.
     *
     *  Must be called **before** ``StreamCapturer::begin()``: stream and event
     *  creation is not permitted while a capture is in flight. Branch streams
     *  are non-blocking, so they carry no implicit dependency on the legacy
     *  default stream; events disable timing, since they are used purely for
     *  ordering.
     *
     *  @param[in] origin        Stream being captured — the one that forks and
     *                           that the branches rejoin.
     *  @param[in] branchCount   Number of independent branches.
     *  @throws aether::Error  If stream or event creation fails.
     */
    CaptureFork(const cudaStream_t& origin, const std::size_t& branchCount)
        : origin_{ origin }
        , forkEvent_{ cudaEventDisableTiming }
    {
        branches_.reserve(branchCount);
        joinEvents_.reserve(branchCount);
        for (std::size_t i = 0; i < branchCount; i++) {
            branches_.emplace_back(/*nonBlocking=*/true);
            joinEvents_.emplace_back(cudaEventDisableTiming);
        }
    }

    /** @brief Not copyable — it owns streams and events. */
    CaptureFork(const CaptureFork&)            = delete;
    /** @brief Not copyable — it owns streams and events. */
    CaptureFork& operator=(const CaptureFork&) = delete;
    /** @brief Not movable — its lifetime is the fork/join scope. */
    CaptureFork(CaptureFork&&)                 = delete;
    /** @brief Not movable — its lifetime is the fork/join scope. */
    CaptureFork& operator=(CaptureFork&&)      = delete;

    /** @brief Open the fork: every branch becomes a sibling of every other.
     *
     *  Call at the fork point, with a capture in flight on the origin stream.
     *  Records the fork event on the origin and makes each branch wait on it,
     *  which both pulls the branches into the capture and makes every node
     *  recorded so far a predecessor of every branch. Calling twice is a no-op.
     *
     *  @throws aether::Error  If the record or a wait fails.
     */
    void fork()
    {
        if (forked_)
            return;
        EAGLE_CHECK_ALWAYS(cudaEventRecord(forkEvent_.cuda(), origin_));
        for (auto& branch : branches_)
            EAGLE_CHECK_ALWAYS(
                cudaStreamWaitEvent(branch.cuda(), forkEvent_.cuda(), 0));
        forked_ = true;
        joined_ = false;
    }

    /** @brief Close the fork: the origin waits for every branch.
     *
     *  Must be called before ``StreamCapturer::end()`` — an unjoined branch
     *  makes the capture fail wholesale. Each branch records its join event
     *  and the origin waits on all of them, so every node launched after the
     *  join depends on all branches.
     *
     *  **Idempotent**: calling it twice, or without a preceding ``fork()``, is
     *  a no-op rather than an error.
     *
     *  @throws aether::Error  If a record or wait fails.
     */
    void join()
    {
        if (!forked_ || joined_)
            return;
        for (std::size_t i = 0; i < branches_.size(); i++)
            EAGLE_CHECK_ALWAYS(
                cudaEventRecord(joinEvents_[i].cuda(), branches_[i].cuda()));
        for (auto& event : joinEvents_)
            EAGLE_CHECK_ALWAYS(
                cudaStreamWaitEvent(origin_, event.cuda(), 0));
        joined_ = true;
    }

    /** @brief Stream to launch branch ``index`` on.
     *
     *  Only meaningful between ``fork()`` and ``join()``; outside that window
     *  it is an ordinary idle stream.
     *
     *  @param[in] index  Branch index in ``[0, size())``.
     *  @return The branch's raw stream handle, for use as a launch argument.
     */
    const cudaStream_t& branch(const std::size_t& index) const
    {
        return branches_[index].cuda();
    }

    /** @brief Number of branches. */
    std::size_t size() const { return branches_.size(); }

    /** @brief True once ``fork()`` has run and ``join()`` has not. */
    bool forked() const { return forked_ && !joined_; }

    /** @brief The origin stream this fork branches from and rejoins. */
    const cudaStream_t& origin() const { return origin_; }

    /** @brief Join automatically if the caller did not.
     *
     *  Makes scoped use correct by construction: destroying an open fork
     *  before ``StreamCapturer::end()`` closes it rather than poisoning the
     *  capture. Never throws — a destructor that escaped during unwinding
     *  would terminate, and a join failure here is already being reported by
     *  the capture that is about to fail.
     */
    ~CaptureFork()
    {
        try {
            join();
        } catch (...) {
            /* swallowed deliberately: see the note above */
        }
    }

protected:
    cudaStream_t origin_;
    Event forkEvent_;
    std::vector<Stream> branches_;
    std::vector<Event> joinEvents_;
    bool forked_ = false;
    bool joined_ = false;
};

} // namespace cuda
} // namespace eagle

#endif
