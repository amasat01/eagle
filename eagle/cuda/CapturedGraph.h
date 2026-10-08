// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/CaptureGuard.h"

#ifndef EAGLE_CPU_ONLY

#include <utility>
#include <vector>

#include <cuda_runtime.h>

namespace eagle {
namespace cuda {

/**
 * @brief Move-only RAII owner of a captured ``cudaGraph_t`` plus
 *        per-kernel ``idealBlockSize`` caps.
 *
 * Mirrors the ownership shape of the rest of the eagle graph layer
 * (``Graph``, ``Launcher``, ``Graph::Storage``): non-copyable;
 * move-only; destructor releases the wrapped ``cudaGraph_t``. The
 * type itself enforces that constructing a ``CapturedGraph`` from a
 * borrowed handle is a programming error — the handle handed in
 * must always represent a transfer of ownership.
 *
 * Pair with ``Graph::addNode(CapturedGraph)`` (the owned-input
 * overload): the by-value parameter is move-constructed from the
 * argument, ``cudaGraphAddChildGraphNode`` clones into the parent,
 * and the parameter's destructor at the end of the addNode call
 * releases the source. The ``Graph::addNode(cudaGraph_t)`` overload
 * — which does *not* take ownership — is the right path for
 * borrowed handles, e.g. a raw ``cudaGraph_t`` exposed by a
 * consumer-owned subgraph builder.
 *
 * ``idealBlockSizes[k]`` is the cap for the k-th kernel in
 * ``cudaGraphGetNodes`` order. ``0`` means grid-only mutation.
 * Vector size must equal the kernel count, or be empty (= grid-only
 * for every kernel).
 */
class CapturedGraph {
public:
    /** @brief Default constructor: empty wrapper (no handle, no caps). */
    CapturedGraph() = default;

    /** @brief Take ownership of an existing ``cudaGraph_t`` and
     *  attach a per-kernel ``idealBlockSize`` table.
     *
     *  Conventional uses: wrap the result of ``StreamCapturer::end()``,
     *  ``cudaGraphCreate``, or ``cudaGraphClone``. Do NOT pass a
     *  borrowed handle — that violates the type's ownership
     *  contract and will lead to a double-free when the source's
     *  original owner also calls ``cudaGraphDestroy``. */
    explicit CapturedGraph(
        cudaGraph_t graph, std::vector<idx_t> idealBlockSizes = {}) noexcept
        : graph_{ graph }
        , idealBlockSizes_{ std::move(idealBlockSizes) }
    {
    }

    /** @brief Copy is forbidden — ownership is exclusive. */
    CapturedGraph(const CapturedGraph&)            = delete;
    CapturedGraph& operator=(const CapturedGraph&) = delete;

    /** @brief Move constructor. Source becomes empty. */
    CapturedGraph(CapturedGraph&& other) noexcept
        : graph_{ std::exchange(other.graph_, nullptr) }
        , idealBlockSizes_{ std::move(other.idealBlockSizes_) }
    {
    }

    /** @brief Move assignment. Releases any handle currently owned
     *  before taking the moved-from one.
     *
     *  STOP-THE-LINE fix: the release is handed to
     *  ``eagle::util::CaptureGuardState::destroyOrDefer`` rather than called
     *  directly -- see ``Stream::destroy_()``'s comment for the full
     *  rationale (a ``cudaGraphDestroy`` firing mid-capture, e.g. from a
     *  Python-GC-triggered move/destroy of an unrelated ``CapturedGraph``,
     *  invalidates whatever capture happens to be active elsewhere in the
     *  process at that moment). Timing is unchanged when no capture is
     *  active (runs immediately, same as before). */
    CapturedGraph& operator=(CapturedGraph&& other) noexcept
    {
        if (this != &other) {
            if (graph_ != nullptr) {
                const cudaGraph_t handle = graph_;
                ::eagle::util::CaptureGuardState::instance().destroyOrDefer(
                    [handle]() { cudaGraphDestroy(handle); });
            }
            graph_           = std::exchange(other.graph_, nullptr);
            idealBlockSizes_ = std::move(other.idealBlockSizes_);
        }
        return *this;
    }

    /** @brief Destructor releases the owned ``cudaGraph_t`` if any.
     *  STOP-THE-LINE fix: see the move-assignment operator's note above --
     *  same deferred-if-capturing release. */
    ~CapturedGraph()
    {
        if (graph_ != nullptr) {
            const cudaGraph_t handle = graph_;
            ::eagle::util::CaptureGuardState::instance().destroyOrDefer(
                [handle]() { cudaGraphDestroy(handle); });
        }
    }

    /** @brief Read-only access to the wrapped ``cudaGraph_t``.
     *  Borrowed view — caller must not destroy. */
    cudaGraph_t graph() const noexcept { return graph_; }

    /** @brief Read-only access to the per-kernel ``idealBlockSize`` caps. */
    const std::vector<idx_t>& idealBlockSizes() const noexcept
    {
        return idealBlockSizes_;
    }

    /** @brief Relinquish ownership of the wrapped handle and return
     *  it. After the call ``*this`` is empty (destructor will not
     *  destroy anything). Use only when ownership is genuinely
     *  transferred to another owner. */
    cudaGraph_t release() noexcept
    {
        return std::exchange(graph_, nullptr);
    }

    /** @brief Whether the wrapper currently owns a handle. */
    explicit operator bool() const noexcept { return graph_ != nullptr; }

private:
    cudaGraph_t graph_ = nullptr;
    std::vector<idx_t> idealBlockSizes_;
};

} // namespace cuda
} // namespace eagle

#endif // EAGLE_CPU_ONLY
