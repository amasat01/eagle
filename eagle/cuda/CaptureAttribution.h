// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include "eagle/cuda/detail/RuntimeCompat.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/throw.h"

#ifndef EAGLE_CPU_ONLY

#include <cstddef>
#include <vector>

#include <cuda_runtime.h>

/* A facility-scoped floor, mirroring
 * eagle/conditional/ConditionalGroup.h's own `#error` guard exactly (see that
 * file's comment for the "header-local #error, not a project-wide gate"
 * rationale: non-consumers of THIS header see zero
 * configure or umbrella-header impact -- it is not pulled into eagle/cuda.h
 * or eagle/eagle.h, named directly like CaptureFork.h/CaptureConditional.h).
 *
 * captureSnapshotNodes needs cudaStreamGetCaptureInfo_v3 (CUDA >= 12.3,
 * 12030) -- the same call CaptureConditional::begin() already uses, proven
 * mid-capture-safe on this driver by probing the call mid-capture
 * (node count strictly grows, one per launch, as capture proceeds).
 * setNodeEnabled's cudaGraphNodeSetEnabled needs only CUDA >= 11.6 (11060,
 * per cuda_runtime_api.h's own `#if __CUDART_API_VERSION >= 11060` guard on
 * the declaration) -- weaker than 12.3, so it is folded into the SAME
 * file-level guard rather than split into two: one #error covering the
 * whole file is simpler than two, and 12.3 already subsumes 11.6.
 * isNodeToggleable (cudaGraphNodeGetType) carries no floor of its own.
 *
 * eagle_core.cu's compiled binding already impose a 12.3 floor as a whole
 * (it unconditionally includes CaptureConditional.h, which transitively
 * hits ConditionalGroup.h's #error), so this guard adds NO NEW effective
 * floor to the shipped Python extension -- it exists so this header stays
 * correct and self-contained for any FUTURE consumer that includes it
 * without also pulling in the conditional facility.
 */
#if defined(CUDART_VERSION) && CUDART_VERSION < 12030
#error "eagle::cuda capture-attribution facility (captureSnapshotNodes / isNodeToggleable / setNodeEnabled): requires CUDA >= 12.3 (CUDART_VERSION >= 12030) for cudaStreamGetCaptureInfo_v3; this toolkit's CUDART_VERSION is older."
#endif

namespace eagle {
namespace cuda {

/**
 * @brief Snapshot the node handles of the graph currently being captured
 *        into, as seen from @p stream: the mechanism
 *        ``GraphPipeline.build()`` uses for per-member node attribution.
 *
 * Queries ``cudaStreamGetCaptureInfo_v3`` for the in-construction
 * ``cudaGraph_t`` and lists its current nodes via ``cudaGraphGetNodes`` --
 * both calls are legal WHILE a ThreadLocal-mode capture is active (proven on
 * this driver by probing the call mid-capture: the observed node count
 * strictly grows, one per launch, as more kernels are captured). Every
 * stream touched by one active capture session -- the origin AND any
 * ``CaptureFork`` branch pulled in via its fork event -- reports the SAME
 * underlying ``cudaGraph_t`` (the mechanism ``CaptureConditional::begin()``
 * already relies on for a fork-branch origin), so
 * calling this immediately before and after one pipeline member's launches
 * -- on whichever stream that member captures on -- and taking the SET
 * DIFFERENCE of the two snapshots yields exactly the nodes that member
 * contributed, with no cross-member ambiguity: capture itself is
 * single-threaded host-side sequencing, so one member's launches fully
 * enter the graph before the next member's capture begins even when the
 * members are siblings under a ``CaptureFork``.
 *
 * Deliberately node-HANDLE-only: never calls
 * ``cudaGraphKernelNodeGetParams`` (the templated-kernel trap
 * ``KernelNodeRecord``'s own comment documents, ``Graph.h``) -- attribution
 * only needs to know WHICH nodes a member owns and, via
 * ``isNodeToggleable``, their TYPE; never their launch parameters.
 *
 * @param[in] stream  A stream that is part of an ACTIVE Global-mode
 *                     capture (the pipeline's main stream, or a
 *                     ``CaptureFork`` branch).
 * @return Node handles in ``cudaGraphGetNodes`` order (not necessarily
 *         capture order; stable within one capture session).
 * @throws aether::Error  If the capture-info or node-listing call
 *         fails, or the node count fails to stabilize (see below).
 * @throws std::runtime_error    If @p stream is not actively capturing.
 *
 * @par STOP-THE-LINE hardening (post-incident, round 2)
 * The naive count-then-fill idiom (query count with ``nodes=nullptr``,
 * allocate exactly that many, fill) trusts that the count is still exact
 * by the second call. CUDA's own docs describe only the OVER-capacity case
 * (``numNodes`` higher than actual: excess entries set to NULL, actual
 * count returned) and are silent on UNDER-capacity. Measured directly on
 * this driver (deliberately under-reporting capacity by 5 against a
 * 300-node graph): the fill call returns ``cudaSuccess`` and reports back
 * EXACTLY the requested capacity, not the true (larger) count -- i.e. it
 * does not overflow the buffer, but it also gives no positive signal that
 * truncation happened. To close that gap without trusting either call in
 * isolation, this re-queries the count (``nullptr``) immediately AFTER
 * the fill and compares against what was actually allocated: if the graph
 * grew during the fill, retry with the new, larger size. Bounded so a
 * pathologically still-growing graph cannot loop forever (never observed;
 * capture is single-threaded host-side sequencing per call in every
 * consumer this facility has).
 */
inline std::vector<cudaGraphNode_t> captureSnapshotNodes(cudaStream_t stream)
{
    cudaStreamCaptureStatus status;
    unsigned long long capId          = 0;
    cudaGraph_t capturedGraph         = nullptr;
    const cudaGraphNode_t* deps       = nullptr;
    const cudaGraphEdgeData* edgeData = nullptr;
    std::size_t numDeps               = 0;
    EAGLE_CHECK_ALWAYS(detail::streamGetCaptureInfo(stream, &status, &capId,
        &capturedGraph, &deps, &edgeData, &numDeps));
    EAGLE_ASSERT(status == cudaStreamCaptureStatusActive,
        "captureSnapshotNodes: stream is not actively capturing");

    std::size_t numNodes = 0;
    EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(capturedGraph, nullptr, &numNodes));

    std::vector<cudaGraphNode_t> nodes;
    static constexpr int kMaxAttempts = 8;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        if (numNodes == 0)
            return nodes;

        const std::size_t capacity = numNodes;
        nodes.assign(capacity, nullptr);
        EAGLE_CHECK_ALWAYS(
            cudaGraphGetNodes(capturedGraph, nodes.data(), &numNodes));

        std::size_t recheck = 0;
        EAGLE_CHECK_ALWAYS(
            cudaGraphGetNodes(capturedGraph, nullptr, &recheck));

        if (recheck <= capacity) {
            nodes.resize(capacity);
            return nodes;
        }
        /* Grew between the count query and now: retry with the larger,
         * freshly-observed size. `nodes` is fully reassigned (`.assign`)
         * next iteration, so no stale entries leak through. */
        numNodes = recheck;
    }
    EAGLE_THROW(std::runtime_error,
        "captureSnapshotNodes: node count did not stabilize after "
        + std::to_string(kMaxAttempts)
        + " attempts -- the graph is growing faster than it can be "
          "queried, which should be impossible for a single-threaded "
          "host-side capture session");
    return nodes; // unreachable (EAGLE_THROW always throws); silences a
                  // missing-return diagnostic on toolchains that do not
                  // propagate [[noreturn]] through the macro-wrapped call.
}

/**
 * @brief Whether @p node's type is one ``setNodeEnabled`` may safely
 *        toggle.
 *
 * ``cudaGraphNodeSetEnabled``'s own documentation (CUDA 12.6
 * ``cuda_runtime_api.h``) states that only kernel, memset and memcpy nodes
 * are currently supported -- this predicate names exactly that set, so a
 * member whose captured node set includes anything else (most pertinently
 * a ``CaptureConditional`` IF node, ``cudaGraphNodeTypeConditional`` --
 * the conditional/skippable facility) is reported non-toggleable at
 * ``GraphPipeline.build()`` time rather than failing later, deep inside a
 * raw driver call at the first toggle attempt.
 */
inline bool isNodeToggleable(cudaGraphNode_t node)
{
    cudaGraphNodeType type;
    EAGLE_CHECK_ALWAYS(cudaGraphNodeGetType(node, &type));
    return type == cudaGraphNodeTypeKernel || type == cudaGraphNodeTypeMemcpy
        || type == cudaGraphNodeTypeMemset;
}

/**
 * @brief Enable or disable one node of an INSTANTIATED exec graph
 *        (the member-enable facility's "mode=enabled" node toggling).
 *
 * A disabled node becomes a driver-level no-op on every subsequent
 * ``cudaGraphLaunch`` of @p exec -- NO recapture, NO change to graph
 * structure. Legal BETWEEN replays; toggling mid-replay is undefined and
 * ordering it correctly against a replay is the caller's responsibility
 * (e.g. via ``GraphPipeline.launch()``'s existing stream-ordering fix).
 *
 * @p node must be a handle from the SAME (non-executable) ``cudaGraph_t``
 * that @p exec was instantiated from -- e.g. one returned by
 * ``captureSnapshotNodes`` at build time, before instantiation. This holds
 * for ``eagle::cuda::Graph``'s captured-graph constructor: ``launcher()``
 * instantiates the adopted graph verbatim (no clone), so node handles
 * captured mid-build remain valid identifiers into it.
 *
 * @throws aether::Error  If the node's type does not support
 *         enable/disable, the node is not part of this exec graph, or any
 *         other driver rejection (call ``isNodeToggleable`` ahead of time
 *         to check the type without risking this call).
 */
inline void setNodeEnabled(
    cudaGraphExec_t exec, cudaGraphNode_t node, bool enabled)
{
    EAGLE_CHECK_ALWAYS(cudaGraphNodeSetEnabled(exec, node, enabled ? 1u : 0u));
}

} // namespace cuda
} // namespace eagle

#endif // EAGLE_CPU_ONLY
