// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/ScratchArena.h"
#include "eagle/conditional/CountGuard.h"
#include "eagle/cpu/Graph.h"
#include "eagle/cuda.h"
#include "eagle/cuda/detail/RuntimeCompat.h"
#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/throw.h"

#include <memory>
#include <vector>

namespace eagle {
namespace conditional {

#ifndef EAGLE_CPU_ONLY

/*
 * CUDA floor for the conditional-node facility,
 * scoped to THIS header (never root CMake, never the umbrella eagle.h): the
 * conditional-node driver APIs this file and its capture-weave sibling
 * (eagle/cuda/CaptureConditional.h, which unconditionally includes this
 * header above) call -- cudaGraphConditionalHandleCreate,
 * cudaGraphSetConditional, cudaGraphNodeTypeConditional/cudaGraphCondTypeIf,
 * cudaStreamGetCaptureInfo_v3 -- require CUDA >= 12.3. A TU that only
 * includes this header (never CaptureConditional.h) still needs the check,
 * so it lives here rather than only in the capture-weave sibling; placing it
 * ALSO in CaptureConditional.h would be redundant, since that header cannot
 * be included without this one pulling this exact check in first (single
 * source of truth for the message). Non-consumers (downstream trajectory libraries, any
 * TU that never names this header) see zero configure or umbrella-header
 * impact -- this is a header-local #error, not a project-wide gate.
 */
#if defined(CUDART_VERSION) && CUDART_VERSION < 12030
#error "eagle::conditional (ConditionalGroup / CaptureConditional): the conditional-node facility requires CUDA >= 12.3 (CUDART_VERSION >= 12030) for cudaGraphConditionalHandleCreate/cudaGraphSetConditional/cudaGraphNodeTypeConditional/cudaStreamGetCaptureInfo_v3; this toolkit's CUDART_VERSION is older."
#endif

namespace detail {

/**
 * @brief eagle-owned device predicate for ``CountGuard``: sets a
 *        conditional handle's branch decision from a device-resident
 *        compare -- never a host round-trip.
 *
 * Templated (rather than a plain free ``__global__`` function) purely so
 * this header stays multi-TU-safe: more than one translation unit in the
 * same link including it would otherwise redefine the same ``__global__``
 * symbol. Function templates merge across TUs the way the rest of eagle's
 * kernel headers already rely on (``eagle::reduce::detail::reduceOnce``).
 * Instantiated as ``setCountGuardKernel<>`` (the sole, default, argument).
 */
template<int = 0>
AETHER_KERNEL() void setCountGuardKernel(cudaGraphConditionalHandle handle,
    const unsigned int* count, const unsigned int* baseline)
{
    const unsigned int base = (baseline != nullptr) ? *baseline : 0u;
    cudaGraphSetConditional(handle, (*count != base) ? 1u : 0u);
}

/**
 * @brief Add the ``setCountGuardKernel`` node evaluating @p guard into
 *        @p graph, depending on @p deps (``numDeps == 0`` permits
 *        ``deps == nullptr``). Returns the new node -- the IF node's sole
 *        dependency.
 *
 * A cold, once-per-build control-plane call: checked with
 * ``EAGLE_CHECK_ALWAYS`` (never the debug-only ``EAGLE_CHECK``), exactly the
 * ``StreamCapturer`` / ``CaptureFork`` convention -- a swallowed failure
 * here would silently leave the branch decision undefined.
 */
inline cudaGraphNode_t addSetCondKernelNode(cudaGraph_t graph,
    cudaGraphConditionalHandle handle, const CountGuard& guard,
    const cudaGraphNode_t* deps, std::size_t numDeps)
{
    cudaGraphNode_t node;
    void* args[] = { (void*)&handle, (void*)&guard.count,
        (void*)&guard.baseline };
    cudaKernelNodeParams kp = {};
    kp.func         = (void*)setCountGuardKernel<>;
    kp.gridDim      = { 1, 1, 1 };
    kp.blockDim     = { 1, 1, 1 };
    kp.kernelParams = args;
    EAGLE_CHECK_ALWAYS(
        cudaGraphAddKernelNode(&node, graph, deps, numDeps, &kp));
    return node;
}

/**
 * @brief Loop head for a WHILE conditional node: reset the eagle-owned
 *        ``[remaining, ran]`` cell for this replay and decide whether the
 *        body runs at all.
 *
 * ``counter[0] = cap`` (iterations still allowed), ``counter[1] = 0``
 * (iterations completed -- the device-visible iteration index a body
 * kernel may read), ``handle = guard && cap > 0``. The reset lives HERE, in
 * a kernel the graph already needs, so no memset node and no per-launch
 * ``cudaGraphCondAssignDefault`` reset is ever woven in. ``cap`` is a
 * kernel argument frozen at build like the graph shape: a bounded
 * iteration count is a structural guarantee, not device state.
 */
template<int = 0>
AETHER_KERNEL() void setLoopHeadKernel(cudaGraphConditionalHandle handle,
    const unsigned int* count, const unsigned int* baseline,
    unsigned int cap, unsigned int* counter)
{
    const unsigned int base = (baseline != nullptr) ? *baseline : 0u;
    counter[0] = cap;
    counter[1] = 0u;
    cudaGraphSetConditional(handle, (*count != base && cap != 0u) ? 1u : 0u);
}

/**
 * @brief Loop tail for a WHILE conditional node: the body's LAST node.
 *        One iteration completed; the loop continues iff the guard still
 *        holds AND iterations remain: ``handle = guard && --remaining > 0``,
 *        ``++ran``.
 */
template<int = 0>
AETHER_KERNEL() void setLoopTailKernel(cudaGraphConditionalHandle handle,
    const unsigned int* count, const unsigned int* baseline,
    unsigned int* counter)
{
    const unsigned int base      = (baseline != nullptr) ? *baseline : 0u;
    const unsigned int remaining = --counter[0];
    ++counter[1];
    cudaGraphSetConditional(handle,
        (*count != base && remaining != 0u) ? 1u : 0u);
}

/**
 * @brief Add the ``setLoopHeadKernel`` node for a WHILE node into @p graph
 *        (the loop sibling of ``addSetCondKernelNode``; same checking
 *        convention). Returns the new node -- the WHILE node's sole
 *        dependency.
 */
inline cudaGraphNode_t addSetLoopHeadKernelNode(cudaGraph_t graph,
    cudaGraphConditionalHandle handle, const CountGuard& guard,
    unsigned int cap, unsigned int* counter, const cudaGraphNode_t* deps,
    std::size_t numDeps)
{
    cudaGraphNode_t node;
    void* args[] = { (void*)&handle, (void*)&guard.count,
        (void*)&guard.baseline, (void*)&cap, (void*)&counter };
    cudaKernelNodeParams kp = {};
    kp.func         = (void*)setLoopHeadKernel<>;
    kp.gridDim      = { 1, 1, 1 };
    kp.blockDim     = { 1, 1, 1 };
    kp.kernelParams = args;
    EAGLE_CHECK_ALWAYS(
        cudaGraphAddKernelNode(&node, graph, deps, numDeps, &kp));
    return node;
}

} // namespace detail

#endif // EAGLE_CPU_ONLY

/**
 * @brief A skippable graph region as ONE native node -- the C++ exemplar of
 *        the conditional-node facility (dual-face: CUDA IF-node build /
 *        CPU host-read execute).
 *
 * Conforms to the EXISTING native-node protocol on both faces
 * (``cuda::NativeNode`` / ``cpu::NativeNode``), so it is contributed to a
 * parent graph exactly like ``ReductionNode`` et al.:
 * ``graph.addNative(ConditionalGroup{guard}, deps)``. Populate ``body()``
 * with the guarded region's own work (``addKernelNode`` / ``addNode`` /
 * nested ``addNative``) BEFORE moving the group into ``addNative`` -- once
 * moved, the local variable is spent.
 *
 * The conditional IS one node in CUDA's node-graph model; the guarded
 * region's many nodes live inside its body graph, entered as that node's
 * sole child. This is why ``buildInto`` (like every other native node)
 * returns exactly one index.
 *
 * @par Body ownership
 * The body is held via ``std::shared_ptr`` -- a by-value ``cuda::Graph`` /
 * ``cpu::Graph`` member would silently break ``addNative``'s registration:
 * ``Graph`` is move-only, but ``addNative`` stores its closure in a
 * ``std::function``, which requires its target to be CopyConstructible and
 * therefore REJECTS a move-only capture. ``shared_ptr`` keeps
 * ``ConditionalGroup`` itself copyable (and hence usable with
 * ``addNative``) while the body graph itself stays exactly where it was
 * built.
 *
 * @par Scratch
 * The body owns its OWN ``ScratchArena`` (it is a full, independent
 * ``Graph`` and reuses the peak-not-sum policy internally, for its own
 * nested natives only). ``reserveScratch`` on the OUTER protocol is
 * therefore a no-op: there is no peak-sharing between the guarded region
 * and the graph it is embedded in.
 *
 * @par Body kernels are not harvested
 * ``buildInto`` enters the body via a raw ``cudaGraphAddChildGraphNode``
 * call, bypassing ``Graph::addNode``'s ``harvestKernels_`` step deliberately
 * -- so kernels inside a ``ConditionalGroup``'s body are exempt from
 * ``Launcher::setLogicalSize``; a resize of the OUTER graph never touches
 * them.
 *
 * @par Nesting fails loudly
 * A ``ConditionalGroup`` nested inside another's ``body()`` is a documented
 * v1 limit, not a checked-and-rejected input: once the nested group's own
 * ``buildInto`` (run from the outer's ``body().finalizeNatives()``) adds an
 * IF node into the outer body's graph, that graph itself becomes
 * conditional-bearing, and this class's own
 * ``cudaGraphAddChildGraphNode`` -- entering that now-conditional-bearing
 * body into ITS parent's IF node -- is rejected by the driver with
 * ``cudaErrorNotSupported`` (801). ``EAGLE_CHECK_ALWAYS`` turns that into a
 * thrown ``aether::Error`` at ``buildInto`` time: loud, not silent.
 * Lift on demand by populating the body with raw graph calls instead of a
 * nested group.
 *
 * @par Cross-mode use is out of scope (v1)
 * Using a ``ConditionalGroup`` as a ``cpu::Graph`` native node inside a
 * build that is NOT ``EAGLE_CPU_ONLY`` has no host body to run (this class
 * only maintains the CUDA body in that configuration) -- ``runHost``
 * ``EAGLE_ASSERT``s loudly rather than silently doing nothing when the
 * guard would have fired.
 *
 * @par Single-launcher only
 * A conditional-bearing ``cuda::Graph`` must be instantiated via
 * ``launcher()`` AT MOST ONCE: a second ``launcher()`` call now THROWS
 * ``aether::Error`` (``Launcher::instantiate_``
 * checks with ``EAGLE_CHECK_ALWAYS``, not the release-mode-silent
 * ``EAGLE_CHECK``) -- the underlying ``cudaGraphInstantiate`` returns
 * ``cudaErrorNotSupported`` (801: only one instantiation of a
 * conditional-bearing graph may exist at a time) and that failure is now
 * reported loudly instead of yielding a silently inert exec, while the first
 * launcher keeps working correctly. Verified empirically
 * (`ConditionalGroupTest.MultiLauncherFanOut`); not lifted in v1.
 *
 * @note CUDA floor: this header (and ``cuda::CaptureConditional``,
 *       which includes it) requires CUDA >= 12.3 -- an older toolkit fails
 *       to compile it with a clear ``#error``, not a silent miscompile.
 */
class ConditionalGroup {
public:
    explicit ConditionalGroup(CountGuard guard)
        : guard_{ guard }
#ifndef EAGLE_CPU_ONLY
        , body_{ std::make_shared<cuda::Graph>() }
#else
        , body_{ std::make_shared<cpu::Graph>() }
#endif
    {
    }

#ifndef EAGLE_CPU_ONLY

    /** @brief The internal CUDA body graph -- populate with
     *  ``addNative`` / ``addKernelNode`` / ``addNode`` before this group is
     *  itself moved into a parent ``Graph::addNative`` call. */
    cuda::Graph& body() { return *body_; }

    /**
     * @brief Phase 2 of the native-node build (see ``NativeNode`` /
     *        ``Graph::addNative``): finalize the body's own native nodes,
     *        then wire ``setCond -> IF -> body`` as ONE node into @p g,
     *        depending on @p deps. Returns the IF node's index via
     *        ``g.adoptNode`` -- the raw-handle currency bridge adds to
     *        ``cuda::Graph`` for exactly this kind of direct graph surgery.
     *
     *  ``body().finalizeNatives()`` is called explicitly here: unlike
     *  ``launcher()``, it is NOT auto-invoked, so a body populated purely
     *  through ``addNative`` would otherwise never capture its subgraph.
     */
    idx_t buildInto(cuda::Graph& g, const std::vector<idx_t>& deps)
    {
        body_->finalizeNatives();

        // A guarded body with no work is refused at build time, matching
        // CaptureConditional::end (where a truly empty IF body hangs the
        // stream when it fires). The body graph always holds its own empty
        // root node, so count only the nodes that do something.
        std::size_t bodyNodes = 0;
        EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(body_->graph(), nullptr, &bodyNodes));
        std::vector<cudaGraphNode_t> nodes(bodyNodes);
        if (bodyNodes > 0)
            EAGLE_CHECK_ALWAYS(
                cudaGraphGetNodes(body_->graph(), nodes.data(), &bodyNodes));
        std::size_t workNodes = 0;
        for (cudaGraphNode_t n : nodes) {
            cudaGraphNodeType type;
            EAGLE_CHECK_ALWAYS(cudaGraphNodeGetType(n, &type));
            if (type != cudaGraphNodeTypeEmpty)
                ++workNodes;
        }
        if (workNodes == 0)
            EAGLE_THROW(std::invalid_argument, "ConditionalGroup::buildInto: the "
                "guarded body holds no work; an IF node with an empty body "
                "never completes when it fires");

        // No cudaGraphCondAssignDefault: setCond writes the handle on every
        // replay before the IF node reads it (see CaptureConditional::begin).
        cudaGraphConditionalHandle handle;
        EAGLE_CHECK_ALWAYS(cudaGraphConditionalHandleCreate(
            &handle, g.graph(), 0, 0));

        std::vector<cudaGraphNode_t> depHandles;
        depHandles.reserve(deps.size());
        for (idx_t d : deps)
            depHandles.push_back(g.nodeHandle(d));

        cudaGraphNode_t nSet = detail::addSetCondKernelNode(g.graph(), handle,
            guard_, depHandles.empty() ? nullptr : depHandles.data(),
            depHandles.size());

        cudaGraphNodeParams cp = {};
        cp.type                = cudaGraphNodeTypeConditional;
        cp.conditional.handle  = handle;
        cp.conditional.type    = cudaGraphCondTypeIf;
        cp.conditional.size    = 1;
        cudaGraphNode_t nCond;
        EAGLE_CHECK_ALWAYS(
            ::eagle::cuda::detail::graphAddNode(&nCond, g.graph(), &nSet, 1, &cp));

        /* The body is a PLAIN graph (unless illegally nested -- see the
         * class doc), so entering it as the IF node's sole child is exactly
         * the [PG]/ARM2-proven explicit-assembly path: 801 does not apply
         * here. Raw call, not g.addNode(), so harvestKernels_ never runs
         * (see the class doc's "body kernels are not harvested"). */
        cudaGraphNode_t nChild;
        EAGLE_CHECK_ALWAYS(cudaGraphAddChildGraphNode(
            &nChild, cp.conditional.phGraph_out[0], nullptr, 0,
            body_->graph()));

        return g.adoptNode(nCond);
    }

#else // EAGLE_CPU_ONLY

    /** @brief The internal host body graph -- populate with ``addNative``
     *  before this group is itself moved into a parent
     *  ``cpu::Graph::addNative`` call. */
    cpu::Graph& body() { return *body_; }

#endif // EAGLE_CPU_ONLY

    /** @brief No-op: see the class doc's "Scratch" section -- the body owns
     *  its own arena, so this group never draws from the PARENT's. Present
     *  only to satisfy the ``NativeNode`` protocol's ``reserveScratch``
     *  slot. */
    void reserveScratch(ScratchArena&) { }

    /**
     * @brief Host face: evaluate the guard by a host read of
     *        ``guard_.count`` / ``guard_.baseline`` and run the host body
     *        iff it passes.
     *
     * In a build where this class only maintains a CUDA body (i.e. NOT
     * ``EAGLE_CPU_ONLY``), there is no host body to run at all -- see the
     * class doc's "Cross-mode use is out of scope" section.
     */
    void runHost(cpu::Graph& /*g*/)
    {
        const unsigned int base = (guard_.baseline != nullptr)
            ? *guard_.baseline
            : 0u;
        const bool fires = (*guard_.count != base);
        if (!fires)
            return;

#ifdef EAGLE_CPU_ONLY
        EAGLE_ASSERT(body_ && body_->nodeCount() > 0,
            "ConditionalGroup::runHost: guard passed but the host body is "
            "empty -- populate body() before adding this group to a Graph");
        body_->run();
#else
        EAGLE_ASSERT(false,
            "ConditionalGroup::runHost: this build only maintains a CUDA "
            "body (EAGLE_CPU_ONLY is not defined); using a ConditionalGroup "
            "as a cpu::Graph native node in a CUDA-capable build is a "
            "mixed-mode use out of scope for v1");
#endif
    }

private:
    CountGuard guard_;
#ifndef EAGLE_CPU_ONLY
    std::shared_ptr<cuda::Graph> body_;
#else
    std::shared_ptr<cpu::Graph> body_;
#endif
};

} // namespace conditional
} // namespace eagle
