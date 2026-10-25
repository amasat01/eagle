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

/**
 * @brief The loop terms of a WHILE conditional node, passed by value to its
 *        head and tail setters (``nullptr`` = the term is absent).
 *
 * - ``cap``: iterations allowed per replay, >= 1; ``kLoopUncapped`` is the
 *   uncapped ``while`` (the guard alone ends it).
 * - ``counter``: the caller-owned device ``unsigned int[2]`` cell
 *   ``[remaining, ran]``, never nullptr; ``ran`` is the device-visible
 *   iteration index (0 during the first iteration).
 * - ``limit``: a runtime trip count, ``ran < *limit`` (``for i in range(n)``).
 * - ``brk``: a break cell the body may set; the head resets it to 0 every
 *   replay and the tail stops the loop once it is nonzero.
 */
struct LoopTerms {
    unsigned int cap            = 0;
    unsigned int* counter       = nullptr;
    const unsigned int* limit   = nullptr;
    unsigned int* brk           = nullptr;
};

/** @brief ``LoopTerms::cap`` of an uncapped loop. */
inline constexpr unsigned int kLoopUncapped = 0xFFFFFFFFu;

namespace detail {

/**
 * @brief The guard every eagle setter evaluates: ``(*count != base) !=
 *        negate``, ``base`` = ``*baseline`` or 0; a null ``count`` always
 *        holds (a loop bounded by its terms alone).
 */
__device__ inline bool guardHolds(const CountGuard& guard, unsigned int negate)
{
    if (guard.count == nullptr)
        return true;
    const unsigned int base = (guard.baseline != nullptr) ? *guard.baseline : 0u;
    return (*guard.count != base) != (negate != 0u);
}

/**
 * @brief eagle-owned device predicate for ``CountGuard``: sets a
 *        conditional handle's branch decision from a device-resident
 *        compare -- never a host round-trip. An IF runs its body on 1; an
 *        IF/ELSE runs arm 0 on 1 and arm 1 on 0.
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
    CountGuard guard, unsigned int negate)
{
    cudaGraphSetConditional(handle, guardHolds(guard, negate) ? 1u : 0u);
}

/** @brief SWITCH predicate: the handle is the arm index ``*value`` (an
 *  index past the last arm runs no arm). */
template<int = 0>
AETHER_KERNEL() void setSwitchKernel(cudaGraphConditionalHandle handle,
    const unsigned int* value)
{
    cudaGraphSetConditional(handle, *value);
}

/**
 * @brief Loop head for a WHILE conditional node: reset the loop state for
 *        this replay (``counter = [cap, 0]``, ``*brk = 0``) and decide
 *        whether the body runs at all: ``guard && cap > 0 && (no limit ||
 *        *limit > 0)``. The reset lives HERE, in a kernel the graph already
 *        needs, so no memset node and no per-launch
 *        ``cudaGraphCondAssignDefault`` reset is ever woven in.
 */
template<int = 0>
AETHER_KERNEL() void setLoopHeadKernel(cudaGraphConditionalHandle handle,
    CountGuard guard, unsigned int negate, LoopTerms t)
{
    t.counter[0] = t.cap;
    t.counter[1] = 0u;
    if (t.brk != nullptr)
        *t.brk = 0u;
    const bool more = t.cap != 0u && (t.limit == nullptr || *t.limit != 0u);
    cudaGraphSetConditional(handle, (more && guardHolds(guard, negate)) ? 1u : 0u);
}

/**
 * @brief Loop tail for a WHILE conditional node: the body's LAST node. One
 *        iteration completed (``--remaining``, ``++ran``); the loop
 *        continues iff the guard still holds, iterations remain, ``ran <
 *        *limit`` and the break cell is clear.
 */
template<int = 0>
AETHER_KERNEL() void setLoopTailKernel(cudaGraphConditionalHandle handle,
    CountGuard guard, unsigned int negate, LoopTerms t)
{
    const unsigned int remaining = --t.counter[0];
    const unsigned int ran       = ++t.counter[1];
    const bool more = remaining != 0u && (t.limit == nullptr || ran < *t.limit)
        && (t.brk == nullptr || *t.brk == 0u);
    cudaGraphSetConditional(handle, (more && guardHolds(guard, negate)) ? 1u : 0u);
}

/** @brief Add a 1x1 setter kernel node running @p func with @p args into
 *  @p graph after @p deps (``numDeps == 0`` permits ``deps == nullptr``).
 *  A cold, once-per-build control-plane call: checked with
 *  ``EAGLE_CHECK_ALWAYS`` (never the debug-only ``EAGLE_CHECK``) -- a
 *  swallowed failure would leave the branch decision undefined. */
inline cudaGraphNode_t addSetterNode(cudaGraph_t graph, void* func, void** args,
    const cudaGraphNode_t* deps, std::size_t numDeps)
{
    cudaGraphNode_t node;
    cudaKernelNodeParams kp = {};
    kp.func         = func;
    kp.gridDim      = { 1, 1, 1 };
    kp.blockDim     = { 1, 1, 1 };
    kp.kernelParams = args;
    EAGLE_CHECK_ALWAYS(
        cudaGraphAddKernelNode(&node, graph, deps, numDeps, &kp));
    return node;
}

/** @brief Add the ``setCountGuardKernel`` node evaluating @p guard; returns
 *  the new node -- the IF node's sole dependency. */
inline cudaGraphNode_t addSetCondKernelNode(cudaGraph_t graph,
    cudaGraphConditionalHandle handle, const CountGuard& guard,
    const cudaGraphNode_t* deps, std::size_t numDeps, unsigned int negate = 0)
{
    void* args[] = { (void*)&handle, (void*)&guard, (void*)&negate };
    return addSetterNode(graph, (void*)setCountGuardKernel<>, args, deps, numDeps);
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
 * ``buildInto`` (run from the outer's ``body().finalize()``) adds an
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
     *  ``body().finalize()`` is called explicitly here: unlike
     *  ``launcher()``, it is NOT auto-invoked, so a body populated purely
     *  through ``addNative`` would otherwise never capture its subgraph.
     */
    idx_t buildInto(cuda::Graph& g, const std::vector<idx_t>& deps)
    {
        body_->finalize();

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
