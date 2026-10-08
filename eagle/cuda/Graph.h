// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cuda/CapturedGraph.h"
#include "eagle/cuda/ComputeBlocks.h"
#include "eagle/cuda/detail/RuntimeCompat.h"
#include "eagle/cuda/NativeNode.h"
#include "eagle/ScratchArena.h"
#include "eagle/cuda/Stream.h"
#include "eagle/cuda/StreamCapturer.h"
#include "eagle/cuda/detail.h"
#include "eagle/util/CaptureGuard.h"

#ifndef EAGLE_CPU_ONLY

#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace eagle {
namespace cuda {

/* Forward declaration so Graph::launcher() can name the return type. The
 * full Launcher definition lives in Launcher.h, which is included at the
 * bottom of this header (mutual-include via guards). */
class Launcher;

/** @brief Node labels */
using LabelsT = std::map<std::string, idx_t>;

/**
 * @brief Per-kernel-node record captured at graph-build time.
 *
 * ``capturedParams`` is snapshotted via one
 * ``cudaGraphKernelNodeGetParams`` call at capture so
 * ``Launcher::setLogicalSize`` never has to re-query at runtime —
 * the runtime call returns a ``func`` pointer that
 * ``cudaGraphExecKernelNodeSetParams`` rejects with
 * ``InvalidDeviceFunction`` for templated kernels.
 *
 * Lifetime: ``node`` and ``capturedParams.kernelParams`` reference
 * internals of the source ``cudaGraph_t``; both remain valid as long
 * as the ``Graph::Storage`` they point into is alive — i.e. as long
 * as either the producing ``Graph`` *or* any ``Launcher`` produced
 * from it still holds a ``shared_ptr<Graph::Storage>``.
 */
struct KernelNodeRecord {
    cudaGraphNode_t node;
    cudaKernelNodeParams capturedParams;
    /** @brief Caller-supplied per-kernel cap (``maxBlockSize``).
     *  ``0`` = grid-only mutation (keep captured blockDim, derive
     *  gridDim from logicalSize). ``> 0`` and ``!= kFixedSize``
     *  enables full re-tune via ``computeBlocks``. ``kFixedSize``
     *  = preserve both gridDim and blockDim from capture (use for
     *  layout-locked kernels whose grid is determined by something
     *  other than the launcher's logical size — e.g. multi-level
     *  reductions whose recursion-level kernels operate on small
     *  fixed-size buffers and would over-write their internal
     *  state if grid grew on a logicalSize patch). */
    idx_t idealBlockSize;
};

/** @brief Sentinel for ``KernelNodeRecord::idealBlockSize`` meaning
 *  "no grid/block patch — preserve capture". Used by
 *  ``Launcher::setLogicalSize`` to skip layout-locked kernels.
 *  Adopted by ``eagle::filtering::FilteringSlice::scanGraph``
 *  (multi-level scan recursion is layout-locked). */
inline constexpr idx_t kFixedSize
    = std::numeric_limits<idx_t>::max();

/**
 * @brief CUDA graph builder and dependency manager.
 *
 * Wraps a ``cudaGraph_t`` with RAII lifetime management and provides
 * methods to add child graphs, kernel nodes, generic nodes, and
 * host-callback nodes. After all nodes are added, call ``launcher()``
 * to instantiate and obtain a ``Launcher`` for repeated execution.
 *
 * Lifetime model: the CUDA-side data lives in a private nested
 * ``Storage`` struct held via ``std::shared_ptr``. ``Graph`` and
 * every ``Launcher`` it produces share ownership of that storage.
 * The source ``cudaGraph_t`` and every captured-child graph are
 * destroyed exactly once — when the last ``shared_ptr<Storage>``
 * (Graph's or Launcher's) drops. This guarantees that node handles
 * and ``kernelParams`` pointers stored in ``KernelNodeRecord``
 * remain valid for any subsequent ``Launcher::setLogicalSize`` call
 * even after the producing ``Graph`` has gone out of scope.
 *
 * Multi-launcher fan-out: ``launcher()`` is ``const`` and may be
 * called multiple times on the same ``Graph`` — each call snapshots
 * the current ``kernelNodes_`` table and instantiates an independent
 * exec graph; all share the same ``Storage`` via ref-count.
 *
 * Copy construction and copy assignment are disabled; use move semantics.
 *
 * @note Available only when ``EAGLE_CPU_ONLY`` is not defined.
 * @see Launcher         — executes a compiled graph.
 * @see StreamCapturer   — records kernel launches into a graph.
 */
class Graph {
    using Self   = Graph;
    using NodesT = detail::Holder<cudaGraphNode_t, true>;
    using DepsT  = NodesT::DepsT;

    friend class Launcher;

    /** @brief Single-owner RAII container for the source ``cudaGraph_t``
     *  and every host-side buffer CUDA holds pointers into.
     *
     *  Held only via ``std::shared_ptr<Storage>`` from inside ``Graph``
     *  and any ``Launcher`` produced from it; non-copyable and
     *  non-movable to make that the only valid handle path. The
     *  destructor releases the source graph and every captured-child
     *  ``cudaGraph_t`` original — fixing the pre-existing leak in the
     *  prior implementation, which destroyed only ``graph`` and
     *  never the captured-child originals stored in ``childGraphs``. */
    struct Storage {
        cudaGraph_t graph = nullptr;
        bool created      = false;

        std::vector<cudaKernelNodeParams> kernelNodeParams;
        std::vector<cudaGraphNodeParams> genericNodeParams;
        std::vector<std::unique_ptr<HostCallback>> hostNodeData;
        std::vector<std::function<void()>> hostNodeLambdas;

        Storage()                          = default;
        Storage(const Storage&)            = delete;
        Storage& operator=(const Storage&) = delete;
        Storage(Storage&&)                 = delete;
        Storage& operator=(Storage&&)      = delete;

        /** @brief STOP-THE-LINE fix: deferred-if-capturing release, same
         *  rationale as ``Stream::destroy_()`` / ``Launcher::destroyInstance_()``
         *  -- ``Storage`` is reached transitively from Python via any
         *  ``Graph`` or ``Launcher`` sharing this ``shared_ptr``, so its
         *  destruction timing is GC-governed exactly like theirs. */
        ~Storage()
        {
            if (created) {
                const cudaGraph_t handle = graph;
                ::eagle::util::CaptureGuardState::instance().destroyOrDefer(
                    [handle]() { cudaGraphDestroy(handle); });
            }
        }
    };

public:
    /** @brief Default constructor allocates the shared ``Storage``,
     *  creates the underlying ``cudaGraph_t``, and adds the master
     *  empty node. */
    Graph()
        : storage_{ std::make_shared<Storage>() }
    {
        create_();
        addEmptyNode({});
    }

    /** @brief Adopt a stream-captured graph as the source graph and replay
     *  it directly — no master node, no child-graph clone, no kernel
     *  harvest.
     *
     *  This is the single-captured-sequence path: ``launcher()``
     *  instantiates the captured ``cudaGraph_t`` verbatim, exactly like a
     *  direct ``cudaGraphInstantiate`` of the capture (the shape a
     *  ``StreamCapturer``'d graph is meant to take). It deliberately skips
     *  ``harvestKernels_`` — calling ``cudaGraphKernelNodeGetParams`` on a
     *  driver-API-launched (e.g. cupy) kernel node is unsafe (see
     *  ``KernelNodeRecord``) and is only needed for ``setLogicalSize``,
     *  which does not apply to a verbatim replay. Consumes ``captured``
     *  (takes ownership of its ``cudaGraph_t``); ``setLogicalSize`` is a
     *  no-op for the resulting Graph — use the default ctor + ``addNode``
     *  when dynamic re-sizing is required. */
    explicit Graph(CapturedGraph captured)
        : storage_{ std::make_shared<Storage>() }
    {
        storage_->graph   = captured.release();
        storage_->created = true;
    }

    /** @brief Copy constructor is forbidden */
    Graph(Graph& other)       = delete;
    Graph(const Graph& other) = delete;

    /** @brief Move constructor — defaulted; transfers the shared_ptr
     *  and bookkeeping members. */
    Graph(Graph&& other) noexcept            = default;

    /** @brief Copy assignment is forbidden */
    Self& operator=(Graph& other)       = delete;
    Self& operator=(const Graph& other) = delete;

    /** @brief Move assignment — defaulted. */
    Self& operator=(Graph&& other) noexcept = default;

    /** @brief Update the stream */
    Self& stream(const cudaStream_t& stream)
    {
        stream_ = stream;
        return *this;
    }

    /** @brief Expose the stream */
    const cudaStream_t& stream() const { return stream_; }

    /** @brief Add the given empty node to the underlined graph */
    template<typename IdxT = std::initializer_list<idx_t>>
    void addEmptyNode(const IdxT& dependencies = {})
    {
        DepsT deps = nodes_.findDependencies<IdxT>(dependencies);
        cudaGraphNode_t& nodeSlot = nodes_.createSlot();
        EAGLE_CHECK_ALWAYS(cudaGraphAddEmptyNode(
            &nodeSlot, storage_->graph, data_(deps), deps.size()));
    }

    /** @brief Add a child graph as node, applying ``idealBlockSize``
     *  uniformly to every kernel inside it. **Borrowed-input
     *  contract**: the consumer does not take ownership of
     *  ``childGraph``; the caller (or upstream producer) is
     *  responsible for its lifetime. ``cudaGraphAddChildGraphNode``
     *  clones the input into the parent immediately, so
     *  ``childGraph`` may be reused, kept, or destroyed by its owner
     *  after this call returns.
     *
     *  ``idealBlockSize == 0`` (default) keeps the captured blockDim at
     *  ``Launcher::setLogicalSize`` time (grid-only mutation); a
     *  positive value enables the full ``computeBlocks`` re-tune.
     *
     *  Use the ``CapturedGraph`` overload when the child is owned
     *  (e.g. produced by ``StreamCapturer::end()`` /
     *  ``cudaGraphClone``) and/or when it has multiple kernels with
     *  distinct caps. */
    template<typename IdxT = std::initializer_list<idx_t>>
    void addNode(cudaGraph_t childGraph, const IdxT& dependencies = {},
        idx_t idealBlockSize = 0)
    {
        DepsT deps = nodes_.findDependencies<IdxT>(dependencies);
        cudaGraphNode_t& nodeSlot = nodes_.createSlot();
        EAGLE_CHECK_ALWAYS(cudaGraphAddChildGraphNode(&nodeSlot, storage_->graph,
            data_(deps), deps.size(), childGraph));
        harvestKernels_(nodeSlot, {}, idealBlockSize);
        /* No retention: childGraph is the producer's responsibility. */
    }

    /** @brief Add a child graph carrying a per-kernel
     *  ``idealBlockSize`` table. **Owned-input contract**:
     *  ``CapturedGraph`` is move-only and represents transferred
     *  ownership of the wrapped ``cudaGraph_t``. The by-value
     *  parameter is move-constructed from the argument; after
     *  ``cudaGraphAddChildGraphNode`` clones it into the parent,
     *  the parameter's destructor releases the source at the end
     *  of this call.
     *
     *  Caps are read positionally from ``child.idealBlockSizes()``.
     *  Empty vector = grid-only for every kernel. */
    template<typename IdxT = std::initializer_list<idx_t>>
    void addNode(CapturedGraph child, const IdxT& dependencies = {})
    {
        DepsT deps = nodes_.findDependencies<IdxT>(dependencies);
        cudaGraphNode_t& nodeSlot = nodes_.createSlot();
        EAGLE_CHECK_ALWAYS(cudaGraphAddChildGraphNode(&nodeSlot, storage_->graph,
            data_(deps), deps.size(), child.graph()));
        harvestKernels_(nodeSlot, child.idealBlockSizes());
        /* ~CapturedGraph runs on `child` at the end of this call,
         * destroying the source handle. The parent already has its
         * clone via cudaGraphAddChildGraphNode. */
    }

    /** @brief Add the given set of sub-nodes to the underlined graph. By
     * default, the nodes are treated as sequentially dependents */
    template<typename IdxT = std::initializer_list<idx_t>>
    void addNodes(const std::initializer_list<cudaGraph_t>& childGraphs,
        const IdxT& dependencies = {})
    {
        idx_t i = 0;
        for (const cudaGraph_t& childGraph : childGraphs) {
            if (i == 0)
                this->addNode(childGraph, dependencies);
            else
                this->addNode(childGraph);
            i++;
        }
    }

    /** @brief Add a generic node through its parameters */
    template<typename IdxT = std::initializer_list<idx_t>>
    void addNode(cudaGraphNodeParams nodeParams, const IdxT& dependencies = {})
    {
        DepsT deps = nodes_.findDependencies<IdxT>(dependencies);
        storage_->genericNodeParams.push_back(std::move(nodeParams));
        cudaGraphNode_t& nodeSlot = nodes_.createSlot();
        EAGLE_CHECK_ALWAYS(detail::graphAddNode(&nodeSlot, storage_->graph, data_(deps),
            deps.size(), &storage_->genericNodeParams.back()));
    }

    /** @brief Add a kernel node.
     *
     *  ``idealBlockSize`` (default ``0`` = grid-only) is the per-kernel
     *  cap; pass ``KernelLaunch::maxBlockSize`` for full re-tuning.
     *  Records the kernel into ``kernelNodes_`` so a future
     *  ``Launcher::setLogicalSize`` can patch gridDim/blockDim. */
    template<typename IdxT = std::initializer_list<idx_t>>
    void addKernelNode(
        cudaKernelNodeParams params, const IdxT& dependencies = {},
        idx_t idealBlockSize = 0)
    {
        DepsT deps = nodes_.findDependencies<IdxT>(dependencies);
        storage_->kernelNodeParams.push_back(std::move(params));
        cudaGraphNode_t& nodeSlot = nodes_.createSlot();
        EAGLE_CHECK_ALWAYS(cudaGraphAddKernelNode(&nodeSlot, storage_->graph,
            data_(deps), deps.size(), &storage_->kernelNodeParams.back()));

        /* Snapshot from graph-internal storage so ``setLogicalSize``
         * never has to call ``cudaGraphKernelNodeGetParams`` at runtime
         * (broken for templated kernels). */
        cudaKernelNodeParams snapshot {};
        EAGLE_CHECK_ALWAYS(cudaGraphKernelNodeGetParams(nodeSlot, &snapshot));
        kernelNodes_.push_back(
            KernelNodeRecord{ nodeSlot, snapshot, idealBlockSize });
    }

    /** @brief Add a host node to the graph.
     *
     *  Internally captures the host-callback launch into a fresh
     *  ``cudaGraph_t`` (owned), wraps it in ``CapturedGraph``, and
     *  routes through the owned-input ``addNode`` overload — that
     *  overload's by-value parameter destructor releases the source
     *  after the parent clones it in. Closes the pre-existing leak
     *  of the captured-graph original. */
    template<typename IdxT = std::initializer_list<idx_t>>
    void addHostNode(
        std::function<void()> lambda, const IdxT& dependencies = {})
    {
        storage_->hostNodeData.emplace_back(
            std::make_unique<HostCallback>(HostCallback{ std::move(lambda) }));
        storage_->hostNodeLambdas.push_back(std::move(lambda));
        StreamCapturer capturer(stream_);
        capturer.begin();
        storage_->hostNodeData.back()->launch(stream_);
        addNode<IdxT>(CapturedGraph{ capturer.end() }, dependencies);
    }

    /** @brief Destructor — defaulted.
     *
     *  The shared ``Storage`` is released through ``storage_`` only
     *  when the last ``shared_ptr<Storage>`` drops. After
     *  ``launcher()`` is called any number of times, this destructor
     *  decrements the ref-count by one; the actual CUDA cleanup runs
     *  inside ``~Storage`` when the last Launcher dies (or here, if
     *  no Launcher was ever produced). */
    ~Graph() = default;

    /** @brief Add a native primitive (reduce / scan / filtering) to this graph
     *  via the two-phase, arena-backed protocol — the transparent alternative to
     *  hand-wiring its subgraph with ``addNode``.
     *
     *  Phase 1 (here): the node reserves its working buffers in this graph's
     *  ``ScratchArena``, which assigns each a slot reused from this node's
     *  dependency ancestors (nodes with disjoint lifetimes share memory — peak,
     *  not sum). Nothing is captured yet. Phase 2 (``finalizeNatives``, run once
     *  before ``launcher()``): the arena commits and each node's subgraph is
     *  captured with the resolved pointers.
     *
     *  A builder never touches the arena; ``deps`` are native-node indices
     *  (values previously returned by ``addNative``). Returns this node's native
     *  index, for use as a later node's dependency.
     *
     *  @tparam N a value type satisfying ``NativeNode`` (copyable, moved in). */
    /// \cond EAGLE_SKIP_DOX
    // (hidden from Doxygen: the requires-clause trips Sphinx's C++ parser;
    // addNative is documented above and on the api_graph docs page.)
    template<typename N>
        requires NativeNode<N>
    idx_t addNative(N node, const std::vector<idx_t>& deps = {})
    {
        const idx_t nativeIdx = idx_t(nativeDeps_.size());

        // Transitive dependency ancestors over the native DAG: a slot may be
        // reused by this node only if its last owner is a guaranteed-complete
        // ancestor (so its scratch lifetime has ended).
        std::set<idx_t> ancestors;
        for (idx_t d : deps) {
            ancestors.insert(d);
            ancestors.insert(
                nativeAncestors_[d].begin(), nativeAncestors_[d].end());
        }

        arena_.beginNode(nativeIdx, ancestors);
        node.reserveScratch(arena_);
        arena_.endNode();

        nativeDeps_.push_back(deps);
        nativeAncestors_.push_back(std::move(ancestors));
        nativeBuilds_.push_back(
            [node = std::move(node)](Graph& g,
                const std::vector<idx_t>& gdeps) mutable -> idx_t {
                return node.buildInto(g, gdeps);
            });
        return nativeIdx;
    }

    /** @brief Phase 2 of the native-node build: commit the scratch arena, then
     *  capture every native node's subgraph (in add order) with resolved
     *  pointers, mapping each node's native dependencies to the graph-node
     *  indices its ancestors produced. Idempotent; a no-op with no native nodes.
     *  Call once after the last ``addNative`` and before ``launcher()``. */
    void finalizeNatives()
    {
        if (nativesFinalized_ || nativeBuilds_.empty())
            return;
        arena_.commit();
        std::vector<idx_t> nativeToGraphNode(nativeBuilds_.size(), 0);
        for (std::size_t i = 0; i < nativeBuilds_.size(); ++i) {
            std::vector<idx_t> gdeps;
            gdeps.reserve(nativeDeps_[i].size());
            for (idx_t d : nativeDeps_[i])
                gdeps.push_back(nativeToGraphNode[d]);
            nativeToGraphNode[i] = nativeBuilds_[i](*this, gdeps);
        }
        nativesFinalized_ = true;
    }

    /** @brief The graph's scratch arena; a native node resolves its reserved
     *  handles through it inside ``buildInto``. */
    ScratchArena& scratchArena() { return arena_; }
    const ScratchArena& scratchArena() const { return arena_; }

    /** @brief Expose the source graph (lifetime tied to ``storage_``). */
    cudaGraph_t graph() const { return storage_ ? storage_->graph : nullptr; }

    /** @brief Construct a Launcher sharing this Graph's source
     *  storage and a snapshot of its current ``kernelNodes_`` table.
     *
     *  May be called more than once; each call instantiates an
     *  independent exec graph and ref-counts the shared ``Storage``.
     *  Subsequent ``addNode`` calls on the producing Graph do not
     *  affect previously-handed-out Launchers' patch tables.
     *
     *  Defined inline at the bottom of Launcher.h (after the full
     *  Launcher class is visible). */
    Launcher launcher() const;

    /** @brief Synchronize */
    inline void synchronize() { EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(stream_)); }

    /** @brief Return the index of the last node in this graph */
    idx_t lastNode() const { return nodes_.size() - 1; }

    /** @brief Raw CUDA handle of node @p id -- graph-surgery escape hatch for
     *  facilities (conditional nodes) that must call cudaGraph* APIs directly. */
    cudaGraphNode_t nodeHandle(idx_t id) const { return nodes_[id]; }

    /** @brief Register an externally-created node so later adds can depend on it. */
    idx_t adoptNode(cudaGraphNode_t node)
    {
        nodes_.createSlot() = node;
        return nodes_.size() - 1;
    }

protected:
    /** @brief Static wrapper for dependency data */
    static cudaGraphNode_t* data_(DepsT& deps)
    {
        return deps.empty() ? nullptr : deps.data();
    }

    /** @brief Append every kernel-type node inside the cloned child at
     *  ``parentChildNode`` to ``kernelNodes_``, snapshotting each
     *  kernel's ``cudaKernelNodeParams`` once at capture time.
     *
     *  Per-kernel ``idealBlockSize`` is sourced from ``caps[k]`` when
     *  ``caps`` is non-empty (size must match the kernel count) or from
     *  ``uniformDefault`` otherwise.
     *
     *  Lifetime: harvested ``inner`` handles and snapshotted
     *  ``kernelParams`` pointers reference internals of the cloned
     *  child sub-graph embedded in ``storage_->graph``; remain valid
     *  for as long as ``Storage`` is alive. */
    void harvestKernels_(cudaGraphNode_t parentChildNode,
        const std::vector<idx_t>& caps, idx_t uniformDefault = 0)
    {
        cudaGraph_t clonedChild = nullptr;
        EAGLE_CHECK_ALWAYS(
            cudaGraphChildGraphNodeGetGraph(parentChildNode, &clonedChild));

        size_t numNodes = 0;
        EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(clonedChild, nullptr, &numNodes));
        if (numNodes == 0)
            return;

        std::vector<cudaGraphNode_t> allNodes(numNodes);
        EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(clonedChild, allNodes.data(), &numNodes));

        const bool useCaps = !caps.empty();
        size_t k = 0;
        for (cudaGraphNode_t inner : allNodes) {
            cudaGraphNodeType type;
            EAGLE_CHECK_ALWAYS(cudaGraphNodeGetType(inner, &type));
            if (type != cudaGraphNodeTypeKernel)
                continue;
            cudaKernelNodeParams params {};
            EAGLE_CHECK_ALWAYS(cudaGraphKernelNodeGetParams(inner, &params));
            const idx_t cap = useCaps ? caps[k] : uniformDefault;
            kernelNodes_.push_back(KernelNodeRecord{ inner, params, cap });
            ++k;
        }
        EAGLE_ASSERT(!useCaps || k == caps.size(),
            "CapturedGraph::idealBlockSizes size must match the number of "
            "kernel nodes in the cloned child graph");
    }

    /** @brief Create the underlying ``cudaGraph_t`` inside ``storage_``. */
    void create_()
    {
        if (!storage_->created) {
            EAGLE_CHECK_ALWAYS(cudaGraphCreate(&storage_->graph, 0));
            storage_->created = true;
        }
    }

    /* Data members */

    /** @brief Shared CUDA-side storage; co-owned by every Launcher
     *  produced from this Graph. */
    std::shared_ptr<Storage> storage_;

    /** @brief Build-only bookkeeping (per-Graph, not shared). */
    NodesT nodes_;
    cudaStream_t stream_ = 0;

    /** @brief Build-time records of harvested kernel nodes; copied
     *  into each produced Launcher at ``launcher()`` time so further
     *  ``addNode`` calls don't mutate previously-handed-out Launchers'
     *  patch tables. */
    std::vector<KernelNodeRecord> kernelNodes_;

    /** @brief Per-graph scratch arena for native nodes (peak-not-sum reuse). */
    ScratchArena arena_;

    /** @brief Native-node phase-1 records, consumed at ``finalizeNatives``:
     *  each node's native dependencies, its transitive ancestor set (for arena
     *  reuse), and a build closure that captures its subgraph in phase 2. */
    std::vector<std::vector<idx_t>> nativeDeps_;
    std::vector<std::set<idx_t>> nativeAncestors_;
    std::vector<std::function<idx_t(Graph&, const std::vector<idx_t>&)>>
        nativeBuilds_;
    bool nativesFinalized_ = false;
};

} // namespace cuda
} // namespace eagle

/* Mutual include: Launcher.h needs the full Graph + Storage definitions,
 * and any caller of ``graph.launcher()`` needs the full Launcher. With
 * include guards this is safe — when Launcher.h re-includes Graph.h
 * during this expansion it returns immediately via the guard. */
#include "eagle/cuda/Launcher.h"

#endif
