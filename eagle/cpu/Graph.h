// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/ScratchArena.h"
#include "eagle/typedefs.h"

#include <concepts>
#include <functional>
#include <set>
#include <utility>
#include <vector>

namespace eagle {
namespace cpu {

/* Forward declaration so ``NativeNode`` can name the executor it runs
 * against; the full definition follows below (the concept only forms an
 * unevaluated reference to it, so the incomplete type is fine here). */
class Graph;

/**
 * @brief Host twin of ``eagle::cuda::NativeNode`` — the protocol a primitive
 *        (reduce / scan / filtering / a custom user node) implements to run
 *        itself under the CPU graph executor ``Graph``.
 *
 * Deliberately symmetric to the CUDA concept: both faces share
 * ``reserveScratch(arena)`` (the ``ScratchArena`` is mode-agnostic — only its
 * backing ``malloc``/``cudaMalloc`` differs), and the host face swaps
 * ``buildInto(cuda::Graph&, deps)`` for ``runHost(Graph&)``. There is no
 * capture/replay split on the host — a node simply executes its OpenMP + SIMD
 * kernels when its closure is invoked (there is no kernel-launch overhead to
 * amortize by capturing).
 *
 * A single node class per primitive satisfies BOTH concepts: ``buildInto`` under
 * ``#ifndef EAGLE_CPU_ONLY`` (the CUDA face), ``runHost`` always compiled.
 */
template<typename N>
concept NativeNode
    = requires(N n, Graph& g, ScratchArena& arena) {
          { n.reserveScratch(arena) };
          { n.runHost(g) };
      };

/**
 * @brief The CPU graph executor — host counterpart of ``eagle::cuda::Graph``.
 *
 * Mirrors ``Graph``'s two-phase native-node surface (``addNative`` →
 * ``finalize`` → ``run``) over ``eagle::cpu::Host`` (OpenMP + SIMD) instead of
 * CUDA graph capture. The arena and the ancestor-liveness reuse are shared
 * verbatim with the CUDA path, so a chain of N nodes collapses to the same slot
 * count on host as on device.
 *
 * **Execution model (v1).** ``run()`` invokes each node's closure in *add
 * order*, which is always a topological order of the native DAG: a node's
 * ``deps`` are indices already returned by a previous ``addNative``, so a node
 * can only depend on one that already exists. Sequential execution therefore
 * satisfies every dependency with no scheduler — and because each node is itself
 * fully parallel across the whole sample batch (OpenMP over sample tiles, SIMD
 * within a tile), there is no idle parallelism for cross-node scheduling to
 * reclaim in the batch-propagation regime.
 *
 * **OpenMP-composable + cache-resident.** Nodes drive the *context-aware*
 * host primitives (``Host::launch`` / ``Host::packetLaunch``), which work-share
 * across an *existing* OpenMP parallel region if ``run()`` is called inside one
 * and run serially otherwise — so a ``Graph`` nests safely inside a caller's
 * parallel region. The full dependency metadata (``nativeDeps_`` /
 * ``nativeAncestors_``) is retained so a future task-parallel ``runTasked()``
 * (``#pragma omp task depend``) can be added without touching the node protocol
 * or the arena. The graph-level ``bytesPerSample`` hint lets every node size its
 * tiles to the same per-sample working set, keeping each thread's sample tile
 * L2-resident *across the whole node chain* — the host analogue of the CUDA
 * graph's launch-overhead amortization, and precisely what sequential execution
 * enables.
 *
 * Move-only (owns a move-only ``ScratchArena``).
 */
class Graph {
public:
    Graph() = default;

    /** @brief Add a host-native primitive (reduce / scan / filtering / custom)
     *  to this graph via the same two-phase, arena-backed protocol as
     *  ``Graph::addNative`` — the CPU-executor twin.
     *
     *  Phase 1 (here): the node reserves its working buffers in this graph's
     *  ``ScratchArena``, which assigns each a slot reused from this node's
     *  dependency ancestors (disjoint-lifetime nodes share memory — peak, not
     *  sum). Nothing runs yet. Phase 2 (``run`` / ``finalize``): the arena
     *  commits once and each node's ``runHost`` executes with the resolved host
     *  pointers.
     *
     *  ``deps`` are native-node indices (values previously returned by
     *  ``addNative``). Returns this node's native index. */
    template<typename N>
        requires NativeNode<N>
    idx_t addNative(N node, const std::vector<idx_t>& deps = {})
    {
        const idx_t nativeIdx = idx_t(runClosures_.size());

        // Transitive dependency ancestors over the native DAG — identical to
        // ``Graph::addNative``: a slot may be reused by this node only if its
        // last owner is a guaranteed-complete ancestor.
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
        runClosures_.push_back(
            [node = std::move(node)](Graph& g) mutable { node.runHost(g); });
        return nativeIdx;
    }

    /** @brief Commit the scratch arena (allocate the single backing buffer at
     *  the computed peak). Idempotent; called implicitly by the first ``run``.
     *  Separated out so a graph can be committed once and ``run`` many times
     *  (the propagation-loop pattern) with stable scratch pointers. */
    void finalize()
    {
        if (!finalized_) {
            arena_.commit();
            finalized_ = true;
        }
    }

    /** @brief Execute every node's ``runHost`` in add order (a topological
     *  order of the native DAG). Commits the arena on first call, then is
     *  freely replayable. Nesting-safe inside a caller's OpenMP parallel
     *  region (nodes use context-aware host primitives). */
    void run()
    {
        finalize();
        for (auto& c : runClosures_)
            c(*this);
    }

    /** @brief The graph's scratch arena; a node resolves its reserved handles
     *  through it inside ``runHost``. */
    ScratchArena& scratchArena() { return arena_; }
    const ScratchArena& scratchArena() const { return arena_; }

    /** @brief Per-sample working-set hint (bytes) nodes pass to
     *  ``Host::launch`` for cross-node L2 residency. ``0`` uses the
     *  compile-time default tile size. */
    idx_t bytesPerSample() const { return bytesPerSample_; }
    Graph& setBytesPerSample(idx_t bytes)
    {
        bytesPerSample_ = bytes;
        return *this;
    }

    /** @brief Number of native nodes added. */
    idx_t nodeCount() const { return idx_t(runClosures_.size()); }

private:
    /** @brief Per-graph scratch arena (peak-not-sum reuse), shared design with
     *  the CUDA path; host ``malloc`` backing in ``EAGLE_CPU_ONLY`` mode. */
    ScratchArena arena_;

    /** @brief Native-node bookkeeping. ``nativeDeps_`` / ``nativeAncestors_``
     *  are retained past build so a future ``runTasked()`` can emit an OpenMP
     *  task-dependency DAG from them without changing the node protocol. */
    std::vector<std::vector<idx_t>> nativeDeps_;
    std::vector<std::set<idx_t>> nativeAncestors_;
    std::vector<std::function<void(Graph&)>> runClosures_;

    idx_t bytesPerSample_ = 0;
    bool finalized_       = false;
};

}  // namespace cpu
}  // namespace eagle
