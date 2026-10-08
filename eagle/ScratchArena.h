// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/CaptureGuard.h" // eagle::util::CaptureGuardState (STOP-THE-LINE fix, free_() below)
#include "eagle/util/DeviceError.h"  // EAGLE_CHECK_ALWAYS (+ transitively CUDA runtime)
#include "eagle/util/throw.h"        // EAGLE_ASSERT

#include <cstddef>
#include <cstdlib>
#include <set>
#include <vector>

namespace eagle {

/**
 * @brief Opaque handle to a reserved scratch region.
 *
 * Returned by ``ScratchArena::reserve`` in phase 1; ``ScratchArena::resolve``
 * turns it into a device (or host, in ``EAGLE_CPU_ONLY`` mode) pointer once the
 * arena is committed.
 */
struct ScratchHandle {
    idx_t slot        = 0;
    std::size_t bytes = 0;
};

/**
 * @brief A per-Graph scratch arena with build-time, ancestor-based slot reuse.
 *
 * The contained, swappable allocator behind ``Graph::addNative``. It packs the
 * working buffers of native nodes so that nodes with **disjoint execution
 * lifetimes share memory** — the arena sizes to the peak concurrently-live
 * scratch, not the sum. Because reuse is decided at build time from the node
 * dependency DAG (a node may alias a slot only if that slot's last user is a
 * guaranteed-complete *ancestor*), the resolved pointers are stable across every
 * graph replay — capture-safe.
 *
 * Two phases:
 *   - **reserve** (``beginNode`` / ``reserve`` / ``endNode``, driven by
 *     ``Graph::addNative``): assign every native node's working buffers to slots,
 *     growing the tracked peak. No device memory yet.
 *   - **commit + resolve** (``Graph::finalizeNatives``): allocate the single
 *     backing buffer once at the computed peak; ``resolve`` hands out pointers.
 *
 * The v1 policy is a greedy first-fit ancestor allocator: correct (never aliases
 * concurrent regions) and peak-not-sum for pipelines / chains / post-join, which
 * covers interleaved compaction/filter graphs. It is intentionally swappable for
 * a smarter policy (e.g. full inter-node liveness) without touching the
 * ``NativeNode`` concept or the ``Graph`` API.
 */
class ScratchArena {
public:
    ScratchArena() = default;
    ScratchArena(const ScratchArena&)            = delete;
    ScratchArena& operator=(const ScratchArena&) = delete;
    ScratchArena(ScratchArena&& o) noexcept { moveFrom_(o); }
    ScratchArena& operator=(ScratchArena&& o) noexcept
    {
        if (this != &o) {
            free_();
            moveFrom_(o);
        }
        return *this;
    }
    ~ScratchArena() { free_(); }

    /** @brief Open a reserve context for the native node @p nodeId whose
     *  transitive dependency ancestors are @p ancestors. A slot may be reused by
     *  this node only if its last owner is in that set (guaranteed complete
     *  before this node starts). Driven by ``Graph::addNative``. */
    void beginNode(idx_t nodeId, std::set<idx_t> ancestors)
    {
        EAGLE_ASSERT(
            !committed_, "ScratchArena: cannot reserve after commit()");
        curNode_      = nodeId;
        curAncestors_ = std::move(ancestors);
    }

    /** @brief Close the current node's reserve context. */
    void endNode() { curAncestors_.clear(); }

    /** @brief Reserve a working region of @p bytes for the current node.
     *
     *  Reuses an existing slot whose last owner is an ancestor of the current
     *  node (so its lifetime has ended — safe to alias), else opens a new slot.
     *  A node never reuses a slot it already claimed this round, so the several
     *  buffers of one primitive stay distinct. Call between ``beginNode`` /
     *  ``endNode``. */
    ScratchHandle reserve(std::size_t bytes)
    {
        EAGLE_ASSERT(
            !committed_, "ScratchArena: cannot reserve after commit()");
        for (Slot& s : slots_) {
            if (s.owner != curNode_ && curAncestors_.count(s.owner)) {
                s.owner = curNode_;
                if (bytes > s.bytes)
                    s.bytes = bytes;  // grow the slot to fit its largest user
                return ScratchHandle{ s.index, bytes };
            }
        }
        Slot s;
        s.index = idx_t(slots_.size());
        s.owner = curNode_;
        s.bytes = bytes;
        slots_.push_back(s);
        return ScratchHandle{ s.index, bytes };
    }

    /** @brief Allocate the single backing buffer, sized to the peak (the sum of
     *  the reused slot sizes). Idempotent. Call once, after all reserves, before
     *  any ``resolve``. */
    void commit()
    {
        if (committed_)
            return;
        offsets_.resize(slots_.size());
        std::size_t off = 0;
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            offsets_[i] = off;
            off += align_(slots_[i].bytes);
        }
        total_ = off;
        if (total_ > 0)
            alloc_();
        committed_ = true;
    }

    /** @brief Resolve a handle to its backing pointer (device memory in CUDA
     *  mode, host memory in ``EAGLE_CPU_ONLY``). Valid only after ``commit``. */
    void* resolve(const ScratchHandle& h) const
    {
        EAGLE_ASSERT(committed_, "ScratchArena: resolve() before commit()");
        return static_cast<char*>(base_) + offsets_[h.slot];
    }

    /** @brief The committed peak size in bytes (0 before commit / no scratch). */
    std::size_t peakBytes() const { return total_; }

    /** @brief The number of distinct slots (the reuse metric — a chain of N
     *  same-shaped nodes collapses to one slot per concurrent buffer). */
    idx_t slotCount() const { return idx_t(slots_.size()); }

private:
    struct Slot {
        idx_t index       = 0;
        idx_t owner       = 0;  // the last node that claimed this slot
        std::size_t bytes = 0;  // grown to the largest user
    };

    static std::size_t align_(std::size_t b)
    {
        return (b + 255u) & ~std::size_t(255u);  // 256B alignment
    }

    void alloc_()
    {
#ifndef EAGLE_CPU_ONLY
        EAGLE_CHECK_ALWAYS(cudaMalloc(&base_, total_));
#else
        base_ = std::malloc(total_);
#endif
    }

    /** @brief STOP-THE-LINE fix (CUDA branch only): ``arena_`` is a direct
     *  member of ``eagle::cuda::Graph``, which is Python-bound and reachable
     *  from ``GraphPipeline`` -- its destruction timing is GC-governed like
     *  ``Stream``/``Launcher``/``Graph::Storage``, so the same
     *  deferred-if-capturing release applies. The host (``EAGLE_CPU_ONLY``)
     *  branch never touches CUDA and needs no deferral. */
    void free_()
    {
        if (!base_)
            return;
#ifndef EAGLE_CPU_ONLY
        {
            void* const handle = base_;
            eagle::util::CaptureGuardState::instance().destroyOrDefer(
                [handle]() { cudaFree(handle); });
        }
#else
        std::free(base_);
#endif
        base_ = nullptr;
    }

    void moveFrom_(ScratchArena& o)
    {
        slots_        = std::move(o.slots_);
        offsets_      = std::move(o.offsets_);
        curAncestors_ = std::move(o.curAncestors_);
        curNode_      = o.curNode_;
        base_         = o.base_;
        total_        = o.total_;
        committed_    = o.committed_;
        o.base_       = nullptr;
        o.total_      = 0;
        o.committed_  = false;
    }

    std::vector<Slot> slots_;
    std::vector<std::size_t> offsets_;
    std::set<idx_t> curAncestors_;
    idx_t curNode_        = 0;
    void* base_           = nullptr;
    std::size_t total_    = 0;
    bool committed_       = false;
};

}  // namespace eagle
