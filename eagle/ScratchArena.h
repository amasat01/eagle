// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/throw.h"        // EAGLE_ASSERT

#include <cstddef>
#include <set>
#include <span>
#include <vector>

namespace eagle {

/**
 * @brief Opaque handle to a reserved scratch region.
 *
 * Returned by ``ScratchArena::reserve`` in phase 1; ``ScratchArena::resolve``
 * turns it into a pointer into the arena's backing once the arena is committed.
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
 *     growing the tracked peak. No memory yet.
 *   - **commit + attach + resolve** (``Graph::finalize``): ``commit`` fixes the
 *     offsets and the peak; the graph allocates ONE buffer of that size, which it
 *     owns (host memory for ``cpu::Graph``, device memory for ``cuda::Graph``),
 *     and ``attach``es a view of it; ``resolve`` hands out pointers into it.
 *
 * The arena never allocates or frees. Its member functions are inline in every
 * file that includes this header, so they must not depend on how that file is
 * compiled: a body that switched on ``EAGLE_CPU_ONLY`` would differ between a
 * host-compiled and a CUDA-compiled file of one program, the linker would keep
 * one copy, and a buffer could reach the other allocator's free.
 *
 * The v1 policy is a greedy first-fit ancestor allocator: correct (never aliases
 * concurrent regions) and peak-not-sum for pipelines / chains / post-join, which
 * covers interleaved compaction/filter graphs. It is intentionally swappable for
 * a smarter policy (e.g. full inter-node liveness) without touching the
 * ``NativeNode`` concept or the ``Graph`` API.
 */
class ScratchArena {
public:
    ScratchArena()                               = default;
    ScratchArena(const ScratchArena&)            = delete;
    ScratchArena& operator=(const ScratchArena&) = delete;
    ScratchArena(ScratchArena&&) noexcept            = default;
    ScratchArena& operator=(ScratchArena&&) noexcept = default;

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

    /** @brief Fix every slot's offset and return the peak: the bytes of the one
     *  buffer the graph must ``attach`` (the sum of the reused slot sizes).
     *  Idempotent. Call once, after all reserves. */
    std::size_t commit()
    {
        if (committed_)
            return total_;
        offsets_.resize(slots_.size());
        std::size_t off = 0;
        for (std::size_t i = 0; i < slots_.size(); ++i) {
            offsets_[i] = off;
            off += align_(slots_[i].bytes);
        }
        total_     = off;
        committed_ = true;
        return total_;
    }

    /** @brief Point the arena at the graph-owned buffer it resolves into. The
     *  arena does not own @p memory; the graph keeps it alive as long as the
     *  arena is used. */
    void attach(std::span<std::byte> memory)
    {
        EAGLE_ASSERT(committed_, "ScratchArena: attach() before commit()");
        EAGLE_ASSERT(memory.size() >= total_, "ScratchArena: the attached buffer is smaller than the peak");
        memory_ = memory;
    }

    /** @brief Resolve a handle to its pointer in the attached buffer. Valid only after ``attach``. */
    void* resolve(const ScratchHandle& h) const
    {
        EAGLE_ASSERT(committed_ && memory_.size() >= total_ && (total_ == 0 || memory_.data() != nullptr),
                     "ScratchArena: resolve() before attach()");
        return memory_.data() + offsets_[h.slot];
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

    std::vector<Slot> slots_;
    std::vector<std::size_t> offsets_;
    std::set<idx_t> curAncestors_;
    idx_t curNode_        = 0;
    std::span<std::byte> memory_;   // a view of the graph-owned buffer
    std::size_t total_    = 0;
    bool committed_       = false;
};

}  // namespace eagle
