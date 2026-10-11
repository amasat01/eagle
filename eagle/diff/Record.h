// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file diff/Record.h
 * @brief The forward half of a reverse through a stepped run: the record slot
 *        (generation word, live and held state), the forward loop that takes
 *        window-start checkpoints, and the column gather a reverse uses to
 *        read a checkpoint in its own sample order. No CUDA.
 *
 * A record points at buffers the next recorded run overwrites. The slot
 * therefore refuses a new recording while the last record is LIVE (neither
 * reversed nor kept), and a reverse refuses a record whose generation is no
 * longer the newest. All slot state changes under one mutex.
 */

#include "eagle/diff/Window.h"

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace eagle {
namespace diff {

/** @brief The single live-record slot of one recording forward. Thread-safe. */
class RecordSlot {
public:
    /** @brief Why a recording could not begin. */
    enum class Begin : int { Ok = 0, Held = 1, Live = 2 };

    /** @brief Start a recording: refused while a reverse holds the last record
     *  (@c Held) or while it is still live (@c Live); otherwise the generation
     *  moves on and older records are stale from here. */
    Begin begin() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (readers_ > 0) return Begin::Held;
        if (live_) return Begin::Live;
        ++generation_;
        live_ = true;
        return Begin::Ok;
    }

    /** @brief A recording that failed part-way leaves nothing live. */
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex_);
        live_ = false;
    }

    /** @brief Hold the record of generation @p g for one read; false (and
     *  nothing held) when it is stale. */
    bool hold(std::uint64_t g) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (g != generation_) return false;
        ++readers_;
        return true;
    }

    /** @brief End one read of generation @p g; @p consumed (a finished
     *  reverse or a keep) frees the slot for the next recording. Returns
     *  false when the generation moved while it was held. */
    bool release(std::uint64_t g, bool consumed) {
        std::lock_guard<std::mutex> lock(mutex_);
        --readers_;
        if (g != generation_) return false;
        if (consumed) live_ = false;
        return true;
    }

    std::uint64_t generation() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return generation_;
    }
    bool live() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return live_;
    }
    bool held() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return readers_ > 0;
    }

private:
    mutable std::mutex mutex_;
    std::uint64_t generation_ = 0;
    bool live_ = false;
    int readers_ = 0;
};

/**
 * @brief The recording forward: launches of at most @p maxK steps that never
 *        cross a window boundary, a checkpoint before every launch that starts
 *        a window, until @p done() or @p maxSteps. Returns the steps run.
 *
 * @param maxK        steps one launch may take (1 for a single-step entry).
 * @param done        true once every sample has finished (checked before each launch).
 * @param checkpoint  checkpoint(k): save the state at the start of window k.
 * @param launch      launch(j, k): run steps j .. j+k-1 on every sample.
 */
template <class Done, class Checkpoint, class Launch>
idx_t recordForward(idx_t maxSteps, idx_t window, idx_t maxK, Done&& done, Checkpoint&& checkpoint,
                    Launch&& launch) {
    const WindowPlan plan = planWindows(maxSteps, window);
    if (maxK < 1) throw std::invalid_argument("recordForward: maxK must be >= 1");
    idx_t j = 0;
    while (j < maxSteps) {
        if (done()) break;
        if (plan.row(j) == 0) checkpoint(plan.windowOf(j));
        const idx_t k = std::min({ maxK, window - plan.row(j), maxSteps - j });
        launch(j, k);
        j += k;
    }
    return j;
}

/**
 * @brief dst[r*dstLd + s] = src[r*srcLd + perm[s]] for every row r < @p rows
 *        and s < @p count: @p count columns of a sample-last plane gathered
 *        into another sample order (and plane width). Element size @p bytes
 *        (1, 2, 4 or 8).
 */
inline void gatherColumns(void* dst, idx_t dstLd, const void* src, idx_t srcLd, const idx_t* perm,
                          idx_t count, idx_t rows, std::size_t bytes) {
    auto go = [&](auto tag) {
        using T = decltype(tag);
        T* d = static_cast<T*>(dst);
        const T* s = static_cast<const T*>(src);
        for (idx_t r = 0; r < rows; ++r) {
            T* dr = d + r * dstLd;
            const T* sr = s + r * srcLd;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (count > 65536)
#endif
            for (idx_t c = 0; c < count; ++c) dr[c] = sr[perm[c]];
        }
    };
    switch (bytes) {
    case 1: go(std::uint8_t{}); break;
    case 2: go(std::uint16_t{}); break;
    case 4: go(std::uint32_t{}); break;
    case 8: go(std::uint64_t{}); break;
    default: throw std::invalid_argument("gatherColumns: element size must be 1, 2, 4 or 8 bytes");
    }
}

}  // namespace diff
}  // namespace eagle
