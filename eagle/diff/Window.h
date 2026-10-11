// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file diff/Window.h
 * @brief How a recorded run of up to @c S steps is cut into reverse windows.
 *        Plain integers, no CUDA.
 *
 * A reverse of a stepped run keeps the state only at window starts (steps
 * 0, W, 2W, ...) and recomputes one window at a time from its start. Memory
 * per sample is about (S/W + W) states, smallest near W = sqrt(S); the full
 * tape (W = S) is the same code with one window.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace eagle {
namespace diff {

using idx_t = std::int64_t;

/** @brief The smallest @c r with r*r >= @p s (exact for every int64 input). */
inline idx_t ceilSqrt(idx_t s) {
    if (s <= 0) return 0;
    idx_t r = static_cast<idx_t>(std::sqrt(static_cast<double>(s)));
    while (r * r < s) ++r;
    while (r > 1 && (r - 1) * (r - 1) >= s) --r;
    return r;
}

/** @brief The default window for runs of up to @p S steps:
 *  clamp(ceil(sqrt(S)), 16, 1024). */
inline idx_t defaultWindow(idx_t S) {
    return std::clamp<idx_t>(ceilSqrt(S), 16, 1024);
}

/** @brief Windows of @c window steps over a run of up to @c steps steps,
 *  aligned to multiples of @c window from step 0. */
struct WindowPlan {
    idx_t steps = 0;
    idx_t window = 1;

    /** @brief How many windows cover @c steps (0 for an empty run). */
    idx_t count() const { return (steps + window - 1) / window; }
    /** @brief The first step of window @p k. */
    idx_t start(idx_t k) const { return k * window; }
    /** @brief How many steps window @p k holds (the last may be short). */
    idx_t length(idx_t k) const { return std::min(window, steps - start(k)); }
    /** @brief The window holding step @p j. */
    idx_t windowOf(idx_t j) const { return j / window; }
    /** @brief The tape row step @p j is stored in. */
    idx_t row(idx_t j) const { return j % window; }
};

/** @brief A checked WindowPlan: @p steps >= 0 and @p window >= 1. */
inline WindowPlan planWindows(idx_t steps, idx_t window) {
    if (steps < 0)
        throw std::invalid_argument("planWindows: steps must be >= 0, got " + std::to_string(steps));
    if (window < 1)
        throw std::invalid_argument("planWindows: window must be >= 1, got " + std::to_string(window));
    return WindowPlan{ steps, window };
}

/**
 * @brief Samples per reverse pass under a memory budget.
 *
 * @param budgetBytes     the bytes a pass may hold.
 * @param perSampleBytes  what one sample costs (its checkpoints and tape).
 * @param n               the batch.
 * @param leaf            the canonical reduction's leaf block (a power of two).
 * @return @p n when the whole batch fits; otherwise the largest leaf * 2^q
 *         that fits, so each pass is a whole subtree of the canonical sum and
 *         the combined result is bitwise the unchunked one; 0 when not even
 *         one leaf block (or, with @p n smaller, one sample) fits.
 */
inline idx_t chunkSamples(idx_t budgetBytes, idx_t perSampleBytes, idx_t n, idx_t leaf) {
    if (perSampleBytes < 1 || n < 1 || leaf < 1 || (leaf & (leaf - 1)) != 0)
        throw std::invalid_argument("chunkSamples: perSampleBytes, n >= 1 and leaf a power of two");
    const idx_t fit = budgetBytes / perSampleBytes;
    if (fit >= n) return n;
    if (fit < leaf) return 0;
    idx_t chunk = leaf;
    while (chunk <= fit / 2) chunk *= 2;
    return chunk;
}

}  // namespace diff
}  // namespace eagle
