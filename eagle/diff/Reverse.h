// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file diff/Reverse.h
 * @brief The window driver of a reverse through a stepped run. No CUDA: the
 *        three operations it sequences (gather a checkpoint, recompute one
 *        step, step the adjoint back once) are the caller's, so the same
 *        driver runs host entries through function pointers and device
 *        launches alike.
 *
 * Samples stop at different steps. The reverse works in its own sample
 * order, sorted by stop step (longest first), so the samples still running
 * at step j are always a PREFIX of that order: every operation at step j
 * runs over its first live(j) samples and leaves the rest untouched. A
 * sample's adjoint therefore starts moving at its own last step.
 *
 * Window by window, last first: gather the window's start state into
 * reverse order, recompute its steps forward (writing one tape row per
 * step), then step the adjoint back through the rows. Every window replays
 * the same compiled step on the same inputs, so the gradient does not
 * depend on the window size, bit for bit.
 */

#include "eagle/diff/Window.h"

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace eagle {
namespace diff {

/** @brief The reverse's sample order: samples sorted by stop step, longest
 *  first, ties in sample order. order[s] is the sample at reverse slot s. */
inline std::vector<idx_t> stopOrder(const idx_t* steps, idx_t n) {
    std::vector<idx_t> order(static_cast<std::size_t>(n));
    std::iota(order.begin(), order.end(), idx_t{ 0 });
    std::stable_sort(order.begin(), order.end(),
                     [&](idx_t a, idx_t b) { return steps[a] > steps[b]; });
    return order;
}

/** @brief live[j] = how many samples take step j (stop > j), for j below
 *  the longest stop; @p steps in any order. */
inline std::vector<idx_t> liveCounts(const idx_t* steps, idx_t n) {
    idx_t longest = 0;
    for (idx_t i = 0; i < n; ++i) {
        if (steps[i] < 0) throw std::invalid_argument("liveCounts: a stop step is negative");
        longest = std::max(longest, steps[i]);
    }
    std::vector<idx_t> stopsAt(static_cast<std::size_t>(longest) + 1, 0);
    for (idx_t i = 0; i < n; ++i) ++stopsAt[static_cast<std::size_t>(steps[i])];
    std::vector<idx_t> live(static_cast<std::size_t>(longest));
    idx_t running = n;
    for (idx_t j = 0; j < longest; ++j) {
        running -= stopsAt[static_cast<std::size_t>(j)];
        live[static_cast<std::size_t>(j)] = running;
    }
    return live;
}

/**
 * @brief Run the reverse, window by window, last window first.
 *
 * @param window     the tape's window W (rows are step mod W).
 * @param live       live[j] from liveCounts: its size is the longest stop.
 * @param gather     gather(k, count): load window k's start state for the
 *                   first @c count reverse slots.
 * @param recompute  recompute(c, len): run the window's steps c .. c+len-1
 *                   forward (tape rows from 0), each over its own live
 *                   prefix; one fused launch or len single steps, as recorded.
 * @param stepBack   stepBack(j, count): the adjoint through step j.
 */
template <class Gather, class Recompute, class StepBack>
void reverseWindows(idx_t window, const std::vector<idx_t>& live, Gather&& gather,
                    Recompute&& recompute, StepBack&& stepBack) {
    const WindowPlan plan = planWindows(static_cast<idx_t>(live.size()), window);
    for (idx_t k = plan.count() - 1; k >= 0; --k) {
        const idx_t c = plan.start(k), len = plan.length(k);
        gather(k, live[static_cast<std::size_t>(c)]);
        recompute(c, len);
        for (idx_t j = c + len - 1; j >= c; --j) stepBack(j, live[static_cast<std::size_t>(j)]);
    }
}

/**
 * @brief Which of the two adjoint planes the next backward launch reads, for a
 *        reverse whose full windows of @c fusedK steps are each ONE launch.
 *
 * A step flips the plane (reads one, writes the other), and so does a fused
 * window, so the plane the run ends on (plane 1) fixes the one it starts on.
 * @c fusedK = 0 is the plain reverse: plane (steps - 1) & 1 for the first step,
 * i.e. step j reads plane j & 1. A window of exactly @c fusedK steps skips its
 * recompute and its per-step launches; a short window keeps both.
 */
struct AdjointChain {
    idx_t fusedK = 0;      ///< steps per fused window, 0 for none.
    idx_t cur = 0;         ///< the plane the next launch reads.
    idx_t start = 0;       ///< first step of the current window.
    idx_t len = 0;         ///< its length.

    AdjointChain(idx_t steps, idx_t window, idx_t fusedKSteps) : fusedK(fusedKSteps) {
        idx_t flips = steps;
        if (fusedK > 0) {
            const WindowPlan plan = planWindows(steps, window);
            idx_t full = 0;
            for (idx_t k = 0; k < plan.count(); ++k) full += plan.length(k) == fusedK ? 1 : 0;
            flips = steps - full * fusedK + full;
        }
        cur = 1 ^ (flips & 1);
    }
    /** @brief Enter window [c, c+len); true when it is one fused launch. */
    bool begin(idx_t c, idx_t length) {
        start = c;
        len = length;
        return fused();
    }
    bool fused() const { return fusedK > 0 && len == fusedK; }
    /** @brief The one step of a fused window that issues the launch. */
    idx_t top() const { return start + len - 1; }
    void flip() { cur ^= 1; }
};

/** @brief acc[s] += g[s] for s < @p count: one step's share of a gradient
 *  that sums over steps (a parameter, a read-only input), in step order. */
template <class T>
void accumulatePrefix(T* acc, const T* g, idx_t count) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (count > 65536)
#endif
    for (idx_t s = 0; s < count; ++s) acc[s] += g[s];
}

/** @brief plane[s] = value for s < @p count: the step index a time-indexed
 *  step's vjp reads, the same for every sample still running. */
template <class T>
void fillPrefix(T* plane, T value, idx_t count) {
    for (idx_t s = 0; s < count; ++s) plane[s] = value;
}

#ifdef __CUDACC__
/** @name Device column ops: the same three column operations on the device,
 *  enqueued on a stream (capturable), for the CUDA backend's window driver. */
///@{
namespace device {

template <class T>
__global__ void gatherColumnsKernel(T* dst, idx_t dstLd, const T* src, idx_t srcLd, const idx_t* index,
                                    idx_t count, idx_t rows) {
    const idx_t i = static_cast<idx_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count * rows) return;
    const idx_t r = i / count, c = i % count;
    dst[r * dstLd + c] = src[r * srcLd + index[c]];
}

template <class T>
__global__ void accumulateKernel(T* acc, const T* g, idx_t ld, idx_t count, idx_t rows) {
    const idx_t i = static_cast<idx_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count * rows) return;
    const idx_t r = i / count, c = i % count;
    acc[r * ld + c] += g[r * ld + c];
}

template <class T>
__global__ void fillKernel(T* plane, T value, idx_t count) {
    const idx_t i = static_cast<idx_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) plane[i] = value;
}

inline unsigned gridFor(idx_t work) { return static_cast<unsigned>((work + 255) / 256); }

template <class T>
void gatherColumns(void* dst, idx_t dstLd, const void* src, idx_t srcLd, const idx_t* index, idx_t count,
                   idx_t rows, cudaStream_t stream) {
    if (count * rows == 0) return;
    gatherColumnsKernel<T><<<gridFor(count * rows), 256, 0, stream>>>(
        static_cast<T*>(dst), dstLd, static_cast<const T*>(src), srcLd, index, count, rows);
}

template <class T>
void accumulatePrefix(T* acc, const T* g, idx_t ld, idx_t count, idx_t rows, cudaStream_t stream) {
    if (count * rows == 0) return;
    accumulateKernel<T><<<gridFor(count * rows), 256, 0, stream>>>(acc, g, ld, count, rows);
}

template <class T>
void fillPrefix(T* plane, T value, idx_t count, cudaStream_t stream) {
    if (count == 0) return;
    fillKernel<T><<<gridFor(count), 256, 0, stream>>>(plane, value, count);
}

}  // namespace device
///@}
#endif

}  // namespace diff
}  // namespace eagle
