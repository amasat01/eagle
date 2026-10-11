// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file canonical.h
 * @brief The canonical fixed-order reduction tree, with a host face and a
 *        device face that compute the SAME tree.
 *
 * **Definition.** Cut the ``n`` samples, in sample order, into *leaf blocks* of
 * ``leafSize`` (= 256) consecutive samples (the last block may be short).
 *
 * 1. *Leaf.* A block is reduced by a fixed in-block tree over its slots
 *    ``0..255``: for ``off = 128, 64, ..., 1`` and every slot ``t < off`` whose
 *    partner ``t + off`` holds a sample, ``slot[t] = OP(slot[t], slot[t+off])``.
 *    A slot past the end of the data is absent, not padded (a padded
 *    ``+0.0`` would turn a ``-0.0`` result into ``+0.0``). This is exactly what
 *    one 256-thread CUDA block does in shared memory, and the host executes the
 *    same steps in the same order.
 * 2. *Pairwise levels.* The leaf partials, in block order, are combined level
 *    by level: node ``i`` of the next level is ``OP(v[2i], v[2i+1])``; an odd
 *    last node is carried up unchanged. Repeat until one value is left.
 *
 * The result is a function of the values in sample order only: it does not
 * depend on the OpenMP thread count, on how the host schedules the leaves, or
 * on the GPU grid. Cutting the input into *chunks* of ``leafSize * 2^q``
 * samples, reducing each chunk with this same tree and combining the chunk
 * results with the SAME pairwise function gives the identical bits (a
 * chunk is exactly a subtree). A left fold over the chunk results, or a chunk
 * size that is not ``leafSize * 2^q``, does not.
 *
 * ``OP`` is the aether functor shape (``OP{}(a, b)``). For a non-associative
 * ``OP`` (floating-point sum) the tree above IS the definition of the answer.
 */

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/throw.h"

#include <algorithm>
#include <vector>

#ifndef EAGLE_CPU_ONLY
#include <cuda_runtime_api.h>
#endif

namespace eagle {
namespace reduce {
namespace canonical {

/** @brief Samples per leaf block; also the CUDA block size of the device leaf.
 *  A power of two. */
inline constexpr idx_t leafSize = 256;

/** @brief True iff ``chunk`` is ``leafSize * 2^q`` for some ``q >= 0`` (the
 *  only chunk sizes that are bitwise-neutral). */
inline constexpr bool isAlignedChunk(idx_t chunk)
{
    return chunk >= leafSize && (chunk & (chunk - 1)) == 0;
}

/** @brief Number of leaf blocks covering ``n`` samples. */
inline constexpr idx_t leafCount(idx_t n)
{
    return (n + leafSize - 1) / leafSize;
}

/**
 * @brief Reduce one leaf block: the fixed in-block tree over ``cnt`` samples
 *        ``in(base) .. in(base + cnt - 1)``, ``1 <= cnt <= leafSize``.
 */
template<typename T, typename OP, typename In>
inline T leaf(const In& in, idx_t base, idx_t cnt)
{
    constexpr idx_t half = leafSize / 2;
    T s[half];
    // Level off = 128 fused with the load: s[t] = OP(in(t), in(t + 128)) for
    // slots with a partner, s[t] = in(t) otherwise (same operands, same order).
    const idx_t paired = cnt > half ? cnt - half : 0;
    const idx_t loaded = cnt < half ? cnt : half;
    for (idx_t t = 0; t < paired; ++t)
        s[t] = OP{}(in(base + t), in(base + t + half));
    for (idx_t t = paired; t < loaded; ++t)
        s[t] = in(base + t);
    for (idx_t off = half / 2; off > 0; off >>= 1) {
        const idx_t lim = cnt > off ? std::min(off, cnt - off) : 0; // has a partner (idx_t may be unsigned)
        for (idx_t t = 0; t < lim; ++t)
            s[t] = OP{}(s[t], s[t + off]);
    }
    return s[0];
}

/** @brief Level-by-level pairwise reduction of ``v[0..m)`` (``m >= 1``), in
 *  place; returns the root. An odd last node is carried. Order-fixed: the
 *  parallel levels only distribute independent pairs. */
template<typename T, typename OP>
inline T pairwise(T* v, idx_t m)
{
    while (m > 1) {
        const idx_t pairs = m / 2;
        // out[i] reads v[2i], v[2i+1] and writes v[i]: independent pairs only
        // when no write lands on a later read, i.e. ascending i. Parallel
        // levels therefore go through a separate buffer.
        if (pairs >= 8192) {
            std::vector<T> nxt((m + 1) / 2);
#pragma omp parallel for schedule(static)
            for (idx_t i = 0; i < pairs; ++i)
                nxt[i] = OP{}(v[2 * i], v[2 * i + 1]);
            if (m & 1)
                nxt[pairs] = v[m - 1];
            std::copy(nxt.begin(), nxt.end(), v);
        } else {
            for (idx_t i = 0; i < pairs; ++i)
                v[i] = OP{}(v[2 * i], v[2 * i + 1]);
            if (m & 1)
                v[pairs] = v[m - 1];
        }
        m = (m + 1) / 2;
    }
    return v[0];
}

/** @brief Pairwise-combine a vector of partials (e.g. chunk results). */
template<typename T, typename OP>
inline T combine(std::vector<T> partials)
{
    EAGLE_ASSERT(!partials.empty(), "canonical::combine: no partials");
    return pairwise<T, OP>(partials.data(), idx_t(partials.size()));
}

/**
 * @brief Host canonical reduction of ``in(begin) .. in(begin + n - 1)``,
 *        ``n >= 1``. OpenMP over leaves; bitwise independent of the thread
 *        count.
 */
template<typename T, typename OP, typename In>
inline T reduceRange(const In& in, idx_t begin, idx_t n)
{
    EAGLE_ASSERT(n >= 1, "canonical::reduceRange: empty range");
    const idx_t nl = leafCount(n);
    std::vector<T> parts(nl);
#pragma omp parallel for schedule(static)
    for (idx_t l = 0; l < nl; ++l) {
        const idx_t b = l * leafSize;
        parts[l]      = leaf<T, OP>(in, begin + b, std::min(leafSize, n - b));
    }
    return pairwise<T, OP>(parts.data(), nl);
}

/** @brief Host canonical reduction of ``in(0) .. in(n-1)``. */
template<typename T, typename OP, typename In>
inline T reduce(const In& in, idx_t n)
{
    return reduceRange<T, OP>(in, 0, n);
}

/**
 * @brief The same reduction processed in chunks of ``chunk`` samples
 *        (``chunk = leafSize * 2^q``; the last chunk may be ragged), chunk
 *        results combined by the pairwise tree. Bitwise equal to ``reduce``.
 */
template<typename T, typename OP, typename In>
inline T reduceChunked(const In& in, idx_t n, idx_t chunk)
{
    EAGLE_ASSERT(n >= 1, "canonical::reduceChunked: empty range");
    EAGLE_ASSERT(isAlignedChunk(chunk),
        "canonical::reduceChunked: chunk must be leafSize * 2^q");
    std::vector<T> parts;
    for (idx_t b = 0; b < n; b += chunk)
        parts.push_back(reduceRange<T, OP>(in, b, std::min(chunk, n - b)));
    return combine<T, OP>(std::move(parts));
}

#ifndef EAGLE_CPU_ONLY

/** @brief A raw device pointer as an ``in(i)`` accessor for the kernels. */
template<typename T>
struct DevicePtr {
    const T* p;
    AETHER_DEVICEHOST() T operator()(idx_t i) const { return p[i]; }
};

/** @brief Device leaf: one 256-thread block reduces one leaf block with the
 *  fixed in-block tree and writes ``out[blockIdx.x]``. Launch with
 *  ``<<<leafCount(n), leafSize>>>``. ``In`` is any device-callable ``in(i)``
 *  (an array view, or :cpp:class:`DevicePtr`). */
template<typename T, typename OP, typename In>
AETHER_KERNEL()
__launch_bounds__(256, 1) void leafKernel(
    In in, T* out, idx_t n)
{
    __shared__ T s[256];
    const idx_t base = idx_t(blockIdx.x) * leafSize;
    const idx_t cnt  = (n - base < leafSize) ? (n - base) : leafSize;
    const idx_t t    = threadIdx.x;
    if (t < cnt)
        s[t] = in(base + t);
    __syncthreads();
    for (idx_t off = leafSize / 2; off > 0; off >>= 1) {
        if (t < off && t + off < cnt)
            s[t] = OP{}(s[t], s[t + off]);
        __syncthreads();
    }
    if (t == 0)
        out[blockIdx.x] = s[0];
}

/** @brief One pairwise level on the device: ``dst[i] = OP(src[2i], src[2i+1])``
 *  (odd last node carried), ``m`` source nodes. */
template<typename T, typename OP>
AETHER_KERNEL() void levelKernel(const T* src, T* dst, idx_t m)
{
    const idx_t outN = (m + 1) / 2;
    for (idx_t i = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; i < outN;
         i += idx_t(gridDim.x) * blockDim.x)
        dst[i] = (2 * i + 1 < m) ? OP{}(src[2 * i], src[2 * i + 1]) : src[2 * i];
}

/** @brief ``*dst = *src``: the one-element store that moves the result to its
 *  destination without a runtime memcpy (capturable). */
template<typename T>
AETHER_KERNEL() void storeKernel(const T* src, T* dst) { *dst = *src; }

/** @brief Elements each of the two device work buffers must hold for ``n``
 *  samples. */
inline idx_t deviceWorkElems(idx_t n) { return leafCount(n); }

/**
 * @brief Enqueue the canonical reduction of the ``n >= 1`` device samples of
 *        ``in`` on ``stream`` using two device work buffers of
 *        ``deviceWorkElems(n)`` elements each.
 * @return device pointer (one of the work buffers) whose element 0 holds the
 *         result once the stream has run.
 */
template<typename T, typename OP, typename In>
inline T* enqueue(const In& in, idx_t n, T* workA, T* workB,
    cudaStream_t stream)
{
    EAGLE_ASSERT(n >= 1, "canonical::enqueue: empty range");
    idx_t m = leafCount(n);
    leafKernel<T, OP, In><<<unsigned(m), unsigned(leafSize), 0, stream>>>(
        in, workA, n);
    T* src = workA;
    T* dst = workB;
    while (m > 1) {
        const idx_t outN = (m + 1) / 2;
        const unsigned g = unsigned(std::min<idx_t>((outN + 255) / 256, 65535));
        levelKernel<T, OP><<<g, 256, 0, stream>>>(src, dst, m);
        std::swap(src, dst);
        m = outN;
    }
    return src;
}

/** @brief ``enqueue`` followed by a one-element store of the result into
 *  ``out`` (a device pointer distinct from the work buffers). */
template<typename T, typename OP, typename In>
inline void enqueueTo(const In& in, idx_t n, T* workA, T* workB, T* out,
    cudaStream_t stream)
{
    const T* res = enqueue<T, OP>(in, n, workA, workB, stream);
    storeKernel<T><<<1, 1, 0, stream>>>(res, out);
}

#endif // EAGLE_CPU_ONLY

} // namespace canonical
} // namespace reduce
} // namespace eagle
