// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file canonical.h
 * @brief The canonical fixed-order reduction tree (the ``ordered`` policy),
 *        with a host face and a device face that compute the SAME tree, and
 *        the ``compensated`` device commit that shares its leaf.
 *
 * **Definition.** Cut the ``n`` samples, in sample order, into *leaves* of
 * ``leafSize = slots * terms`` (= 256 * 16) consecutive samples (the last
 * leaf may be short).
 *
 * 1. *Slot walk.* Slot ``t`` of a leaf holds samples ``t, t + 256, ...``
 *    (at most ``terms`` of them) folded in that order. A slot with no sample
 *    is absent.
 * 2. *In-leaf tree.* For ``off = 128, 64, ..., 1`` and every slot ``t < off``
 *    whose partner ``t + off`` is present, ``slot[t] = join(slot[t],
 *    slot[t + off])``. An absent partner is carried, never padded.
 * 3. *Pairwise levels.* The leaf partials, in leaf order, are combined level
 *    by level: node ``i`` of the next level is ``join(v[2i], v[2i+1])``; an
 *    odd last node is carried up unchanged.
 *
 * For a floating-point sum a partial is a (hi, comp) pair: the slot walk is
 * a Neumaier walk, ``join`` is the compensated pair-combine, and the value is
 * ``hi + comp``. Every other operator (integer sums, max, all) folds plainly;
 * those are exact, so the tree only fixes the cost.
 *
 * The result is a function of the values in sample order only: it does not
 * depend on the OpenMP thread count, on the GPU grid, or on host vs device.
 * Chunks of ``leafSize * 2^q`` samples are subtrees, so combining their
 * partials with the same pairwise levels gives the identical bits.
 *
 * ``compensated`` keeps the leaf and commits each leaf partial with two-sum
 * atomics into one (hi, comp) pair per column instead of running the levels:
 * one launch fewer, not bitwise run to run. The host runs ``ordered`` for
 * both policies.
 */

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/throw.h"

#include "aether/accum/atomic.h"
#include "aether/expr/Reduce.h"

#include <algorithm>
#include <cstdint>
#include <type_traits>
#include <vector>

#ifndef EAGLE_CPU_ONLY
#include <cuda_runtime_api.h>
#endif

namespace eagle {
namespace reduce {

/** @brief How a floating-point sum is finished. Integer sums, max and all are
 *  exact and ignore it. ``auto_`` is ``compensated`` in this release. */
enum class Policy : std::uint32_t { compensated = 0, ordered = 1, auto_ = 2 };

/** @brief The policy ``p`` runs as. */
inline constexpr Policy resolve(Policy p)
{
    return p == Policy::auto_ ? Policy::compensated : p;
}

namespace canonical {

namespace eft = aether::accum::atomic::detail; // the compensated primitives

/** @brief Slots per leaf; also the CUDA block size of the device leaf. */
inline constexpr idx_t slots = 256;
/** @brief Samples one slot folds (its stride is ``slots``). */
inline constexpr idx_t terms = 16;
/** @brief Samples per leaf. A power of two. */
inline constexpr idx_t leafSize = slots * terms;

/** @brief True iff ``chunk`` is ``leafSize * 2^q`` for some ``q >= 0`` (the
 *  only chunk sizes that are bitwise-neutral). */
inline constexpr bool isAlignedChunk(idx_t chunk)
{
    return chunk >= leafSize && (chunk & (chunk - 1)) == 0;
}

/** @brief Number of leaves covering ``n`` samples. */
inline constexpr idx_t leafCount(idx_t n)
{
    return (n + leafSize - 1) / leafSize;
}

/** @brief True when ``OP`` over ``T`` is a floating-point sum (compensated
 *  partials). */
template<typename T, typename OP>
inline constexpr bool compensated_v
    = std::is_floating_point_v<T> && std::is_same_v<OP, aether::SumOp<T>>;

/** @brief A partial of an exact operator: one plain value. */
template<typename T, typename OP, bool C = compensated_v<T, OP>>
struct Partial {
    T h;
    AETHER_DEVICEHOST() static Partial of(T x, T) { return { x }; }
    AETHER_DEVICEHOST() void add(T x) { h = OP {}(h, x); }
    AETHER_DEVICEHOST() void add(T x, T) { add(x); }
    AETHER_DEVICEHOST() static Partial join(const Partial& a, const Partial& b)
    {
        return { OP {}(a.h, b.h) };
    }
    AETHER_DEVICEHOST() T value() const { return h; }
};

/** @brief A partial of a floating-point sum: (hi, comp). A non-finite ``hi``
 *  drops ``comp``, so the value then follows IEEE through ``hi``. */
template<typename T, typename OP>
struct Partial<T, OP, true> {
    T h, c;
    AETHER_DEVICEHOST() static Partial of(T x, T xc)
    {
        return { x, eft::isFinite_(x) ? xc : T(0) };
    }
    AETHER_DEVICEHOST() void add(T x) { eft::neumaierAdd_(h, c, x); }
    AETHER_DEVICEHOST() void add(T x, T xc)
    {
        eft::neumaierAdd_(h, c, x);
        c = eft::isFinite_(h) ? c + xc : T(0);
    }
    AETHER_DEVICEHOST() static Partial join(const Partial& a, const Partial& b)
    {
        Partial o;
        eft::pairCombine_(a.h, a.c, b.h, b.c, o.h, o.c);
        return o;
    }
    AETHER_DEVICEHOST() T value() const { return eft::isFinite_(h) ? h + c : h; }
};

/** @brief "No companion input": the terms are plain values. */
struct NoComp { };

/** @brief Fold sample ``i`` (and its companion, if any) into ``p``. */
template<typename P, typename In, typename Cin>
inline void fold(P& p, bool first, const In& in, const Cin& cin, idx_t i)
{
    if constexpr (std::is_same_v<Cin, NoComp>) {
        if (first)
            p = P::of(in(i), std::decay_t<decltype(in(i))>(0));
        else
            p.add(in(i));
    } else {
        if (first)
            p = P::of(in(i), cin(i));
        else
            p.add(in(i), cin(i));
    }
}

/**
 * @brief Reduce one leaf: ``cnt`` samples ``in(base) .. in(base + cnt - 1)``,
 *        ``1 <= cnt <= leafSize``, to its partial.
 */
template<typename T, typename OP, typename In, typename Cin = NoComp>
inline Partial<T, OP> leaf(const In& in, idx_t base, idx_t cnt, const Cin& cin = {})
{
    using P          = Partial<T, OP>;
    const idx_t live = cnt < slots ? cnt : slots;
    P s[slots];
    for (idx_t t = 0; t < live; ++t)
        for (idx_t k = t; k < cnt; k += slots)
            fold(s[t], k == t, in, cin, base + k);
    for (idx_t off = slots / 2; off > 0; off >>= 1) {
        const idx_t lim = live > off ? std::min(off, live - off) : 0; // has a partner
        for (idx_t t = 0; t < lim; ++t)
            s[t] = P::join(s[t], s[t + off]);
    }
    return s[0];
}

/** @brief Level-by-level pairwise reduction of ``v[0..m)`` (``m >= 1``), in
 *  place; returns the root. An odd last node is carried. */
template<typename P>
inline P pairwise(P* v, idx_t m)
{
    while (m > 1) {
        const idx_t pairs = m / 2;
        // Parallel levels go through a separate buffer (out[i] would land on
        // a later read in place).
        if (pairs >= 8192) {
            std::vector<P> nxt((m + 1) / 2);
#pragma omp parallel for schedule(static)
            for (idx_t i = 0; i < pairs; ++i)
                nxt[i] = P::join(v[2 * i], v[2 * i + 1]);
            if (m & 1)
                nxt[pairs] = v[m - 1];
            std::copy(nxt.begin(), nxt.end(), v);
        } else {
            for (idx_t i = 0; i < pairs; ++i)
                v[i] = P::join(v[2 * i], v[2 * i + 1]);
            if (m & 1)
                v[pairs] = v[m - 1];
        }
        m = (m + 1) / 2;
    }
    return v[0];
}

/** @brief Pairwise-combine partials (e.g. chunk results, in chunk order). */
template<typename P>
inline P combine(std::vector<P> partials)
{
    EAGLE_ASSERT(!partials.empty(), "canonical::combine: no partials");
    return pairwise(partials.data(), idx_t(partials.size()));
}

/**
 * @brief Host canonical partial of ``in(begin) .. in(begin + n - 1)``,
 *        ``n >= 1``. OpenMP over leaves; bitwise independent of the thread
 *        count.
 */
template<typename T, typename OP, typename In, typename Cin = NoComp>
inline Partial<T, OP> partialRange(const In& in, idx_t begin, idx_t n, const Cin& cin = {})
{
    EAGLE_ASSERT(n >= 1, "canonical::partialRange: empty range");
    const idx_t nl = leafCount(n);
    std::vector<Partial<T, OP>> parts(nl);
#pragma omp parallel for schedule(static)
    for (idx_t l = 0; l < nl; ++l) {
        const idx_t b = l * leafSize;
        parts[l]      = leaf<T, OP>(in, begin + b, std::min(leafSize, n - b), cin);
    }
    return pairwise(parts.data(), nl);
}

/** @brief Host canonical reduction of ``in(begin) .. in(begin + n - 1)``. */
template<typename T, typename OP, typename In>
inline T reduceRange(const In& in, idx_t begin, idx_t n)
{
    return partialRange<T, OP>(in, begin, n).value();
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
 *        partials combined by the pairwise levels. Bitwise equal to ``reduce``.
 */
template<typename T, typename OP, typename In>
inline T reduceChunked(const In& in, idx_t n, idx_t chunk)
{
    EAGLE_ASSERT(n >= 1, "canonical::reduceChunked: empty range");
    EAGLE_ASSERT(isAlignedChunk(chunk),
        "canonical::reduceChunked: chunk must be leafSize * 2^q");
    std::vector<Partial<T, OP>> parts;
    for (idx_t b = 0; b < n; b += chunk)
        parts.push_back(partialRange<T, OP>(in, b, std::min(chunk, n - b)));
    return combine(std::move(parts)).value();
}

/** @brief Column ``j`` of a row-major ``(rows, cols)`` host plane as ``in(i)``. */
template<typename In>
struct Column {
    In in;
    idx_t cols, j;
    auto operator()(idx_t i) const { return in(i * cols + j); }
};

#ifndef EAGLE_CPU_ONLY

/** @brief A raw device pointer as an ``in(i)`` accessor for the kernels. */
template<typename T>
struct DevicePtr {
    const T* p;
    AETHER_DEVICEHOST() T operator()(idx_t i) const { return p[i]; }
};

/**
 * @brief Device leaf: block ``(l, y)`` reduces leaf ``l`` of column
 *        ``col0 + y`` of a row-major ``(rows, cols)`` plane with the slot walk
 *        and the in-leaf tree of the definition. ``acc == nullptr``: writes
 *        the partial to ``out[col * leafCount(rows) + l]`` (``ordered``);
 *        otherwise commits it into ``acc[col]`` with two-sum atomics
 *        (``compensated``). Launch with ``slots`` threads.
 */
template<typename T, typename OP, typename In, typename Cin>
AETHER_KERNEL()
__launch_bounds__(256) void leafKernel(In in, Cin cin, idx_t rows, idx_t cols, idx_t col0,
    Partial<T, OP>* out, Partial<T, OP>* acc)
{
    using P = Partial<T, OP>;
    __shared__ P s[slots];
    const idx_t col  = col0 + idx_t(blockIdx.y);
    const idx_t base = idx_t(blockIdx.x) * leafSize;
    const idx_t cnt  = (rows - base < leafSize) ? (rows - base) : leafSize;
    const idx_t live = cnt < slots ? cnt : slots;
    const idx_t t    = threadIdx.x;
    const auto x = [&](idx_t i) { return in(i * cols + col); };
    if (t < live) {
        // loads first (independent), then the fixed walk
        T v[terms];
#pragma unroll
        for (idx_t k = 0; k < terms; ++k)
            v[k] = (t + k * slots < cnt) ? x(base + t + k * slots) : T(0);
        P p;
        if constexpr (std::is_same_v<Cin, NoComp>) {
            p = P::of(v[0], T(0));
#pragma unroll
            for (idx_t k = 1; k < terms; ++k)
                if (t + k * slots < cnt)
                    p.add(v[k]);
        } else {
            const auto xc = [&](idx_t i) { return cin(i * cols + col); };
            p = P::of(v[0], xc(base + t));
#pragma unroll
            for (idx_t k = 1; k < terms; ++k)
                if (t + k * slots < cnt)
                    p.add(v[k], xc(base + t + k * slots));
        }
        s[t] = p;
    }
    __syncthreads();
    for (idx_t off = slots / 2; off > 0; off >>= 1) {
        if (t < off && t + off < live)
            s[t] = P::join(s[t], s[t + off]);
        __syncthreads();
    }
    if (t != 0)
        return;
    if constexpr (compensated_v<T, OP>) {
        if (acc != nullptr) {
            const T old = atomicAdd(&acc[col].h, s[0].h);
            atomicAdd(&acc[col].c, eft::twoSumErrT_(old, s[0].h, old + s[0].h) + s[0].c);
            return;
        }
    }
    out[col * idx_t(gridDim.x) + idx_t(blockIdx.x)] = s[0];
}

/** @brief One pairwise level of every column: ``m`` source nodes per column,
 *  ``(m + 1) / 2`` destination nodes (odd last node carried). */
template<typename P>
AETHER_KERNEL() void levelKernel(const P* src, P* dst, idx_t m, idx_t cols)
{
    const idx_t outN = (m + 1) / 2;
    for (idx_t g = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; g < outN * cols;
         g += idx_t(gridDim.x) * blockDim.x) {
        const idx_t j = g / outN, i = g - j * outN;
        const P* s    = src + j * m;
        dst[g]        = (2 * i + 1 < m) ? P::join(s[2 * i], s[2 * i + 1]) : s[2 * i];
    }
}

/** @brief Zero the ``cols`` (hi, comp) accumulators of ``compensated``. */
template<typename P>
AETHER_KERNEL() void zeroKernel(P* acc, idx_t cols)
{
    for (idx_t j = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; j < cols;
         j += idx_t(gridDim.x) * blockDim.x)
        acc[j] = P::of(0, 0);
}

/** @brief Write the ``cols`` roots: ``out[j] = value`` or, with ``pairOut``,
 *  ``out[j] = hi`` and ``out[cols + j] = comp``. */
template<typename T, typename P>
AETHER_KERNEL() void storeKernel(const P* root, T* out, idx_t cols, bool pairOut)
{
    for (idx_t j = idx_t(blockIdx.x) * blockDim.x + threadIdx.x; j < cols;
         j += idx_t(gridDim.x) * blockDim.x) {
        if constexpr (sizeof(P) == 2 * sizeof(T)) {
            if (pairOut) {
                out[j]        = root[j].h;
                out[cols + j] = root[j].c;
                continue;
            }
        }
        out[j] = root[j].value();
    }
}

/** @brief Elements of ``T`` each of the two device work buffers must hold
 *  for ``rows`` samples in each of ``cols`` columns. */
inline idx_t deviceWorkElems(idx_t rows, idx_t cols = 1)
{
    return 2 * leafCount(rows) * cols;
}

/**
 * @brief Enqueue the reduction of each column of the row-major device plane
 *        ``in`` (``rows >= 1`` by ``cols >= 1``) on ``stream`` into
 *        ``out[0, cols)`` (``2 * cols`` elements, hi then comp, with
 *        ``pairOut``). ``cin`` is the companion plane (``NoComp`` for none).
 *        Two device work buffers of ``deviceWorkElems(rows, cols)`` elements
 *        each. Capturable.
 */
template<typename T, typename OP, typename In, typename Cin = NoComp>
inline void enqueueTo(const In& in, const Cin& cin, idx_t rows, idx_t cols, Policy policy,
    T* workA, T* workB, T* out, bool pairOut, cudaStream_t stream)
{
    using P = Partial<T, OP>;
    EAGLE_ASSERT(rows >= 1 && cols >= 1, "canonical::enqueueTo: empty plane");
    const bool atomic = compensated_v<T, OP> && resolve(policy) == Policy::compensated;
    const idx_t m0    = leafCount(rows);
    const unsigned g  = unsigned(std::min<idx_t>((cols + 255) / 256, 65535));
    P* src            = reinterpret_cast<P*>(workA);
    P* dst            = reinterpret_cast<P*>(workB);
    if (atomic)
        zeroKernel<P><<<g, 256, 0, stream>>>(src, cols);
    for (idx_t c0 = 0; c0 < cols; c0 += 65535) {
        const dim3 grid(unsigned(m0), unsigned(std::min<idx_t>(cols - c0, 65535)));
        leafKernel<T, OP, In, Cin><<<grid, unsigned(slots), 0, stream>>>(
            in, cin, rows, cols, c0, src, atomic ? src : nullptr);
    }
    for (idx_t m = atomic ? 1 : m0; m > 1; m = (m + 1) / 2) {
        const idx_t outN = (m + 1) / 2;
        const unsigned gl = unsigned(std::min<idx_t>((outN * cols + 255) / 256, 65535));
        levelKernel<P><<<gl, 256, 0, stream>>>(src, dst, m, cols);
        std::swap(src, dst);
    }
    storeKernel<T, P><<<g, 256, 0, stream>>>(src, out, cols, pairOut);
}

/** @brief The ``ordered`` 1-D reduction of ``in(0) .. in(n-1)``, enqueued;
 *  returns the device pointer (a work buffer) whose element 0 holds the result
 *  once the stream has run. */
template<typename T, typename OP, typename In>
inline T* enqueue(const In& in, idx_t n, T* workA, T* workB, cudaStream_t stream)
{
    using P = Partial<T, OP>;
    EAGLE_ASSERT(n >= 1, "canonical::enqueue: empty range");
    idx_t m = leafCount(n);
    leafKernel<T, OP, In, NoComp><<<unsigned(m), unsigned(slots), 0, stream>>>(
        in, NoComp {}, n, 1, 0, reinterpret_cast<P*>(workA), nullptr);
    P* src = reinterpret_cast<P*>(workA);
    P* dst = reinterpret_cast<P*>(workB);
    for (; m > 1; m = (m + 1) / 2) {
        const unsigned g = unsigned(std::min<idx_t>(((m + 1) / 2 + 255) / 256, 65535));
        levelKernel<P><<<g, 256, 0, stream>>>(src, dst, m, 1);
        std::swap(src, dst);
    }
    // the value goes to the buffer not holding the root
    T* res = reinterpret_cast<T*>(src == reinterpret_cast<P*>(workA) ? workB : workA);
    storeKernel<T, P><<<1, 1, 0, stream>>>(src, res, 1, false);
    return res;
}

/** @brief The ``ordered`` 1-D reduction of ``in(0) .. in(n-1)`` into
 *  ``*out`` (a device pointer distinct from the work buffers). */
template<typename T, typename OP, typename In>
inline void enqueueTo(const In& in, idx_t n, T* workA, T* workB, T* out,
    cudaStream_t stream)
{
    enqueueTo<T, OP>(in, NoComp {}, n, 1, Policy::ordered, workA, workB, out, false, stream);
}

#endif // EAGLE_CPU_ONLY

} // namespace canonical
} // namespace reduce
} // namespace eagle
