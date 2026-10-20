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
 *
 * **Fixed-point f64 leaf.** On devices with slow FP64 (``fixedLeaf``) a
 * double sum uses ``FixedPartial`` instead of (hi, comp): each term is added
 * exactly into a 128-bit integer with a per-partial binary unit, using only
 * integer instructions. The same tree, the same walk; host and device run the
 * same integer arithmetic, so the two agree bit for bit. Both policies run
 * the pairwise levels on this path (bitwise run to run).
 */

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/throw.h"

#include "aether/accum/atomic.h"
#include "aether/expr/Reduce.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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

/** @brief ``min(a, b)``, the exact fold beside ``aether::MaxOp`` (``a < b ? a : b``). */
template<typename T>
struct MinOp {
    AETHER_DEVICEHOST() constexpr T operator()(T a, T b) const { return a < b ? a : b; }
};

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

/**
 * @brief A partial of a double sum on the fixed-point path: the value is
 *        ``v * 2^u`` with ``v`` a signed 128-bit integer (``hi:lo``).
 *
 * A term's least significant bit is placed at most 53 bits above the unit,
 * so the unit sits 106 bits (u^2) below the largest term seen: a term (or a
 * partial, when two units are aligned) loses at most one unit, which keeps
 * every sum within the (1 + n*k*u) ULP contract and exact whenever the terms
 * span fewer than 2^53 in magnitude. A slot folds at most 2 * ``terms`` terms
 * of at most 2^106 units (no overflow); ``join`` halves both operands and
 * raises the unit when either reaches 2^125, so a partial never overflows.
 * Infinities and NaN set sticky flags and the value follows IEEE.
 */
struct FixedPartial {
    __extension__ typedef __int128 I;
    __extension__ typedef unsigned __int128 U;
    static constexpr std::int32_t empty = -100000; // the unit of a partial that saw no term
    static constexpr std::int32_t posInf = 1, negInf = 2, nan = 4;

    std::uint64_t lo;
    std::int64_t hi;
    std::int32_t u, fl;

    AETHER_DEVICEHOST() AETHER_FORCEINLINE() I get() const { return I(U(std::uint64_t(hi)) << 64 | U(lo)); }
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() void set(I v)
    {
        lo = std::uint64_t(U(v));
        hi = std::int64_t(v >> 64);
    }
    /** @brief ``v >> d`` (floor), any ``d >= 0``. */
    AETHER_DEVICEHOST() AETHER_FORCEINLINE() static I shr(I v, std::int32_t d) { return v >> (d < 127 ? d : 127); }

    AETHER_DEVICEHOST() AETHER_FORCEINLINE() static std::uint64_t bits(double x)
    {
#ifdef __CUDA_ARCH__
        return std::uint64_t(__double_as_longlong(x));
#else
        std::uint64_t b;
        std::memcpy(&b, &x, sizeof b);
        return b;
#endif
    }

    AETHER_DEVICEHOST() AETHER_FORCEINLINE() static double ofBits(std::uint64_t b)
    {
#ifdef __CUDA_ARCH__
        return __longlong_as_double(std::int64_t(b));
#else
        double x;
        std::memcpy(&x, &b, sizeof x);
        return x;
#endif
    }

    AETHER_DEVICEHOST() static FixedPartial of(double x, double xc)
    {
        FixedPartial p { 0, 0, empty, 0 };
        p.add(x);
        p.add(xc);
        return p;
    }
    AETHER_DEVICEHOST() void add(double x, double xc)
    {
        add(x);
        add(xc);
    }
    /** @brief Add one term exactly (truncated only below the unit). */
    AETHER_DEVICEHOST() void add(double x)
    {
        const std::uint64_t b = bits(x);
        const std::int32_t e  = std::int32_t((b >> 52) & 0x7FF);
        if (e == 0x7FF) {
            fl |= (b << 12) != 0 ? nan : ((b >> 63) != 0 ? negInf : posInf);
            return;
        }
        const std::uint64_t m = (b & 0xFFFFFFFFFFFFFull) | (e != 0 ? (1ull << 52) : 0ull);
        if (m == 0)
            return;
        const std::int32_t ex = (e != 0 ? e : 1) - 1075; // weight of m's last bit
        I v                   = get();
        if (ex - 53 > u) {
            v = shr(v, ex - 53 - u);
            u = ex - 53;
        }
        const std::int32_t sh = ex - u; // <= 53
        const I t = sh >= 0 ? I(U(m) << sh) : (-sh < 64 ? I(m >> -sh) : I(0));
        set((b >> 63) != 0 ? v - t : v + t);
    }
    /** @brief Align the units and add; symmetric in its operands. */
    AETHER_DEVICEHOST() static FixedPartial join(const FixedPartial& a, const FixedPartial& b)
    {
        I x = a.get(), y = b.get();
        std::int32_t u = a.u;
        if (b.u > u) {
            x = shr(x, b.u - u);
            u = b.u;
        } else {
            y = shr(y, u - b.u);
        }
        // |x| or |y| >= 2^125: halve both so the sum stays below 2^127
        if (std::uint64_t(std::int64_t(x >> 125) + 1) > 1 || std::uint64_t(std::int64_t(y >> 125) + 1) > 1) {
            x >>= 1;
            y >>= 1;
            ++u;
        }
        FixedPartial o;
        o.set(x + y);
        o.u  = u;
        o.fl = a.fl | b.fl;
        return o;
    }
    /** @brief The value rounded once to nearest-even (subnormals included). */
    AETHER_DEVICEHOST() double value() const
    {
        if (fl != 0) {
            if ((fl & nan) != 0 || (fl & (posInf | negInf)) == (posInf | negInf))
                return ofBits(0x7FF8000000000000ull);
            return ofBits((fl & posInf) != 0 ? 0x7FF0000000000000ull : 0xFFF0000000000000ull);
        }
        const I v = get();
        if (v == 0)
            return 0.0;
        const bool neg = v < 0;
        const U mag    = neg ? U(0) - U(v) : U(v);
        const std::uint64_t mh = std::uint64_t(mag >> 64), ml = std::uint64_t(mag);
#ifdef __CUDA_ARCH__
        const std::int32_t lz = mh != 0 ? __clzll(std::int64_t(mh)) : 64 + __clzll(std::int64_t(ml));
#else
        const std::int32_t lz = mh != 0 ? __builtin_clzll(mh) : 64 + __builtin_clzll(ml);
#endif
        const std::int32_t len  = 128 - lz;
        const std::int32_t sub  = len + u + 1074; // bits of weight >= 2^-1074
        const std::int32_t keep = sub < 53 ? sub : 53;
        const std::int32_t drop = len - keep;
        double r;
        if (drop <= 0) {
            r = ldexp(double(std::uint64_t(mag)), u);
        } else {
            U q = drop < 128 ? mag >> drop : U(0);
            const U rem = drop < 128 ? mag - (q << drop) : mag;
            if (drop <= 128) {
                const U half = U(1) << (drop - 1);
                if (rem > half || (rem == half && (q & 1) != 0))
                    ++q;
            }
            r = ldexp(double(std::uint64_t(q)), u + drop);
        }
        return neg ? -r : r;
    }
};

/** @brief FP32:FP64 throughput ratio from which a device takes the
 *  fixed-point f64 leaf (1:32 and 1:64 parts; 1:2 parts keep (hi, comp)). */
inline constexpr int fixedLeafRatio = 8;

/** @brief True when a device with FP32:FP64 throughput ratio
 *  ``fp32PerFp64`` reduces double sums on the fixed-point path. */
inline constexpr bool fixedLeaf(int fp32PerFp64)
{
    return fp32PerFp64 >= fixedLeafRatio;
}

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
template<typename T, typename OP, typename P = Partial<T, OP>, typename In, typename Cin = NoComp>
inline P leaf(const In& in, idx_t base, idx_t cnt, const Cin& cin = {})
{
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
template<typename T, typename OP, typename P = Partial<T, OP>, typename In, typename Cin = NoComp>
inline P partialRange(const In& in, idx_t begin, idx_t n, const Cin& cin = {})
{
    EAGLE_ASSERT(n >= 1, "canonical::partialRange: empty range");
    const idx_t nl = leafCount(n);
    std::vector<P> parts(nl);
#pragma omp parallel for schedule(static)
    for (idx_t l = 0; l < nl; ++l) {
        const idx_t b = l * leafSize;
        parts[l]      = leaf<T, OP, P>(in, begin + b, std::min(leafSize, n - b), cin);
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
template<typename T, typename OP, typename P, typename In, typename Cin>
AETHER_KERNEL()
__launch_bounds__(256) void leafKernel(In in, Cin cin, idx_t rows, idx_t cols, idx_t col0, P* out, P* acc)
{
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
    if constexpr (compensated_v<T, OP> && std::is_same_v<P, Partial<T, OP>>) {
        if (acc != nullptr) {
            const T old = atomicAdd(&acc[col].h, s[0].h);
            atomicAdd(&acc[col].c, eft::twoSumErrT_(old, s[0].h, old + s[0].h) + s[0].c);
            return;
        }
    }
    out[col * idx_t(gridDim.x) + idx_t(blockIdx.x)] = s[0];
}

/** @brief Nodes one ``levelsKernel`` block reduces (nine pairwise levels). */
inline constexpr idx_t levelGroup = 2 * slots;

/** @brief Nine pairwise levels of every column: block ``(g, y)`` reduces
 *  source nodes ``[g * levelGroup, ...)`` of column ``col0 + y`` (``m`` per
 *  column) to destination node ``g`` (``gridDim.x`` per column). A group is
 *  an aligned subtree of the levels, so the bits are those of level-by-level
 *  pairwise (an odd last node is carried). Launch with ``slots`` threads. */
template<typename P>
AETHER_KERNEL() __launch_bounds__(256) void levelsKernel(const P* src, P* dst, idx_t m, idx_t col0)
{
    __shared__ P s[levelGroup];
    const idx_t col  = col0 + idx_t(blockIdx.y);
    const idx_t base = idx_t(blockIdx.x) * levelGroup;
    const idx_t cnt  = (m - base < levelGroup) ? (m - base) : levelGroup;
    const idx_t t    = threadIdx.x;
    for (idx_t i = t; i < cnt; i += slots)
        s[i] = src[col * m + base + i];
    __syncthreads();
    for (idx_t off = 1; off < levelGroup; off <<= 1) {
        const idx_t i = t * 2 * off;
        if (i + off < cnt)
            s[i] = P::join(s[i], s[i + off]);
        __syncthreads();
    }
    if (t == 0)
        dst[col * idx_t(gridDim.x) + idx_t(blockIdx.x)] = s[0];
}

/** @brief Enqueue the pairwise levels of ``m`` nodes per column, ping-ponging
 *  ``src``/``dst``; returns the buffer holding the ``cols`` roots. */
template<typename P>
inline P* enqueueLevels(P* src, P* dst, idx_t m, idx_t cols, cudaStream_t stream)
{
    for (; m > 1; m = (m + levelGroup - 1) / levelGroup) {
        const idx_t groups = (m + levelGroup - 1) / levelGroup;
        for (idx_t c0 = 0; c0 < cols; c0 += 65535) {
            const dim3 grid(unsigned(groups), unsigned(std::min<idx_t>(cols - c0, 65535)));
            levelsKernel<P><<<grid, unsigned(slots), 0, stream>>>(src, dst, m, c0);
        }
        std::swap(src, dst);
    }
    return src;
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
                out[cols + j] = eft::isFinite_(root[j].h) ? root[j].c : T(0);
                continue;
            }
        }
        out[j] = root[j].value();
    }
}

/** @brief Elements of ``T`` each of the two device work buffers must hold
 *  for ``rows`` samples in each of ``cols`` columns (room for a
 *  ``FixedPartial`` per leaf, three doubles). */
inline idx_t deviceWorkElems(idx_t rows, idx_t cols = 1)
{
    return 3 * leafCount(rows) * cols;
}

/** @brief ``fixedLeaf`` of CUDA device ``device``. */
inline bool fixedLeafDevice(int device)
{
    int ratio = 0;
    if (cudaDeviceGetAttribute(&ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device) != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return fixedLeaf(ratio);
}

/** @brief ``enqueueTo`` with the partial type ``P`` fixed. */
template<typename T, typename OP, typename P, typename In, typename Cin>
inline void enqueueWith(const In& in, const Cin& cin, idx_t rows, idx_t cols, bool atomic,
    T* workA, T* workB, T* out, bool pairOut, cudaStream_t stream)
{
    const idx_t m0   = leafCount(rows);
    const unsigned g = unsigned(std::min<idx_t>((cols + 255) / 256, 65535));
    P* src           = reinterpret_cast<P*>(workA);
    P* dst           = reinterpret_cast<P*>(workB);
    if (atomic)
        zeroKernel<P><<<g, 256, 0, stream>>>(src, cols);
    for (idx_t c0 = 0; c0 < cols; c0 += 65535) {
        const dim3 grid(unsigned(m0), unsigned(std::min<idx_t>(cols - c0, 65535)));
        leafKernel<T, OP, P, In, Cin><<<grid, unsigned(slots), 0, stream>>>(
            in, cin, rows, cols, c0, src, atomic ? src : nullptr);
    }
    src = enqueueLevels(src, dst, atomic ? 1 : m0, cols, stream);
    storeKernel<T, P><<<g, 256, 0, stream>>>(src, out, cols, pairOut);
}

/**
 * @brief Enqueue the reduction of each column of the row-major device plane
 *        ``in`` (``rows >= 1`` by ``cols >= 1``) on ``stream`` into
 *        ``out[0, cols)`` (``2 * cols`` elements, hi then comp, with
 *        ``pairOut``). ``cin`` is the companion plane (``NoComp`` for none).
 *        ``fixed``: a double sum without ``pairOut`` runs the fixed-point
 *        leaf (``FixedPartial``) and the pairwise levels under every policy.
 *        Two device work buffers of ``deviceWorkElems(rows, cols)`` elements
 *        each. Capturable.
 */
template<typename T, typename OP, typename In, typename Cin = NoComp>
inline void enqueueTo(const In& in, const Cin& cin, idx_t rows, idx_t cols, Policy policy,
    T* workA, T* workB, T* out, bool pairOut, cudaStream_t stream, bool fixed = false)
{
    EAGLE_ASSERT(rows >= 1 && cols >= 1, "canonical::enqueueTo: empty plane");
    if constexpr (std::is_same_v<T, double> && compensated_v<T, OP>) {
        if (fixed && !pairOut) {
            enqueueWith<T, OP, FixedPartial>(in, cin, rows, cols, false, workA, workB, out, false, stream);
            return;
        }
    }
    const bool atomic = compensated_v<T, OP> && resolve(policy) == Policy::compensated;
    enqueueWith<T, OP, Partial<T, OP>>(in, cin, rows, cols, atomic, workA, workB, out, pairOut, stream);
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
    leafKernel<T, OP, P, In, NoComp><<<unsigned(m), unsigned(slots), 0, stream>>>(
        in, NoComp {}, n, 1, 0, reinterpret_cast<P*>(workA), nullptr);
    P* src = enqueueLevels(reinterpret_cast<P*>(workA), reinterpret_cast<P*>(workB), m, idx_t(1), stream);
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
