// SPDX-License-Identifier: Apache-2.0

// The device scan: one pass over the values with decoupled look-back, the kernels behind ``eg_scan`` on a device context
// (included by device/Loop.cu only; nvcc). A tile of ``kBlock`` threads scans ``items`` values per thread and publishes its
// total; tiles are taken in ticket order, so a tile only ever waits for tiles that run before it. The sums are laid out so
// that their grouping depends on the extent alone, never on which tile finished first: a tile's prefix is the fold of every
// earlier GROUP of ``kGroup`` tiles (published, in group order, by the last tile of the group before it) then the fold of the
// totals of the tiles before it in its own group, taken in one fixed order (``fixed_fold``). A real sum is therefore the same
// bits on every run.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace eagle {
namespace door {
namespace scan {

constexpr int kBlock = 128;   ///< threads of a tile (a multiple of the warp, at most 32 warps)
constexpr int kGroup = 128;   ///< tiles per group; a group publishes the fold up to itself once, in group order
constexpr int kWarp = 32;

enum Fold { kSum = 0, kMin = 1, kMax = 2 };

/** @brief The fold ``F`` over ``T``: an integer sum wraps; a real minimum or maximum skips a NaN (``fmin``, ``fmax``). */
template <class T, int F>
struct Op {
    __host__ __device__ static T identity() {
        if constexpr (F == kSum) return T(0);
        constexpr bool low = F == kMax;
        if constexpr (std::is_floating_point_v<T>) return low ? -INFINITY : INFINITY;
        else {
            using U = std::make_unsigned_t<T>;
            const T lowest = std::is_signed_v<T> ? static_cast<T>(static_cast<U>(~static_cast<U>(0)) ^ static_cast<U>(~static_cast<U>(0) >> 1)) : T(0);
            return low ? lowest : static_cast<T>(~lowest);
        }
    }
    __host__ __device__ static T apply(T a, T b) {
        if constexpr (F == kSum) {
            if constexpr (std::is_integral_v<T>) {
                using U = std::make_unsigned_t<T>;
                return static_cast<T>(static_cast<U>(a) + static_cast<U>(b));
            } else {
                return a + b;
            }
        } else if constexpr (std::is_floating_point_v<T>) {
            if constexpr (sizeof(T) == 4) return F == kMin ? ::fminf(a, b) : ::fmaxf(a, b);
            else return F == kMin ? ::fmin(a, b) : ::fmax(a, b);
        } else {
            return F == kMin ? (b < a ? b : a) : (a < b ? b : a);
        }
    }
};

/** @brief The bits of a tile's published total: a 4-byte value rides in the same 8-byte word as its flag, an 8-byte value
 *  takes a 16-byte word; either is written and read as one volatile access, so a reader sees the value its flag names. */
template <class T>
struct Cell {
    static constexpr int words = sizeof(T) == 8 ? 2 : 1;
    __device__ static void put(void* cells, long long i, T v) {
        if constexpr (sizeof(T) == 8) {
            unsigned long long x;
            __builtin_memcpy(&x, &v, 8);
            asm volatile("st.volatile.global.v2.u64 [%0], {%1, %2};" ::"l"(static_cast<ulonglong2*>(cells) + i), "l"(x), "l"(1ull) : "memory");
        } else {
            unsigned x;
            __builtin_memcpy(&x, &v, 4);
            asm volatile("st.volatile.global.u64 [%0], %1;" ::"l"(static_cast<unsigned long long*>(cells) + i),
                         "l"((1ull << 32) | x) : "memory");
        }
    }
    /** One look at cell ``i``: true and its value when it is published. */
    __device__ static bool peek(const void* cells, long long i, T& v) {
        if constexpr (sizeof(T) == 8) {
            unsigned long long x, f;
            asm volatile("ld.volatile.global.v2.u64 {%0, %1}, [%2];" : "=l"(x), "=l"(f) : "l"(static_cast<const ulonglong2*>(cells) + i) : "memory");
            __builtin_memcpy(&v, &x, 8);
            return f != 0ull;
        } else {
            unsigned long long w;
            asm volatile("ld.volatile.global.u64 %0, [%1];" : "=l"(w) : "l"(static_cast<const unsigned long long*>(cells) + i) : "memory");
            const unsigned x = static_cast<unsigned>(w);
            __builtin_memcpy(&v, &x, 4);
            return (w >> 32) != 0ull;
        }
    }
};

/** @brief The fold of the published cells ``first .. last - 1``, in one order that depends on the cell count alone: lane ``l`` of
 *  the warp folds the cells ``first + l``, ``first + l + 32``, ... in turn, then the lanes fold by a fixed butterfly. A lane looks
 *  at ``kLook`` of its cells at once and waits for the ones not yet published. Every lane of the warp calls it and every lane
 *  returns the result. */
template <class T, int F>
__device__ T fixed_fold(const void* cells, long long first, long long last) {
    constexpr int kLook = 2;
    T mine = Op<T, F>::identity();
    for (long long i = first + (threadIdx.x & (kWarp - 1)); i < last; i += static_cast<long long>(kWarp) * kLook) {
        T seen[kLook];
        bool ready[kLook];
#pragma unroll
        for (int k = 0; k < kLook; ++k) {
            const long long j = i + static_cast<long long>(k) * kWarp;
            ready[k] = j >= last || Cell<T>::peek(cells, j, seen[k]);
        }
#pragma unroll
        for (int k = 0; k < kLook; ++k) {
            const long long j = i + static_cast<long long>(k) * kWarp;
            if (j < last) {
                while (!ready[k]) ready[k] = Cell<T>::peek(cells, j, seen[k]);
                mine = Op<T, F>::apply(mine, seen[k]);
            }
        }
    }
    for (int d = kWarp / 2; d > 0; d >>= 1) mine = Op<T, F>::apply(mine, __shfl_xor_sync(0xffffffffu, mine, d));
    return mine;
}

/** @brief Items per thread of a tile of ``T``: 12 four-byte values (staged through shared memory) or 8 eight-byte values (read
 *  and written as 16-byte vectors straight from memory); measured against the toolkit's scan at four million values. */
template <class T>
__host__ __device__ constexpr int items_of() { return sizeof(T) == 8 ? 8 : 12; }

template <class T>
__host__ __device__ constexpr long long tile_values() { return static_cast<long long>(kBlock) * items_of<T>(); }

/** @brief The tiles of a scan over ``n`` values. */
template <class T>
inline int tiles_of(std::int64_t n) { return static_cast<int>((n + tile_values<T>() - 1) / tile_values<T>()); }

/** @brief The bytes of working memory one scan over ``n`` values needs (to be zeroed before the launch): the ticket, the tile
 *  totals and the group totals. */
template <class T>
inline std::size_t state_bytes(std::int64_t n) {
    const std::size_t tiles = static_cast<std::size_t>(tiles_of<T>(n));
    const std::size_t cell = sizeof(T) == 8 ? 16 : 8;
    return 256 + (tiles + (tiles + kGroup - 1) / kGroup) * cell;
}

/** @brief One scan: ``out[i]`` folds ``in[0..i]`` (or ``in[0..i-1]`` when ``exclusive``) with ``F``, started from its identity. */
template <class T, int F>
__global__ void __launch_bounds__(kBlock) scan_tiles(const T* in, T* out, long long n, bool exclusive, void* state) {
    using O = Op<T, F>;
    constexpr int kItems = items_of<T>();
    constexpr int kWarps = kBlock / kWarp;
    __shared__ T stage[sizeof(T) == 8 ? 1 : kBlock * kItems];
    __shared__ T warp_fold[kWarps];
    __shared__ T tile_prefix;
    __shared__ unsigned tile_id;
    unsigned* ticket = static_cast<unsigned*>(state);
    char* cells = static_cast<char*>(state) + 256;
    const long long tiles = (n + tile_values<T>() - 1) / tile_values<T>();
    const long long groups = (tiles + kGroup - 1) / kGroup;
    (void)groups;
    void* tile_cells = cells;
    void* group_cells = cells + tiles * Cell<T>::words * 8;
    if (threadIdx.x == 0) tile_id = atomicAdd(ticket, 1u);
    __syncthreads();
    const long long t = tile_id;
    const long long base = t * tile_values<T>();
    const T identity = O::identity();

    T v[kItems];
    if constexpr (sizeof(T) == 8) {
        struct alignas(16) Pair { T x[2]; };
        const long long mine = base + static_cast<long long>(threadIdx.x) * kItems;
        if (mine + kItems <= n) {
            const Pair* src = reinterpret_cast<const Pair*>(in + mine);
            for (int k = 0; k < kItems / 2; ++k) {
                const Pair p = src[k];
                v[2 * k] = p.x[0];
                v[2 * k + 1] = p.x[1];
            }
        } else {
            for (int j = 0; j < kItems; ++j) v[j] = mine + j < n ? in[mine + j] : identity;
        }
    } else {
        for (int i = 0; i < kItems; ++i) {
            const long long at = base + i * kBlock + threadIdx.x;
            stage[i * kBlock + threadIdx.x] = at < n ? in[at] : identity;
        }
        __syncthreads();
        for (int j = 0; j < kItems; ++j) v[j] = stage[threadIdx.x * kItems + j];
    }
    T run = identity;    // v[j] becomes the fold of this thread's values up to j
    for (int j = 0; j < kItems; ++j) { run = O::apply(run, v[j]); v[j] = run; }
    const int lane = threadIdx.x & (kWarp - 1), warp = threadIdx.x / kWarp;
    T seen = run;    // the fold of the warp's threads up to this one
    for (int d = 1; d < kWarp; d <<= 1) {
        const T up = __shfl_up_sync(0xffffffffu, seen, d);
        if (lane >= d) seen = O::apply(up, seen);
    }
    const T before_in_warp = __shfl_up_sync(0xffffffffu, seen, 1);
    if (lane == kWarp - 1) warp_fold[warp] = seen;
    __syncthreads();
    if (warp == 0) {
        T w = lane < kWarps ? warp_fold[lane] : identity;
        for (int d = 1; d < kWarp; d <<= 1) {
            const T up = __shfl_up_sync(0xffffffffu, w, d);
            if (lane >= d) w = O::apply(up, w);
        }
        if (lane < kWarps) warp_fold[lane] = w;
    }
    __syncthreads();
    const T total = warp_fold[kWarps - 1];
    const T before_warp = warp > 0 ? warp_fold[warp - 1] : identity;
    const T before_thread = O::apply(before_warp, lane > 0 ? before_in_warp : identity);

    // The tile's prefix: the earlier groups in order, then the earlier tiles of this group in order.
    if (warp == 0) {
        if (lane == 0) Cell<T>::put(tile_cells, t, total);
        const long long group = t / kGroup, first = group * kGroup;
        const T inner = fixed_fold<T, F>(tile_cells, first, t);
        T outer = identity;    // the fold of every earlier group, published by the last tile of the group before this one
        if (group > 0) {
            while (!Cell<T>::peek(group_cells, group - 1, outer)) {}
        }
        if (lane == 0) tile_prefix = O::apply(outer, inner);
        if (t == first + kGroup - 1 || t == tiles - 1) {    // the last tile of its group publishes the fold up to and including the group
            if (lane == 0) Cell<T>::put(group_cells, group, O::apply(outer, O::apply(inner, total)));
        }
    }
    __syncthreads();
    const T start = O::apply(tile_prefix, before_thread);

    if constexpr (sizeof(T) == 8) {
        struct alignas(16) Pair { T x[2]; };
        const long long mine = base + static_cast<long long>(threadIdx.x) * kItems;
        T res[kItems];
        for (int j = 0; j < kItems; ++j) res[j] = exclusive ? (j == 0 ? start : O::apply(start, v[j - 1])) : O::apply(start, v[j]);
        if (mine + kItems <= n) {
            Pair* dst = reinterpret_cast<Pair*>(out + mine);
            for (int k = 0; k < kItems / 2; ++k) {
                Pair p;
                p.x[0] = res[2 * k];
                p.x[1] = res[2 * k + 1];
                dst[k] = p;
            }
        } else {
            for (int j = 0; j < kItems; ++j)
                if (mine + j < n) out[mine + j] = res[j];
        }
    } else {
        for (int j = 0; j < kItems; ++j)
            stage[threadIdx.x * kItems + j] = exclusive ? (j == 0 ? start : O::apply(start, v[j - 1])) : O::apply(start, v[j]);
        __syncthreads();
        for (int i = 0; i < kItems; ++i) {
            const long long at = base + i * kBlock + threadIdx.x;
            if (at < n) out[at] = stage[i * kBlock + threadIdx.x];
        }
    }
}

/** @brief Queue the clearing of the working memory ``state`` and the scan of ``n`` values on ``stream`` (``n`` above 0). */
template <class T, int F>
inline void scan_queue(cudaStream_t stream, const T* in, T* out, std::int64_t n, bool exclusive, void* state) {
    cudaMemsetAsync(state, 0, state_bytes<T>(n), stream);
    scan_tiles<T, F><<<tiles_of<T>(n), kBlock, 0, stream>>>(in, out, n, exclusive, state);
}

}  // namespace scan
}  // namespace door
}  // namespace eagle
