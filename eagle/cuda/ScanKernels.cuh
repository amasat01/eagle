// SPDX-License-Identifier: Apache-2.0

// The device scan and compaction: one pass over the values with decoupled look-back, the kernels behind ``eg_scan`` and
// ``eg_compact`` on a device context, and the scan every device caller in eagle uses (nvcc only; included by cuda/Scan.h and by
// the door's device/Loop.cu). A tile of ``kBlock`` threads scans ``items`` values per thread and publishes its total; tiles are taken in ticket order, so a tile only ever waits for tiles that run before it. The sums are laid out so
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
namespace cuda {
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

/** @brief Mask bytes per thread of a compaction tile (one 16-byte vector). */
constexpr int kCompactItems = 16;
constexpr long long kCompactTile = static_cast<long long>(kBlock) * kCompactItems;

/** @brief The tiles of a compaction over ``n`` lanes. */
inline int compact_tiles_of(std::int64_t n) { return static_cast<int>((n + kCompactTile - 1) / kCompactTile); }

/** @brief The bytes of working memory for ``tiles`` tiles whose totals take ``cell_bytes`` each: the ticket and the finished-tile
 *  count, the tile totals and the group totals. It is all zero before a scan or compaction and is left all zero by it: one of at
 *  most one group of tiles has its last tile clear the few cells it used (no launch of its own); a larger one is followed by a
 *  clearing launch, which costs nothing next to its own run. */
inline std::size_t state_bytes_of(std::size_t tiles, std::size_t cell_bytes) {
    return 256 + (((tiles + (tiles + kGroup - 1) / kGroup) * cell_bytes + 15) / 16) * 16;    // the last tile clears in 16-byte steps
}

/** @brief The working memory of a scan over ``n`` values of ``T``. */
template <class T>
inline std::size_t state_bytes(std::int64_t n) { return state_bytes_of(static_cast<std::size_t>(tiles_of<T>(n)), sizeof(T) == 8 ? 16 : 8); }

/** @brief The working memory of a compaction over ``n`` lanes. */
inline std::size_t compact_state_bytes(std::int64_t n) { return state_bytes_of(static_cast<std::size_t>(compact_tiles_of(n)), 8); }

/** @brief The tile this block works on: tiles are drawn in the order blocks start, so a tile only waits for tiles drawn before it.
 *  A scan of one tile has no other tile to wait for and touches no working memory at all. */
__device__ __forceinline__ long long take_tile(void* state, long long tiles, unsigned* shared_slot) {
    if (tiles == 1) return 0;
    if (threadIdx.x == 0) *shared_slot = atomicAdd(static_cast<unsigned*>(state), 1u);
    __syncthreads();
    return *shared_slot;
}

/** @brief The block's fold of one value per thread: returns the fold of the values of the threads before this one and sets
 *  ``total`` to the fold of all of them. ``warp_fold`` is ``kBlock / kWarp`` values of shared memory. */
template <class T, int F>
__device__ T block_offsets(T mine, T* warp_fold, T& total) {
    using O = Op<T, F>;
    constexpr int kWarps = kBlock / kWarp;
    const int lane = threadIdx.x & (kWarp - 1), warp = threadIdx.x / kWarp;
    T seen = mine;    // the fold of the warp's threads up to this one
    for (int d = 1; d < kWarp; d <<= 1) {
        const T up = __shfl_up_sync(0xffffffffu, seen, d);
        if (lane >= d) seen = O::apply(up, seen);
    }
    const T before_in_warp = __shfl_up_sync(0xffffffffu, seen, 1);
    if (lane == kWarp - 1) warp_fold[warp] = seen;
    __syncthreads();
    if (warp == 0) {
        T w = lane < kWarps ? warp_fold[lane] : O::identity();
        for (int d = 1; d < kWarp; d <<= 1) {
            const T up = __shfl_up_sync(0xffffffffu, w, d);
            if (lane >= d) w = O::apply(up, w);
        }
        if (lane < kWarps) warp_fold[lane] = w;
    }
    __syncthreads();
    total = warp_fold[kWarps - 1];
    const T before_warp = warp > 0 ? warp_fold[warp - 1] : O::identity();
    return O::apply(before_warp, lane > 0 ? before_in_warp : O::identity());
}

/** @brief Publish tile ``t``'s ``total`` and leave in ``shared_prefix`` the fold of every earlier tile: the earlier groups in order
 *  (the last tile of the group before this one publishes their fold), then the earlier tiles of this group in order. The last
 *  tile of a group publishes the fold up to and including the group. One warp calls it (the first); ``cells`` is the state past
 *  its header, laid out as ``tiles`` tile totals then the group folds. */
template <class T, int F>
__device__ void tile_prefix(void* cells, long long tiles, long long t, T total, T* shared_prefix) {
    using O = Op<T, F>;
    const int lane = threadIdx.x & (kWarp - 1);
    void* tile_cells = cells;
    void* group_cells = static_cast<char*>(cells) + tiles * Cell<T>::words * 8;
    if (lane == 0) Cell<T>::put(tile_cells, t, total);
    const long long group = t / kGroup, first = group * kGroup;
    const T inner = fixed_fold<T, F>(tile_cells, first, t);
    T outer = O::identity();    // the fold of every earlier group
    if (group > 0) {
        while (!Cell<T>::peek(group_cells, group - 1, outer)) {}
    }
    if (lane == 0) *shared_prefix = O::apply(outer, inner);
    if (t == first + kGroup - 1 || t == tiles - 1) {
        if (lane == 0) Cell<T>::put(group_cells, group, O::apply(outer, O::apply(inner, total)));
    }
}

/** @brief The end of a tile. A scan of one group of tiles clears its own working memory: the tile that finishes last knows every
 *  other tile has read what it needs (a tile's reads of the cells were consumed before its own count). */
__device__ __forceinline__ void finish_tile(void* state, long long tiles, int cell_words, bool self_clean, bool* shared_last) {
    if (!self_clean || tiles == 1) return;
    unsigned* ticket = static_cast<unsigned*>(state);
    unsigned* finished = ticket + 1;
    __syncthreads();
    if (threadIdx.x == 0) *shared_last = atomicAdd(finished, 1u) == static_cast<unsigned>(tiles - 1);
    __syncthreads();
    if (*shared_last) {
        char* cells = static_cast<char*>(state) + 256;
        const long long groups = (tiles + kGroup - 1) / kGroup;
        const long long cell_bytes = (tiles + groups) * cell_words * 8;
        for (long long at = static_cast<long long>(threadIdx.x) * 16; at < cell_bytes; at += static_cast<long long>(kBlock) * 16)
            *reinterpret_cast<ulonglong2*>(cells + at) = make_ulonglong2(0ull, 0ull);
        if (threadIdx.x == 0) { *ticket = 0u; *finished = 0u; }
    }
}

/** @brief One scan: ``out[i]`` folds ``in[0..i]`` (or ``in[0..i-1]`` when ``exclusive``) with ``F``, started from its identity.
 *  ``vector``: ``in`` and ``out`` are 16-byte aligned, so eight-byte values move as 16-byte vectors. */
template <class T, int F>
__global__ void __launch_bounds__(kBlock) scan_tiles(const T* in, T* out, long long n, bool exclusive, bool vector, bool self_clean, void* state) {
    using O = Op<T, F>;
    constexpr int kItems = items_of<T>();
    __shared__ T stage[sizeof(T) == 8 ? 1 : kBlock * kItems];
    __shared__ T warp_fold[kBlock / kWarp];
    __shared__ T shared_prefix;
    __shared__ unsigned tile_slot;
    __shared__ bool last_tile;
    const long long tiles = (n + tile_values<T>() - 1) / tile_values<T>();
    const long long t = take_tile(state, tiles, &tile_slot);
    const long long base = t * tile_values<T>();
    const T identity = O::identity();

    T v[kItems];
    if constexpr (sizeof(T) == 8) {
        struct alignas(16) Pair { T x[2]; };
        const long long mine = base + static_cast<long long>(threadIdx.x) * kItems;
        if (vector && mine + kItems <= n) {
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
    T total;
    const T before_thread = block_offsets<T, F>(run, warp_fold, total);
    if (tiles == 1) {
        if (threadIdx.x == 0) shared_prefix = identity;
    } else if (threadIdx.x < kWarp) {
        tile_prefix<T, F>(static_cast<char*>(state) + 256, tiles, t, total, &shared_prefix);
    }
    __syncthreads();
    const T start = O::apply(shared_prefix, before_thread);

    if constexpr (sizeof(T) == 8) {
        struct alignas(16) Pair { T x[2]; };
        const long long mine = base + static_cast<long long>(threadIdx.x) * kItems;
        T res[kItems];
        for (int j = 0; j < kItems; ++j) res[j] = exclusive ? (j == 0 ? start : O::apply(start, v[j - 1])) : O::apply(start, v[j]);
        if (vector && mine + kItems <= n) {
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
    finish_tile(state, tiles, Cell<T>::words, self_clean, &last_tile);
}

/** @brief One compaction: the index of every lane whose mask byte is non-zero goes to ``kept`` in lane order and their number to
 *  ``count``; the rest of ``kept`` is not written. The tile's kept lanes start at the number kept by the tiles before it, so no
 *  intermediate array is written. ``vector``: ``keep`` is 16-byte aligned. */
inline __global__ void __launch_bounds__(kBlock) compact_tiles(const std::uint8_t* keep, std::uint32_t* kept, std::uint32_t* count, long long n, bool vector,
                                                        bool self_clean, void* state) {
    __shared__ std::uint32_t warp_fold[kBlock / kWarp];
    __shared__ std::uint32_t shared_prefix;
    __shared__ unsigned tile_slot;
    __shared__ bool last_tile;
    const long long tiles = (n + kCompactTile - 1) / kCompactTile;
    const long long t = take_tile(state, tiles, &tile_slot);
    const long long mine = t * kCompactTile + static_cast<long long>(threadIdx.x) * kCompactItems;
    std::uint8_t mask[kCompactItems];
    if (vector && mine + kCompactItems <= n) {
        const uint4 word = *reinterpret_cast<const uint4*>(keep + mine);
        __builtin_memcpy(mask, &word, kCompactItems);
    } else {
        for (int j = 0; j < kCompactItems; ++j) mask[j] = mine + j < n ? keep[mine + j] : std::uint8_t(0);
    }
    std::uint32_t mine_kept = 0;
    for (int j = 0; j < kCompactItems; ++j) mine_kept += mask[j] != 0;
    std::uint32_t total;
    const std::uint32_t before_thread = block_offsets<std::uint32_t, kSum>(mine_kept, warp_fold, total);
    if (tiles == 1) {
        if (threadIdx.x == 0) shared_prefix = 0;
    } else if (threadIdx.x < kWarp) {
        tile_prefix<std::uint32_t, kSum>(static_cast<char*>(state) + 256, tiles, t, total, &shared_prefix);
    }
    __syncthreads();
    std::uint32_t at = shared_prefix + before_thread;
    for (int j = 0; j < kCompactItems; ++j)
        if (mask[j] != 0) kept[at++] = static_cast<std::uint32_t>(mine + j);
    if (t == tiles - 1 && threadIdx.x == 0) *count = shared_prefix + total;
    finish_tile(state, tiles, 1, self_clean, &last_tile);
}

/** @brief Queue the scan of ``n`` values (above 0) on ``stream``; ``state`` is ``state_bytes`` of zeroed working memory and is zero
 *  again when the queued work has run. */
template <class T, int F>
inline void scan_queue(cudaStream_t stream, const T* in, T* out, std::int64_t n, bool exclusive, void* state) {
    const bool self_clean = tiles_of<T>(n) <= kGroup;
    const bool vector = reinterpret_cast<std::uintptr_t>(in) % 16 == 0 && reinterpret_cast<std::uintptr_t>(out) % 16 == 0;
    scan_tiles<T, F><<<tiles_of<T>(n), kBlock, 0, stream>>>(in, out, n, exclusive, vector, self_clean, state);
    if (!self_clean) cudaMemsetAsync(state, 0, state_bytes<T>(n), stream);
}

/** @brief Queue the compaction of the ``n`` mask bytes (above 0) on ``stream``; ``state`` is ``compact_state_bytes`` of zeroed
 *  working memory and is zero again when the queued work has run. */
inline void compact_queue(cudaStream_t stream, const std::uint8_t* keep, std::uint32_t* kept, std::uint32_t* count, std::int64_t n, void* state) {
    const bool self_clean = compact_tiles_of(n) <= kGroup;
    const bool vector = reinterpret_cast<std::uintptr_t>(keep) % 16 == 0;
    compact_tiles<<<compact_tiles_of(n), kBlock, 0, stream>>>(keep, kept, count, n, vector, self_clean, state);
    if (!self_clean) cudaMemsetAsync(state, 0, compact_state_bytes(n), stream);
}

}  // namespace scan
}  // namespace cuda
}  // namespace eagle
