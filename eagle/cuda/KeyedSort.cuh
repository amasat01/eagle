// SPDX-License-Identifier: Apache-2.0

// The device stable sort by integer key behind ``eg_order_by_key`` (nvcc only; included by the door's device/Loop.cu). The
// lanes are ordered by least-significant-digit radix passes over the bits a key bound needs: each pass counts the digits of every
// tile of lanes, scans the counts (digit-major, with the single-pass scan of ``cuda/ScanKernels.cuh``) and scatters the lanes to
// their places, ranking the lanes of a tile that share a digit in lane order, so every pass is stable and the result is one value
// on every run. The first place of each key then comes from the runs of equal keys in the sorted order: a compaction finds the
// run starts, the run lengths are the counts and an exclusive scan of the counts is ``starts``. Keys outside [0, bound) are
// counted by the first pass.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "eagle/cuda/ScanKernels.cuh"

namespace eagle {
namespace cuda {
namespace order {

constexpr int kBlock = 256;                                  ///< threads of a tile: eight warps
constexpr int kWarps = kBlock / scan::kWarp;
constexpr int kItems = 16;                                   ///< lanes per thread
constexpr long long kTile = static_cast<long long>(kBlock) * kItems;
constexpr unsigned kMaxWidth = 8;                            ///< bits of a digit: at most 256 bins

/** @brief What a sort of ``n`` lanes with the key bound ``bound`` does: the bits its keys need, the passes over them and the tiles. */
struct Shape {
    long long n;
    long long bound;
    unsigned bits;     ///< ``ceil(log2(bound))``: 0 for a bound of 1
    unsigned passes;   ///< radix passes: ``ceil(bits / kMaxWidth)``
    unsigned width;    ///< bits of every digit but the last: ``ceil(bits / passes)``
    int tiles;
    unsigned bins;     ///< ``1 << width``
};

inline Shape shape_of(long long n, long long bound) {
    Shape s{ n, bound, 0, 0, 0, static_cast<int>((n + kTile - 1) / kTile), 1 };
    while ((1ll << s.bits) < bound) ++s.bits;
    s.passes = (s.bits + kMaxWidth - 1) / kMaxWidth;
    s.width = s.passes == 0 ? 0 : (s.bits + s.passes - 1) / s.passes;
    s.bins = 1u << s.width;
    return s;
}

/** @brief Round ``bytes`` up to a multiple of 16. */
constexpr std::size_t up16(std::size_t bytes) { return (bytes + 15) / 16 * 16; }

/** @brief The bytes of zeroed working memory (``scan::state_bytes``-style, left zero) the largest scan or compaction of a sort needs. */
inline std::size_t state_bytes(const Shape& s) {
    std::size_t most = scan::compact_state_bytes(s.n);
    const std::size_t counts = scan::state_bytes<std::uint32_t>(static_cast<long long>(s.bins) * s.tiles);
    const std::size_t starts = scan::state_bytes<std::uint32_t>(s.bound + 1);
    if (counts > most) most = counts;
    if (starts > most) most = starts;
    return up16(most);
}

/** @brief Where the temporary arrays of a sort lie in its scratch (not zeroed). */
struct Layout {
    std::size_t bad = 0;              ///< a 32-bit count of keys outside the bound
    std::size_t runs = 16;            ///< a 32-bit count of runs of equal keys
    std::size_t key_a, idx_a, key_b, idx_b, counts, mask, boundaries, bytes;
};

inline Layout layout_of(const Shape& s) {
    Layout l;
    const std::size_t lanes = up16(static_cast<std::size_t>(s.n) * 4);
    l.key_a = 32;
    l.idx_a = l.key_a + lanes;
    l.key_b = l.idx_a + lanes;
    l.idx_b = l.key_b + lanes;
    l.counts = l.idx_b + lanes;
    l.mask = l.counts + up16(static_cast<std::size_t>(s.bins) * static_cast<std::size_t>(s.tiles) * 4);
    l.boundaries = l.mask + up16(static_cast<std::size_t>(s.n));
    l.bytes = l.boundaries + lanes;
    return l;
}

/** @brief The lanes of the warp whose digit equals this lane's, among the lanes of ``active``. */
__device__ __forceinline__ unsigned match_digit(unsigned digit, unsigned width, unsigned active) {
    unsigned same = active;
    for (unsigned b = 0; b < width; ++b) {
        const unsigned bit = (digit >> b) & 1u;
        const unsigned ones = __ballot_sync(0xffffffffu, bit);
        same &= bit ? ones : ~ones;
    }
    return same;
}

template <class Key>
__device__ __forceinline__ bool outside(Key v, unsigned long long bound) {
    if constexpr (std::is_signed_v<Key>) return v < 0 || static_cast<unsigned long long>(v) >= bound;
    else return static_cast<unsigned long long>(v) >= bound;
}

/** @brief The digit counts of every tile: ``counts[digit * tiles + tile]``. The first pass reads the typed keys and counts the keys
 *  outside the bound into ``bad``; later passes read the 32-bit keys of the pass before. */
template <class Key, bool First>
__global__ void __launch_bounds__(kBlock) upsweep(const Key* typed, const std::uint32_t* keys, long long n, int tiles, unsigned shift,
                                                  unsigned width, unsigned long long bound, std::uint32_t* counts, std::uint32_t* bad) {
    __shared__ unsigned histogram[1u << kMaxWidth];
    for (unsigned d = threadIdx.x; d < (1u << width); d += kBlock) histogram[d] = 0;
    __syncthreads();
    const int warp = threadIdx.x / scan::kWarp, lane = threadIdx.x % scan::kWarp;
    const long long base = static_cast<long long>(blockIdx.x) * kTile + static_cast<long long>(warp) * kItems * scan::kWarp;
    const unsigned mask = (1u << width) - 1u;
    unsigned lanes_below = (1u << lane) - 1u;
    (void)lanes_below;
    for (int j = 0; j < kItems; ++j) {
        const long long at = base + static_cast<long long>(j) * scan::kWarp + lane;
        const bool valid = at < n;
        std::uint32_t key = 0;
        bool bad_key = false;
        if constexpr (First) {
            if (valid) {
                const Key v = typed[at];
                bad_key = outside<Key>(v, bound);
                key = static_cast<std::uint32_t>(v);
            }
        } else {
            if (valid) key = keys[at];
        }
        const unsigned digit = (key >> shift) & mask;
        const unsigned active = __ballot_sync(0xffffffffu, valid);
        const unsigned same = match_digit(digit, width, active);
        if (valid && (same & ((1u << lane) - 1u)) == 0) atomicAdd(&histogram[digit], static_cast<unsigned>(__popc(same)));
        if constexpr (First) {
            const unsigned bad_lanes = __ballot_sync(0xffffffffu, bad_key);
            if (lane == 0 && bad_lanes != 0) atomicAdd(bad, static_cast<unsigned>(__popc(bad_lanes)));
        }
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < (1u << width); d += kBlock) counts[static_cast<long long>(d) * tiles + blockIdx.x] = histogram[d];
}

/** @brief Place every lane of a tile at its sorted place for this digit: ``offsets[digit * tiles + tile]`` (the scan of the counts) plus
 *  the number of lanes of the tile that come before it with the same digit. The first pass reads the typed keys and numbers the
 *  lanes itself; later passes read the 32-bit keys and lane numbers of the pass before. */
template <class Key, bool First>
__global__ void __launch_bounds__(kBlock) downsweep(const Key* typed, const std::uint32_t* keys, const std::uint32_t* lanes, long long n, int tiles,
                                                    unsigned shift, unsigned width, const std::uint32_t* offsets, std::uint32_t* out_keys,
                                                    std::uint32_t* out_lanes) {
    __shared__ unsigned warp_count[kWarps][1u << kMaxWidth];
    __shared__ unsigned tile_offset[1u << kMaxWidth];
    const unsigned bins = 1u << width;
    for (unsigned d = threadIdx.x; d < bins; d += kBlock) {
        for (int w = 0; w < kWarps; ++w) warp_count[w][d] = 0;
        tile_offset[d] = offsets[static_cast<long long>(d) * tiles + blockIdx.x];
    }
    __syncthreads();
    const int warp = threadIdx.x / scan::kWarp, lane = threadIdx.x % scan::kWarp;
    const long long base = static_cast<long long>(blockIdx.x) * kTile + static_cast<long long>(warp) * kItems * scan::kWarp;
    const unsigned mask = (1u << width) - 1u;
    std::uint32_t key[kItems], number[kItems];
    unsigned digit[kItems], same[kItems], rank[kItems];
    // The loads and the matches of the rounds do not depend on each other: they are issued together, and only the counters chain.
    for (int j = 0; j < kItems; ++j) {
        const long long at = base + static_cast<long long>(j) * scan::kWarp + lane;
        key[j] = 0;
        number[j] = 0;
        if (at < n) {
            if constexpr (First) {
                key[j] = static_cast<std::uint32_t>(typed[at]);
                number[j] = static_cast<std::uint32_t>(at);
            } else {
                key[j] = keys[at];
                number[j] = lanes[at];
            }
        }
        digit[j] = (key[j] >> shift) & mask;
    }
    for (int j = 0; j < kItems; ++j) {
        const long long at = base + static_cast<long long>(j) * scan::kWarp + lane;
        same[j] = match_digit(digit[j], width, __ballot_sync(0xffffffffu, at < n));
    }
    for (int j = 0; j < kItems; ++j) {
        const long long at = base + static_cast<long long>(j) * scan::kWarp + lane;
        const unsigned before = warp_count[warp][digit[j]];
        rank[j] = before + static_cast<unsigned>(__popc(same[j] & ((1u << lane) - 1u)));
        __syncwarp();
        if (at < n && (same[j] & ((1u << lane) - 1u)) == 0) warp_count[warp][digit[j]] = before + static_cast<unsigned>(__popc(same[j]));
        __syncwarp();
    }
    __syncthreads();
    for (unsigned d = threadIdx.x; d < bins; d += kBlock) {
        unsigned running = 0;
        for (int w = 0; w < kWarps; ++w) {
            const unsigned c = warp_count[w][d];
            warp_count[w][d] = running;
            running += c;
        }
    }
    __syncthreads();
    for (int j = 0; j < kItems; ++j) {
        const long long at = base + static_cast<long long>(j) * scan::kWarp + lane;
        if (at < n) {
            const unsigned place = tile_offset[digit[j]] + warp_count[warp][digit[j]] + rank[j];
            out_keys[place] = key[j];
            out_lanes[place] = number[j];
        }
    }
}

/** @brief ``flags[j]`` is 1 where a run of equal keys starts in the sorted 32-bit keys. */
__global__ void run_starts(const std::uint32_t* sorted, long long n, std::uint8_t* flags) {
    for (long long j = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; j < n; j += static_cast<long long>(gridDim.x) * blockDim.x)
        flags[j] = j == 0 || sorted[j] != sorted[j - 1];
}

/** @brief ``counts[key]`` is the length of the run of ``key``, for the runs ``0 .. *runs - 1`` that start at ``starts_at``. */
__global__ void run_lengths(const std::uint32_t* starts_at, const std::uint32_t* runs, const std::uint32_t* sorted, long long n, unsigned long long bound,
                            std::uint32_t* counts) {
    const long long total = *runs;
    for (long long r = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; r < total; r += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long first = starts_at[r];
        const long long end = r + 1 < total ? starts_at[r + 1] : n;
        const unsigned long long key = sorted[first];
        if (key < bound) counts[key] = static_cast<std::uint32_t>(end - first);
    }
}

/** @brief A bound of 1: ``order`` is the lanes in order and the keys outside [0, 1) are counted. */
template <class Key>
__global__ void single_key(const Key* typed, long long n, std::uint32_t* order, std::uint32_t* bad) {
    const int lane = threadIdx.x % scan::kWarp;
    for (long long j = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; j - lane < n; j += static_cast<long long>(gridDim.x) * blockDim.x) {
        const bool valid = j < n;
        bool bad_key = false;
        if (valid) {
            order[j] = static_cast<std::uint32_t>(j);
            bad_key = outside<Key>(typed[j], 1ull);
        }
        const unsigned bad_lanes = __ballot_sync(0xffffffffu, bad_key);
        if (lane == 0 && bad_lanes != 0) atomicAdd(bad, static_cast<unsigned>(__popc(bad_lanes)));
    }
}

/** @brief ``starts`` of a bound of 1: [0, n]. */
__global__ void single_starts(long long n, std::uint32_t* starts) {
    starts[0] = 0;
    starts[1] = static_cast<std::uint32_t>(n);
}

inline unsigned grid_of(long long n) { return static_cast<unsigned>(n / 256 + 1 < 65535ll * 4 ? n / 256 + 1 : 65535ll * 4); }

/** @brief Queue the sort of the ``n`` (above 0) lanes whose keys are ``typed`` into ``order`` and ``starts`` on ``stream``.
 *  ``state`` is ``state_bytes`` of zeroed working memory (left zero), ``temp`` is ``layout_of(shape).bytes`` of memory in any state.
 *  The number of keys outside [0, bound) is left in the 32-bit word at ``temp`` (a count that is 0 when every key is in range). */
template <class Key>
inline void queue(cudaStream_t stream, const Key* typed, const Shape& s, std::uint32_t* order, std::uint32_t* starts, void* state, void* temp) {
    const Layout at = layout_of(s);
    char* t = static_cast<char*>(temp);
    auto* bad = reinterpret_cast<std::uint32_t*>(t + at.bad);
    cudaMemsetAsync(bad, 0, 4, stream);
    if (s.bits == 0) {
        single_key<Key><<<grid_of(s.n), 256, 0, stream>>>(typed, s.n, order, bad);
        single_starts<<<1, 1, 0, stream>>>(s.n, starts);
        return;
    }
    std::uint32_t* keys[2] = { reinterpret_cast<std::uint32_t*>(t + at.key_a), reinterpret_cast<std::uint32_t*>(t + at.key_b) };
    std::uint32_t* lanes[2] = { reinterpret_cast<std::uint32_t*>(t + at.idx_a), reinterpret_cast<std::uint32_t*>(t + at.idx_b) };
    auto* counts = reinterpret_cast<std::uint32_t*>(t + at.counts);
    const long long cells = static_cast<long long>(s.bins) * s.tiles;
    unsigned shift = 0;
    for (unsigned p = 0; p < s.passes; ++p, shift += s.width) {
        const unsigned width = p + 1 == s.passes ? s.bits - shift : s.width;
        const unsigned long long bound = static_cast<unsigned long long>(s.bound);
        const std::uint32_t* in_keys = keys[(p + 1) & 1];
        const std::uint32_t* in_lanes = lanes[(p + 1) & 1];
        std::uint32_t* out_keys = keys[p & 1];
        std::uint32_t* out_lanes = p + 1 == s.passes ? order : lanes[p & 1];
        if (p == 0) upsweep<Key, true><<<s.tiles, kBlock, 0, stream>>>(typed, nullptr, s.n, s.tiles, shift, width, bound, counts, bad);
        else upsweep<Key, false><<<s.tiles, kBlock, 0, stream>>>(nullptr, in_keys, s.n, s.tiles, shift, width, bound, counts, bad);
        scan::scan_queue<std::uint32_t, scan::kSum>(stream, counts, counts, static_cast<long long>(1u << width) * s.tiles, true, state);
        if (p == 0) downsweep<Key, true><<<s.tiles, kBlock, 0, stream>>>(typed, nullptr, nullptr, s.n, s.tiles, shift, width, counts, out_keys, out_lanes);
        else downsweep<Key, false><<<s.tiles, kBlock, 0, stream>>>(nullptr, in_keys, in_lanes, s.n, s.tiles, shift, width, counts, out_keys, out_lanes);
        (void)cells;
    }
    const std::uint32_t* sorted = keys[(s.passes - 1) & 1];
    auto* flags = reinterpret_cast<std::uint8_t*>(t + at.mask);
    auto* run_at = reinterpret_cast<std::uint32_t*>(t + at.boundaries);
    auto* runs = reinterpret_cast<std::uint32_t*>(t + at.runs);
    run_starts<<<grid_of(s.n), 256, 0, stream>>>(sorted, s.n, flags);
    scan::compact_queue(stream, flags, run_at, runs, s.n, state);
    cudaMemsetAsync(starts, 0, static_cast<std::size_t>(s.bound + 1) * 4, stream);
    run_lengths<<<grid_of(s.n), 256, 0, stream>>>(run_at, runs, sorted, s.n, static_cast<unsigned long long>(s.bound), starts);
    scan::scan_queue<std::uint32_t, scan::kSum>(stream, starts, starts, s.bound + 1, true, state);
}

}  // namespace order
}  // namespace cuda
}  // namespace eagle
