// SPDX-License-Identifier: Apache-2.0

// The device stable sort by integer key behind ``eg_order_by_key`` (nvcc only; included by the door's device/Loop.cu). The
// lanes are ordered by least-significant-digit radix passes over the bits a key bound needs. One read of the keys counts the
// digits of every pass (and the keys outside [0, bound)); each pass then scatters the lanes of a tile to their places, ranking the
// lanes of a tile that share a digit in lane order and finding the lanes of the tiles before it by a look-back over their counts,
// so every pass is stable and the result is one value on every run. The first place of each key (``starts``) is the digit's first
// place for a single pass, and a binary search of the sorted keys over the most significant digit's lanes for more.
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
constexpr unsigned kMaxPasses = 4;                           ///< passes over a 32-bit key
constexpr int kHistogramBlocks = 64;                          ///< blocks that count the digits, each over several tiles
constexpr long long kFewKeys = 16;                             ///< a bound this many times below the lanes gets ``first_places``
constexpr long long kShortStretch = 8;                       ///< keys a lane writes alone in ``fill_starts``

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

/** @brief Where the temporary arrays of a sort lie in its scratch (not zeroed). */
struct Layout {
    std::size_t bad = 0;              ///< a 32-bit count of keys outside the bound
    std::size_t key_a, idx_a, key_b, idx_b, digits, ticket, status, bytes;
};

inline Layout layout_of(const Shape& s) {
    Layout l;
    const std::size_t lanes = up16(static_cast<std::size_t>(s.n) * 4);
    l.key_a = 32;
    l.idx_a = l.key_a + lanes;
    l.key_b = l.idx_a + lanes;
    l.idx_b = l.key_b + lanes;
    l.digits = l.idx_b + lanes;                                                            // kMaxPasses * 256 words
    l.ticket = l.digits + kMaxPasses * (1u << kMaxWidth) * 4;                              // 16 bytes, then the status
    l.status = l.ticket + 16;
    l.bytes = l.status + up16(static_cast<std::size_t>(s.bins) * static_cast<std::size_t>(s.tiles) * 8);
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

/** @brief The bits of the digit of pass ``pass``: every pass takes ``shape.width`` bits but the last, which takes what is left. */
__host__ __device__ constexpr unsigned digit_bits(unsigned pass, unsigned passes, unsigned width, unsigned bits) {
    return pass + 1 == passes ? bits - pass * width : width;
}

/** @brief ``digits[pass * 256 + d]`` counts the lanes whose digit of that pass is ``d``, for every pass in one read of the typed keys,
 *  and ``bad`` counts the keys outside [0, bound). The blocks take the tiles in turn, so that few of them add their counts to the
 *  global ones. */
template <class Key>
__global__ void __launch_bounds__(kBlock) histogram(const Key* typed, long long n, long long tiles, unsigned passes, unsigned width, unsigned bits,
                                                    unsigned long long bound, std::uint32_t* digits, std::uint32_t* bad) {
    __shared__ unsigned local[kMaxPasses][1u << kMaxWidth];
    for (unsigned e = threadIdx.x; e < kMaxPasses * (1u << kMaxWidth); e += kBlock) (&local[0][0])[e] = 0;
    __syncthreads();
    unsigned outside_count = 0;
    for (long long tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const long long base = tile * kTile;
        Key held[kItems];  // all the loads are issued before the first count waits for one
#pragma unroll
        for (int j = 0; j < kItems; ++j) {
            const long long at = base + static_cast<long long>(j) * kBlock + threadIdx.x;
            if (at < n) held[j] = typed[at];
        }
#pragma unroll
        for (int j = 0; j < kItems; ++j) {
            if (base + static_cast<long long>(j) * kBlock + threadIdx.x >= n) continue;
            outside_count += outside<Key>(held[j], bound);
            const std::uint32_t key = static_cast<std::uint32_t>(held[j]);
            for (unsigned p = 0; p < passes; ++p) atomicAdd(&local[p][(key >> (p * width)) & ((1u << digit_bits(p, passes, width, bits)) - 1u)], 1u);
        }
    }
    if (outside_count != 0) atomicAdd(bad, outside_count);
    __syncthreads();
    for (unsigned p = 0; p < passes; ++p)
        for (unsigned d = threadIdx.x; d < (1u << digit_bits(p, passes, width, bits)); d += kBlock)
            if (local[p][d] != 0) atomicAdd(&digits[p * (1u << kMaxWidth) + d], local[p][d]);
}

/** @brief One block per pass: ``digits`` of the pass becomes its exclusive scan, the first place of every digit in the lanes sorted by
 *  that pass. A single pass is also the sort's ``starts`` (``bound + 1`` words), as the keys are the digits. */
__global__ void __launch_bounds__(kBlock) digit_bases(std::uint32_t* digits, unsigned passes, unsigned width, unsigned bits, long long n,
                                                      long long bound, std::uint32_t* starts) {
    __shared__ unsigned base[1u << kMaxWidth];
    __shared__ unsigned warp_total[kWarps];
    const unsigned pass = blockIdx.x;
    std::uint32_t* mine = digits + pass * (1u << kMaxWidth);
    const unsigned bins = 1u << digit_bits(pass, passes, width, bits);
    const unsigned count = threadIdx.x < bins ? mine[threadIdx.x] : 0;
    const int lane = threadIdx.x % scan::kWarp, warp = threadIdx.x / scan::kWarp;
    unsigned inclusive = count;
    for (int offset = 1; offset < scan::kWarp; offset <<= 1) {
        const unsigned below = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += below;
    }
    if (lane == scan::kWarp - 1) warp_total[warp] = inclusive;
    __syncthreads();
    unsigned before = 0;
    for (int w = 0; w < warp; ++w) before += warp_total[w];
    base[threadIdx.x] = before + inclusive - count;
    __syncthreads();
    if (threadIdx.x < bins) mine[threadIdx.x] = base[threadIdx.x];
    if (passes == 1)
        for (long long k = threadIdx.x; k <= bound; k += kBlock) starts[k] = k < bins ? base[k] : static_cast<std::uint32_t>(n);
}

/** @brief Status of a tile's count of one digit in ``scatter``: ``kAggregate`` is the tile's own count, ``kInclusive`` the count of all
 *  the tiles up to and including it; both ride in the high word of one 64-bit store, so a reader sees a flag with its value. */
constexpr unsigned long long kAggregate = 1ull << 32;
constexpr unsigned long long kInclusive = 2ull << 32;

/** @brief Place every lane of a tile at its sorted place for this digit: the digit's first place (``bases``) plus the lanes of the
 *  tiles before it with the same digit (found by a look-back over their counts, ``status``) plus the lanes of the tile before it with
 *  the same digit. Tiles are taken in order from ``ticket``, so the tiles looked back at have all begun. The first pass reads the typed
 *  keys and numbers the lanes itself; later passes read the 32-bit keys and lane numbers of the pass before. */
template <class Key, bool First>
__global__ void __launch_bounds__(kBlock) scatter(const Key* typed, const std::uint32_t* keys, const std::uint32_t* lanes, long long n, unsigned shift,
                                                  unsigned width, unsigned bins, const std::uint32_t* bases, unsigned long long* status,
                                                  std::uint32_t* ticket, std::uint32_t* out_keys, std::uint32_t* out_lanes) {
    __shared__ unsigned warp_count[kWarps][1u << kMaxWidth];
    __shared__ unsigned tile_offset[1u << kMaxWidth];
    __shared__ unsigned digit_start[1u << kMaxWidth];
    __shared__ unsigned warp_total[kWarps];
    __shared__ std::uint32_t stage_key[kTile], stage_lane[kTile];
    __shared__ unsigned taken;
    if (threadIdx.x == 0) taken = atomicAdd(ticket, 1u);
    for (unsigned d = threadIdx.x; d < bins; d += kBlock)
        for (int w = 0; w < kWarps; ++w) warp_count[w][d] = 0;
    __syncthreads();
    const long long tile = taken;
    const int warp = threadIdx.x / scan::kWarp, lane = threadIdx.x % scan::kWarp;
    const long long base = tile * kTile + static_cast<long long>(warp) * kItems * scan::kWarp;
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
    // Each digit's count over the whole tile, and each warp's start inside its digit (exclusive over the warps).
    unsigned total = 0;
    if (threadIdx.x < bins) {
        unsigned running = 0;
        for (int w = 0; w < kWarps; ++w) {
            const unsigned c = warp_count[w][threadIdx.x];
            warp_count[w][threadIdx.x] = running;
            running += c;
        }
        total = running;
        // The tile's own count is published before the look-back, so that no tile waits for one that waits for it.
        status[tile * bins + threadIdx.x] = (tile == 0 ? kInclusive : kAggregate) | total;
    }
    // Where each digit starts inside the tile once it is sorted: the exclusive scan of the digit totals over the threads.
    unsigned inclusive = total;
    for (int offset = 1; offset < scan::kWarp; offset <<= 1) {
        const unsigned below = __shfl_up_sync(0xffffffffu, inclusive, offset);
        if (lane >= offset) inclusive += below;
    }
    if (lane == scan::kWarp - 1) warp_total[warp] = inclusive;
    __syncthreads();
    unsigned before = 0;
    for (int w = 0; w < warp; ++w) before += warp_total[w];
    if (threadIdx.x < bins) {
        digit_start[threadIdx.x] = before + inclusive - total;
        // The lanes of the tiles before this one with this digit: the counts of the tiles back to one that knows its inclusive count.
        unsigned running = 0;
        for (long long t = tile - 1; t >= 0; --t) {
            unsigned long long word;
            do word = *reinterpret_cast<volatile unsigned long long*>(&status[t * bins + threadIdx.x]);
            while (word == 0);
            running += static_cast<unsigned>(word);
            if (word >= kInclusive) break;
        }
        if (tile != 0) status[tile * bins + threadIdx.x] = kInclusive | (running + total);
        tile_offset[threadIdx.x] = bases[threadIdx.x] + running;
    }
    __syncthreads();
    // The tile is held sorted by digit in shared memory, so that the writes below are runs of consecutive places.
    for (int j = 0; j < kItems; ++j) {
        const long long at = base + static_cast<long long>(j) * scan::kWarp + lane;
        if (at < n) {
            const unsigned local = digit_start[digit[j]] + warp_count[warp][digit[j]] + rank[j];
            stage_key[local] = key[j];
            stage_lane[local] = number[j];
        }
    }
    __syncthreads();
    const long long left = n - tile * kTile;
    const int held = left < kTile ? static_cast<int>(left) : static_cast<int>(kTile);
    for (int at = threadIdx.x; at < held; at += kBlock) {
        const std::uint32_t sorted_key = stage_key[at];
        const unsigned d = (sorted_key >> shift) & mask;
        const unsigned place = tile_offset[d] + static_cast<unsigned>(at) - digit_start[d];
        out_keys[place] = sorted_key;
        out_lanes[place] = stage_lane[at];
    }
}

/** @brief ``starts[key]`` is the first place of ``key`` in the sorted keys, for ``key`` in [0, bound]: a binary search of the sorted
 *  keys, over the lanes of the key's most significant digit, whose first places are ``top_bases``. It serves the bounds that are few
 *  beside the lanes, where one thread a key is cheaper than one thread a lane. */
__global__ void first_places(const std::uint32_t* sorted, long long n, long long bound, unsigned top_shift, unsigned top_bins,
                             const std::uint32_t* top_bases, std::uint32_t* starts) {
    for (long long k = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; k <= bound; k += static_cast<long long>(gridDim.x) * blockDim.x) {
        if (k == bound) {
            starts[k] = static_cast<std::uint32_t>(n);
            continue;
        }
        const unsigned top = static_cast<unsigned>(k >> top_shift);
        unsigned low = top_bases[top], high = top + 1 < top_bins ? top_bases[top + 1] : static_cast<unsigned>(n);
        while (low < high) {
            const unsigned middle = low + (high - low) / 2;
            if (sorted[middle] < k) low = middle + 1;
            else high = middle;
        }
        starts[k] = low;
    }
}

/** @brief ``starts[key]`` is the first place of ``key`` in the sorted keys, for ``key`` in [0, bound]: every place that starts a run of
 *  equal keys gives its place to the keys from the run before it, as far as its own key, and the keys past the last run start at ``n``.
 *  A warp writes the long stretches together, one lane's stretch at a time. */
__global__ void fill_starts(const std::uint32_t* sorted, long long n, long long bound, std::uint32_t* starts) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i - threadIdx.x % scan::kWarp <= n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        const long long key = i < n ? static_cast<long long>(sorted[i]) : bound;
        const long long before = i == 0 || i > n ? -1 : static_cast<long long>(sorted[i - 1]);
        const long long last = key < bound ? key : bound;
        const bool starts_run = i <= n && key > before && before < bound;
        const long long first = before + 1;
        const long long length = starts_run ? last - first + 1 : 0;
        for (long long k = first; k < first + length && length <= kShortStretch; ++k) starts[k] = static_cast<std::uint32_t>(i);
        unsigned long_lanes = __ballot_sync(0xffffffffu, length > kShortStretch);
        while (long_lanes != 0) {
            const int owner = __ffs(long_lanes) - 1;
            long_lanes &= long_lanes - 1;
            const long long from = __shfl_sync(0xffffffffu, first, owner), count = __shfl_sync(0xffffffffu, length, owner);
            const long long place = __shfl_sync(0xffffffffu, i, owner);
            for (long long k = threadIdx.x % scan::kWarp; k < count; k += scan::kWarp) starts[from + k] = static_cast<std::uint32_t>(place);
        }
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
 *  ``temp`` is ``layout_of(shape).bytes`` of memory in any state.
 *  The number of keys outside [0, bound) is left in the 32-bit word at ``temp`` (a count that is 0 when every key is in range). */
template <class Key>
inline void queue(cudaStream_t stream, const Key* typed, const Shape& s, std::uint32_t* order, std::uint32_t* starts, void* temp) {
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
    auto* digits = reinterpret_cast<std::uint32_t*>(t + at.digits);
    auto* ticket = reinterpret_cast<std::uint32_t*>(t + at.ticket);
    auto* status = reinterpret_cast<unsigned long long*>(t + at.status);
    const std::size_t status_bytes = static_cast<std::size_t>(s.bins) * static_cast<std::size_t>(s.tiles) * 8;
    cudaMemsetAsync(digits, 0, kMaxPasses * (1u << kMaxWidth) * 4, stream);
    histogram<Key><<<s.tiles < kHistogramBlocks ? s.tiles : kHistogramBlocks, kBlock, 0, stream>>>(typed, s.n, s.tiles, s.passes, s.width, s.bits, static_cast<unsigned long long>(s.bound), digits, bad);
    digit_bases<<<s.passes, kBlock, 0, stream>>>(digits, s.passes, s.width, s.bits, s.n, s.bound, starts);
    unsigned shift = 0;
    for (unsigned p = 0; p < s.passes; ++p, shift += s.width) {
        const unsigned width = digit_bits(p, s.passes, s.width, s.bits);
        const std::uint32_t* in_keys = keys[(p + 1) & 1];
        const std::uint32_t* in_lanes = lanes[(p + 1) & 1];
        std::uint32_t* out_keys = keys[p & 1];
        std::uint32_t* out_lanes = p + 1 == s.passes ? order : lanes[p & 1];
        cudaMemsetAsync(ticket, 0, 16 + status_bytes, stream);  // the ticket and the status that follows it
        const std::uint32_t* bases = digits + p * (1u << kMaxWidth);
        if (p == 0) scatter<Key, true><<<s.tiles, kBlock, 0, stream>>>(typed, nullptr, nullptr, s.n, shift, width, 1u << width, bases, status, ticket, out_keys, out_lanes);
        else scatter<Key, false><<<s.tiles, kBlock, 0, stream>>>(nullptr, in_keys, in_lanes, s.n, shift, width, 1u << width, bases, status, ticket, out_keys, out_lanes);
    }
    if (s.passes > 1) {
        const unsigned top = s.passes - 1;
        const std::uint32_t* sorted = keys[top & 1];
        if (s.bound * kFewKeys <= s.n)
            first_places<<<grid_of(s.bound + 1), 256, 0, stream>>>(sorted, s.n, s.bound, top * s.width, 1u << digit_bits(top, s.passes, s.width, s.bits),
                                                                  digits + top * (1u << kMaxWidth), starts);
        else fill_starts<<<grid_of(s.n + 1), 256, 0, stream>>>(sorted, s.n, s.bound, starts);
    }
}

}  // namespace order
}  // namespace cuda
}  // namespace eagle
