// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file launch.h
 * @brief The device twins (nvcc only): one ``__global__`` per op over ``body<Op, T>``, and
 *        ``enqueue``, which only ENQUEUES on a stream (capture-legal: no allocation, no sync).
 *        Each op adds exactly one kernel node; ``Cumsum`` adds a memset of its scratch and one kernel
 *        (a decoupled look-back scan over 1024-element tiles, so there is no lane ceiling).
 */

#ifndef __CUDACC__
#error "eagle/array/launch.h is for nvcc"
#endif

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "eagle/array/kernels.h"

namespace eagle::array {
template <> struct IsFloat<__half> : std::true_type {};
template <> struct IsFloat<__nv_bfloat16> : std::true_type {};
}  // namespace eagle::array

namespace eagle::array::device {

template <Op op, class T>
__global__ void kernel(Args a)
{
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (j < a.n) body<op, T>(a, j);
}

template <Op op, class T>
inline void launch(const Args& a, cudaStream_t s)
{
    if (a.n <= 0) return;
    const std::int64_t g = (a.n + 255) / 256;
    if (g > 2147483647LL) throw std::invalid_argument("eagle array op: n is past the launch grid");
    kernel<op, T><<<static_cast<unsigned>(g), 256, 0, s>>>(a);
}

constexpr int kScanThreads = 256;
constexpr int kScanPerThread = static_cast<int>(kCumsumTile) / kScanThreads;

/// Decoupled look-back inclusive scan. Scratch (zeroed): ticket u32 | status u32[t] | aggregate u64[t] | prefix u64[t].
/// ``Flags``: the scanned values are ``x != 0`` and the kernel compacts the indices (FlatNz) instead of writing the scan.
template <class T, bool Flags = false>
__global__ void cumsum_kernel(Args a)
{
    __shared__ std::uint64_t sh[kScanThreads];
    __shared__ std::uint64_t exclusive;
    __shared__ unsigned tile_id;
    unsigned char* sc = static_cast<unsigned char*>(const_cast<void*>(a.p2));
    const std::int64_t tiles = cumsum_tiles(a.n);
    unsigned* ticket = reinterpret_cast<unsigned*>(sc);
    volatile unsigned* status = reinterpret_cast<volatile unsigned*>(sc + 16);
    volatile std::uint64_t* aggr = reinterpret_cast<volatile std::uint64_t*>(sc + 16 + ((tiles * 4 + 7) / 8) * 8);
    volatile std::uint64_t* prefix = aggr + tiles;
    if (threadIdx.x == 0) tile_id = atomicAdd(ticket, 1u);
    __syncthreads();
    const std::int64_t tile = tile_id;
    const T* x = static_cast<const T*>(a.p1);
    T* d = Flags ? nullptr : static_cast<T*>(a.p0);
    const std::int64_t base = tile * kCumsumTile + static_cast<std::int64_t>(threadIdx.x) * kScanPerThread;
    std::uint64_t v[kScanPerThread];
    std::uint64_t local = 0;
#pragma unroll
    for (int k = 0; k < kScanPerThread; ++k) {
        if constexpr (Flags) v[k] = base + k < a.n ? (nonzero(x[base + k]) ? 1ull : 0ull) : 0ull;
        else v[k] = base + k < a.n ? static_cast<std::uint64_t>(static_cast<std::int64_t>(x[base + k])) : 0ull;
        local += v[k];
    }
    sh[threadIdx.x] = local;
    __syncthreads();
    for (int off = 1; off < kScanThreads; off <<= 1) {   // Hillis-Steele over the thread sums
        const std::uint64_t add = threadIdx.x >= off ? sh[threadIdx.x - off] : 0ull;
        __syncthreads();
        sh[threadIdx.x] += add;
        __syncthreads();
    }
    if (threadIdx.x == kScanThreads - 1) {
        const std::uint64_t total = sh[kScanThreads - 1];
        if (tile == 0) { prefix[0] = total; __threadfence(); status[0] = 2; exclusive = 0; }
        else {
            aggr[tile] = total; __threadfence(); status[tile] = 1;
            std::uint64_t acc = 0;
            for (std::int64_t k = tile - 1; k >= 0; --k) {
                unsigned st;
                while ((st = status[k]) == 0) { }
                __threadfence();
                if (st == 2) { acc += prefix[k]; break; }
                acc += aggr[k];
            }
            exclusive = acc;
            prefix[tile] = acc + total; __threadfence(); status[tile] = 2;
        }
    }
    __syncthreads();
    std::uint64_t run = exclusive + (threadIdx.x ? sh[threadIdx.x - 1] : 0ull);
#pragma unroll
    for (int k = 0; k < kScanPerThread; ++k) {
        run += v[k];
        if (base + k < a.n) {
            if constexpr (Flags) {
                if (v[k]) static_cast<std::int32_t*>(a.p0)[run - 1] = static_cast<std::int32_t>(base + k);
                if (base + k == a.n - 1) *static_cast<std::int64_t*>(const_cast<void*>(a.p3)) = static_cast<std::int64_t>(run);
            } else d[base + k] = static_cast<T>(run);
        }
    }
}

template <class T, bool Flags = false>
inline void cumsum(const Args& a, cudaStream_t s)
{
    if (a.n <= 0) return;
    if (a.p2 == nullptr) throw std::invalid_argument("eagle cumsum: p2 must be zero-fillable scratch of cumsum_scratch_bytes(n) bytes");
    if (cudaMemsetAsync(const_cast<void*>(a.p2), 0, cumsum_scratch_bytes(a.n), s) != cudaSuccess)
        throw std::runtime_error("eagle cumsum: scratch memset failed");
    cumsum_kernel<T, Flags><<<static_cast<unsigned>(cumsum_tiles(a.n)), kScanThreads, 0, s>>>(a);
}

/// Count the nonzero elements into ``*p0`` (int64): a memset of the cell and one kernel (warp ballot, one atomic per warp).
template <class T>
__global__ void count_kernel(Args a)
{
    const std::int64_t j = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool flag = j < a.n && nonzero(static_cast<const T*>(a.p1)[j]);
    const unsigned ballot = __ballot_sync(0xffffffffu, flag);
    if ((threadIdx.x & 31) == 0 && ballot)
        atomicAdd(static_cast<unsigned long long*>(a.p0), static_cast<unsigned long long>(__popc(ballot)));
}

template <class T>
inline void count(const Args& a, cudaStream_t s)
{
    if (cudaMemsetAsync(a.p0, 0, 8, s) != cudaSuccess) throw std::runtime_error("eagle count_nonzero: cell memset failed");
    if (a.n > 0) count_kernel<T><<<static_cast<unsigned>((a.n + 255) / 256), 256, 0, s>>>(a);
}

/// Enqueue ``op`` over ``a.n`` elements of dtype ``a.dtype`` on ``s``.
inline void enqueue(Op op, const Args& a, cudaStream_t s)
{
    check_op(op, a.dtype);
    with_type<__half, __nv_bfloat16>(a.dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        switch (op) {
        case Op::Add: launch<Op::Add, T>(a, s); break;
        case Op::Fill: launch<Op::Fill, T>(a, s); break;
        case Op::Sub: launch<Op::Sub, T>(a, s); break;
        case Op::Mul: launch<Op::Mul, T>(a, s); break;
        case Op::Max: launch<Op::Max, T>(a, s); break;
        case Op::Eq: launch<Op::Eq, T>(a, s); break;
        case Op::Ne: launch<Op::Ne, T>(a, s); break;
        case Op::Gt: launch<Op::Gt, T>(a, s); break;
        case Op::Where: launch<Op::Where, T>(a, s); break;
        case Op::Clip: launch<Op::Clip, T>(a, s); break;
        case Op::Arange: launch<Op::Arange, T>(a, s); break;
        case Op::Linspace: launch<Op::Linspace, T>(a, s); break;
        case Op::Take: launch<Op::Take, T>(a, s); break;
        case Op::Scatter: launch<Op::Scatter, T>(a, s); break;
        case Op::AddS: launch<Op::AddS, T>(a, s); break;
        case Op::SubS: launch<Op::SubS, T>(a, s); break;
        case Op::MulS: launch<Op::MulS, T>(a, s); break;
        case Op::MaxS: launch<Op::MaxS, T>(a, s); break;
        case Op::EqS: launch<Op::EqS, T>(a, s); break;
        case Op::NeS: launch<Op::NeS, T>(a, s); break;
        case Op::GtS: launch<Op::GtS, T>(a, s); break;
        case Op::RSubS: launch<Op::RSubS, T>(a, s); break;
        case Op::LtS: launch<Op::LtS, T>(a, s); break;
        case Op::Cumsum: cumsum<T>(a, s); break;
        case Op::CountNz: count<T>(a, s); break;
        case Op::FlatNz: cumsum<T, true>(a, s); break;
        }
    });
}

}  // namespace eagle::array::device
