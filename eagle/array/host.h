// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file host.h
 * @brief The host twins of the array ops: ``#pragma omp parallel for simd`` over the SAME
 *        ``body<Op, T>`` the device kernels use. Build with ``-ffp-contract=off``.
 */

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "eagle/array/kernels.h"

namespace eagle::array::host {

template <Op op, class T>
inline void run(const Args& a)
{
#pragma omp parallel for simd schedule(static)
    for (std::int64_t j = 0; j < a.n; ++j) body<op, T>(a, j);
}

/// Inclusive prefix sum, modular in ``T``: per-thread chunk sums, a serial combine, a second pass.
template <class T>
inline void cumsum(const Args& a)
{
    const T* x = static_cast<const T*>(a.p1);
    T* d = static_cast<T*>(a.p0);
    const std::int64_t n = a.n;
    if (n <= 0) return;
    int threads = std::max(1, std::min<int>(omp_get_max_threads(), static_cast<int>((n + 4095) / 4096)));
    std::vector<T> sums(static_cast<std::size_t>(threads) + 1, T(0));
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        const std::int64_t lo = n * t / threads, hi = n * (t + 1) / threads;
        T acc = T(0);
        for (std::int64_t j = lo; j < hi; ++j) { acc = static_cast<T>(acc + x[j]); d[j] = acc; }
        sums[static_cast<std::size_t>(t) + 1] = acc;
#pragma omp barrier
#pragma omp single
        for (int k = 1; k <= threads; ++k) sums[static_cast<std::size_t>(k)] = static_cast<T>(sums[static_cast<std::size_t>(k)] + sums[static_cast<std::size_t>(k) - 1]);
        const T off = sums[static_cast<std::size_t>(t)];
        if (off != T(0))
            for (std::int64_t j = lo; j < hi; ++j) d[j] = static_cast<T>(d[j] + off);
    }
}

template <class T>
inline void count_nonzero(const Args& a)
{
    const T* x = static_cast<const T*>(a.p1);
    std::int64_t c = 0;
#pragma omp parallel for simd reduction(+ : c) schedule(static)
    for (std::int64_t j = 0; j < a.n; ++j) c += nonzero(x[j]) ? 1 : 0;
    *static_cast<std::int64_t*>(a.p0) = c;
}

/// Compaction: per-thread chunk counts, a serial prefix, then each thread writes its chunk's indices.
template <class T>
inline void flatnonzero(const Args& a)
{
    const T* x = static_cast<const T*>(a.p1);
    std::int32_t* out = static_cast<std::int32_t*>(a.p0);
    const std::int64_t n = a.n;
    int threads = std::max(1, std::min<int>(omp_get_max_threads(), static_cast<int>((n + 4095) / 4096)));
    std::vector<std::int64_t> start(static_cast<std::size_t>(threads) + 1, 0);
#pragma omp parallel num_threads(threads)
    {
        const int t = omp_get_thread_num();
        const std::int64_t lo = n * t / threads, hi = n * (t + 1) / threads;
        std::int64_t c = 0;
        for (std::int64_t j = lo; j < hi; ++j) c += nonzero(x[j]) ? 1 : 0;
        start[static_cast<std::size_t>(t) + 1] = c;
#pragma omp barrier
#pragma omp single
        for (int k = 1; k <= threads; ++k) start[static_cast<std::size_t>(k)] += start[static_cast<std::size_t>(k) - 1];
        std::int64_t at = start[static_cast<std::size_t>(t)];
        for (std::int64_t j = lo; j < hi; ++j)
            if (nonzero(x[j])) out[at++] = static_cast<std::int32_t>(j);
    }
    *static_cast<std::int64_t*>(const_cast<void*>(a.p3)) = start[static_cast<std::size_t>(threads)];
}

/// Run ``op`` on the host over ``a.n`` elements of dtype ``a.dtype``.
inline void run(Op op, const Args& a)
{
    check_op(op, a.dtype);
    with_type(a.dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        switch (op) {
        case Op::Add: run<Op::Add, T>(a); break;
        case Op::Fill: run<Op::Fill, T>(a); break;
        case Op::Sub: run<Op::Sub, T>(a); break;
        case Op::Mul: run<Op::Mul, T>(a); break;
        case Op::Max: run<Op::Max, T>(a); break;
        case Op::Eq: run<Op::Eq, T>(a); break;
        case Op::Ne: run<Op::Ne, T>(a); break;
        case Op::Gt: run<Op::Gt, T>(a); break;
        case Op::Where: run<Op::Where, T>(a); break;
        case Op::Clip: run<Op::Clip, T>(a); break;
        case Op::Arange: run<Op::Arange, T>(a); break;
        case Op::Linspace: run<Op::Linspace, T>(a); break;
        case Op::Take: run<Op::Take, T>(a); break;
        case Op::Scatter: run<Op::Scatter, T>(a); break;
        case Op::AddS: run<Op::AddS, T>(a); break;
        case Op::SubS: run<Op::SubS, T>(a); break;
        case Op::MulS: run<Op::MulS, T>(a); break;
        case Op::MaxS: run<Op::MaxS, T>(a); break;
        case Op::EqS: run<Op::EqS, T>(a); break;
        case Op::NeS: run<Op::NeS, T>(a); break;
        case Op::GtS: run<Op::GtS, T>(a); break;
        case Op::RSubS: run<Op::RSubS, T>(a); break;
        case Op::LtS: run<Op::LtS, T>(a); break;
        case Op::Cumsum: cumsum<T>(a); break;
        case Op::CountNz: count_nonzero<T>(a); break;
        case Op::FlatNz: flatnonzero<T>(a); break;
        }
    });
}

/// ``dst[j] = static_cast<D>(src[j])`` over ``n`` contiguous elements; dtype codes 0-9 (bool is a 0/1 byte).
inline void cast(void* dst, int dst_dtype, const void* src, int src_dtype, std::int64_t n)
{
    if (dst_dtype < 0 || dst_dtype >= kComputeDTypes || src_dtype < 0 || src_dtype >= kComputeDTypes)
        throw std::invalid_argument("eagle cast: only the compute dtypes (codes 0-9) cast");
    const auto typed = [](int code, auto&& f) {
        if (code == 0) return f(Tag<bool>{});
        return with_type(code, f);
    };
    typed(dst_dtype, [&](auto dt) {
        using D = typename decltype(dt)::type;
        typed(src_dtype, [&](auto st) {
            using S = typename decltype(st)::type;
            D* d = static_cast<D*>(dst);
            const S* s = static_cast<const S*>(src);
#pragma omp parallel for simd schedule(static)
            for (std::int64_t j = 0; j < n; ++j) d[j] = static_cast<D>(s[j]);
        });
    });
}

}  // namespace eagle::array::host
