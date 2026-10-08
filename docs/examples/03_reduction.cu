// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// 03_reduction.cu  (CUDA mode)
//
// Sum an aether SoA array with EAGLE's multi-level GPU reduction. The pure-C++
// twin (03_reduction.cpp) computes the same result with the cache-padded
// OpenMP reduction — same array, same operator, one dual-mode API.
#include <cstdio>

#include <eagle/eagle.h>

int main()
{
    using T   = double;
    using Sum = aether::SumOp<T>;
    constexpr eagle::idx_t N = 1000;

    // Fill a device-backed array with 1, 2, ..., N on the host, then upload.
    auto arr = eagle::makeArray<T>(N);
    {
        auto a = arr.hostView();
        for (eagle::idx_t i = 0; i < N; ++i)
            a(i) = static_cast<T>(i + 1);
    }
    arr.upload();

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // [cell:reduce]
    // Blocking multi-level reduction on the GPU: returns the scalar sum.
    const T total = eagle::cuda::Reduction::reduceBlocking<T, Sum>(arr, T(0), stream);
    // [cell:reduce:end]

    cudaStreamDestroy(stream);

    const T expected = static_cast<T>(N) * static_cast<T>(N + 1) / T(2);
    const bool ok    = (total == expected);
    std::printf("03_reduction (CUDA): sum(1..%ld)=%.0f (expected %.0f) : %s\n",
        static_cast<long>(N), total, expected, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
