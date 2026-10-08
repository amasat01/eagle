// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// 03_reduction.cpp  (pure-C++ / OpenMP mode)
//
// The C++-mode twin of 03_reduction.cu: the same aether SoA array reduced with
// EAGLE's cache-padded OpenMP reduction. Built when eagle is configured with
// EAGLE_CPP_MODE (which defines EAGLE_CPU_ONLY and removes the CUDA path).
#include <cstdio>

#include <eagle/eagle.h>

int main()
{
    using T   = double;
    using Sum = aether::SumOp<T>;
    constexpr eagle::idx_t N = 1000;

    auto arr = eagle::makeArray<T>(N);
    {
        auto a = arr.hostView();
        for (eagle::idx_t i = 0; i < N; ++i)
            a(i) = static_cast<T>(i + 1);
    }

    // [cell:reduce]
    // Cache-padded OpenMP reduction over the host-resident array.
    const T total = eagle::cpu::Reduction<T, Sum>::reduce(arr.hostView().as_const(), T(0));
    // [cell:reduce:end]

    const T expected = static_cast<T>(N) * static_cast<T>(N + 1) / T(2);
    const bool ok    = (total == expected);
    std::printf("03_reduction (C++): sum(1..%ld)=%.0f (expected %.0f) : %s\n",
        static_cast<long>(N), total, expected, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
