// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// 02_host_dispatch.cu
//
// The pure-C++ path: dispatch one task per index through EAGLE's OpenMP + SIMD
// host launcher. `eagle::cpu::Host::launch` batches consecutive indices into
// aether SIMD packets — the same call compiles and runs identically in CUDA
// mode (this file) and in pure-C++ mode (the .cpp twin), which is EAGLE's
// dual-mode promise.
//
// IMPORTANT: `launch` only WORK-SHARES across OpenMP threads when the caller
// is already inside an existing `#pragma omp parallel` region; called bare,
// as below, it runs serially (SIMD-vectorised, single-threaded). This
// standalone example has no enclosing region, so it is a *correctness* demo,
// not a *threading* demo — see `eagle::cpu::Host::launch`'s doc comment
// (eagle/cpu/Host.h) for the full contract: wrapping several `launch` calls
// in your own `#pragma omp parallel` region is what gets them threaded in
// production code.
#include <cstdio>
#include <vector>

#include <eagle/eagle.h>

int main()
{
    constexpr eagle::idx_t N = 1024;
    std::vector<double> x(N, 0.0);

    // [cell:launch]
    // One host task per index i; i.global() is this task's flat index. We write
    // the sequence of odd numbers 1, 3, 5, ... whose prefix sums are N*N.
    eagle::cpu::Host::launch(N, [&](const auto& i) {
        const eagle::idx_t k = i.global();
        x[k] = 2.0 * static_cast<double>(k) + 1.0;
    });
    // [cell:launch:end]

    double sum = 0.0;
    for (eagle::idx_t k = 0; k < N; ++k)
        sum += x[k];

    const double expected = static_cast<double>(N) * static_cast<double>(N);
    const bool ok         = (sum == expected);
    std::printf("02_host_dispatch: N=%ld sum=%.0f (expected %.0f) : %s\n",
        static_cast<long>(N), sum, expected, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
