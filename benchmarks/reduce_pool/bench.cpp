// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// One compensated float64 sum through eagle's canonical host reduction, timed.
//
//   bench <team> <n>   prints the median microseconds of one reduction (and the result's bits)
//
// The same source builds against two trees (see run.sh): the tree that runs the reduction's loops on
// eagle's host pool (`team` is an argument of `reduce`), and the tree that ran them with OpenMP
// (`-DEAGLE_BENCH_OMP`: the team is omp_set_num_threads').
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "eagle/reduce/canonical.h"

#ifdef EAGLE_BENCH_OMP
#include <omp.h>
#endif

namespace {

namespace canon = eagle::reduce::canonical;
using Clock     = std::chrono::steady_clock;

struct VecIn {
    const double* p;
    double operator()(eagle::idx_t i) const { return p[i]; }
};

double reduceOnce(const std::vector<double>& x, int team)
{
#ifdef EAGLE_BENCH_OMP
    omp_set_num_threads(team);
    return canon::reduce<double, aether::SumOp<double>>(VecIn{ x.data() }, eagle::idx_t(x.size()));
#else
    return canon::reduce<double, aether::SumOp<double>>(VecIn{ x.data() }, eagle::idx_t(x.size()), team);
#endif
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 3)
        return 2;
    const int team = std::atoi(argv[1]);
    const long n   = std::atol(argv[2]);
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<double> x(n);
    for (auto& v : x)
        v = nd(rng);
    double sink = 0;
    for (int w = 0; w < 5; ++w)
        sink += reduceOnce(x, team);
    // enough calls that the timer is not the measurement: about 30 ms of work, at least 9 calls
    const auto t0 = Clock::now();
    sink += reduceOnce(x, team);
    const double one = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    const int calls  = std::max(9, static_cast<int>(30000.0 / std::max(one, 1.0)));
    std::vector<double> us(calls);
    for (double& u : us) {
        const auto s = Clock::now();
        sink += reduceOnce(x, team);
        u = std::chrono::duration<double, std::micro>(Clock::now() - s).count();
    }
    std::nth_element(us.begin(), us.begin() + calls / 2, us.end());
    std::uint64_t bits;
    const double result = reduceOnce(x, team);
    std::memcpy(&bits, &result, sizeof(bits));
    std::printf("%.3f %016llx\n", us[calls / 2], static_cast<unsigned long long>(bits));
    return sink == 12345.678 ? 1 : 0;   // keeps the sums alive
}
