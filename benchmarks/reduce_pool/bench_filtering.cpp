// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The host scan, compaction and reorder of eagle's filtering, timed.
//
//   bench_filtering <scan|compact|reorder> <team> <n>   prints the median microseconds of one call (and a checksum)
//
// The same source builds against two trees (see run.sh): the tree whose loops run on eagle's host pool (`team` is an
// argument), and the tree whose loops ran under OpenMP (`-DEAGLE_BENCH_OMP`: the team is omp_set_num_threads').
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include "eagle/filtering/Reorder.h"

#ifdef EAGLE_BENCH_OMP
#include <omp.h>
#define EAGLE_BENCH_TEAM(team) omp_set_num_threads(team)
#define EAGLE_BENCH_ARG(team)
#else
#define EAGLE_BENCH_TEAM(team) (void)(team)
#define EAGLE_BENCH_ARG(team) , team
#endif

namespace {

using Clock = std::chrono::steady_clock;
using eagle::idx_t;
namespace flt = eagle::filtering;

/** @brief One call of the chosen operation over fixed data; returns a value the data determines. */
struct Case {
    std::int64_t n;
    std::vector<std::uint8_t> mask;
    std::vector<idx_t> in, out, sums;
    std::vector<std::int32_t> perm, inv, map;
    std::vector<double> plane;
    std::vector<std::uint8_t> scratch;

    explicit Case(std::int64_t n_)
        : n(n_), mask(std::size_t(n)), in(std::size_t(n)), out(std::size_t(n)), sums(std::size_t(n) / 1024 + 2),
          perm(std::size_t(n)), inv(std::size_t(n)), map(std::size_t(n)), plane(std::size_t(n)),
          scratch(flt::reorderScratchBytes(n, 8) + 16)
    {
        std::mt19937 rng(7);
        std::bernoulli_distribution keep(0.6);
        for (std::int64_t i = 0; i < n; ++i) {
            mask[std::size_t(i)] = keep(rng) ? 1 : 0;
            in[std::size_t(i)]   = idx_t(rng() % 13);
            plane[std::size_t(i)] = double(i);
        }
    }

    void* aligned() { return scratch.data() + ((16 - reinterpret_cast<std::uintptr_t>(scratch.data()) % 16) % 16); }

    std::uint64_t scan(int team)
    {
        EAGLE_BENCH_TEAM(team);
        eagle::cpu::Scan::scan<idx_t, aether::SumOp<idx_t>, true>(
            eagle::spanView<idx_t>(in.data(), n, eagle::hostDevice()).as_const(),
            eagle::spanView<idx_t>(out.data(), n, eagle::hostDevice()),
            eagle::spanView<idx_t>(sums.data(), idx_t(sums.size()), eagle::hostDevice()), idx_t(0) EAGLE_BENCH_ARG(team));
        return std::uint64_t(out[std::size_t(n) - 1]);
    }

    std::uint64_t compact(int team)
    {
        EAGLE_BENCH_TEAM(team);
        std::vector<idx_t> keep(flt::compactScratchBytes(n) / sizeof(idx_t));
        std::uint32_t count = 0;
        flt::compactHost(mask.data(), 0u, map.data(), &count, keep.data(), n EAGLE_BENCH_ARG(team));
        return count;
    }

    std::uint64_t reorder(int team)
    {
        EAGLE_BENCH_TEAM(team);
        std::iota(perm.begin(), perm.end(), 0);
        std::iota(inv.begin(), inv.end(), 0);
        std::iota(map.begin(), map.end(), 0);
        std::uint32_t count = std::uint32_t(n), span = std::uint32_t(n), fire = 1;
        const flt::ReorderPlane planes[1] = { { plane.data(), 8 } };
        flt::reorderHost(planes, 1, mask.data(), 0u, perm.data(), inv.data(), map.data(), &count, &span, &fire, aligned(),
            n, 0u EAGLE_BENCH_ARG(team));
        return count;
    }
};

}  // namespace

int main(int argc, char** argv)
{
    if (argc != 4)
        return 2;
    const std::string op = argv[1];
    const int team       = std::atoi(argv[2]);
    const long n         = std::atol(argv[3]);
    Case c(n);
    const auto once = [&]() -> std::uint64_t {
        return op == "scan" ? c.scan(team) : op == "compact" ? c.compact(team) : c.reorder(team);
    };
    std::uint64_t sink = 0;
    for (int w = 0; w < 5; ++w)
        sink += once();
    const auto t0 = Clock::now();
    sink += once();
    const double one = std::chrono::duration<double, std::micro>(Clock::now() - t0).count();
    const int calls  = std::max(9, static_cast<int>(30000.0 / std::max(one, 1.0)));
    std::vector<double> us(calls);
    for (double& u : us) {
        const auto s = Clock::now();
        sink += once();
        u = std::chrono::duration<double, std::micro>(Clock::now() - s).count();
    }
    std::nth_element(us.begin(), us.begin() + calls / 2, us.end());
    std::printf("%.3f %016llx\n", us[calls / 2], static_cast<unsigned long long>(once()));
    return sink == 0xffffffffull ? 1 : 0;
}
