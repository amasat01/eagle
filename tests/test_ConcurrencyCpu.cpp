// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Host faces used by many threads at once on distinct data give the serial
 * result bit for bit: the canonical reduction tree (each caller opens its own
 * OpenMP team) and the compaction index map. C++-only mode; the CUDA build
 * runs the device twins. */
#include <atomic>
#include <cstdint>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#include "eagle/eagle.h"
#include "eagle/reduce/canonical.h"

#include "TestBase.h"

namespace eagle_tests {
namespace ConcurrencyCpuTest {

namespace canon = eagle::reduce::canonical;
constexpr int kThreads = 8;

template <class F>
void inThreads(F&& body)
{
    std::atomic<int> ready { 0 };
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back([&, t] {
            ready.fetch_add(1);
            while (ready.load() < kThreads) std::this_thread::yield();
            body(t);
        });
    for (auto& t : threads) t.join();
}

struct VecIn {
    const double* p;
    double operator()(eagle::idx_t i) const { return p[i]; }
};

TEST(ConcurrencyCpu, CanonicalReduceOnDistinctDataEqualsTheSerialBitsInManyThreads)
{
    const eagle::idx_t n = 70001;
    std::vector<std::vector<double>> data(kThreads, std::vector<double>(n));
    std::vector<double> want(kThreads), got(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        std::mt19937_64 rng(100 + t);
        std::normal_distribution<double> nd(0.0, 1.0);
        for (auto& x : data[t]) x = nd(rng) * 1e3;
        want[t] = canon::reduce<double, aether::SumOp<double>>(VecIn { data[t].data() }, n);
    }
    inThreads([&](int t) {
        for (int rep = 0; rep < 5; ++rep)
            got[t] = canon::reduce<double, aether::SumOp<double>>(VecIn { data[t].data() }, n);
    });
    for (int t = 0; t < kThreads; ++t)
        EXPECT_EQ(0, std::memcmp(&want[t], &got[t], sizeof(double))) << t;
}

TEST(ConcurrencyCpu, CompactOnDistinctBuffersEqualsTheSerialMapInManyThreads)
{
    using eagle::filtering::compactHost;
    using eagle::filtering::compactScratchBytes;
    const std::int64_t n = 100003;
    struct Job {
        std::vector<std::uint8_t> mask;
        std::vector<std::int32_t> map;
        std::vector<eagle::idx_t> scratch;
        std::uint32_t count = 0;
        void run(std::int64_t n)
        {
            map.assign(std::size_t(n), -1);
            scratch.resize(compactScratchBytes(n) / sizeof(eagle::idx_t));
            compactHost(mask.data(), 0u, map.data(), &count, scratch.data(), n);
        }
    };
    std::vector<Job> want(kThreads), got(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        std::mt19937 rng(7 + t);
        std::bernoulli_distribution keep(0.6);
        want[t].mask.resize(std::size_t(n));
        for (auto& b : want[t].mask) b = keep(rng) ? 1 : 0;
        got[t].mask = want[t].mask;
        want[t].run(n);
    }
    inThreads([&](int t) {
        for (int rep = 0; rep < 3; ++rep) got[t].run(n);
    });
    for (int t = 0; t < kThreads; ++t) {
        EXPECT_EQ(want[t].count, got[t].count) << t;
        EXPECT_EQ(want[t].map, got[t].map) << t;
    }
}

}  // namespace ConcurrencyCpuTest
}  // namespace eagle_tests
