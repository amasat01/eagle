// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file CanonicalReduceCases.h
 * @brief Host-face rows of the canonical reduction tree
 *        (``eagle/reduce/canonical.h``), shared by the C++-only build
 *        (test_CanonicalReduce.cpp) and the CUDA build
 *        (test_CanonicalReduce.cu, beside the device row).
 *
 * Rows: bitwise thread-count invariance (OpenMP 1/2/4/8, non-power-of-two
 * sizes); the tree against an independent recursive oracle; chunked ==
 * unchunked incl. ragged chunks; planted wrong orders (left fold, chunk of
 * 3 leaves) MUST differ on the corpus; the host leaf equals the device-leaf
 * definition run on CPU-simulated blocks; ``cpu::Reduction`` routes floating
 * sums through the tree.
 */

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <omp.h>

#include "eagle/eagle.h"
#include "eagle/reduce/canonical.h"

#include "TestBase.h"

namespace eagle_tests {
namespace CanonicalReduceTest {

namespace canon = eagle::reduce::canonical;
using idx_t     = eagle::idx_t;

template<typename T>
inline bool sameBits(T a, T b)
{
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/** @brief Adversarial corpus: normal values scaled by 10^[-8, 8]. */
template<typename T>
inline std::vector<T> corpus(idx_t n, std::uint64_t seed)
{
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::uniform_int_distribution<int> ex(-8, 8);
    std::vector<T> v(n);
    for (auto& x : v)
        x = T(nd(rng) * std::pow(10.0, ex(rng)));
    return v;
}

template<typename T>
struct VecIn {
    const T* p;
    T operator()(idx_t i) const { return p[i]; }
};

/** @brief Independent oracle: leaf by recursive stride-halving over a
 *  padded-with-absence slot array, pairwise over leaves — written without
 *  sharing code with canonical.h. */
template<typename T>
inline T oracle(const std::vector<T>& x)
{
    const std::size_t n = x.size();
    std::vector<T> parts;
    for (std::size_t b = 0; b < n; b += 256) {
        const std::size_t cnt = std::min<std::size_t>(256, n - b);
        std::vector<T> s(x.begin() + b, x.begin() + b + cnt);
        for (std::size_t off = 128; off >= 1; off /= 2)
            for (std::size_t t = 0; t < off; ++t)
                if (t + off < cnt)
                    s[t] = s[t] + s[t + off];
        parts.push_back(s[0]);
    }
    while (parts.size() > 1) {
        std::vector<T> nx;
        for (std::size_t i = 0; i < parts.size(); i += 2)
            nx.push_back(i + 1 < parts.size() ? parts[i] + parts[i + 1]
                                              : parts[i]);
        parts = nx;
    }
    return parts[0];
}

template<typename T>
inline T leftFold(const std::vector<T>& v)
{
    T s = v[0];
    for (std::size_t i = 1; i < v.size(); ++i)
        s = s + v[i];
    return s;
}

/** @brief Sizes incl. non-powers of two, leaf and chunk boundaries. */
inline std::vector<idx_t> sizes()
{
    return { 1, 2, 3, 255, 256, 257, 511, 512, 513, 1000, 4097, 12345, 65536,
        65537, 100003, (1 << 20) + 37 };
}

class CanonicalReduceTest : public Test { };

TEST_F(CanonicalReduceTest, ThreadCountInvariance_Bitwise)
{
    const int saved = omp_get_max_threads();
    for (idx_t n : sizes()) {
        auto x = corpus<double>(n, 1000 + n);
        auto arr = eagle::makeArray<double>(n);
        const eagle::GRefArrT<double> w = arr.hostView();
        for (idx_t i = 0; i < n; ++i)
            w(i) = x[i];
        const eagle::CRefArrT<double> r = arr.hostView().as_const();
        double ref = 0;
        bool have  = false;
        for (int th : { 1, 2, 4, 8 }) {
            omp_set_num_threads(th);
            const double got
                = eagle::cpu::Reduction<double, aether::SumOp<double>>::reduce(
                    r, 0.0);
            if (!have) {
                ref  = got;
                have = true;
            }
            EXPECT_TRUE(sameBits(got, ref)) << "n=" << n << " threads=" << th;
            EXPECT_TRUE(sameBits(got, oracle(x))) << "n=" << n;
        }
    }
    omp_set_num_threads(saved);
}

TEST_F(CanonicalReduceTest, ThreadCountInvariance_Float)
{
    const int saved = omp_get_max_threads();
    for (idx_t n : { idx_t(257), idx_t(70001), idx_t(300000) }) {
        auto x = corpus<float>(n, 7 + n);
        VecIn<float> in { x.data() };
        float ref = 0;
        for (int th : { 1, 2, 4, 8 }) {
            omp_set_num_threads(th);
            const float got
                = canon::reduce<float, aether::SumOp<float>>(in, n);
            if (th == 1)
                ref = got;
            EXPECT_TRUE(sameBits(got, ref)) << "n=" << n << " threads=" << th;
        }
        EXPECT_TRUE(sameBits(ref, oracle(x)));
    }
    omp_set_num_threads(saved);
}

TEST_F(CanonicalReduceTest, ChunkedEqualsUnchunked_Bitwise)
{
    using Op = aether::SumOp<double>;
    for (idx_t n : sizes()) {
        if (n > 200000)
            continue; // the larger sizes are covered by the thread rows
        auto x = corpus<double>(n, 31 + n);
        VecIn<double> in { x.data() };
        const double full = canon::reduce<double, Op>(in, n);
        for (int q = 0; q <= 5; ++q) {
            const idx_t chunk = canon::leafSize << q;
            EXPECT_TRUE(
                sameBits(canon::reduceChunked<double, Op>(in, n, chunk), full))
                << "n=" << n << " chunk=" << chunk;
        }
    }
    // an unaligned chunk is refused
    auto x = corpus<double>(1000, 3);
    VecIn<double> in { x.data() };
    EXPECT_ANY_THROW((canon::reduceChunked<double, aether::SumOp<double>>(
        in, 1000, canon::leafSize * 3)));
}

TEST_F(CanonicalReduceTest, PlantedWrongOrders_Differ)
{
    using Op = aether::SumOp<double>;
    int badFold = 0, badTree3 = 0, badFlat = 0, total = 0;
    for (idx_t n : { idx_t(256 * 48), idx_t(256 * 50), idx_t(256 * 64),
             idx_t(256 * 40 + 17), idx_t(20011) }) {
        for (int trial = 0; trial < 10; ++trial) {
            auto x = corpus<double>(n, 500 + 13 * trial + n);
            VecIn<double> in { x.data() };
            const double full = canon::reduce<double, Op>(in, n);
            ++total;
            // left fold of chunk partials (chunk = 16 leaves)
            std::vector<double> parts;
            for (idx_t b = 0; b < n; b += 256 * 16)
                parts.push_back(canon::reduceRange<double, Op>(
                    in, b, std::min<idx_t>(256 * 16, n - b)));
            badFold += !sameBits(leftFold(parts), full);
            // chunks of 3 leaves, pairwise-combined (not a subtree)
            std::vector<double> p3;
            for (idx_t b = 0; b < n; b += 256 * 3)
                p3.push_back(canon::reduceRange<double, Op>(
                    in, b, std::min<idx_t>(256 * 3, n - b)));
            badTree3 += !sameBits(canon::combine<double, Op>(p3), full);
            // a plain left fold over all samples
            badFlat += !sameBits(leftFold(x), full);
        }
    }
    // non-vacuity: every planted wrong order must trip somewhere
    EXPECT_GT(badFold, 0);
    EXPECT_GT(badTree3, 0);
    EXPECT_GT(badFlat, 0);
    RecordProperty("planted_fold", badFold);
    RecordProperty("planted_chunk3", badTree3);
    RecordProperty("planted_flat", badFlat);
    RecordProperty("trials", total);
}

/** @brief The device leaf, run on the CPU: 256 "threads" with a shared array,
 *  phased exactly like leafKernel (load; sync; per offset: guarded combine;
 *  sync). */
template<typename T>
inline T simulatedDeviceLeaf(const T* data, idx_t cnt)
{
    T shared[256];
    for (idx_t t = 0; t < 256; ++t) // phase 0: each thread loads its sample
        if (t < cnt)
            shared[t] = data[t];
    for (idx_t off = 128; off > 0; off >>= 1) // one barrier-separated phase
        for (idx_t t = 0; t < 256; ++t)
            if (t < off && t + off < cnt)
                shared[t] = shared[t] + shared[t + off];
    return shared[0];
}

TEST_F(CanonicalReduceTest, HostLeafEqualsSimulatedDeviceLeaf)
{
    for (idx_t cnt : { idx_t(1), idx_t(2), idx_t(3), idx_t(100), idx_t(127),
             idx_t(128), idx_t(129), idx_t(255), idx_t(256) }) {
        for (int trial = 0; trial < 20; ++trial) {
            auto x = corpus<double>(cnt, 77 * trial + cnt);
            VecIn<double> in { x.data() };
            EXPECT_TRUE(sameBits(
                (canon::leaf<double, aether::SumOp<double>>(in, 0, cnt)),
                simulatedDeviceLeaf(x.data(), cnt)))
                << "cnt=" << cnt;
        }
    }
}

TEST_F(CanonicalReduceTest, CpuReduction_InitAndEmpty)
{
    using Op = aether::SumOp<double>;
    auto x = corpus<double>(5000, 9);
    auto arr = eagle::makeArray<double>(5000);
    const eagle::GRefArrT<double> w = arr.hostView();
    for (idx_t i = 0; i < 5000; ++i)
        w(i) = x[i];
    const eagle::CRefArrT<double> r = arr.hostView().as_const();
    EXPECT_TRUE(sameBits(eagle::cpu::Reduction<double, Op>::reduce(r, 1.5),
        1.5 + oracle(x)));
}

} // namespace CanonicalReduceTest
} // namespace eagle_tests
