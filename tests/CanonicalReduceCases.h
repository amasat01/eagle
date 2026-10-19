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
 * sizes); the tree against an independent oracle; chunked == unchunked incl.
 * ragged chunks; the plain (uncompensated) tree misses 1 ULP on a fixture the
 * tree meets (non-vacuity of the contract rows); the contract on exact
 * fixtures; columns equal flat reduces; ``auto`` resolves to compensated; the
 * host leaf equals the device-leaf phases run on CPU-simulated blocks;
 * ``cpu::Reduction`` routes floating sums through the tree.
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

template<typename T>
struct Pair {
    T h, c;
};

template<typename T>
inline Pair<T> join(Pair<T> a, Pair<T> b)
{
    const T s = a.h + b.h, bb = s - a.h, e = (a.h - (s - bb)) + (b.h - bb);
    return { s, std::isfinite(s) ? (a.c + b.c) + e : T(0) };
}

/** @brief Independent oracle of the ordered tree: per leaf of 4096, slot t
 *  folds t, t+256, ... by Neumaier steps, then the in-leaf tree, then
 *  pairwise levels — written without sharing code with canonical.h. */
template<typename T>
inline T oracle(const std::vector<T>& x)
{
    const std::size_t n = x.size();
    std::vector<Pair<T>> parts;
    for (std::size_t b = 0; b < n; b += 4096) {
        const std::size_t cnt = std::min<std::size_t>(4096, n - b);
        std::vector<Pair<T>> s;
        for (std::size_t t = 0; t < std::min<std::size_t>(cnt, 256); ++t) {
            Pair<T> p { x[b + t], T(0) };
            for (std::size_t k = t + 256; k < cnt; k += 256)
                p = join(p, Pair<T> { x[b + k], T(0) }); // a Neumaier step is a join with (x, 0)
            s.push_back(p);
        }
        for (std::size_t off = 128; off >= 1; off /= 2)
            for (std::size_t t = 0; t < off; ++t)
                if (t + off < s.size())
                    s[t] = join(s[t], s[t + off]);
        parts.push_back(s[0]);
    }
    while (parts.size() > 1) {
        std::vector<Pair<T>> nx;
        for (std::size_t i = 0; i < parts.size(); i += 2)
            nx.push_back(i + 1 < parts.size() ? join(parts[i], parts[i + 1]) : parts[i]);
        parts = nx;
    }
    return std::isfinite(parts[0].h) ? parts[0].h + parts[0].c : parts[0].h;
}

/** @brief A plain add: not ``SumOp``, so the tree folds it uncompensated. */
template<typename T>
struct PlainAdd {
    AETHER_DEVICEHOST() T operator()(T a, T b) const { return a + b; }
};

/** @brief The pre-compensation tree (plain adds, leaf 256, stride halving). */
template<typename T>
inline T plainTree(const std::vector<T>& x)
{
    std::vector<T> parts;
    for (std::size_t b = 0; b < x.size(); b += 256) {
        const std::size_t cnt = std::min<std::size_t>(256, x.size() - b);
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
            nx.push_back(i + 1 < parts.size() ? parts[i] + parts[i + 1] : parts[i]);
        parts = nx;
    }
    return parts[0];
}

/** @brief Exact fixture: mirrored pairs +-x cancelling exactly plus a
 *  2^-10-grid residual. Returns the values; ``exact`` and the reached
 *  ``kappa`` through the out-parameters. */
template<typename T>
inline std::vector<T> fixture(idx_t n, double kappa, double& exact, double& reached)
{
    std::mt19937_64 rng(n);
    std::uniform_int_distribution<int> unit(1, 8);
    std::uniform_real_distribution<double> mag(1.0, 2.0);
    std::vector<T> x(n);
    std::vector<double> big(n);
    long long units = 0;
    idx_t nbig      = 0;
    for (idx_t i = 0; i < n; ++i) {
        const idx_t j = std::min(i, n - 1 - i);
        big[j]        = mag(rng);
        if (j % 4 == 0 || i == n - 1 - i) {
            const int u = unit(rng);
            units += u;
            x[i] = T(std::ldexp(double(u), -10));
        } else {
            ++nbig;
        }
    }
    exact              = std::ldexp(double(units), -10);
    const double scale = kappa * exact / (1.5 * double(std::max<idx_t>(nbig, 1)));
    double abs         = 0;
    for (idx_t i = 0; i < n; ++i) {
        const idx_t j = std::min(i, n - 1 - i);
        if (!(j % 4 == 0 || i == n - 1 - i))
            x[i] = (i < n - 1 - i) ? T(big[j] * scale) : -T(big[j] * scale);
        abs += std::fabs(double(x[i]));
    }
    reached = abs / exact;
    return x;
}

template<typename T>
inline double ulps(T got, double exact)
{
    const T e = T(exact);
    return std::fabs(double(got) - exact) / double(std::nextafter(e, T(INFINITY)) - e);
}

/** @brief Sizes incl. non-powers of two, leaf and chunk boundaries. */
inline std::vector<idx_t> sizes()
{
    return { 1, 2, 3, 255, 256, 257, 4095, 4096, 4097, 8193, 12345, 65536,
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
        for (int q = 0; q <= 3; ++q) {
            const idx_t chunk = canon::leafSize << q;
            EXPECT_TRUE(
                sameBits(canon::reduceChunked<double, Op>(in, n, chunk), full))
                << "n=" << n << " chunk=" << chunk;
        }
    }
    // an unaligned chunk is refused
    auto x = corpus<double>(10000, 3);
    VecIn<double> in { x.data() };
    EXPECT_ANY_THROW((canon::reduceChunked<double, aether::SumOp<double>>(
        in, 10000, canon::leafSize * 3)));
}

TEST_F(CanonicalReduceTest, PlainTree_ExceedsContract)
{
    // non-vacuity of the contract rows: on the kappa = 2^10 f32 fixture the
    // plain tree (old order, and the new order without compensation) misses
    // 1 ULP; the canonical tree meets it.
    double exact = 0, kappa = 0;
    const idx_t n = 1000000;
    auto x        = fixture<float>(n, 1024.0, exact, kappa);
    VecIn<float> in { x.data() };
    const double oldTree = ulps(plainTree(x), exact);
    const double newOrder = ulps(canon::reduce<float, PlainAdd<float>>(in, n), exact);
    EXPECT_GT(oldTree, 1.0);
    EXPECT_GT(newOrder, 1.0);
    EXPECT_LE(ulps(canon::reduce<float, aether::SumOp<float>>(in, n), exact), 1.0);
    RecordProperty("plain_tree_ulp", int(oldTree));
    RecordProperty("plain_new_order_ulp", int(newOrder));
}

template<typename T>
inline void withinContract(double u, double oneUlpUpTo)
{
    for (idx_t n : { idx_t(257), idx_t(2048 * 3 + 1), idx_t(1000000) }) {
        for (double kappa : { 1.0, 1024.0, 1048576.0 }) {
            double exact = 0, k = 0;
            auto x = fixture<T>(n, kappa, exact, k);
            VecIn<T> in { x.data() };
            const double err = ulps(canon::reduce<T, aether::SumOp<T>>(in, n), exact);
            EXPECT_LE(err, 1.0 + double(n) * k * u) << "n=" << n << " kappa=" << kappa;
            if (double(n) * k <= oneUlpUpTo) {
                EXPECT_LE(err, 1.0) << "n=" << n << " kappa=" << kappa;
            }
        }
    }
    // non-finite terms follow IEEE through hi
    auto x = corpus<T>(9001, 5);
    VecIn<T> in { x.data() };
    x[1000] = T(INFINITY);
    EXPECT_EQ((canon::reduce<T, aether::SumOp<T>>(in, 9001)), T(INFINITY));
    x[7000] = -T(INFINITY);
    EXPECT_TRUE(std::isnan(canon::reduce<T, aether::SumOp<T>>(in, 9001)));
}

TEST_F(CanonicalReduceTest, Compensated_WithinContract_f32)
{
    withinContract<float>(std::ldexp(1.0, -24), std::ldexp(1.0, 20));
}

TEST_F(CanonicalReduceTest, Compensated_WithinContract_f64)
{
    withinContract<double>(std::ldexp(1.0, -53), std::ldexp(1.0, 48));
}

TEST_F(CanonicalReduceTest, Columns_EqualsPerColumnFlat)
{
    const idx_t rows = 9001, cols = 7;
    auto x = corpus<double>(rows * cols, 21);
    VecIn<double> in { x.data() };
    for (idx_t j = 0; j < cols; ++j) {
        std::vector<double> col(rows);
        for (idx_t i = 0; i < rows; ++i)
            col[i] = x[i * cols + j];
        const canon::Column<VecIn<double>> cj { in, cols, j };
        EXPECT_TRUE(sameBits(
            (canon::partialRange<double, aether::SumOp<double>>(cj, 0, rows).value()), oracle(col)))
            << "column " << j;
    }
}

TEST_F(CanonicalReduceTest, Policy_Auto_FollowsRule)
{
    // no device card ships: auto is compensated
    using eagle::reduce::Policy;
    static_assert(eagle::reduce::resolve(Policy::auto_) == Policy::compensated);
    static_assert(eagle::reduce::resolve(Policy::ordered) == Policy::ordered);
    EXPECT_EQ(eagle::reduce::resolve(Policy::compensated), Policy::compensated);
}

/** @brief The device leaf, run on the CPU: 256 "threads" with a shared array,
 *  phased like leafKernel (each thread walks its slot; sync; per offset:
 *  guarded join; sync). */
template<typename T>
inline T simulatedDeviceLeaf(const T* data, idx_t cnt)
{
    Pair<T> shared[256];
    const idx_t live = std::min<idx_t>(cnt, 256);
    for (idx_t t = 0; t < live; ++t) { // phase 0: each thread walks its slot
        Pair<T> p { data[t], T(0) };
        for (idx_t k = t + 256; k < cnt; k += 256)
            p = join(p, Pair<T> { data[k], T(0) });
        shared[t] = p;
    }
    for (idx_t off = 128; off > 0; off >>= 1) // one barrier-separated phase
        for (idx_t t = 0; t < 256; ++t)
            if (t < off && t + off < live)
                shared[t] = join(shared[t], shared[t + off]);
    return std::isfinite(shared[0].h) ? shared[0].h + shared[0].c : shared[0].h;
}

TEST_F(CanonicalReduceTest, HostLeafEqualsSimulatedDeviceLeaf)
{
    for (idx_t cnt : { idx_t(1), idx_t(2), idx_t(3), idx_t(100), idx_t(255),
             idx_t(256), idx_t(257), idx_t(1000), idx_t(4095), idx_t(4096) }) {
        for (int trial = 0; trial < 20; ++trial) {
            auto x = corpus<double>(cnt, 77 * trial + cnt);
            VecIn<double> in { x.data() };
            EXPECT_TRUE(sameBits(
                (canon::leaf<double, aether::SumOp<double>>(in, 0, cnt).value()),
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
