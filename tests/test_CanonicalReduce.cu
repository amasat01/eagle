// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_CanonicalReduce.cu
 * @brief The host rows (CanonicalReduceCases.h) plus the device rows: the GPU
 *        canonical reduction equals the host one bit for bit; the fixed-point
 *        f64 leaf is exact on the cancellation ladder, follows IEEE on
 *        specials, and its device face equals its host face bit for bit.
 *        (Device rows — need a GPU.)
 */

#include <cuda_runtime_api.h>

#include "CanonicalReduceCases.h"

namespace eagle_tests {
namespace CanonicalReduceTest {

template<typename T, typename OP>
static T deviceCanonical(const std::vector<T>& x)
{
    const idx_t n = idx_t(x.size());
    auto arr      = eagle::makeArray<T>(n);
    const eagle::GRefArrT<T> w = arr.hostView();
    for (idx_t i = 0; i < n; ++i)
        w(i) = x[i];
    arr.upload();
    const idx_t nl = canon::deviceWorkElems(n);
    auto a         = eagle::makeArray<T>(nl);
    auto b         = eagle::makeArray<T>(nl);
    a.upload();
    b.upload();
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());

    cudaStream_t stream;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));
    T* res = canon::enqueue<T, OP>(arr.deviceView().as_const(), n,
        a.deviceView().data(), b.deviceView().data(), stream);
    EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(stream));
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));
    auto& out = (res == a.deviceView().data()) ? a : b;
    out.download();
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    return out.hostView()(0);
}

TEST_F(CanonicalReduceTest, CUDA_SumEqualsHost_Bitwise)
{
    for (idx_t n : sizes()) {
        auto x = corpus<double>(n, 4242 + n);
        VecIn<double> in { x.data() };
        const double host
            = canon::reduce<double, aether::SumOp<double>>(in, n);
        EXPECT_TRUE(sameBits(
            deviceCanonical<double, aether::SumOp<double>>(x), host))
            << "n=" << n;
    }
}

TEST_F(CanonicalReduceTest, CUDA_MinEqualsHost_Bitwise)
{
    for (idx_t n : sizes()) {
        auto x = corpus<double>(n, 99 + n);
        x[n / 2] = -std::numeric_limits<double>::infinity();
        if (n > 3)
            x[n / 3] = std::numeric_limits<double>::quiet_NaN();
        VecIn<double> in { x.data() };
        EXPECT_TRUE(sameBits((deviceCanonical<double, canon::MinOp<double>>(x)),
            (canon::reduce<double, canon::MinOp<double>>(in, n))))
            << "n=" << n;
    }
}

TEST_F(CanonicalReduceTest, CUDA_SumEqualsHost_Float_Bitwise)
{
    for (idx_t n : { idx_t(257), idx_t(70001), idx_t(300000) }) {
        auto x = corpus<float>(n, 17 + n);
        VecIn<float> in { x.data() };
        const float host = canon::reduce<float, aether::SumOp<float>>(in, n);
        EXPECT_TRUE(sameBits(
            deviceCanonical<float, aether::SumOp<float>>(x), host))
            << "n=" << n;
    }
}

// ---- the fixed-point f64 leaf ----

/** @brief Cancellation-ladder fixture: pairs x at r and exactly -x at n-1-r,
 *  |x| = m * 2^e with a full 52-bit mantissa m and e uniform in
 *  [G - ER, G + ER]; a quarter of the pairs (and the middle) instead carry
 *  v * 2^-10 on both partners. Returns the values; ``units`` = the exact sum
 *  in units of 2^-10. */
inline std::vector<double> ladder(idx_t n, int ER, int G, long long& units)
{
    const auto mix = [](std::uint64_t x) {
        x += 0x9E3779B97F4A7C15ull;
        x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
        x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
        return x ^ (x >> 31);
    };
    std::vector<double> x(n);
    units = 0;
    for (idx_t r = 0; r < n; ++r) {
        const idx_t m = n - 1 - r, j = std::min(r, m);
        const std::uint64_t h = mix(std::uint64_t(j) * 0x100000001B3ull ^ mix(7));
        if (j == 0 || m == r || (h & 3) == 0) {
            const long long v = 1 + (long long)((h >> 8) % 4096);
            x[r] = std::ldexp(double(v), -10);
            units += v;
        } else {
            const std::uint64_t h2 = mix(h), h3 = mix(h2);
            const double mant = 1.0 + double(h2 & ((1ull << 52) - 1)) / double(1ull << 52);
            const double v = std::ldexp(mant, int((h3 >> 1) % std::uint64_t(2 * ER + 1)) - ER + G);
            x[r] = ((h3 & 1) != 0) == (r < m) ? -v : v;
        }
    }
    return x;
}

inline double hostFixed(const std::vector<double>& x, idx_t rows = 0, idx_t cols = 1, idx_t j = 0,
    const double* comp = nullptr)
{
    using P = canon::FixedPartial;
    rows    = rows == 0 ? idx_t(x.size()) : rows;
    const canon::Column<VecIn<double>> cj { { x.data() }, cols, j };
    if (comp != nullptr) {
        const canon::Column<VecIn<double>> cc { { comp }, cols, j };
        return canon::partialRange<double, aether::SumOp<double>, P>(cj, 0, rows, cc).value();
    }
    return canon::partialRange<double, aether::SumOp<double>, P>(cj, 0, rows).value();
}

/** @brief The device fixed-point reduction of each column of a (rows, cols)
 *  plane (+ companion); ``reps`` launches, every result must agree. */
inline std::vector<double> deviceFixed(const std::vector<double>& x, idx_t rows, idx_t cols,
    const std::vector<double>* comp = nullptr, int reps = 1)
{
    const idx_t w = canon::deviceWorkElems(rows, cols);
    double *in, *cin = nullptr, *wa, *wb, *out;
    EAGLE_CHECK_ALWAYS(cudaMalloc(&in, x.size() * sizeof(double)));
    EAGLE_CHECK_ALWAYS(cudaMemcpy(in, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
    if (comp != nullptr) {
        EAGLE_CHECK_ALWAYS(cudaMalloc(&cin, x.size() * sizeof(double)));
        EAGLE_CHECK_ALWAYS(cudaMemcpy(cin, comp->data(), x.size() * sizeof(double), cudaMemcpyHostToDevice));
    }
    EAGLE_CHECK_ALWAYS(cudaMalloc(&wa, w * sizeof(double)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&wb, w * sizeof(double)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&out, cols * sizeof(double)));
    std::vector<double> got(cols), first;
    for (int r = 0; r < reps; ++r) {
        EAGLE_CHECK_ALWAYS(cudaMemset(out, 0xFF, cols * sizeof(double)));
        if (cin != nullptr)
            canon::enqueueTo<double, aether::SumOp<double>>(canon::DevicePtr<double> { in },
                canon::DevicePtr<double> { cin }, rows, cols, eagle::reduce::Policy::compensated, wa, wb,
                out, false, cudaStream_t(0), true);
        else
            canon::enqueueTo<double, aether::SumOp<double>>(canon::DevicePtr<double> { in }, canon::NoComp {},
                rows, cols, eagle::reduce::Policy::compensated, wa, wb, out, false, cudaStream_t(0), true);
        EAGLE_CHECK_ALWAYS(cudaMemcpy(got.data(), out, cols * sizeof(double), cudaMemcpyDeviceToHost));
        if (r == 0)
            first = got;
        else
            EXPECT_EQ(std::memcmp(first.data(), got.data(), cols * sizeof(double)), 0) << "rep " << r;
    }
    for (double* p : { in, cin, wa, wb, out })
        if (p != nullptr)
            EXPECT_EQ(cudaFree(p), cudaSuccess);
    return first;
}

TEST_F(CanonicalReduceTest, Fixed_ExactOnLadder)
{
    // 0 ULP: the base fixture (spread 2^+-30) and the kappa ladder G0..G40
    for (idx_t n : { idx_t(163840), idx_t(655360) + 3 }) {
        long long units = 0;
        const auto x = ladder(n, 30, 0, units);
        EXPECT_TRUE(sameBits(hostFixed(x), std::ldexp(double(units), -10))) << "base n=" << n;
    }
    for (int G : { 0, 10, 20, 30, 40 }) {
        long long units = 0;
        const auto x = ladder(655360, 0, G, units);
        EXPECT_TRUE(sameBits(hostFixed(x), std::ldexp(double(units), -10))) << "G=" << G;
    }
    for (idx_t n : { idx_t(257), idx_t(2048 * 3 + 1), idx_t(1000000) })
        for (double kappa : { 1.0, 1024.0, 1048576.0 }) {
            double exact = 0, k = 0;
            const auto x = fixture<double>(n, kappa, exact, k);
            EXPECT_TRUE(sameBits(hostFixed(x), exact)) << "n=" << n << " kappa=" << kappa;
        }
}

TEST_F(CanonicalReduceTest, Fixed_IeeeSpecials)
{
    const double inf = std::numeric_limits<double>::infinity();
    auto x = corpus<double>(9001, 5);
    x[1000] = inf;
    EXPECT_EQ(hostFixed(x), inf);
    x[7000] = -inf;
    EXPECT_TRUE(std::isnan(hostFixed(x)));
    x[7000] = 1.0;
    x[3] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(std::isnan(hostFixed(x)));
    // signed zero is not kept; subnormals round once; overflow is +Inf
    EXPECT_TRUE(sameBits(hostFixed({ -0.0, -0.0 }), 0.0));
    const double tiny = std::numeric_limits<double>::denorm_min();
    EXPECT_TRUE(sameBits(hostFixed({ tiny, 3 * tiny, -tiny }), 3 * tiny));
    const double big = std::numeric_limits<double>::max();
    EXPECT_EQ(hostFixed({ big, big }), inf);
    EXPECT_TRUE(sameBits(hostFixed({ big, big, -big }), big));
}

TEST_F(CanonicalReduceTest, Fixed_RuleFromRatio)
{
    static_assert(canon::fixedLeaf(32) && canon::fixedLeaf(64));
    static_assert(!canon::fixedLeaf(2) && !canon::fixedLeaf(1));
}

TEST_F(CanonicalReduceTest, CUDA_FixedEqualsHost_Bitwise)
{
    for (idx_t n : sizes()) {
        const auto x = corpus<double>(n, 4242 + n);
        EXPECT_TRUE(sameBits(deviceFixed(x, n, 1)[0], hostFixed(x))) << "n=" << n;
    }
    // the ladder, exact on the device too, bitwise x20 run to run
    for (int G : { 0, 10, 20, 30, 40 }) {
        long long units = 0;
        const auto x = ladder(655360, G == 0 ? 30 : 0, G, units);
        const double want = std::ldexp(double(units), -10);
        EXPECT_TRUE(sameBits(deviceFixed(x, idx_t(x.size()), 1, nullptr, 20)[0], want)) << "G=" << G;
    }
    // specials on the device
    auto s = corpus<double>(9001, 5);
    s[1000] = std::numeric_limits<double>::infinity();
    EXPECT_EQ(deviceFixed(s, 9001, 1)[0], std::numeric_limits<double>::infinity());
    s[7000] = -std::numeric_limits<double>::infinity();
    EXPECT_TRUE(std::isnan(deviceFixed(s, 9001, 1)[0]));
    // columns with a companion plane
    const idx_t rows = 10007, cols = 5;
    const auto x = corpus<double>(rows * cols, 77), c = corpus<double>(rows * cols, 78);
    const auto got = deviceFixed(x, rows, cols, &c);
    for (idx_t j = 0; j < cols; ++j)
        EXPECT_TRUE(sameBits(got[j], hostFixed(x, rows, cols, j, c.data()))) << "column " << j;
}

} // namespace CanonicalReduceTest
} // namespace eagle_tests
