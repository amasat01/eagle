// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_CanonicalReduce.cu
 * @brief The host rows (CanonicalReduceCases.h) plus the device row: the GPU
 *        canonical reduction equals the host one bit for bit. (Device row —
 *        needs a GPU.)
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

} // namespace CanonicalReduceTest
} // namespace eagle_tests
