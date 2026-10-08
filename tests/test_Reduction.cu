// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include "eagle/cuda/Reduction.h"
#include "eagle/cpu/Reduction.h"

#include "TestBase.h"

#include <aether/banded/banded.h>

namespace eagle_tests {
namespace ReductionTest {

using idx_t = eagle::idx_t;

class ReductionTest : public Test {
public:
    ReductionTest()
        : Test()
    {
        // eagle::util::setLogLevel(eagle::util::ALL);
    }

    /** @brief Device Reduction */
    template<typename T, typename OP, bool inPlace = false>
    void testCudaReduction(aether::Array<T>& arr, const T& expected,
        const T& init = 0, const double& eps = 0)
    {
        arr.upload();
        EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());

        cudaStream_t stream;
        EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));
        T actual;
        if constexpr (inPlace) {
            actual = eagle::cuda::Reduction::reduceBlocking<T, OP>(
                arr, arr, init);
        } else {
            actual = eagle::cuda::Reduction::reduceBlocking<T, OP>(
                arr, init, stream);
        }
        EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));

        if (eps == 0) {
            ASSERT_EQ(actual, expected);
        } else {
            ASSERT_NEAR(actual, expected, eps);
        }
    }

    /** @brief Host Reduction */
    template<typename T, typename OP>
    void testOmpReduction(aether::Array<T>& arr, const T& expected,
        const T& init = 0, const double& eps = 0)
    {
        const eagle::CRefArrT<T> aref = arr.hostView().as_const();
        T actual = eagle::cpu::Reduction<T, OP>::reduce(aref, init);

        if (eps == 0) {
            ASSERT_EQ(actual, expected);
        } else {
            ASSERT_NEAR(actual, expected, eps);
        }
    }

    template<typename T, typename OP>
    T dumbCPUReduction(const aether::Array<T>& arr, const T& init = 0)
    {
        T result                      = init;
        const eagle::CRefArrT<T> aref = arr.hostView();
        for (idx_t i = 0; i < idx_t(arr.samples()); i++) {
            result = OP{}(result, aref(i));
        }
        return result;
    }
};

using doubleSumOp = aether::SumOp<double>;
using floatMaxOp  = aether::MaxOp<float>;
using intSumOp    = aether::SumOp<int>;
using logicalAnd  = aether::LogicalAndOp;
/* The emulated arm's carrier: `aether::banded::Band`, three
 * bare FP32 limbs, warp-transported by eagle/cuda/WarpShuffle.h as its raw
 * per-limb bit patterns. `BandedReal` is aether's PUBLIC host-only
 * `double` ingest/egress terminal pair (`Band` itself walls off every mix
 * with a native float/double on purpose). */
using SdT         = aether::banded::Band;
using BandedRealT = aether::banded::BandedReal;

/** @brief `double` -> the emulated carrier (host only). @see BandedReal. */
static SdT bandOf(double v) { return SdT(BandedRealT::fromDouble(v)); }
/** @brief …and back. Exact inverse of `bandOf` on every tier-1 value. */
static double asDouble(SdT b) { return BandedRealT(b).toDouble(); }
using sdSumOp     = aether::SumOp<SdT>;
using sdMaxOp     = aether::MaxOp<SdT>;
using doubleMaxOp = aether::MaxOp<double>;
using intMaxOp    = aether::MaxOp<int>;

TEST_F(ReductionTest, CUDA_OmpIntegers_TinyArray)
{
    constexpr idx_t testSize = 16;
    constexpr int value      = 1;
    auto arr = eagle::makeArray<int>(testSize, value);
    testCudaReduction<int, intSumOp>(arr, testSize * value);
}

TEST_F(ReductionTest, CUDA_WithIntegers_TinyArray)
{
    constexpr idx_t testSize = 16;
    constexpr int value      = 1;
    auto arr = eagle::makeArray<int>(testSize, value);
    testCudaReduction<int, intSumOp>(arr, testSize * value);
}

TEST_F(ReductionTest, OMP_WithIntegers_BigArray)
{
    const idx_t testSize = isMinimalMode() ? static_cast<idx_t>(10'000)
                                           : static_cast<idx_t>(1e8 + 31);
    constexpr int value      = 1;
    auto arr = eagle::makeArray<int>(testSize, value);
    testOmpReduction<int, intSumOp>(arr, testSize * value);
}

TEST_F(ReductionTest, CUDA_WithIntegers_BigArray)
{
    const idx_t testSize = isMinimalMode() ? static_cast<idx_t>(10'000)
                                           : static_cast<idx_t>(1e8 + 31);
    constexpr int value      = 1;
    auto arr = eagle::makeArray<int>(testSize, value);
    testCudaReduction<int, intSumOp>(arr, testSize * value);
}

TEST_F(ReductionTest, CUDA_InPlace_ThrowsError)
{
    const idx_t testSize = isMinimalMode() ? static_cast<idx_t>(10'000)
                                           : static_cast<idx_t>(1e8 + 31);
    constexpr int value      = 1;
    auto arr = eagle::makeArray<int>(testSize, value);
    auto stmt = [this, &arr, testSize]() {
        testCudaReduction<int, intSumOp, true>(arr, testSize * value);
    };
    EXPECT_THROW(stmt(), std::runtime_error);
}

TEST_F(ReductionTest, OMP_SumWithDoubles)
{
    const idx_t testSize = isMinimalMode() ? static_cast<idx_t>(10'000)
                                           : static_cast<idx_t>(1e6 + 37);
    constexpr double value   = -0.0000031;
    auto arr = eagle::makeArray<double>(testSize, value);
    eagle::GRefArrT<double> aref = arr.hostView();

    for (idx_t i = 0; i < idx_t(arr.samples()); i++) {
        aref(i) = randReal();
    }

    using OP = doubleSumOp;
    testOmpReduction<double, OP>(
        arr, dumbCPUReduction<double, OP>(arr), 0, 1e-6);
}

TEST_F(ReductionTest, CUDA_SumWithDoubles)
{
    const idx_t testSize = isMinimalMode() ? static_cast<idx_t>(10'000)
                                           : static_cast<idx_t>(1e6 + 37);
    constexpr double value   = -0.0000031;
    auto arr = eagle::makeArray<double>(testSize, value);
    eagle::GRefArrT<double> aref = arr.hostView();

    for (idx_t i = 0; i < idx_t(arr.samples()); i++) {
        aref(i) = randReal();
    }

    using OP = doubleSumOp;
    testCudaReduction<double, OP>(
        arr, dumbCPUReduction<double, OP>(arr), 0, 1e-6);
}

TEST_F(ReductionTest, OMP_MaxWithFloats)
{
    constexpr idx_t testSize = 1e4;
    constexpr float haystack = -0.00031;
    constexpr float needle   = 42;
    auto arr = eagle::makeArray<float>(testSize, haystack);
    eagle::GRefArrT<float> aref = arr.hostView();

    aref(123) = needle;

    using OP = floatMaxOp;
    testOmpReduction<float, OP>(arr, needle);
    testOmpReduction<float, OP>(arr, dumbCPUReduction<float, OP>(arr));
}

TEST_F(ReductionTest, CUDA_MaxWithFloats)
{
    constexpr idx_t testSize = 1e4;
    constexpr float haystack = -0.00031;
    constexpr float needle   = 42;
    auto arr = eagle::makeArray<float>(testSize, haystack);
    eagle::GRefArrT<float> aref = arr.hostView();

    aref(123) = needle;

    using OP = floatMaxOp;
    testCudaReduction<float, OP>(arr, needle);
    testCudaReduction<float, OP>(arr, dumbCPUReduction<float, OP>(arr));
}

/* Pins the integral device leg (aether's AETHER_MATH_BINARY_ORD): max<int> must
 * compile and run in device code — the suite was double/float-only here, which
 * is how a host-only fmax leg once survived unnoticed. */
TEST_F(ReductionTest, CUDA_MaxWithIntegers)
{
    constexpr idx_t testSize = 1e4;
    constexpr int haystack   = -31;
    constexpr int needle     = 42;
    auto arr = eagle::makeArray<int>(testSize, haystack);
    eagle::GRefArrT<int> aref = arr.hostView();

    aref(123) = needle;

    using OP = intMaxOp;
    testCudaReduction<int, OP>(arr, needle);
    testCudaReduction<int, OP>(arr, dumbCPUReduction<int, OP>(arr));
}

TEST_F(ReductionTest, OMP_AndWithBools)
{
    using OP                 = logicalAnd;
    constexpr idx_t testSize = 1e4;
    auto arr = eagle::makeArray<bool>(testSize, true);
    eagle::GRefArrT<bool> aref = arr.hostView();

    testOmpReduction<bool, OP>(arr, true, true);
    aref(123) = false;
    testOmpReduction<bool, OP>(arr, false, true);
}

TEST_F(ReductionTest, CUDA_AndWithBools)
{
    using OP                 = logicalAnd;
    constexpr idx_t testSize = 1e4;
    auto arr = eagle::makeArray<bool>(testSize, true);
    eagle::GRefArrT<bool> aref = arr.hostView();

    testCudaReduction<bool, OP>(arr, true, true);
    aref(123) = false;
    testCudaReduction<bool, OP>(arr, false, true);
}

/* ------------------ banded carrier (FP64 emulation) ----------------------- */
// Device-only per the emulation policy (no OMP twins). The warp stage
// transports the carrier as its three raw FP32 limb bit patterns
// (eagle/cuda/WarpShuffle.h) — max parity vs the double twin is EXACT
// (comparisons are picks of exactly-loaded values); sum parity is
// tolerance-based (emulated adds are correctly rounded but can flip RNE ties
// — never assert bit-exact arithmetic parity, the numerics contract's item 1).
//
// The TEST NAMES below still say "SoftDouble": they are pinned by the
// committed name manifest (tests/expected_tests_cuda.txt) and a rename is a
// re-mint, which is a build-maintenance call, not this port's. The CARRIER is
// what moved.
//
// DEVIATION (aether rule): `Band`'s exponent range IS FP32's, and the storage
// codec admits only tier 1's exact window [2^-94, 2^126)
// (aether/banded/BandCell8.h — `BandedReal::fromDouble` THROWS outside it).
// An FP64-range carrier would put the max-reduction identity element at
// -1e300; here it moves to -1e30. It is still a strict lower bound on
// every filled value below (which live in [-1250, +1500]), so the assertion
// this test makes is unchanged.

TEST_F(ReductionTest, CUDA_SoftDoubleMax_MatchesDoubleReference)
{
    constexpr idx_t testSize = 100000;
    auto d = eagle::makeArray<double>(testSize);
    auto s = eagle::makeArray<SdT>(testSize);
    auto dref = d.hostView();
    auto sref = s.hostView();
    for (idx_t i = 0; i < testSize; i++) {
        // deterministic mixed-sign, mixed-magnitude fill
        const double v = ((i * 2654435761u) % 2000003) * 1.375e-3 - 1250.0;
        dref(i)        = v;
        sref(i)        = bandOf(v);
    }
    d.upload();
    s.upload();
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());

    cudaStream_t stream;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));
    const double dm = eagle::cuda::Reduction::reduceBlocking<double,
        doubleMaxOp>(d, -1e300, stream);
    const SdT sm = eagle::cuda::Reduction::reduceBlocking<SdT, sdMaxOp>(
        s, bandOf(-1e30), stream);
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));

    EXPECT_EQ(asDouble(sm), dm);
}

TEST_F(ReductionTest, CUDA_SoftDoubleSum_MatchesDoubleReference)
{
    constexpr idx_t testSize = 100000;
    auto d = eagle::makeArray<double>(testSize);
    auto s = eagle::makeArray<SdT>(testSize);
    auto dref = d.hostView();
    auto sref = s.hostView();
    for (idx_t i = 0; i < testSize; i++) {
        // positive values in [0.5, 2.5) — no cancellation in the fold
        const double v = 0.5 + ((i * 40503u) % 8192) * (2.0 / 8192.0);
        dref(i)        = v;
        sref(i)        = bandOf(v);
    }
    d.upload();
    s.upload();
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());

    cudaStream_t stream;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));
    const double ds = eagle::cuda::Reduction::reduceBlocking<double,
        doubleSumOp>(d, 0.0, stream);
    const SdT ss = eagle::cuda::Reduction::reduceBlocking<SdT, sdSumOp>(
        s, bandOf(0.0), stream);
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));

    // ~1.5e5 magnitude sum; 1e-9 abs tol = ~1e-14 relative, orders above
    // any tie-flip accumulation, orders below a transport/fold bug.
    EXPECT_NEAR(asDouble(ss), ds, 1e-9);
}

} // namespace ReductionTest
} // namespace eagle_tests
