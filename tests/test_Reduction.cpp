// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "eagle/cuda/Reduction.h"
#include "eagle/cpu/Reduction.h"

#include "TestBase.h"

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

TEST_F(ReductionTest, OMP_WithIntegers_BigArray)
{
    const idx_t testSize = isMinimalMode() ? static_cast<idx_t>(10'000)
                                           : static_cast<idx_t>(1e8 + 31);
    constexpr int value  = 1;
    auto arr = eagle::makeArray<int>(testSize, value);
    testOmpReduction<int, intSumOp>(arr, testSize * value);
}

TEST_F(ReductionTest, OMP_SumWithDoubles)
{
    const idx_t testSize   = isMinimalMode() ? static_cast<idx_t>(10'000)
                                             : static_cast<idx_t>(1e6 + 37);
    constexpr double value = -0.0000031;
    auto arr                    = eagle::makeArray<double>(testSize, value);
    const eagle::GRefArrT<double> aref = arr.hostView();

    for (idx_t i = 0; i < idx_t(arr.samples()); i++) {
        aref(i) = randReal();
    }

    using OP = doubleSumOp;
    testOmpReduction<double, OP>(
        arr, dumbCPUReduction<double, OP>(arr), 0, 1e-6);
}

TEST_F(ReductionTest, OMP_MaxWithFloats)
{
    constexpr idx_t testSize = 1e4;
    constexpr float haystack = -0.00031;
    constexpr float needle   = 42;
    auto arr                          = eagle::makeArray<float>(testSize, haystack);
    const eagle::GRefArrT<float> aref = arr.hostView();

    aref(123) = needle;

    using OP = floatMaxOp;
    testOmpReduction<float, OP>(arr, needle);
    testOmpReduction<float, OP>(arr, dumbCPUReduction<float, OP>(arr));
}

TEST_F(ReductionTest, OMP_AndWithBools)
{
    using OP                 = logicalAnd;
    constexpr idx_t testSize = 1e4;
    auto arr                         = eagle::makeArray<bool>(testSize, true);
    const eagle::GRefArrT<bool> aref = arr.hostView();

    testOmpReduction<bool, OP>(arr, true, true);
    aref(123) = false;
    testOmpReduction<bool, OP>(arr, false, true);
}

} // namespace ReductionTest
} // namespace eagle_tests
