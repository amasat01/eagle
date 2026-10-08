// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

// CUDA headers define AETHER_NOINLINE() which interferes with libstdc++'s use of
// `__attribute((AETHER_NOINLINE()))`. In order to avoid compilation error,
// temporarily unset AETHER_NOINLINE() when we include affected libstdc++ header.
// Only applies when compiling with clang(d).
// See https://github.com/llvm/llvm-project/issues/62939#issuecomment-1563455451

#ifdef __clang__
#pragma push_macro("AETHER_NOINLINE()")
#undef AETHER_NOINLINE()
#endif
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <random>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop
#ifdef __clang__
#pragma pop_macro("AETHER_NOINLINE()")
#endif

#include "eagle/typedefs.h"
#include "eagle/eagle.h"

namespace eagle_tests {
#define TEST_SIZE 320 + 1
#define TEST_N_BLOCKS (TEST_SIZE + EAGLE_BLOCKSIZE - 1) / EAGLE_BLOCKSIZE

using eagle::mInt_t;
using eagle::Real;

/** @brief Base class for test suites */
class Test : public ::testing::Test {
public:
    Test()
        : rng_(rd_())
        , uni_(0., 1.)
        , uniInt_(0, 100000)
        , bool_(0.5)
    {
        // Deliberately THROWING here: device-error residue at test START is a
        // real defect and gtest catches fixture-constructor exceptions, so it
        // reds THIS test visibly instead of aborting the process.
        resetDeviceErrors_();
        eagle::util::setLogLevel(eagle::util::OFF);
    }

    // The destructor twin must NEVER throw (a throwing destructor is
    // std::terminate: the whole suite dies rc 134 with no gtest summary —
    // exactly how NestedGroupThrowsNotSupported's deliberate 801, surfacing
    // at the teardown sync on the CI runner's newer driver, killed the CUDA
    // debug job). Same doctrine as EAGLE_CHECK_NOTHROW
    // (eagle/util/DeviceError.h): drain + log, never fatal. Residue that
    // survives the drain still fails the NEXT test's constructor check.
    ~Test() { drainDeviceErrorsNothrow_(); }

    /** @brief True when EAGLE_TEST_MINIMAL=1 is set in the environment.
     *  Expensive tests (reduction/scan on large arrays) should branch on this
     *  to keep sanitize wall-time tractable. */
    static bool isMinimalMode()
    {
        const char* e = std::getenv("EAGLE_TEST_MINIMAL");
        return e != nullptr && e[0] == '1' && e[1] == '\0';
    }

protected:
    void resetDeviceErrors_()
    {
        EAGLE_KERNEL_PRE();
        EAGLE_KERNEL_POST();
    }

    void drainDeviceErrorsNothrow_() noexcept
    {
        try {
            resetDeviceErrors_();
        } catch (const std::exception& e) {
            std::fprintf(stderr,
                "TestBase teardown: device error drained (swallowed): %s\n",
                e.what());
        } catch (...) {
            std::fprintf(stderr,
                "TestBase teardown: non-std device error drained (swallowed)\n");
        }
    }

    /** @brief Random `Real` sampled from a uniform distribution [0; 1) */
    Real randReal() { return uni_(rng_); }
    Real randReal(const Real& min, const Real& max)
    {
        Real val = randReal();
        return val * (max - min) + min;
    }
    mInt_t randInt() { return uniInt_(rng_); }
    bool randBool() { return bool_(rng_); }

    // Random seed
    std::random_device rd_;
    // Random Number Generator
    std::mt19937 rng_;
    // Uniform distribution
    std::uniform_real_distribution<Real> uni_;
    // Uniform real distribution
    std::uniform_int_distribution<> uniInt_;
    // Bernoully distribution
    std::bernoulli_distribution bool_;
};
} // namespace eagle_tests
