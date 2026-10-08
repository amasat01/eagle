// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* DeviceProps gate (CUDA-free half): the device-property facility's
 * pure arithmetic -- eagle::deriveFields()'s formulas, the fmaLanesPerSM /
 * fp64Ratio compute-capability tables (including their fallback path for an
 * unlisted cc), and eagle::cpu::deviceProps()'s documented host-values path --
 * every one of which is plain C++ over already-known integers, no CUDA API
 * call anywhere. Deliberately NOT "TestBase.h" (same reasoning as
 * test_HostDispatchContract.cpp: that header pulls in the full eagle.h
 * umbrella, whose __device__/__host__-qualified bodies only nvcc parses, and
 * in CUDA mode this .cpp is compiled by CXX/g++, not nvcc) -- eagle/
 * DeviceProps.h and eagle/cpu/DeviceProps.h are CUDA-free by design
 * (mirroring eagle/launch/Traits.h), so this TU needs no forced
 * EAGLE_CPU_ONLY define either, unlike its siblings that do
 * reach eagle/typedefs.h. Dual-registered into BOTH build modes
 * (CMakeLists.txt), same as test_ConformanceCorpus.cpp. The CUDA-mode-only
 * counterpart (eagle::cuda::deviceProps against real hardware, including the
 * dev-box known-answer) lives in test_DeviceProps.cu. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "eagle/DeviceProps.h"
#include "eagle/cpu/DeviceProps.h"

namespace eagle_tests {
namespace DevicePropsTest {

class DevicePropsTest : public ::testing::Test { };

// --------------------------------------------------------------------------
// eagle::cpu::deviceProps() -- the host-props path.
// --------------------------------------------------------------------------

TEST_F(DevicePropsTest, CpuRawFieldSanity)
{
    const eagle::DeviceProps props = eagle::cpu::deviceProps();
    EXPECT_GT(props.sm_count, 0)
        << "cpu::deviceProps() must report at least one logical CPU";
    EXPECT_GT(props.warp_size, 0);
    EXPECT_EQ(props.cc_major, 0)
        << "the host profile documents cc 0.0 as its no-CUDA sentinel";
    EXPECT_EQ(props.cc_minor, 0);
}

TEST_F(DevicePropsTest, CpuDerivedFieldsAreFiniteAndPositive)
{
    const eagle::DeviceProps props = eagle::cpu::deviceProps();
    EXPECT_GT(props.peak_bytes_per_s, 0.0);
    EXPECT_GT(props.peak_flops_sp, 0.0);
    EXPECT_GT(props.peak_flops_dp, 0.0);
    EXPECT_GT(props.fp64_ratio, 0.0);
}

TEST_F(DevicePropsTest, CpuRidgeMatchesFlopsOverBytesExactly)
{
    const eagle::DeviceProps props = eagle::cpu::deviceProps();
    EXPECT_DOUBLE_EQ(props.ridgeFlopsPerByte("float32"),
        props.peak_flops_sp / props.peak_bytes_per_s);
    EXPECT_DOUBLE_EQ(props.ridgeFlopsPerByte("float64"),
        props.peak_flops_dp / props.peak_bytes_per_s);
}

// --------------------------------------------------------------------------
// eagle::deriveFields() / the compute-capability table fallback paths.
// --------------------------------------------------------------------------

TEST_F(DevicePropsTest, FmaLanesFallsBackTo64ForAnUnknownComputeCapability)
{
    EXPECT_EQ(eagle::fmaLanesPerSM(/*ccMajor=*/99, /*ccMinor=*/9), 64);
}

TEST_F(DevicePropsTest, Fp64RatioFallsBackTo32ForAnUnknownComputeCapability)
{
    EXPECT_EQ(eagle::fp64Ratio(/*ccMajor=*/99, /*ccMinor=*/9), 32);
}

TEST_F(DevicePropsTest, Fp64RatioKnownAnswerPascalConsumer)
{
    // Same table entry the CUDA-mode known-answer test exercises live
    // against the dev box (test_DeviceProps.cu) -- pinned here too,
    // CUDA-free, as a table-content regression guard independent of any GPU
    // being present.
    EXPECT_EQ(eagle::fp64Ratio(/*ccMajor=*/6, /*ccMinor=*/1), 32);
}

TEST_F(DevicePropsTest, DeriveFieldsRidgeMatchesFlopsOverBytesExactlyOnAHandBuiltProfile)
{
    eagle::DeviceProps props;
    props.cc_major              = 8;
    props.cc_minor              = 0;
    props.sm_count               = 108;
    props.clock_rate_khz        = 1'410'000;
    props.memory_clock_rate_khz = 1'215'000;
    props.memory_bus_width_bits = 5120;
    props = eagle::deriveFields(props);

    EXPECT_DOUBLE_EQ(props.ridgeFlopsPerByte("float32"),
        props.peak_flops_sp / props.peak_bytes_per_s);
    EXPECT_DOUBLE_EQ(props.ridgeFlopsPerByte("float64"),
        props.peak_flops_dp / props.peak_bytes_per_s);
    // An A100-class profile: fp64Ratio(8, 0) == 2 (datacenter table entry).
    EXPECT_DOUBLE_EQ(props.fp64_ratio, 2.0);
    EXPECT_DOUBLE_EQ(props.peak_flops_dp, props.peak_flops_sp / 2.0);
}

TEST_F(DevicePropsTest, RidgeFlopsPerByteIsZeroWhenPeakBytesPerSIsZero)
{
    // A degenerate/unpopulated profile must not divide by zero.
    eagle::DeviceProps props;
    props = eagle::deriveFields(props);
    EXPECT_EQ(props.peak_bytes_per_s, 0.0);
    EXPECT_EQ(props.ridgeFlopsPerByte("float32"), 0.0);
    EXPECT_EQ(props.ridgeFlopsPerByte("float64"), 0.0);
}

} // namespace DevicePropsTest
} // namespace eagle_tests
