// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* DeviceProps gate (CUDA-hardware half): eagle::cuda::deviceProps()
 * against the REAL device -- raw-field sanity, derived-field consistency
 * (ridge == flops/bytes exactly, the same arithmetic the CPU-side twin in
 * test_DeviceProps.cpp checks against a synthetic profile), and the dev-box
 * known-answer (Quadro P2000: cc == (6, 1), fp64_ratio == 32). The CUDA-free
 * half (fmaLanesPerSM/fp64Ratio table + fallback, eagle::cpu::deviceProps())
 * lives in test_DeviceProps.cpp, dual-registered into both build modes. */
#include "eagle/cuda/DeviceProps.h"

#include "TestBase.h"

namespace eagle_tests {
namespace DevicePropsCudaTest {

class DevicePropsCudaTest : public Test { };

TEST_F(DevicePropsCudaTest, RawFieldSanity)
{
    const eagle::DeviceProps props = eagle::cuda::deviceProps(0);
    EXPECT_GT(props.sm_count, 0);
    EXPECT_EQ(props.warp_size, 32);
    EXPECT_GT(props.cc_major, 0);
    EXPECT_GT(props.clock_rate_khz, 0);
    EXPECT_GT(props.memory_clock_rate_khz, 0);
    EXPECT_GT(props.memory_bus_width_bits, 0);
    EXPECT_GT(props.shared_mem_per_block, 0u);
    EXPECT_GT(props.regs_per_block, 0);
}

TEST_F(DevicePropsCudaTest, DerivedFieldsAreFiniteAndPositive)
{
    const eagle::DeviceProps props = eagle::cuda::deviceProps(0);
    EXPECT_GT(props.peak_bytes_per_s, 0.0);
    EXPECT_GT(props.peak_flops_sp, 0.0);
    EXPECT_GT(props.peak_flops_dp, 0.0);
    EXPECT_GT(props.fp64_ratio, 0.0);
}

TEST_F(DevicePropsCudaTest, RidgeMatchesFlopsOverBytesExactly)
{
    const eagle::DeviceProps props = eagle::cuda::deviceProps(0);
    EXPECT_DOUBLE_EQ(props.ridgeFlopsPerByte("float32"),
        props.peak_flops_sp / props.peak_bytes_per_s);
    EXPECT_DOUBLE_EQ(props.ridgeFlopsPerByte("float64"),
        props.peak_flops_dp / props.peak_bytes_per_s);
}

/* Known-answer pin for the dev-box GPU (a Quadro P2000, sm_61) -- this
 * repo's standing target-hardware fact ("P2000 = dev/test").
 * Gated on the device NAME, not on cc: any other GPU (a CI runner, say)
 * skips with its identity in the message, while a P2000 reporting anything
 * but (6, 1, 32x) still FAILS -- the assertions stay live exactly where the
 * known answer applies. Gating on cc itself would make the cc assertions
 * tautological. The hardware-independent checks live in the three tests
 * above. */
TEST_F(DevicePropsCudaTest, KnownAnswerDevBoxIsPascalConsumerFp64Ratio32)
{
    const eagle::DeviceProps props = eagle::cuda::deviceProps(0);
    if (props.name.find("P2000") == std::string::npos) {
        GTEST_SKIP() << "known-answer pin is for the Quadro P2000 dev box; "
                        "this device is '" << props.name << "'";
    }
    EXPECT_EQ(props.cc_major, 6);
    EXPECT_EQ(props.cc_minor, 1);
    EXPECT_DOUBLE_EQ(props.fp64_ratio, 32.0);
}

} // namespace DevicePropsCudaTest
} // namespace eagle_tests
