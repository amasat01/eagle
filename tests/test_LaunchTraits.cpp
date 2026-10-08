// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Unit tests for eagle::launch::Traits / eagle::launch::minBlocksPerSM.
 * Expected values are HAND-DERIVED in the comments
 * beside each assertion -- never recomputed via the minBlocksPerSM()
 * expression itself, so a broken implementation cannot rubber-stamp its own
 * answer.
 *
 * Deliberately NOT "TestBase.h": that header pulls in the full eagle.h
 * umbrella (cuda.h/reduce.h/...), whose __device__/__host__-qualified
 * bodies only nvcc parses -- and in CUDA mode this `.cpp` is compiled by
 * CXX (g++), not nvcc (same trap test_ConformanceCorpus.cpp documents).
 * eagle/launch/Traits.h is itself CUDA-free, but it reaches
 * eagle/typedefs.h -> <aether/aether.h>, whose AETHER_DEVICE()/AETHER_HOST()
 * macros already expand to nothing for a host-compiler TU of a CUDA-mode
 * build (aether/macros.h's AETHER_DEVICE_COMPILER axis -- no per-source
 * AETHER_CPP_MODE escape needed);
 * this TU is still explicitly compiled with -DEAGLE_CPU_ONLY
 * (CMakeLists.txt COMPILE_DEFINITIONS on this one source) to select eagle's
 * own host-face path, independent of whichever EAGLE_CPP_MODE the ambient
 * build uses, exactly like the eagle_probe_launch_traits_cpu_only compile
 * probe. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "eagle/launch/Traits.h"

namespace eagle_tests {
namespace LaunchTraitsTest {

using eagle::launch::kSmMaxThreads;
using eagle::launch::kSmRegFile;
using eagle::launch::minBlocksPerSM;
using eagle::launch::Traits;

/* Traits<Max,MinPerSM> is a pure value carrier: the two template args come
 * straight back out as the matching static members. */
static_assert(Traits<256, 4>::maxBlockSize == 256);
static_assert(Traits<256, 4>::minBlocksPerSM == 4);
static_assert(Traits<128, 8>::maxBlockSize == 128);
static_assert(Traits<128, 8>::minBlocksPerSM == 8);

/* SM machine-model constants (sm_61..sm_90; extension point for a later
 * arch, not encoded here). */
static_assert(kSmRegFile == 65536);
static_assert(kSmMaxThreads == 2048);

/* minBlocksPerSM(maxBlockSize, regsPerThreadBudget)
 *   = kSmRegFile / (regsPerThreadBudget * maxBlockSize).
 * Hand-derived (independent arithmetic, not the impl expression) -- these
 * match this budget's static_asserts (independently, without calling it):
 *   65536 / (128 * 128) = 65536 / 16384 = 4
 *   65536 / (128 * 256) = 65536 / 32768 = 2
 */
static_assert(minBlocksPerSM(128, 128) == 4);
static_assert(minBlocksPerSM(256, 128) == 2);

/* Runtime-visible mirror of the same facts, so a broken build shows up as a
 * FAILED gtest case (not just a compile error) when the binary is run
 * directly. */
TEST(LaunchTraitsTest, TraitsCarriesTemplateArgsAsStaticMembers)
{
    EXPECT_EQ((Traits<256, 4>::maxBlockSize), 256);
    EXPECT_EQ((Traits<256, 4>::minBlocksPerSM), 4);
    EXPECT_EQ((Traits<128, 8>::maxBlockSize), 128);
    EXPECT_EQ((Traits<128, 8>::minBlocksPerSM), 8);
}

TEST(LaunchTraitsTest, MinBlocksPerSMMatchesHandDerivedValues)
{
    // 65536 / (128 * 128) = 4
    EXPECT_EQ(minBlocksPerSM(128, 128), 4);
    // 65536 / (128 * 256) = 2
    EXPECT_EQ(minBlocksPerSM(256, 128), 2);
}

TEST(LaunchTraitsTest, MachineModelConstants)
{
    EXPECT_EQ(kSmRegFile, 65536);
    EXPECT_EQ(kSmMaxThreads, 2048);
}

} // namespace LaunchTraitsTest
} // namespace eagle_tests
