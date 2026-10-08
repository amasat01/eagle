// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Contract-guard test for eagle::cpu::Host::launch's region-required
 * work-sharing contract. Host.h hard-codes `parallel=false`: `launch` NEVER
 * opens its own `#pragma omp parallel` region -- work is shared across an
 * EXISTING region's threads (`packetBatchedFor` -> `packetFor`, a
 * `#pragma omp for schedule(static) nowait` loop in aether's
 * `aether/backend/cpu/Tiled.h`) and otherwise the whole range runs SERIALLY
 * on the calling thread. A consumer that assumes `launch` spreads across
 * threads unconditionally gets a silent full serialization: it compiles,
 * computes correct values, and loses all threading. Values alone cannot see
 * this -- only thread identity can, hence this dedicated instrument, with a
 * companion negative control below for the OTHER half of the same
 * documented contract.
 *
 * Deliberately NOT "TestBase.h": that header pulls in the full eagle.h
 * umbrella (cuda.h/reduce.h/...), whose __device__/__host__-qualified
 * bodies only nvcc parses -- and in CUDA mode this .cpp is compiled by CXX
 * (g++), not nvcc (same trap test_ConformanceCorpus.cpp / test_LaunchTraits
 * .cpp / test_ConditionalGroupCpu.cpp document). eagle/cpu/Host.h itself is
 * CUDA-free, but it reaches eagle/typedefs.h -> <aether/aether.h>, whose
 * AETHER_DEVICE()/AETHER_HOST() macros already expand to nothing for a
 * host-compiler TU of a CUDA-mode build (the AETHER_DEVICE_COMPILER axis --
 * no per-source AETHER_CPP_MODE needed, unlike a per-source CPU-only
 * hatch); this TU is still explicitly compiled with -DEAGLE_CPU_ONLY
 * (CMakeLists.txt COMPILE_DEFINITIONS on this one source) to select eagle's
 * own host-face path, independent of whichever EAGLE_CPP_MODE the ambient
 * build uses -- dual-registered into BOTH build modes per the
 * test_LaunchTraits.cpp recipe. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "eagle/cpu/Host.h"

#include <omp.h>
#include <set>
#include <vector>

namespace eagle_tests {
namespace HostDispatchContractTest {

using eagle::idx_t;

class HostDispatchContractTest : public ::testing::Test { };

TEST_F(HostDispatchContractTest, SpreadsAcrossThreadsInsideParallelRegion)
{
    // omp_set_num_threads is a process-global mutation of the OpenMP
    // runtime; gtest runs this suite shuffled -- restore what we found.
    const int previousThreads = omp_get_max_threads();
    omp_set_num_threads(4);
    ASSERT_GE(omp_get_max_threads(), 2)
        << "the OpenMP runtime refuses to give this process more than one "
           "thread, so thread spreading cannot be observed here -- that is "
           "an environment failure, not a code failure";

    // Large enough that the tile scheduler gives every thread several
    // tiles (it trims the tile size so each thread gets at least two).
    const idx_t nSamples = 65536;
    std::vector<int> owner(nSamples, -1);

#pragma omp parallel
    {
        eagle::cpu::Host::launch(nSamples, [&](const auto& i) {
            owner[i.global()] = omp_get_thread_num();
        });
    }

    omp_set_num_threads(previousThreads);

    std::set<int> distinct;
    for (idx_t i = 0; i < nSamples; ++i) {
        ASSERT_GE(owner[i], 0)
            << "sample " << i << " was never visited by Host::launch";
        distinct.insert(owner[i]);
    }

    EXPECT_GT(distinct.size(), 1u)
        << "Host::launch ran the whole range on ONE thread despite an "
           "enclosing '#pragma omp parallel' region -- the work-share arm "
           "(the tile-batched path) did not activate.";
}

TEST_F(HostDispatchContractTest, RunsSerialWithoutEnclosingParallelRegion)
{
    // Companion negative control: the OTHER half of the SAME documented
    // contract (eagle/cpu/Host.h doc comment on `launch`) -- called bare,
    // with no enclosing '#pragma omp parallel', `launch` must run the WHOLE
    // range on the calling thread, even with multiple threads available.
    // This is exactly the silent-serialization trap the old
    // docs/examples/02_host_dispatch.cpp comment ("spreads them across
    // OpenMP threads" unconditionally) would have hidden from a reader.
    const int previousThreads = omp_get_max_threads();
    omp_set_num_threads(4);

    ASSERT_FALSE(omp_in_parallel())
        << "test precondition violated: must start outside any OpenMP "
           "region for this to be a meaningful negative control";

    const idx_t nSamples = 65536;
    std::vector<int> owner(nSamples, -1);

    eagle::cpu::Host::launch(nSamples, [&](const auto& i) {
        owner[i.global()] = omp_get_thread_num();
    });

    omp_set_num_threads(previousThreads);

    std::set<int> distinct;
    for (idx_t i = 0; i < nSamples; ++i) {
        ASSERT_GE(owner[i], 0)
            << "sample " << i << " was never visited by Host::launch";
        distinct.insert(owner[i]);
    }

    EXPECT_EQ(distinct.size(), 1u)
        << "Host::launch spread across " << distinct.size()
        << " threads with NO enclosing '#pragma omp parallel' region -- "
           "either the documented serial-outside-a-region contract changed "
           "(update eagle/cpu/Host.h's doc comment to match) or this "
           "instrument is unsound.";
}

} // namespace HostDispatchContractTest
} // namespace eagle_tests
