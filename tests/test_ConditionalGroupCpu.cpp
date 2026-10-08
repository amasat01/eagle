// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* CPU parity twin for eagle::conditional::ConditionalGroup.
 *
 * Deliberately NOT "TestBase.h": that header pulls in the full eagle.h
 * umbrella (cuda.h/reduce.h/...), whose __device__/__host__-qualified
 * bodies only nvcc parses -- and in CUDA mode this .cpp is compiled by CXX
 * (g++), not nvcc (same trap test_ConformanceCorpus.cpp / test_LaunchTraits
 * .cpp document). eagle/conditional.h itself is CUDA-free once
 * EAGLE_CPU_ONLY is defined (every CUDA-only header along its include chain
 * self-guards to empty), so this TU is explicitly compiled with
 * -DEAGLE_CPU_ONLY (CMakeLists.txt COMPILE_DEFINITIONS on this one source),
 * independent of whichever EAGLE_CPP_MODE the ambient build uses --
 * dual-registered into BOTH build modes per the test_LaunchTraits.cpp
 * recipe.
 *
 * Same three-leg skip-proof shape as the CUDA G-BEHAV gate, ported to the
 * host executor: the tracer is a native node whose runHost increments a
 * host counter with no liveness check of its own; the "replay" is
 * cpu::Graph::run(), called three times on the SAME graph with only the
 * guard's host word mutated between calls -- never rebuilt.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "eagle/conditional.h"

#include <stdexcept>
#include <utility>

namespace eagle_tests {
namespace ConditionalGroupCpuTest {

using eagle::CountGuard;
using eagle::conditional::ConditionalGroup;

namespace {

/* The unguarded tracer: runHost increments unconditionally, with no
 * fencepost / liveness check of its own -- the host-side analogue of the
 * CUDA test's bare atomicAdd kernel. */
struct TracerNode {
    int* counter;
    void reserveScratch(eagle::ScratchArena&) { }
    void runHost(eagle::cpu::Graph&) { (*counter)++; }
};

}  // namespace

// G-CPU primary: the three-leg skip-proof against cpu::Graph::run(), ONE
// graph, no rebuild between legs.
TEST(ConditionalGroupCpuTest, ThreeLegSkipProof)
{
    unsigned int count = 1;
    int tracer          = 0;

    eagle::cpu::Graph g;
    ConditionalGroup group(CountGuard{ &count, nullptr });
    group.body().addNative(TracerNode{ &tracer }, {});
    g.addNative(std::move(group), {});

    // Leg 1: guard TRUE (count=1 != baseline 0) -> body fires -> tracer==1.
    g.run();
    ASSERT_EQ(tracer, 1) << "leg 1 (guard TRUE) must fire the body exactly once";

    // Leg 2: mutate ONLY the guard's host word -> body must NOT fire ->
    // tracer stays at 1 (the skip-proof: the tracer has no liveness check
    // of its own).
    count = 0;
    g.run();
    ASSERT_EQ(tracer, 1) << "leg 2 (guard FALSE) must leave the tracer silent";

    // Leg 3: restore guard TRUE, SAME graph, no rebuild -> tracer==2. Kills
    // a build-time bake of the predicate (the most likely wrong
    // implementation): that would have frozen the leg-1 decision.
    count = 1;
    g.run();
    ASSERT_EQ(tracer, 2) << "leg 3 (guard restored TRUE) must re-fire the body";
}

// Pinned prose: "runHost EAGLE_ASSERTs a nonempty host body when the guard
// passes" -- a ConditionalGroup whose body was never populated must fail
// loudly (not silently do nothing) the moment its guard fires.
TEST(ConditionalGroupCpuTest, RunHostAssertsOnEmptyBodyWhenGuardPasses)
{
    unsigned int count = 1;  // guard TRUE: base 0, count != 0
    eagle::cpu::Graph g;
    ConditionalGroup group(CountGuard{ &count, nullptr });  // body() never touched
    g.addNative(std::move(group), {});

    ASSERT_THROW(g.run(), std::runtime_error)
        << "an empty host body must assert loudly when the guard passes, "
           "never silently no-op";
}

}  // namespace ConditionalGroupCpuTest
}  // namespace eagle_tests
