// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* CPU parity/reference twin (mirrors test_ConditionalGroupCpu.cpp's
 * own header note) for eagle::compose::GraphComposer's CPU face.
 *
 * Deliberately NOT "TestBase.h": that header pulls in the full eagle.h
 * umbrella (cuda.h/reduce.h/...), whose __device__/__host__-qualified
 * bodies only nvcc parses -- and in CUDA mode this .cpp is compiled by CXX
 * (g++), not nvcc. eagle/compose/GraphComposer.h itself is CUDA-free once
 * EAGLE_CPU_ONLY is defined (every CUDA-only header along its include chain
 * self-guards to empty), so this TU is explicitly compiled with
 * -DEAGLE_CPU_ONLY (CMakeLists.txt COMPILE_DEFINITIONS on this one source),
 * independent of whichever EAGLE_CPP_MODE the ambient build uses --
 * dual-registered into BOTH build modes, same recipe as
 * test_ConditionalGroupCpu.cpp / test_LaunchTraits.cpp.
 *
 * The CPU face is the REFERENCE implementation: every mode
 * ("sequenced"/"enabled"/"rebuild") resolves to the SAME ordered host
 * dispatch -- there is no captured super-graph, no node to toggle, on this
 * face. The tests below exercise that dispatch, the fired-history list,
 * the two-level RECURSION LOCK nesting, and the pinned rejection set.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "eagle/compose/GraphComposer.h"

#include <memory>
#include <stdexcept>
#include <vector>

namespace eagle_tests {
namespace GraphComposerCpuTest {

using eagle::compose::GraphComposer;

namespace {

/* The unguarded tracer: runHost increments unconditionally -- same shape
 * as test_ConditionalGroupCpu.cpp's own TracerNode. */
struct TracerNode {
    int* counter;
    void reserveScratch(eagle::ScratchArena&) { }
    void runHost(eagle::cpu::Graph&) { (*counter)++; }
};

} // namespace

/* ================================================================
 * 1. mode="sequenced": ordered dispatch, in registration order, over
 *    a plain callable AND a registered cpu::Graph member.
 * ================================================================ */
TEST(GraphComposerCpuTest, SequencedOrderedDispatch)
{
    std::vector<int> log;
    int graphCounter = 0;

    eagle::cpu::Graph g;
    g.addNative(TracerNode{ &graphCounter }, {});

    GraphComposer composer("sequenced");
    composer.registerCallable([&log]() { log.push_back(0); }, "callable");
    composer.registerGraph(std::move(g), "graph");
    composer.build();
    composer.launch(3);

    ASSERT_EQ(log.size(), 3u);
    for (int v : log)
        EXPECT_EQ(v, 0);
    EXPECT_EQ(graphCounter, 3);
}

/* ================================================================
 * 2. Two-level nesting: a nested composer's launch(n) is called ONCE
 *    (it owns its own internal replay count), never n times.
 * ================================================================ */
TEST(GraphComposerCpuTest, TwoLevelNestingDispatchesInOrder)
{
    std::vector<int> log;

    auto inner = std::make_shared<GraphComposer>("sequenced");
    inner->registerCallable([&log]() { log.push_back(1); }, "inner-step");
    inner->build();

    GraphComposer outer("sequenced");
    outer.registerNested(inner, "inner");
    outer.registerCallable([&log]() { log.push_back(2); }, "outer-step");
    outer.build();
    outer.launch(2);

    /* Registration order: inner (2 fires from its OWN n=2 replay) then
     * the outer's own callable (2 fires). */
    ASSERT_EQ(log.size(), 4u);
    EXPECT_EQ(log[0], 1);
    EXPECT_EQ(log[1], 1);
    EXPECT_EQ(log[2], 2);
    EXPECT_EQ(log[3], 2);
}

/* ================================================================
 * 3. Flags + fired-history parity across modes: "enabled" and
 *    "rebuild" degenerate to the SAME host masking as "sequenced" on
 *    this face -- no node-enable, no recapture concept exists
 *    here to differ by.
 * ================================================================ */
TEST(GraphComposerCpuTest, FlagsGateDispatchIdenticallyAcrossModes)
{
    for (const char* mode : { "sequenced", "enabled", "rebuild" }) {
        SCOPED_TRACE(std::string("mode=") + mode);
        int a = 0;
        int b = 0;
        GraphComposer composer(mode);
        composer.registerCallable([&a]() { a++; });
        composer.registerCallable([&b]() { b++; });
        composer.build();

        composer.launch(1);
        EXPECT_EQ(a, 1);
        EXPECT_EQ(b, 1);

        composer.setRouting({ true, false });
        composer.launch(1);
        EXPECT_EQ(a, 2);
        EXPECT_EQ(b, 1) << "member 1 disabled -- must not fire";

        ASSERT_EQ(composer.firedHistory().size(), 2u);
        EXPECT_EQ(composer.firedHistory()[0], (std::vector<eagle::idx_t>{ 0, 1 }));
        EXPECT_EQ(composer.firedHistory()[1], (std::vector<eagle::idx_t>{ 0 }));
    }
}

TEST(GraphComposerCpuTest, ResetFiredHistoryClears)
{
    GraphComposer composer("sequenced");
    composer.registerCallable([]() { });
    composer.build();
    composer.launch(1);
    ASSERT_EQ(composer.firedHistory().size(), 1u);
    composer.resetFiredHistory();
    EXPECT_TRUE(composer.firedHistory().empty());
}

/* ================================================================
 * 4. Rejections -- the pinned set, CPU face.
 * ================================================================ */
TEST(GraphComposerCpuTest, RejectsNestingUnderNonSequencedOuter)
{
    auto inner = std::make_shared<GraphComposer>("sequenced");
    inner->registerCallable([]() { });
    inner->build();

    GraphComposer flatOuter("enabled");
    EXPECT_THROW(flatOuter.registerNested(inner), std::invalid_argument);
}

TEST(GraphComposerCpuTest, RejectsNestingNonSequencedOrUnbuiltInner)
{
    GraphComposer outer("sequenced");

    auto flatInner = std::make_shared<GraphComposer>("enabled");
    EXPECT_THROW(outer.registerNested(flatInner), std::invalid_argument);

    auto unbuiltInner = std::make_shared<GraphComposer>("sequenced");
    EXPECT_THROW(outer.registerNested(unbuiltInner), std::invalid_argument);
}

TEST(GraphComposerCpuTest, RejectsDuplicateMember)
{
    GraphComposer outer("sequenced");
    auto inner = std::make_shared<GraphComposer>("sequenced");
    inner->registerCallable([]() { });
    inner->build();

    outer.registerNested(inner, "dup");
    EXPECT_THROW(outer.registerNested(inner, "dup2"), std::invalid_argument);
    EXPECT_THROW(outer.registerCallable([]() { }, "dup"), std::invalid_argument);
}

TEST(GraphComposerCpuTest, RejectsLaunchBeforeBuild)
{
    GraphComposer composer("sequenced");
    composer.registerCallable([]() { });
    EXPECT_THROW(composer.launch(), std::runtime_error);
}

TEST(GraphComposerCpuTest, RejectsRoutingLengthMismatch)
{
    GraphComposer composer("sequenced");
    composer.registerCallable([]() { });
    composer.registerCallable([]() { });
    composer.build();
    EXPECT_THROW(composer.setRouting({ true }), std::invalid_argument);
}

TEST(GraphComposerCpuTest, RejectsRegisterAfterBuildAndDoubleBuild)
{
    GraphComposer composer("sequenced");
    composer.registerCallable([]() { });
    composer.build();
    EXPECT_THROW(composer.registerCallable([]() { }), std::runtime_error);
    EXPECT_THROW(composer.build(), std::runtime_error);
}

TEST(GraphComposerCpuTest, RejectsBuildWithNoMembers)
{
    GraphComposer composer("sequenced");
    EXPECT_THROW(composer.build(), std::runtime_error);
}

TEST(GraphComposerCpuTest, RejectsGraphMemberAndPreLaunchUnderFlatModes)
{
    GraphComposer flat("enabled");
    eagle::cpu::Graph g;
    int counter = 0;
    g.addNative(TracerNode{ &counter }, {});
    EXPECT_THROW(flat.registerGraph(std::move(g)), std::invalid_argument)
        << "a built cpu::Graph member is sequenced-only";
    EXPECT_THROW(
        flat.registerCallable([]() { }, {}, []() { }), std::invalid_argument)
        << "pre_launch is sequenced-only";
}

TEST(GraphComposerCpuTest, RejectsUnknownMode)
{
    EXPECT_THROW(GraphComposer("bogus"), std::invalid_argument);
    EXPECT_THROW(GraphComposer("conditional"), std::invalid_argument)
        << "conditional mode is DEFERRED on the C++ face";
}

} // namespace GraphComposerCpuTest
} // namespace eagle_tests
