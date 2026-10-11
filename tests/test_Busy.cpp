// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* The busy guard: header-only, CUDA-free, compiled by CXX in both build
 * modes. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "eagle/util/Busy.h"

#include <atomic>
#include <string>
#include <thread>

namespace eagle_tests {
namespace BusyTest {

using eagle::util::Busy;
using eagle::util::BusyGuard;

TEST(BusyTest, SecondEntrantIsToldToWaitOrUseItsOwnObject)
{
    BusyGuard g("eagle.Graph");
    g.enter("launch");
    try {
        g.enter("build");
        FAIL() << "a second entrant must be refused";
    } catch (const Busy& e) {
        EXPECT_STREQ(e.what(),
            "eagle.Graph.build: another call is still running on this Graph; "
            "give each thread its own Graph, or wait for it");
    }
    EXPECT_EQ(g.current(), "launch");
    g.leave();
    EXPECT_EQ(g.current(), "");
}

TEST(BusyTest, LeavingAllowsReentryAndScopeLeavesOnDestruction)
{
    BusyGuard g("x.Y");
    {
        auto s = g.scope("a");
        EXPECT_FALSE(g.tryEnter("b"));
    }
    EXPECT_TRUE(g.tryEnter("b"));
    g.leave();
    {
        auto s = g.scope("c");
        auto moved = std::move(s);
        EXPECT_FALSE(g.tryEnter("d"));
    }
    EXPECT_TRUE(g.tryEnter("d"));
}

TEST(BusyTest, ForcedOverlapAcrossThreadsExactlyOneWins)
{
    BusyGuard g("o");
    std::atomic<int> inside{ 0 }, refused{ 0 }, ran{ 0 };
    std::atomic<bool> release{ false };
    auto work = [&] {
        try {
            auto s = g.scope("run");
            ++inside;
            while (!release.load()) std::this_thread::yield();
            ++ran;
        } catch (const Busy&) {
            ++refused;
        }
    };
    std::thread a(work);
    while (inside.load() == 0) std::this_thread::yield();
    std::thread b(work), c(work);
    b.join();
    c.join();
    EXPECT_EQ(refused.load(), 2);
    release = true;
    a.join();
    EXPECT_EQ(ran.load(), 1);
    EXPECT_TRUE(g.tryEnter("again"));
}

}  // namespace BusyTest
}  // namespace eagle_tests
