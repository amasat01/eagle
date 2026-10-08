// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Registration-consistency unit tests for eagle::util::Observer move
 * assignment and re-observe. The observable's list and the observer's own
 * `ref_` must never disagree: a stale list entry is written through on the
 * observable's next move (`finalizeMove` -> `notifyObservers`, a raw pointer
 * store), and the observer's destructor -- keyed on its own null `ref_` --
 * never removes it. Found live in practice: `releaseEphCache`-style
 * release-by-assignment left two dangling observers on a session's ephemeris
 * unit per host run, detonating as an invalid write on the unit's next move.
 *
 * Deliberately NOT "TestBase.h": that header pulls in the full eagle.h
 * umbrella, whose __device__/__host__-qualified bodies only nvcc parses --
 * and in CUDA mode this `.cpp` is compiled by CXX (g++), not nvcc (same trap
 * test_ConformanceCorpus.cpp documents). Observer.h/Observable.h reach
 * eagle/typedefs.h -> <aether/aether.h>, so this TU is explicitly compiled
 * with -DEAGLE_CPU_ONLY (CMakeLists.txt COMPILE_DEFINITIONS on this one
 * source), independent of the ambient EAGLE_CPP_MODE, exactly like
 * test_LaunchTraits.cpp. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "eagle/util/Observable.h"

namespace eagle_tests {
namespace ObserverRegistrationTests {

/** @brief Minimal observable: the machinery under test, nothing else. The
 *  move constructor mirrors the real observables' idiom (aether-backed arrays
 *  call `finalizeMove` from theirs) so the notify-on-move path is exercised. */
struct Probe : eagle::util::Observable<Probe> {
    Probe() = default;
    Probe(Probe&& other) { finalizeMove(std::move(other)); }
};
using ObsT = Probe::ObserverT;

TEST(ObserverRegistrationTest, ReleaseByMoveAssignDeregisters)
{
    Probe a;
    ObsT  obs(a);
    ASSERT_EQ(a.observers().size(), 1u);
    obs = ObsT();
    EXPECT_TRUE(a.observers().empty())
        << "release-by-assignment left a dangling entry on the observable";
}

TEST(ObserverRegistrationTest, MoveAssignLeavesOldReferent)
{
    Probe a;
    Probe b;
    ObsT  source(a);
    ObsT  target(b);
    target = std::move(source);
    EXPECT_TRUE(b.observers().empty())
        << "the target's old referent kept a pointer to the reassigned "
           "observer";
    ASSERT_EQ(a.observers().size(), 1u);
    EXPECT_EQ(a.observers().front(), &target);
}

TEST(ObserverRegistrationTest, SameReferentMoveAssignKeepsOneEntry)
{
    Probe a;
    ObsT  source(a);
    ObsT  target(a);
    ASSERT_EQ(a.observers().size(), 2u);
    target = std::move(source);
    ASSERT_EQ(a.observers().size(), 1u);
    EXPECT_EQ(a.observers().front(), &target);
}

TEST(ObserverRegistrationTest, ReobserveLeavesOldReferent)
{
    Probe a;
    Probe b;
    ObsT  obs(a);
    obs.observe(b);
    EXPECT_TRUE(a.observers().empty())
        << "re-observing abandoned the old referent's list entry";
    ASSERT_EQ(b.observers().size(), 1u);
    EXPECT_EQ(b.observers().front(), &obs);
}

/** @brief The detonation pattern end to end: a released observer must be
 *  untouched by the observable's later move. Under the defect this test
 *  writes through freed/stale storage (caught by the list assertions above
 *  in-process, by valgrind at system level). */
TEST(ObserverRegistrationTest, ObservableMoveDoesNotTouchReleasedObserver)
{
    Probe a;
    ObsT  obs(a);
    obs = ObsT();
    Probe b(std::move(a));
    EXPECT_TRUE(b.observers().empty());
}

} // namespace ObserverRegistrationTests
} // namespace eagle_tests
