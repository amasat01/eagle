// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Acceptance test: release-mode CUDA-status discard
 * is a defect, not an acceptable hot-path trade. eagle's plain, release-
 * mode-silent `EAGLE_CHECK` has been deleted outright; every product call
 * site it used to guard (host-API calls only -- syncs, async memcpys, event
 * record/wait, cudaLaunchHostFunc, one cudaMalloc) now goes through the
 * unconditional `EAGLE_CHECK_ALWAYS`.
 *
 * This exercises one of those formerly-EAGLE_CHECK sites --
 * `ScratchArena::alloc_()`'s `cudaMalloc` (ScratchArena.h) -- through
 * `ScratchArena`'s PUBLIC API only (`beginNode`/`reserve`/`endNode`/
 * `commit`), never touching a raw handle. The failure trigger is
 * deterministic rather than an empirical hardware/driver quirk: a
 * `reserve()` request far beyond any real device's memory guarantees
 * `cudaErrorMemoryAllocation`, unlike the handle-poisoning techniques used
 * elsewhere in this suite (test_DeviceErrorNothrow.cu) for teardown-only
 * proofs.
 *
 * Two checks, mirroring existing idioms in this suite:
 *  1. Type (test_DeviceErrorNothrow.cu's EXPECT_THROW/EXPECT_NO_THROW
 *     contrast idiom): `commit()` must now THROW the typed
 *     `aether::Error` -- pre-conversion, the release-mode-silent
 *     `EAGLE_CHECK` would have discarded the status and left the arena
 *     silently un-backed.
 *  2. Message identity (ConditionalGroupTest.MultiLauncherFanOut's
 *     thrownWhat idiom): the message must anchor on the STABLE
 *     CUDA error NAME (`cudaErrorMemoryAllocation`), not only the
 *     English description, which carries no stability guarantee.
 */
#include <string>

#include "eagle/eagle.h"

#include "TestBase.h"

#ifndef EAGLE_CPU_ONLY

namespace eagle_tests {
namespace DeviceErrorReleaseGateTest {

TEST(DeviceErrorReleaseGate, FormerEagleCheckSiteThrowsTyped)
{
    // 1 PiB: no real device has this much memory -- a deterministic
    // cudaErrorMemoryAllocation, not a driver/hardware-dependent guess.
    const std::size_t hugeRequest = static_cast<std::size_t>(1) << 50;

    eagle::ScratchArena arena;
    arena.beginNode(0, {});
    arena.reserve(hugeRequest);
    arena.endNode();

    // 1. Type: must throw the typed aether::Error, not silently
    //    swallow the failing cudaMalloc.
    EXPECT_THROW(arena.commit(), aether::Error)
        << "ScratchArena::commit() must throw aether::Error on a "
           "failing cudaMalloc -- pre-conversion, this site used the "
           "release-mode-silent EAGLE_CHECK and would have discarded the "
           "status instead.";

    // 2. Message identity: `committed_` never got set on the
    //    first (thrown) attempt, so retrying `commit()` deterministically
    //    re-fails the same way and lets us inspect the message.
    bool threw = false;
    std::string thrownWhat;
    try {
        arena.commit();
    } catch (const aether::Error& e) {
        threw      = true;
        thrownWhat = e.what();
    }
    ASSERT_TRUE(threw)
        << "expected aether::Error from a retried commit()";
    EXPECT_NE(thrownWhat.find("cudaErrorMemoryAllocation"), std::string::npos)
        << "expected the stable cudaErrorMemoryAllocation NAME in the "
           "message, not just the English description; got: "
        << thrownWhat;

    // The two failed cudaMalloc calls left the runtime's per-thread last
    // error set; consume it so a later test's cudaGetLastError() check does
    // not report this test's deliberate failure as its own.
    EXPECT_EQ(cudaGetLastError(), cudaErrorMemoryAllocation);
}

} // namespace DeviceErrorReleaseGateTest
} // namespace eagle_tests

#endif // EAGLE_CPU_ONLY
