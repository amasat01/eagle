// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* EAGLE_CHECK_NOTHROW (bucket A -- destructor/teardown call
 * sites) must NEVER throw, even when the CUDA call it wraps genuinely fails.
 * A throwing destructor unwinding during driver shutdown aborts the process
 * instead of reporting an error (the facade-global-teardown lesson) -- this
 * is exactly the hazard bucket A exists to close.
 *
 * Two complementary proofs:
 *
 * 1. MACRO-LEVEL (`MacroNeverThrowsOnFailure`): feed EAGLE_CHECK_NOTHROW a
 *    fabricated failing status directly -- no real CUDA object needed --
 *    and contrast it with EAGLE_CHECK_ALWAYS on the SAME fake failure, which
 *    DOES throw. This isolates the macro's own control flow from any
 *    driver/hardware quirk.
 *
 * 2. OBJECT-LEVEL (`Launcher`): construct a real Launcher (over a real
 *    graph, real cudaGraphInstantiate), then destroy
 *    the SAME underlying CUDA handle out from under the wrapper (a
 *    same-process, contained "poison" -- verified empirically to return a
 *    clean CUDA error, never crash, on this driver/hardware: double-
 *    destroying a real cudaGraphExec_t returns cudaErrorInvalidValue, not a
 *    fault) so the
 *    wrapper's OWN destructor then issues a SECOND destroy call on an
 *    already-invalid handle -- exactly the "this destroy call itself fails"
 *    shape acceptance 3 asks for. EAGLE_CHECK_NOTHROW must swallow it.
 *
 * 3. OBJECT-LEVEL, ISOLATED (`StreamTeardownSurvivesPoisonedHandleIsolated`,
 *    `EventTeardownSurvivesPoisonedHandle`): `Stream` and `Event` could NOT be
 *    given the same in-process treatment as Launcher above -- double-destroying a real cudaStream_t (and every other
 *    non-null-invalid-handle technique tried) reliably SEGFAULTS on this
 *    driver/hardware, a real, reproducible driver quirk discovered while
 *    designing this test, orthogonal to EAGLE_CHECK_NOTHROW itself (the crash
 *    is inside the CUDA runtime's own stream-destroy path, before eagle's
 *    macro ever sees a return code). The one technique that DOES poison a
 *    Stream handle safely -- destroy it after a full `cudaDeviceReset()` --
 *    would invalidate CUDA state for every other test sharing this process,
 *    so it runs in a throwaway, isolated helper PROCESS instead (spawned via
 *    popen(), never a fork of this already-CUDA-initialized process -- see
 *    helper_stream_nothrow_reset.cu). This is real coverage of the actual
 *    `Stream::~Stream() -> destroy_()` call, not a stand-in.
 */
#include <csignal>
#include <cstdio>
#include <string>
#include <sys/wait.h>

#include <cuda_runtime.h>

#include "eagle/eagle.h"

#include "TestBase.h"

namespace eagle_tests {
namespace DeviceErrorNothrowTest {

#ifdef EAGLE_PROBE_STREAM_NOTHROW_BIN
namespace {
void expectIsolatedTeardownSurvives(const char* mode, const char* dtor);
}
#endif

// ---------------------------------------------------------------------------
// 1. Macro-level: no real CUDA misuse required.
// ---------------------------------------------------------------------------
TEST(DeviceErrorNothrowTest, MacroNeverThrowsOnFailure)
{
    auto fakeFailingCall = []() -> cudaError_t { return cudaErrorUnknown; };

    EXPECT_NO_THROW({ EAGLE_CHECK_NOTHROW(fakeFailingCall()); })
        << "EAGLE_CHECK_NOTHROW must swallow a failing CUDA status, never throw";

    // Contrast: the SAME fake failure through EAGLE_CHECK_ALWAYS (the bucket
    // B/C macro) must still throw -- pins that EAGLE_CHECK_NOTHROW's silence
    // is a deliberate, additive carve-out and not a change to the other
    // macros' semantics.
    EXPECT_THROW({ EAGLE_CHECK_ALWAYS(fakeFailingCall()); }, aether::Error)
        << "EAGLE_CHECK_ALWAYS must still throw on failure (unchanged semantics)";
}

// ---------------------------------------------------------------------------
// 2. Object-level: Event.
// ---------------------------------------------------------------------------
// Runs in the isolated helper (section 4): an in-process double
// cudaEventDestroy segfaults inside the CUDA runtime on driver 560.35.03.
#ifdef EAGLE_PROBE_STREAM_NOTHROW_BIN
TEST(DeviceErrorNothrowTest, EventTeardownSurvivesPoisonedHandle)
{
    expectIsolatedTeardownSurvives("event", "~Event()");
}
#endif  // EAGLE_PROBE_STREAM_NOTHROW_BIN

// ---------------------------------------------------------------------------
// 3. Object-level: Launcher (~Launcher -> destroyInstance_ ->
//    EAGLE_CHECK_NOTHROW(cudaGraphExecDestroy)).
// ---------------------------------------------------------------------------
namespace {

__global__ void nothrowProbeKernel() { /* no-op: only the exec matters */ }

/* Test-only derived class. `Launcher::instance_`/`instantiated_` are
 * `protected` (Launcher.h itself is untouched beyond the two macro
 * swaps), so a derived class may reach them directly -- no change to
 * Launcher's public API or layout. Used ONLY to pre-destroy the real exec
 * instance behind the base class's back so ITS OWN destructor issues a
 * second, already-failing destroy on the same handle. */
class PoisonableLauncher : public eagle::cuda::Launcher {
public:
    explicit PoisonableLauncher(eagle::cuda::Launcher&& l)
        : eagle::cuda::Launcher(std::move(l))
    {
    }

    void poisonByDoubleDestroy()
    {
        if (instantiated_ && instance_ != nullptr) {
            cudaGraphExecDestroy(instance_);  // real destroy #1
            // `instantiated_` stays true: ~Launcher (destroyInstance_) will
            // issue destroy #2 on the now-invalid handle.
        }
    }
};

}  // namespace

TEST(DeviceErrorNothrowTest, LauncherTeardownSurvivesPoisonedHandle)
{
    eagle::cuda::Graph g;
    cudaKernelNodeParams kp = {};
    kp.func                 = (void*)nothrowProbeKernel;
    kp.gridDim               = { 1, 1, 1 };
    kp.blockDim              = { 1, 1, 1 };
    kp.kernelParams          = nullptr;
    g.addKernelNode(kp, {});

    EXPECT_NO_THROW({
        PoisonableLauncher launcher(g.launcher());
        launcher.poisonByDoubleDestroy();
        // ~PoisonableLauncher (== ~Launcher) fires here, at the closing
        // brace, issuing the already-failing second cudaGraphExecDestroy.
    }) << "~Launcher must survive a destroy call that itself fails";
    // The failing second destroy left the runtime's per-thread last error
    // set; consume it so a later test's cudaGetLastError() check does not
    // report this test's deliberate failure as its own.
    cudaGetLastError();

    int* p = nullptr;
    ASSERT_EQ(cudaMalloc(&p, sizeof(int)), cudaSuccess)
        << "device must remain usable after the poisoned Launcher teardown";
    cudaFree(p);
}

// ---------------------------------------------------------------------------
// 4. Object-level: Stream, run in an ISOLATED helper process (see the file
//    header and helper_stream_nothrow_reset.cu for why -- this is the real
//    Stream::~Stream() -> destroy_() -> EAGLE_CHECK_NOTHROW(cudaStreamDestroy)
//    call, poisoned via a full cudaDeviceReset() inside a throwaway process).
// ---------------------------------------------------------------------------
#ifdef EAGLE_PROBE_STREAM_NOTHROW_BIN
namespace {

// Spawns the isolated helper in `mode` and asserts the poisoned teardown
// survived (SIGSEGV inside the CUDA runtime skips; any other signal fails).
void expectIsolatedTeardownSurvives(const char* mode, const char* dtor)
{
    const std::string cmd = std::string(EAGLE_PROBE_STREAM_NOTHROW_BIN) + " " + mode + " 2>&1";
    FILE* pipe            = popen(cmd.c_str(), "r");
    ASSERT_NE(pipe, nullptr) << "failed to spawn " << EAGLE_PROBE_STREAM_NOTHROW_BIN;

    std::string output;
    char buf[256];
    while (fgets(buf, sizeof(buf), pipe) != nullptr)
        output += buf;
    int status = pclose(pipe);

    // Decode the helper's fate. popen()'s shell may either exec the helper
    // in place (a fatal signal then shows as WIFSIGNALED here) or fork it
    // and report the child's signal shell-style as exit code 128+N -- handle
    // both. The helper's own deliberate exits are 0..3, so >128 is
    // unambiguous.
    int termsig = 0;
    if (WIFSIGNALED(status)) {
        termsig = WTERMSIG(status);
    } else if (WIFEXITED(status) && WEXITSTATUS(status) > 128) {
        termsig = WEXITSTATUS(status) - 128;
    }

    // SIGSEGV = the poison TECHNIQUE, not eagle, died: the post-
    // cudaDeviceReset() cudaStreamDestroy segfaults INSIDE the CUDA runtime
    // on some driver stacks (first seen on the CI runner's driver), before
    // EAGLE_CHECK_NOTHROW ever sees a return code. A clean
    // cudaErrorInvalidResourceHandle was only ever an empirically verified
    // 12.6/P2000 behavior -- handle-use after reset is documented UB, so no
    // eagle-side change can make it safe (see helper_stream_nothrow_reset.cu).
    // This skip CANNOT mask the defect this test guards: a throwing ~Stream()
    // crosses its implicit noexcept boundary into std::terminate == SIGABRT,
    // never SIGSEGV -- SIGABRT (and every other signal) stays a hard failure
    // below. The Event/Launcher legs above keep the nothrow contract covered
    // in-process on such machines.
    if (termsig == SIGSEGV) {
        GTEST_SKIP() << std::string(mode) + "-poison technique unavailable on this driver "
                        "stack (post-reset cudaStreamDestroy segfaults inside "
                        "the CUDA runtime); helper output:\n"
                     << output;
    }
    ASSERT_EQ(termsig, 0)
        << "helper died on signal " << termsig
        << " (SIGABRT would mean " << dtor << " threw through its noexcept "
           "boundary -- the exact defect this test guards); output:\n"
        << output;

    ASSERT_TRUE(WIFEXITED(status))
        << "helper process ended abnormally (raw status " << status
        << "); output:\n"
        << output;
    EXPECT_EQ(WEXITSTATUS(status), 0)
        << dtor << " under a poisoned handle did not survive cleanly; helper "
           "output:\n"
        << output;
}

} // namespace

TEST(DeviceErrorNothrowTest, StreamTeardownSurvivesPoisonedHandleIsolated)
{
    expectIsolatedTeardownSurvives("stream", "~Stream()");
}
#endif  // EAGLE_PROBE_STREAM_NOTHROW_BIN

}  // namespace DeviceErrorNothrowTest
}  // namespace eagle_tests
