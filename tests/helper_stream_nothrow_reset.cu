// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Standalone helper process for the Stream teardown-under-poison proof.
 *
 * NOT a gtest binary and NOT part of `eagle_tests` -- spawned via popen() from
 * `DeviceErrorNothrowTest.StreamTeardownSurvivesPoisonedHandleIsolated`
 * (test_DeviceErrorNothrow.cu) as a fresh, isolated process. Never a fork() of
 * the already-CUDA-initialized `eagle_tests` process -- popen()'s fork+immediate
 * exec discards the parent's CUDA state before this program ever runs, which is
 * what makes this safe (a raw in-process fork after CUDA init is unsupported).
 *
 * Why this needs its own process at all: `eagle::cuda::Stream::~Stream()` ->
 * `destroy_()` only issues its `EAGLE_CHECK_NOTHROW(cudaStreamDestroy(cuda_))`
 * call when `cuda_ != nullptr` -- so exercising that line for real requires a
 * genuinely non-null, invalid handle. Every technique tried that produces a
 * non-null invalid `cudaStream_t` on this driver/hardware (CUDA 12.6, P2000)
 * SEGFAULTS: a fabricated handle value, and double-destroying a real one
 * (`cudaStreamDestroy` twice on the same handle) both crash inside the CUDA
 * runtime itself, before eagle's macro ever sees a return code (verified with
 * standalone probes; GraphExec does NOT share this failure mode -- its
 * double-destroy returns a clean error, see
 * LauncherTeardownSurvivesPoisonedHandle in test_DeviceErrorNothrow.cu). The
 * one technique that IS safe: invalidate the handle via a full
 * `cudaDeviceReset()` (verified: the post-reset destroy then returns
 * `cudaErrorInvalidResourceHandle`, 400, cleanly). A device-wide reset would
 * corrupt every other test sharing eagle_tests' single process, so it is
 * confined here, in a throwaway process that does nothing else.
 *
 * Addendum: the reset technique's clean 400 is ITSELF a
 * driver/hardware-specific behavior -- on the CI runner's newer stack the
 * post-reset destroy segfaults inside the CUDA runtime just like the other
 * techniques (handle-use after reset is documented UB). The spawning test
 * therefore treats SIGSEGV as "technique unavailable here" and skips; a
 * SIGABRT (throwing ~Stream, the guarded defect) stays a hard failure. */
#include <cstdio>
#include <cstring>
#include <exception>

#include <cuda_runtime.h>

#include "eagle/cuda/stream/Event.h"
#include "eagle/cuda/stream/Stream.h"

// `event` as argv[1] poisons an Event instead of a Stream: its in-process
// double-destroy segfaults inside the CUDA runtime on driver 560.35.03.
int main(int argc, char** argv)
{
    const bool event = argc > 1 && std::strcmp(argv[1], "event") == 0;
    try {
        eagle::cuda::Stream* s = event ? nullptr : new eagle::cuda::Stream();
        eagle::cuda::Event* e  = event ? new eagle::cuda::Event() : nullptr;
        cudaError_t rrc        = cudaDeviceReset();
        if (rrc != cudaSuccess) {
            std::fprintf(
                stderr, "PROBE SETUP FAILED: cudaDeviceReset -> %d (%s)\n",
                (int)rrc, cudaGetErrorString(rrc));
            return 3;
        }
        // ~Stream() -> destroy_(): `cuda_` is still the same non-null handle
        // it always was; cudaDeviceReset() only invalidated what it points to.
        // The real EAGLE_CHECK_NOTHROW(cudaStreamDestroy(cuda_)) call now
        // genuinely fails -- it must be swallowed, not thrown.
        delete s;
        delete e;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "PROBE FAIL: teardown threw: %s\n", e.what());
        return 1;
    } catch (...) {
        std::fprintf(stderr, "PROBE FAIL: teardown threw a non-std exception\n");
        return 1;
    }

    // Recovery check: the device must still be usable afterward (a fresh
    // primary context is lazily recreated on first use post-reset).
    int* p          = nullptr;
    cudaError_t rc = cudaMalloc(&p, sizeof(int));
    if (rc != cudaSuccess) {
        std::fprintf(stderr,
            "PROBE FAIL: device unusable after poisoned teardown: %s\n",
            cudaGetErrorString(rc));
        return 2;
    }
    cudaFree(p);

    std::printf("PROBE OK\n");
    return 0;
}
