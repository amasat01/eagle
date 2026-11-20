// SPDX-License-Identifier: Apache-2.0

// The device functions of fixture_reverse.c's step x <- a x + b, t <- t + 1, its one-step taping twin and its step vjp (built to a fatbin for
// sm_61 and embedded in libfixture_reverse.so): the same arithmetic as its host entries, one thread per sample over the partition triple a
// DEVICE launch appends.
#include "plugin/gref_layout.h"

using eagle::plugin::GRefMirror;
using eagle::plugin::ScalarHandle;

namespace {

__device__ double* real(const ScalarHandle& h) { return static_cast<double*>(h.data); }
__device__ unsigned char* flag(const ScalarHandle& h) { return static_cast<unsigned char*>(h.data); }

}  // namespace

#define SAMPLE(i)                                                                                              \
    const long long flat_ = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;                    \
    if (flat_ >= count) return;                                                                                \
    const long long i = base + flat_;                                                                          \
    (void)nSamples

#define TRIPLE long long base, long long count, long long nSamples

extern "C" {

// slots: x, t, terminated, finished_count, T, a, b
__global__ void lin(ScalarHandle x, ScalarHandle t, ScalarHandle done, ScalarHandle finished, ScalarHandle T, double a, double b, TRIPLE) {
    SAMPLE(i);
    if (flag(done)[i] != 0) return;
    real(x)[i] = a * real(x)[i] + b;
    real(t)[i] += 1.0;
    if (real(t)[i] >= real(T)[i]) {
        flag(done)[i] = 1;
        atomicAdd(static_cast<unsigned*>(finished.data), 1u);
    }
}

// slots: applied_steps, x, t, tape (window 1: row 0 holds t, row 1 holds x), T, finished_count, terminated, a, b, tape_on
__global__ void lin_tape1(ScalarHandle steps, ScalarHandle x, ScalarHandle t, GRefMirror tape, ScalarHandle T, ScalarHandle finished,
                          ScalarHandle done, double a, double b, long long on, TRIPLE) {
    SAMPLE(i);
    if (flag(done)[i] != 0) return;
    if (on != 0) {
        tape.data_[0 * static_cast<long long>(tape.compStride_) + i] = real(t)[i];
        tape.data_[1 * static_cast<long long>(tape.compStride_) + i] = real(x)[i];
    }
    real(x)[i] = a * real(x)[i] + b;
    real(t)[i] += 1.0;
    static_cast<long long*>(steps.data)[i] += 1;
    if (real(t)[i] >= real(T)[i]) {
        flag(done)[i] = 1;
        atomicAdd(static_cast<unsigned*>(finished.data), 1u);
    }
}

// slots: bar_a, bar_b, bar_x, bar_t, bar_x_next, bar_t_next, x (taped), a, b
__global__ void lin_vjp(ScalarHandle ga, ScalarHandle gb, ScalarHandle gx, ScalarHandle gt, ScalarHandle gxn, ScalarHandle gtn, ScalarHandle x,
                        double a, double b, TRIPLE) {
    SAMPLE(i);
    (void)b;
    real(ga)[i] = real(x)[i] * real(gxn)[i];
    real(gb)[i] = real(gxn)[i];
    real(gx)[i] = a * real(gxn)[i];
    real(gt)[i] = real(gtn)[i];
}

}
