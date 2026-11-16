// SPDX-License-Identifier: Apache-2.0

// The device fixture's kernel (built to a fatbin for sm_61 and embedded in libfixture_device.so):
// y = fma(a, x, v[0]) + n * v[1], every operation rounded on its own (the test's plain-C oracle
// computes the same), over the partition triple a DEVICE launch appends.
#include "plugin/gref_layout.h"

using eagle::plugin::GRefMirror;
using eagle::plugin::ScalarHandle;

extern "C" __global__ void axpy(ScalarHandle y, ScalarHandle x, GRefMirror v, double a, unsigned n,
                                long long base, long long count, long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    (void)nSamples;
    const double* xs = static_cast<const double*>(x.data);
    const double* vs = v.data_;
    static_cast<double*>(y.data)[i] =
        __dadd_rn(__fma_rn(a, xs[i], vs[i]), __dmul_rn(static_cast<double>(n), vs[v.compStride_ + i]));
}

// dcells: the rows of 4 samples hold the min (tot) and the max (mx) of x (the row's first sample resets them).
extern "C" __global__ void dcells(ScalarHandle tot, ScalarHandle mx, ScalarHandle x, long long base, long long count,
                                  long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    (void)nSamples;
    if (i % 4 != 0) return;
    double* t = static_cast<double*>(tot.data);
    double* m = static_cast<double*>(mx.data);
    const double* xs = static_cast<const double*>(x.data);
    t[i / 4] = 1e300;
    m[i / 4] = -1e300;
    for (long long j = i; j < i + 4 && j < nSamples; ++j) {
        if (xs[j] < t[i / 4]) t[i / 4] = xs[j];
        if (xs[j] > m[i / 4]) m[i / 4] = xs[j];
    }
}

// dkeyed: acc[i / 4] += x[i] and comp[i / 4] += 0.5 for the 4 samples of the key the lane leads; the fold:
// acc += comp, comp = 0.125 over the target's slots.
extern "C" __global__ void dkeyed(ScalarHandle acc, ScalarHandle comp, ScalarHandle x, long long base, long long count,
                                  long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    if (i % 4 != 0) return;
    const double* xs = static_cast<const double*>(x.data);
    for (long long j = i; j < i + 4 && j < nSamples; ++j) {
        static_cast<double*>(acc.data)[i / 4] += xs[j];
        static_cast<double*>(comp.data)[i / 4] += 0.5;
    }
}

extern "C" __global__ void dkeyed__fold_acc(ScalarHandle acc, ScalarHandle comp, ScalarHandle x, long long base,
                                            long long count, long long nSamples) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    const long long i = base + flat;
    (void)x;
    (void)nSamples;
    static_cast<double*>(acc.data)[i] += static_cast<double*>(comp.data)[i];
    static_cast<double*>(comp.data)[i] = 0.125;
}
