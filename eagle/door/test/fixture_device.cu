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
