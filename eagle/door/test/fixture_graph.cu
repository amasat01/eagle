// SPDX-License-Identifier: Apache-2.0

// The device functions of fixture_graph.c (built to a fatbin for sm_61 and embedded in libfixture_graph.so): the same closed-form
// units and vjps as its host entries, one thread per sample over the partition triple a DEVICE launch appends.
#include "plugin/gref_layout.h"

using eagle::plugin::GRefMirror;
using eagle::plugin::ScalarHandle;

namespace {

__device__ double* at(const ScalarHandle& h) { return static_cast<double*>(h.data); }
__device__ double& cell(const GRefMirror& m, int c, long long i) { return m.data_[c * static_cast<long long>(m.compStride_) + i]; }

}  // namespace

#define SAMPLE(i)                                                                                              \
    const long long flat_ = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;                    \
    if (flat_ >= count) return;                                                                                \
    const long long i = base + flat_;                                                                          \
    (void)nSamples

#define TRIPLE long long base, long long count, long long nSamples

extern "C" {

__global__ void scale(ScalarHandle x, ScalarHandle w, ScalarHandle y, TRIPLE) {
    SAMPLE(i);
    at(y)[i] = at(w)[0] * at(x)[i];
}
__global__ void scale_vjp(ScalarHandle x, ScalarHandle w, ScalarHandle bar_y, ScalarHandle bar_x, ScalarHandle bar_w, TRIPLE) {
    SAMPLE(i);
    at(bar_x)[i] = at(w)[0] * at(bar_y)[i];
    at(bar_w)[i] = at(x)[i] * at(bar_y)[i];
}
// slots: x, w, bar_y, bar_x, bar_w (the running sum), bar_w_c (its compensation). Thread 0 sums the partition's samples in order, as the host entry does.
__global__ void scale_vjp_comp(ScalarHandle x, ScalarHandle w, ScalarHandle bar_y, ScalarHandle bar_x, ScalarHandle bar_w, ScalarHandle bar_w_c, TRIPLE) {
    const long long flat = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat < count) at(bar_x)[base + flat] = at(w)[0] * at(bar_y)[base + flat];
    if (flat != 0) return;
    double sum = at(bar_w)[0], c = at(bar_w_c)[0];
    for (long long i = base; i < base + count; ++i) {
        const double term = at(x)[i] * at(bar_y)[i] - c, t = sum + term;
        c = (t - sum) - term;
        sum = t;
    }
    at(bar_w)[0] = sum;
    at(bar_w_c)[0] = c;
    (void)nSamples;
}
// the fold after the launch: the sum minus its compensation, over the elements of the target
__global__ void scale_vjp_comp_fold(ScalarHandle x, ScalarHandle w, ScalarHandle bar_y, ScalarHandle bar_x, ScalarHandle bar_w, ScalarHandle bar_w_c, TRIPLE) {
    SAMPLE(i);
    (void)x, (void)w, (void)bar_y, (void)bar_x;
    at(bar_w)[i] -= at(bar_w_c)[i];
}
// slots: x, c (uint32 cell: a WHILE's counter, 0-based), y: y = (c + 1) x
__global__ void cscale(ScalarHandle x, ScalarHandle c, ScalarHandle y, TRIPLE) {
    SAMPLE(i);
    at(y)[i] = static_cast<double>(*static_cast<unsigned*>(c.data) + 1u) * at(x)[i];
}
// slots: c (uint32 cell), bar_y, bar_x
__global__ void cscale_vjp(ScalarHandle c, ScalarHandle bar_y, ScalarHandle bar_x, TRIPLE) {
    SAMPLE(i);
    at(bar_x)[i] = static_cast<double>(*static_cast<unsigned*>(c.data) + 1u) * at(bar_y)[i];
}
// slots: x, n (int32), y
__global__ void iscale(ScalarHandle x, ScalarHandle n, ScalarHandle y, TRIPLE) {
    SAMPLE(i);
    at(y)[i] = static_cast<double>(static_cast<int*>(n.data)[i]) * at(x)[i];
}
// slots: n (int32), bar_y, bar_x
__global__ void iscale_vjp(ScalarHandle n, ScalarHandle bar_y, ScalarHandle bar_x, TRIPLE) {
    SAMPLE(i);
    at(bar_x)[i] = static_cast<double>(static_cast<int*>(n.data)[i]) * at(bar_y)[i];
}
__global__ void square(ScalarHandle a, ScalarHandle o, TRIPLE) {
    SAMPLE(i);
    at(o)[i] = at(a)[i] * at(a)[i];
}
__global__ void square_vjp(ScalarHandle a, ScalarHandle bar_o, ScalarHandle bar_a, TRIPLE) {
    SAMPLE(i);
    at(bar_a)[i] = 2.0 * at(a)[i] * at(bar_o)[i];
}
__global__ void mul2(ScalarHandle a, ScalarHandle b, ScalarHandle o, TRIPLE) {
    SAMPLE(i);
    at(o)[i] = at(a)[i] * at(b)[i];
}
__global__ void mul2_vjp(ScalarHandle a, ScalarHandle b, ScalarHandle bar_o, ScalarHandle bar_a, ScalarHandle bar_b, TRIPLE) {
    SAMPLE(i);
    at(bar_a)[i] = at(b)[i] * at(bar_o)[i];
    at(bar_b)[i] = at(a)[i] * at(bar_o)[i];
}
__global__ void fan(ScalarHandle x, GRefMirror v, TRIPLE) {
    SAMPLE(i);
    for (int c = 0; c < 3; ++c) cell(v, c, i) = (c + 1) * at(x)[i];
}
__global__ void fan_vjp(GRefMirror bar_v, ScalarHandle bar_x, TRIPLE) {
    SAMPLE(i);
    double g = 0.0;
    for (int c = 0; c < 3; ++c) g += (c + 1) * cell(bar_v, c, i);
    at(bar_x)[i] = g;
}
__global__ void norm2(GRefMirror v, ScalarHandle o, TRIPLE) {
    SAMPLE(i);
    double t = 0.0;
    for (int c = 0; c < 3; ++c) t += cell(v, c, i) * cell(v, c, i);
    at(o)[i] = t;
}
__global__ void norm2_vjp(GRefMirror v, ScalarHandle bar_o, GRefMirror bar_v, TRIPLE) {
    SAMPLE(i);
    for (int c = 0; c < 3; ++c) cell(bar_v, c, i) = 2.0 * cell(v, c, i) * at(bar_o)[i];
}

__global__ void tick(ScalarHandle c, TRIPLE) {
    SAMPLE(i);
    static_cast<int*>(c.data)[i] -= 1;
}

}  // extern "C"
