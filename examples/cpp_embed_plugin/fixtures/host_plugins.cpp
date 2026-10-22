// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// CPU twins of the three CUDA fixtures (mattrace.cu, toy_energy.cu, toy_energy_vjp.cu):
// one shared object exporting one `<id>_host` entry per plugin. Like the CUDA
// fixtures they depend only on the binary ABI (plugin/gref_abi.h), not on eagle itself.
// Each entry gets the same params[] the CUDA kernel would get as arguments, in the
// order of the sidecar's arg_spec, and runs its own OpenMP loop.
#include "plugin/gref_abi.h"

#include <cmath>
#include <cstdint>

using namespace eagle::plugin;

namespace {
double* handle_data(void* const* params, int k) {
    return static_cast<double*>(static_cast<const ScalarHandle*>(params[k])->data);
}
const GRefMirror& gref(void* const* params, int k) {
    return *static_cast<const GRefMirror*>(params[k]);
}
}  // namespace

// arg_spec: [mutable out, mat_in M, nsamples]
extern "C" void mattrace_host(void* const* params, std::int32_t n) {
    double* out = handle_data(params, 0);
    const double* M = gref(params, 1).data_;
    const std::uint64_t so = gref(params, 1).compStride_;
#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i) out[i] = M[0 * so + i] + M[4 * so + i] + M[8 * so + i];
}

// arg_spec: [vec_in x, mutable e, uniform p, nsamples]
extern "C" void toy_energy_host(void* const* params, std::int32_t n) {
    const double* x = gref(params, 0).data_;
    const std::uint64_t so = gref(params, 0).compStride_;
    double* e = handle_data(params, 1);
    const double p = *static_cast<const double*>(params[2]);
#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i) {
        const double r2 = x[0 * so + i] * x[0 * so + i] + x[1 * so + i] * x[1 * so + i] +
                          x[2 * so + i] * x[2 * so + i];
        e[i] = p * std::exp(-0.5 * r2);
    }
}

// arg_spec: [vec_in x, per_sample e_bar, uniform p, mutable x_bar, mutable p_bar, nsamples]
extern "C" void toy_energy_vjp_host(void* const* params, std::int32_t n) {
    const double* x = gref(params, 0).data_;
    const std::uint64_t so = gref(params, 0).compStride_;
    const double* e_bar = handle_data(params, 1);
    const double p = *static_cast<const double*>(params[2]);
    double* xb = gref(params, 3).data_;
    const std::uint64_t xbso = gref(params, 3).compStride_;
    double* p_bar = handle_data(params, 4);
#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i) {
        const double x0 = x[0 * so + i], x1 = x[1 * so + i], x2 = x[2 * so + i];
        const double g = std::exp(-0.5 * (x0 * x0 + x1 * x1 + x2 * x2));
        const double eb = e_bar[i];
        xb[0 * xbso + i] = eb * (-x0) * p * g;
        xb[1 * xbso + i] = eb * (-x1) * p * g;
        xb[2 * xbso + i] = eb * (-x2) * p * g;
        p_bar[i] = eb * g;
    }
}
