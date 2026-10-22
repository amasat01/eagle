// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The float32 twin of toy_energy_host (host_plugins.cpp): every plane and the by-value
// uniform are 4-byte floats, as the sidecar's scalar_type says.
#include "plugin/gref_abi.h"

#include <cmath>
#include <cstdint>

using namespace eagle::plugin;

// arg_spec: [vec_in x, mutable e, uniform p, nsamples]
extern "C" void toy_energy32_host(void* const* params, std::int32_t n) {
    const GRefMirror& X = *static_cast<const GRefMirror*>(params[0]);
    const float* x = reinterpret_cast<const float*>(X.data_);
    const std::uint64_t so = X.compStride_;
    float* e = static_cast<float*>(static_cast<const ScalarHandle*>(params[1])->data);
    const float p = *static_cast<const float*>(params[2]);
#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i) {
        const float r2 = x[0 * so + i] * x[0 * so + i] + x[1 * so + i] * x[1 * so + i] +
                         x[2 * so + i] * x[2 * so + i];
        e[i] = p * std::exp(-0.5f * r2);
    }
}
