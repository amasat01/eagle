// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// A hand-written CPU plugin fixture for test_HostPlugin.cpp — the `.so` a
// deployed plugin would emit for a matrix-input pure kernel ``out[i] = trace(M[i])`` (M a
// 3x3 matrix input). Like host_plugin_addvec.cpp it depends ONLY on the binary ABI
// (gref_abi.h PODs): a matrix binds through the SAME 40-byte GRefMirror as a vector
// (a matrix GRef is a width-``R*C`` vector GRef), laid out flat ``(R*C, N)`` SoA with
// ``dim = r*C + c`` (sample-fastest), so component ``(r, c)`` of sample ``i`` lives at
// ``data_[compStride_ * (r*C + c) + i]``. The trace reads flat dims 0, 4, 8. No
// eagle, no CUDA — the plugin owns its own OpenMP loop.
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

// arg_spec order (must match the sidecar the host registers): a scalar `mutable`
// output handle, a `mat_in` matrix GRef mirror, then `nsamples`.
extern "C" void mattrace_host(void* const* params, std::int32_t n)
{
    double* out = static_cast<double*>(
        static_cast<const ScalarHandle*>(params[0])->data);
    const GRefMirror* M    = static_cast<const GRefMirror*>(params[1]);
    const double* data     = M->data_;
    const std::uint64_t so = M->compStride_;  // SoA plane stride == N

#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i)
        out[i] = data[0 * so + i]   // (0,0)
            + data[4 * so + i]      // (1,1)
            + data[8 * so + i];     // (2,2)
}
