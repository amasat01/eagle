// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Hand-written CUDA plugin fixture -- device `bind_matrix` proof
// kernel. The device twin of eagle/tests/fixtures/host_plugin_mattrace.cpp: a pure
// kernel `mattrace(M: mat_in[3,3]) -> out = trace(M)`, ABI-only (gref_layout.h), no
// exactly like fixtures/device_plugin_softdouble.cu is for the
// SoftDouble dtype pin. A matrix binds through the SAME 40-byte GRefMirror as a
// vector (flat (R*C, N) SoA, dim = r*C + c, sample-fastest), so `M.data_` /
// `M.compStride_` read exactly like a vector's -- proving
// `eagle::cuda::PluginRegistry::bind_matrix`
// (plugin/plugin_registry/registry.h) packs a `mat_in` argument correctly for a
// real driver-loaded launch.
#include "plugin/gref_layout.h"  // device TU, layout-only

#include <cstdint>

using namespace eagle::plugin;

// arg_spec: [("mutable","out"), ("mat_in","M"), ("nsamples","nsamples")]
extern "C" __global__ void raptor_kernel(
    ScalarHandle out, GRefMirror M, std::uint32_t n)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double* data     = M.data_;
    const std::uint64_t so = M.compStride_;
    double* o               = static_cast<double*>(out.data);
    o[i] = data[0 * so + i]    // (0,0)
         + data[4 * so + i]    // (1,1)
         + data[8 * so + i];   // (2,2)
}
