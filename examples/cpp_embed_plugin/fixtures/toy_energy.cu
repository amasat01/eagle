// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Hand-written CUDA plugin fixture -- the primal half of the toy VJP pair (a CUDA demo
// exercising a derivative artifact). `toy_energy(x: vec_in[3], p: uniform) ->
// e = p * exp(-0.5 * |x|^2)`, ABI-only (gref_layout.h) -- the same closed form
// as the code generator's own downstream-toy-extension example, reimplemented
// by hand so this example never imports the code generator.
#include "plugin/gref_layout.h"  // device TU, layout-only

#include <cstdint>
#include <math.h>

using namespace eagle::plugin;

// arg_spec: [("vec_in","x"), ("mutable","e"), ("uniform","p"), ("nsamples","nsamples")]
extern "C" __global__ void raptor_kernel(
    GRefMirror x, ScalarHandle e, double p, std::uint32_t n)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double* xd        = x.data_;
    const std::uint64_t so  = x.compStride_;
    const double x0 = xd[0 * so + i], x1 = xd[1 * so + i], x2 = xd[2 * so + i];
    const double r2 = x0 * x0 + x1 * x1 + x2 * x2;
    static_cast<double*>(e.data)[i] = p * exp(-0.5 * r2);
}
