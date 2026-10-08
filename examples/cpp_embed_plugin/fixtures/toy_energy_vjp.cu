// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Hand-written CUDA plugin fixture -- the derivative half of the toy VJP pair. The
// custom VJP of `toy_energy` (fixtures/toy_energy.cu): given the primal input
// `x`, the uniform `p`, and the upstream cotangent `e_bar`, writes
//
//   x_bar[k] = e_bar * d(e)/d(x_k) = e_bar * (-x_k) * e
//   p_bar    = e_bar * d(e)/d(p)   = e_bar * exp(-0.5 * |x|^2)
//
// Phase A derivative artifacts are recompute-only (no residual buffers): this
// kernel re-derives `e` from `x`/`p` itself rather than taking it as an input,
// exactly like a real generated/custom VJP (see sidecar.h's `Derivative`
// comment: "the artifact re-reads its primal inputs"). ABI-only (gref_layout.h).
// The optional `derivative` sidecar block (this fixture's .json)
// carries the metadata a consumer would branch on; the launch itself ignores it.
#include "plugin/gref_layout.h"  // device TU, layout-only

#include <cstdint>
#include <math.h>

using namespace eagle::plugin;

// arg_spec: [("vec_in","x"), ("per_sample","e_bar"), ("uniform","p"),
//            ("mutable","x_bar"), ("mutable","p_bar"), ("nsamples","nsamples")]
extern "C" __global__ void raptor_kernel(
    GRefMirror x, ScalarHandle e_bar, double p,
    GRefMirror x_bar, ScalarHandle p_bar, std::uint32_t n)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double* xd        = x.data_;
    const std::uint64_t so  = x.compStride_;
    const double x0 = xd[0 * so + i], x1 = xd[1 * so + i], x2 = xd[2 * so + i];
    const double r2    = x0 * x0 + x1 * x1 + x2 * x2;
    const double e_val = p * exp(-0.5 * r2);
    const double eb     = static_cast<const double*>(e_bar.data)[i];

    double* xb              = x_bar.data_;
    const std::uint64_t xbso = x_bar.compStride_;
    xb[0 * xbso + i] = eb * (-x0) * e_val;
    xb[1 * xbso + i] = eb * (-x1) * e_val;
    xb[2 * xbso + i] = eb * (-x2) * e_val;
    static_cast<double*>(p_bar.data)[i] = eb * exp(-0.5 * r2);
}
