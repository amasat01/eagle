// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Hand-written CUDA plugin fixture — the aether-era stand-in for a generated
// "vector" gravity acceleration, same semantics as
// the original generator script's `gravity` kernel: a = -mu * r /
// |r|^3, accumulated into `out`. ABI-only (gref_layout.h) —
// ported from an earlier generated source under ABI v2 (a prior-era
// plugin cannot coexist with the ported eagle host by construction — it
// reads garbage through the new 32/40-byte mirrors).
#include "plugin/gref_layout.h"  // device TU, layout-only

#include <cstdint>
#include <math.h>

using namespace eagle::plugin;

// arg_spec: [("out","out"), ("vec_in","position"), ("terminated","terminated"),
//            ("uniform","mu")]
extern "C" __global__
void raptor_kernel(
    GRefMirror out, GRefMirror position, ScalarHandle terminated, double mu)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out.samples_) return;
    const auto* term = static_cast<const unsigned char*>(terminated.data);
    if (term != nullptr && term[i] != 0) return;

    const double* pd        = position.data_;
    const std::uint64_t so  = position.compStride_;
    const double p0 = pd[0 * so + i], p1 = pd[1 * so + i], p2 = pd[2 * so + i];
    const double rNorm = sqrt(p0 * p0 + p1 * p1 + p2 * p2);
    const double rCubedNormInv = 1.0 / (rNorm * rNorm * rNorm);

    double* od               = out.data_;
    const std::uint64_t oso  = out.compStride_;
    const double coeff = rCubedNormInv * (-mu);
    od[0 * oso + i] += coeff * p0;
    od[1 * oso + i] += coeff * p1;
    od[2 * oso + i] += coeff * p2;
}
