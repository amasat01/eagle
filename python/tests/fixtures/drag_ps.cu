// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Hand-written CUDA plugin fixture — the aether-era stand-in for a generated
// "vector" per-sample-drag acceleration, same semantics as
// the original generator script's `drag_ps` kernel: a = -0.5 * cd *
// area / mass * |v| * v, accumulated into `out`. ABI-only (gref_layout.h) —
// ported from an earlier generated source under ABI v2
// (a prior-era plugin cannot coexist with the ported eagle host by
// construction — it reads garbage through the new 32/40-byte mirrors).
#include "plugin/gref_layout.h"  // device TU, layout-only

#include <cstdint>
#include <math.h>

using namespace eagle::plugin;

// arg_spec: [("out","out"), ("vec_in","velocity"), ("per_sample","mass"),
//            ("per_sample","area"), ("per_sample","cd"), ("terminated","terminated")]
extern "C" __global__
void raptor_kernel(
    GRefMirror out, GRefMirror velocity, ScalarHandle mass, ScalarHandle area,
    ScalarHandle cd, ScalarHandle terminated)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= out.samples_) return;
    const auto* term = static_cast<const unsigned char*>(terminated.data);
    if (term != nullptr && term[i] != 0) return;

    const double* vd        = velocity.data_;
    const std::uint64_t so  = velocity.compStride_;
    const double v0 = vd[0 * so + i], v1 = vd[1 * so + i], v2 = vd[2 * so + i];
    const double vNorm = sqrt(v0 * v0 + v1 * v1 + v2 * v2);

    const double m  = static_cast<const double*>(mass.data)[i];
    const double a  = static_cast<const double*>(area.data)[i];
    const double cD = static_cast<const double*>(cd.data)[i];
    const double coeff = static_cast<double>(-0.5) * cD * a / m * vNorm;

    double* od               = out.data_;
    const std::uint64_t oso  = out.compStride_;
    od[0 * oso + i] += coeff * v0;
    od[1 * oso + i] += coeff * v1;
    od[2 * oso + i] += coeff * v2;
}
