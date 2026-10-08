// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Hand-written CUDA plugin fixture — the aether-era stand-in for a generated
// "pure" (read-modify-write) kernel shape: a running counter bumped by a
// per-launch rate, same semantics as the original generator script's
// `bump` kernel. ABI-only (gref_layout.h) — ported from an
// earlier generated source under ABI v2 (a prior-era plugin cannot
// coexist with the ported eagle host by construction — it reads garbage
// through the new 32/40-byte mirrors — so the "mixed-state simulation" this
// fixture previously stood in for is moot post-port).
#include "plugin/gref_layout.h"  // device TU, layout-only

#include <cstdint>

using namespace eagle::plugin;

// arg_spec: [("mutable","counter"), ("terminated","terminated"),
//            ("uniform","rate"), ("nsamples","nsamples")]
extern "C" __global__
void raptor_kernel(
    ScalarHandle counter, ScalarHandle terminated, double rate, std::uint32_t n)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto* term = static_cast<const unsigned char*>(terminated.data);
    if (term != nullptr && term[i] != 0) return;

    double* c = static_cast<double*>(counter.data);
    c[i] = c[i] + rate;
}
