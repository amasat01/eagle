// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// A hand-written CUDA plugin fixture for test_PluginRegistryDtype.cu's
// SoftdoubleLoadsAndLaunches check -- the PTX a generated deploy would emit for a
// pure kernel `advance(step, val: Mutable[float])` (see
// eagle/plugin/pure_inject/pure_inject_demo.cu's documented `advance`
// kernel), but hand-written and ABI-only: no code generator dependency, exactly like
// fixtures/host_plugin_addvec.cpp is for the CPU host path. The device
// PluginRegistry (plugin/plugin_registry/registry.h) routes purely on the
// sidecar's `scalar_type` TAG and the gref_layout.h PODs — it never inspects the
// kernel's own arithmetic — so a raw-`double` kernel is a faithful stand-in
// for a real SoftDouble artifact: SoftDouble is bit-identical `double` at
// every buffer boundary (a static_assert(sizeof(SoftDouble) == 8) enforces
// it), and registry.h is itself deliberately dependency-free.
#include "plugin/gref_layout.h"  // device TU, layout-only

#include <cstdint>

using namespace eagle::plugin;

// arg_spec: [("mutable","val"), ("terminated","terminated"), ("uniform","step"),
// ("nsamples","nsamples")] — mirrors the pure_inject_demo.cu `advance` shape.
// `nsamples` is packed as `unsigned` by PluginRegistry::inject
// (`counts.push_back(unsigned(n))` in registry.h), so the last parameter here
// is `std::uint32_t`, not a signed int.
extern "C" __global__ void raptor_kernel(
    ScalarHandle val, ScalarHandle terminated, double step, std::uint32_t n)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto* term = static_cast<const unsigned char*>(terminated.data);
    if (term != nullptr && term[i] != 0) return;
    auto* v = static_cast<double*>(val.data);
    v[i] = v[i] + step;
}
