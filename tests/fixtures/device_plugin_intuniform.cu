// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// A hand-written CUDA plugin fixture for test_PluginRegistryUniformInt.cu's
// MixedFloatAndIntUniformsLoadAndLaunch check -- the PTX a generated deploy would emit for a
// pure kernel taking BOTH a float64 and an int64 uniform.
// Hand-written and ABI-only: no code generator dependency, exactly like
// fixtures/device_plugin_softdouble.cu and fixtures/host_plugin_intuniform.cpp.
//
// THE PARAMETER TYPES ARE THE POINT. A generated kernel spells its integer quantity
// `using Int = long long;` and emits a uniform as
// `AETHER_GRID_CONSTANT() <type> p_<name>`, so an int uniform occupies 8 SIGNED bytes. This
// fixture declares exactly that (`long long k`), beside a float64 uniform
// (`double gain`), so the launch proves the registry packs each kind into its own
// by-value slot in arg_spec order. `nsamples` is packed as `unsigned` by
// PluginRegistry::inject (`counts.push_back(unsigned(n))`), so the last parameter is
// `std::uint32_t`, matching the softdouble fixture.
//
// GRID_CONSTANT is deliberately NOT spelled here: it is an aether macro
// (`const __grid_constant__` on sm_70+) and this fixture avoids it by construction,
// like every other plugin fixture. It is a parameter-passing OPTIMIZATION, not part of
// the ABI the registry packs — cuLaunchKernel hands the same by-value bytes either way.
#include "plugin/gref_layout.h"  // device TU, layout-only

#include <cstdint>

using namespace eagle::plugin;

// arg_spec: [("mutable","val"), ("uniform","gain"), ("uniform","k"),
// ("nsamples","nsamples")].
extern "C" __global__ void raptor_kernel(
    ScalarHandle val, double gain, long long k, std::uint32_t n)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    auto* v = static_cast<double*>(val.data);
    v[i] = v[i] * gain + static_cast<double>(k);
}
