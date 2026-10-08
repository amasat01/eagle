// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// A hand-written CPU plugin fixture for test_HostPlugin.cpp — the `.so` a
// deployed plugin would emit for a pure kernel ``out[i] = a[i] + b[i]``. It
// depends ONLY on the binary ABI (gref_abi.h PODs): no eagle headers, no
// CUDA. The host entry does its OWN OpenMP parallel
// loop over the N samples — the plugin owns its parallelism, the registry only
// hands it the packed args and the sample count. This is the "outer thread loop
// inside the plugin" half of the story.
#include "plugin/gref_abi.h"

#include <cstdint>

// The plugin ABI PODs live in the device-neutral ``eagle::plugin`` protocol namespace.
using namespace eagle::plugin;

// arg_spec order (must match the sidecar the host registers): a scalar `mutable`
// output handle, two `per_sample` input handles, then `nsamples`. Each params[i]
// points to the same POD the device registry would hand cuLaunchKernel.
extern "C" void addvec_host(void* const* params, std::int32_t n)
{
    double* out = static_cast<double*>(
        static_cast<const ScalarHandle*>(params[0])->data);
    const double* a = static_cast<const double*>(
        static_cast<const ScalarHandle*>(params[1])->data);
    const double* b = static_cast<const double*>(
        static_cast<const ScalarHandle*>(params[2])->data);

#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}
