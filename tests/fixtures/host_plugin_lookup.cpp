// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// A hand-written CPU plugin fixture for test_HostPlugin.cpp — the `.so` a
// deployed plugin would emit for a pure lookup kernel ``acc[i] = tab[idx(i)]``.
// Like host_plugin_addvec.cpp it depends ONLY on the binary ABI (gref_abi.h
// PODs): no eagle headers, no CUDA. The table ``tab`` is a consolidated read-only
// buffer — on the host it is packed as a flat ScalarHandle over the caller's own
// memory (no upload), read at a user-computed index. Here the index is just
// ``i % K`` (K = 16, matching the golden ``Table[16]``) so the gather is a closed
// form the test can check; the real index math is the device codegen's concern.
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

// arg_spec order (must match the sidecar the host registers): a scalar `mutable`
// output handle, a `lookup` table handle, then `nsamples`. Each params[i] points to
// the same POD the device registry would hand cuLaunchKernel.
extern "C" void lookup_host(void* const* params, std::int32_t n)
{
    double* acc = static_cast<double*>(
        static_cast<const ScalarHandle*>(params[0])->data);
    const double* tab = static_cast<const double*>(
        static_cast<const ScalarHandle*>(params[1])->data);
    constexpr std::int32_t K = 16;  // the declared table length (Table[16])

#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i)
        acc[i] = tab[i % K];
}
