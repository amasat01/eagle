// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Hand-written CPU plugin fixtures for test_HostPluginUniformInt.cpp — the `.so` a
// deployed plugin would emit for two pure kernels that take an INTEGER uniform.
// Like the other host_plugin_* fixtures it depends ONLY on the binary ABI
// (gref_abi.h PODs): no eagle headers, no CUDA.
//
// WHY THE SIGNATURE SPELLS `long long`. A generated kernel's integer quantity is
// spelled `long long` by the code generator, emitted as
// `AETHER_GRID_CONSTANT() Int p_<name>` for a uniform. These entries therefore read the
// uniform slot as a `long long`, i.e. the exact 8 signed bytes the registry packs
// through `bind_uniform_int` — which is what makes the tests an ABI assertion rather
// than a value round-trip: if the registry ever packed a `double` into this slot, the
// bit pattern read back here would be an astronomically different integer, and if it
// packed 32 bits the high half would be lost.
#include "plugin/gref_abi.h"

#include <cstdint>

// The plugin ABI PODs live in the device-neutral ``eagle::plugin`` protocol namespace.
using namespace eagle::plugin;

// arg_spec order: a scalar `mutable` output handle, one int64 `uniform`, then
// `nsamples`. Writes the uniform straight through, so the test reads back exactly the
// integer it bound — including values no 32-bit slot could carry.
extern "C" void intuniform_host(void* const* params, std::int32_t n)
{
    double* out = static_cast<double*>(
        static_cast<const ScalarHandle*>(params[0])->data);
    const long long k = *static_cast<const long long*>(params[1]);

#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i)
        out[i] = static_cast<double>(k);
}

// The MIXED row set: a scalar `mutable` output handle, a `per_sample` input handle, a
// float64 `uniform`, an int64 `uniform`, then `nsamples`. The two uniform kinds are
// packed from two SEPARATE stable-storage vectors inside the registry, so this entry
// is what proves both land in their own params[] slot, in arg_spec order, with the
// right interpretation — an interleaving no single-kind kernel can exercise.
extern "C" void mixedscale_host(void* const* params, std::int32_t n)
{
    double* out = static_cast<double*>(
        static_cast<const ScalarHandle*>(params[0])->data);
    const double* a = static_cast<const double*>(
        static_cast<const ScalarHandle*>(params[1])->data);
    const double gain = *static_cast<const double*>(params[2]);
    const long long k = *static_cast<const long long*>(params[3]);

#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i)
        out[i] = a[i] * gain + static_cast<double>(k);
}
