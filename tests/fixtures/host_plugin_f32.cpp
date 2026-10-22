// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Hand-written float32 CPU plugin fixtures for test_HostPlugin.cpp: the `.so` a plugin
// built with ``scalar_type: float32`` would emit. Binary ABI only (gref_abi.h), no
// eagle headers, no CUDA. Every buffer and the by-value uniform are 4-byte floats.
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

// out[i] = a[i] + b[i]: a scalar `mutable` output, two `per_sample` inputs, `nsamples`.
extern "C" void addvec32_host(void* const* params, std::int32_t n)
{
    float* out = static_cast<float*>(static_cast<const ScalarHandle*>(params[0])->data);
    const float* a = static_cast<const float*>(static_cast<const ScalarHandle*>(params[1])->data);
    const float* b = static_cast<const float*>(static_cast<const ScalarHandle*>(params[2])->data);
#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

// out[i] = trace(M[i]): a scalar `mutable` output, a 3x3 `mat_in`, `nsamples`.
extern "C" void mattrace32_host(void* const* params, std::int32_t n)
{
    float* out = static_cast<float*>(static_cast<const ScalarHandle*>(params[0])->data);
    const GRefMirror* M = static_cast<const GRefMirror*>(params[1]);
    const float* data = reinterpret_cast<const float*>(M->data_);
    const std::uint64_t so = M->compStride_;
#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i)
        out[i] = data[0 * so + i] + data[4 * so + i] + data[8 * so + i];
}

// out[i] = x[i] * gain + tab[i % 4]: a scalar `mutable` output, a `vec_in` GRef, a real
// `uniform`, a `lookup` table of 4, `nsamples`.
extern "C" void vecgain32_host(void* const* params, std::int32_t n)
{
    float* out = static_cast<float*>(static_cast<const ScalarHandle*>(params[0])->data);
    const GRefMirror* X = static_cast<const GRefMirror*>(params[1]);
    const float* x = reinterpret_cast<const float*>(X->data_);
    const float gain = *static_cast<const float*>(params[2]);
    const float* tab = static_cast<const float*>(static_cast<const ScalarHandle*>(params[3])->data);
#pragma omp parallel for
    for (std::int32_t i = 0; i < n; ++i)
        out[i] = x[i] * gain + tab[i % 4];
}

// A finishing kernel: marks every sample with k[i] > 0 finished (writes the mask) and counts
// them into the registry-owned finished_count cell. arg_spec: mutable k (float), terminated
// mask, lookup finished_count (uint32), nsamples.
extern "C" void countfin32_host(void* const* params, std::int32_t n)
{
    const float* k = static_cast<const float*>(static_cast<const ScalarHandle*>(params[0])->data);
    unsigned char* mask = static_cast<unsigned char*>(static_cast<const ScalarHandle*>(params[1])->data);
    std::uint32_t* count = static_cast<std::uint32_t*>(static_cast<const ScalarHandle*>(params[2])->data);
    for (std::int32_t i = 0; i < n; ++i)
        if (k[i] > 0.0f) {
            mask[i] = 1;
            ++*count;
        }
}
