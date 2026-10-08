// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The aether-abi/2 layout self-check's OTHER RED arm: a correctly TAGGED v2
// plugin that exports no `eagle_layout_sizes` at all. A v2 artifact MUST
// carry it, so its absence is a load refusal naming the
// symbol — never a lenient "absent, so trust the tag".
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

extern "C" void exec_nolayout_host(void* const* params, std::int64_t base,
                                   std::int64_t count, std::int64_t /*nSamples*/)
{
    double* y = static_cast<double*>(
        static_cast<const ScalarHandle*>(params[0])->data);
    for (std::int64_t i = base; i < base + count; ++i) y[i] = 1.0;
}
