// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The aether-abi/2 layout self-check's RED arm:
// a plugin whose exported `eagle_layout_sizes` DISAGREES with the loading host's.
//
// The disagreement is in the LAST field — `sizeof(partition triple)` reported as
// 16 rather than 24 — because that is the failure the tag alone can never catch:
// the artifact's tag is a perfectly correct "aether-abi/2", and a loader that
// trusted it would decode a three-int64 triple out of two and then compute
// silently wrong answers. The refusal must name the field.
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

extern "C" const std::uint64_t eagle_layout_sizes[5] = {
    sizeof(GRefMirror), sizeof(ScalarHandle), sizeof(IntHandle),
    sizeof(EAGLE_ABI_INDEX_T), 16u,   // WRONG on purpose: the triple is 3 x int64
};

extern "C" void exec_bad_host(void* const* params, std::int64_t base,
                              std::int64_t count, std::int64_t /*nSamples*/)
{
    double* y = static_cast<double*>(
        static_cast<const ScalarHandle*>(params[0])->data);
    for (std::int64_t i = base; i < base + count; ++i) y[i] = 0.0;
}
