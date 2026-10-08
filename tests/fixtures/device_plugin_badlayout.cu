// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The DEVICE face of the layout self-check's RED arm: a correctly
// aether-abi/2-tagged PTX module whose exported
// `eagle_layout_sizes` disagrees with the loading host's in the LAST field
// (`sizeof(partition triple)` reported as 16 rather than 24). The loader must
// resolve the symbol with `cuModuleGetGlobal`, stage it, and refuse NAMING the
// field — the tag alone can never catch this.
#include "plugin/gref_layout.h"  // device TU, layout-only

#include <cstdint>

using namespace eagle::plugin;

extern "C" __device__ unsigned long long eagle_layout_sizes[5] = {
    sizeof(GRefMirror), sizeof(ScalarHandle), sizeof(IntHandle),
    sizeof(EAGLE_ABI_INDEX_T), 16ull,   // WRONG on purpose
};

extern "C" __global__ void exec_bad(ScalarHandle y, std::uint32_t /*n*/,
                                    long long base, long long count,
                                    long long /*nSamples*/)
{
    const long long flat =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return;
    static_cast<double*>(y.data)[base + flat] = 0.0;
}
