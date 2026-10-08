// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Hand-written aether-abi/2 DEVICE plugin fixtures for test_ExecContractDevice.cu
// — the PTX twin of host_plugin_execv2.cpp, body for body (ABI-only, no aether,
// no code generator). Compiled to raw PTX at configure time and loaded through
// `cuModuleLoadData`, exactly as a deployed plugin artifact is.
//
// Two things make these v2 rather than v1:
//   * every kernel takes the int64 PARTITION TRIPLE after its role args, and its
//     global sample index is `base + flat` with the early-out on `count` — the
//     flat-index shape a generated kernel already emits, re-based;
//   * the module exports `eagle_layout_sizes` as a `__device__` global, which the
//     loader resolves with `cuModuleGetGlobal` (there is no dlsym on a PTX module
//     — there is no dynamic symbol lookup on a PTX module).
#include "plugin/gref_layout.h"  // device TU, layout-only

#include <cstdint>

using namespace eagle::plugin;

// The layout self-check export — derived from this build's own PODs, for
// the same reason the host fixture derives it (the MISMATCH arm is a separate
// fixture, device_plugin_badlayout.cu).
extern "C" __device__ unsigned long long eagle_layout_sizes[5] = {
    sizeof(GRefMirror), sizeof(ScalarHandle), sizeof(IntHandle),
    sizeof(EAGLE_ABI_INDEX_T), sizeof(PartitionTriple),
};

static constexpr long long kLanes = 4;

// The ONE index seam: a flat thread index, an early-out on the partition's
// own `count`, and a global sample index of `base + flat`.
__device__ __forceinline__ bool sample_index(long long base, long long count,
                                             long long& i)
{
    const long long flat =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (flat >= count) return false;
    i = base + flat;
    return true;
}

extern "C" __global__ void exec_local(ScalarHandle y, ScalarHandle x, double a,
                                      double b, std::uint32_t /*n*/,
                                      long long base, long long count,
                                      long long /*nSamples*/)
{
    long long i;
    if (!sample_index(base, count, i)) return;
    static_cast<double*>(y.data)[i] =
        a * static_cast<const double*>(x.data)[i] + b;
}

extern "C" __global__ void exec_mapreduce(ScalarHandle partial, ScalarHandle x,
                                          std::uint32_t /*n*/, long long base,
                                          long long count, long long /*nSamples*/)
{
    long long i;
    if (!sample_index(base, count, i)) return;
    // ELEMENTWISE (L1): one per-sample contribution into the accumulate plane.
    // The COMBINE is eagle's, in a fixed order — never an in-kernel atomic.
    static_cast<double*>(partial.data)[i] = static_cast<const double*>(x.data)[i];
}

extern "C" __global__ void exec_scatter(ScalarHandle acc, ScalarHandle x,
                                        std::uint32_t /*n*/, long long base,
                                        long long count, long long /*nSamples*/)
{
    long long i;
    if (!sample_index(base, count, i)) return;
    // cross_sample_write: the target lane is NOT this sample's own column. On the
    // device that means an atomic, which is exactly why L3 places this class on
    // single-device structures only until a partial-accum combine is ruled.
    atomicAdd(static_cast<double*>(acc.data) + (i % kLanes),
              static_cast<const double*>(x.data)[i]);
}

extern "C" __global__ void exec_wide(ScalarHandle wout, ScalarHandle win,
                                     ScalarHandle aout, std::uint32_t /*n*/,
                                     long long base, long long count,
                                     long long /*nSamples*/)
{
    long long i;
    if (!sample_index(base, count, i)) return;
    const double v = static_cast<const double*>(win.data)[i];
    static_cast<double*>(wout.data)[i] = 2.0 * v;
    static_cast<double*>(aout.data)[i] = v + 1.0;
}

// The triple made OBSERVABLE: thread 0 of the launch records what eagle handed it.
extern "C" __global__ void exec_triple(ScalarHandle t, std::uint32_t /*n*/,
                                       long long base, long long count,
                                       long long nSamples)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double* d = static_cast<double*>(t.data);
    d[0] = double(base);
    d[1] = double(count);
    d[2] = double(nSamples);
}

// The CROSS-SAMPLE READ twin of exec_gather_host: sample `i` reads an
// absolute index that is not its own, out of a table that is WHOLE on every rank.
extern "C" __global__ void exec_gather(ScalarHandle y, ScalarHandle table,
                                       std::uint32_t /*n*/, long long base,
                                       long long count, long long nSamples)
{
    long long i;
    if (!sample_index(base, count, i)) return;
    static_cast<double*>(y.data)[i] =
        static_cast<const double*>(table.data)[(i * 7 + 3) % nSamples];
}

// The LEGACY (aether-abi/1) kernel — no triple, the whole-view shape.
extern "C" __global__ void exec_legacy(ScalarHandle y, ScalarHandle x,
                                       std::uint32_t n)
{
    const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    static_cast<double*>(y.data)[i] = static_cast<const double*>(x.data)[i] + 1.0;
}
