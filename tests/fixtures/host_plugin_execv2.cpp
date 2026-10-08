// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Hand-written aether-abi/2 CPU plugin fixtures for tests/test_ExecContract.cpp
// (ABI-only, no eagle machinery, no aether, no CUDA -- exactly what a
// generated deploy would emit, minus the generator itself).
//
// WHAT IS DIFFERENT FROM host_plugin_addvec.cpp (the v1 fixture beside it):
//   * the entry takes the int64 PARTITION TRIPLE `{base, count, nSamples}` after
//     the packed params, and is a SERIAL range over `[base, base + count)` —
//     there is NO `#pragma omp parallel for` in this
//     file. Threading and tiling are eagle's (`eagle::exec::HostTeam`), and each
//     tile calls these entries with its OWN triple, which is precisely what keeps
//     wide/accum columns disjoint;
//   * the object exports `eagle_layout_sizes` -- the layout self-check.
//
// The three declared ACCESS CLASSES of the contract (L3) each get a body:
//   `exec_local`      sample_local        y[i] = a*x[i] + b
//   `exec_mapreduce`  mapreduce(sum)      partial[i] = x[i]   (eagle folds)
//   `exec_scatter`    cross_sample_write  acc[i % kLanes] += x[i]
// plus `exec_wide` (all three wide roles at once), `exec_triple` (the triple made
// observable) and `exec_legacy` (a v1 entry, for the legacy bridge).
#include "plugin/gref_abi.h"

#include <cstdint>

using namespace eagle::plugin;

// ---------------------------------------------------------------------------
// The layout self-check export.
// ---------------------------------------------------------------------------
// Derived from this build's own PODs rather than hand-written: a real generated
// artifact computes it the same way (from the aether it was compiled against),
// and hand-written numbers here would only ever check the fixture's own typos. The
// MISMATCH arm is a separate fixture with deliberately wrong numbers
// (host_plugin_badlayout.cpp) — that is where the mechanism is proven.
extern "C" const std::uint64_t eagle_layout_sizes[5] = {
    sizeof(GRefMirror), sizeof(ScalarHandle), sizeof(IntHandle),
    sizeof(EAGLE_ABI_INDEX_T), sizeof(PartitionTriple),
};

// The number of accumulate lanes exec_scatter writes into — a cross-sample write:
// every sample contributes to a lane that is NOT its own column.
static constexpr std::int64_t kLanes = 4;

static inline double* handle_data(void* const* params, int i)
{
    return static_cast<double*>(
        static_cast<const ScalarHandle*>(params[i])->data);
}
static inline const double* const_handle_data(void* const* params, int i)
{
    return static_cast<const double*>(
        static_cast<const ScalarHandle*>(params[i])->data);
}

// arg_spec: [mutable y, per_sample x, uniform a, uniform b, nsamples]
extern "C" void exec_local_host(void* const* params, std::int64_t base,
                                std::int64_t count, std::int64_t /*nSamples*/)
{
    double* y       = handle_data(params, 0);
    const double* x = const_handle_data(params, 1);
    const double a  = *static_cast<const double*>(params[2]);
    const double b  = *static_cast<const double*>(params[3]);
    for (std::int64_t i = base; i < base + count; ++i)
        y[i] = a * x[i] + b;
}

// arg_spec: [accum_out partial, per_sample x, nsamples]
//
// The body stays ELEMENTWISE (L1): it writes ONE per-sample contribution into the
// accumulate plane and never reduces. The COMBINE is eagle's own
// `eagle::exec::fold`, which is what makes the combine ORDER a property of the
// partitioning rather than of the plugin.
extern "C" void exec_mapreduce_host(void* const* params, std::int64_t base,
                                    std::int64_t count, std::int64_t /*nSamples*/)
{
    double* partial = handle_data(params, 0);
    const double* x = const_handle_data(params, 1);
    for (std::int64_t i = base; i < base + count; ++i)
        partial[i] = x[i];
}

// arg_spec: [wide_out acc, per_sample x, nsamples]
extern "C" void exec_scatter_host(void* const* params, std::int64_t base,
                                  std::int64_t count, std::int64_t /*nSamples*/)
{
    double* acc     = handle_data(params, 0);
    const double* x = const_handle_data(params, 1);
    for (std::int64_t i = base; i < base + count; ++i)
        acc[i % kLanes] += x[i];
}

// arg_spec: [wide_out wout, wide_in win, accum_out aout, nsamples]
extern "C" void exec_wide_host(void* const* params, std::int64_t base,
                               std::int64_t count, std::int64_t /*nSamples*/)
{
    double* wout      = handle_data(params, 0);
    const double* win = const_handle_data(params, 1);
    double* aout      = handle_data(params, 2);
    for (std::int64_t i = base; i < base + count; ++i) {
        wout[i] = 2.0 * win[i];
        aout[i] = win[i] + 1.0;
    }
}

// arg_spec: [wide_out t, nsamples] — the triple made OBSERVABLE. Every invocation
// writes its own `{base, count, nSamples}` into the first three slots of the
// plane, so a test can assert what eagle actually handed the body.
extern "C" void exec_triple_host(void* const* params, std::int64_t base,
                                 std::int64_t count, std::int64_t nSamples)
{
    double* t = handle_data(params, 0);
    t[0] = double(base);
    t[1] = double(count);
    t[2] = double(nSamples);
}

// arg_spec: [mutable y, per_sample table, nsamples]
//
// The CROSS-SAMPLE READ body (`cross_sample_read`): sample `i`
// reads a sample that is NOT its own, at an absolute index derived from the TRUE
// `nSamples` -- the shape the code generator's Staged/lookup reads already have today. It is the
// body that makes RankPartition's replicated-input ruling checkable: the answer is
// bit-identical under any partitioning only because every rank holds the WHOLE
// table, and it would silently change the moment eagle sliced a shared role.
extern "C" void exec_gather_host(void* const* params, std::int64_t base,
                                 std::int64_t count, std::int64_t nSamples)
{
    double* y           = handle_data(params, 0);
    const double* table = const_handle_data(params, 1);
    for (std::int64_t i = base; i < base + count; ++i)
        y[i] = table[(i * 7 + 3) % nSamples];
}

// The LEGACY (aether-abi/1) entry — the old shape, unchanged, for the legacy bridge.
// arg_spec: [mutable y, per_sample x, nsamples]
extern "C" void exec_legacy_host(void* const* params, std::int32_t n)
{
    double* y       = handle_data(params, 0);
    const double* x = const_handle_data(params, 1);
    for (std::int32_t i = 0; i < n; ++i)
        y[i] = x[i] + 1.0;
}
