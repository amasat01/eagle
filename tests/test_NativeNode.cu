// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include "eagle/filtering/FilteringSliceNode.h"
#include "eagle/filtering/ScanNode.h"
#include "eagle/cpu/Graph.h"
#include "eagle/reduce/ReductionNode.h"
#include "eagle/cuda/Reduction.h"
#include "eagle/cpu/Reduction.h"
#include "eagle/util/ObservableArray.h"

#include "TestBase.h"

namespace eagle_tests {
namespace NativeNodeTest {

using idx_t = eagle::idx_t;
using SumOp  = aether::SumOp<double>;
using RN     = eagle::reduce::ReductionNode<double, SumOp>;
using ScanOp = aether::SumOp<idx_t>;
using SN     = eagle::filtering::ScanNode<idx_t, ScanOp, true>;  // inclusive

class NativeNodeTest : public Test {
public:
    /** @brief Fill @p arr with a ramp starting at @p base; upload; return the
     *  host sum (the reduction reference). */
    static double fillRamp(aether::Array<double>& arr, double base)
    {
        const eagle::GRefArrT<double> h = arr.hostView();
        double s = 0.0;
        for (idx_t i = 0; i < idx_t(arr.samples()); ++i) {
            const double v = base + double(i);
            h(i)           = v;
            s += v;
        }
        arr.upload();
        return s;
    }
};

// A chain (B depends on A) lets the arena reuse A's work buffer for B: one slot,
// not two — and both reductions still produce the correct, independent result.
TEST_F(NativeNodeTest, ChainReusesScratchAndReducesCorrectly)
{
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    const idx_t n = 4096;
    aether::Array<double> a = eagle::makeArray<double>(n);
    aether::Array<double> b = eagle::makeArray<double>(n);
    const double sa = fillRamp(a, 1.0);
    const double sb = fillRamp(b, 1000.0);
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());

    cudaStream_t stream;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));

    double ra = 0.0, rb = 0.0;
    eagle::cuda::Graph g;
    const idx_t nA
        = g.addNative(RN(&ra, a.deviceView().as_const(), n, 0.0, stream), {});
    g.addNative(RN(&rb, b.deviceView().as_const(), n, 0.0, stream), { nA });
    g.finalizeNatives();

    // The whole point of the graph-level arena: B reuses A's dead scratch.
    ASSERT_EQ(g.scratchArena().slotCount(), (idx_t)1);

    eagle::cuda::Launcher launcher = g.launcher();
    launcher.launch();
    launcher.synchronize();

    ASSERT_NEAR(ra, sa, 1e-6);
    ASSERT_NEAR(rb, sb, 1e-6);
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));
}

// Two independent reductions (no declared dependency) get distinct slots — the
// allocator never aliases regions whose lifetimes it cannot prove disjoint.
TEST_F(NativeNodeTest, IndependentReductionsDoNotShareScratch)
{
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    const idx_t n = 4096;
    aether::Array<double> a = eagle::makeArray<double>(n);
    aether::Array<double> b = eagle::makeArray<double>(n);
    const double sa = fillRamp(a, 1.0);
    const double sb = fillRamp(b, 1000.0);
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());

    cudaStream_t stream;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));

    double ra = 0.0, rb = 0.0;
    eagle::cuda::Graph g;
    g.addNative(RN(&ra, a.deviceView().as_const(), n, 0.0, stream), {});
    g.addNative(RN(&rb, b.deviceView().as_const(), n, 0.0, stream), {});
    g.finalizeNatives();

    ASSERT_EQ(g.scratchArena().slotCount(), (idx_t)2);

    eagle::cuda::Launcher launcher = g.launcher();
    launcher.launch();
    launcher.synchronize();

    ASSERT_NEAR(ra, sa, 1e-6);
    ASSERT_NEAR(rb, sb, 1e-6);
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));
}

// A native reduction matches the free-builder reduceBlocking bit-for-bit path.
TEST_F(NativeNodeTest, MatchesFreeBuilder)
{
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    const idx_t n = 10000;
    aether::Array<double> a = eagle::makeArray<double>(n);
    const double sa = fillRamp(a, 0.5);
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());

    cudaStream_t stream;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));

    // Reference via the free builder's blocking Array overload (the GRef
    // overload has a pre-existing makeBuffer(GRef) mismatch, unrelated here).
    const double ref
        = eagle::cuda::Reduction::reduceBlocking<double, SumOp>(
            a, 0.0, stream);

    double rn = 0.0;
    eagle::cuda::Graph g;
    g.addNative(RN(&rn, a.deviceView().as_const(), n, 0.0, stream), {});
    g.finalizeNatives();
    eagle::cuda::Launcher launcher = g.launcher();
    launcher.launch();
    launcher.synchronize();

    ASSERT_NEAR(rn, ref, 1e-9);
    ASSERT_NEAR(rn, sa, 1e-6);
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));
}

// Two chained scans share one BLOCKSUMS slot; both produce the correct inclusive
// prefix over distinct inputs (a second native primitive on the same protocol).
TEST_F(NativeNodeTest, ScanChainReusesScratchAndScansCorrectly)
{
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    const idx_t n = 2048;
    aether::Array<idx_t> ina  = eagle::makeArray<idx_t>(n);
    aether::Array<idx_t> outa = eagle::makeArray<idx_t>(n);
    aether::Array<idx_t> inb  = eagle::makeArray<idx_t>(n);
    aether::Array<idx_t> outb = eagle::makeArray<idx_t>(n);
    {
        const eagle::GRefArrT<idx_t> ha = ina.hostView();
        const eagle::GRefArrT<idx_t> hb = inb.hostView();
        for (idx_t i = 0; i < n; ++i) {
            ha(i) = 1;
            hb(i) = 2;
        }
    }
    ina.upload();
    inb.upload();
    outa.upload();
    outb.upload();
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());

    cudaStream_t stream;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));

    eagle::cuda::Graph g;
    const idx_t nA
        = g.addNative(SN(ina.deviceView().as_const(), outa.deviceView(), stream), {});
    g.addNative(SN(inb.deviceView().as_const(), outb.deviceView(), stream), { nA });
    g.finalizeNatives();

    // Chain (B depends on A) reuses A's BLOCKSUMS buffer: one slot, not two.
    ASSERT_EQ(g.scratchArena().slotCount(), (idx_t)1);

    eagle::cuda::Launcher launcher = g.launcher();
    launcher.launch();
    launcher.synchronize();

    outa.download(stream);
    outb.download(stream);
    EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(stream));

    const eagle::GRefArrT<idx_t> oa = outa.hostView();
    const eagle::GRefArrT<idx_t> ob = outb.hostView();
    for (idx_t i = 0; i < n; ++i) {
        ASSERT_EQ(oa(i), i + 1);        // inclusive scan of ones
        ASSERT_EQ(ob(i), 2 * (i + 1));  // inclusive scan of twos
    }
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));
}

// A compaction (predicate -> scan -> scatter) as a native node, its whole 3N
// scanner scratch drawn from the arena: two chained compactions share one 3N slot,
// and each index map is verified against a known keep-even predicate (identity
// data means slice[j] reads back the j-th kept index == 2j).
TEST_F(NativeNodeTest, FilteringSliceChainReusesScratchAndCompactsCorrectly)
{
    using ArrT   = eagle::util::observable::Array<idx_t>;
    using SliceT = eagle::util::Slice<ArrT>;
    using FSN    = eagle::filtering::FilteringSliceNode<ArrT>;

    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    const idx_t n    = 2048;
    const idx_t keep = n / 2;  // even indices

    ArrT dataA(n), dataB(n);
    aether::Array<idx_t> predA = eagle::makeArray<idx_t>(n);
    aether::Array<idx_t> predB = eagle::makeArray<idx_t>(n);
    {
        typename ArrT::GRef ha                 = dataA.hostView();
        typename ArrT::GRef hb                 = dataB.hostView();
        const eagle::GRefArrT<idx_t> pa        = predA.hostView();
        const eagle::GRefArrT<idx_t> pb        = predB.hostView();
        for (idx_t i = 0; i < n; ++i) {
            ha(i) = i;  // identity payload -> slice[j] reads back the kept index
            hb(i) = i;
            pa(i) = (i % 2 == 0) ? 1 : 0;  // keep the even indices
            pb(i) = (i % 2 == 0) ? 1 : 0;
        }
    }
    SliceT sliceA(dataA), sliceB(dataB);
    sliceA.upload();
    sliceB.upload();
    predA.upload();
    predB.upload();
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());

    cudaStream_t stream;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));

    eagle::cuda::Graph g;
    const idx_t nA = g.addNative(
        FSN(sliceA.deviceRef(), predA.deviceView().data(), n, stream), {});
    g.addNative(
        FSN(sliceB.deviceRef(), predB.deviceView().data(), n, stream), { nA });
    g.finalizeNatives();

    // Both compactions share one 3N scanner slot (peak, not sum).
    ASSERT_EQ(g.scratchArena().slotCount(), (idx_t)1);

    eagle::cuda::Launcher launcher = g.launcher();
    launcher.launch();
    launcher.synchronize();

    sliceA.download(stream);
    sliceB.download(stream);
    EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(stream));

    typename SliceT::GRef sa = sliceA.hostRef();
    typename SliceT::GRef sb = sliceB.hostRef();
    ASSERT_EQ(sa.size(), keep);
    ASSERT_EQ(sb.size(), keep);
    for (idx_t j = 0; j < keep; ++j) {
        eagle::SampleIndex idx = eagle::SampleIndex::make(j);
        ASSERT_EQ(sa[idx], 2 * j);  // j-th kept even index
        ASSERT_EQ(sb[idx], 2 * j);
    }
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));
}

// Dual-backend equality: the SAME ReductionNode class run through the CUDA graph
// (device data) and through the host executor (host-mirrored data) agree — one
// binary proving the two backends compute the same result on the same protocol.
TEST_F(NativeNodeTest, HostAndDeviceReductionAgree)
{
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    const idx_t n = 4096;
    aether::Array<double> a = eagle::makeArray<double>(n);
    const double sa = fillRamp(a, 1.0);  // fills host, then uploads
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());

    // Device path — CUDA graph.
    cudaStream_t stream;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));
    double rd = 0.0;
    eagle::cuda::Graph g;
    g.addNative(RN(&rd, a.deviceView().as_const(), n, 0.0, stream), {});
    g.finalizeNatives();
    eagle::cuda::Launcher launcher = g.launcher();
    launcher.launch();
    launcher.synchronize();

    // Host path — CPU executor over the same (host-resident) ramp.
    double rh = 0.0;
    eagle::cpu::Graph hg;
    hg.addNative(RN(&rh, a.hostView().as_const(), n, 0.0), {});
    hg.run();

    ASSERT_NEAR(rd, rh, 1e-9);  // backends agree
    ASSERT_NEAR(rh, sa, 1e-6);  // and both match the reference
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));
}

}  // namespace NativeNodeTest
}  // namespace eagle_tests
