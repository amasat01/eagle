// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "eagle/filtering/FilteringSliceNode.h"
#include "eagle/filtering/ScanNode.h"
#include "eagle/cpu/Graph.h"
#include "eagle/reduce/ReductionNode.h"
#include "eagle/cuda/Reduction.h"
#include "eagle/cpu/Reduction.h"
#include "eagle/util/ObservableArray.h"

#include "TestBase.h"

namespace eagle_tests {
namespace HostNativeNodeTest {

using idx_t  = eagle::idx_t;
using SumOp  = aether::SumOp<double>;
using RN     = eagle::reduce::ReductionNode<double, SumOp>;
using ScanOp = aether::SumOp<idx_t>;
using SN     = eagle::filtering::ScanNode<idx_t, ScanOp, true>;  // inclusive

// Host twin of NativeNodeTest (test_NativeNode.cu): the SAME node classes and the
// SAME graph-level arena, run through the CPU executor (eagle::cpu::Graph)
// instead of a CUDA graph. The arena's ancestor-liveness reuse is mode-agnostic,
// so every slot-count assertion matches its CUDA sibling byte-for-byte; only the
// backing store (host malloc) and the compute kernels (OpenMP twins) differ.
class HostNativeNodeTest : public Test {
public:
    /** @brief Fill @p arr with a ramp starting at @p base (host storage); return
     *  the host sum (the reduction reference). No upload — host path. */
    static double fillRamp(aether::Array<double>& arr, double base)
    {
        const eagle::GRefArrT<double> h = arr.hostView();
        double s = 0.0;
        for (idx_t i = 0; i < idx_t(arr.samples()); ++i) {
            const double v = base + double(i);
            h(i)           = v;
            s += v;
        }
        return s;
    }
};

// A chain (B depends on A) lets the arena reuse A's slot for B: one slot, not two
// — the same reuse the CUDA path proves — and both host reductions are correct.
TEST_F(HostNativeNodeTest, ChainReusesScratchAndReducesCorrectly)
{
    const idx_t n = 4096;
    aether::Array<double> a = eagle::makeArray<double>(n);
    aether::Array<double> b = eagle::makeArray<double>(n);
    const double sa = fillRamp(a, 1.0);
    const double sb = fillRamp(b, 1000.0);

    double ra = 0.0, rb = 0.0;
    eagle::cpu::Graph g;
    const idx_t nA = g.addNative(RN(&ra, a.hostView().as_const(), n, 0.0), {});
    g.addNative(RN(&rb, b.hostView().as_const(), n, 0.0), { nA });
    g.finalize();

    // Same peak-not-sum reuse metric as the CUDA arena.
    ASSERT_EQ(g.scratchArena().slotCount(), (idx_t)1);

    g.run();

    ASSERT_NEAR(ra, sa, 1e-6);
    ASSERT_NEAR(rb, sb, 1e-6);
}

// Two independent reductions (no declared dependency) get distinct slots — the
// allocator never aliases regions whose lifetimes it cannot prove disjoint.
TEST_F(HostNativeNodeTest, IndependentReductionsDoNotShareScratch)
{
    const idx_t n = 4096;
    aether::Array<double> a = eagle::makeArray<double>(n);
    aether::Array<double> b = eagle::makeArray<double>(n);
    const double sa = fillRamp(a, 1.0);
    const double sb = fillRamp(b, 1000.0);

    double ra = 0.0, rb = 0.0;
    eagle::cpu::Graph g;
    g.addNative(RN(&ra, a.hostView().as_const(), n, 0.0), {});
    g.addNative(RN(&rb, b.hostView().as_const(), n, 0.0), {});
    g.finalize();

    ASSERT_EQ(g.scratchArena().slotCount(), (idx_t)2);

    g.run();

    ASSERT_NEAR(ra, sa, 1e-6);
    ASSERT_NEAR(rb, sb, 1e-6);
}

// A native reduction run on the host executor matches the free cpu::Reduction call.
TEST_F(HostNativeNodeTest, MatchesFreeOmpReduce)
{
    const idx_t n = 10000;
    aether::Array<double> a = eagle::makeArray<double>(n);
    const double sa = fillRamp(a, 0.5);

    const double ref = eagle::cpu::Reduction<double, SumOp>::reduce(
        a.hostView().as_const(), 0.0);

    double rn = 0.0;
    eagle::cpu::Graph g;
    g.addNative(RN(&rn, a.hostView().as_const(), n, 0.0), {});
    g.run();

    ASSERT_NEAR(rn, ref, 1e-9);
    ASSERT_NEAR(rn, sa, 1e-6);
}

// Two chained scans share one BLOCKSUMS slot (the arena IS exercised on host —
// runHost resolves the slot as a host pointer); both produce the correct
// inclusive prefix over distinct inputs.
TEST_F(HostNativeNodeTest, ScanChainReusesScratchAndScansCorrectly)
{
    const idx_t n = 2048;
    aether::Array<idx_t> ina = eagle::makeArray<idx_t>(n);
    aether::Array<idx_t> outa = eagle::makeArray<idx_t>(n);
    aether::Array<idx_t> inb = eagle::makeArray<idx_t>(n);
    aether::Array<idx_t> outb = eagle::makeArray<idx_t>(n);
    {
        const eagle::GRefArrT<idx_t> ha = ina.hostView();
        const eagle::GRefArrT<idx_t> hb = inb.hostView();
        for (idx_t i = 0; i < n; ++i) {
            ha(i) = 1;
            hb(i) = 2;
        }
    }

    eagle::cpu::Graph g;
    const idx_t nA = g.addNative(
        SN(ina.hostView().as_const(), outa.hostView()), {});
    g.addNative(SN(inb.hostView().as_const(), outb.hostView()), { nA });
    g.finalize();

    ASSERT_EQ(g.scratchArena().slotCount(), (idx_t)1);

    g.run();

    const eagle::GRefArrT<idx_t> oa = outa.hostView();
    const eagle::GRefArrT<idx_t> ob = outb.hostView();
    for (idx_t i = 0; i < n; ++i) {
        ASSERT_EQ(oa(i), i + 1);        // inclusive scan of ones
        ASSERT_EQ(ob(i), 2 * (i + 1));  // inclusive scan of twos
    }
}

// A compaction (predicate -> scan -> scatter) as a native node run on the host
// executor, its whole 3N scanner scratch drawn from the arena: two chained
// compactions share one 3N slot, and each index map is verified against a known
// keep-even predicate (identity data means slice[j] reads back the kept index).
TEST_F(HostNativeNodeTest, FilteringSliceChainReusesScratchAndCompactsCorrectly)
{
    using ArrT   = eagle::util::observable::Array<idx_t>;
    using SliceT = eagle::util::Slice<ArrT>;
    using FSN    = eagle::filtering::FilteringSliceNode<ArrT>;

    const idx_t n    = 2048;
    const idx_t keep = n / 2;  // even indices

    ArrT dataA(n), dataB(n);
    aether::Array<idx_t> predA = eagle::makeArray<idx_t>(n);
    aether::Array<idx_t> predB = eagle::makeArray<idx_t>(n);
    {
        typename ArrT::GRef ha           = dataA.hostRef();
        typename ArrT::GRef hb           = dataB.hostRef();
        const eagle::GRefArrT<idx_t> pa  = predA.hostView();
        const eagle::GRefArrT<idx_t> pb  = predB.hostView();
        for (idx_t i = 0; i < n; ++i) {
            ha(i) = i;  // identity payload -> slice[j] reads back the kept index
            hb(i) = i;
            pa(i) = (i % 2 == 0) ? 1 : 0;  // keep the even indices
            pb(i) = (i % 2 == 0) ? 1 : 0;
        }
    }
    SliceT sliceA(dataA), sliceB(dataB);

    eagle::cpu::Graph g;
    const idx_t nA = g.addNative(
        FSN(sliceA.hostRef(), predA.hostView().data(), n), {});
    g.addNative(
        FSN(sliceB.hostRef(), predB.hostView().data(), n), { nA });
    g.finalize();

    // Both compactions share one 3N scanner slot (peak, not sum).
    ASSERT_EQ(g.scratchArena().slotCount(), (idx_t)1);

    g.run();

    typename SliceT::GRef sa = sliceA.hostRef();
    typename SliceT::GRef sb = sliceB.hostRef();
    ASSERT_EQ(sa.size(), keep);
    ASSERT_EQ(sb.size(), keep);
    for (idx_t j = 0; j < keep; ++j) {
        eagle::SampleIndex idx = eagle::SampleIndex::make(j);
        ASSERT_EQ(sa[idx], 2 * j);  // j-th kept even index
        ASSERT_EQ(sb[idx], 2 * j);
    }
}

}  // namespace HostNativeNodeTest
}  // namespace eagle_tests
