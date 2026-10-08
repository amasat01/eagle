// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "TestBase.h"

#include "eagle/compose/GraphComposer.h"
#include "eagle/cuda.h"

#ifndef EAGLE_CPU_ONLY

namespace eagle_tests {
namespace GraphComposerTest {

using namespace eagle;
using namespace eagle::cuda;
using eagle::compose::GraphComposer;

/* ================================================================
 * Helper kernel: accumulates 1.0 into every element of buf[0..n) --
 * launching the owning Launcher N times must leave buf[i] == N (a
 * disabled/never-captured member must leave its buffer untouched at
 * whatever sentinel it started at), the same "replay conformance"
 * idiom test_CaptureFork.cu's ReplaysMatchHostOracle uses.
 * ================================================================ */
__global__ void incKernel(double* buf, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        buf[tid] += 1.0;
}

static cuda::Launcher buildIncLauncher(const cudaStream_t& stream, double* buf, int n)
{
    StreamCapturer cap(stream);
    cap.begin();
    incKernel<<<1, n, 0, stream>>>(buf, n);
    Graph g;
    g.stream(stream);
    g.addNode(CapturedGraph{ cap.end() });
    return g.launcher();
}

class GraphComposerFixture : public Test {
protected:
    static constexpr int N = 32;

    double* alloc(double init = 0.0)
    {
        double* d = nullptr;
        EAGLE_CHECK_ALWAYS(cudaMalloc(&d, N * sizeof(double)));
        std::vector<double> h(N, init);
        EAGLE_CHECK_ALWAYS(
            cudaMemcpy(d, h.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        // per-thread default stream: no ordering with the composer's streams
        EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
        bufs_.push_back(d);
        return d;
    }

    std::vector<double> download(double* d)
    {
        std::vector<double> h(N);
        EAGLE_CHECK_ALWAYS(
            cudaMemcpy(h.data(), d, N * sizeof(double), cudaMemcpyDeviceToHost));
        return h;
    }

    void expectAll(double* d, double want)
    {
        std::vector<double> h = download(d);
        for (int i = 0; i < N; i++)
            EXPECT_DOUBLE_EQ(h[i], want) << "element " << i;
    }

    void TearDown() override
    {
        for (double* d : bufs_)
            EAGLE_CHECK_ALWAYS(cudaFree(d));
    }

    std::vector<double*> bufs_;
};

/* ================================================================
 * 1. mode="sequenced": bitwise-identical to launching each member's
 *    Launcher directly (no composer at all) -- the composer must add
 *    no observable numeric difference for stream-owning members.
 * ================================================================ */
TEST_F(GraphComposerFixture, SequencedBitwiseVsDirectLaunch)
{
    constexpr int REPS = 5;
    double* bufA        = alloc();
    double* bufB        = alloc();

    Stream sA(/*nonBlocking=*/true);
    Stream sB(/*nonBlocking=*/true);

    GraphComposer composer("sequenced");
    composer.registerLauncher(buildIncLauncher(sA.cuda(), bufA, N), "A");
    composer.registerLauncher(buildIncLauncher(sB.cuda(), bufB, N), "B");
    composer.build();
    composer.launch(REPS);

    /* Direct reference: independent buffers, independent Launchers, no
     * composer -- REPS direct launch()+synchronize() calls each. */
    double* refA = alloc();
    double* refB = alloc();
    Stream rsA(/*nonBlocking=*/true);
    Stream rsB(/*nonBlocking=*/true);
    Launcher lA = buildIncLauncher(rsA.cuda(), refA, N);
    Launcher lB = buildIncLauncher(rsB.cuda(), refB, N);
    for (int i = 0; i < REPS; i++)
        lA.launch();
    lA.synchronize();
    for (int i = 0; i < REPS; i++)
        lB.launch();
    lB.synchronize();

    EXPECT_EQ(download(bufA), download(refA));
    EXPECT_EQ(download(bufB), download(refB));
    expectAll(bufA, double(REPS));
    expectAll(bufB, double(REPS));

    ASSERT_EQ(composer.firedHistory().size(), 1u);
    EXPECT_EQ(composer.firedHistory()[0], (std::vector<idx_t>{ 0, 1 }));
}

/* ================================================================
 * 2. Two-level nesting, bitwise: an INNER sequenced composer (two
 *    stream-owning members) registered as a member of an OUTER
 *    sequenced composer alongside a third, independent member --
 *    every buffer must match what launching all three Launchers
 *    directly, REPS times, would produce.
 * ================================================================ */
TEST_F(GraphComposerFixture, TwoLevelNestingBitwise)
{
    constexpr int REPS = 4;
    double* bufX        = alloc();
    double* bufY        = alloc();
    double* bufZ        = alloc();

    Stream sX(/*nonBlocking=*/true);
    Stream sY(/*nonBlocking=*/true);
    Stream sZ(/*nonBlocking=*/true);

    auto inner = std::make_shared<GraphComposer>("sequenced");
    inner->registerLauncher(buildIncLauncher(sX.cuda(), bufX, N), "X");
    inner->registerLauncher(buildIncLauncher(sY.cuda(), bufY, N), "Y");
    inner->build();

    GraphComposer outer("sequenced");
    outer.registerNested(inner, "inner");
    outer.registerLauncher(buildIncLauncher(sZ.cuda(), bufZ, N), "Z");
    outer.build();
    outer.launch(REPS);

    expectAll(bufX, double(REPS));
    expectAll(bufY, double(REPS));
    expectAll(bufZ, double(REPS));

    /* The nested composer's OWN fired history also recorded the replay
     * (it is a real, independent GraphComposer -- its bookkeeping does
     * not depend on being nested). */
    ASSERT_EQ(inner->firedHistory().size(), 1u);
    EXPECT_EQ(inner->firedHistory()[0], (std::vector<idx_t>{ 0, 1 }));
}

/* ================================================================
 * 3. mode="enabled": capture-once, output tracks flags -- toggling
 *    via setRouting() changes replay output with NO recapture (build()
 *    may only run once; a second capture would throw).
 * ================================================================ */
TEST_F(GraphComposerFixture, EnabledTogglesCaptureOnceOutputTracksFlags)
{
    double* bufA = alloc();
    double* bufB = alloc();
    double* bufC = alloc();

    GraphComposer composer("enabled");
    composer.registerCallable(
        [bufA](cudaStream_t s) { incKernel<<<1, N, 0, s>>>(bufA, N); }, "A");
    composer.registerCallable(
        [bufB](cudaStream_t s) { incKernel<<<1, N, 0, s>>>(bufB, N); }, "B");
    composer.registerCallable(
        [bufC](cudaStream_t s) { incKernel<<<1, N, 0, s>>>(bufC, N); }, "C");
    composer.build(); // ONE capture; a second build() would throw

    composer.launch(1);
    expectAll(bufA, 1.0);
    expectAll(bufB, 1.0);
    expectAll(bufC, 1.0);

    composer.setRouting({ true, false, true });
    composer.launch(1);
    expectAll(bufA, 2.0); // toggled on: fired again
    expectAll(bufB, 1.0); // toggled OFF: untouched by this replay
    expectAll(bufC, 2.0);

    EXPECT_THROW(composer.build(), std::runtime_error)
        << "mode=\"enabled\" captures exactly once";

    ASSERT_EQ(composer.firedHistory().size(), 2u);
    EXPECT_EQ(composer.firedHistory()[1], (std::vector<idx_t>{ 0, 2 }));
}

/* ================================================================
 * 4. mode="rebuild": every setRouting() call recaptures from scratch,
 *    containing ONLY the active members -- an inactive member's
 *    buffer is untouched because its work was never even captured
 *    this time, not merely disabled.
 * ================================================================ */
TEST_F(GraphComposerFixture, RebuildRecapturesOnRoutingChange)
{
    double* bufA = alloc();
    double* bufB = alloc();

    GraphComposer composer("rebuild");
    composer.registerCallable(
        [bufA](cudaStream_t s) { incKernel<<<1, N, 0, s>>>(bufA, N); }, "A");
    composer.registerCallable(
        [bufB](cudaStream_t s) { incKernel<<<1, N, 0, s>>>(bufB, N); }, "B");
    composer.build();

    composer.launch(1);
    expectAll(bufA, 1.0);
    expectAll(bufB, 1.0);

    composer.setRouting({ true, false }); // recapture: only A is present now
    composer.launch(1);
    expectAll(bufA, 2.0);
    expectAll(bufB, 1.0); // never recaptured this round -- untouched

    composer.setRouting({ false, false }); // recapture: empty -- launch is a no-op
    EXPECT_NO_THROW(composer.launch(1));
    expectAll(bufA, 2.0);
    expectAll(bufB, 1.0);
}

/* ================================================================
 * 5. Rejections -- the pinned set: non-sequenced nesting (both
 *    directions), duplicate member, launch-before-build, and the
 *    routing flag-length contract.
 * ================================================================ */
TEST_F(GraphComposerFixture, RejectsNestingUnderNonSequencedOuter)
{
    auto inner = std::make_shared<GraphComposer>("sequenced");
    inner->registerCallable([](cudaStream_t) { });
    inner->build();

    GraphComposer flatOuter("enabled");
    EXPECT_THROW(flatOuter.registerNested(inner), std::invalid_argument);
}

TEST_F(GraphComposerFixture, RejectsNestingNonSequencedOrUnbuiltInner)
{
    GraphComposer outer("sequenced");

    auto flatInner = std::make_shared<GraphComposer>("enabled");
    EXPECT_THROW(outer.registerNested(flatInner), std::invalid_argument)
        << "the nested composer must itself be mode=\"sequenced\"";

    auto unbuiltInner = std::make_shared<GraphComposer>("sequenced");
    EXPECT_THROW(outer.registerNested(unbuiltInner), std::invalid_argument)
        << "the nested composer must already be build()-ed";
}

TEST_F(GraphComposerFixture, RejectsDuplicateMember)
{
    GraphComposer outer("sequenced");
    auto inner = std::make_shared<GraphComposer>("sequenced");
    inner->registerCallable([](cudaStream_t) { });
    inner->build();

    outer.registerNested(inner, "dup");
    EXPECT_THROW(outer.registerNested(inner, "dup2"), std::invalid_argument)
        << "the same nested composer may not be registered twice";
    EXPECT_THROW(
        outer.registerCallable([](cudaStream_t) { }, "dup"),
        std::invalid_argument)
        << "duplicate member name";
}

TEST_F(GraphComposerFixture, RejectsLaunchBeforeBuild)
{
    GraphComposer composer("sequenced");
    composer.registerCallable([](cudaStream_t) { });
    EXPECT_THROW(composer.launch(), std::runtime_error);
}

TEST_F(GraphComposerFixture, RejectsRoutingLengthMismatch)
{
    GraphComposer composer("sequenced");
    composer.registerCallable([](cudaStream_t) { });
    composer.registerCallable([](cudaStream_t) { });
    composer.build();
    EXPECT_THROW(composer.setRouting({ true }), std::invalid_argument);
}

TEST_F(GraphComposerFixture, RejectsLauncherAndPreLaunchUnderFlatModes)
{
    double* buf = alloc();
    GraphComposer flat("enabled");
    EXPECT_THROW(
        flat.registerLauncher(buildIncLauncher(0, buf, N)), std::invalid_argument)
        << "a built Launcher member is sequenced-only";
    EXPECT_THROW(
        flat.registerCallable([](cudaStream_t) { }, {}, []() { }),
        std::invalid_argument)
        << "pre_launch is sequenced-only";
}

} // namespace GraphComposerTest
} // namespace eagle_tests

#endif
