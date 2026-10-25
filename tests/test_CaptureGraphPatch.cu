// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Graph patching (Launcher::setKernelParams / setMemcpySource / tryExecUpdate),
// cross-branch fork waits, the generalised loop tail, the negated guard, the
// decision-cell copy and the IF/ELSE + SWITCH arms, on eagle's capture weaves.

#include "TestBase.h"

#include "eagle/cuda.h"
#include "eagle/cuda/CaptureAttribution.h"
#include "eagle/cuda/CaptureConditional.h"
#include "eagle/cuda/CaptureFork.h"
#include "eagle/cuda/Launcher.h"
#include "eagle/cuda/detail/RuntimeCompat.h"

#ifndef EAGLE_CPU_ONLY

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace eagle_tests {
namespace CaptureGraphPatchTest {

using namespace eagle;
using namespace eagle::cuda;
using eagle::conditional::kLoopUncapped;
using eagle::conditional::LoopTerms;

__global__ void writeKernel(unsigned int* dst, const unsigned int* src) { *dst = *src; }
__global__ void accumKernel(unsigned int* dst, const unsigned int* src) { *dst += *src; }
__global__ void setKernel(unsigned int* dst, unsigned int v) { *dst = v; }
__global__ void decKernel(unsigned int* p) { --*p; }
// Body of a loop: records the device index cell and raises the break cell at `at`.
__global__ void indexKernel(const unsigned int* cell, unsigned int* out, unsigned int* brk, unsigned int at)
{
    out[cell[1]] = cell[1] + 1u;
    if (brk != nullptr && cell[1] == at)
        *brk = 1u;
}

/** Device words, zeroed; freed with the fixture. */
struct Cells {
    unsigned int* p = nullptr;
    explicit Cells(std::size_t n)
    {
        EAGLE_CHECK_ALWAYS(cudaMalloc(&p, n * sizeof(unsigned int)));
        EAGLE_CHECK_ALWAYS(cudaMemset(p, 0, n * sizeof(unsigned int)));
    }
    ~Cells() { cudaFree(p); }
    unsigned int* operator+(std::size_t i) const { return p + i; }
    void set(std::size_t i, unsigned int v) const
    {
        EAGLE_CHECK_ALWAYS(cudaMemcpy(p + i, &v, sizeof v, cudaMemcpyHostToDevice));
    }
    unsigned int get(std::size_t i) const
    {
        unsigned int v = 0;
        EAGLE_CHECK_ALWAYS(cudaMemcpy(&v, p + i, sizeof v, cudaMemcpyDeviceToHost));
        return v;
    }
};

/** Nodes added to the capture seen from @p s since @p before. */
std::vector<cudaGraphNode_t> added(cudaStream_t s, const std::vector<cudaGraphNode_t>& before)
{
    std::vector<cudaGraphNode_t> out;
    for (cudaGraphNode_t n : captureSnapshotNodes(s))
        if (std::find(before.begin(), before.end(), n) == before.end())
            out.push_back(n);
    return out;
}

/** Instantiate @p graph through eagle's Graph and replay it once. */
struct Built {
    Graph graph;
    Launcher launcher;
    explicit Built(cudaGraph_t g)
        : graph(CapturedGraph(g))
        , launcher(graph.launcher())
    {
    }
    void run()
    {
        launcher.launch();
        launcher.synchronize();
    }
};

class GraphPatch : public Test { };

// A root node, a fork-branch node, a WHILE-body node and a memcpy node, each
// re-pointed on the exec: every output follows the new source, no re-capture.
// RED: drop the `args[i] = params[i]` loop in setKernelParams -> outputs stay 5.
TEST_F(GraphPatch, PatchRootForkBranchAndWhileBodyNodes)
{
    Cells c(16); // 0 srcA, 1 srcB, 2 root, 3 br0, 4 br1, 5 loop acc, 6 live, 7..8 cell, 9 mcpy dst
    c.set(0, 5);
    c.set(1, 7);
    Stream origin;
    CaptureFork fork(origin.cuda(), 2);
    CaptureConditional loop(origin.cuda(), CountGuard{ c + 6, nullptr }, 3u, c + 7);
    c.set(6, 1);

    StreamCapturer capturer(origin.cuda());
    EXPECT_EQ(capturer.graph(), nullptr);
    capturer.begin();
    EXPECT_NE(capturer.graph(), nullptr);
    auto before = captureSnapshotNodes(origin.cuda());
    writeKernel<<<1, 1, 0, origin.cuda()>>>(c + 2, c + 0);
    const auto rootNodes = added(origin.cuda(), before);
    before = captureSnapshotNodes(origin.cuda());
    EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(c + 9, c + 0, sizeof(unsigned int), cudaMemcpyDefault, origin.cuda()));
    const auto copyNodes = added(origin.cuda(), before);
    fork.fork();
    writeKernel<<<1, 1, 0, fork.branch(0)>>>(c + 3, c + 0);
    before = captureSnapshotNodes(fork.branch(1));
    writeKernel<<<1, 1, 0, fork.branch(1)>>>(c + 4, c + 0);
    const auto branchNodes = added(fork.branch(1), before);
    fork.join();
    cudaStream_t body = loop.begin();
    accumKernel<<<1, 1, 0, body>>>(c + 5, c + 0);
    loop.end();
    Built built(capturer.end());
    EXPECT_EQ(capturer.graph(), nullptr);

    ASSERT_EQ(rootNodes.size(), 1u);
    ASSERT_EQ(copyNodes.size(), 1u);
    ASSERT_EQ(branchNodes.size(), 1u);
    const auto bodyNodes   = loop.bodyNodes(0);
    const auto setterNodes = loop.setterNodes();
    ASSERT_EQ(bodyNodes.size(), 2u);   // accum + eagle's tail
    ASSERT_EQ(setterNodes.size(), 2u); // head + tail
    EXPECT_NE(std::find(bodyNodes.begin(), bodyNodes.end(), setterNodes[1]), bodyNodes.end());
    const cudaGraphNode_t accumNode = bodyNodes[0] == setterNodes[1] ? bodyNodes[1] : bodyNodes[0];

    built.run();
    EXPECT_EQ(c.get(2), 5u);
    EXPECT_EQ(c.get(4), 5u);
    EXPECT_EQ(c.get(5), 15u);
    EXPECT_EQ(c.get(9), 5u);

    const unsigned int* srcB = c + 1;
    void* params[] = { nullptr, (void*)&srcB }; // keep dst, re-point src
    for (cudaGraphNode_t n : { rootNodes[0], branchNodes[0], accumNode })
        built.launcher.setKernelParams(n, params, 2);
    built.launcher.setMemcpySource(copyNodes[0], c + 1);
    c.set(5, 0);
    built.run();
    EXPECT_EQ(c.get(2), 7u);
    EXPECT_EQ(c.get(3), 5u) << "an unpatched node keeps its captured arguments";
    EXPECT_EQ(c.get(4), 7u);
    EXPECT_EQ(c.get(5), 21u);
    EXPECT_EQ(c.get(9), 7u);
    EXPECT_EQ(c.get(0), 5u) << "the old source is untouched";

    void* tooMany[] = { nullptr, nullptr, nullptr };
    EXPECT_THROW(built.launcher.setKernelParams(rootNodes[0], tooMany, 3), std::invalid_argument);
}

// A same-topology re-capture updates the exec in place; another shape is refused
// and leaves the exec working.
TEST_F(GraphPatch, ExecUpdateAcceptedAndRefused)
{
    Cells c(4);
    c.set(0, 5);
    c.set(1, 7);
    Stream s;
    auto capture = [&](int kernels, const unsigned int* src) {
        StreamCapturer capturer(s.cuda());
        capturer.begin();
        for (int k = 0; k < kernels; ++k)
            writeKernel<<<1, 1, 0, s.cuda()>>>(c + 2, src);
        return capturer.end();
    };
    Built built(capture(1, c + 0));
    built.run();
    EXPECT_EQ(c.get(2), 5u);

    CapturedGraph same(capture(1, c + 1));
    EXPECT_TRUE(built.launcher.tryExecUpdate(same.graph()));
    built.run();
    EXPECT_EQ(c.get(2), 7u);

    CapturedGraph other(capture(2, c + 0));
    EXPECT_FALSE(built.launcher.tryExecUpdate(other.graph()));
    c.set(2, 0);
    built.run();
    EXPECT_EQ(c.get(2), 7u) << "a refused update leaves the exec as it was";
}

// mark/wait is ONE captured edge and no node; the waiter reads the producer's write.
TEST_F(GraphPatch, ForkMarkWaitAddsOneEdge)
{
    Cells c(4);
    c.set(0, 9);
    auto build = [&](bool withWait, std::size_t& nodes, std::size_t& edges) {
        Stream origin;
        CaptureFork fork(origin.cuda(), 2, withWait ? 1 : 0);
        StreamCapturer capturer(origin.cuda());
        capturer.begin();
        fork.fork();
        writeKernel<<<1, 1, 0, fork.branch(0)>>>(c + 1, c + 0);
        if (withWait)
            fork.wait(1, fork.mark(0));
        writeKernel<<<1, 1, 0, fork.branch(1)>>>(c + 2, c + 1);
        if (withWait)
            EXPECT_THROW(fork.mark(0), std::out_of_range);
        fork.join();
        cudaGraph_t g = capturer.end();
        EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(g, nullptr, &nodes));
        EAGLE_CHECK_ALWAYS(detail::graphGetEdges(g, nullptr, nullptr, &edges));
        return g;
    };
    std::size_t n0 = 0, e0 = 0, n1 = 0, e1 = 0;
    CapturedGraph plain(build(false, n0, e0));
    Built built(build(true, n1, e1));
    EXPECT_EQ(n1, n0);
    EXPECT_EQ(e1, e0 + 1);
    built.run();
    EXPECT_EQ(c.get(2), 9u);
}

/** Capture one WHILE weave with @p guard / @p terms around @p body. */
template <class Body>
cudaGraph_t captureLoop(cudaStream_t s, CountGuard guard, unsigned int negate, LoopTerms terms, Body body)
{
    CaptureConditional loop(s, CondKind::While, guard, 0, negate, terms);
    StreamCapturer capturer(s);
    capturer.begin();
    body(loop.begin());
    loop.end();
    return capturer.end();
}

// One tail for every loop form: counted by a runtime trip count with the index
// cell, a mid-body break, and an uncapped while.
TEST_F(GraphPatch, LoopTailCountedBreakUncapped)
{
    Cells c(64); // 0..1 cell, 2 limit, 3 brk, 4 live, 8.. out
    Stream s;
    {   // counted: for i in range(*limit), no guard, no cap
        Built built(captureLoop(s.cuda(), CountGuard{ nullptr, nullptr }, 0,
            LoopTerms{ kLoopUncapped, c + 0, c + 2, nullptr },
            [&](cudaStream_t b) { indexKernel<<<1, 1, 0, b>>>(c + 0, c + 8, nullptr, 0); }));
        for (unsigned int n : { 7u, 0u, 3u }) {
            c.set(2, n);
            built.run();
            EXPECT_EQ(c.get(1), n) << "ran == the runtime trip count";
        }
        for (unsigned int i = 0; i < 7; ++i)
            EXPECT_EQ(c.get(8 + i), i + 1) << "the index cell is i in iteration i";
    }
    {   // break at i == 4 under cap 100; the head clears a stale break cell
        Built built(captureLoop(s.cuda(), CountGuard{ nullptr, nullptr }, 0,
            LoopTerms{ 100u, c + 0, nullptr, c + 3 },
            [&](cudaStream_t b) { indexKernel<<<1, 1, 0, b>>>(c + 0, c + 8, c + 3, 4); }));
        for (int rep = 0; rep < 2; ++rep) {
            built.run();
            EXPECT_EQ(c.get(1), 5u);
            EXPECT_EQ(c.get(3), 1u);
        }
    }
    {   // uncapped while live: the guard alone ends it
        Built built(captureLoop(s.cuda(), CountGuard{ c + 4, nullptr }, 0,
            LoopTerms{ kLoopUncapped, c + 0, nullptr, nullptr },
            [&](cudaStream_t b) { decKernel<<<1, 1, 0, b>>>(c + 4); }));
        c.set(4, 300);
        built.run();
        EXPECT_EQ(c.get(1), 300u);
        EXPECT_EQ(c.get(0), kLoopUncapped - 300u);
    }
}

// negate inverts the guard: IF runs on a zero cell, WHILE runs until the cell is set.
TEST_F(GraphPatch, NegatedGuard)
{
    Cells c(8); // 0 flag, 1 tracer, 2..3 cell
    Stream s;
    CaptureConditional weave(s.cuda(), CondKind::If, CountGuard{ c + 0, nullptr }, 0, 1);
    StreamCapturer capturer(s.cuda());
    capturer.begin();
    setKernel<<<1, 1, 0, weave.begin()>>>(c + 1, 1);
    weave.end();
    Built ifNot(capturer.end());
    c.set(0, 0);
    ifNot.run();
    EXPECT_EQ(c.get(1), 1u);
    c.set(0, 1);
    c.set(1, 0);
    ifNot.run();
    EXPECT_EQ(c.get(1), 0u);

    // while not done: the body sets done in iteration 2 -> 3 iterations
    Built whileNot(captureLoop(s.cuda(), CountGuard{ c + 0, nullptr }, 1,
        LoopTerms{ 50u, c + 2, nullptr, nullptr },
        [&](cudaStream_t b) { indexKernel<<<1, 1, 0, b>>>(c + 2, c + 4, c + 0, 2); }));
    c.set(0, 0);
    whileNot.run();
    EXPECT_EQ(c.get(3), 3u);
}

// `if c1: A (clears c1) elif c2: B else: C` as negated IFs: reading the live cell
// runs A then C; one decision copy at the chain head runs exactly A.
TEST_F(GraphPatch, DecisionCopyFixesElifArmWritingItsGuard)
{
    Cells c(8); // 0 c1, 1 c2, 2 tracer, 3 d1, 4 d2
    Stream s;
    auto chain = [&](bool decisionCopy) {
        const unsigned int* g1 = decisionCopy ? c + 3 : c + 0;
        const unsigned int* g2 = decisionCopy ? c + 4 : c + 1;
        CaptureConditional ifA(s.cuda(), CondKind::If, CountGuard{ g1, nullptr });
        CaptureConditional orElse(s.cuda(), CondKind::If, CountGuard{ g1, nullptr }, 0, 1);
        CaptureConditional ifB(orElse.bodyStream(), CondKind::If, CountGuard{ g2, nullptr });
        CaptureConditional ifC(orElse.bodyStream(), CondKind::If, CountGuard{ g2, nullptr }, 0, 1);
        StreamCapturer capturer(s.cuda());
        capturer.begin();
        setKernel<<<1, 1, 0, s.cuda()>>>(c + 2, 0);
        if (decisionCopy)
            EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(c + 3, c + 0, 2 * sizeof(unsigned int), cudaMemcpyDefault, s.cuda()));
        cudaStream_t a = ifA.begin();
        setKernel<<<1, 1, 0, a>>>(c + 2, 1);
        setKernel<<<1, 1, 0, a>>>(c + 0, 0); // arm A clears its own guard
        ifA.end();
        orElse.begin();
        setKernel<<<1, 1, 0, ifB.begin()>>>(c + 2, 2);
        ifB.end();
        setKernel<<<1, 1, 0, ifC.begin()>>>(c + 2, 3);
        ifC.end();
        orElse.end();
        Built built(capturer.end());
        int right = 0;
        for (unsigned int rep = 0; rep < 20; ++rep) {
            c.set(0, 1);
            c.set(1, rep & 1u);
            built.run();
            right += c.get(2) == 1u;
        }
        return right;
    };
    EXPECT_EQ(chain(false), 0) << "the live-cell lowering runs both arms";
    EXPECT_EQ(chain(true), 20);
}

// Below driver 12.8 IF/ELSE and SWITCH are refused at construction, before any capture.
TEST_F(GraphPatch, ArmsRefusedBelowDriver1280)
{
    int version = 0;
    EAGLE_CHECK_ALWAYS(cudaDriverGetVersion(&version));
    EXPECT_EQ(conditionalArmsSupported(), version >= kConditionalArmsDriver);
    if (conditionalArmsSupported())
        GTEST_SKIP() << "driver " << version << " provides IF/ELSE and SWITCH";
    Cells c(1);
    Stream s;
    EXPECT_THROW(CaptureConditional(s.cuda(), CondKind::IfElse, CountGuard{ c + 0, nullptr }), std::runtime_error);
    EXPECT_THROW(CaptureConditional(s.cuda(), CondKind::Switch, CountGuard{ c + 0, nullptr }, 3), std::runtime_error);
    EXPECT_THROW(CaptureConditional(s.cuda(), CondKind::If, CountGuard{ c + 0, nullptr }, 2), std::invalid_argument);
}

// IF/ELSE runs exactly one arm; SWITCH runs arm *count (none past the last).
// Runs on a driver >= 12.8 only.
TEST_F(GraphPatch, IfElseAndSwitchArms)
{
    if (!conditionalArmsSupported())
        GTEST_SKIP() << "IF/ELSE and SWITCH need driver >= 12.8";
    Cells c(4); // 0 guard / index, 1 tracer
    Stream s;
    auto build = [&](CondKind kind, unsigned int arms) {
        CaptureConditional weave(s.cuda(), kind, CountGuard{ c + 0, nullptr }, arms);
        StreamCapturer capturer(s.cuda());
        capturer.begin();
        setKernel<<<1, 1, 0, s.cuda()>>>(c + 1, 99);
        weave.begin();
        for (unsigned int k = 0; k < weave.arms(); ++k)
            setKernel<<<1, 1, 0, weave.armStream(k)>>>(c + 1, 10 + k);
        weave.end();
        return capturer.end();
    };
    Built ifElse(build(CondKind::IfElse, 0));
    c.set(0, 1);
    ifElse.run();
    EXPECT_EQ(c.get(1), 10u);
    c.set(0, 0);
    ifElse.run();
    EXPECT_EQ(c.get(1), 11u);

    Built sw(build(CondKind::Switch, 3));
    for (unsigned int k : { 0u, 2u, 1u }) {
        c.set(0, k);
        sw.run();
        EXPECT_EQ(c.get(1), 10u + k);
    }
    c.set(0, 3);
    sw.run();
    EXPECT_EQ(c.get(1), 99u);
}

} // namespace CaptureGraphPatchTest
} // namespace eagle_tests

#endif // EAGLE_CPU_ONLY
