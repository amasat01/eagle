// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Lifetime regression tests: every test builds a Graph in an inner
 * scope, moves the Launcher out, exits the scope, and only THEN
 * exercises the Launcher. The shared_ptr<Graph::Storage> lifetime
 * model must keep node handles and kernelParams pointers valid for
 * setLogicalSize / launch even after the producing Graph dies.
 *
 * These would have flagged a real downstream regression at eagle CI
 * level. */
#include "TestBase.h"

#include "eagle/cuda.h"

#ifndef EAGLE_CPU_ONLY

#include <set>

namespace eagle_tests {
namespace GraphLifetimeTest {

using namespace eagle;
using namespace eagle::cuda;

/* ================================================================
 * Helper kernels
 * ================================================================ */
__global__ void writeKernelLT(int* buf, int offset, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        buf[tid] = tid + offset;
}

__global__ void writeKernelLT2(int* buf, int offset, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        buf[tid] = tid * 2 + offset;
}

/* Build an OWNED captured cudaGraph_t with a single kernel write.
 * The returned handle is owned by the caller (use to wrap in
 * CapturedGraph or pass to addNode(cudaGraph_t) and destroy
 * manually). */
static cudaGraph_t captureSingleKernel_(
    const cudaStream_t& stream, int* buf, int offset, int n, int blockDim)
{
    StreamCapturer capturer(stream);
    capturer.begin();
    const int gridDim = (n + blockDim - 1) / blockDim;
    writeKernelLT<<<gridDim, blockDim, 0, stream>>>(buf, offset, n);
    return capturer.end();
}

/* Build an OWNED captured cudaGraph_t with two distinct kernels. */
static cudaGraph_t captureTwoKernels_(
    const cudaStream_t& stream, int* buf, int offset, int n, int blockDim)
{
    StreamCapturer capturer(stream);
    capturer.begin();
    const int gridDim = (n + blockDim - 1) / blockDim;
    writeKernelLT<<<gridDim, blockDim, 0, stream>>>(buf, offset, n);
    writeKernelLT2<<<gridDim, blockDim, 0, stream>>>(buf, offset, n);
    return capturer.end();
}

class GraphLifetimeFixture : public Test {
protected:
    static constexpr int N = 64;
    int* d_buf             = nullptr;
    std::vector<int> h_buf;

    void SetUp() override
    {
        h_buf.assign(N, -1);
        EAGLE_CHECK_ALWAYS(cudaMalloc(&d_buf, N * sizeof(int)));
        EAGLE_CHECK_ALWAYS(cudaMemset(d_buf, 0, N * sizeof(int)));
        // The memset rides the per-thread default stream, which has no ordering
        // with the eagle Streams the test then writes on: finish it first.
        EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    }
    void TearDown() override
    {
        if (d_buf)
            EAGLE_CHECK_ALWAYS(cudaFree(d_buf));
    }
    void downloadBuf()
    {
        EAGLE_CHECK_ALWAYS(cudaMemcpy(h_buf.data(), d_buf, N * sizeof(int),
            cudaMemcpyDeviceToHost));
    }
};

/* ================================================================
 * 1. Source Graph destroyed before launch — top-level kernel node.
 *    addKernelNode places kernels at the top of storage_->graph;
 *    Launcher must keep that storage alive via shared_ptr.
 * ================================================================ */
TEST_F(GraphLifetimeFixture, SourceGraphDestroyed_TopLevelKernel_LaunchOK)
{
    Stream s;
    Launcher launcher;
    {
        Graph g;
        g.stream(s.cuda());
        cudaKernelNodeParams params {};
        void* args[]      = { (void*)&d_buf, (void*)&N };
        params.func           = (void*)writeKernelLT;
        params.gridDim        = dim3(1, 1, 1);
        params.blockDim       = dim3(N, 1, 1);
        params.sharedMemBytes = 0;
        params.kernelParams   = args;
        params.extra          = nullptr;
        const int offset      = 7;
        void* args2[]     = { (void*)&d_buf, (void*)&offset, (void*)&N };
        params.kernelParams   = args2;
        g.addKernelNode(params);
        launcher = g.launcher();
    } /* g dies here; Launcher must hold storage alive */

    launcher.launch();
    launcher.synchronize();
    downloadBuf();
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(h_buf[i], i + 7);
}

/* ================================================================
 * 2. Source Graph destroyed before setLogicalSize — top-level
 *    kernel. The failure mode this guards: setLogicalSize patches
 *    the exec graph using rec.node + rec.capturedParams.kernelParams
 *    that point into source_->graph internals. Must remain valid.
 * ================================================================ */
TEST_F(GraphLifetimeFixture, SourceGraphDestroyed_TopLevelKernel_SetLogicalSizeOK)
{
    Stream s;
    Launcher launcher;
    const int offset = 11;
    void* args[]     = { (void*)&d_buf, (void*)&offset, (void*)&N };
    {
        Graph g;
        g.stream(s.cuda());
        cudaKernelNodeParams params {};
        params.func           = (void*)writeKernelLT;
        params.gridDim        = dim3(1, 1, 1);
        params.blockDim       = dim3(32, 1, 1);
        params.sharedMemBytes = 0;
        params.kernelParams   = args;
        params.extra          = nullptr;
        g.addKernelNode(params, std::initializer_list<idx_t>{},
            /*idealBlockSize=*/0); /* grid-only patch */
        launcher = g.launcher();
    }

    /* Patch logical size after Graph death → exercises
     * cudaGraphExecKernelNodeSetParams against handles that point
     * into the now-Launcher-owned shared Storage. */
    EXPECT_NO_FATAL_FAILURE(launcher.setLogicalSize(N));
    launcher.launch();
    launcher.synchronize();
    downloadBuf();
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(h_buf[i], i + 11);
}

/* ================================================================
 * 3. Source Graph destroyed before setLogicalSize — captured-child
 *    kernel. The exact path a Slice scan/scatter hits via
 *    + addNode(CapturedGraph). Must work after Graph dies.
 * ================================================================ */
TEST_F(GraphLifetimeFixture, SourceGraphDestroyed_CapturedChildKernel_SetLogicalSizeOK)
{
    Stream s;
    Launcher launcher;
    {
        Graph g;
        g.stream(s.cuda());
        CapturedGraph cap{
            captureSingleKernel_(s.cuda(), d_buf, 13, N, /*blockDim=*/32),
            std::vector<idx_t>{ 0 } /* grid-only */
        };
        g.addNode(std::move(cap));
        launcher = g.launcher();
    } /* g dies; cap was already consumed inside addNode */

    EXPECT_NO_FATAL_FAILURE(launcher.setLogicalSize(N));
    launcher.launch();
    launcher.synchronize();
    downloadBuf();
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(h_buf[i], i + 13);
}

/* ================================================================
 * 4. Mixed: top-level kernel + captured child, Graph dies, both
 *    kernels patchable.
 * ================================================================ */
TEST_F(GraphLifetimeFixture, SourceGraphDestroyed_MixedTopLevelAndCaptured)
{
    Stream s;
    Launcher launcher;
    const int offset1 = 3;
    void* args[] = { (void*)&d_buf, (void*)&offset1, (void*)&N };
    {
        Graph g;
        g.stream(s.cuda());
        /* Top-level kernel first */
        cudaKernelNodeParams params {};
        params.func           = (void*)writeKernelLT;
        params.gridDim        = dim3(1, 1, 1);
        params.blockDim       = dim3(N, 1, 1);
        params.sharedMemBytes = 0;
        params.kernelParams   = args;
        params.extra          = nullptr;
        g.addKernelNode(params);
        /* Then captured child (overwrites with offset 21) */
        g.addNode(CapturedGraph{
            captureSingleKernel_(s.cuda(), d_buf, 21, N, 32),
            std::vector<idx_t>{ 0 } });
        launcher = g.launcher();
    }

    ASSERT_EQ(launcher.kernelNodeCount(), 2);
    EXPECT_NO_FATAL_FAILURE(launcher.setLogicalSize(N));
    launcher.launch();
    launcher.synchronize();
    downloadBuf();
    /* Captured child runs after the top-level kernel → buffer
     * shows the second offset (21). */
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(h_buf[i], i + 21);
}

/* ================================================================
 * 5. Multiple captured children with distinct per-kernel caps,
 *    Graph dies, each launcher kernel keeps its own cap.
 * ================================================================ */
TEST_F(GraphLifetimeFixture, SourceGraphDestroyed_MultipleCapturedChildren_PerKernelCaps)
{
    Stream s;
    Launcher launcher;
    {
        Graph g;
        g.stream(s.cuda());
        g.addNode(CapturedGraph{
            captureSingleKernel_(s.cuda(), d_buf, 31, N, 32),
            std::vector<idx_t>{ 64 } });
        g.addNode(CapturedGraph{
            captureSingleKernel_(s.cuda(), d_buf, 41, N, 32),
            std::vector<idx_t>{ 256 } });
        launcher = g.launcher();
    }

    ASSERT_EQ(launcher.kernelNodeCount(), 2);
    /* Caps survived the Graph drop. */
    EXPECT_EQ(launcher.kernelNodes()[0].idealBlockSize, 64);
    EXPECT_EQ(launcher.kernelNodes()[1].idealBlockSize, 256);

    EXPECT_NO_FATAL_FAILURE(launcher.setLogicalSize(N));
    launcher.launch();
    launcher.synchronize();
    downloadBuf();
    /* Second child overwrites first → offset 41 is final. */
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(h_buf[i], i + 41);
}

/* ================================================================
 * 6. Moved Launcher after source Graph destroyed.
 *    a = g.launcher(); b = std::move(a); drop g; exercise b.
 *    Verifies move-ops carry the shared_ptr correctly.
 * ================================================================ */
TEST_F(GraphLifetimeFixture, MovedLauncher_AfterSourceDestroyed)
{
    Stream s;
    Launcher b;
    {
        Graph g;
        g.stream(s.cuda());
        g.addNode(CapturedGraph{
            captureSingleKernel_(s.cuda(), d_buf, 47, N, 32),
            std::vector<idx_t>{ 0 } });
        Launcher a = g.launcher();
        b = std::move(a);
        /* a is moved-from; both g and a die at the end of this scope. */
    }

    EXPECT_NO_FATAL_FAILURE(b.setLogicalSize(N));
    b.launch();
    b.synchronize();
    downloadBuf();
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(h_buf[i], i + 47);
}

/* ================================================================
 * 7. Source Graph destroyed; addHostNode lambda must still fire.
 *    HostCallback closure lives in storage_->hostNodeData — must
 *    survive Graph death via shared_ptr.
 * ================================================================ */
TEST_F(GraphLifetimeFixture, SourceGraphDestroyed_HostNodeUserDataAlive)
{
    Stream s;
    int counter = 0;
    Launcher launcher;
    {
        Graph g;
        g.stream(s.cuda());
        g.addHostNode([&counter]() { counter++; });
        launcher = g.launcher();
    } /* g dies; the captured lambda must still be safe to invoke */

    launcher.launch();
    launcher.synchronize();
    EXPECT_EQ(counter, 1);
}

/* ================================================================
 * 8. eagle Integrator-style two-snapshot: two Launchers from the
 *    same Graph, both outlive the Graph. Both must work, no
 *    double-free on Storage.
 * ================================================================ */
TEST_F(GraphLifetimeFixture, MultipleLaunchersOutliveSource)
{
    Stream s;
    Launcher la, lb;
    {
        Graph g;
        g.stream(s.cuda());
        g.addNode(CapturedGraph{
            captureSingleKernel_(s.cuda(), d_buf, 53, N, 32),
            std::vector<idx_t>{ 0 } });
        la = g.launcher();
        /* Add another node, then snapshot again — Integrator
         * pattern. */
        g.addNode(CapturedGraph{
            captureSingleKernel_(s.cuda(), d_buf, 59, N, 32),
            std::vector<idx_t>{ 0 } });
        lb = g.launcher();
    }

    /* la has 1 kernel; lb has 2. */
    EXPECT_EQ(la.kernelNodeCount(), 1);
    EXPECT_EQ(lb.kernelNodeCount(), 2);

    /* la's exec graph contains only the first child → buffer shows 53. */
    la.launch();
    la.synchronize();
    downloadBuf();
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(h_buf[i], i + 53);

    /* lb's exec graph contains both children → final buffer shows 59. */
    lb.launch();
    lb.synchronize();
    downloadBuf();
    for (int i = 0; i < N; ++i)
        EXPECT_EQ(h_buf[i], i + 59);
}

/* Recursively walk a cudaGraph_t (top-level + child sub-graphs)
 * and collect every node handle. */
static void collectAllNodes_(cudaGraph_t g, std::set<cudaGraphNode_t>& out)
{
    if (g == nullptr)
        return;
    size_t n = 0;
    EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(g, nullptr, &n));
    if (n == 0)
        return;
    std::vector<cudaGraphNode_t> nodes(n);
    EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(g, nodes.data(), &n));
    for (cudaGraphNode_t node : nodes) {
        out.insert(node);
        cudaGraphNodeType ty;
        EAGLE_CHECK_ALWAYS(cudaGraphNodeGetType(node, &ty));
        if (ty == cudaGraphNodeTypeGraph) {
            cudaGraph_t child = nullptr;
            EAGLE_CHECK_ALWAYS(cudaGraphChildGraphNodeGetGraph(node, &child));
            collectAllNodes_(child, out);
        }
    }
}

/* ================================================================
 * 9. Structural guard: every harvested kernelNodes_[i].node is
 *    reachable from the source graph (top-level + child sub-graphs).
 *    Catches future harvest regressions where the wrong graph is
 *    queried.
 * ================================================================ */
TEST_F(GraphLifetimeFixture, HarvestedNodeHandle_BelongsToInstantiatedGraph)
{
    Stream s;
    Launcher launcher;
    {
        Graph g;
        g.stream(s.cuda());
        g.addNode(CapturedGraph{
            captureTwoKernels_(s.cuda(), d_buf, 67, N, 32),
            std::vector<idx_t>{ 0, 0 } });
        g.addNode(CapturedGraph{
            captureSingleKernel_(s.cuda(), d_buf, 71, N, 32),
            std::vector<idx_t>{ 0 } });
        launcher = g.launcher();
    } /* graph dies; storage alive via launcher's shared_ptr */

    ASSERT_EQ(launcher.kernelNodeCount(), 3);

    std::set<cudaGraphNode_t> reachable;
    collectAllNodes_(launcher.graph(), reachable);

    for (const KernelNodeRecord& rec : launcher.kernelNodes()) {
        EXPECT_NE(reachable.find(rec.node), reachable.end())
            << "harvested kernel node not reachable from source graph";
    }
}

} // namespace GraphLifetimeTest
} // namespace eagle_tests

#endif
