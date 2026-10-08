// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Tests for per-kernel ``idealBlockSize`` preservation in
 * ``eagle::cuda::Graph``. */
#include "TestBase.h"

#include "eagle/cuda.h"
#include "eagle/cuda/CapturedGraph.h"

#ifndef EAGLE_CPU_ONLY

namespace eagle_tests {
namespace GraphPerKernelIdealsTest {

using namespace eagle;
using namespace eagle::cuda;

/* Two no-op kernels with distinct signatures for deterministic harvest
 * order; never launched, only inspected. */
__global__ void kernelA(int* buf, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        buf[tid] = tid;
}

__global__ void kernelB(int* buf, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        buf[tid] = tid + 1000;
}

/* Helper: capture a child graph with a single kernel (kernelA). */
static cudaGraph_t captureSingleKernel(
    const cudaStream_t& stream, int* buf, int n, int blockDim)
{
    StreamCapturer capturer(stream);
    capturer.begin();
    const int gridDim = (n + blockDim - 1) / blockDim;
    kernelA<<<gridDim, blockDim, 0, stream>>>(buf, n);
    return capturer.end();
}

/* Helper: capture a child graph with two distinct kernels (A, B). */
static cudaGraph_t captureTwoKernels(
    const cudaStream_t& stream, int* buf, int n, int blockDimA, int blockDimB)
{
    StreamCapturer capturer(stream);
    capturer.begin();
    const int gridA = (n + blockDimA - 1) / blockDimA;
    const int gridB = (n + blockDimB - 1) / blockDimB;
    kernelA<<<gridA, blockDimA, 0, stream>>>(buf, n);
    kernelB<<<gridB, blockDimB, 0, stream>>>(buf, n);
    return capturer.end();
}

class GraphPerKernelIdealsFixture : public Test {
protected:
    static constexpr int N = 64;
    int* d_buf             = nullptr;

    void SetUp() override
    {
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
};

/* Test 1: addKernelNode preserves the supplied idealBlockSize per record. */
TEST_F(GraphPerKernelIdealsFixture, PerKernelIdealsPreservedViaAddKernelNode)
{
    Stream s;
    Graph g;
    g.stream(s.cuda());

    const int blockDimA = 32;
    const int blockDimB = 64;
    const int gridDim   = 1;

    /* kernelA: idealBlockSize = 128. */
    {
        cudaKernelNodeParams params {};
        void* args[] = { (void*)&d_buf, (void*)&N };
        params.func           = (void*)kernelA;
        params.gridDim        = dim3(gridDim, 1, 1);
        params.blockDim       = dim3(blockDimA, 1, 1);
        params.sharedMemBytes = 0;
        params.kernelParams   = args;
        params.extra          = nullptr;
        g.addKernelNode(params, std::initializer_list<idx_t>{},
            /*idealBlockSize=*/128);
    }

    /* kernelB: idealBlockSize = 256. */
    {
        cudaKernelNodeParams params {};
        void* args[] = { (void*)&d_buf, (void*)&N };
        params.func           = (void*)kernelB;
        params.gridDim        = dim3(gridDim, 1, 1);
        params.blockDim       = dim3(blockDimB, 1, 1);
        params.sharedMemBytes = 0;
        params.kernelParams   = args;
        params.extra          = nullptr;
        g.addKernelNode(params, std::initializer_list<idx_t>{},
            /*idealBlockSize=*/256);
    }

    Launcher launcher = g.launcher();
    ASSERT_EQ(launcher.kernelNodeCount(), 2);
    EXPECT_EQ(launcher.kernelNodes()[0].idealBlockSize, 128);
    EXPECT_EQ(launcher.kernelNodes()[1].idealBlockSize, 256);
}

/* Test 2: legacy single-cap addNode(cudaGraph_t, deps, cap) writes the
 * same cap to every kernel inside the captured child. */
TEST_F(GraphPerKernelIdealsFixture, HarvestUniformChildSingleKernel)
{
    Stream s;
    Graph g;
    g.stream(s.cuda());

    cudaGraph_t child = captureSingleKernel(s.cuda(), d_buf, N, /*blockDim=*/32);
    g.addNode(child, std::initializer_list<idx_t>{}, /*idealBlockSize=*/256);

    Launcher launcher = g.launcher();
    ASSERT_EQ(launcher.kernelNodeCount(), 1);
    EXPECT_EQ(launcher.kernelNodes()[0].idealBlockSize, 256);
}

/* Test 3: REGRESSION — multi-kernel CapturedGraph caps {128, 256}
 * must be preserved positionally, NOT smeared uniformly. */
TEST_F(GraphPerKernelIdealsFixture, HarvestPerKernelChildPreservesEachCap)
{
    Stream s;
    Graph g;
    g.stream(s.cuda());

    cudaGraph_t child = captureTwoKernels(
        s.cuda(), d_buf, N, /*blockDimA=*/32, /*blockDimB=*/64);
    CapturedGraph cap { child, std::vector<idx_t>{ 128, 256 } };
    g.addNode(std::move(cap));

    Launcher launcher = g.launcher();
    ASSERT_EQ(launcher.kernelNodeCount(), 2);
    EXPECT_EQ(launcher.kernelNodes()[0].idealBlockSize, 128)
        << "first kernel must keep its own cap, not the second's";
    EXPECT_EQ(launcher.kernelNodes()[1].idealBlockSize, 256)
        << "second kernel must keep its own cap, not the first's";
}

/* Test 4: Empty idealBlockSizes vector = grid-only for every kernel. */
TEST_F(GraphPerKernelIdealsFixture, HarvestPerKernelEmptyCapsTreatedAsZero)
{
    Stream s;
    Graph g;
    g.stream(s.cuda());

    cudaGraph_t child = captureTwoKernels(
        s.cuda(), d_buf, N, /*blockDimA=*/32, /*blockDimB=*/64);
    CapturedGraph cap { child, std::vector<idx_t>{} };
    g.addNode(std::move(cap));

    Launcher launcher = g.launcher();
    ASSERT_EQ(launcher.kernelNodeCount(), 2);
    EXPECT_EQ(launcher.kernelNodes()[0].idealBlockSize, 0);
    EXPECT_EQ(launcher.kernelNodes()[1].idealBlockSize, 0);
}

/* Test 5: setLogicalSize uses the per-kernel cap, not a uniform value.
 * Use very large nActive so the policy clamps each kernel to its OWN
 * cap (different caps → different patched blockDim). */
TEST_F(GraphPerKernelIdealsFixture, SetLogicalSizeUsesPerKernelCap)
{
    Stream s;
    Graph g;
    g.stream(s.cuda());

    cudaGraph_t child = captureTwoKernels(
        s.cuda(), d_buf, N, /*blockDimA=*/32, /*blockDimB=*/64);
    /* Disparate caps so the two kernels diverge under setLogicalSize. */
    CapturedGraph cap { child, std::vector<idx_t>{ 64, 256 } };
    g.addNode(std::move(cap));

    Launcher launcher = g.launcher();
    ASSERT_EQ(launcher.kernelNodeCount(), 2);

    constexpr idx_t nActive = 1'000'000;
    launcher.setLogicalSize(nActive);

    /* setLogicalSize patches the EXEC graph in place; the snapshot is
     * unchanged. Re-run the policy to confirm caps would diverge. */
    EXPECT_EQ(launcher.kernelNodes()[0].idealBlockSize, 64);
    EXPECT_EQ(launcher.kernelNodes()[1].idealBlockSize, 256);

    idx_t bs0 = 0, nb0 = 0;
    idx_t bs1 = 0, nb1 = 0;
    computeBlocks(nActive, nb0, bs0, /*ideal=*/64);
    computeBlocks(nActive, nb1, bs1, /*ideal=*/256);
    EXPECT_NE(bs0, bs1);
    EXPECT_EQ(bs0, 64);
    EXPECT_EQ(bs1, 256);
}

/* Test 6: Same caps fed to two separate parents are preserved in
 * both — harvest is non-destructive on the producer's cap table. */
TEST_F(GraphPerKernelIdealsFixture, HarvestNeverOverridesExisting)
{
    Stream s;
    int* d_buf2 = nullptr;
    EAGLE_CHECK_ALWAYS(cudaMalloc(&d_buf2, N * sizeof(int)));

    cudaGraph_t childA = captureTwoKernels(
        s.cuda(), d_buf, N, /*blockDimA=*/32, /*blockDimB=*/64);
    cudaGraph_t childB = captureTwoKernels(
        s.cuda(), d_buf2, N, /*blockDimA=*/32, /*blockDimB=*/64);

    Graph g1, g2;
    g1.stream(s.cuda());
    g2.stream(s.cuda());

    /* Same caps fed to two different parents. */
    CapturedGraph capA { childA, std::vector<idx_t>{ 128, 512 } };
    CapturedGraph capB { childB, std::vector<idx_t>{ 128, 512 } };
    g1.addNode(std::move(capA));
    g2.addNode(std::move(capB));

    Launcher l1 = g1.launcher();
    Launcher l2 = g2.launcher();
    ASSERT_EQ(l1.kernelNodeCount(), 2);
    ASSERT_EQ(l2.kernelNodeCount(), 2);
    EXPECT_EQ(l1.kernelNodes()[0].idealBlockSize, 128);
    EXPECT_EQ(l1.kernelNodes()[1].idealBlockSize, 512);
    EXPECT_EQ(l2.kernelNodes()[0].idealBlockSize, 128);
    EXPECT_EQ(l2.kernelNodes()[1].idealBlockSize, 512);

    EAGLE_CHECK_ALWAYS(cudaFree(d_buf2));
}

} // namespace GraphPerKernelIdealsTest
} // namespace eagle_tests

#endif
