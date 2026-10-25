// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_Transfer.cu
 * @brief ``eagle/exec/Transfer.h``, the CUDA face: a copy in every direction,
 *        page-locked host memory, and a host function on a stream, also as nodes
 *        of a captured graph replayed (C++-only mode runs the host face in
 *        test_Transfer.cpp).
 *
 * RED: a host function that ran at capture time instead of at every replay
 * leaves the count at 1 after three replays.
 */

#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "eagle/exec/Transfer.h"

#include "TestBase.h"

namespace eagle_tests {
namespace TransferTest {

using eagle::exec::CopyKind;

TEST(TransferTest, MemcpyCopiesInEveryDirection)
{
    const std::size_t n = 37;
    std::vector<double> src(n), back(n, 0.0), plain(n, 0.0);
    std::iota(src.begin(), src.end(), 1.0);
    void* pinned = eagle::exec::hostAlloc(n * sizeof(double));
    double *d1 = nullptr, *d2 = nullptr;
    ASSERT_EQ(cudaMalloc(&d1, n * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d2, n * sizeof(double)), cudaSuccess);
    cudaStream_t s;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);

    eagle::exec::memcpyAsync(pinned, src.data(), n * sizeof(double), CopyKind::HostToHost, s);
    eagle::exec::memcpyAsync(d1, pinned, n * sizeof(double), CopyKind::HostToDevice, s);
    eagle::exec::memcpyAsync(d2, d1, n * sizeof(double), CopyKind::DeviceToDevice, s);
    eagle::exec::memcpyAsync(back.data(), d2, n * sizeof(double), CopyKind::DeviceToHost, s);
    ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
    EXPECT_EQ(back, src);

    cudaStreamDestroy(s);
    cudaFree(d1);
    cudaFree(d2);
    eagle::exec::hostFree(pinned);
}

TEST(TransferTest, MemcpyOntoItselfOrOfNothingEnqueuesNothing)
{
    cudaStream_t s;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    std::vector<double> a{ 1.0, 2.0, 3.0 };
    // Inside a capture a node would show; an empty capture instantiates only when nothing was added.
    ASSERT_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    eagle::exec::memcpyAsync(a.data(), a.data(), a.size() * sizeof(double), CopyKind::HostToHost, s);
    eagle::exec::memcpyAsync(a.data(), nullptr, 0, CopyKind::HostToHost, s);
    cudaGraph_t g = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(s, &g), cudaSuccess);
    std::size_t nodes = 99;
    ASSERT_EQ(cudaGraphGetNodes(g, nullptr, &nodes), cudaSuccess);
    EXPECT_EQ(nodes, 0u);
    EXPECT_EQ(a, (std::vector<double>{ 1.0, 2.0, 3.0 }));
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
}

TEST(TransferTest, HostAllocGivesPageLockedMemoryAndFreeAcceptsNull)
{
    void* p = eagle::exec::hostAlloc(1000);
    ASSERT_NE(p, nullptr);
    cudaPointerAttributes attr{};
    ASSERT_EQ(cudaPointerGetAttributes(&attr, p), cudaSuccess);
    EXPECT_EQ(attr.type, cudaMemoryTypeHost);
    std::memset(p, 0xAB, 1000);
    eagle::exec::hostFree(p);
    eagle::exec::hostFree(nullptr);
}

TEST(TransferTest, HostFuncRunsOnEveryReplayOfACapturedGraphAndNotAtCapture)
{
    cudaStream_t s;
    ASSERT_EQ(cudaStreamCreate(&s), cudaSuccess);
    int hits = 0;
    ASSERT_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), cudaSuccess);
    eagle::exec::hostFunc(s, [](void* arg) { ++*static_cast<int*>(arg); }, &hits);
    cudaGraph_t g = nullptr;
    ASSERT_EQ(cudaStreamEndCapture(s, &g), cudaSuccess);
    EXPECT_EQ(hits, 0);
    cudaGraphExec_t exec = nullptr;
    ASSERT_EQ(cudaGraphInstantiate(&exec, g, 0), cudaSuccess);
    for (int k = 1; k <= 3; ++k) {
        ASSERT_EQ(cudaGraphLaunch(exec, s), cudaSuccess);
        ASSERT_EQ(cudaStreamSynchronize(s), cudaSuccess);
        EXPECT_EQ(hits, k);
    }
    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(g);
    cudaStreamDestroy(s);
}

}  // namespace TransferTest
}  // namespace eagle_tests
