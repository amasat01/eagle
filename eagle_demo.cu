// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// EAGLE CUDA-mode smoke demo: capture a trivial kernel launch into a
// CUDA graph via the StreamCapturer, instantiate a Launcher, replay it,
// and verify the device buffer.
#include <cstdio>
#include <vector>

#include <eagle/eagle.h>

__global__ void addOne(int* buf, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        buf[tid] += 1;
}

int main()
{
    constexpr int N = 64;
    int* dBuf       = nullptr;
    if (cudaMalloc(&dBuf, N * sizeof(int)) != cudaSuccess)
        return 1;
    cudaMemset(dBuf, 0, N * sizeof(int));

    eagle::cuda::Stream stream;
    eagle::cuda::Graph graph;
    graph.stream(stream.cuda());

    eagle::cuda::StreamCapturer capturer(stream.cuda());
    capturer.begin();
    addOne<<<1, N, 0, stream.cuda()>>>(dBuf, N);
    graph.addNode(capturer.end());

    eagle::cuda::Launcher launcher = graph.launcher();
    launcher.launch();
    launcher.synchronize();

    std::vector<int> host(N, -1);
    cudaMemcpy(host.data(), dBuf, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dBuf);

    for (int i = 0; i < N; ++i)
        if (host[i] != 1) {
            std::printf("eagle_demo (CUDA): FAIL at %d\n", i);
            return 1;
        }

    std::printf("eagle_demo (CUDA): OK\n");
    return 0;
}
