// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// 01_capture_replay.cu
//
// The flagship EAGLE pattern: record a stream of kernel launches once, assemble
// them into a CUDA graph, instantiate a Launcher, and replay the instantiated
// graph many times with no per-launch API overhead.
//
// This is a CUDA-only example (graph capture has no pure-C++ analogue). Build:
//   nvcc -std=c++20 -arch=sm_61 -I<prefix>/include -DEAGLE_BLOCKSIZE=256 \
//        -DCUDA_API_PER_THREAD_DEFAULT_STREAM=1 -Xcompiler -fPIE \
//        01_capture_replay.cu -o 01_capture_replay
#include <cstdio>
#include <vector>

#include <eagle/eagle.h>

// [cell:kernel]
// A trivial elementwise kernel — the "work" we want to capture and replay.
__global__ void addOne(int* buf, int n)
{
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        buf[tid] += 1;
}
// [cell:kernel:end]

int main()
{
    constexpr int N       = 64;
    constexpr int REPLAYS = 10;

    int* dBuf = nullptr;
    if (cudaMalloc(&dBuf, N * sizeof(int)) != cudaSuccess)
        return 1;
    cudaMemset(dBuf, 0, N * sizeof(int));

    // [cell:capture]
    // A stream to record on, and a Graph that will own the captured work.
    eagle::cuda::Stream stream;
    eagle::cuda::Graph graph;
    graph.stream(stream.cuda());

    // Everything launched on the stream between begin() and end() is recorded
    // into a cudaGraph_t instead of executing eagerly.
    eagle::cuda::StreamCapturer capturer(stream.cuda());
    capturer.begin();
    addOne<<<1, N, 0, stream.cuda()>>>(dBuf, N);
    graph.addNode(capturer.end());
    // [cell:capture:end]

    // [cell:replay]
    // Instantiate the graph once, then replay it REPLAYS times. Each launch()
    // is a single cudaGraphLaunch — no kernel-config marshalling per call.
    eagle::cuda::Launcher launcher = graph.launcher();
    for (int r = 0; r < REPLAYS; ++r)
        launcher.launch();
    launcher.synchronize();
    // [cell:replay:end]

    std::vector<int> host(N, -1);
    cudaMemcpy(host.data(), dBuf, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(dBuf);

    bool ok = true;
    for (int i = 0; i < N; ++i)
        ok &= (host[i] == REPLAYS);

    std::printf("01_capture_replay: %d replays -> buf[0]=%d (expected %d) : %s\n",
        REPLAYS, host[0], REPLAYS, ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
