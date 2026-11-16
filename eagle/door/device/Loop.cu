// SPDX-License-Identifier: Apache-2.0

// The captured loop of a DEVICE runner over eagle's capture seam: the stream
// capture (StreamCapturer), one WHILE weave (CaptureConditional: a head setter,
// the body on its own stream, the tail setter), the graph (CapturedGraph),
// instantiated once and launched per run, as the old path's band loop
// (``repeat_while`` under ``GraphPipeline.build``).
#include "eagle/door/device/Loop.h"

#include "eagle/cuda/CaptureConditional.h"
#include "eagle/cuda/CapturedGraph.h"
#include "eagle/cuda/StreamCapturer.h"
#include "eagle/reduce/canonical.h"

namespace eagle {
namespace door {

struct DeviceLoop {
    cuda::CapturedGraph graph;
    cudaGraphExec_t exec = nullptr;
};

DeviceLoop* loop_capture(std::uint64_t stream, const unsigned* finished, const unsigned* total, unsigned* cells,
                         unsigned cap, const std::function<void(std::uint64_t)>& body) {
    const auto origin = reinterpret_cast<cudaStream_t>(stream);
    auto* loop = new DeviceLoop;
    try {
        cuda::CaptureConditional weave(origin, CountGuard{ finished, total }, cap, cells);
        cuda::StreamCapturer capturer(origin);
        capturer.begin();
        try {
            body(reinterpret_cast<std::uint64_t>(weave.begin()));
            weave.end();
        } catch (...) {
            (void)capturer.end();
            throw;
        }
        loop->graph = cuda::CapturedGraph(capturer.end());
        EAGLE_CHECK_ALWAYS(cudaGraphInstantiate(&loop->exec, loop->graph.graph(), 0));
    } catch (...) {
        delete loop;
        throw;
    }
    return loop;
}

namespace {

__global__ void maskCountKernel(const unsigned char* mask, long long n, unsigned width, unsigned* counter) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        bool set = false;
        for (unsigned b = 0; b < width; ++b) set = set || mask[i * width + b] != 0;
        if (set) atomicAdd(counter, 1u);
    }
}

}  // namespace

std::uint64_t cells_new(unsigned count) {
    void* p = nullptr;
    EAGLE_CHECK_ALWAYS(cudaMalloc(&p, count * sizeof(unsigned)));
    const cudaError_t z = cudaMemset(p, 0, count * sizeof(unsigned));
    if (z != cudaSuccess) (void)cudaFree(p);
    EAGLE_CHECK_ALWAYS(z);
    return reinterpret_cast<std::uint64_t>(p);
}

void cells_free(std::uint64_t cells) noexcept {
    if (cells != 0) (void)cudaFree(reinterpret_cast<void*>(cells));
}

void cells_write(std::uint64_t stream, std::uint64_t cells, const unsigned* values, unsigned count) {
    EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(reinterpret_cast<void*>(cells), values, count * sizeof(unsigned),
                                       cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
    stream_wait(stream);   // the host values are the caller's stack
}

void cells_read(std::uint64_t stream, std::uint64_t cells, unsigned* values, unsigned count) {
    EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(values, reinterpret_cast<const void*>(cells), count * sizeof(unsigned),
                                       cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    stream_wait(stream);
}

void mask_count(std::uint64_t stream, const void* mask, std::int64_t n, unsigned width, unsigned* counter) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    EAGLE_CHECK_ALWAYS(cudaMemsetAsync(counter, 0, sizeof(unsigned), s));
    if (n <= 0) return;
    const long long blocks = (n + 255) / 256 < 1024 ? (n + 255) / 256 : 1024;
    maskCountKernel<<<static_cast<unsigned>(blocks), 256, 0, s>>>(static_cast<const unsigned char*>(mask), n, width,
                                                                   counter);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

std::int64_t cell_work_elems(std::int64_t rows) {
    return static_cast<std::int64_t>(reduce::canonical::deviceWorkElems(static_cast<idx_t>(rows)));
}

namespace {

template <class T, class OP>
void fold_as(cudaStream_t stream, const void* in, std::int64_t rows, void* out, void* wa, void* wb) {
    namespace canon = reduce::canonical;
    canon::enqueueTo<T, OP>(canon::DevicePtr<T>{ static_cast<const T*>(in) }, canon::NoComp{}, static_cast<idx_t>(rows), 1,
                            reduce::Policy::ordered, static_cast<T*>(wa), static_cast<T*>(wb), static_cast<T*>(out), false,
                            stream);
}

}  // namespace

void cell_fold(std::uint64_t stream, const void* in, std::int64_t rows, unsigned bits, bool max, void* out, void* wa,
               void* wb) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    if (bits == 32) {
        if (max) fold_as<float, aether::MaxOp<float>>(s, in, rows, out, wa, wb);
        else fold_as<float, reduce::canonical::MinOp<float>>(s, in, rows, out, wa, wb);
    } else {
        if (max) fold_as<double, aether::MaxOp<double>>(s, in, rows, out, wa, wb);
        else fold_as<double, reduce::canonical::MinOp<double>>(s, in, rows, out, wa, wb);
    }
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

void stream_wait(std::uint64_t stream) {
    EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

void loop_launch(DeviceLoop* loop, std::uint64_t stream) {
    EAGLE_CHECK_ALWAYS(cudaGraphLaunch(loop->exec, reinterpret_cast<cudaStream_t>(stream)));
}

void loop_free(DeviceLoop* loop) noexcept {
    if (loop == nullptr) return;
    if (loop->exec != nullptr) (void)cudaGraphExecDestroy(loop->exec);
    delete loop;
}

}  // namespace door
}  // namespace eagle
