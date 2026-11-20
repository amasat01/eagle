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
#include "eagle/filtering/Compact.h"
#include "eagle/reduce/canonical.h"

#include "eagle/filtering/Reorder.h"
#include "eagle/filtering/ReorderClasses.h"
#include "eagle/diff/Reverse.h"
#include "eagle/events/Sensitivity.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <thread>

namespace eagle {
namespace door {

struct DeviceLoop {
    cuda::CapturedGraph graph;
    cudaGraphExec_t exec = nullptr;
};

DeviceLoop* loop_capture(std::uint64_t stream, const unsigned* finished, const unsigned* total, unsigned* cells,
                         unsigned cap, const std::function<void(std::uint64_t)>& body, const std::vector<LoopGate>& gates) {
    const auto origin = reinterpret_cast<cudaStream_t>(stream);
    auto* loop = new DeviceLoop;
    try {
        cuda::CaptureConditional weave(origin, CountGuard{ finished, total }, cap, cells);
        std::vector<std::unique_ptr<cuda::CaptureConditional>> ifs;    // the gates: IF nodes inside the loop body, made before the capture
        for (const LoopGate& g : gates)
            ifs.push_back(std::make_unique<cuda::CaptureConditional>(weave.bodyStream(), CountGuard{ g.count, g.base }));
        cuda::StreamCapturer capturer(origin);
        capturer.begin();
        try {
            body(reinterpret_cast<std::uint64_t>(weave.begin()));
            for (std::size_t i = 0; i < gates.size(); ++i) {
                gates[i].run(reinterpret_cast<std::uint64_t>(ifs[i]->begin()));
                ifs[i]->end();
            }
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

__global__ void compactDoneKernel(const unsigned* finished, unsigned* done_at_last, unsigned* compactions) {
    *done_at_last = *finished;
    ++*compactions;
}

__global__ void staleLiveKernel(unsigned char* stale, const unsigned char* dead, std::int64_t n) {
    const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n && dead[i] != 0) stale[i] = 0;
}

__global__ void reorderCountKernel(const unsigned* fire, unsigned* reorders) { *reorders += *fire != 0 ? 1u : 0u; }

__global__ void maskCountKernel(const unsigned char* mask, long long n, unsigned width, unsigned* counter) {
    for (long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<long long>(gridDim.x) * blockDim.x) {
        bool set = false;
        for (unsigned b = 0; b < width; ++b) set = set || mask[i * width + b] != 0;
        if (set) atomicAdd(counter, 1u);
    }
}

}  // namespace

void* host_pinned_new(std::size_t bytes) {
    void* p = nullptr;
    EAGLE_CHECK_ALWAYS(cudaHostAlloc(&p, bytes, cudaHostAllocDefault));
    return p;
}

void host_pinned_free(void* data) noexcept {
    if (data != nullptr) (void)cudaFreeHost(data);
}

void* alloc_async(std::uint64_t stream, std::size_t bytes, bool zero) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    void* p = nullptr;
    EAGLE_CHECK_ALWAYS(cudaMallocAsync(&p, bytes, s));
    if (!zero) return p;
    const cudaError_t z = cudaMemsetAsync(p, 0, bytes, s);
    if (z != cudaSuccess) (void)cudaFreeAsync(p, s);
    EAGLE_CHECK_ALWAYS(z);
    return p;
}

void free_async(std::uint64_t alloc_stream, std::uint64_t current, void* data) noexcept {
    if (data == nullptr) return;
    const auto freed_on = reinterpret_cast<cudaStream_t>(alloc_stream);
    if (current != alloc_stream) {    // the block may still be read by work queued on the stream now in scope
        cudaEvent_t done = nullptr;
        if (cudaEventCreateWithFlags(&done, cudaEventDisableTiming) == cudaSuccess) {
            (void)cudaEventRecord(done, reinterpret_cast<cudaStream_t>(current));
            (void)cudaStreamWaitEvent(freed_on, done, 0);
            (void)cudaEventDestroy(done);
        }
    }
    (void)cudaFreeAsync(data, freed_on);
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

void event_op(std::uint64_t stream, unsigned bytes, unsigned op, const void* planes) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    if (bytes == 8) events::device::apply(*static_cast<const events::EventPlanes<double>*>(planes), op, s);
    else events::device::apply(*static_cast<const events::EventPlanes<float>*>(planes), op, s);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

std::size_t compact_scratch_bytes(std::int64_t n) { return filtering::compactScratchBytes(n); }

void compact_step(std::uint64_t stream, const void* mask, std::int64_t n, std::int32_t* map, unsigned* count,
                  void* scratch, unsigned* live32, const unsigned* span, unsigned* fire, float theta) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    const auto* flags = static_cast<const std::uint8_t*>(mask);
    if (live32 != nullptr) {
        const filtering::ReorderTrigger trigger{ live32, span, fire, theta };
        filtering::compactDevice(flags, filtering::kCompactMaskIsDrop, map, count, scratch, n, s, &trigger);
    } else {
        filtering::compactDevice(flags, filtering::kCompactMaskIsDrop, map, count, scratch, n, s);
    }
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

void compact_mark(std::uint64_t stream, const unsigned* finished, unsigned* done_at_last, unsigned* compactions) {
    compactDoneKernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(finished, done_at_last, compactions);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

namespace {

std::vector<filtering::ReorderPlane> table(const std::vector<DevicePlane>& planes) {
    std::vector<filtering::ReorderPlane> out;
    for (const DevicePlane& p : planes) out.push_back({ p.data, p.bytes });
    return out;
}

}  // namespace

std::size_t reorder_scratch_bytes(std::int64_t n, const std::vector<DevicePlane>& planes) {
    const auto t = table(planes);
    return filtering::reorderScratchBytes(n, t.data(), static_cast<std::uint32_t>(t.size()), 0);
}

void reorder_step(std::uint64_t stream, const std::vector<DevicePlane>& planes, const void* mask, std::int32_t* perm,
                  std::int32_t* inv, std::int32_t* map, unsigned* count, unsigned* span, unsigned* fire, void* scratch,
                  std::int64_t n, unsigned* reorders) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    reorderCountKernel<<<1, 1, 0, s>>>(fire, reorders);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
    const auto t = table(planes);
    filtering::reorderDevice(t.data(), static_cast<std::uint32_t>(t.size()), static_cast<const std::uint8_t*>(mask),
                             filtering::kCompactMaskIsDrop, perm, inv, map, count, span, fire, scratch, n, s);
}

void restore_step(std::uint64_t stream, const std::vector<DevicePlane>& planes, std::int32_t* perm, std::int32_t* inv,
                  void* scratch, std::int64_t n) {
    const auto t = table(planes);
    filtering::restoreDevice(t.data(), static_cast<std::uint32_t>(t.size()), perm, inv, scratch, n, reinterpret_cast<cudaStream_t>(stream));
}

std::size_t class_sort_scratch_bytes(std::int64_t n, const std::vector<DevicePlane>& planes, unsigned classes) {
    std::uint32_t widest = 4u;
    for (const DevicePlane& p : planes) widest = std::max(widest, p.bytes);
    return filtering::classSortScratchBytes(n, widest, classes);
}

void class_sort_step(std::uint64_t stream, const std::vector<DevicePlane>& planes, const void* mask, unsigned classes, const void* class_plane,
                     std::int32_t* perm, std::int32_t* inv, std::int32_t* map, unsigned* count, unsigned* span, unsigned* offsets, unsigned* sorts,
                     void* scratch, std::int64_t n) {
    const auto t = table(planes);
    filtering::reorderClassesDevice(t.data(), static_cast<std::uint32_t>(t.size()), static_cast<const std::uint8_t*>(mask), filtering::kCompactMaskIsDrop,
                                    classes, static_cast<const std::uint8_t*>(class_plane), perm, inv, map, count, span, nullptr, offsets, sorts,
                                    scratch, n, reinterpret_cast<cudaStream_t>(stream));
}

void compact_keep_step(std::uint64_t stream, const void* mask, std::int64_t n, std::int32_t* map, unsigned* count, void* scratch) {
    filtering::compactDevice(static_cast<const std::uint8_t*>(mask), 0u, map, count, scratch, n, reinterpret_cast<cudaStream_t>(stream));
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

void stale_live_step(std::uint64_t stream, void* stale, const void* dead, std::int64_t n) {
    if (n <= 0) return;
    staleLiveKernel<<<static_cast<unsigned>((n + 255) / 256), 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        static_cast<unsigned char*>(stale), static_cast<const unsigned char*>(dead), n);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

void reset_step(std::uint64_t stream, std::int32_t* perm, std::int32_t* inv, std::int32_t* map, unsigned* count, unsigned* span, std::int64_t n) {
    filtering::resetDevice(perm, inv, map, count, span, n, reinterpret_cast<cudaStream_t>(stream));
}

namespace {

/** @brief Sample ``k`` of the seam's (rows, cols) row-major walk, read from a (cols, n) samples-last plane. */
template <class T>
struct SamplesLast {
    const T* p;
    idx_t n, cols;
    __host__ __device__ T operator()(idx_t k) const { return p[(k % cols) * n + k / cols]; }
};

template <class T, class OP>
void reduce_as(cudaStream_t s, const void* in, std::int64_t n, std::int64_t cols, void* out, void* comp, bool pair, void* wa, void* wb) {
    namespace canon = reduce::canonical;
    const auto rows = static_cast<idx_t>(n), width = static_cast<idx_t>(cols);
    const SamplesLast<T> x{ static_cast<const T*>(in), rows, width };
    T* a = static_cast<T*>(wa);
    T* b = static_cast<T*>(wb);
    if (pair) {    // (hi, comp) land as 2 * cols values: split them into the two planes
        const auto stream = reinterpret_cast<std::uint64_t>(s);
        T* both = static_cast<T*>(alloc_async(stream, 2 * static_cast<std::size_t>(cols) * sizeof(T), false));
        canon::enqueueTo<T, OP>(x, canon::NoComp{}, rows, width, reduce::Policy::ordered, a, b, both, true, s);
        EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(out, both, static_cast<std::size_t>(cols) * sizeof(T), cudaMemcpyDeviceToDevice, s));
        EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(comp, both + cols, static_cast<std::size_t>(cols) * sizeof(T), cudaMemcpyDeviceToDevice, s));
        free_async(stream, stream, both);
    } else if (comp != nullptr) {
        canon::enqueueTo<T, OP>(x, SamplesLast<T>{ static_cast<const T*>(comp), rows, width }, rows, width, reduce::Policy::ordered, a, b,
                                static_cast<T*>(out), false, s);
    } else {
        canon::enqueueTo<T, OP>(x, canon::NoComp{}, rows, width, reduce::Policy::ordered, a, b, static_cast<T*>(out), false, s);
    }
}

template <class T>
void reduce_op(cudaStream_t s, int op, const void* in, std::int64_t n, std::int64_t cols, void* out, void* comp, bool pair, void* wa, void* wb) {
    if (op == 0) reduce_as<T, aether::SumOp<T>>(s, in, n, cols, out, comp, pair, wa, wb);
    else if (op == 1) reduce_as<T, aether::MaxOp<T>>(s, in, n, cols, out, comp, pair, wa, wb);
    else reduce_as<T, reduce::canonical::MinOp<T>>(s, in, n, cols, out, comp, pair, wa, wb);
}

}  // namespace

void reduce_step(std::uint64_t stream, int code, int op, const void* in, std::int64_t n, std::int64_t cols, void* out, void* comp,
                 bool pair) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    const std::size_t elems = static_cast<std::size_t>(reduce::canonical::deviceWorkElems(static_cast<idx_t>(n), static_cast<idx_t>(cols)));
    void *wa = nullptr, *wb = nullptr;
    const std::size_t bytes = elems * 8;    // room for the widest partial of any element type
    try {
        wa = alloc_async(stream, bytes, false);
        wb = alloc_async(stream, bytes, false);
        switch (code) {
            case 0: reduce_op<float>(s, op, in, n, cols, out, comp, pair, wa, wb); break;
            case 1: reduce_op<double>(s, op, in, n, cols, out, comp, pair, wa, wb); break;
            case 2: reduce_op<std::int32_t>(s, op, in, n, cols, out, comp, pair, wa, wb); break;
            case 3: reduce_op<std::int64_t>(s, op, in, n, cols, out, comp, pair, wa, wb); break;
            default: reduce_as<std::uint8_t, aether::LogicalAndOp>(s, in, n, cols, out, nullptr, false, wa, wb);
        }
        EAGLE_CHECK_ALWAYS(cudaGetLastError());
        cudaStreamCaptureStatus capturing = cudaStreamCaptureStatusNone;
        EAGLE_CHECK_ALWAYS(cudaStreamIsCapturing(s, &capturing));
        if (capturing == cudaStreamCaptureStatusNone) EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(s));    // a capture cannot wait: its replay leaves the result in place
    } catch (...) {
        free_async(stream, stream, wa);
        free_async(stream, stream, wb);
        throw;
    }
    free_async(stream, stream, wa);
    free_async(stream, stream, wb);
}

void copy_step(std::uint64_t stream, void* dst, const void* src, std::size_t bytes) {
    EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, reinterpret_cast<cudaStream_t>(stream)));
}

void write_bytes(std::uint64_t stream, void* dst, const void* src, std::size_t bytes) {
    EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream)));
    stream_wait(stream);
}

void read_bytes(std::uint64_t stream, void* dst, const void* src, std::size_t bytes) {
    EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(stream)));
    stream_wait(stream);
}

void zero_bytes(std::uint64_t stream, void* dst, std::size_t bytes) {
    EAGLE_CHECK_ALWAYS(cudaMemsetAsync(dst, 0, bytes, reinterpret_cast<cudaStream_t>(stream)));
}

namespace {

template <class T>
__global__ void weighKernel(T* data, std::int64_t count, std::int64_t run, std::int64_t bins, std::int64_t edge, T scale, T interior) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x) {
        const std::int64_t bin = i / run % bins;
        data[i] *= bin > 0 && bin < edge ? interior : scale;
    }
}

template <class T>
void weigh_as(cudaStream_t s, void* data, std::int64_t count, std::int64_t run, std::int64_t bins, std::int64_t edge, double scale,
              double weight) {
    const std::int64_t blocks = std::min<std::int64_t>((count + 255) / 256, 4096);
    weighKernel<T><<<static_cast<unsigned>(blocks), 256, 0, s>>>(static_cast<T*>(data), count, run, bins, edge, static_cast<T>(scale),
                                                               static_cast<T>(scale * weight));
}

}  // namespace

void fft_weigh(std::uint64_t stream, unsigned bits, void* data, std::int64_t count, std::int64_t run, std::int64_t bins,
               std::int64_t edge, double scale, double weight) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    if (count <= 0) return;
    if (bits == 64) weigh_as<double>(s, data, count, run, bins, edge, scale, weight);
    else weigh_as<float>(s, data, count, run, bins, edge, scale, weight);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

namespace {

template <class T>
__global__ void scatterColumnsKernel(T* dst, std::int64_t dstLd, const T* src, std::int64_t srcLd, const std::int64_t* index,
                                     std::int64_t count, std::int64_t rows) {
    for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count * rows;
         i += static_cast<std::int64_t>(gridDim.x) * blockDim.x)
        dst[(i / count) * dstLd + index[i % count]] = src[(i / count) * srcLd + i % count];
}

template <class T>
void columns_as(cudaStream_t s, int op, void* dst, std::int64_t dst_ld, const void* src, std::int64_t src_ld,
                const std::int64_t* index, std::int64_t count, std::int64_t rows) {
    if (op == 0) {
        diff::device::gatherColumns<T>(dst, dst_ld, src, src_ld, index, count, rows, s);
        return;
    }
    const std::int64_t blocks = (count * rows + 255) / 256 < 1024 ? (count * rows + 255) / 256 : 1024;
    scatterColumnsKernel<T><<<static_cast<unsigned>(blocks), 256, 0, s>>>(static_cast<T*>(dst), dst_ld, static_cast<const T*>(src),
                                                                         src_ld, index, count, rows);
}

}  // namespace

void diff_columns(std::uint64_t stream, int op, void* dst, std::int64_t dst_ld, const void* src, std::int64_t src_ld,
                  const std::int64_t* index, std::int64_t count, std::int64_t rows, unsigned bytes, std::int64_t value) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    if (count <= 0 || rows <= 0) return;
    if (op == 1 && bytes == 8) diff::device::accumulatePrefix(static_cast<double*>(dst), static_cast<const double*>(src), dst_ld, count, rows, s);
    else if (op == 1) diff::device::accumulatePrefix(static_cast<float*>(dst), static_cast<const float*>(src), dst_ld, count, rows, s);
    else if (op == 2 && bytes == 8) diff::device::fillPrefix(static_cast<std::int64_t*>(dst), static_cast<std::int64_t>(value), count, s);
    else if (op == 2) diff::device::fillPrefix(static_cast<std::int32_t*>(dst), static_cast<std::int32_t>(value), count, s);
    else if (bytes == 1) columns_as<std::uint8_t>(s, op, dst, dst_ld, src, src_ld, index, count, rows);
    else if (bytes == 2) columns_as<std::uint16_t>(s, op, dst, dst_ld, src, src_ld, index, count, rows);
    else if (bytes == 4) columns_as<std::uint32_t>(s, op, dst, dst_ld, src, src_ld, index, count, rows);
    else columns_as<std::uint64_t>(s, op, dst, dst_ld, src, src_ld, index, count, rows);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

std::int64_t free_bytes() {
    std::size_t free = 0, total = 0;
    EAGLE_CHECK_ALWAYS(cudaMemGetInfo(&free, &total));
    return static_cast<std::int64_t>(free);
}

std::int64_t cell_work_elems(std::int64_t rows) {
    return static_cast<std::int64_t>(reduce::canonical::deviceWorkElems(static_cast<idx_t>(rows)));
}

namespace {

template <class T, class OP>
void fold_as(cudaStream_t stream, const void* in, const void* comp, std::int64_t rows, void* out, void* wa, void* wb) {
    namespace canon = reduce::canonical;
    if (comp != nullptr)    // a compensated sum: the rows' residuals ride as the input companion
        canon::enqueueTo<T, OP>(canon::DevicePtr<T>{ static_cast<const T*>(in) }, canon::DevicePtr<T>{ static_cast<const T*>(comp) },
                                static_cast<idx_t>(rows), 1, reduce::Policy::ordered, static_cast<T*>(wa), static_cast<T*>(wb),
                                static_cast<T*>(out), false, stream);
    else
        canon::enqueueTo<T, OP>(canon::DevicePtr<T>{ static_cast<const T*>(in) }, canon::NoComp{}, static_cast<idx_t>(rows), 1,
                                reduce::Policy::ordered, static_cast<T*>(wa), static_cast<T*>(wb), static_cast<T*>(out), false,
                                stream);
}

}  // namespace

void cell_fold(std::uint64_t stream, const void* in, const void* comp, std::int64_t rows, unsigned bits, int op, void* out, void* wa,
               void* wb) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    if (bits == 32) {
        if (op == 0) fold_as<float, aether::SumOp<float>>(s, in, comp, rows, out, wa, wb);
        else if (op == 1) fold_as<float, aether::MaxOp<float>>(s, in, nullptr, rows, out, wa, wb);
        else fold_as<float, reduce::canonical::MinOp<float>>(s, in, nullptr, rows, out, wa, wb);
    } else {
        if (op == 0) fold_as<double, aether::SumOp<double>>(s, in, comp, rows, out, wa, wb);
        else if (op == 1) fold_as<double, aether::MaxOp<double>>(s, in, nullptr, rows, out, wa, wb);
        else fold_as<double, reduce::canonical::MinOp<double>>(s, in, nullptr, rows, out, wa, wb);
    }
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

void stream_wait(std::uint64_t stream) {
    EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)));
}

struct DeviceCapture {
    cuda::StreamCapturer capturer;
};

DeviceCapture* capture_begin(std::uint64_t stream) {
    auto capture = std::make_unique<DeviceCapture>(DeviceCapture{ cuda::StreamCapturer(reinterpret_cast<cudaStream_t>(stream)) });
    capture->capturer.begin();
    return capture.release();
}

DeviceLoop* capture_end(DeviceCapture* capture) {
    const std::unique_ptr<DeviceCapture> owned(capture);
    auto loop = std::make_unique<DeviceLoop>();
    loop->graph = cuda::CapturedGraph(owned->capturer.end());
    EAGLE_CHECK_ALWAYS(cudaGraphInstantiate(&loop->exec, loop->graph.graph(), 0));
    return loop.release();
}

void capture_abort(DeviceCapture* capture) noexcept {
    if (capture == nullptr) return;
    const std::unique_ptr<DeviceCapture> owned(capture);
    try {
        cuda::CapturedGraph dropped(owned->capturer.end());
    } catch (...) {
    }
    (void)cudaGetLastError();    // the error of an unfinished capture (an unjoined fork) is not the next call's
}

void loop_launch(DeviceLoop* loop, std::uint64_t stream) {
    EAGLE_CHECK_ALWAYS(cudaGraphLaunch(loop->exec, reinterpret_cast<cudaStream_t>(stream)));
}

void loop_free(DeviceLoop* loop) noexcept {
    if (loop == nullptr) return;
    if (loop->exec != nullptr) (void)cudaGraphExecDestroy(loop->exec);
    delete loop;
}

std::uint64_t loop_exec(const DeviceLoop* loop) { return reinterpret_cast<std::uint64_t>(loop->exec); }

std::uint64_t loop_graph(const DeviceLoop* loop) { return reinterpret_cast<std::uint64_t>(loop->graph.graph()); }

void loop_reinstantiate(DeviceLoop* loop) {
    cudaGraphExec_t fresh = nullptr;
    EAGLE_CHECK_ALWAYS(cudaGraphInstantiate(&fresh, loop->graph.graph(), 0));
    if (loop->exec != nullptr) (void)cudaGraphExecDestroy(loop->exec);
    loop->exec = fresh;
}

std::vector<std::uint64_t> capture_kernels(std::uint64_t stream, unsigned count) {
    cudaStreamCaptureStatus status;
    unsigned long long id = 0;
    cudaGraph_t graph = nullptr;
    const cudaGraphNode_t* tail = nullptr;
    const cudaGraphEdgeData* edge_data = nullptr;
    std::size_t n_tail = 0;
    EAGLE_CHECK_ALWAYS(cuda::detail::streamGetCaptureInfo(reinterpret_cast<cudaStream_t>(stream), &status, &id, &graph, &tail, &edge_data, &n_tail));
    std::vector<std::uint64_t> nodes(count);
    if (count == 0) return nodes;
    EAGLE_ASSERT(n_tail == 1, "capture_kernels: the launches did not end in one node");
    std::size_t n_edges = 0;
    EAGLE_CHECK_ALWAYS(cuda::detail::graphGetEdges(graph, nullptr, nullptr, &n_edges));
    std::vector<cudaGraphNode_t> from(n_edges), to(n_edges);
    if (n_edges != 0) EAGLE_CHECK_ALWAYS(cuda::detail::graphGetEdges(graph, from.data(), to.data(), &n_edges));
    cudaGraphNode_t at = tail[0];
    for (unsigned k = count; k-- > 0;) {
        nodes[k] = reinterpret_cast<std::uint64_t>(at);
        if (k == 0) break;
        const auto edge = std::find(to.begin(), to.begin() + static_cast<std::ptrdiff_t>(n_edges), at);
        EAGLE_ASSERT(edge != to.begin() + static_cast<std::ptrdiff_t>(n_edges), "capture_kernels: a partition launch has no predecessor");
        at = from[static_cast<std::size_t>(edge - to.begin())];
    }
    return nodes;
}

struct DevicePool {
    cuda::Stream side{ true };    // scratch_alloc's stream: never captured
    std::vector<std::unique_ptr<cuda::Stream>> streams;
    std::vector<std::unique_ptr<cuda::Event>> events;
    unsigned streams_taken = 0, events_taken = 0;
    cudaStream_t stream() { return streams.at(streams_taken++)->cuda(); }
    cudaEvent_t event() { return events.at(events_taken++)->cuda(); }
};

DevicePool* pool_new(unsigned streams, unsigned events) {
    auto pool = std::make_unique<DevicePool>();
    for (unsigned i = 0; i < streams; ++i) pool->streams.push_back(std::make_unique<cuda::Stream>(true));
    for (unsigned i = 0; i < events; ++i) pool->events.push_back(std::make_unique<cuda::Event>(cudaEventDisableTiming));
    return pool.release();
}

void pool_free(DevicePool* pool) noexcept { delete pool; }

namespace {

__global__ void saveTestKernel(const unsigned* cell, const unsigned* baseline, const unsigned* iter, unsigned* dst, unsigned capacity) {
    unsigned k = iter != nullptr ? *iter : 0u;
    if (k >= capacity) k = capacity - 1;
    dst[3 * k] = 1u;
    dst[3 * k + 1] = *cell;
    dst[3 * k + 2] = baseline != nullptr ? *baseline : 0u;
}

__global__ void tableCopyKernel(const unsigned char* src, std::size_t bytes, unsigned char* const* table, const unsigned* iter, unsigned capacity, unsigned window) {
    unsigned k = *iter;
    if (k % window != 0) return;
    k /= window;
    if (k >= capacity) k = capacity - 1;
    unsigned char* dst = table[k];
    const std::size_t first = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x, stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t i = first; i < bytes; i += stride) dst[i] = src[i];
}

}  // namespace

void save_test_step(std::uint64_t stream, const unsigned* cell, const unsigned* baseline, const unsigned* iter, unsigned* dst, unsigned capacity) {
    saveTestKernel<<<1, 1, 0, reinterpret_cast<cudaStream_t>(stream)>>>(cell, baseline, iter, dst, capacity);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

void table_copy_step(std::uint64_t stream, const void* src, std::size_t bytes, void* const* table, const unsigned* iter, unsigned capacity, unsigned window) {
    if (bytes == 0) return;
    const unsigned blocks = static_cast<unsigned>(std::min<std::size_t>((bytes + 255) / 256, 1024));
    tableCopyKernel<<<blocks, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const unsigned char*>(src), bytes,
                                                                               reinterpret_cast<unsigned char* const*>(table), iter, capacity, window);
    EAGLE_CHECK_ALWAYS(cudaGetLastError());
}

void read_many(std::uint64_t stream, const std::vector<ReadItem>& items) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    for (const ReadItem& r : items)
        if (r.bytes != 0) EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(r.host, r.device, r.bytes, cudaMemcpyDeviceToHost, s));
    stream_wait(stream);
}

bool pool_has(const DevicePool* pool, unsigned streams, unsigned events) {
    return pool->streams_taken + streams <= pool->streams.size() && pool->events_taken + events <= pool->events.size();
}

/** @brief A conditional: its eagle weave, and the pool streams it took (given back when it ends). */
struct DeviceWeave {
    std::unique_ptr<cuda::CaptureConditional> cond;
    DevicePool* pool = nullptr;
    unsigned streams_before = 0;
    ~DeviceWeave() { if (pool != nullptr) pool->streams_taken = streams_before; }
};

/** @brief A fork: the pool streams and events it took (given back when it ends). */
struct DeviceFork {
    DevicePool* pool;
    cudaStream_t origin;
    cudaEvent_t forked;
    std::vector<cudaStream_t> branches;
    std::vector<cudaEvent_t> joins;
    unsigned streams_before, events_before;
    DeviceFork(DevicePool* p, cudaStream_t o)
        : pool(p), origin(o), forked(p->event()), streams_before(p->streams_taken), events_before(p->events_taken - 1) {}
    ~DeviceFork() {
        pool->streams_taken = streams_before;
        pool->events_taken = events_before;
    }
};

bool cond_arms_supported() { return cuda::conditionalArmsSupported(); }

DeviceWeave* cond_begin(DevicePool* pool, std::uint64_t origin, int kind, const unsigned* cell, const unsigned* baseline, bool negate,
                        unsigned arms, unsigned cap, unsigned* loop_cells) {
    const auto stream = reinterpret_cast<cudaStream_t>(origin);
    auto weave = std::make_unique<DeviceWeave>();
    weave->streams_before = pool->streams_taken;
    std::vector<cudaStream_t> borrowed;
    for (unsigned k = 0; k < arms; ++k) borrowed.push_back(pool->stream());
    weave->pool = pool;
    const conditional::LoopTerms terms{ cap == 0 ? conditional::kLoopUncapped : cap, loop_cells };
    weave->cond = std::make_unique<cuda::CaptureConditional>(stream, static_cast<cuda::CondKind>(kind), CountGuard{ cell, baseline }, arms,
                                                             negate ? 1u : 0u, terms, borrowed);
    weave->cond->begin();
    return weave.release();
}

std::uint64_t cond_stream(const DeviceWeave* weave, unsigned arm) {
    return reinterpret_cast<std::uint64_t>(weave->cond->armStream(arm));
}

void cond_end(DeviceWeave* weave) {
    const std::unique_ptr<DeviceWeave> owned(weave);
    owned->cond->end();
}

void cond_drop(DeviceWeave* weave) noexcept { delete weave; }

DeviceFork* fork_begin(DevicePool* pool, std::uint64_t origin) {
    auto fork = std::make_unique<DeviceFork>(pool, reinterpret_cast<cudaStream_t>(origin));
    EAGLE_CHECK_ALWAYS(cudaEventRecord(fork->forked, fork->origin));
    fork_branch(fork.get());
    return fork.release();
}

std::uint64_t fork_branch(DeviceFork* fork) {
    const cudaStream_t s = fork->pool->stream();
    fork->branches.push_back(s);
    fork->joins.push_back(fork->pool->event());
    EAGLE_CHECK_ALWAYS(cudaStreamWaitEvent(s, fork->forked, 0));
    return reinterpret_cast<std::uint64_t>(s);
}

std::uint64_t fork_stream(const DeviceFork* fork, unsigned branch) {
    return reinterpret_cast<std::uint64_t>(fork->branches.at(branch));
}

void fork_wait(DeviceFork* fork, unsigned waiter, unsigned producer) {
    const cudaEvent_t done = fork->pool->event();
    EAGLE_CHECK_ALWAYS(cudaEventRecord(done, fork->branches.at(producer)));
    EAGLE_CHECK_ALWAYS(cudaStreamWaitEvent(fork->branches.at(waiter), done, 0));
}

void fork_end(DeviceFork* fork) {
    const std::unique_ptr<DeviceFork> owned(fork);
    for (std::size_t i = 0; i < owned->branches.size(); ++i) {
        EAGLE_CHECK_ALWAYS(cudaEventRecord(owned->joins[i], owned->branches[i]));
        EAGLE_CHECK_ALWAYS(cudaStreamWaitEvent(owned->origin, owned->joins[i], 0));
    }
}

void fork_drop(DeviceFork* fork) noexcept { delete fork; }

}  // namespace door
}  // namespace eagle
