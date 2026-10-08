// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The CUDA-aware half of the generic DLPack layer.
//
// aether's `aether/interop/Buffer.h` owns the buffer record (zero-copy view,
// access, owner, producer, keep-alive), the `Requirements` validation and
// the DLPack import/export; it makes no CUDA call. This header adds what
// needs the CUDA runtime: the DLPack 1.0 stream contract on export, the
// re-fence a consumer issues before each launch that reads an imported
// buffer, and the device checks of the host arm.
//
// STREAM CODES (DLPack 1.0, CUDA). A stream argument is optional:
//   none or 1   the legacy default stream (cudaStreamLegacy)
//   2           the per-thread default stream (cudaStreamPerThread)
//   -1          no ordering: the caller takes responsibility
//   0           refused (ambiguous by the DLPack specification)
//   any other   a cudaStream_t handle
// Ordering is one cudaEventRecord on the producer's stream plus one
// cudaStreamWaitEvent on the consumer's. The layer never blocks the host.
// Events come from a per-device pool and are reused: a wait captures the
// event's state when it is enqueued, so the event is free again right after.
//
// HOST ARM. A CPU buffer takes no stream: a stream argument other than none
// is refused with a named message. Requirements on the device type refuse a
// CPU buffer where CUDA is required and the reverse.
//
// A pure C++ build (EAGLE_CPU_ONLY) keeps the host arm and the stream-code
// parsing; any request that needs a CUDA stream is refused there.
#pragma once

#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <aether/dtype/dlpack.h>
#include <aether/interop/Buffer.h>

#ifndef EAGLE_CPU_ONLY
#include <cuda_runtime.h>
#endif

#include "gref_layout.h"

namespace eagle {
namespace interop {

using aether::interop::Access;
using aether::interop::BufferView;
using aether::interop::Requirements;

// The DLPack stream codes with a fixed meaning.
inline constexpr std::intptr_t kStreamLegacy    = 1;
inline constexpr std::intptr_t kStreamPerThread = 2;
inline constexpr std::intptr_t kStreamNoOrder   = -1;

// The error every refusal of this layer raises.
struct InteropError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

// A stream argument resolved against the DLPack codes.
struct ResolvedStream {
    bool order = true;            // false for -1: no ordering requested
    std::intptr_t handle = 0;     // the raw cudaStream_t value (when `order`)
};

// Resolve a DLPack stream argument for a CUDA buffer. `0` is refused.
inline ResolvedStream resolve_stream(std::optional<std::intptr_t> stream)
{
    const std::intptr_t code = stream.value_or(kStreamLegacy);
    if (code == 0)
        throw InteropError("stream: 0 is ambiguous and refused by DLPack; pass 1 for the legacy default "
                           "stream or 2 for the per-thread default stream");
    if (code == kStreamNoOrder)
        return ResolvedStream{ false, 0 };
    if (code < -1)
        throw InteropError("stream: required -1, 1, 2 or a stream handle, got " + std::to_string(code));
    return ResolvedStream{ true, code };
}

// Refuse a stream argument for a CPU buffer.
inline void check_host_stream(std::optional<std::intptr_t> stream)
{
    if (stream.has_value())
        throw InteropError("stream: a CPU buffer takes no stream, got " + std::to_string(*stream));
}

#ifndef EAGLE_CPU_ONLY

namespace detail {

inline cudaStream_t as_cuda_stream(std::intptr_t code)
{
    if (code == kStreamLegacy)
        return cudaStreamLegacy;
    if (code == kStreamPerThread)
        return cudaStreamPerThread;
    return reinterpret_cast<cudaStream_t>(code);
}

inline void check_cuda(cudaError_t st, const char* what)
{
    if (st != cudaSuccess)
        throw std::runtime_error(std::string("eagle::interop: ") + what + ": " + cudaGetErrorString(st));
}

// Restores the calling thread's current device on scope exit.
class DeviceScope {
public:
    explicit DeviceScope(int device)
    {
        check_cuda(cudaGetDevice(&previous_), "cudaGetDevice");
        if (device != previous_)
            check_cuda(cudaSetDevice(device), "cudaSetDevice");
    }
    ~DeviceScope() { (void)cudaSetDevice(previous_); }
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;

private:
    int previous_ = 0;
};

} // namespace detail

// A per-device pool of timing-free events. Its events are never destroyed:
// the pool lives for the whole process and destroying an event after the
// CUDA context is gone is an error.
class EventPool {
public:
    static EventPool& instance()
    {
        static EventPool* pool = new EventPool();
        return *pool;
    }

    // An event of the current device (created only when the pool is empty).
    cudaEvent_t acquire(int device)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& free = free_[device];
            if (!free.empty()) {
                cudaEvent_t ev = free.back();
                free.pop_back();
                return ev;
            }
            ++created_;
        }
        cudaEvent_t ev = nullptr;
        detail::check_cuda(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreateWithFlags");
        return ev;
    }

    void release(int device, cudaEvent_t ev)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        free_[device].push_back(ev);
    }

    // How many events the pool has created so far.
    std::size_t created() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return created_;
    }

private:
    EventPool() = default;
    mutable std::mutex mutex_;
    std::unordered_map<int, std::vector<cudaEvent_t>> free_;
    std::size_t created_ = 0;
};

// Order everything queued so far on `producer` before anything queued later
// on `consumer` (both DLPack stream arguments, see the codes above), on
// `device`. A no-op when either side is -1 or when both name the same
// stream. Never blocks the host.
inline void fence(std::optional<std::intptr_t> producer, std::optional<std::intptr_t> consumer, int device)
{
    const ResolvedStream p = resolve_stream(producer);
    const ResolvedStream c = resolve_stream(consumer);
    if (!p.order || !c.order || p.handle == c.handle)
        return;
    detail::DeviceScope scope(device);
    EventPool& pool = EventPool::instance();
    cudaEvent_t ev = pool.acquire(device);
    const cudaError_t rec = cudaEventRecord(ev, detail::as_cuda_stream(p.handle));
    const cudaError_t wait = rec == cudaSuccess ? cudaStreamWaitEvent(detail::as_cuda_stream(c.handle), ev, 0) : rec;
    pool.release(device, ev);
    detail::check_cuda(rec, "cudaEventRecord");
    detail::check_cuda(wait, "cudaStreamWaitEvent");
}

#endif // EAGLE_CPU_ONLY

// A buffer plus the stream its contents are ordered on: the producer side of
// every later export and re-fence. `stream` is empty for a CPU buffer.
struct OrderedBuffer {
    BufferView buffer;
    std::optional<std::intptr_t> stream;
};

// A fence implemented outside this build, per DLPack device type: a pure C++
// build (EAGLE_CPU_ONLY) that reaches a device runtime through a separately
// loaded backend installs one here for each device type it can route. It orders
// `producer` before `consumer` (DLPack stream arguments, empty = none given) on
// device (`dl_device_type`, `device_id`) and throws on failure.
using FenceProvider = void (*)(std::int32_t dl_device_type, std::int32_t device_id,
    std::optional<std::intptr_t> producer, std::optional<std::intptr_t> consumer);

namespace detail {
inline FenceProvider* fence_providers()
{
    static FenceProvider table[64] = {};
    return table;
}
} // namespace detail

// Install (or, with nullptr, remove) the fence provider of DLPack device type
// `dl_device_type`. Install before any buffer of that type is re-fenced.
inline void set_fence_provider(std::int32_t dl_device_type, FenceProvider provider)
{
    if (dl_device_type < 0 || dl_device_type >= 64)
        throw std::invalid_argument("set_fence_provider: DLPack device type out of range");
    detail::fence_providers()[dl_device_type] = provider;
}

// The fence provider of `dl_device_type`, or nullptr.
inline FenceProvider fence_provider(std::int32_t dl_device_type)
{
    if (dl_device_type < 0 || dl_device_type >= 64)
        return nullptr;
    return detail::fence_providers()[dl_device_type];
}

// Re-fence before a launch that reads `b` on `consumer`: for producers that
// keep writing between launches on the stream the buffer was ordered on. A
// no-op for a CPU buffer (which takes no consumer stream).
inline void refence(const OrderedBuffer& b, std::optional<std::intptr_t> consumer)
{
    if (b.buffer.view.device.type() == kDLCPU) {
        check_host_stream(consumer);
        return;
    }
#ifndef EAGLE_CPU_ONLY
    fence(b.stream, consumer, b.buffer.view.device.id());
#else
    const auto type = static_cast<std::int32_t>(b.buffer.view.device.type());
    if (FenceProvider provider = fence_provider(type)) {
        provider(type, static_cast<std::int32_t>(b.buffer.view.device.id()), b.stream, consumer);
        return;
    }
    throw InteropError("device: a CUDA buffer needs a CUDA build of eagle, this build is CPU-only");
#endif
}

// Export `b` as a versioned DLPack tensor for a consumer that will use it on
// `consumer` (a DLPack stream argument). CUDA: fences the buffer's stream
// onto `consumer` first. CPU: `consumer` must be empty. Zero-copy; the
// read-only flag follows `BufferView::writable()`.
inline DLManagedTensorVersioned* export_versioned(const OrderedBuffer& b, std::optional<std::intptr_t> consumer)
{
    refence(b, consumer);
    return aether::interop::exportDLPack(b.buffer);
}

// The legacy-struct twin of `export_versioned` (refuses a read-only buffer,
// see `aether::interop::exportDLPackLegacy`).
inline DLManagedTensor* export_legacy(const OrderedBuffer& b, std::optional<std::intptr_t> consumer)
{
    refence(b, consumer);
    return aether::interop::exportDLPackLegacy(b.buffer);
}

// The plugin mirror of a validated (W, N) float64 buffer: C-contiguous,
// 8-byte aligned, CPU or CUDA. Writable when `writable` is set (a `Mutable`
// role); any access otherwise (a `lookup` role reads only).
inline plugin::GRefMirror gref_from_buffer(const BufferView& b, std::int64_t width, bool writable)
{
    Requirements req;
    req.dtype = aether::DType(DLDataType{ kDLFloat, 64, 1 });
    req.contiguous = true;
    req.alignment = 8;
    req.writable = writable;
    if (b.view.rank != 2) {
        throw InteropError("shape: required a rank-2 (W, N) buffer, got rank " + std::to_string(b.view.rank));
    }
    if (static_cast<std::int64_t>(b.view.extents[0]) != width) {
        throw InteropError("shape: required " + std::to_string(width) + " components, got "
            + std::to_string(static_cast<std::int64_t>(b.view.extents[0])));
    }
    if (b.view.device.type() != kDLCPU && b.view.device.type() != kDLCUDA) {
        throw InteropError("device: required cpu or cuda, got "
            + aether::interop::deviceTypeName(b.view.device.type()));
    }
    const std::vector<std::string> r = aether::interop::refusals(b, req);
    if (!r.empty()) {
        std::string msg = "eagle::interop::gref_from_buffer: ";
        for (std::size_t i = 0; i < r.size(); ++i)
            msg += (i ? "; " : "") + r[i];
        throw InteropError(msg);
    }
    plugin::GRefMirror g;
    const auto n = static_cast<std::uint64_t>(b.view.extents[1]);
    g.data_ = static_cast<double*>(b.view.data);
    g.samples_ = n;
    g.compStride_ = n;
    g.sampleStride_ = 1;
    g.deviceType_ = b.view.device.type() == kDLCPU ? plugin::kEagleAbiDeviceCPU : plugin::kEagleAbiDeviceCUDA;
    g.deviceId_ = b.view.device.id();
    return g;
}

} // namespace interop
} // namespace eagle
