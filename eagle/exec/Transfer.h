// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The four stream-ordered host/device operations the backend seam exposes, in
// one place with both faces: a copy by direction, pinned host memory, and a
// host function on a stream. The CUDA face (`cudaMemcpyAsync`, `cudaHostAlloc`,
// `cudaLaunchHostFunc`) is compiled only without EAGLE_CPU_ONLY. The host face
// is the same call on plain memory: `memcpy`, an aligned `malloc`, a direct
// call, and the stream is ignored because every host launch is synchronous.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <new>

#ifndef EAGLE_CPU_ONLY
#include <cuda_runtime_api.h>

#include "eagle/util/DeviceError.h"
#endif

namespace eagle {
namespace exec {

/** @brief Direction of a copy; the values are the seam's `EAGLE_BACKEND_COPY_*`. */
enum class CopyKind : std::int32_t { HostToDevice = 0, DeviceToHost = 1, DeviceToDevice = 2, HostToHost = 3 };

/** @brief A host function: called once per execution, with the pointer given at enqueue. */
using HostFn = void (*)(void*);

/** @brief Whether @p kind is one of the four directions. */
inline bool validCopyKind(std::int32_t kind) { return kind >= 0 && kind <= 3; }

/**
 * @brief Copy @p bytes from @p src to @p dst, ordered on @p stream (a raw
 *        `cudaStream_t`; ignored by the host face).
 *
 * Capture-legal: inside a capture it becomes a memcpy node. Source and
 * destination are distinct, non-overlapping storage, or the same pointer, which
 * is a no-op and enqueues nothing. A zero length is a no-op too.
 */
inline void memcpyAsync(void* dst, const void* src, std::size_t bytes, CopyKind kind, void* stream)
{
    if (bytes == 0 || dst == src)
        return;
#ifdef EAGLE_CPU_ONLY
    (void)kind;
    (void)stream;
    std::memcpy(dst, src, bytes);
#else
    static constexpr cudaMemcpyKind kinds[] = { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
        cudaMemcpyDeviceToDevice, cudaMemcpyHostToHost };
    EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(dst, src, bytes, kinds[static_cast<std::int32_t>(kind)],
        static_cast<cudaStream_t>(stream)));
#endif
}

/** @brief @p bytes of host memory the device can DMA from (page-locked); the host face is aligned `malloc`. */
inline void* hostAlloc(std::size_t bytes)
{
    void* p = nullptr;
#ifdef EAGLE_CPU_ONLY
    constexpr std::size_t align = 64;
    p = std::aligned_alloc(align, (bytes + align - 1) / align * align);
    if (p == nullptr)
        throw std::bad_alloc();
#else
    EAGLE_CHECK_ALWAYS(cudaHostAlloc(&p, bytes, cudaHostAllocPortable));
#endif
    return p;
}

/** @brief Release memory from hostAlloc() (null is fine). */
inline void hostFree(void* p)
{
#ifdef EAGLE_CPU_ONLY
    std::free(p);
#else
    if (p != nullptr)
        EAGLE_CHECK_ALWAYS(cudaFreeHost(p));
#endif
}

/**
 * @brief Call `fn(arg)` once the work enqueued before it on @p stream is done.
 *
 * Capture-legal: inside a capture it becomes a host node that runs on every
 * replay. @p fn must not call CUDA, and @p arg must outlive every execution.
 * The host face calls it now.
 */
inline void hostFunc(void* stream, HostFn fn, void* arg)
{
#ifdef EAGLE_CPU_ONLY
    (void)stream;
    fn(arg);
#else
    EAGLE_CHECK_ALWAYS(cudaLaunchHostFunc(static_cast<cudaStream_t>(stream), reinterpret_cast<cudaHostFn_t>(fn), arg));
#endif
}

}  // namespace exec
}  // namespace eagle
