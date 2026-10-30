// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file DriverEntry.h
 * @brief Driver-API entry points resolved through the CUDA runtime, so nothing
 *        that includes eagle links libcuda.
 */

#ifndef EAGLE_CPU_ONLY

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <stdexcept>
#include <string>

namespace eagle {
namespace cuda {
namespace detail {

/**
 * @brief Resolve a Driver-API entry point at first use, through the CUDA runtime.
 *
 * Nothing that includes this header links libcuda: a process on a machine with no
 * NVIDIA driver loads fine and can run the host structures. Only asking for device
 * execution needs the driver, and that is where a missing driver or GPU is
 * reported, in words a user can act on.
 *
 * The symbol is asked for at its CUDA 12.0 ABI, never at the toolkit's own
 * CUDA_VERSION: a CUDA 13 driver refuses a version above its own (a 13.0 driver
 * answers "invalid argument" for 13040), so a build with a newer toolkit than the
 * installed driver -- CUDA minor-version compatibility -- would lose every launch.
 * The entry points eagle uses have not changed their ABI since 12.0.
 */
inline constexpr unsigned int DRIVER_ENTRY_ABI = 12000;

template <class Fn>
Fn driver_entry(const char* symbol) {
#ifdef CUDA_API_PER_THREAD_DEFAULT_STREAM
    constexpr unsigned long long flags = cudaEnablePerThreadDefaultStream;
#else
    constexpr unsigned long long flags = cudaEnableLegacyStream;
#endif
    void* fn = nullptr;
#if CUDART_VERSION >= 12050
    cudaDriverEntryPointQueryResult found{};
    const cudaError_t err =
        cudaGetDriverEntryPointByVersion(symbol, &fn, DRIVER_ENTRY_ABI, flags, &found);
    const bool ok = err == cudaSuccess && found == cudaDriverEntryPointSuccess;
#elif CUDART_VERSION >= 12000
    cudaDriverEntryPointQueryResult found{};
    const cudaError_t err = cudaGetDriverEntryPoint(symbol, &fn, flags, &found);
    const bool ok = err == cudaSuccess && found == cudaDriverEntryPointSuccess;
#else
    const cudaError_t err = cudaGetDriverEntryPoint(symbol, &fn, flags);
    const bool ok = err == cudaSuccess;
#endif
    if (!ok || fn == nullptr) {
        const std::string why = err != cudaSuccess
            ? std::string(cudaGetErrorString(err))
            : std::string("the driver does not provide ") + symbol;
        const bool noDriver = err == cudaErrorInsufficientDriver || err == cudaErrorNoDevice ||
                              err == cudaErrorInitializationError;
        if (noDriver)
            throw std::runtime_error(
                "eagle: device execution needs an NVIDIA driver and a CUDA-capable GPU, "
                "and none is usable here (" + why + "). Host execution "
                "(eagle::exec::HostTeam) does not need one.");
        throw std::runtime_error(std::string("eagle: the NVIDIA driver could not provide ") + symbol +
                                 " (" + why + ").");
    }
    return reinterpret_cast<Fn>(fn);
}

/** @brief `cuGetErrorString`'s text for @p r, or "?" when even that is unavailable. */
inline const char* driver_error_text(CUresult r) {
    static const auto get = driver_entry<decltype(&::cuGetErrorString)>("cuGetErrorString");
    const char* msg = nullptr;
    get(r, &msg);
    return msg ? msg : "?";
}

/**
 * @brief The context an out-of-memory failure needs, read at the moment of failure: the
 *        local bytes per thread of @p fn (0: unknown) and the current device's free/total
 *        memory, as text for the message. Never throws.
 */
inline std::string oom_context(CUfunction fn) {
    std::string out;
    try {
        if (fn != nullptr) {
            static const auto attr = driver_entry<decltype(&::cuFuncGetAttribute)>("cuFuncGetAttribute");
            int local = 0;
            if (attr(&local, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, fn) == CUDA_SUCCESS)
                out += "kernel local memory " + std::to_string(local) + " bytes per thread; ";
        }
        static const auto info = driver_entry<decltype(&::cuMemGetInfo)>("cuMemGetInfo");
        std::size_t freeBytes = 0, totalBytes = 0;
        if (info(&freeBytes, &totalBytes) == CUDA_SUCCESS)
            out += "GPU memory " + std::to_string(freeBytes >> 20) + " MiB free of "
                + std::to_string(totalBytes >> 20) + " MiB total";
    } catch (...) {
    }
    return out;
}

} // namespace detail
} // namespace cuda
} // namespace eagle

#endif // EAGLE_CPU_ONLY
