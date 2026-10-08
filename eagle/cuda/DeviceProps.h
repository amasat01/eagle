// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/DeviceProps.h"
#include "eagle/util/DeviceError.h"

#ifndef EAGLE_CPU_ONLY

#include <cuda_runtime_api.h>

namespace eagle {
namespace cuda {

/**
 * @brief Query ``cudaGetDeviceProperties`` for @p device and derive the
 * roofline fields through :func:`eagle::deriveFields`. The one
 * live-hardware source of an :class:`eagle::DeviceProps` — ``eagle::cpu::
 * deviceProps()`` (``eagle/cpu/DeviceProps.h``) is the GPU-free twin.
 *
 * @note Available only when ``EAGLE_CPU_ONLY`` is not defined.
 */
inline DeviceProps deviceProps(int device = 0)
{
    cudaDeviceProp prop{};
    EAGLE_CHECK_ALWAYS(cudaGetDeviceProperties(&prop, device));

    DeviceProps props;
    props.name                  = prop.name;
    props.cc_major              = prop.major;
    props.cc_minor              = prop.minor;
    props.sm_count               = prop.multiProcessorCount;
    // CUDA 13 dropped the two clock rates from cudaDeviceProp; the attribute
    // query answers on every toolkit.
    EAGLE_CHECK_ALWAYS(cudaDeviceGetAttribute(&props.clock_rate_khz, cudaDevAttrClockRate, device));
    EAGLE_CHECK_ALWAYS(cudaDeviceGetAttribute(&props.memory_clock_rate_khz, cudaDevAttrMemoryClockRate, device));
    props.memory_bus_width_bits = prop.memoryBusWidth;
    props.shared_mem_per_block  = prop.sharedMemPerBlock;
    props.shared_mem_per_sm     = prop.sharedMemPerMultiprocessor;
    props.regs_per_block         = prop.regsPerBlock;
    props.regs_per_sm            = prop.regsPerMultiprocessor;
    props.warp_size               = prop.warpSize;

    return deriveFields(props);
}

} // namespace cuda
} // namespace eagle

#endif // EAGLE_CPU_ONLY
