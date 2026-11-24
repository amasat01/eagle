// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"

namespace eagle {
namespace filtering {
namespace detail {

/* Scan core routines of the host scanner (``cpu::Scan``). */
struct ScanCore {
    using Self = ScanCore;

    static constexpr unsigned int warpSize = 32;

    static constexpr unsigned int blockSize = 256;

    template<unsigned int blockSize_ = blockSize>
    AETHER_DEVICEHOST()
    static idx_t countBlocks(const idx_t N, const idx_t bsize = blockSize_)
    {
        return (N + bsize - 1) / bsize;
    }
};

} // namespace detail
} // namespace filtering
} // namespace eagle
