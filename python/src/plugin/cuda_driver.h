// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file cuda_driver.h
 * @brief Plugin-private view of the driver-API translation layer
 *        (``cuda_driver.cpp``): what the seam entries need to know about the
 *        driver beyond the CUDA runtime calls the layer implements.
 */

#include <cstdint>

namespace eagle_cuda_plugin {

/**
 * @brief The capability groups (``EAGLE_BACKEND_CAP_*`` bits of @p wanted)
 *        whose driver entry points all resolve. A group with a missing entry
 *        point (an older driver, or a name listed in ``EAGLE_CUDA_DRIVER_HIDE``)
 *        is cleared. With no loadable driver at all, @p wanted is returned
 *        unchanged: every group then reports the missing driver itself, as a
 *        typed unavailable status, when it is used.
 */
std::uint64_t driverGroups(std::uint64_t wanted);

} // namespace eagle_cuda_plugin
