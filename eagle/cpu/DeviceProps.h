// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/DeviceProps.h"

#include <thread>

namespace eagle {
namespace cpu {

/**
 * @brief Documented host values standing in for a real device query — no GPU
 * needed. ``sm_count`` is the one HONEST sysconf-backed field
 * (``std::thread::hardware_concurrency()``, the logical-CPU count, used as
 * this facility's SM-count analogue); every other raw field below is a
 * DOCUMENTED SYNTHETIC — there is no portable, dependency-free way to query a
 * host's clock speed or memory bandwidth from a header-only facility — chosen
 * so the derived ridge point stays FINITE and MODERATE: a classifier built on
 * this profile degrades toward calling most nodes roofline-bound (class M,
 * "add_concurrent siblings, never restructure") rather than confidently
 * memory- or compute-bound on hardware it never actually measured. This is a
 * WEAK bias from the chosen constants, not a formal guarantee.
 */
inline DeviceProps deviceProps()
{
    DeviceProps props;
    props.name = "cpu (OpenMP host)";
    props.cc_major = 0; // documented sentinel: no CUDA compute capability on a host profile
    props.cc_minor = 0;
    const unsigned hw = std::thread::hardware_concurrency();
    props.sm_count = hw > 0 ? static_cast<int>(hw) : 1; // honest sysconf-backed
    props.clock_rate_khz = 3'000'000; // documented synthetic: 3 GHz-class host core
    props.memory_clock_rate_khz = 2'666'000; // documented synthetic: DDR4-2666-class
    props.memory_bus_width_bits = 128; // documented synthetic: dual-channel width
    props.shared_mem_per_block = 0; // documented synthetic: not modeled on a host profile
    props.shared_mem_per_sm = 0;
    props.regs_per_block = 0; // documented synthetic: not modeled on a host profile
    props.regs_per_sm = 0;
    props.warp_size = 1; // documented synthetic: a host thread never diverges

    return deriveFields(props);
}

} // namespace cpu
} // namespace eagle
