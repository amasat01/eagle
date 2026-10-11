// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// How many threads of one kernel an SM holds, and where a register bound pays.
//
// A warp's registers are granted in units of 256, and each of the SM's
// sub-partitions (four on every architecture here, two on cc 6.0) owns an equal
// share of the register file. So the warps an SM holds step down at certain
// register counts ("edges"), and `registerFile / registersPerThread` overstates
// them between edges. This is the one place residency is computed; every launch
// decision (block size, persistent geometry, capacity, the cost model's lane
// rate) reads it. Plain integer arithmetic, identical on host and device builds.
#pragma once

#include <algorithm>

namespace eagle {
namespace policy {

/** @brief Lanes per warp. */
inline constexpr int RESIDENCY_WARP_LANES = 32;
/** @brief Registers per thread one 256-register grant covers (256 / 32 lanes). */
inline constexpr int REGISTER_UNIT = 8;

/** @brief The device facts residency reads. A zero field is "not reported". */
struct ResidencyDevice {
    int ccMajor = 0;
    int ccMinor = 0;
    int regsPerSm = 0;        ///< 32-bit registers in the SM's file
    int maxThreadsPerSm = 0;  ///< resident-thread limit; 0 leaves the register file as the only limit
};

/** @brief Scheduler sub-partitions per SM: two on cc 6.0, four everywhere else. */
inline constexpr int subPartitions(int ccMajor, int ccMinor) {
    return (ccMajor == 6 && ccMinor == 0) ? 2 : 4;
}

/** @brief Resident blocks an SM allows whatever their size. */
inline constexpr int maxBlocksPerSm(int ccMajor, int ccMinor) {
    if (ccMajor == 7 && ccMinor == 5) return 16;
    if ((ccMajor == 8 && ccMinor >= 6) || ccMajor == 12) return 24;
    return 32;
}

/** @brief Warps one SM holds at @p regs registers per thread; 0 when unknown. */
inline int residentWarps(const ResidencyDevice& dev, int regs) {
    if (regs <= 0 || dev.regsPerSm <= 0) return 0;
    const int parts = subPartitions(dev.ccMajor, dev.ccMinor);
    const int unitsPerPart = dev.regsPerSm / (parts * 256);
    const int unitsPerWarp = (regs + REGISTER_UNIT - 1) / REGISTER_UNIT;
    int warps = parts * (unitsPerPart / unitsPerWarp);
    if (dev.maxThreadsPerSm > 0) warps = std::min(warps, dev.maxThreadsPerSm / RESIDENCY_WARP_LANES);
    return warps;
}

/** @brief Threads one SM holds at @p regs registers per thread, ignoring block shape. */
inline int residentThreads(const ResidencyDevice& dev, int regs) {
    return residentWarps(dev, regs) * RESIDENCY_WARP_LANES;
}

/** @brief Blocks of @p block threads one SM holds at @p regs registers; 0 when none fit. */
inline int residentBlocks(const ResidencyDevice& dev, int regs, int block) {
    if (block <= 0 || block > 1024) return 0;
    const int perBlock = (block + RESIDENCY_WARP_LANES - 1) / RESIDENCY_WARP_LANES;
    return std::min(residentWarps(dev, regs) / perBlock, maxBlocksPerSm(dev.ccMajor, dev.ccMinor));
}

/**
 * @brief The register count a kernel at @p regs could drop to for a larger
 *        residency, when that is at most REGISTER_UNIT registers away; else 0.
 *
 * The edge is the top of the previous 8-register unit (41..48 -> 40,
 * 169..176 -> 168), taken only when the SM really holds more warps there.
 */
inline int registerEdge(const ResidencyDevice& dev, int regs) {
    if (regs <= 0 || regs > 255) return 0;
    const int edge = ((regs - 1) / REGISTER_UNIT) * REGISTER_UNIT;
    return edge > 0 && residentWarps(dev, edge) > residentWarps(dev, regs) ? edge : 0;
}

/** @brief What a compiled kernel needs per thread. */
struct KernelResources {
    int regs = 0;
    int localBytes = 0;
};

/**
 * @brief Keep a kernel recompiled under a register bound of @p edge only when
 *        it spills nothing it did not spill before and fits the bound.
 *
 * "Local size unchanged" is the test, never "small spill": a bound that adds
 * even one spilled double is refused.
 */
inline bool keepBounded(const KernelResources& free, const KernelResources& bounded, int edge) {
    return edge > 0 && bounded.localBytes == free.localBytes && bounded.regs <= edge;
}

}  // namespace policy
}  // namespace eagle
