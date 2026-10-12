// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// How a launch is sized and paced: the threads a persistent launch puts on each
// SM, the steps one work-stealing chunk runs, and the steps between two
// compactions of a banded run.
//
// Every rule reads only the device class (compute capability, FP64 rate), the
// precision, the kernel's registers and per-step operation count, the batch
// size n, the step fill f and the mean steps m; none of them reads which kernel
// it is sizing. Plain arithmetic, identical on
// host and device builds, so the host loop and the device policy kernel share
// one answer.
#pragma once

#include "eagle/policy/Residency.h"

#include <algorithm>
#include <climits>
#include <cstdint>

namespace eagle {
namespace policy {

/** @brief Cycles of dependent arithmetic a step's chain exposes to one warp. */
inline constexpr int PERSIST_CHAIN_CYCLES = 16;
/** @brief Fewest warps per scheduler partition a persistent launch ever uses. */
inline constexpr int PERSIST_MIN_WARPS = 2;
/** @brief Threads per SM are granted in multiples of this. */
inline constexpr int PERSIST_THREAD_STEP = 64;
/** @brief The step fill assumed until a run has measured one. */
inline constexpr double PERSIST_FIRST_FILL = 0.5;
/** @brief Fewest and most steps one work-stealing chunk runs. */
inline constexpr int PERSIST_CHUNK_MIN = 2;
inline constexpr int PERSIST_CHUNK_MAX = 32;
/** @brief Cycles a chunk head costs a warp: the work fetch and its same-address atomic. */
inline constexpr double PERSIST_HEAD_CYCLES = 256.0;

/** @brief Compaction's fixed cost: nine small kernels, in seconds. */
inline constexpr double COMPACTION_FLOOR_S = 27e-6;
/** @brief Compaction's cost per live sample, in seconds. */
inline constexpr double COMPACTION_PER_SAMPLE_S = 32e-12;

/** @brief The device facts the launch rules read. A zero field is "not reported". */
struct LaunchDevice {
    ResidencyDevice residency;
    int sms = 0;
    int fp64Ratio = 0;      ///< FP32:FP64 throughput ratio (1 when not reported)
    int clockKhz = 0;
};

/** @brief Scheduler partitions and FP32 lanes per SM, where they differ from 4 x 32. */
struct SmLayout { int partitions; int lanes; };

inline constexpr SmLayout smLayout(int ccMajor, int ccMinor) {
    if (ccMajor == 6 && ccMinor == 0) return { 2, 64 };
    if ((ccMajor == 7 && (ccMinor == 0 || ccMinor == 2 || ccMinor == 5)) || (ccMajor == 8 && ccMinor == 0))
        return { 4, 64 };
    return { 4, 128 };
}

/** @brief Cycles one warp instruction of the step's precision takes to issue on the SM. */
inline int issueCycles(const LaunchDevice& dev, bool fp64) {
    const SmLayout layout = smLayout(dev.residency.ccMajor, dev.residency.ccMinor);
    int issue = std::max(1, RESIDENCY_WARP_LANES * layout.partitions / layout.lanes);
    if (fp64) issue *= std::max(1, dev.fp64Ratio);
    return issue;
}

/** @brief Threads one warp-per-partition layer of the SM holds (32 lanes x partitions). */
inline int warpLayer(const LaunchDevice& dev) {
    return RESIDENCY_WARP_LANES * smLayout(dev.residency.ccMajor, dev.residency.ccMinor).partitions;
}

/**
 * @brief Threads per SM at which the step's dependent chain keeps the issue
 *        pipe busy: ceil(1 + chain / issue) warps per partition (5 on a
 *        full-rate FP64 part, 2 on a 1:32 or 1:64 one).
 */
inline int persistSaturation(const LaunchDevice& dev, bool fp64) {
    const int warps = 1 + (PERSIST_CHAIN_CYCLES + issueCycles(dev, fp64) - 1) / issueCycles(dev, fp64);
    return warpLayer(dev) * warps;
}

/** @brief Threads per SM a persistent launch asks for before n and the registers cut it: the saturation count plus the fetch warp. */
inline int persistDemand(const LaunchDevice& dev, bool fp64) {
    return persistSaturation(dev, fp64) + warpLayer(dev);
}

/**
 * @brief Threads per SM of a persistent launch of @p n samples at @p regs
 *        registers and step fill @p f; 0 when the SM count is unknown.
 *
 * The saturation count, held between two bounds: at least PERSIST_MIN_WARPS
 * warps per partition, at most the residency ceiling and the drain limit
 * n * f / (2 * SMs) (past it, lanes outnumber the work and sit idle at the
 * end). When the ceiling is under the saturation count the launch takes the
 * whole ceiling.
 *
 * Once the chunk is known, one more warp per partition is added to cover the
 * latency of a chunk head's fetch. @p chunk and @p ops (the step's
 * floating-point operations) are 0 when unknown, which adds nothing.
 */
inline int persistThreads(const LaunchDevice& dev, bool fp64, int regs, long long n, double f,
                          int chunk = 0, double ops = 0.0) {
    if (dev.sms <= 0) return 0;
    const int resident = residentThreads(dev.residency, regs);
    const int ceiling = resident > 0 ? resident : INT_MAX;
    const double drain = static_cast<double>(n) * f / (2.0 * dev.sms);
    const int lowest = std::min(PERSIST_MIN_WARPS * warpLayer(dev), ceiling);
    const double hi = std::min<double>(ceiling, drain);
    // a head's fetch latency (~2000 cycles) is never covered by a chunk's arithmetic: always one warp more
    const int fetchWarps = (chunk > 0 && ops > 0.0) ? 1 : 0;
    const double saturation = persistSaturation(dev, fp64) + fetchWarps * warpLayer(dev);
    const double want = std::max<double>(lowest, std::min<double>(saturation, hi));
    const long long step = PERSIST_THREAD_STEP;
    const long long rounded = (static_cast<long long>(want + step - 1) / step) * step;
    return static_cast<int>(std::min<long long>(rounded, ceiling));
}

/**
 * @brief Steps one work-stealing chunk runs for samples that take @p meanSteps
 *        on average, each step @p ops floating-point operations of the given precision.
 *
 * The power of two nearest sqrt(2 * m * H / (ops * issue)), held to
 * [PERSIST_CHUNK_MIN, PERSIST_CHUNK_MAX]: the minimiser of the lanes a chunk
 * leaves idle at its end (c / 2m) against the head's cost per step
 * (H / (c * ops * issue)), H being PERSIST_HEAD_CYCLES. Written on the square
 * so no root is taken. An unknown @p ops (0) takes the longest chunk.
 */
inline int persistChunk(const LaunchDevice& dev, bool fp64, double ops, double meanSteps) {
    if (!(ops > 0.0)) return PERSIST_CHUNK_MAX;
    const double x2 = 2.0 * meanSteps * PERSIST_HEAD_CYCLES / (ops * issueCycles(dev, fp64));
    int chunk = PERSIST_CHUNK_MIN;
    while (chunk < PERSIST_CHUNK_MAX && x2 >= 2.0 * chunk * chunk) chunk *= 2;
    return chunk;
}

/**
 * @brief Steps to the next compaction of a banded run: the power of two nearest
 *        sqrt(2R / h), held inside [@p kMin, @p kMax].
 *
 * @p rate is R, the compaction's cost in steps of a full-live launch;
 * h = newly / (live * k) is the rate at which samples finish. Balancing the
 * waste of dead lanes (h * e / 2) against the compaction cost (R / e) gives
 * e = sqrt(2R / h). Written on e^2 so no square root is taken: the host loop and
 * the device policy kernel round the same way.
 */
inline long long bandK(double rate, long long newly, long long live, long long k, long long kMax,
                       long long kMin) {
    if (newly <= 0) return kMax;
    const double e2 = 2.0 * rate * static_cast<double>(live) * static_cast<double>(k) / static_cast<double>(newly);
    long long p = 1;
    while (p < (1LL << 30) && e2 >= 2.0 * static_cast<double>(p) * static_cast<double>(p)) p *= 2;
    return std::min(std::max(p, kMin), kMax);
}

/**
 * @brief R from the cost model: one compaction of @p live samples over one
 *        step of a full-live launch (@p stepOps issue slots per sample); 0 when
 *        the clock or the SM count is unknown.
 */
inline double bandRate(const LaunchDevice& dev, bool fp64, long long live, double stepOps) {
    if (dev.clockKhz <= 0 || dev.sms <= 0 || stepOps <= 0) return 0.0;
    const SmLayout layout = smLayout(dev.residency.ccMajor, dev.residency.ccMinor);
    const double warps = static_cast<double>(live) / RESIDENCY_WARP_LANES;
    const double stepS = warps * stepOps * issueCycles(dev, fp64) /
                         (static_cast<double>(layout.partitions) * dev.sms * dev.clockKhz * 1e3);
    return (COMPACTION_FLOOR_S + COMPACTION_PER_SAMPLE_S * static_cast<double>(live)) / stepS;
}

/** @brief True when a launch asking for @p demandThreads would not fit at the unbounded residency. */
inline bool boundDemanded(int demandThreads, int residentUnbounded) {
    return residentUnbounded > 0 && demandThreads > residentUnbounded;
}

}  // namespace policy
}  // namespace eagle
