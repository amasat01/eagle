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
#include <cmath>
#include <cstdint>

namespace eagle {
namespace policy {

/** @brief Cycles of dependent arithmetic a step's chain exposes to one warp. */
inline constexpr int PERSIST_CHAIN_CYCLES = 8;
/** @brief Fewest warps per scheduler partition a persistent launch ever uses. */
inline constexpr int PERSIST_MIN_WARPS = 2;
/** @brief Threads per SM are granted in multiples of this. */
inline constexpr int PERSIST_THREAD_STEP = 64;
/** @brief The step fill assumed until a run has measured one. */
inline constexpr double PERSIST_FIRST_FILL = 0.5;
/** @brief Fewest and most steps one work-stealing chunk runs. */
inline constexpr int PERSIST_CHUNK_MIN = 2;
inline constexpr int PERSIST_CHUNK_MAX = 32;
/** @brief Cycles of fetch latency the warps of a persistent launch must cover. */
inline constexpr double PERSIST_FETCH_CYCLES = 2000.0;
/** @brief Most warps per partition added to cover a chunk head's fetch. */
inline constexpr int PERSIST_FETCH_WARPS_MAX = 3;

/** @brief Compaction's fixed cost: nine small kernels, in seconds. */
inline constexpr double COMPACTION_FLOOR_S = 27e-6;
/** @brief Compaction's cost per live sample, in seconds. */
inline constexpr double COMPACTION_PER_SAMPLE_S = 32e-12;
/** @brief The gap a compaction adds between its kernels in a captured graph, in seconds. */
inline constexpr double GRAPH_NODE_S = 62e-6;
/**
 * @brief Share of a step's arithmetic still exposed to dead lanes when the step
 *        is bound by memory: the rest hides under the loads.
 */
inline constexpr double BAND_HIDDEN_WASTE = 0.25;

/** @brief The device facts the launch rules read. A zero field is "not reported". */
struct LaunchDevice {
    ResidencyDevice residency;
    int sms = 0;
    int fp64Ratio = 0;      ///< FP32:FP64 throughput ratio (1 when not reported)
    int clockKhz = 0;
    int memClockKhz = 0;    ///< memory clock; with busWidthBits gives the bandwidth, 0 = not reported
    int busWidthBits = 0;
};

/** @brief Scheduler partitions and FP32 lanes per SM, where they differ from 4 x 32. */
struct SmLayout { int partitions; int lanes; };

inline constexpr SmLayout smLayout(int ccMajor, int ccMinor) {
    if (ccMajor == 6 && ccMinor == 0) return { 2, 64 };
    if ((ccMajor == 7 && (ccMinor == 0 || ccMinor == 2 || ccMinor == 5)) || (ccMajor == 8 && ccMinor == 0))
        return { 4, 64 };
    return { 4, 128 };
}

/**
 * @brief SM cycles one warp's claim of work holds the chip: the same-address
 *        atomic every claim contends on. Pascal's L2 serialises them about an
 *        order of magnitude slower than Volta and later.
 */
inline constexpr int claimCycles(int ccMajor) { return ccMajor < 7 ? 12 : 1; }

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
 * Once the chunk is known, one to PERSIST_FETCH_WARPS_MAX more warps per
 * partition are added to cover the latency of a chunk head's fetch. @p chunk and @p ops (the step's
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
    // a head's fetch latency is covered by as many warps as its chunk's arithmetic falls short of it
    int fetchWarps = 0;
    if (chunk > 0 && ops > 0.0) {
        const double cover = chunk * ops * issueCycles(dev, fp64);
        fetchWarps = static_cast<int>(std::clamp(std::ceil(PERSIST_FETCH_CYCLES / cover), 1.0,
                                                 static_cast<double>(PERSIST_FETCH_WARPS_MAX)));
    }
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
 * The power of two nearest sqrt(8 * m * SMs * claimCycles / (ops * issue)),
 * held to [PERSIST_CHUNK_MIN, PERSIST_CHUNK_MAX]: the minimiser of the lane
 * steps a chunk wastes masked at its end (c / 2 per lane) against the claims
 * (m * t_claim / (32 c)), every claim holding the chip claimCycles cycles
 * and the SMs sharing the one counter. Written on the square so no root is
 * taken. An unknown @p ops (0) or SM count takes the longest chunk.
 */
inline int persistChunk(const LaunchDevice& dev, bool fp64, double ops, double meanSteps) {
    if (!(ops > 0.0) || dev.sms <= 0) return PERSIST_CHUNK_MAX;
    const double x2 = 8.0 * meanSteps * dev.sms * claimCycles(dev.residency.ccMajor) /
                      (ops * issueCycles(dev, fp64));
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
 * @brief Seconds of one step of @p live samples in a full-live launch on the
 *        arithmetic alone: @p stepOps issue slots per sample over the SMs'
 *        issue pipes; 0 when the clock or the SM count is unknown.
 */
inline double bandComputeStepS(const LaunchDevice& dev, bool fp64, long long live, double stepOps) {
    if (dev.clockKhz <= 0 || dev.sms <= 0 || stepOps <= 0) return 0.0;
    const SmLayout layout = smLayout(dev.residency.ccMajor, dev.residency.ccMinor);
    const double warps = static_cast<double>(live) / RESIDENCY_WARP_LANES;
    return warps * stepOps * issueCycles(dev, fp64) /
           (static_cast<double>(layout.partitions) * dev.sms * dev.clockKhz * 1e3);
}

/**
 * @brief Seconds one step of @p live samples takes to move @p bytesPerSample
 *        bytes each through device memory (memClockKhz * busWidthBits * 2 / 8
 *        bytes per second); 0 when the memory clock, the bus width or the
 *        bytes are unknown.
 */
inline double bandMemoryStepS(const LaunchDevice& dev, long long live, double bytesPerSample) {
    if (dev.memClockKhz <= 0 || dev.busWidthBits <= 0 || bytesPerSample <= 0) return 0.0;
    const double bandwidth = static_cast<double>(dev.memClockKhz) * 1e3 * dev.busWidthBits * 2.0 / 8.0;
    return static_cast<double>(live) * bytesPerSample / bandwidth;
}

/**
 * @brief Seconds of one full-live step: the longer of its arithmetic and its
 *        memory traffic; 0 when the clock or the SM count is unknown.
 */
inline double bandStepS(const LaunchDevice& dev, bool fp64, long long live, double stepOps,
                        double bytesPerSample = 0.0) {
    const double compute = bandComputeStepS(dev, fp64, live, stepOps);
    return compute > 0.0 ? std::max(compute, bandMemoryStepS(dev, live, bytesPerSample)) : 0.0;
}

/**
 * @brief R from the cost model: one compaction of @p live samples over the
 *        step it is weighed against; 0 when the clock or the SM count is
 *        unknown.
 *
 * The compaction costs its fixed part, its per-sample part and the graph gap.
 * What it saves is the arithmetic dead lanes would have run, so it is priced
 * against the compute time of a step: a step bound by compute is the whole
 * step, a step bound by memory hides most of it (only BAND_HIDDEN_WASTE of
 * the compute time is exposed). The memory term is read from
 * @p bytesPerSample (0: not reported, the step is taken as compute-bound).
 */
inline double bandRate(const LaunchDevice& dev, bool fp64, long long live, double stepOps,
                       double bytesPerSample = 0.0) {
    const double compute = bandComputeStepS(dev, fp64, live, stepOps);
    if (compute <= 0.0) return 0.0;
    const double memory = bandMemoryStepS(dev, live, bytesPerSample);
    const double exposed = memory > compute ? BAND_HIDDEN_WASTE * compute : compute;
    return (COMPACTION_FLOOR_S + COMPACTION_PER_SAMPLE_S * static_cast<double>(live) + GRAPH_NODE_S) / exposed;
}

/** @brief Share of a persistent launch's issue slots that do useful work once its lanes keep fetching. */
inline constexpr double ENTRY_PERSIST_EFFICIENCY = 0.85;
/** @brief Share of the step budget a fused launch runs on average (its lanes stop at their own last step). */
inline constexpr double ENTRY_FUSED_SHARE = 0.9;
/** @brief Persist over fused time below which the persistent entry is taken without timing the other. */
inline constexpr double ENTRY_PERSIST_BELOW = 0.8;
/** @brief Persist over fused time above which the fused entry is taken without timing the other. */
inline constexpr double ENTRY_FUSED_ABOVE = 1.25;

/**
 * @brief Seconds one thread's step of @p ops operations costs when every SM's
 *        issue pipes are full; 0 when the clock, SM count or @p ops is unknown.
 */
inline double threadStepS(const LaunchDevice& dev, bool fp64, double ops) {
    if (dev.clockKhz <= 0 || dev.sms <= 0 || !(ops > 0.0)) return 0.0;
    const SmLayout layout = smLayout(dev.residency.ccMajor, dev.residency.ccMinor);
    return ops * issueCycles(dev, fp64) /
           (static_cast<double>(RESIDENCY_WARP_LANES) * layout.partitions * dev.sms * dev.clockKhz * 1e3);
}

/**
 * @brief Seconds one warp takes to run a step of @p ops operations alone: its
 *        dependent chain, PERSIST_CHAIN_CYCLES cycles an operation; 0 when
 *        the clock or @p ops is unknown.
 */
inline double chainStepS(const LaunchDevice& dev, double ops) {
    if (dev.clockKhz <= 0 || !(ops > 0.0)) return 0.0;
    return ops * PERSIST_CHAIN_CYCLES / (static_cast<double>(dev.clockKhz) * 1e3);
}

/**
 * @brief Predicted wall of the persistent entry over the fused entry for a
 *        batch of @p n samples whose longest sample ran @p steps steps and
 *        whose steps fill a fraction @p fill of that; 0 when unknown.
 *
 * The fused entry runs the batch in waves of @p residentThreads (all the
 * threads the device holds at the entry's registers); a wave takes the longer
 * of its dependent chain and its issue slots, both over the share of the budget
 * the lanes run. The persistent entry issues only the steps the samples take
 * (n * fill * steps) at ENTRY_PERSIST_EFFICIENCY, plus one chain for the tail.
 * Compare the result with ENTRY_PERSIST_BELOW and ENTRY_FUSED_ABOVE.
 */
inline double persistOverFused(const LaunchDevice& dev, bool fp64, long long residentThreads, long long n,
                               long long steps, double fill, double ops) {
    const double tThr = threadStepS(dev, fp64, ops);
    const double tLat = chainStepS(dev, ops);
    if (tThr <= 0.0 || tLat <= 0.0 || n <= 0 || steps <= 0 || !(fill > 0.0) || residentThreads <= 0) return 0.0;
    const double waves = std::ceil(static_cast<double>(n) / static_cast<double>(residentThreads));
    const double perWave = static_cast<double>(n) / waves;
    const double fused = waves * std::max(ENTRY_FUSED_SHARE * steps * tLat,
                                          perWave * ENTRY_FUSED_SHARE * steps * tThr);
    const double persist = static_cast<double>(n) * fill * steps * tThr / ENTRY_PERSIST_EFFICIENCY + steps * tLat;
    return persist / fused;
}

/** @brief True when a launch asking for @p demandThreads would not fit at the unbounded residency. */
inline bool boundDemanded(int demandThreads, int residentUnbounded) {
    return residentUnbounded > 0 && demandThreads > residentUnbounded;
}

}  // namespace policy
}  // namespace eagle
