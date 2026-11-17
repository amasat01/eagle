// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Fused or split? The pure rule that picks how a K-way dispatch group runs.
//
//   switch     one kernel; each sample runs its own branch inline. Always
//              correct, and the default.
//   segmented  samples are sorted into K runs by branch and each run executes
//              one branch alone, so no lane pays for a branch it does not take.
//
// The rule compares the predicted time of the two over the branch mix actually
// seen (`Mix`) and the structural cost of each branch (`CostVector`, counted
// from the kernel's IR). It enters `segmented` when the predicted gain reaches
// ENTER_GAIN and leaves it when the gain falls under LEAVE_GAIN; in between it
// keeps what it has, so a mix hovering at the threshold does not flip every
// time it is looked at. Everything here is plain arithmetic: no allocation, no
// state beyond the arguments, identical on host and device builds.
#pragma once

#include "eagle/DeviceProps.h"
#include "eagle/policy/Residency.h"
#include "eagle/policy/WeightRows.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace eagle {
namespace policy {

/** @brief Most branches one dispatch group may have. */
inline constexpr int MAX_BRANCHES = 16;
/** @brief Switch to `segmented` once the predicted gain reaches this. */
inline constexpr double ENTER_GAIN = 1.15;
/** @brief Fall back to `switch` once the predicted gain drops below this. */
inline constexpr double LEAVE_GAIN = 1.05;
/** @brief Steps between two looks at the mix. */
inline constexpr int DECISION_EVERY = 16;
/** @brief Lanes that run in lockstep on a device. */
inline constexpr int WARP_LANES = 32;

// Host compute is the slower of two limits: the throughput of the operations
// (`ops / rate`) and the dependent chain through them (`chain * chainSeconds`):
// a core cannot start an operation before the one it reads has finished, so work
// that is one long chain runs at the latency of its operations however many
// there are. A loop over samples the compiler vectorises runs its operations
// `vectorWidth` times faster and hides the chain across its lanes, so it pays none;
// one it leaves scalar pays the chain in full.
//
// The weights are rows of WeightRows.h, picked by the machine class (see weightRowIndex); the host rows were fitted by
// benchmarks/reproduce/fit_profile.py to the calibration of a 4-core Xeon W-2125 (relative RMS 0.19 over the isolated groups). The device
// built-in row is a conservative prior until a device calibration fits within 0.20.
/// Warps per SM above which the issue pipe stays busy: a kernel resident in fewer runs the lanes slower.
/// 8 is the knee measured on a P2000 (1:32 FP64 rate).
inline constexpr int ISSUE_KNEE_WARPS = 8;

enum class Policy { Switch, Segmented };
enum class Target { Host, Device };

/** @brief Work of one branch, per sample: counts of operations by class. */
struct BranchCost {
    double arith32 = 0;    ///< add/mul/compare/select ... on 32-bit or integer values
    double arith64 = 0;    ///< the same on 64-bit floats
    double special32 = 0;  ///< transcendental / library calls on 32-bit values
    double special64 = 0;  ///< the same on 64-bit floats
    double loads = 0;      ///< plane elements read
    double gathers = 0;    ///< indexed (scattered) plane reads
    double chain32 = 0;         ///< longest dependent path of arith32 ops
    double chain64 = 0;         ///< the same for arith64
    double chainSpecial32 = 0;  ///< dependent special32 ops on that path
    double chainSpecial64 = 0;  ///< the same for special64
    bool vectorised = false;    ///< a host loop over samples running only this branch is vectorised
    /// plane elements written; negative: not counted, and every plane read is taken to be written too
    double stores = -1;
};

/** @brief The branches of one dispatch group, as counted from its IR. */
struct CostVector {
    int count = 0;
    /// The fused host loop is vectorised: it evaluates every branch and selects,
    /// so it costs the sum of the branches whatever the mix (and its chain is the
    /// longest branch's: the branches are independent). Otherwise it is a scalar
    /// switch that runs each sample's own branch.
    bool fusedVectorised = false;
    int regsPerThread = 0;  ///< registers the compiled kernel needs per thread; 0 when unknown (full rate)
    std::array<BranchCost, MAX_BRANCHES> branch{};
};

/** @brief How many samples took each branch (exact integer counts), and the run around them. */
struct Mix {
    std::array<std::int64_t, MAX_BRANCHES> counts{};
    std::int64_t fusedSteps = 1;      ///< steps one fused launch covers; the split path launches per step
    std::int64_t remainingSteps = DECISION_EVERY;  ///< steps left to pay one class sort over
    double mixed32 = -1;              ///< measured fraction of 32-lane groups holding two or more
                                      ///< branches; negative when unmeasured (the iid prior is used)

    /** @brief Samples counted over the first @p branches branches. */
    std::int64_t total(int branches) const {
        std::int64_t sum = 0;
        for (int j = 0; j < branches; ++j) sum += counts[j];
        return sum;
    }
};

/** @brief Per-target constants; see defaultWeights for where the numbers come from. */
struct Weights {
    Target target = Target::Host;
    double arith32 = 1;       ///< one 32-bit operation, in 64-bit operations
    double special32 = 20;    ///< one 32-bit transcendental, in 64-bit operations
    double special64 = 20;    ///< one 64-bit transcendental, in 64-bit operations
    double elemBytes = 8;     ///< bytes of one plane element
    double launchSeconds = 5e-6;          ///< device: one kernel launch / host: one team barrier
    double splitSeconds = 0;              ///< per sample, per split step: run bounds, guard, skipped lanes
    double sortSeconds = 0;               ///< per sample: one class sort
    double laneRate = 0;                  ///< 64-bit operations per second; 0 takes half the device's
                                          ///< peak single-precision flop rate (an FMA is two flops)
    double memoryRate = 0;                ///< bytes per second; 0 takes the device's peak
    double cacheBytes = 0;                ///< cache that holds a gathered plane; 0 treats every gather as memory
    double cacheRatio = 4;                ///< a gather served from that cache runs this much faster
    double chainSeconds = 0;              ///< host: one dependent operation, across all cores; 0 ignores chains
    double vectorWidth = 1;               ///< host: speed-up of a vectorised loop over a scalar one
};

/** @brief What the rule remembers between two looks. */
struct HysteresisState {
    Policy current = Policy::Switch;
    int sinceDecision = 0;
};

/** @brief One row of EG_WEIGHT_ROWS. */
struct WeightRow {
    int target, vendor, family, model;
    double arith32, special32, special64, launchSeconds, splitSeconds, sortSeconds, laneRatePerCore, memoryRate, cacheRatio,
        chainSecondsPerCore, vectorWidth;
};

#define EG_WEIGHT_ROW_INIT(...) { __VA_ARGS__ },
inline constexpr WeightRow WEIGHT_ROWS[] = { EG_WEIGHT_ROWS(EG_WEIGHT_ROW_INIT) };
#undef EG_WEIGHT_ROW_INIT
inline constexpr int WEIGHT_ROW_COUNT = static_cast<int>(sizeof(WEIGHT_ROWS) / sizeof(WEIGHT_ROWS[0]));

/** @brief The row of the class (@p vendor, @p family, @p model) of @p target, or -1 for the built-in row: an unknown class takes it, nothing nearer. */
inline int weightRowIndex(Target target, int vendor, int family, int model) {
    for (int i = 0; i < WEIGHT_ROW_COUNT; ++i) {
        const WeightRow& r = WEIGHT_ROWS[i];
        if (r.target == static_cast<int>(target) && (r.vendor || r.family || r.model) && r.vendor == vendor && r.family == family && r.model == model)
            return i;
    }
    return -1;
}

/** @brief The built-in row of @p target (the row whose class is all zero). */
inline const WeightRow& builtinWeightRow(Target target) {
    for (const WeightRow& r : WEIGHT_ROWS)
        if (r.target == static_cast<int>(target) && !r.vendor && !r.family && !r.model) return r;
    throw std::logic_error("WeightRows.h: a target has no built-in row");
}

/** @brief The facts of the live machine a row does not hold: physical cores (host; 1 on a device) and the bytes of cache a gathered plane may sit in. */
struct Machine {
    double physicalCores = 1;
    double cacheBytes = 0;
};

/** @brief The weights of @p row on @p machine: the per-core rates scaled by the physical cores. */
inline Weights weightsOfRow(const WeightRow& row, const Machine& machine) {
    Weights w;
    w.target = static_cast<Target>(row.target);
    w.arith32 = row.arith32;
    w.special32 = row.special32;
    w.special64 = row.special64;
    w.launchSeconds = row.launchSeconds;
    w.splitSeconds = row.splitSeconds;
    w.sortSeconds = row.sortSeconds;
    w.laneRate = row.laneRatePerCore * machine.physicalCores;
    w.memoryRate = row.memoryRate;
    w.cacheBytes = machine.cacheBytes;
    w.cacheRatio = row.cacheRatio;
    w.chainSeconds = row.chainSecondsPerCore / machine.physicalCores;
    w.vectorWidth = row.vectorWidth;
    return w;
}

/**
 * @brief The built-in weights of @p target on the machine the rows were fitted on (a 4-core Xeon W-2125 with 12.7e6 bytes of cache; a
 *        Quadro P2000 for the device, whose cache is the live L2), until a fitted profile replaces them.
 */
inline Weights defaultWeights(Target target) {
    return weightsOfRow(builtinWeightRow(target), target == Target::Host ? Machine{ 4, 12.7e6 } : Machine{ 1, 0 });
}

namespace detail {

/** @brief What one branch costs one sample, per step: lane operations, dependent-chain seconds, state bytes, gathered bytes. */
struct Work { double ops; double chain; double stateBytes; double gatherBytes; };

/** @brief Speed-up of a host loop that is @p vectorised; a device (and a scalar loop) has none. */
inline double hostWidth(bool vectorised, const Weights& w) {
    return (w.target == Target::Host && vectorised) ? std::max(w.vectorWidth, 1.0) : 1.0;
}

inline Work branchWork(const BranchCost& b, const Weights& w, const DeviceProps& props, bool vectorised) {
    // A 64-bit op costs fp64_ratio plain ones on a device; a host core has no such penalty.
    const double wide = (w.target == Target::Device && props.fp64_ratio > 1.0) ? props.fp64_ratio : 1.0;
    const double width = hostWidth(vectorised, w);
    const double path = b.chain32 + b.chain64 + b.chainSpecial32 * w.special32 + b.chainSpecial64 * w.special64;
    return { (b.arith32 * w.arith32 + b.arith64 * wide + b.special32 * w.special32 + b.special64 * w.special64 * wide) / width,
             // A vectorised loop overlaps the chain across its lanes; a scalar one pays it in full.
             (w.target == Target::Host && !vectorised) ? path * w.chainSeconds : 0.0,
             // Planes read are read once; planes written are written once (an uncounted write count
             // assumes every plane read is also written).
             (b.stores < 0.0 ? 2.0 * b.loads : b.loads + b.stores) * w.elemBytes,
             b.gathers * w.elemBytes };
}

/** @brief Lane operations per second: the configured rate, slowed when too few warps are resident. */
inline double laneRate(const CostVector& cost, const DeviceProps& props, const Weights& w) {
    double rate = w.laneRate > 0.0 ? w.laneRate : 0.5 * props.peak_flops_sp;
    if (w.target == Target::Device && cost.regsPerThread > 0) {
        const ResidencyDevice dev{ props.cc_major, props.cc_minor, props.regs_per_sm, 0 };
        const int warps = residentWarps(dev, cost.regsPerThread);
        if (warps > 0 && warps < ISSUE_KNEE_WARPS) rate *= static_cast<double>(warps) / ISSUE_KNEE_WARPS;
    }
    return rate;
}

/** @brief Seconds per byte of state traffic and of gathered traffic over @p n samples. */
struct ByteRates { double state; double gather; };

inline ByteRates byteRates(double n, const DeviceProps& props, const Weights& w) {
    const double memory = w.memoryRate > 0.0 ? w.memoryRate : props.peak_bytes_per_s;
    if (memory <= 0.0) return { 0.0, 0.0 };
    const bool cached = w.cacheBytes > 0.0 && n * w.elemBytes <= w.cacheBytes;
    return { 1.0 / memory, 1.0 / (cached ? memory * w.cacheRatio : memory) };
}

/** @brief One launch over @p n samples: the slowest of its arithmetic, its dependent chain and its memory traffic, plus the launch. */
inline double launchSeconds(double n, const Work& work, double rate, const ByteRates& bytes, double launch) {
    const double compute = rate > 0.0 ? n * work.ops / rate : 0.0;
    const double traffic = n * (work.stateBytes * bytes.state + work.gatherBytes * bytes.gather);
    return std::max({ compute, n * work.chain, traffic }) + launch;
}

}  // namespace detail

/**
 * @brief Share of 32-lane groups that execute branch @p j, taken by a share
 *        @p p of the samples: a group runs every branch any of its lanes takes.
 *
 * With a measured mixed-group share `m`, the mixed groups are taken to run
 * every branch and the pure ones to split as the samples do; without one,
 * lanes are drawn independently: `1 - (1 - p)^32`.
 */
inline double warpShare(double p, double mixed32) {
    if (p <= 0.0) return 0.0;
    if (mixed32 < 0.0) return 1.0 - std::pow(1.0 - p, WARP_LANES);
    return std::min(1.0, p * (1.0 - mixed32) + mixed32);
}

/** @brief Seconds per step of the two paths for one mix. */
struct PathSeconds { double fused; double split; };

/**
 * @brief Predicted seconds per step of the fused `switch` kernel and of the
 *        split one for this mix.
 *
 * A launch takes the slowest of its arithmetic, its dependent chain (host only) and
 * its memory traffic, plus the launch itself. Per sample and step, in lane operations and bytes:
 * - fused: one kernel of `fusedSteps` steps. Its carried planes are read and
 *   written once per launch, so their traffic is divided by `fusedSteps`;
 *   gathered reads cost every step, at memory speed or, when the gathered plane
 *   fits the cache, at the cache's. On a device a 32-lane group pays for every
 *   branch any of its lanes takes (warpShare); a vectorised host loop
 *   evaluates every branch and selects, so it costs the sum whatever the mix; a
 *   scalar one runs each sample's own branch.
 * - split: one launch per branch over its own samples, every step, state
 *   traffic included, plus `splitSeconds` per sample and one class sort paid
 *   over the remaining steps.
 * The lane rate is the device's, slowed when the kernel's registers leave fewer
 * than ISSUE_KNEE_WARPS warps resident.
 *
 * @return both zero when the mix is empty
 */
inline PathSeconds predictedSeconds(const CostVector& cost, const Mix& mix, const DeviceProps& props,
                                    const Weights& w) {
    if (cost.count < 2 || cost.count > MAX_BRANCHES)
        throw std::invalid_argument("predictedSeconds: a dispatch group has 2 to 16 branches");
    const std::int64_t total = mix.total(cost.count);
    if (total <= 0) return { 0.0, 0.0 };

    const double n = static_cast<double>(total);
    const double fusedSteps = static_cast<double>(std::max<std::int64_t>(mix.fusedSteps, 1));
    const double remaining = static_cast<double>(std::max<std::int64_t>(mix.remainingSteps, 1));
    const double rate = detail::laneRate(cost, props, w);
    const detail::ByteRates bytes = detail::byteRates(n, props, w);

    detail::Work fused{ 0, 0, 0, 0 };
    double split = n * (w.splitSeconds + w.sortSeconds / remaining);
    for (int j = 0; j < cost.count; ++j) {
        const double p = static_cast<double>(mix.counts[j]) / static_cast<double>(total);
        const detail::Work work = detail::branchWork(cost.branch[j], w, props, cost.branch[j].vectorised);
        // The fused loop has one width for every branch: its own flag, not the branch's.
        const detail::Work inFused = detail::branchWork(cost.branch[j], w, props, cost.fusedVectorised);
        const double share = w.target == Target::Device ? warpShare(p, mix.mixed32)
                           : cost.fusedVectorised ? 1.0 : p;
        fused.ops += share * inFused.ops;
        fused.chain = cost.fusedVectorised ? std::max(fused.chain, inFused.chain) : fused.chain + share * inFused.chain;
        fused.stateBytes += share * inFused.stateBytes;
        fused.gatherBytes += share * inFused.gatherBytes;
        const detail::ByteRates own = detail::byteRates(p * n, props, w);
        split += detail::launchSeconds(p * n, work, rate, own, w.launchSeconds);
    }
    fused.stateBytes /= fusedSteps;
    const double fusedTime = detail::launchSeconds(n, fused, rate, bytes, 0.0)
                           + w.launchSeconds / fusedSteps;
    return { fusedTime, split };
}

/**
 * @brief Predicted time of the fused `switch` kernel over the split one for
 *        this mix (see predictedSeconds); above 1 the split wins.
 *
 * @return 1 when the mix is empty (nothing to compare)
 */
inline double predictedGain(const CostVector& cost, const Mix& mix, const DeviceProps& props,
                            const Weights& w) {
    const PathSeconds t = predictedSeconds(cost, mix, props, w);
    return t.split > 0.0 ? t.fused / t.split : 1.0;
}

/** @brief The band: enter at ENTER_GAIN, leave under LEAVE_GAIN, otherwise keep @p current. */
inline Policy decide(double gain, Policy current) {
    if (gain >= ENTER_GAIN) return Policy::Segmented;
    if (gain < LEAVE_GAIN) return Policy::Switch;
    return current;
}

/** @brief The policy for this mix: `predictedGain` through the hysteresis band
 *         (an empty mix says nothing, so the current policy stays). */
inline Policy resolve(const CostVector& cost, const Mix& mix, const DeviceProps& props,
                      const Weights& w, const HysteresisState& state) {
    if (mix.total(cost.count) <= 0) return state.current;
    return decide(predictedGain(cost, mix, props, w), state.current);
}

/**
 * @brief Count @p steps more steps; true when it is time to look at the mix
 *        again (every DECISION_EVERY steps), restarting the count.
 */
inline bool due(HysteresisState& state, int steps = 1) {
    state.sinceDecision += steps;
    if (state.sinceDecision < DECISION_EVERY) return false;
    state.sinceDecision = 0;
    return true;
}

}  // namespace policy
}  // namespace eagle
