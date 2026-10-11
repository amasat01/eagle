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

// Measured defaults (see defaultWeights), per sample.
inline constexpr double HOST_LANE_RATE = 28e9;         ///< lane ops per second, all cores, vectorised
inline constexpr double HOST_PLANE_SECONDS = 0.7e-9;    ///< one launch's state read + write
inline constexpr double HOST_SPLIT_SECONDS = 0.2e-9;    ///< one split step's bookkeeping
inline constexpr double HOST_SORT_SECONDS = 25e-9;      ///< one class sort
inline constexpr double DEVICE_PLANE_SECONDS = 0.3e-9;
inline constexpr double DEVICE_SPLIT_SECONDS = 0.05e-9;
inline constexpr double DEVICE_SORT_SECONDS = 10e-9;

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
};

/** @brief The branches of one dispatch group, as counted from its IR. */
struct CostVector {
    int count = 0;
    /// The fused host loop is vectorised: it evaluates every branch and selects,
    /// so it costs the sum of the branches whatever the mix.
    bool fusedVectorised = false;
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
    double special = 20;       ///< one transcendental, in plain arithmetic ops
    double load = 1;           ///< one plane element read, in arithmetic ops
    double gather = 4;         ///< extra cost of an indexed read over a plain one
    double simdLoss = 3;       ///< host, unvectorised fused loop: slowdown of its vectorisable work
    double launchSeconds = 5e-6;          ///< device: one kernel launch / host: one team barrier
    double planeSeconds = 0;              ///< per sample, per launch: reading and writing the state
    double splitSeconds = 0;              ///< per sample, per split step: run bounds, guard, skipped lanes
    double sortSeconds = 0;               ///< per sample: one class sort
    double laneRate = 0;                  ///< lane operations per second; 0 takes half the device's
                                          ///< peak flop rate (an FMA is two flops)
};

/** @brief What the rule remembers between two looks. */
struct HysteresisState {
    Policy current = Policy::Switch;
    int sinceDecision = 0;
};

/**
 * @brief Weights for @p target, measured on a 4-core Xeon W-2125 host and a
 *        Quadro P2000 (N = 2^21, fp64, warmed), until a fitted profile replaces them.
 */
inline Weights defaultWeights(Target target) {
    Weights w;
    w.target = target;
    if (target == Target::Host) {
        w.launchSeconds = 2e-6;
        w.laneRate = HOST_LANE_RATE;
        w.planeSeconds = HOST_PLANE_SECONDS;
        w.splitSeconds = HOST_SPLIT_SECONDS;
        w.sortSeconds = HOST_SORT_SECONDS;
    } else {
        w.planeSeconds = DEVICE_PLANE_SECONDS;
        w.splitSeconds = DEVICE_SPLIT_SECONDS;
        w.sortSeconds = DEVICE_SORT_SECONDS;
    }
    return w;
}

namespace detail {

/** @brief Cost of the vectorisable part (arithmetic, plain loads) and the scalar part of one branch. */
struct Split { double vector; double scalar; };

inline Split branchWork(const BranchCost& b, const Weights& w, const DeviceProps& props) {
    // A 64-bit op costs fp64_ratio plain ones on a device; a host core has no such penalty.
    const double wide = (w.target == Target::Device && props.fp64_ratio > 1.0) ? props.fp64_ratio : 1.0;
    // A load costs what the device's ridge point says one operand costs in flops.
    const double ridge = props.ridgeFlopsPerByte("float32");
    const double load = (w.target == Target::Device && ridge > 0.0) ? std::max(w.load, 2.0 * ridge) : w.load;
    return { b.arith32 + b.arith64 * wide + b.loads * load,
             (b.special32 + b.special64 * wide) * w.special + b.gathers * load * w.gather };
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

/**
 * @brief Predicted time of the fused `switch` kernel over the split one for
 *        this mix; above 1 the split wins.
 *
 * Work per sample and step, in lane operations, turned into seconds at the
 * device's lane rate:
 * - fused: on the host a vectorised loop runs every branch (sum of the
 *   branches); an unvectorised one runs its own branch but loses `simdLoss`
 *   on the vectorisable work. On a device a 32-lane group pays for every
 *   branch any of its lanes takes (warpShare).
 * - split: each sample runs its own branch only.
 *
 * Fixed terms: the fused kernel reads and writes the state once per launch of
 * `fusedSteps` steps; the split path once per step, plus `splitSeconds` per
 * sample per step, K launches per step, and one class sort paid over the
 * remaining steps.
 *
 * @return 1 when the mix is empty (nothing to compare)
 */
inline double predictedGain(const CostVector& cost, const Mix& mix, const DeviceProps& props,
                            const Weights& w) {
    if (cost.count < 2 || cost.count > MAX_BRANCHES)
        throw std::invalid_argument("predictedGain: a dispatch group has 2 to 16 branches");
    const std::int64_t total = mix.total(cost.count);
    if (total <= 0) return 1.0;

    double fused = 0, split = 0;
    for (int j = 0; j < cost.count; ++j) {
        const double p = static_cast<double>(mix.counts[j]) / static_cast<double>(total);
        const detail::Split work = detail::branchWork(cost.branch[j], w, props);
        const double c = work.vector + work.scalar;
        if (w.target == Target::Device)
            fused += warpShare(p, mix.mixed32) * c;
        else if (cost.fusedVectorised)
            fused += c;
        else
            fused += p * (work.vector * w.simdLoss + work.scalar);
        split += p * c;
    }

    const double rate = w.laneRate > 0.0 ? w.laneRate : 0.5 * props.peak_flops_sp;
    if (rate <= 0.0) return split > 0.0 ? fused / split : 1.0;
    const double n = static_cast<double>(total);
    const double fusedSteps = static_cast<double>(std::max<std::int64_t>(mix.fusedSteps, 1));
    const double remaining = static_cast<double>(std::max<std::int64_t>(mix.remainingSteps, 1));
    const double fusedTime = fused * n / rate + (n * w.planeSeconds + w.launchSeconds) / fusedSteps;
    const double splitTime = split * n / rate + n * (w.planeSeconds + w.splitSeconds)
                           + cost.count * w.launchSeconds + n * w.sortSeconds / remaining;
    return fusedTime / splitTime;
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
