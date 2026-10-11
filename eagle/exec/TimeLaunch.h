// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Wall time of repeated launches, for calibrating the cost model: every
// repetition is returned, never a summary, so the caller decides what to drop
// (the first, cold one) and how to aggregate. The host face times the team run
// with steady_clock; the device face lives behind the backend seam
// (`eagle_backend_x_time_launch`), timed with CUDA event pairs.
#pragma once

#include "eagle/exec/HostTeam.h"
#include "eagle/exec/Partition.h"

#include <chrono>
#include <stdexcept>
#include <utility>
#include <vector>

namespace eagle {
namespace exec {

/** @brief Seconds taken by each of @p reps calls of @p run (steady_clock). */
template <class Run>
std::vector<double> timeRepetitions(int reps, Run&& run) {
    if (reps < 1) throw std::invalid_argument("timeRepetitions: reps must be at least 1");
    std::vector<double> seconds;
    seconds.reserve(static_cast<std::size_t>(reps));
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        run();
        const auto t1 = std::chrono::steady_clock::now();
        seconds.push_back(std::chrono::duration<double>(t1 - t0).count());
    }
    return seconds;
}

/** @brief Seconds of each of @p reps host team runs of @p entry over @p part. */
inline std::vector<double> timeHostLaunch(HostEntryV2 entry, void* const* args, const Partition& part,
                                          int reps) {
    return timeRepetitions(reps, [&] { HostTeam::run(entry, args, part); });
}

}  // namespace exec
}  // namespace eagle
