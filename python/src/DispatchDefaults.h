// SPDX-License-Identifier: Apache-2.0

// The default dispatch weights the Python binding serves. This table serves only the old Python dispatch policy and leaves with it; the door's
// built-in rows (eagle/door/eagle_door_rows.h) are the source of truth, and eagle/door/test/weight_rows_test.cpp fails when the two drift.
#pragma once

#include "eagle/policy/Dispatch.h"

namespace eagle_python {

/** @brief The weights of the built-in host row on the machine it was fitted on (4 cores, 12.7e6 bytes of cache), or of the built-in device row. */
inline eagle::policy::Weights defaultDispatchWeights(eagle::policy::Target target) {
    eagle::policy::Weights w;
    w.target = target;
    if (target == eagle::policy::Target::Host) {
        w.arith32 = 1.495, w.special32 = 25.2, w.special64 = 122.0, w.launchSeconds = 1.4e-6, w.splitSeconds = 0.35e-9, w.sortSeconds = 13.2e-9;
        w.laneRate = 24.5e9, w.memoryRate = 52.1e9, w.cacheBytes = 12.7e6, w.cacheRatio = 5.09, w.chainSeconds = 0.23e-9, w.vectorWidth = 8;
    } else {
        w.arith32 = 1, w.special32 = 20, w.special64 = 20, w.launchSeconds = 5e-6, w.splitSeconds = 0.05e-9, w.sortSeconds = 10e-9;
        w.laneRate = 0, w.memoryRate = 0, w.cacheBytes = 0, w.cacheRatio = 4, w.chainSeconds = 0, w.vectorWidth = 1;
    }
    return w;
}

}  // namespace eagle_python
