// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file eagle.h
 * @brief Top-level umbrella header for the EAGLE graph-launch engine.
 *
 * Pulls in the full public surface: shared typedefs, the CUDA-graph
 * capture/launch machinery, the OpenMP+SIMD host dispatcher, the
 * observer/slice utilities, and the scan/scatter/compaction and
 * reduction primitives. Device-only pieces self-guard behind
 * ``EAGLE_CPU_ONLY`` inside each header.
 */

#include "eagle/typedefs.h"

#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"
#include "eagle/util/Threads.h"
#include "eagle/util/Observer.h"
#include "eagle/util/Observable.h"
#include "eagle/util/ObservableArray.h"
#include "eagle/util/Slice.h"

#include "eagle/cpu/Host.h"

#include "eagle/device_props.h"

#include "eagle/compose.h"

#include "eagle/stream.h"

#include "eagle/cuda.h"

#include "eagle/filtering.h"
#include "eagle/reduce.h"
