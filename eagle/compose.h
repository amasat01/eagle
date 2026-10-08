// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file compose.h
 * @brief Umbrella header for the graph-composition facility:
 *        ``eagle::compose::GraphComposer``.
 *
 * Mirrors ``eagle/device_props.h``'s aggregation pattern: a thin umbrella
 * so ``#include <eagle/eagle.h>`` sees the facility, unconditionally
 * includable (the underlying header self-guards its CUDA-only surface
 * behind ``EAGLE_CPU_ONLY``, mirroring every other dual-face header this
 * umbrella style aggregates).
 */

#include "eagle/compose/GraphComposer.h"
