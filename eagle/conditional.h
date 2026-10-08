// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file conditional.h
 * @brief Umbrella header for skippable graph regions (conditional
 *        nodes).
 *
 * What is this: lets a captured CUDA graph skip a region at replay time
 * instead of always running it -- e.g. an event check that only sometimes
 * needs its follow-up work. Use it when you want that "if" decided on the
 * device, inside a captured graph, without rebuilding or re-launching the
 * graph. Needs CUDA >= 12.3 (the conditional-node driver APIs this pulls
 * in are not available before that; see ``conditional/ConditionalGroup.h``'s
 * own CUDA-floor check for the exact APIs and error).
 *
 * Pulls in the declarative predicate (``CountGuard``), the dual-face
 * explicit-builder facility (``conditional::ConditionalGroup``), and the
 * CUDA-only capture-weave sibling (``cuda::CaptureConditional``, which
 * self-guards behind ``EAGLE_CPU_ONLY`` like the rest of the CUDA-only
 * graph layer). Not pulled into the top-level ``eagle.h`` umbrella -- name
 * this header directly to use the facility.
 */

#include "eagle/conditional/CountGuard.h"
#include "eagle/conditional/ConditionalGroup.h"
#include "eagle/cuda/CaptureConditional.h"
