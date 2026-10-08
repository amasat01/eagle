// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace eagle {

/**
 * @brief Declarative predicate for a skippable graph region.
 *
 * The gated region runs iff ``*count != (baseline ? *baseline : 0)``.
 * Backends synthesize the guard evaluation themselves from this data alone
 * -- there is no user-supplied predicate kernel or lambda in v1. The CUDA
 * face (``eagle::conditional::ConditionalGroup::buildInto``,
 * ``eagle::cuda::CaptureConditional``) reads both words device-side via an
 * eagle-owned ``setCond`` kernel, with zero host round-trip; the CPU face
 * (``ConditionalGroup::runHost``) reads them host-side at dispatch.
 *
 * Covers both real v1 consumers: the fencepost-pair liveness check
 * (``type_offsets[t+1]`` vs ``type_offsets[t]``) and a plain nonzero count
 * (``n_active``).
 *
 * Mode-free by design -- no CUDA includes, so this header may be named from
 * a translation unit that has never seen ``<cuda_runtime.h>``.
 *
 * @warning ``count`` and (if non-null) ``baseline`` must be device-resident
 *          ``unsigned int`` words on the CUDA face, or host-resident on the
 *          CPU face, and their lifetime must outlive every replay of the
 *          graph the guard is wired into. The caller owns that lifetime;
 *          nothing here retains it.
 */
struct CountGuard {
    /** @brief Never nullptr -- the live count read at every evaluation. */
    const unsigned int* count;
    /** @brief nullptr reads as 0 (the common "nonzero" case, e.g. n_active). */
    const unsigned int* baseline = nullptr;
};

} // namespace eagle
