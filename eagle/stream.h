// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file stream.h
 * @brief Backend-generic non-owning stream handle dispatch.
 *
 * Exposes ``eagle::StreamRef`` — a cheap, non-owning handle naming the active
 * backend's stream. On CUDA it references a ``cudaStream_t`` (the default
 * stream unless constructed otherwise) and forwards ``synchronize()`` /
 * ``native()`` to the driver; under ``EAGLE_CPU_ONLY`` it collapses to a
 * synchronous host no-op. This is the *single* backend selection point for
 * streams: downstream code names only ``eagle::StreamRef`` and never guards a
 * raw ``cudaStream_t`` at the call site.
 *
 * Deliberately distinct from the owning RAII ``eagle::cuda::Stream`` (which
 * *creates* a real stream): the alias name mirrors the non-owning backend
 * types ``cpu::StreamRef`` / ``cuda::StreamRef`` and avoids colliding with
 * ``Stream`` in consumers that pull in both ``eagle`` and ``eagle::cuda``.
 */

#include "eagle/cpu/StreamRef.h"

#ifndef EAGLE_CPU_ONLY
#include "eagle/cuda/stream/StreamRef.h"
#endif

namespace eagle {

#ifdef EAGLE_CPU_ONLY
/** @brief Backend-generic non-owning stream handle (host backend). */
using StreamRef = cpu::StreamRef;
#else
/** @brief Backend-generic non-owning stream handle (CUDA backend). */
using StreamRef = cuda::StreamRef;
#endif

} // namespace eagle
