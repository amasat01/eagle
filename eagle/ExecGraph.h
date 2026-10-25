// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file ExecGraph.h
 * @brief ``ExecGraph`` -- what ``cpu::Graph`` and ``cuda::Graph`` both provide --
 *        and ``eagle::Graph``, the executor of the current build.
 */

#include "eagle/ScratchArena.h"
#include "eagle/cpu/Graph.h"
#include "eagle/typedefs.h"

#ifndef EAGLE_CPU_ONLY
#include "eagle/cuda/Graph.h"
#endif

#include <concepts>
#include <vector>

namespace eagle {

/**
 * @brief The surface shared by the host and CUDA graph executors: move-only,
 *        a scratch arena, the native-node dependency table, and ``finalize()``
 *        to commit the arena (CUDA also captures the native nodes there).
 *        Code written against it builds for either executor.
 */
template<typename G>
concept ExecGraph
    = std::default_initializable<G> && std::movable<G> && !std::copyable<G>
      && requires(G g, const G cg) {
             { g.finalize() };
             { g.scratchArena() } -> std::same_as<ScratchArena&>;
             { cg.scratchArena() } -> std::same_as<const ScratchArena&>;
             { cg.nativeDeps() }
                 -> std::same_as<const std::vector<std::vector<idx_t>>&>;
         };

static_assert(ExecGraph<cpu::Graph>);

#ifdef EAGLE_CPU_ONLY
/** @brief The graph executor of this build: ``cpu::Graph``. */
using Graph = cpu::Graph;
#else
static_assert(ExecGraph<cuda::Graph>);

/** @brief The graph executor of this build: ``cuda::Graph``. */
using Graph = cuda::Graph;
#endif

}   // namespace eagle
