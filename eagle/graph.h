// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file graph.h
 * @brief Umbrella header for the graph view: a generic ``Description`` any
 *        eagle graph can be rendered from, stride-aware hazard-edge
 *        derivation, a deterministic layered layout, and ASCII / Mermaid / DOT
 *        renderings -- plus ``describe()`` for ``cpu::Graph``,
 *        ``cuda::Graph``, ``GraphComposer`` and ``ConditionalGroup``.
 *
 * CUDA-free and header-only; ``describe(cuda::Graph)`` is present only in a
 * build that has CUDA.
 */

#include "eagle/graph/Describe.h"
#include "eagle/graph/Description.h"
#include "eagle/graph/Edges.h"
#include "eagle/graph/Layout.h"
#include "eagle/graph/Render.h"
