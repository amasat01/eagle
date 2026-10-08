// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file RuntimeCompat.h
 * @brief The CUDA runtime calls whose signatures changed in CUDA 13, under one
 *        spelling: CUDA 13 adds an edge-data argument to cudaGraphAddNode,
 *        cudaGraphGetEdges and cudaStreamUpdateCaptureDependencies, and renames
 *        cudaStreamGetCaptureInfo_v3 to cudaStreamGetCaptureInfo. eagle passes no
 *        edge data, so both toolkits get the same call.
 */

#include "eagle/typedefs.h"

#ifndef EAGLE_CPU_ONLY

#include <cuda_runtime_api.h>

#include <cstddef>

namespace eagle {
namespace cuda {
namespace detail {

inline cudaError_t graphAddNode(cudaGraphNode_t* node, cudaGraph_t graph, const cudaGraphNode_t* deps,
    std::size_t n, cudaGraphNodeParams* params)
{
#if CUDART_VERSION >= 13000
    return cudaGraphAddNode(node, graph, deps, nullptr, n, params);
#else
    return cudaGraphAddNode(node, graph, deps, n, params);
#endif
}

inline cudaError_t graphGetEdges(cudaGraph_t graph, cudaGraphNode_t* from, cudaGraphNode_t* to, std::size_t* n)
{
#if CUDART_VERSION >= 13000
    return cudaGraphGetEdges(graph, from, to, nullptr, n);
#else
    return cudaGraphGetEdges(graph, from, to, n);
#endif
}

inline cudaError_t streamUpdateCaptureDependencies(cudaStream_t stream, cudaGraphNode_t* deps, std::size_t n,
    unsigned int flags)
{
#if CUDART_VERSION >= 13000
    return cudaStreamUpdateCaptureDependencies(stream, deps, nullptr, n, flags);
#else
    return cudaStreamUpdateCaptureDependencies(stream, deps, n, flags);
#endif
}

inline cudaError_t streamGetCaptureInfo(cudaStream_t stream, cudaStreamCaptureStatus* status,
    unsigned long long* id, cudaGraph_t* graph, const cudaGraphNode_t** deps, const cudaGraphEdgeData** edges,
    std::size_t* n)
{
#if CUDART_VERSION >= 13000
    return cudaStreamGetCaptureInfo(stream, status, id, graph, deps, edges, n);
#else
    return cudaStreamGetCaptureInfo_v3(stream, status, id, graph, deps, edges, n);
#endif
}

} // namespace detail
} // namespace cuda
} // namespace eagle

#endif // EAGLE_CPU_ONLY
