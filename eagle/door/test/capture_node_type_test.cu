// SPDX-License-Identifier: Apache-2.0

// The capture-attribution facility reads the type of every node of a graph. A graph holding a conditional node must answer: the runtime's
// cudaGraphNodeGetType gives cudaErrorUnknown for one (CUDA 12.6 runtime over a 560 driver), the driver's cuGraphNodeGetType names it.
// Usage: capture_node_type_test (needs a device).
#include "eagle/cuda/CaptureAttribution.h"

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    std::printf("%s  %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

}  // namespace

int main() {
    cudaGraph_t graph = nullptr;
    int* cell = nullptr;
    bool built = cudaMalloc(&cell, sizeof(int)) == cudaSuccess && cudaGraphCreate(&graph, 0) == cudaSuccess;
    cudaGraphConditionalHandle handle;
    built = built && cudaGraphConditionalHandleCreate(&handle, graph, 1, cudaGraphCondAssignDefault) == cudaSuccess;
    cudaGraphNodeParams params = {};
    params.type = cudaGraphNodeTypeConditional;
    params.conditional.handle = handle;
    params.conditional.type = cudaGraphCondTypeIf;
    params.conditional.size = 1;
    cudaGraphNode_t node = nullptr;
    built = built && cudaGraphAddNode(&node, graph, nullptr, 0, &params) == cudaSuccess;
    check(built, "a graph with an IF conditional node is built");
    if (!built) return 1;
    cudaMemsetParams memset_params = {};    // the body: a memset node, legal in a conditional body
    memset_params.dst = cell;
    memset_params.elementSize = 4;
    memset_params.width = 1;
    memset_params.height = 1;
    cudaGraphNode_t body_node = nullptr;
    check(cudaGraphAddMemsetNode(&body_node, params.conditional.phGraph_out[0], nullptr, 0, &memset_params) == cudaSuccess,
          "the conditional's body takes a memset node");
    bool typed = false;
    try {
        typed = eagle::cuda::detail::nodeType(node) == cudaGraphNodeTypeConditional;
    } catch (...) {
    }
    check(typed, "the type of a conditional node is read: conditional (the runtime's cudaGraphNodeGetType answers cudaErrorUnknown)");
    bool toggleable = true, refused_none = false;
    try {
        toggleable = eagle::cuda::isNodeToggleable(node);
        refused_none = eagle::cuda::detail::bodyRefusedNodeType(graph) == nullptr;
    } catch (...) {
    }
    check(!toggleable && refused_none, "a graph with a conditional node is not toggleable and holds no node a conditional body refuses");
    cudaGraphDestroy(graph);
    cudaFree(cell);
    std::printf("capture_node_type_test: %d failure(s)\n", failures);
    return failures != 0;
}
