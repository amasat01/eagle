// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// 04_graph_view.cpp
//
// Print a C++ graph: describe() turns an eagle::cpu::Graph into a generic
// Description, and the same layout and renderers that draw a Python
// eagle.Graph draw it. CUDA-free; the CUDA twin of describe() is the
// eagle::cuda::Graph overload, used the same way.
#include <cstdio>
#include <iostream>

#include <eagle/eagle.h>

namespace {

// A native node needs reserveScratch() and runHost(); this one does nothing,
// because only the graph's shape matters here.
struct Step {
    void reserveScratch(eagle::ScratchArena&) {}
    void runHost(eagle::cpu::Graph&) {}
};

}  // namespace

int main()
{
    // [cell:build]
    // four nodes: 0 feeds 1 and 2, which both feed 3
    eagle::cpu::Graph g;
    const eagle::idx_t a = g.addNative(Step{}, {});
    const eagle::idx_t b = g.addNative(Step{}, {a});
    const eagle::idx_t c = g.addNative(Step{}, {a});
    g.addNative(Step{}, {b, c});
    // [cell:build:end]

    // [cell:describe]
    const eagle::graph::Description d = eagle::graph::describe(g);
    const eagle::graph::Layout lay    = eagle::graph::layout(d);
    std::cout << eagle::graph::ascii(d, lay);
    // [cell:describe:end]

    // [cell:export]
    std::cout << eagle::graph::mermaid(d);
    // [cell:export:end]
    return 0;
}
