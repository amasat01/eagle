// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"

#include <concepts>
#include <vector>

namespace eagle {

class ScratchArena;  // forward (eagle:: top-level, device-neutral working-buffer allocator)

namespace cuda {

class Graph;         // forward: the graph a native node contributes itself to

/**
 * @brief The protocol an eagle primitive (scan / reduce / filtering) implements
 *        to contribute itself to a Graph via ``Graph::addNative``.
 *
 * A native node participates in a two-phase, capture-safe build that a builder
 * never sees (it only ever calls ``graph.addNative(node, deps)``):
 *
 *   1. **reserve** — ``reserveScratch(arena)``: the node declares its
 *      working-buffer sizes to the graph's ``ScratchArena``, which assigns each
 *      one a *reusable* slot (nodes with disjoint execution lifetimes share
 *      memory — peak, not sum). The node stashes the returned handles. No device
 *      memory exists yet; nothing is captured.
 *   2. **build** — ``buildInto(graph, deps)``: after the arena has computed the
 *      peak and allocated once, the node appends its subgraph, sourcing scratch
 *      from ``arena.resolve(handle)`` (stable pointers, valid for every replay),
 *      wiring the supplied dependencies. Returns the index of its last graph
 *      node, so downstream nodes can depend on it.
 *
 * The arena and the two phases are internal to ``Graph::addNative`` /
 * ``Graph::finalizeNatives``; the reuse policy inside the arena is a contained,
 * swappable component. This keeps the node authors and graph builders free of any
 * allocation concern — the whole point of the graph-level arena.
 */
/// \cond EAGLE_SKIP_DOX
// (hidden from Doxygen: 1.9.x emits C++20 concepts in a form Sphinx's C++
// parser rejects; the protocol is documented in the block comment above and
// summarised on the api_graph docs page.)
template<typename N>
concept NativeNode = requires(N n, Graph& g, ScratchArena& arena,
    const std::vector<idx_t>& deps) {
    { n.reserveScratch(arena) };
    { n.buildInto(g, deps) } -> std::convertible_to<idx_t>;
};
/// \endcond

}  // namespace cuda
}  // namespace eagle
