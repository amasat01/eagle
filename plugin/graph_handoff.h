// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Handing a CPU graph to another runtime: the host table.
 *
 * `Graph.host_table()` builds an immutable table of the graph's host stages;
 * its `.handle` is a `const eagle_host_table*`. A runtime that already has an
 * OpenMP parallel region calls `eagle_host_table_run` from EVERY thread of that
 * region: the stages' tiles are shared across those threads, with the barrier
 * of a worksharing loop after each node. Called from only some of the threads
 * it deadlocks. Called outside any region, it opens its own.
 *
 * The table holds the pointers of the planes bound when it was made (a
 * snapshot). `bind` or a rebuild of the graph makes it stale; a stale table
 * is refused, never run. Keep the handle (or the Graph) alive while it runs.
 * The entry never takes the GIL.
 *
 * Plain C: no aether, no CUDA, no Python. */
#ifndef EAGLE_PLUGIN_GRAPH_HANDOFF_H
#define EAGLE_PLUGIN_GRAPH_HANDOFF_H

#include "eagle/exec/HostTableAbi.h"

#endif
