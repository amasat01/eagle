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

#ifdef __cplusplus
extern "C" {
#endif

#define EAGLE_HOST_TABLE_OK 0
#define EAGLE_HOST_TABLE_STALE 1    /* the graph was bound again or rebuilt since the table was made */
#define EAGLE_HOST_TABLE_FAILED 2   /* a node threw; the planes are partly written */

typedef struct eagle_host_table {
    int (*run)(const struct eagle_host_table*);
} eagle_host_table;

/* 0 on success, else an EAGLE_HOST_TABLE_* code. Every thread of the caller's region calls it. */
static inline int eagle_host_table_run(const eagle_host_table* table) { return table->run(table); }

/* The words for a nonzero status. */
static inline const char* eagle_host_table_message(int status)
{
    return status == EAGLE_HOST_TABLE_OK ? "ok"
         : status == EAGLE_HOST_TABLE_STALE
             ? "eagle host table: the graph was bound again or rebuilt since the table was made; call host_table() again"
             : "eagle host table: a node failed";
}

#ifdef __cplusplus
}
#endif
#endif
