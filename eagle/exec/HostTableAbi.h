// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* The host table's C ABI: the handle a runtime receives and the call it makes. The handoff contract is in
 * plugin/graph_handoff.h, installed as eagle/plugin/graph_handoff.h. Plain C: no aether, no CUDA, no Python. */
#ifndef EAGLE_EXEC_HOST_TABLE_ABI_H
#define EAGLE_EXEC_HOST_TABLE_ABI_H

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
