/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file eg_door.h
 * @brief Eagle's C door ``eagle-door/1``: the one API a generated
 *        binding calls to run units. C11, POD only, no C++ crosses.
 *
 * RULES (every function below follows them).
 *  - Every function takes the context first and returns an ``eg_status``.
 *  - A non-OK status leaves its reason in the context, as text for a person;
 *    ``eg_error`` is the one function that reads it. Nothing else is text:
 *    settings, options and reports are structs, enums and arrays.
 *  - Arrays of unknown length use the two-call protocol: call with a NULL
 *    buffer and capacity 0 to get the length, then call again with a buffer.
 *  - Structs start with ``uint32_t struct_size``. INPUT: the caller zero-fills
 *    and sets its own ``sizeof``; eagle reads the fields that fit and takes
 *    the rest as zero. OUTPUT: the caller zero-fills and sets its ``sizeof``;
 *    eagle writes ``min(caller, eagle)`` bytes and writes that size back.
 *    Structs never travel in arrays (parallel arrays instead). A ``flags``
 *    bit eagle does not know is refused (EG_BAD_REQUEST), never ignored.
 *  - Versioning: major 1 is additive only (functions, struct tail fields,
 *    enum values and flag bits are appended). A caller names the major it was
 *    built for in ``eg_ctx_settings``; another major is refused.
 *  - No callbacks, except the ones a caller hands over on purpose: a host node's ``eg_host_fn`` and ``eg_host_vjp_fn``
 *    (see ``eg_host_node_new``) and its release function, and a plane's deleter. Every callback runs on the thread that
 *    made the door call that reached it, inside that call. A binding may release its interpreter lock around every call.
 *  - No global mutable state: everything lives in a context. The one process
 *    resource is the host worker pool, shared by all contexts; each context
 *    states its own thread count, which is passed per launch. A context and
 *    its handles may be used from any number of threads: calls on one context
 *    run one at a time, a second thread waiting until the running call returns
 *    (no order between waiting threads). ``eg_error`` reports the reason of the
 *    CALLING thread's own last non-OK call, so another thread's call never
 *    overwrites it. A plane's deleter runs inside a call and must not call the
 *    door on that context (it is refused BAD_REQUEST, not waited for).
 *    Different contexts run concurrently.
 *  - Fork: a context belongs to the process that made it. In a forked child
 *    every call on it is refused except ``eg_error`` and ``eg_ctx_free``
 *    (which then frees host memory only: no device call, no thread join).
 *  - Handles belong to the context that made them; a handle of another
 *    context is refused. Every ``*_free`` is NULL-safe and returns EG_OK.
 *  - Arrays cross by DLPack only (``DLManagedTensorVersioned``, DLPack >= 1.0).
 *  - ``eg_ctx_info``, ``eg_device_props`` and ``eg_plan_counters`` are each both a struct tag and a
 *    function name; C++ callers write ``struct eg_ctx_info`` for the struct.
 *
 * GROUPS. The HOST group is complete. The DEVICE group is declared with one
 * line each; its calls map onto the ``eagle-backend/1`` seam handles.
 */
#ifndef EG_DOOR_H
#define EG_DOOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- version ------------------------------------------------------------- */

#define EG_DOOR_VERSION_MAJOR 1u
#define EG_DOOR_VERSION_MINOR 0u

/* ---- status: one value per refusal kind ---------------------------------- */

typedef int32_t eg_status;
/** Done. */
#define EG_OK 0
/** NOT WRITTEN: eagle does not do this (a feature, flag, dtype, role or
 *  device kind this build lacks). The request may be valid elsewhere. */
#define EG_NOT_WRITTEN 1
/** BAD REQUEST: the caller's input is wrong (a bad handle, a missing or
 *  mistyped binding, a shape that disagrees with the unit's meta, a buffer
 *  too small, a wrong thread or process, an unknown flag). */
#define EG_BAD_REQUEST 2
/** FAULT: the request was valid but failed while running (a driver error,
 *  out of memory, a library that does not load, no usable device). */
#define EG_FAULT 3

/* ---- opaque handles ------------------------------------------------------ */

typedef struct eg_ctx_s eg_ctx;       /**< all state: settings, host team, device, stream, caches, errors */
typedef struct eg_unit_s eg_unit;     /**< one loaded unit (its stage units included) with its typed meta */
typedef struct eg_plane_s eg_plane;   /**< one array taken from DLPack; reference-counted inside eagle */
typedef struct eg_plan_s eg_plan;     /**< a unit under one structure over fixed partitions, plus its bindings */
typedef struct eg_runner_s eg_runner; /**< a run-until-done loop over bound plans; keeps its record */
typedef struct eg_graph_s eg_graph;   /**< a recorded sequence of door calls, replayed as one */
typedef struct eg_gemm_s eg_gemm;     /**< a matrix product: stored shapes, flags and width, and on a device the stream's cuBLAS handle */
typedef struct eg_fft_s eg_fft;       /**< a transform plan: kind, norm, shape and axes, and on a device its cuFFT plans */
typedef struct eg_host_node_s eg_host_node;   /**< a caller's callback (and optional vjp) the door runs as one node, see ``eg_host_node_new`` */

/** DLPack >= 1.0 managed tensor (dlpack.h). Only a pointer crosses. */
struct DLManagedTensorVersioned;

/* ---- enums --------------------------------------------------------------- */

/** How a plan runs. */
typedef int32_t eg_structure;
#define EG_STRUCTURE_HOST_TEAM 0   /**< eagle's host team over contiguous tiles */
#define EG_STRUCTURE_HOST_SERIAL 1 /**< one serial call per partition: the correctness reference */
#define EG_STRUCTURE_DEVICE 2      /**< one device launch per partition (device group) */

/** Order of a sum over samples. */
typedef int32_t eg_reduce_policy;
#define EG_REDUCE_COMPENSATED 0 /**< compensated, canonical order (the default) */
#define EG_REDUCE_FIXED 1       /**< plain sum in a fixed leaf order */

typedef int32_t eg_reduce_op;
#define EG_REDUCE_SUM 0
#define EG_REDUCE_MAX 1
#define EG_REDUCE_MIN 2
#define EG_REDUCE_ALL 3 /**< over a uint8 mask */

typedef int32_t eg_steps_reverse;
#define EG_STEPS_REVERSE_PER_STEP 0
#define EG_STEPS_REVERSE_FUSED 1
#define EG_STEPS_REVERSE_AUTO 2 /**< FUSED or PER_STEP, chosen at eg_runner_new from the unit's measured table; the choice is
                                     reported by eg_runner_figures */

typedef int32_t eg_events;
#define EG_EVENTS_STEP 0   /**< a sample stops at the step its mask was set */
#define EG_EVENTS_REFINE 1 /**< locate each stop inside its stop step; a plain run takes exactly the derived units
                              TAPE (window 1), REFINE_NEWTON and STATE_GUARD_VJP; a recorded run adds the reverse ones */

/** Per-sample event status of a refine run (``eg_run_events``, int8). After an
 *  eg_run that returned OK a caller sees only 1 and 2, and 3 with
 *  EG_NO_EVENT_ZERO: a run where any sample has 4 or 5, or 3 under
 *  EG_NO_EVENT_REFUSE, returns BAD_REQUEST naming the first such sample. */
#define EG_EVENT_PENDING 0  /**< not evaluated */
#define EG_EVENT_REFINED 1  /**< found inside the stop step */
#define EG_EVENT_AT_STEP 2  /**< found at the step's end, within tolerance */
#define EG_EVENT_NONE 3     /**< never stopped, stopped before any step, or no guard crossed */
#define EG_EVENT_GRAZE 4    /**< a tangent crossing: the event time is not differentiable */
#define EG_EVENT_TWO 5      /**< two guards cross inside the tolerance in one step */

typedef int32_t eg_no_event;
#define EG_NO_EVENT_REFUSE 0 /**< a sample with no crossing in its stop step is refused */
#define EG_NO_EVENT_ZERO 1   /**< its stop stays the stepped one */

/** What a derived unit is to a runner. Hawk builds every derived unit before
 *  the runner (its derivative transform stays in hawk); eagle only runs them. */
typedef int32_t eg_derived_role;
#define EG_DERIVED_TAPE 0            /**< the one-step taping twin (record) */
#define EG_DERIVED_TAPE_FUSED 1      /**< the K-step taping twin (record) */
#define EG_DERIVED_STEP_VJP 2        /**< the per-step vjp (reverse) */
#define EG_DERIVED_FUSED_VJP 3       /**< the K-step vjp (steps_reverse fused) */
#define EG_DERIVED_REFINE_ENTRY 4    /**< the event refine entry (events refine) */
#define EG_DERIVED_REFINE_NEWTON 5   /**< refine entry + its jvp in the event time */
#define EG_DERIVED_REFINE_VJP 6      /**< vjp of the refine entry (refine + record) */
#define EG_DERIVED_STATE_GUARD 7     /**< the state guard (refine) */
#define EG_DERIVED_STATE_GUARD_VJP 8 /**< vjp of the state guard (refine + record) */

/** The launch shape a run took. */
typedef int32_t eg_run_mode;
#define EG_MODE_HOST 0      /**< the host loop */
#define EG_MODE_BAND 1      /**< device: the WHILE-graph / policy path */
#define EG_MODE_FUSED_ONE 2 /**< device: one fused launch, no graph */
#define EG_MODE_PERSIST 3   /**< device: the persistent entry, no graph */

typedef int32_t eg_cond_kind;
#define EG_COND_IF 0
#define EG_COND_IF_ELSE 1
#define EG_COND_WHILE 2
#define EG_COND_SWITCH 3

/** Capability bits (``eg_ctx_info.capabilities``). */
#define EG_CAP_HOST (UINT64_C(1) << 0)
#define EG_CAP_DEVICE (UINT64_C(1) << 1)
#define EG_CAP_COND_IF_ELSE (UINT64_C(1) << 2)
#define EG_CAP_COND_SWITCH (UINT64_C(1) << 3)

/* ---- unit meta: constant data inside every unit library ------------------ */

/**
 * Every unit library links ONE generated C translation unit of constant data
 * next to its kernel texts (which stay byte-identical) and exports ONE symbol:
 *
 *     const struct eg_library_meta* eg_unit_library_meta(void);
 *
 * (C linkage, default visibility, no side effects, no constructors). Eagle
 * reaches it with dlsym on the library's own handle, so two libraries never
 * collide. The tree is versioned like the seam's OUTPUT structs: each struct
 * starts with ``struct_size``; eagle reads the fields that fit and treats
 * fields past ``struct_size`` as absent. ``schema_major`` must equal
 * EG_META_SCHEMA_MAJOR; within a major the tree is append-only. Structs are
 * reached through pointers or pointer arrays, never as arrays of structs.
 * The structs below ``eg_unit_meta`` (body, launch, finish, team) are
 * forward-declared here and defined by the typed-meta design.
 */
#define EG_META_SCHEMA_MAJOR 1u
#define EG_UNIT_LIBRARY_META_SYMBOL "eg_unit_library_meta"
/** Bytes of a unit digest. Hawk defines what it hashes; eagle only compares. */
#define EG_DIGEST_BYTES 32

struct eg_unit_body;   /**< the typed walk (eg_unit_meta.h): slots, cells, keyed and team planes, ... */
struct eg_unit_launch; /**< lane space, block, box, plane axes, wide placement */
struct eg_unit_finish; /**< mask slot, steps (or EG_STEPS_AUTO), steps_max */
struct eg_unit_team;   /**< a stage's team record */
struct eg_unit_derived; /**< what a derived unit is to its primal: tape, reserved slots, slot sources */
struct eg_unit_gemm;    /**< the matrix products of a dense-contraction unit (eg_unit_meta.h) */

/** A unit named file-independently: its name and digest, never a path. */
struct eg_unit_ref {
    uint32_t struct_size;
    uint32_t reserved0;
    const char* name;
    uint8_t digest[EG_DIGEST_BYTES];
};

/** One stage of a split unit: the stage unit by reference, its cells and team. */
struct eg_unit_stage {
    uint32_t struct_size;
    int32_t n_cells_in, n_cells_out;
    int32_t reserved0;
    const struct eg_unit_ref* unit;
    const char* const* cells_in;
    const char* const* cells_out;
    const struct eg_unit_team* team; /**< NULL: no team */
};

/** One unit of the library. */
struct eg_unit_meta {
    uint32_t struct_size;
    uint32_t flags;                     /**< EG_UNIT_CALLER_CARDS (eg_unit_meta.h); other bits are refused */
    const char* name;
    uint8_t digest[EG_DIGEST_BYTES];
    int32_t pattern;                    /**< enum of the typed-meta design */
    int32_t scalar_type;                /**< 0 float64, 1 float32, 2 banded: a REAL slot is an 8-byte cell crossing as unsigned 64-bit, made with eg_banded_pack (refused on a host structure as not written) */
    const char* host_entry;             /**< symbol of the serial v2 entry, NULL: none */
    const char* device_image;           /**< symbol of the embedded device image, NULL: none */
    const char* device_function;        /**< kernel name inside that image */
    uint64_t device_image_bytes;        /**< 0 when ``device_image_size`` names the size */
    int32_t device_image_format;        /**< EG_IMAGE_* of eg_unit_meta.h: 0 none, 1 ptx, 2 cubin, 3 fatbin */
    int32_t n_stages;                   /**< 0: not a split unit */
    const struct eg_unit_stage* const* stages;
    const struct eg_unit_body* body;
    const struct eg_unit_launch* launch;
    const struct eg_unit_finish* finish; /**< NULL: no finish */
    const char* device_image_size;      /**< symbol of a ``uint64_t`` holding the image's byte count
                                             (written at compile time); NULL when ``device_image`` is NULL */
    const struct eg_unit_derived* derived; /**< NULL: a primal unit (eg_unit_meta.h) */
    const char* device_compiler;        /**< the compiler that made the image, e.g. "nvcc 12.6"; NULL: no image */
    const char* device_flags;           /**< its flags, as given; NULL: no image */
    int32_t device_ptx_isa;             /**< PTX ISA of the image's PTX, major * 10 + minor (85 = 8.5); 0: no PTX.
                                             eg_unit_load refuses an image with no machine code for the device
                                             whose PTX is newer than the driver takes */
    int32_t reserved1;
    const struct eg_unit_gemm* gemm;    /**< NULL: none; the matrix products a DEVICE plan runs instead of a kernel (eg_unit_meta.h) */
};

/** The top of the tree: what ``eg_unit_library_meta`` returns. */
struct eg_library_meta {
    uint32_t struct_size;
    uint32_t schema_major, schema_minor;
    uint32_t n_units;
    const struct eg_unit_meta* const* units;
};

/** The exported symbol's type (eagle dlsyms EG_UNIT_LIBRARY_META_SYMBOL). */
typedef const struct eg_library_meta* (*eg_unit_library_meta_fn)(void);

/** The function every unit library defines (EG_UNIT_LIBRARY_META_SYMBOL);
 *  declared here so a unit library including this header has its definition
 *  checked. Eagle itself never links it: it reaches it by dlsym. */
const struct eg_library_meta* eg_unit_library_meta(void);

/* ---- structs ------------------------------------------------------------- */

/** INPUT to ``eg_unit_load``. */
struct eg_unit_load_desc {
    uint32_t struct_size;
    uint32_t flags;                 /**< none defined: must be 0 */
    const char* path;               /**< the unit library (hawk's publishing step chose it) */
    const struct eg_unit_ref* unit; /**< which unit of the library, by name + digest */
    eg_unit* const* deps;           /**< already-loaded units its stages name */
    int32_t n_deps;
    int32_t reserved0;
};

/** Settings flags. */
#define EG_CTX_DEVICE (1u << 0) /**< ``device`` is a device ordinal (device group); clear: host only */

/** INPUT to ``eg_ctx_new``. Zero-filled with ``abi_major`` set, it is a host-only context. */
struct eg_ctx_settings {
    uint32_t struct_size;
    uint32_t flags;        /**< EG_CTX_* */
    uint32_t abi_major;    /**< EG_DOOR_VERSION_MAJOR the caller was built for */
    int32_t device;        /**< with EG_CTX_DEVICE: the device ordinal, >= 0; else must be 0 */
};

/** OUTPUT of ``eg_ctx_info``. */
struct eg_ctx_info {
    uint32_t struct_size;
    uint32_t abi_major, abi_minor;
    int32_t device;          /**< -1 host only */
    uint64_t capabilities;   /**< EG_CAP_* */
    int32_t host_threads;    /**< threads a host launch of this context uses now */
    int32_t device_type;     /**< DLPack device type of ``device`` (0 when host only) */
    int64_t stream;          /**< the DLPack stream code a producer must order against
                                  (``__dlpack__(stream=)``); -1 when host only */
    int32_t weight_vendor;   /**< the class key the dispatch weights were looked up by: host, 1 Intel / 2 AMD / 0 other (CPUID vendor), 0 also when the host has no CPUID class (not x86, or CPUID unreadable); device, 0 */
    int32_t weight_family;   /**< host: the CPUID display family (read for any x86 vendor), 0 with no CPUID class; device: the compute capability major */
    int32_t weight_model;    /**< host: the CPUID display model, 0 with no CPUID class (vendor, family and model all 0: no class key, the built-in row); device: the compute capability minor */
    int32_t weight_row;      /**< the index of the EG_WEIGHT_ROWS row taken, -1 for the built-in row of the target (no row of that class) */
};

/** ``eg_plan_desc.flags``: split the range evenly into ``n_partitions`` parts (``part_base`` and ``part_count`` NULL):
 *  the first ``range % n_partitions`` parts take one more, a range shorter than the parts leaves the last ones empty. */
#define EG_PLAN_SPLIT_EVENLY (1u << 0)

/** INPUT to ``eg_plan_new``. The range the partitions cut is the samples [0, n_samples); for a thread-box unit it is
 *  its threads, [0, ceil(n_samples / (t * stride)) * stride), so a partition counts threads and never splits a box.
 *  A unit that reads an index set or a prefix count takes ONE partition, whose [base, base + count) are POSITIONS of
 *  the set (the first entries of its map, or the first lanes of a prefix); a launch runs those below the live count. */
struct eg_plan_desc {
    uint32_t struct_size;
    uint32_t flags;               /**< EG_PLAN_SPLIT_EVENLY or 0 */
    eg_unit* unit;                /**< the source unit; its stages ride in its meta */
    eg_structure structure;
    eg_reduce_policy policy;      /**< order of the unit's sums over samples */
    int64_t n_samples;
    int64_t n_partitions;         /**< 0: one partition, the whole range; with EG_PLAN_SPLIT_EVENLY the number of parts (>= 1) */
    const int64_t* part_base;     /**< n_partitions entries, else NULL (always NULL with EG_PLAN_SPLIT_EVENLY) */
    const int64_t* part_count;    /**< n_partitions entries, else NULL (always NULL with EG_PLAN_SPLIT_EVENLY) */
};

/** INPUT to ``eg_bind`` and ``eg_reverse``: planes and uniforms by SLOT, the
 *  index in the unit meta's slot table (hawk resolves names to slots). */
struct eg_bind_desc {
    uint32_t struct_size;
    uint32_t flags;                     /**< none defined: must be 0 */
    int32_t n_planes;
    int32_t n_uniforms;
    const int32_t* plane_slots;         /**< n_planes slot indices */
    eg_plane* const* planes;            /**< n_planes planes (eagle takes a reference) */
    const int32_t* uniform_slots;       /**< n_uniforms slot indices */
    const void* const* uniform_values;  /**< each read at the slot's wire size (at most 8 bytes: bool 1, int32/uint32 4,
                                             int64 8, a real 4 or 8 by the unit's scalar); bytes beyond it are ignored */
};

/** Runner flags. */
#define EG_RUN_RECORD (1u << 0)  /**< record each run for ``eg_reverse`` */
#define EG_RUN_REORDER (1u << 1) /**< reorder the bound planes at ``reorder_theta`` */

/** INPUT to ``eg_runner_new``. */
struct eg_runner_desc {
    uint32_t struct_size;
    uint32_t flags;                     /**< EG_RUN_* */
    int32_t n_plans;                    /**< plans of one step, launched in order */
    int32_t n_derived;
    eg_plan* const* plans;              /**< bound plans; the first carries the finish mask */
    const eg_derived_role* derived_roles;
    eg_unit* const* derived_units;      /**< what EG_RUN_RECORD / EG_EVENTS_REFINE need */
    int64_t max_steps;                  /**< a compacting run stops after ceil(max_steps / every) cadences of ``every`` steps, a
                                             non-compacting K-step run after ceil(max_steps / K) launches, so a sample may take up
                                             to the next whole cadence or launch; an AUTO run cuts its last launch so no sample
                                             exceeds max_steps. The report's steps counts steps launched (a sample finished
                                             inside a launch does not advance it). */
    int64_t every;                      /**< compaction cadence in steps; 0: the runner's choice */
    int64_t window;                     /**< record window; 0: clamp(ceil(sqrt(max_steps)), 16, 1024); -1: every step */
    uint64_t memory_budget;             /**< bytes for the record; 0: half the free memory */
    double reorder_theta;               /**< read only with EG_RUN_REORDER */
    eg_steps_reverse steps_reverse;     /**< PER_STEP, FUSED (needs EG_RUN_RECORD, the K-step vjp and window K) or AUTO:
                                             a record of a K-step unit (finish.steps K > 1, window 0 or K, events STEP)
                                             takes the row of the unit's table for this target (host_steps_reverse for a
                                             host runner, device_steps_reverse for a device one) nearest (state width,
                                             K) by log2 distance, ties to the smaller (width, K, lo, hi); FUSED when
                                             the run's regime (n, or n / (SMs * resident threads per SM) for a
                                             per-resident table) lies in the row, and the window becomes K. Any other
                                             AUTO, and a unit with no table or no rows, is PER_STEP. A table that is
                                             per-resident on a runner whose device cannot state its SM count and
                                             resident threads is BAD_REQUEST. AUTO takes the derived units of FUSED; the
                                             K-step vjp is left unused when PER_STEP is chosen */
    eg_events events;
    eg_no_event on_no_event;
    int32_t reserved0;
    eg_graph* step_graph;               /**< per-step work, or NULL: a graph recorded on the runner's own side (a host graph for a host
                                             runner, a device graph for a device one) that runs once per step, BEFORE the step's plans,
                                             on every launch, on the planes it was recorded over. It may hold launches, copies, reduces, transforms,
                                             products, forks and, on the host, conditionals; never a run of a runner (a device graph
                                             with a conditional is NOT_WRITTEN: the driver refuses a conditional node inside the loop's
                                             own conditional body). Each plan must advance exactly one step per launch (finish steps 1): a unit that
                                             advances K > 1 steps per launch, or an automatic count, is BAD_REQUEST, as the per-step hook
                                             nodes of eagle's Python runner were (read, not run: it refused them the same way).
                                             On the host the graph is replayed ahead of each launch, with an active set ahead of each of
                                             the ``every`` launches between compactions, over all samples, not the live prefix, and
                                             stops with the plans when every sample finished. On a device it is captured into the loop
                                             as a child graph of the loop body, so it runs wherever the plans do: for each launch of every
                                             whole interval the loop runs, the last one included even when the batch finished inside it
                                             (eagle's Python device loop captured its hook nodes in the same body, read, not run), and a plan of it rebound after the
                                             first run makes the loop capture again at the next run; a device graph holding host nodes
                                             is BAD_REQUEST. The runner holds a reference to the graph, released at eg_runner_free:
                                             eg_graph_free drops the caller's reference and the graph lives until its last runner is
                                             freed. BAD_REQUEST too: a graph of another context or side,
                                             a graph ended with EG_RECORD_DIFF, and, with EG_RUN_REORDER, a per-sample plane only the
                                             graph touches (the reorder moves the plans' planes, not it). With EG_RUN_RECORD or events
                                             REFINE it is NOT_WRITTEN: a recording run stores nothing of the graph and so cannot reverse
                                             it. */
};

/** Report flags. */
#define EG_REPORT_DONE (1u << 0)        /**< finished == n */
#define EG_REPORT_EXACT_STEPS (1u << 1) /**< ``steps`` is exact */
#define EG_REPORT_SLOT_FILL (1u << 4)   /**< ``slot_fill`` is set */

/** OUTPUT of ``eg_run``. */
struct eg_run_report {
    uint32_t struct_size;
    uint32_t flags;          /**< EG_REPORT_* */
    eg_run_mode mode;
    int32_t reserved0;
    int64_t n;               /**< samples */
    int64_t finished;        /**< samples whose mask is set at the end */
    int64_t launches;        /**< loop iterations */
    int64_t steps;           /**< steps the loop ran */
    int64_t compactions;
    int64_t reorders;
    int64_t mapped;          /**< samples launched over, summed over launches */
    double build_s;          /**< one-time capture wall; 0 on host */
    double slot_fill;        /**< with EG_REPORT_SLOT_FILL only */
};

/** OUTPUT of ``eg_runner_figures``. ``block``, ``chunk`` and ``grid`` are -1 where the runner has no such launch. */
struct eg_runner_figures {
    uint32_t struct_size;
    uint32_t flags;          /**< none defined */
    eg_run_mode mode;        /**< EG_MODE_HOST for a host runner; a device runner's mode as in ``eg_run_report`` */
    int32_t block;           /**< threads per block of the launch the runner's unit made */
    int32_t chunk;           /**< PERSIST only: steps one persistent thread takes per sample claim */
    eg_steps_reverse steps_reverse; /**< PER_STEP or FUSED: the reverse mode of a recorded runner (steps_reverse AUTO resolved
                                         at eg_runner_new); PER_STEP for a runner without EG_RUN_RECORD */
    int64_t grid;            /**< blocks of that launch */
    /* appended: a struct_size that ends before a member reads 0 */
    int32_t reverse_passes;  /**< the memory passes of the last ``eg_reverse`` of this runner; 0 before one */
    int32_t reverse_bitwise; /**< 1 once an ``eg_reverse`` ran: its sums over samples are the canonical compensated pair, bitwise; 0 before one */
};

/** Entry flags. */
#define EG_ENTRY_DEVICE (1u << 0)  /**< the plan has a device function: its figures are set */
#define EG_ENTRY_PERSIST (1u << 1) /**< the image has a persist entry: ``persist_*`` are set */
#define EG_ENTRY_GEMM (1u << 2)    /**< a gemm unit the door routes: ``eg_launch`` queues its cuBLAS products, not the
                                        device function (EG_ENTRY_DEVICE is clear and every figure 0). Set only for
                                        a DEVICE plan of one partition with no index or active set, cells or
                                        companions (eg_unit_meta.h, gemm units); any other DEVICE plan of a gemm unit
                                        launches its own kernel. */

/**
 * OUTPUT of ``eg_plan_entry_figures``: what the driver reports for the plan's
 * device function when its unit is loaded. Fields of an entry the plan does
 * not have are 0 and their flag is clear. The text size of an entry is not
 * reported: the driver does not state it.
 */
struct eg_entry_figures {
    uint32_t struct_size;
    uint32_t flags;          /**< EG_ENTRY_* */
    int32_t regs;            /**< registers per thread of the device function */
    int32_t max_threads;     /**< most threads per block of the device function */
    int32_t persist_regs, persist_max_threads;   /**< the same for the persist entry */
    int32_t local_bytes;     /**< local memory per thread of the device function (spills and stack) */
    int32_t shared_bytes;    /**< static shared memory per block of the device function */
    int32_t const_bytes;     /**< constant memory of the device function */
    int32_t reserved0;
};

/** OUTPUT of ``eg_plan_counters``. */
struct eg_plan_counters {
    uint32_t struct_size;
    uint32_t flags;          /**< none defined: 0 */
    int64_t finished;        /**< the finished counter's value, -1 when the unit has none */
    int64_t steps;           /**< the steps word's value, -1 when the unit has none */
};

/** Reduce flags. */
#define EG_REDUCE_PAIR (1u << 0) /**< write the (sum, compensation) pair */

/** INPUT to ``eg_reduce``: ``in`` is (columns, n) samples-last; ``out`` holds columns. */
struct eg_reduce_desc {
    uint32_t struct_size;
    uint32_t flags;          /**< EG_REDUCE_PAIR */
    eg_plane* in;
    eg_plane* out;
    eg_plane* comp;          /**< compensation companion, or NULL */
    eg_reduce_op op;
    eg_reduce_policy policy;
};

/** ``eg_cond_desc.flags``: IF, IF_ELSE and WHILE test the cell for zero instead of non-zero. SWITCH refuses it. */
#define EG_COND_NEGATE (1u << 0)

/** INPUT to ``eg_record_cond_begin``. The test reads ``cell[cell_index]`` and, when ``baseline`` is set, ``baseline[baseline_index]``
 *  (both at replay time): IF, IF_ELSE and WHILE take the arm when ``cell[cell_index] != baseline[baseline_index]`` (``!= 0`` with no
 *  baseline), or when they are equal under EG_COND_NEGATE. SWITCH takes ``cell[cell_index]`` as the arm index and refuses a
 *  baseline. A ``struct_size`` that ends before ``max_iters`` reads 0, before ``cell_index`` reads index 0 and no baseline. */
struct eg_cond_desc {
    uint32_t struct_size;
    uint32_t flags;          /**< EG_COND_NEGATE, or 0 */
    eg_cond_kind kind;
    int32_t n_arms;          /**< SWITCH only: arm count */
    eg_plane* cell;          /**< a one-dimensional int32 or uint32 plane (a mask reduction or a kernel writes uint32): truth, or the arm index */
    int64_t max_iters;       /**< WHILE only: the loop stops without error after this many iterations; 0: no cap. Another kind refuses
                                  a nonzero value. */
    int64_t cell_index;      /**< the element of ``cell`` the test reads: 0 or more, below its length */
    eg_plane* baseline;      /**< NULL: compare with 0; else a plane of the dtype of ``cell`` (IF, IF_ELSE, WHILE only) */
    int64_t baseline_index;  /**< the element of ``baseline`` read; ignored without a baseline */
};

/** OUTPUT of ``eg_device_props`` (device group): the driver's raw facts of one device and the roofline figures derived
 *  from them (peak rates and ridge points: the figures the dispatch rule reads). */
struct eg_device_props {
    uint32_t struct_size;
    int32_t cc_major, cc_minor;
    int32_t sm_count, warp_size;
    int32_t regs_per_sm, regs_per_block;     /**< 32-bit registers */
    int32_t reserved0;
    int64_t shared_mem_per_block, shared_mem_per_sm; /**< bytes */
    char name[256];                          /**< the driver's device name, NUL-terminated */
    int32_t clock_rate_khz;                  /**< core clock */
    int32_t memory_clock_rate_khz;
    int32_t memory_bus_width_bits;
    int32_t reserved1;
    double peak_bytes_per_s;                 /**< 2 x memory clock (Hz) x bus width (bytes) */
    double peak_flops_sp, peak_flops_dp;     /**< SMs x clock (Hz) x FMA lanes per SM x 2; dp = sp / fp64_ratio */
    double fp64_ratio;                       /**< single- to double-precision throughput, by compute capability */
    double ridge_flops_per_byte_sp;          /**< peak_flops_sp / peak_bytes_per_s (0 when no bandwidth is known) */
    double ridge_flops_per_byte_dp;
};

/* ======================================================================== */
/* HOST GROUP                                                               */
/* ======================================================================== */

/* ---- context ------------------------------------------------------------- */

/**
 * Make a context. On any refusal a context is still made (when memory allows)
 * so its reason can be read with ``eg_error``; every other call on it then
 * returns the same status. ``*out`` is NULL only on EG_FAULT out of memory.
 * Replaces: module import of ``_core`` + the backend load in ``_backend.py``,
 * the module-level registries, caches and locks of the Python package.
 * Calls today: ``eagle::exec::detail::HostPool`` (lazily, at the first team
 * launch); with EG_CTX_DEVICE the seam's ``probe`` / ``set_device``.
 * Status: OK; BAD_REQUEST (other ABI major, unknown flag, struct too small,
 * ``device`` != 0 without EG_CTX_DEVICE, a negative ordinal); NOT_WRITTEN
 * (EG_CTX_DEVICE, no device backend built); BAD_REQUEST also for EG_CTX_DEVICE with "no CUDA driver could be opened"
 * or "the driver sees 0 device(s)"; FAULT (a driver that failed after it opened).
 */
eg_status eg_ctx_new(const struct eg_ctx_settings* settings, eg_ctx** out);

/** Free a context and every handle still made from it. NULL-safe. In a
 *  forked child frees host memory only. With DLPack exports outstanding
 *  (eg_plane_dlpack) the context is only CLOSED: every later call on it except
 *  ``eg_error`` is refused BAD_REQUEST "the context is closed", and everything
 *  is freed when the last export's deleter runs, on the thread that runs it.
 *  The context pointer is not used again after eg_ctx_free. Status: OK. */
eg_status eg_ctx_free(eg_ctx* ctx);

/**
 * The reason of the calling thread's last non-OK status on ``ctx``, UTF-8, two-call protocol:
 * ``*len`` gets the byte count without the terminator; with ``cap > len``
 * the text and a terminator are written. The reason stays until the next
 * non-OK call of this thread on ``ctx``; a refusal of ``eg_error`` itself never
 * overwrites it. With no reason stored for this thread,
 * ``*len`` is 0. The ONE function that reads a reason.
 * Replaces: Python exceptions raised in the package; the seam's thread-local
 * ``eagle_backend_last_error()`` (eagle copies it into the context at once).
 * Status: OK; BAD_REQUEST (ctx NULL, or 0 < cap <= len).
 */
eg_status eg_error(eg_ctx* ctx, char* buf, size_t cap, size_t* len);

/**
 * What this context is: versions, capabilities, host threads in effect, device
 * and the DLPack stream code producers must order against.
 * Replaces: ``eagle.__version__``, ``_core.host_threads``, ``eagle.cuda``
 * probes, ``abi.py`` version checks.
 * Calls today: ``eagle::exec::host_max_threads`` (HostPool.h), seam ``version``
 * and ``capabilities``. Status: OK; BAD_REQUEST.
 */
eg_status eg_ctx_info(eg_ctx* ctx, struct eg_ctx_info* out);

/**
 * Cap this context's host launches at ``threads`` (0: the default budget:
 * ``OMP_NUM_THREADS`` read once, else the physical cores the process may run
 * on, capped by the cgroup quota). Affects this context only.
 * Replaces: ``_core.set_host_threads`` (process-wide today) and
 * ``_core.set_thread_host_team`` (per thread today).
 * Calls today: ``host_threads_setting`` / ``detail::host_thread_team``; the
 * door passes the count per launch instead of setting either.
 * Status: OK; BAD_REQUEST (threads < 0).
 */
eg_status eg_host_threads(eg_ctx* ctx, int32_t threads);

/* ---- units --------------------------------------------------------------- */

/**
 * Load the unit ``desc->unit`` (name + digest) from the library at
 * ``desc->path``. Eagle opens the library, calls EG_UNIT_LIBRARY_META_SYMBOL,
 * finds the unit whose name AND digest match, checks the meta tree, resolves
 * the host entry and (on a device context only) the device image. Each stage
 * the meta names is matched by name + digest against ``desc->deps``: the
 * caller loads stage units first and passes them; eagle never searches files
 * and never calls back. No sidecar file is read. Needs no device to check.
 * Replaces: ``host_launch.HostPlugin.__init__`` (ctypes CDLL + sidecar
 * checks), ``loaded.LoadedKernel``, ``sidecar.validate_sidecar``/``read_*``,
 * ``abi.check_aether_abi``, ``roles.check_launch_certified_pattern``, the
 * stage assembly of ``_split.py``.
 * Calls today: nothing compiled (Python + ctypes); device images through seam
 * ``module_load`` / ``module_function``.
 * Status: OK; BAD_REQUEST (no unit with that name + digest, a stage missing
 * from ``deps`` (named with its digest), meta of another schema major, meta
 * that contradicts itself); NOT_WRITTEN (a pattern, role or scalar type this
 * build lacks); FAULT (the library does not load, or lacks the symbol).
 * A library at a path this process loaded before is read again when the file there is another file (device and
 * inode); the same file loaded twice is one library.
 */
eg_status eg_unit_load(eg_ctx* ctx, const struct eg_unit_load_desc* desc, eg_unit** out);

/** Release a unit (its library stays open while a plan uses it). NULL-safe. */
eg_status eg_unit_free(eg_ctx* ctx, eg_unit* unit);

/* ---- planes -------------------------------------------------------------- */

/**
 * Take a DLPack tensor as a plane. On OK eagle owns ``tensor`` and calls its
 * deleter exactly once, when the last reference drops, which happens only
 * inside eg_plane_free, eg_bind (a replaced slot), eg_plan_free,
 * eg_runner_free or eg_ctx_free, on the calling thread. On any other status
 * the caller still owns it and no deleter runs. Writable unless the tensor's
 * READ_ONLY flag is set.
 * Replaces: ``interop.py`` array-interface reads, ``arrays.py``,
 * ``marshal.py`` address packing, ``_plan_planes`` shape checks.
 * Calls today: ``plugin/dlpack_bridge.h`` (``fromDLPack``-style checks).
 * Status: OK; BAD_REQUEST (null data with nonzero size, device of another
 * context); NOT_WRITTEN (a dtype or device type eagle does not run).
 */
eg_status eg_plane_wrap(eg_ctx* ctx, struct DLManagedTensorVersioned* tensor, eg_plane** out);

/** Drop the caller's reference to a plane. NULL-safe. */
eg_status eg_plane_free(eg_ctx* ctx, eg_plane* plane);

/** Where a plane eagle makes lives (``eg_plane_desc.kind``). */
typedef int32_t eg_mem;
#define EG_MEM_HOST 0   /**< ordinary host memory (DLPack kDLCPU) */
#define EG_MEM_PINNED 1 /**< page-locked host memory the device reads by DMA (kDLCUDAHost); device contexts only */
#define EG_MEM_DEVICE 2 /**< device memory of the context's device (kDLCUDA); device contexts only */

/** ``eg_plane_desc.flags``. */
#define EG_PLANE_ZERO (1u << 0) /**< the bytes start at zero; without it they are unspecified */

/** INPUT to ``eg_plane_new``. The dtype is the DLPack code and bit width a
 *  plane may carry (as ``eg_plane_wrap`` accepts): float 64/32, int 64/32/8,
 *  uint 32, bool 8, one lane. */
struct eg_plane_desc {
    uint32_t struct_size;
    uint32_t flags;          /**< EG_PLANE_* */
    eg_mem kind;
    int32_t ndim;
    uint32_t dtype_code;     /**< DLDataTypeCode: kDLInt 0, kDLUInt 1, kDLFloat 2, kDLBool 6 */
    uint32_t dtype_bits;
    const int64_t* shape;    /**< ndim extents, each >= 0 (ndim 0: one element) */
};

/**
 * Make a plane eagle owns, in host, pinned or device memory, through the one
 * allocator eagle uses for its own planes. ``*out`` is the caller's reference,
 * dropped by eg_plane_free like any plane's; the memory is freed when the last
 * reference drops (device and pinned memory after the work queued on the
 * context's device). Allocation is not a recorded node: a device context
 * refuses it between eg_record_begin and eg_record_end.
 * Replaces: the binding's ``install_allocator`` (array_binding.h).
 * Calls today: ``eagle_backend_mem_alloc`` / ``_release`` (eagle_cuda.cu).
 * Status: OK; BAD_REQUEST (unknown ``kind`` or flag bit, ``ndim`` < 0 or
 * ``shape`` NULL with ``ndim`` > 0, a negative extent, a size that overflows,
 * a call inside a device recording); NOT_WRITTEN (a dtype eagle does not run,
 * PINNED or DEVICE on a host-only context); FAULT (the allocation failed).
 */
eg_status eg_plane_new(eg_ctx* ctx, const struct eg_plane_desc* desc, eg_plane** out);

/**
 * Export a plane as a DLPack tensor, the inverse of eg_plane_wrap: ``*out``
 * views the plane's memory (kDLCPU, kDLCUDAHost for a pinned plane, kDLCUDA)
 * and holds a reference to it, so eg_plane_free may follow at once. The
 * consumer calls ``*out``'s deleter once when done; it may run on any thread
 * (it waits for a call in progress on the context) and the memory is freed
 * when the last reference drops. eg_ctx_free with an export unreleased closes
 * the context (see eg_ctx_free); the last export's release frees it, and the
 * exported memory stays valid until then.
 * Replaces: ``export_dlpack`` (array_binding.h) over eagle's own allocations.
 * Calls today: ``export_dlpack`` (array_binding.h).
 * Status: OK; BAD_REQUEST (``out`` NULL, a plane not of this context or already
 * freed by the caller).
 */
eg_status eg_plane_dlpack(eg_ctx* ctx, eg_plane* plane, struct DLManagedTensorVersioned** out);

/* ---- plans --------------------------------------------------------------- */

/**
 * A plan: the unit under one structure over fixed partitions. Eagle makes the
 * planes the caller never passes (companions, cell scratch, team scratch,
 * the finished counter) when the plan is bound.
 * Replaces: ``plan.plan``, ``plan.Plan``, ``_split`` stage plans.
 * Calls today: ``eagle::exec::Partition`` (Partition.h),
 * ``exec::check_placement``. Status: OK; BAD_REQUEST (partitions that
 * overlap or leave the range, the range named; EG_PLAN_SPLIT_EVENLY with arrays or with n_partitions < 1); NOT_WRITTEN (DEVICE without the device group).
 */
eg_status eg_plan_new(eg_ctx* ctx, const struct eg_plan_desc* desc, eg_plan** out);

/** Free a plan and its bindings. NULL-safe. */
eg_status eg_plan_free(eg_ctx* ctx, eg_plan* plan);

/**
 * Bind planes and uniforms by slot. Later binds replace earlier ones slot by
 * slot. Shapes, dtypes, devices and write access are checked against the
 * unit meta here, once, not at launch.
 * Binding the target of a compensated sum zeroes its companion, eagle's own
 * plane, whatever it held (in place when the target's shape and dtype are
 * unchanged, else made again); the caller cannot bind a companion
 * (BAD_REQUEST). So a caller that
 * zeroes a target in place and binds that slot again restarts the residual too
 * (the old ``BoundPlan.clear``). An ordered scatter's TARGET scratch plane is
 * made again the same way, sized by the target. A bind touches only the slots
 * it names: every other slot keeps its plane and its state between launches.
 * Refused BAD_REQUEST while recording (between eg_record_begin and
 * eg_record_end): bind before recording.
 * A unit with an active set (its map and count slots) takes the caller's planes there on a plan no runner owns: the
 * map one dimension of the slot's integer dtype, the count one cell (1,). A plain eg_launch then runs the positions
 * the map lists, in the order of the map and in any order, up to the count (refused BAD_REQUEST by name when the count
 * is above the map's length); each position at most once is the caller's promise, which is not scanned. A plan a
 * runner owns refuses the slots, and eg_runner_new refuses a plan that holds the caller's map or count.
 * Replaces: ``Plan.bind``, ``_plan_binding.BoundPlan``, ``_plan_pack``,
 * ``marshal``, ``_lane_bind`` (team grids, chunk counts), ``_cell_rows``.
 * Calls today: nothing compiled (Python packs ``ctypes`` addresses).
 * Status: OK; BAD_REQUEST (unknown slot, wrong shape/dtype/device, read-only
 * plane in a written slot, an eagle-made plane passed).
 */
eg_status eg_bind(eg_ctx* ctx, eg_plan* plan, const struct eg_bind_desc* desc);

/**
 * Zero the named OUTPUT slots' bound planes in place, and what eagle owns for
 * each: its compensation companion (zeroed) and an ordered scatter's target
 * scratch plane (made again, as eg_bind does for a rebound target). On a
 * DEVICE plan the zeroing is queued on the context's stream, on a host plan it
 * is done before return. A call that fails changes nothing. ``n_slots == 0``
 * is OK and does nothing. Companions are always eagle's (a caller cannot bind
 * one), so every companion of a named target is cleared with it. Call it
 * between launches; it is not a node of a recording.
 * Replaces: ``BoundPlan.clear``.
 * Calls today: nothing compiled (Python fills each plane).
 * Status: OK; BAD_REQUEST (``n_slots`` < 0, ``slots`` NULL with ``n_slots`` > 0,
 * a slot out of range, not an output, an eagle-made plane, unbound or named
 * twice, a call while recording).
 */
eg_status eg_clear(eg_ctx* ctx, eg_plan* plan, int32_t n_slots, const int32_t* slots);

/**
 * Write ``m`` into the bound index count plane of a PREFIX set (an index
 * count slot with no map slot), in the context's stream order with no host
 * wait: on a DEVICE plan the write is queued behind the work already issued,
 * on a host plan it is done before return. ``m == 0`` is legal and runs
 * nothing; the capacity is the sample count. A lookup set's count is the
 * caller's own plane: BAD_REQUEST, bind it again. A call that fails changes
 * nothing. Call it between launches or replays, not while recording. A
 * recorded launch of an index set (host or DEVICE) reads the count plane at
 * every replay, never a copy taken at record time: the grid is the capacity,
 * the unit's own guard skips the lanes at or above the count, so a replay runs
 * at the count last written, with no new recording. A replay does not check the
 * count (a plain DEVICE launch reads it back and does); a lookup set's count
 * is the caller's plane to keep within its map's entries.
 * Replaces: ``_set_count``, ``_index_set.write_count``.
 * Calls today: nothing compiled (Python writes the plane).
 * Status: OK; BAD_REQUEST (a unit with no index count slot, a lookup set, the
 * count slot unbound, ``m`` above the sample count, a call while recording).
 */
eg_status eg_set_count(eg_ctx* ctx, eg_plan* plan, uint32_t m);

/**
 * Host float64 to 8-byte banded cells: ``out[i]`` is the cell of ``in[i]``,
 * for ``n`` values, on the host, before return. Stops at the first value
 * outside the band (its reason names the element and carries the band's own
 * text); ``out`` is then undefined. ``n == 0`` is OK and touches nothing.
 * Replaces: the float64 to banded conversion of the Python array binding.
 * Calls today: ``aether::banded::BandedReal::fromDouble(x).toBits()``.
 * Status: OK; BAD_REQUEST (``in`` or ``out`` NULL with ``n`` > 0, a value
 * outside the band).
 */
eg_status eg_banded_pack(eg_ctx* ctx, const double* in, uint64_t* out, uint64_t n);

/**
 * Banded cells to host float64: ``out[i]`` is the value of ``in[i]``, for
 * ``n`` cells, on the host, before return. Every cell decodes. ``n == 0`` is
 * OK and touches nothing.
 * Replaces: the banded to float64 conversion of the Python array binding.
 * Calls today: ``aether::banded::BandedReal::fromBits(w).toDouble()``.
 * Status: OK; BAD_REQUEST (``in`` or ``out`` NULL with ``n`` > 0).
 */
eg_status eg_banded_unpack(eg_ctx* ctx, const uint64_t* in, double* out, uint64_t n);

/**
 * One float64 to one banded cell, as ``eg_banded_pack`` with ``n == 1``;
 * ``*out`` is untouched on a refusal.
 * Replaces: the scalar float64 to banded conversion of the Python binding.
 * Calls today: ``aether::banded::BandedReal::fromDouble(x).toBits()``.
 * Status: OK; BAD_REQUEST (``out`` NULL, ``x`` outside the band).
 */
eg_status eg_banded_pack_scalar(eg_ctx* ctx, double x, uint64_t* out);

/**
 * Launch the bound plan once over each of its partitions. ``*tiles`` gets the
 * host tiles run (0 for DEVICE). A DEVICE plan of a unit with a gemm record
 * (eg_unit_meta.h) queues its cuBLAS products instead of a kernel.
 * Counters count per launch: eagle zeroes its own finished counter and its own mix counters at the start of each
 * eg_launch (and of each replay of a recorded one), so ``eg_plan_counters`` reads what this launch counted; samples
 * finished before it (a mask set at bind) are not counted. A mix plane the caller binds is never zeroed: the unit adds
 * into it. Inside ``eg_run`` the finished counter keeps counting over the runner's batch (``eg_runner_reset`` clears it).
 * Replaces: ``BoundPlan.launch`` / ``Plan.run`` / ``exec.HostTeam.run``.
 * Calls today: ``eagle::exec::HostTeam::run`` (HostTeam.h) via
 * ``_core.run_host`` (eagle_core.cpp); ``run_serial`` for HOST_SERIAL.
 * Status: OK; BAD_REQUEST (a slot unbound); FAULT.
 */
eg_status eg_launch(eg_ctx* ctx, eg_plan* plan, uint64_t* tiles);

/**
 * Read the device entries of a plan's unit as the driver reported them at
 * load. A host plan, or a unit without a device function (a split unit), has
 * no device entry: EG_ENTRY_DEVICE is clear and every figure is 0. A routed gemm unit sets EG_ENTRY_GEMM instead.
 * Replaces: ``BoundPlan.entry_reports`` (registers, local memory, demand).
 * Calls today: nothing compiled (Python reads the driver itself).
 * Status: OK; BAD_REQUEST (``out`` NULL or its ``struct_size`` too small, a
 * handle of another context).
 */
eg_status eg_plan_entry_figures(eg_ctx* ctx, const eg_plan* plan, struct eg_entry_figures* out);

/**
 * Read a plan's counters now (per launch: see ``eg_launch``): the finished counter (``out->finished``, the samples the
 * last launch finished, -1 when the unit finishes nothing), the steps
 * word of a unit with steps AUTO (``out->steps``, -1 when it has none), and the mix counters of every AUTO dispatch
 * group, concatenated in the unit's dispatch order, k per group, into ``mix``. Two-call protocol as ``eg_error``:
 * ``*n_mix`` always gets the count; with ``mix_cap`` 0 nothing is written; with ``0 < mix_cap < *n_mix`` the call is
 * refused. Each value is read from the plane bound at its slot, eagle's or the caller's. On a DEVICE plan the read is
 * synchronous on the context's stream, after the work queued there. ``struct_size`` as ``eg_plan_entry_figures``.
 * Status: OK; BAD_REQUEST (``out`` or ``n_mix`` NULL, ``mix`` NULL with ``mix_cap`` > 0, ``mix_cap`` negative or too
 * small, a handle of another context).
 */
eg_status eg_plan_counters(eg_ctx* ctx, const eg_plan* plan, struct eg_plan_counters* out, uint32_t* mix, int32_t mix_cap,
                           int32_t* n_mix);

/* ---- runs ---------------------------------------------------------------- */

/**
 * A run-until-done loop over bound plans. All loop policy (cadence,
 * compaction, reorder, entry choice, fused steps, persist sizing, record
 * windows, event refine) is decided here, from the unit meta and the
 * context, never by the caller.
 * Replaces: ``until_done`` / ``Runner.__init__`` (``_until_done.py``),
 * ``_active_set``, ``_index_set``, ``diff/_run.record_planes``,
 * ``diff/events``.
 * Calls today: ``band_k``, ``band_rate``, ``entry_ratio``, ``persist_*``,
 * ``dispatch_*``, ``diff_default_window``, ``diff_chunk_samples``
 * (eagle_core.cpp bindings of the C++ policies).
 * A unit whose finish steps are AUTO runs one step per launch in a step of several plans (the runner writes 1 into
 * its steps word; a caller-bound steps word is refused); a fixed K above 1 there is refused.
 * Status: OK; BAD_REQUEST (a plan without a finish, every/reorder on a plan
 * without an active set, record without its derived units); NOT_WRITTEN.
 */
eg_status eg_runner_new(eg_ctx* ctx, const struct eg_runner_desc* desc, eg_runner** out);

/** Free a runner and its record. NULL-safe. */
eg_status eg_runner_free(eg_ctx* ctx, eg_runner* runner);

/**
 * Put the runner back as eg_runner_new left it, so the next eg_run starts the
 * batch over: the planes a reordering run left permuted return to sample
 * order, then the done mask (the first plan's finish mask plane, the caller's,
 * zeroed in place) and the finished and step counters are cleared, an active
 * set returns to the identity (every sample live, a reordering set's
 * permutation the identity and its counters zero), and an automatic group's
 * mix counters and a segmented group's sort state are cleared. The last
 * recorded run and the last refine results are dropped: eg_reverse then
 * refuses "no recorded run to reverse" and eg_run_events "no refine run". The
 * caller rebinds or rewrites the state planes it wants run again. On a DEVICE
 * runner the writes are queued on the context's stream with no host wait; on a
 * host runner they are done before return.
 * Replaces: ``Runner.reset``, ``ActiveSet.reset``, the dispatch runner's reset.
 * Calls today: nothing compiled (Python clears each piece).
 * Status: OK; BAD_REQUEST (a plan of the runner with a slot unbound, a call
 * while recording).
 */
eg_status eg_runner_reset(eg_ctx* ctx, eg_runner* runner);

/**
 * Run until every sample is done or ``max_steps``; fill ``report``.
 * Replaces: ``Runner.run`` and the host loop of ``_host_loop.run_host``.
 * Calls today: ``HostTeam::run`` per launch; ``diff::recordForward`` and
 * ``events_*_host`` when recording or refining.
 * Status: OK; BAD_REQUEST (refine found no crossing with EG_NO_EVENT_REFUSE);
 * FAULT.
 */
eg_status eg_run(eg_ctx* ctx, eg_runner* runner, struct eg_run_report* report);

/**
 * The mode and the resolved launch of the runner's last ``eg_run``. A host
 * runner has mode EG_MODE_HOST and no launch: ``block``, ``chunk`` and ``grid``
 * are -1. A device runner reports the launch of its first plan's unit
 * (``block`` and ``grid``; for PERSIST the persistent entry's, with its
 * ``chunk``): -1 before the launch is resolved (a BAND or FUSED_ONE runner at
 * its first ``eg_run``, a PERSIST runner at ``eg_runner_new``) and for a unit
 * with no device function of its own (a split unit launches per stage).
 * ``chunk`` is -1 outside PERSIST. No fill, rate or step count is reported: the
 * door does not hold them.
 * Replaces: ``Runner.launch_policy``.
 * Calls today: ``_launch_policy`` (Python).
 * Status: OK; BAD_REQUEST (``out`` NULL or its ``struct_size`` too small, a
 * handle of another context).
 */
eg_status eg_runner_figures(eg_ctx* ctx, const eg_runner* runner, struct eg_runner_figures* out);

/**
 * Write the last EG_EVENTS_REFINE run's per-sample results, both from that
 * same run: ``t_stop`` (n samples, the element type of the primal's time
 * plane) and ``status`` (n samples, int8, EG_EVENT_*). Either may be NULL.
 * Replaces: ``Runner.event_status`` and the event-time read.
 * Status: OK; BAD_REQUEST with the reason "no refine run" (the runner has
 * not run with EG_EVENTS_REFINE) or "plane shape or dtype" (a wrong plane).
 */
eg_status eg_run_events(eg_ctx* ctx, eg_runner* runner, eg_plane* t_stop, eg_plane* status);

/**
 * Reverse the last recorded run: ``seeds`` binds cotangents by slot,
 * ``grads`` binds the caller's gradient planes by slot (overwritten).
 * Both use the PRIMAL unit's slot indices: a seed on a state output, a grad on
 * a state input or a parameter. Eagle maps them to the derived units through
 * their links (eg_unit_derived). A (PRIMAL_SLOT, GRAD) link with ``reduced`` 1 is
 * summed over samples in SAMPLE order (gathered back after a reordering run)
 * with the canonical ordered tree of leaf 4096, always as a compensated pair
 * (sum, compensation); the sum lands in the caller's plane of one value. When
 * ``memory_budget`` splits the reverse into passes, each pass sums its steps
 * that way and the passes' pairs combine in pass order: bitwise the unsplit
 * sum only with one pass. Eagle owns the run-time bindings no caller sees: the forward
 * binds the tape as a one-row dummy with ``tape_on_slot`` 0 and the recompute
 * binds the real tape with 1; state adjoints alternate between two buffers
 * eagle makes, by step parity; the twin's lookup planes (finished count, map)
 * are zeroed or identity-filled; a refine run tapes one row per sample.
 * Replaces: ``diff.reverse(runner.record, ...)`` (``diff/_reverse.py``,
 * ``diff/_run.py``). Calls today: ``diff::recordForward`` replay,
 * ``diff_reverse_host`` (eagle_core.cpp). Status: OK; BAD_REQUEST (runner
 * not recorded, unknown slot, a slot with no link, a dtype mismatch);
 * NOT_WRITTEN (a role with no old-path oracle yet); FAULT.
 */
eg_status eg_reverse(eg_ctx* ctx, eg_runner* runner, const struct eg_bind_desc* seeds,
                     const struct eg_bind_desc* grads);

/* ---- graph operations ---------------------------------------------------- */

/**
 * Reduce ``in`` over samples into ``out``. Recordable.
 * Replaces: ``_graph_reduce`` reduce nodes. Calls today:
 * ``reduce_canonical_host`` (eagle_core.cpp, ``eagle::reduce::canonical``).
 * Status: OK; BAD_REQUEST (shape/dtype); NOT_WRITTEN (an op/dtype pair).
 */
eg_status eg_reduce(eg_ctx* ctx, const struct eg_reduce_desc* desc);

/**
 * Copy ``src`` into ``dst`` (same shape and dtype). Two planes on one side
 * copy on that side; one host plane and one device plane of the context's
 * device copy across, either way, on the context's stream, and the call
 * returns once the copy has finished: the host plane may be pageable or
 * pinned and is the caller's again on return. ``eg_sync`` orders it like any
 * op. A same-side copy is recordable; a host<->device copy is not: inside a
 * recording it is refused NOT_WRITTEN, so copy before or after the recording.
 * Replaces: ``_transfer`` copy nodes.
 * Status: OK; BAD_REQUEST (shape, dtype, access); NOT_WRITTEN (a host<->device
 * copy while recording).
 */
eg_status eg_copy(eg_ctx* ctx, eg_plane* src, eg_plane* dst);

/* ---- transforms ---------------------------------------------------------- */

typedef int32_t eg_fft_kind;
#define EG_FFT_C2C 0 /**< complex to complex; forward unless EG_FFT_INVERSE */
#define EG_FFT_R2C 1 /**< real to complex, forward */
#define EG_FFT_C2R 2 /**< complex to real, inverse; the input is left unchanged */

/** numpy's ``norm``, with n the product of the transformed lengths (the REAL side for R2C and C2R). */
typedef int32_t eg_fft_norm;
#define EG_FFT_NORM_BACKWARD 0 /**< forward unscaled, inverse scaled by 1/n */
#define EG_FFT_NORM_FORWARD 1  /**< forward scaled by 1/n, inverse unscaled */
#define EG_FFT_NORM_ORTHO 2    /**< both scaled by 1/sqrt(n) */

/** ``eg_fft_desc.flags``. */
#define EG_FFT_INVERSE (1u << 0) /**< C2C only: the inverse direction */
#define EG_FFT_ADJOINT (1u << 1) /**< the plan is the adjoint of the transform described: its ``in`` has that
                                      transform's output shape and its ``out`` that transform's input shape */

/**
 * INPUT to ``eg_fft_new``. Complex data is a float plane whose last axis has
 * extent 2 (real part, imaginary part); ``shape`` counts complex elements, or
 * for R2C and C2R the elements of the REAL side, and never the trailing 2. A
 * real transform halves the last of ``axes``: its complex side has
 * ``shape[axis] / 2 + 1`` bins there (n even: bins 0 and n/2 are real).
 */
struct eg_fft_desc {
    uint32_t struct_size;
    uint32_t flags;          /**< EG_FFT_* */
    eg_fft_kind kind;
    eg_fft_norm norm;
    uint32_t dtype_bits;     /**< 32 or 64: the float width of both planes */
    int32_t ndim;            /**< data axes, >= 1 */
    const int64_t* shape;    /**< ndim extents; a transformed axis needs one >= 1 */
    int32_t n_axes;          /**< >= 1 */
    const int32_t* axes;     /**< n_axes distinct axes in [0, ndim) */
};

/**
 * Make a transform plan for the context's target: a host context runs
 * pocketfft, a device context cuFFT on the context's stream. cuFFT is opened
 * at the first device plan (libcufft.so.12, then .11) and is not linked, so
 * the door loads without it. With EG_FFT_ADJOINT the plan is the adjoint
 * T^H of the transform T the descriptor names, as the real inner product
 * over the float data: C2C runs the other direction with norm BACKWARD and
 * FORWARD swapped (ORTHO stays); R2C^H is C2R and C2R^H is R2C, norm swapped,
 * and the door applies the Hermitian weight, which for the bins k of the
 * halved axis with 0 < k < last (last = m - 1 for even n, m for odd n, with
 * m = n / 2 + 1) halves the input of R2C^H before it runs and doubles the
 * output of C2R^H after. So the jvp of a transform is a run of its own plan
 * and its vjp a run of its adjoint plan; no elementwise multiply is the
 * caller's. Allocates (device: cuFFT work area and the C2R input copy), so it
 * is not recordable: refused between eg_record_begin and eg_record_end.
 * Replaces: ``aether.fft.plan`` through ``eagle.fft`` and its Hermitian
 * weight steps. Calls today: ``aether::fft::c2c`` / ``r2c`` / ``c2r``,
 * and cuFFT through ``FftPlan`` (cufft_plan.h).
 * Status: OK; BAD_REQUEST (unknown flag, kind or norm; ``dtype_bits`` not a
 * float width; ``ndim`` < 1, ``shape`` NULL, a negative extent or a
 * transformed axis of length 0; no axes, a repeated or out-of-range axis;
 * INVERSE on R2C or C2R; a call while recording); NOT_WRITTEN (float width 16
 * or 128, a size past the 32-bit counts cuFFT takes); FAULT (cuFFT not
 * loadable, naming the files tried; a cuFFT or CUDA error).
 */
eg_status eg_fft_new(eg_ctx* ctx, const struct eg_fft_desc* desc, eg_fft** out);

/**
 * Run the plan once, ``in`` -> ``out``. Device: queued on the context's
 * stream with no allocation and no wait, so legal inside a recording (the
 * graph keeps the plan alive); host: run before return, or kept and replayed
 * while recording. ``in`` == ``out`` is allowed for C2C only. The planes are
 * on the plan's target (host or device), compact, ``out`` writable, float of
 * the plan's width, with the shapes the descriptor implies: C2C (shape, 2)
 * both; R2C (shape) in, (shape with the halved axis n/2+1, 2) out; C2R the
 * reverse; an ADJOINT plan the shapes of the transform it runs.
 * Replaces: ``eagle.fft._lower`` (``plan.execute`` and the weight multiplies).
 * Calls today: ``aether::fft::c2c`` / ``r2c`` / ``c2r``, ``FftPlan::execute``.
 * Status: OK; BAD_REQUEST (a plane of the wrong width, shape, target or
 * access, R2C or C2R with ``in`` overlapping ``out``, a C2C overlap that is
 * not exact); FAULT (a cuFFT or CUDA error).
 */
eg_status eg_fft_run(eg_ctx* ctx, eg_fft* plan, eg_plane* in, eg_plane* out);

/** Drop the caller's reference to a plan; a recording that runs it keeps it. NULL-safe. */
eg_status eg_fft_free(eg_ctx* ctx, eg_fft* plan);

/* ---- matrix products ----------------------------------------------------- */

/** ``eg_gemm_desc.flags`` (also ``eg_unit_gemm_call.flags``). */
#define EG_GEMM_TRANS_A (1u << 0)    /**< op(A) = A^T: A is stored (k, m) */
#define EG_GEMM_TRANS_B (1u << 1)    /**< op(B) = B^T: B is stored (n, k) */
#define EG_GEMM_TRANS_C (1u << 2)    /**< C is stored through a transposed view: (n, m) */
#define EG_GEMM_ACCUMULATE (1u << 3) /**< C += product (beta 1); without: C = product (beta 0) */

/**
 * INPUT to ``eg_gemm_new``: the product ``C = op(A) op(B)`` (``C += `` with EG_GEMM_ACCUMULATE) of three row-major
 * matrices, stated as STORED shapes: op(A) is (m, k) and A is stored (k, m) under EG_GEMM_TRANS_A, op(B) is (k, n) and
 * B is stored (n, k) under EG_GEMM_TRANS_B, C is stored (m, n), or (n, m) under EG_GEMM_TRANS_C.
 */
struct eg_gemm_desc {
    uint32_t struct_size;
    uint32_t flags;                /**< EG_GEMM_* */
    uint32_t dtype_bits;           /**< 32 or 64: the float width of the three planes */
    int32_t reserved0;
    int64_t a_rows, a_cols, b_rows, b_cols, c_rows, c_cols; /**< the stored shapes, each in 1 .. 2^31 - 1 */
};

/**
 * Make a matrix product for the context's target: a host context runs a plain ordered triple loop (reference speed,
 * deterministic, the same bits at any thread count), a device context cuBLAS (opened at the first device product or
 * gemm plan, libcublas.so.13 then .12, never linked) on the context's stream, with the handle made here as
 * ``eg_plan_new`` makes a gemm plan's. A reduce against ones is a product with ``b_cols`` 1 whose ones plane the
 * caller owns. Allocates (the handle), so it is not recordable: refused between eg_record_begin and eg_record_end.
 * Replaces: ``eagle.gemm.plan.lower_dense_gemm`` caller-built plans and the old host twin.
 * Calls today: ``cublasSgemm`` / ``cublasDgemm`` through ``gemm_launch`` (device/Gemm.cpp); the host loop is new.
 * Status: OK; BAD_REQUEST (``out`` or ``desc`` NULL, ``struct_size`` too small, unknown flag bits, ``dtype_bits`` not
 * 32 or 64, an extent below 1 or above 2^31 - 1, contracted extents that disagree, a C stored other than the product's,
 * a call while recording); FAULT (cuBLAS not loadable, naming the files tried; a cuBLAS or CUDA error).
 */
eg_status eg_gemm_new(eg_ctx* ctx, const struct eg_gemm_desc* desc, eg_gemm** out);

/**
 * Run the product once over the planes ``a``, ``b``, ``c`` (the plan's stored shapes). Device: queued on the context's
 * stream with no allocation and no wait, so legal inside a recording; host: run before return, or kept and replayed
 * while recording. Each plane is 2-D, on the product's target, float of its width, writable for ``c``, with ``rows``
 * rows as stated and a row stride (its second extent) of at least the stated columns, so a wider plane serves a column
 * slice. ``c`` must not overlap ``a`` or ``b``. A recording keeps the handle's shape, not the handle: ``eg_gemm_free``
 * before the graph runs is safe; the planes stay the caller's, alive until the graph has run.
 * Replaces: ``eagle.gemm`` run steps. Calls today: ``gemm_launch`` (device/Gemm.cpp).
 * Status: OK; BAD_REQUEST (a NULL or foreign handle, a plane of the wrong rank, rows, row stride, width, target or
 * access, ``c`` overlapping an input); FAULT (a cuBLAS or CUDA error).
 */
eg_status eg_gemm_run(eg_ctx* ctx, eg_gemm* gemm, eg_plane* a, eg_plane* b, eg_plane* c);

/** Drop the caller's reference to a product; a host recording that runs it keeps it. NULL-safe. */
eg_status eg_gemm_free(eg_ctx* ctx, eg_gemm* gemm);

/* ---- host nodes ---------------------------------------------------------- */

/**
 * The caller's function of a host node. ``planes`` holds ``n`` borrowed views, in the order the planes were declared at
 * ``eg_host_call``: each entry is a ``const DLTensor*`` of dlpack.h (typed ``void`` here: this header includes no DLPack header),
 * kDLCPU, or kDLCUDA for a device plane. A view is valid only during the call. The function reads the planes
 * declared EG_ACCESS_READ and writes the ones declared EG_ACCESS_WRITE; a write to any other plane is the caller's fault, which the
 * door neither detects nor repairs. It must not call the door on this context: the context is busy, and such a call is refused
 * BAD_REQUEST naming the host node. It runs on the thread that called ``eg_host_call``, ``eg_graph_launch``, ``eg_run`` or
 * ``eg_graph_reverse``, never on a driver thread. Nonzero return = failure.
 */
typedef int32_t (*eg_host_fn)(void* user, int32_t n, const void* const* planes);

/**
 * The reverse of a host node's function, called by ``eg_graph_reverse`` for a graph ended with EG_RECORD_DIFF. All four arrays
 * hold ``n`` entries, one per declared plane, each a ``DLTensor*`` as in ``eg_host_fn``, and a view is valid only during the call. ``primals[i]`` is the plane declared with
 * EG_ACCESS_READ as the node read it (the door keeps a copy when a later node overwrites it), else NULL. ``adjoints_in[i]`` is
 * the adjoint of a float plane declared with EG_ACCESS_WRITE, else NULL. ``adjoints_out[i]`` is a fresh zeroed view for a float
 * plane declared with EG_ACCESS_READ, else NULL: the function writes into it the adjoint it pulls back to that plane, and the door
 * adds it to the running adjoint of the plane (a plane declared READ|WRITE has its running adjoint replaced by it, as the forward
 * overwrote the plane). A plane that is not float has no adjoint: its ``adjoints_in`` and ``adjoints_out`` entries are NULL. The
 * restrictions of ``eg_host_fn`` apply. Nonzero return = failure.
 */
typedef int32_t (*eg_host_vjp_fn)(void* user, int32_t n, const void* const* primals,
                                  const void* const* adjoints_in, void* const* adjoints_out);

/** Called once with ``user`` when the last reference to the host node drops (see ``eg_host_node_free``), on the thread that drops
 *  it, or at ``eg_ctx_free`` for a node still alive then. It must not call the door on this context. */
typedef void (*eg_host_release_fn)(void* user);

/** How ``eg_host_call`` declares a plane: the access is the hazard the door orders the node by. READ|WRITE is read, then written in place. */
#define EG_ACCESS_READ 1u
#define EG_ACCESS_WRITE 2u

/** INPUT to ``eg_host_node_new``. */
struct eg_host_node_desc {
    uint32_t struct_size;
    uint32_t flags;               /**< none defined: 0 */
    const char* name;             /**< shown by ``eg_graph_describe`` and in refusals and failures; copied; required, not empty */
    eg_host_fn fn;                /**< required */
    eg_host_vjp_fn vjp;           /**< NULL: none. A graph ended with EG_RECORD_DIFF and reversed with a gradient flowing through the node is refused by name */
    void* user;                   /**< handed back to every callback */
    eg_host_release_fn release;   /**< NULL: none */
};

/**
 * Make a host node: one opaque callback the door runs where a launch would run, for work the door cannot express as a unit (a
 * caller's own function, a custom reverse step, a node kind of the caller's). The door never asks the function about targets, shapes
 * or scheduling: what it reads and writes is what ``eg_host_call`` declares, and ordering, hazard edges, forks, waits, the
 * description and the reverse's copies all follow from those declarations.
 * A host node is reference-counted like a plan. ``eg_host_node_free`` drops the caller's reference; a recorded graph holds its own,
 * so a free while a graph holds the node is deferred, and ``release(user)`` runs exactly once, when the last reference drops.
 * Status: OK; BAD_REQUEST (``desc`` or ``out`` NULL, ``name`` NULL or empty, ``fn`` NULL, an unknown flag).
 */
eg_status eg_host_node_new(eg_ctx* ctx, const struct eg_host_node_desc* desc, eg_host_node** out);

/** Drop the caller's reference to a host node; a recorded graph that runs it keeps it. NULL-safe. */
eg_status eg_host_node_free(eg_ctx* ctx, eg_host_node* node);

/**
 * Run the host node now, or record it while recording, exactly as ``eg_launch`` does for a plan. ``planes`` and ``access`` hold
 * ``n_planes`` entries; each plane is declared EG_ACCESS_READ, EG_ACCESS_WRITE or both, and no plane is declared twice.
 * Not recording: on a device context the door first synchronizes the context's stream, so every device plane is ready; the callback
 * then runs on this thread, and may read and write device planes in place, also with its own library enqueued on
 * ``eg_ctx_info.stream``; the door's next work on that stream is ordered after what the callback enqueued there. Recording on a
 * host context: the node is kept wherever a launch may be recorded (an arm, a fork branch, a WHILE body), holds a reference to the
 * node and the planes, and is replayed in recording order by ``eg_graph_launch``; ``eg_graph_describe`` reports it as
 * EG_GNODE_HOST with its name and the declared planes. Recording on a device context: the host node is a segment boundary of the
 * graph, like ``eg_run`` of a DEVICE runner: the capture ends before it, ``eg_graph_launch`` synchronizes the stream, calls the
 * function on its own thread and starts the next capture after it. Inside a conditional arm, a fork branch or a WHILE of a device
 * recording it is refused NOT_WRITTEN by name, the recording left open.
 * Reverse: with ``vjp`` set, a host node in an EG_RECORD_DIFF graph is a leaf of the reverse (see ``eg_host_vjp_fn``). A host node
 * with no vjp is refused by ``eg_graph_reverse``, BAD_REQUEST naming the node, when a gradient flows through it, and skipped when
 * no gradient reaches the planes it writes. A host node inside a WHILE body of an EG_RECORD_DIFF recording that writes planes the
 * tape tracks is refused NOT_WRITTEN at ``eg_record_end``. On a device context an EG_RECORD_DIFF graph keeps the node as a segment
 * of the reverse's stream too: the door synchronizes it before the vjp.
 * Failure: a nonzero return of the function stops the call: the door function that ran it (``eg_host_call``, ``eg_graph_launch``,
 * ``eg_graph_reverse``) returns FAULT "host node '<name>' failed (<code>)"; a replay stops there, and the planes written so far stay
 * written. The caller keeps its own exception.
 * Status: OK; BAD_REQUEST (a NULL or foreign handle, ``n_planes`` below 0 or planes or access missing, an access other than 1, 2 or 3,
 * a plane declared twice, a host-only context given a device plane); NOT_WRITTEN (the device refusals above); FAULT (the function failed).
 */
eg_status eg_host_call(eg_ctx* ctx, eg_host_node* node, int32_t n_planes, eg_plane* const* planes, const uint32_t* access);

/**
 * Start recording: until ``eg_record_end``, ``eg_launch``, ``eg_run``,
 * ``eg_reduce``, ``eg_copy``, ``eg_fft_run``, ``eg_gemm_run``, ``eg_host_call`` and the conditional calls are recorded, not
 * run. Host: an ordered list replayed in C. Device: a capture on the
 * context's stream (so device work the caller issues on that stream between
 * begin and end is captured too).
 *
 * On a device context, ``eg_launch`` of a host plan, ``eg_run`` of a host runner, ``eg_reduce`` and ``eg_copy`` of host
 * planes are recorded as HOST nodes of the capture: the driver's host-function launch calls one function inside this
 * library, on the context's host team, in stream order at each replay. No caller code runs, and the step makes no CUDA
 * call. The graph holds a reference to every plan, runner and plane a host node runs until ``eg_graph_free`` (which waits
 * for the context's stream to be idle first) or ``eg_ctx_free``. A host node cannot return an error to a caller: the first
 * failure is kept on the graph, and the next ``eg_graph_launch`` of it or ``eg_sync`` of the context returns FAULT naming
 * the step, and clears it. ``eg_reverse`` is not recorded on a device context (NOT_WRITTEN).
 * A recorded ``eg_run`` replays as a restart: at every ``eg_graph_launch``, plain or EG_RECORD_DIFF, ``eg_runner_reset`` and then the
 * run, so each launch runs the batch from its start and, for an EG_RUN_RECORD runner, keeps its own record.
 * The ``eg_run`` of a DEVICE runner on a device context is not a host node: the capture ends there, and ``eg_graph_launch`` replays the graph as
 * its captures and those runs in the order recorded, each run eagerly on the context's stream (it reads back and synchronizes, which a captured
 * graph cannot). A graph with no such run is one capture. The description and the rebinding of a plan's arguments cover every capture.
 * A run inside a conditional arm, a fork branch or a WHILE is refused NOT_WRITTEN, the recording left open: record it outside them.
 * One rule for every refusal between ``eg_record_begin`` and ``eg_record_end``: a refused recording call changes nothing, the recording is as it was
 * before the call (a race refused at ``eg_record_fork_end`` leaves the fork open). The caller may go on, or end the recording with ``eg_record_abort``.
 * Replaces: ``graph.Graph.build`` / ``pipeline.GraphPipeline`` and the
 * node-by-node host arms. Status: OK; BAD_REQUEST (already recording).
 */
eg_status eg_record_begin(eg_ctx* ctx);

/** Discard the open recording, from any state (an open fork or conditional, an empty branch, a half-recorded run): every plan, runner,
 *  transform plan and plane it holds is released, and on a device context its capture is ended and destroyed with its pending driver
 *  error cleared, so a launch, a bind or a new ``eg_record_begin`` that follows works. No graph results. Status: OK when it discarded a
 *  recording; BAD_REQUEST "eg_record_abort: no open recording" otherwise. */
eg_status eg_record_abort(eg_ctx* ctx);

/** Open a conditional on ``desc->cell``; its first arm is being recorded. A fork may open inside an arm and a
 *  conditional inside a fork branch; a fork opened inside an arm must be closed before the arm or the conditional ends.
 *  Status: OK; BAD_REQUEST (not recording; a cell that is not a one-element int32 or uint32 plane; EG_COND_NEGATE on SWITCH;
 *  ``max_iters`` below 0 or on a kind other than WHILE); NOT_WRITTEN (IF_ELSE / SWITCH without the capability bit). */
eg_status eg_record_cond_begin(eg_ctx* ctx, const struct eg_cond_desc* desc);

/** Close the current arm and start the next: IF and WHILE have one arm, so a
 *  second is BAD_REQUEST ("all are open"); IF_ELSE takes its second arm here and
 *  a third is refused; SWITCH moves to the next case and is refused past its
 *  ``n_arms``-th. With no conditional open: BAD_REQUEST. Status: OK; BAD_REQUEST. */
eg_status eg_record_cond_arm(eg_ctx* ctx);

/** Close the innermost conditional. Status: OK; BAD_REQUEST (also a fork still open inside it). */
eg_status eg_record_cond_end(eg_ctx* ctx);

/**
 * Open a fork and its branch 0. The nodes recorded until ``eg_record_fork_branch`` or ``eg_record_fork_end`` form that
 * branch. Forks nest; a fork inside a conditional arm and a conditional inside a branch are legal.
 * Status: OK; BAD_REQUEST (not recording).
 */
eg_status eg_record_fork_begin(eg_ctx* ctx);

/** Close the current branch and open the next. Status: OK; BAD_REQUEST (no fork open, or a conditional still open inside the
 *  branch; an empty branch). */
eg_status eg_record_fork_branch(eg_ctx* ctx);

/**
 * Close the last branch and join: the next node waits for every branch. Before joining, the door checks the hazards between
 * the branches: two nodes in different branches of this fork that touch overlapping memory, at least one of them writing,
 * with no ``eg_record_wait`` between them, are a race and refused BAD_REQUEST naming both node ordinals (0-based, in
 * recording order, a conditional's own test counting as a node reading its cell and baseline), both slots and the hazard (RAW, WAR or
 * WAW). The hazards come from each node's slots (a unit slot's role: read, write or both) and from the memory of the planes
 * (eagle/graph/Edges.h). A refused race leaves the fork open. Replay: the branches may run at the same time (a device graph: parallel branches; the host: in
 * recording order, one legal schedule); the values equal the run in recording order. Status: OK; BAD_REQUEST (no fork open,
 * a conditional still open inside the branch, an empty branch, a race).
 */
eg_status eg_record_fork_end(eg_ctx* ctx);

/**
 * Inside a fork branch: the nodes recorded after this call run after the nodes of the earlier sibling branches that write
 * ``planes``, which removes the hazards between those nodes. Status: OK; BAD_REQUEST (outside a fork branch; a plane no
 * earlier sibling branch writes; a plane this branch has written; a NULL or foreign plane; ``n_planes`` below 1).
 */
eg_status eg_record_wait(eg_ctx* ctx, int32_t n_planes, eg_plane* const* planes);

#define EG_RECORD_DIFF (1u << 0) /**< plan the recording for ``eg_graph_reverse`` (launches, copies, sum reduces, transforms and products) */

/** One recorded launch a gradient flows through, and the vjp unit that reverses it. ``launch`` is the 0-based ordinal of the
 *  ``eg_launch`` inside the recording (a plan launched twice is two ordinals). The vjp is a unit of the launch's library whose
 *  derived record (``eg_unit_derived``, ``of_unit`` the launch's unit) links each of its slots to a slot of the launch:
 *  (PRIMAL_SLOT, VALUE) the primal value, (PRIMAL_SLOT, ADJOINT_IN) the cotangent of a written plane, (PRIMAL_SLOT, GRAD) the
 *  gradient of a read (``reduced`` 1: a one-cell shared parameter plane, summed over samples). The links decide; no other link
 *  (a tape row, a step plane, a time start copy, ADJOINT_OUT) and no by-value uniform gradient is written. ``vjp`` is a unit of
 *  the SAME library as the launch: the derived record names its primal by index in its own library's table, so a vjp of another
 *  library has no identity to check and is refused. Put a kernel and its vjp in one library bundle. */
struct eg_node_vjp {
    int32_t launch;
    eg_unit* vjp;
};

/** INPUT to ``eg_record_end``. */
struct eg_record_end_desc {
    uint32_t struct_size;
    uint32_t flags;                    /**< EG_RECORD_DIFF, or 0 */
    int32_t n_vjps;                    /**< EG_RECORD_DIFF: one per recorded launch a gradient flows through */
    const struct eg_node_vjp* vjps;
};

/** End recording; ``*out`` replays it. ``desc`` NULL: a plain graph. With EG_RECORD_DIFF every recorded call must be an
 *  ``eg_launch`` on a plan with no split, SEGMENTED or companion unit, an ``eg_copy``, ``eg_reduce``, ``eg_fft_run`` or
 *  ``eg_gemm_run``, an ``eg_run`` of an EG_RUN_RECORD runner (refine or not), or a fork, an IF, IF_ELSE or SWITCH around them,
 *  or a WHILE around the others; these are the leaves. The reverse runs over the leaves
 *  in reverse recording order; fork, branch, join and wait are structure only. What each kind reverses to (the door makes these
 *  transposes at this call, with no ``eg_node_vjp``, on the target of the graph, and frees them with it): a copy copies the adjoint
 *  of its target back into the adjoint of its source (a source that is not a float plane has no adjoint); a 1-D sum reduce gives
 *  every sample the adjoint of the sum; a transform of any kind, norm and direction applies its adjoint transform (the Hermitian
 *  weights of a real kind and the norm swap included) to the adjoint of its output; a product gives A the adjoint of C times op(B)
 *  transposed, B op(A) transposed times the adjoint of C, and, when it accumulates, passes the adjoint of C on to C's earlier value
 *  (the three products run as the forward does: the reference loop on the host, cuBLAS on a device). A run is reversed by ``eg_reverse``
 *  of its runner, seeded with the adjoints of the state planes the run overwrote and giving gradients for every plane it reads; the
 *  adjoint of a state plane is then the run's gradient of that state (the run replaced it), those of the other planes it reads gain
 *  the run's gradients, and a plane with no link gets none. The runner's record is reversed once per ``eg_graph_launch``; its own
 *  memory passes are in its ``eg_runner_figures``. A plane a run only reads must not be overwritten by a later leaf (NOT_WRITTEN by name).
 *  None of these has a scatter-add.
 *  A reduce whose op is not sum is refused BAD_REQUEST with the text "has no gradient eagle can pull back (a tie has no single
 *  answer); only op='sum' reverses"; a 2-D or companion reduce NOT_WRITTEN with "eagle does not pull back yet; only a 1-D op='sum'
 *  reduce reverses"; a run of a runner made without EG_RUN_RECORD NOT_WRITTEN with "the run of runner k in a DIFF recording needs EG_RUN_RECORD to
 *  be reversed", the same runner run twice BAD_REQUEST ending "record the second run with a second runner", a run inside a WHILE NOT_WRITTEN
 *  with "a phase count must be static: unroll the phases into the graph". The text after the
 *  call's name is the same on every target, for the caller to carry unchanged. A device recording refuses a call that runs on the
 *  host (a host plan, the copy or reduce of host planes, the run of a host runner) NOT_WRITTEN by name. Each replay saves what every conditional's test saw
 *  (``cell[cell_index]`` and the baseline element) and a leaf under an arm takes part in the reverse only if that arm ran, decided
 *  from the saved values with NEGATE applied, never from the live cells. A WHILE needs ``max_iters`` of 1 or more (0 is refused:
 *  the cap bounds the tape); the tape plans its body unrolled ``max_iters`` times, each copy running only if its iteration did.
 *  The planes the later leaves overwrite are copied at each ``eg_graph_launch`` just before the overwriting leaf, so the reverse reads
 *  what each leaf read; that memory is allocated here (FAULT by name when it cannot be). A WHILE keeps the planes its body writes
 *  only at the start of every ``w``-th iteration, ``w = min(max_iters, clamp(ceil(sqrt(max_iters)), 16, 1024))``, and the reverse
 *  rebuilds each window's states on private copies by re-launching the body's plans, newest window first: the tape holds
 *  ``ceil(max_iters / w) + w`` states of those planes instead of ``max_iters``, and the gradient is the unrolled plan's. One
 *  ``eg_node_vjp`` per ``eg_launch`` ordinal serves every unrolled copy of it. Status: OK; BAD_REQUEST (not recording, a conditional or fork open,
 *  a vjp whose links the launch cannot satisfy, two names of one plane in a node, overlapping planes); NOT_WRITTEN (a
 *  node or unit the reverse does not cover); FAULT (capture failed; a snapshot buffer not allocated). */
eg_status eg_record_end(eg_ctx* ctx, const struct eg_record_end_desc* desc, eg_graph** out);

/** Replay a recorded graph once. Calls today: seam ``launcher_launch`` on a
 *  device. A plan rebound (``eg_bind``, ``eg_clear``) outside a recording is read at its new binding by the next replay of a
 *  plain graph, host and device, with no re-record; a graph ended with EG_RECORD_DIFF refuses it (its tape holds the planes
 *  it was recorded with). Status: OK; BAD_REQUEST (a rebound plan of an EG_RECORD_DIFF graph); FAULT (also a failure a host
 *  node of an earlier replay kept, see ``eg_record_begin``). */
eg_status eg_graph_launch(eg_ctx* ctx, eg_graph* graph);

/** ``eg_graph_reverse_desc.flags``: refuse a gradient that is not bit-reproducible (see ``bitwise``) instead of reporting it. */
#define EG_REVERSE_STRICT (1u << 0)

/** INPUT to ``eg_graph_reverse``. Planes are matched by handle, not by slot: graph planes are shared across nodes. */
struct eg_graph_reverse_desc {
    uint32_t struct_size;
    uint32_t flags;                    /**< EG_REVERSE_STRICT, or 0 */
    int32_t n_seeds;
    eg_plane* const* outputs;          /**< planes written by the graph */
    eg_plane* const* seeds;            /**< dJ/d(outputs[i]), one value per element; an output with no seed has cotangent zero */
    int32_t n_grads;
    eg_plane* const* inputs;           /**< planes the graph reads before anything writes them */
    eg_plane* const* grads;            /**< dJ/d(inputs[i]) is written into grads[i]; an input not listed gets nothing */
    /* appended: a struct_size that ends before a member leaves it unwritten */
    uint8_t* bitwise;                  /**< OUT, ``n_grads`` entries, or NULL: 1 when no cross-sample scatter-add feeds grads[i], else 0 */
    int32_t* passes;                   /**< OUT, or NULL: the memory passes the reverse ran in; 1 (a graph has no memory budget) */
};

/** Reverse the graph's LAST ``eg_graph_launch``, eagerly on the host. The sum over samples of a shared parameter's gradient (a
 *  reduced GRAD link) is always the compensated pair in sample order, on the host and on a device: the same bits on every run, every
 *  thread count and both targets, so there is no policy member and such a gradient is bitwise. A gradient is not bitwise when a
 *  cross-sample scatter-add feeds it: a vjp slot of role ACCUM_OUT taints the adjoint it writes, and the taint passes through every
 *  earlier leaf that reads a tainted adjoint, through arms and WHILE copies; ``bitwise[i]`` is then 0, and under EG_REVERSE_STRICT
 *  the call is refused by name (the input and the launch ordinal of the tainting vjp) before any gradient is written. An input
 *  that is not a float plane is refused by name. Leaves run in reverse order (see ``eg_record_end``); a
 *  leaf that did not run, an arm not taken or a WHILE iteration that did not happen, is the identity: what it would write keeps
 *  its adjoint for the earlier writer or the input plane, and it adds nothing to what it only reads. A plane read by several
 *  leaves gets its adjoints summed in leaf order, the seed first. The caller's promise: no plane the record reads live was
 *  changed since that launch, except through the graph's own plans (which the door checks: see below). Status: OK;
 *  BAD_REQUEST (graph not ended with EG_RECORD_DIFF, never launched, a call while recording, a plan of the graph rebound
 *  (eg_bind, eg_clear) or launched outside the graph since its last launch, a launch with a gradient flowing through it and
 *  no vjp, a seed or gradient plane that does not match); NOT_WRITTEN (a device context or graph). */
eg_status eg_graph_reverse(eg_ctx* ctx, eg_graph* graph, const struct eg_graph_reverse_desc* desc);

/** What a node of a graph description is. */
typedef int32_t eg_gnode_kind;
#define EG_GNODE_LAUNCH 0  /**< ``eg_launch`` */
#define EG_GNODE_RUN 1     /**< ``eg_run`` */
#define EG_GNODE_REDUCE 2  /**< ``eg_reduce`` */
#define EG_GNODE_COPY 3    /**< ``eg_copy`` */
#define EG_GNODE_FFT 4     /**< ``eg_fft_run`` */
#define EG_GNODE_GEMM 5    /**< ``eg_gemm_run`` */
#define EG_GNODE_COND 6    /**< a conditional's test: it reads its cell and baseline */
#define EG_GNODE_ARM 7     /**< one arm of a conditional */
#define EG_GNODE_FORK 8    /**< ``eg_record_fork_begin`` */
#define EG_GNODE_BRANCH 9  /**< one branch of a fork */
#define EG_GNODE_JOIN 10   /**< ``eg_record_fork_end``: what the next node waits for */
#define EG_GNODE_WAIT 11   /**< ``eg_record_wait`` */
#define EG_GNODE_HOST 12   /**< ``eg_host_call``: ``unit`` is the host node's name, the planes are the declared ones */

/** What an edge of a graph description is. */
typedef int32_t eg_gedge_kind;
#define EG_GEDGE_RAW 0   /**< the later node reads what the earlier wrote (derived from the planes) */
#define EG_GEDGE_WAR 1   /**< the later node overwrites what the earlier read */
#define EG_GEDGE_WAW 2   /**< both write the plane */
#define EG_GEDGE_WAIT 3  /**< a writer to the ``eg_record_wait`` node that names its plane */
#define EG_GEDGE_JOIN 4  /**< a node of a fork to its join */
#define EG_GEDGE_ORDER 5 /**< recording order the replay keeps without a hazard: a container to its content, a wait to what follows it in its branch, a join to the next node */

/** One node of a graph description. ``planes``, ``access`` and ``slot`` hold ``n_planes`` entries. */
struct eg_graph_node {
    uint32_t struct_size;
    eg_gnode_kind kind;
    int32_t parent;       /**< the enclosing node: an ARM's COND, a BRANCH's FORK, else the ARM or BRANCH the call was recorded in; -1 at top level */
    int32_t level;        /**< the schedule level: nodes of one level may run at the same time */
    int32_t arm;          /**< ARM: its index; COND: its ``eg_cond_kind``; else -1 */
    int32_t n_planes;     /**< the planes the node reads or writes (a plane bound to two slots appears twice) */
    const char* unit;     /**< LAUNCH: the unit's name; RUN: the first plan's unit name; HOST: the host node's name; else NULL */
    eg_plane* const* planes;
    const uint8_t* access;  /**< per plane: 1 read, 2 write, 3 both */
    const int32_t* slot;    /**< per plane: the unit slot of a LAUNCH, else -1 */
    /* appended: a struct_size that ends before a member reads it as 0 */
    uint32_t cond_flags;    /**< COND: the ``eg_cond_desc.flags`` it was opened with (EG_COND_NEGATE) */
    int32_t has_baseline;   /**< COND: 1 when its test compares with a baseline plane (the second plane), else 0 */
    int64_t max_iters;      /**< COND: the WHILE cap, 0 for none */
    int64_t cell_index;     /**< COND: the element of the cell the test reads */
    int32_t n_plans;        /**< RUN: the plans of the runner (``unit`` is the first plan's unit name) */
    int32_t launch;         /**< LAUNCH: its ordinal among the ``eg_launch`` calls of the recording, the number ``eg_node_vjp.launch`` takes; else -1 */
};

/** One edge of a graph description: ``from`` must finish before ``to`` starts. ``plane`` is NULL for JOIN and ORDER. */
struct eg_graph_edge {
    int32_t from, to;
    eg_gedge_kind kind;
    int32_t from_slot, to_slot;  /**< the unit slots the hazard goes through, -1 for a node that is not a launch */
    eg_plane* plane;
};

/** OUTPUT of ``eg_graph_describe``. */
struct eg_graph_description {
    uint32_t struct_size;
    uint32_t flags;                        /**< EG_RECORD_DIFF when the graph was ended so, else 0 */
    int32_t n_nodes;
    const struct eg_graph_node* nodes;     /**< recording order: a node's parent and every edge point to lower indices */
    int32_t n_edges;
    const struct eg_graph_edge* edges;
    int32_t n_levels;                      /**< 0 for an empty graph, else the highest ``level`` plus one */
    uint32_t node_size;                    /**< the door's ``sizeof(struct eg_graph_node)``: step ``nodes`` by this, not by the caller's own size */
    uint32_t edge_size;                    /**< the door's ``sizeof(struct eg_graph_edge)``: step ``edges`` by this */
};

/**
 * Describe a recorded graph for drawing: its nodes (a fork, its branches and join, a conditional and its arms, every recorded
 * call), the hazard, wait, join and order edges, and the schedule level of each node (the longest path from a root). The same
 * description is kept for a host and a device graph: the door builds it at record time, it is not read back from a driver.
 * Read-only and callback-free; the arrays are the door's and stay valid until ``eg_graph_free`` of the graph or the next
 * ``eg_graph_describe`` of it, so a caller copies what it keeps. Node indices are the ordinals a race refusal names. A node or edge struct grows by appended members, so step the arrays by ``node_size`` and ``edge_size``. A host graph replays in recording order, which respects every
 * level. Status: OK; BAD_REQUEST (a NULL, freed or foreign graph; ``out`` NULL or with a ``struct_size`` below the door's ``sizeof(struct eg_graph_description)``).
 */
eg_status eg_graph_describe(eg_ctx* ctx, const eg_graph* graph, struct eg_graph_description* out);

/** Free a graph. NULL-safe. */
eg_status eg_graph_free(eg_ctx* ctx, eg_graph* graph);

/* ======================================================================== */
/* DEVICE GROUP (declarations; behaviour mapped onto eagle-backend/1)       */
/* ======================================================================== */

/*
 * Device-group calls without a device. In a door built WITHOUT the device
 * group every call below returns EG_NOT_WRITTEN ("the device group is not
 * built"), except ``eg_sync``, which is EG_OK (nothing to wait for), and
 * ``eg_device_count``, which is EG_OK with 0. In a door built WITH it, on a
 * host context (made without EG_CTX_DEVICE): ``eg_sync`` is EG_OK,
 * ``eg_stream_use`` is EG_BAD_REQUEST ("host-only context": it has no
 * stream), ``eg_device_count`` and ``eg_device_props`` answer.
 */

/** Number of devices the driver sees (seam ``device_count``): 0, with EG_OK, when the driver library cannot be
 *  opened or ``cuInit`` reports no device, and in a door built without the device group. Any context answers.
 *  Status: OK; BAD_REQUEST (``count`` NULL); FAULT (a driver that failed after it opened). */
eg_status eg_device_count(eg_ctx* ctx, int32_t* count);

/** Properties of device ``device`` (an ordinal below eg_device_count; seam ``device_props``), read from the driver; any
 *  context answers for any device. Status: OK; BAD_REQUEST (``out`` NULL or with a ``struct_size`` below the door's
 *  ``sizeof(struct eg_device_props)``, an ordinal the driver does not see, no driver); NOT_WRITTEN (a door built
 *  without the device group); FAULT (a driver that failed after it opened). */
eg_status eg_device_props(eg_ctx* ctx, int32_t device, struct eg_device_props* out);

/** Make ``stream`` (a raw stream of the context's device, 0: eagle's own) the
 *  context's stream scope (seam ``stream_create`` over an external stream).
 *  Device memory eagle makes while a stream is in scope is released on that
 *  stream, after the work queued there: the caller keeps a stream it passed
 *  alive until every plane and plan made while it was in scope is freed. */
eg_status eg_stream_use(eg_ctx* ctx, uint64_t stream);

/** Block until the context's stream is idle (seam ``stream_synchronize``). Status: OK; FAULT (also the failure a host
 *  node of a device graph kept while the stream ran, see ``eg_record_begin``). */
eg_status eg_sync(eg_ctx* ctx);

#ifdef __cplusplus
}
#endif

#endif /* EG_DOOR_H */
