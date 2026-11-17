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
 *  - No callbacks: eagle never calls back into the caller. A binding may
 *    release its interpreter lock around every call.
 *  - No global mutable state: everything lives in a context. The one process
 *    resource is the host worker pool, shared by all contexts; each context
 *    states its own thread count, which is passed per launch. A context and
 *    every handle made from it are used by ONE thread at a time; a second
 *    thread entering the same context while a call runs is refused
 *    (EG_BAD_REQUEST), never raced. That refusal carries NO reason: the
 *    stored reason belongs to the thread inside and is left as it was, so
 *    ``eg_error`` reports none for it. Different contexts run concurrently.
 *  - Fork: a context belongs to the process that made it. In a forked child
 *    every call on it is refused except ``eg_error`` and ``eg_ctx_free``
 *    (which then frees host memory only: no device call, no thread join).
 *  - Handles belong to the context that made them; a handle of another
 *    context is refused. Every ``*_free`` is NULL-safe and returns EG_OK.
 *  - Arrays cross by DLPack only (``DLManagedTensorVersioned``, DLPack >= 1.0).
 *  - ``eg_ctx_info`` and ``eg_device_props`` are each both a struct tag and a
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
#define EG_STEPS_REVERSE_AUTO 2 /**< chosen from the unit meta's measured table */

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
    uint32_t flags;                     /**< none defined yet */
    const char* name;
    uint8_t digest[EG_DIGEST_BYTES];
    int32_t pattern;                    /**< enum of the typed-meta design */
    int32_t scalar_type;                /**< 0 float64, 1 float32, 2 banded (refused on a host structure as not written) */
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
};

/** INPUT to ``eg_plan_new``. */
struct eg_plan_desc {
    uint32_t struct_size;
    uint32_t flags;               /**< none defined: must be 0 */
    eg_unit* unit;                /**< the source unit; its stages ride in its meta */
    eg_structure structure;
    eg_reduce_policy policy;      /**< order of the unit's sums over samples */
    int64_t n_samples;
    int64_t n_partitions;         /**< 0: one partition, the whole range */
    const int64_t* part_base;     /**< n_partitions entries, else NULL */
    const int64_t* part_count;    /**< n_partitions entries, else NULL */
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
    const void* const* uniform_values;  /**< each read at the slot's declared dtype */
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
    int64_t max_steps;
    int64_t every;                      /**< compaction cadence in steps; 0: the runner's choice */
    int64_t window;                     /**< record window; 0: clamp(ceil(sqrt(max_steps)), 16, 1024); -1: every step */
    uint64_t memory_budget;             /**< bytes for the record; 0: half the free memory */
    double reorder_theta;               /**< read only with EG_RUN_REORDER */
    eg_steps_reverse steps_reverse;
    eg_events events;
    eg_no_event on_no_event;
    int32_t reserved0;
    eg_graph* step_graph;               /**< device only: recorded device work run once per step inside the
                                             captured loop, or NULL; a host runner refuses a non-NULL one */
};

/** Report flags. */
#define EG_REPORT_DONE (1u << 0)        /**< finished == n */
#define EG_REPORT_EXACT_STEPS (1u << 1) /**< ``steps`` is exact */
#define EG_REPORT_PROBE (1u << 2)       /**< a run taken to time the entry not in use */
#define EG_REPORT_TIMED (1u << 3)       /**< the run carried the timing pair */
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

/** INPUT to ``eg_record_cond_begin``. */
struct eg_cond_desc {
    uint32_t struct_size;
    uint32_t flags;          /**< none defined: must be 0 */
    eg_cond_kind kind;
    int32_t n_arms;          /**< SWITCH only: arm count */
    eg_plane* cell;          /**< the int32 cell a kernel sets: truth, or the arm index */
};

/** OUTPUT of ``eg_device_props`` (device group). */
struct eg_device_props {
    uint32_t struct_size;
    int32_t cc_major, cc_minor;
    int32_t sm_count, warp_size;
    int32_t regs_per_sm, regs_per_block;
    int32_t reserved0;
    int64_t shared_mem_per_block, shared_mem_per_sm;
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
 * (EG_CTX_DEVICE, no device backend built); FAULT (backend load).
 */
eg_status eg_ctx_new(const struct eg_ctx_settings* settings, eg_ctx** out);

/** Free a context and every handle still made from it. NULL-safe. In a
 *  forked child frees host memory only. Status: OK. */
eg_status eg_ctx_free(eg_ctx* ctx);

/**
 * The reason of the last non-OK status on ``ctx``, UTF-8, two-call protocol:
 * ``*len`` gets the byte count without the terminator; with ``cap > len``
 * the text and a terminator are written. The reason stays until the next
 * non-OK call on ``ctx``; a refusal of ``eg_error`` itself never overwrites
 * it, and a busy-context refusal writes none. With no reason stored,
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

/* ---- plans --------------------------------------------------------------- */

/**
 * A plan: the unit under one structure over fixed partitions. Eagle makes the
 * planes the caller never passes (companions, cell scratch, team scratch,
 * the finished counter) when the plan is bound.
 * Replaces: ``plan.plan``, ``plan.Plan``, ``_split`` stage plans.
 * Calls today: ``eagle::exec::Partition`` (Partition.h),
 * ``exec::check_placement``. Status: OK; BAD_REQUEST (partitions that
 * overlap or leave the range); NOT_WRITTEN (DEVICE without the device group).
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
 * Launch the bound plan once over each of its partitions. ``*tiles`` gets the
 * host tiles run (0 for DEVICE).
 * Replaces: ``BoundPlan.launch`` / ``Plan.run`` / ``exec.HostTeam.run``.
 * Calls today: ``eagle::exec::HostTeam::run`` (HostTeam.h) via
 * ``_core.run_host`` (eagle_core.cpp); ``run_serial`` for HOST_SERIAL.
 * Status: OK; BAD_REQUEST (a slot unbound); FAULT.
 */
eg_status eg_launch(eg_ctx* ctx, eg_plan* plan, uint64_t* tiles);

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
 * Status: OK; BAD_REQUEST (a plan without a finish, every/reorder on a plan
 * without an active set, record without its derived units); NOT_WRITTEN.
 */
eg_status eg_runner_new(eg_ctx* ctx, const struct eg_runner_desc* desc, eg_runner** out);

/** Free a runner and its record. NULL-safe. */
eg_status eg_runner_free(eg_ctx* ctx, eg_runner* runner);

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
 * Copy ``src`` into ``dst`` (same shape and dtype; any device pair the
 * context reaches). Recordable. Replaces: ``_transfer`` copy nodes.
 * Status: OK; BAD_REQUEST.
 */
eg_status eg_copy(eg_ctx* ctx, eg_plane* src, eg_plane* dst);

/**
 * Start recording: until ``eg_record_end``, ``eg_launch``, ``eg_run``,
 * ``eg_reduce``, ``eg_copy`` and the conditional calls are recorded, not
 * run. Host: an ordered list replayed in C. Device: a capture on the
 * context's stream (so device work the caller issues on that stream between
 * begin and end is captured too).
 * Replaces: ``graph.Graph.build`` / ``pipeline.GraphPipeline`` and the
 * node-by-node host arms. Status: OK; BAD_REQUEST (already recording).
 */
eg_status eg_record_begin(eg_ctx* ctx);

/** Open a conditional on ``desc->cell``; its first arm is being recorded.
 *  Calls today: seam ``conditional_*``. Status: OK; BAD_REQUEST;
 *  NOT_WRITTEN (IF_ELSE / SWITCH without the capability bit). */
eg_status eg_record_cond_begin(eg_ctx* ctx, const struct eg_cond_desc* desc);

/** Close the current arm and start the next. Status: OK; BAD_REQUEST. */
eg_status eg_record_cond_arm(eg_ctx* ctx);

/** Close the innermost conditional. Status: OK; BAD_REQUEST. */
eg_status eg_record_cond_end(eg_ctx* ctx);

/** End recording; ``*out`` replays it. Status: OK; BAD_REQUEST (not
 *  recording, a conditional open); FAULT (capture failed). */
eg_status eg_record_end(eg_ctx* ctx, eg_graph** out);

/** Replay a recorded graph once. Calls today: seam ``launcher_launch`` on a
 *  device. Status: OK; FAULT. */
eg_status eg_graph_launch(eg_ctx* ctx, eg_graph* graph);

/** Free a graph. NULL-safe. */
eg_status eg_graph_free(eg_ctx* ctx, eg_graph* graph);

/* ======================================================================== */
/* DEVICE GROUP (declarations; behaviour mapped onto eagle-backend/1)       */
/* ======================================================================== */

/** Number of devices the backend sees (seam ``device_count``). */
eg_status eg_device_count(eg_ctx* ctx, int32_t* count);

/** Properties of a device (seam ``device_props``). */
eg_status eg_device_props(eg_ctx* ctx, int32_t device, struct eg_device_props* out);

/** Make ``stream`` (a raw stream of the context's device, 0: eagle's own) the
 *  context's stream scope (seam ``stream_create`` over an external stream). */
eg_status eg_stream_use(eg_ctx* ctx, uint64_t stream);

/** Block until the context's stream is idle (seam ``stream_synchronize``). */
eg_status eg_sync(eg_ctx* ctx);

/** While recording, add an external device graph as one child node; eagle
 *  takes no ownership (seam ``capture_add_child``). */
eg_status eg_record_child(eg_ctx* ctx, uint64_t graph);

#ifdef __cplusplus
}
#endif

#endif /* EG_DOOR_H */
