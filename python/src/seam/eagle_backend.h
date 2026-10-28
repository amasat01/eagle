/* Copyright 2026 Alessandro Masat
 * SPDX-License-Identifier: Apache-2.0
 */

/**
 * @file eagle_backend.h
 * @brief The ``eagle-backend/1`` C seam between ``eagle._core`` (g++, no device
 *        runtime) and a device backend plugin ``libeagle_<kind>.<ext>``.
 *
 * Pure C, POD only, nothing C++ crosses. Every entry returns an
 * ``eagle_backend_status`` (unless stated); a nonzero status leaves its message
 * in the backend's thread-local ``eagle_backend_last_error()``.
 *
 * KIND. A kind is an execution model (``cuda``; later ``hip``, ``level_zero``,
 * ``vulkan``, ``metal``, ``opencl``, ``webgpu``), never a toolkit generation: a
 * CUDA 13 build is kind ``cuda`` with other ``build_info`` facts. The plugin of
 * kind K is the file ``libeagle_<K>.so`` (``.dylib`` macOS, ``.dll`` Windows), and
 * ``$EAGLE_BACKEND_<K>`` names an exact file to load instead (no fall-through on
 * failure). A future major 2 is ``libeagle_<K>.2.<ext>``.
 *
 * VERSIONING. ``eagle_backend_version()`` is ``(major << 16) | minor``. The core
 * accepts exactly major 1; the minor is informational. Within major 1 the seam is
 * additive only: new symbols are bound one by one (their presence is the
 * capability), struct fields are appended, status values and capability bits are
 * appended, flag bits are assigned once.
 *
 * MANDATORY CORE. Every backend exports the seven meta symbols below.
 *
 * CAPABILITY GROUPS. ``eagle_backend_capabilities`` returns one bit per group.
 * A set bit promises the group's 1.0 symbols (a set bit with one of them missing
 * is a load refusal naming the symbol); a clear bit makes the group's calls raise
 * ``eagle.BackendUnavailable("... does not provide <group>")``.
 *
 * | bit | group       | 1.0 symbols (``eagle_backend_`` prefix omitted)        |
 * |-----|-------------|-------------------------------------------------------|
 * | 0   | stream      | stream_create, stream_ptr, stream_synchronize, stream_release |
 * | 1   | capturer    | capturer_create, capturer_begin, capturer_end, capturer_release |
 * | 2   | captured    | captured_is_valid, captured_release                    |
 * | 3   | fork        | fork_create, fork_fork, fork_join, fork_branch, fork_origin, fork_forked, fork_size, fork_release |
 * | 4   | conditional | conditional_create, conditional_begin, conditional_body_stream, conditional_is_loop, conditional_end, conditional_release |
 * | 5   | attribution | capture_snapshot_nodes, is_node_toggleable (+ capture_add_child, bound per symbol) |
 * | 6   | graph       | graph_create, graph_from_captured, graph_stream, graph_add_node, graph_launcher, graph_last_node, graph_release |
 * | 7   | launcher    | launcher_launch, launcher_synchronize, launcher_stream, launcher_set_logical_size, launcher_kernel_node_count, launcher_set_node_enabled, launcher_release |
 * | 8   | (reserved)  | the composer group, when it is promoted from ``x_``     |
 * | 9   | exec        | run_device                                            |
 * | 10  | device      | device_props                                          |
 * | 11  | interop     | fence, event_pool_created (+ memset_device, memcpy_device, memcpy, host_alloc, host_free, host_func, bound per symbol) |
 * |     | (minor 7)   | device_count, set_device, get_device, launch_status, device_synchronize, stream_is_capturing, module_load, module_load_jit, module_function, module_release, link_ptx, gemm, gemm_release, function_attribute, launch_raw, cell_op, memcpy_strided, mem_alloc, mem_release, mem_drain, pointer_device, stream_sync, event_create, event_create_timed, event_elapsed, event_record, event_wait, event_release: bound per symbol, no group bit |
 * | 12  | filtering   | compact_device (+ reorder_device, bound per symbol)   |
 * | 13  | conditional IF/ELSE | none: ``conditional_desc.kind`` IF_ELSE is accepted (driver >= 12.8) |
 * | 14  | conditional SWITCH  | none: ``conditional_desc.kind`` SWITCH is accepted (driver >= 12.8)  |
 * | 15..62 | reserved | new families only (additions to a family are per-symbol) |
 * | 63  | never assigned                                                       |
 *
 * EXPERIMENTAL (``eagle_backend_x_*``): optional one by one, may change in any
 * release, never locked: the composer, ``captured_debug_dot``, the two
 * capture-guard counters and ``restore_device``.
 *
 * STRUCTS. Never cross by value, never in arrays (flat parallel arrays / CSR
 * instead). Each starts with ``uint32_t struct_size``. An INPUT desc: the caller
 * zero-fills and sets its own ``sizeof``; the callee reads only the fields that
 * fit and takes the rest as their zero default. An OUTPUT struct: the caller
 * zero-fills and sets its ``sizeof``; the callee writes ``min(caller, callee)``
 * bytes, never beyond, and writes back that size; the caller treats fields past
 * the returned size as ABSENT. Padding is explicit.
 *
 * FLAGS. A desc's ``flags`` bits are assigned once; a bit the backend does not
 * know is refused with ``EAGLE_BACKEND_UNSUPPORTED``, never ignored.
 *
 * DEVICES. Every creator names its device: ``device >= 0`` is that ordinal;
 * ``-1`` is the device of the stream the desc carries or, for a creator that
 * takes no stream, the device of the calling thread's current DRIVER context
 * (which every runtime in the process shares; device 0 when there is none). A
 * handle is bound to its device for life, whatever is current later.
 *
 * HANDLES AND LIFETIMES. Handles are opaque and single-thread-affine. Every
 * ``*_release`` is NULL-safe and returns nothing. A call documented as CONSUMING
 * a handle takes it on ``EAGLE_BACKEND_OK`` only; on any error the caller still
 * owns it. Raw stream / node / function values cross as ``uint64_t``.
 * ``eagle_backend_last_error()`` is valid until the next seam call on the same
 * thread (copy it at once). A ``const char*`` owned by a handle stays valid until
 * the next call that mutates that handle. Buffers the backend allocates are freed
 * with ``eagle_backend_free``.
 *
 * CALLBACKS run on the calling thread, synchronously, inside the seam call that
 * triggers them; the core owns the GIL state. The backend holds no lock across a
 * callback. A callback returning nonzero makes the call unwind and return
 * ``EAGLE_BACKEND_CALLBACK_FAILED``.
 *
 * STREAM CODES (``fence``): the integers are the DLPack ``__dlpack__(stream=)``
 * values of the given DLPack device type, resolved by the backend (CUDA: 1 legacy
 * default, 2 per-thread default, -1 no ordering, 0 refused, else a raw stream).
 * ``EAGLE_BACKEND_STREAM_NONE`` means none was given (the type's default applies).
 */
#ifndef EAGLE_BACKEND_H
#define EAGLE_BACKEND_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Seam major version: the core accepts exactly this major. */
#define EAGLE_BACKEND_VERSION_MAJOR 1u
/** Seam minor version this header describes (informational). */
#define EAGLE_BACKEND_VERSION_MINOR 7u
/** The packed version a backend built from this header reports. */
#define EAGLE_BACKEND_VERSION ((EAGLE_BACKEND_VERSION_MAJOR << 16) | EAGLE_BACKEND_VERSION_MINOR)

/** Status of every seam call. Values are append-only; a caller maps a value it
 *  does not know to a generic error carrying ``eagle_backend_last_error()``. */
typedef int32_t eagle_backend_status;

#define EAGLE_BACKEND_OK 0
#define EAGLE_BACKEND_ERROR 1
#define EAGLE_BACKEND_INVALID_ARGUMENT 2
#define EAGLE_BACKEND_INDEX 3
#define EAGLE_BACKEND_INTEROP 4
#define EAGLE_BACKEND_UNAVAILABLE 5
#define EAGLE_BACKEND_CALLBACK_FAILED 6
#define EAGLE_BACKEND_BAD_HANDLE 7
/** The backend provides the group but not this option / flag / field. */
#define EAGLE_BACKEND_UNSUPPORTED 8

/** Capability bits (see the table in the file doc). */
#define EAGLE_BACKEND_CAP_STREAM (UINT64_C(1) << 0)
#define EAGLE_BACKEND_CAP_CAPTURER (UINT64_C(1) << 1)
#define EAGLE_BACKEND_CAP_CAPTURED (UINT64_C(1) << 2)
#define EAGLE_BACKEND_CAP_FORK (UINT64_C(1) << 3)
#define EAGLE_BACKEND_CAP_CONDITIONAL (UINT64_C(1) << 4)
#define EAGLE_BACKEND_CAP_ATTRIBUTION (UINT64_C(1) << 5)
#define EAGLE_BACKEND_CAP_GRAPH (UINT64_C(1) << 6)
#define EAGLE_BACKEND_CAP_LAUNCHER (UINT64_C(1) << 7)
/* bit 8 reserved: the composer group on promotion */
#define EAGLE_BACKEND_CAP_EXEC (UINT64_C(1) << 9)
#define EAGLE_BACKEND_CAP_DEVICE (UINT64_C(1) << 10)
#define EAGLE_BACKEND_CAP_INTEROP (UINT64_C(1) << 11)
#define EAGLE_BACKEND_CAP_FILTERING (UINT64_C(1) << 12)
#define EAGLE_BACKEND_CAP_CONDITIONAL_IF_ELSE (UINT64_C(1) << 13)
#define EAGLE_BACKEND_CAP_CONDITIONAL_SWITCH (UINT64_C(1) << 14)

/** ``eagle_backend_conditional_desc.kind`` (0: WHILE with ``counter_ptr``, else IF). */
#define EAGLE_BACKEND_COND_IF 1u
#define EAGLE_BACKEND_COND_IF_ELSE 2u
#define EAGLE_BACKEND_COND_WHILE 3u
#define EAGLE_BACKEND_COND_SWITCH 4u

/** "No stream given" in a ``fence`` stream argument. */
#define EAGLE_BACKEND_STREAM_NONE INT64_MIN

/** ``eagle_backend_stream_desc.flags``: no implicit sync with the legacy default stream. */
#define EAGLE_BACKEND_STREAM_NON_BLOCKING 1u

typedef struct eagle_backend_stream_s* eagle_backend_stream;
typedef struct eagle_backend_capturer_s* eagle_backend_capturer;
typedef struct eagle_backend_captured_s* eagle_backend_captured;
typedef struct eagle_backend_fork_s* eagle_backend_fork;
typedef struct eagle_backend_conditional_s* eagle_backend_conditional;
typedef struct eagle_backend_graph_s* eagle_backend_graph;
typedef struct eagle_backend_launcher_s* eagle_backend_launcher;
typedef struct eagle_backend_composer_s* eagle_backend_composer;

/** A callback that issues work on ``stream``; nonzero = failed. */
typedef int32_t (*eagle_backend_step_fn)(void* ctx, uint64_t stream);
/** A callback without arguments; nonzero = failed. */
typedef int32_t (*eagle_backend_void_fn)(void* ctx);

/* ---- structs (used by tag: a tag may share its name with a function) ------- */

struct eagle_backend_build_info {
    uint32_t struct_size;
    uint32_t abi_version;     /**< the backend's ``eagle_backend_version()`` */
    int32_t cudart_version;   /**< device-runtime version built against (0 if none) */
    int32_t reserved0;
    char eagle_version[32];   /**< the eagle release the backend was built from */
    char archs[128];          /**< device code carried, e.g. "61,90/90" (SASS / PTX) */
    char backend_kind[16];    /**< the kind: "cuda", ... */
};

struct eagle_backend_device_props {
    uint32_t struct_size;
    int32_t device;
    char name[256];
    int32_t cc_major, cc_minor, sm_count, clock_rate_khz, memory_clock_rate_khz, memory_bus_width_bits;
    int32_t regs_per_block, regs_per_sm, warp_size, reserved0;
    int64_t shared_mem_per_block, shared_mem_per_sm;
    double peak_bytes_per_s, peak_flops_sp, peak_flops_dp, fp64_ratio;
    double ridge_flops_per_byte_sp, ridge_flops_per_byte_dp;
    /* Appended in minor 7: device memory in bytes, read with the device current. */
    int64_t total_mem_bytes, free_mem_bytes;
};

struct eagle_backend_stream_desc {
    uint32_t struct_size;
    uint32_t flags;           /**< EAGLE_BACKEND_STREAM_* */
    int32_t device;
    int32_t reserved0;
};

struct eagle_backend_capturer_desc {
    uint32_t struct_size;
    uint32_t flags;
    int32_t device;
    int32_t reserved0;
    uint64_t stream;          /**< the stream to capture */
};

struct eagle_backend_fork_desc {
    uint32_t struct_size;
    uint32_t flags;
    int32_t device;
    int32_t reserved0;
    uint64_t origin;          /**< the stream the branches fork from */
    uint64_t branch_count;
    uint64_t wait_count;      /**< marks ``fork_mark`` may make (minor 5) */
};

struct eagle_backend_conditional_desc {
    uint32_t struct_size;
    uint32_t flags;
    int32_t device;
    uint32_t loop_cap;        /**< WHILE iteration cap (with counter_ptr) */
    uint64_t origin;          /**< the stream the region would be captured on */
    uint64_t count_ptr;       /**< device uint32* */
    uint64_t baseline_ptr;    /**< device uint32*, 0 = none */
    uint64_t counter_ptr;     /**< device uint32[2] for a WHILE loop, 0 = IF */
    /* minor 5 */
    uint32_t kind;            /**< EAGLE_BACKEND_COND_*, 0 = from counter_ptr */
    uint32_t arms;            /**< 0 = the kind's own; SWITCH: the arm count */
    uint32_t negate;          /**< nonzero inverts the guard */
    uint32_t reserved1;
    uint64_t brk_ptr;         /**< WHILE: device uint32 break cell, 0 = none */
    uint64_t limit_ptr;       /**< WHILE: device uint32 trip count, 0 = none */
};

struct eagle_backend_graph_desc {
    uint32_t struct_size;
    uint32_t flags;
    int32_t device;
    int32_t reserved0;
};

/** One kernel launch over one partition (``run_device``). */
struct eagle_backend_launch_desc {
    uint32_t struct_size;
    uint32_t flags;
    int32_t device;
    uint32_t block;              /**< threads per block */
    uint64_t function;           /**< the kind's kernel handle (CUDA: CUfunction) */
    const uint64_t* params;      /**< nparams addresses of the argument storage */
    const uint64_t* param_sizes; /**< nparams byte sizes, or NULL when unknown */
    int64_t nparams;
    int64_t base, count, n_samples; /**< the partition triple */
    uint64_t stream;
};

struct eagle_backend_composer_desc {
    uint32_t struct_size;
    uint32_t flags;
    int32_t device;
    int32_t reserved0;
    const char* mode;         /**< "sequenced" | "enabled" | "rebuild" */
};

/* ---- mandatory core --------------------------------------------------------- */

uint32_t eagle_backend_version(void);
const char* eagle_backend_last_error(void);
eagle_backend_status eagle_backend_build_info(struct eagle_backend_build_info* info);
/** OK when a device is usable, UNAVAILABLE (message in last_error) otherwise. */
eagle_backend_status eagle_backend_probe(void);
void eagle_backend_free(void* p);
eagle_backend_status eagle_backend_capabilities(uint64_t* groups);
/** The DLPack device-type codes this backend serves: up to ``cap`` written to
 *  ``types``, the full number to ``count``. */
eagle_backend_status eagle_backend_device_types(int32_t* types, int32_t cap, int32_t* count);

/* ---- stream (bit 0) ---------------------------------------------------------- */

eagle_backend_status eagle_backend_stream_create(const struct eagle_backend_stream_desc* desc, eagle_backend_stream* out);
eagle_backend_status eagle_backend_stream_ptr(eagle_backend_stream h, uint64_t* stream);
eagle_backend_status eagle_backend_stream_synchronize(eagle_backend_stream h);
void eagle_backend_stream_release(eagle_backend_stream h);

/* ---- capturer (bit 1) -------------------------------------------------------- */

eagle_backend_status eagle_backend_capturer_create(const struct eagle_backend_capturer_desc* desc, eagle_backend_capturer* out);
eagle_backend_status eagle_backend_capturer_begin(eagle_backend_capturer h);
eagle_backend_status eagle_backend_capturer_end(eagle_backend_capturer h, eagle_backend_captured* out);
/** The graph the in-flight capture records into (0: none). Added in minor 5. */
eagle_backend_status eagle_backend_capturer_graph(eagle_backend_capturer h, uint64_t* graph);
void eagle_backend_capturer_release(eagle_backend_capturer h);

/* ---- captured (bit 2) -------------------------------------------------------- */

eagle_backend_status eagle_backend_captured_is_valid(eagle_backend_captured h, int32_t* valid);
void eagle_backend_captured_release(eagle_backend_captured h);

/* ---- fork (bit 3) ------------------------------------------------------------ */

eagle_backend_status eagle_backend_fork_create(const struct eagle_backend_fork_desc* desc, eagle_backend_fork* out);
eagle_backend_status eagle_backend_fork_fork(eagle_backend_fork h);
eagle_backend_status eagle_backend_fork_join(eagle_backend_fork h);
/** INDEX when ``i`` is out of range. */
eagle_backend_status eagle_backend_fork_branch(eagle_backend_fork h, uint64_t i, uint64_t* stream);
eagle_backend_status eagle_backend_fork_origin(eagle_backend_fork h, uint64_t* stream);
eagle_backend_status eagle_backend_fork_forked(eagle_backend_fork h, int32_t* forked);
eagle_backend_status eagle_backend_fork_size(eagle_backend_fork h, uint64_t* n);
void eagle_backend_fork_release(eagle_backend_fork h);
/** Mark branch ``producer``'s tip; ``*mark`` indexes it. INDEX past ``wait_count``
 *  marks. Added in minor 5; bound per symbol. */
eagle_backend_status eagle_backend_fork_mark(eagle_backend_fork h, uint64_t producer, uint64_t* mark);
/** Branch ``waiter`` waits for ``mark``: one captured edge. Added in minor 5. */
eagle_backend_status eagle_backend_fork_wait(eagle_backend_fork h, uint64_t waiter, uint64_t mark);

/* ---- conditional (bit 4) ----------------------------------------------------- */

eagle_backend_status eagle_backend_conditional_create(const struct eagle_backend_conditional_desc* desc, eagle_backend_conditional* out);
eagle_backend_status eagle_backend_conditional_begin(eagle_backend_conditional h, uint64_t* body_stream);
eagle_backend_status eagle_backend_conditional_body_stream(eagle_backend_conditional h, uint64_t* body_stream);
eagle_backend_status eagle_backend_conditional_is_loop(eagle_backend_conditional h, int32_t* is_loop);
eagle_backend_status eagle_backend_conditional_end(eagle_backend_conditional h);
void eagle_backend_conditional_release(eagle_backend_conditional h);
/** Arm ``k``'s capture stream. Added in minor 5; bound per symbol. */
eagle_backend_status eagle_backend_conditional_arm_stream(eagle_backend_conditional h, uint64_t k, uint64_t* stream);
/** ``which`` >= 0: the nodes of arm ``which``'s body graph; -1: the setter nodes
 *  (head, + the tail of a WHILE). ``*nodes`` is backend-allocated
 *  (``eagle_backend_free``). Added in minor 5. */
eagle_backend_status eagle_backend_conditional_nodes(eagle_backend_conditional h, int64_t which, uint64_t** nodes,
    int64_t* count);

/* ---- attribution (bit 5) ----------------------------------------------------- */

/** The node handles of the graph ``stream`` is capturing into; ``*nodes`` is
 *  backend-allocated (``eagle_backend_free``). */
eagle_backend_status eagle_backend_capture_snapshot_nodes(uint64_t stream, uint64_t** nodes, int64_t* count);
eagle_backend_status eagle_backend_is_node_toggleable(uint64_t node, int32_t* toggleable);
/** Insert the graph ``child`` (a ``cudaGraph_t`` another library owns) at the
 *  capture tip of ``stream`` as one child-graph node; ``conditional_body`` != 0
 *  refuses, by name, a node type CUDA forbids inside a conditional body. Added in
 *  minor 3; bound per symbol, so its presence is the capability. */
eagle_backend_status eagle_backend_capture_add_child(uint64_t stream, uint64_t child, int32_t conditional_body);

/* ---- graph (bit 6) ----------------------------------------------------------- */

eagle_backend_status eagle_backend_graph_create(const struct eagle_backend_graph_desc* desc, eagle_backend_graph* out);
/** CONSUMES ``captured`` (on OK); INVALID_ARGUMENT for an empty capture. */
eagle_backend_status eagle_backend_graph_from_captured(eagle_backend_captured captured, eagle_backend_graph* out);
eagle_backend_status eagle_backend_graph_stream(eagle_backend_graph h, uint64_t stream);
/** CONSUMES ``captured`` (on OK). */
eagle_backend_status eagle_backend_graph_add_node(eagle_backend_graph h, eagle_backend_captured captured);
eagle_backend_status eagle_backend_graph_launcher(eagle_backend_graph h, eagle_backend_launcher* out);
eagle_backend_status eagle_backend_graph_last_node(eagle_backend_graph h, int64_t* index);
void eagle_backend_graph_release(eagle_backend_graph h);

/* ---- launcher (bit 7) -------------------------------------------------------- */

eagle_backend_status eagle_backend_launcher_launch(eagle_backend_launcher h);
eagle_backend_status eagle_backend_launcher_synchronize(eagle_backend_launcher h);
eagle_backend_status eagle_backend_launcher_stream(eagle_backend_launcher h, uint64_t stream);
eagle_backend_status eagle_backend_launcher_set_logical_size(eagle_backend_launcher h, int64_t logical_size);
eagle_backend_status eagle_backend_launcher_kernel_node_count(eagle_backend_launcher h, int64_t* count);
eagle_backend_status eagle_backend_launcher_set_node_enabled(eagle_backend_launcher h, uint64_t node, int32_t enabled);
void eagle_backend_launcher_release(eagle_backend_launcher h);
/** Patch kernel node ``node``'s first ``n`` arguments from the addresses in
 *  ``params`` (0 keeps one); later arguments keep their captured values. Added
 *  in minor 5, like the two below; bound per symbol. */
eagle_backend_status eagle_backend_launcher_set_kernel_params(eagle_backend_launcher h, uint64_t node,
    const uint64_t* params, int64_t n);
/** Re-point memcpy node ``node``'s source; everything else stays as captured. */
eagle_backend_status eagle_backend_launcher_set_memcpy_source(eagle_backend_launcher h, uint64_t node, uint64_t src);
/** Update the exec from ``captured`` (not consumed); ``*accepted`` = 0 when the
 *  driver rejects it (the exec is unchanged). */
eagle_backend_status eagle_backend_launcher_exec_update(eagle_backend_launcher h, eagle_backend_captured captured,
    int32_t* accepted);

/* ---- exec (bit 9) ------------------------------------------------------------ */

/** One launch over one partition; ``*launched`` = 1, or 0 for an empty partition. */
eagle_backend_status eagle_backend_run_device(const struct eagle_backend_launch_desc* desc, int32_t* launched);

/* ---- device (bit 10) --------------------------------------------------------- */

eagle_backend_status eagle_backend_device_props(int32_t device, struct eagle_backend_device_props* props);

/* ---- interop (bit 11) -------------------------------------------------------- */

/** Order ``producer`` before ``consumer`` on device (``dl_device_type``,
 *  ``device_id``) without blocking the host; codes per the file doc. */
eagle_backend_status eagle_backend_fence(int32_t dl_device_type, int32_t device_id, int64_t producer, int64_t consumer);
eagle_backend_status eagle_backend_event_pool_created(int64_t* count);
/** Zero ``bytes`` bytes at the device address ``ptr``, enqueued on ``stream``
 *  (a raw 0 is the legacy default stream) on the stream's device: no
 *  synchronisation, no allocation, capture-legal. Added in minor 2; bound per
 *  symbol, so its presence is the capability. */
eagle_backend_status eagle_backend_memset_device(uint64_t ptr, uint64_t bytes, uint64_t stream);
/** Copy ``bytes`` bytes from the device address ``src`` to the device address
 *  ``dst`` (distinct, non-overlapping), enqueued on ``stream`` like
 *  ``memset_device``. Added in minor 4; bound per symbol. */
eagle_backend_status eagle_backend_memcpy_device(uint64_t dst, uint64_t src, uint64_t bytes, uint64_t stream);

/** Directions of ``eagle_backend_memcpy``. */
#define EAGLE_BACKEND_COPY_H2D 0
#define EAGLE_BACKEND_COPY_D2H 1
#define EAGLE_BACKEND_COPY_D2D 2
#define EAGLE_BACKEND_COPY_H2H 3

/** Copy ``bytes`` bytes from ``src`` to ``dst`` in direction ``kind``
 *  (``EAGLE_BACKEND_COPY_*``), enqueued on ``stream`` like ``memset_device``:
 *  capture-legal, a memcpy node inside a capture. The two ranges are distinct
 *  and non-overlapping, or the same address, which enqueues nothing. A host
 *  range need not be pinned (an unpinned one is staged, with no overlap). A
 *  backend whose memory is unified may make every copy a no-op. Added in
 *  minor 6; bound per symbol. */
eagle_backend_status eagle_backend_memcpy(uint64_t dst, uint64_t src, uint64_t bytes, int32_t kind, uint64_t stream);
/** Allocate ``bytes`` bytes of page-locked host memory the device can DMA
 *  from; its address goes to ``*out`` (0 for ``bytes == 0``). Added in minor 6;
 *  bound per symbol. */
eagle_backend_status eagle_backend_host_alloc(uint64_t bytes, uint64_t* out);
/** Free an address from ``host_alloc`` (0 is fine). Added in minor 6; bound per
 *  symbol. */
eagle_backend_status eagle_backend_host_free(uint64_t ptr);
/** Call ``fn(arg)`` on a backend thread once the work enqueued before it on
 *  ``stream`` is done; capture-legal, a host node inside a capture that runs on
 *  every replay. ``fn`` must not call the backend, and ``arg`` must outlive
 *  every execution. Added in minor 6; bound per symbol. */
eagle_backend_status eagle_backend_host_func(uint64_t stream, void (*fn)(void*), void* arg);

/* ---- minor 7: device scope, modules, raw launch, built-in cell ops ----------- */

/** Number of devices the backend sees. Added in minor 7; bound per symbol, as
 *  are the nine after it. */
eagle_backend_status eagle_backend_device_count(int32_t* count);
/** Make @p device the calling thread's current device (its primary context). */
eagle_backend_status eagle_backend_set_device(int32_t device);
/** ``*device`` = the calling thread's current device ordinal (the selected one while the thread has no context
 *  yet: 0 by default). Reads no device properties and creates no context. */
eagle_backend_status eagle_backend_get_device(int32_t* device);
/** Read (and clear) the calling thread's pending launch error on @p device (-1: the current one): success when the
 *  launches issued so far left none, else a failure whose ``eagle_backend_last_error()`` names the CUDA error. A
 *  launch-error query, not a wait: it never synchronises. */
eagle_backend_status eagle_backend_launch_status(int32_t device);
/** Block the host until all work on @p device (-1: the current one) is done. */
eagle_backend_status eagle_backend_device_synchronize(int32_t device);
/** ``*capturing`` = 1 when @p stream (0: the legacy default) is inside a capture. */
eagle_backend_status eagle_backend_stream_is_capturing(uint64_t stream, int32_t* capturing);

/** Load a device module from @p image (``bytes`` bytes of cubin, fatbin or
 *  PTX text) in the primary context of @p device (-1: the current one).
 *  The module handle goes to ``*module``; free it with ``module_release`` on the
 *  same device. */
eagle_backend_status eagle_backend_module_load(const void* image, uint64_t bytes, int32_t device, uint64_t* module);
/** ``module_load`` of PTX text, JIT-compiled under a cap of @p max_registers registers per thread
 *  (``CU_JIT_MAX_REGISTERS``; 0: no cap). A cubin ignores the cap. */
eagle_backend_status eagle_backend_module_load_jit(const void* image, uint64_t bytes, int32_t device,
                                                   uint32_t max_registers, uint64_t* module);
/** Link PTX text (@p image, ``bytes`` bytes) for @p device under a cap of @p max_registers registers per thread
 *  (0: no cap) and copy the resulting cubin to @p out (``capacity`` bytes; may be null to only measure). ``*size``
 *  is the cubin's size; the call fails if @p out is too small. */
eagle_backend_status eagle_backend_link_ptx(const void* image, uint64_t bytes, int32_t device, uint32_t max_registers,
                                            void* out, uint64_t capacity, uint64_t* size);
/** The function @p name of @p module; its raw handle goes to ``*function``. Valid
 *  until the module is released. */
eagle_backend_status eagle_backend_module_function(uint64_t module, const char* name, uint64_t* function);
/** Unload @p module (0 is fine) on @p device. */
void eagle_backend_module_release(uint64_t module, int32_t device);
/** One ``CUfunction`` attribute (the driver's ``CUfunction_attribute`` code: 0 max threads per block, 1 static
 *  shared bytes, 2 constant bytes, 3 local bytes, 4 registers) of @p function (``cuFuncGetAttribute``). */
eagle_backend_status eagle_backend_function_attribute(uint64_t function, int32_t attribute, int32_t* value);

/** One launch of a raw function (``launch_raw``). */
struct eagle_backend_launch_raw_desc {
    uint32_t struct_size;
    uint32_t flags;           /**< zero */
    int32_t device;           /**< -1: the stream's device */
    uint32_t shared_bytes;    /**< dynamic shared memory */
    uint64_t function;        /**< from ``module_function`` */
    uint64_t stream;          /**< 0: the legacy default stream */
    uint32_t grid[3];
    uint32_t block[3];
    uint64_t args;            /**< ``void*[arg_count]``, each element the address of one argument value */
    uint32_t arg_count;
    uint32_t reserved;        /**< zero */
};
/** ``cuLaunchKernel`` of ``desc->function``: enqueue only, capture-legal. */
eagle_backend_status eagle_backend_launch_raw(const struct eagle_backend_launch_raw_desc* desc);

/** ``eagle_backend_cell_op_desc.op`` values. Append-only. Integer kernels, so a
 *  result is bit-equal to the plain C expression it names. ``p`` are device
 *  addresses, ``i`` integer arguments, ``n`` the element count.
 *
 *  | op | name              | p                                  | i |
 *  |----|-------------------|------------------------------------|---|
 *  | 0  | BUMP              | p0 u32[1]: ++                      | |
 *  | 1  | MASK_COUNT        | p0 u8[n] mask, p1 u32[1]: += count of set bytes | |
 *  | 2  | POLICY            | p0 finished u32[1], p1 total u32[1], p2 cells i64[], p3 word i64[1], p4 go u32[1], p5 hist i64[]: one policy step | i0 k_min, i1 k_max, i2..i7 the cell indices k, fin_at, done, cap, r, mapped, i8 rate scale |
 *  | 3  | IDENTITY          | p0 i32[n]: a[j] = j                | |
 *  | 4  | COUNT_FIRED       | p0 fire u32[1], p1 reorders u32[1]: += (fire != 0) | |
 *  | 5  | GROUP_AND         | p0 u8[i0] mask, p1 u8[n] drop: drop[g] = AND of the group's lanes (lanes past i0 count as set) | i0 lanes, i1 group |
 *  | 6  | SCALE_COUNT       | p0 u32[1] c, p1 u32[1] o: o = c * group | i0 group |
 *  | 7  | CAST              | p0 dst[n], p1 src[n] (contiguous)  | i0 dst dtype, i1 src dtype (``EAGLE_BACKEND_DTYPE_*``) |
 *  | 8  | CONTIGUOUS_GATHER | p0 dst (contiguous), p1 src (first element) | i0 elem bytes (1,2,4,8), i1 ndim (1..4), i2..i5 shape, i6..i9 src strides in bytes (signed); n = product of shape |
 *  | 9  | ADD               | p0 dst[n], p1 a[n], p2 b[n]        | i0 dtype (any dtype but bool; minor 7: u8, u32, u64, f16, bf16 join) |
 */
#define EAGLE_BACKEND_CELL_BUMP 0u
#define EAGLE_BACKEND_CELL_MASK_COUNT 1u
#define EAGLE_BACKEND_CELL_POLICY 2u
#define EAGLE_BACKEND_CELL_IDENTITY 3u
#define EAGLE_BACKEND_CELL_COUNT_FIRED 4u
#define EAGLE_BACKEND_CELL_GROUP_AND 5u
#define EAGLE_BACKEND_CELL_SCALE_COUNT 6u
#define EAGLE_BACKEND_CELL_CAST 7u
#define EAGLE_BACKEND_CELL_CONTIGUOUS_GATHER 8u
#define EAGLE_BACKEND_CELL_ADD 9u
/* 10-23: the array ops (eagle/array/kernels.h defines each, one body shared by the device kernel and the
 * host twin). dtype = ``i[9]`` (``EAGLE_BACKEND_DTYPE_*``), n = elements, scalar parameters ``i[0]``/``i[1]``
 * (integral dtypes) or ``f[0]``/``f[1]`` (float dtypes). Operands per op are listed in kernels.h. */
#define EAGLE_BACKEND_CELL_FILL 10u
#define EAGLE_BACKEND_CELL_SUB 11u
#define EAGLE_BACKEND_CELL_MUL 12u
#define EAGLE_BACKEND_CELL_MAX 13u
#define EAGLE_BACKEND_CELL_EQ 14u
#define EAGLE_BACKEND_CELL_NE 15u
#define EAGLE_BACKEND_CELL_GT 16u
#define EAGLE_BACKEND_CELL_WHERE 17u
#define EAGLE_BACKEND_CELL_CLIP 18u
#define EAGLE_BACKEND_CELL_ARANGE 19u
#define EAGLE_BACKEND_CELL_LINSPACE 20u
#define EAGLE_BACKEND_CELL_TAKE 21u
#define EAGLE_BACKEND_CELL_CUMSUM 22u
#define EAGLE_BACKEND_CELL_SCATTER 23u
/* 24 COUNT_NONZERO: p0 int64[1] (zeroed by the op, then the count), p1 x[n] (dtype i[9]).
 * 25 FLATNONZERO:   p0 int32[n] (the indices of the nonzero elements, first count slots), p1 x[n], p2 zeroed-by-the-op
 *                   scan scratch of cumsum_scratch_bytes(n), p3 int64[1] (the count). Both: any compute dtype. */
#define EAGLE_BACKEND_CELL_COUNT_NONZERO 24u
#define EAGLE_BACKEND_CELL_FLATNONZERO 25u
/* 26-34: the scalar-operand forms (minor 7): p0 dst[n], p1 x[n], the scalar in ``i[0]`` (integral dtypes) or ``f[0]``
 * (float dtypes), dtype ``i[9]``. ADDS/SUBS/MULS/MAXS = x op s; EQS/NES/GTS = u8[n] of x op s; RSUBS = s - x;
 * LTS = u8[n] of x < s (the scalar-on-the-left spellings of subtraction and greater-than). One kernel node each. */
#define EAGLE_BACKEND_CELL_ADDS 26u
#define EAGLE_BACKEND_CELL_SUBS 27u
#define EAGLE_BACKEND_CELL_MULS 28u
#define EAGLE_BACKEND_CELL_MAXS 29u
#define EAGLE_BACKEND_CELL_EQS 30u
#define EAGLE_BACKEND_CELL_NES 31u
#define EAGLE_BACKEND_CELL_GTS 32u
#define EAGLE_BACKEND_CELL_RSUBS 33u
#define EAGLE_BACKEND_CELL_LTS 34u
/* 35 STRIDED_COPY (minor 7): p0 dst, p1 src (each the first element, any strides), i0 elem bytes (1,2,4,8), i1 ndim
 * (1..4), i2..i5 shape, i6..i9 dst strides in bytes (signed), j[0..3] src strides in bytes (signed); n = product of
 * shape. One kernel node, no allocation: both offsets come from the logical index. A contiguous side is the
 * contiguous strides, so this is also the gather and the scatter. */
#define EAGLE_BACKEND_CELL_STRIDED_COPY 35u

/** ``cell_op`` dtype codes. */
#define EAGLE_BACKEND_DTYPE_BOOL 0
#define EAGLE_BACKEND_DTYPE_UINT8 1
#define EAGLE_BACKEND_DTYPE_INT32 2
#define EAGLE_BACKEND_DTYPE_UINT32 3
#define EAGLE_BACKEND_DTYPE_INT64 4
#define EAGLE_BACKEND_DTYPE_UINT64 5
#define EAGLE_BACKEND_DTYPE_FLOAT32 6
#define EAGLE_BACKEND_DTYPE_FLOAT64 7
#define EAGLE_BACKEND_DTYPE_FLOAT16 8   /* minor 7: array ops and casts; arithmetic needs sm_53 */
#define EAGLE_BACKEND_DTYPE_BFLOAT16 9  /* minor 7: array ops and casts; arithmetic needs sm_80 */

struct eagle_backend_cell_op_desc {
    uint32_t struct_size;
    uint32_t flags;           /**< zero */
    uint32_t op;              /**< EAGLE_BACKEND_CELL_* */
    int32_t device;           /**< -1: the stream's device */
    uint64_t stream;          /**< 0: the legacy default stream */
    int64_t n;                /**< element or thread count, per op */
    uint64_t p[6];
    int64_t i[10];
    double f[2];              /**< appended in minor 7: float scalars of the array ops */
    int64_t j[4];             /**< appended in minor 7: the source byte strides of STRIDED_COPY */
};
/** Enqueue the built-in op ``desc->op`` (the plugin carries SASS for every ship
 *  arch; no JIT). Capture-legal: enqueue only, no allocation, no sync. */
eagle_backend_status eagle_backend_cell_op(const struct eagle_backend_cell_op_desc* desc);

/** ``eagle_backend_gemm_desc.dtype`` values. */
#define EAGLE_BACKEND_GEMM_F32 0
#define EAGLE_BACKEND_GEMM_F64 1

/** A batched GEMM in cuBLAS's column-major terms: ``C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i]`` for
 *  ``batch`` matrices ``stride_*`` ELEMENTS apart (``batch`` 1: one product; the strides are then unused).
 *  ``op_*`` 0: as is, 1: transposed. ``a``, ``b``, ``c`` are device addresses. */
struct eagle_backend_gemm_desc {
    uint32_t struct_size;
    uint32_t dtype;           /**< EAGLE_BACKEND_GEMM_* */
    int32_t device;           /**< -1: the stream's device */
    uint32_t op_a, op_b;
    uint64_t stream;          /**< 0: the legacy default stream */
    int64_t m, n, k;
    int64_t lda, ldb, ldc;
    uint64_t a, b, c;
    int64_t batch, stride_a, stride_b, stride_c;
    double alpha, beta;
};
/** Enqueue the GEMM on ``desc->stream`` through cuBLAS, which the plugin loads at run time (``dlopen``; the
 *  plugin links none). One cuBLAS handle per (device, stream) is created at its first call (which allocates:
 *  call once before a capture) and reused; capture-legal after that. Fails as unavailable, naming the
 *  libraries tried, when cuBLAS is absent. */
eagle_backend_status eagle_backend_gemm(const struct eagle_backend_gemm_desc* desc);
/** Destroy the cuBLAS handle of (@p device, @p stream), if any (call once the stream's work is done). */
void eagle_backend_gemm_release(int32_t device, uint64_t stream);

/** Strided copy (``cuMemcpy2DAsync``): ``height`` rows of ``width_bytes``, row
 *  starts ``dst_pitch`` / ``src_pitch`` bytes apart, direction ``kind``
 *  (``EAGLE_BACKEND_COPY_*``, not H2H), enqueued on ``stream`` like ``memcpy``. */
eagle_backend_status eagle_backend_memcpy_strided(uint64_t dst, uint64_t dst_pitch, uint64_t src, uint64_t src_pitch,
    uint64_t width_bytes, uint64_t height, int32_t kind, uint64_t stream);

/* ---- minor 7: stream-ordered memory (the array layer's device storage) ------- */

/** ``mem_alloc`` kinds. */
#define EAGLE_BACKEND_MEM_DEVICE 0
#define EAGLE_BACKEND_MEM_PINNED 1

/** Allocate ``bytes`` of ``kind`` memory on ``device`` (-1: the stream's). A device allocation is
 *  stream-ordered on ``stream`` (0: the legacy default stream) and aligned to 256 bytes; its address goes to
 *  ``*ptr`` and the handle to free it to ``*chunk``. Refused by name while ``stream`` is capturing (an
 *  allocation is not capture-legal). ``bytes == 0`` gives a null ``*ptr`` and ``*chunk``. */
eagle_backend_status eagle_backend_mem_alloc(int32_t kind, int32_t device, uint64_t bytes, uint64_t stream,
    uint64_t* chunk, uint64_t* ptr);
/** Free ``chunk`` (0 is fine), stream-ordered on its allocation stream. While that stream is capturing the
 *  free is DEFERRED (queued) and runs at the next ``mem_alloc`` or ``mem_drain`` outside a capture. */
void eagle_backend_mem_release(uint64_t chunk);
/** Run every deferred free now; ``*left`` (may be null) gets the number still queued (0 unless a stream is
 *  still capturing). */
eagle_backend_status eagle_backend_mem_drain(int64_t* left);
/** The device ordinal of the allocation holding ``ptr`` (``-1``: host or unknown memory). */
eagle_backend_status eagle_backend_pointer_device(uint64_t ptr, int32_t* device);

/** Block the host until everything queued on ``stream`` (0: the legacy default stream) is done, on ``device``
 *  (-1: the stream's). Only that stream, not the device. Not capture-legal: inside a capture it fails by name. */
eagle_backend_status eagle_backend_stream_sync(int32_t device, uint64_t stream);

/** A timing-free event on @p device (-1: the current one); its raw handle goes to ``*event``. */
eagle_backend_status eagle_backend_event_create(int32_t device, uint64_t* event);
/** Like ``event_create`` but the event keeps its timestamp, so ``event_elapsed`` can measure between two. */
eagle_backend_status eagle_backend_event_create_timed(int32_t device, uint64_t* event);
/** Milliseconds between two recorded timed events, to ``*ms``. Both must have completed (synchronize the later
 *  one first): an unfinished event fails by name. Not capture-legal. */
eagle_backend_status eagle_backend_event_elapsed(uint64_t start, uint64_t end, float* ms);
/** Record @p event on @p stream (0: the legacy default stream). Enqueue only, capture-legal. */
eagle_backend_status eagle_backend_event_record(uint64_t event, uint64_t stream);
/** Make @p stream wait for the work @p event last recorded. Enqueue only, capture-legal. */
eagle_backend_status eagle_backend_event_wait(uint64_t stream, uint64_t event);
/** Destroy @p event (0 is fine) on @p device (-1: the current one). */
void eagle_backend_event_release(uint64_t event, int32_t device);

/* ---- filtering (bit 12) ------------------------------------------------------ */

/** ``eagle_backend_compact_desc.flags``: a set mask byte DROPS the sample (a
 *  ``terminated`` mask); without it a set byte keeps the sample. */
#define EAGLE_BACKEND_COMPACT_MASK_IS_DROP (UINT32_C(1) << 0)

/** Active-set compaction of ``n`` samples (``compact_device``). Every buffer is a
 *  raw device address on ``device``. */
struct eagle_backend_compact_desc {
    uint32_t struct_size;
    uint32_t flags;          /**< EAGLE_BACKEND_COMPACT_* */
    int32_t device;
    uint32_t reserved;       /**< zero */
    uint64_t mask;           /**< n bytes, one per sample */
    uint64_t index_map;      /**< int32_t[n]: OUT, the kept indices, ascending, in [0, *count) */
    uint64_t count;          /**< uint32_t[1]: OUT, the number of kept samples */
    uint64_t scratch;        /**< scratch_bytes of working memory; 0 = query */
    int64_t n;               /**< sample count, 0 <= n <= 2^31 - 1 */
    uint64_t stream;         /**< the backend's queue to enqueue on */
    int64_t scratch_bytes;   /**< OUT when scratch == 0: the bytes scratch must hold */
    /* Appended: the reorder trigger. A desc whose struct_size ends before
     * ``live32``, or whose ``live32`` is 0, asks for no trigger. */
    uint64_t live32;         /**< uint32_t[1]: OUT, 32-sample groups holding a kept sample */
    uint64_t span;           /**< uint32_t[1]: samples [0, *span) may still be kept */
    uint64_t fire;           /**< uint32_t[1]: OUT, 1 when *count < theta * 32 * *live32 && *count < *span && *span >= 4096 */
    float theta;             /**< the locality threshold, in (0, 1) */
    uint32_t reserved2;      /**< zero */
};

/** Compact the samples ``mask`` keeps into ``index_map[0, *count)``, ascending.
 *
 *  QUERY: with ``scratch == 0`` the call only writes ``desc->scratch_bytes`` and
 *  returns OK; nothing is enqueued and no other field is read but ``n``.
 *  CAPTURABILITY CONTRACT: otherwise the call only ENQUEUES on ``desc->stream`` —
 *  no allocation, no synchronisation, no host read of device memory — so it is
 *  stream-capturable on every backend that has capture, and merely asynchronous
 *  on one that has not. ``n > 2^31 - 1`` returns ``EAGLE_BACKEND_UNSUPPORTED``. */
eagle_backend_status eagle_backend_compact_device(struct eagle_backend_compact_desc* desc);

/** One plane of a reorder: a device address of ``n`` elements of ``elem_bytes``
 *  (1, 2, 4, 8 or 16), aligned to it. */
struct eagle_backend_reorder_plane {
    uint64_t data;
    uint32_t elem_bytes;
    uint32_t reserved;       /**< zero */
};

/** A physical reorder (``reorder_device``) or its undo (``x_restore_device``).
 *  Every buffer is a raw device address on ``device``. */
struct eagle_backend_reorder_desc {
    uint32_t struct_size;
    uint32_t flags;          /**< EAGLE_BACKEND_COMPACT_MASK_IS_DROP */
    int32_t device;
    uint32_t n_planes;
    uint64_t planes;         /**< host array of n_planes eagle_backend_reorder_plane, read at enqueue */
    uint64_t mask;           /**< n bytes, one per sample; one of the planes */
    uint64_t perm;           /**< int32_t[n]: slot -> sample id */
    uint64_t inv;            /**< int32_t[n]: sample id -> slot */
    uint64_t index_map;      /**< int32_t[n]: OUT, the identity over [0, *count) */
    uint64_t count;          /**< uint32_t[1]: OUT, the number of kept samples */
    uint64_t span;           /**< uint32_t[1]: IN/OUT, [0, *span) moves; set to *count */
    uint64_t fire;           /**< uint32_t[1]: 0 = unconditional; else the kernels exit while *fire == 0, and it is cleared */
    uint64_t scratch;        /**< scratch_bytes of working memory; 0 = query */
    int64_t n;               /**< sample count, 0 <= n <= 2^31 - 1 */
    uint64_t stream;         /**< the backend's queue to enqueue on */
    int64_t scratch_bytes;   /**< OUT when scratch == 0: the bytes scratch must hold (for every plane's element size) */
};

/** Permute every plane IN PLACE over ``[0, *span)`` so the samples ``mask``
 *  keeps fill ``[0, *count)`` in ascending sample order, the dropped ones after
 *  them; ``perm`` is permuted with them and ``inv`` rebuilt; ``index_map``
 *  becomes the identity over ``[0, *count)``; ``*span = *count``.
 *
 *  QUERY and CAPTURABILITY CONTRACT as for ``compact_device`` (the query reads
 *  ``n``, ``n_planes`` and the plane table's element sizes). Bound per symbol:
 *  added within major 1, so its presence is the capability. */
eagle_backend_status eagle_backend_reorder_device(struct eagle_backend_reorder_desc* desc);

/* ---- filtering: K classes (bound per symbol, like reorder_device) ------------- */

/** ``eagle_backend_reorder_classes_desc.flags``: sort the slots ``[0, *count)``
 *  among themselves by class only (the live prefix of a reverse pass); without
 *  it the sort is by ``(live, class)`` over ``[0, *span)``. An older backend
 *  that does not know the bit refuses the call (UNSUPPORTED). */
#define EAGLE_BACKEND_CLASSES_WITHIN_LIVE (UINT32_C(1) << 1)
/** The most classes (``classes`` in [2, 255]; the class plane is one byte per sample). */
#define EAGLE_BACKEND_MAX_CLASSES 255

/** ``compact_device`` plus the K-class trigger. The first block repeats
 *  ``eagle_backend_compact_desc`` field for field; ``live32``, ``span`` and
 *  ``fire`` are REQUIRED here. Every buffer is a raw device address. */
struct eagle_backend_compact_classes_desc {
    uint32_t struct_size;
    uint32_t flags;          /**< EAGLE_BACKEND_COMPACT_MASK_IS_DROP */
    int32_t device;
    uint32_t reserved;       /**< zero */
    uint64_t mask;
    uint64_t index_map;
    uint64_t count;
    uint64_t scratch;        /**< 0 = query */
    int64_t n;
    uint64_t stream;
    int64_t scratch_bytes;   /**< OUT when scratch == 0 */
    uint64_t live32;         /**< uint32_t[1]: OUT */
    uint64_t span;           /**< uint32_t[1]: IN */
    uint64_t fire;           /**< uint32_t[1]: OUT, the liveness trigger of compact_device */
    float theta;             /**< the liveness threshold, in (0, 1) */
    uint32_t reserved2;      /**< zero */
    /* the class fields */
    uint64_t class_plane;    /**< uint8_t[n], each sample's class in [0, classes) */
    uint32_t classes;        /**< K, in [2, 255] */
    float class_theta;       /**< the mixed-group threshold, in (0, 1) */
    uint64_t offsets;        /**< uint32_t[classes + 1]: reserved for the sort; not written here (0 allowed) */
    uint64_t mixed32;        /**< uint32_t[1]: OUT, 32-slot groups holding two or more live classes */
    uint64_t stale_count;    /**< uint32_t[1]: IN, samples whose class changed since the last sort; 0 = none */
    uint64_t class_fire;     /**< uint32_t[1]: OUT, (mixed32 > 0 && mixed32 >= class_theta * live32 && span >= 4096)
                              *   || stale_count > 0.05 * count || fire */
    uint64_t reserved3;      /**< zero */
};

/** ``compact_device`` with the class trigger (see ``eagle_backend_compact_device``
 *  for the query and capturability contract; both hold here). Bound per symbol:
 *  its presence is the capability. */
eagle_backend_status eagle_backend_compact_classes_device(struct eagle_backend_compact_classes_desc* desc);

/** ``reorder_device`` generalised to K classes. The first block repeats
 *  ``eagle_backend_reorder_desc`` field for field. ``class_plane`` must be one of
 *  the planes (it moves with them). */
struct eagle_backend_reorder_classes_desc {
    uint32_t struct_size;
    uint32_t flags;          /**< EAGLE_BACKEND_COMPACT_MASK_IS_DROP | EAGLE_BACKEND_CLASSES_WITHIN_LIVE */
    int32_t device;
    uint32_t n_planes;
    uint64_t planes;
    uint64_t mask;
    uint64_t perm;
    uint64_t inv;
    uint64_t index_map;
    uint64_t count;
    uint64_t span;
    uint64_t fire;           /**< 0 = unconditional; else the kernels exit while *fire == 0, and it is cleared */
    uint64_t scratch;        /**< 0 = query */
    int64_t n;
    uint64_t stream;
    int64_t scratch_bytes;   /**< OUT when scratch == 0 */
    /* the class fields */
    uint64_t class_plane;    /**< uint8_t[n] */
    uint32_t classes;        /**< K, in [2, 255] */
    uint32_t reserved3;      /**< zero */
    uint64_t offsets;        /**< uint32_t[classes + 1]: OUT, exclusive prefix over the LIVE samples */
    uint64_t class_sorts;    /**< uint32_t[1]: incremented once when the permutation ran */
};

/** Permute every plane IN PLACE by a stable counting sort on ``(live, class)``
 *  over ``[0, *span)`` (dead last), then ``*count = *span = live``; with
 *  WITHIN_LIVE, over ``[0, *count)`` by class only, ``count``/``span`` unchanged.
 *  ``perm`` moves, ``inv`` is rebuilt, ``index_map`` is the identity over
 *  ``[0, live)``, ``offsets`` is written. Query and capturability as for
 *  ``reorder_device`` (the query's ``scratch_bytes`` includes the class
 *  working set). Bound per symbol. */
eagle_backend_status eagle_backend_reorder_classes_device(struct eagle_backend_reorder_classes_desc* desc);

/** Reset to the identity: ``perm``, ``inv`` and (when non-zero) ``index_map`` become
 *  ``0..n-1`` and ``*count = *span = n``, in ONE kernel. Only ``device``, ``perm``,
 *  ``inv``, ``index_map``, ``count``, ``span``, ``n`` and ``stream`` of
 *  ``eagle_backend_reorder_desc`` are read (no planes, scratch or ``fire``). Enqueue
 *  only and capture-legal. Added in minor 1; bound per symbol, so its presence is
 *  the capability. */
eagle_backend_status eagle_backend_reset_device(struct eagle_backend_reorder_desc* desc);

/** Residency classes reported by ``eagle_backend_pointer_residency``. */
#define EAGLE_BACKEND_RESIDENCY_HOST 0
#define EAGLE_BACKEND_RESIDENCY_DEVICE 1
#define EAGLE_BACKEND_RESIDENCY_MANAGED 2
#define EAGLE_BACKEND_RESIDENCY_MAPPED_HOST 3
#define EAGLE_BACKEND_RESIDENCY_PAGEABLE 4
#define EAGLE_BACKEND_RESIDENCY_OTHER_DEVICE 5

/** Classify the raw address ``ptr`` as seen from ``device`` (-1 = the current one):
 *  one of ``EAGLE_BACKEND_RESIDENCY_*`` is written to ``*cls``. MAPPED_HOST is pinned
 *  host memory the device can address (a planes-in-host-memory buffer for
 *  ``reorder_device``); PAGEABLE is host memory it cannot. A setup-time query, not a
 *  per-call check. Added in minor 1; bound per symbol, so its presence is the
 *  capability. */
eagle_backend_status eagle_backend_pointer_residency(int32_t device, uint64_t ptr, int32_t* cls);

/* ---- experimental (eagle_backend_x_*, optional, unlocked) -------------------- */

/** Write the captured graph as Graphviz dot to ``path`` (the caller reads it). */
eagle_backend_status eagle_backend_x_captured_debug_dot(eagle_backend_captured h, const char* path, uint32_t flags);
eagle_backend_status eagle_backend_x_capture_guard_depth(int64_t* depth);
eagle_backend_status eagle_backend_x_capture_guard_pending_count(int64_t* count);
/** Undo every reorder since the last restore: each plane back in sample order,
 *  ``perm`` and ``inv`` the identity (``mask``, ``index_map``, ``count``,
 *  ``span`` and ``fire`` are not read). Enqueue only; eager use. */
eagle_backend_status eagle_backend_x_restore_device(struct eagle_backend_reorder_desc* desc);

/** Element types and operators of ``x_reduce_canonical``. */
#define EAGLE_BACKEND_REDUCE_F32 0u
#define EAGLE_BACKEND_REDUCE_F64 1u
#define EAGLE_BACKEND_REDUCE_I32 2u
#define EAGLE_BACKEND_REDUCE_I64 3u
#define EAGLE_BACKEND_REDUCE_U8  4u   /**< one byte per element; only with OP_ALL */
#define EAGLE_BACKEND_REDUCE_OP_SUM 0u
#define EAGLE_BACKEND_REDUCE_OP_MAX 1u
#define EAGLE_BACKEND_REDUCE_OP_ALL 2u
#define EAGLE_BACKEND_REDUCE_OP_MIN 3u
#define EAGLE_BACKEND_REDUCE_COMPENSATED 0u
#define EAGLE_BACKEND_REDUCE_ORDERED     1u
#define EAGLE_BACKEND_REDUCE_AUTO        2u
#define EAGLE_BACKEND_REDUCE_PAIR_OUT    1u

/** The canonical fixed-order reduction of ``n`` device elements into one. */
struct eagle_backend_reduce_desc {
    uint32_t struct_size;
    uint32_t dtype;          /**< EAGLE_BACKEND_REDUCE_F32 .. U8 */
    uint32_t op;             /**< EAGLE_BACKEND_REDUCE_OP_SUM / MAX / ALL */
    int32_t device;
    uint64_t in;             /**< n elements, device address */
    uint64_t out;            /**< one element, device address (may not alias work or in) */
    uint64_t work_a;         /**< work_elems elements, device address; 0 = query */
    uint64_t work_b;         /**< work_elems elements, device address */
    int64_t n;               /**< sample count, n >= 1 */
    uint64_t stream;         /**< the backend's queue to enqueue on */
    int64_t work_elems;      /**< OUT when work_a == 0: the elements EACH work buffer must hold */
    /* read only when struct_size covers them (older callers: ordered, 1 column, no comp) */
    uint32_t policy;         /**< EAGLE_BACKEND_REDUCE_COMPENSATED / ORDERED / AUTO (float sums only) */
    uint32_t flags;          /**< EAGLE_BACKEND_REDUCE_PAIR_OUT: out receives hi[columns] then comp[columns] */
    int64_t columns;         /**< in is row-major (n, columns); each column reduces to out[column] */
    uint64_t comp_in;        /**< 0, or the companion (n, columns) plane: each term is in + comp_in (float sums) */
};

/** Reduce each column of ``in`` (``n`` rows) with the canonical tree (leaves of
 *  256 slots by 16 strided samples, a fixed in-leaf tree, then level-by-level
 *  pairwise; see ``eagle/reduce/canonical.h``) and write it to ``out``. Under
 *  ORDERED the value is a function of the values in sample order only and
 *  equals the host canonical reduction bit for bit; under COMPENSATED (float
 *  sums) the leaf partials are committed by two-sum atomics instead. A float64
 *  sum without PAIR_OUT on a device with slow FP64 (``fixedLeaf`` in
 *  canonical.h) runs the fixed-point leaf and the pairwise levels under every
 *  policy, equal to the host fixed-point reduction bit for bit. Enqueue only (kernel launches ending in a one-element
 *  store: capturable). Experimental. */
eagle_backend_status eagle_backend_x_reduce_canonical(struct eagle_backend_reduce_desc* desc);

/** Time ``reps`` launches of ``desc`` (as ``run_device``), each between a pair of
 *  CUDA events on ``desc->stream``, and write the seconds of each to
 *  ``seconds[0, reps)``. Synchronises the stream once, after the last pair: for
 *  calibration, never inside a run. Experimental. */
eagle_backend_status eagle_backend_x_time_launch(const struct eagle_backend_launch_desc* desc, int32_t reps,
    double* seconds);

/** Column operations of a reverse window driver (``x_diff_columns``). */
#define EAGLE_BACKEND_DIFF_GATHER     0u   /**< dst[r*dst_ld+c] = src[r*src_ld+index[c]] */
#define EAGLE_BACKEND_DIFF_ACCUMULATE 1u   /**< dst[r*dst_ld+c] += src[r*dst_ld+c] (4: f32, 8: f64) */
#define EAGLE_BACKEND_DIFF_FILL       2u   /**< dst[c] = value (4: i32, 8: i64) */
#define EAGLE_BACKEND_DIFF_COPY       3u   /**< dst[0, count*elem_bytes) = src[...], device to device */
#define EAGLE_BACKEND_DIFF_READ       4u   /**< HOST dst = device src (count*elem_bytes bytes); waits for the stream */

/** One column operation over ``count`` columns of ``rows`` rows of
 *  sample-last device planes. Element size ``elem_bytes`` (gather: 1, 2, 4
 *  or 8). */
struct eagle_backend_diff_desc {
    uint32_t struct_size;
    uint32_t op;             /**< EAGLE_BACKEND_DIFF_GATHER / ACCUMULATE / FILL */
    int32_t device;
    uint32_t elem_bytes;
    uint64_t dst;            /**< device address */
    uint64_t src;            /**< device address (gather, accumulate) */
    uint64_t index;          /**< int64 device column index (gather) */
    int64_t dst_ld;          /**< row stride of dst, in elements */
    int64_t src_ld;          /**< row stride of src, in elements (gather) */
    int64_t count;
    int64_t rows;
    int64_t value;           /**< fill */
    uint64_t stream;
};

/** Enqueue one column operation (capturable, except READ, which waits for
 *  the stream and returns the bytes in host memory). Experimental. */
eagle_backend_status eagle_backend_x_diff_columns(struct eagle_backend_diff_desc* desc);

/** One per-sample event operation (``x_event_columns``): ``op`` is an
 *  ``eagle::events::Op`` (classify, newton, settle, graze, commit, seed,
 *  select; ``eagle/events/Refine.h``, ``Sensitivity.h``). Every address is a
 *  device plane of ``n`` samples (row pointers for ``a``/``b``/``c``), 0 when
 *  the operation does not use it. */
#define EAGLE_BACKEND_EVENT_GUARDS     8
#define EAGLE_BACKEND_EVENT_COMPONENTS 32
struct eagle_backend_event_desc {
    uint32_t struct_size;
    uint32_t op;
    int32_t device;
    uint32_t elem_bytes;     /**< 4: f32, 8: f64 */
    int64_t n;
    int32_t guards;
    int32_t components;
    uint64_t g[EAGLE_BACKEND_EVENT_GUARDS];
    uint64_t dg[EAGLE_BACKEND_EVENT_GUARDS];
    uint64_t gbar[EAGLE_BACKEND_EVENT_GUARDS];
    uint64_t a[EAGLE_BACKEND_EVENT_COMPONENTS];
    uint64_t b[EAGLE_BACKEND_EVENT_COMPONENTS];
    uint64_t c[EAGLE_BACKEND_EVENT_COMPONENTS];
    uint64_t terminated;     /**< uint8 */
    uint64_t applied;        /**< int64 */
    uint64_t t0, h;
    uint64_t status, which;  /**< int8 */
    uint64_t tau, lo, hi, t_stop;
    uint64_t stream;
};

/** Enqueue one per-sample event operation (capturable). Experimental. */
eagle_backend_status eagle_backend_x_event_columns(struct eagle_backend_event_desc* desc);

eagle_backend_status eagle_backend_x_composer_create(const struct eagle_backend_composer_desc* desc, eagle_backend_composer* out);
/** CONSUMES ``launcher`` (on OK). ``name`` / ``pre`` may be NULL. */
eagle_backend_status eagle_backend_x_composer_register_launcher(eagle_backend_composer h, eagle_backend_launcher launcher,
    const char* name, eagle_backend_void_fn pre, void* pre_ctx, int64_t* index);
eagle_backend_status eagle_backend_x_composer_register_callable(eagle_backend_composer h, eagle_backend_step_fn step,
    void* step_ctx, const char* name, eagle_backend_void_fn pre, void* pre_ctx, int64_t* index);
/** ``nested`` is SHARED (not consumed); the caller keeps its handle. */
eagle_backend_status eagle_backend_x_composer_register_nested(eagle_backend_composer h, eagle_backend_composer nested,
    const char* name, eagle_backend_void_fn pre, void* pre_ctx, int64_t* index);
eagle_backend_status eagle_backend_x_composer_build(eagle_backend_composer h);
eagle_backend_status eagle_backend_x_composer_set_routing(eagle_backend_composer h, const uint8_t* pattern, int64_t n);
eagle_backend_status eagle_backend_x_composer_launch(eagle_backend_composer h, int64_t n);
/** CSR: ``*offsets`` has ``calls + 1`` entries; both backend-allocated. */
eagle_backend_status eagle_backend_x_composer_fired_history(eagle_backend_composer h, int64_t** flat, int64_t** offsets,
    int64_t* calls);
eagle_backend_status eagle_backend_x_composer_reset_fired_history(eagle_backend_composer h);
eagle_backend_status eagle_backend_x_composer_mode(eagle_backend_composer h, const char** mode);
eagle_backend_status eagle_backend_x_composer_num_members(eagle_backend_composer h, int64_t* n);
eagle_backend_status eagle_backend_x_composer_member_name(eagle_backend_composer h, int64_t i, const char** name);
void eagle_backend_x_composer_release(eagle_backend_composer h);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* EAGLE_BACKEND_H */
