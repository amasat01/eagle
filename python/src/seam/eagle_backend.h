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
 * | 5   | attribution | capture_snapshot_nodes, is_node_toggleable             |
 * | 6   | graph       | graph_create, graph_from_captured, graph_stream, graph_add_node, graph_launcher, graph_last_node, graph_release |
 * | 7   | launcher    | launcher_launch, launcher_synchronize, launcher_stream, launcher_set_logical_size, launcher_kernel_node_count, launcher_set_node_enabled, launcher_release |
 * | 8   | (reserved)  | the composer group, when it is promoted from ``x_``     |
 * | 9   | exec        | run_device                                            |
 * | 10  | device      | device_props                                          |
 * | 11  | interop     | fence, event_pool_created                             |
 * | 12  | filtering   | compact_device (+ reorder_device, bound per symbol)   |
 * | 13..62 | reserved | new families only (additions to a family are per-symbol) |
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
#define EAGLE_BACKEND_VERSION_MINOR 0u
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
    char archs[128];          /**< device code carried, e.g. "61-real,90-virtual" */
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

/* ---- conditional (bit 4) ----------------------------------------------------- */

eagle_backend_status eagle_backend_conditional_create(const struct eagle_backend_conditional_desc* desc, eagle_backend_conditional* out);
eagle_backend_status eagle_backend_conditional_begin(eagle_backend_conditional h, uint64_t* body_stream);
eagle_backend_status eagle_backend_conditional_body_stream(eagle_backend_conditional h, uint64_t* body_stream);
eagle_backend_status eagle_backend_conditional_is_loop(eagle_backend_conditional h, int32_t* is_loop);
eagle_backend_status eagle_backend_conditional_end(eagle_backend_conditional h);
void eagle_backend_conditional_release(eagle_backend_conditional h);

/* ---- attribution (bit 5) ----------------------------------------------------- */

/** The node handles of the graph ``stream`` is capturing into; ``*nodes`` is
 *  backend-allocated (``eagle_backend_free``). */
eagle_backend_status eagle_backend_capture_snapshot_nodes(uint64_t stream, uint64_t** nodes, int64_t* count);
eagle_backend_status eagle_backend_is_node_toggleable(uint64_t node, int32_t* toggleable);

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

/* ---- experimental (eagle_backend_x_*, optional, unlocked) -------------------- */

/** Write the captured graph as Graphviz dot to ``path`` (the caller reads it). */
eagle_backend_status eagle_backend_x_captured_debug_dot(eagle_backend_captured h, const char* path, uint32_t flags);
eagle_backend_status eagle_backend_x_capture_guard_depth(int64_t* depth);
eagle_backend_status eagle_backend_x_capture_guard_pending_count(int64_t* count);
/** Undo every reorder since the last restore: each plane back in sample order,
 *  ``perm`` and ``inv`` the identity (``mask``, ``index_map``, ``count``,
 *  ``span`` and ``fire`` are not read). Enqueue only; eager use. */
eagle_backend_status eagle_backend_x_restore_device(struct eagle_backend_reorder_desc* desc);

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
