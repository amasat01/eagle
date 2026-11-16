/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file eg_unit_meta.h
 * @brief The typed unit meta below ``eg_unit_meta`` of eg_door.h
 *        (``eagle-door/1``, EG_META_SCHEMA_MAJOR 1): the body, launch, finish
 *        and team records and every record and closed set they need.
 *
 * RULES (every struct below follows them, on top of eg_door.h's meta rules).
 *  - Constant data only. Every record is ``const``, reached by a pointer or a
 *    pointer array with a count; no function pointers, no code runs at load.
 *    The only relocations are pointers to other constant data of the same
 *    library (names, arrays, records).
 *  - Each record starts with ``uint32_t struct_size``. Eagle reads the members
 *    that fit; a member past ``struct_size`` is absent and reads as its
 *    documented absent value (-1 for a slot or table index, 0 for a count,
 *    NULL for a pointer). Within major 1 members are appended at the end only.
 *    The absent value serves version skew only (a meta older than eagle). A
 *    generated meta writes every member, and a hand-written one must too: a C
 *    initializer that omits a member leaves 0, which names slot 0, not -1.
 *  - Closed sets are ``int32_t`` with the values defined here. A value or a
 *    flag bit eagle does not know is refused by ``eg_unit_load`` with
 *    EG_NOT_WRITTEN naming the unit and the member; it is never ignored.
 *  - Lists are (pointer, count). Records travel as pointer arrays; pairs and
 *    plain numbers travel as parallel arrays of the same count.
 *  - A slot is named by its INDEX in its unit's ``eg_unit_body.slots`` (the
 *    index ``eg_bind`` takes). A name is kept only where a person reads it in
 *    a reason, where hawk resolves a declaration, or where two units meet:
 *    across the units of one kernel (a source unit and its stage units) a
 *    team plane, a keyed plane or a cell is joined by NAME, as eg_door.h's
 *    ``eg_unit_stage`` cells are. ``eg_unit_load`` resolves every such name
 *    ONCE into index maps (source slot -> stage slot, per stage); no name is
 *    looked up after the load call returns.
 *  - Device image: ``eg_unit_meta.device_image`` names the image's BYTES
 *    symbol; its length is read from the SIZE symbol the meta names (a
 *    ``const uint64_t`` named by ``eg_unit_meta.device_image_size``), so the
 *    meta is generated before the compile. ``device_image_bytes`` is written 0
 *    by the generator. The compile step must export both symbols with C linkage
 *    and default visibility from the unit library. ``device_image_size`` is NULL
 *    when ``device_image`` is NULL.
 *  - EG_SCALAR_BANDED on a host structure is refused by ``eg_plan_new`` with
 *    EG_NOT_WRITTEN.
 *  - Names are NUL-terminated ASCII. Flags are 0 / 1. Absence is NULL, -1 or a
 *    count of 0 as each member states; never a text.
 *  - Contradictions inside the tree (an index out of range, a slot of the
 *    wrong role or dtype for its use, a count that disagrees) are refused by
 *    ``eg_unit_load`` with EG_BAD_REQUEST ("meta that contradicts itself").
 *  - The same tree serves the host and the device unit: per-target facts
 *    (the thread box, a companion's fold entry) have one member per target;
 *    everything else is shared. No symbol or function name is derived from
 *    another by a rule: each is a member.
 */
#ifndef EG_UNIT_META_H
#define EG_UNIT_META_H

#include "eg_door.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ---- closed sets --------------------------------------------------------- */

/** ``eg_unit_meta.pattern``: how eagle loads and launches a unit. */
typedef int32_t eg_unit_pattern;
#define EG_PATTERN_PURE 0 /**< the serial v2 entry over samples (the one pattern written) */

/** ``eg_unit_meta.scalar_type``: what Real is on the wire. eg_door.h states
 *  0 and 1; BANDED is appended (additive within major 1). */
typedef int32_t eg_scalar_type;
#define EG_SCALAR_FLOAT64 0
#define EG_SCALAR_FLOAT32 1
#define EG_SCALAR_BANDED 2 /**< banded accumulators; a host structure refuses it as EG_NOT_WRITTEN */

/** ``eg_unit_meta.device_image_format`` (the values eg_door.h states). */
typedef int32_t eg_image_format;
#define EG_IMAGE_NONE 0   /**< no device image (a unit naming ``device_image`` with NONE is refused, EG_BAD_REQUEST) */
#define EG_IMAGE_PTX 1
#define EG_IMAGE_CUBIN 2
#define EG_IMAGE_FATBIN 3

/** What a slot holds; decides how eagle binds it (contract Role, same order). */
typedef int32_t eg_slot_role;
#define EG_ROLE_MUTABLE 0     /**< a plane read and written in place */
#define EG_ROLE_PER_SAMPLE 1  /**< one value per sample */
#define EG_ROLE_VEC_IN 2      /**< a vector read in one piece */
#define EG_ROLE_LOOKUP 3      /**< a table indexed at run time; its length is the caller's */
#define EG_ROLE_WIDE_OUT 4    /**< several values per sample out; runtime length */
#define EG_ROLE_UNIFORM 5     /**< one value for all samples, by value */
#define EG_ROLE_MAT_IN 6      /**< a matrix read in one piece */
#define EG_ROLE_ACCUM_OUT 7   /**< accumulated over samples; runtime length */
#define EG_ROLE_TERMINATED 8  /**< the mask that marks a sample finished */
#define EG_ROLE_WIDE_IN 9     /**< several values per sample in; runtime length */
#define EG_ROLE_NSAMPLES 10   /**< the sample count, by value; eagle passes it */
#define EG_ROLE_OUT 11        /**< a plain output plane */

/** The element type ON THE WIRE, at its real width: the width the entry text
 *  binds, not the IR leaf's (what eg_bind checks a plane's dtype against; a
 *  uniform's value is read at it; an index-set map is 32 or 64 bits by it). */
typedef int32_t eg_wire_dtype;
#define EG_DTYPE_REAL 0         /**< the unit's scalar_type (an f64 or f32 leaf) */
#define EG_DTYPE_INT64 1        /**< signed 64-bit, and nothing else */
#define EG_DTYPE_INT32 2        /**< signed 32-bit, and nothing else */
#define EG_DTYPE_UINT32 3       /**< unsigned 32-bit (the finished counter, a mix counter) */
#define EG_DTYPE_BOOL 4
#define EG_DTYPE_BAND_PARTIAL 5 /**< 16 bytes: a partial sum of a BANDED unit's team plane */

/** How a unit's loads and stores relate samples (eg_plan_new's structure check). */
typedef int32_t eg_access;
#define EG_ACCESS_SAMPLE_LOCAL 0
#define EG_ACCESS_CROSS_SAMPLE_READ 1
#define EG_ACCESS_CROSS_SAMPLE_WRITE 2

/** What one block of a team stage's launch stands for (contract Grid, same order). */
typedef int32_t eg_team_grid;
#define EG_GRID_LANES 0   /**< blocks = ceil(n / (lanes * warps)) */
#define EG_GRID_KEYS 1    /**< blocks = keys of ``keyed`` * groups */
#define EG_GRID_CHUNKS 2  /**< blocks = lane chunks of an ordered scatter into ``of_slot`` */
#define EG_GRID_TARGETS 3 /**< blocks = the cells of ``of_slot`` */

/** What one ``per_lane`` slice of a team plane stands for. */
typedef int32_t eg_plane_unit;
#define EG_UNIT_LANE 0   /**< one slice per lane */
#define EG_UNIT_KEY 1    /**< one per key of ``keyed`` */
#define EG_UNIT_TARGET 2 /**< chunks * cells of ``target_slot`` (an ordered scatter's partial sums) */
#define EG_UNIT_ONCE 3   /**< one slice for the whole launch */

/** The SHAPE of a plane eagle makes. */
typedef int32_t eg_plane_layout;
#define EG_LAYOUT_FLAT 0        /**< (per_lane * units,) */
#define EG_LAYOUT_PLANE_MAJOR 1 /**< (per_lane, units), (units,) when per_lane is 1 */

/** Whether a plane holds a placed wide value, and the UNIT COUNT that follows. */
typedef int32_t eg_wide_layout;
#define EG_WIDE_NONE 0        /**< not a wide value: units from ``unit``; filled with ``fill`` */
#define EG_WIDE_PLAIN 1       /**< a wide value: units from ``unit``; written before read, never filled */
#define EG_WIDE_WARP_TILED 2  /**< a wide value tiled by warp: device units rounded up to whole warps of 32, host one unit; never filled */

/** A dispatch group's strategy. */
typedef int32_t eg_dispatch_strategy;
#define EG_DISPATCH_PREDICATED 0
#define EG_DISPATCH_SWITCH 1
#define EG_DISPATCH_SEGMENTED 2 /**< the class, stale and offsets slots are eagle's */
#define EG_DISPATCH_AUTO 3      /**< decided at run time from the mix slot */

/** The batch a steps_reverse row is measured in. */
typedef int32_t eg_steps_regime;
#define EG_REGIME_LANES 0        /**< plain sample count n (host rows) */
#define EG_REGIME_PER_RESIDENT 1 /**< n / (SMs * resident threads per SM) (device rows) */

/* ---- flags --------------------------------------------------------------- */

#define EG_SLOT_PRIOR (1u << 0)       /**< the body reads the launch-start value: eagle keeps it between launches and records it */
#define EG_SLOT_PARAM_PLANE (1u << 1) /**< a parameter read at index 0 only: eg_reverse reduces its gradient over samples */
#define EG_SLOT_GATHER (1u << 2)      /**< a matrix read only at lane indices: its column count is not the lane count */
#define EG_SLOT_SHARED (1u << 3)      /**< a parameter plane with NO lane axis: one value for every lane, bound flat (eg_bind sizes it by its shape, not by lanes) */

#define EG_FINISH_ENTRY_COUNTS (1u << 0) /**< the fast entries count samples finished on entry: the runner zeroes the counter, it does not count the mask */

/* ---- records ------------------------------------------------------------- */

/** One argument slot of a unit's entry, at its position in the argument list.
 *  For a split kernel's source unit, the slots are what the caller binds over
 *  all stages; eagle passes each to the stage slot of the same name. */
struct eg_unit_slot {
    uint32_t struct_size;
    uint32_t flags;           /**< EG_SLOT_*; read by eg_bind, eg_runner_new (PRIOR), eg_reverse (PARAM_PLANE) */
    eg_slot_role role;        /**< read by eg_bind (what may be passed, written, made) and eg_launch (packing) */
    eg_wire_dtype dtype;      /**< read by eg_bind (dtype check; a uniform's value width) and eg_launch (Real vs Int handles) */
    int32_t rank;             /**< 0 scalar, 1 vector, 2 matrix: the per-sample component; eg_bind and eg_launch */
    int32_t like_slot;        /**< an adjoint plane sized like this LOOKUP slot (eg_bind shape), -1: none */
    int32_t n_gather;         /**< entries of ``gather``: 0, or the launch's n_lane_axes */
    int32_t reserved0;
    const int64_t* shape;     /**< ``rank`` extents, NULL when rank is 0; the plane is (*shape, lanes) unless the role has a runtime length */
    const uint8_t* gather;    /**< with EG_SLOT_GATHER: 1 for each lane axis the compact matrix varies along (eg_bind's column count); NULL: unchecked */
    const char* name;         /**< the declared name: hawk resolves names to slots; eagle's reasons quote it; joins a stage slot */
};

/** A companion plane of a compensated keyed sum: eagle makes it zeroed at
 *  eg_bind (the caller never passes it) and folds it into its target. */
struct eg_unit_companion {
    uint32_t struct_size;
    int32_t slot;             /**< the companion's slot (eagle-made) */
    int32_t target_slot;      /**< the slot it compensates */
    int32_t reserved0;
    const char* host_fold_entry;      /**< host symbol of the entry that folds it into the target (eg_launch after the unit), NULL: none */
    const char* device_fold_function; /**< the fold function's name inside the unit's device image, NULL: none */
};

/** A cell: a reduction over samples kept in rows of ``cell_row`` samples, folded
 *  into the cell between launches (eg_bind makes the rows; eg_launch folds). */
struct eg_unit_cell {
    uint32_t struct_size;
    eg_reduce_op op;          /**< the fold (eg_door.h's reduce op); another op is refused NOT_WRITTEN */
    uint32_t internal;        /**< 1: only a later stage reads it and the caller never declared it: eagle's own, not handed back */
    int32_t columns;          /**< components of the cell, >= 1 */
    const int32_t* slots;     /**< ``columns`` row-plane slots in THIS unit, NULL: this unit binds none (a source unit's internal cell) */
    const char* name;         /**< joins eg_unit_stage cells_in / cells_out; reasons */
    const int32_t* comp_slots; /**< ``columns`` row-plane slots of a SUM cell's compensation in THIS unit, NULL: none (refused on MAX / MIN) */
};

/** A plane keyed by kept lane axes, owned by eagle between stages (eg_bind sizes
 *  it width * product of the axes; a runtime axis is as long as the first lane axis). */
struct eg_unit_keyed {
    uint32_t struct_size;
    int32_t n_axes;
    int32_t payload_rank;
    int32_t reserved0;
    int64_t width;                   /**< entries of one key's payload */
    const char* const* axis_labels;  /**< n_axes labels, an entry NULL: unlabelled */
    const int64_t* axis_extents;     /**< n_axes extents, -1: runtime */
    const int64_t* payload;          /**< payload_rank extents of one key's payload, NULL when 0 */
    const char* name;                /**< joins the stage slot of that name; reasons */
};

/** A scratch plane eagle makes once per kernel and lane count (eg_bind) and
 *  every stage unit binds by name. */
struct eg_unit_team_plane {
    uint32_t struct_size;
    eg_wire_dtype dtype;      /**< REAL, INT32 or BAND_PARTIAL */
    eg_plane_unit unit;       /**< what a slice stands for */
    eg_plane_layout layout;   /**< the plane's own shape */
    eg_wide_layout wide;      /**< wide placement: the unit count and whether it is filled */
    int32_t keyed;            /**< unit KEY: index in this body's keyed table, else -1 */
    int32_t target_slot;      /**< unit TARGET: the slot whose cells the slices stand for, else -1 */
    int32_t reserved0;
    int64_t per_lane;         /**< elements of one slice */
    int64_t fill;             /**< value at the start of every run; read only when ``fill_promised`` is 1 and ``wide`` is NONE */
    const char* name;         /**< joins the stage slot of that name; reasons (the scratch byte refusal) */
    int32_t target_stage;     /**< unit TARGET: index in the kernel's stages of the unit whose body holds ``target_slot``; -1: this unit */
    int32_t fill_promised;    /**< 1: ``fill`` is promised at the start of every run; 0: no fill is promised and ``fill`` is not read */
};

/** One dispatch group of the unit, as the runner's entry choice reads it. */
struct eg_unit_dispatch {
    uint32_t struct_size;
    eg_dispatch_strategy strategy;
    int32_t mix_slot;         /**< AUTO: the uint32 branch-mix counters eagle makes and reads, else -1 */
    int32_t class_slot;       /**< SEGMENTED: per-sample class plane eagle makes, else -1 */
    int32_t stale_slot;       /**< SEGMENTED with a dynamic selector: per-sample stale mask eagle makes, else -1 */
    int32_t offsets_slot;     /**< SEGMENTED: the K+1 run offsets eagle makes, else -1 */
    const struct eg_unit_ref* classify; /**< SEGMENTED, static selector: the classify unit (passed in deps), NULL: none */
    const char* name;         /**< the group's name: hawk's resolve step and reasons */
};

/** The typed walk of a unit: its slots and the planes eagle makes for it. */
struct eg_unit_body {
    uint32_t struct_size;
    eg_access access;         /**< eg_plan_new: structure and partition checks */
    int32_t n_slots;
    int32_t n_companions;
    int32_t n_cells;
    int32_t n_keyed;
    int32_t n_team_planes;
    int32_t n_dispatches;
    int64_t cell_row;         /**< samples per row of a cell (eg_bind; one run has one row width), 0 when n_cells is 0 */
    int64_t scatter_chunks;   /**< the most chunks of an ordered scatter (team grid CHUNKS, plane unit TARGET), 0: none */
    const struct eg_unit_slot* const* slots;             /**< in argument order: the index IS the slot (eg_bind_desc) */
    const struct eg_unit_companion* const* companions;
    const struct eg_unit_cell* const* cells;
    const struct eg_unit_keyed* const* keyed;            /**< the one keyed table; team records and planes index it */
    const struct eg_unit_team_plane* const* team_planes;
    const struct eg_unit_dispatch* const* dispatches;
};

/** Named axes of a sink: the plane binds as (*slot shape, *axes). */
struct eg_unit_plane_axes {
    uint32_t struct_size;
    int32_t slot;
    int32_t n_axes;
    int32_t reserved0;
    const char* const* labels;   /**< n_axes labels: a lane axis's label shares its extent; another is a free extent the caller states */
    const int64_t* extents;      /**< n_axes extents, -1: runtime */
};

/** A thread box: each thread serves ``t`` samples ``stride`` apart; a launch of
 *  n lanes runs ceil(n / (t * stride)) * stride threads, and base / count count threads. */
struct eg_unit_box {
    uint32_t struct_size;
    int32_t axis;             /**< the sample axis the box runs along, >= 0 */
    int64_t t;                /**< samples per thread, >= 2 */
    int64_t stride;           /**< >= 1 */
};

/** The launch shape of a unit. */
struct eg_unit_launch {
    uint32_t struct_size;
    int32_t n_lane_axes;              /**< 0: one unlabelled runtime axis */
    int32_t block;                    /**< device threads per block a launch must use (emit-fixed tile or launch bounds), 0: eagle's choice */
    int32_t active_map_slot;          /**< the guard is an active set: the map eagle compacts (makes), -1: none */
    int32_t active_count_slot;        /**< its live-lane count (eagle-made), -1: none */
    int32_t index_map_slot;           /**< the caller's index set map (eg_bind range- and duplicate-checks it; width = its dtype), -1: none */
    int32_t index_count_slot;         /**< the caller's prefix count; set without a map slot: a prefix set, -1: none */
    int32_t n_plane_axes;
    int64_t group;                    /**< lanes per compaction group (eg_runner_new), >= 1 */
    const char* const* lane_labels;   /**< n_lane_axes labels, an entry NULL: unlabelled; hawk maps label=extent to n_samples */
    const int64_t* lane_extents;      /**< n_lane_axes extents, -1: runtime (the first axis only) */
    const struct eg_unit_plane_axes* const* plane_axes; /**< eg_bind's shape check of named-axis sinks */
    const struct eg_unit_box* host_box;   /**< the host entry's box, NULL: none */
    const struct eg_unit_box* device_box; /**< the device function's box, NULL: none */
};

/** The measured rows of steps_reverse AUTO: a row is (state width, K, narrowest
 *  regime, widest regime); the nearest row by log2 distance picks FUSED when the
 *  run's regime lies inside it. One table per target serves every unit (the
 *  generated source points every unit at the same object). */
struct eg_unit_steps_table {
    uint32_t struct_size;
    eg_steps_regime regime;
    int32_t n_rows;
    int32_t reserved0;
    const int64_t* widths;
    const int64_t* ks;
    const double* lo;
    const double* hi;
};

/** ``eg_unit_finish.steps``: the runner picks the steps of each launch, up to ``steps_max``. */
#define EG_STEPS_AUTO (-1)

/** How a finishing unit stops a sample and steps (eg_runner_new, eg_run). */
struct eg_unit_finish {
    uint32_t struct_size;
    uint32_t flags;           /**< EG_FINISH_* */
    int32_t mask_slot;        /**< the TERMINATED slot it marks (the plan carrying the finish) */
    int32_t counter_slot;     /**< the uint32 LOOKUP cell it counts newly finished samples into (eagle-made) */
    int32_t steps_slot;       /**< steps AUTO: the LOOKUP word the runner sets per launch (eagle-made), else -1 */
    int32_t reserved0;
    int64_t steps;            /**< steps one launch takes, >= 1; EG_STEPS_AUTO; 0 (absent) is refused, EG_BAD_REQUEST */
    int64_t steps_max;        /**< AUTO: the most steps one launch takes, >= 1; else 0 */
    int64_t step_ops;         /**< one step's operation count (cadence, chunk length), -1: not counted */
    int64_t fp64_ops_ir;      /**< one step's 64-bit float operations counted in the IR, -1: not counted */
    const struct eg_unit_steps_table* host_steps_reverse;   /**< NULL: no rows, AUTO picks PER_STEP */
    const struct eg_unit_steps_table* device_steps_reverse; /**< NULL: no rows, AUTO picks PER_STEP */
};

/** The team a stage runs on (eg_unit_stage.team). Slot indices are into the
 *  SOURCE unit's body (the unit whose meta lists the stage) unless a ``*_stage``
 *  member names a later stage's unit. */
struct eg_unit_team {
    uint32_t struct_size;
    eg_team_grid grid;
    int32_t threads;          /**< threads of one team; a block runs threads * warps */
    int32_t warps;            /**< warps of one team of a warp per lane, else 1 */
    int32_t lanes;            /**< lanes one block serves, else 1 */
    int32_t groups;           /**< KEYS: blocks per key, else 1 */
    int32_t keyed;            /**< KEYS: index in the keyed table, else -1 */
    int32_t of_slot;          /**< CHUNKS, TARGETS: the slot the scatter adds into, else -1 */
    int32_t status_plane;     /**< CHUNKS: index in team_planes of the out-of-range flags eagle checks after a run, else -1 */
    int32_t of_stage;         /**< CHUNKS, TARGETS: index in the kernel's stages of the unit whose body holds ``of_slot``; -1: the source unit */
};

/* ---- derived units ------------------------------------------------------- */

/** Where a slot of a reverse unit comes from. ``of_slot`` is a PRIMAL slot index
 *  unless stated otherwise. */
typedef int32_t eg_derived_source;
#define EG_SRC_PRIMAL 0            /**< the primal's slot ``of_slot``, unchanged */
#define EG_SRC_STATE_ADJOINT_IN 1  /**< the adjoint of state ``of_slot``, read at this step's parity */
#define EG_SRC_STATE_ADJOINT_OUT 2 /**< the next-step adjoint of ``of_slot``, written to the other parity */
#define EG_SRC_PARAM_GRAD 3        /**< the accumulated gradient of parameter or uniform ``of_slot`` */
#define EG_SRC_TAPED 4             /**< the tape rows of ``of_slot`` (the state itself when it is not taped) */
#define EG_SRC_STEPS 5             /**< a step plane: ``of_slot`` = the taping twin's ``steps_slot``, or -1 for one eagle makes only because this unit reads it */
#define EG_SRC_EVENT 6             /**< an event multiplier of a refine run: ``of_slot`` = the event's index in the unit's event list */

/** One taped plane: rows [row0, row0 + rows) of the tape hold primal slot ``slot``. */
struct eg_unit_taped {
    uint32_t struct_size;
    int32_t slot, row0, rows;
};

/** One slot of a reverse unit and its source. */
struct eg_unit_link {
    uint32_t struct_size;
    int32_t slot;             /**< in THIS unit */
    eg_derived_source source;
    int32_t of_slot;          /**< see eg_derived_source */
    uint32_t reduced;         /**< PARAM_GRAD only: 1 when the per-sample term is summed over samples
                                   (eg_reverse); 0 for a per-sample parameter's gradient */
};

/** Events refine: the refined plane of a primal state. */
struct eg_unit_refined {
    uint32_t struct_size;
    int32_t state_slot;       /**< in the primal */
    int32_t refined_slot;     /**< in THIS unit */
};

/** What a derived unit is to its primal (eg_unit_meta.derived). In 0.8.0 the
 *  caller states these facts and the library range-checks them; eagle checks
 *  the role against the runner's need, every index and ``of_unit``. Slots are
 *  in THIS unit unless stated; -1: absent. */
struct eg_unit_derived {
    uint32_t struct_size;
    eg_derived_role role;     /**< EG_DERIVED_* (eg_door.h) */
    int32_t of_unit;          /**< the primal's index in the library table (a taping twin shares its name) */
    int32_t tape_slot;
    int32_t n_taped;
    int32_t tape_rows;        /**< rows per recorded step: the sum of the taped rows */
    int32_t steps_slot;       /**< applied steps, int64 per sample */
    int32_t tape_on_slot;
    int32_t time0_slot;       /**< the time plane's start copy */
    int32_t tau_slot;         /**< events refine: the per-sample step */
    int32_t fused_steps_slot; /**< a fused unit: the int64 word of steps this launch fuses, set by the runner */
    int32_t n_links;
    int32_t n_refined;
    int32_t reserved0;
    const struct eg_unit_taped* const* taped;
    const struct eg_unit_link* const* links;     /**< one per slot of a reverse unit; NULL on a taping twin */
    const struct eg_unit_refined* const* refined;
};

#ifdef __cplusplus
}
#endif

#endif /* EG_UNIT_META_H */
