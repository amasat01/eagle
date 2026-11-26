/* SPDX-License-Identifier: Apache-2.0 */

/**
 * @file eg_unit_meta.h
 * @brief The typed unit meta below ``eg_unit_meta`` of eg_door.h
 *        (``eagle-door/1``, EG_META_SCHEMA_MAJOR 1): the body, launch, finish,
 *        team and group records and every record and closed set they need.
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
#define EG_SCALAR_BANDED 2 /**< banded accumulators: every REAL slot is an 8-byte cell crossing as unsigned 64-bit, made with eg_banded_pack; a host structure refuses it as EG_NOT_WRITTEN */

/** ``eg_unit_meta.device_image_format`` (the values eg_door.h states). */
typedef int32_t eg_image_format;
#define EG_IMAGE_NONE 0   /**< no device image (a unit naming ``device_image`` with NONE is refused, EG_BAD_REQUEST) */
#define EG_IMAGE_PTX 1
#define EG_IMAGE_CUBIN 2
#define EG_IMAGE_FATBIN 3

/** What a slot holds; decides how eagle binds it (contract Role, same order). */
typedef int32_t eg_slot_role;
#define EG_ROLE_MUTABLE 0     /**< a plane read and written in place */
#define EG_ROLE_PLANE_IN 1   /**< a plane read per sample, whatever the rank of its value (the slot's shape says it) */
#define EG_ROLE_LOOKUP 3      /**< a table indexed at run time; its length is the caller's */
#define EG_ROLE_PLANE_OUT 4   /**< a plane the unit writes; runtime length. Without EG_SLOT_ZEROED the caller's plane is any compact plane of one dimension or more and its handle carries the trailing extent; with it the plane is an accumulate target (see EG_SLOT_ZEROED) */
#define EG_ROLE_UNIFORM 5     /**< one value for all samples, by value */
#define EG_ROLE_TERMINATED 8  /**< the mask that marks a sample finished */
#define EG_ROLE_WIDE_IN 9     /**< several values per sample in; runtime length: as a PLANE_OUT slot that is not zeroed, a read clamps to the trailing extent */
#define EG_ROLE_NSAMPLES 10   /**< the sample count, by value; eagle passes it */
#define EG_ROLE_OUT 11        /**< a plain output plane */
#define EG_ROLE_EXTENT 12     /**< a runtime axis's extent, by value (INT64) */

/** The element type ON THE WIRE, at its real width: the width the entry text
 *  binds, not the IR leaf's (what eg_bind checks a plane's dtype against; a
 *  uniform's value is read at it; an index-set map is 32 or 64 bits by it). */
typedef int32_t eg_wire_dtype;
#define EG_DTYPE_REAL 0         /**< the unit's scalar_type: f64, f32, or in a banded unit an 8-byte cell crossing as unsigned 64-bit (eg_banded_pack) */
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

/** ``eg_unit_group.membership``: which lanes make up one group. */
typedef int32_t eg_group_membership;
#define EG_GROUP_COUNT 0 /**< ``count`` consecutive lanes */
#define EG_GROUP_AXES 1  /**< the lanes along the lane axes ``axes`` names */
#define EG_GROUP_KEY 2   /**< the lanes of equal ``key_slot`` value */

/** The SHAPE of a plane eagle makes. */
typedef int32_t eg_plane_layout;
#define EG_LAYOUT_FLAT 0        /**< (per_lane * units,) */
#define EG_LAYOUT_PLANE_MAJOR 1 /**< (per_lane, units), (units,) when per_lane is 1 */

/** Whether a plane holds a placed wide value, and the UNIT COUNT that follows. Value 1 is retired (every scratch
 *  plane a placement makes is tiled by warp) and refused. */
typedef int32_t eg_wide_layout;
#define EG_WIDE_NONE 0        /**< not a wide value: units from ``unit``; filled with ``fill`` */
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
#define EG_UNIT_CALLER_CARDS (1u << 0) /**< eg_unit_meta.flags: lowered under card documents the caller stated, not the shipped
                                          cards; eagle loads such a unit as any other and only carries the bit */
#define EG_UNIT_CHECKED (1u << 1)      /**< eg_unit_meta.flags: the unit is built with the checks of its generated code and states ``fault_slot`` (a
                                          meta without the bit is unchecked whatever its ``fault_slot`` holds, so a zero-filled member is safe) */

#define EG_SLOT_PRIOR (1u << 0)       /**< the body reads the slot before writing it: a caller must give it an initial value;
                                          eagle validates the bit and does not act on it */
#define EG_SLOT_PARAM_PLANE (1u << 1) /**< a LOOKUP slot that is a parameter read at index 0 only: its gradient is the sum over
                                          samples into that one cell; eg_reverse of such a slot is not written in this build and is
                                          refused by name */
#define EG_SLOT_GATHER (1u << 2)      /**< compact along the lane axes ``gather`` keeps (m = the product of their extents):
                                          a rank-2 PLANE_IN is bound as (rows, cols, m); a LOOKUP, read flat by the kernel's
                                          index over the kept axes, holds product(shape) x m elements */
#define EG_SLOT_SHARED (1u << 3)      /**< a parameter plane with NO lane axis: one value for every lane, bound flat (eg_bind sizes it by its shape, not by lanes) */
#define EG_SLOT_ZEROED (1u << 4)     /**< a PLANE_OUT slot the runtime zeroes before the run and folds nothing else into: an accumulate target, addressed by key, so its handle counts the plane's own elements, whatever the lane count; refused on any other role */

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
    int32_t rank;             /**< the length of shape: the rank of the per-sample component; eg_bind and eg_launch */
    int32_t like_slot;        /**< an adjoint plane sized like this LOOKUP slot (eg_bind shape: the planes bound to the two slots have the same ndim and extents), -1: none */
    int32_t n_gather;         /**< entries of ``gather``: the launch's n_lane_axes for a GATHER slot of a launch with lane axes, else 0 */
    int32_t reserved0;
    const int64_t* shape;     /**< ``rank`` extents, NULL when rank is 0; the plane is (*shape, lanes) unless the role has a runtime length */
    const uint8_t* gather;    /**< with EG_SLOT_GATHER: 1 for each lane axis the compact plane varies along (m, the count eg_bind checks); required, with ``n_gather`` equal to ``n_lane_axes``, on a unit whose launch has lane axes */
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
 *  into the cell between launches (eg_bind makes the rows; eg_launch folds). A launch
 *  writes the cell as the reduction over the lanes it RAN: the rows those lanes lie in
 *  (an index or prefix set: its positions below the live count) fold, and over no lane
 *  the cell takes the op's identity (sum 0, min +inf, max -inf, ALL true), never what it
 *  held. Partitions that leave a gap between them are folded over their span. */
struct eg_unit_cell {
    uint32_t struct_size;
    eg_reduce_op op;          /**< the fold (eg_door.h's reduce op); another op is refused NOT_WRITTEN */
    uint32_t internal;        /**< 1: only a later stage reads it and the caller never declared it: eagle's own, not handed back */
    int32_t columns;          /**< components of the cell, >= 1 */
    const int32_t* slots;     /**< ``columns`` row-plane slots in THIS unit, NULL: this unit binds none (a source unit's internal cell) */
    const char* name;         /**< joins eg_unit_stage cells_in / cells_out; reasons */
    const int32_t* comp_slots; /**< ``columns`` row-plane slots of a SUM cell's compensation in THIS unit, NULL: none. A MAX or MIN cell names here the rows
                                    *   of the lane that won each row (a real holding a whole number), and then folds pairs (value, lane): the larger (smaller)
                                    *   value wins, equal values go to the lower lane, a NaN never wins, and no winner has the lane ``nSamples``. */
    uint32_t position;        /**< 1: the cell takes the lane of the winner instead of its value (needs ``comp_slots``; MAX and MIN only) */
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

/** One dispatch group of the unit, as the runner's entry choice reads it.
 *
 *  PREDICATED and SWITCH are plain kernels. AUTO runs fused: the kernel counts its branch mix into ``mix_slot`` (k uint32), which eagle makes
 *  zeroed when the caller leaves the slot out.
 *
 *  SEGMENTED is carried by the group's units, each with its own record of the group (same ``name`` and ``k``; a load that finds the names or the
 *  ks of the units disagree is refused BAD_REQUEST by name):
 *    - each branch unit ``<name>__k<j>``: ``branches`` (the k refs in order), ``offsets_slot``; a dynamic selector also ``class_slot``,
 *      ``stale_slot`` and ``sweep`` (classify NULL), a static one ``classify`` (class_slot and stale_slot -1, sweep NULL);
 *    - the classify unit ``<name>__cls`` (static selector): ``class_slot`` is its class plane (a per-sample INT64 MUTABLE or OUT slot), every
 *      other slot -1, ``classify``, ``branches`` and ``sweep`` NULL;
 *    - the sweep unit (dynamic selector): an ordinary SWITCH record with ``k``; its launch has ``active_map_slot`` and ``active_count_slot``.
 *  A group is planned from a branch unit loaded with the group's other units in deps (a branch may name itself). The plan's slots are that
 *  unit's, then each other unit's slot no earlier slot names, in order (branches, then the classify or sweep unit): the group's caller slots
 *  are joined by name, and the planes the records name are eagle's.
 *  One launch of the plan is one step: the classify unit runs (static selector), the samples are sorted stably by (live, class) (live: the
 *  TERMINATED slot of the branch unit reads zero; no such slot: every sample is live), branch j runs over the first run-length-of-class-j threads
 *  and reads its run from the offsets, and a dynamic selector's sweep runs over the stale LIVE samples (the active set eagle fills each step); the
 *  sort is redone when more stale samples than 5% of the live ones remain. The caller's planes are in sample order before and after the launch. */
struct eg_unit_dispatch {
    uint32_t struct_size;
    eg_dispatch_strategy strategy;
    int32_t mix_slot;         /**< AUTO: the uint32 branch-mix counters eagle makes (unless the caller binds some) and the kernel fills, else -1 */
    int32_t class_slot;       /**< a classify unit: its per-sample class plane; a dynamic branch unit: the class plane eagle makes; else -1 */
    int32_t stale_slot;       /**< SEGMENTED with a dynamic selector: per-sample stale mask eagle makes, else -1 */
    int32_t offsets_slot;     /**< a branch unit: the K+1 run offsets eagle makes, else -1 */
    const struct eg_unit_ref* classify; /**< a static-selector branch unit: the classify unit (passed in deps), NULL: none */
    const char* name;         /**< the group's name: hawk's resolve step and reasons */
    int32_t k;                /**< the group's branches: the mix is k uint32, the offsets k+1 int64 */
    int32_t reserved0;
    const struct eg_unit_ref* const* branches; /**< a branch unit: k refs (passed in deps); branches[j] runs over the run of class j, the
                                                    selector value j; else NULL */
    const struct eg_unit_ref* sweep; /**< a dynamic-selector branch unit: the fused unit run over an active set of the stale LIVE samples (passed
                                          in deps; the set eagle fills each step); else NULL. A branch record is static (stale_slot -1,
                                          classify set, sweep NULL) or dynamic (stale_slot and sweep set, classify NULL); any other mix is refused */
    const struct eg_branch_cost* const* branch_costs; /**< AUTO: k entries, branches[j]'s cost; NULL: not stated (AUTO runs as built) */
    uint32_t cost_flags;      /**< EG_DISPATCH_FUSED_VECTORISED, EG_DISPATCH_VECTOR_STATED; 0: none stated */
};

#define EG_BRANCH_VECTORISED (1u << 0)   /**< eg_branch_cost.flags: this branch's host loop vectorises */
#define EG_BRANCH_HAS_STORES (1u << 1)   /**< eg_branch_cost.flags: ``stores`` is stated */
#define EG_BRANCH_VECTOR_STATED (1u << 2) /**< eg_branch_cost.flags: EG_BRANCH_VECTORISED is a verdict, not unknown */
#define EG_DISPATCH_FUSED_VECTORISED (1u << 0) /**< eg_unit_dispatch.cost_flags: the fused host loop vectorises */
#define EG_DISPATCH_VECTOR_STATED (1u << 1)    /**< eg_unit_dispatch.cost_flags: EG_DISPATCH_FUSED_VECTORISED is a verdict, not unknown */

/** One branch's operation counts per sample, as the library counts them: the inputs of the run-time dispatch
 *  decision. The counts are per step of the branch; the weights that price them are not kernel facts and are not here. */
struct eg_branch_cost {
    uint32_t struct_size;
    uint32_t flags;           /**< EG_BRANCH_* */
    double arith32, arith64;  /**< arithmetic operations at 32 and 64 bits */
    double special32, special64; /**< special functions (exp, sin, ...) at 32 and 64 bits */
    double loads, gathers;    /**< plane reads at the lane index, and reads at a computed index */
    double chain32, chain64;  /**< the longest dependent arithmetic chain at 32 and 64 bits */
    double chain_special32, chain_special64; /**< the special functions on that chain */
    double stores;            /**< plane writes; read only with EG_BRANCH_HAS_STORES */
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

/** Named axes of a plane (LOOKUP, WIDE_IN, PLANE_IN or PLANE_OUT): the plane
 * binds as (*slot shape, *axes), last axis fastest. */
struct eg_unit_plane_axes {
    uint32_t struct_size;
    int32_t slot;
    int32_t n_axes;
    int32_t reserved0;
    const char* const* labels;   /**< n_axes labels: a lane axis's label shares its extent; another is a free extent the caller states */
    const int64_t* extents;      /**< n_axes extents, -1: runtime */
    const int32_t*
        extent_slots; /**< n_axes slots, NULL: none; for a runtime axis the
                         EG_ROLE_EXTENT slot eagle fills with that axis's extent
                         at eg_bind, -1 for a static axis or a runtime axis that
                         shares a lane axis's extent */
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
    int32_t steps_slot;       /**< steps AUTO: the one-cell int64 LOOKUP word the runner sets per launch (eagle makes it unless the caller binds a plane there), else -1 */
    int32_t reserved0;
    int64_t steps;            /**< steps one launch takes, >= 1; EG_STEPS_AUTO; 0 (absent) is refused, EG_BAD_REQUEST */
    int64_t steps_max;        /**< AUTO: the most steps one launch takes, >= 1; else 0 */
    int64_t step_ops;         /**< one step's operation count (cadence, chunk length), -1: not counted */
    int64_t fp64_ops_ir;      /**< one step's 64-bit float operations counted in the IR, -1: not counted */
    const struct eg_unit_steps_table* host_steps_reverse;   /**< NULL: no rows, AUTO picks PER_STEP */
    const struct eg_unit_steps_table* device_steps_reverse; /**< NULL: no rows, AUTO picks PER_STEP */
    int32_t n_events;         /**< the comparisons of the stop rule (its `or` leaves, in operand order), counted by the
                                   library; 0: none. Event i of a GUARD link is the i-th of them */
    int32_t reserved1;
    const char* device_persist_function; /**< AUTO: the persistent entry in the device image, NULL: not stated */
    const char* device_range_function;   /**< AUTO: the contiguous-range entry in the device image, NULL: not stated */
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

/** Where a group's scratch plane lives (``eg_unit_group_scratch.realisation``). */
typedef int32_t eg_scratch_realisation;
#define EG_SCRATCH_LOCAL 0 /**< an array of the unit's own text, on the stack of the thread that runs the group */
#define EG_SCRATCH_PLANE 1 /**< a plane eagle makes at ``eg_plan_new``: ``slot`` names the LOOKUP slot the entry reads it through */

/** A scratch plane of a group: memory local to one group, which only the lanes
 * of that group read and write and the caller never sees. A group of
 * ``max_size`` lanes holds ``max_size`` values of ``shape`` in each scratch
 * plane. */
struct eg_unit_group_scratch {
    uint32_t struct_size;
    eg_wire_dtype dtype; /**< REAL, INT64, INT32 or BOOL: the type of one scalar
                            of a value (a unit's scratch is never a BAND_PARTIAL
                            or UINT32 plane) */
    int32_t rank;        /**< the length of ``shape``: 0 for a scalar value */
    eg_scratch_realisation
        realisation; /**< where the plane lives: LOCAL (the unit's own text
                        holds it) or PLANE (eagle makes it at ``eg_plan_new``).
                        A zero-filled member is LOCAL */
    const int64_t* shape; /**< ``rank`` extents of one value, each >= 1, NULL
                             when rank is 0 */
    int64_t elements; /**< scalars per group: the group's size bound times the
                         product of ``shape``, >= 1 */
    int64_t bytes;    /**< bytes per group: ``elements`` times the width of
                         ``dtype`` (REAL is the unit's ``scalar_type``: 4 bytes for
                         FLOAT32, else 8; INT64 8; INT32 4; BOOL 1) */
    const char* name; /**< the declared name; reasons quote it */
    int32_t slot; /**< PLANE: the unit's LOOKUP slot of this plane's element
                     type that eagle makes, else -1. The plane holds one slice
                     per group of the launch, ``elements`` scalars each; the
                     slice of group ``g`` starts ``g * elements`` scalars in
                     and holds component ``k`` of the values from ``k *
                     max_size`` on (one array per component, as a LOCAL plane
                     is held) */
    int32_t reserved1;
};

/** The group of a unit (``eg_unit_meta.group``, NULL: the unit has none): a
 * group is a set of lanes that run their body in phases, a phase being the code
 * between two barriers, and that read and write the scratch planes of their
 * group. The record says WHAT a group is and what one costs; how many threads
 * serve it and where its scratch lives are the backend's choice at launch and
 * are not stated. The lane axes ``axes`` indexes are the unit's launch lane
 * axes (``eg_unit_launch.lane_labels`` and ``lane_extents``). A unit that
 * states a group is a plain unit: it states no stages, derived role, gemm,
 * finish, dispatch group, active or index set or thread box. */
struct eg_unit_group {
    uint32_t struct_size;
    eg_group_membership membership;
    int32_t phases; /**< the phases of the body, >= 1 */
    int32_t n_axes; /**< AXES: the lane axes that span a group, >= 1, else 0 */
    int32_t key_slot; /**< KEY: the slot of the per-lane key (a rank-0 INT32 or
                         INT64 PLANE_IN slot of this unit), else -1 */
    int32_t n_scratch;
    int64_t count; /**< COUNT: the lanes of a group, >= 1, else 0 */
    int64_t
        max_size; /**< the size bound of a group in lanes: ``count`` for COUNT;
                     for AXES the product of the extents of the spanned axes, -1
                     when one is a run-time extent; for KEY the stated bound,
                     -1: not stated. A scratch plane needs a bound */
    int64_t
        scratch_bytes;   /**< the bytes of all scratch planes of one group: the
                            sum of the planes' ``bytes``, 0 without scratch */
    const int32_t* axes; /**< AXES: ``n_axes`` distinct indices into the lane
                            axes, in the order they span the group, else NULL */
    const struct eg_unit_group_scratch* const*
        scratch; /**< ``n_scratch`` planes with distinct names, NULL when 0 */
    const int32_t* distinct_slots; /**< the slots of the planes the unit's host
                         text reaches through restrict pointers, so it assumes
                         them distinct memory: ``n_distinct`` ascending slot
                         indices of PLANE_IN or written (MUTABLE, PLANE_OUT,
                         OUT) plane slots, NULL when 0. ``eg_bind`` refuses a
                         call that binds overlapping memory to two of them when
                         one is written. Absent (an older record): none */
    int32_t n_distinct;            /**< the length of ``distinct_slots`` */
    int32_t reserved1;
};

/* ---- derived units ------------------------------------------------------- */
/* Unreleased: until 0.8.0 pre these records change in place (no append rule yet). */

/** What quantity a link refers to; ``of_slot`` per value. */
typedef int32_t eg_link_what;
#define EG_WHAT_PRIMAL_SLOT 0 /**< primal slot ``of_slot`` (element type: that slot's) */
#define EG_WHAT_TIME0 1       /**< the start copy of the primal time plane ``of_slot`` (the time plane's real type) */
#define EG_WHAT_STEPS 2       /**< a step plane: ``of_slot`` = the taping twin's ``steps_slot``, or -1 for one eagle makes;
                                   on a refine unit, the stop step's index max(applied steps - 1, 0) (an integer type) */
#define EG_WHAT_TAU 3         /**< the candidate or located time inside the stop step, in [0, h]; ``of_slot`` -1 (the time plane's real type) */
#define EG_WHAT_TIME_STEP 4   /**< the length h of the ACCEPTED step, never a proposed next one, from primal slot ``of_slot`` (a UNIFORM,
                                   which eagle fills into the plane, or a PLANE_IN plane); always bound per sample (the time plane's
                                   real type). A refine kernel with no step slot is refused at build time */
#define EG_WHAT_GUARD 5       /**< guard ``of_slot``: the of_slot-th comparison of the primal's stop rule in operand order,
                                   below the primal's finish.n_events; zero at the crossing (the scalar type) */
#define EG_WHAT_TAPE_ROW 6    /**< the tape rows of primal state ``of_slot``: the pre-step state, or the state itself when it is
                                   not taped; the reverse unit reads the same slot either way; on FUSED_VJP, the state at the
                                   window's start (that slot's type) */
#define EG_WHAT_REFINED 7     /**< primal state ``of_slot`` at the candidate time (that slot's type) */

/** In which form the unit holds it. */
typedef int32_t eg_link_form;
#define EG_FORM_VALUE 0       /**< the quantity itself, read or (an output slot) written; a PLANE_IN slot of a primal
                                   UNIFORM is a plane eagle fills with that value; on FUSED_VJP, the primal's TERMINATED
                                   slot is the window's entry flag, a plane eagle makes all zero */
#define EG_FORM_DOT 1         /**< its derivative in the candidate time */
#define EG_FORM_ADJOINT_IN 2  /**< the cotangent arriving at it, filled by eagle (rules below) */
#define EG_FORM_ADJOINT_OUT 3 /**< the unit's output cotangent of it */
#define EG_FORM_GRAD 4        /**< an accumulated gradient; ``reduced`` 1: summed over samples (rules below) */
#define EG_FORM_SEED 5        /**< a plane eagle fills with ones (a tangent seed) */
#define EG_FORM_DISCARD 6     /**< written by the unit, read by nobody */

/**
 * The one table of valid (role, what, form) links: ``X(role, what, form)``, bare
 * names, one row per line. A triple not listed is refused BAD_REQUEST by eagle
 * and by the library alike; both read these rows. Fill and sum rules eagle owns:
 *  - (GUARD, ADJOINT_IN): 1 for the event that stopped the sample, else 0, on
 *    STATE_GUARD_VJP; -(xbar . refined DOT) / guard DOT for the stopping event,
 *    else 0, on REFINE_VJP (xbar: the end-state cotangent).
 *  - (REFINED, ADJOINT_IN) on REFINE_VJP: xbar for a located sample, else 0.
 *  - (PRIMAL_SLOT, ADJOINT_OUT) on REFINE_VJP: the pre-stop cotangent of a
 *    located sample; the step reverse then runs one step fewer for it.
 *  - (TIME0, GRAD): added per sample into the time plane's gradient.
 *  - (.., GRAD) with ``reduced`` 1: the compensated pair on STEP_VJP and
 *    FUSED_VJP (eg_reverse); one canonical sum, added to the step reverse's
 *    sum, on REFINE_VJP.
 * Cardinality: exactly one TIME_STEP link on REFINE_NEWTON; at most one
 * (TIME0, VALUE) per unit, on a twin only on ``time0_slot``; ``reduced`` is 0
 * unless the form is GRAD; a slot the unit's own meta names as eagle's (the
 * finish counter, the active map or count) takes no link.
 */
#define EG_LINK_RULES(X) \
    X(EG_DERIVED_TAPE, EG_WHAT_PRIMAL_SLOT, EG_FORM_VALUE) \
    X(EG_DERIVED_TAPE, EG_WHAT_TIME0, EG_FORM_VALUE) \
    X(EG_DERIVED_TAPE_FUSED, EG_WHAT_PRIMAL_SLOT, EG_FORM_VALUE) \
    X(EG_DERIVED_TAPE_FUSED, EG_WHAT_TIME0, EG_FORM_VALUE) \
    X(EG_DERIVED_STEP_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_VALUE) \
    X(EG_DERIVED_STEP_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_ADJOINT_IN) \
    X(EG_DERIVED_STEP_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_ADJOINT_OUT) \
    X(EG_DERIVED_STEP_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_GRAD) \
    X(EG_DERIVED_STEP_VJP, EG_WHAT_TAPE_ROW, EG_FORM_VALUE) \
    X(EG_DERIVED_STEP_VJP, EG_WHAT_STEPS, EG_FORM_VALUE) \
    X(EG_DERIVED_STEP_VJP, EG_WHAT_TIME0, EG_FORM_GRAD) \
    X(EG_DERIVED_FUSED_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_VALUE) \
    X(EG_DERIVED_FUSED_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_ADJOINT_IN) \
    X(EG_DERIVED_FUSED_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_ADJOINT_OUT) \
    X(EG_DERIVED_FUSED_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_GRAD) \
    X(EG_DERIVED_FUSED_VJP, EG_WHAT_TAPE_ROW, EG_FORM_VALUE) \
    X(EG_DERIVED_FUSED_VJP, EG_WHAT_STEPS, EG_FORM_VALUE) \
    X(EG_DERIVED_FUSED_VJP, EG_WHAT_TIME0, EG_FORM_GRAD) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_PRIMAL_SLOT, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_TAPE_ROW, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_STEPS, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_TIME0, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_TIME_STEP, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_TAU, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_TAU, EG_FORM_SEED) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_GUARD, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_GUARD, EG_FORM_DOT) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_REFINED, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_NEWTON, EG_WHAT_REFINED, EG_FORM_DOT) \
    X(EG_DERIVED_STATE_GUARD_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_VALUE) \
    X(EG_DERIVED_STATE_GUARD_VJP, EG_WHAT_REFINED, EG_FORM_VALUE) \
    X(EG_DERIVED_STATE_GUARD_VJP, EG_WHAT_GUARD, EG_FORM_ADJOINT_IN) \
    X(EG_DERIVED_STATE_GUARD_VJP, EG_WHAT_REFINED, EG_FORM_ADJOINT_OUT) \
    X(EG_DERIVED_STATE_GUARD_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_GRAD) \
    X(EG_DERIVED_STATE_GUARD_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_DISCARD) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_TAPE_ROW, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_STEPS, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_TIME0, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_TIME0, EG_FORM_GRAD) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_TIME_STEP, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_TIME_STEP, EG_FORM_GRAD) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_TAU, EG_FORM_VALUE) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_TAU, EG_FORM_DISCARD) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_GUARD, EG_FORM_ADJOINT_IN) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_REFINED, EG_FORM_ADJOINT_IN) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_ADJOINT_OUT) \
    X(EG_DERIVED_REFINE_VJP, EG_WHAT_PRIMAL_SLOT, EG_FORM_GRAD)

/** One taped plane: rows [row0, row0 + rows) of the tape hold primal slot ``slot``. */
struct eg_unit_taped {
    uint32_t struct_size;
    int32_t slot, row0, rows;
};

/** One slot of a derived unit: what it refers to and in which form. */
struct eg_unit_link {
    uint32_t struct_size;
    int32_t slot;             /**< in THIS unit */
    eg_link_what what;
    int32_t of_slot;          /**< see eg_link_what */
    uint32_t reduced;         /**< GRAD only: 1 when summed over samples (eg_reverse); 0 for a per-sample gradient */
    eg_link_form form;
};

/** What a derived unit is to its primal (eg_unit_meta.derived). In 0.8.0 the
 *  caller states these facts and the library range-checks them; eagle checks
 *  the role against the runner's need, every index, ``of_unit`` and every link
 *  against EG_LINK_RULES. Slots are in THIS unit unless stated; -1: absent. A
 *  taping twin builds its record window into its code: the tape slot's declared
 *  rows are window * tape_rows, and eg_runner_new refuses another ``window``
 *  (0 included) BAD_REQUEST; a recorded refine run takes window 1 only. */
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
    int32_t fused_steps_slot; /**< the int64 word of steps a fused launch fuses, set by the runner, on any derived unit
                                   that reads one (the K-step twin; the K-step reverse unit if it reads one); a LOOKUP
                                   int64 with no link. A launch takes at most the twin's finish.steps_max when its
                                   finish.steps is AUTO, else finish.steps. A FUSED_VJP compiles K in: steps_reverse
                                   FUSED needs the primal's finish.steps equal to the record window (K) */
    int32_t n_links;
    int32_t reserved0;
    const struct eg_unit_taped* const* taped;
    const struct eg_unit_link* const* links;     /**< every slot of a reverse or refine unit; on a taping twin each ordinary
                                                      slot, none on its other reserved slots */
};

/* ---- gemm units ---------------------------------------------------------- */

/**
 * A unit that is one dense contraction states ``eg_unit_meta.gemm``: the matrix products the library lowered it to.
 * The library states them; eagle lowers nothing. A unit with the member still carries its full kernel
 * (``host_entry``) and its device function, because a HOST plan of it runs that kernel's own loop, bit for bit as
 * without the member, and so does a DEVICE plan the door cannot route. A DEVICE plan the door can route (one
 * partition; not split, segmented, with cells, companions, an index set or an active set) runs the stated products
 * through cuBLAS on the context's stream INSTEAD of the kernel, in a recording or not, and sets EG_ENTRY_GEMM in
 * ``eg_plan_entry_figures``; a DEVICE runner launches the kernel. cuBLAS is opened at run time (libcublas.so.13, then
 * .12), never linked; a routable DEVICE plan whose cuBLAS cannot be opened is EG_FAULT naming the files tried, never
 * a silent fall back to the kernel. Both widths run (``scalar_type`` float32: S, float64: D); a banded unit (EG_SCALAR_BANDED) with the
 * member is refused EG_NOT_WRITTEN at ``eg_unit_load``, and the library never states one.
 *
 * Every plane is one dense row-major matrix view of its slot: rows as stated, row stride = the plane's elements / rows
 * (so a plane wider than the stated columns serves a column slice), columns as stated. alpha is 1, beta 0 (``C =
 * product``) or 1 (EG_GEMM_ACCUMULATE: ``C += product``), batch 1.
 *
 * Refusals, all by reason text. ``eg_unit_load``: a slot out of range or of a role that takes no plane, ``c_slot`` not
 * a written role, a slot whose dtype is not the unit's scalar (EG_BAD_REQUEST); an unknown flag bit (EG_NOT_WRITTEN);
 * ``n_calls`` < 1, an extent < 1 that is not EG_GEMM_PER_LANE, EG_GEMM_PER_LANE with ``lane_fanin`` 0 (EG_BAD_REQUEST).
 * ``eg_plan_new``: the contracted extents disagree, C's stored shape is not the product's (or its transpose), an
 * extent that does not resolve to 1 or more or ``n_samples`` not divisible by ``lane_fanin``, an extent or stride
 * beyond 2^31 - 1 (EG_BAD_REQUEST); cuBLAS missing on a routable DEVICE plan (EG_FAULT).
 * ``eg_bind``: a plane whose element count is not a multiple of the rows, or whose row stride is smaller than the
 * stated columns, or beyond 2^31 - 1 (EG_BAD_REQUEST).
 *
 * The cuBLAS handle of the context's stream is made at ``eg_plan_new`` of the first DEVICE plan of a gemm unit and
 * again at ``eg_stream_use``; ``eg_launch`` never makes one, so a gemm is legal in a recording.
 */

/** An extent that is ``n_samples / eg_unit_gemm.lane_fanin``, resolved at ``eg_plan_new``. */
#define EG_GEMM_PER_LANE (-1)


/** One matrix product ``C = op(A) * op(B)`` over three slots of the unit; op(A) is (m, k), op(B) is (k, n). */
struct eg_unit_gemm_call {
    uint32_t struct_size;
    uint32_t flags;                /**< EG_GEMM_*; other bits are refused */
    int32_t a_slot, b_slot, c_slot; /**< slots of ``eg_unit_body.slots``; ``c_slot`` is a written role */
    int32_t reserved0;
    int64_t a_rows, a_cols, b_rows, b_cols, c_rows, c_cols; /**< the STORED shapes: each >= 1 or EG_GEMM_PER_LANE */
};

/** The record of ``eg_unit_meta.gemm`` (NULL there: none): a forward unit states 1 call, an adjoint unit 2. */
struct eg_unit_gemm {
    uint32_t struct_size;
    int32_t n_calls;                              /**< >= 1 */
    int64_t lane_fanin;                           /**< the other operand's free extent: a PER_LANE extent is n_samples / it; 0: none */
    const struct eg_unit_gemm_call* const* calls; /**< n_calls records, run in order */
};

#ifdef __cplusplus
}
#endif

#endif /* EG_UNIT_META_H */
