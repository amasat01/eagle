// SPDX-License-Identifier: Apache-2.0

// The door's internal state: the context and the handles made from it.
//
// Every exported function of eg_door.h is a shim in abi.cpp that enters the
// context through `call` below. The shim refuses a second thread, a forked
// child and a context refused at creation BEFORE the body runs; the body
// refuses by throwing `Refusal`, and `call` turns every exception into a
// status plus the reason text in the context. No exception leaves `call`.
#pragma once

#include "eagle/door/eg_door.h"
#include "eagle/exec/Partition.h"
#include "plugin/dlpack.h"
#include "plugin/gref_layout.h"

struct eg_unit_meta;

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <unistd.h>

namespace eagle {
namespace door {

/** @brief A refusal of the request: the status kind and the reason text. */
struct Refusal : std::runtime_error {
    eg_status status;
    Refusal(eg_status s, const std::string& reason) : std::runtime_error(reason), status(s) {}
};

/** @brief Refuse with @p status and @p reason (never returns). */
[[noreturn]] inline void refuse(eg_status status, const std::string& reason) {
    throw Refusal(status, reason);
}

/**
 * @brief Read a versioned INPUT struct: zero-fill, copy the bytes that fit.
 *
 * A struct whose ``struct_size`` cannot hold ``struct_size`` and ``flags`` is
 * refused; the rest of a short struct reads as zero.
 */
template <class T>
T read_input(const T* in, const char* what) {
    if (in == nullptr) refuse(EG_BAD_REQUEST, std::string(what) + " is NULL");
    if (in->struct_size < 2 * sizeof(std::uint32_t))
        refuse(EG_BAD_REQUEST, std::string(what) + ": struct_size " +
               std::to_string(in->struct_size) + " is too small");
    T out{};
    const std::size_t n = in->struct_size < sizeof(T) ? in->struct_size : sizeof(T);
    std::memcpy(static_cast<void*>(&out), static_cast<const void*>(in), n);
    return out;
}

/**
 * @brief Write @p value into the OUTPUT struct @p out: the bytes its ``struct_size`` covers, that size written back.
 */
template <class T>
void write_output(T* out, T value, const char* what) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, std::string(what) + " is NULL");
    if (out->struct_size < sizeof(std::uint32_t))
        refuse(EG_BAD_REQUEST, std::string(what) + ": struct_size " + std::to_string(out->struct_size) + " is too small");
    const std::uint32_t n = std::min<std::uint32_t>(out->struct_size, sizeof(T));
    value.struct_size = n;
    std::memcpy(static_cast<void*>(out), static_cast<const void*>(&value), n);
}

/** @brief One argument slot of a loaded unit, as eg_bind and eg_launch read it. */
struct SlotFacts {
    std::int32_t role = 0;
    std::int32_t dtype = 0;
    std::string name;
    std::int32_t rank = 0;
    std::vector<std::int64_t> shape;          ///< ``rank`` extents; the plane is (*shape, n)
    bool mirror = false;                      ///< passed as the 40-byte GRefMirror, else the 32-byte handle
    bool gathered = false;                    ///< a compact (rows, cols, m) matrix read at lane indices: m is not the lane count
    std::vector<std::uint8_t> keep;           ///< gathered: 1 for each lane axis the matrix varies along (empty: m unchecked)
    bool shared = false;                      ///< a LOOKUP with no lane axis: one value for every lane, bound flat
    /** @brief One named axis of an ACCUM_OUT plane: a static extent, or -1 and the lane axis whose extent it shares. */
    struct Axis { std::int64_t extent = -1; int lane = -1; };
    std::vector<Axis> axes;                   ///< ACCUM_OUT: the plane's named axes (empty: any compact plane)
    bool made = false;                        ///< eagle's own plane (a companion): never bound by the caller
    bool cell = false;                        ///< a cell's slot: the caller binds the finished (1,) cell, the kernel reads eagle's rows
    std::int32_t keys_of = -1;                ///< a WIDE_OUT slot a KEYS stage writes (a block sum's output): the keyed record whose keys it holds, else -1
    std::int32_t like = -1;                   ///< a plane sized like the bound plane of this LOOKUP slot (its ndim and extents), else -1
    bool optional_made = false;               ///< eagle makes the plane unless the caller binds one (the AUTO mix counters)
    std::int64_t min_elements = 0;            ///< a caller's plane of an optional_made slot holds at least this many elements
};

/** @brief A compensation companion eagle makes beside the target the caller binds, and the fold it runs after each launch. */
struct CompanionFacts {
    std::int32_t slot = -1, target = -1;
    eagle::exec::HostEntryV2 fold = nullptr;  ///< the host fold entry, nullptr: the unit's library has none
    std::string device_fold_name;             ///< the fold function's name in the device image, empty: none
    std::uint64_t device_fold = 0;            ///< that function (a CUfunction) once the image is loaded, 0: none
};

/** @brief One component of a cell: its row slot and the fold (SUM is refused: the meta states no compensation slot). */
struct CellFacts {
    std::int32_t slot = -1;
    eg_reduce_op op = EG_REDUCE_SUM;
    std::int32_t comp = -1;                   ///< SUM: the row slot of its compensation, else -1
    std::string name;                         ///< the cell's name (a column's slot is named <cell>_<k> when the cell has several)
};

/** @brief One cell column of a split unit's stages: the rows its stage writes and where they fold (a source slot of the caller's, or a
 *  cell plane eagle makes and later stages read by this name). */
struct SplitCell {
    std::int32_t stage = 0, row_slot = -1, comp_slot = -1, source = -1;
    eg_reduce_op op = EG_REDUCE_SUM;
    std::string name;
};

/** @brief One measured row of steps_reverse AUTO: the narrowest and widest regime where FUSED won at (width, k). */
struct StepsRow {
    std::int64_t width = 0, k = 0;
    double lo = 0.0, hi = 0.0;
};

/** @brief A target's measured rows of steps_reverse AUTO, copied from eg_unit_steps_table; no rows: AUTO picks PER_STEP. */
struct StepsTable {
    bool per_resident = false;       ///< the regime is n / (SMs * resident threads per SM), else the sample count n
    std::vector<StepsRow> rows;
};

/** @brief A unit's finish record, as eg_runner_new and eg_run read it. */
struct FinishFacts {
    bool present = false;
    std::int32_t mask_slot = -1;     ///< the TERMINATED slot it marks
    std::int32_t counter_slot = -1;  ///< the uint32 LOOKUP cell eagle makes
    std::int64_t steps = 0;          ///< steps one launch takes, >= 1; EG_STEPS_AUTO
    std::int32_t steps_slot = -1;    ///< AUTO: the int64 LOOKUP word the runner sets per launch
    std::int64_t steps_max = 0;      ///< AUTO: the most steps one launch takes
    std::uint32_t flags = 0;         ///< EG_FINISH_*
    std::int64_t ops = 0;            ///< one step's operation count the launch rules read (fp64_ops_ir, else step_ops), 0: none
    std::int32_t n_events = 0;       ///< the stop rule's comparisons (a GUARD link's event is below it)
    StepsTable host_reverse, device_reverse;   ///< the measured rows of steps_reverse AUTO, per target
};

/** @brief A derived unit's record (eg_unit_derived), copied at eg_unit_load; eg_runner_new checks it
 *  against the runner's primal. */
struct DerivedFacts {
    struct Taped { std::int32_t slot = -1, row0 = 0, rows = 0; };
    struct Link { std::int32_t slot = -1, what = -1, of_slot = -1; std::uint32_t reduced = 0; std::int32_t form = -1; };
    bool present = false;
    std::int32_t role = -1, of_unit = -1, tape_slot = -1, tape_rows = 0, steps_slot = -1, tape_on_slot = -1;
    std::int32_t time0_slot = -1, fused_steps_slot = -1;
    std::vector<Taped> taped;
    std::vector<Link> links;
};
/** @brief A team plane of a split unit (eg_unit_team_plane), copied at eg_unit_load. */
struct TeamPlaneFacts {
    std::int32_t dtype = 0, unit = 0, layout = 0, wide = 0;
    std::int64_t per_lane = 1, fill = 0;
    bool fill_promised = false;
    std::string name;
    std::int32_t keyed = -1;                  ///< unit KEY: the record in the unit's keyed table, else -1
    std::int32_t target = -1;                 ///< unit TARGET: the source slot whose cells the slices stand for, else -1
};

/** @brief A record of the keyed table (eg_unit_keyed): ``width`` payload entries over the kept lane axes. */
struct KeyedFacts {
    std::int64_t width = 1;
    std::vector<std::int64_t> extents;        ///< per kept axis, -1: runtime
    std::string name;
};

/** @brief One stage of a split unit: its unit, its team (LANES grid only on the host) and the join of its slots, made once at load. */
struct StageFacts {
    eg_unit_s* unit = nullptr;                ///< the stage's unit; a reference held unless ``self``
    bool self = false;                        ///< the stage is the source unit itself
    bool held = false;                        ///< a reference to ``unit`` is held (a dep)
    std::string ref_name;                     ///< the stage reference, resolved against deps at load
    std::array<std::uint8_t, EG_DIGEST_BYTES> ref_digest{};
    bool team = false;
    std::int32_t grid = 0, threads = 1, warps = 1, lanes = 1, groups = 1;
    std::int32_t keyed = -1;                  ///< grid KEYS: the record in the source unit's keyed table, else -1
    std::int32_t of_slot = -1;                ///< grids CHUNKS, TARGETS: the source slot the ordered scatter adds into, else -1
    std::int32_t status_plane = -1;           ///< grid CHUNKS: the team plane of out-of-range flags checked after a launch, else -1
    std::vector<std::int32_t> source_of;      ///< per stage slot: the source slot it joins, else -1
    std::vector<std::int32_t> plane_of;       ///< per stage slot: the team plane it joins, else -1 (both -1: the sample count)
    std::vector<std::int32_t> row_of, comp_of, read_of;   ///< per stage slot: the split cell whose rows / compensation rows it is, or whose plane it reads
    std::int64_t cell_row = 0;                ///< samples per row of the stage's cells
    std::vector<std::string> cells_in, cells_out;
    /** @brief A companion of the stage's unit, folded after the stage runs: its index in that unit's companions, and the plane it folds
     *  into (its target's source slot, or its team plane). */
    struct Fold { std::size_t companion = 0; std::int32_t source = -1, plane = -1; };
    std::vector<Fold> folds;
};

/** @brief A unit reference of a dispatch record, resolved against deps by unit_load. */
struct RefFacts {
    std::string name;
    std::array<std::uint8_t, EG_DIGEST_BYTES> digest{};
};

/** @brief A dispatch group of a unit (eg_unit_dispatch), copied at eg_unit_load. */
struct DispatchFacts {
    std::int32_t strategy = 0, k = 2, mix = -1, cls = -1, stale = -1, offsets = -1;
    std::string name;
    bool classify_record = false;             ///< SEGMENTED: a classify unit's own record (``cls`` is its class slot)
    bool branch_record = false;               ///< SEGMENTED: a branch unit's record (it names the group's units)
    std::vector<RefFacts> branches;
    RefFacts classify, sweep;
    bool has_classify = false, has_sweep = false;
};

/** @brief The segmented group a branch unit loaded with its group's units in deps runs: the units, and per unit the join of its
 *  slots to the plan's (the branch unit's slots, then the slots of the other units no earlier slot names). */
struct SegFacts {
    static constexpr std::int32_t kMap = -2, kCount = -3;   ///< a sweep's active map and count: eagle's own, joined to no plan slot
    bool present = false;
    bool dynamic = false;                     ///< a stale slot and a sweep, no classify unit
    std::int32_t k = 0, class_slot = -1, stale_slot = -1, offsets_slot = -1, mask_slot = -1;   ///< plan slots; mask: the TERMINATED slot, -1: every sample is live
    std::string name;
    std::vector<eg_unit_s*> branches;         ///< k units in class order; a reference may be the unit itself
    eg_unit_s* classify = nullptr;
    eg_unit_s* sweep = nullptr;
    std::vector<eg_unit_s*> held;             ///< the units a reference is held on (deps)
    std::vector<std::vector<std::int32_t>> join;   ///< per member (branches, then classify or sweep), per its slot: the plan slot, or kMap / kCount
    std::int32_t access = 0;                  ///< the strictest access of the units launched over the same partitions
};

/** @brief The element count of @p t (the product of its shape). */
inline std::int64_t elements(const DLTensor& t) {
    std::int64_t n = 1;
    for (int i = 0; i < t.ndim; ++i) n *= t.shape[i];
    return n;
}

/** @brief True when @p t is one compact row-major range (no strides, or the compact ones). */
inline bool compact(const DLTensor& t) {
    if (t.strides == nullptr) return true;
    std::int64_t expect = 1;
    for (int i = t.ndim - 1; i >= 0; --i) {
        if (t.shape[i] > 1 && t.strides[i] != expect) return false;
        expect *= t.shape[i];
    }
    return true;
}

}  // namespace door
}  // namespace eagle

/** @brief One loaded unit: its library, its entry and the slot facts copied from its meta. */
struct eg_unit_s {
    eg_ctx* ctx = nullptr;
    void* library = nullptr;                  ///< dlopen handle, closed when ``refs`` drops to 0
    eagle::exec::HostEntryV2 host_entry = nullptr;
    std::string name;
    std::int32_t scalar_type = 0;
    std::int32_t access = 0;
    std::vector<eagle::door::SlotFacts> slots;
    eagle::door::FinishFacts finish;
    eagle::door::DerivedFacts derived;        ///< a derived unit (eg_unit_meta.derived), else not present
    std::int32_t index = -1;                  ///< the unit index in its library table
    std::array<std::uint8_t, EG_DIGEST_BYTES> digest{};
    std::vector<eagle::door::TeamPlaneFacts> team_planes;
    std::int64_t scatter_chunks = 0;          ///< the most chunks of an ordered scatter (body.scatter_chunks), 0: none
    std::vector<eagle::door::KeyedFacts> keyed;    ///< the keyed table of a split unit's body
    std::vector<std::string> source_cells;    ///< a split unit's cells (no row slots: the stages own the rows)
    std::vector<eagle::door::SplitCell> split_cells;
    std::vector<eagle::door::StageFacts> stages;   ///< non-empty: a split unit, run stage by stage
    std::vector<eagle::door::DispatchFacts> dispatches;   ///< the unit's dispatch groups
    eagle::door::SegFacts seg;                ///< a branch unit loaded with its group's units in deps
    std::vector<eagle::door::CompanionFacts> companions;
    std::vector<eagle::door::CellFacts> cells;
    std::int64_t cell_row = 0;                ///< samples per row of a cell's rows, 0: no cells
    std::int32_t active_map_slot = -1;        ///< the int32 LOOKUP of live sample slots a runner makes, -1: none
    std::int32_t active_count_slot = -1;      ///< the uint32 LOOKUP count of its live prefix
    std::int64_t group = 1;                   ///< lanes per group of the active set: the map holds group indices
    std::vector<std::int64_t> lane_extents;   ///< one per lane axis, -1: runtime (the launch's n over the static ones)
    std::int64_t box_t = 0, box_stride = 1;   ///< a host thread box: each thread serves box_t lanes box_stride apart; 0: none
    std::int32_t index_map_slot = -1;         ///< the caller's int32/int64 LOOKUP of lanes to run (an index set), -1: none
    std::int32_t index_count_slot = -1;       ///< the caller's uint32 LOOKUP count: lanes run (with no map: the prefix)
    std::int32_t block = 0;                   ///< device threads per block the meta states, 0: eagle's choice
    std::int64_t dev_box_t = 0, dev_box_stride = 1;   ///< a device thread box: each thread serves dev_box_t lanes dev_box_stride apart; 0: none
    std::uint64_t module = 0;                 ///< the loaded device image (a CUmodule), 0: none
    std::uint64_t function = 0;               ///< its device function (a CUfunction), 0: none
    int regs = 0, max_threads = 0;            ///< the device function's registers per thread and most threads per block
    std::uint64_t persist_fn = 0, range_fn = 0;   ///< AUTO: the image's ``<function>_persist`` / ``_range`` entries, 0: absent
    int persist_regs = 0, persist_max_threads = 0;
    int local_bytes = 0, shared_bytes = 0, const_bytes = 0;   ///< the device function's local (per thread), static shared and constant memory
    int refs = 1;                             ///< the caller's reference plus one per plan
    bool caller_ref = true;
};

namespace eagle {
namespace door {

/** @brief The keys of keyed record @p k over @p n lanes: the product of the kept axes' extents, a runtime axis as long as the first lane
 *  axis (``n`` over the static extents of the other lane axes). */
inline std::int64_t key_count(const eg_unit_s& unit, const KeyedFacts& k, std::int64_t n) {
    std::int64_t rest = 1;
    for (std::size_t i = 1; i < unit.lane_extents.size(); ++i) rest *= std::max<std::int64_t>(1, unit.lane_extents[i]);
    const std::int64_t runtime = n / rest;
    std::int64_t keys = 1;
    for (std::int64_t e : k.extents) keys *= e < 0 ? runtime : e;
    return keys;
}

/** @brief The chunks an ordered scatter of @p n lanes into @p cells cells adds in: ``clamp(n / cells, 1, cap)``. */
inline std::int64_t chunks_of(std::int64_t n, std::int64_t cells, std::int64_t cap) {
    return std::min(cap, std::max<std::int64_t>(1, n / std::max<std::int64_t>(cells, 1)));
}

}  // namespace door
}  // namespace eagle

/** @brief One DLPack tensor taken by eagle: reference-counted, deleter called once. */
struct eg_plane_s {
    eg_ctx* ctx = nullptr;
    DLManagedTensorVersioned* tensor = nullptr;
    int refs = 1;                             ///< the caller's reference plus one per binding and one per unreleased export
    bool caller_ref = true;
    bool pinned = false;                      ///< eagle-made page-locked host memory: the tensor says kDLCPU, an export kDLCUDAHost
};

namespace eagle {
namespace door {

/** @brief What a SEGMENTED plan keeps between launches: the planes eagle makes for the group and the sort of the last launch. */
struct SegState {
    std::vector<std::int64_t> offsets;        ///< the k+1 run offsets of the live samples
    std::vector<std::int64_t> cls;            ///< the class of the sample at each slot of the arrangement
    std::vector<std::uint8_t> stale;          ///< dynamic: the samples whose branch moved, by slot of the arrangement
    std::vector<std::int32_t> perm;           ///< slot -> sample of the kept sort (empty: the identity)
    std::vector<std::int32_t> map;            ///< dynamic: the stale live slots the sweep runs over
    std::vector<std::int64_t> run_len;        ///< per class: its run in the sort
    std::uint32_t count = 0;                  ///< the live prefix of ``map``
    std::int64_t stale_now = 0;               ///< stale live samples after the last launch
    bool sorted = false;
    plugin::ScalarHandle map_h{}, count_h{};
    std::vector<std::uint8_t> sortkey;        ///< DEVICE dynamic: the class of each sample (sample order) at the last sort
    std::vector<void*> dev_bufs;              ///< DEVICE: the device memory of the sort and the sorted copies, made at the first launch, freed with the plan
};

}  // namespace door
}  // namespace eagle

/** @brief A unit under one structure over fixed partitions, plus its bindings. */
struct eg_plan_s {
    /** @brief One slot's binding: a plane (by handle) or a uniform (by value). */
    struct Binding {
        eg_plane_s* plane = nullptr;
        bool bound = false;
        eagle::plugin::ScalarHandle handle{};
        eagle::plugin::GRefMirror mirror{};
        alignas(8) unsigned char value[16] = {};
    };
    eg_ctx* ctx = nullptr;
    eg_unit_s* unit = nullptr;
    eg_structure structure = EG_STRUCTURE_HOST_TEAM;
    std::int64_t n_samples = 0;
    std::vector<eagle::exec::Partition> parts;
    std::vector<Binding> bindings;            ///< one per slot; never resized after eg_plan_new
    std::vector<void*> params;                ///< the entry's argument list, into ``bindings``
    std::vector<eg_plane_s*> rows;            ///< one eagle-made row plane per cell component (``unit->cells``), one reference each
    std::vector<eg_plane_s*> comps;           ///< the compensation rows of each cell column (nullptr: none), parallel to ``rows``
    std::vector<eg_plane_s*> outs;            ///< a split unit: the cell plane eagle makes for each internal cell column (nullptr: the caller's)
    std::vector<eg_plane_s*> team;            ///< one eagle-made scratch plane per team plane of a split unit, one reference each
    std::vector<eg_plane_s*> work;            ///< DEVICE: the two reduction work buffers of each cell component, one reference each
    std::vector<std::vector<eg_plane_s*>> segteam;   ///< a SEGMENTED plan: the scratch planes of each member that is a split unit other than the planned unit (by member index)
    std::vector<eg_plane_s*> segdev;          ///< a DEVICE SEGMENTED plan: the offsets and class planes eagle makes in device memory
    std::vector<eg_plane_s*> mix;             ///< the eagle-made mix counters of each AUTO group (``unit->dispatches`` order), one reference each
    mutable eagle::door::SegState seg;        ///< a SEGMENTED plan: its planes and sort
    /** @brief DEVICE: the scratch planes of SEGMENTED member @p mi, the unit @p m: the planned unit's are ``team``, another member's ``segteam``. */
    const std::vector<eg_plane_s*>& seg_team(const eg_unit_s& m, std::size_t mi) const { return &m == unit ? team : segteam[mi]; }
    std::uint32_t finished = 0;               ///< the eagle-made finished counter of a finishing unit
    std::int64_t steps_word = 0;              ///< the eagle-made steps word of an AUTO unit
    std::uint64_t device_counter = 0;         ///< DEVICE with a finish: the device cell of the finished counter
    std::uint64_t device_steps = 0;           ///< DEVICE with steps AUTO: the device int64 steps word
    eg_runner_s* active_owner = nullptr;      ///< the runner whose active set the plan reads
    int refs = 1;                             ///< the caller's reference plus one per runner
    bool caller_ref = true;
};

namespace eagle {
namespace door {

/** @brief The active set of a runner whose units read one: the live-slot map and its count,
 *  the compaction counters, and (EG_RUN_REORDER) the permutation of the bound planes. */
struct ActiveState {
    std::int64_t n = 0;
    std::int64_t group = 1;                   ///< lanes per group: a group is live while any of its lanes is
    std::int64_t every_launches = 1;          ///< launches of the step between two compactions
    std::int64_t every_steps = 1;             ///< steps between two compactions
    std::vector<std::int32_t> map;            ///< ascending live slots; never resized (plans point into it)
    std::uint32_t count = 0;                  ///< the live prefix launched
    std::uint32_t done_at_last = 0;           ///< the finished count at the last compaction
    bool reorder = false;                     ///< EG_RUN_REORDER over at least MIN_REORDER_SPAN samples
    double theta = 0.0;
    std::vector<std::int32_t> perm, inv;      ///< slot -> sample and sample -> slot, while reordered
    std::int64_t span = 0;                    ///< slots a reorder may still move
    bool fire = false;                        ///< the last compaction found a reorder worth running
    std::int64_t compactions = 0, reorders = 0, restored_at = 0;
};

}  // namespace door
}  // namespace eagle

namespace eagle {
namespace door {

struct DeviceLoop;
struct DeviceRecording;   ///< a capture in progress on the context's stream (device/Device.cpp)
struct DeviceGraph;       ///< a recorded device graph: the instantiated capture and the plans it launched
struct RunRecord;         ///< an EG_RUN_RECORD runner: its derived plans and last record (host/Reverse.cpp)

/** @brief The device part of a DEVICE runner (device/Device.cpp): its captured loop and the device
 *  cells it reads; freed with the runner (no device call in a forked child). */
struct DeviceRun {
    eg_ctx* ctx = nullptr;                    ///< set: a DEVICE runner
    DeviceLoop* loop = nullptr;               ///< captured at the first eg_run
    std::uint64_t cells = 0;                  ///< device [remaining, ran, total, live count, finished at last compaction, compactions] (uint32)
    std::uint64_t perm = 0, inv = 0, rcells = 0, rscratch = 0;   ///< a reordering set's permutation, [live32, span, fire, reorders, zero] and scratch
    std::size_t rscratch_bytes = 0;           ///< the bytes rscratch holds (a recorded run grows it for its own planes)
    std::uint64_t map = 0, scratch = 0;       ///< an active set's device live-slot map (int32) and its compaction scratch
    double build_s = 0.0;                     ///< the capture's wall
    int mode = EG_MODE_BAND;                  ///< steps AUTO: EG_MODE_FUSED_ONE or EG_MODE_PERSIST (no graph)
    unsigned block = 0, grid = 0, chunk = 0;  ///< the persist launch
    std::uint64_t fast = 0;                   ///< the persist entry's device [next, steps] (uint32)
    std::vector<unsigned char> bound;         ///< the plans' argument bytes the loop was captured over
    DeviceRun() = default;
    DeviceRun(const DeviceRun&) = delete;
    DeviceRun& operator=(const DeviceRun&) = delete;
    ~DeviceRun();
};

}  // namespace door
}  // namespace eagle

/** @brief A run-until-done loop over bound plans (the first carries the finish). */
struct eg_runner_s {
    std::vector<eg_plan_s*> plans;            ///< one reference each
    std::int64_t max_steps = 0;
    int refs = 1;                             ///< the caller's reference plus one per recorded graph
    bool caller_ref = true;
    bool active = false;                      ///< the plans read an active set
    eg_steps_reverse steps_reverse = EG_STEPS_REVERSE_PER_STEP;   ///< a recorded runner's reverse mode, AUTO resolved
    eagle::door::ActiveState act;
    eagle::door::DeviceRun dev;               ///< a DEVICE runner's loop and cells
    eagle::door::RunRecord* record = nullptr; ///< EG_RUN_RECORD: the derived plans and the last record
};

namespace eagle {
namespace door {

/** @brief The device of a device context and the facts a launch reads (device/Device.cpp). */
struct DeviceState {
    std::int32_t ordinal = -1;                ///< -1: a host-only context
    int handle = 0;                           ///< the driver's device (a CUdevice)
    std::uint64_t context = 0;                ///< its primary context (a CUcontext), retained while the context lives
    std::uint64_t stream = 0;                 ///< the stream in scope (a CUstream): eagle's own or the caller's (eg_stream_use)
    std::uint64_t own_stream = 0;             ///< eagle's own stream, destroyed with the context
    int sms = 0, regs_per_sm = 0, smem_per_sm = 0, threads_per_sm = 0, cc_major = 0, cc_minor = 0;
    int fp64_ratio = 1;                       ///< FP32:FP64 throughput ratio (1 when not reported)
    std::unordered_map<void*, std::uint64_t> alloc_stream;   ///< each live device_alloc block -> the stream it was allocated on
};

}  // namespace door
}  // namespace eagle

struct eg_graph_s;      // a recorded list of door calls (host/Record.cpp)
struct eg_recording_s;  // the recording in progress on a context (host/Record.cpp)

/** @brief All door state of one caller. */
struct eg_ctx_s {
    long pid = 0;                             ///< the process that made the context
    std::atomic<bool> busy{ false };          ///< a call is running on the context
    eg_status refused = EG_OK;                ///< a refused eg_ctx_new: every call returns it
    std::int32_t host_threads = 0;            ///< 0: the default budget
    std::string reason;                       ///< the last non-OK status's reason
    bool reason_lost = false;                 ///< the reason could not be stored (out of memory)
    std::unordered_set<eg_unit_s*> units;
    std::unordered_set<eg_plane_s*> planes;
    std::unordered_set<eg_plan_s*> plans;
    std::unordered_set<eg_runner_s*> runners;
    std::unordered_set<eg_graph_s*> graphs;
    int exports = 0;                          ///< the DLPack exports (eg_plane_dlpack) not yet released by their consumers; changed under `busy`
    bool closed = false;                      ///< eg_ctx_free ran with exports outstanding: the last export's release frees the context
    eg_recording_s* rec = nullptr;            ///< between eg_record_begin and eg_record_end
    eagle::door::DeviceState dev;             ///< EG_CTX_DEVICE: the device and its stream
    eagle::door::DeviceRecording* dev_rec = nullptr;              ///< a device context between eg_record_begin and eg_record_end
    std::unordered_set<eagle::door::DeviceGraph*> dev_graphs;     ///< the device graphs eg_record_end made

    /** @brief Store @p text as the reason, never throwing. */
    void set_reason(const char* text) noexcept {
        try {
            reason = text;
            reason_lost = false;
        } catch (...) {
            reason_lost = true;
        }
    }
};

namespace eagle {
namespace door {

/** @brief True in a process other than the one that made @p ctx. */
inline bool forked(const eg_ctx& ctx) { return ctx.pid != static_cast<long>(::getpid()); }

/**
 * @brief Run @p body on @p ctx under the door's rules and return its status.
 *
 * Refused before @p body runs: a NULL context, a second thread (no reason is
 * written: the reason belongs to the thread inside), a forked child unless
 * @p in_child, and a context refused at creation unless @p in_refused.
 */
template <class F>
eg_status call(eg_ctx* ctx, F&& body, bool in_child = false, bool in_refused = false) noexcept {
    if (ctx == nullptr) return EG_BAD_REQUEST;
    bool idle = false;
    if (!ctx->busy.compare_exchange_strong(idle, true, std::memory_order_acquire))
        return EG_BAD_REQUEST;
    eg_status status = EG_OK;
    if (!in_child && forked(*ctx)) {
        status = EG_BAD_REQUEST;
        ctx->set_reason("this context belongs to another process: a forked child may only call "
                        "eg_error and eg_ctx_free");
    } else if (!in_refused && ctx->refused != EG_OK) {
        status = ctx->refused;
    } else if (ctx->closed) {
        status = EG_BAD_REQUEST;
        ctx->set_reason("the context is closed: eg_ctx_free ran while an export was outstanding");
    } else {
        try {
            body(*ctx);
        } catch (const Refusal& r) {
            status = r.status;
            ctx->set_reason(r.what());
        } catch (const std::bad_alloc&) {
            status = EG_FAULT;
            ctx->set_reason("out of memory");
        } catch (const std::exception& e) {
            status = EG_FAULT;
            ctx->set_reason(e.what());
        } catch (...) {
            status = EG_FAULT;
            ctx->set_reason("an unknown failure inside eagle");
        }
    }
    ctx->busy.store(false, std::memory_order_release);
    return status;
}

/** @brief Refuse @p handle unless @p set (a context's registry) holds it. */
template <class H>
void check_handle(const std::unordered_set<H*>& set, const H* handle, const char* what) {
    if (handle == nullptr) refuse(EG_BAD_REQUEST, std::string(what) + " is NULL");
    if (set.count(const_cast<H*>(handle)) == 0)
        refuse(EG_BAD_REQUEST, std::string(what) +
               " is not a live handle of this context (freed, or made by another context)");
}

// Bodies of the exported functions (Context.cpp, Unit.cpp, Plane.cpp, Plan.cpp).
eg_status ctx_new(const eg_ctx_settings* settings, eg_ctx** out) noexcept;
eg_status error_read(eg_ctx* ctx, char* buf, std::size_t cap, std::size_t* len) noexcept;
void ctx_free_all(eg_ctx& ctx);
bool ctx_close(eg_ctx& ctx);
void ctx_info(eg_ctx& ctx, struct eg_ctx_info* out);
void unit_load(eg_ctx& ctx, const eg_unit_load_desc* desc, eg_unit** out);
void unit_release(eg_ctx& ctx, eg_unit_s* unit);
void plane_wrap(eg_ctx& ctx, DLManagedTensorVersioned* tensor, eg_plane** out);
void plane_release(eg_ctx& ctx, eg_plane_s* plane);
eg_plane_s* make_plane(eg_ctx& ctx, const DLDataType& dtype, std::vector<std::int64_t> shape, bool device = false,
                       bool pinned = false, bool zero = true);
void plane_new(eg_ctx& ctx, const eg_plane_desc* desc, eg_plane** out);
void plane_dlpack(eg_ctx& ctx, eg_plane_s* plane, DLManagedTensorVersioned** out);
void plan_new(eg_ctx& ctx, const eg_plan_desc* desc, eg_plan** out);
void plan_free(eg_ctx& ctx, eg_plan_s* plan);
void plan_release(eg_ctx& ctx, eg_plan_s* plan);
void check_bound(const eg_plan_s& plan);
void bind(eg_ctx& ctx, eg_plan_s* plan, const eg_bind_desc* desc);
void clear(eg_ctx& ctx, eg_plan_s* plan, std::int32_t n_slots, const std::int32_t* slots);
void set_count(eg_ctx& ctx, eg_plan_s* plan, std::uint32_t m);
void banded_pack(eg_ctx& ctx, const double* in, std::uint64_t* out, std::uint64_t n);
void banded_unpack(eg_ctx& ctx, const std::uint64_t* in, double* out, std::uint64_t n);
void banded_pack_scalar(eg_ctx& ctx, double x, std::uint64_t* out);
void bind_raw(eg_plan_s& plan, int slot, void* data, std::uint64_t count);
void launch(eg_ctx& ctx, eg_plan_s* plan, std::uint64_t* tiles);
void runner_new(eg_ctx& ctx, const eg_runner_desc* desc, eg_runner** out);
void runner_free(eg_ctx& ctx, eg_runner_s* runner);
void record_new(eg_ctx& ctx, eg_runner_s& runner, const eg_runner_desc& d);
void record_free(eg_ctx& ctx, eg_runner_s& runner);
void record_reset(eg_runner_s& runner);
void runner_reset(eg_ctx& ctx, eg_runner_s* runner);
void active_counters(ActiveState& a);
void active_identity(ActiveState& a);
void record_forward(eg_ctx& ctx, eg_runner_s& runner, eg_run_report* report);
/** @brief eg_run_events (host/Reverse.cpp). */
void run_events(eg_ctx& ctx, eg_runner_s* runner, eg_plane_s* t_stop, eg_plane_s* status);
void reverse(eg_ctx& ctx, eg_runner_s* runner, const eg_bind_desc* seeds, const eg_bind_desc* grads);
void runner_release(eg_ctx& ctx, eg_runner_s* runner);
void run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report);
void run_loop(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report);
void reduce(eg_ctx& ctx, const eg_reduce_desc* desc);
void copy(eg_ctx& ctx, eg_plane_s* src, eg_plane_s* dst);
bool record_launch(eg_ctx& ctx, eg_plan_s* plan);
bool record_run(eg_ctx& ctx, eg_runner_s* runner);
void record_begin(eg_ctx& ctx);
void record_cond_begin(eg_ctx& ctx, const eg_cond_desc* desc);
void record_cond_arm(eg_ctx& ctx);
void record_cond_end(eg_ctx& ctx);
void record_end(eg_ctx& ctx, eg_graph** out);
void graph_launch(eg_ctx& ctx, eg_graph_s* graph);
void graph_free(eg_ctx& ctx, eg_graph_s* graph);
void record_drop_all(eg_ctx& ctx);
bool device_planes(const eg_ctx& ctx, const char* op, std::initializer_list<const eg_plane_s*> planes);
bool device_plans(const eg_ctx& ctx, const eg_runner_desc* desc);
/** @brief The derived units of an AUTO runner that chose PER_STEP: the caller's set without the K-step vjp. */
struct KeptDerived {
    std::vector<eg_derived_role> roles;
    std::vector<eg_unit*> units;
};
void check_runner(const eg_ctx& ctx, eg_runner_desc& d, bool device, KeptDerived& kept);
/** @brief A plane the reorder moves: its first byte and element size. */
struct Owned {
    char* data;
    std::size_t bytes;
};
/** @brief The per-sample planes a reorder of @p runner moves (the mask first, a (D, n) plane as D rows), each once. */
std::vector<Owned> owned_planes(const eg_runner_s& runner);
/** @brief Refuse an index set's live count @p live above the entries its plan holds: the map's, else the sample count's. */
void check_index_live(const eg_plan_s& plan, std::uint64_t live);
/** @brief The fewest samples a reorder is built for and fires over. */
inline constexpr std::int64_t kMinReorderSpan = 4096;
/** @brief Whether the plans of @p d read an active set (all or none must); refuses a request that does not fit one. */
bool check_active(const eg_runner_desc& d);
/** @brief The steps between two compactions of a fixed-step active set (``every``, else the default cadence). */
std::int64_t active_every_steps(const eg_runner_desc& d);
/** @brief Recompute the active map and count of @p a from the mask plane (@p mask, elements of @p bytes): the plain loop's compaction. */
void active_compact(ActiveState& a, const char* mask, std::size_t bytes);
/** @brief Move the live slots of the span to the front, every plane of @p owned (the mask first) and ``perm`` with them. */
void active_reorder(ActiveState& a, const std::vector<Owned>& owned);
/** @brief Put every plane of @p owned back in sample order and ``perm`` / ``inv`` to the identity. */
void active_restore(ActiveState& a, const std::vector<Owned>& owned);
void check_report(const eg_run_report* report);
void write_report(eg_run_report* report, eg_run_report r);

/** @brief eg_runner_figures: fill @p out with the mode and the resolved launch of @p runner. */
void runner_figures(eg_ctx& ctx, const eg_runner_s* runner, struct eg_runner_figures* out);
/** @brief eg_plan_entry_figures: fill @p out with the device entries of the unit of @p plan. */
void plan_entry_figures(eg_ctx& ctx, const eg_plan_s* plan, eg_entry_figures* out);

}  // namespace door
}  // namespace eagle
