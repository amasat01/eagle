// SPDX-License-Identifier: Apache-2.0

// The door's internal state: the context and the handles made from it.
//
// Every exported function of eg_door.h is a shim in abi.cpp that enters the
// context through `call` below. The shim queues a second thread, and refuses a forked
// child and a context refused at creation BEFORE the body runs; the body
// refuses by throwing `Refusal`, and `call` turns every exception into a
// status plus the reason text in the context. No exception leaves `call`.
#pragma once

#include "eagle/door/eg_door.h"
#include "eagle/door/eg_unit_meta.h"
#include "eagle/cpu/DeviceProps.h"
#include "eagle/exec/Partition.h"
#include "eagle/policy/Dispatch.h"
#include "plugin/dlpack.h"
#include "plugin/gref_layout.h"

struct eg_unit_meta;

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <mutex>
#include <new>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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
    bool param_plane = false;                 ///< a LOOKUP read at index 0 only (a parameter in a one-cell plane); validated, not acted on
    /** @brief One named axis of a zeroed PLANE_OUT plane: a static extent, or -1 and the lane axis whose extent it shares. */
    struct Axis {
        std::string label;
        std::int64_t extent      = -1;
        int lane                 = -1;
        std::int32_t extent_slot = -1;
    };
    std::vector<Axis> axes; ///< a plane with named axes: they follow its
                            ///< payload extents (empty: any compact plane)
    bool made = false;                        ///< eagle's own plane (a companion): never bound by the caller
    bool cell = false;                        ///< a cell's slot: the caller binds the finished (1,) cell, the kernel reads eagle's rows
    bool zeroed = false;                      ///< a PLANE_OUT slot the runtime zeroes before the run: an accumulate target
    std::int32_t keys_of = -1;                ///< a kept PLANE_OUT slot a KEYS stage writes (a block sum's output): the keyed record whose keys it holds, else -1
    std::int32_t like = -1;                   ///< a plane sized like the bound plane of this LOOKUP slot (its ndim and extents), else -1
    bool optional_made = false;               ///< eagle makes the plane unless the caller binds one (the AUTO mix counters)
    std::int64_t min_elements = 0;            ///< a caller's plane of an optional_made slot holds at least this many elements
};

/** @brief A written plane the runtime zeroes before the run (an accumulate target). */
inline bool zeroed_out(const SlotFacts& s) { return s.role == EG_ROLE_PLANE_OUT && s.zeroed; }
/** @brief A written plane the caller's own values stay in (a store target). */
inline bool kept_out(const SlotFacts& s) { return s.role == EG_ROLE_PLANE_OUT && !s.zeroed; }

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
    std::int32_t comp = -1;                   ///< SUM: the row slot of its compensation; MAX, MIN: of the winners' lanes; else -1
    bool position = false;                    ///< MAX, MIN: the cell takes the lane of the winner, not its value
    std::string name;                         ///< the cell's name (a column's slot is named <cell>_<k> when the cell has several)
};

/** @brief One cell column of a split unit's stages: the rows its stage writes and where they fold (a source slot of the caller's, or a
 *  cell plane eagle makes and later stages read by this name). */
struct SplitCell {
    bool position = false;                    ///< MAX, MIN: the cell takes the lane of the winner, not its value (see CellFacts)
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
/** @brief One matrix product of a unit's gemm record (eg_unit_gemm_call), copied at eg_unit_load. A plan holds its own copy
 *  with the extents resolved and ``m``, ``n``, ``k`` set (op(A) is (m, k), op(B) is (k, n)). */
struct GemmCall {
    std::uint32_t flags = 0;
    std::int32_t a = -1, b = -1, c = -1;      ///< slots of the unit
    std::int64_t a_rows = 0, a_cols = 0, b_rows = 0, b_cols = 0, c_rows = 0, c_cols = 0;   ///< stored shapes; -1 until a plan resolves it
    std::int64_t m = 0, n = 0, k = 0;
};

/** @brief A unit's gemm record (eg_unit_meta.gemm): the products a DEVICE plan runs instead of a kernel. */
struct GemmFacts {
    bool present = false;
    std::int64_t lane_fanin = 0;
    std::vector<GemmCall> calls;
};

/** @brief A scratch plane of a unit's group (eg_unit_group_scratch), copied at
 * eg_unit_load. */
struct GroupScratchFacts {
    std::string name;
    std::int32_t dtype    = 0;
    std::int64_t elements = 0, bytes = 0; ///< per group
    std::int32_t realisation = 0; ///< EG_SCRATCH_LOCAL or EG_SCRATCH_PLANE
    std::int32_t slot
        = -1; ///< PLANE: the slot eagle makes and binds the plane to, else -1
};

/** @brief A unit's group record (eg_unit_meta.group), checked at eg_unit_load; the device launch reads it
 * (a block per group, its scratch in dynamic shared memory); the host launch reads it for the groups of a range. */
struct GroupFacts {
    bool present            = false;
    std::int32_t membership = 0; ///< EG_GROUP_*
    std::int32_t phases     = 0;
    std::int32_t key_slot = -1; ///< KEY: the slot of the per-lane key, else -1
    std::int64_t count    = 0;  ///< COUNT: the lanes of a group, else 0
    std::int64_t max_size
        = -1; ///< the size bound of a group in lanes, -1: not stated
    std::int64_t scratch_bytes
        = 0; ///< the bytes of all scratch planes of one group
    std::vector<std::int32_t> axes; ///< AXES: the lane axes that span a group
    std::vector<GroupScratchFacts> scratch;
    std::vector<std::int32_t>
        distinct; ///< the slots the unit assumes are distinct memory, ascending
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
    std::vector<eg_branch_cost> costs;        ///< AUTO: the cost of each branch (k entries), or empty: not stated
    std::uint32_t cost_flags = 0;             ///< AUTO: EG_DISPATCH_FUSED_VECTORISED, EG_DISPATCH_VECTOR_STATED
    bool names_set = false;                   ///< AUTO: the record names its segmented set (``branches`` and ``classify``)
};

/** @brief The segmented group a branch unit loaded with its group's units in deps runs: the units, and per unit the join of its
 *  slots to the plan's (the branch unit's slots, then the slots of the other units no earlier slot names). */
struct SegFacts {
    static constexpr std::int32_t kMap = -2, kCount = -3;   ///< a sweep's active map and count: eagle's own, joined to no plan slot
    bool present = false;
    bool dynamic = false;                     ///< a stale slot and a sweep, no classify unit
    std::size_t own_slots = 0;                ///< the planned unit's own slot count: its kernel takes that many arguments, not the group's extra slots
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

/** @brief The 40-byte mirror of the team plane @p d (made by eagle for @p t) that a mirror slot reads: the plane's own pitches,
 *  plane-major (per_lane, units) or lane-major (units, per_lane). */
inline plugin::GRefMirror team_mirror(const DLTensor& d, const TeamPlaneFacts& t, std::int32_t device_type) {
    const std::uint64_t units = static_cast<std::uint64_t>(elements(d) / t.per_lane);
    plugin::GRefMirror m;
    m.data_ = reinterpret_cast<double*>(static_cast<char*>(d.data) + d.byte_offset);
    m.samples_ = units;
    m.compStride_ = t.layout == EG_LAYOUT_PLANE_MAJOR ? units : 1;
    m.sampleStride_ = t.layout == EG_LAYOUT_PLANE_MAJOR ? 1 : static_cast<std::uint64_t>(t.per_lane);
    m.deviceType_ = device_type;
    m.deviceId_ = 0;
    return m;
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
    eagle::door::GemmFacts gemm;              ///< a dense-contraction unit (eg_unit_meta.gemm), else not present
    eagle::door::GroupFacts
        group_record; ///< a unit with a group (eg_unit_meta.group), else not
                      ///< present
    std::int32_t index = -1;                  ///< the unit index in its library table
    std::int32_t fault_slot = -1;             ///< a checked unit: the two-cell int64 LOOKUP the door makes per plan for the generated checks, else -1
    std::array<std::uint8_t, EG_DIGEST_BYTES> digest{};
    std::vector<eagle::door::TeamPlaneFacts> team_planes;
    std::int64_t scatter_chunks = 0;          ///< the most chunks of an ordered scatter (body.scatter_chunks), 0: none
    std::vector<eagle::door::KeyedFacts> keyed;    ///< the keyed table of a split unit's body
    std::vector<std::string> source_cells;    ///< a split unit's cells (no row slots: the stages own the rows)
    std::vector<eagle::door::SplitCell> split_cells;
    std::vector<eagle::door::StageFacts> stages;   ///< non-empty: a split unit, run stage by stage
    std::vector<eagle::door::DispatchFacts> dispatches;   ///< the unit's dispatch groups
    eagle::door::SegFacts seg;                ///< a branch unit loaded with its group's units in deps
    eg_unit_s* auto_seg = nullptr;            ///< an AUTO group that names its segmented set: the branch-0 unit loaded with the set in deps (a reference held)
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
    std::int64_t sorts = 0;                   ///< sorts of the samples by class made since the plan was made (a launch re-sorts when the batch went stale)
    bool sorted = false;
    plugin::ScalarHandle map_h{}, count_h{};
    std::vector<std::uint8_t> sortkey;        ///< DEVICE dynamic: the class of each sample (sample order) at the last sort
    std::vector<void*> dev_bufs;              ///< DEVICE: the device memory of the sort and the sorted copies, made at the first launch, freed with the plan
};

}  // namespace door
}  // namespace eagle

namespace eagle {
namespace door {
/**
 * @brief The first failure of a host step that runs inside a device graph. Such a step cannot return to a caller: it
 * stores its failure here, on the graph, and the next ``eg_graph_launch`` or ``eg_sync`` of the context returns it.
 */
struct StepFailure {
    std::mutex lock;
    bool failed = false;
    std::string reason;

    /** @brief Keep @p text unless an earlier failure is kept. Never throws. */
    void set(const std::string& text) noexcept {
        try {
            const std::lock_guard<std::mutex> guard(lock);
            if (!failed) failed = true, reason = text;
        } catch (...) {
        }
    }
    /** @brief Move the kept failure into @p out and clear it; false when there is none. */
    bool take(std::string& out) {
        const std::lock_guard<std::mutex> guard(lock);
        if (!failed) return false;
        out = std::move(reason);
        failed = false;
        reason.clear();
        return true;
    }
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
    std::vector<eagle::door::GemmCall> gemm;  ///< the unit's gemm calls with this plan's extents (empty: not a gemm unit)
    std::vector<Binding> bindings;            ///< one per slot; never resized after eg_plan_new
    std::vector<void*> params;                ///< the entry's argument list, into ``bindings``
    std::vector<eg_plane_s*> rows;            ///< one eagle-made row plane per cell component (``unit->cells``), one reference each
    std::vector<eg_plane_s*> comps;           ///< the compensation rows of each cell column (nullptr: none), parallel to ``rows``
    std::vector<eg_plane_s*> outs;            ///< a split unit: the cell plane eagle makes for each internal cell column (nullptr: the caller's)
    std::vector<eg_plane_s*> team;            ///< one eagle-made scratch plane per team plane of a split unit, one reference each
    std::vector<eg_plane_s*> group_scratch;   ///< one eagle-made plane per group scratch plane realised as a plane (``unit->group_record.scratch``, in order, the others skipped), one reference each
    eg_plane_s* fault = nullptr;              ///< a checked unit: the two int64 cells (slot ordinal, index) of the generated checks, one reference
    std::vector<eg_plane_s*> work;            ///< DEVICE: the two reduction work buffers of each cell component, one reference each
    std::vector<std::vector<eg_plane_s*>> segteam;   ///< a SEGMENTED plan: the scratch planes of each member that is a split unit other than the planned unit (by member index)
    std::vector<eg_plane_s*> segdev;          ///< a DEVICE SEGMENTED plan: the offsets and class planes eagle makes in device memory
    std::vector<eg_plane_s*> mix;             ///< the eagle-made mix counters of each AUTO group (``unit->dispatches`` order), one reference each
    mutable eagle::door::SegState seg;        ///< a SEGMENTED plan: its planes and sort
    /** @brief DEVICE: the scratch planes of SEGMENTED member @p mi, the unit @p m: the planned unit's are ``team``, another member's ``segteam``. */
    const std::vector<eg_plane_s*>& seg_team(const eg_unit_s& m, std::size_t mi) const { return &m == unit ? team : segteam[mi]; }
    std::uint32_t finished = 0;               ///< the eagle-made finished counter of a finishing unit
    std::int64_t steps_word = 0;              ///< eagle's own steps word of an AUTO unit, read when the caller binds none (`steps_cell`)
    std::uint64_t device_counter = 0;         ///< DEVICE with a finish: the device cell of the finished counter
    std::uint64_t device_steps = 0;           ///< DEVICE with steps AUTO: eagle's own device int64 steps word (`steps_cell`)
    eg_runner_s* active_owner = nullptr;      ///< the runner whose active set the plan reads
    int refs = 1;                             ///< the caller's reference plus one per runner
    bool caller_ref = true;
    std::uint64_t binds = 0;                  ///< eg_bind and eg_clear calls on this plan
    std::uint64_t launches = 0;               ///< eg_launch calls on this plan (a replay of a graph is not one)
};

/** @brief The slots a plain launch of @p p reads as its set of positions: an index set's, or the active set of a plan no runner owns
 *  (the caller binds the map and the count); -1: none (a prefix set has a count and no map). */
inline std::int32_t set_map_slot(const eg_plan_s& p) {
    return p.unit->index_count_slot >= 0 ? p.unit->index_map_slot : p.active_owner == nullptr ? p.unit->active_map_slot : -1;
}
inline std::int32_t set_count_slot(const eg_plan_s& p) {
    return p.unit->index_count_slot >= 0 ? p.unit->index_count_slot : p.active_owner == nullptr ? p.unit->active_count_slot : -1;
}

/** @brief The steps one launch advances the finisher of a step of @p n_plans plans: its own, but one when its steps are AUTO and the step has
 *  several plans (the runner writes 1 into the steps word at every launch). */
inline std::int64_t launch_steps(const eagle::door::FinishFacts& f, std::size_t n_plans) {
    return f.steps == EG_STEPS_AUTO && n_plans > 1 ? 1 : f.steps;
}

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
    int threads = 0;                          ///< the host threads a compaction uses, set by the run; 0: the default budget
    std::vector<std::uint64_t> scratch;       ///< the compaction's working memory, kept between compactions and grown with the lane count
    std::vector<std::uint8_t> narrowed;       ///< a mask wider than a byte, one byte per lane, kept between compactions
};

}  // namespace door
}  // namespace eagle

namespace eagle {
namespace door {

struct DeviceLoop;
struct DeviceFft;         ///< the cuFFT plans of a device transform (device/Fft.cpp)
struct DeviceRecording;   ///< a capture in progress on the context's stream (device/Device.cpp)
struct GraphDiff;         ///< the tape of a graph ended with EG_RECORD_DIFF (host/Graph.h)
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
    bool stepped = false;                     ///< a SEGMENTED group: no captured loop, one host-driven launch of the group per step
    unsigned block = 0, grid = 0, chunk = 0;  ///< the persist launch
    double fill = -1.0;                       ///< the step fill the persist launch was sized for; -1: none
    std::uint64_t fast = 0;                   ///< the persist entry's device [next, steps] (uint32)
    std::vector<unsigned char> bound;         ///< the plans' argument bytes the loop was captured over
    DeviceRun() = default;
    DeviceRun(const DeviceRun&) = delete;
    DeviceRun& operator=(const DeviceRun&) = delete;
    ~DeviceRun();
};

/** @brief The figures of the band policy after a host run of an automatic unit: what ``eg_runner_figures`` reads (the cells of ``Runner.policy``). */
struct BandFigures {
    double rate = -1.0;
    std::int64_t k = -1, finished_at_launch = -1, steps_done = -1, max_steps = -1;
    int go = -1;
};

/** @brief Add @p launches to the bin of ``eg_run_report.launches_by_k`` that holds @p k steps per launch. */
void note_launches_by_k(eg_run_report& report, std::int64_t k, std::int64_t launches);

/** @brief What a runner over an AUTO group that names its segmented set keeps (the look): the plan of the segmented set, stepped while the group
 *  runs segmented, and the band's state. */
struct AutoLook {
    eg_plan_s* seg = nullptr;                 ///< the segmented set's plan over the runner's planes, one reference
    const DispatchFacts* group = nullptr;     ///< the AUTO record
    eagle::policy::CostVector cost;           ///< its branch costs as the rule reads them
    eagle::policy::HysteresisState state;     ///< the policy in force; fused again after a look that leaves the band
    double gain = 1.0;                        ///< the gain of the last look
    std::int64_t looks = 0;                   ///< looks made in the last run
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
    std::int32_t reverse_passes = 0;          ///< the memory passes of the last eg_reverse (a step graph's reverse of each step adds one); 0 before one
    bool reverse_not_bitwise = false;         ///< a step graph's reverse of the last eg_reverse wrote an adjoint a cross-sample scatter-add feeds
    eg_steps_reverse steps_reverse = EG_STEPS_REVERSE_PER_STEP;   ///< a recorded runner's reverse mode, AUTO resolved
    eagle::door::ActiveState act;
    eagle::door::DeviceRun dev;               ///< a DEVICE runner's loop and cells
    eagle::door::RunRecord* record = nullptr; ///< EG_RUN_RECORD: the derived plans and the last record
    eagle::door::AutoLook* look = nullptr;    ///< a runner over an AUTO group that names its segmented set: the look that decides each run
    eagle::door::BandFigures band;            ///< the band policy's cells after the last host run of an automatic unit
    std::vector<eg_step_graph> step_graphs;   ///< the graphs run once per step, in step order (one reference held to each, released with the runner)
};

namespace eagle {
namespace door {

/**
 * @brief The stream in scope of one context: a fact of the calling thread, so two threads that each call eg_stream_use
 * and then launch never put a launch on each other's stream. A thread that has not set one reads the context's own
 * stream. Assigning the context's own stream (or 0) puts the thread back on the default. A thread's entry for a context
 * stays until it sets that context's default again; it holds two integers.
 */
class ThreadStream {
public:
    ThreadStream();
    operator std::uint64_t() const;
    ThreadStream& operator=(std::uint64_t stream);
    /** The stream a thread reads before it sets one (the context's own); also drops the calling thread's entry. */
    void set_default(std::uint64_t stream);

private:
    std::uint64_t serial_;                    ///< names this context's entry in every thread's table
    std::uint64_t default_ = 0;
};

/** @brief The device of a device context and the facts a launch reads (device/Device.cpp). */
struct DeviceState {
    std::int32_t ordinal = -1;                ///< -1: a host-only context
    int handle = 0;                           ///< the driver's device (a CUdevice)
    std::uint64_t context = 0;                ///< its primary context (a CUcontext), retained while the context lives
    ThreadStream stream;                      ///< the stream in scope for the calling thread (a CUstream): eagle's own, a queue of eg_stream_new, or the caller's (eg_stream_use)
    std::uint64_t own_stream = 0;             ///< eagle's own stream, destroyed with the context
    std::unordered_map<std::uint64_t, std::uint64_t> made_streams;   ///< eg_stream_new handle (bit 63 set, never a driver pointer) -> its CUstream
    std::uint64_t next_made_stream = 0;       ///< the counter the next handle is made from
    std::unordered_set<std::uint64_t> adopted_streams;               ///< raw streams the caller passed to eg_stream_use (never destroyed here)
    std::unordered_map<const void*, std::uint64_t> events;           ///< eg_event handle -> its CUevent (device/Device.cpp)
    struct ScanState {
        void* data = nullptr;
        std::size_t bytes = 0;
        std::size_t clean = 0;    ///< the first ``clean`` bytes are zero (what a scan, a compaction or a sort asks of its working memory)
    };
    std::unordered_map<std::uint64_t, ScanState> scan_state;         ///< stream -> the working memory of its eager scans, grown on demand (device/Device.cpp)
    int sms = 0, regs_per_sm = 0, smem_per_sm = 0, smem_per_block = 0, threads_per_sm = 0, cc_major = 0, cc_minor = 0;
    int clock_khz = 0, memory_clock_khz = 0, bus_bits = 0, l2_bytes = 0;   ///< the device's clocks, memory bus and L2: what its peak rates are derived from
    int fp64_ratio = 1;                       ///< FP32:FP64 throughput ratio (1 when not reported)
    std::unordered_map<void*, std::uint64_t> alloc_stream;   ///< each live device_alloc block -> the stream it was allocated on
    std::unordered_map<std::uint64_t, std::uint64_t> gemm_handles;   ///< stream -> its cuBLAS handle, made when a gemm unit is planned (device/Gemm.cpp)
};

}  // namespace door
}  // namespace eagle

struct eg_graph_s;      // a recorded list of door calls (host/Record.cpp)
struct eg_recording_s;  // the recording in progress on a context (host/Record.cpp)

/**
 * @brief A transform plan as it RUNS: an EG_FFT_ADJOINT descriptor is resolved at eg_fft_new into the kind, direction and
 * norm of the transform that computes it, plus the Hermitian weight of the bins along the halved axis.
 */
struct eg_fft_s {
    eg_ctx* ctx = nullptr;
    bool device = false;                      ///< the context's target: cuFFT, else pocketfft
    eg_fft_kind kind = EG_FFT_C2C;            ///< as run: R2C^H runs C2R and C2R^H runs R2C
    bool forward = true;                      ///< the c2c direction (R2C is forward, C2R inverse)
    eg_fft_norm norm = EG_FFT_NORM_BACKWARD;  ///< as run
    std::uint32_t bits = 64;                  ///< the float width of both planes
    std::vector<std::int64_t> shape;          ///< complex for C2C, the real side's for R2C and C2R
    std::vector<std::int32_t> axes;           ///< the last is the halved axis of a real transform
    std::int64_t bins = 0, edge = 0, run = 0; ///< real kinds: m = n/2+1, last (bins 0 < k < last are interior), floats of one bin
    double weight = 1.0;                      ///< factor on the interior bins: C2R scales its input by it, R2C its output
    eagle::door::DeviceFft* dev = nullptr;    ///< a device plan's cuFFT plans
    void* work = nullptr;                     ///< a device C2R's copy of its input (cuFFT's c2r destroys it)
    int refs = 1;                             ///< the caller's reference plus one per recording that runs it
    bool caller_ref = true;
};

/** @brief A direct matrix product (eg_gemm_new): one GemmCall with its extents resolved, the width and the target. */
/** One event (eg_event_new): the steady-clock time it was last recorded at; ``recorded`` says whether it ever was. */
struct eg_event_s {
    std::chrono::steady_clock::time_point stamp;
    bool recorded = false;
};

struct eg_gemm_s {
    eg_ctx* ctx = nullptr;
    bool device = false;                      ///< the context's target: cuBLAS, else the reference loop
    bool f32 = false;                         ///< the float width of the three planes
    eagle::door::GemmCall call;               ///< the stored shapes, flags and m, n, k; its slots are unused
    int refs = 1;                             ///< the caller's reference plus one per recording that runs it
    bool caller_ref = true;
};

/** @brief A caller's callback the door runs as one node (eg_host_node_new). */
struct eg_host_node_s {
    eg_ctx* ctx = nullptr;
    std::string name;
    eg_host_fn fn = nullptr;
    eg_host_vjp_fn vjp = nullptr;             ///< NULL: no reverse
    void* user = nullptr;
    eg_host_release_fn release = nullptr;     ///< called once with ``user`` when the last reference drops
    int refs = 1;                             ///< the caller's reference plus one per recorded graph that runs it
    bool caller_ref = true;
};

namespace eagle {
namespace door {
/** @brief The last non-OK reason of one calling thread; @c lost when its text could not be stored (out of memory). */
struct Reason {
    std::string text;
    bool lost = false;
};
}  // namespace door
}  // namespace eagle

/** @brief All door state of one caller. */
struct eg_ctx_s {
    long pid = 0;                             ///< the process that made the context
    std::mutex lock;                          ///< one call runs at a time; a second thread waits here (never taken in a forked child)
    std::atomic<std::thread::id> holder;      ///< the thread inside `lock`, else the null id: a re-entry on that thread is refused, not waited for
    eg_status refused = EG_OK;                ///< a refused eg_ctx_new: every call returns it
    std::int32_t host_threads = 0;            ///< 0: the default budget
    std::unordered_map<std::thread::id, eagle::door::Reason> reasons;   ///< each calling thread's last non-OK reason, under `lock`
    bool reason_lost = false;                 ///< a thread's first reason could not even get an entry (out of memory)
    std::unordered_set<eg_unit_s*> units;
    std::unordered_set<eg_plane_s*> planes;
    std::unordered_set<eg_plan_s*> plans;
    std::unordered_set<eg_runner_s*> runners;
    std::unordered_set<eg_graph_s*> graphs;
    std::unordered_set<eg_fft_s*> ffts;
    std::unordered_set<eg_gemm_s*> gemms;
    std::unordered_set<eg_host_node_s*> host_nodes;
    std::unordered_set<eg_event_s*> events;
    const std::string* in_host_node = nullptr;   ///< the name of the host node whose callback is running on this context, else null
    int exports = 0;                          ///< the DLPack exports (eg_plane_dlpack) not yet released by their consumers; changed under `lock`
    bool closed = false;                      ///< eg_ctx_free ran with exports outstanding: the last export's release frees the context
    eg_recording_s* rec = nullptr;            ///< between eg_record_begin and eg_record_end
    eagle::door::DeviceState dev;             ///< EG_CTX_DEVICE: the device and its stream
    eagle::door::DeviceRecording* dev_rec = nullptr;              ///< a device context between eg_record_begin and eg_record_end
    std::unordered_set<eagle::door::DeviceGraph*> dev_graphs;     ///< the device graphs eg_record_end made
    eagle::policy::Weights weights;           ///< the dispatch rule's weights: the row of this machine's class (-1: built-in) on its live facts
    eagle::DeviceProps props;                 ///< the properties the rule reads: the host's documented ones, or the device's from the driver
    std::int32_t weight_vendor = 0, weight_family = 0, weight_model = 0, weight_row = -1;   ///< the class key read and the row index taken
    bool gemm_by_cublas[2] = { true, true };  ///< DEVICE: whether this device class routes a routable gemm unit of a 32-bit (0) or 64-bit (1) scalar to cuBLAS

    /** @brief Store @p text as the calling thread's reason, never throwing. */
    void set_reason(const char* text) noexcept {
        try {
            eagle::door::Reason& r = reasons[std::this_thread::get_id()];
            try {
                r.text = text;
                r.lost = false;
            } catch (...) {
                r.lost = true;
            }
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
 * @brief The context's call lock, held for a scope. A forked child holds nothing: it has one thread, and a lock
 * that another thread of the parent held at the fork would never open. A thread that already holds the lock (a
 * caller's deleter, run inside a call, calling the door again) is not waited for: it would wait on itself, so
 * `reentered()` is true and the lock is neither taken nor released here.
 */
class Hold {
public:
    explicit Hold(eg_ctx& ctx) : ctx_(ctx) {
        if (forked(ctx)) return;
        const std::thread::id self = std::this_thread::get_id();
        if (ctx.holder.load(std::memory_order_relaxed) == self) {
            reentered_ = true;
            return;
        }
        ctx.lock.lock();
        ctx.holder.store(self, std::memory_order_relaxed);
        held_ = true;
    }
    ~Hold() { release(); }
    Hold(const Hold&) = delete;
    Hold& operator=(const Hold&) = delete;
    /** @brief True when this thread already held the context's lock. */
    bool reentered() const { return reentered_; }
    /** @brief Release before the context may be freed. */
    void release() {
        if (!held_) return;
        held_ = false;
        ctx_.holder.store(std::thread::id{}, std::memory_order_relaxed);
        ctx_.lock.unlock();
    }

private:
    eg_ctx& ctx_;
    bool held_ = false;
    bool reentered_ = false;
};

/**
 * @brief Run @p body on @p ctx under the door's rules and return its status.
 *
 * A second thread waits until the running call returns. Refused before
 * @p body runs: a NULL context, a call made from inside a call on the same
 * thread, a forked child unless @p in_child, and a context refused at
 * creation unless @p in_refused.
 */
template <class F>
eg_status call(eg_ctx* ctx, F&& body, bool in_child = false, bool in_refused = false) noexcept {
    if (ctx == nullptr) return EG_BAD_REQUEST;
    Hold hold(*ctx);
    eg_status status = EG_OK;
    if (hold.reentered()) {
        status = EG_BAD_REQUEST;
        if (ctx->in_host_node != nullptr)
            ctx->set_reason(("this context is busy inside the callback of host node '" + *ctx->in_host_node + "': a host node's callback must not call the door on its context").c_str());
        else
            ctx->set_reason("this context is already inside a call on this thread (a plane's deleter must not call the door)");
    } else if (!in_child && forked(*ctx)) {
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
    return status;
}

/** @brief The rows of a unit's cells that a launch over the samples [@p base, @p end) wrote: [first, end) in rows of
 *  ``cell_row`` samples; empty when the launch ran no lane. A cell is the reduction over the rows written. */
inline std::pair<std::int64_t, std::int64_t> rows_written(const eg_plan_s& plan, std::int64_t base, std::int64_t end) {
    const std::int64_t row = plan.unit->cell_row;
    if (end <= base || row < 1) return { 0, 0 };
    return { base / row, (end + row - 1) / row };
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
/** @brief Why the caller may not bind slot @p slot of @p unit: the tail of the
 * refusal ("is the finished counter: ..."), or NULL when the caller may bind
 * it. The one place that decides it, read by eg_bind and by eg_unit_made_slots.
 */
const char* made_reason(const eg_unit_s& unit, std::int32_t slot);
/** @brief The slots the library supplies: those the caller may not bind and
 * those it may leave unbound (the AUTO mix counters), in slot order. */
std::vector<std::int32_t> made_slots(const eg_unit_s& unit);
void unit_made_slots(eg_ctx& ctx, const eg_unit_s* unit, std::int32_t cap,
    std::int32_t* slots, std::int32_t* n);
/** @brief Drop one load of a unit library; the last one unloads it and closes the file it was read from. */
void close_library(void* library);
void plane_wrap(eg_ctx& ctx, DLManagedTensorVersioned* tensor, eg_plane** out);
void plane_release(eg_ctx& ctx, eg_plane_s* plane);
eg_plane_s* make_plane(eg_ctx& ctx, const DLDataType& dtype, std::vector<std::int64_t> shape, bool device = false,
                       bool pinned = false, bool zero = true);
void plane_new(eg_ctx& ctx, const eg_plane_desc* desc, eg_plane** out);
void plane_dlpack(eg_ctx& ctx, eg_plane_s* plane, DLManagedTensorVersioned** out);
void plan_new(eg_ctx& ctx, const eg_plan_desc* desc, eg_plan** out);
void plan_free(eg_ctx& ctx, eg_plan_s* plan);
void plan_release(eg_ctx& ctx, eg_plan_s* plan);
void plan_counters(eg_ctx& ctx, const eg_plan_s* plan, struct eg_plan_counters* out, std::uint32_t* mix, std::int32_t mix_cap, std::int32_t* n_mix);
/** @brief The steps word of an AUTO plan (its unit has ``finish.steps_slot``): the caller's bound plane, else eagle's own cell (a DEVICE plan's is a device address). */
void* steps_cell(const eg_plan_s& plan);
void check_bound(const eg_plan_s& plan);
void bind(eg_ctx& ctx, eg_plan_s* plan, const eg_bind_desc* desc);
void clear(eg_ctx& ctx, eg_plan_s* plan, std::int32_t n_slots, const std::int32_t* slots);
void set_count(eg_ctx& ctx, eg_plan_s* plan, std::uint32_t m);
void banded_pack(eg_ctx& ctx, const double* in, std::uint64_t* out, std::uint64_t n);
void banded_unpack(eg_ctx& ctx, const std::uint64_t* in, double* out, std::uint64_t n);
void banded_pack_scalar(eg_ctx& ctx, double x, std::uint64_t* out);
void bind_raw(eg_plan_s& plan, int slot, void* data, std::uint64_t count);
/** @brief The sample count the plane @p t bound at @p slot of @p plan states in its handle or mirror (eg_bind's rule). */
std::uint64_t bound_count(const eg_plan_s& plan, int slot, const DLTensor& t);
/** @brief Copy @p bytes on one side: host memory, or device to device on @p dev (host/Reverse.cpp). */
void copy_bytes(eg_ctx* dev, void* dst, const void* src, std::size_t bytes);
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
/** @brief eg_reverse; with @p handles false its planes need not be handles of the context (a graph's reverse binds buffers of its tape). */
void reverse(eg_ctx& ctx, eg_runner_s* runner, const eg_bind_desc* seeds, const eg_bind_desc* grads, bool handles = true);
/** @brief The primal slots of recorded runner @p runner (EG_RUN_RECORD, or refine with it) that carry a float plane: the taped states, which the
 *  run reads and overwrites, and the other planes it reads and gives a gradient for. */
struct RunSlots {
    std::vector<int> states, reads;
};
RunSlots recorded_run_slots(const eg_runner_s& runner);
/** @brief Whether @p runner's run can be reversed: an EG_RUN_RECORD runner (events refine included, when it was made with the flag). */
bool run_reversible(const eg_runner_s& runner);
/** @brief The run a recorded graph replays: the runner back at its start (eg_runner_reset), then its run, taped when it is an EG_RUN_RECORD runner. */
void run_from_start(eg_ctx& ctx, eg_runner_s* runner);
void runner_release(eg_ctx& ctx, eg_runner_s* runner);
void run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report);
void run_loop(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report);
/** @brief The look of a runner over an AUTO group with a segmented set (Look.cpp): check the request, attach the look to a new runner, run, reset, free. */
void look_check(const eg_ctx& ctx, const eg_runner_desc& desc);
void look_attach(eg_ctx& ctx, eg_runner_s& runner);
void look_run(eg_ctx& ctx, eg_runner_s* runner, eg_run_report* report);
void look_reset(eg_ctx& ctx, eg_runner_s& runner);
void look_free(eg_ctx& ctx, eg_runner_s& runner);
void seg_reset(eg_ctx& ctx, eg_plan_s& plan);
void reduce(eg_ctx& ctx, const eg_reduce_desc* desc);
void scan(eg_ctx& ctx, const eg_scan_desc* desc);
void compact(eg_ctx& ctx, const eg_compact_desc* desc);
void order_by_key(eg_ctx& ctx, const eg_order_desc* desc);
void reorder(eg_ctx& ctx, const eg_reorder_desc* desc);
void copy(eg_ctx& ctx, eg_plane_s* src, eg_plane_s* dst);
void fft_new(eg_ctx& ctx, const eg_fft_desc* desc, eg_fft** out);
void fft_run(eg_ctx& ctx, eg_fft_s* plan, eg_plane_s* in, eg_plane_s* out);
void fft_release(eg_ctx& ctx, eg_fft_s* plan);
/** @brief Free every transform plan of @p ctx (the context is closing). */
void fft_free_all(eg_ctx& ctx);
/** @brief The transform that is the adjoint of @p plan as run, made on the same target and owned by the caller's reference alone (EG_RECORD_DIFF's tape frees it). */
eg_fft_s* fft_adjoint(eg_ctx& ctx, const eg_fft_s& plan);
/** @brief eg_fft_run while recording: keep the run, false when the context is not recording. */
bool record_fft(eg_ctx& ctx, eg_fft_s* plan, eg_plane_s* in, eg_plane_s* out);

/** @brief The largest extent or row stride cuBLAS takes (a C int). */
constexpr std::int64_t kGemmMaxExtent = 2147483647;
/** @brief Set @p c's m, n, k from its stored shapes and flags; refuse (BAD_REQUEST, naming @p at) shapes whose contracted extents disagree or whose C is stored differently. */
void gemm_shape(GemmCall& c, const std::string& at);
/** @brief Refuse (BAD_REQUEST, naming @p at) a plane @p t that is not a view of @p rows rows whose row stride (its elements / @p rows) covers @p cols and that cuBLAS takes. */
void check_gemm_plane(const std::string& at, std::int64_t rows, std::int64_t cols, const DLTensor& t);
/** @brief The products that give the gradient of A and of B for the product @p g: dA from dC and B, dB from A and dC, each over the stored planes of the forward. */
std::pair<eg_gemm_s, eg_gemm_s> gemm_adjoints(const eg_gemm_s& g);
/** @brief Whether a DEVICE plan of @p plan's unit queues the cuBLAS products (a gemm unit the door can route, see eg_door.h). */
bool gemm_routed(const eg_plan_s& plan);
/** @brief Read the gemm route of @p ctx's device class from @p rows (a device context; refused EG_FAULT when the table is malformed). Run once at eg_ctx_new. */
void resolve_gemm_routes(eg_ctx& ctx, const eagle::policy::GemmRouteRow* rows, int count);
void gemm_new(eg_ctx& ctx, const eg_gemm_desc* desc, eg_gemm** out);
void gemm_run(eg_ctx& ctx, eg_gemm_s* gemm, eg_plane_s* a, eg_plane_s* b, eg_plane_s* c);
void gemm_release(eg_ctx& ctx, eg_gemm_s* gemm);
/** @brief eg_gemm_run while recording on the host: keep the checked run, false when the context is not recording. */
bool record_gemm(eg_ctx& ctx, eg_gemm_s* gemm, eg_plane_s* a, eg_plane_s* b, eg_plane_s* c);
bool record_launch(eg_ctx& ctx, eg_plan_s* plan);
bool record_run(eg_ctx& ctx, eg_runner_s* runner);
void record_begin(eg_ctx& ctx);
void record_cond_begin(eg_ctx& ctx, const eg_cond_desc* desc);
/** @brief Refuse the counter, limit and brk of @p d (and a WHILE that has no way to stop) as eg_record_cond_begin does on host and device. */
void check_loop_control(eg_ctx& ctx, const eg_cond_desc& d);
void record_cond_arm(eg_ctx& ctx);
void record_cond_end(eg_ctx& ctx);
void record_fork_begin(eg_ctx& ctx);
void record_fork_branch(eg_ctx& ctx);
void record_fork_end(eg_ctx& ctx);
void record_wait(eg_ctx& ctx, std::int32_t n_planes, eg_plane_s* const* planes);
void record_end(eg_ctx& ctx, const eg_record_end_desc* desc, eg_graph** out);
void record_abort(eg_ctx& ctx);
/** @brief eg_graph_reverse; @p from_step: the seeds and gradients are planes of the caller's own (views over its buffers, not handles) and the
 *  tests of the replay were restored by step_tape_restore, not read from the device (a runner's step graph, host/Reverse.cpp). */
void graph_reverse(eg_ctx& ctx, eg_graph_s* graph, const eg_graph_reverse_desc* desc, bool from_step = false);
/** @brief Zero @p bytes at @p data, on the context's stream for device memory. */
void zero_memory(eg_ctx& ctx, bool device, void* data, std::size_t bytes);
/** @brief A plan of a derived unit over eagle's memory: one whole partition of @p n samples (host/Reverse.cpp). */
eg_plan_s* inner_plan(eg_ctx& ctx, eg_unit_s* unit, eg_structure structure, std::int64_t n);
void inner_free(eg_ctx& ctx, eg_plan_s*& p);
/** @brief The sum of @p x[0, count) as a canonical compensated pair in sample order, on @p threads threads (host/Reverse.cpp). */
double sample_sum(const double* x, std::int64_t count, int threads);
float sample_sum(const float* x, std::int64_t count, int threads);
/** @brief Bind the by-value uniform of @p slot of @p p. */
void bind_value(eg_plan_s& p, int slot, const void* value, std::size_t bytes);
/** @brief eg_host_node_new / eg_host_node_release / eg_host_call (host/HostNode.cpp, host/Record.cpp). */
void host_node_new(eg_ctx& ctx, const eg_host_node_desc* desc, eg_host_node** out);
void host_node_release(eg_ctx& ctx, eg_host_node_s* node);
void host_call(eg_ctx& ctx, eg_host_node_s* node, std::int32_t n_planes, eg_plane_s* const* planes, const std::uint32_t* access);
/** @brief Checked units (host/Fault.cpp): set the fault cell of @p plan (or of every plan of a runner) to -1 before a launch, replay or run; read it after
 *  (a device plan: after the work queued on the stream) and refuse EG_FAULT "unit 'u': slot k ('name'): index i outside its N elements" when a check fired.
 *  Both are no-ops for a plan of an unchecked unit. */
void fault_reset(eg_ctx& ctx, const eg_plan_s& plan);
void fault_check(eg_ctx& ctx, const eg_plan_s& plan);
void fault_reset(eg_ctx& ctx, const eg_runner_s& runner);
void fault_check(eg_ctx& ctx, const eg_runner_s& runner);
/** @brief The five event calls on a host context (Context.cpp); a device context never reaches them (device_events_check). */
void event_new(eg_ctx& ctx, eg_event** out);
void event_free(eg_ctx& ctx, eg_event* event);
void event_record(eg_ctx& ctx, eg_event* event);
void event_wait(eg_ctx& ctx, eg_event* event);
void event_elapsed(eg_ctx& ctx, const eg_event* start, const eg_event* stop, double* ms);
void graph_launch(eg_ctx& ctx, eg_graph_s* graph);
void graph_free(eg_ctx& ctx, eg_graph_s* graph);
/** @brief A reference to host graph @p graph for a runner that runs it per step, and its release (the last one frees the graph). */
void graph_retain(eg_graph_s* graph);
void graph_release(eg_ctx& ctx, eg_graph_s* graph);
/** @brief Hold and release a step graph, host or device (a graph freed by its caller lives until its last runner is freed). */
void step_graph_retain(const eg_ctx& ctx, eg_graph_s* graph);
void step_graph_release(eg_ctx& ctx, eg_graph_s* graph);
/** @brief Run the step's graphs and plans in step order: each graph of @p runner ahead of the plan it names, through @p run_graph and @p run_plan. */
template <class GraphFn, class PlanFn>
void run_step(const eg_runner_s& runner, GraphFn run_graph, PlanFn run_plan) {
    std::size_t g = 0;
    for (std::size_t p = 0; p <= runner.plans.size(); ++p) {
        for (; g < runner.step_graphs.size() && static_cast<std::size_t>(runner.step_graphs[g].before_plan) == p; ++g) run_graph(runner.step_graphs[g].graph);
        if (p < runner.plans.size()) run_plan(runner.plans[p]);
    }
}
void graph_describe(eg_ctx& ctx, const eg_graph_s* graph, eg_graph_description* out);
void record_drop_all(eg_ctx& ctx);
bool device_planes(const eg_ctx& ctx, const char* op,
    std::initializer_list<const eg_plane_s*> planes,
    bool across_members = false);
/** @brief Whether @p src and @p dst of @p ctx are one host plane and one device plane (eg_copy's only mixed case). */
bool planes_cross_sides(const eg_ctx& ctx, const eg_plane_s* src, const eg_plane_s* dst);
bool device_plans(const eg_ctx& ctx, const eg_runner_desc* desc);
/** @brief The derived units of an AUTO runner that chose PER_STEP: the caller's set without the K-step vjp. */
struct KeptDerived {
    std::vector<eg_derived_role> roles;
    std::vector<eg_unit*> units;
};
void check_runner(eg_ctx& ctx, eg_runner_desc& d, bool device, KeptDerived& kept);
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
