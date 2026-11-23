// SPDX-License-Identifier: Apache-2.0

// Plans: a unit under one structure over fixed partitions; bindings by slot,
// checked once at eg_bind; eg_launch runs eagle's host team over them, or one
// device launch per partition (device/Device.cpp).
#include "eagle/door/Context.h"
#include "eagle/door/device/Seam.h"
#include "eagle/door/eg_unit_meta.h"

#include "eagle/door/host/Team.h"

#include <algorithm>

namespace eagle {
namespace door {

namespace {

/** @brief @p parts contiguous partitions of [0, @p range), each over @p n samples: the first range % parts take one
 *  more, as eagle's Python plan splits. A range shorter than @p parts leaves the last partitions empty. */
std::vector<exec::Partition> split_evenly(std::int64_t range, std::int64_t parts, std::int64_t n) {
    std::vector<exec::Partition> out;
    std::int64_t base = 0, remaining = range;
    for (std::int64_t left = parts; left > 0; --left) {
        const std::int64_t count = (remaining + left - 1) / left;
        out.push_back(exec::Partition{ base, count, n });
        base += count;
        remaining -= count;
    }
    return out;
}

/** @brief The groups of a launch over @p n lanes of a unit with a group: the unit's entry runs a range of GROUPS, a group wholly in one
 *  call. A COUNT group takes @p count consecutive lanes (the last may be partial); an AXES group takes the lanes along the spanned
 *  lane axes, the other axes tell the groups apart, so the launch must hold a whole number of them. */
std::int64_t group_count(const eg_unit_s& unit, std::int64_t n) {
    const GroupFacts& g = unit.group_record;
    if (g.membership == EG_GROUP_KEY)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': a unit with a group by key is not launched by this backend yet");
    if (g.membership == EG_GROUP_COUNT) return (n + g.count - 1) / g.count;
    std::int64_t fixed = 1;    // the lanes of the static axes: a run-time axis takes the launch's n over them
    for (std::int64_t e : unit.lane_extents) fixed *= e > 0 ? e : 1;
    if (n % fixed != 0)
        refuse(EG_BAD_REQUEST, "unit '" + unit.name + "': n_samples " + std::to_string(n) + " is not a whole number of the " +
               std::to_string(fixed) + " lanes its static lane axes hold");
    std::int64_t size = 1;
    for (std::int32_t axis : g.axes) {
        const std::int64_t e = unit.lane_extents[static_cast<std::size_t>(axis)];
        size *= e > 0 ? e : n / fixed;
    }
    return size > 0 ? n / size : 0;
}

/** @brief The partitions of @p d: the whole range, an even split of it, or the caller's, checked to stay inside
 *  the range without overlap. The range is the samples [0, n), for a thread box the threads
 *  [0, ceil(n / (t * stride)) * stride): each thread owns whole boxes, so no partition splits one, and for a unit with a group
 *  the groups [0, group_count). */
std::vector<exec::Partition> read_partitions(const eg_plan_desc& d) {
    const std::int64_t n = d.n_samples;
    const bool device = d.structure == EG_STRUCTURE_DEVICE;
    const std::int64_t box_t = device ? d.unit->dev_box_t : d.unit->box_t, box_stride = device ? d.unit->dev_box_stride : d.unit->box_stride;
    const bool grouped = d.unit->group_record.present;
    const std::int64_t range = grouped ? group_count(*d.unit, n)
                               : box_t > 0 ? (n + box_t * box_stride - 1) / (box_t * box_stride) * box_stride : n;
    const std::string range_text = grouped ? "the groups [0, " + std::to_string(range) + ") of unit '" + d.unit->name + "'"
                                   : box_t > 0 ? "the thread range [0, " + std::to_string(range) + ") of unit '" + d.unit->name + "' (" +
                                                   std::to_string(box_t) + " lanes per thread, stride " + std::to_string(box_stride) + ")"
                                             : "the range [0, " + std::to_string(n) + ")";
    if ((d.flags & EG_PLAN_SPLIT_EVENLY) != 0) {
        if (d.part_base != nullptr || d.part_count != nullptr)
            refuse(EG_BAD_REQUEST, "eg_plan_desc: EG_PLAN_SPLIT_EVENLY takes no part_base or part_count");
        if (d.n_partitions < 1)
            refuse(EG_BAD_REQUEST, "eg_plan_desc: EG_PLAN_SPLIT_EVENLY needs n_partitions >= 1, got " + std::to_string(d.n_partitions));
        return split_evenly(range, d.n_partitions, n);
    }
    if (d.n_partitions == 0) return { exec::Partition{ 0, range, n } };
    if (d.n_partitions < 0 || d.part_base == nullptr || d.part_count == nullptr)
        refuse(EG_BAD_REQUEST, "eg_plan_desc: n_partitions " + std::to_string(d.n_partitions) +
               " needs part_base and part_count");
    std::vector<exec::Partition> parts;
    for (std::int64_t i = 0; i < d.n_partitions; ++i) {
        const std::int64_t base = d.part_base[i], count = d.part_count[i];
        if (base < 0 || count < 1 || base > range || count > range - base)
            refuse(EG_BAD_REQUEST, "partition " + std::to_string(i) + " (base " +
                   std::to_string(base) + ", count " + std::to_string(count) +
                   ") leaves " + range_text);
        parts.push_back(exec::Partition{ base, count, n });
    }
    std::vector<exec::Partition> sorted = parts;
    std::sort(sorted.begin(), sorted.end(),
              [](const exec::Partition& a, const exec::Partition& b) { return a.base < b.base; });
    for (std::size_t i = 1; i < sorted.size(); ++i)
        if (sorted[i].base < sorted[i - 1].base + sorted[i - 1].count)
            refuse(EG_BAD_REQUEST, "partitions overlap at " + std::to_string(sorted[i].base));
    return parts;
}

/** @brief The bytes a uniform of wire dtype @p dtype occupies in a unit of @p scalar. */
std::size_t uniform_bytes(std::int32_t dtype, std::int32_t scalar) {
    switch (dtype) {
        case EG_DTYPE_REAL: return scalar == EG_SCALAR_FLOAT32 ? 4 : 8;
        case EG_DTYPE_INT64: return 8;
        case EG_DTYPE_INT32:
        case EG_DTYPE_UINT32: return 4;
        default: return 1;  // EG_DTYPE_BOOL (eg_unit_load refuses the rest)
    }
}

/** @brief The wire element type of a REAL slot of a unit of @p scalar: float32, float64, or for a banded unit
 *  the 8-byte cell as unsigned 64-bit. */
DLDataType real_dtype(std::int32_t scalar) {
    if (scalar == EG_SCALAR_BANDED) return DLDataType{ kDLUInt, 64, 1 };
    return DLDataType{ kDLFloat, static_cast<std::uint8_t>(scalar == EG_SCALAR_FLOAT32 ? 32 : 64), 1 };
}

/** @brief True for the roles whose plane has a runtime row count (the handle's extent is the trailing one). */
bool wide(const SlotFacts& s) { return s.role == EG_ROLE_WIDE_IN || kept_out(s); }

/** @brief True when @p t carries the element type slot @p s of a unit of @p scalar declares. */
bool dtype_ok(const DLDataType& t, const SlotFacts& s, std::int32_t scalar) {
    if (t.lanes != 1) return false;
    switch (s.dtype) {
        case EG_DTYPE_REAL: {
            const DLDataType real = real_dtype(scalar);
            return s.role != EG_ROLE_TERMINATED && t.code == real.code && t.bits == real.bits;
        }
        case EG_DTYPE_INT64: return t.code == kDLInt && t.bits == 64;
        case EG_DTYPE_INT32: return t.code == kDLInt && t.bits == 32;
        case EG_DTYPE_UINT32: return t.code == kDLUInt && t.bits == 32;
        default: return (t.code == kDLBool || t.code == kDLUInt) && t.bits == 8;  // EG_DTYPE_BOOL (a mask too)
    }
}

/** @brief The extent of each named axis of the plane @p t bound at slot @p s, and the refusal of a shape that does not fit.
 *  The rule, for every plane with axes: (i) an array of the exact shape (*payload, *axes) is accepted (a runtime axis takes the
 *  array's extent; a lane axis's label shares that axis's extent); (ii) any other array is a compact plane, accepted when its
 *  element count is the payload times the stated axes times at most one runtime axis, whose extent is the quotient; a count that
 *  does not divide is refused, and with two or more runtime axes only (i) is accepted. */
std::vector<std::int64_t> axis_extents(const eg_plan_s& plan, const SlotFacts& s, const DLTensor& t, const std::string& what) {
    const std::size_t payload = s.shape.size(), named = s.axes.size();
    if (t.ndim < 1) refuse(EG_BAD_REQUEST, what + ": a plane of one dimension or more is expected");
    std::int64_t fixed = 1;
    for (std::int64_t e : plan.unit->lane_extents) fixed *= e > 0 ? e : 1;
    std::vector<std::int64_t> want(named);    // the extent the unit states for each axis, -1 for a runtime one
    for (std::size_t i = 0; i < named; ++i) {
        const SlotFacts::Axis& a = s.axes[i];
        want[i] = a.lane < 0 ? a.extent
                  : plan.unit->lane_extents[static_cast<std::size_t>(a.lane)] > 0
                      ? plan.unit->lane_extents[static_cast<std::size_t>(a.lane)] : plan.n_samples / fixed;
    }
    std::vector<std::int64_t> got(named);
    std::string exact;    // why the exact shape does not fit, empty when it does
    if (t.ndim != static_cast<int>(payload + named)) {
        exact = "a plane of " + std::to_string(payload + named) + " dimension(s) is expected";
    } else {
        for (std::size_t i = 0; exact.empty() && i < payload; ++i)
            if (t.shape[i] != s.shape[i])
                exact = "axis " + std::to_string(i) + " has extent " + std::to_string(t.shape[i]) + ", the slot's payload has " +
                        std::to_string(s.shape[i]);
        for (std::size_t i = 0; i < named; ++i) {
            got[i] = t.shape[payload + i];
            if (exact.empty() && want[i] >= 0 && got[i] != want[i])
                exact = "axis " + std::to_string(payload + i) + " ('" + s.axes[i].label + "') has extent " + std::to_string(got[i]) +
                        ", the unit's named axis has " + std::to_string(want[i]);
        }
        if (exact.empty()) return got;
    }
    std::size_t runtime = 0;
    std::int64_t known = 1;
    for (std::int64_t e : s.shape) known *= e;
    for (std::int64_t e : want) {
        if (e < 0) ++runtime;
        else known *= e;
    }
    if (runtime > 1)
        refuse(EG_BAD_REQUEST, what + ": " + exact + "; with " + std::to_string(runtime) +
               " runtime axes only the shape (*payload, *axes) is accepted");
    const std::int64_t total = elements(t);
    if (total % known != 0 || (runtime == 0 && total != known))
        refuse(EG_BAD_REQUEST, what + ": " + exact + "; " + std::to_string(total) + " elements do not fit the plane either: the payload and the stated axes take " +
               std::to_string(known) + (runtime == 0 ? "" : " each, the runtime axis the rest"));
    for (std::size_t i = 0; i < named; ++i) got[i] = want[i] >= 0 ? want[i] : total / known;
    return got;
}

/** @brief Check the shape of a plane with named axes: see axis_extents. A plane that states none is any compact plane. */
void check_sink_shape(const eg_plan_s& plan, const SlotFacts& s, const DLTensor& t, const std::string& what) {
    if (t.ndim < 1) refuse(EG_BAD_REQUEST, what + ": a plane of one dimension or more is expected");
    if (s.shape.size() + s.axes.size() == 0) return;
    axis_extents(plan, s, t, what);
}

/** @brief Check @p plane against slot @p slot of @p plan (dtype, shape, device, write access). */
void check_plane(const eg_plan_s& plan, int slot, const eg_plane_s& plane) {
    const SlotFacts& s = plan.unit->slots[slot];
    const DLTensor& t = plane.tensor->dl_tensor;
    const std::string what = "slot " + std::to_string(slot) + " ('" + s.name + "')";
    if ((t.device.device_type == kDLCUDA) != (plan.structure == EG_STRUCTURE_DEVICE))
        refuse(EG_BAD_REQUEST, what + (t.device.device_type == kDLCUDA ? ": a device plane in a host plan"
                                                                       : ": a host plane in a DEVICE plan"));
    if (plan.unit->scalar_type == EG_SCALAR_BANDED && s.dtype == EG_DTYPE_REAL && s.role != EG_ROLE_TERMINATED && t.dtype.code == kDLFloat)
        refuse(EG_BAD_REQUEST, what + ": a banded cell is not a double; pack it with eg_banded_pack");
    if (!dtype_ok(t.dtype, s, plan.unit->scalar_type))
        refuse(EG_BAD_REQUEST, what + ": dtype (code " + std::to_string(t.dtype.code) + ", bits " +
               std::to_string(t.dtype.bits) + ") is not the slot's");
    if (s.gathered) {                    // compact along the kept lane axes: m = the product of their extents
        std::int64_t m = 1, fixed = 1;
        for (std::int64_t e : plan.unit->lane_extents) fixed *= e > 0 ? e : 1;
        for (std::size_t i = 0; i < s.keep.size(); ++i)
            if (s.keep[i]) m *= plan.unit->lane_extents[i] > 0 ? plan.unit->lane_extents[i] : plan.n_samples / fixed;
        if (s.role == EG_ROLE_LOOKUP) {  // read flat by the kernel's index over the kept axes: (*shape) values per kept lane
            std::int64_t want = m;
            for (std::int64_t e : s.shape) want *= e;
            if (t.ndim < 1 || elements(t) != want)
                refuse(EG_BAD_REQUEST, what + ": a table of " + std::to_string(want) + " elements is expected (" + std::to_string(m) +
                       " for the lane axes it varies along)");
        } else if (t.ndim != 3 || t.shape[0] != s.shape[0] || t.shape[1] != s.shape[1] || t.shape[2] != m) {    // (rows, cols, m)
            refuse(EG_BAD_REQUEST, what + ": a plane of shape (" + std::to_string(s.shape[0]) + ", " + std::to_string(s.shape[1]) +
                   ", " + std::to_string(m) + ") is expected (m: the columns of the lane axes it varies along)");
        }
    } else if (s.keys_of >= 0) {                // a block sum's output: width entries per key, the keys' count along the last axis
        const KeyedFacts& k = plan.unit->keyed[static_cast<std::size_t>(s.keys_of)];
        const std::int64_t want = k.width * key_count(*plan.unit, k, plan.n_samples);
        if (t.ndim < 1 || elements(t) != want)
            refuse(EG_BAD_REQUEST, what + ": a block sum's output of " + std::to_string(want) + " elements (" + std::to_string(k.width) + " per key) is expected");
        check_sink_shape(plan, s, t, what);
    } else if (!s.axes.empty()) { // a plane with named axes: (*payload, *axes),
                                  // whatever its role
        check_sink_shape(plan, s, t, what);
    } else if (wide(s)) { // the caller's layout: any compact plane; the
                          // kernel's reads clamp to its trailing extent
        if (t.ndim < 1) refuse(EG_BAD_REQUEST, what + ": a plane of at least one dimension is expected");
    } else if (s.role
        == EG_ROLE_LOOKUP) { // a table of the caller's: any compact shape, read
                             // by element count
        if (t.ndim < 1) refuse(EG_BAD_REQUEST, what + ": a table needs at least one dimension");
        if (s.shared && s.rank > 0) {    // a shared parameter has its natural shape, bound flat
            bool same = t.ndim == s.rank;
            for (int i = 0; same && i < s.rank; ++i) same = t.shape[i] == s.shape[static_cast<std::size_t>(i)];
            if (!same) refuse(EG_BAD_REQUEST, what + ": a shared parameter of its declared shape is expected");
        }
        if (s.optional_made && elements(t) < s.min_elements)
            refuse(EG_BAD_REQUEST, what + ": the mix counters of a dispatch group hold at least " + std::to_string(s.min_elements) + " elements (one per branch)");
        if (slot == set_map_slot(plan) && t.ndim != 1)
            refuse(EG_BAD_REQUEST, what + ": the index map is one dimension");
        if (slot == set_count_slot(plan) && (t.ndim != 1 || t.shape[0] != 1))
            refuse(EG_BAD_REQUEST, what + ": the index count is one cell, shape (1,)");
        if (slot == plan.unit->finish.steps_slot && (t.ndim != 1 || t.shape[0] != 1))
            refuse(EG_BAD_REQUEST, what + ": the steps word is one cell, shape (1,)");
    } else if (zeroed_out(s)) { // a cell: the finished (1,); else the caller's
                                // length, as (*payload, *named axes) the meta
                                // states
        if (s.cell) {
            if (t.ndim != 1 || t.shape[0] != 1) refuse(EG_BAD_REQUEST, what + ": a cell is one element, shape (1,)");
        } else {
            check_sink_shape(plan, s, t, what);
        }
    } else { // (*shape, n): the slot's extents, then one value per sample
        bool ok = t.ndim == s.rank + 1 && t.shape[s.rank] == plan.n_samples;
        for (int i = 0; ok && i < s.rank; ++i) ok = t.shape[i] == s.shape[static_cast<std::size_t>(i)];
        if (!ok) {
            std::string want = "(";
            for (std::int64_t e : s.shape) want += std::to_string(e) + ", ";
            refuse(EG_BAD_REQUEST, what + ": a plane of shape " + want + std::to_string(plan.n_samples) + ") is expected");
        }
    }
    if (!compact(t)) refuse(EG_BAD_REQUEST, what + ": the plane is not contiguous");
    const bool written = s.role == EG_ROLE_MUTABLE || s.role == EG_ROLE_OUT || s.role == EG_ROLE_PLANE_OUT ||
                         slot == plan.unit->finish.mask_slot || slot == plan.unit->finish.steps_slot;
    if (written && (plane.tensor->flags & DLPACK_FLAG_BITMASK_READ_ONLY))
        refuse(EG_BAD_REQUEST, what + ": a read-only plane in a written slot");
}

/** @brief The planes an index set will have after @p d: refuse a count above the map, a lane outside
 *  [0, n) and a lane listed twice (``_index_set.check_index_set``). */
void check_index_set(const eg_plan_s& plan, const eg_bind_desc& d) {
    const auto prospect = [&](int slot) -> const eg_plane_s* {
        for (int i = 0; i < d.n_planes; ++i)
            if (d.plane_slots[i] == slot) return d.planes[i];
        return plan.bindings[slot].plane;
    };
    const eg_plane_s *m = prospect(plan.unit->index_map_slot), *c = prospect(plan.unit->index_count_slot);
    if (m == nullptr || c == nullptr) return;
    const DLTensor& mt = m->tensor->dl_tensor;
    const bool device = plan.structure == EG_STRUCTURE_DEVICE;    // the caller's planes are device memory: read them back
    const auto fetch = [&](const DLTensor& t, std::size_t bytes, void* out) {
        const char* at = static_cast<const char*>(t.data) + t.byte_offset;
        if (device) device_read(*plan.ctx, at, out, bytes);
        else std::memcpy(out, at, bytes);
    };
    std::uint32_t count = 0;
    fetch(c->tensor->dl_tensor, sizeof(count), &count);
    const std::uint64_t held = count;
    if (held > static_cast<std::uint64_t>(mt.shape[0]))
        refuse(EG_BAD_REQUEST, "the index count is " + std::to_string(held) + ", but the index map holds " +
               std::to_string(mt.shape[0]) + " entries");
    std::vector<char> words(static_cast<std::size_t>((held * mt.dtype.bits / 8 + 3) / 4 * 4));    // the listed entries, whole words
    if (!words.empty()) fetch(mt, words.size(), words.data());
    const char* data = words.data();
    std::vector<bool> seen(static_cast<std::size_t>(plan.n_samples), false);
    for (std::uint64_t i = 0; i < held; ++i) {
        const std::int64_t lane = mt.dtype.bits == 64 ? reinterpret_cast<const std::int64_t*>(data)[i]
                                             : reinterpret_cast<const std::int32_t*>(data)[i];
        if (lane < 0 || lane >= plan.n_samples)
            refuse(EG_BAD_REQUEST, "the index set: map[" + std::to_string(i) + "] = " + std::to_string(lane) +
                   " is outside the dense lanes [0, " + std::to_string(plan.n_samples) + ")");
        if (seen[static_cast<std::size_t>(lane)])
            refuse(EG_BAD_REQUEST, "the index set: map[" + std::to_string(i) + "] = " + std::to_string(lane) +
                   " repeats an earlier entry; each lane may be listed once");
        seen[static_cast<std::size_t>(lane)] = true;
    }
}

/** @brief The unit's gemm calls with the extents of a plan over @p n_samples lanes resolved and m, n, k set; refuses a record the
 *  shapes contradict (the contracted extents, C's stored shape, an extent that is empty or too large for cuBLAS). */
std::vector<GemmCall> resolve_gemm(const eg_unit_s& unit, std::int64_t n_samples) {
    const GemmFacts& g = unit.gemm;
    std::vector<GemmCall> calls = g.calls;
    for (std::size_t i = 0; i < calls.size(); ++i) {
        GemmCall& c = calls[i];
        const std::string at = "unit '" + unit.name + "' gemm calls[" + std::to_string(i) + "]";
        for (std::int64_t* e : { &c.a_rows, &c.a_cols, &c.b_rows, &c.b_cols, &c.c_rows, &c.c_cols }) {
            if (*e == EG_GEMM_PER_LANE) {
                if (n_samples % g.lane_fanin != 0)
                    refuse(EG_BAD_REQUEST, at + ": n_samples " + std::to_string(n_samples) + " is not divisible by lane_fanin " + std::to_string(g.lane_fanin));
                *e = n_samples / g.lane_fanin;
            }
            if (*e < 1 || *e > kGemmMaxExtent)
                refuse(EG_BAD_REQUEST, at + ": the extent " + std::to_string(*e) + " (n_samples " + std::to_string(n_samples) + ") is outside 1 .. " +
                       std::to_string(kGemmMaxExtent));
        }
        gemm_shape(c, at);
    }
    return calls;
}

/** @brief Whether a DEVICE plan of the gemm unit @p unit over @p n_parts partitions can be the cuBLAS products: one product per
 *  launch, so not several partitions, a split or segmented unit, cells or companions, an index set or an active set. */
bool gemm_routable(const eg_unit_s& unit, std::size_t n_parts) {
    return n_parts == 1 && unit.stages.empty() && !unit.seg.present && unit.cells.empty() && unit.companions.empty() &&
           unit.index_map_slot < 0 && unit.index_count_slot < 0 && unit.active_map_slot < 0;
}

/** @brief Check every entry of @p d before any binding changes (a bind is all or none). */
void check_bind(const eg_ctx& ctx, const eg_plan_s& plan, const eg_bind_desc& d) {
    if (d.flags != 0)
        refuse(EG_BAD_REQUEST, "eg_bind_desc.flags: unknown bits " + std::to_string(d.flags));
    if (d.n_planes < 0 || d.n_uniforms < 0 ||
        (d.n_planes > 0 && (d.plane_slots == nullptr || d.planes == nullptr)) ||
        (d.n_uniforms > 0 && (d.uniform_slots == nullptr || d.uniform_values == nullptr)))
        refuse(EG_BAD_REQUEST, "eg_bind_desc: counts without their arrays");
    const int n_slots = static_cast<int>(plan.bindings.size());
    std::vector<bool> seen(n_slots, false);
    const auto take = [&](int slot, bool plane) {
        if (slot < 0 || slot >= n_slots)
            refuse(EG_BAD_REQUEST, "unknown slot " + std::to_string(slot) + " (the unit has " +
                   std::to_string(n_slots) + ")");
        const SlotFacts& s = plan.unit->slots[slot];
        if (slot == plan.unit->finish.counter_slot)
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(slot) + " ('" + s.name +
                   "') is the finished counter: eagle makes it, do not pass it");
        if (slot == plan.unit->fault_slot)
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(slot) + " ('" + s.name +
                   "') is the fault cell of a checked unit: eagle makes it, do not pass it");
        if (s.role == EG_ROLE_NSAMPLES)
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(slot) + " ('" + s.name +
                   "') is the sample count: eagle passes it, do not bind it");
        if (s.role == EG_ROLE_EXTENT)
            refuse(EG_BAD_REQUEST,
                "slot " + std::to_string(slot) + " ('" + s.name
                    + "') is the extent of a plane's runtime axis: eagle reads "
                      "it from that plane, do not bind it");
        if (plan.active_owner != nullptr && (slot == plan.unit->active_map_slot || slot == plan.unit->active_count_slot))
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(slot) + " ('" + s.name +
                   "') is part of the active set of a runner: the runner makes it, do not pass it");
        if (s.made)
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(slot) + " ('" + s.name +
                   "') is a companion or compensation plane: eagle makes it, do not pass it");
        if ((s.role == EG_ROLE_UNIFORM) == plane)
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(slot) + " ('" + s.name + "') takes " +
                   (plane ? "a uniform, not a plane" : "a plane, not a uniform"));
        if (seen[slot]) refuse(EG_BAD_REQUEST, "slot " + std::to_string(slot) + " bound twice in one call");
        seen[slot] = true;
    };
    for (int i = 0; i < d.n_planes; ++i) {
        take(d.plane_slots[i], true);
        check_handle(ctx.planes, d.planes[i], "eg_bind_desc.planes[i]");
        check_plane(plan, d.plane_slots[i], *d.planes[i]);
    }
    for (int i = 0; i < d.n_uniforms; ++i) {
        take(d.uniform_slots[i], false);
        if (d.uniform_values[i] == nullptr)
            refuse(EG_BAD_REQUEST, "uniform for slot " + std::to_string(d.uniform_slots[i]) + " is NULL");
    }
    if (plan.unit->index_map_slot >= 0) check_index_set(plan, d);
    const auto bound_plane = [&](int slot) -> const eg_plane_s* {    // this call's plane for the slot, else the one it holds
        for (int i = 0; i < d.n_planes; ++i)
            if (d.plane_slots[i] == slot) return d.planes[i];
        return plan.bindings[static_cast<std::size_t>(slot)].plane;
    };
    for (int s = 0; s < n_slots; ++s) {    // a plane sized like a LOOKUP's: checked when both are known, so whichever of the two is bound last (or again) checks the pair
        const std::int32_t like = plan.unit->slots[static_cast<std::size_t>(s)].like;
        if (like < 0) continue;
        const eg_plane_s *p = bound_plane(s), *l = bound_plane(like);
        if (p == nullptr || l == nullptr || (!seen[s] && !seen[like])) continue;
        const DLTensor &a = p->tensor->dl_tensor, &b = l->tensor->dl_tensor;
        const auto shape_of = [](const DLTensor& t) {
            std::string r = "(";
            for (int i = 0; i < t.ndim; ++i) r += (i ? ", " : "") + std::to_string(t.shape[i]);
            return r + ")";
        };
        bool same = a.ndim == b.ndim;
        for (int i = 0; same && i < a.ndim; ++i) same = a.shape[i] == b.shape[i];
        if (!same)
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(s) + " ('" + plan.unit->slots[static_cast<std::size_t>(s)].name + "') is sized like slot " + std::to_string(like) + " ('" +
                   plan.unit->slots[static_cast<std::size_t>(like)].name + "'): its plane is " + shape_of(a) + ", that slot's is " + shape_of(b));
    }
    for (std::size_t i = 0; i < plan.gemm.size(); ++i) {    // a gemm slot's plane, as a view of the rows the call states
        const GemmCall& c = plan.gemm[i];
        const std::int32_t slots[3] = { c.a, c.b, c.c };
        const std::int64_t rows[3] = { c.a_rows, c.b_rows, c.c_rows }, cols[3] = { c.a_cols, c.b_cols, c.c_cols };
        for (int j = 0; j < 3; ++j)
            if (seen[static_cast<std::size_t>(slots[j])])
                check_gemm_plane("unit '" + plan.unit->name + "' gemm calls[" + std::to_string(i) + "] slot " + std::to_string(slots[j]) + " ('" +
                                     plan.unit->slots[static_cast<std::size_t>(slots[j])].name + "')",
                                 rows[j], cols[j], bound_plane(slots[j])->tensor->dl_tensor);
    }
}

/** @brief The device tag a view of @p plan carries (the device id stays 0, as ``_plan_pack`` writes it). */
std::int32_t abi_device(const eg_plan_s& plan) {
    return plan.structure == EG_STRUCTURE_DEVICE ? plugin::kEagleAbiDeviceCUDA : plugin::kEagleAbiDeviceCPU;
}

/** @brief Bind slot @p slot of @p plan to the 40-byte mirror of a compact (components, n) plane: the
 *  component pitch is n and the sample pitch 1 (``_plan_pack._gref_mirror_box`` of a compact plane). */
void bind_mirror(eg_plan_s& plan, int slot, void* data, std::uint64_t n) {
    eg_plan_s::Binding& b = plan.bindings[slot];
    b.mirror.data_ = static_cast<double*>(data);
    b.mirror.samples_ = n;
    b.mirror.compStride_ = n;
    b.mirror.sampleStride_ = 1;
    b.mirror.deviceType_ = abi_device(plan);
    b.mirror.deviceId_ = 0;
    b.bound = true;
    plan.params[slot] = &b.mirror;
}

/** @brief Bind slot @p slot of @p plan to a plain handle over @p data of @p count elements. */
void bind_handle(eg_plan_s& plan, int slot, void* data, std::uint64_t count) {
    eg_plan_s::Binding& b = plan.bindings[slot];
    b.handle.data = data;
    b.handle.samples = count;
    b.handle.stride = 1;
    b.handle.deviceType = abi_device(plan);
    b.handle.deviceId = 0;
    b.bound = true;
    plan.params[slot] = &b.handle;
}

}  // namespace

/** @brief The scratch plane of team plane @p t: per_lane elements per unit, the units counted by the plane's unit (a TARGET plane's from @p cells, the
 *  cells of the caller's target plane); a placed wide value is one unit on the host, whole warps of 32 on the device. Full of its fill, made once. */
static eg_plane_s* make_team_plane(eg_ctx& ctx, const eg_unit_s& unit, const TeamPlaneFacts& t, std::int64_t n, bool device, std::int64_t cells) {
    const std::int64_t count = t.unit == EG_UNIT_ONCE ? 1 : t.unit == EG_UNIT_KEY ? key_count(unit, unit.keyed[static_cast<std::size_t>(t.keyed)], n)
                             : t.unit == EG_UNIT_TARGET ? chunks_of(n, cells, unit.scatter_chunks) * cells : n;
    const std::int64_t units = t.wide != EG_WIDE_WARP_TILED ? count : device ? (count + 31) / 32 * 32 : 1;
    const std::int64_t size = std::max<std::int64_t>(1, t.per_lane * units);
    const DLDataType elem = t.dtype == EG_DTYPE_INT32 ? DLDataType{ kDLInt, 32, 1 }
                                                      : real_dtype(unit.scalar_type);
    std::vector<std::int64_t> shape{ size };
    if (t.layout == EG_LAYOUT_PLANE_MAJOR && t.per_lane != 1) shape = { t.per_lane, size / t.per_lane };
    return make_plane(ctx, elem, shape, device);
}

/** @brief eg_plan_new: a host or DEVICE plan of the unit over checked partitions. */
void plan_new(eg_ctx& ctx, const eg_plan_desc* desc, eg_plan** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_plan_new: out is NULL");
    *out = nullptr;
    const eg_plan_desc d = read_input(desc, "eg_plan_desc");
    if ((d.flags & ~EG_PLAN_SPLIT_EVENLY) != 0)
        refuse(EG_BAD_REQUEST, "eg_plan_desc.flags: unknown bits " + std::to_string(d.flags & ~EG_PLAN_SPLIT_EVENLY));
    check_handle(ctx.units, d.unit, "eg_plan_desc.unit");
    if (d.unit->group_record.present && d.structure == EG_STRUCTURE_DEVICE)
        refuse(EG_NOT_WRITTEN,
            "unit '" + d.unit->name
                + "': a unit with a group is not launched by this backend yet");
    for (const StageFacts& st : d.unit->stages)
        if (st.unit->group_record.present)
            refuse(EG_NOT_WRITTEN,
                "unit '" + d.unit->name + "' stage '" + st.ref_name
                    + "': a unit with a group is not launched by this backend "
                      "yet");
    const bool device = d.structure == EG_STRUCTURE_DEVICE;
    if (device) device_plan_check(ctx, *d.unit);
    else if (d.structure != EG_STRUCTURE_HOST_TEAM && d.structure != EG_STRUCTURE_HOST_SERIAL)
        refuse(EG_BAD_REQUEST, "unknown structure " + std::to_string(d.structure));
    if (d.policy != EG_REDUCE_COMPENSATED && d.policy != EG_REDUCE_FIXED)
        refuse(EG_BAD_REQUEST, "unknown reduce policy " + std::to_string(d.policy));
    if (d.n_samples < 0) refuse(EG_BAD_REQUEST, "n_samples " + std::to_string(d.n_samples) + " < 0");
    if (!device && d.unit->scalar_type == EG_SCALAR_BANDED)
        refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': a BANDED unit on a host structure");
    if (!device && d.unit->host_entry == nullptr && d.unit->stages.empty())
        refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "' has no host entry");
    if (!device) {
        for (const CompanionFacts& c : d.unit->companions)
            if (c.fold == nullptr)
                refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': the companion '" + d.unit->slots[c.slot].name +
                       "' has no host fold entry");
        for (const StageFacts& st : d.unit->stages)
            for (const CompanionFacts& c : st.unit->companions)
                if (c.fold == nullptr)
                    refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "' stage '" + st.ref_name + "': the companion '" + st.unit->slots[c.slot].name +
                           "' has no host fold entry");
    }
    for (const DispatchFacts& f : d.unit->dispatches)    // a branch unit is planned with its group's units in deps
        if (f.branch_record && !d.unit->seg.present)
            refuse(EG_BAD_REQUEST, "unit '" + d.unit->name + "' is one branch of the SEGMENTED dispatch group '" + f.name +
                   "': load it with the group's units in deps to plan the group");
    if (d.unit->seg.present) {
        const SegFacts& g = d.unit->seg;
        for (const eg_unit_s* m : g.branches)    // a branch's own scratch planes run on the device; the host runs the sweep's only
            if (!device && !m->team_planes.empty())
                refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': the branch '" + m->name + "' of dispatch group '" + g.name + "' has scratch planes: scratch planes of a segmented member on the host are not written");
        if (!device && g.sweep != nullptr && !g.sweep->stages.empty() && g.sweep->stages[0].team)
            refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': the sweep unit '" + g.sweep->name + "' of dispatch group '" + g.name + "' is a team stage, which is not written on the host");
        if (device && d.unit->dev_box_t > 0) refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': a SEGMENTED dispatch group over a device thread box is not written");
        std::vector<const eg_unit_s*> units(g.branches.begin(), g.branches.end());
        units.push_back(g.dynamic ? g.sweep : g.classify);
        for (const eg_unit_s* m : units)
            if (device ? m->function == 0 : m->host_entry == nullptr)
                refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': the unit '" + m->name + "' of dispatch group '" + g.name + "' has no " + (device ? "device function" : "host entry"));
        if (d.unit->box_t > 0 || d.unit->index_count_slot >= 0 || d.unit->active_map_slot >= 0)
            refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': a SEGMENTED dispatch group over a thread box, an index set or an active set is not written");
    }
    std::vector<exec::Partition> parts = read_partitions(d);
    if (d.unit->seg.present && (parts.size() != 1 || parts[0].base != 0 || parts[0].count != d.n_samples))
        refuse(EG_BAD_REQUEST, "unit '" + d.unit->name + "' runs a SEGMENTED dispatch group over the sorted samples: one whole partition");
    for (const exec::Partition& p : parts)    // a row of a cell has one writer: a partition starts on a row
        if (d.unit->cell_row > 0 && p.base % d.unit->cell_row != 0)
            refuse(EG_BAD_REQUEST, "unit '" + d.unit->name + "' reduces over samples in rows of " + std::to_string(d.unit->cell_row) +
                   ": a partition starts on a multiple of it, got base " + std::to_string(p.base));
    for (const StageFacts& st : d.unit->stages)    // a row of a stage's cell has one writer too
        for (const exec::Partition& p : parts)
            if (st.cell_row > 0 && !st.unit->cells.empty() && p.base % st.cell_row != 0)
                refuse(EG_BAD_REQUEST, "unit '" + d.unit->name + "' stage '" + st.ref_name + "' reduces over samples in rows of " + std::to_string(st.cell_row) +
                       ": a partition starts on a multiple of it, got base " + std::to_string(p.base));
    if ((d.unit->index_count_slot >= 0 || d.unit->active_map_slot >= 0) && parts.size() != 1)
        refuse(EG_BAD_REQUEST, "unit '" + d.unit->name + "' reads an index set or an active set: one partition (positions of the set)");
    static constexpr exec::Access kAccess[] = { exec::Access::SampleLocal,
        exec::Access::CrossSampleRead, exec::Access::CrossSampleWrite };
    try {
        exec::check_placement(kAccess[d.unit->seg.present ? d.unit->seg.access : d.unit->access], device ? exec::Structure::DeviceKernel : exec::Structure::HostTeam,
                              static_cast<int>(parts.size()));
    } catch (const std::runtime_error& e) {
        refuse(EG_BAD_REQUEST, e.what());
    }
    std::vector<GemmCall> gemm;
    if (d.unit->gemm.present) {
        gemm = resolve_gemm(*d.unit, d.n_samples);
        if (device && gemm_routable(*d.unit, parts.size())) device_gemm_ready(ctx);
    }
    auto* plan = new eg_plan_s;
    plan->ctx = &ctx;
    plan->unit = d.unit;
    plan->structure = d.structure;
    plan->n_samples = d.n_samples;
    try {
        plan->parts = std::move(parts);
        plan->gemm = std::move(gemm);
        plan->bindings.resize(d.unit->slots.size());
        plan->params.resize(d.unit->slots.size());
        if (d.unit->finish.present && d.structure == EG_STRUCTURE_DEVICE) {    // the counter: one device cell
            device_plan_cells(*plan);
            bind_handle(*plan, d.unit->finish.counter_slot, reinterpret_cast<void*>(plan->device_counter), 1);
        } else if (d.unit->finish.present) {    // the finished counter: one uint32 cell (its element count)
            bind_handle(*plan, d.unit->finish.counter_slot, &plan->finished, 1);
        }
        for (std::size_t i = 0; i < d.unit->slots.size(); ++i)    // eagle passes the sample count
            if (d.unit->slots[i].role == EG_ROLE_NSAMPLES) {
                eg_plan_s::Binding& b = plan->bindings[i];
                const std::uint32_t n32 = static_cast<std::uint32_t>(d.n_samples);
                std::memcpy(b.value, &n32, sizeof(n32));
                b.bound = true;
                plan->params[i] = b.value;
            }
        if (d.unit->fault_slot >= 0) {    // a checked unit: the cells its generated checks claim
            plan->fault = make_plane(ctx, DLDataType{ kDLInt, 64, 1 }, { 2 }, device);
            bind_handle(*plan, d.unit->fault_slot, plan->fault->tensor->dl_tensor.data, 2);
        }
        if (d.unit->finish.present && d.unit->finish.steps_slot >= 0)    // the steps word of an AUTO unit
            bind_handle(*plan, d.unit->finish.steps_slot, steps_cell(*plan), 1);
        if (!d.unit->cells.empty()) {    // the rows of every cell component, zeroed
            const DLDataType real = real_dtype(d.unit->scalar_type);
            const std::int64_t rows = std::max<std::int64_t>(1, (d.n_samples + d.unit->cell_row - 1) / d.unit->cell_row);
            for (const CellFacts& c : d.unit->cells) {
                plan->rows.push_back(make_plane(ctx, real, { rows }, device));
                plan->comps.push_back(c.comp >= 0 ? make_plane(ctx, real, { rows }, device) : nullptr);
                bind_handle(*plan, c.slot, plan->rows.back()->tensor->dl_tensor.data, static_cast<std::uint64_t>(rows));
                plan->bindings[c.slot].bound = false;    // bound when the caller binds the cell
                if (c.comp >= 0) bind_handle(*plan, c.comp, plan->comps.back()->tensor->dl_tensor.data, static_cast<std::uint64_t>(rows));
            }
            if (device) device_cell_work(*plan);
        }
        std::vector<std::int64_t> sizes;    // the scratch the stages share: per_lane elements per unit; a placed wide value is one unit on the host, whole warps of 32 on the device
        for (const TeamPlaneFacts& t : d.unit->team_planes) {
            const std::int64_t count = t.unit == EG_UNIT_ONCE ? 1 : t.unit == EG_UNIT_KEY ? key_count(*d.unit, d.unit->keyed[static_cast<std::size_t>(t.keyed)], d.n_samples) : d.n_samples;
            const std::int64_t units = t.wide != EG_WIDE_WARP_TILED ? count : device ? (count + 31) / 32 * 32 : 1;
            sizes.push_back(t.unit == EG_UNIT_TARGET ? 0 : std::max<std::int64_t>(1, t.per_lane * units));    // a TARGET plane is made when its target is bound
            if (device && t.fill_promised && t.wide == EG_WIDE_NONE && t.dtype == EG_DTYPE_REAL && d.unit->scalar_type != EG_SCALAR_FLOAT32 && t.fill != 0)
                refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': team plane '" + t.name + "' is refilled with a nonzero double on the device, which is not written");
        }
        const auto elem_of = [&](const TeamPlaneFacts& t) {
            return t.dtype == EG_DTYPE_INT32 ? DLDataType{ kDLInt, 32, 1 }
                                             : real_dtype(d.unit->scalar_type);
        };
        if (device && !sizes.empty()) {    // the planes together must fit the device (the old bind's refusal, naming the stage that needs most)
            std::vector<std::int64_t> bytes;
            std::int64_t total = 0;
            for (std::size_t k = 0; k < sizes.size(); ++k) {
                bytes.push_back(sizes[k] * elem_of(d.unit->team_planes[k]).bits / 8);
                total += bytes.back();
            }
            const std::int64_t free_bytes = device_free_bytes(ctx);
            if (total > free_bytes) {
                const auto group = [](std::int64_t v) {
                    std::string s = std::to_string(v);
                    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) s.insert(static_cast<std::size_t>(i), ",");
                    return s;
                };
                std::string list, stage;
                std::int64_t most = -1;
                for (std::size_t k = 0; k < bytes.size(); ++k) list += (k ? ", " : "") + d.unit->team_planes[k].name + " " + group(bytes[k]);
                for (const StageFacts& st : d.unit->stages) {
                    if (!st.team) continue;
                    std::int64_t need = 0;
                    for (std::int32_t p : st.plane_of) need += p >= 0 ? bytes[static_cast<std::size_t>(p)] : 0;
                    if (need > most) { most = need; stage = st.ref_name; }
                }
                refuse(EG_BAD_REQUEST, "unit '" + d.unit->name + "' needs " + group(total) + " bytes of team scratch (" + list + ") at " +
                       group(d.n_samples) + " lanes, but the device has " + group(free_bytes) + " bytes free; stage '" + stage +
                       "' alone needs " + group(most));
            }
        }
        for (std::size_t k = 0; k < d.unit->team_planes.size(); ++k) {
            const TeamPlaneFacts& t = d.unit->team_planes[k];
            plan->team.push_back(t.unit == EG_UNIT_TARGET ? nullptr : make_team_plane(ctx, *d.unit, t, d.n_samples, device, 0));
        }
        for (const DispatchFacts& f : d.unit->dispatches) {    // an AUTO group's mix counters: eagle's, zeroed, unless the caller binds some
            if (f.strategy != EG_DISPATCH_AUTO) continue;
            plan->mix.push_back(make_plane(ctx, DLDataType{ kDLUInt, 32, 1 }, { f.k }, device));
            bind_handle(*plan, f.mix, plan->mix.back()->tensor->dl_tensor.data, static_cast<std::uint64_t>(f.k));
        }
        if (d.unit->seg.present) {    // a SEGMENTED group: the planes eagle makes for it
            const SegFacts& g = d.unit->seg;
            SegState& st = plan->seg;
            const DLDataType i64{ kDLInt, 64, 1 };
            if (device) {    // the offsets and class planes live in device memory; the sort's own buffers are made at the first launch
                plan->segdev.push_back(make_plane(ctx, i64, { g.k + 1 }, true));
                plan->segdev.push_back(make_plane(ctx, i64, { std::max<std::int64_t>(d.n_samples, 1) }, true));
                bind_handle(*plan, g.offsets_slot, plan->segdev[0]->tensor->dl_tensor.data, static_cast<std::uint64_t>(g.k) + 1);
                bind_handle(*plan, g.class_slot, plan->segdev[1]->tensor->dl_tensor.data, static_cast<std::uint64_t>(d.n_samples));
            } else {
                st.offsets.assign(static_cast<std::size_t>(g.k) + 1, 0);
                st.cls.assign(static_cast<std::size_t>(d.n_samples), 0);
                bind_handle(*plan, g.offsets_slot, st.offsets.data(), static_cast<std::uint64_t>(g.k) + 1);
                bind_handle(*plan, g.class_slot, st.cls.data(), static_cast<std::uint64_t>(d.n_samples));
            }
            st.run_len.assign(static_cast<std::size_t>(g.k), 0);
            std::vector<const eg_unit_s*> members(g.branches.begin(), g.branches.end());
            members.push_back(g.dynamic ? g.sweep : g.classify);
            for (const eg_unit_s* m : members) {    // a member that is a split unit: its scratch planes, made here (a placed wide value: one unit on the host, whole warps of 32 on the device)
                plan->segteam.emplace_back();
                if (m == d.unit) continue;    // the planned unit's own scratch is plan->team
                for (const TeamPlaneFacts& t : m->team_planes) {
                    const std::int64_t count = t.unit == EG_UNIT_ONCE ? 1 : d.n_samples;
                    const std::int64_t units = t.wide != EG_WIDE_WARP_TILED ? count : device ? (count + 31) / 32 * 32 : 1;
                    const std::int64_t size = std::max<std::int64_t>(1, t.per_lane * units);
                    std::vector<std::int64_t> shape{ size };
                    if (t.layout == EG_LAYOUT_PLANE_MAJOR && t.per_lane != 1) shape = { t.per_lane, size / t.per_lane };
                    const DLDataType elem = t.dtype == EG_DTYPE_INT32 ? DLDataType{ kDLInt, 32, 1 }
                                                                      : real_dtype(d.unit->scalar_type);
                    plan->segteam.back().push_back(make_plane(ctx, elem, shape, device));
                }
            }
            if (g.dynamic && device) {    // the stale mask, the sweep's active map and its count live in device memory
                plan->segdev.push_back(make_plane(ctx, DLDataType{ kDLBool, 8, 1 }, { std::max<std::int64_t>(d.n_samples, 1) }, true));
                plan->segdev.push_back(make_plane(ctx, DLDataType{ kDLInt, 32, 1 }, { std::max<std::int64_t>(d.n_samples, 1) }, true));
                plan->segdev.push_back(make_plane(ctx, DLDataType{ kDLUInt, 32, 1 }, { 1 }, true));
                bind_handle(*plan, g.stale_slot, plan->segdev[2]->tensor->dl_tensor.data, static_cast<std::uint64_t>(d.n_samples));
                const auto handle = [](void* data, std::uint64_t count) {
                    plugin::ScalarHandle h{};
                    h.data = data, h.samples = count, h.stride = 1, h.deviceType = plugin::kEagleAbiDeviceCUDA, h.deviceId = 0;
                    return h;
                };
                st.map_h = handle(plan->segdev[3]->tensor->dl_tensor.data, static_cast<std::uint64_t>(d.n_samples));
                st.count_h = handle(plan->segdev[4]->tensor->dl_tensor.data, 1);
                st.sortkey.assign(static_cast<std::size_t>(d.n_samples), 0);
            } else if (g.dynamic) {
                st.stale.assign(static_cast<std::size_t>(d.n_samples), 0);
                st.map.assign(static_cast<std::size_t>(d.n_samples), 0);
                bind_handle(*plan, g.stale_slot, st.stale.data(), static_cast<std::uint64_t>(d.n_samples));
                const auto handle = [](void* data, std::uint64_t count) {
                    plugin::ScalarHandle h{};
                    h.data = data, h.samples = count, h.stride = 1, h.deviceType = plugin::kEagleAbiDeviceCPU, h.deviceId = 0;
                    return h;
                };
                st.map_h = handle(st.map.data(), static_cast<std::uint64_t>(d.n_samples));
                st.count_h = handle(&st.count, 1);
            }
        }
        for (std::size_t i = 0; i < d.unit->slots.size(); ++i)    // a source slot of a split unit that eagle makes (scratch, a compensation row): bound from the start
            if (!d.unit->stages.empty() && d.unit->slots[i].made) plan->bindings[i].bound = true;
        if (!d.unit->split_cells.empty()) {    // the rows of each stage's cell columns, their compensation rows, and the cell planes eagle folds into
            const DLDataType real = real_dtype(d.unit->scalar_type);
            for (const SplitCell& sc : d.unit->split_cells) {
                const StageFacts& st = d.unit->stages[static_cast<std::size_t>(sc.stage)];
                const std::int64_t rows = std::max<std::int64_t>(1, (d.n_samples + st.cell_row - 1) / st.cell_row);
                plan->rows.push_back(make_plane(ctx, real, { rows }, device));
                plan->comps.push_back(sc.comp_slot >= 0 ? make_plane(ctx, real, { rows }, device) : nullptr);
                plan->outs.push_back(sc.source < 0 ? make_plane(ctx, real, { 1 }, device) : nullptr);
            }
            if (device) device_cell_work(*plan);
        }
        ctx.plans.insert(plan);
    } catch (...) {
        for (eg_plane_s* r : plan->mix) plane_release(ctx, r);
        for (eg_plane_s* r : plan->segdev) plane_release(ctx, r);
        for (auto& v : plan->segteam) for (eg_plane_s* r : v) plane_release(ctx, r);
        for (void* b : plan->seg.dev_bufs) device_free(ctx, b);
        for (eg_plane_s* r : plan->team) if (r != nullptr) plane_release(ctx, r);
        for (eg_plane_s* r : plan->comps) if (r != nullptr) plane_release(ctx, r);
        for (eg_plane_s* r : plan->outs) if (r != nullptr) plane_release(ctx, r);
        for (eg_plane_s* r : plan->work) plane_release(ctx, r);
        for (eg_plane_s* r : plan->rows) plane_release(ctx, r);
        for (eg_plan_s::Binding& b : plan->bindings)
            if (b.plane != nullptr) plane_release(ctx, b.plane);
        device_plan_free(*plan);
        delete plan;
        throw;
    }
    ++d.unit->refs;
    *out = plan;
}

/** @brief eg_plan_free: drop the plan's plane and unit references. */
void plan_free(eg_ctx& ctx, eg_plan_s* plan) {
    ctx.plans.erase(plan);
    for (eg_plan_s::Binding& b : plan->bindings)
        if (b.plane != nullptr) plane_release(ctx, b.plane);
    if (plan->fault != nullptr) plane_release(ctx, plan->fault);
    for (eg_plane_s* r : plan->work) plane_release(ctx, r);
    for (eg_plane_s* r : plan->rows) plane_release(ctx, r);
    for (eg_plane_s* r : plan->team) if (r != nullptr) plane_release(ctx, r);
    for (eg_plane_s* r : plan->mix) plane_release(ctx, r);
    for (eg_plane_s* r : plan->segdev) plane_release(ctx, r);
    for (auto& v : plan->segteam) for (eg_plane_s* r : v) plane_release(ctx, r);
    for (void* b : plan->seg.dev_bufs) device_free(ctx, b);
    for (eg_plane_s* r : plan->comps) if (r != nullptr) plane_release(ctx, r);
    for (eg_plane_s* r : plan->outs) if (r != nullptr) plane_release(ctx, r);
    unit_release(ctx, plan->unit);
    device_plan_free(*plan);
    delete plan;
}

/** @brief Drop one reference to @p plan (the caller's or a runner's); the last frees it. */
void plan_release(eg_ctx& ctx, eg_plan_s* plan) {
    if (--plan->refs == 0) plan_free(ctx, plan);
}

/** @brief Refuse @p plan while a slot is unbound. */
void check_bound(const eg_plan_s& plan) {
    for (std::size_t i = 0; i < plan.bindings.size(); ++i)
        if (!plan.bindings[i].bound)
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(i) + " ('" + plan.unit->slots[i].name +
                   "') is unbound");
}

/** @brief Give each runtime axis of the plane bound at @p slot the argument that carries its extent, from the shape of @p t. */
void fill_extents(eg_plan_s& plan, int slot, const DLTensor& t) {
    const SlotFacts& sf = plan.unit->slots[static_cast<std::size_t>(slot)];
    if (sf.axes.empty()) return;
    const std::vector<std::int64_t> extents = axis_extents(plan, sf, t, "slot " + std::to_string(slot) + " ('" + sf.name + "')");
    for (std::size_t a = 0; a < sf.axes.size(); ++a) {
        if (sf.axes[a].extent_slot < 0) continue;
        eg_plan_s::Binding& e = plan.bindings[static_cast<std::size_t>(sf.axes[a].extent_slot)];
        const std::int64_t extent = extents[a];
        std::memcpy(e.value, &extent, sizeof(extent));
        e.bound = true;
        plan.params[static_cast<std::size_t>(sf.axes[a].extent_slot)] = e.value;
    }
}

/** @brief Restart what eagle owns beside the target in @p slot, whose plane is @p t: its compensation companion,
 *  zeroed (in place when the target's shape and dtype are those it was made for, else made again), and an
 *  ordered scatter's target scratch, made again sized by the target. */
void renew_owned(eg_ctx& ctx, eg_plan_s& plan, int slot, const DLTensor& t) {
    for (const CompanionFacts& c : plan.unit->companions) {     // a new target: its companion, zeroed, of its shape and dtype
        if (c.target != slot) continue;
        eg_plane_s* held = plan.bindings[c.slot].plane;
        if (held != nullptr) {    // the same shape and dtype as before: zero the companion in place (a rebind is the old clear: no new allocation)
            const DLTensor& h = held->tensor->dl_tensor;
            bool same = h.ndim == t.ndim && h.dtype.code == t.dtype.code && h.dtype.bits == t.dtype.bits;
            for (int a = 0; same && a < t.ndim; ++a) same = h.shape[a] == t.shape[a];
            if (same) {
                const std::size_t bytes = static_cast<std::size_t>(elements(h)) * (h.dtype.bits / 8);
                if (plan.structure == EG_STRUCTURE_DEVICE) device_zero(ctx, h.data, bytes);
                else std::memset(h.data, 0, bytes);
                continue;
            }
        }
        eg_plane_s* comp = make_plane(ctx, t.dtype, std::vector<std::int64_t>(t.shape, t.shape + t.ndim),
                                      plan.structure == EG_STRUCTURE_DEVICE);
        eg_plane_s* prev = plan.bindings[c.slot].plane;
        plan.bindings[c.slot].plane = comp;
        bind_handle(plan, c.slot, comp->tensor->dl_tensor.data, static_cast<std::uint64_t>(plan.n_samples));
        fill_extents(plan, c.slot, comp->tensor->dl_tensor);
        if (prev != nullptr) plane_release(ctx, prev);
    }
    for (std::size_t k = 0; k < plan.unit->team_planes.size(); ++k) {    // an ordered scatter's partial sums: sized by the cells of the target just bound
        const TeamPlaneFacts& tp = plan.unit->team_planes[k];
        if (tp.unit != EG_UNIT_TARGET || tp.target != slot) continue;
        eg_plane_s* fresh = make_team_plane(ctx, *plan.unit, tp, plan.n_samples, plan.structure == EG_STRUCTURE_DEVICE, elements(t));
        eg_plane_s* prev = plan.team[k];
        plan.team[k] = fresh;
        if (prev != nullptr) plane_release(ctx, prev);
    }
}

/** @brief eg_clear: zero each named output plane and what eagle owns beside it, in the stream order of @p ctx. */
void clear(eg_ctx& ctx, eg_plan_s* plan, std::int32_t n_slots, const std::int32_t* slots) {
    check_handle(ctx.plans, plan, "plan");
    if (ctx.rec != nullptr || ctx.dev_rec != nullptr)
        refuse(EG_BAD_REQUEST, "eg_clear while recording (eg_record_begin .. eg_record_end): clear between launches, outside the recording");
    if (n_slots < 0 || (n_slots > 0 && slots == nullptr)) refuse(EG_BAD_REQUEST, "eg_clear: n_slots without its array");
    ++plan->binds;
    const int n_all = static_cast<int>(plan->bindings.size());
    std::vector<bool> seen(n_all, false);
    for (std::int32_t i = 0; i < n_slots; ++i) {
        const int slot = slots[i];
        if (slot < 0 || slot >= n_all)
            refuse(EG_BAD_REQUEST, "unknown slot " + std::to_string(slot) + " (the unit has " + std::to_string(n_all) + ")");
        const SlotFacts& s = plan->unit->slots[slot];
        const std::string name = "slot " + std::to_string(slot) + " ('" + s.name + "')";
        if (s.made) refuse(EG_BAD_REQUEST, name + " is a companion or compensation plane: eagle clears it with its target, do not pass it");
        if (s.role != EG_ROLE_MUTABLE && s.role != EG_ROLE_OUT && s.role != EG_ROLE_PLANE_OUT)
            refuse(EG_BAD_REQUEST, name + " is not an output plane");
        if (plan->bindings[static_cast<std::size_t>(slot)].plane == nullptr) refuse(EG_BAD_REQUEST, name + " is unbound");
        if (seen[slot]) refuse(EG_BAD_REQUEST, name + " cleared twice in one call");
        seen[slot] = true;
    }
    for (std::int32_t i = 0; i < n_slots; ++i) {
        const DLTensor& t = plan->bindings[static_cast<std::size_t>(slots[i])].plane->tensor->dl_tensor;
        void* data = static_cast<char*>(t.data) + t.byte_offset;
        const std::size_t bytes = static_cast<std::size_t>(elements(t)) * (t.dtype.bits / 8);
        if (plan->structure == EG_STRUCTURE_DEVICE) device_zero(ctx, data, bytes);
        else std::memset(data, 0, bytes);
        renew_owned(ctx, *plan, slots[i], t);
    }
}

/** @brief eg_set_count: write @p m into the bound index count plane, in the stream order of @p ctx. */
void set_count(eg_ctx& ctx, eg_plan_s* plan, std::uint32_t m) {
    check_handle(ctx.plans, plan, "plan");
    if (ctx.rec != nullptr || ctx.dev_rec != nullptr)
        refuse(EG_BAD_REQUEST, "eg_set_count while recording (eg_record_begin .. eg_record_end): set the count between replays");
    const eg_unit_s& u = *plan->unit;
    if (u.index_count_slot < 0) refuse(EG_BAD_REQUEST, "unit '" + u.name + "' has no index count slot");
    if (u.index_map_slot >= 0)
        refuse(EG_BAD_REQUEST, "unit '" + u.name + "' is a lookup set: its count is the caller's plane, bind it again");
    const eg_plane_s* count = plan->bindings[static_cast<std::size_t>(u.index_count_slot)].plane;
    if (count == nullptr) refuse(EG_BAD_REQUEST, "the index count slot ('" + u.slots[static_cast<std::size_t>(u.index_count_slot)].name + "') is unbound");
    const std::uint64_t capacity = static_cast<std::uint64_t>(plan->n_samples);
    if (m > capacity)
        refuse(EG_BAD_REQUEST, "the index count " + std::to_string(m) + " is above the sample count's " + std::to_string(capacity) + " entries");
    const bool device = plan->structure == EG_STRUCTURE_DEVICE;
    const DLTensor& c = count->tensor->dl_tensor;
    void* at = static_cast<char*>(c.data) + c.byte_offset;
    if (device) device_write(ctx, at, &m, sizeof(m));
    else std::memcpy(at, &m, sizeof(m));
}

bool gemm_routed(const eg_plan_s& plan) {
    return plan.structure == EG_STRUCTURE_DEVICE && plan.unit->gemm.present && gemm_routable(*plan.unit, plan.parts.size()) &&
           plan.ctx->gemm_by_cublas[plan.unit->scalar_type == EG_SCALAR_FLOAT64];
}

/** @brief eg_plan_entry_figures: the device entries of the plan's unit as read at load; a host plan, a routed gemm unit or a
 *  unit with no device function of its own has none (a routed gemm unit sets EG_ENTRY_GEMM). */
void plan_entry_figures(eg_ctx& ctx, const eg_plan_s* plan, eg_entry_figures* out) {
    check_handle(ctx.plans, plan, "plan");
    const eg_unit_s& u = *plan->unit;
    eg_entry_figures f{};
    if (gemm_routed(*plan)) {
        f.flags = EG_ENTRY_GEMM;
    } else if (plan->structure == EG_STRUCTURE_DEVICE && u.function != 0) {
        f.flags = EG_ENTRY_DEVICE | (u.persist_fn != 0 ? EG_ENTRY_PERSIST : 0u);
        f.regs = u.regs;
        f.max_threads = u.max_threads;
        f.persist_regs = u.persist_regs;
        f.persist_max_threads = u.persist_max_threads;
        f.local_bytes = u.local_bytes;
        f.shared_bytes = u.shared_bytes;
        f.const_bytes = u.const_bytes;
    }
    if (u.fault_slot >= 0) f.flags |= EG_ENTRY_CHECKED;
    write_output(out, f, "eg_entry_figures");
}

/** @brief The cell of the steps word @p plan's kernel reads: the caller's plane when bound, else eagle's own (a host
 *  address, or for a DEVICE plan a device address). */
void* steps_cell(const eg_plan_s& plan) {
    const eg_plane_s* bound = plan.bindings[static_cast<std::size_t>(plan.unit->finish.steps_slot)].plane;
    if (bound != nullptr) return static_cast<char*>(bound->tensor->dl_tensor.data) + bound->tensor->dl_tensor.byte_offset;
    return plan.structure == EG_STRUCTURE_DEVICE ? reinterpret_cast<void*>(plan.device_steps) : static_cast<void*>(const_cast<std::int64_t*>(&plan.steps_word));
}

namespace {
/** @brief Read @p bytes at @p at into @p out: device memory (ordered after the context's queued work) for a DEVICE plan, else host memory. */
void read_cell(const eg_plan_s& plan, const void* at, void* out, std::size_t bytes) {
    if (plan.structure == EG_STRUCTURE_DEVICE) device_read(*plan.ctx, at, out, bytes);
    else std::memcpy(out, at, bytes);
}
}  // namespace

/** @brief eg_plan_counters: the finished counter, the steps word and every AUTO group's mix counters, read now. */
void plan_counters(eg_ctx& ctx, const eg_plan_s* plan, struct eg_plan_counters* out, std::uint32_t* mix, std::int32_t mix_cap, std::int32_t* n_mix) {
    check_handle(ctx.plans, plan, "plan");
    if (n_mix == nullptr) refuse(EG_BAD_REQUEST, "eg_plan_counters: n_mix is NULL");
    const eg_unit_s& u = *plan->unit;
    std::int64_t count = 0;
    for (const DispatchFacts& f : u.dispatches)
        if (f.strategy == EG_DISPATCH_AUTO) count += f.k;
    *n_mix = static_cast<std::int32_t>(count);
    if (mix_cap > 0 && mix == nullptr) refuse(EG_BAD_REQUEST, "eg_plan_counters: mix is NULL with mix_cap > 0");
    if (mix_cap < 0 || (mix_cap > 0 && mix_cap < count))
        refuse(EG_BAD_REQUEST, "eg_plan_counters: mix_cap " + std::to_string(mix_cap) + " is below the " + std::to_string(count) +
               " mix counters of the unit's AUTO groups");
    struct eg_plan_counters c{};
    c.finished = -1;
    c.steps = -1;
    if (u.finish.present) {
        std::uint32_t finished = 0;
        read_cell(*plan, plan->structure == EG_STRUCTURE_DEVICE ? reinterpret_cast<const void*>(plan->device_counter) : &plan->finished, &finished, sizeof(finished));
        c.finished = finished;
        if (u.finish.steps_slot >= 0) read_cell(*plan, steps_cell(*plan), &c.steps, sizeof(c.steps));
    }
    if (mix_cap > 0) {
        std::size_t group = 0;
        for (const DispatchFacts& f : u.dispatches) {
            if (f.strategy != EG_DISPATCH_AUTO) continue;
            const eg_plane_s* bound = plan->bindings[static_cast<std::size_t>(f.mix)].plane;
            const DLTensor& t = (bound != nullptr ? bound : plan->mix[group])->tensor->dl_tensor;
            read_cell(*plan, static_cast<const char*>(t.data) + t.byte_offset, mix, static_cast<std::size_t>(f.k) * sizeof(std::uint32_t));
            mix += f.k;
            ++group;
        }
    }
    write_output(out, c, "eg_plan_counters");
}

/** @brief The sample count a plane bound at @p slot of @p plan states in its handle or mirror: a lookup its length, a keyed output its
 *  key count, a wide or gathered plane its trailing extent, else the plan's sample count. */
std::uint64_t bound_count(const eg_plan_s& plan, int slot, const DLTensor& t) {
    const SlotFacts& sf = plan.unit->slots[static_cast<std::size_t>(slot)];
    if (sf.mirror) return static_cast<std::uint64_t>(sf.gathered ? t.shape[t.ndim - 1] : plan.n_samples);
    if (sf.role == EG_ROLE_LOOKUP) return static_cast<std::uint64_t>(elements(t));
    if (zeroed_out(sf)) return static_cast<std::uint64_t>(elements(t));    // an accumulate target is a flat plane of its own length, not of the lane count
    if (sf.keys_of >= 0) return static_cast<std::uint64_t>(key_count(*plan.unit, plan.unit->keyed[static_cast<std::size_t>(sf.keys_of)], plan.n_samples));
    return static_cast<std::uint64_t>(wide(sf) ? t.shape[t.ndim - 1] : plan.n_samples);
}

/** @brief eg_bind: check every entry, then replace the named slots. */
void bind(eg_ctx& ctx, eg_plan_s* plan, const eg_bind_desc* desc) {
    check_handle(ctx.plans, plan, "plan");
    if (ctx.rec != nullptr || ctx.dev_rec != nullptr)
        refuse(EG_BAD_REQUEST, "eg_bind while recording (eg_record_begin .. eg_record_end): bind before recording, a recorded launch replays the planes it was recorded with");
    const eg_bind_desc d = read_input(desc, "eg_bind_desc");
    check_bind(ctx, *plan, d);
    ++plan->binds;
    for (int i = 0; i < d.n_planes; ++i) {
        eg_plan_s::Binding& b = plan->bindings[d.plane_slots[i]];
        eg_plane_s* old = b.plane;
        b.plane = d.planes[i];
        ++b.plane->refs;
        const DLTensor& t = b.plane->tensor->dl_tensor;
        char* data = static_cast<char*>(t.data) + t.byte_offset;
        const SlotFacts& sf = plan->unit->slots[d.plane_slots[i]];
        if (sf.cell) {
            b.bound = true;          // the kernel reads eagle's rows (made at eg_plan_new); this plane receives the folded cell
        } else if (sf.mirror) {
            bind_mirror(*plan, d.plane_slots[i], data, bound_count(*plan, d.plane_slots[i], t));
        } else {
            bind_handle(*plan, d.plane_slots[i], data, bound_count(*plan, d.plane_slots[i], t));
        }
        if (old != nullptr) plane_release(ctx, old);
        renew_owned(ctx, *plan, d.plane_slots[i], t);
        fill_extents(*plan, d.plane_slots[i], t);
    }
    for (int i = 0; i < d.n_uniforms; ++i) {
        const int slot = d.uniform_slots[i];
        eg_plan_s::Binding& b = plan->bindings[slot];
        std::memcpy(b.value, d.uniform_values[i],
                    uniform_bytes(plan->unit->slots[slot].dtype, plan->unit->scalar_type));
        b.bound = true;
        plan->params[slot] = b.value;
    }
}

/** @brief Whether @p desc names a DEVICE plan (the runner then runs on the device). */
bool device_plans(const eg_ctx& ctx, const eg_runner_desc* desc) {
    const eg_runner_desc d = read_input(desc, "eg_runner_desc");
    for (std::int32_t i = 0; d.plans != nullptr && i < d.n_plans; ++i)
        if (ctx.plans.count(d.plans[i]) != 0 && d.plans[i]->structure == EG_STRUCTURE_DEVICE) return true;
    return false;
}

/** @brief Bind @p slot of @p plan to eagle's own memory of @p count samples (a runner's active set, a derived plan). */
void bind_raw(eg_plan_s& plan, int slot, void* data, std::uint64_t count) {
    if (plan.unit->slots[static_cast<std::size_t>(slot)].mirror) bind_mirror(plan, slot, data, count);
    else bind_handle(plan, slot, data, count);
}

/** @brief eg_launch: the host team (or the serial reference) over each partition; a DEVICE plan
 *  queues one kernel launch per partition on the context's stream (eg_sync waits for them). */
void launch(eg_ctx& ctx, eg_plan_s* plan, std::uint64_t* tiles) {
    check_handle(ctx.plans, plan, "plan");
    check_bound(*plan);
    ++plan->launches;
    if (plan->structure == EG_STRUCTURE_DEVICE) {
        if (ctx.dev_rec != nullptr) {    // captured on the stream in scope
            device_record_launch(ctx, *plan);
            if (tiles != nullptr) *tiles = 0;
            return;
        }
        fault_reset(ctx, *plan);
        device_launch_call(*plan);
        fault_check(ctx, *plan);
        if (tiles != nullptr) *tiles = 0;
        return;
    }
    if (record_launch(ctx, plan)) {
        if (tiles != nullptr) *tiles = 0;
        return;
    }
    plan->finished = 0;    // an eg_launch counts the samples it finishes: eg_plan_counters reads this launch's count
    fault_reset(ctx, *plan);
    const std::uint64_t run = host::run_plan(*plan, host::team_threads(ctx));
    fault_check(ctx, *plan);
    if (tiles != nullptr) *tiles = run;
}

}  // namespace door
}  // namespace eagle
