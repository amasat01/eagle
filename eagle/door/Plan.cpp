// SPDX-License-Identifier: Apache-2.0

// Plans: a unit under one structure over fixed partitions; bindings by slot,
// checked once at eg_bind; eg_launch runs eagle's host team over them, or one
// device launch per partition (device/Device.cpp).
#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"

#include "eagle/door/host/Team.h"

#include <algorithm>

namespace eagle {
namespace door {

namespace {

/** @brief The partitions of @p d: the whole range, or the caller's, checked to stay
 *  inside [0, n) without overlap. */
std::vector<exec::Partition> read_partitions(const eg_plan_desc& d) {
    const std::int64_t n = d.n_samples;
    const bool device = d.structure == EG_STRUCTURE_DEVICE;
    const std::int64_t box_t = device ? d.unit->dev_box_t : d.unit->box_t, box_stride = device ? d.unit->dev_box_stride : d.unit->box_stride;
    if (box_t > 0) {    // a thread box: base and count count threads, ceil(n / (t * stride)) * stride of them
        if (d.n_partitions != 0)
            refuse(EG_BAD_REQUEST, "unit '" + d.unit->name + "' computes " + std::to_string(box_t) +
                   " lanes per thread (a thread box), so its partitions count threads, not lanes; pass none");
        const std::int64_t span = box_t * box_stride;
        return { exec::Partition{ 0, (n + span - 1) / span * box_stride, n } };
    }
    if (d.n_partitions == 0) return { exec::Partition::whole(n) };
    if (d.n_partitions < 0 || d.part_base == nullptr || d.part_count == nullptr)
        refuse(EG_BAD_REQUEST, "eg_plan_desc: n_partitions " + std::to_string(d.n_partitions) +
               " needs part_base and part_count");
    std::vector<exec::Partition> parts;
    for (std::int64_t i = 0; i < d.n_partitions; ++i) {
        const std::int64_t base = d.part_base[i], count = d.part_count[i];
        if (base < 0 || count < 1 || base > n || count > n - base)
            refuse(EG_BAD_REQUEST, "partition " + std::to_string(i) + " (base " +
                   std::to_string(base) + ", count " + std::to_string(count) +
                   ") leaves the range [0, " + std::to_string(n) + ")");
        parts.push_back(exec::Partition{ base, count, n });
    }
    std::vector<exec::Partition> sorted = parts;
    std::sort(sorted.begin(), sorted.end(),
              [](const exec::Partition& a, const exec::Partition& b) { return a.base < b.base; });
    for (std::size_t i = 1; i < sorted.size(); ++i)
        if (sorted[i].base < sorted[i - 1].base + sorted[i - 1].count)
            refuse(EG_BAD_REQUEST, "partitions overlap at sample " + std::to_string(sorted[i].base));
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

/** @brief True for the roles whose plane has a runtime row count (the handle's extent is the trailing one). */
bool wide(const SlotFacts& s) { return s.role == EG_ROLE_WIDE_IN || s.role == EG_ROLE_WIDE_OUT; }

/** @brief True when @p t carries the element type slot @p s of a unit of @p scalar declares. */
bool dtype_ok(const DLDataType& t, const SlotFacts& s, std::int32_t scalar) {
    if (t.lanes != 1) return false;
    switch (s.dtype) {
        case EG_DTYPE_REAL: return s.role != EG_ROLE_TERMINATED && t.code == kDLFloat && t.bits == (scalar == EG_SCALAR_FLOAT32 ? 32 : 64);
        case EG_DTYPE_INT64: return t.code == kDLInt && t.bits == 64;
        case EG_DTYPE_INT32: return t.code == kDLInt && t.bits == 32;
        case EG_DTYPE_UINT32: return t.code == kDLUInt && t.bits == 32;
        default: return (t.code == kDLBool || t.code == kDLUInt) && t.bits == 8;  // EG_DTYPE_BOOL (a mask too)
    }
}

/** @brief Check @p plane against slot @p slot of @p plan (dtype, shape, device, write access). */
void check_plane(const eg_plan_s& plan, int slot, const eg_plane_s& plane) {
    const SlotFacts& s = plan.unit->slots[slot];
    const DLTensor& t = plane.tensor->dl_tensor;
    const std::string what = "slot " + std::to_string(slot) + " ('" + s.name + "')";
    if ((t.device.device_type == kDLCUDA) != (plan.structure == EG_STRUCTURE_DEVICE))
        refuse(EG_BAD_REQUEST, what + (t.device.device_type == kDLCUDA ? ": a device plane in a host plan"
                                                                       : ": a host plane in a DEVICE plan"));
    if (!dtype_ok(t.dtype, s, plan.unit->scalar_type))
        refuse(EG_BAD_REQUEST, what + ": dtype (code " + std::to_string(t.dtype.code) + ", bits " +
               std::to_string(t.dtype.bits) + ") is not the slot's");
    if (s.gathered) {                    // (rows, cols, m): m the caller's column count, checked against the lane axes it varies along
        const std::int64_t m = t.ndim == 3 ? t.shape[2] : -1;
        std::int64_t want = 1, fixed = 1;
        for (std::int64_t e : plan.unit->lane_extents) fixed *= e > 0 ? e : 1;
        for (std::size_t i = 0; i < s.keep.size(); ++i)
            if (s.keep[i]) want *= plan.unit->lane_extents[i] > 0 ? plan.unit->lane_extents[i] : plan.n_samples / fixed;
        if (t.ndim != 3 || t.shape[0] != s.shape[0] || t.shape[1] != s.shape[1] || m < 1 || (!s.keep.empty() && m != want))
            refuse(EG_BAD_REQUEST, what + ": a plane of shape (" + std::to_string(s.shape[0]) + ", " + std::to_string(s.shape[1]) +
                   ", " + (s.keep.empty() ? std::string("m") : std::to_string(want)) + ") is expected (m: the columns of the lane axes it varies along)");
    } else if (s.keys_of >= 0) {                // a block sum's output: width entries per key, the keys' count along the last axis
        const KeyedFacts& k = plan.unit->keyed[static_cast<std::size_t>(s.keys_of)];
        const std::int64_t want = k.width * key_count(*plan.unit, k, plan.n_samples);
        if (t.ndim < 1 || elements(t) != want)
            refuse(EG_BAD_REQUEST, what + ": a block sum's output of " + std::to_string(want) + " elements (" + std::to_string(k.width) + " per key) is expected");
    } else if (wide(s)) {                       // (rows, n): the rows are the caller's, one value per sample along the last axis
        if (t.ndim != 2 || t.shape[1] != plan.n_samples)
            refuse(EG_BAD_REQUEST, what + ": a plane of shape (rows, " + std::to_string(plan.n_samples) + ") is expected");
    } else if (s.role == EG_ROLE_LOOKUP) {      // a table of the caller's: any compact shape, read by element count
        if (t.ndim < 1) refuse(EG_BAD_REQUEST, what + ": a table needs at least one dimension");
        if (s.shared && s.rank > 0) {    // a shared parameter has its natural shape, bound flat
            bool same = t.ndim == s.rank;
            for (int i = 0; same && i < s.rank; ++i) same = t.shape[i] == s.shape[static_cast<std::size_t>(i)];
            if (!same) refuse(EG_BAD_REQUEST, what + ": a shared parameter of its declared shape is expected");
        }
        if (slot == plan.unit->index_map_slot && t.ndim != 1)
            refuse(EG_BAD_REQUEST, what + ": the index map is one dimension");
        if (slot == plan.unit->index_count_slot && (t.ndim != 1 || t.shape[0] != 1))
            refuse(EG_BAD_REQUEST, what + ": the index count is one cell, shape (1,)");
    } else if (s.role == EG_ROLE_ACCUM_OUT) {   // a cell: the finished (1,); else the caller's length, along the named axes the meta states
        if (s.cell) {
            if (t.ndim != 1 || t.shape[0] != 1) refuse(EG_BAD_REQUEST, what + ": a cell is one element, shape (1,)");
        } else {
            if (t.ndim < 1 || (!s.axes.empty() && t.ndim != static_cast<int>(s.axes.size())))
                refuse(EG_BAD_REQUEST, what + ": a plane of " + std::to_string(s.axes.empty() ? 1 : s.axes.size()) + " dimension(s) or more is expected");
            std::int64_t fixed = 1;
            for (std::int64_t e : plan.unit->lane_extents) fixed *= e > 0 ? e : 1;
            for (std::size_t i = 0; i < s.axes.size(); ++i) {    // a lane axis's label shares that axis's extent
                const SlotFacts::Axis& a = s.axes[i];
                const std::int64_t want = a.lane < 0 ? a.extent
                                          : plan.unit->lane_extents[static_cast<std::size_t>(a.lane)] > 0
                                              ? plan.unit->lane_extents[static_cast<std::size_t>(a.lane)] : plan.n_samples / fixed;
                if (want >= 0 && t.shape[i] != want)
                    refuse(EG_BAD_REQUEST, what + ": axis " + std::to_string(i) + " has extent " + std::to_string(t.shape[i]) +
                           ", the unit's named axis has " + std::to_string(want));
            }
        }
    } else {                             // (*shape, n): the slot's extents, then one value per sample
        bool ok = t.ndim == s.rank + 1 && t.shape[s.rank] == plan.n_samples;
        for (int i = 0; ok && i < s.rank; ++i) ok = t.shape[i] == s.shape[static_cast<std::size_t>(i)];
        if (!ok) {
            std::string want = "(";
            for (std::int64_t e : s.shape) want += std::to_string(e) + ", ";
            refuse(EG_BAD_REQUEST, what + ": a plane of shape " + want + std::to_string(plan.n_samples) + ") is expected");
        }
    }
    if (!compact(t)) refuse(EG_BAD_REQUEST, what + ": the plane is not contiguous");
    const bool written = s.role == EG_ROLE_MUTABLE || s.role == EG_ROLE_OUT || s.role == EG_ROLE_WIDE_OUT || s.role == EG_ROLE_ACCUM_OUT ||
                         slot == plan.unit->finish.mask_slot;
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
        if (s.role == EG_ROLE_NSAMPLES)
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(slot) + " ('" + s.name +
                   "') is the sample count: eagle passes it, do not bind it");
        if (slot == plan.unit->finish.steps_slot)
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(slot) + " ('" + s.name +
                   "') is the steps word: eagle makes it, do not pass it");
        if (slot == plan.unit->active_map_slot || slot == plan.unit->active_count_slot)
            refuse(EG_BAD_REQUEST, "slot " + std::to_string(slot) + " ('" + s.name +
                   "') is part of the active set: the runner makes it, do not pass it");
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

/** @brief eg_plan_new: a host or DEVICE plan of the unit over checked partitions. */
void plan_new(eg_ctx& ctx, const eg_plan_desc* desc, eg_plan** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_plan_new: out is NULL");
    *out = nullptr;
    const eg_plan_desc d = read_input(desc, "eg_plan_desc");
    if (d.flags != 0) refuse(EG_BAD_REQUEST, "eg_plan_desc.flags: unknown bits " + std::to_string(d.flags));
    check_handle(ctx.units, d.unit, "eg_plan_desc.unit");
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
    if (!device)
        for (const CompanionFacts& c : d.unit->companions)
            if (c.fold == nullptr)
                refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': the companion '" + d.unit->slots[c.slot].name +
                       "' has no host fold entry");
    std::vector<exec::Partition> parts = read_partitions(d);
    for (const exec::Partition& p : parts)    // a row of a cell has one writer: a partition starts on a row
        if (d.unit->cell_row > 0 && p.base % d.unit->cell_row != 0)
            refuse(EG_BAD_REQUEST, "unit '" + d.unit->name + "' reduces over samples in rows of " + std::to_string(d.unit->cell_row) +
                   ": a partition starts on a multiple of it, got base " + std::to_string(p.base));
    for (const StageFacts& st : d.unit->stages)    // a row of a stage's cell has one writer too
        for (const exec::Partition& p : parts)
            if (st.cell_row > 0 && !st.unit->cells.empty() && p.base % st.cell_row != 0)
                refuse(EG_BAD_REQUEST, "unit '" + d.unit->name + "' stage '" + st.ref_name + "' reduces over samples in rows of " + std::to_string(st.cell_row) +
                       ": a partition starts on a multiple of it, got base " + std::to_string(p.base));
    if (d.unit->index_count_slot >= 0 && (parts.size() != 1 || parts[0].base != 0 || parts[0].count != d.n_samples))
        refuse(EG_BAD_REQUEST, "unit '" + d.unit->name + "' reads an index set: one whole partition");
    static constexpr exec::Access kAccess[] = { exec::Access::SampleLocal,
        exec::Access::CrossSampleRead, exec::Access::CrossSampleWrite };
    try {
        exec::check_placement(kAccess[d.unit->access], device ? exec::Structure::DeviceKernel : exec::Structure::HostTeam,
                              static_cast<int>(parts.size()));
    } catch (const std::runtime_error& e) {
        refuse(EG_BAD_REQUEST, e.what());
    }
    auto* plan = new eg_plan_s;
    plan->ctx = &ctx;
    plan->unit = d.unit;
    plan->structure = d.structure;
    plan->n_samples = d.n_samples;
    try {
        plan->parts = std::move(parts);
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
        if (d.unit->finish.present && d.unit->finish.steps_slot >= 0)    // the steps word of an AUTO unit
            bind_handle(*plan, d.unit->finish.steps_slot,
                        d.structure == EG_STRUCTURE_DEVICE ? reinterpret_cast<void*>(plan->device_steps)
                                                           : static_cast<void*>(&plan->steps_word), 1);
        if (!d.unit->cells.empty()) {    // the rows of every cell component, zeroed
            const DLDataType real{ kDLFloat, static_cast<std::uint8_t>(d.unit->scalar_type == EG_SCALAR_FLOAT32 ? 32 : 64), 1 };
            const std::int64_t rows = std::max<std::int64_t>(1, (d.n_samples + d.unit->cell_row - 1) / d.unit->cell_row);
            for (const CellFacts& c : d.unit->cells) {
                plan->rows.push_back(make_plane(ctx, real, { rows }, device));
                plan->comps.push_back(c.comp >= 0 ? make_plane(ctx, real, { rows }, device) : nullptr);
                bind_handle(*plan, c.slot, plan->rows.back()->tensor->dl_tensor.data, static_cast<std::uint64_t>(d.n_samples));
                plan->bindings[c.slot].bound = false;    // bound when the caller binds the cell
                if (c.comp >= 0) bind_handle(*plan, c.comp, plan->comps.back()->tensor->dl_tensor.data, static_cast<std::uint64_t>(d.n_samples));
            }
            if (device) device_cell_work(*plan);
        }
        std::vector<std::int64_t> sizes;    // the scratch the stages share: per_lane elements per unit; a placed wide value is one unit on the host, whole warps of 32 on the device
        for (const TeamPlaneFacts& t : d.unit->team_planes) {
            const std::int64_t count = t.unit == EG_UNIT_ONCE ? 1 : t.unit == EG_UNIT_KEY ? key_count(*d.unit, d.unit->keyed[static_cast<std::size_t>(t.keyed)], d.n_samples) : d.n_samples;
            const std::int64_t units = t.wide != EG_WIDE_WARP_TILED ? count : device ? (count + 31) / 32 * 32 : 1;
            sizes.push_back(std::max<std::int64_t>(1, t.per_lane * units));
            if (device && t.fill_promised && t.wide == EG_WIDE_NONE && t.dtype == EG_DTYPE_REAL && d.unit->scalar_type != EG_SCALAR_FLOAT32 && t.fill != 0)
                refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': team plane '" + t.name + "' is refilled with a nonzero double on the device, which is not written");
        }
        const auto elem_of = [&](const TeamPlaneFacts& t) {
            return t.dtype == EG_DTYPE_INT32 ? DLDataType{ kDLInt, 32, 1 }
                                             : DLDataType{ kDLFloat, static_cast<std::uint8_t>(d.unit->scalar_type == EG_SCALAR_FLOAT32 ? 32 : 64), 1 };
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
            const std::int64_t size = sizes[k];
            const DLDataType elem = elem_of(t);
            std::vector<std::int64_t> shape{ size };
            if (t.layout == EG_LAYOUT_PLANE_MAJOR && t.per_lane != 1) shape = { t.per_lane, size / t.per_lane };
            plan->team.push_back(make_plane(ctx, elem, shape, device));
        }
        for (std::size_t i = 0; i < d.unit->slots.size(); ++i)    // a source slot of a split unit that eagle makes (scratch, a compensation row): bound from the start
            if (!d.unit->stages.empty() && d.unit->slots[i].made) plan->bindings[i].bound = true;
        if (!d.unit->split_cells.empty()) {    // the rows of each stage's cell columns, their compensation rows, and the cell planes eagle folds into
            const DLDataType real{ kDLFloat, static_cast<std::uint8_t>(d.unit->scalar_type == EG_SCALAR_FLOAT32 ? 32 : 64), 1 };
            for (const SplitCell& sc : d.unit->split_cells) {
                const StageFacts& st = d.unit->stages[static_cast<std::size_t>(sc.stage)];
                const std::int64_t rows = std::max<std::int64_t>(1, (d.n_samples + st.cell_row - 1) / st.cell_row);
                plan->rows.push_back(make_plane(ctx, real, { rows }, false));
                plan->comps.push_back(sc.comp_slot >= 0 ? make_plane(ctx, real, { rows }, false) : nullptr);
                plan->outs.push_back(sc.source < 0 ? make_plane(ctx, real, { 1 }, false) : nullptr);
            }
        }
        ctx.plans.insert(plan);
    } catch (...) {
        for (eg_plane_s* r : plan->team) plane_release(ctx, r);
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
    for (eg_plane_s* r : plan->work) plane_release(ctx, r);
    for (eg_plane_s* r : plan->rows) plane_release(ctx, r);
    for (eg_plane_s* r : plan->team) plane_release(ctx, r);
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

/** @brief eg_bind: check every entry, then replace the named slots. */
void bind(eg_ctx& ctx, eg_plan_s* plan, const eg_bind_desc* desc) {
    check_handle(ctx.plans, plan, "plan");
    const eg_bind_desc d = read_input(desc, "eg_bind_desc");
    check_bind(ctx, *plan, d);
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
            bind_mirror(*plan, d.plane_slots[i], data,
                        static_cast<std::uint64_t>(sf.gathered ? t.shape[t.ndim - 1] : plan->n_samples));
        } else {
            bind_handle(*plan, d.plane_slots[i], data,
                        sf.role == EG_ROLE_LOOKUP ? static_cast<std::uint64_t>(elements(t))
                        : sf.keys_of >= 0 ? static_cast<std::uint64_t>(key_count(*plan->unit, plan->unit->keyed[static_cast<std::size_t>(sf.keys_of)], plan->n_samples))
                                          : static_cast<std::uint64_t>(plan->n_samples));
        }
        if (old != nullptr) plane_release(ctx, old);
        for (const CompanionFacts& c : plan->unit->companions) {     // a new target: its companion, zeroed, of its shape and dtype
            if (c.target != d.plane_slots[i]) continue;
            eg_plane_s* comp = make_plane(ctx, t.dtype, std::vector<std::int64_t>(t.shape, t.shape + t.ndim),
                                          plan->structure == EG_STRUCTURE_DEVICE);
            eg_plane_s* prev = plan->bindings[c.slot].plane;
            plan->bindings[c.slot].plane = comp;
            bind_handle(*plan, c.slot, comp->tensor->dl_tensor.data, static_cast<std::uint64_t>(plan->n_samples));
            if (prev != nullptr) plane_release(ctx, prev);
        }
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
    if (plan->structure != EG_STRUCTURE_DEVICE)
        device_quiet(ctx, "eg_launch of a host plan", EG_NOT_WRITTEN, kHostWorkInGraph);
    check_bound(*plan);
    if (plan->structure == EG_STRUCTURE_DEVICE) {
        if (ctx.dev_rec != nullptr) {    // captured on the stream in scope
            device_record_launch(ctx, *plan);
            if (tiles != nullptr) *tiles = 0;
            return;
        }
        device_launch(*plan);
        if (tiles != nullptr) *tiles = 0;
        return;
    }
    if (record_launch(ctx, plan)) {
        if (tiles != nullptr) *tiles = 0;
        return;
    }
    const std::uint64_t run = host::run_plan(*plan, host::team_threads(ctx));
    if (tiles != nullptr) *tiles = run;
}

}  // namespace door
}  // namespace eagle
