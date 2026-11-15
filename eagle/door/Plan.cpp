// SPDX-License-Identifier: Apache-2.0

// Plans: a unit under one host structure over fixed partitions; bindings by
// slot, checked once at eg_bind; eg_launch runs eagle's host team over them.
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

/** @brief True when @p t is one compact row-major range (no strides, or the compact ones). */
bool compact(const DLTensor& t) {
    if (t.strides == nullptr) return true;
    std::int64_t expect = 1;
    for (int i = t.ndim - 1; i >= 0; --i) {
        if (t.shape[i] > 1 && t.strides[i] != expect) return false;
        expect *= t.shape[i];
    }
    return true;
}

/** @brief True when @p t carries the element type slot @p s of a unit of @p scalar declares. */
bool dtype_ok(const DLDataType& t, const SlotFacts& s, std::int32_t scalar) {
    if (t.lanes != 1) return false;
    if (s.role == EG_ROLE_TERMINATED) return (t.code == kDLBool || t.code == kDLUInt) && t.bits == 8;
    switch (s.dtype) {
        case EG_DTYPE_REAL: return t.code == kDLFloat && t.bits == (scalar == EG_SCALAR_FLOAT32 ? 32 : 64);
        case EG_DTYPE_INT64: return t.code == kDLInt && t.bits == 64;
        case EG_DTYPE_INT32: return t.code == kDLInt && t.bits == 32;
        case EG_DTYPE_UINT32: return t.code == kDLUInt && t.bits == 32;
        default: return (t.code == kDLBool || t.code == kDLUInt) && t.bits == 8;  // EG_DTYPE_BOOL
    }
}

/** @brief Check @p plane against slot @p slot of @p plan (dtype, shape, device, write access). */
void check_plane(const eg_plan_s& plan, int slot, const eg_plane_s& plane) {
    const SlotFacts& s = plan.unit->slots[slot];
    const DLTensor& t = plane.tensor->dl_tensor;
    const std::string what = "slot " + std::to_string(slot) + " ('" + s.name + "')";
    if (!dtype_ok(t.dtype, s, plan.unit->scalar_type))
        refuse(EG_BAD_REQUEST, what + ": dtype (code " + std::to_string(t.dtype.code) + ", bits " +
               std::to_string(t.dtype.bits) + ") is not the slot's");
    if (s.role == EG_ROLE_LOOKUP) {      // a table of the caller's: any compact shape, read by element count
        if (t.ndim < 1) refuse(EG_BAD_REQUEST, what + ": a table needs at least one dimension");
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
    const bool written = s.role == EG_ROLE_MUTABLE || s.role == EG_ROLE_OUT ||
                         slot == plan.unit->finish.mask_slot;
    if (written && (plane.tensor->flags & DLPACK_FLAG_BITMASK_READ_ONLY))
        refuse(EG_BAD_REQUEST, what + ": a read-only plane in a written slot");
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
}

/** @brief Bind slot @p slot of @p plan to the 40-byte mirror of a compact (components, n) plane: the
 *  component pitch is n and the sample pitch 1 (``_plan_pack._gref_mirror_box`` of a compact plane). */
void bind_mirror(eg_plan_s& plan, int slot, void* data, std::uint64_t n) {
    eg_plan_s::Binding& b = plan.bindings[slot];
    b.mirror.data_ = static_cast<double*>(data);
    b.mirror.samples_ = n;
    b.mirror.compStride_ = n;
    b.mirror.sampleStride_ = 1;
    b.mirror.deviceType_ = plugin::kEagleAbiDeviceCPU;
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
    b.handle.deviceType = plugin::kEagleAbiDeviceCPU;
    b.handle.deviceId = 0;
    b.bound = true;
    plan.params[slot] = &b.handle;
}

}  // namespace

/** @brief eg_plan_new: a host plan of the unit over checked partitions. */
void plan_new(eg_ctx& ctx, const eg_plan_desc* desc, eg_plan** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_plan_new: out is NULL");
    *out = nullptr;
    const eg_plan_desc d = read_input(desc, "eg_plan_desc");
    if (d.flags != 0) refuse(EG_BAD_REQUEST, "eg_plan_desc.flags: unknown bits " + std::to_string(d.flags));
    check_handle(ctx.units, d.unit, "eg_plan_desc.unit");
    if (d.structure == EG_STRUCTURE_DEVICE)
        refuse(EG_NOT_WRITTEN, "structure DEVICE: the device group is not built");
    if (d.structure != EG_STRUCTURE_HOST_TEAM && d.structure != EG_STRUCTURE_HOST_SERIAL)
        refuse(EG_BAD_REQUEST, "unknown structure " + std::to_string(d.structure));
    if (d.policy != EG_REDUCE_COMPENSATED && d.policy != EG_REDUCE_FIXED)
        refuse(EG_BAD_REQUEST, "unknown reduce policy " + std::to_string(d.policy));
    if (d.n_samples < 0) refuse(EG_BAD_REQUEST, "n_samples " + std::to_string(d.n_samples) + " < 0");
    if (d.unit->scalar_type == EG_SCALAR_BANDED)
        refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "': a BANDED unit on a host structure");
    if (d.unit->host_entry == nullptr)
        refuse(EG_NOT_WRITTEN, "unit '" + d.unit->name + "' has no host entry");
    std::vector<exec::Partition> parts = read_partitions(d);
    static constexpr exec::Access kAccess[] = { exec::Access::SampleLocal,
        exec::Access::CrossSampleRead, exec::Access::CrossSampleWrite };
    try {
        exec::check_placement(kAccess[d.unit->access], exec::Structure::HostTeam,
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
        if (d.unit->finish.present)    // the finished counter: one uint32 cell (its element count)
            bind_handle(*plan, d.unit->finish.counter_slot, &plan->finished, 1);
        for (std::size_t i = 0; i < d.unit->slots.size(); ++i)    // eagle passes the sample count
            if (d.unit->slots[i].role == EG_ROLE_NSAMPLES) {
                eg_plan_s::Binding& b = plan->bindings[i];
                const std::uint32_t n32 = static_cast<std::uint32_t>(d.n_samples);
                std::memcpy(b.value, &n32, sizeof(n32));
                b.bound = true;
                plan->params[i] = b.value;
            }
        if (d.unit->finish.present && d.unit->finish.steps_slot >= 0)    // the steps word of an AUTO unit
            bind_handle(*plan, d.unit->finish.steps_slot, &plan->steps_word, 1);
        ctx.plans.insert(plan);
    } catch (...) {
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
    unit_release(ctx, plan->unit);
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
        if (sf.mirror) {
            bind_mirror(*plan, d.plane_slots[i], data, static_cast<std::uint64_t>(plan->n_samples));
        } else {
            std::int64_t elements = 1;       // a table is read by its element count, the rest by n
            for (int e = 0; e < t.ndim; ++e) elements *= t.shape[e];
            bind_handle(*plan, d.plane_slots[i], data,
                        sf.role == EG_ROLE_LOOKUP ? static_cast<std::uint64_t>(elements)
                                                  : static_cast<std::uint64_t>(plan->n_samples));
        }
        if (old != nullptr) plane_release(ctx, old);
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

/** @brief Bind @p slot of @p plan to eagle's own memory (a runner's active set). */
void bind_raw(eg_plan_s& plan, int slot, void* data, std::uint64_t count) { bind_handle(plan, slot, data, count); }

/** @brief eg_launch: the host team (or the serial reference) over each partition. */
void launch(eg_ctx& ctx, eg_plan_s* plan, std::uint64_t* tiles) {
    check_handle(ctx.plans, plan, "plan");
    check_bound(*plan);
    if (record_launch(ctx, plan)) {
        if (tiles != nullptr) *tiles = 0;
        return;
    }
    const std::uint64_t run = host::run_plan(*plan, host::team_threads(ctx));
    if (tiles != nullptr) *tiles = run;
}

}  // namespace door
}  // namespace eagle
