// SPDX-License-Identifier: Apache-2.0

// eg_unit_load: open the unit library, read its constant meta, check it, and
// keep the facts eg_bind and eg_launch need (on a device context the device
// image is loaded too, device/Device.cpp). Every member past a record's
// struct_size reads as its documented absent value.
#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"
#include "plugin/gref_abi.h"

#include <dlfcn.h>

#include <numeric>

namespace eagle {
namespace door {

namespace {

/** @brief Read a meta record: @p absent, then the bytes the record's struct_size covers. */
template <class T>
T read_record(const T* in, const T& absent, const std::string& what) {
    if (in == nullptr) refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + what + " is NULL");
    if (in->struct_size < sizeof(std::uint32_t))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + what + " struct_size " +
               std::to_string(in->struct_size));
    T out = absent;
    const std::size_t n = in->struct_size < sizeof(T) ? in->struct_size : sizeof(T);
    std::memcpy(static_cast<void*>(&out), static_cast<const void*>(in), n);
    return out;
}

/** @brief Check one slot and copy its facts (the roles and dtypes this slice runs). */
SlotFacts read_slot(const eg_unit_slot* in, const std::string& unit, int index) {
    eg_unit_slot absent{};
    absent.like_slot = -1;
    const std::string what = "unit '" + unit + "' slot " + std::to_string(index);
    const eg_unit_slot s = read_record(in, absent, what);
    const std::string named = what + " ('" + (s.name ? s.name : "?") + "')";
    if (s.flags & ~(EG_SLOT_PRIOR | EG_SLOT_PARAM_PLANE | EG_SLOT_GATHER | EG_SLOT_SHARED))
        refuse(EG_NOT_WRITTEN, named + ": unknown flag bits " + std::to_string(s.flags));
    if ((s.flags & EG_SLOT_GATHER) && (s.role != EG_ROLE_MAT_IN || s.rank != 2))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + named + " is GATHER but not a rank-2 MAT_IN");
    if ((s.flags & EG_SLOT_SHARED) && s.role != EG_ROLE_LOOKUP)
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + named + " is SHARED but not a LOOKUP");
    if (s.n_gather < 0 || (s.n_gather > 0 && (s.gather == nullptr || !(s.flags & EG_SLOT_GATHER))))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + named + " n_gather " + std::to_string(s.n_gather) +
               " without a gather pattern on a GATHER slot");
    if (s.role < EG_ROLE_MUTABLE || s.role > EG_ROLE_OUT)
        refuse(EG_NOT_WRITTEN, named + ": unknown role " + std::to_string(s.role));
    if (s.dtype < EG_DTYPE_REAL || s.dtype > EG_DTYPE_BAND_PARTIAL)
        refuse(EG_NOT_WRITTEN, named + ": unknown dtype " + std::to_string(s.dtype));
    if (s.rank < 0 || s.rank > 2 || (s.rank > 0 && s.shape == nullptr))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + named + " rank " +
               std::to_string(s.rank));
    const bool plane = s.role == EG_ROLE_MUTABLE || s.role == EG_ROLE_PER_SAMPLE || s.role == EG_ROLE_OUT ||
                       s.role == EG_ROLE_VEC_IN || s.role == EG_ROLE_MAT_IN;
    if (!plane && s.role != EG_ROLE_UNIFORM && s.role != EG_ROLE_TERMINATED && s.role != EG_ROLE_LOOKUP &&
        s.role != EG_ROLE_ACCUM_OUT && s.role != EG_ROLE_NSAMPLES && s.role != EG_ROLE_WIDE_IN && s.role != EG_ROLE_WIDE_OUT)
        refuse(EG_NOT_WRITTEN, named + ": role " + std::to_string(s.role) + " is not written yet (this build binds "
               "MUTABLE, PER_SAMPLE, VEC_IN, MAT_IN, OUT, WIDE_IN, WIDE_OUT, ACCUM_OUT, LOOKUP, UNIFORM, NSAMPLES and TERMINATED)");
    if (s.rank != 0 && !plane && !(s.flags & EG_SLOT_SHARED))
        refuse(EG_NOT_WRITTEN, named + ": a slot of rank > 0 outside the plane roles is not written yet");
    if (s.rank != 0 && s.role == EG_ROLE_PER_SAMPLE)
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + named + " is PER_SAMPLE of rank > 0");
    if (s.role == EG_ROLE_NSAMPLES && s.dtype != EG_DTYPE_UINT32)
        refuse(EG_NOT_WRITTEN, named + ": an NSAMPLES slot of a dtype other than UINT32 is not written");
    if (s.role == EG_ROLE_TERMINATED && (s.dtype == EG_DTYPE_REAL || s.dtype == EG_DTYPE_BAND_PARTIAL))
        refuse(EG_NOT_WRITTEN, named + ": a TERMINATED mask of a floating dtype is not written (BOOL, INT32, UINT32 "
               "and INT64 are)");
    if (s.dtype == EG_DTYPE_BAND_PARTIAL)
        refuse(EG_NOT_WRITTEN, named + ": a BAND_PARTIAL slot is not written");
    SlotFacts facts;
    facts.role = s.role;
    facts.dtype = s.dtype;
    facts.name = s.name ? s.name : "";
    facts.rank = s.rank;
    for (int i = 0; i < s.rank; ++i) {
        if (s.shape[i] < 1)
            refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + named + " extent " + std::to_string(s.shape[i]));
        facts.shape.push_back(s.shape[i]);
    }
    facts.mirror = s.role == EG_ROLE_OUT || s.role == EG_ROLE_VEC_IN || s.role == EG_ROLE_MAT_IN ||
                   (s.role == EG_ROLE_MUTABLE && s.rank > 0);
    facts.gathered = (s.flags & EG_SLOT_GATHER) != 0;
    facts.shared = (s.flags & EG_SLOT_SHARED) != 0;
    if (facts.gathered) facts.keep.assign(s.gather, s.gather + s.n_gather);
    return facts;
}

/** @brief Check the finish record against the slots and copy it into @p unit; every
 *  LOOKUP slot must be the finished counter (the one LOOKUP eagle makes so far). */
void read_finish(const eg_unit_meta& m, eg_unit_s& unit) {
    const int n_slots = static_cast<int>(unit.slots.size());
    const auto role_of = [&](std::int32_t slot) { return slot >= 0 && slot < n_slots ? unit.slots[slot].role : -1; };
    if (m.finish != nullptr) {
        eg_unit_finish absent{};
        absent.mask_slot = absent.counter_slot = absent.steps_slot = -1;
        const eg_unit_finish f = read_record(m.finish, absent, "unit '" + unit.name + "' finish");
        const std::string what = "meta that contradicts itself: unit '" + unit.name + "' finish ";
        if (f.flags & ~EG_FINISH_ENTRY_COUNTS)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "' finish: unknown flag bits " + std::to_string(f.flags));
        if (role_of(f.mask_slot) != EG_ROLE_TERMINATED)
            refuse(EG_BAD_REQUEST, what + "mask_slot " + std::to_string(f.mask_slot) + " is not a TERMINATED slot");
        if (role_of(f.counter_slot) != EG_ROLE_LOOKUP || unit.slots[f.counter_slot].dtype != EG_DTYPE_UINT32)
            refuse(EG_BAD_REQUEST, what + "counter_slot " + std::to_string(f.counter_slot) +
                   " is not a uint32 LOOKUP slot");
        if (f.steps == EG_STEPS_AUTO) {
            if (role_of(f.steps_slot) != EG_ROLE_LOOKUP || unit.slots[f.steps_slot].dtype != EG_DTYPE_INT64 ||
                f.steps_slot == f.counter_slot)
                refuse(EG_BAD_REQUEST, what + "steps_slot " + std::to_string(f.steps_slot) +
                       " is not an int64 LOOKUP slot (steps AUTO)");
            if (f.steps_max < 1) refuse(EG_BAD_REQUEST, what + "steps_max " + std::to_string(f.steps_max) + " (>= 1 with steps AUTO)");
        } else {
            if (f.steps < 1) refuse(EG_BAD_REQUEST, what + "steps " + std::to_string(f.steps) + " (>= 1 or AUTO)");
            if (f.steps_slot != -1)
                refuse(EG_BAD_REQUEST, what + "steps_slot " + std::to_string(f.steps_slot) + " with fixed steps");
        }
        if (f.n_events < 0) refuse(EG_BAD_REQUEST, what + "n_events " + std::to_string(f.n_events) + " (>= 0)");
        unit.finish = FinishFacts{ true, f.mask_slot, f.counter_slot, f.steps, f.steps_slot, f.steps_max, f.flags,
                                   f.fp64_ops_ir > 0 ? f.fp64_ops_ir : f.step_ops > 0 ? f.step_ops : 0, f.n_events };
    }
}

/** @brief Read the companions and cells of @p body: the slots eagle makes, the fold entries and the rows' fold. */
void read_made(const eg_unit_body& body, void* library, bool device_only, eg_unit_s& unit) {
    const std::string who = "meta that contradicts itself: unit '" + unit.name + "' ";
    const std::int32_t n = static_cast<std::int32_t>(unit.slots.size());
    const auto accum = [&](std::int32_t slot, const char* what) -> SlotFacts& {
        if (slot < 0 || slot >= n || unit.slots[slot].role != EG_ROLE_ACCUM_OUT || unit.slots[slot].dtype != EG_DTYPE_REAL)
            refuse(EG_BAD_REQUEST, who + what + " slot " + std::to_string(slot) + " is not a float ACCUM_OUT slot");
        return unit.slots[slot];
    };
    if (body.n_companions < 0 || (body.n_companions > 0 && body.companions == nullptr) || body.n_cells < 0 ||
        (body.n_cells > 0 && body.cells == nullptr))
        refuse(EG_BAD_REQUEST, who + "counts a companion or cell list it does not carry");
    for (int i = 0; i < body.n_companions; ++i) {
        const eg_unit_companion c = read_record(body.companions[i], eg_unit_companion{}, "unit '" + unit.name + "' companion");
        SlotFacts& comp = accum(c.slot, "companion");
        accum(c.target_slot, "companion target");
        if (c.slot == c.target_slot || comp.made)
            refuse(EG_BAD_REQUEST, who + "companion slot " + std::to_string(c.slot) + " is its own target or listed twice");
        if (c.host_fold_entry == nullptr && c.device_fold_function == nullptr)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': the companion '" + comp.name + "' has no host fold entry and no device fold function");
        void* sym = c.host_fold_entry != nullptr ? ::dlsym(library, c.host_fold_entry) : nullptr;
        if (c.host_fold_entry != nullptr && sym == nullptr && !device_only)
            refuse(EG_FAULT, "unit '" + unit.name + "': the library lacks the fold entry '" + c.host_fold_entry + "'");
        comp.made = true;
        unit.companions.push_back({ c.slot, c.target_slot, reinterpret_cast<exec::HostEntryV2>(sym),
                                    c.device_fold_function != nullptr ? c.device_fold_function : "", 0 });
    }
    if (body.n_cells > 0 && (unit.stages.empty() || body.cell_row != 0)) {    // a split unit's cells have no rows of its own: its stages state theirs
        if (body.cell_row < 1) refuse(EG_BAD_REQUEST, who + "cell_row " + std::to_string(body.cell_row) + " with cells");
        unit.cell_row = body.cell_row;
    }
    for (int i = 0; i < body.n_cells; ++i) {
        const eg_unit_cell cell = read_record(body.cells[i], eg_unit_cell{}, "unit '" + unit.name + "' cell");
        const std::string name = cell.name ? cell.name : "?";
        if (cell.op != EG_REDUCE_SUM && cell.op != EG_REDUCE_MAX && cell.op != EG_REDUCE_MIN)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': cell '" + name + "' folds with op " + std::to_string(cell.op) +
                   ", which is not written (SUM, MAX and MIN are)");
        if (cell.slots == nullptr && !unit.stages.empty() && cell.columns >= 1) {    // a split unit's cell: its stages own the rows
            unit.source_cells.push_back(name);
            continue;
        }
        if (cell.columns < 1 || cell.slots == nullptr)
            refuse(EG_BAD_REQUEST, who + "cell '" + name + "' has " + std::to_string(cell.columns) + " columns and no row slots");
        if (cell.op == EG_REDUCE_SUM && cell.comp_slots == nullptr)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': the cell '" + name + "' sums, and the cell's compensation slot is not stated in the unit meta (comp_slots is NULL)");
        if (cell.op != EG_REDUCE_SUM && cell.comp_slots != nullptr)
            refuse(EG_BAD_REQUEST, who + "cell '" + name + "' folds with MAX or MIN and names compensation slots");
        for (int c = 0; c < cell.columns; ++c) {
            SlotFacts& row = accum(cell.slots[c], "cell");
            if (row.made || row.cell) refuse(EG_BAD_REQUEST, who + "cell slot " + std::to_string(cell.slots[c]) + " is listed twice");
            std::int32_t comp = -1;
            if (cell.comp_slots != nullptr) {
                comp = cell.comp_slots[c];
                SlotFacts& cs = accum(comp, "cell compensation");
                if (cs.made || cs.cell || comp == cell.slots[c]) refuse(EG_BAD_REQUEST, who + "cell compensation slot " + std::to_string(comp) + " is listed twice");
                cs.made = true;
            }
            row.cell = true;
            unit.cells.push_back({ cell.slots[c], cell.op, comp, name });
        }
    }
}

/** @brief Check the dispatch groups of the unit and keep them. PREDICATED and SWITCH run as plain kernels; AUTO runs fused over mix counters eagle makes
 *  unless the caller binds some; a SEGMENTED group is a branch unit's record (it names the group's units, resolved against deps by
 *  resolve_segmented) or a classify unit's record (its class slot). */
void read_dispatch(const eg_unit_body& body, eg_unit_s& unit) {
    const std::string who = "meta that contradicts itself: unit '" + unit.name + "' ";
    if (body.n_dispatches < 0 || (body.n_dispatches > 0 && body.dispatches == nullptr))
        refuse(EG_BAD_REQUEST, who + "n_dispatches " + std::to_string(body.n_dispatches));
    const auto slot_is = [&](std::int32_t slot, bool written, std::int32_t dtype) {
        if (slot < 0 || slot >= static_cast<std::int32_t>(unit.slots.size())) return false;
        const SlotFacts& s = unit.slots[static_cast<std::size_t>(slot)];
        const bool role = written ? (s.role == EG_ROLE_MUTABLE || s.role == EG_ROLE_OUT) : s.role == EG_ROLE_LOOKUP;
        return role && s.dtype == dtype && s.rank == 0;
    };
    const auto ref_of = [](const eg_unit_ref* r) {
        RefFacts f;
        f.name = r->name != nullptr ? r->name : "";
        std::memcpy(f.digest.data(), r->digest, EG_DIGEST_BYTES);
        return f;
    };
    for (int i = 0; i < body.n_dispatches; ++i) {
        eg_unit_dispatch absent{};
        absent.mix_slot = absent.class_slot = absent.stale_slot = absent.offsets_slot = -1;
        absent.k = 2; /* a record written before k existed: the least a group has */
        const eg_unit_dispatch d = read_record(body.dispatches[i], absent, "unit '" + unit.name + "' dispatch");
        if (d.name == nullptr) refuse(EG_BAD_REQUEST, who + "has a dispatch group without a name");
        const std::string group = d.name;
        const std::string at = who + "dispatch group '" + group + "' ";
        const bool plain = d.strategy == EG_DISPATCH_PREDICATED || d.strategy == EG_DISPATCH_SWITCH;
        if (!plain && d.strategy != EG_DISPATCH_SEGMENTED && d.strategy != EG_DISPATCH_AUTO)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': dispatch group '" + group + "' has strategy " + std::to_string(d.strategy) +
                   ", which is not written");
        const bool seg = d.strategy == EG_DISPATCH_SEGMENTED;
        if (d.k < 2) refuse(EG_BAD_REQUEST, at + "has k " + std::to_string(d.k) + ", a group has at least 2 branches");
        if (!seg && (d.branches != nullptr || d.sweep != nullptr))
            refuse(EG_BAD_REQUEST, at + "names branches or a sweep and is not SEGMENTED");
        const bool cls_record = seg && d.branches == nullptr && d.classify == nullptr && d.sweep == nullptr && d.stale_slot == -1 &&
                                d.offsets_slot == -1 && d.class_slot != -1;
        if (seg && !cls_record && !((d.stale_slot == -1 && d.classify != nullptr && d.sweep == nullptr) ||
                                    (d.stale_slot != -1 && d.classify == nullptr && d.sweep != nullptr)))
            refuse(EG_BAD_REQUEST, at + "is SEGMENTED with neither a static selector (no stale slot, a classify "
                   "unit, no sweep) nor a dynamic one (a stale slot and a sweep, no classify unit)");
        if ((d.mix_slot != -1) != (d.strategy == EG_DISPATCH_AUTO))
            refuse(EG_BAD_REQUEST, at + "has a mix slot unless it is AUTO, and AUTO has one");
        if (!seg && (d.class_slot != -1 || d.stale_slot != -1 || d.offsets_slot != -1 || d.classify != nullptr))
            refuse(EG_BAD_REQUEST, at + "names SEGMENTED members and is not SEGMENTED");
        if (d.mix_slot != -1 && !slot_is(d.mix_slot, false, EG_DTYPE_UINT32))
            refuse(EG_BAD_REQUEST, at + "mix slot " + std::to_string(d.mix_slot) + " is not a uint32 LOOKUP slot");
        DispatchFacts f;
        f.strategy = d.strategy, f.k = d.k, f.mix = d.mix_slot, f.name = group;
        if (d.strategy == EG_DISPATCH_AUTO) {    // fused; the counters are eagle's unless the caller binds some
            SlotFacts& m = unit.slots[static_cast<std::size_t>(d.mix_slot)];
            m.optional_made = true;
            m.min_elements = d.k;
        }
        if (cls_record) {
            if (!slot_is(d.class_slot, true, EG_DTYPE_INT64))
                refuse(EG_BAD_REQUEST, at + "classify record: class slot " + std::to_string(d.class_slot) + " is not a per-sample int64 MUTABLE or OUT slot");
            unit.slots[static_cast<std::size_t>(d.class_slot)].made = true;
            f.classify_record = true;
            f.cls = d.class_slot;
        } else if (seg) {
            if (d.branches == nullptr) refuse(EG_BAD_REQUEST, at + "is SEGMENTED and names no branches (k refs)");
            for (std::int32_t j = 0; j < d.k; ++j) {
                if (d.branches[j] == nullptr || d.branches[j]->name == nullptr) refuse(EG_BAD_REQUEST, at + "names a branch " + std::to_string(j) + " without a unit");
                f.branches.push_back(ref_of(d.branches[j]));
            }
            if (!slot_is(d.offsets_slot, false, EG_DTYPE_INT64))
                refuse(EG_BAD_REQUEST, at + "offsets slot " + std::to_string(d.offsets_slot) + " is not an int64 LOOKUP slot");
            const bool dynamic = d.stale_slot != -1;
            if (dynamic) {
                if (!slot_is(d.class_slot, true, EG_DTYPE_INT64))
                    refuse(EG_BAD_REQUEST, at + "class slot " + std::to_string(d.class_slot) + " is not a per-sample int64 MUTABLE or OUT slot");
                if (!slot_is(d.stale_slot, true, EG_DTYPE_BOOL))
                    refuse(EG_BAD_REQUEST, at + "stale slot " + std::to_string(d.stale_slot) + " is not a per-sample bool MUTABLE or OUT slot");
                if (d.class_slot == d.stale_slot || d.class_slot == d.offsets_slot)
                    refuse(EG_BAD_REQUEST, at + "gives one slot two of its class, stale and offsets planes");
                unit.slots[static_cast<std::size_t>(d.class_slot)].made = true;
                unit.slots[static_cast<std::size_t>(d.stale_slot)].made = true;
                f.has_sweep = true;
                f.sweep = ref_of(d.sweep);
            } else {
                if (d.class_slot != -1) refuse(EG_BAD_REQUEST, at + "has a static selector and a class slot (the classify unit owns it)");
                f.has_classify = true;
                f.classify = ref_of(d.classify);
            }
            unit.slots[static_cast<std::size_t>(d.offsets_slot)].made = true;
            f.branch_record = true;
            f.cls = d.class_slot, f.stale = d.stale_slot, f.offsets = d.offsets_slot;
        }
        unit.dispatches.push_back(std::move(f));
    }
    for (const DispatchFacts& f : unit.dispatches)
        if ((f.branch_record || f.classify_record) && unit.dispatches.size() > 1)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': a SEGMENTED dispatch group beside a second dispatch group is not written");
}

/** @brief Copy and check the keyed table of the body: the keys a block sum or a KEY team plane runs over. A record names no plane of its own here:
 *  whether eagle makes one is decided where the stages are joined. */
void read_keyed(const eg_unit_body& body, eg_unit_s& unit) {
    const std::string who = "meta that contradicts itself: unit '" + unit.name + "' ";
    if (body.n_keyed < 0 || (body.n_keyed > 0 && body.keyed == nullptr)) refuse(EG_BAD_REQUEST, who + "n_keyed " + std::to_string(body.n_keyed));
    for (int i = 0; i < body.n_keyed; ++i) {
        const eg_unit_keyed k = read_record(body.keyed[i], eg_unit_keyed{}, "unit '" + unit.name + "' keyed");
        const std::string at = who + "keyed record " + std::to_string(i);
        if (k.name == nullptr) refuse(EG_BAD_REQUEST, at + " has no name");
        if (k.n_axes < 0 || (k.n_axes > 0 && k.axis_extents == nullptr)) refuse(EG_BAD_REQUEST, at + " n_axes " + std::to_string(k.n_axes));
        if (k.width < 1) refuse(EG_BAD_REQUEST, at + " width " + std::to_string(k.width));
        KeyedFacts f;
        f.width = k.width;
        f.name = k.name;
        for (int a = 0; a < k.n_axes; ++a) {
            if (k.axis_extents[a] < 1 && k.axis_extents[a] != -1) refuse(EG_BAD_REQUEST, at + " extent " + std::to_string(k.axis_extents[a]));
            f.extents.push_back(k.axis_extents[a]);
        }
        unit.keyed.push_back(std::move(f));
    }
}

/** @brief Copy and check the team planes of the body (a split unit's scratch). Only the planes a host plan makes are written. */
void read_team_planes(const eg_unit_body& body, eg_unit_s& unit) {
    const std::string who = "meta that contradicts itself: unit '" + unit.name + "' ";
    if (body.n_team_planes < 0 || (body.n_team_planes > 0 && body.team_planes == nullptr))
        refuse(EG_BAD_REQUEST, who + "n_team_planes " + std::to_string(body.n_team_planes));
    unit.scatter_chunks = std::max<std::int64_t>(0, body.scatter_chunks);
    for (int i = 0; i < body.n_team_planes; ++i) {
        eg_unit_team_plane absent{};
        absent.keyed = absent.target_slot = absent.target_stage = -1;
        const eg_unit_team_plane t = read_record(body.team_planes[i], absent, "unit '" + unit.name + "' team plane");
        if (t.name == nullptr) refuse(EG_BAD_REQUEST, who + "has a team plane without a name");
        const std::string name = t.name;
        const std::string at = "unit '" + unit.name + "': team plane '" + name + "'";
        if (t.dtype == EG_DTYPE_BAND_PARTIAL) refuse(EG_NOT_WRITTEN, at + " holds BAND_PARTIAL elements, which are not written");
        if (t.dtype != EG_DTYPE_REAL && t.dtype != EG_DTYPE_INT32) refuse(EG_BAD_REQUEST, who + "team plane '" + name + "' has dtype " + std::to_string(t.dtype));
        if (t.unit < EG_UNIT_LANE || t.unit > EG_UNIT_ONCE || t.layout < EG_LAYOUT_FLAT || t.layout > EG_LAYOUT_PLANE_MAJOR)
            refuse(EG_NOT_WRITTEN, at + " has an unknown unit or layout");
        if (t.wide == 1) refuse(EG_BAD_REQUEST, who + "team plane '" + name + "' is wide value 1, which is retired (a placed wide value is WARP_TILED)");
        if (t.wide != EG_WIDE_NONE && t.wide != EG_WIDE_WARP_TILED) refuse(EG_NOT_WRITTEN, at + " has an unknown wide layout " + std::to_string(t.wide));
        if (t.per_lane < 1) refuse(EG_BAD_REQUEST, who + "team plane '" + name + "' per_lane " + std::to_string(t.per_lane));
        if (t.unit == EG_UNIT_TARGET) {
            if (t.target_slot < 0 || t.target_slot >= static_cast<std::int32_t>(unit.slots.size()))
                refuse(EG_BAD_REQUEST, who + "team plane '" + name + "' has TARGET units and target_slot " + std::to_string(t.target_slot) + ", outside the " +
                       std::to_string(unit.slots.size()) + " slots of the unit");
            if (t.target_stage != -1) refuse(EG_NOT_WRITTEN, at + " has TARGET units whose target is a slot of the unit of stage " + std::to_string(t.target_stage) + ", which is not written");
            if (body.scatter_chunks < 1) refuse(EG_BAD_REQUEST, who + "has TARGET units and no chunk cap (scatter_chunks " + std::to_string(body.scatter_chunks) + ")");
        }
        if (t.unit == EG_UNIT_KEY && (t.keyed < 0 || t.keyed >= static_cast<std::int32_t>(unit.keyed.size())))
            refuse(EG_BAD_REQUEST, who + "team plane '" + name + "' has KEY units and keyed " + std::to_string(t.keyed) + ", outside the keyed table of " +
                   std::to_string(unit.keyed.size()));
        unit.team_planes.push_back({ t.dtype, t.unit, t.layout, t.wide, t.per_lane, t.fill, t.fill_promised != 0, name, t.unit == EG_UNIT_KEY ? t.keyed : -1,
                                     t.unit == EG_UNIT_TARGET ? t.target_slot : -1 });
    }
}

/** @brief Copy the stages of a split unit and check their records; the units are resolved against deps by unit_load. */
void read_stages(const eg_unit_meta& m, eg_unit_s& unit) {
    for (int i = 0; i < m.n_stages; ++i) {
        const std::string at = "unit '" + unit.name + "' stage " + std::to_string(i);
        eg_unit_stage absent{};
        const eg_unit_stage st = read_record(m.stages[i], absent, at);
        if (st.unit == nullptr || st.unit->name == nullptr) refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + at + " names no unit");
        if (st.n_cells_in < 0 || st.n_cells_out < 0 || (st.n_cells_in > 0 && st.cells_in == nullptr) || (st.n_cells_out > 0 && st.cells_out == nullptr))
            refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + at + " counts cells it does not carry");
        StageFacts f;
        for (int k = 0; k < st.n_cells_in; ++k) f.cells_in.push_back(st.cells_in[k] ? st.cells_in[k] : "?");
        for (int k = 0; k < st.n_cells_out; ++k) f.cells_out.push_back(st.cells_out[k] ? st.cells_out[k] : "?");
        f.ref_name = st.unit->name;
        std::memcpy(f.ref_digest.data(), st.unit->digest, EG_DIGEST_BYTES);
        if (st.team != nullptr) {
            eg_unit_team absent_team{};
            absent_team.keyed = absent_team.of_slot = absent_team.status_plane = absent_team.of_stage = -1;
            absent_team.warps = absent_team.lanes = absent_team.groups = 1;
            const eg_unit_team t = read_record(st.team, absent_team, at + " team");
            if (t.grid < EG_GRID_LANES || t.grid > EG_GRID_TARGETS) refuse(EG_NOT_WRITTEN, at + ": unknown team grid " + std::to_string(t.grid));
            if (t.threads < 1 || t.warps < 1 || t.lanes < 1 || t.groups < 1)
                refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + at + " team has threads/warps/lanes/groups below 1");
            f.team = true;
            f.grid = t.grid, f.threads = t.threads, f.warps = t.warps, f.lanes = t.lanes, f.groups = t.groups;
            if (t.grid == EG_GRID_KEYS) f.keyed = t.keyed;    // checked against the keyed table once the body is read
            if (t.grid == EG_GRID_CHUNKS || t.grid == EG_GRID_TARGETS) {    // checked against the slots and the team planes once the body is read
                if (t.of_stage != -1) refuse(EG_NOT_WRITTEN, at + ": a team of grid " + std::string(t.grid == EG_GRID_CHUNKS ? "CHUNKS" : "TARGETS") +
                                                                " into a slot of the unit of stage " + std::to_string(t.of_stage) + " is not written");
                f.of_slot = t.of_slot;
                if (t.grid == EG_GRID_CHUNKS) f.status_plane = t.status_plane;
            }
        }
        unit.stages.push_back(std::move(f));
    }
}

/** @brief Check the body and launch records and copy the slot facts into @p unit. */
void read_body(const eg_unit_meta& m, void* library, bool device_only, eg_unit_s& unit) {
    const eg_unit_body body = read_record(m.body, eg_unit_body{}, "unit '" + unit.name + "' body");
    if (body.access < EG_ACCESS_SAMPLE_LOCAL || body.access > EG_ACCESS_CROSS_SAMPLE_WRITE)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': unknown access " + std::to_string(body.access));
    unit.access = body.access;
    read_keyed(body, unit);
    if (body.n_slots < 0 || (body.n_slots > 0 && body.slots == nullptr))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' n_slots " +
               std::to_string(body.n_slots));
    for (int i = 0; i < body.n_slots; ++i) unit.slots.push_back(read_slot(body.slots[i], unit.name, i));
    read_made(body, library, device_only, unit);
    read_dispatch(body, unit);
    read_team_planes(body, unit);
    read_finish(m, unit);
    if (m.launch == nullptr) return;
    eg_unit_launch absent{};
    absent.active_map_slot = absent.active_count_slot = -1;
    absent.index_map_slot = absent.index_count_slot = -1;
    const eg_unit_launch launch = read_record(m.launch, absent, "unit '" + unit.name + "' launch");
    if (launch.block < 0)
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' block " + std::to_string(launch.block));
    unit.block = launch.block;
    if (launch.index_map_slot != -1 && launch.index_count_slot == -1)
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' names an index map without its count");
    if (launch.index_count_slot != -1) {
        const auto lookup = [&](std::int32_t slot, bool wide_ok, std::int32_t dtype, const char* what) {
            if (slot < 0 || slot >= static_cast<std::int32_t>(unit.slots.size()) ||
                unit.slots[slot].role != EG_ROLE_LOOKUP ||
                !(unit.slots[slot].dtype == dtype || (wide_ok && unit.slots[slot].dtype == EG_DTYPE_INT64)))
                refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' " + what + " slot " +
                       std::to_string(slot) + " is not a LOOKUP slot of its dtype");
        };
        lookup(launch.index_count_slot, false, EG_DTYPE_UINT32, "index count");
        if (launch.index_map_slot != -1) lookup(launch.index_map_slot, true, EG_DTYPE_INT32, "index map");
        if (launch.active_map_slot != -1)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': an index set and an active set in one unit is not written");
        if (launch.group > 1)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': an index set over groups of lanes is not written yet");
        unit.index_map_slot = launch.index_map_slot;
        unit.index_count_slot = launch.index_count_slot;
    }
    if ((launch.active_map_slot == -1) != (launch.active_count_slot == -1))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name +
               "' names an active map without its count (or the count without the map)");
    if (launch.active_map_slot != -1) {
        const auto lookup = [&](std::int32_t slot, std::int32_t dtype, const char* what) {
            if (slot < 0 || slot >= static_cast<std::int32_t>(unit.slots.size()) ||
                unit.slots[slot].role != EG_ROLE_LOOKUP || unit.slots[slot].dtype != dtype)
                refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' " + what +
                       " slot " + std::to_string(slot) + " is not a LOOKUP slot of its dtype");
        };
        lookup(launch.active_map_slot, EG_DTYPE_INT32, "active map");
        lookup(launch.active_count_slot, EG_DTYPE_UINT32, "active count");
        if (launch.active_map_slot == launch.active_count_slot || launch.active_map_slot == unit.finish.counter_slot ||
            launch.active_count_slot == unit.finish.counter_slot)
            refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name +
                   "' gives one slot two of the finished counter, the active map and the active count");
        if (launch.group < 1)
            refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' group " + std::to_string(launch.group));
        unit.group = launch.group;
        unit.active_map_slot = launch.active_map_slot;
        unit.active_count_slot = launch.active_count_slot;
    }
    if (launch.n_lane_axes < 0 || (launch.n_lane_axes > 0 && launch.lane_extents == nullptr))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' n_lane_axes " +
               std::to_string(launch.n_lane_axes) + " without lane_extents");
    for (int i = 0; i < launch.n_lane_axes; ++i)
        unit.lane_extents.push_back(launch.lane_extents[i]);
    for (const SlotFacts& f : unit.slots)
        if (f.gathered && !f.keep.empty() && static_cast<std::int32_t>(f.keep.size()) != launch.n_lane_axes)
            refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' slot '" + f.name +
                   "' has a gather pattern of " + std::to_string(f.keep.size()) + " axes, the launch has " +
                   std::to_string(launch.n_lane_axes));
    if (launch.n_plane_axes < 0 || (launch.n_plane_axes > 0 && launch.plane_axes == nullptr))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' n_plane_axes " +
               std::to_string(launch.n_plane_axes) + " without plane_axes");
    for (int i = 0; i < launch.n_plane_axes; ++i) {
        const eg_unit_plane_axes pa = read_record(launch.plane_axes[i], eg_unit_plane_axes{}, "unit '" + unit.name + "' plane axes");
        if (pa.slot < 0 || pa.slot >= static_cast<std::int32_t>(unit.slots.size()) || pa.n_axes < 1 || pa.labels == nullptr ||
            pa.extents == nullptr)
            refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' plane axes of slot " + std::to_string(pa.slot) +
                   " (a slot of the unit, one axis or more, with labels and extents)");
        SlotFacts& f = unit.slots[pa.slot];
        if (f.role != EG_ROLE_ACCUM_OUT && f.role != EG_ROLE_WIDE_OUT)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': named plane axes on slot '" + f.name + "' (not an ACCUM_OUT or WIDE_OUT slot) are not written");
        for (int a = 0; a < pa.n_axes; ++a) {
            SlotFacts::Axis axis;
            axis.extent = pa.extents[a];
            for (int l = 0; pa.labels[a] != nullptr && launch.lane_labels != nullptr && l < launch.n_lane_axes; ++l)
                if (launch.lane_labels[l] != nullptr && std::strcmp(launch.lane_labels[l], pa.labels[a]) == 0) axis.lane = l;
            f.axes.push_back(axis);
        }
    }
    for (const bool device : { false, true }) {    // the thread box of each side
        const eg_unit_box* stated = device ? launch.device_box : launch.host_box;
        if (stated == nullptr) continue;
        const char* side = device ? "device" : "host";
        const eg_unit_box box = read_record(stated, eg_unit_box{}, "unit '" + unit.name + "' " + side + "_box");
        if (box.axis < 0 || box.t < 2 || box.stride < 1)
            refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' " + side + "_box needs axis >= 0, t >= 2 "
                   "and stride >= 1, got " + std::to_string(box.axis) + ", " + std::to_string(box.t) + ", " +
                   std::to_string(box.stride));
        if (unit.active_map_slot != -1 || unit.index_count_slot != -1)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': a " + side + " thread box with an active set or an index set is not written");
        (device ? unit.dev_box_t : unit.box_t) = box.t;
        (device ? unit.dev_box_stride : unit.box_stride) = box.stride;
    }
}

/** @brief Find the unit named by @p ref (name AND digest) in the library's tree. */
eg_unit_meta find_unit(const eg_library_meta* lib, const eg_unit_ref& ref, std::int32_t& index) {
    const eg_library_meta top = read_record(lib, eg_library_meta{}, "eg_library_meta");
    if (top.schema_major != EG_META_SCHEMA_MAJOR)
        refuse(EG_BAD_REQUEST, "unit meta of schema major " + std::to_string(top.schema_major) +
               ": this eagle reads major " + std::to_string(EG_META_SCHEMA_MAJOR));
    if (top.n_units > 0 && top.units == nullptr)
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: n_units without units");
    for (std::uint32_t i = 0; i < top.n_units; ++i) {
        const eg_unit_meta m = read_record(top.units[i], eg_unit_meta{}, "unit " + std::to_string(i));
        if (m.name != nullptr && std::strcmp(m.name, ref.name) == 0 &&
            std::memcmp(m.digest, ref.digest, EG_DIGEST_BYTES) == 0) {
            index = static_cast<std::int32_t>(i);
            return m;
        }
    }
    refuse(EG_BAD_REQUEST, std::string("no unit named '") + ref.name +
           "' with the given digest in the library");
}

/**
 * @brief Check the size block the unit's library exports beside a host entry:
 * ``eagle_abi_tag`` must be the v2 tag (the entry eg_launch calls is the v2
 * serial range) and ``eagle_layout_sizes`` must equal eagle's own layout sizes,
 * field by field (the check ``plugin/gref_abi.h`` states). A missing symbol or
 * a mismatch is refused BAD_REQUEST, naming the symbol or the field and both values.
 */
void check_size_block(void* library, const std::string& unit) {
    const auto* tag = static_cast<const char*>(::dlsym(library, "eagle_abi_tag"));
    if (tag == nullptr)
        refuse(EG_BAD_REQUEST, "unit '" + unit + "': the library does not export 'eagle_abi_tag'");
    if (plugin::abi_version_of(tag) != 2)
        refuse(EG_BAD_REQUEST, "unit '" + unit + "': eagle_abi_tag is '" + tag + "', this door runs '" +
               EAGLE_AETHER_ABI_V2 + "'");
    const auto* sizes = static_cast<const std::uint64_t*>(::dlsym(library, plugin::kEagleLayoutSymbol));
    if (sizes == nullptr)
        refuse(EG_BAD_REQUEST, "unit '" + unit + "': the library does not export '" +
               plugin::kEagleLayoutSymbol + "'");
    try {
        plugin::check_layout_sizes(sizes, "unit '" + unit + "'");
    } catch (const std::runtime_error& e) {
        refuse(EG_BAD_REQUEST, e.what());
    }
}

/** @brief The one table of valid (role, what, form) links: EG_LINK_RULES, expanded, never copied. */
struct LinkRule { std::int32_t role, what, form; };
constexpr LinkRule kLinkRules[] = {
#define EG_LINK_ROW(role, what, form) { role, what, form },
    EG_LINK_RULES(EG_LINK_ROW)
#undef EG_LINK_ROW
};

/** @brief Refuse a link whose (role, what, form) is not a row of EG_LINK_RULES, or whose ``reduced`` is set off GRAD. */
void check_link(const std::string& unit, std::int32_t role, std::int32_t index, const eg_unit_link& l) {
    const std::string at = "unit '" + unit + "' links[" + std::to_string(index) + "]";
    bool role_has_rows = false;
    for (const LinkRule& r : kLinkRules) {
        if (r.role != role) continue;
        role_has_rows = true;
        if (r.what == l.what && r.form == l.form) {
            if (l.reduced != 0 && l.form != EG_FORM_GRAD)
                refuse(EG_BAD_REQUEST, at + ".reduced " + std::to_string(l.reduced) + ": only a GRAD link is reduced");
            return;
        }
    }
    if (!role_has_rows) refuse(EG_NOT_WRITTEN, at + ": derived role " + std::to_string(role) + " takes no links in this build of the door");
    refuse(EG_BAD_REQUEST, at + ": role " + std::to_string(role) + " takes no link of what " + std::to_string(l.what) +
           " in form " + std::to_string(l.form) + " (not a row of EG_LINK_RULES)");
}

/** @brief Copy the derived record (eg_unit_derived); eg_runner_new checks it against the primal. */
void read_derived(const eg_unit_meta& m, eg_unit_s& unit) {
    if (m.derived == nullptr) return;
    eg_unit_derived absent{};
    const eg_unit_derived d = read_record(m.derived, absent, "unit '" + unit.name + "' derived");
    const std::string what = "meta that contradicts itself: unit '" + unit.name + "' derived ";
    if (d.n_taped < 0 || (d.n_taped > 0 && d.taped == nullptr)) refuse(EG_BAD_REQUEST, what + "n_taped without taped");
    if (d.n_links < 0 || (d.n_links > 0 && d.links == nullptr)) refuse(EG_BAD_REQUEST, what + "n_links without links");
    DerivedFacts& f = unit.derived;
    f.present = true;
    f.role = d.role;
    f.of_unit = d.of_unit;
    f.tape_slot = d.tape_slot;
    f.tape_rows = d.tape_rows;
    f.steps_slot = d.steps_slot;
    f.tape_on_slot = d.tape_on_slot;
    f.time0_slot = d.time0_slot;
    f.fused_steps_slot = d.fused_steps_slot;
    for (std::int32_t i = 0; i < d.n_taped; ++i) {
        const eg_unit_taped t = read_record(d.taped[i], eg_unit_taped{}, what + "taped[" + std::to_string(i) + "]");
        f.taped.push_back({ t.slot, t.row0, t.rows });
    }
    int time0_values = 0, time_steps = 0;
    for (std::int32_t i = 0; i < d.n_links; ++i) {
        const eg_unit_link l = read_record(d.links[i], eg_unit_link{}, what + "links[" + std::to_string(i) + "]");
        check_link(unit.name, d.role, i, l);
        if (l.what == EG_WHAT_TIME0 && l.form == EG_FORM_VALUE && ++time0_values > 1)
            refuse(EG_BAD_REQUEST, "unit '" + unit.name + "' links[" + std::to_string(i) + "]: a second (TIME0, VALUE) link");
        if (d.role == EG_DERIVED_REFINE_NEWTON && l.what == EG_WHAT_TIME_STEP && ++time_steps > 1)
            refuse(EG_BAD_REQUEST, "unit '" + unit.name + "' links[" + std::to_string(i) + "]: a second (TIME_STEP, VALUE) link (a Newton unit takes exactly one)");
        f.links.push_back({ l.slot, l.what, l.of_slot, l.reduced, l.form });
    }
}

/** @brief Check the unit-level members and resolve the host entry. */
void read_unit(const eg_unit_meta& m, void* library, eg_unit_s& unit) {
    if (m.flags != 0)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': unknown flag bits " + std::to_string(m.flags));
    if (m.pattern != EG_PATTERN_PURE)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': pattern " + std::to_string(m.pattern) +
               " is not written");
    if (m.scalar_type < EG_SCALAR_FLOAT64 || m.scalar_type > EG_SCALAR_BANDED)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': unknown scalar_type " +
               std::to_string(m.scalar_type));
    unit.scalar_type = m.scalar_type;
    if (m.n_stages < 0 || (m.n_stages > 0 && m.stages == nullptr))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' n_stages " +
               std::to_string(m.n_stages));
    read_stages(m, unit);
    if (m.device_image_format < EG_IMAGE_NONE || m.device_image_format > EG_IMAGE_FATBIN)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': unknown device_image_format " +
               std::to_string(m.device_image_format));
    if (m.device_image != nullptr && m.device_image_format == EG_IMAGE_NONE)
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name +
               "' names a device image of format NONE");
    // A library built for the device alone (the device image, no host code) loads on a device context as the
    // old device plugin does (it drops host_entry); a host plan of it is refused at eg_plan_new.
    const bool device_only = unit.ctx->dev.ordinal >= 0 && m.device_image != nullptr;
    if (m.host_entry != nullptr) {
        void* sym = ::dlsym(library, m.host_entry);
        if (sym == nullptr && !device_only)
            refuse(EG_FAULT, "unit '" + unit.name + "': the library lacks its host entry '" +
                   m.host_entry + "'");
        unit.host_entry = reinterpret_cast<exec::HostEntryV2>(sym);
        if (sym != nullptr) check_size_block(library, unit.name);
    }
    read_body(m, library, device_only, unit);
    for (std::size_t i = 0; i < unit.stages.size(); ++i)
        if (unit.stages[i].grid == EG_GRID_KEYS && (unit.stages[i].keyed < 0 || unit.stages[i].keyed >= static_cast<std::int32_t>(unit.keyed.size())))
            refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' stage " + std::to_string(i) + " team has grid KEYS and keyed " +
                   std::to_string(unit.stages[i].keyed) + ", outside the keyed table of " + std::to_string(unit.keyed.size()));
    for (std::size_t i = 0; i < unit.stages.size(); ++i) {
        const StageFacts& st = unit.stages[i];
        if (st.grid != EG_GRID_CHUNKS && st.grid != EG_GRID_TARGETS) continue;
        const std::string at = "meta that contradicts itself: unit '" + unit.name + "' stage " + std::to_string(i) + " team has grid " + (st.grid == EG_GRID_CHUNKS ? "CHUNKS" : "TARGETS");
        if (st.of_slot < 0 || st.of_slot >= static_cast<std::int32_t>(unit.slots.size()))
            refuse(EG_BAD_REQUEST, at + " and of_slot " + std::to_string(st.of_slot) + ", outside the " + std::to_string(unit.slots.size()) + " slots of the unit");
        if (st.grid != EG_GRID_CHUNKS) continue;
        if (unit.scatter_chunks < 1) refuse(EG_BAD_REQUEST, at + " and no chunk cap (scatter_chunks " + std::to_string(unit.scatter_chunks) + ")");
        if (st.status_plane < 0 || st.status_plane >= static_cast<std::int32_t>(unit.team_planes.size()) || unit.team_planes[static_cast<std::size_t>(st.status_plane)].dtype != EG_DTYPE_INT32)
            refuse(EG_BAD_REQUEST, at + " and status_plane " + std::to_string(st.status_plane) + ", which is not an INT32 team plane");
    }
    if (unit.stages.empty() && !unit.team_planes.empty())
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' has team planes and no stages");
    if (!unit.stages.empty() && (!unit.cells.empty() || unit.derived.present))
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': a split unit with cell rows of its own or a derived role is not written");
    read_derived(m, unit);
    if (unit.ctx->dev.ordinal >= 0 && m.device_image != nullptr) device_unit_load(unit, m, library);
}

/** @brief Check the load description (flags, path, unit reference, deps). */
eg_unit_ref read_desc(const eg_unit_load_desc& d) {
    if (d.flags != 0)
        refuse(EG_BAD_REQUEST, "eg_unit_load_desc.flags: unknown bits " + std::to_string(d.flags));
    if (d.path == nullptr) refuse(EG_BAD_REQUEST, "eg_unit_load_desc.path is NULL");
    if (d.n_deps < 0 || (d.n_deps > 0 && d.deps == nullptr))
        refuse(EG_BAD_REQUEST, "eg_unit_load_desc: n_deps " + std::to_string(d.n_deps) +
               " without deps");
    if (d.unit == nullptr) refuse(EG_BAD_REQUEST, "eg_unit_load_desc.unit is NULL");
    eg_unit_ref ref{};
    const std::size_t n = d.unit->struct_size < sizeof(ref) ? d.unit->struct_size : sizeof(ref);
    std::memcpy(static_cast<void*>(&ref), static_cast<const void*>(d.unit), n);
    if (ref.name == nullptr) refuse(EG_BAD_REQUEST, "eg_unit_load_desc.unit names no unit");
    return ref;
}

/** @brief Resolve the stages of the split unit @p unit against @p deps (or the unit itself) and join the slot names ONCE: per stage slot the
 *  source slot, the team plane, the cell rows (or their compensation) it is, or the cell it reads; the sample count has none. */
void resolve_stages(eg_ctx& ctx, const eg_unit_load_desc& d, eg_unit_s& unit) {
    const std::string who = "meta that contradicts itself: unit '" + unit.name + "' ";
    try {
        for (std::size_t i = 0; i < unit.stages.size(); ++i) {
            StageFacts& st = unit.stages[i];
            const std::string at = "unit '" + unit.name + "' stage " + std::to_string(i) + " ('" + st.ref_name + "')";
            if (st.ref_name == unit.name && st.ref_digest == unit.digest) {
                if (unit.host_entry == nullptr && unit.function == 0) refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + at + " is the unit itself, which has no entry to run");
                st.unit = &unit;
                st.self = true;
            } else {
                for (std::int32_t k = 0; k < d.n_deps; ++k) {
                    check_handle(ctx.units, d.deps[k], "eg_unit_load_desc.deps[i]");
                    if (d.deps[k]->name == st.ref_name && d.deps[k]->digest == st.ref_digest) st.unit = d.deps[k];
                }
                if (st.unit == nullptr) {
                    std::string hex;
                    for (std::uint8_t b : st.ref_digest) { static const char* h = "0123456789abcdef"; hex += h[b >> 4]; hex += h[b & 15]; }
                    refuse(EG_BAD_REQUEST, at + " (digest " + hex + ") is not in deps");
                }
                eg_unit_s& su = *st.unit;
                if (su.host_entry == nullptr && su.function == 0) refuse(EG_BAD_REQUEST, at + " has no entry to run (no host entry, no device function)");
                if (!su.stages.empty() || su.derived.present)
                    refuse(EG_NOT_WRITTEN, at + ": a stage that is itself split or a derived unit is not written");
                if (su.finish.present) {    // the source's finish is the unit's: a stage's own record must name the same mask and counter slots (joined by name below), and is not read again
                    const auto slot_name = [](const eg_unit_s& x, std::int32_t i) { return i >= 0 ? x.slots[static_cast<std::size_t>(i)].name : std::string(); };
                    if (!unit.finish.present || slot_name(su, su.finish.mask_slot) != slot_name(unit, unit.finish.mask_slot) ||
                        slot_name(su, su.finish.counter_slot) != slot_name(unit, unit.finish.counter_slot) || su.finish.steps != unit.finish.steps)
                        refuse(EG_NOT_WRITTEN, at + ": a finishing stage whose finish (mask slot, counter slot, steps) is not the source unit's is not written");
                }
                ++su.refs;
                st.held = true;
            }
            const eg_unit_s& su = *st.unit;
            st.cell_row = su.cell_row;
            for (const CellFacts& c : su.cells) {    // the cell columns this stage writes: where each folds is joined below
                SplitCell sc;
                sc.stage = static_cast<std::int32_t>(i);
                sc.row_slot = c.slot;
                sc.comp_slot = c.comp;
                sc.op = c.op;
                sc.name = su.slots[static_cast<std::size_t>(c.slot)].name;
                for (std::size_t k = 0; k < unit.slots.size(); ++k)
                    if (unit.slots[k].name == sc.name) sc.source = static_cast<std::int32_t>(k);
                unit.split_cells.push_back(sc);
            }
        }
        for (std::size_t k = 0; k < unit.keyed.size(); ++k) {    // a keyed record is eagle's plane iff its name is no slot of the unit and is a slot of a stage; else a key spec
            const KeyedFacts& kf = unit.keyed[k];
            bool of_unit = false, of_stage = false;
            for (const SlotFacts& sl : unit.slots) of_unit = of_unit || sl.name == kf.name;
            for (const StageFacts& st : unit.stages)
                for (const SlotFacts& sl : st.unit->slots) {
                    if (sl.name != kf.name) continue;
                    of_stage = true;
                    if (sl.dtype != EG_DTYPE_REAL)
                        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': the keyed plane '" + kf.name + "' that eagle makes is read or written as a slot that is not of the unit's scalar type");
                }
            if (of_unit || !of_stage) continue;
            bool as_plane = false;    // the record names its own KEY team plane: that plane is the one (made once, zeroed or filled as it says)
            for (const TeamPlaneFacts& t : unit.team_planes) {
                if (t.name != kf.name) continue;
                if (t.unit != EG_UNIT_KEY || t.keyed != static_cast<std::int32_t>(k))
                    refuse(EG_BAD_REQUEST, who + "the keyed record '" + kf.name + "' names a team plane that is not its KEY plane");
                as_plane = true;
            }
            if (as_plane) continue;
            // eagle's plane: width entries per key, zeroed at the start of every launch, bound by name to the stages that name it (a team plane of KEY units)
            TeamPlaneFacts t;
            t.dtype = EG_DTYPE_REAL, t.unit = EG_UNIT_KEY, t.layout = EG_LAYOUT_FLAT, t.wide = EG_WIDE_NONE;
            t.per_lane = kf.width, t.fill = 0, t.fill_promised = true, t.name = kf.name, t.keyed = static_cast<std::int32_t>(k);
            unit.team_planes.push_back(t);
            for (const StageFacts& st : unit.stages)    // the compensation companion of that plane (a stage's companion whose target it is): eagle's too, of the same shape, zeroed with it
                for (const CompanionFacts& c : st.unit->companions) {
                    const std::string& cn = st.unit->slots[static_cast<std::size_t>(c.slot)].name;
                    bool named = false;
                    for (const SlotFacts& sl : unit.slots) named = named || sl.name == cn;
                    for (const TeamPlaneFacts& tp : unit.team_planes) named = named || tp.name == cn;
                    if (st.unit->slots[static_cast<std::size_t>(c.target)].name != kf.name || named) continue;
                    t.name = cn;
                    unit.team_planes.push_back(t);
                }
        }
        for (std::size_t i = 0; i < unit.stages.size(); ++i) {
            StageFacts& st = unit.stages[i];
            const std::string at = "unit '" + unit.name + "' stage " + std::to_string(i) + " ('" + st.ref_name + "')";
            const eg_unit_s& su = *st.unit;
            const std::size_t n_slots = su.slots.size();
            st.source_of.assign(n_slots, -1);
            st.plane_of.assign(n_slots, -1);
            st.row_of.assign(n_slots, -1);
            st.comp_of.assign(n_slots, -1);
            st.read_of.assign(n_slots, -1);
            for (std::size_t c = 0; c < unit.split_cells.size(); ++c) {
                const SplitCell& sc = unit.split_cells[c];
                if (sc.stage != static_cast<std::int32_t>(i)) continue;
                st.row_of[static_cast<std::size_t>(sc.row_slot)] = static_cast<std::int32_t>(c);
                if (sc.comp_slot >= 0) st.comp_of[static_cast<std::size_t>(sc.comp_slot)] = static_cast<std::int32_t>(c);
            }
            for (std::size_t j = 0; j < n_slots; ++j) {
                const std::string& nm = su.slots[j].name;
                if (st.row_of[j] >= 0) continue;
                if (st.comp_of[j] >= 0) {    // a compensation row: eagle's; a source slot of that name is hidden from the caller
                    for (std::size_t k = 0; k < unit.slots.size(); ++k)
                        if (unit.slots[k].name == nm) unit.slots[k].made = true;
                    continue;
                }
                for (std::size_t c = 0; c < unit.split_cells.size(); ++c)    // a slot that reads a cell an earlier stage folds
                    if (unit.split_cells[c].name == nm && unit.split_cells[c].stage < static_cast<std::int32_t>(i)) st.read_of[j] = static_cast<std::int32_t>(c);
                if (st.read_of[j] >= 0) continue;
                for (std::size_t k = 0; k < unit.slots.size(); ++k)
                    if (unit.slots[k].name == nm) st.source_of[j] = static_cast<std::int32_t>(k);
                for (std::size_t k = 0; k < unit.team_planes.size(); ++k)
                    if (unit.team_planes[k].name == nm) st.plane_of[j] = static_cast<std::int32_t>(k);
                if (st.source_of[j] >= 0 && st.plane_of[j] >= 0) {    // a stage that is the unit itself lists its scratch as a slot: eagle's, never the caller's
                    unit.slots[static_cast<std::size_t>(st.source_of[j])].made = true;
                    st.source_of[j] = -1;
                }
                if (st.source_of[j] < 0 && st.plane_of[j] < 0 && su.slots[j].role != EG_ROLE_NSAMPLES)
                    refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + at + " slot '" + nm + "' joins no source slot, no team plane and no cell");
            }
            for (const std::string& out : st.cells_out) {    // the cells the stage says it folds are cells its unit has
                bool found = false;
                for (const CellFacts& c : su.cells) found = found || c.name == out;
                if (!found) refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + at + " folds the cell '" + out + "', which its unit does not have");
            }
            for (const std::string& in : st.cells_in) {    // the cells it reads are source cells an earlier stage folds
                bool found = false;
                for (const std::string& s : unit.source_cells) found = found || s == in;
                if (!found) refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + at + " reads the cell '" + in + "', which the unit does not declare");
            }
        }
        bool self = false;
        for (const StageFacts& st : unit.stages) self = self || st.self;
        if (!unit.companions.empty() && !self)
            refuse(EG_BAD_REQUEST, who + "lists companions, and none of its stages is the unit itself that would fold them");
        for (std::size_t i = 0; i < unit.stages.size(); ++i) {    // the companions of a stage's unit: each is folded into its target after the stage runs
            StageFacts& st = unit.stages[i];
            const eg_unit_s& su = *st.unit;
            for (std::size_t ci = 0; ci < su.companions.size(); ++ci) {
                const CompanionFacts& c = su.companions[ci];
                const std::string of = "the companion '" + su.slots[static_cast<std::size_t>(c.slot)].name + "' of '" + su.slots[static_cast<std::size_t>(c.target)].name + "'";
                const std::string at = "unit '" + unit.name + "' stage " + std::to_string(i) + " ('" + st.ref_name + "'): " + of;
                if (st.team) refuse(EG_NOT_WRITTEN, at + " is on a team stage, which is not written");
                const std::int32_t cs = st.source_of[static_cast<std::size_t>(c.slot)], ts = st.source_of[static_cast<std::size_t>(c.target)];
                const std::int32_t cp = st.plane_of[static_cast<std::size_t>(c.slot)], tp = st.plane_of[static_cast<std::size_t>(c.target)];
                if (cs >= 0 && ts >= 0) {    // the caller's plane and the slot eagle makes beside it at eg_bind: the unit's own companion, as a plain unit's
                    unit.slots[static_cast<std::size_t>(cs)].made = true;
                    if (!st.self) unit.companions.push_back({ cs, ts, c.fold, c.device_fold_name, c.device_fold });
                    st.folds.push_back({ ci, ts, -1 });
                } else if (cp >= 0 && tp >= 0) {    // a keyed plane eagle makes and the plane beside it, both zeroed at the start of every launch
                    const TeamPlaneFacts& cpf = unit.team_planes[static_cast<std::size_t>(cp)];
                    const TeamPlaneFacts& tpf = unit.team_planes[static_cast<std::size_t>(tp)];
                    if (tpf.unit != EG_UNIT_KEY || cpf.unit != EG_UNIT_KEY || tpf.keyed != cpf.keyed || tpf.per_lane != cpf.per_lane)
                        refuse(EG_BAD_REQUEST, who + of + " and its target are team planes that do not hold the same keys");
                    if (!tpf.fill_promised || tpf.fill != 0 || !cpf.fill_promised || cpf.fill != 0 || tpf.dtype != EG_DTYPE_REAL || cpf.dtype != EG_DTYPE_REAL)
                        refuse(EG_NOT_WRITTEN, at + " is beside a team plane that is not a float plane zeroed at every launch");
                    st.folds.push_back({ ci, -1, tp });
                } else {
                    refuse(EG_NOT_WRITTEN, at + ": its target and it are not both slots of the unit, nor both planes eagle makes for a keyed record");
                }
            }
        }
        for (const StageFacts& st : unit.stages) {    // a block sum's output: the WIDE_OUT slot a KEYS stage writes holds that stage's keys
            if (st.grid != EG_GRID_KEYS) continue;
            for (std::size_t j = 0; j < st.unit->slots.size(); ++j) {
                if (st.source_of[j] < 0 || st.unit->slots[j].role != EG_ROLE_WIDE_OUT) continue;
                SlotFacts& src = unit.slots[static_cast<std::size_t>(st.source_of[j])];
                if (src.keys_of >= 0 && src.keys_of != st.keyed)
                    refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': slot '" + src.name + "' is written by KEYS stages of two keyed records, which is not written");
                src.keys_of = st.keyed;
            }
        }
        for (std::size_t k = 0; k < unit.slots.size(); ++k) {
            bool joined = unit.slots[k].made || unit.slots[k].role == EG_ROLE_NSAMPLES;    // the sample count joins no stage slot of its own
            for (const StageFacts& st : unit.stages)
                for (std::int32_t s : st.source_of) joined = joined || s == static_cast<std::int32_t>(k);
            for (const SplitCell& c : unit.split_cells) joined = joined || c.source == static_cast<std::int32_t>(k);
            if (!joined) refuse(EG_BAD_REQUEST, who + "slot '" + unit.slots[k].name + "' joins no stage slot");
        }
        for (std::size_t k = 0; k < unit.team_planes.size(); ++k) {
            bool joined = false;
            for (const StageFacts& st : unit.stages)
                for (std::int32_t p : st.plane_of) joined = joined || p == static_cast<std::int32_t>(k);
            if (!joined) refuse(EG_BAD_REQUEST, who + "team plane '" + unit.team_planes[k].name + "' joins no stage slot");
        }
    } catch (...) {
        for (StageFacts& st : unit.stages)
            if (st.held) --st.unit->refs;
        unit.stages.clear();
        unit.split_cells.clear();
        throw;
    }
}

/** @brief A branch unit loaded with its group's units in deps: resolve the references (a branch may be the unit itself), check every unit's own
 *  record against the group's (name and k), and join the units' slots to the plan's: the branch unit's slots, then each other unit's slot that no
 *  earlier slot names, in unit order (branches, then the classify unit or the sweep). Eagle's own planes (offsets, class, stale, the sweep's map
 *  and count) join no caller slot. A branch loaded with no deps stays a plain unit (it is one of another unit's deps); planning it is refused. */
void resolve_segmented(eg_ctx& ctx, const eg_unit_load_desc& d, eg_unit_s& unit) {
    const DispatchFacts* group = nullptr;
    for (const DispatchFacts& f : unit.dispatches)
        if (f.branch_record) group = &f;
    if (group == nullptr || d.n_deps == 0) return;
    const std::string who = "meta that contradicts itself: unit '" + unit.name + "' dispatch group '" + group->name + "' ";
    SegFacts& S = unit.seg;
    try {
        const auto find = [&](const RefFacts& r, const char* what) -> eg_unit_s* {
            if (r.name == unit.name && r.digest == unit.digest) return &unit;
            for (std::int32_t k = 0; k < d.n_deps; ++k) {
                check_handle(ctx.units, d.deps[k], "eg_unit_load_desc.deps[i]");
                if (d.deps[k]->name == r.name && d.deps[k]->digest == r.digest) return d.deps[k];
            }
            refuse(EG_BAD_REQUEST, "unit '" + unit.name + "': the " + what + " '" + r.name + "' of dispatch group '" + group->name + "' is not in deps");
        };
        const auto member = [&](eg_unit_s* m, const char* what) {
            if (m != &unit) {
                ++m->refs;
                S.held.push_back(m);
            }
            if (!m->companions.empty() || !m->cells.empty() || m->derived.present || m->index_count_slot >= 0 || m->finish.steps_slot >= 0)
                refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': the " + what + " '" + m->name + "' of dispatch group '" + group->name +
                       "' has cells, companions, a derived role, an index set or a steps word: not written in a SEGMENTED group");
            if (!m->stages.empty() && m != &unit && (m->stages.size() != 1 || !m->stages[0].self))    // a member that is one stage of itself (the planned unit's stages are checked below)
                refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': the " + what + " '" + m->name + "' of dispatch group '" + group->name +
                       "' is split in more than one stage: not written in a SEGMENTED group");
            for (const TeamPlaneFacts& t : m->team_planes)
                if (t.unit == EG_UNIT_KEY || t.fill_promised)
                    refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': the " + what + " '" + m->name + "' of dispatch group '" + group->name +
                           "' has the keyed or refilled scratch plane '" + t.name + "': not written in a SEGMENTED group");
        };
        const auto record_of = [&](const eg_unit_s& m, bool branch, bool classify, std::int32_t strategy) -> const DispatchFacts* {
            for (const DispatchFacts& f : m.dispatches)
                if (f.name == group->name && f.k == group->k && f.strategy == strategy && f.branch_record == branch && f.classify_record == classify) return &f;
            return nullptr;
        };
        S.name = group->name;
        S.k = group->k;
        S.dynamic = group->has_sweep;
        S.offsets_slot = group->offsets;
        S.class_slot = group->cls;
        S.stale_slot = group->stale;
        S.access = unit.access;
        std::vector<const DispatchFacts*> records;
        for (std::int32_t j = 0; j < group->k; ++j) {
            eg_unit_s* b = find(group->branches[static_cast<std::size_t>(j)], "branch");
            const DispatchFacts* r = record_of(*b, true, false, EG_DISPATCH_SEGMENTED);
            if (r == nullptr || r->has_sweep != S.dynamic)
                refuse(EG_BAD_REQUEST, who + "names branch '" + b->name + "', whose own record is not that group's (its name, k or selector kind differ)");
            member(b, "branch");
            S.branches.push_back(b);
            records.push_back(r);
            S.access = std::max(S.access, b->access);
        }
        if (!unit.stages.empty()) {    // the planned unit lends its scratch planes to the branches: its stages are the group's branches, in order
            bool same = unit.stages.size() == S.branches.size();
            for (std::size_t j = 0; same && j < S.branches.size(); ++j) same = unit.stages[j].unit == S.branches[j];
            if (!same)
                refuse(EG_NOT_WRITTEN, who + "is split in stages that are not its branches in order: not written in a SEGMENTED group");
        }
        eg_unit_s* other = nullptr;
        const DispatchFacts* other_rec = nullptr;
        if (S.dynamic) {
            other = S.sweep = find(group->sweep, "sweep unit");
            other_rec = record_of(*other, false, false, EG_DISPATCH_SWITCH);
            if (other_rec == nullptr)
                refuse(EG_BAD_REQUEST, who + "names sweep unit '" + other->name + "', which carries no SWITCH record of that name and k");
            if (other->active_map_slot < 0)
                refuse(EG_BAD_REQUEST, who + "names sweep unit '" + other->name + "', which has no active set (it runs over the stale samples)");
        } else {
            other = S.classify = find(group->classify, "classify unit");
            other_rec = record_of(*other, false, true, EG_DISPATCH_SEGMENTED);
            if (other_rec == nullptr)
                refuse(EG_BAD_REQUEST, who + "names classify unit '" + other->name + "', which carries no classify record of that name and k");
        }
        member(other, S.dynamic ? "sweep unit" : "classify unit");
        S.access = std::max(S.access, other->access);
        std::int32_t mask = -1;
        for (std::size_t i = 0; i < unit.slots.size() && mask < 0; ++i)
            if (unit.slots[i].role == EG_ROLE_TERMINATED) mask = static_cast<std::int32_t>(i);
        S.mask_slot = mask;
        const auto compatible = [&](const SlotFacts& a, const SlotFacts& b, const eg_unit_s& m) {
            if (a.role != b.role || a.dtype != b.dtype || a.rank != b.rank || a.shape != b.shape)
                refuse(EG_BAD_REQUEST, who + "slot '" + b.name + "' of '" + m.name + "' is not the slot of that name in '" + unit.name + "' (role, dtype or shape differ)");
        };
        std::vector<eg_unit_s*> members = S.branches;
        members.push_back(other);
        for (std::size_t mi = 0; mi < members.size(); ++mi) {
            const eg_unit_s& m = *members[mi];
            const DispatchFacts* rec = mi < records.size() ? records[mi] : other_rec;
            std::vector<std::int32_t> join(m.slots.size(), -1);
            for (std::size_t j = 0; j < m.slots.size(); ++j) {
                const std::int32_t js = static_cast<std::int32_t>(j);
                if (&m == &unit) {
                    join[j] = js;
                    continue;
                }
                if (mi >= records.size() && S.dynamic && js == m.active_map_slot) { join[j] = SegFacts::kMap; continue; }
                if (mi >= records.size() && S.dynamic && js == m.active_count_slot) { join[j] = SegFacts::kCount; continue; }
                if (mi < records.size() && js == rec->offsets) { join[j] = S.offsets_slot; continue; }
                if (mi < records.size() && S.dynamic && js == rec->cls) { join[j] = S.class_slot; continue; }
                if (mi < records.size() && S.dynamic && js == rec->stale) { join[j] = S.stale_slot; continue; }
                if (mi >= records.size() && !S.dynamic && js == rec->cls) {    // the classify unit's class plane: one for the group, eagle's
                    if (S.class_slot < 0) {
                        SlotFacts cs = m.slots[j];
                        cs.made = true;
                        unit.slots.push_back(std::move(cs));
                        S.class_slot = static_cast<std::int32_t>(unit.slots.size()) - 1;
                    }
                    join[j] = S.class_slot;
                    continue;
                }
                const SlotFacts& sl = m.slots[j];
                if (sl.made && &m != &unit) continue;    // the member's own plane (the scratch of its split): eagle makes it per member, no caller slot
                for (std::size_t u = 0; u < unit.slots.size() && join[j] < 0; ++u)
                    if (unit.slots[u].name == sl.name && !(sl.role == EG_ROLE_NSAMPLES && unit.slots[u].role != EG_ROLE_NSAMPLES)) {
                        compatible(unit.slots[u], sl, m);
                        join[j] = static_cast<std::int32_t>(u);
                    }
                if (join[j] < 0) {    // a slot only this unit has: the group's too, after the branch unit's own
                    unit.slots.push_back(sl);
                    unit.slots.back().made = false;
                    join[j] = static_cast<std::int32_t>(unit.slots.size()) - 1;
                }
            }
            S.join.push_back(std::move(join));
        }
        for (std::size_t mi = 0; mi < members.size(); ++mi)    // the unit itself: its slots now include the group's extra ones
            if (members[mi] == &unit) {
                S.join[mi].resize(unit.slots.size());
                std::iota(S.join[mi].begin(), S.join[mi].end(), 0);
            }
        S.present = true;
    } catch (...) {
        for (eg_unit_s* h : S.held) --h->refs;
        S = SegFacts{};
        throw;
    }
}

}  // namespace

/** @brief eg_unit_load: open, find, check, resolve; the library stays open while referenced. */
void unit_load(eg_ctx& ctx, const eg_unit_load_desc* desc, eg_unit** out) {
    if (out == nullptr) refuse(EG_BAD_REQUEST, "eg_unit_load: out is NULL");
    *out = nullptr;
    const eg_unit_ref ref = read_desc(read_input(desc, "eg_unit_load_desc"));
    void* library = ::dlopen(desc->path, RTLD_NOW | RTLD_LOCAL);
    if (library == nullptr) {
        const char* why = ::dlerror();
        refuse(EG_FAULT, std::string("the unit library '") + desc->path + "' does not load: " +
               (why ? why : "unknown"));
    }
    try {
        void* sym = ::dlsym(library, EG_UNIT_LIBRARY_META_SYMBOL);
        if (sym == nullptr)
            refuse(EG_FAULT, std::string("the unit library '") + desc->path + "' lacks " +
                   EG_UNIT_LIBRARY_META_SYMBOL);
        const eg_library_meta* lib = reinterpret_cast<eg_unit_library_meta_fn>(sym)();
        std::int32_t index = -1;
        const eg_unit_meta m = find_unit(lib, ref, index);
        auto* unit = new eg_unit_s;
        unit->ctx = &ctx;
        unit->library = library;
        unit->name = ref.name;
        unit->index = index;
        std::memcpy(unit->digest.data(), m.digest, EG_DIGEST_BYTES);
        try {
            read_unit(m, library, *unit);
            if (!unit->stages.empty()) resolve_stages(ctx, read_input(desc, "eg_unit_load_desc"), *unit);
            for (const SlotFacts& s : unit->slots)
                if (s.role == EG_ROLE_WIDE_OUT && !s.axes.empty() && s.keys_of < 0)
                    refuse(EG_NOT_WRITTEN, "unit '" + unit->name + "': named plane axes on slot '" + s.name + "', which no KEYS stage writes, are not written");
            resolve_segmented(ctx, read_input(desc, "eg_unit_load_desc"), *unit);
            ctx.units.insert(unit);
        } catch (...) {
            device_unit_free(*unit);
            delete unit;
            throw;
        }
        *out = unit;
    } catch (...) {
        ::dlclose(library);
        throw;
    }
}

/** @brief Drop one reference to @p unit; the last one closes its library. */
void unit_release(eg_ctx& ctx, eg_unit_s* unit) {
    if (--unit->refs > 0) return;
    ctx.units.erase(unit);
    for (StageFacts& st : unit->stages)
        if (st.held) unit_release(ctx, st.unit);
    for (eg_unit_s* h : unit->seg.held) unit_release(ctx, h);
    device_unit_free(*unit);
    ::dlclose(unit->library);
    delete unit;
}

}  // namespace door
}  // namespace eagle
