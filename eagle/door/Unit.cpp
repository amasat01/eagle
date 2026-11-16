// SPDX-License-Identifier: Apache-2.0

// eg_unit_load: open the unit library, read its constant meta, check it, and
// keep the facts eg_bind and eg_launch need (on a device context the device
// image is loaded too, device/Device.cpp). Every member past a record's
// struct_size reads as its documented absent value.
#include "eagle/door/Context.h"
#include "eagle/door/eg_unit_meta.h"
#include "plugin/gref_abi.h"

#include <dlfcn.h>

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

/** @brief A count of the meta that this slice of the door does not take yet. */
void refuse_count(std::int64_t n, const std::string& unit, const char* member) {
    if (n < 0) refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit + "' " +
                      member + " is " + std::to_string(n));
    if (n > 0) refuse(EG_NOT_WRITTEN, "unit '" + unit + "': " + member + " > 0 is not written "
                      "in this build of the door");
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
        unit.finish = FinishFacts{ true, f.mask_slot, f.counter_slot, f.steps, f.steps_slot, f.steps_max, f.flags,
                                   f.fp64_ops_ir > 0 ? f.fp64_ops_ir : f.step_ops > 0 ? f.step_ops : 0 };
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
    if (body.n_cells > 0) {
        if (body.cell_row < 1) refuse(EG_BAD_REQUEST, who + "cell_row " + std::to_string(body.cell_row) + " with cells");
        unit.cell_row = body.cell_row;
    }
    for (int i = 0; i < body.n_cells; ++i) {
        const eg_unit_cell cell = read_record(body.cells[i], eg_unit_cell{}, "unit '" + unit.name + "' cell");
        const std::string name = cell.name ? cell.name : "?";
        if (cell.op != EG_REDUCE_SUM && cell.op != EG_REDUCE_MAX && cell.op != EG_REDUCE_MIN)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': cell '" + name + "' folds with op " + std::to_string(cell.op) +
                   ", which is not written (SUM, MAX and MIN are)");
        if (cell.columns < 1 || cell.slots == nullptr)
            refuse(EG_BAD_REQUEST, who + "cell '" + name + "' has " + std::to_string(cell.columns) + " columns and no row slots");
        for (int c = 0; c < cell.columns; ++c) {
            SlotFacts& row = accum(cell.slots[c], "cell");
            if (row.made || row.cell) refuse(EG_BAD_REQUEST, who + "cell slot " + std::to_string(cell.slots[c]) + " is listed twice");
            if (cell.op == EG_REDUCE_SUM)
                refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': the cell '" + name + "' sums, and the cell's compensation slot is not "
                       "stated in the unit meta (eg_unit_cell has no member for it)");
            row.cell = true;
            unit.cells.push_back({ cell.slots[c], cell.op });
        }
    }
}

/** @brief Check the body and launch records and copy the slot facts into @p unit. */
void read_body(const eg_unit_meta& m, void* library, bool device_only, eg_unit_s& unit) {
    const eg_unit_body body = read_record(m.body, eg_unit_body{}, "unit '" + unit.name + "' body");
    if (body.access < EG_ACCESS_SAMPLE_LOCAL || body.access > EG_ACCESS_CROSS_SAMPLE_WRITE)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': unknown access " + std::to_string(body.access));
    unit.access = body.access;
    refuse_count(body.n_keyed, unit.name, "n_keyed");
    refuse_count(body.n_team_planes, unit.name, "n_team_planes");
    refuse_count(body.n_dispatches, unit.name, "n_dispatches");
    if (body.n_slots < 0 || (body.n_slots > 0 && body.slots == nullptr))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' n_slots " +
               std::to_string(body.n_slots));
    for (int i = 0; i < body.n_slots; ++i) unit.slots.push_back(read_slot(body.slots[i], unit.name, i));
    read_made(body, library, device_only, unit);
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
        if (f.role != EG_ROLE_ACCUM_OUT)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': named plane axes on slot '" + f.name + "' (not an ACCUM_OUT slot) are not written");
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
    if (m.n_stages > 0)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': a split unit (n_stages > 0) is not "
               "written yet: the stage layout is not final");
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
        try {
            read_unit(m, library, *unit);
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
    device_unit_free(*unit);
    ::dlclose(unit->library);
    delete unit;
}

}  // namespace door
}  // namespace eagle
