// SPDX-License-Identifier: Apache-2.0

// eg_unit_load: open the unit library, read its constant meta, check it, and
// keep the facts eg_bind and eg_launch need. Every member past a record's
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
    if (s.flags & (EG_SLOT_GATHER | EG_SLOT_SHARED))
        refuse(EG_NOT_WRITTEN, named + ": GATHER and SHARED slots are not written yet");
    if (s.role < EG_ROLE_MUTABLE || s.role > EG_ROLE_OUT)
        refuse(EG_NOT_WRITTEN, named + ": unknown role " + std::to_string(s.role));
    if (s.dtype < EG_DTYPE_REAL || s.dtype > EG_DTYPE_BAND_PARTIAL)
        refuse(EG_NOT_WRITTEN, named + ": unknown dtype " + std::to_string(s.dtype));
    if (s.rank < 0 || s.rank > 2 || (s.rank > 0 && s.shape == nullptr))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: " + named + " rank " +
               std::to_string(s.rank));
    const bool plane = s.role == EG_ROLE_MUTABLE || s.role == EG_ROLE_PER_SAMPLE ||
                       s.role == EG_ROLE_OUT;
    if (!plane && s.role != EG_ROLE_UNIFORM && s.role != EG_ROLE_TERMINATED && s.role != EG_ROLE_LOOKUP)
        refuse(EG_NOT_WRITTEN, named + ": role " + std::to_string(s.role) + " is not written yet "
               "(this build binds MUTABLE, PER_SAMPLE, OUT, UNIFORM, TERMINATED and the finished counter)");
    if (s.rank != 0)
        refuse(EG_NOT_WRITTEN, named + ": a slot of rank > 0 is not written yet");
    if (plane && s.dtype != EG_DTYPE_REAL)
        refuse(EG_NOT_WRITTEN, named + ": a plane of a non-REAL dtype is not written yet");
    if (s.role == EG_ROLE_TERMINATED && s.dtype != EG_DTYPE_BOOL)
        refuse(EG_NOT_WRITTEN, named + ": a TERMINATED mask of a non-BOOL dtype is not written yet");
    if (s.dtype == EG_DTYPE_BAND_PARTIAL)
        refuse(EG_NOT_WRITTEN, named + ": a BAND_PARTIAL uniform is not written");
    return SlotFacts{ s.role, s.dtype, s.name ? s.name : "" };
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
        if (f.steps == EG_STEPS_AUTO)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': finish steps AUTO is not written in this "
                   "build of the door");
        if (f.steps < 1) refuse(EG_BAD_REQUEST, what + "steps " + std::to_string(f.steps) + " (>= 1 or AUTO)");
        if (f.steps_slot != -1)
            refuse(EG_BAD_REQUEST, what + "steps_slot " + std::to_string(f.steps_slot) + " with fixed steps");
        unit.finish = FinishFacts{ true, f.mask_slot, f.counter_slot, f.steps };
    }
}

/** @brief Every LOOKUP slot must be one eagle makes: the finished counter or the active set. */
void check_lookups(const eg_unit_s& unit) {
    for (int i = 0; i < static_cast<int>(unit.slots.size()); ++i)
        if (unit.slots[i].role == EG_ROLE_LOOKUP && i != unit.finish.counter_slot &&
            i != unit.active_map_slot && i != unit.active_count_slot)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "' slot " + std::to_string(i) + " ('" +
                   unit.slots[i].name + "'): a LOOKUP slot other than the finished counter and the "
                   "active set is not written yet");
}

/** @brief Check the body and launch records and copy the slot facts into @p unit. */
void read_body(const eg_unit_meta& m, eg_unit_s& unit) {
    const eg_unit_body body = read_record(m.body, eg_unit_body{}, "unit '" + unit.name + "' body");
    if (body.access < EG_ACCESS_SAMPLE_LOCAL || body.access > EG_ACCESS_CROSS_SAMPLE_WRITE)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': unknown access " + std::to_string(body.access));
    unit.access = body.access;
    refuse_count(body.n_companions, unit.name, "n_companions");
    refuse_count(body.n_cells, unit.name, "n_cells");
    refuse_count(body.n_keyed, unit.name, "n_keyed");
    refuse_count(body.n_team_planes, unit.name, "n_team_planes");
    refuse_count(body.n_dispatches, unit.name, "n_dispatches");
    if (body.n_slots < 0 || (body.n_slots > 0 && body.slots == nullptr))
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: unit '" + unit.name + "' n_slots " +
               std::to_string(body.n_slots));
    for (int i = 0; i < body.n_slots; ++i) unit.slots.push_back(read_slot(body.slots[i], unit.name, i));
    read_finish(m, unit);
    if (m.launch == nullptr) {
        check_lookups(unit);
        return;
    }
    eg_unit_launch absent{};
    absent.active_map_slot = absent.active_count_slot = -1;
    absent.index_map_slot = absent.index_count_slot = -1;
    const eg_unit_launch launch = read_record(m.launch, absent, "unit '" + unit.name + "' launch");
    if (launch.index_map_slot != -1 || launch.index_count_slot != -1)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': index sets are not written yet");
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
        if (launch.group > 1)
            refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': an active set over groups of lanes is not written yet");
        unit.active_map_slot = launch.active_map_slot;
        unit.active_count_slot = launch.active_count_slot;
    }
    check_lookups(unit);
    refuse_count(launch.n_lane_axes > 1 ? 1 : 0, unit.name, "n_lane_axes above 1");
    refuse_count(launch.n_plane_axes, unit.name, "n_plane_axes");
    if (launch.host_box != nullptr)
        refuse(EG_NOT_WRITTEN, "unit '" + unit.name + "': a host thread box is not written yet");
}

/** @brief Find the unit named by @p ref (name AND digest) in the library's tree. */
eg_unit_meta find_unit(const eg_library_meta* lib, const eg_unit_ref& ref) {
    const eg_library_meta top = read_record(lib, eg_library_meta{}, "eg_library_meta");
    if (top.schema_major != EG_META_SCHEMA_MAJOR)
        refuse(EG_BAD_REQUEST, "unit meta of schema major " + std::to_string(top.schema_major) +
               ": this eagle reads major " + std::to_string(EG_META_SCHEMA_MAJOR));
    if (top.n_units > 0 && top.units == nullptr)
        refuse(EG_BAD_REQUEST, "meta that contradicts itself: n_units without units");
    for (std::uint32_t i = 0; i < top.n_units; ++i) {
        const eg_unit_meta m = read_record(top.units[i], eg_unit_meta{}, "unit " + std::to_string(i));
        if (m.name != nullptr && std::strcmp(m.name, ref.name) == 0 &&
            std::memcmp(m.digest, ref.digest, EG_DIGEST_BYTES) == 0)
            return m;
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
    if (m.host_entry != nullptr) {
        void* sym = ::dlsym(library, m.host_entry);
        if (sym == nullptr)
            refuse(EG_FAULT, "unit '" + unit.name + "': the library lacks its host entry '" +
                   m.host_entry + "'");
        unit.host_entry = reinterpret_cast<exec::HostEntryV2>(sym);
        check_size_block(library, unit.name);
    }
    read_body(m, unit);
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
        const eg_unit_meta m = find_unit(lib, ref);
        auto* unit = new eg_unit_s;
        unit->ctx = &ctx;
        unit->library = library;
        unit->name = ref.name;
        try {
            read_unit(m, library, *unit);
            ctx.units.insert(unit);
        } catch (...) {
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
    ::dlclose(unit->library);
    delete unit;
}

}  // namespace door
}  // namespace eagle
