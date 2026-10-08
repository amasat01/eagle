// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The uniform BINDING-TYPE vocabulary shared by the two plugin registries
// (plugin_registry/registry.h and host_registry.h) — the int-role phase.
//
// WHY A SHARED HEADER. Both registries pack a ``uniform`` arg_spec row BY VALUE into
// the kernel's parameter slot, and both now have two possible spellings for that slot
// (float64 or int64). The compare and the refusal wording therefore live in ONE place,
// exactly like ``check_aether_abi`` (gref_abi.h) and ``check_launch_certified_pattern``
// (roles.h): a second inline copy is a second spelling, and a second spelling drifts.
//
// ---------------------------------------------------------------------------
// THE INT WIDTH IS 64-BIT
// ---------------------------------------------------------------------------
// A generated kernel's integer quantity is the code generator's own alias, threaded into every
// kernel template right after ``using Real``:
//
//     using Int = long long;   // the code generator's alias definition
//
// so an integer uniform's parameter declaration is ``AETHER_GRID_CONSTANT() Int p_<name>`` —
// an 8-byte SIGNED value. That is the width these registries must pack, and it is
// corroborated end to end by every other int-typed slot in the stack: an int
// ``Mutable`` handle is the code generator's own 64-bit integer array
// handle (the code generator's pure path), eagle's device marshal fills ``cp.int64``
// (eagle/python/eagle/marshal.py), and the code generator's host provider fills ``np.int64``
// (the code generator's host provider).
//
// aether has no ``Int`` typedef — the near-miss names are aether's own
// 32-bit unsigned index types (array indices and sizes, which is what
// the ``nsamples`` role packs, NOT a user integer) and its 32-bit int
// container alias, which no generated kernel signature names.
// Reading the width off either of those would have produced a 32-bit uniform and a
// silently misread parameter slot, so the alias above is the only authority.
//
// ---------------------------------------------------------------------------
// HOW A UNIFORM'S TYPE IS KNOWN HERE (and the one seam that is NOT closed yet)
// ---------------------------------------------------------------------------
// An arg_spec row is a ``[role, name]`` PAIR — it carries no dtype (sidecar.h
// ``ArgEntry``), and the C++ sidecar reader does not parse the sidecar's top-level
// ``params`` block at all, which today is a bare NAME LIST anyway
// (the code generator's deploy compile step). So there is nothing on the wire for a C++ loader to key
// an int uniform off: the registries learn a uniform's type the same way they learn
// every other binding — from the CALLER, who wrote the host program against a kernel
// whose signature it knows (this is already true of ``bind_vector`` vs ``bind_handle``,
// which are chosen by the caller and merely CROSS-CHECKED against the row's role).
//
// What the caller cannot be trusted with is BOTH: binding one name through both
// binders leaves the packer to choose which 8 bytes the kernel meant, which is the
// silent-wrong-answer failure this vocabulary exists to refuse. Hence
// ``check_uniform_kind_free`` below, called by both binders in both registries, in
// both directions.
//
// The remaining seam, stated rather than hidden: once the sidecar's ``params`` block
// becomes decl-carrying ``{name, dtype}`` (the Python
// consumers window) and ``parse_sidecar`` grows a reader for it, the registries can
// additionally cross-check the CALLER's choice against the ARTIFACT's declaration —
// the ``mutable`` role's ``sidecar.mutables`` name->dtype lookup is the pattern to
// copy, and this file is where that check belongs. That is a sidecar.h change and is
// outside this window's editable surface; nothing here presumes it, and no
// placeholder field is frozen for it.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

namespace eagle {
namespace plugin {

/// The by-value ABI type of an INTEGER uniform: 64-bit signed, matching the
/// ``using Int = long long;`` every generated kernel declares (see the header note).
using UniformInt = std::int64_t;
static_assert(sizeof(UniformInt) == sizeof(long long),
              "UniformInt must be the 8-byte slot a generated kernel's "
              "'AETHER_GRID_CONSTANT() Int p_<name>' parameter occupies");

/// Which of the two by-value spellings a uniform binding occupies.
enum class UniformKind { Real, Int };

/// The wire/sidecar spelling of a uniform kind, for an error message.
inline const char* uniform_kind_name(UniformKind kind) {
    return kind == UniformKind::Int ? "int64" : "float64";
}

/// The binder a caller must use for a uniform kind, for an error message.
inline const char* uniform_binder_name(UniformKind kind) {
    return kind == UniformKind::Int ? "bind_uniform_int" : "bind_uniform";
}

/// Refuse a uniform binding that would give ONE name TWO by-value types.
///
/// Called by both binders of both registries before they record anything, so the
/// refusal fires in both directions (int over an existing float64 binding, and
/// float64 over an existing int one) with one wording. Re-binding the SAME kind is
/// untouched — it overwrites, exactly as ``bind_uniform`` always has.
///
/// @p subject names the registry face in the message (e.g. "plugin registry",
/// "host plugin registry"), matching the ``check_aether_abi`` convention.
inline void check_uniform_kind_free(UniformKind incoming, bool bound_real,
                                    bool bound_int, const std::string& name,
                                    const std::string& subject) {
    const bool clash = (incoming == UniformKind::Int) ? bound_real : bound_int;
    if (!clash) return;
    const UniformKind held =
        (incoming == UniformKind::Int) ? UniformKind::Real : UniformKind::Int;
    throw std::runtime_error(subject + ": uniform '" + name + "' is already bound as " +
        uniform_kind_name(held) + " and cannot also be bound as " +
        uniform_kind_name(incoming) + "; a uniform occupies ONE by-value kernel "
        "parameter slot, declared as exactly one of 'AETHER_GRID_CONSTANT() Real p_" + name +
        "' (float64) or 'AETHER_GRID_CONSTANT() Int p_" + name + "' (int64), so two bindings "
        "of one name would leave the packer to guess which 8 bytes the kernel meant. "
        "Bind it once, through the binder that matches the artifact's declaration (" +
        uniform_binder_name(held) + " for " + uniform_kind_name(held) + ", " +
        uniform_binder_name(incoming) + " for " + uniform_kind_name(incoming) + ").");
}

}  // namespace plugin
}  // namespace eagle
