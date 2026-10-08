// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Host-side ABI mirrors + validation for the AETHER array views a generated/hawk
// kernel receives by value. This header deliberately includes **no aether
// headers** — the whole point of the precompiled host is that accepting a
// kernelized plugin needs only the binary ABI (these PODs), the PTX, and its
// metadata sidecar; not the aether source, nvcc, or CuPy.
//
// SPLIT HISTORY: the device-clean half of this file -- the
// GRefMirror/ScalarHandle/IntHandle/PartitionTriple mirror PODs, the
// kEagleAbi* constants, EAGLE_ABI_INDEX_T, and the sizeof/alignof pins — moved
// to `plugin/gref_layout.h`, which this file now `#include`s. The reason is
// the NVRTC profile: a device translation unit compiled through NVRTC
// reaches every payload header from memory, and NVRTC's EDG front end
// has no host `<stdexcept>`/`<string>` at all — a device TU may never reach
// them, not even behind an unused function. eagle's own device-side plugin
// fixtures and every HAWK-emitted device kernel now include ONLY
// `gref_layout.h`; this file — `#include <stdexcept>`, `#include <string>`,
// exceptions — is a HOST-only header. No symbol is renamed: everything that
// used to live here is still reachable exactly the same way, either from this
// file directly (the validation functions) or transitively through the
// `gref_layout.h` include below (the PODs and constants).
//
// The layouts are pinned by static_asserts in `gref_layout.h` and cross-checked
// against the real aether types by tests/test_gref_layout.py and
// tests/test_handle_layout.py.
#pragma once

#include "gref_layout.h"

#include <stdexcept>
#include <string>

namespace eagle {
namespace plugin {

// Binary-ABI version tag of the POD mirrors in `gref_layout.h`. MUST match
// The code generator's AETHER_ABI_VERSION: a sidecar/manifest built with a different
// value is rejected at load (see plugin_registry/registry.h and plugin.py). The
// static_asserts in `gref_layout.h` guard the layout at compile time; this tag
// guards the artifact-vs-loader boundary at run time. BUMP BOTH this and
// config.AETHER_ABI_VERSION whenever the GRef/HandleT layout changes.
#define EAGLE_AETHER_ABI "aether-abi/1"

// The HETEROGENEOUS EXECUTION contract's tag. Same POD
// mirrors, one added obligation: a v2 artifact takes the int64 partition TRIPLE
// `{base, count, nSamples}` after its role args and MUST export
// `eagle_layout_sizes` (below) so a layout mismatch fails loud instead of decoding
// garbage.
#define EAGLE_AETHER_ABI_V2 "aether-abi/2"

// The ONE binary-ABI gate every C++ door calls.
//
// ``aether_abi`` is a compatibility GATE, not a schema discriminant: it does not
// select a variant, it certifies that the artifact's POD layout (GRefMirror /
// ScalarHandle above) matches this host's. That is exactly the kind of fact
// that must never be inferred leniently, so the check is PRESENCE-required —
// an EMPTY tag is rejected too, not treated as "absent -> trust it". Every
// launching entry point in the seven-door register (plugin/roles.h) that reads
// a manifest or a sidecar routes its tag through here: four call sites, ONE
// spelling, ONE message shape. An inline re-implementation at a new door would
// be a second spelling that can drift out of step with this one — the failure
// class this one-check architecture exists to prevent.
//
// ``what`` names the artifact and its level, e.g. "plugin manifest",
// "plugin 'gravity' sidecar", "host plugin 'addvec': sidecar". The message
// carries the shared substrings ``built for`` and ``aether_abi`` that the
// conformance corpus and the C++ hardening tests check.
// The ABI generation a tag names: 1 for `aether-abi/1`, 2 for `aether-abi/2`, and
// 0 for anything else (absent, empty, or a version this build cannot speak). The
// value DISPATCHES: a v2 plugin takes the partition triple and carries a layout
// self-check symbol, a v1 one takes neither and may only ever be launched
// whole-view (the legacy bridge).
inline int abi_version_of(const std::string& tag) {
    if (tag == EAGLE_AETHER_ABI) return 1;
    if (tag == EAGLE_AETHER_ABI_V2) return 2;
    return 0;
}

// A "'aether-abi/1', 'aether-abi/2'" rendering of the tags this build speaks.
inline std::string aether_abi_tags_joined() {
    return "'" EAGLE_AETHER_ABI "', '" EAGLE_AETHER_ABI_V2 "'";
}

inline void check_aether_abi(const std::string& tag, const std::string& what) {
    // BOTH generations load: v1 is today's whole-view,
    // single-device behaviour byte for byte, v2 the partitioned contract. The
    // presence-required semantics are unchanged — an EMPTY tag is still rejected,
    // never treated as "absent -> trust it" — and so is the "built for" leading
    // clause the conformance corpus pins across the language boundary.
    if (abi_version_of(tag) == 0)
        throw std::runtime_error(what + " built for aether_abi '" + tag +
            "' but this host expects one of " + aether_abi_tags_joined() +
            "; rebuild the plugin");
}

// The human name of field @p i — what a refusal says instead of an index.
inline const char* layout_field_name(std::size_t i) {
    switch (i) {
        case 0: return "sizeof(GRefMirror)";
        case 1: return "sizeof(ScalarHandle)";
        case 2: return "sizeof(IntHandle)";
        case 3: return "sizeof(aether::idx_t)";
        case 4: return "sizeof(partition triple)";
        default: return "<unknown layout field>";
    }
}

// This build's own layout sizes, in the array's field order (`gref_layout.h`'s
// `kEagleLayoutFieldCount`/`kEagleLayoutSymbol`).
inline void expected_layout_sizes(std::uint64_t out[kEagleLayoutFieldCount]) {
    out[0] = sizeof(GRefMirror);
    out[1] = sizeof(ScalarHandle);
    out[2] = sizeof(IntHandle);
    out[3] = sizeof(EAGLE_ABI_INDEX_T);
    out[4] = sizeof(PartitionTriple);
}

// Compare an artifact's exported sizes against this host's, refusing the FIRST
// disagreement by name. `what` names the artifact and its level, exactly like
// `check_aether_abi`'s.
inline void check_layout_sizes(const std::uint64_t* got, const std::string& what) {
    if (got == nullptr)
        throw std::runtime_error(what + ": aether-abi/2 requires the exported '" +
            std::string(kEagleLayoutSymbol) + "' layout self-check symbol, which "
            "this artifact does not export; rebuild the plugin");
    std::uint64_t want[kEagleLayoutFieldCount];
    expected_layout_sizes(want);
    for (std::size_t i = 0; i < kEagleLayoutFieldCount; ++i)
        if (got[i] != want[i])
            throw std::runtime_error(what + ": aether-abi/2 layout mismatch in " +
                layout_field_name(i) + ": the artifact was built with " +
                std::to_string(got[i]) + " bytes, this host has " +
                std::to_string(want[i]) + "; rebuild the plugin against this "
                "aether configuration");
}

}  // namespace plugin
}  // namespace eagle
