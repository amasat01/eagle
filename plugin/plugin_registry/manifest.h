// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Minimal reader for a plugin-set ``manifest.json``.
//
// An Bundle (Python) emits one manifest describing N plugins, each with its own
// artifact + sidecar. Like sidecar.h, this is a small purpose-built scan over
// our own fixed, machine-generated schema rather than a general JSON parser — no
// third-party dependency, no CuPy. The schema is:
//
//   { "version": 1,
//     "plugins": [
//       { "id": "gravity", "order": 0, "enabled": true,
//         "artifact": "gravity.ptx", "sidecar": "gravity.json", "format": "ptx" },
//       ... ] }
//
// Plugins are returned in declared order, which is the injection order.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "../gref_abi.h"  // abi_version_of / the aether-abi tag macros
#include "../sidecar.h"   // reuse _slurp / _string_value / _string_array / _has_key

namespace eagle {
namespace plugin {

struct ManifestEntry {
    std::string id;
    int order = 0;
    bool enabled = true;
    std::string artifact;   // filename, relative to the manifest's directory
    std::string sidecar;    // filename, relative to the manifest's directory
    std::string format;     // "ptx" | "cubin" | "fatbin"
};

struct Manifest {
    std::string dir;                      // manifest's directory (trailing '/')
    std::string aether_abi;                 // binary-ABI tag (empty if absent)
    int schema_version = 0;               // plugin-schema version (0 = absent -> v1)
    std::string pattern;                  // "vector" | "pure" | "neural_block" (the plugin family)
    bool has_blocks = false;              // a top-level "blocks[]" key is present (parse-pure record)
    // ---- the schema-v2 EXECUTION AXIS --------------
    // FLAT top-level keys, not a nested ``execution`` object: this reader is a
    // purpose-built string scanner over a strict-key document (see the file header),
    // and every other axis it already reads (``pattern``, ``aether_abi``) is flat.
    // raptor's schema is the source of record for the vocabulary
    // (``raptor/schema/manifest.py``'s ``EXEC_TARGETS`` / ``EXEC_ACCESS_CLASSES`` /
    // ``EXEC_OPS`` / ``EXEC_KEYS``); this is its C++ half. PRESENCE is recorded
    // separately from the value because absence and empty mean different things
    // here: a v2 document missing the axis is REFUSED, never defaulted.
    bool has_exec_targets = false;
    bool has_exec_access  = false;
    bool has_exec_op      = false;
    std::vector<std::string> exec_targets;  // "device" | "host"
    std::string exec_access;                // L3's access class
    std::string exec_op;                    // "sum"|"times"|"max"|"land" (mapreduce only)
    std::vector<ManifestEntry> plugins;   // in declared (== injection) order
};

// The schema-v2 execution-axis vocabularies. KEEP IN SYNC with raptor
// (``raptor/schema/manifest.py``: ``EXEC_TARGETS``, ``EXEC_ACCESS_CLASSES``,
// ``EXEC_OPS``) — raptor owns the SHAPE, this header is the
// C++ loader's copy of the vocabulary it must gate on.
inline constexpr const char* const kExecTargets[] = { "device", "host" };
inline constexpr const char* const kExecAccessClasses[] = {
    "cross_sample_read", "cross_sample_write", "mapreduce", "sample_local",
};
inline constexpr const char* const kExecOps[] = { "land", "max", "sum", "times" };

inline std::string _joined(const char* const* v, std::size_t n) {
    std::string s;
    for (std::size_t i = 0; i < n; ++i) { if (i) s += ", "; s += v[i]; }
    return s;
}
inline std::string exec_targets_joined() { return _joined(kExecTargets, 2); }
inline std::string exec_access_classes_joined() { return _joined(kExecAccessClasses, 4); }
inline std::string exec_ops_joined() { return _joined(kExecOps, 4); }

inline bool is_valid_exec_target(const std::string& t) {
    for (const char* v : kExecTargets) if (t == v) return true;
    return false;
}
inline bool is_valid_exec_access(const std::string& a) {
    for (const char* v : kExecAccessClasses) if (a == v) return true;
    return false;
}
inline bool is_valid_exec_op(const std::string& o) {
    for (const char* v : kExecOps) if (o == v) return true;
    return false;
}

// The bool that follows ``"key":`` (bounded to its value token so a later key's
// value cannot leak in).
inline bool _bool_value(const std::string& s, const std::string& key) {
    auto k = s.find("\"" + key + "\"");
    if (k == std::string::npos) throw std::runtime_error("manifest missing key: " + key);
    auto c = s.find(':', k) + 1;
    auto end = s.find_first_of(",}", c);
    const std::string val = s.substr(c, (end == std::string::npos ? s.size() : end) - c);
    return val.find("true") != std::string::npos;
}

// The integer that follows ``"key":`` (atoi stops at the first non-digit).
inline int _int_value(const std::string& s, const std::string& key) {
    auto k = s.find("\"" + key + "\"");
    if (k == std::string::npos) throw std::runtime_error("manifest missing key: " + key);
    auto c = s.find(':', k) + 1;
    return std::atoi(s.c_str() + c);
}

inline Manifest parse_manifest(const std::string& path) {
    const std::string s = _slurp(path);
    Manifest m;
    auto slash = path.find_last_of('/');
    m.dir = (slash == std::string::npos) ? std::string("./")
                                         : path.substr(0, slash + 1);
    // Optional binary-ABI tag (absent in an older manifest -> empty string).
    if (s.find("\"aether_abi\"") != std::string::npos)
        m.aether_abi = _string_value(s, "aether_abi");
    // Optional kind tag: the plugin family ("vector" | "pure").
    if (s.find("\"pattern\"") != std::string::npos)
        m.pattern = _string_value(s, "pattern");
    // Parse-pure record; the VALIDATION lives in validate_manifest_formats:
    // presence of a top-level "blocks" array marks a neural-block manifest carriage.
    m.has_blocks = s.find("\"blocks\"") != std::string::npos;
    // Optional plugin-schema version; the legacy "version" key is a fallback.
    // Absent (both) -> 0, treated as v1 by the registries (backward-lenient).
    if (s.find("\"schema_version\"") != std::string::npos)
        m.schema_version = _int_value(s, "schema_version");
    else if (s.find("\"version\"") != std::string::npos)
        m.schema_version = _int_value(s, "version");

    // The schema-v2 execution axis. Parsed unconditionally and
    // validation-free, like every other field this scanner reads; the version-
    // conditional REQUIREDNESS and the vocabulary gates live in
    // ``validate_execution_axis`` below, which both registries call.
    if (_has_key(s, "exec_targets")) {
        m.has_exec_targets = true;
        m.exec_targets = _string_array(s, "exec_targets");
    }
    if (_has_key(s, "exec_access")) {
        m.has_exec_access = true;
        m.exec_access = _string_value(s, "exec_access");
    }
    if (_has_key(s, "exec_op")) {
        m.has_exec_op = true;
        m.exec_op = _string_value(s, "exec_op");
    }

    auto p = s.find("\"plugins\"");
    if (p == std::string::npos) throw std::runtime_error("manifest missing plugins");
    auto open = s.find('[', p);
    if (open == std::string::npos) throw std::runtime_error("malformed plugins array");
    int depth = 0;
    std::size_t close = std::string::npos;
    for (std::size_t i = open; i < s.size(); ++i) {
        if (s[i] == '[') ++depth;
        else if (s[i] == ']' && --depth == 0) { close = i; break; }
    }
    if (close == std::string::npos)
        throw std::runtime_error("truncated plugins array (no matching ']')");

    // Walk each { ... } object between the array brackets.
    std::size_t i = open;
    while (true) {
        auto ob = s.find('{', i);
        if (ob == std::string::npos || ob >= close) break;
        int d = 0;
        std::size_t oe = std::string::npos;
        for (std::size_t j = ob; j < s.size(); ++j) {
            if (s[j] == '{') ++d;
            else if (s[j] == '}' && --d == 0) { oe = j; break; }
        }
        if (oe == std::string::npos || oe > close)
            throw std::runtime_error("truncated plugin object");
        const std::string obj = s.substr(ob, oe - ob + 1);
        ManifestEntry e;
        e.id = _string_value(obj, "id");
        e.artifact = _string_value(obj, "artifact");
        e.sidecar = _string_value(obj, "sidecar");
        e.format = _string_value(obj, "format");
        e.order = _int_value(obj, "order");
        e.enabled = _bool_value(obj, "enabled");
        m.plugins.push_back(e);
        i = oe + 1;
    }
    if (m.plugins.empty()) throw std::runtime_error("manifest has no plugins");
    return m;
}

// ---------------------------------------------------------------------------
// The schema-v2 EXECUTION AXIS gate
// ---------------------------------------------------------------------------
// Shared by BOTH registries, exactly like ``validate_manifest_formats`` above, and
// for the same reason: a check with two spellings is a check that drifts.
//
// The rules, and the WHY of each:
//   * the ``schema_version`` and the ``aether_abi`` tag must AGREE ("v1
//     loaders refuse v2 plugins and vice versa"). A v1-tagged document stamped
//     schema 2 would be read as partitionable while its artifacts take no triple.
//   * schema v1 must carry NONE of the execution keys — the legacy bridge is
//     whole-view/single-device BY THE AXIS'S ABSENCE, so a v1 document naming one
//     is a strict-key violation, never a silent upgrade.
//   * schema v2 REQUIRES ``exec_targets`` + ``exec_access``: legality is DERIVED
//     from the declaration and never guessed, so absence is a load refusal,
//     not a default.
//   * ``exec_op`` is required IFF ``exec_access == "mapreduce"`` and forbidden
//     otherwise (L3's ``mapreduce(op)`` is the only class with a combine).
//
// Message shapes mirror raptor's ``check_execution_axis`` leading clauses, which is
// what keeps the C++ and Python refusals recognisably the same rule.
inline void validate_execution_axis(const Manifest& m, const std::string& what) {
    const int version = (m.schema_version == 0) ? 1 : m.schema_version;
    if (!m.aether_abi.empty()) {
        const int tag_version = abi_version_of(m.aether_abi);
        if (tag_version != version)
            throw std::runtime_error(what + ": schema v" + std::to_string(version) +
                " requires aether_abi '" + (version == 2 ? EAGLE_AETHER_ABI_V2
                                                         : EAGLE_AETHER_ABI) +
                "'; got '" + m.aether_abi + "'");
    }
    if (version < 2) {
        std::string present;
        if (m.has_exec_access)  present += (present.empty() ? "" : ", ") + std::string("exec_access");
        if (m.has_exec_op)      present += (present.empty() ? "" : ", ") + std::string("exec_op");
        if (m.has_exec_targets) present += (present.empty() ? "" : ", ") + std::string("exec_targets");
        if (!present.empty())
            throw std::runtime_error(what + ": schema v1 must not carry execution "
                "key(s) " + present + " (the execution axis is schema-v2-only; set "
                "schema_version=2 and aether_abi='" EAGLE_AETHER_ABI_V2 "' to use it)");
        return;
    }
    std::string missing;
    if (!m.has_exec_access)  missing += "exec_access";
    if (!m.has_exec_targets) missing += (missing.empty() ? "" : ", ") + std::string("exec_targets");
    if (!missing.empty())
        throw std::runtime_error(what + ": schema v2 requires execution key(s) " +
            missing + " (absence = load refused, L8)");

    bool targets_ok = !m.exec_targets.empty();
    for (std::size_t i = 0; targets_ok && i < m.exec_targets.size(); ++i) {
        if (!is_valid_exec_target(m.exec_targets[i])) targets_ok = false;
        for (std::size_t j = 0; j < i; ++j)
            if (m.exec_targets[i] == m.exec_targets[j]) targets_ok = false;
    }
    if (!targets_ok) {
        std::string got;
        for (std::size_t i = 0; i < m.exec_targets.size(); ++i) {
            if (i) got += ", ";
            got += "'" + m.exec_targets[i] + "'";
        }
        throw std::runtime_error(what + ": exec_targets must be a non-empty list "
            "drawn from " + exec_targets_joined() + " with no duplicates; got [" +
            got + "]");
    }
    if (!is_valid_exec_access(m.exec_access))
        throw std::runtime_error(what + ": exec_access '" + m.exec_access +
            "' is not a supported access class (supported: " +
            exec_access_classes_joined() + ")");
    if (m.exec_access == "mapreduce") {
        if (!m.has_exec_op)
            throw std::runtime_error(what + ": exec_access='mapreduce' requires "
                "exec_op (one of " + exec_ops_joined() + ")");
    } else if (m.has_exec_op) {
        throw std::runtime_error(what + ": exec_op is forbidden unless "
            "exec_access='mapreduce' (exec_access='" + m.exec_access + "')");
    }
    if (m.has_exec_op && !is_valid_exec_op(m.exec_op))
        throw std::runtime_error(what + ": exec_op '" + m.exec_op + "' is not a "
            "supported reduction op (supported: " + exec_ops_joined() + ")");
}

// Validate every declared entry's ``format`` against the canonical schema-v1
// vocabulary (plugin/roles.h ``kManifestFormats`` — matches the producer of
// record, the code generator's bundling). ``format`` is a schema DISCRIMINANT
// it selects HOW the registry must load the artifact bytes a manifest
// entry names, so an unrecognized value must be refused rather than handed to
// ``cuModuleLoadData`` (or dlopen'd) as though it were a known container. Shared
// by both registries (mirrors ``validate_sidecar``'s "one check, not N" property):
// called over every entry BEFORE any sidecar parse / module load, so a bad
// manifest fails fast with no partial load — the manifest-level twin of the
// duplicate-id structural check both registries already run at this point.
inline void validate_manifest_formats(const Manifest& m) {
    // The orphaned-blocks guard: a "blocks[]" section is neural-block machinery
    // reachable ONLY under pattern "neural_block" — and the pattern gate already
    // refused that family before this shared helper runs (both registries'
    // `check_launch_certified_pattern` call). A "blocks[]" reaching here means a producer
    // emitted descriptors beside a kernel manifest; a kernel-manifest loader would
    // SILENTLY ignore them and load bare kernels (a live mis-read hazard). Refuse loudly,
    // sharing the Python door's leading clause verbatim.
    if (m.has_blocks)
        throw std::runtime_error("manifest carrying blocks[] must declare pattern "
            "'neural_block'; got pattern '" + m.pattern + "' (block descriptors would "
            "be silently ignored by a kernel-manifest loader)");
    for (const auto& e : m.plugins)
        if (!is_valid_manifest_format(e.format))
            throw std::runtime_error("manifest plugin '" + e.id +
                "': unknown artifact format '" + e.format + "' (supported: " +
                manifest_formats_joined() + "); upgrade eagle");
}

}  // namespace plugin
}  // namespace eagle
