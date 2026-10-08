// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Minimal reader for a PTX-plugin's ``.json`` sidecar.
//
// The sidecar is the metadata a deployed plugin ships next to its PTX; a host
// reconstructs the launch signature from it without any Python dependency.
// We only need two fields — the extern-C kernel name and the ordered arg_spec —
// so this is a small purpose-built scan over our own fixed, machine-generated
// schema rather than a general JSON parser.
#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "roles.h"   // the canonical arg-spec role vocabulary + schema version

namespace eagle {
namespace plugin {

struct ArgEntry {
    // ``role`` is one of the canonical schema-v1 arg-spec roles enumerated in
    // plugin/roles.h (out | vec_in | mat_in | per_sample | lookup | mutable |
    // terminated | uniform | nsamples). validate_sidecar() rejects anything
    // outside that set at load time.
    std::string role;
    std::string name;
};

// A pure kernel's writable per-sample state (the ``mutable`` role) — the pure output.
// ``dtype`` picks the by-value ABI: a "vector" / "matrix" Mutable rides the ``(W, N)`` /
// ``(R*C, N)`` GRef path (like ``out``/``vec_in``/``mat_in``); a "float"/"int" one a flat
// scalar handle. ``width`` is the SoA row count (1 for a scalar, R*C for a matrix). A
// "matrix" Mutable's (R, C) is recorded in ``Sidecar::mat_shapes`` (keyed by name).
struct MutableInfo {
    std::string name;
    std::string dtype;  // "float" | "int" | "vector" | "matrix"
    int width = 1;
};

// A declared read-only buffer (a lookup table or a shared constant). ``count``
// (== prod(shape)) is the flat element count the registry allocates + uploads at
// consolidation and binds by value as a flat scalar handle of that many Reals.
struct BufferInfo {
    std::string name;
    std::string kind;   // "lookup"
    std::string dtype;  // "float"
    std::size_t count = 0;
};

// Optional derivative-kernel metadata (an external autodiff tool's output; schema-v1 ADDITIVE field).
// Present only for a VJP/JVP derivative artifact — GENERATED (``build_vjp``/``build_jvp``)
// or CUSTOM (``set_custom_vjp``/``set_custom_jvp``); the two are structurally
// indistinguishable here (one emission path). A primal / ordinary kernel leaves
// ``present`` false. Phase A is recompute-only: the artifact re-reads its primal inputs,
// so ``residual_policy`` is always "recompute" and ``residuals`` is empty — a populated
// ``residuals`` list is a FUTURE schema revision (validate_sidecar rejects it, naming the
// schema-bump rule). The device / host launch ignores this block entirely: the artifact
// still runs as an ordinary pure kernel without a loader reading it.
struct Derivative {
    bool present = false;
    std::string kind;             // "vjp" | "jvp"
    std::string primal;           // the primal kernel's logical id
    std::vector<std::string> wrt; // differentiated input names
    std::string residual_policy;  // "recompute" (Phase A)
    bool residuals_populated = false;  // Phase A: always false (residuals == [])
};

// One ``*_exec`` reference on a ``neural_block`` descriptor:
// ``{"kind": "kernel", "kernel": "<id>"}``. It names WHICH artifact implements a leg of
// the block (forward / vjp / jvp) and HOW the reference must be read. The referenced
// artifact is an ordinary ``pure`` plugin that loads through the normal doors — this is
// a NAME, never a launch target in itself.
struct ExecRef {
    bool present = false;
    std::string kind;    // value-strict against roles.h kExecRefKinds
    std::string kernel;  // the referenced artifact's logical id
    // Any top-level key OTHER than "kind"/"kernel" seen while parsing this object
    // an exec ref is a fixed two-key
    // shape, so a third key is a defect the validator must see. Populated at parse
    // time (parse_sidecar), because by the time validate_sidecar runs the raw text
    // is gone — this is the same "capture now, check later" split every other
    // ExecRef field already follows.
    std::vector<std::string> unknown_keys;
};

// The parse-side surface of a ``neural_block`` DESCRIPTOR (C++ gets parse +
// validate, NOT a public descriptor type). Filled by parse_sidecar
// for every sidecar — the fields simply stay absent on a kernel sidecar — and consumed
// ONLY by validate_sidecar's pattern-conditional clause. A typed C++ descriptor, and the
// layer/buffer types beside it, land with their first C++ consumer: freezing a
// struct layout nobody reads is the compile-probe lesson inverted.
//
// Everything is keyed by name because the requiredness loop ITERATES roles.h's
// ``kNeuralRequiredFields`` rather than hand-writing a check per field — so ABSENT must
// be distinguishable from zero, which is what the map (not a struct of ints) buys.
struct NeuralInfo {
    std::map<std::string, long long> ints;  // present integer fields only
    bool has_scatter_policy = false;
    std::string scatter_policy;
    std::map<std::string, ExecRef> execs;   // keyed by kNeuralExecRefFields name
};

struct Sidecar {
    std::string kernel;
    std::string aether_abi;  // binary-ABI tag (empty if an older sidecar lacks it)
    // Plugin family ("vector" | "pure"), schema-v1's variant DISCRIMINANT.
    // Empty if an older/hand-built sidecar never stamped one — validate_sidecar
    // stays absence-lenient (pre-freeze sidecars), value-strict otherwise.
    std::string pattern;
    int schema_version = 0;   // plugin-schema version (0 = absent -> treated as v1)
    std::string scalar_type;  // "float64"|"float32"|"softdouble" ("" = pre-dtype
                              // sidecar, treated as float64)
    // Optional dlopen'd host entry-point symbol for the CPU plugin path (the
    // reserved ``launch`` capability, host face). Empty -> the HostPluginRegistry
    // defaults to ``<kernel>_host``. The device path ignores this field.
    std::string host_entry;
    std::vector<ArgEntry> arg_spec;
    std::vector<BufferInfo> buffers;  // declared buffers (may be empty)
    std::vector<MutableInfo> mutables;  // pure-kernel writable state (may be empty)
    // (R, C) per matrix argument name — both ``mat_in`` inputs (from the sidecar's
    // ``mat_shapes`` object) and matrix Mutables (from their ``shape``). A host packs a
    // matrix through the SAME GRef mirror as a vector, so this is used only for the
    // bind-time shape guard; empty for a matrix-free kernel (backward-lenient).
    std::map<std::string, std::pair<int, int>> mat_shapes;
    // Optional derivative-kernel metadata (absent on a primal / ordinary kernel).
    Derivative derivative;
    // The ``neural_block`` descriptor surface (inert on a kernel sidecar).
    NeuralInfo neural;
};

inline std::string _slurp(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open sidecar: " + path);
    std::ostringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// Position of the KEY occurrence of ``"key"`` in ``s`` at or after ``from`` — an
// occurrence whose next non-whitespace character is ':'. A bare ``"kernel"`` appearing
// as a VALUE (``"kind": "kernel"``) is skipped, which a plain ``find`` cannot do. This
// is half of the fix; the other half (masking nested objects) is in
// parse_sidecar, because a nested KEY of the same name is still a real key.
inline std::size_t _key_pos(const std::string& s, const std::string& key,
                            std::size_t from = 0) {
    const std::string q = "\"" + key + "\"";
    for (auto k = s.find(q, from); k != std::string::npos; k = s.find(q, k + 1)) {
        auto after = s.find_first_not_of(" \t\r\n", k + q.size());
        if (after != std::string::npos && s[after] == ':') return k;
    }
    return std::string::npos;
}

// Whether ``s`` declares ``key`` as a key at all (not merely as some value's text).
inline bool _has_key(const std::string& s, const std::string& key) {
    return _key_pos(s, key) != std::string::npos;
}

// The top-level key names of a FLAT ``{...}`` object whose values are all quoted
// strings (an exec reference: only ``kind``/``kernel`` are defined, but an
// unrecognized key must be SEEN before it can be rejected). Every quoted token in
// ``obj`` whose next non-whitespace character is ':' is a key — the same "is this
// occurrence a KEY" test ``_key_pos`` uses per-name, generalized here from "does the
// object contain THIS key" to "what keys does the object contain". A value token is
// never mistaken for a key, because a value's closing quote is followed by ',' or
// '}', not ':'.
inline std::vector<std::string> _object_keys(const std::string& obj) {
    std::vector<std::string> keys;
    for (std::size_t i = 0; i < obj.size();) {
        auto q1 = obj.find('"', i);
        if (q1 == std::string::npos) break;
        auto q2 = obj.find('"', q1 + 1);
        if (q2 == std::string::npos) break;
        auto after = obj.find_first_not_of(" \t\r\n", q2 + 1);
        if (after != std::string::npos && obj[after] == ':')
            keys.push_back(obj.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return keys;
}

// Return the quoted string value that follows ``"key"`` in ``s`` (the next
// double-quoted token after the key's colon).
inline std::string _string_value(const std::string& s, const std::string& key) {
    auto k = _key_pos(s, key);
    if (k == std::string::npos) throw std::runtime_error("sidecar missing key: " + key);
    auto q1 = s.find('"', s.find(':', k) + 1);
    auto q2 = s.find('"', q1 + 1);
    return s.substr(q1 + 1, q2 - q1 - 1);
}

// Return the quoted strings of the array that follows ``"key"`` in ``s`` (e.g. the
// ``wrt`` name list). Absent key -> empty vector.
inline std::vector<std::string> _string_array(const std::string& s,
                                              const std::string& key) {
    std::vector<std::string> out;
    auto k = _key_pos(s, key);
    if (k == std::string::npos) return out;
    auto open = s.find('[', k);
    if (open == std::string::npos) return out;
    auto close = s.find(']', open);
    if (close == std::string::npos) return out;
    for (std::size_t i = open + 1; i < close;) {
        auto q1 = s.find('"', i);
        if (q1 == std::string::npos || q1 >= close) break;
        auto q2 = s.find('"', q1 + 1);
        out.push_back(s.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    return out;
}

// Whether the array that follows ``"key"`` in ``s`` has any non-whitespace content
// between its ``[`` and ``]`` (i.e. is a *populated* list). ``residuals: []`` -> false.
inline bool _array_populated(const std::string& s, const std::string& key) {
    auto k = _key_pos(s, key);
    if (k == std::string::npos) return false;
    auto open = s.find('[', k);
    if (open == std::string::npos) return false;
    auto close = s.find(']', open);
    if (close == std::string::npos) return false;
    for (std::size_t i = open + 1; i < close; ++i)
        if (!std::isspace(static_cast<unsigned char>(s[i]))) return true;
    return false;
}

inline Sidecar parse_sidecar(const std::string& path) {
    const std::string raw = _slurp(path);
    Sidecar sc;

    // ---- NESTING SAFETY, and it is not optional --------------------
    // This scanner finds a key by text, so a NESTED object's keys are visible to every
    // top-level scan. On a ``neural_block`` descriptor that is not theoretical: an exec
    // reference is literally ``{"kind": "kernel", "kernel": "<id>"}``, so a descriptor
    // whose ``forward_exec`` precedes its own ``kernel`` key parses its wire identity as
    // the FORWARD kernel's id. Reproduced against this very header before the fix
    // (``forward_exec`` first -> kernel=[rhs_fwd]; ``kernel`` first -> correct), and the
    // corpus golden is deliberately written in the adversarial order so the confusion
    // case is EXECUTED rather than reasoned about.
    //
    // The fix is structural, not a smarter ``kernel`` scan: bound every nested ``*_exec``
    // object FIRST, parse each one from the raw text, then BLANK those regions (spaces,
    // so every offset is preserved) and scan all top-level keys from the masked copy. A
    // sidecar with no exec objects is masked nowhere and parses exactly as before.
    std::string s = raw;
    for (const char* key : kNeuralExecRefFields) {
        auto kp = _key_pos(raw, key);
        if (kp == std::string::npos) continue;
        auto ob = raw.find('{', kp);
        if (ob == std::string::npos) continue;
        int d = 0;
        std::size_t oe = std::string::npos;
        for (std::size_t j = ob; j < raw.size(); ++j) {
            if (raw[j] == '{') ++d;
            else if (raw[j] == '}' && --d == 0) { oe = j; break; }
        }
        if (oe == std::string::npos) continue;  // truncated -> leave absent, reject later
        const std::string obj = raw.substr(ob, oe - ob + 1);
        ExecRef ref;
        ref.present = true;
        if (_has_key(obj, "kind"))   ref.kind   = _string_value(obj, "kind");
        if (_has_key(obj, "kernel")) ref.kernel = _string_value(obj, "kernel");
        for (const auto& k : _object_keys(obj))
            if (k != "kind" && k != "kernel") ref.unknown_keys.push_back(k);
        sc.neural.execs[key] = ref;
        std::fill(s.begin() + static_cast<std::ptrdiff_t>(kp),
                  s.begin() + static_cast<std::ptrdiff_t>(oe) + 1, ' ');
    }

    sc.kernel = _string_value(s, "kernel");
    // Optional binary-ABI tag (absent in an older sidecar -> empty string).
    if (_has_key(s, "aether_abi"))
        sc.aether_abi = _string_value(s, "aether_abi");
    // Optional plugin-family discriminant (absent -> empty string; validate_sidecar
    // decides whether that is lenient). Parsing stays validation-free here, like
    // every other field this scanner reads — see validate_sidecar for the check.
    if (_has_key(s, "pattern"))
        sc.pattern = _string_value(s, "pattern");
    if (_has_key(s, "scalar_type"))
        sc.scalar_type = _string_value(s, "scalar_type");
    // Optional host entry-point symbol for the CPU plugin path.
    if (_has_key(s, "host_entry"))
        sc.host_entry = _string_value(s, "host_entry");
    // Optional plugin-schema version (absent -> 0, treated as v1 downstream).
    if (auto sv = _key_pos(s, "schema_version"); sv != std::string::npos)
        sc.schema_version = std::atoi(s.c_str() + s.find(':', sv) + 1);

    // ``neural_block`` descriptor fields. Parsed unconditionally (parsing stays
    // validation-free and pattern-agnostic, like every other field here); they are read
    // only by validate_sidecar's pattern-conditional clause, so a kernel sidecar that
    // happens to carry one of these names is unaffected. The integer scan's failure mode
    // is safe BY CONSTRUCTION and pinned by a corpus row: ``strtoll`` of a non-number
    // yields 0, which the positive-integer check rejects. Which fields are integers is
    // DERIVED from roles.h (everything required that is neither the scatter contract nor
    // an exec reference), so the set cannot drift from the requiredness list.
    for (const char* f : kNeuralRequiredFields) {
        const std::string field = f;
        if (field == "scatter_policy") continue;
        bool is_exec = false;
        for (const char* e : kNeuralExecRefFields)
            if (field == e) is_exec = true;
        if (is_exec) continue;
        if (auto p = _key_pos(s, field); p != std::string::npos)
            sc.neural.ints[field] =
                std::strtoll(s.c_str() + s.find(':', p) + 1, nullptr, 10);
    }
    if (_has_key(s, "scatter_policy")) {
        sc.neural.has_scatter_policy = true;
        sc.neural.scatter_policy = _string_value(s, "scatter_policy");
    }

    // Bound the arg_spec array: from "arg_spec" find its opening '[' and the
    // matching ']' (tracking depth), then pair up the quoted strings inside.
    auto a = _key_pos(s, "arg_spec");
    if (a == std::string::npos) throw std::runtime_error("sidecar missing arg_spec");
    auto open = s.find('[', a);
    if (open == std::string::npos) throw std::runtime_error("malformed arg_spec");
    int depth = 0;
    std::size_t close = std::string::npos;
    for (std::size_t i = open; i < s.size(); ++i) {
        if (s[i] == '[') ++depth;
        else if (s[i] == ']' && --depth == 0) { close = i; break; }
    }
    if (close == std::string::npos)
        throw std::runtime_error("truncated arg_spec (no matching ']')");
    std::vector<std::string> toks;
    for (std::size_t i = open; i < close;) {
        auto q1 = s.find('"', i);
        if (q1 == std::string::npos || q1 >= close) break;
        auto q2 = s.find('"', q1 + 1);
        toks.push_back(s.substr(q1 + 1, q2 - q1 - 1));
        i = q2 + 1;
    }
    if (toks.size() % 2 != 0) throw std::runtime_error("malformed arg_spec");
    for (std::size_t i = 0; i + 1 < toks.size(); i += 2)
        sc.arg_spec.push_back({toks[i], toks[i + 1]});

    // Optional: declared extension-hatch buffers. Absence (an older sidecar or a
    // force with no buffers) leaves ``buffers`` empty. Walk each { ... } object in
    // the "buffers" array and pull name/kind/dtype (strings) + count (int).
    auto b = _key_pos(s, "buffers");
    if (b != std::string::npos) {
        auto bopen = s.find('[', b);
        std::size_t bclose = std::string::npos;
        if (bopen != std::string::npos) {
            int bd = 0;
            for (std::size_t i = bopen; i < s.size(); ++i) {
                if (s[i] == '[') ++bd;
                else if (s[i] == ']' && --bd == 0) { bclose = i; break; }
            }
        }
        for (std::size_t i = bopen; bclose != std::string::npos && i < bclose;) {
            auto ob = s.find('{', i);
            if (ob == std::string::npos || ob >= bclose) break;
            int d = 0;
            std::size_t oe = std::string::npos;
            for (std::size_t j = ob; j < s.size(); ++j) {
                if (s[j] == '{') ++d;
                else if (s[j] == '}' && --d == 0) { oe = j; break; }
            }
            if (oe == std::string::npos || oe > bclose) break;
            const std::string obj = s.substr(ob, oe - ob + 1);
            BufferInfo bi;
            bi.name = _string_value(obj, "name");
            bi.kind = _string_value(obj, "kind");
            bi.dtype = _string_value(obj, "dtype");
            auto c = _key_pos(obj, "count");
            bi.count = (c == std::string::npos) ? 0
                : std::strtoull(obj.c_str() + obj.find(':', c) + 1, nullptr, 10);
            sc.buffers.push_back(bi);
            i = oe + 1;
        }
    }

    // Optional: a pure kernel's writable ``mutables`` (name/dtype/width). Absence (a
    // vector sidecar) leaves it empty. Walk each { ... } object like "buffers".
    auto m = _key_pos(s, "mutables");
    if (m != std::string::npos) {
        auto mopen = s.find('[', m);
        std::size_t mclose = std::string::npos;
        if (mopen != std::string::npos) {
            int md = 0;
            for (std::size_t i = mopen; i < s.size(); ++i) {
                if (s[i] == '[') ++md;
                else if (s[i] == ']' && --md == 0) { mclose = i; break; }
            }
        }
        for (std::size_t i = mopen; mclose != std::string::npos && i < mclose;) {
            auto ob = s.find('{', i);
            if (ob == std::string::npos || ob >= mclose) break;
            int d = 0;
            std::size_t oe = std::string::npos;
            for (std::size_t j = ob; j < s.size(); ++j) {
                if (s[j] == '{') ++d;
                else if (s[j] == '}' && --d == 0) { oe = j; break; }
            }
            if (oe == std::string::npos || oe > mclose) break;
            const std::string obj = s.substr(ob, oe - ob + 1);
            MutableInfo mi;
            mi.name = _string_value(obj, "name");
            mi.dtype = _string_value(obj, "dtype");
            auto w = _key_pos(obj, "width");
            mi.width = (w == std::string::npos) ? 1
                : int(std::strtol(obj.c_str() + obj.find(':', w) + 1, nullptr, 10));
            // A matrix Mutable carries its (R, C) ``shape`` — record it for the
            // bind-time shape guard (a scalar/vector Mutable has no ``shape`` key).
            if (auto sh = _key_pos(obj, "shape"); sh != std::string::npos) {
                auto lb = obj.find('[', sh);
                auto cm = obj.find(',', lb);
                sc.mat_shapes[mi.name] = {
                    int(std::strtol(obj.c_str() + lb + 1, nullptr, 10)),
                    int(std::strtol(obj.c_str() + cm + 1, nullptr, 10)),
                };
            }
            sc.mutables.push_back(mi);
            i = oe + 1;
        }
    }

    // Optional: ``mat_shapes`` — a {name: [R, C]} object of the ``mat_in`` matrix-input
    // shapes. Absence (a matrix-free sidecar) leaves the map as populated by any matrix
    // Mutables above. Walk the object, pairing each quoted key with its ``[R, C]``.
    if (auto ms = _key_pos(s, "mat_shapes"); ms != std::string::npos) {
        auto oopen = s.find('{', ms);
        std::size_t oclose = std::string::npos;
        if (oopen != std::string::npos) {
            int od = 0;
            for (std::size_t i = oopen; i < s.size(); ++i) {
                if (s[i] == '{') ++od;
                else if (s[i] == '}' && --od == 0) { oclose = i; break; }
            }
        }
        for (std::size_t i = oopen; oclose != std::string::npos && i < oclose;) {
            auto q1 = s.find('"', i);
            if (q1 == std::string::npos || q1 >= oclose) break;
            auto q2 = s.find('"', q1 + 1);
            const std::string key = s.substr(q1 + 1, q2 - q1 - 1);
            auto lb = s.find('[', q2);
            if (lb == std::string::npos || lb >= oclose) break;
            auto cm = s.find(',', lb);
            sc.mat_shapes[key] = {
                int(std::strtol(s.c_str() + lb + 1, nullptr, 10)),
                int(std::strtol(s.c_str() + cm + 1, nullptr, 10)),
            };
            i = s.find(']', cm) + 1;
        }
    }

    // Optional: the ``derivative`` block (a VJP/JVP derivative artifact's metadata).
    // Absence (a primal / ordinary kernel, or an older sidecar) leaves it inert —
    // backward-lenient. Bound the { ... } object, then pull its scalar keys + arrays.
    if (auto d = _key_pos(s, "derivative"); d != std::string::npos) {
        auto ob = s.find('{', d);
        if (ob != std::string::npos) {
            int dd = 0;
            std::size_t oe = std::string::npos;
            for (std::size_t j = ob; j < s.size(); ++j) {
                if (s[j] == '{') ++dd;
                else if (s[j] == '}' && --dd == 0) { oe = j; break; }
            }
            if (oe != std::string::npos) {
                const std::string obj = s.substr(ob, oe - ob + 1);
                sc.derivative.present = true;
                sc.derivative.kind = _string_value(obj, "kind");
                if (_has_key(obj, "primal"))
                    sc.derivative.primal = _string_value(obj, "primal");
                if (_has_key(obj, "residual_policy"))
                    sc.derivative.residual_policy =
                        _string_value(obj, "residual_policy");
                sc.derivative.wrt = _string_array(obj, "wrt");
                sc.derivative.residuals_populated =
                    _array_populated(obj, "residuals");
            }
        }
    }

    return sc;
}

// The ``neural_block`` family name. A literal here (rather than a constant) is
// DISPATCH — this clause validates exactly one family — the ``_require_pattern``
// precedent. Membership in ``kRecognizedPatterns`` is what makes the
// value legal; this is what selects the clause.
inline constexpr const char* kNeuralBlockPattern = "neural_block";

// One ``{"kind": ..., "kernel": ...}`` exec reference: strict shape, value-strict
// ``kind``. An exec ref names WHICH artifact implements a leg of the block and HOW the
// reference must be read, so an unrecognized ``kind`` is refused rather than dispatched
// on as though it were a plain kernel.
inline void _validate_exec_ref(const ExecRef& ref, const std::string& field,
                               const std::string& ctx) {
    // CLOSED shape: an exec ref is a fixed
    // two-key object, so a third key is refused by name rather than silently ignored
    // — the same "closed set" discipline the requiredness/forbidden-field loops below
    // already apply to the descriptor as a whole.
    if (!ref.unknown_keys.empty()) {
        std::vector<std::string> unknown = ref.unknown_keys;
        std::sort(unknown.begin(), unknown.end());
        std::string rendered;
        for (std::size_t i = 0; i < unknown.size(); ++i) {
            if (i) rendered += ", ";
            rendered += "'" + unknown[i] + "'";
        }
        throw std::runtime_error(ctx + ": neural_block " + field +
            " carries unknown key(s) [" + rendered + "] (exec references are closed "
            "to 'kind', 'kernel')");
    }
    if (!is_valid_exec_ref_kind(ref.kind))
        throw std::runtime_error(ctx + ": neural_block " + field + ".kind '" +
            ref.kind + "' is not a supported exec reference kind (supported: " +
            exec_ref_kinds_joined() + "); upgrade eagle");
    if (ref.kernel.empty())
        throw std::runtime_error(ctx + ": neural_block " + field +
            ".kernel must be a non-empty kernel id");
}

// The pattern-conditional ``neural_block`` clause of the shared validator —
// the C++ twin of ``eagle.sidecar._validate_neural_block``, pinned check for check and
// leading-clause for leading-clause by the shared conformance corpus.
//
// A descriptor is not a kernel: it carries no ``arg_spec`` entries, no artifact of its
// own, and nothing launches it (``neural_block`` is RECOGNIZED but never
// LAUNCH-CERTIFIED). It declares the block's shape and NAMES the kernels that implement
// it; those are ordinary ``pure`` artifacts loaded through the normal doors.
inline void _validate_neural_block(const Sidecar& sc, const std::string& ctx) {
    // The descriptor's wire identity. ``parse_sidecar`` already
    // required the KEY; this rejects an empty value.
    if (sc.kernel.empty())
        throw std::runtime_error(ctx + ": a 'neural_block' descriptor must carry a "
            "non-empty 'kernel' (its wire identity)");
    // ABSENCE-STRICT, unlike the general lenient check above: the leniencies exist for
    // pre-freeze backward compatibility, and a family minted after the freeze has no
    // backward to be compatible with. ALLOWLIST form (never a rejected-value literal),
    // which is what keeps the neural tree's SoftDouble grep meaningful.
    if (sc.scalar_type != "float32" && sc.scalar_type != "float64")
        throw std::runtime_error(ctx + ": neural_block requires scalar_type in "
            "(float32, float64); got '" + sc.scalar_type + "' (a neural block's "
            "arithmetic is native-float only)");
    // Present and EMPTY. ``parse_sidecar`` throws if the key is absent (which is why the
    // Python clause requires it explicitly — one behaviour, two enforcement points);
    // a POPULATED arg_spec means the producer built a kernel sidecar and stamped it as a
    // descriptor.
    if (!sc.arg_spec.empty())
        throw std::runtime_error(ctx + ": a 'neural_block' descriptor must carry an "
            "EMPTY arg_spec ([]); got " + std::to_string(sc.arg_spec.size()) +
            " entries (a descriptor binds nothing — the kernels it references carry "
            "their own arg_spec)");

    // Presence is ITERATED over roles.h's constant, never hand-written per field, so the
    // requiredness lives in exactly one place (single-sourced). Sorted,
    // so that a multi-defect descriptor reports the SAME missing field as Python does.
    std::vector<std::string> required(std::begin(kNeuralRequiredFields),
                                      std::end(kNeuralRequiredFields));
    std::sort(required.begin(), required.end());
    for (const auto& f : required) {
        bool present = false;
        if (f == "scatter_policy")          present = sc.neural.has_scatter_policy;
        else if (sc.neural.execs.count(f))  present = true;
        else                                present = sc.neural.ints.count(f) > 0;
        if (!present)
            throw std::runtime_error(ctx + ": missing required neural_block field '" +
                f + "' (schema-v1 neural_block requires " +
                neural_required_fields_joined() + ")");
    }

    // Widths and counts. ``param_width`` is the one that may legitimately be zero: a
    // block with no learned parameters is meaningful, a block with no inputs or no state
    // is not. The scanner is type-blind by construction (like every other integer it
    // reads), so a non-numeric value arrives here as 0 and is rejected by exactly this
    // check — the safe-by-construction failure mode pins it.
    // The upper bound every neural_block integer field must respect — uint32
    // representability, mirroring the runtime tier's cap on the DERIVED
    // N_BLOCKS/PARAM_POOL_SIZE (buffers.py's ``_UINT32_CAP``). The descriptor tier had
    // no analogous cap on its own fanin/fanout/*_width fields until this mint
    // closed the gap.
    constexpr long long kNeuralIntUpperBound = 1LL << 32;  // 2**32
    for (const auto& [field, value] : sc.neural.ints) {
        const bool non_negative = (field == "param_width");
        if ((non_negative ? (value < 0) : (value <= 0)) ||
            value >= kNeuralIntUpperBound)
            throw std::runtime_error(ctx + ": neural_block " + field + " must be " +
                (non_negative ? "a non-negative integer" : "a positive integer") +
                " representable in a uint32 (< 2**32); got " + std::to_string(value));
    }
    // A wire-checkable width lock: the block's outputs are committed into the
    // per-block state row, so they cannot be wider than it.
    const long long ow = sc.neural.ints.at("output_width");
    const long long sw = sc.neural.ints.at("state_width");
    if (ow > sw)
        throw std::runtime_error(ctx + ": neural_block output_width (" +
            std::to_string(ow) + ") must not exceed state_width (" +
            std::to_string(sw) + ")");

    if (!is_valid_scatter_policy(sc.neural.scatter_policy))
        throw std::runtime_error(ctx + ": neural_block scatter_policy '" +
            sc.neural.scatter_policy + "' is not a supported terminal-write contract "
            "(supported: " + scatter_policies_joined() + "); upgrade eagle");

    std::vector<std::string> execFields(std::begin(kNeuralExecRefFields),
                                        std::end(kNeuralExecRefFields));
    std::sort(execFields.begin(), execFields.end());
    for (const auto& f : execFields) {
        auto it = sc.neural.execs.find(f);
        if (it != sc.neural.execs.end()) _validate_exec_ref(it->second, f, ctx);
    }

    // Forbidden kernel-machinery fields — ITERATED over the constant, same rule as
    // presence above. Each would be silently IGNORED on a descriptor (the mis-read
    // hazard), so a producer that copied one must fail loudly. The populated-check is a
    // lookup with NO default branch: adding a name to roles.h without teaching this
    // function how to see it is a loud runtime failure, never a silently unchecked field.
    std::vector<std::string> forbidden(std::begin(kNeuralForbiddenFields),
                                       std::end(kNeuralForbiddenFields));
    std::sort(forbidden.begin(), forbidden.end());
    for (const auto& f : forbidden) {
        bool populated;
        if      (f == "aether_abi")   populated = !sc.aether_abi.empty();
        else if (f == "derivative") populated = sc.derivative.present;
        else if (f == "buffers")    populated = !sc.buffers.empty();
        else if (f == "mutables")   populated = !sc.mutables.empty();
        else if (f == "mat_shapes") populated = !sc.mat_shapes.empty();
        else if (f == "host_entry") populated = !sc.host_entry.empty();
        else throw std::runtime_error(ctx + ": internal — kNeuralForbiddenFields names "
            "'" + f + "' but validate_sidecar has no populated-check for it");
        if (populated)
            throw std::runtime_error(ctx + ": a 'neural_block' descriptor must not "
                "carry '" + f + "' (forbidden on a descriptor: " +
                neural_forbidden_fields_joined() + "); it belongs on the kernel "
                "artifact the descriptor references");
    }
}

// Validate a parsed sidecar against the schema-v1 contract, before any binding:
//   * forward-strict schema version — a ``schema_version`` newer than this build
//     supports (plugin/roles.h ``kPluginSchemaVersion``) is rejected; absent (0) is
//     treated as v1 (backward-lenient, so pre-freeze sidecars stay loadable);
//   * canonical roles — every ``arg_spec`` role must be in the vocabulary
//     (``is_valid_role``), the same set the Python loader enforces.
// ``ctx`` names the artifact in the error (e.g. "plugin 'gravity'").
//
// THIS IS THE SHARED VALIDATOR — the first gate at every door that reads a
// sidecar, and at doors #3/#5 the ONLY family gate there is. The NORMATIVE
// entry-point register (all seven doors, each with its gates and its launch set)
// lives in plugin/roles.h next to the pattern sets it governs; that table is the
// review anchor and is the one to
// edit when a launcher is added. Restated here as the enumeration a reader of
// THIS file needs — which doors run the checks below:
//
//   #1 plugin_registry/registry.h ``from_manifest``  — runs it per sidecar.
//   #2 host_registry.h ``load``                      — runs it via #3 per entry.
//   #3 host_registry.h ``add_plugin``                — runs it; SOLE family gate.
//   #4 python .. loaded.py ``LoadedVector``/``LoadedPure`` — Python twin, then
//      ``_require_pattern``.
//   #5 python .. host_launch.py ``HostPluginLibrary`` — Python twin; SOLE
//      family gate (the Python mirror of #3).
//   #6 python .. registry.py ``load_manifest``       — Python twin, then the
//      ``loaders`` dict dispatch.
//   #7 plugin/plugin_host.cpp                        — runs it FIRST, before its
//      family dispatch and its ``check_aether_abi``, and before ``cuInit``.
//
// A seventh row exists because an earlier audit stopped at six: plugin_host had
// launched with zero gates since it was written. If you add an
// eighth door, add it in BOTH places.
inline void validate_sidecar(const Sidecar& sc, const std::string& ctx) {
    // Forward-strict schema gate. The ceiling is ``kPluginMaxSchemaVersion``
    // (v1 AND v2 both load, v3+ does not), NOT ``kPluginSchemaVersion`` — that
    // constant is the cross-repo version, not this loader's ceiling; see
    // roles.h for why the two are separate constants.
    if (sc.schema_version > kPluginMaxSchemaVersion)
        throw std::runtime_error(ctx + ": plugin schema v" +
            std::to_string(sc.schema_version) + " is newer than this host supports "
            "(v" + std::to_string(kPluginMaxSchemaVersion) + "); upgrade eagle");
    for (const auto& a : sc.arg_spec)
        if (!is_valid_role(a.role))
            throw std::runtime_error(ctx + ": unknown arg role '" + a.role +
                "' (not in the schema-v1 vocabulary)");
    // Forward-strict dtype (mirrors Python's enum check against
    // eagle.dtypes.SCALAR_TYPES in LoadedKernel.__init__): reject a
    // scalar_type this build does not recognize at all, rather than let it
    // silently fall through the registries' float32/softdouble literal
    // compares and be treated as an ordinary float64 kernel. Empty ("" — a
    // pre-dtype sidecar that predates this field) stays backward-lenient.
    if (!sc.scalar_type.empty() && sc.scalar_type != "float64" &&
        sc.scalar_type != "float32" && sc.scalar_type != "softdouble")
        throw std::runtime_error(ctx + ": unknown sidecar scalar_type '" +
            sc.scalar_type + "' (supported: float64, float32, softdouble)");
    // Sidecar-level ``pattern`` ("the discriminant rule"): value-strict, like
    // the manifest-level gate in both registries' ``from_manifest``/``load`` — an
    // unrecognized value is rejected naming the supported set, telling the caller
    // to upgrade eagle. This is the ONLY place that closes the ``add_plugin`` path
    // (host_registry.h): a caller who bypasses a manifest and hands a Sidecar
    // straight to ``add_plugin`` never goes through either registry's manifest-level
    // check, so without this a hand-built or future (e.g. neural-ABI) sidecar with
    // an unrecognized ``pattern`` would load and launch as a bare ordinary kernel
    // with zero diagnostics. Absence stays lenient (pre-freeze sidecars never
    // stamped one) — this loader never dispatches on ``pattern`` either way.
    // The set checked here is RECOGNIZED, not launch-certified: this is the
    // SHARED structural validator, so it admits every family it can validate. A
    // launching loader gates additionally on ``kLaunchCertifiedPatterns``.
    if (!sc.pattern.empty() && !is_recognized_pattern(sc.pattern))
        throw std::runtime_error(ctx + ": unknown sidecar pattern '" + sc.pattern +
            "' (supported: " + recognized_patterns_joined() + "); upgrade eagle");
    // The pattern-conditional ``neural_block`` clause — the ONE validator of
    // the descriptor wire shape, in both languages. It runs BEFORE the derivative and
    // buffer checks below because it FORBIDS both of those blocks on a descriptor: a
    // confused producer that copied kernel machinery onto a descriptor must be told
    // that, not nitpicked on the shape of a block it should not carry at all.
    if (sc.pattern == kNeuralBlockPattern) _validate_neural_block(sc, ctx);
    // Optional derivative block, Phase A shape check: ``kind`` must be vjp|jvp,
    // and ``residuals`` must be empty (recompute-only). A populated ``residuals`` list
    // is a future schema revision — reject it, naming the schema-bump rule, rather than
    // silently accept metadata this host cannot honor.
    if (sc.derivative.present) {
        const auto& d = sc.derivative;
        if (d.kind != "vjp" && d.kind != "jvp")
            throw std::runtime_error(ctx + ": derivative.kind '" + d.kind +
                "' is not one of vjp|jvp");
        if (d.residuals_populated)
            throw std::runtime_error(ctx + ": derivative.residuals is populated, but "
                "Phase A derivative artifacts are recompute-only (residuals must be "
                "[]); a populated residuals list requires a schema_version bump");
    }
    // Declared-buffer `kind`: value-strict against the canonical
    // vocabulary (plugin/roles.h kBufferKinds). Both registries' `declared_tables()`
    // filter on `kind == "lookup"` as DISPATCH (which buffers to consolidate), not
    // validation — so without this check an unrecognized kind would silently drop
    // out of that filter and never be consolidated/bound, rather than fail loudly at
    // load. Mirrors the Python `eagle.sidecar.validate_sidecar` buffer-kind check.
    for (const auto& b : sc.buffers)
        if (!is_valid_buffer_kind(b.kind))
            throw std::runtime_error(ctx + ": buffer '" + b.name + "' has unknown "
                "kind '" + b.kind + "' (supported: " + buffer_kinds_joined() +
                "); upgrade eagle");
}

}  // namespace plugin
}  // namespace eagle
