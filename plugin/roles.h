// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// Canonical plugin arg-spec ROLE vocabulary + schema version (the C++ half).
//
// The sidecar ``arg_spec`` is an ordered list of ``[role, name]`` pairs; ``role``
// is one of the fixed strings below. This header is the C++ half of the schema-v1
// plugin protocol's role vocabulary; the Python half is
// ``eagle/python/eagle/roles.py`` and a cross-check test
// (``python/tests/test_roles_vocab.py``) asserts the two lists are identical
// (order-independent). Every loader validates each arg_spec role against this set
// at LOAD time (forward-strict: an unknown role is rejected before any launch),
// so the C++ ``PluginRegistry`` and the Python ``Loaded*`` accept exactly the same
// vocabulary.
//
// NOTE on ``mat_in``: it is a valid schema-v1 role (a pure kernel's matrix input)
// recognized by every loader. A matrix binds through the SAME 32-byte GRef mirror as a
// vector (a matrix GRef is a width-``R*C`` vector GRef; the R x C shape is in-kernel
// flat indexing only), so both the device registry and the CPU host ``PluginRegistry``
// (host_registry.h ``run``) pack it via ``make_gref``. The host runner additionally
// validates the bound matrix's ``(R, C)`` against the sidecar's ``mat_shapes`` /
// Mutable ``shape``.
//
// No third-party dependency (matches sidecar.h / manifest.h): plain data,
// ``std::string`` throughout (the host tree's style).
#pragma once

#include <string>

namespace eagle {
namespace plugin {

// The 12 canonical arg-spec roles. KEEP IN SYNC with
// ``eagle/python/eagle/roles.py::ROLES`` (order-independent; the cross-check test
// enforces set equality). Per-kind subsets (vector=6, pure=8) union to exactly this
// set -- see the ``*_arg_spec`` producers in the code generator. ``wide_in``/``wide_out``
// (the code generator's wide-buffer vocabulary role + its VJP
// scatter gradient) are ADDITIVE (schema v1, not a version bump): both ride the
// SAME plain scalar-handle ABI as ``per_sample``/``lookup``/``mutable`` (see
// ``eagle/python/eagle/roles.py::classify_arg``'s ``WIDE_IN``/``WIDE_OUT`` tags).
//
// ``accum_out`` is the TWELFTH role and it closes a live DISAGREEMENT, not a gap
// The code generator's real
// producer has been emitting an ``accum_out`` role and renaming it to ``wide_out``
// on the wire (the code generator, "eagle.roles.classify_arg has
// never heard of accum_out") because this register did not carry it — so the
// register and its own producer described different vocabularies. Its ABI is the
// same plain scalar handle ``wide_in``/``wide_out`` ride; what it NAMES is the
// cross-sample accumulate plane (``AccumWrite``), which is exactly the surface
// This classifies and places, so it must be nameable.
//
// The same finding's other half: ``wide_in``/``wide_out`` were in this register but
// NEITHER C++ registry could launch them (seven role arms, then throw). All three
// are launchable arms now, in both registries.
inline constexpr const char* const kPluginArgRoles[] = {
    "out", "vec_in", "mat_in", "per_sample", "lookup", "mutable",
    "terminated", "uniform", "nsamples", "wide_in", "wide_out", "accum_out",
};

// Is ``role`` one of the canonical schema-v1 arg-spec roles?
inline bool is_valid_role(const std::string& role) {
    for (const char* r : kPluginArgRoles)
        if (role == r) return true;
    return false;
}

// The current + maximum plugin-schema version this build understands. A sidecar or
// manifest tagged with a HIGHER ``schema_version`` is rejected (forward-strict); an
// untagged artifact (``schema_version`` absent, parsed as 0) is treated as v1
// (backward-lenient — pre-freeze artifacts stay loadable). This is orthogonal to
// the ``"aether-abi/1"`` tag, which is the ABI-stable binary-layout contract.
inline constexpr int kPluginSchemaVersion = 1;

// The HIGHEST ``schema_version`` this build ACCEPTS -- the
// TRANSITION BRIDGE: schema v1 AND v2 both load on the aether branches, v3+ does
// not. Deliberately a SEPARATE constant from ``kPluginSchemaVersion`` rather than a
// bump of it, mirroring raptor's own split (``raptor/schema/manifest.py``'s
// ``SCHEMA_VERSION`` vs ``MAX_SCHEMA_VERSION``): ``kPluginSchemaVersion`` is the
// A cross-repo constant (raptor / eagle-Python / this header / the code generator's sanctioned
// literal copy) and moving it would break agreement across the other repos
// does not own, while the loader ceiling is a per-build property that must move
// with the loader. A schema-v2 document additionally carries the FLAT execution
// keys (``exec_targets`` / ``exec_access`` / ``exec_op``) and is refused
// without them; a v1 document must carry NONE of them.
inline constexpr int kPluginMaxSchemaVersion = 2;

// The plugin FAMILIES a manifest/sidecar ``pattern`` may name (schema v1): the
// variants a generated ``Bundle`` can emit. Unlike an ordinary optional key, ``pattern``
// is a schema DISCRIMINANT — it SELECTS which variant a bundle is, so a loader that
// does not recognize a value must refuse it rather than fall through and bind an
// unfamiliar variant as though it were an ordinary kernel (see
// "the discriminant rule").
//
// A prior design decision splits that one set in two -- NESTED, both
// value-strict, and this header is the single source of both:
//
//   * RECOGNIZED (``kRecognizedPatterns``) — the families the SHARED sidecar
//     validator (``sidecar.h`` ``validate_sidecar``) can structurally validate. A
//     value outside it is rejected naming the supported set and telling the caller
//     to upgrade eagle.
//   * LAUNCH-CERTIFIED (``kLaunchCertifiedPatterns``) — the families a LAUNCHING
//     entry point will actually bind and run.
//
// LAUNCH-CERTIFIED is a subset of RECOGNIZED. The two DIVERGED when
// ``neural_block`` was added: the neural
// descriptor family — is RECOGNIZED (the shared ``validate_sidecar`` structurally
// validates it) but NOT launch-certified (no loader in this build binds or runs a
// descriptor; its referenced kernels are ordinary ``pure`` artifacts that load
// through the normal doors). One rule governed that widening:
// admitting a family to RECOGNIZED (a one-line edit here) and adding the
// launch-certification checks at every launching entry point were ONE change.
// Widening RECOGNIZED alone would have re-opened that hole — an ``arg_spec``-driven
// loader binding an unfamiliar variant as a bare kernel through ``add_plugin``,
// whose only gate IS ``validate_sidecar``.
//
// Because the sets now differ, every launching door speaks TWO messages,
// and they are single-sourced in ``check_launch_certified_pattern`` below:
//
//   * pattern NOT in RECOGNIZED  -> "is not a supported plugin family (expected one
//     of ...); upgrade eagle" — a newer eagle is exactly what would load it.
//   * pattern in RECOGNIZED but not certified -> "is recognized but not launchable by
//     this loader", with NO upgrade suffix: this build knows the family perfectly
//     well, and upgrading would not make this loader run it.
//
// ---------------------------------------------------------------------------
// THE ENTRY-POINT REGISTER (NORMATIVE)
// ---------------------------------------------------------------------------
// "every launching entry point" is not a figure of speech: it is the SEVEN doors
// below, and this list is the register of record. Two consecutive door audits
// enumerated it short -- one review missed the host ``load``'s schema gate, another
// missed ``plugin_host.cpp`` entirely (it had launched ungated since it was
// written). The corrective is structural rather than a promise to be more
// careful: a new launcher that does not add its row here is a DIFF-VISIBLE
// omission at review, so the next miss cannot be silent. Any change to the
// pattern sets above must walk all seven rows.
//
//   #1 plugin_registry/registry.h ``PluginRegistry::from_manifest`` (device).
//      Gates: ``check_aether_abi`` (manifest, then every sidecar), forward-strict
//      ``schema_version``, launch-certified ``pattern``, duplicate-id and
//      manifest-``format`` structural checks, shared ``validate_sidecar`` per
//      sidecar, float32 capability refusal.   Launch set: LAUNCH-CERTIFIED.
//   #2 host_registry.h ``PluginRegistry::load`` (CPU host, manifest-level).
//      Gates: the #1 set, mirrored, then #3 per entry.   Launch set:
//      LAUNCH-CERTIFIED.
//   #3 host_registry.h ``PluginRegistry::add_plugin`` (CPU host, manifest-
//      BYPASSING — a sidecar + ``.so`` handed over directly). Gates: shared
//      ``validate_sidecar``, then ``check_launch_certified_pattern`` (added in
//      step 1b — before the widening this door had NO family gate beyond the
//      validator, which is precisely why the atomicity clause exists), then
//      ``check_aether_abi`` and the float64-only policy.   Launch set:
//      LAUNCH-CERTIFIED.
//   #4 eagle/python/eagle/loaded.py ``LoadedVector`` / ``LoadedPure``. Gates:
//      shared ``validate_sidecar`` via ``LoadedKernel``, then ``_require_pattern``
//      — a hardcoded literal per class, retained DISPATCH (each class binds
//      exactly one family, so the literal is what it means; a ``neural_block``
//      descriptor is refused there as "not a vector/pure plugin").   Launch
//      set: the one family each class names (``vector`` / ``pure``).
//   #5 eagle/python/eagle/host_launch.py ``HostPluginLibrary`` — the Python
//      mirror of #3 (a ``.so`` + sidecar, no manifest). Gates: as #3, through
//      ``eagle.roles.check_launch_certified_pattern``.   Launch set:
//      LAUNCH-CERTIFIED.
//   #6 eagle/python/eagle/registry.py ``load_manifest`` — dispatches on
//      ``pattern`` through a ``loaders`` dict whose keys ARE the gate (an
//      unlisted family raises). Source-text-pinned equal to the Python
//      launch-certified set by python/tests/test_roles_vocab.py.   Launch set:
//      LAUNCH-CERTIFIED.
//   #7 plugin/plugin_host.cpp — the standalone Driver-API host; the only
//      launching executable in the tree with no registry behind it. Gates:
//      shared ``validate_sidecar``, family dispatch, then
//      ``check_aether_abi``, all before ``cuInit`` so a refusal never touches the
//      GPU.   Launch set: ``vector`` only (it binds vector-shaped role sets).
//
// KEEP IN SYNC with ``eagle/python/eagle/roles.py``'s ``RECOGNIZED_PATTERNS`` /
// ``LAUNCH_CERTIFIED_PATTERNS`` (order-independent), and with the Python ``loaders``
// dict in ``eagle/python/eagle/registry.py::load_manifest`` (door #6, so its keys
// are the launch-certified set).
inline constexpr const char* const kRecognizedPatterns[] = {
    "vector", "pure", "neural_block",
};

// The families a launching loader will bind and run (subset of the recognized set).
inline constexpr const char* const kLaunchCertifiedPatterns[] = { "vector", "pure" };

// Is ``pattern`` a family the shared sidecar validator can structurally validate?
inline bool is_recognized_pattern(const std::string& pattern) {
    for (const char* p : kRecognizedPatterns)
        if (pattern == p) return true;
    return false;
}

// Is ``pattern`` a family this build's launching loaders will bind and run?
inline bool is_launch_certified_pattern(const std::string& pattern) {
    for (const char* p : kLaunchCertifiedPatterns)
        if (pattern == p) return true;
    return false;
}

// A human-readable "vector, pure" rendering of ``kRecognizedPatterns``, for an error
// message naming the supported set.
inline std::string recognized_patterns_joined() {
    std::string s;
    for (std::size_t i = 0;
         i < sizeof(kRecognizedPatterns) / sizeof(kRecognizedPatterns[0]); ++i) {
        if (i) s += ", ";
        s += kRecognizedPatterns[i];
    }
    return s;
}

// The same rendering for ``kLaunchCertifiedPatterns`` (the launching loaders' gate).
inline std::string launch_certified_patterns_joined() {
    std::string s;
    for (std::size_t i = 0;
         i < sizeof(kLaunchCertifiedPatterns) / sizeof(kLaunchCertifiedPatterns[0]);
         ++i) {
        if (i) s += ", ";
        s += kLaunchCertifiedPatterns[i];
    }
    return s;
}

// Refuse ``pattern`` at a LAUNCHING door, with the two-branch message this
// function locks. THE ONE SPELLING: every launching door that gates on the certified set
// calls this rather than re-writing the compare and the wording (the ``check_aether_abi``
// architecture — a fourth inline copy is a fourth spelling).
//
// ``subject`` names the artifact and is followed directly by " pattern '<value>'",
// so callers pass e.g. "plugin manifest", "host manifest", or "host plugin 'k':".
// An EMPTY pattern is lenient (a pre-freeze artifact never stamped one, and no
// caller of this helper dispatches on the value).
//
// Door #7 (plugin_host.cpp) deliberately does NOT call this: its launch set is
// ``vector`` alone rather than the certified set, so it spells its own dispatch —
// but it emits the SAME "recognized but not launchable by this loader" substring,
// which is what the conformance corpus and the named host tests check.
inline void check_launch_certified_pattern(const std::string& pattern,
                                           const std::string& subject) {
    if (pattern.empty() || is_launch_certified_pattern(pattern)) return;
    if (!is_recognized_pattern(pattern))
        throw std::runtime_error(subject + " pattern '" + pattern +
            "' is not a supported plugin family (expected one of " +
            launch_certified_patterns_joined() + "); upgrade eagle");
    // RECOGNIZED but not certified: this build understands the family, it simply
    // does not launch it. No "upgrade eagle" — a newer eagle is not the remedy,
    // and saying so would send the caller down a road that does not exist.
    throw std::runtime_error(subject + " pattern '" + pattern +
        "' is recognized but not launchable by this loader (expected one of " +
        launch_certified_patterns_joined() + ")");
}

// ---------------------------------------------------------------------------
// The ``neural_block`` descriptor vocabulary (schema v1)
// ---------------------------------------------------------------------------
// A ``neural_block`` sidecar is a DESCRIPTOR, not a kernel: it carries no arg_spec,
// no artifact of its own, and nothing launches it. It names the kernels that
// implement it through ``*_exec`` references; those are ordinary ``pure`` artifacts
// that load through the normal doors. Everything below is single-sourced here and
// mirrored in ``eagle/python/eagle/roles.py``, with set-equality parity tests in
// python/tests/test_roles_vocab.py.

// The ``kind`` discriminant of an exec reference (``{"kind": ..., "kernel": ...}``).
// One value in v1: the referenced artifact is a plain plugin kernel. A future
// aggregate/plan-bundle kind RESHAPES the object (a digest payload alongside
// ``kind``), which is a meaning change and therefore a ``schema_version`` bump
// (an additive change). KEEP IN SYNC with ``EXEC_REF_KINDS``.
inline constexpr const char* const kExecRefKinds[] = { "kernel" };

// The declared TERMINAL-WRITE CONTRACT of a block's scatter.
//
// ``scatter_policy`` declares what the committed result MEANS — never the mechanism
// EAGLE uses to commit it. The v1 sole value ``unique_write`` says: every
// (target, slot) is written by exactly one source per step, so the commit is a plain
// store, the result is deterministic with zero atomics, and the bit-exact gate
// applies. Mechanism (atomic add, block-reduce-then-atomic, segmented reduce)
// is EAGLE-internal, evidence-driven and WIRE-INVISIBLE FOREVER: changing executors
// is never a schema event, and no mechanism spelling may enter this vocabulary.
//
// "accumulate" is the second value -- contract-named,
// every (target, slot) may be written by more than
// one source per step, and the committed result is the carried base plus an
// order-unspecified sum of contributions (band-gated, never bit-exact — the
// bit-exact gate is unique_write-only). The vocabulary value and its
// demonstrated dual-mode executor are the same obligation, landed across
// this work: the value here landed first (GPU-free); the executor is
// package 2/3's (ForwardPipeline::commit's accumulate variant).
// KEEP IN SYNC with ``SCATTER_POLICIES``.
inline constexpr const char* const kScatterPolicies[] = { "unique_write", "accumulate" };

// The fields a ``neural_block`` descriptor MUST carry, beyond the general required
// set (``schema_version``, ``kernel``, ``pattern``, ``scalar_type``, and an
// ``arg_spec`` that is present and EMPTY). The validators ITERATE this list rather
// than hand-writing a presence ``if`` per field, so the list is the only place the
// requiredness can drift. KEEP IN SYNC with ``NEURAL_REQUIRED_FIELDS``.
// A schema-key rename (both-ends): ``fanin``/``fanout`` ->
// ``in_degree``/``out_degree``. SCHEMA_VERSION stays 1 (pre-release hard break).
inline constexpr const char* const kNeuralRequiredFields[] = {
    "in_degree", "out_degree", "input_width", "output_width", "state_width",
    "param_width", "scatter_policy", "forward_exec",
};

// The keys whose values are exec references. ``forward_exec`` is required (it is in
// ``kNeuralRequiredFields`` above); the two derivative refs are optional-additive,
// strictly shaped when present. The C++ parser also uses this list to BOUND each
// nested object before scanning top-level keys — see parse_sidecar.
// KEEP IN SYNC with ``NEURAL_EXEC_REF_FIELDS``.
inline constexpr const char* const kNeuralExecRefFields[] = {
    "forward_exec", "vjp_exec", "jvp_exec",
};

// Kernel-machinery fields a descriptor must NOT carry. Each would otherwise be
// silently IGNORED on a descriptor — a mis-read hazard — so a confused
// producer must fail loudly instead. ``aether_abi`` because the referenced artifacts
// carry their own tags (a descriptor copy is drift); ``derivative``
// because it lives on the VJP artifact, not on the block that references it; and
// ``buffers`` / ``mutables`` / ``mat_shapes`` / ``host_entry`` because a descriptor
// binds nothing. The rule is "absent or empty": eagle's own representation uses an
// empty string / empty container for absence throughout, so a present-but-empty
// value is indistinguishable from absence by construction, and any real copy of
// one of these fields is non-empty. KEEP IN SYNC with ``NEURAL_FORBIDDEN_FIELDS``.
inline constexpr const char* const kNeuralForbiddenFields[] = {
    "aether_abi", "derivative", "buffers", "mutables", "mat_shapes", "host_entry",
};

// Is ``kind`` a recognized exec-reference kind?
inline bool is_valid_exec_ref_kind(const std::string& kind) {
    for (const char* k : kExecRefKinds)
        if (kind == k) return true;
    return false;
}

// Is ``policy`` a recognized terminal-write contract?
inline bool is_valid_scatter_policy(const std::string& policy) {
    for (const char* p : kScatterPolicies)
        if (policy == p) return true;
    return false;
}

// A human-readable "kernel" rendering of ``kExecRefKinds``.
inline std::string exec_ref_kinds_joined() {
    std::string s;
    for (std::size_t i = 0; i < sizeof(kExecRefKinds) / sizeof(kExecRefKinds[0]); ++i) {
        if (i) s += ", ";
        s += kExecRefKinds[i];
    }
    return s;
}

// A human-readable rendering of ``kNeuralRequiredFields`` (declaration order).
inline std::string neural_required_fields_joined() {
    std::string s;
    for (std::size_t i = 0;
         i < sizeof(kNeuralRequiredFields) / sizeof(kNeuralRequiredFields[0]); ++i) {
        if (i) s += ", ";
        s += kNeuralRequiredFields[i];
    }
    return s;
}

// A human-readable rendering of ``kNeuralForbiddenFields`` (declaration order).
inline std::string neural_forbidden_fields_joined() {
    std::string s;
    for (std::size_t i = 0;
         i < sizeof(kNeuralForbiddenFields) / sizeof(kNeuralForbiddenFields[0]); ++i) {
        if (i) s += ", ";
        s += kNeuralForbiddenFields[i];
    }
    return s;
}

// A human-readable "unique_write" rendering of ``kScatterPolicies``.
inline std::string scatter_policies_joined() {
    std::string s;
    for (std::size_t i = 0;
         i < sizeof(kScatterPolicies) / sizeof(kScatterPolicies[0]); ++i) {
        if (i) s += ", ";
        s += kScatterPolicies[i];
    }
    return s;
}

// The declared-buffer ``kind`` vocabulary (schema v1). A read-only lookup table /
// shared constant is the only kind a loader can bind, and every loader's (both
// registries' ``declared_tables``) ``kind == "lookup"`` comprehension is DISPATCH,
// not validation — an unrecognized kind would silently vanish from the binding set
// rather than fail. ``validate_sidecar`` (sidecar.h) closes that filter by rejecting
// the value up front. KEEP IN SYNC with
// ``eagle/python/eagle/roles.py``'s ``BUFFER_KINDS`` (order-independent; the
// cross-check test enforces set equality).
inline constexpr const char* const kBufferKinds[] = { "lookup" };

// Is ``kind`` one of the canonical schema-v1 buffer kinds?
inline bool is_valid_buffer_kind(const std::string& kind) {
    for (const char* k : kBufferKinds)
        if (kind == k) return true;
    return false;
}

// A human-readable "lookup" rendering of ``kBufferKinds``, for an error message
// naming the supported set.
inline std::string buffer_kinds_joined() {
    std::string s;
    for (std::size_t i = 0; i < sizeof(kBufferKinds) / sizeof(kBufferKinds[0]); ++i) {
        if (i) s += ", ";
        s += kBufferKinds[i];
    }
    return s;
}

// The manifest-entry ``format`` vocabulary (schema v1) — the artifact container a
// plugin manifest entry names. Matches the producer of record's vocabulary
// (the code generator's deploy bundling). A manifest entry's
// ``format`` is a schema DISCRIMINANT: it selects
// how the registry must load the artifact bytes it names (``cuModuleLoadData`` for
// PTX text vs. a binary cubin/fatbin), so an unrecognized value must be refused
// rather than handed to the loader as though it were one of the known kinds.
// KEEP IN SYNC with ``eagle/python/eagle/roles.py``'s ``MANIFEST_FORMATS``
// (order-independent; the cross-check test enforces set equality).
inline constexpr const char* const kManifestFormats[] = { "ptx", "cubin", "fatbin" };

// Is ``format`` one of the canonical schema-v1 manifest-entry artifact formats?
inline bool is_valid_manifest_format(const std::string& format) {
    for (const char* f : kManifestFormats)
        if (format == f) return true;
    return false;
}

// A human-readable "ptx, cubin, fatbin" rendering of ``kManifestFormats``, for an
// error message naming the supported set.
inline std::string manifest_formats_joined() {
    std::string s;
    for (std::size_t i = 0;
         i < sizeof(kManifestFormats) / sizeof(kManifestFormats[0]); ++i) {
        if (i) s += ", ";
        s += kManifestFormats[i];
    }
    return s;
}

}  // namespace plugin
}  // namespace eagle
