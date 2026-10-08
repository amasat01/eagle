// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "plugin/host_registry.h"
#include "plugin/plugin_registry/manifest.h"
#include "plugin/sidecar.h"

#include "TestBase.h"

#include <array>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Schema hardening for C++ sidecar/manifest loading:
//   1. ignore-unknown: an unrecognized top-level sidecar/manifest key, and an
//      unrecognized `launch` subkey, load cleanly and are ignored;
//   2. compat matrix: v1 loads, v1 + an unknown optional key loads, a newer
//      schema_version is a clean reject naming the upgrade rule;
//   3. the frozen forward-strict field list;
//   4. the scalar_type asymmetry fix — C++ now rejects an unrecognized
//      scalar_type, mirroring Python's SCALAR_TYPES enum check.
// Companion: eagle/python/tests/test_schema_hardening.py (same six-item
// split, same FROZEN_STRICT_FIELDS membership).
//
// `pattern` is a schema DISCRIMINANT, so both C++ registries reject an
// unrecognized manifest `pattern` at load (registry level, not the parser —
// see plugin_registry/registry.h `from_manifest` and host_registry.h
// `load`), and the shared `validate_sidecar` also value-checks the
// sidecar-level `pattern` (closing the `add_plugin` path, which bypasses
// manifests entirely). Both stay absence-lenient in C++ (pre-freeze
// manifests/sidecars).
using namespace eagle::plugin;

namespace eagle_tests {
namespace SchemaHardeningTest {

// A minimal well-formed vector-kernel sidecar; `extra` is spliced in as
// additional top-level JSON content just before the closing brace (mirrors
// test_Sidecar.cpp's sidecarJson() convention).
static std::string vectorSidecarJson(const std::string& extra)
{
    return std::string(R"({
  "format": "ptx",
  "schema_version": 1,
  "pattern": "vector",
  "aether_abi": "aether-abi/1",
  "scalar_type": "float64",
  "kernel": "raptor_kernel",
  "vector_inputs": ["position"],
  "params": ["mu"],
  "per_sample": [],
  "accumulate": true,
  "sink": "outVec-scratch",
  "vec_widths": {"out": 3, "position": 3},
  "arg_spec": [
    ["out", "out"], ["vec_in", "position"],
    ["terminated", "terminated"], ["uniform", "mu"]
  ],
  "buffers": [])")
        + extra + "\n}\n";
}

// A minimal well-formed `neural_block` DESCRIPTOR sidecar — the C++ twin of the
// conformance row01 golden, and identical to it field for field. `extra` is spliced
// in as additional top-level JSON just before the closing brace.
//
// Key order matters: `forward_exec` must come before the top-level `kernel`,
// so the nested object's own "kernel" key and value precede the descriptor's
// identity. A parser that isn't nesting-safe would misread `kernel` as
// "mlp_block_fwd". Every test below asserting `sc.kernel == "mlp_block"`
// guards against that. Do not reorder these keys.
static std::string neuralSidecarJson(const std::string& extra)
{
    return std::string(R"({
  "schema_version": 1,
  "pattern": "neural_block",
  "scalar_type": "float64",
  "forward_exec": {"kind": "kernel", "kernel": "mlp_block_fwd"},
  "vjp_exec": {"kind": "kernel", "kernel": "mlp_block_vjp"},
  "kernel": "mlp_block",
  "in_degree": 4,
  "out_degree": 2,
  "input_width": 8,
  "output_width": 3,
  "state_width": 6,
  "param_width": 24,
  "scatter_policy": "unique_write",
  "arg_spec": [])")
        + extra + "\n}\n";
}

static std::string writeTmp(const std::string& name, const std::string& body)
{
    const std::string path = ::testing::TempDir() + name;
    std::ofstream f(path, std::ios::binary);
    f << body;
    f.close();
    return path;
}

// ----------------------------------------------------------------------- //
// Item 1 — ignore-unknown (sidecar side): an unrecognized top-level key AND
// an unrecognized `launch` subkey load cleanly through the same hand-written
// scanner the doc/JSON-Schema claim tolerates them (additionalProperties:
// true) — nothing executable proved that before this test.
// ----------------------------------------------------------------------- //
TEST(SchemaHardeningTest, ParseSidecarIgnoresUnknownTopLevelKeyAndLaunchSubkey)
{
    const std::string extra = R"(,
  "a_future_producer_key_this_loader_has_never_seen": {"whatever": true},
  "launch": {"block": 128, "a_hint_from_a_future_minor_revision": 7})";
    const Sidecar sc = parse_sidecar(writeTmp("ignore_unknown.json", vectorSidecarJson(extra)));

    EXPECT_EQ(sc.kernel, "raptor_kernel");
    EXPECT_EQ(sc.aether_abi, "aether-abi/1");
    EXPECT_EQ(sc.scalar_type, "float64");
    ASSERT_EQ(sc.arg_spec.size(), 4u);
    EXPECT_EQ(sc.arg_spec[0].role, "out");
    EXPECT_NO_THROW(validate_sidecar(sc, "plugin 'gravity'"));
}

// ----------------------------------------------------------------------- //
// Item 1 — ignore-unknown (manifest side): an unrecognized top-level manifest
// key AND an unrecognized per-entry subkey load cleanly.
// ----------------------------------------------------------------------- //
TEST(SchemaHardeningTest, ParseManifestIgnoresUnknownTopLevelKeyAndEntrySubkey)
{
    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "pattern": "vector",
  "a_future_manifest_key_this_loader_has_never_seen": {"nested": [1, 2, 3]},
  "plugins": [
    {
      "id": "gravity",
      "order": 0,
      "enabled": true,
      "artifact": "gravity.ptx",
      "sidecar": "gravity.json",
      "format": "ptx",
      "a_future_entry_subkey_this_loader_has_never_seen": ["x", "y"]
    }
  ]
})";
    const Manifest m = parse_manifest(writeTmp("ignore_unknown_manifest.json", manifestJson));

    EXPECT_EQ(m.aether_abi, "aether-abi/1");
    EXPECT_EQ(m.schema_version, 1);
    EXPECT_EQ(m.pattern, "vector");
    ASSERT_EQ(m.plugins.size(), 1u);
    EXPECT_EQ(m.plugins[0].id, "gravity");
    EXPECT_EQ(m.plugins[0].artifact, "gravity.ptx");
    EXPECT_TRUE(m.plugins[0].enabled);
}

// ----------------------------------------------------------------------- //
// Item 2 — compat matrix (sidecar-level). The manifest-level schema_version
// gate lives at BOTH C++ manifest doors — the CUDA PluginRegistry
// (registry.h's `from_manifest` schema_version gate, exercised by
// PluginRegistryManifestTest.RejectsNewerManifestSchemaVersion,
// test_PluginRegistryManifest.cu) and the host PluginRegistry
// (host_registry.h::load, exercised by the shared conformance corpus's
// row04b_manifest_legacy_version_newer via `cpp_load_manifest`). The host
// door previously lacked this check. The stale pointer at
// test_PluginRegistryDtype.cu was also wrong: that file only ever stamps
// schema_version 1.
// ----------------------------------------------------------------------- //
TEST(SchemaHardeningTest, V1SidecarLoadsCleanly)
{
    const Sidecar sc = parse_sidecar(writeTmp("v1.json", vectorSidecarJson("")));
    EXPECT_EQ(sc.schema_version, 1);
    EXPECT_NO_THROW(validate_sidecar(sc, "plugin 'gravity'"));
}

TEST(SchemaHardeningTest, V1SidecarPlusUnknownOptionalKeyLoadsCleanly)
{
    const std::string extra = R"(,
  "an_unforeseen_optional_hint": 42)";
    const Sidecar sc = parse_sidecar(writeTmp("v1_plus_unknown.json", vectorSidecarJson(extra)));
    EXPECT_NO_THROW(validate_sidecar(sc, "plugin 'gravity'"));
}

TEST(SchemaHardeningTest, NewerSchemaVersionIsRejectedNamingTheUpgradeRule)
{
    const std::string body =
        std::string(R"({
  "format": "ptx",
  "schema_version": )") + std::to_string(kPluginMaxSchemaVersion + 1) + R"(,
  "pattern": "vector",
  "aether_abi": "aether-abi/1",
  "scalar_type": "float64",
  "kernel": "raptor_kernel",
  "vector_inputs": [], "params": [], "per_sample": [],
  "accumulate": true, "sink": "outVec-scratch", "vec_widths": {},
  "arg_spec": [["out", "out"], ["terminated", "terminated"]],
  "buffers": []
})";
    const Sidecar sc = parse_sidecar(writeTmp("v_future.json", body));
    try {
        validate_sidecar(sc, "plugin 'gravity'");
        FAIL() << "expected validate_sidecar to reject a newer schema_version";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("upgrade eagle"), std::string::npos) << msg;
    }
}

// ----------------------------------------------------------------------- //
// Item 3 — the frozen forward-strict field list.
//
// KEEP IN SYNC with the Python mirror,
// eagle/python/tests/test_schema_hardening.py::FROZEN_STRICT_FIELDS.
//
// If you are touching kFrozenStrictFields because you added or removed a
// strict field: first make the version-bump call. Any change to the MEANING
// or REQUIREDNESS of an EXISTING field, or any NEW field a consumer must
// read to launch a kernel safely, bumps schema_version
// (plugin/roles.h::kPluginSchemaVersion / eagle.roles.SCHEMA_VERSION,
// single-sourced). A purely additive OPTIONAL hint (e.g. the reserved
// `launch` block) does not need one — see plugin_schema.rst
// "Compatibility policy".
//
// `pattern` here is the MANIFEST-level field. It is forward-strict
// (unrecognized value rejected) in every loader: Python's
// registry.py::load_manifest, and both C++ registries
// (plugin_registry/registry.h::from_manifest, host_registry.h::load). The
// C++ manifest PARSER (plugin/plugin_registry/manifest.h) still never
// validates it, exactly like it never validates aether_abi or schema_version —
// parsers stay validation-free by design; the check lives at registry level.
// See plugin_schema.rst's "Compatibility policy" for the remaining, narrower
// asymmetry this does NOT close: Python's manifest-level `pattern` is
// additionally absence-STRICT (dispatch needs it to pick a Loaded* class);
// the C++ registries stay absence-lenient (pre-freeze manifests never
// stamped one, and neither registry dispatches on it).
//
// Each entry also carries its CLASS: "gate" (aether_abi, schema_version — a
// compatibility/version certification, not a schema-variant selector) |
// "discriminant" (SELECTS which schema VARIANT an artifact is;
// single-sourced vocabulary in plugin/roles.h + eagle.roles, ONE check per
// field in the shared validation layer) | "structural" (a required-key /
// well-formedness check, not a variant selector). Buffer `kind` closes the
// silent `kind == "lookup"` dispatch filter; `derivative.kind` has been
// strict throughout; `format` is the manifest-entry artifact container.
// Naming membership explicitly here (rather than leaving it implicit) is
// what lets the frozen guard mechanically detect a strict-field addition.
// `exec_ref kind` and `scatter_policy` are the two reserved discriminant
// slots for the `neural_block` descriptor's own strict surface — the v1
// discriminant register is closed at seven. `neural_block required fields`
// is ONE entry standing for the whole roles.h `kNeuralRequiredFields`
// constant, which both validators ITERATE: naming the list rather than its
// eight members is what keeps this a register instead of a second spelling
// of the vocabulary.
struct FrozenField { const char* name; const char* cls; };
static const std::array<FrozenField, 15> kFrozenStrictFields = {{
    { "schema_version", "gate" },
    { "kernel", "structural" },
    { "pattern", "discriminant" },
    { "aether_abi", "gate" },
    { "arg_spec role", "structural" },
    { "scalar_type", "discriminant" },
    { "buffer kind", "discriminant" },
    { "derivative.kind", "discriminant" },
    { "format", "discriminant" },
    { "forward_exec", "structural" },
    { "vjp_exec", "optional-additive" },
    { "jvp_exec", "optional-additive" },
    { "exec_ref kind", "discriminant" },
    { "scatter_policy", "discriminant" },
    { "neural_block required fields", "structural" },
}};

TEST(SchemaHardeningTest, FrozenStrictFieldListIsExactlyFifteen)
{
    ASSERT_EQ(kFrozenStrictFields.size(), 15u)
        << "kFrozenStrictFields changed size. Before editing it: does this "
           "change the meaning/requiredness of an existing field, or add a "
           "new field a consumer must read to launch safely? If yes, bump "
           "schema_version (plugin/roles.h::kPluginSchemaVersion). A purely "
           "additive optional hint does not need one. See "
           "plugin_schema.rst 'Compatibility policy'.";
    const std::set<std::string> want = {
        "schema_version", "kernel", "pattern", "aether_abi", "arg_spec role",
        "scalar_type", "buffer kind", "derivative.kind", "format",
        "forward_exec", "vjp_exec", "jvp_exec",
        "exec_ref kind", "scatter_policy", "neural_block required fields",
    };
    std::set<std::string> got;
    for (const auto& f : kFrozenStrictFields) got.insert(f.name);
    EXPECT_EQ(got, want);
}

TEST(SchemaHardeningTest, FrozenFieldClassesAreAllRecognized)
{
    const std::set<std::string> validClasses = {
        "gate", "discriminant", "structural", "optional-additive",
    };
    for (const auto& f : kFrozenStrictFields)
        EXPECT_TRUE(validClasses.count(f.cls) > 0)
            << "'" << f.name << "' has unrecognized class '" << f.cls << "'";
}

TEST(SchemaHardeningTest, FrozenFieldClassesMatchTheirRegisters)
{
    // The discriminant register is closed for v1 at seven: `exec_ref kind`
    // and `scatter_policy` are the two most recently added slots.
    // Everything else here is a gate, a structural check, or one of the two
    // optional-additive derivative exec references.
    std::set<std::string> discriminants, gates, structural, additive;
    for (const auto& f : kFrozenStrictFields) {
        if (std::string(f.cls) == "discriminant") discriminants.insert(f.name);
        else if (std::string(f.cls) == "gate") gates.insert(f.name);
        else if (std::string(f.cls) == "structural") structural.insert(f.name);
        else if (std::string(f.cls) == "optional-additive") additive.insert(f.name);
    }
    EXPECT_EQ(discriminants, (std::set<std::string>{
        "pattern", "scalar_type", "buffer kind", "derivative.kind", "format",
        "exec_ref kind", "scatter_policy",
    }));
    EXPECT_EQ(gates, (std::set<std::string>{ "schema_version", "aether_abi" }));
    EXPECT_EQ(structural, (std::set<std::string>{
        "kernel", "arg_spec role", "forward_exec",
        "neural_block required fields",
    }));
    EXPECT_EQ(additive, (std::set<std::string>{ "vjp_exec", "jvp_exec" }));
}

TEST(SchemaHardeningTest, FrozenField_SchemaVersion_RejectsNewer)
{
    Sidecar sc;
    sc.kernel         = "gravity";
    // "Newer than this host supports" is kPluginMaxSchemaVersion + 1, NOT
    // kPluginSchemaVersion + 1 — the cross-repo version and the loader
    // ceiling are separate constants, and schema v2 LOADS. Using the wrong
    // one here would make this row assert that a version the loader accepts
    // is rejected.
    sc.schema_version = kPluginMaxSchemaVersion + 1;
    sc.arg_spec       = { ArgEntry{ "out", "out" } };
    try {
        validate_sidecar(sc, "plugin 'gravity'");
        FAIL() << "expected validate_sidecar to reject a newer schema_version";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("upgrade eagle"), std::string::npos) << e.what();
    }
}

TEST(SchemaHardeningTest, FrozenField_Kernel_IsARequiredSidecarKey)
{
    // No "kernel" key at all -> parse_sidecar throws naming it.
    const std::string body = R"({
  "arg_spec": [["out", "out"]]
})";
    try {
        parse_sidecar(writeTmp("frozen_kernel_missing.json", body));
        FAIL() << "expected parse_sidecar to throw for a missing 'kernel' key";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("kernel"), std::string::npos) << e.what();
    }
}

TEST(SchemaHardeningTest, FrozenField_AetherAbi_RejectsMismatch)
{
    Sidecar sc;
    sc.kernel      = "addvec";
    sc.aether_abi    = "aether-abi/999-stale";
    sc.arg_spec    = { ArgEntry{ "mutable", "out" }, ArgEntry{ "per_sample", "a" },
                       ArgEntry{ "per_sample", "b" }, ArgEntry{ "nsamples", "n" } };
    sc.mutables    = { MutableInfo{ "out", "float", 1 } };
    eagle::cpu::PluginRegistry reg;
    try {
        reg.add_plugin(sc, HOST_PLUGIN_ADDVEC_SO);
        FAIL() << "expected add_plugin to reject the AETHER ABI mismatch";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("aether_abi"), std::string::npos) << e.what();
    }
}

// `add_plugin` is the manifest-BYPASSING entry point (loads a sidecar +
// `.so` directly, like HostAddPluginRejectsUnknownSidecarPatternNeuralBlock
// above), so `aether_abi` must be presence-required here too, not just at
// the two manifest-level gates (HostLoadRejectsMissingAetherAbi below, and
// the device registry's `from_manifest`). Before this fix, an ABSENT
// aether_abi (the sidecar default: an empty string) was silently accepted —
// only a MISMATCHED non-empty value was rejected (see
// FrozenField_AetherAbi_RejectsMismatch above). The check fires before
// dlopen, so "unused.so" need not exist.
TEST(SchemaHardeningTest, HostAddPluginRejectsMissingAetherAbi)
{
    Sidecar sc;  // sc.aether_abi left default-constructed (empty)
    sc.kernel   = "gravity";
    sc.arg_spec = { ArgEntry{ "out", "out" } };
    EXPECT_TRUE(sc.aether_abi.empty());
    eagle::cpu::PluginRegistry reg;
    try {
        reg.add_plugin(sc, "unused.so");
        FAIL() << "expected add_plugin to reject the missing aether_abi";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("'gravity'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("aether_abi"), std::string::npos) << msg;
    }
}

TEST(SchemaHardeningTest, FrozenField_ArgSpecRole_RejectsUnknown)
{
    const std::string body = R"({
  "kernel": "gravity",
  "arg_spec": [["bogus_role", "x"]]
})";
    const Sidecar sc = parse_sidecar(writeTmp("frozen_bogus_role.json", body));
    try {
        validate_sidecar(sc, "plugin 'gravity'");
        FAIL() << "expected validate_sidecar to reject the unknown role";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("unknown arg role"), std::string::npos) << e.what();
    }
}

// The item-4 fix: a scalar_type outside {float64, float32, softdouble} is
// now rejected, instead of silently falling through the registries'
// literal float32/softdouble compares as though it were an ordinary
// float64 kernel.
TEST(SchemaHardeningTest, FrozenField_ScalarType_RejectsUnrecognizedValue)
{
    const std::string extra;
    std::string body = vectorSidecarJson(extra);
    const std::string needle = "\"float64\"";
    const auto pos = body.find(needle);
    ASSERT_NE(pos, std::string::npos);
    body.replace(pos, needle.size(), "\"int8_not_a_real_scalar_type\"");
    const Sidecar sc = parse_sidecar(writeTmp("frozen_bad_scalar_type.json", body));
    EXPECT_EQ(sc.scalar_type, "int8_not_a_real_scalar_type");
    try {
        validate_sidecar(sc, "plugin 'gravity'");
        FAIL() << "expected validate_sidecar to reject the unrecognized scalar_type";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("unknown sidecar scalar_type"), std::string::npos)
            << e.what();
    }
}

// A recognized-but-unsupported-here value (e.g. "float32"/"softdouble") is
// NOT what item 4 rejects — those stay valid schema-v1 values; a registry
// may separately refuse to bind them for its own capability reasons (D-B).
TEST(SchemaHardeningTest, ScalarTypeFloat32StillValidatesCleanly)
{
    const std::string extra;
    std::string body = vectorSidecarJson(extra);
    const std::string needle = "\"float64\"";
    const auto pos = body.find(needle);
    ASSERT_NE(pos, std::string::npos);
    body.replace(pos, needle.size(), "\"float32\"");
    const Sidecar sc = parse_sidecar(writeTmp("scalar_type_f32.json", body));
    EXPECT_NO_THROW(validate_sidecar(sc, "plugin 'gravity'"));
}

// parse_manifest validates NOTHING — a bogus `pattern` parses through
// untouched, exactly like a bogus/mismatched aether_abi or an out-of-range
// schema_version would (see FrozenField_AetherAbi_RejectsMismatch and
// NewerSchemaVersionIsRejectedNamingTheUpgradeRule above, both of which
// construct their Sidecar/Manifest and validate at the REGISTRY, never at
// parse). `pattern` IS rejected end to end — see
// PluginRegistryManifestTest.RejectsUnknownManifestPatternNeuralBlock
// (eagle/tests/test_PluginRegistryManifest.cu, device) and
// HostLoadRejectsUnknownManifestPatternNeuralBlock below (host) — just not by
// this function; parse_manifest stays validation-free by design, mirroring
// where every other frozen-list gate lives.
TEST(SchemaHardeningTest, FrozenField_Pattern_ParseManifestValidatesNothing)
{
    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "pattern": "not_a_real_pattern",
  "plugins": [
    {"id": "gravity", "order": 0, "enabled": true,
     "artifact": "gravity.ptx", "sidecar": "gravity.json", "format": "ptx"}
  ]
})";
    const Manifest m = parse_manifest(writeTmp("frozen_bad_pattern.json", manifestJson));
    EXPECT_EQ(m.pattern, "not_a_real_pattern");
}

// ----------------------------------------------------------------------- //
// Sidecar-level `pattern` DISCRIMINANT strictness (validate_sidecar,
// the shared layer both registries call). Mirrors the ScalarType tests
// above: parse_sidecar stays validation-free, validate_sidecar rejects.
// ----------------------------------------------------------------------- //
TEST(SchemaHardeningTest, FrozenField_Pattern_RejectsUnknownSidecarValue)
{
    std::string body = vectorSidecarJson("");
    const std::string needle = "\"vector\"";
    const auto pos = body.find(needle);
    ASSERT_NE(pos, std::string::npos);
    body.replace(pos, needle.size(), "\"not_a_real_pattern\"");
    const Sidecar sc = parse_sidecar(writeTmp("bad_sidecar_pattern.json", body));
    EXPECT_EQ(sc.pattern, "not_a_real_pattern");  // parsed, not rejected by parse
    try {
        validate_sidecar(sc, "plugin 'gravity'");
        FAIL() << "expected validate_sidecar to reject the unrecognized pattern "
                  "'not_a_real_pattern'";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("not_a_real_pattern"), std::string::npos) << msg;
        EXPECT_NE(msg.find("unknown sidecar pattern"), std::string::npos) << msg;
        EXPECT_NE(msg.find("upgrade eagle"), std::string::npos) << msg;
    }
}

// Absence stays lenient (pre-freeze sidecars never stamped one; neither C++
// registry dispatches a variant class off `pattern`).
TEST(SchemaHardeningTest, FrozenField_Pattern_SidecarAbsenceStaysLenient)
{
    Sidecar sc;
    sc.kernel   = "gravity";
    sc.arg_spec = { ArgEntry{ "out", "out" } };
    EXPECT_TRUE(sc.pattern.empty());
    EXPECT_NO_THROW(validate_sidecar(sc, "plugin 'gravity'"));
}

// The DECISIVE proof for item 2: the public `add_plugin` path bypasses
// manifests entirely (a caller can hand it a hand-built Sidecar straight),
// so a manifest-only fix would leave it wide open. `add_plugin` calls
// `validate_sidecar` before dlopen, so the reject fires before any `.so` is
// ever touched — `"unused.so"` need not exist.
TEST(SchemaHardeningTest, HostAddPluginRejectsUnrecognizedSidecarPattern)
{
    Sidecar sc;
    sc.kernel   = "gravity";
    sc.aether_abi = EAGLE_AETHER_ABI;
    sc.pattern  = "not_a_real_pattern";
    sc.arg_spec = { ArgEntry{ "out", "out" } };
    eagle::cpu::PluginRegistry reg;
    try {
        reg.add_plugin(sc, "unused.so");
        FAIL() << "expected add_plugin to reject the unrecognized sidecar pattern";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("not_a_real_pattern"), std::string::npos) << msg;
        EXPECT_NE(msg.find("unknown sidecar pattern"), std::string::npos) << msg;
        EXPECT_NE(msg.find("upgrade eagle"), std::string::npos) << msg;
    }
}

// The OTHER branch of the same door, and the half that did not exist before the
// atomic widening that added this branch: `neural_block` is RECOGNIZED —
// `validate_sidecar` above accepts the descriptor as structurally valid schema v1
// — and then `add_plugin`'s own launch certification refuses it, because nothing
// in this build binds or runs a descriptor. NO "upgrade eagle": a newer eagle is
// not the remedy, and saying so would send the caller down a road that does not
// exist. This test is the C++ half of what conformance row 1 pins at the shared
// corpus level; door #7's identical substring is pinned by the named subprocess
// tests in python/tests/test_cpp_host.py.
TEST(SchemaHardeningTest, HostAddPluginRejectsRecognizedButUnlaunchableNeuralBlock)
{
    const Sidecar sc = parse_sidecar(
        writeTmp("neural_descriptor.json", neuralSidecarJson("")));
    ASSERT_EQ(sc.pattern, "neural_block");
    EXPECT_NO_THROW(validate_sidecar(sc, "descriptor"));  // RECOGNIZED: valid
    eagle::cpu::PluginRegistry reg;
    try {
        reg.add_plugin(sc, "unused.so");
        FAIL() << "expected add_plugin to refuse to launch a neural_block descriptor";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("neural_block"), std::string::npos) << msg;
        EXPECT_NE(msg.find("recognized but not launchable"), std::string::npos) << msg;
        EXPECT_EQ(msg.find("upgrade eagle"), std::string::npos) << msg;
    }
}

// ----------------------------------------------------------------------- //
// host_registry.h::load(), the CPU twin of
// PluginRegistry::from_manifest (eagle/tests/test_PluginRegistryManifest.cu).
// `load()` had ZERO prior test coverage; these are its first tests. Every
// case below constructs a `Manifest` directly (its fields are public) so no
// file I/O is needed for the guards that fire before the per-entry loop.
// ----------------------------------------------------------------------- //
static ManifestEntry oneEntry(const std::string& id)
{
    ManifestEntry e;
    e.id       = id;
    e.order    = 0;
    e.enabled  = true;
    e.artifact = id + ".so";
    e.sidecar  = id + ".json";
    e.format   = "ptx";
    return e;
}

TEST(SchemaHardeningTest, HostLoadRejectsUnrecognizedManifestPattern)
{
    Manifest m;
    m.aether_abi = EAGLE_AETHER_ABI;
    m.pattern  = "not_a_real_pattern";
    m.plugins  = { oneEntry("gravity") };
    eagle::cpu::PluginRegistry reg;
    try {
        reg.load(m, "unused_dir");
        FAIL() << "expected load() to reject the unrecognized manifest pattern";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("not_a_real_pattern"), std::string::npos) << msg;
        EXPECT_NE(msg.find("not a supported plugin family"), std::string::npos) << msg;
        EXPECT_NE(msg.find("upgrade eagle"), std::string::npos) << msg;
    }
}

// The second branch at the manifest level: RECOGNIZED but not launch-certified.
TEST(SchemaHardeningTest, HostLoadRejectsRecognizedButUnlaunchableNeuralBlock)
{
    Manifest m;
    m.aether_abi = EAGLE_AETHER_ABI;
    m.pattern  = "neural_block";
    m.plugins  = { oneEntry("gravity") };
    eagle::cpu::PluginRegistry reg;
    try {
        reg.load(m, "unused_dir");
        FAIL() << "expected load() to refuse to launch a neural_block manifest";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("neural_block"), std::string::npos) << msg;
        EXPECT_NE(msg.find("recognized but not launchable"), std::string::npos) << msg;
        EXPECT_EQ(msg.find("upgrade eagle"), std::string::npos) << msg;
    }
}

// Absence stays lenient: the pattern gate does not fire, so load() reaches
// the NEXT failure (no sidecar file at "unused_dir/gravity.json"), not a
// pattern-related one.
TEST(SchemaHardeningTest, HostLoadAbsentManifestPatternStaysLenient)
{
    Manifest m;
    m.aether_abi = EAGLE_AETHER_ABI;
    m.plugins  = { oneEntry("gravity") };
    eagle::cpu::PluginRegistry reg;
    try {
        reg.load(m, "unused_dir");
        FAIL() << "expected load() to throw (no such sidecar file)";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_EQ(msg.find("supported plugin family"), std::string::npos) << msg;
        EXPECT_EQ(msg.find("pattern"), std::string::npos) << msg;
    }
}

// Duplicate-id rejection, the host mirror of
// PluginRegistryManifestTest.RejectsDuplicatePluginId — the same invariant,
// same error style, checked before any sidecar/artifact I/O.
TEST(SchemaHardeningTest, HostLoadRejectsDuplicatePluginId)
{
    Manifest m;
    m.aether_abi = EAGLE_AETHER_ABI;
    m.pattern  = "vector";
    ManifestEntry e2 = oneEntry("gravity");
    e2.enabled       = false;  // fires regardless of the second entry's enabled flag
    m.plugins        = { oneEntry("gravity"), e2 };
    eagle::cpu::PluginRegistry reg;
    try {
        reg.load(m, "unused_dir");
        FAIL() << "expected load() to reject the duplicate plugin id";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("duplicate"), std::string::npos) << msg;
        EXPECT_NE(msg.find("'gravity'"), std::string::npos) << msg;
    }
}

// Item 4: `aether_abi` is now presence-required in load(), like the device
// registry's from_manifest — an EMPTY aether_abi is rejected, not silently
// trusted.
TEST(SchemaHardeningTest, HostLoadRejectsMissingAetherAbi)
{
    Manifest m;  // aether_abi left default-constructed (empty)
    m.plugins = { oneEntry("gravity") };
    eagle::cpu::PluginRegistry reg;
    try {
        reg.load(m, "unused_dir");
        FAIL() << "expected load() to reject the missing aether_abi";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("aether_abi"), std::string::npos) << e.what();
    }
}


// ----------------------------------------------------------------------- //
// Behavioural checks for the six most recently frozen entries, plus the
// nesting-safety regression check for parsing exec_ref objects. The register
// above records MEMBERSHIP; these prove the strictness is real — a register
// entry with no behavioural check backing it is just as mute as a check
// with no register entry.
//
// Companion: python/tests/test_schema_hardening.py's identical block, and
// conformance rows 1 / 1b / 5 which drive BOTH languages off one fixture.
// ----------------------------------------------------------------------- //

// The golden with the line declaring `field` deleted (one key per line, by
// construction). The scanner is not a JSON parser, so a dangling comma is
// immaterial — what matters is that the KEY is gone.
static std::string neuralJsonWithout(const std::string& field)
{
    const std::string  needle = "\"" + field + "\":";
    std::istringstream in(neuralSidecarJson(""));
    std::string        out, line;
    while (std::getline(in, line))
        if (line.find(needle) == std::string::npos) out += line + "\n";
    return out;
}

// The golden with the line declaring `field` replaced by `replacement`.
static std::string neuralJsonReplacing(const std::string& field,
                                        const std::string& replacement)
{
    const std::string  needle = "\"" + field + "\":";
    std::istringstream in(neuralSidecarJson(""));
    std::string        out, line;
    while (std::getline(in, line))
        out += (line.find(needle) == std::string::npos ? line : replacement) + "\n";
    return out;
}

// The counter-gate for every reject below: each doctors exactly ONE field of
// this fixture, so the only thing that can make one fail is the field under
// test. It also carries the nesting-order assertion, because the golden's
// key order is adversarial by design.
TEST(SchemaHardeningTest, NeuralGoldenParsesAndValidates)
{
    const Sidecar sc =
        parse_sidecar(writeTmp("neural_golden.json", neuralSidecarJson("")));
    EXPECT_EQ(sc.kernel, "mlp_block");  // NOT "mlp_block_fwd"
    EXPECT_NO_THROW(validate_sidecar(sc, "descriptor"));
}

// A dedicated check for the nesting-safety fix, so a regression names the
// actual defect. `_string_value` finds keys by text, so the exec object's nested `"kernel"`
// key — and the literal value "kernel" its `kind` carries — both shadow the
// descriptor's identity unless parse_sidecar bounds and masks the nested
// objects FIRST. Both orderings are asserted: the benign one always parsed
// correctly, which is exactly why the bug survived until it was executed.
TEST(SchemaHardeningTest, ParseSidecarIsNestingSafeForExecRefs)
{
    const Sidecar adversarial =
        parse_sidecar(writeTmp("pc5_adversarial.json", neuralSidecarJson("")));
    EXPECT_EQ(adversarial.kernel, "mlp_block");
    EXPECT_EQ(adversarial.neural.execs.at("forward_exec").kernel, "mlp_block_fwd");
    EXPECT_EQ(adversarial.neural.execs.at("forward_exec").kind, "kernel");
    EXPECT_EQ(adversarial.neural.execs.at("vjp_exec").kernel, "mlp_block_vjp");

    const std::string benign = R"({
  "schema_version": 1,
  "pattern": "neural_block",
  "scalar_type": "float64",
  "kernel": "mlp_block",
  "forward_exec": {"kind": "kernel", "kernel": "mlp_block_fwd"},
  "in_degree": 4, "out_degree": 2,
  "input_width": 8, "output_width": 3, "state_width": 6, "param_width": 24,
  "scatter_policy": "unique_write",
  "arg_spec": []
}
)";
    const Sidecar sc = parse_sidecar(writeTmp("pc5_benign.json", benign));
    EXPECT_EQ(sc.kernel, "mlp_block");
    EXPECT_EQ(sc.neural.execs.at("forward_exec").kernel, "mlp_block_fwd");
}

TEST(SchemaHardeningTest, FrozenField_ForwardExec_IsRequired)
{
    const Sidecar sc = parse_sidecar(
        writeTmp("neural_no_fwd.json", neuralJsonWithout("forward_exec")));
    try {
        validate_sidecar(sc, "descriptor");
        FAIL() << "expected a missing forward_exec to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what())
                      .find("missing required neural_block field 'forward_exec'"),
                  std::string::npos)
            << e.what();
    }
}

// ONE frozen entry stands for the whole roles.h constant, so this test walks
// it: every member must be individually required, or the register entry
// overstates what is checked.
TEST(SchemaHardeningTest, FrozenField_NeuralRequiredFields_AreIterated)
{
    for (const char* f : kNeuralRequiredFields) {
        const std::string field = f;
        const Sidecar sc = parse_sidecar(writeTmp(
            "neural_without_" + field + ".json", neuralJsonWithout(field)));
        try {
            validate_sidecar(sc, "descriptor");
            FAIL() << "expected a missing '" << field << "' to be rejected";
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what())
                          .find("missing required neural_block field '" + field + "'"),
                      std::string::npos)
                << e.what();
        }
    }
}

TEST(SchemaHardeningTest, FrozenField_ExecRefKind_RejectsUnknown)
{
    const Sidecar sc = parse_sidecar(writeTmp(
        "neural_bad_exec_kind.json",
        neuralJsonReplacing(
            "forward_exec",
            R"(  "forward_exec": {"kind": "plan_bundle", "kernel": "k"},)")));
    try {
        validate_sidecar(sc, "descriptor");
        FAIL() << "expected an unrecognized forward_exec.kind to be rejected";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("forward_exec.kind"), std::string::npos) << msg;
        EXPECT_NE(msg.find("plan_bundle"), std::string::npos) << msg;
    }
}

// The dead spelling `unique_target` is the value under test here and
// in conformance row 5: a "helpful" loader that accepted both spellings would
// fail both.
TEST(SchemaHardeningTest, FrozenField_ScatterPolicy_RejectsUnknown)
{
    const Sidecar sc = parse_sidecar(writeTmp(
        "neural_bad_scatter.json",
        neuralJsonReplacing("scatter_policy",
                            R"(  "scatter_policy": "unique_target",)")));
    try {
        validate_sidecar(sc, "descriptor");
        FAIL() << "expected an unrecognized scatter_policy to be rejected";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("scatter_policy"), std::string::npos) << msg;
        EXPECT_NE(msg.find("unique_target"), std::string::npos) << msg;
    }
}

// optional-ADDITIVE: absent is fine, present is validated as strictly as
// `forward_exec`.
TEST(SchemaHardeningTest, FrozenField_JvpExec_IsOptionalButStrict)
{
    const Sidecar absent = parse_sidecar(
        writeTmp("neural_no_jvp.json", neuralSidecarJson("")));
    EXPECT_EQ(absent.neural.execs.count("jvp_exec"), 0u);
    EXPECT_NO_THROW(validate_sidecar(absent, "descriptor"));

    const Sidecar present = parse_sidecar(writeTmp(
        "neural_jvp.json",
        neuralSidecarJson(R"(,
  "jvp_exec": {"kind": "kernel", "kernel": "mlp_block_jvp"})")));
    EXPECT_EQ(present.neural.execs.at("jvp_exec").kernel, "mlp_block_jvp");
    EXPECT_NO_THROW(validate_sidecar(present, "descriptor"));

    const Sidecar bad = parse_sidecar(writeTmp(
        "neural_bad_jvp.json",
        neuralSidecarJson(R"(,
  "jvp_exec": {"kind": "not_a_kind", "kernel": "d"})")));
    try {
        validate_sidecar(bad, "descriptor");
        FAIL() << "expected an unrecognized jvp_exec.kind to be rejected";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("jvp_exec.kind"), std::string::npos)
            << e.what();
    }
}


}  // namespace SchemaHardeningTest
}  // namespace eagle_tests
