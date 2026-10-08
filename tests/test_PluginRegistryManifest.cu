// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "plugin/plugin_registry/registry.h"

#include "TestBase.h"

#include <fstream>
#include <string>

// Manifest-level duplicate plugin id rejection ("duplicate ids are
// rejected"), C++/CUDA side. The Python producer
// (the code generator's deploy bundling) already guarantees a Bundle-built manifest
// never carries a duplicate id (auto-derived collisions get a numeric
// suffix), so this guards the OTHER path: a manifest assembled/edited by
// hand. PluginRegistry::from_manifest throws before any Driver-API call
// (before cuModuleLoadData) for both cases below, so — like
// test_PluginRegistryDtype.cu's RejectsFloat32NamingRealReasonAndAlternative
// — this is pure host-side JSON validation exercised through the real class,
// not a device smoke test; no live CUDA context is needed.
// Companion: eagle/python/eagle/registry.py::load_manifest (same check,
// same "duplicate plugin id" framing), exercised by
// eagle/python/tests/test_plugin_registry.py.
//
// Manifest-level `pattern` DISCRIMINANT
// strictness, added the same way and in the same file: also pure host-side
// JSON validation, before any Driver-API call.
using namespace eagle::plugin;

namespace eagle_tests {
namespace PluginRegistryManifestTest {

static std::string writeTmp(const std::string& name, const std::string& body)
{
    const std::string path = ::testing::TempDir() + name;
    std::ofstream f(path, std::ios::binary);
    f << body;
    f.close();
    return path;
}

// Two entries share the id "gravity"; neither artifact/sidecar file is ever
// created — the duplicate-id reject fires before from_manifest reads either.
static std::string writeManifestWithDuplicateId()
{
    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "pattern": "vector",
  "plugins": [
    {"id": "gravity", "order": 0, "enabled": true,
     "artifact": "gravity.ptx", "sidecar": "gravity.json", "format": "ptx"},
    {"id": "gravity", "order": 1, "enabled": false,
     "artifact": "gravity2.ptx", "sidecar": "gravity2.json", "format": "ptx"}
  ]
})";
    return writeTmp("dup_id_manifest.json", manifestJson);
}

TEST(PluginRegistryManifestTest, RejectsDuplicatePluginId)
{
    const std::string manifestPath = writeManifestWithDuplicateId();
    try {
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
        FAIL() << "expected from_manifest to reject the duplicate plugin id";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("duplicate"), std::string::npos) << msg;
        EXPECT_NE(msg.find("'gravity'"), std::string::npos) << msg;
    }
}

// The duplicate check fires regardless of the SECOND entry's `enabled` flag
// (above, entry 1 is disabled) and regardless of ORDER: a duplicate later in
// the list is caught just the same as one earlier.
static std::string writeManifestWithThreeEntriesLastDuplicate()
{
    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "pattern": "vector",
  "plugins": [
    {"id": "gravity", "order": 0, "enabled": true,
     "artifact": "gravity.ptx", "sidecar": "gravity.json", "format": "ptx"},
    {"id": "drag_ps", "order": 1, "enabled": true,
     "artifact": "drag.ptx", "sidecar": "drag.json", "format": "ptx"},
    {"id": "gravity", "order": 2, "enabled": true,
     "artifact": "gravity3.ptx", "sidecar": "gravity3.json", "format": "ptx"}
  ]
})";
    return writeTmp("dup_id_manifest_3.json", manifestJson);
}

TEST(PluginRegistryManifestTest, RejectsDuplicatePluginIdNotAdjacent)
{
    const std::string manifestPath = writeManifestWithThreeEntriesLastDuplicate();
    EXPECT_THROW(
        eagle::cuda::PluginRegistry::from_manifest(manifestPath),
        std::runtime_error);
}

// A manifest with distinct ids is unaffected by the new check (still throws
// later, at the missing-artifact cuModuleLoadData — proving the guard is
// specific to the id collision, not a blanket reject of every manifest).
TEST(PluginRegistryManifestTest, DistinctIdsAreNotRejectedByTheDuplicateGuard)
{
    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "pattern": "vector",
  "plugins": [
    {"id": "gravity", "order": 0, "enabled": true,
     "artifact": "gravity_missing.ptx", "sidecar": "gravity_missing.json",
     "format": "ptx"}
  ]
})";
    const std::string manifestPath = writeTmp("distinct_id_manifest.json", manifestJson);
    try {
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
        FAIL() << "expected from_manifest to throw (missing sidecar file)";
    } catch (const std::runtime_error& e) {
        // Reaches the sidecar-read failure, NOT the duplicate-id message —
        // proves the new guard did not fire for a manifest with no collision.
        EXPECT_EQ(std::string(e.what()).find("duplicate"), std::string::npos)
            << e.what();
    }
}

// ----------------------------------------------------------------------- //
// Manifest-level `pattern` DISCRIMINANT strictness. The probe this
// closes: a manifest stamped with an unfamiliar `pattern` is schema-legal
// today under every OTHER C++ gate (aether_abi, schema_version=1, roles,
// scalar_type) — `pattern` is the only field that names which variant it
// actually is, and nothing previously read it here.
//
// Since a later atomic widening
// DEV-1) this door speaks TWO messages, and both are pinned below:
// UNRECOGNIZED -> "not a supported plugin family ...; upgrade eagle", and
// RECOGNIZED-but-uncertified (`neural_block`, the neural descriptor family)
// -> "recognized but not launchable by this loader", with NO upgrade suffix,
// because this build knows the family and simply does not run descriptors.
// ----------------------------------------------------------------------- //
static std::string writeManifestWithPattern(const std::string& name,
                                              const std::string& pattern_json_or_absent)
{
    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,)" + pattern_json_or_absent + R"(
  "plugins": [
    {"id": "gravity", "order": 0, "enabled": true,
     "artifact": "gravity.ptx", "sidecar": "gravity.json", "format": "ptx"}
  ]
})";
    return writeTmp(name, manifestJson);
}

TEST(PluginRegistryManifestTest, RejectsUnrecognizedManifestPattern)
{
    const std::string manifestPath = writeManifestWithPattern(
        "unrecognized_pattern_manifest.json", R"(
  "pattern": "not_a_real_pattern",)");
    try {
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
        FAIL() << "expected from_manifest to reject the unrecognized pattern";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("not_a_real_pattern"), std::string::npos) << msg;
        EXPECT_NE(msg.find("not a supported plugin family"), std::string::npos) << msg;
        EXPECT_NE(msg.find("vector"), std::string::npos) << msg;
        EXPECT_NE(msg.find("pure"), std::string::npos) << msg;
        EXPECT_NE(msg.find("upgrade eagle"), std::string::npos) << msg;
    }
}

// The second branch: `neural_block` used to be this file's inert placeholder for
// "a family this build has never heard of". The step-1b widening made it REAL —
// RECOGNIZED by the shared validator, never launch-certified — so the message it
// earns here changed, and this test is where that change is pinned rather than
// silently absorbed.
TEST(PluginRegistryManifestTest, RejectsRecognizedButUnlaunchableNeuralBlockManifest)
{
    const std::string manifestPath = writeManifestWithPattern(
        "neural_block_manifest.json", R"(
  "pattern": "neural_block",)");
    try {
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
        FAIL() << "expected from_manifest to refuse to launch a neural_block manifest";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("neural_block"), std::string::npos) << msg;
        EXPECT_NE(msg.find("recognized but not launchable"), std::string::npos) << msg;
        EXPECT_EQ(msg.find("upgrade eagle"), std::string::npos) << msg;
    }
}

// A "blocks[]" section under a NON-neural pattern is the live
// hazard — a kernel-manifest loader would silently ignore the descriptors and
// load bare kernels. The shared manifest-validation helper (validate_manifest_formats,
// reached AFTER the pattern gate lets "pure"/"vector" through) refuses it loudly with
// the same leading clause the Python door emits. Device twin of corpus row 25b.
TEST(PluginRegistryManifestTest, RejectsOrphanedBlocksSection)
{
    const std::string manifestPath = writeManifestWithPattern(
        "orphaned_blocks_pure_manifest.json", R"(
  "pattern": "pure",
  "blocks": [{"kernel": "orphan"}],)");
    try {
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
        FAIL() << "expected from_manifest to reject blocks[] under a non-neural pattern";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("carrying blocks[] must declare pattern 'neural_block'"),
                  std::string::npos) << msg;
        EXPECT_NE(msg.find("pure"), std::string::npos) << msg;
    }
}

// A recognized value ("vector"/"pure") is unaffected by the new gate — proven
// by every other test in this file (all stamp `"pattern": "vector"` and reach
// their own, later, failure).

// Absence stays lenient in C++ (pre-freeze manifests never stamped one): the
// pattern gate does not fire, so the manifest reaches the NEXT failure
// (missing sidecar file), not a pattern-related one.
TEST(PluginRegistryManifestTest, AbsentManifestPatternStaysLenient)
{
    const std::string manifestPath = writeManifestWithPattern(
        "absent_pattern_manifest.json", "");
    try {
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
        FAIL() << "expected from_manifest to throw (missing sidecar file)";
    } catch (const std::runtime_error& e) {
        // Reaches the sidecar-read failure, NOT a pattern-related message.
        const std::string msg = e.what();
        EXPECT_EQ(msg.find("supported plugin family"), std::string::npos) << msg;
        EXPECT_EQ(msg.find("pattern"), std::string::npos) << msg;
    }
}

// ----------------------------------------------------------------------- //
// The DEVICE registry's manifest-entry
// `format` gate (plugin_registry/manifest.h's `validate_manifest_formats`,
// shared with the host registry) and its manifest-level `schema_version` gate
// (registry.h's `from_manifest`) were, until this check, exercised by ZERO tests: every
// existing .cu manifest fixture in this file stamps `"format": "ptx"` and
// `"schema_version": 1`, and the corpus's C++ manifest loader
// (`cpp_load_manifest`) maps to the HOST registry only (this driver TU is
// CUDA-free by design — see test_ConformanceCorpus.cpp's banner), so neither
// device call site was ever reached. The two tests below are the first-ever
// executions of both gates.
// ----------------------------------------------------------------------- //

// A single declared entry stamped an artifact `format` this build does not
// recognize. `from_manifest` checks every entry's format (manifest.h
// `validate_manifest_formats`) BEFORE any sidecar parse / module load, so the
// referenced gravity.ptx/gravity.json need not exist.
TEST(PluginRegistryManifestTest, RejectsUnknownManifestFormat)
{
    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "pattern": "vector",
  "plugins": [
    {"id": "gravity", "order": 0, "enabled": true,
     "artifact": "gravity.ptx", "sidecar": "gravity.json",
     "format": "not_a_manifest_format"}
  ]
})";
    const std::string manifestPath
        = writeTmp("unknown_format_manifest.json", manifestJson);
    try {
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
        FAIL() << "expected from_manifest to reject the unrecognized format "
                  "'not_a_manifest_format'";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("unknown artifact format"), std::string::npos) << msg;
        EXPECT_NE(msg.find("not_a_manifest_format"), std::string::npos) << msg;
        EXPECT_NE(msg.find("upgrade eagle"), std::string::npos) << msg;
    }
}

// A manifest stamped a schema_version newer than this build supports —
// kPluginMaxSchemaVersion + 1: the loader
// CEILING and the cross-repo constant (kPluginSchemaVersion) are separate
// constants now, and schema v2 LOADS, so using the wrong one here would have asserted
// that a version this loader accepts is rejected.
// `from_manifest`'s forward-strict schema gate fires before the pattern check,
// the duplicate-id check, and the format check, so the referenced
// gravity.ptx/gravity.json need not exist either.
TEST(PluginRegistryManifestTest, RejectsNewerManifestSchemaVersion)
{
    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": )" + std::to_string(kPluginMaxSchemaVersion + 1) + R"(,
  "pattern": "vector",
  "plugins": [
    {"id": "gravity", "order": 0, "enabled": true,
     "artifact": "gravity.ptx", "sidecar": "gravity.json", "format": "ptx"}
  ]
})";
    const std::string manifestPath
        = writeTmp("newer_schema_version_manifest.json", manifestJson);
    try {
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
        FAIL() << "expected from_manifest to reject the newer schema_version";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("upgrade eagle"), std::string::npos) << msg;
    }
}

}  // namespace PluginRegistryManifestTest
}  // namespace eagle_tests
