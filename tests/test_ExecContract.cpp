// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The HETEROGENEOUS EXECUTION CONTRACT — host/CUDA-free rows (aether-abi/2).
//
// Normative source: the heterogeneous execution contract spec, with the
// defects found by an adversarial review closed by the rows below.
//
// This is a `.cpp` with no CUDA in it and it is compiled and run in BOTH build
// modes (tests/CMakeLists.txt adds it outside the mode-conditional glob, with a
// forced EAGLE_CPU_ONLY, exactly like test_HostDispatchContract.cpp): the loader
// gates, the placement rules and the host execution structure are the same product
// on both arms, so they must be gated on both. The device face — the PTX layout
// symbol, the device launch triple, the partition-identity and host/device-twin
// rows — is test_ExecContractDevice.cu, which the CUDA gate carries.
//
// FIXTURES (hand-written, ABI-only): fixtures/host_plugin_execv2.cpp
// (the v2 pair's host half: sample_local / mapreduce / cross_sample_write bodies,
// plus the wide-role and triple probes and a legacy v1 entry),
// fixtures/host_plugin_badlayout.cpp and fixtures/host_plugin_nolayout.cpp (the
// layout self-check's two RED arms).

#include "eagle/exec/HostTeam.h"
#include "eagle/exec/Partition.h"
#include "plugin/host_registry.h"
#include "plugin/plugin_registry/manifest.h"

// Deliberately NOT "TestBase.h" — same reasoning as test_ConformanceCorpus.cpp:
// that header pulls eagle's device headers, whose __device__/__host__ attributes
// only nvcc parses, and in CUDA mode this `.cpp` is compiled by g++.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

using namespace eagle::plugin;
namespace ex = eagle::exec;

namespace eagle_tests {
namespace ExecContractTest {

// ---------------------------------------------------------------------------
// Fixture staging helpers
// ---------------------------------------------------------------------------
static std::string writeTmp(const std::string& name, const std::string& body)
{
    const std::string path = ::testing::TempDir() + name;
    std::ofstream f(path, std::ios::binary);
    f << body;
    f.close();
    return path;
}

// A schema-v2 sidecar for one of the fixture bodies.
static Sidecar v2Sidecar(const std::string& kernel, const std::string& argSpec)
{
    const std::string json = R"({
  "kernel": ")" + kernel + R"(",
  "aether_abi": "aether-abi/2",
  "schema_version": 2,
  "pattern": "pure",
  "scalar_type": "float64",
  "arg_spec": )" + argSpec + R"(
})";
    return parse_sidecar(writeTmp("ec_" + kernel + "_sidecar.json", json));
}

// A schema-v1 sidecar (the legacy bridge) for the same fixture object.
static Sidecar v1Sidecar(const std::string& kernel, const std::string& argSpec)
{
    const std::string json = R"({
  "kernel": ")" + kernel + R"(",
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "pattern": "pure",
  "scalar_type": "float64",
  "arg_spec": )" + argSpec + R"(
})";
    return parse_sidecar(writeTmp("ec_v1_" + kernel + "_sidecar.json", json));
}

// The manifest a loader-level row hands `eagle::cpu::PluginRegistry::load`.
// `execKeys` is spliced in verbatim so a row can state exactly which of the flat
// execution keys it declares (including none, or an invalid one).
static std::string writeManifest(const std::string& tag, const std::string& abi,
                                 int schemaVersion, const std::string& execKeys,
                                 const std::string& sidecarName)
{
    const std::string json = R"({
  "schema_version": )" + std::to_string(schemaVersion) + R"(,
  "pattern": "pure",
  "aether_abi": ")" + abi + R"(",
)" + execKeys + R"(  "plugins": [
    {"id": ")" + tag + R"(", "order": 0, "enabled": true,
     "artifact": ")" + std::string(HOST_PLUGIN_EXECV2_SO) + R"(",
     "sidecar": ")" + sidecarName + R"(", "format": "ptx"}
  ]
})";
    return writeTmp("ec_" + tag + "_manifest.json", json);
}

// arg_spec literals, shared by the sidecar builders and the fixture bodies.
static const char* kLocalArgs =
    R"([["mutable","y"],["per_sample","x"],["uniform","a"],["uniform","b"],["nsamples","nsamples"]])";
static const char* kMapreduceArgs =
    R"([["accum_out","partial"],["per_sample","x"],["nsamples","nsamples"]])";
static const char* kScatterArgs =
    R"([["wide_out","acc"],["per_sample","x"],["nsamples","nsamples"]])";
static const char* kWideArgs =
    R"([["wide_out","wout"],["wide_in","win"],["accum_out","aout"],["nsamples","nsamples"]])";
static const char* kTripleArgs = R"([["wide_out","t"],["nsamples","nsamples"]])";
static const char* kLegacyArgs =
    R"([["mutable","y"],["per_sample","x"],["nsamples","nsamples"]])";

// The RULED band (fork F-d): S x 2 x eps(float64), S = the number of elements
// folded, applied RELATIVELY. Derived from one anchor, never fitted to an observed
// difference — the calibrated-bound rule.
static double band(std::size_t S)
{
    return double(S) * 2.0 * std::numeric_limits<double>::epsilon();
}
static ::testing::AssertionResult withinBand(double a, double b, std::size_t S)
{
    const double scale = std::max({ 1.0, std::fabs(a), std::fabs(b) });
    const double tol = band(S) * scale;
    if (std::fabs(a - b) <= tol) return ::testing::AssertionSuccess();
    return ::testing::AssertionFailure()
        << "|" << a << " - " << b << "| = " << std::fabs(a - b)
        << " exceeds the ruled band S*2*eps*scale = " << tol << " (S=" << S << ")";
}

// ---------------------------------------------------------------------------
// The role register and its two disagreeing halves
// ---------------------------------------------------------------------------
TEST(ExecContractTest, RoleRegisterCarriesAccumOut)
{
    EXPECT_TRUE(is_valid_role("accum_out"));
    EXPECT_TRUE(is_valid_role("wide_in"));
    EXPECT_TRUE(is_valid_role("wide_out"));
    EXPECT_EQ(sizeof(kPluginArgRoles) / sizeof(kPluginArgRoles[0]), 12u);
    EXPECT_FALSE(is_valid_role("no_such_role"));
}

// ---------------------------------------------------------------------------
// The loader: both tags, the schema ceiling, the flat execution keys
// ---------------------------------------------------------------------------
TEST(ExecContractTest, BothAbiTagsAreAccepted)
{
    EXPECT_NO_THROW(check_aether_abi(EAGLE_AETHER_ABI, "probe"));
    EXPECT_NO_THROW(check_aether_abi(EAGLE_AETHER_ABI_V2, "probe"));
    // Presence-required semantics are UNCHANGED: an empty tag is still refused.
    EXPECT_THROW(check_aether_abi("", "probe"), std::runtime_error);
    try {
        check_aether_abi("aether-abi/3", "probe");
        FAIL() << "expected aether-abi/3 to be refused";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("built for"), std::string::npos) << msg;
        EXPECT_NE(msg.find("aether-abi/1"), std::string::npos) << msg;
        EXPECT_NE(msg.find("aether-abi/2"), std::string::npos) << msg;
    }
}

TEST(ExecContractTest, AbiVersionOfResolvesTheGeneration)
{
    EXPECT_EQ(abi_version_of(EAGLE_AETHER_ABI), 1);
    EXPECT_EQ(abi_version_of(EAGLE_AETHER_ABI_V2), 2);
    EXPECT_EQ(abi_version_of("aether-abi/3"), 0);
    EXPECT_EQ(abi_version_of(""), 0);
}

TEST(ExecContractTest, SchemaPinStaysOneWhileTheCeilingIsTwo)
{
    // The cross-repo constant does NOT move (raptor / eagle-Python / this
    // header / the code generator's sanctioned copy); the LOADER ceiling does.
    EXPECT_EQ(kPluginSchemaVersion, 1);
    EXPECT_EQ(kPluginMaxSchemaVersion, 2);
}

TEST(ExecContractTest, ManifestSchemaBeyondMaxIsRefused)
{
    const Sidecar sc = v2Sidecar("exec_local", kLocalArgs);
    const std::string path = writeManifest("beyondmax", "aether-abi/2",
        kPluginMaxSchemaVersion + 1,
        "  \"exec_targets\": [\"host\"],\n  \"exec_access\": \"sample_local\",\n",
        "ec_exec_local_sidecar.json");
    const Manifest m = parse_manifest(path);
    eagle::cpu::PluginRegistry reg;
    try {
        reg.load(m, ::testing::TempDir());
        FAIL() << "expected a schema v3 manifest to be refused";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("upgrade eagle"), std::string::npos)
            << e.what();
    }
    (void)sc;
}

TEST(ExecContractTest, ManifestSchemaTwoParsesTheFlatExecutionKeys)
{
    v2Sidecar("exec_mapreduce", kMapreduceArgs);
    const std::string path = writeManifest("flatkeys", "aether-abi/2", 2,
        "  \"exec_targets\": [\"device\", \"host\"],\n"
        "  \"exec_access\": \"mapreduce\",\n  \"exec_op\": \"sum\",\n",
        "ec_exec_mapreduce_sidecar.json");
    const Manifest m = parse_manifest(path);
    EXPECT_TRUE(m.has_exec_targets);
    EXPECT_TRUE(m.has_exec_access);
    EXPECT_TRUE(m.has_exec_op);
    ASSERT_EQ(m.exec_targets.size(), 2u);
    EXPECT_EQ(m.exec_targets[0], "device");
    EXPECT_EQ(m.exec_targets[1], "host");
    EXPECT_EQ(m.exec_access, "mapreduce");
    EXPECT_EQ(m.exec_op, "sum");
    EXPECT_NO_THROW(validate_execution_axis(m, "probe"));
}

TEST(ExecContractTest, ManifestSchemaTwoWithoutTheExecutionAxisIsRefused)
{
    v2Sidecar("exec_local", kLocalArgs);
    const std::string path = writeManifest("noaxis", "aether-abi/2", 2, "",
                                           "ec_exec_local_sidecar.json");
    const Manifest m = parse_manifest(path);
    try {
        validate_execution_axis(m, "host manifest");
        FAIL() << "expected a v2 manifest with no execution axis to be refused";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("exec_access"), std::string::npos) << msg;
        EXPECT_NE(msg.find("exec_targets"), std::string::npos) << msg;
        EXPECT_NE(msg.find("absence = load refused"), std::string::npos) << msg;
    }
}

TEST(ExecContractTest, ManifestSchemaOneCarryingExecutionKeysIsRefused)
{
    v2Sidecar("exec_local", kLocalArgs);
    const std::string path = writeManifest("v1axis", "aether-abi/1", 1,
        "  \"exec_targets\": [\"host\"],\n  \"exec_access\": \"sample_local\",\n",
        "ec_exec_local_sidecar.json");
    const Manifest m = parse_manifest(path);
    try {
        validate_execution_axis(m, "host manifest");
        FAIL() << "expected a v1 manifest carrying the execution axis to be refused";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("schema-v2-only"), std::string::npos)
            << e.what();
    }
}

TEST(ExecContractTest, ExecutionAxisVocabularyIsValueStrict)
{
    v2Sidecar("exec_local", kLocalArgs);
    struct Row { const char* keys; const char* substr; };
    const Row rows[] = {
        { "  \"exec_targets\": [\"gpu\"],\n  \"exec_access\": \"sample_local\",\n",
          "exec_targets must be a non-empty list" },
        { "  \"exec_targets\": [],\n  \"exec_access\": \"sample_local\",\n",
          "exec_targets must be a non-empty list" },
        { "  \"exec_targets\": [\"host\", \"host\"],\n  \"exec_access\": \"sample_local\",\n",
          "with no duplicates" },
        { "  \"exec_targets\": [\"host\"],\n  \"exec_access\": \"telepathy\",\n",
          "is not a supported access class" },
        { "  \"exec_targets\": [\"host\"],\n  \"exec_access\": \"mapreduce\",\n",
          "requires exec_op" },
        { "  \"exec_targets\": [\"host\"],\n  \"exec_access\": \"sample_local\",\n"
          "  \"exec_op\": \"sum\",\n",
          "exec_op is forbidden unless" },
        { "  \"exec_targets\": [\"host\"],\n  \"exec_access\": \"mapreduce\",\n"
          "  \"exec_op\": \"convolve\",\n",
          "is not a supported reduction op" },
    };
    int row = 0;
    for (const Row& r : rows) {
        SCOPED_TRACE(std::string("execution-axis row: ") + r.keys);
        const std::string path = writeManifest("vocab" + std::to_string(row++),
            "aether-abi/2", 2, r.keys, "ec_exec_local_sidecar.json");
        const Manifest m = parse_manifest(path);
        try {
            validate_execution_axis(m, "host manifest");
            ADD_FAILURE() << "expected a refusal for: " << r.keys;
        } catch (const std::runtime_error& e) {
            EXPECT_NE(std::string(e.what()).find(r.substr), std::string::npos)
                << e.what();
        }
    }
}

TEST(ExecContractTest, SchemaVersionAndAbiTagMustAgree)
{
    v2Sidecar("exec_local", kLocalArgs);
    const std::string path = writeManifest("mixed", "aether-abi/1", 2,
        "  \"exec_targets\": [\"host\"],\n  \"exec_access\": \"sample_local\",\n",
        "ec_exec_local_sidecar.json");
    const Manifest m = parse_manifest(path);
    try {
        validate_execution_axis(m, "host manifest");
        FAIL() << "expected a schema-v2 / aether-abi/1 manifest to be refused";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("aether-abi/2"), std::string::npos)
            << e.what();
    }
}

// ---------------------------------------------------------------------------
// The layout self-check (host face: dlsym)
// ---------------------------------------------------------------------------
TEST(ExecContractTest, LayoutSelfCheckExpectsThisBuildsSizes)
{
    std::uint64_t want[kEagleLayoutFieldCount];
    expected_layout_sizes(want);
    EXPECT_EQ(want[0], 40u);
    EXPECT_EQ(want[1], 32u);
    EXPECT_EQ(want[2], 32u);
    EXPECT_EQ(want[3], sizeof(EAGLE_ABI_INDEX_T));  // review finding C1
    EXPECT_EQ(want[4], 24u);
    EXPECT_NO_THROW(check_layout_sizes(want, "probe"));
}

TEST(ExecContractTest, HostV2PluginWithWrongLayoutIsRefusedNamingTheField)
{
    const Sidecar sc = v2Sidecar("exec_bad", R"([["mutable","y"]])");
    eagle::cpu::PluginRegistry reg;
    try {
        reg.add_plugin(sc, HOST_PLUGIN_BADLAYOUT_SO);
        FAIL() << "expected the wrong-layout plugin to be refused";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("sizeof(partition triple)"), std::string::npos) << msg;
        EXPECT_NE(msg.find("16"), std::string::npos) << msg;
        EXPECT_NE(msg.find("24"), std::string::npos) << msg;
    }
    EXPECT_EQ(reg.size(), 0u);
}

TEST(ExecContractTest, HostV2PluginWithoutTheLayoutSymbolIsRefused)
{
    const Sidecar sc = v2Sidecar("exec_nolayout", R"([["mutable","y"]])");
    eagle::cpu::PluginRegistry reg;
    try {
        reg.add_plugin(sc, HOST_PLUGIN_NOLAYOUT_SO);
        FAIL() << "expected a v2 plugin with no layout symbol to be refused";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("eagle_layout_sizes"), std::string::npos)
            << e.what();
    }
}

TEST(ExecContractTest, HostV1PluginNeedsNoLayoutSymbol)
{
    // A legacy artifact predates the self-check and must keep loading.
    const Sidecar sc = v1Sidecar("exec_nolayout", R"([["mutable","y"]])");
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sc, HOST_PLUGIN_NOLAYOUT_SO));
    EXPECT_EQ(reg.abi_version(0), 1);
}

// ---------------------------------------------------------------------------
// The three wide/accum roles are LAUNCHABLE
// ---------------------------------------------------------------------------
TEST(ExecContractTest, HostRegistryLaunchesTheWideAndAccumRoles)
{
    constexpr std::int64_t N = 64;
    std::vector<double> win(N), wout(N, 0.0), aout(N, 0.0);
    for (std::int64_t i = 0; i < N; ++i) win[std::size_t(i)] = double(i) * 0.5;

    const Sidecar sc = v2Sidecar("exec_wide", kWideArgs);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO));
    EXPECT_EQ(reg.abi_version(0), 2);
    reg.bind_handle("win", win.data());
    reg.bind_handle("wout", wout.data());
    reg.bind_handle("aout", aout.data());
    ASSERT_EQ(reg.run(std::int32_t(N)), 1);
    for (std::int64_t i = 0; i < N; ++i) {
        ASSERT_DOUBLE_EQ(wout[std::size_t(i)], 2.0 * win[std::size_t(i)]);
        ASSERT_DOUBLE_EQ(aout[std::size_t(i)], win[std::size_t(i)] + 1.0);
    }
}

TEST(ExecContractTest, HostRegistryRefusesAnUnknownRoleStill)
{
    const Sidecar sc = v2Sidecar("exec_wide", R"([["telepathy","t"]])");
    eagle::cpu::PluginRegistry reg;
    // The role is refused at LOAD by the shared validator, before any launch.
    EXPECT_THROW(reg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO), std::runtime_error);
}

// ---------------------------------------------------------------------------
// The launch args: the int64 triple reaches the body
// ---------------------------------------------------------------------------
TEST(ExecContractTest, HostV2EntryReceivesThePartitionTriple)
{
    constexpr std::int64_t N = 100;
    std::vector<double> t(3, -1.0);
    const Sidecar sc = v2Sidecar("exec_triple", kTripleArgs);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO));
    reg.bind_handle("t", t.data());

    // Whole view: {0, N, N}.
    ASSERT_EQ(reg.run(std::int32_t(N)), 1);
    EXPECT_DOUBLE_EQ(t[0], 0.0);
    EXPECT_DOUBLE_EQ(t[1], double(N));
    EXPECT_DOUBLE_EQ(t[2], double(N));

    // A partition: base and count move, nSamples does NOT (L2 — every expression
    // that bakes the sample count receives the TRUE total).
    ASSERT_EQ(reg.run_partition(ex::Partition{ 40, 25, N }), 1);
    EXPECT_DOUBLE_EQ(t[0], 40.0);
    EXPECT_DOUBLE_EQ(t[1], 25.0);
    EXPECT_DOUBLE_EQ(t[2], double(N));
}

TEST(ExecContractTest, LegacyV1PluginRefusesANonWholePartition)
{
    constexpr std::int64_t N = 32;
    std::vector<double> x(N, 1.0), y(N, 0.0);
    const Sidecar sc = v1Sidecar("exec_legacy", kLegacyArgs);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO));
    reg.bind_handle("x", x.data());
    reg.bind_handle("y", y.data());
    ASSERT_EQ(reg.run(std::int32_t(N)), 1);           // whole view still works
    EXPECT_DOUBLE_EQ(y[0], 2.0);
    try {
        reg.run_partition(ex::Partition{ 0, 16, N });
        FAIL() << "expected a legacy plugin to refuse a non-whole partition";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("WHOLE-VIEW"), std::string::npos)
            << e.what();
    }
    EXPECT_THROW(reg.entry_v2(0), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Placement legality
// ---------------------------------------------------------------------------
TEST(ExecContractTest, PlacementAllowsEveryImplementedStructureForLocalAndMapreduce)
{
    // RankPartition joined the implemented set, so it is enumerated here with
    // the other two rather than in a structure-specific row of its own.
    for (ex::Access a : { ex::Access::SampleLocal, ex::Access::CrossSampleRead,
                          ex::Access::MapReduce }) {
        for (ex::Structure s : { ex::Structure::DeviceKernel,
                                 ex::Structure::HostTeam,
                                 ex::Structure::RankPartition }) {
            EXPECT_NO_THROW(ex::check_placement(a, s, 1));
            EXPECT_NO_THROW(ex::check_placement(a, s, 4));
        }
    }
}

TEST(ExecContractTest, PlacementAllowsCrossSampleReadUnderRankPartition)
{
    // The ruling made explicit: a `cross_sample_read` body is legal across ranks
    // BECAUSE RankPartition replicates its inputs, so the lookup/Staged tables it
    // reads are whole on every rank — L3's stated condition, met by construction.
    // Pinned in its own row so a future change that partitions those tables has to
    // delete a rule, not quietly widen a loop.
    for (int np : { 1, 2, 8 })
        EXPECT_NO_THROW(ex::check_placement(ex::Access::CrossSampleRead,
                                            ex::Structure::RankPartition, np));
    EXPECT_NO_THROW(ex::check_placement("cross_sample_read", "rank_partition", 2));
}

TEST(ExecContractTest, PlacementRefusesCrossSampleWriteAcrossPartitions)
{
    // Legal WHOLE: the deterministic accumulate stays as shipped.
    EXPECT_NO_THROW(ex::check_placement(ex::Access::CrossSampleWrite,
                                        ex::Structure::HostTeam, 1));
    for (ex::Structure s : { ex::Structure::DeviceKernel,
                             ex::Structure::HostTeam }) {
        try {
            ex::check_placement(ex::Access::CrossSampleWrite, s, 2);
            ADD_FAILURE() << "expected cross_sample_write x 2 partitions to refuse";
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("cross_sample_write"), std::string::npos) << msg;
            EXPECT_NE(msg.find("partitions"), std::string::npos) << msg;
        }
    }
}

TEST(ExecContractTest, PlacementRefusesCrossSampleWriteUnderRankPartition)
{
    // F-e, and refused at EVERY world size including one: a placement that is legal
    // at `-np 1` and illegal at `-np 2` is a trap, since the structure exists to be
    // run at more than one rank.
    for (int np : { 1, 2, 8 }) {
        try {
            ex::check_placement(ex::Access::CrossSampleWrite,
                                ex::Structure::RankPartition, np);
            ADD_FAILURE() << "expected cross_sample_write on rank_partition to refuse"
                          << " at npartitions=" << np;
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("cross_sample_write"), std::string::npos) << msg;
            EXPECT_NE(msg.find("rank_partition"), std::string::npos) << msg;
            EXPECT_NE(msg.find("F-e"), std::string::npos) << msg;
        }
    }
    EXPECT_THROW(ex::check_placement("cross_sample_write", "rank_partition", 1),
                 std::runtime_error);
}

TEST(ExecContractTest, PlacementRefusesTheUnimplementedStructures)
{
    // Only DeviceGroup is left in this class now -- and the message no longer
    // names the rank-partition case, because it shipped.
    const ex::Structure s = ex::Structure::DeviceGroup;
    for (int np : { 1, 4 }) {
        try {
            ex::check_placement(ex::Access::SampleLocal, s, np);
            ADD_FAILURE() << "expected " << ex::to_string(s) << " to refuse";
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("NCCL not implemented"), std::string::npos) << msg;
            EXPECT_NE(msg.find(ex::to_string(s)), std::string::npos) << msg;
        }
    }
}

TEST(ExecContractTest, LegacyV1PluginIsRefusedUnderTheRankStructure)
{
    // On the STRUCTURE axis: a v1 artifact takes no triple, so it cannot be
    // cut across ranks at all — and the refusal must not depend on the world size,
    // where a 1-rank share happens to BE the whole view.
    for (ex::Structure s : { ex::Structure::RankPartition,
                             ex::Structure::DeviceGroup }) {
        try {
            ex::check_legacy_structure(1, s, "probe");
            ADD_FAILURE() << "expected a v1 plugin on " << ex::to_string(s)
                          << " to refuse";
        } catch (const std::runtime_error& e) {
            const std::string msg = e.what();
            EXPECT_NE(msg.find("aether-abi/1"), std::string::npos) << msg;
            EXPECT_NE(msg.find("L13"), std::string::npos) << msg;
            EXPECT_NE(msg.find(ex::to_string(s)), std::string::npos) << msg;
        }
    }
    // A v2 plugin passes, and a v1 plugin on a single-device structure is exactly
    // today's legacy bridge — untouched.
    EXPECT_NO_THROW(ex::check_legacy_structure(2, ex::Structure::RankPartition, "probe"));
    EXPECT_NO_THROW(ex::check_legacy_structure(1, ex::Structure::HostTeam, "probe"));
    EXPECT_NO_THROW(ex::check_legacy_structure(1, ex::Structure::DeviceKernel, "probe"));
}

TEST(ExecContractTest, PlacementParsesTheWireSpellingsAndRefusesUnknownOnes)
{
    EXPECT_NO_THROW(ex::check_placement("sample_local", "host_team", 2));
    EXPECT_THROW(ex::check_placement("cross_sample_write", "device_kernel", 2),
                 std::runtime_error);
    EXPECT_NO_THROW(ex::check_placement("sample_local", "rank_partition", 2));
    EXPECT_THROW(ex::check_placement("sample_local", "device_group", 1),
                 std::runtime_error);
    EXPECT_THROW(ex::check_placement("telepathy", "host_team", 1),
                 std::runtime_error);
    EXPECT_THROW(ex::check_placement("sample_local", "quantum_mesh", 1),
                 std::runtime_error);
    EXPECT_THROW(ex::check_placement("sample_local", "host_team", 0),
                 std::runtime_error);
}

TEST(ExecContractTest, FoldIsAFixedAscendingCombine)
{
    const double v[] = { 1.0, 2.0, 4.0, 8.0 };
    EXPECT_DOUBLE_EQ(ex::fold(ex::ReduceOp::Sum, v, 4), 15.0);
    EXPECT_DOUBLE_EQ(ex::fold(ex::ReduceOp::Times, v, 4), 64.0);
    EXPECT_DOUBLE_EQ(ex::fold(ex::ReduceOp::Max, v, 4), 8.0);
    EXPECT_DOUBLE_EQ(ex::fold(ex::ReduceOp::LAnd, v, 4), 1.0);
    // The identity of an EMPTY fold is the operator's own, never zero for all.
    EXPECT_DOUBLE_EQ(ex::fold(ex::ReduceOp::Times, v, 0), 1.0);
    EXPECT_DOUBLE_EQ(ex::fold(ex::ReduceOp::Sum, v, 0), 0.0);
    EXPECT_EQ(ex::op_from_string("sum"), ex::ReduceOp::Sum);
    EXPECT_THROW(ex::op_from_string("convolve"), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Partition identity, the mapreduce band, and the host team's schedule
// ---------------------------------------------------------------------------
namespace {
// Load `exec_local` and run it over an arbitrary list of partitions, returning y.
std::vector<double> runLocal(const std::vector<ex::Partition>& parts,
                             std::int64_t N, bool team, std::size_t bytesPerSample)
{
    std::vector<double> x(static_cast<std::size_t>(N)), y(std::size_t(N), 0.0);
    for (std::int64_t i = 0; i < N; ++i)
        x[std::size_t(i)] = 1.0 / double(i + 1);
    const Sidecar sc = v2Sidecar("exec_local", kLocalArgs);
    eagle::cpu::PluginRegistry reg;
    reg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO);
    reg.bind_handle("x", x.data());
    reg.bind_handle("y", y.data());
    reg.bind_uniform("a", 3.25);
    reg.bind_uniform("b", -0.5);
    auto packed = reg.pack(0, N);
    for (const ex::Partition& p : parts) {
        if (team)
            ex::HostTeam::run(reg.entry_v2(0), packed.params, p, bytesPerSample);
        else
            ex::HostTeam::run_serial(reg.entry_v2(0), packed.params.data(), p);
    }
    return y;
}
}  // namespace

TEST(ExecContractTest, HostPartitionIdentityIsBitExact)
{
    // A sample_local body is BIT-identical whole vs partitioned vs teamed.
    constexpr std::int64_t N = 1000;
    const auto whole = runLocal({ ex::Partition::whole(N) }, N, false, 0);
    const auto split = runLocal({ ex::Partition{ 0, 400, N },
                                  ex::Partition{ 400, 600, N } }, N, false, 0);
    const auto teamed = runLocal({ ex::Partition::whole(N) }, N, true, 0);
    ASSERT_EQ(whole.size(), std::size_t(N));
    for (std::size_t i = 0; i < whole.size(); ++i) {
        ASSERT_EQ(std::memcmp(&whole[i], &split[i], sizeof(double)), 0)
            << "partitioned run differs at sample " << i;
        ASSERT_EQ(std::memcmp(&whole[i], &teamed[i], sizeof(double)), 0)
            << "teamed run differs at sample " << i;
    }
    // Non-vacuity: the values are not all the trivially-equal default.
    EXPECT_NE(whole[0], whole[1]);
}

TEST(ExecContractTest, HostTeamTilesCoverEverySampleExactlyOnce)
{
    constexpr std::int64_t N = 5000;
    std::vector<double> hits(std::size_t(N), 0.0), x(std::size_t(N), 0.0);
    // x[i] = i + 1 makes the SLOT identity observable: a tile that re-based its
    // local index at 0 (the collision class this test names) would write
    // the wrong value into the wrong slot and leave the tail at its 0 seed.
    for (std::int64_t i = 0; i < N; ++i) x[std::size_t(i)] = double(i) + 1.0;
    const Sidecar sc = v2Sidecar("exec_mapreduce", kMapreduceArgs);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO));
    reg.bind_handle("x", x.data());
    reg.bind_handle("partial", hits.data());
    auto packed = reg.pack(0, N);
    const std::size_t tiles =
        ex::HostTeam::run(reg.entry_v2(0), packed.params, ex::Partition::whole(N), 0);
    EXPECT_EQ(tiles, ex::HostTeam::tile_count(ex::Partition::whole(N), 0));
    EXPECT_GT(tiles, 1u) << "the schedule must actually be tiled for this row to "
                            "say anything (tile size "
                         << ex::HostTeam::tile_size(0) << ")";
    for (std::size_t i = 0; i < hits.size(); ++i)
        ASSERT_DOUBLE_EQ(hits[i], double(i) + 1.0) << "sample " << i << " carries "
            "the wrong tile's value or none at all (review finding H2's collision "
            "class)";
}

TEST(ExecContractTest, HostMapreduceWholeVsPartitionsWithinBand)
{
    constexpr std::int64_t N = 4096;
    std::vector<double> x(static_cast<std::size_t>(N)), partial(std::size_t(N), 0.0);
    for (std::int64_t i = 0; i < N; ++i)
        x[std::size_t(i)] = 1.0 / double(i + 1);   // ordering-sensitive by design

    const Sidecar sc = v2Sidecar("exec_mapreduce", kMapreduceArgs);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO));
    reg.bind_handle("x", x.data());
    reg.bind_handle("partial", partial.data());
    auto packed = reg.pack(0, N);

    // WHOLE: one partition, one fold over all N contributions.
    ex::HostTeam::run_serial(reg.entry_v2(0), packed.params.data(),
                             ex::Partition::whole(N));
    const double whole = ex::fold(ex::ReduceOp::Sum, partial.data(), std::size_t(N));

    // TWO PARTITIONS: eagle folds each partition, then combines the partials — a
    // DIFFERENT association of the same terms, which is exactly why this class is
    // band-gated rather than bit-exact.
    std::fill(partial.begin(), partial.end(), 0.0);
    const ex::Partition p0{ 0, 1500, N }, p1{ 1500, N - 1500, N };
    ex::HostTeam::run_serial(reg.entry_v2(0), packed.params.data(), p0);
    ex::HostTeam::run_serial(reg.entry_v2(0), packed.params.data(), p1);
    const double r0 = ex::fold(ex::ReduceOp::Sum, partial.data() + p0.base,
                               std::size_t(p0.count));
    const double r1 = ex::fold(ex::ReduceOp::Sum, partial.data() + p1.base,
                               std::size_t(p1.count));
    const double partials[2] = { r0, r1 };
    const double split = ex::fold(ex::ReduceOp::Sum, partials, 2);

    EXPECT_TRUE(withinBand(whole, split, std::size_t(N)));
    // Non-vacuity: the band must be tight enough that a DROPPED partition fails it.
    EXPECT_FALSE(withinBand(whole, r0, std::size_t(N)))
        << "the band is so wide that losing a whole partition still passes";
}

TEST(ExecContractTest, HostTeamRunsTheMapreduceBodyWithinTheSameBand)
{
    constexpr std::int64_t N = 4096;
    std::vector<double> x(static_cast<std::size_t>(N)), partial(std::size_t(N), 0.0);
    for (std::int64_t i = 0; i < N; ++i) x[std::size_t(i)] = 1.0 / double(i + 1);
    const Sidecar sc = v2Sidecar("exec_mapreduce", kMapreduceArgs);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO));
    reg.bind_handle("x", x.data());
    reg.bind_handle("partial", partial.data());
    auto packed = reg.pack(0, N);
    ex::HostTeam::run(reg.entry_v2(0), packed.params, ex::Partition::whole(N), 0);
    const double teamed = ex::fold(ex::ReduceOp::Sum, partial.data(), std::size_t(N));

    std::fill(partial.begin(), partial.end(), 0.0);
    ex::HostTeam::run_serial(reg.entry_v2(0), packed.params.data(),
                             ex::Partition::whole(N));
    const double serial = ex::fold(ex::ReduceOp::Sum, partial.data(), std::size_t(N));
    EXPECT_TRUE(withinBand(teamed, serial, std::size_t(N)));
}

TEST(ExecContractTest, CrossSampleWriteAccumulatesWholeView)
{
    // The shipped capability the contract CLASSIFIES rather than refuses (L3):
    // a single-structure, single-partition cross-sample accumulate.
    constexpr std::int64_t N = 64;
    std::vector<double> x(static_cast<std::size_t>(N)), acc(4, 0.0);
    for (std::int64_t i = 0; i < N; ++i) x[std::size_t(i)] = double(i);
    const Sidecar sc = v2Sidecar("exec_scatter", kScatterArgs);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO));
    reg.bind_handle("x", x.data());
    reg.bind_handle("acc", acc.data());
    ASSERT_EQ(reg.run(std::int32_t(N)), 1);
    double want[4] = { 0, 0, 0, 0 };
    for (std::int64_t i = 0; i < N; ++i) want[i % 4] += double(i);
    for (int l = 0; l < 4; ++l) EXPECT_DOUBLE_EQ(acc[std::size_t(l)], want[l]);
}

TEST(ExecContractTest, HostTeamTileSizeFollowsTheHostLaunchRule)
{
    // The tile size is `Host::launch`'s own L2-derived rule, not a new one.
    EXPECT_EQ(ex::HostTeam::tile_size(0), aether::optimalTileSize<double>(0));
    EXPECT_EQ(ex::HostTeam::tile_size(512), aether::optimalTileSize<double>(512));
    EXPECT_EQ(ex::HostTeam::tile_count(ex::Partition::whole(0), 0), 0u);
    const std::size_t tile = ex::HostTeam::tile_size(0);
    EXPECT_EQ(ex::HostTeam::tile_count(ex::Partition::whole(std::int64_t(tile)), 0), 1u);
    EXPECT_EQ(ex::HostTeam::tile_count(ex::Partition::whole(std::int64_t(tile) + 1), 0), 2u);
}

TEST(ExecContractTest, EmptyPartitionRunsNothing)
{
    const Sidecar sc = v2Sidecar("exec_triple", kTripleArgs);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO));
    std::vector<double> t(3, -7.0);
    reg.bind_handle("t", t.data());
    EXPECT_EQ(ex::HostTeam::run(reg.entry_v2(0), reg.pack(0, 10).params,
                                ex::Partition{ 0, 0, 10 }, 0), 0u);
    EXPECT_DOUBLE_EQ(t[0], -7.0);  // untouched
}

}  // namespace ExecContractTest
}  // namespace eagle_tests
