// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "plugin/plugin_registry/registry.h"

#include "TestBase.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

// v0.2.3 merge-door coverage — eagle::cuda::PluginRegistry::add_plugin (called on an
// ALREADY-POPULATED registry, i.e. the merge path) and set_plugin_enabled. Both shipped
// in 0.2.3 (registry.h's "Exactly `add_plugin` on a fresh registry ... is ONE
// implementation" note, and the `set_plugin_enabled` doc above its definition) but, until
// this file, neither was exercised by ANY eagle test: test_PluginRegistryManifest.cu's
// merge-shaped tests (RejectsDuplicatePluginId*) all throw before touching the Driver API,
// so the actual append-order / staged-then-spliced machinery in `add_plugin`'s bottom half
// was never run.
//
// Reuses the two device-plugin PTX fixtures already built for
// test_PluginRegistryDtype.cu (DEVICE_PLUGIN_SOFTDOUBLE_PTX: `val[i] += step`) and
// test_PluginRegistryUniformInt.cu (DEVICE_PLUGIN_INTUNIFORM_PTX: `val[i] = val[i]*gain +
// k`) — no new .cu fixture needed. Both kernels' `mutable val` binds to the SAME name
// ("val"), so merging one of each into one registry and injecting them lets a single
// device buffer's final value witness which plugin ran, in which order — append order is
// asserted PHYSICALLY (via the composed arithmetic), not just via `size()`.
using namespace eagle::plugin;

namespace eagle_tests {
namespace PluginRegistryMergeTest {

static std::string writeTmp(const std::string& name, const std::string& body)
{
    const std::string path = ::testing::TempDir() + name;
    std::ofstream f(path, std::ios::binary);
    f << body;
    f.close();
    return path;
}

static std::string readFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

static void cuCheck(CUresult r, const char* what)
{
    if (r != CUDA_SUCCESS) {
        const char* m = nullptr; cuGetErrorString(r, &m);
        FAIL() << what << " failed: " << (m ? m : "?");
    }
}

// Same primary-context bootstrap as test_PluginRegistryDtype.cu /
// test_PluginRegistryUniformInt.cu: the Runtime creates the context, the Driver API
// (from_manifest / add_plugin / inject) shares it.
static void ensurePrimaryContext()
{
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    ASSERT_EQ(cudaFree(0), cudaSuccess);
    cuCheck(cuInit(0), "cuInit");
    CUcontext ctx = nullptr;
    cuCheck(cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
    if (ctx == nullptr) {
        CUdevice d;
        cuCheck(cuDeviceGet(&d, 0), "cuDeviceGet");
        cuCheck(cuDevicePrimaryCtxRetain(&ctx, d), "cuDevicePrimaryCtxRetain");
        cuCheck(cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
    }
}

// Stages a copy of the softdouble fixture's PTX + a matching sidecar under
// `tag`-prefixed filenames (both in TempDir(), like every other fixture in this
// binary), so several tests below can each use their OWN filenames and never collide
// with test_PluginRegistryDtype.cu's hardcoded "sd.ptx" / "sd_sidecar.json", nor with
// each other. Returns the two filenames (artifact, sidecar), relative to TempDir() —
// exactly what a manifest entry needs.
static std::pair<std::string, std::string> stageSoftdoubleFixture(const std::string& tag)
{
    const std::string artifact = tag + ".ptx";
    const std::string sidecar = tag + "_sidecar.json";
    writeTmp(artifact, readFile(DEVICE_PLUGIN_SOFTDOUBLE_PTX));
    const std::string sidecarJson = R"({
  "kernel": "raptor_kernel",
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "scalar_type": "softdouble",
  "arg_spec": [["mutable", "val"], ["terminated", "terminated"],
               ["uniform", "step"], ["nsamples", "nsamples"]]
})";
    writeTmp(sidecar, sidecarJson);
    return {artifact, sidecar};
}

// The int-uniform fixture's twin: `val[i] = val[i]*gain + k`.
static std::pair<std::string, std::string> stageIntUniformFixture(const std::string& tag)
{
    const std::string artifact = tag + ".ptx";
    const std::string sidecar = tag + "_sidecar.json";
    writeTmp(artifact, readFile(DEVICE_PLUGIN_INTUNIFORM_PTX));
    const std::string sidecarJson = R"({
  "kernel": "raptor_kernel",
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "scalar_type": "float64",
  "arg_spec": [["mutable", "val"], ["uniform", "gain"],
               ["uniform", "k"], ["nsamples", "nsamples"]]
})";
    writeTmp(sidecar, sidecarJson);
    return {artifact, sidecar};
}

struct StagedEntry {
    std::string id;
    std::string artifact;  // filename relative to the manifest's own directory
    std::string sidecar;   // ditto — may deliberately name a file never written
    std::string format = "ptx";
};

// Assembles + writes a manifest.json (in TempDir(), like every fixture manifest in
// this binary) naming `entries` in order — the manifest's declared (== injection ==
// merge) order.
static std::string writeManifest(const std::string& manifestTag,
                                  const std::vector<StagedEntry>& entries)
{
    std::string plugins;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        const auto& e = entries[i];
        plugins += "    {\"id\": \"" + e.id + "\", \"order\": " + std::to_string(i) +
            ", \"enabled\": true, \"artifact\": \"" + e.artifact +
            "\", \"sidecar\": \"" + e.sidecar + "\", \"format\": \"" + e.format + "\"}";
        plugins += (i + 1 < entries.size()) ? ",\n" : "\n";
    }
    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "plugins": [
)" + plugins + R"(  ]
})";
    return writeTmp(manifestTag + "_manifest.json", manifestJson);
}

// ----------------------------------------------------------------------- //
// 1) size()/active()/order = append order, verified PHYSICALLY: from_manifest
// loads "sd" (val += step) first, add_plugin merges "iu" (val = val*gain + k)
// second, and one inject() must run them in exactly that sequence.
// ----------------------------------------------------------------------- //
TEST(PluginRegistryMergeTest, MergeAppendsInOrderAndComposesSequentially)
{
    ensurePrimaryContext();

    const auto sd = stageSoftdoubleFixture("merge_order_sd");
    const std::string manifestA = writeManifest("merge_order_a",
        {{"sd", sd.first, sd.second}});

    eagle::cuda::PluginRegistry registry =
        eagle::cuda::PluginRegistry::from_manifest(manifestA);
    ASSERT_EQ(registry.size(), 1u);

    const auto iu = stageIntUniformFixture("merge_order_iu");
    const std::string manifestB = writeManifest("merge_order_b",
        {{"iu", iu.first, iu.second}});
    registry.add_plugin(manifestB);

    ASSERT_EQ(registry.size(), 2u);
    EXPECT_EQ(registry.active(), 2);
    EXPECT_TRUE(registry.will_launch(0));
    EXPECT_TRUE(registry.will_launch(1));

    constexpr int N = 1024;
    constexpr double step = 0.125;
    constexpr double gain = 0.25;
    const UniformInt k = 7;

    std::vector<double> val0(N);
    for (int i = 0; i < N; ++i) val0[i] = double(i) * 0.5 - 3.0;

    double* d_val = nullptr;
    unsigned char* d_term = nullptr;
    ASSERT_EQ(cudaMalloc(&d_val, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_term, N), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_val, val0.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_term, 0, N), cudaSuccess);

    registry.bind_handle("val", d_val);
    registry.bind_handle("terminated", d_term);
    registry.bind_uniform("step", step);
    registry.bind_uniform("gain", gain);
    registry.bind_uniform_int("k", k);

    const int launched = registry.inject(CUstream(nullptr), N);
    EXPECT_EQ(launched, 2);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> got(N);
    ASSERT_EQ(cudaMemcpy(got.data(), d_val, N * sizeof(double),
                          cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_val);
    cudaFree(d_term);

    // If merge order were reversed (iu before sd) this would instead be
    // (val0*gain+k)+step — a different number for every sample, so the check is
    // discriminating, not incidentally satisfied either way.
    for (int i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(got[i], (val0[i] + step) * gain + double(k))
            << "sample " << i;
}

// ----------------------------------------------------------------------- //
// 2) Cross-manifest id collision: merging a manifest whose id is ALREADY loaded
// (from an earlier, distinct manifest) is refused with the merge-specific message,
// and the refusal touches nothing — the registry is exactly as it was.
// ----------------------------------------------------------------------- //
TEST(PluginRegistryMergeTest, CrossManifestIdCollisionRefused)
{
    ensurePrimaryContext();

    const auto sd = stageSoftdoubleFixture("merge_dup_sd_first");
    const std::string manifestA = writeManifest("merge_dup_a",
        {{"sd", sd.first, sd.second}});
    eagle::cuda::PluginRegistry registry =
        eagle::cuda::PluginRegistry::from_manifest(manifestA);
    ASSERT_EQ(registry.size(), 1u);

    // A second, distinct manifest that happens to reuse the id "sd". Its artifact/
    // sidecar are never created — the collision check fires before either is read,
    // exactly like test_PluginRegistryManifest.cu's within-manifest duplicate pins.
    const std::string manifestC = writeManifest("merge_dup_c",
        {{"sd", "merge_dup_c_missing.ptx", "merge_dup_c_missing.json"}});

    try {
        registry.add_plugin(manifestC);
        FAIL() << "expected add_plugin to reject the cross-manifest id collision";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("already loaded in this registry"), std::string::npos) << msg;
        EXPECT_NE(msg.find("'sd'"), std::string::npos) << msg;
    }

    // Untouched: still exactly the one plugin manifestA loaded.
    EXPECT_EQ(registry.size(), 1u);
    EXPECT_TRUE(registry.will_launch(0));
}

// ----------------------------------------------------------------------- //
// 3) set_plugin_enabled(index, on) is per-plugin (registry/merge order), and
// toggling one index never moves the other. Proven both via will_launch()/active()
// and, end to end, via which kernel(s) actually ran.
// ----------------------------------------------------------------------- //
TEST(PluginRegistryMergeTest, SetPluginEnabledTogglesExactlyOnePlugin)
{
    ensurePrimaryContext();

    const auto sd = stageSoftdoubleFixture("merge_toggle_sd");
    const std::string manifestA = writeManifest("merge_toggle_a",
        {{"sd", sd.first, sd.second}});
    eagle::cuda::PluginRegistry registry =
        eagle::cuda::PluginRegistry::from_manifest(manifestA);

    const auto iu = stageIntUniformFixture("merge_toggle_iu");
    const std::string manifestB = writeManifest("merge_toggle_b",
        {{"iu", iu.first, iu.second}});
    registry.add_plugin(manifestB);
    ASSERT_EQ(registry.size(), 2u);
    ASSERT_EQ(registry.active(), 2);

    constexpr int N = 256;
    constexpr double step = 0.125;
    constexpr double gain = 0.25;
    const UniformInt k = 7;

    std::vector<double> val0(N);
    for (int i = 0; i < N; ++i) val0[i] = double(i) * 0.5 - 3.0;

    double* d_val = nullptr;
    unsigned char* d_term = nullptr;
    ASSERT_EQ(cudaMalloc(&d_val, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_term, N), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_term, 0, N), cudaSuccess);
    registry.bind_handle("val", d_val);
    registry.bind_handle("terminated", d_term);
    registry.bind_uniform("step", step);
    registry.bind_uniform("gain", gain);
    registry.bind_uniform_int("k", k);

    auto resetVal = [&] {
        ASSERT_EQ(cudaMemcpy(d_val, val0.data(), N * sizeof(double),
                              cudaMemcpyHostToDevice), cudaSuccess);
    };
    auto readVal = [&](std::vector<double>& out) {
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(out.data(), d_val, N * sizeof(double),
                              cudaMemcpyDeviceToHost), cudaSuccess);
    };

    // Disable index 0 ("sd"): index 1 ("iu") is unaffected.
    registry.set_plugin_enabled(0, false);
    EXPECT_FALSE(registry.will_launch(0));
    EXPECT_TRUE(registry.will_launch(1));
    EXPECT_EQ(registry.active(), 1);

    resetVal();
    EXPECT_EQ(registry.inject(CUstream(nullptr), N), 1);
    std::vector<double> got(N);
    readVal(got);
    for (int i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(got[i], val0[i] * gain + double(k)) << "sample " << i;

    // Re-enable index 0: both plugins launch again, in the original order.
    registry.set_plugin_enabled(0, true);
    EXPECT_TRUE(registry.will_launch(0));
    EXPECT_TRUE(registry.will_launch(1));
    EXPECT_EQ(registry.active(), 2);

    resetVal();
    EXPECT_EQ(registry.inject(CUstream(nullptr), N), 2);
    readVal(got);
    for (int i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(got[i], (val0[i] + step) * gain + double(k))
            << "sample " << i;

    // Disable index 1 ("iu") instead: index 0 ("sd") is unaffected — the toggle is
    // symmetric, not merely "index 0 happens to work".
    registry.set_plugin_enabled(1, false);
    EXPECT_TRUE(registry.will_launch(0));
    EXPECT_FALSE(registry.will_launch(1));
    EXPECT_EQ(registry.active(), 1);

    resetVal();
    EXPECT_EQ(registry.inject(CUstream(nullptr), N), 1);
    readVal(got);
    for (int i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(got[i], val0[i] + step) << "sample " << i;

    cudaFree(d_val);
    cudaFree(d_term);
}

// ----------------------------------------------------------------------- //
// 4) Transactional guarantee: a manifest merged SECOND that fails partway through
// its own plugins[] leaves the registry exactly as it was before the call — even
// though its FIRST entry loaded successfully (staged, then never spliced in,
// per registry.h's "STAGED, then spliced" comment on add_plugin).
// ----------------------------------------------------------------------- //
TEST(PluginRegistryMergeTest, FailingMergeLeavesRegistryUnchanged)
{
    ensurePrimaryContext();

    const auto sd = stageSoftdoubleFixture("merge_txn_sd");
    const std::string manifestA = writeManifest("merge_txn_a",
        {{"sd", sd.first, sd.second}});
    eagle::cuda::PluginRegistry registry =
        eagle::cuda::PluginRegistry::from_manifest(manifestA);
    ASSERT_EQ(registry.size(), 1u);

    // manifestD has TWO entries: the first ("iu2") is a real, loadable plugin (it
    // would succeed if merged alone — reaches cuModuleLoadData/cuModuleGetFunction
    // and gets staged); the second ("broken") names a sidecar file that is never
    // created, so parse_sidecar throws mid-loop, AFTER "iu2" already loaded.
    const auto iu2 = stageIntUniformFixture("merge_txn_iu2");
    const std::string manifestD = writeManifest("merge_txn_d", {
        {"iu2", iu2.first, iu2.second},
        {"broken", "merge_txn_broken.ptx", "merge_txn_broken_MISSING.json"},
    });

    EXPECT_THROW(registry.add_plugin(manifestD), std::runtime_error);

    // Neither "iu2" (which loaded fine) nor "broken" made it in: still exactly the
    // one plugin manifestA contributed.
    ASSERT_EQ(registry.size(), 1u);
    EXPECT_EQ(registry.active(), 1);
    EXPECT_TRUE(registry.will_launch(0));

    // "sd" (manifestA's plugin) still works end to end after the failed merge.
    constexpr int N = 128;
    constexpr double step = 0.125;
    std::vector<double> val0(N);
    for (int i = 0; i < N; ++i) val0[i] = double(i) * 0.5 - 3.0;

    double* d_val = nullptr;
    unsigned char* d_term = nullptr;
    ASSERT_EQ(cudaMalloc(&d_val, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_term, N), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_val, val0.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_term, 0, N), cudaSuccess);
    registry.bind_handle("val", d_val);
    registry.bind_handle("terminated", d_term);
    registry.bind_uniform("step", step);

    EXPECT_EQ(registry.inject(CUstream(nullptr), N), 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<double> got(N);
    ASSERT_EQ(cudaMemcpy(got.data(), d_val, N * sizeof(double),
                          cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_val);
    cudaFree(d_term);

    for (int i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(got[i], val0[i] + step) << "sample " << i;
}

}  // namespace PluginRegistryMergeTest
}  // namespace eagle_tests
