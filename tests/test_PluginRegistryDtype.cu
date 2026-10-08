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

// The CUDA-mode PluginRegistry's float32 rejection is
// CORRECT (this registry's entire binding surface — bind_vector/
// bind_uniform/consolidate — is double-typed, and a float32 kernel genuinely
// needs float-typed buffers and 4-byte uniform packing this registry does
// not have) but used to carry the CPU-host registry's error text verbatim:
// "not supported by the C++ host runner; deploy for the CUDA backend"
// (backwards advice on the CUDA path itself). This test pins the corrected
// message by substring so a future edit cannot silently regress back to the
// dead-end wording. It does not touch a live CUDA context:
// PluginRegistry::from_manifest throws before any Driver-API call (before
// cuModuleLoadData), so this is pure host-side JSON validation exercised
// through the real class, not a device smoke test.
//
// softdouble: the load-and-run probe the comment above asked for. SoftDouble
// is bit-identical IEEE float64 (a static_assert(sizeof(SoftDouble) == 8)
// enforces it), so it loads and launches through
// this registry's double-typed binding surface with zero widening — the
// rejection this file used to check was not backed by the float32 kind of
// "wrong buffer layout" reason and has been lifted in registry.h.
// PluginRegistryDtypeTest.SoftdoubleLoadsAndLaunches (below) pins that: a
// hand-written, ABI-only device fixture (fixtures/device_plugin_softdouble.cu
// — SoftDouble IS `double` at the wire, so a raw-double kernel is a faithful
// stand-in; the registry itself never inspects the kernel's own arithmetic,
// only the sidecar's `scalar_type` tag) tagged `scalar_type=softdouble` loads
// via cuModuleLoadData, resolves its `raptor_kernel` symbol, and launches
// through PluginRegistry::inject unmodified.
using namespace eagle::plugin;

namespace eagle_tests {
namespace PluginRegistryDtypeTest {

static std::string writeTmp(const std::string& name, const std::string& body)
{
    const std::string path = ::testing::TempDir() + name;
    std::ofstream f(path, std::ios::binary);
    f << body;
    f.close();
    return path;
}

// Writes a manifest + one sidecar (both in TempDir()) declaring `scalar_type`,
// referencing an artifact file that is deliberately never created — the
// scalar_type reject fires before from_manifest ever reads the artifact.
static std::string writeManifestWithScalarType(
    const std::string& tag, const std::string& scalarType)
{
    const std::string sidecarName = tag + "_sidecar.json";
    const std::string sidecarJson = R"({
  "kernel": "raptor_kernel",
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "scalar_type": ")" + scalarType + R"(",
  "arg_spec": [["out", "out"], ["terminated", "terminated"]]
})";
    writeTmp(sidecarName, sidecarJson);

    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "plugins": [
    {"id": ")" + tag + R"(", "order": 0, "enabled": true,
     "artifact": ")" + tag + R"(.ptx", "sidecar": ")" + sidecarName + R"(",
     "format": "ptx"}
  ]
})";
    return writeTmp(tag + "_manifest.json", manifestJson);
}

TEST(PluginRegistryDtypeTest, RejectsFloat32NamingRealReasonAndAlternative)
{
    const std::string manifestPath = writeManifestWithScalarType("f32", "float32");
    try {
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
        FAIL() << "expected from_manifest to reject the float32 plugin";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("binds float64 buffers only"), std::string::npos) << msg;
        EXPECT_NE(msg.find("'float32'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("Python/torch device path"), std::string::npos) << msg;
        EXPECT_NE(msg.find("eagle.LoadedVector"), std::string::npos) << msg;
        // The dead-end wording this replaced must be gone.
        EXPECT_EQ(msg.find("deploy for the CUDA backend"), std::string::npos) << msg;
        EXPECT_EQ(msg.find("not supported by the C++ host runner"), std::string::npos) << msg;
    }
}

// Stages a manifest + sidecar (both in TempDir()) around the CMake-built
// device_plugin_softdouble.cu fixture (path baked in as
// DEVICE_PLUGIN_SOFTDOUBLE_PTX): copies its compiled PTX in beside a sidecar
// declaring scalar_type=softdouble with the fixture's real arg_spec
// (`mutable val`, `terminated`, `uniform step`, `nsamples`) — unlike
// writeManifestWithScalarType above, the artifact here is real, so
// from_manifest runs all the way through cuModuleLoadData /
// cuModuleGetFunction, not just the metadata reject.
static std::string writeSoftdoubleManifest()
{
    const std::string ptxBytes = [] {
        std::ifstream f(DEVICE_PLUGIN_SOFTDOUBLE_PTX, std::ios::binary);
        std::ostringstream ss; ss << f.rdbuf();
        return ss.str();
    }();
    writeTmp("sd.ptx", ptxBytes);

    const std::string sidecarJson = R"({
  "kernel": "raptor_kernel",
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "scalar_type": "softdouble",
  "arg_spec": [["mutable", "val"], ["terminated", "terminated"],
               ["uniform", "step"], ["nsamples", "nsamples"]]
})";
    writeTmp("sd_sidecar.json", sidecarJson);

    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "plugins": [
    {"id": "sd", "order": 0, "enabled": true,
     "artifact": "sd.ptx", "sidecar": "sd_sidecar.json", "format": "ptx"}
  ]
})";
    return writeTmp("sd_manifest.json", manifestJson);
}

static void cuCheck(CUresult r, const char* what)
{
    if (r != CUDA_SUCCESS) {
        const char* m = nullptr; cuGetErrorString(r, &m);
        FAIL() << what << " failed: " << (m ? m : "?");
    }
}

TEST(PluginRegistryDtypeTest, SoftdoubleLoadsAndLaunches)
{
    // Same primary-context bootstrap as pure_inject_demo.cu: the Runtime
    // creates the context, the Driver API (from_manifest / inject) shares it.
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

    const std::string manifestPath = writeSoftdoubleManifest();

    // The load itself: this is the guarded call — it used to throw here.
    eagle::cuda::PluginRegistry registry =
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
    ASSERT_EQ(registry.size(), 1u);
    EXPECT_EQ(registry.active(), 1);

    constexpr int N = 1024;
    constexpr double step = 0.125;

    std::vector<double> val0(N);
    for (int i = 0; i < N; ++i) val0[i] = double(i) * 0.01 - 5.0;

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

    // The launch itself (eager, no graph capture -- this check is about
    // load+launch, not the graph-injection path already covered by
    // pure_inject_demo.cu). Stream 0 (the default stream) is a valid CUstream.
    const int launched = registry.inject(CUstream(nullptr), N);
    EXPECT_EQ(launched, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> got(N);
    ASSERT_EQ(cudaMemcpy(got.data(), d_val, N * sizeof(double),
                          cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_val);
    cudaFree(d_term);

    // The kernel really ran: val[i] == val0[i] + step for every sample.
    for (int i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(got[i], val0[i] + step) << "sample " << i;
}

}  // namespace PluginRegistryDtypeTest
}  // namespace eagle_tests
