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

// The int-role phase — the device PluginRegistry's
// ADDITIVE int64 uniform path, beside the float64 one. The CUDA-mode twin of
// test_HostPluginUniformInt.cpp; the two registries share the binding vocabulary and
// the refusal wording (plugin/plugin_registry/uniform_binding.h), so the pins here are
// deliberately the SAME shape as the host ones.
//
// THE WIDTH. A generated kernel's integer quantity is the code generator's
// ``using Int = long long;``, so an int uniform's
// parameter is ``AETHER_GRID_CONSTANT() Int p_<name>`` — 8 SIGNED bytes, which is what
// ``eagle::plugin::UniformInt`` is. The launch check below carries 2**32 + 1 through the
// slot precisely so a 32-bit packing (what a 32-bit ``idx_t`` would have given) fails it
// by truncation rather than passing on a small value.
//
// The two refusal pins need NO CUDA context: the binding maps are registry-level and
// the guard fires before anything is recorded — the same "pure host-side validation
// exercised through the real class" as test_PluginRegistryDtype.cu's float32 check.
using namespace eagle::plugin;

namespace eagle_tests {
namespace PluginRegistryUniformIntTest {

// REFUSAL DIRECTION 1 — int64 over an existing float64 binding of the same name. A
// uniform occupies ONE by-value parameter slot, so two bindings of one name would
// leave the packer to guess which 8 bytes the kernel meant.
TEST(PluginRegistryUniformIntTest, RefusesIntBindingOverAFloat64Uniform)
{
    eagle::cuda::PluginRegistry reg;
    reg.bind_uniform("k", 1.5);
    try {
        reg.bind_uniform_int("k", 7);
        FAIL() << "expected bind_uniform_int to refuse the double-typed name";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("uniform 'k'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("already bound as float64"), std::string::npos) << msg;
        EXPECT_NE(msg.find("cannot also be bound as int64"), std::string::npos) << msg;
        EXPECT_NE(msg.find("bind_uniform_int"), std::string::npos) << msg;
    }
}

// REFUSAL DIRECTION 2 — float64 over an existing int64 binding. The mirror image; a
// one-directional guard is exactly the kind that lets the other migration order past.
TEST(PluginRegistryUniformIntTest, RefusesFloat64BindingOverAnIntUniform)
{
    eagle::cuda::PluginRegistry reg;
    reg.bind_uniform_int("k", 7);
    try {
        reg.bind_uniform("k", 1.5);
        FAIL() << "expected bind_uniform to refuse the int-typed name";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("uniform 'k'"), std::string::npos) << msg;
        EXPECT_NE(msg.find("already bound as int64"), std::string::npos) << msg;
        EXPECT_NE(msg.find("cannot also be bound as float64"), std::string::npos) << msg;
        EXPECT_NE(msg.find("bind_uniform"), std::string::npos) << msg;
    }
}

// The refusal is name-scoped: two DIFFERENT names may take different kinds (which is
// what a mixed kernel needs), and re-binding the SAME kind still overwrites, exactly as
// bind_uniform always has.
TEST(PluginRegistryUniformIntTest, DifferentNamesMayTakeDifferentKinds)
{
    eagle::cuda::PluginRegistry reg;
    reg.bind_uniform("gain", 0.25);
    reg.bind_uniform_int("k", 11);
    reg.bind_uniform("gain", 0.5);
    reg.bind_uniform_int("k", 12);
    SUCCEED();
}

static std::string writeTmp(const std::string& name, const std::string& body)
{
    const std::string path = ::testing::TempDir() + name;
    std::ofstream f(path, std::ios::binary);
    f << body;
    f.close();
    return path;
}

// Stages a manifest + sidecar (both in TempDir()) around the CMake-built
// device_plugin_intuniform.cu fixture (path baked in as DEVICE_PLUGIN_INTUNIFORM_PTX),
// whose kernel takes `(ScalarHandle val, double gain, long long k, uint32 n)` — the
// MIXED row set. Same staging recipe as test_PluginRegistryDtype.cu's softdouble check.
static std::string writeIntUniformManifest()
{
    const std::string ptxBytes = [] {
        std::ifstream f(DEVICE_PLUGIN_INTUNIFORM_PTX, std::ios::binary);
        std::ostringstream ss; ss << f.rdbuf();
        return ss.str();
    }();
    writeTmp("iu.ptx", ptxBytes);

    const std::string sidecarJson = R"({
  "kernel": "raptor_kernel",
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "scalar_type": "float64",
  "arg_spec": [["mutable", "val"], ["uniform", "gain"],
               ["uniform", "k"], ["nsamples", "nsamples"]]
})";
    writeTmp("iu_sidecar.json", sidecarJson);

    const std::string manifestJson = R"({
  "aether_abi": "aether-abi/1",
  "schema_version": 1,
  "plugins": [
    {"id": "iu", "order": 0, "enabled": true,
     "artifact": "iu.ptx", "sidecar": "iu_sidecar.json", "format": "ptx"}
  ]
})";
    return writeTmp("iu_manifest.json", manifestJson);
}

static void cuCheck(CUresult r, const char* what)
{
    if (r != CUDA_SUCCESS) {
        const char* m = nullptr; cuGetErrorString(r, &m);
        FAIL() << what << " failed: " << (m ? m : "?");
    }
}

// The end-to-end device check: a kernel with BOTH a float64 and an int64 uniform loads,
// binds through the two binders, and launches through inject() — each kind landing in
// its own by-value parameter slot, in arg_spec order. The int value is 2**32 + 1, so a
// 32-bit packing would deliver 1 and fail this by ~4.3e9 rather than pass by luck.
TEST(PluginRegistryUniformIntTest, MixedFloatAndIntUniformsLoadAndLaunch)
{
    // Same primary-context bootstrap as test_PluginRegistryDtype.cu: the Runtime
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

    const std::string manifestPath = writeIntUniformManifest();
    eagle::cuda::PluginRegistry registry =
        eagle::cuda::PluginRegistry::from_manifest(manifestPath);
    ASSERT_EQ(registry.size(), 1u);

    constexpr int N = 1024;
    constexpr double gain = 0.25;
    const UniformInt k = (UniformInt(1) << 32) + 1;  // 4294967297

    std::vector<double> val0(N);
    for (int i = 0; i < N; ++i) val0[i] = double(i) * 0.5;

    double* d_val = nullptr;
    ASSERT_EQ(cudaMalloc(&d_val, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_val, val0.data(), N * sizeof(double),
                          cudaMemcpyHostToDevice), cudaSuccess);

    registry.bind_handle("val", d_val);
    registry.bind_uniform("gain", gain);
    registry.bind_uniform_int("k", k);

    const int launched = registry.inject(CUstream(nullptr), N);
    EXPECT_EQ(launched, 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> got(N);
    ASSERT_EQ(cudaMemcpy(got.data(), d_val, N * sizeof(double),
                          cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_val);

    for (int i = 0; i < N; ++i)
        EXPECT_DOUBLE_EQ(got[i], val0[i] * gain + double(k)) << "sample " << i;
}

}  // namespace PluginRegistryUniformIntTest
}  // namespace eagle_tests
