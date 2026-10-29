// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// The HETEROGENEOUS EXECUTION CONTRACT — DEVICE rows (aether-abi/2).
//
// The CUDA face of test_ExecContract.cpp: the PTX layout self-check
// (cuModuleGetGlobal), the device launch triple, partition
// identity under `eagle::exec::DeviceKernel`, the mapreduce band, and the
// host/device TWIN — the one row that needs both arms in one process and is
// therefore carried by the CUDA gate alone.
//
// FIXTURES (ABI-only, nvcc'd to raw PTX at configure time):
// fixtures/device_plugin_execv2.cu (the v2 pair's device half — body for body the
// twin of fixtures/host_plugin_execv2.cpp) and fixtures/device_plugin_badlayout.cu
// (the wrong-sizes arm). The MISSING-symbol arm reuses the existing
// fixtures/device_plugin_softdouble.cu PTX, which predates the self-check and
// therefore exports nothing.

#include "eagle/exec/DeviceKernel.h"
#include "eagle/exec/HostTeam.h"
#include "plugin/host_registry.h"
#include "plugin/plugin_registry/registry.h"

#include "TestBase.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using namespace eagle::plugin;
namespace ex = eagle::exec;

namespace eagle_tests {
namespace ExecContractDeviceTest {

static std::string writeTmp(const std::string& name, const std::string& body)
{
    const std::string path = ::testing::TempDir() + name;
    std::ofstream f(path, std::ios::binary);
    f << body;
    f.close();
    return path;
}

static std::string slurp(const char* path)
{
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

// Stage `ptxSource` beside a sidecar + manifest naming `kernel`, and return the
// manifest path. Everything lands in TempDir(), so the manifest's own directory
// resolves both relative names.
static std::string stage(const std::string& tag, const char* ptxSource,
                         const std::string& kernel, const std::string& argSpec,
                         const std::string& abi, int schemaVersion,
                         const std::string& execKeys)
{
    writeTmp(tag + ".ptx", slurp(ptxSource));
    const std::string sidecar = R"({
  "kernel": ")" + kernel + R"(",
  "aether_abi": ")" + abi + R"(",
  "schema_version": )" + std::to_string(schemaVersion) + R"(,
  "pattern": "pure",
  "scalar_type": "float64",
  "arg_spec": )" + argSpec + R"(
})";
    writeTmp(tag + "_sidecar.json", sidecar);
    const std::string manifest = R"({
  "schema_version": )" + std::to_string(schemaVersion) + R"(,
  "pattern": "pure",
  "aether_abi": ")" + abi + R"(",
)" + execKeys + R"(  "plugins": [
    {"id": ")" + tag + R"(", "order": 0, "enabled": true,
     "artifact": ")" + tag + R"(.ptx", "sidecar": ")" + tag + R"(_sidecar.json",
     "format": "ptx"}
  ]
})";
    return writeTmp(tag + "_manifest.json", manifest);
}

static const char* kLocalArgs =
    R"([["mutable","y"],["per_sample","x"],["uniform","a"],["uniform","b"],["nsamples","nsamples"]])";
static const char* kMapreduceArgs =
    R"([["accum_out","partial"],["per_sample","x"],["nsamples","nsamples"]])";
static const char* kWideArgs =
    R"([["wide_out","wout"],["wide_in","win"],["accum_out","aout"],["nsamples","nsamples"]])";
static const char* kTripleArgs = R"([["wide_out","t"],["nsamples","nsamples"]])";
static const char* kLegacyArgs =
    R"([["mutable","y"],["per_sample","x"],["nsamples","nsamples"]])";
static const char* kSampleLocalKeys =
    "  \"exec_targets\": [\"device\", \"host\"],\n  \"exec_access\": \"sample_local\",\n";
static const char* kMapreduceKeys =
    "  \"exec_targets\": [\"device\", \"host\"],\n  \"exec_access\": \"mapreduce\",\n"
    "  \"exec_op\": \"sum\",\n";

static void cuCheck(CUresult r, const char* what)
{
    if (r != CUDA_SUCCESS) {
        const char* m = nullptr; cuGetErrorString(r, &m);
        FAIL() << what << " failed: " << (m ? m : "?");
    }
}

// The same primary-context bootstrap the other device-registry tests use.
static void ensureContext()
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
// The layout self-check, device face
// ---------------------------------------------------------------------------
TEST(ExecContractDeviceTest, DeviceV2PluginWithWrongLayoutIsRefusedNamingTheField)
{
    ensureContext();
    const std::string path = stage("dbad", DEVICE_PLUGIN_BADLAYOUT_PTX, "exec_bad",
        R"([["mutable","y"],["nsamples","nsamples"]])", "aether-abi/2", 2,
        kSampleLocalKeys);
    try {
        eagle::cuda::PluginRegistry::from_manifest(path);
        FAIL() << "expected the wrong-layout PTX module to be refused";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("sizeof(partition triple)"), std::string::npos) << msg;
        EXPECT_NE(msg.find("16"), std::string::npos) << msg;
        EXPECT_NE(msg.find("24"), std::string::npos) << msg;
    }
}

TEST(ExecContractDeviceTest, DeviceV2PluginWithoutTheLayoutSymbolIsRefused)
{
    ensureContext();
    // The pre-self-check softdouble fixture, re-tagged v2: it exports nothing.
    const std::string path = stage("dnolayout", DEVICE_PLUGIN_SOFTDOUBLE_PTX,
        "raptor_kernel",
        R"([["mutable","val"],["terminated","terminated"],["uniform","step"],["nsamples","nsamples"]])",
        "aether-abi/2", 2, kSampleLocalKeys);
    try {
        eagle::cuda::PluginRegistry::from_manifest(path);
        FAIL() << "expected a v2 module with no layout symbol to be refused";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("eagle_layout_sizes"), std::string::npos)
            << e.what();
    }
}

TEST(ExecContractDeviceTest, DeviceManifestSchemaBeyondMaxIsRefused)
{
    ensureContext();
    const std::string path = stage("dv3", DEVICE_PLUGIN_EXECV2_PTX, "exec_local",
        kLocalArgs, "aether-abi/2", kPluginMaxSchemaVersion + 1, kSampleLocalKeys);
    try {
        eagle::cuda::PluginRegistry::from_manifest(path);
        FAIL() << "expected a schema v3 manifest to be refused";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("upgrade eagle"), std::string::npos)
            << e.what();
    }
}

TEST(ExecContractDeviceTest, DeviceManifestSchemaTwoWithoutTheExecutionAxisIsRefused)
{
    ensureContext();
    const std::string path = stage("dnoaxis", DEVICE_PLUGIN_EXECV2_PTX, "exec_local",
        kLocalArgs, "aether-abi/2", 2, "");
    try {
        eagle::cuda::PluginRegistry::from_manifest(path);
        FAIL() << "expected a v2 manifest with no execution axis to be refused";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("absence = load refused"),
                  std::string::npos) << e.what();
    }
}

// ---------------------------------------------------------------------------
// The wide/accum roles launch on the device too
// ---------------------------------------------------------------------------
TEST(ExecContractDeviceTest, DeviceRegistryLaunchesTheWideAndAccumRoles)
{
    ensureContext();
    constexpr int N = 256;
    const std::string path = stage("dwide", DEVICE_PLUGIN_EXECV2_PTX, "exec_wide",
        kWideArgs, "aether-abi/2", 2, kSampleLocalKeys);
    eagle::cuda::PluginRegistry reg =
        eagle::cuda::PluginRegistry::from_manifest(path);
    ASSERT_EQ(reg.size(), 1u);
    EXPECT_EQ(reg.abi_version(0), 2);

    std::vector<double> win(N);
    for (int i = 0; i < N; ++i) win[std::size_t(i)] = double(i) * 0.25;
    double *d_win = nullptr, *d_wout = nullptr, *d_aout = nullptr;
    ASSERT_EQ(cudaMalloc(&d_win, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_wout, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_aout, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_win, win.data(), N * sizeof(double),
                         cudaMemcpyHostToDevice), cudaSuccess);
    reg.bind_handle("win", d_win);
    reg.bind_handle("wout", d_wout);
    reg.bind_handle("aout", d_aout);
    EXPECT_EQ(reg.inject(CUstream(nullptr), N), 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    std::vector<double> wout(N), aout(N);
    ASSERT_EQ(cudaMemcpy(wout.data(), d_wout, N * sizeof(double),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(aout.data(), d_aout, N * sizeof(double),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_win); cudaFree(d_wout); cudaFree(d_aout);
    for (int i = 0; i < N; ++i) {
        ASSERT_DOUBLE_EQ(wout[std::size_t(i)], 2.0 * win[std::size_t(i)]);
        ASSERT_DOUBLE_EQ(aout[std::size_t(i)], win[std::size_t(i)] + 1.0);
    }
}

// ---------------------------------------------------------------------------
// The launch triple reaches the device body
// ---------------------------------------------------------------------------
TEST(ExecContractDeviceTest, DeviceV2KernelReceivesThePartitionTriple)
{
    ensureContext();
    constexpr int N = 100;
    const std::string path = stage("dtriple", DEVICE_PLUGIN_EXECV2_PTX, "exec_triple",
        kTripleArgs, "aether-abi/2", 2, kSampleLocalKeys);
    eagle::cuda::PluginRegistry reg =
        eagle::cuda::PluginRegistry::from_manifest(path);
    double* d_t = nullptr;
    ASSERT_EQ(cudaMalloc(&d_t, 3 * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemset(d_t, 0, 3 * sizeof(double)), cudaSuccess);
    reg.bind_handle("t", d_t);

    EXPECT_EQ(reg.inject(CUstream(nullptr), N), 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    double t[3] = {};
    ASSERT_EQ(cudaMemcpy(t, d_t, 3 * sizeof(double), cudaMemcpyDeviceToHost),
              cudaSuccess);
    EXPECT_DOUBLE_EQ(t[0], 0.0);
    EXPECT_DOUBLE_EQ(t[1], double(N));
    EXPECT_DOUBLE_EQ(t[2], double(N));

    EXPECT_EQ(reg.inject_partition(CUstream(nullptr), ex::Partition{ 40, 25, N }), 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(t, d_t, 3 * sizeof(double), cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_t);
    EXPECT_DOUBLE_EQ(t[0], 40.0);
    EXPECT_DOUBLE_EQ(t[1], 25.0);
    EXPECT_DOUBLE_EQ(t[2], double(N));   // the TRUE total, never the partition's
}

TEST(ExecContractDeviceTest, DeviceGridIsDerivedFromThePartitionCount)
{
    EXPECT_EQ(ex::DeviceKernel::grid(0, 256), 0u);
    EXPECT_EQ(ex::DeviceKernel::grid(1, 256), 1u);
    EXPECT_EQ(ex::DeviceKernel::grid(256, 256), 1u);
    EXPECT_EQ(ex::DeviceKernel::grid(257, 256), 2u);
    EXPECT_THROW(ex::DeviceKernel::grid(10, 0), std::runtime_error);
}

TEST(ExecContractDeviceTest, LegacyV1DevicePluginRefusesANonWholePartition)
{
    ensureContext();
    constexpr int N = 64;
    const std::string path = stage("dlegacy", DEVICE_PLUGIN_EXECV2_PTX, "exec_legacy",
        kLegacyArgs, "aether-abi/1", 1, "");
    eagle::cuda::PluginRegistry reg =
        eagle::cuda::PluginRegistry::from_manifest(path);
    EXPECT_EQ(reg.abi_version(0), 1);
    std::vector<double> x(N, 2.0);
    double *d_x = nullptr, *d_y = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_y, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, x.data(), N * sizeof(double), cudaMemcpyHostToDevice),
              cudaSuccess);
    reg.bind_handle("x", d_x);
    reg.bind_handle("y", d_y);
    EXPECT_EQ(reg.inject(CUstream(nullptr), N), 1);   // whole view still works
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<double> y(N);
    ASSERT_EQ(cudaMemcpy(y.data(), d_y, N * sizeof(double), cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_x); cudaFree(d_y);
    EXPECT_DOUBLE_EQ(y[0], 3.0);
    try {
        reg.inject_partition(CUstream(nullptr), ex::Partition{ 0, 32, N });
        FAIL() << "expected a legacy plugin to refuse a non-whole partition";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("WHOLE-VIEW"), std::string::npos)
            << e.what();
    }
}

// ---------------------------------------------------------------------------
// Partition identity, the mapreduce band, the host/device twin
// ---------------------------------------------------------------------------
namespace {
// Run `exec_local` on the device over `parts` and return y.
std::vector<double> deviceLocal(const std::vector<ex::Partition>& parts,
                                const std::vector<double>& x, const char* tag)
{
    const int N = int(x.size());
    const std::string path = stage(tag, DEVICE_PLUGIN_EXECV2_PTX, "exec_local",
        kLocalArgs, "aether-abi/2", 2, kSampleLocalKeys);
    eagle::cuda::PluginRegistry reg =
        eagle::cuda::PluginRegistry::from_manifest(path);
    double *d_x = nullptr, *d_y = nullptr;
    EXPECT_EQ(cudaMalloc(&d_x, N * sizeof(double)), cudaSuccess);
    EXPECT_EQ(cudaMalloc(&d_y, N * sizeof(double)), cudaSuccess);
    EXPECT_EQ(cudaMemcpy(d_x, x.data(), N * sizeof(double), cudaMemcpyHostToDevice),
              cudaSuccess);
    EXPECT_EQ(cudaMemset(d_y, 0, N * sizeof(double)), cudaSuccess);
    reg.bind_handle("x", d_x);
    reg.bind_handle("y", d_y);
    reg.bind_uniform("a", 3.25);
    reg.bind_uniform("b", -0.5);
    for (const ex::Partition& p : parts)
        reg.inject_partition(CUstream(nullptr), p);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<double> y(static_cast<std::size_t>(N), 0.0);
    EXPECT_EQ(cudaMemcpy(y.data(), d_y, N * sizeof(double), cudaMemcpyDeviceToHost),
              cudaSuccess);
    cudaFree(d_x); cudaFree(d_y);
    return y;
}
}  // namespace

TEST(ExecContractDeviceTest, DevicePartitionIdentityIsBitExact)
{
    ensureContext();
    constexpr int N = 1000;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[std::size_t(i)] = 1.0 / double(i + 1);
    const auto whole = deviceLocal({ ex::Partition::whole(N) }, x, "didw");
    const auto split = deviceLocal({ ex::Partition{ 0, 400, N },
                                     ex::Partition{ 400, 600, N } }, x, "dids");
    for (std::size_t i = 0; i < whole.size(); ++i)
        ASSERT_EQ(std::memcmp(&whole[i], &split[i], sizeof(double)), 0)
            << "partitioned device run differs at sample " << i;
    EXPECT_NE(whole[0], whole[1]);
}

TEST(ExecContractDeviceTest, DeviceMapreduceWholeVsPartitionsWithinBand)
{
    ensureContext();
    constexpr int N = 4096;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[std::size_t(i)] = 1.0 / double(i + 1);
    const std::string path = stage("dmr", DEVICE_PLUGIN_EXECV2_PTX, "exec_mapreduce",
        kMapreduceArgs, "aether-abi/2", 2, kMapreduceKeys);
    eagle::cuda::PluginRegistry reg =
        eagle::cuda::PluginRegistry::from_manifest(path);
    double *d_x = nullptr, *d_p = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMalloc(&d_p, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, x.data(), N * sizeof(double), cudaMemcpyHostToDevice),
              cudaSuccess);
    reg.bind_handle("x", d_x);
    reg.bind_handle("partial", d_p);

    ASSERT_EQ(cudaMemset(d_p, 0, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(reg.inject(CUstream(nullptr), N), 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const double whole = ex::DeviceKernel::combine_partials(
        ex::ReduceOp::Sum, reinterpret_cast<CUdeviceptr>(d_p), std::size_t(N));

    ASSERT_EQ(cudaMemset(d_p, 0, N * sizeof(double)), cudaSuccess);
    const ex::Partition p0{ 0, 1500, N }, p1{ 1500, N - 1500, N };
    ASSERT_EQ(reg.inject_partition(CUstream(nullptr), p0), 1);
    ASSERT_EQ(reg.inject_partition(CUstream(nullptr), p1), 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    const double r0 = ex::DeviceKernel::combine_partials(ex::ReduceOp::Sum,
        reinterpret_cast<CUdeviceptr>(d_p + p0.base), std::size_t(p0.count));
    const double r1 = ex::DeviceKernel::combine_partials(ex::ReduceOp::Sum,
        reinterpret_cast<CUdeviceptr>(d_p + p1.base), std::size_t(p1.count));
    cudaFree(d_x); cudaFree(d_p);
    const double partials[2] = { r0, r1 };
    const double split = ex::fold(ex::ReduceOp::Sum, partials, 2);

    EXPECT_TRUE(withinBand(whole, split, std::size_t(N)));
    EXPECT_FALSE(withinBand(whole, r0, std::size_t(N)))
        << "the band is so wide that losing a whole partition still passes";
}

TEST(ExecContractDeviceTest, HostDeviceArmsAgreeWithinBand)
{
    ensureContext();
    constexpr int N = 2048;
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i) x[std::size_t(i)] = 1.0 / double(i + 1);

    // DEVICE arm.
    const auto dev = deviceLocal({ ex::Partition::whole(N) }, x, "dtwin");

    // HOST arm — the SAME body text, compiled for the host, driven by the team.
    std::vector<double> y(std::size_t(N), 0.0);
    const std::string sidecarJson = std::string(R"({
  "kernel": "exec_local",
  "aether_abi": "aether-abi/2",
  "schema_version": 2,
  "pattern": "pure",
  "scalar_type": "float64",
  "arg_spec": )") + kLocalArgs + R"(
})";
    const Sidecar sc = parse_sidecar(writeTmp("twin_sidecar.json", sidecarJson));
    eagle::cpu::PluginRegistry hreg;
    ASSERT_NO_THROW(hreg.add_plugin(sc, HOST_PLUGIN_EXECV2_SO));
    hreg.bind_handle("x", const_cast<double*>(x.data()));
    hreg.bind_handle("y", y.data());
    hreg.bind_uniform("a", 3.25);
    hreg.bind_uniform("b", -0.5);
    auto packed = hreg.pack(0, N);
    ex::HostTeam::run(hreg.entry_v2(0), packed.params, ex::Partition::whole(N), 0);

    // The RULED host/device band: S x 2 x eps with S = the number
    // of floating-point combines the body performs on one sample (a multiply and
    // an add) — derived from the one anchor, never fitted to what was observed.
    // The two arms may legitimately differ by an FMA contraction on one side.
    for (std::size_t i = 0; i < y.size(); ++i)
        ASSERT_TRUE(withinBand(dev[i], y[i], 2u))
            << "host/device twin differs at sample " << i;
    EXPECT_NE(y[0], y[1]);
}

// ---------------------------------------------------------------------------
// A kernel with a THREAD BOX (`launch.box`, T = 4, stride = 8: a span is 32 lanes): the device
// registry reads the box from the sidecar and counts a partition in threads.
// ---------------------------------------------------------------------------
namespace {
constexpr std::int64_t kBoxSpan = 32;

// stage() with a `launch` block in the sidecar.
std::string stageBoxed(const std::string& tag)
{
    writeTmp(tag + ".ptx", slurp(DEVICE_PLUGIN_EXECV2_PTX));
    writeTmp(tag + "_sidecar.json", R"({
  "kernel": "exec_boxed",
  "aether_abi": "aether-abi/2",
  "schema_version": 2,
  "pattern": "pure",
  "scalar_type": "float64",
  "launch": {"block": 128, "box": {"axis": 0, "T": 4, "stride": 8}},
  "arg_spec": [["mutable","y"],["nsamples","nsamples"]]
})");
    return writeTmp(tag + "_manifest.json", R"({
  "schema_version": 2,
  "pattern": "pure",
  "aether_abi": "aether-abi/2",
  "exec_targets": ["device"],
  "exec_access": "sample_local",
  "plugins": [
    {"id": ")" + tag + R"(", "order": 0, "enabled": true,
     "artifact": ")" + tag + R"(.ptx", "sidecar": ")" + tag + R"(_sidecar.json",
     "format": "ptx"}
  ]
})");
}

// `parts` (in lanes) injected, in order, on a fresh y of N lanes.
std::vector<double> boxedDevice(const std::vector<ex::Partition>& parts, std::int64_t N, const char* tag)
{
    eagle::cuda::PluginRegistry reg = eagle::cuda::PluginRegistry::from_manifest(stageBoxed(tag));
    double* d_y = nullptr;
    EXPECT_EQ(cudaMalloc(&d_y, std::size_t(N) * sizeof(double)), cudaSuccess);
    EXPECT_EQ(cudaMemset(d_y, 0, std::size_t(N) * sizeof(double)), cudaSuccess);
    reg.bind_handle("y", d_y);
    try {
        for (const ex::Partition& p : parts)
            reg.inject_partition(CUstream(nullptr), p, 64);
    } catch (...) {
        cudaFree(d_y);
        throw;
    }
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<double> y(std::size_t(N), 0.0);
    EXPECT_EQ(cudaMemcpy(y.data(), d_y, std::size_t(N) * sizeof(double), cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_y);
    return y;
}

// Each part alone writes exactly its own lanes once; all of them together write every lane once.
void expectDeviceOwnership(const std::vector<ex::Partition>& parts, std::int64_t N)
{
    for (const ex::Partition& p : parts) {
        const auto y = boxedDevice({ p }, N, "dboxp");
        for (std::int64_t l = 0; l < N; ++l)
            ASSERT_DOUBLE_EQ(y[std::size_t(l)], (l >= p.base && l < p.base + p.count) ? 1.0 : 0.0)
                << "lane " << l << " of partition [" << p.base << ", " << p.base + p.count << ")";
    }
    const auto all = boxedDevice(parts, N, "dboxa");
    for (std::int64_t l = 0; l < N; ++l)
        ASSERT_DOUBLE_EQ(all[std::size_t(l)], 1.0) << "lane " << l << " of the whole set";
}
}  // namespace

TEST(ExecContractDeviceTest, DeviceBoxedWholeAndAlignedPartitionsOwnTheirLanes)
{
    ensureContext();
    for (std::int64_t rows : { 31, 32, 64 }) {          // 31 rows: the last span has 3 of T = 4
        const std::int64_t N = rows * 8;
        expectDeviceOwnership({ ex::Partition::whole(N) }, N);
        const std::int64_t half = N / kBoxSpan / 2 * kBoxSpan;
        expectDeviceOwnership({ ex::Partition{ 0, half, N }, ex::Partition{ half, N - half, N } }, N);
        const std::int64_t q = std::max<std::int64_t>(1, N / kBoxSpan / 4) * kBoxSpan;
        std::vector<ex::Partition> four;
        for (std::int64_t b = 0; b < N; b += q)
            four.push_back(ex::Partition{ b, std::min(q, N - b), N });
        expectDeviceOwnership(four, N);
    }
}

TEST(ExecContractDeviceTest, DeviceBoxedPartialRangeRunsOnlyItsSpans)
{
    ensureContext();
    const std::int64_t N = 31 * 8;
    const ex::Partition mid{ 2 * kBoxSpan, 3 * kBoxSpan, N };
    const auto y = boxedDevice({ mid }, N, "dboxm");
    for (std::int64_t l = 0; l < N; ++l)
        ASSERT_DOUBLE_EQ(y[std::size_t(l)], (l >= mid.base && l < mid.base + mid.count) ? 1.0 : 0.0)
            << "lane " << l;
    const ex::Partition tail{ 4 * kBoxSpan, N - 4 * kBoxSpan, N };
    const auto z = boxedDevice({ tail }, N, "dboxt");
    for (std::int64_t l = 0; l < N; ++l)
        ASSERT_DOUBLE_EQ(z[std::size_t(l)], l >= tail.base ? 1.0 : 0.0) << "lane " << l;
}

TEST(ExecContractDeviceTest, DeviceBoxedPartitionOffASpanIsRefusedByName)
{
    ensureContext();
    const std::int64_t N = 31 * 8;
    for (const ex::Partition p : { ex::Partition{ 0, 100, N }, ex::Partition{ 10, 64, N },
                                   ex::Partition{ kBoxSpan, 5, N } }) {
        try {
            boxedDevice({ p }, N, "dboxr");
            FAIL() << "expected the partition [" << p.base << ", " << p.base + p.count << ") to be refused";
        } catch (const std::runtime_error& e) {
            const std::string what = e.what();
            EXPECT_NE(what.find("thread box"), std::string::npos) << what;
            EXPECT_NE(what.find("32"), std::string::npos) << what;
        }
    }
    // fewer rows than T: whole only
    expectDeviceOwnership({ ex::Partition::whole(3 * 8) }, 3 * 8);
    EXPECT_THROW(boxedDevice({ ex::Partition{ 0, 8, 24 } }, 24, "dboxf"), std::runtime_error);
}

}  // namespace ExecContractDeviceTest
}  // namespace eagle_tests
