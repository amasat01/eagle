// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `eagle::exec::RankPartition<DeviceKernel>` — the DEVICE rows of the 2-rank MPI
// bed (MECHANISM only,
// and deliberately NO CUDA-aware MPI).
//
// THE BED'S HARDWARE SHAPE: both ranks use GPU 0 (`CUDA_VISIBLE_DEVICES=0`),
// each in its OWN context. That is a single-node bed by construction, and it is
// enough for the mechanism -- one contiguous sub-partition per rank,
// launched by the rank's own `cuLaunchKernel`, gathered host-staged — is exactly
// the same mechanism on two GPUs as on one. Nothing here is a performance claim,
// and nothing here would be one on better hardware.
//
// THE STAGING IS AETHER'S. The output plane is an `aether::Array`, so
// the D2H and H2D halves of the gather are `Array::download`/`Array::upload` —
// `aether::copyAsync` underneath — and `eagle/exec/` contains no `cudaMemcpy` of
// its own. `DeviceInnerStagesThroughAether` is that rule as an executable audit,
// with its matcher proven on a control string first: a grep that finds nothing is
// only evidence when it has been shown it can find something.
//
// FIXTURES: tests/fixtures/device_plugin_execv2.cu, nvcc'd to raw PTX by
// tests/CMakeLists.txt — the same artifact the main CUDA suite's rows load.

#include "eagle/exec/RankPartition.h"
#include "eagle/exec/DeviceKernel.h"
#include "eagle/exec/Partition.h"
#include "plugin/plugin_registry/registry.h"

#include <mpi.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <dirent.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace eagle::plugin;
namespace ex = eagle::exec;

namespace eagle_tests {
namespace MpiBedDeviceTest {

using RPD = ex::RankPartition<ex::DeviceKernel>;

static int worldRank()
{
    int r = 0;
    EXPECT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &r), MPI_SUCCESS);
    return r;
}

// Rank-prefixed, for the same reason the host rows' staging is: TempDir() is shared
// by both ranks, and two processes writing one file can leave a half-written JSON
// that fails to parse on one rank only.
static std::string writeTmp(const std::string& name, const std::string& body)
{
    const std::string path =
        ::testing::TempDir() + "r" + std::to_string(worldRank()) + "_" + name;
    std::ofstream f(path, std::ios::binary);
    f << body;
    f.close();
    return path;
}

static std::string slurp(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

// Stage the v2 PTX beside a sidecar + manifest naming @p kernel; returns the
// manifest path. Mirrors tests/test_ExecContractDevice.cu's own `stage`.
static std::string stage(const std::string& tag, const std::string& kernel,
                         const std::string& argSpec)
{
    writeTmp(tag + ".ptx", slurp(DEVICE_PLUGIN_EXECV2_PTX));
    const std::string rankTag = "r" + std::to_string(worldRank()) + "_" + tag;
    const std::string sidecar = R"({
  "kernel": ")" + kernel + R"(",
  "aether_abi": "aether-abi/2",
  "schema_version": 2,
  "pattern": "pure",
  "scalar_type": "float64",
  "arg_spec": )" + argSpec + R"(
})";
    writeTmp(tag + "_sidecar.json", sidecar);
    const std::string manifest = R"({
  "schema_version": 2,
  "pattern": "pure",
  "aether_abi": "aether-abi/2",
  "exec_targets": ["device", "host"],
  "exec_access": "sample_local",
  "plugins": [
    {"id": ")" + tag + R"(", "order": 0, "enabled": true,
     "artifact": ")" + rankTag + R"(.ptx", "sidecar": ")" + rankTag + R"(_sidecar.json",
     "format": "ptx"}
  ]
})";
    return writeTmp(tag + "_manifest.json", manifest);
}

static const char* kLocalArgs =
    R"([["mutable","y"],["per_sample","x"],["uniform","a"],["uniform","b"],["nsamples","nsamples"]])";

static void cuCheck(CUresult r, const char* what)
{
    if (r != CUDA_SUCCESS) {
        const char* m = nullptr; cuGetErrorString(r, &m);
        FAIL() << what << " failed: " << (m ? m : "?");
    }
}

// The same primary-context bootstrap the other device-registry tests use. Each
// rank is its own process, so each gets its own context on the same GPU 0.
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

// This rank's expected slice, stated from the contract rather than read back from
// `RankPartition::local` — see the same helper in test_MpiBed.cpp for why asking
// the function under test what to expect hides exactly the defect that matters.
static void expectedSlice(std::int64_t n, std::int64_t& base, std::int64_t& count)
{
    int rank = 0, size = 0;
    ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
    ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &size), MPI_SUCCESS);
    const std::int64_t q = n / size, rem = n % size, r = rank;
    count = q + (r < rem ? 1 : 0);
    base  = r * q + (r < rem ? r : rem);
}

static constexpr double kSentinel = -12345.0;

static ::testing::AssertionResult identicalOnAllRanks(const double* v,
                                                      std::size_t n)
{
    std::vector<double> ref(v, v + n);
    const int rc = MPI_Bcast(ref.data(), int(n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS)
        return ::testing::AssertionFailure() << "MPI_Bcast failed: " << rc;
    for (std::size_t i = 0; i < n; ++i)
        if (std::memcmp(&v[i], &ref[i], sizeof(double)) != 0)
            return ::testing::AssertionFailure()
                << "element " << i << " differs from rank 0's: " << v[i]
                << " vs " << ref[i];
    return ::testing::AssertionSuccess();
}

// Strip C and C++ comments. The audit is about what the CODE does, and the first
// version of it went red on this very header's DOCSTRING, which explains that the
// staging deliberately avoids a raw memcpy — the word, not the call. Scanning prose
// cuts both ways: a comment can also CLAIM compliance the code does not have. The
// stripper is exercised on a control pair below before it is trusted here, so its
// two failure directions (dropping real code, keeping commented code) are pinned.
static std::string stripComments(const std::string& src)
{
    std::string out;
    out.reserve(src.size());
    enum { Code, Line, Block } st = Code;
    for (std::size_t i = 0; i < src.size(); ++i) {
        const char c = src[i];
        const char n = (i + 1 < src.size()) ? src[i + 1] : '\0';
        if (st == Code) {
            if (c == '/' && n == '/') { st = Line;  ++i; continue; }
            if (c == '/' && n == '*') { st = Block; ++i; continue; }
            out.push_back(c);
        } else if (st == Line) {
            if (c == '\n') { st = Code; out.push_back(c); }
        } else {
            if (c == '*' && n == '/') { st = Code; ++i; }
        }
    }
    return out;
}

// The file names in eagle/exec/, enumerated from the DIRECTORY rather than listed
// here: a hand-written list would go blind the day a new structure header lands,
// which is precisely when this kind of audit matters.
static std::vector<std::string> execHeaders()
{
    std::vector<std::string> out;
    DIR* d = ::opendir(EAGLE_EXEC_SOURCE_DIR);
    if (d == nullptr) return out;
    while (const dirent* e = ::readdir(d)) {
        const std::string name = e->d_name;
        if (name.size() > 2 && name.compare(name.size() - 2, 2, ".h") == 0)
            out.push_back(name);
    }
    ::closedir(d);
    return out;
}

}  // namespace MpiBedDeviceTest
}  // namespace eagle_tests

using namespace eagle_tests::MpiBedDeviceTest;

// ---------------------------------------------------------------------------
// sample_local under a DEVICE inner is BIT-identical whole vs across ranks
// ---------------------------------------------------------------------------
TEST(RankPartition, SampleLocalDeviceBitExactVsWhole)
{
    ensureContext();
    constexpr std::int64_t N = 1000;
    const ex::Partition whole = ex::Partition::whole(N);
    const ex::Partition mine  = RPD::local(whole, MPI_COMM_WORLD);
    std::int64_t expBase = 0, expCount = 0;
    expectedSlice(N, expBase, expCount);
    ASSERT_EQ(mine.base, expBase);
    ASSERT_EQ(mine.count, expCount);
    ASSERT_EQ(mine.nSamples, N);

    std::vector<double> x(static_cast<std::size_t>(N));
    for (std::int64_t i = 0; i < N; ++i) x[std::size_t(i)] = 1.0 / double(i + 1);

    // The ORACLE: the whole run on this rank's own GPU, from the replicated input.
    std::vector<double> oracle(static_cast<std::size_t>(N), kSentinel);
    {
        const std::string path = stage("dmpiw", "exec_local", kLocalArgs);
        eagle::cuda::PluginRegistry reg =
            eagle::cuda::PluginRegistry::from_manifest(path);
        double *d_x = nullptr, *d_y = nullptr;
        ASSERT_EQ(cudaMalloc(&d_x, N * sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMalloc(&d_y, N * sizeof(double)), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(d_x, x.data(), N * sizeof(double),
                             cudaMemcpyHostToDevice), cudaSuccess);
        reg.bind_handle("x", d_x);
        reg.bind_handle("y", d_y);
        reg.bind_uniform("a", 3.25);
        reg.bind_uniform("b", -0.5);
        ASSERT_EQ(reg.inject_partition(CUstream(nullptr), whole), 1);
        ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(oracle.data(), d_y, N * sizeof(double),
                             cudaMemcpyDeviceToHost), cudaSuccess);
        cudaFree(d_x); cudaFree(d_y);
    }

    // The DISTRIBUTED run. The OUTPUT plane is an aether Array: that is what makes
    // the gather's D2H/H2D halves `aether::copyAsync` rather than a memcpy of
    // eagle's own — see RankPartition::allgather_staged.
    const std::string path = stage("dmpid", "exec_local", kLocalArgs);
    eagle::cuda::PluginRegistry reg =
        eagle::cuda::PluginRegistry::from_manifest(path);
    ASSERT_EQ(reg.abi_version(0), 2);
    ASSERT_NO_THROW(RPD::check(ex::Access::SampleLocal, reg.abi_version(0),
                               MPI_COMM_WORLD));

    aether::Array<double> y(static_cast<std::size_t>(N));
    {
        auto h = y.hostView();
        for (std::int64_t i = 0; i < N; ++i) h(i) = kSentinel;
    }
    y.upload();
    double* d_x = nullptr;
    ASSERT_EQ(cudaMalloc(&d_x, N * sizeof(double)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(d_x, x.data(), N * sizeof(double), cudaMemcpyHostToDevice),
              cudaSuccess);
    reg.bind_handle("x", d_x);
    reg.bind_handle("y", y.deviceView().data());
    reg.bind_uniform("a", 3.25);
    reg.bind_uniform("b", -0.5);

    auto packed = reg.pack(0, N);
    ASSERT_EQ(RPD::run(reg.function(0), packed.params, whole, MPI_COMM_WORLD,
                       CUstream(nullptr)), 1);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

    // PRE-GATHER: this rank's kernel covered its OWN sub-partition and no other.
    // Read back with a copy of the TEST's own, so the check does not depend on the
    // staging path it is meant to be independent of.
    std::vector<double> pre(static_cast<std::size_t>(N), 0.0);
    ASSERT_EQ(cudaMemcpy(pre.data(), y.deviceView().data(), N * sizeof(double),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    for (std::int64_t i = 0; i < N; ++i) {
        const bool mineHere = (i >= expBase && i < expBase + expCount);
        if (mineHere) {
            ASSERT_NE(pre[std::size_t(i)], kSentinel)
                << "rank " << worldRank() << " left its own sample " << i << " unwritten";
        } else {
            ASSERT_DOUBLE_EQ(pre[std::size_t(i)], kSentinel)
                << "rank " << worldRank() << " wrote sample " << i
                << ", which belongs to another rank";
        }
    }

    RPD::allgather_staged(y, whole, MPI_COMM_WORLD, aether::Stream(nullptr));

    // Read the DEVICE copy, not the host chunk: that is what proves BOTH halves of
    // the staging ran — a gather that only came home D2H would leave the device
    // plane still holding this rank's slice and sentinels.
    std::vector<double> got(static_cast<std::size_t>(N), 0.0);
    ASSERT_EQ(cudaMemcpy(got.data(), y.deviceView().data(), N * sizeof(double),
                         cudaMemcpyDeviceToHost), cudaSuccess);
    cudaFree(d_x);

    for (std::size_t i = 0; i < got.size(); ++i)
        ASSERT_EQ(std::memcmp(&got[i], &oracle[i], sizeof(double)), 0)
            << "the rank-partitioned device run differs from the whole run at sample "
            << i;
    EXPECT_TRUE(identicalOnAllRanks(got.data(), got.size()));
    EXPECT_NE(got[0], got[1]);
}

// ---------------------------------------------------------------------------
// The device inner stages through aether, and this audit can fail
// ---------------------------------------------------------------------------
TEST(RankPartition, DeviceInnerStagesThroughAether)
{
    // The matcher, proven on a KNOWN answer before it is trusted on the real
    // sources: a scan that reports "no hits" is evidence only once it has been
    // shown to hit.
    auto contains = [](const std::string& hay, const std::string& needle) {
        return hay.find(needle) != std::string::npos;
    };
    const std::string control =
        "    cudaMemcpy(dst, src, n, cudaMemcpyDeviceToHost);\n";
    ASSERT_TRUE(contains(stripComments(control), "cudaMemcpy"))
        << "the audit matcher cannot find a raw memcpy even when there is one";
    // ...and the comment stripper keeps real code while dropping prose about it.
    ASSERT_FALSE(contains(stripComments("// cudaMemcpy(a, b, n, k);\nint x = 1;\n"),
                          "cudaMemcpy"))
        << "the stripper leaves a line comment in, so the audit judges prose";
    ASSERT_FALSE(contains(stripComments("/* cudaMemcpy(a, b, n, k); */\n"),
                          "cudaMemcpy"))
        << "the stripper leaves a block comment in, so the audit judges prose";
    ASSERT_TRUE(contains(stripComments("/* prose */\n" + control), "cudaMemcpy"))
        << "the stripper ate real code along with the comment";

    // The enumeration must actually enumerate. A directory scan that came back
    // empty would report a clean audit over nothing.
    const std::vector<std::string> headers = execHeaders();
    ASSERT_GE(headers.size(), 4u)
        << "eagle/exec/ enumeration returned " << headers.size() << " headers";
    for (const char* required : { "Partition.h", "HostTeam.h", "DeviceKernel.h",
                                  "RankPartition.h" }) {
        EXPECT_NE(std::find(headers.begin(), headers.end(), std::string(required)),
                  headers.end())
            << "eagle/exec/" << required << " was not enumerated by the audit";
    }

    // No raw `cudaMemcpy` anywhere in the execution structures.
    std::string rankPartitionSrc;
    for (const std::string& name : headers) {
        const std::string raw =
            slurp(std::string(EAGLE_EXEC_SOURCE_DIR) + "/" + name);
        ASSERT_FALSE(raw.empty()) << "eagle/exec/" << name << " read back empty";
        const std::string src = stripComments(raw);
        EXPECT_FALSE(contains(src, "cudaMemcpy"))
            << "eagle/exec/" << name << " performs a raw cudaMemcpy (aether "
               "data moves through aether::copyAsync)";
        if (name == "RankPartition.h") rankPartitionSrc = src;
    }
    ASSERT_FALSE(rankPartitionSrc.empty());

    // RankPartition's gather in particular does NO copying of its own — not the
    // runtime API, not the driver API, not std::memcpy.
    for (const char* forbidden : { "cudaMemcpy", "cuMemcpy", "memcpy(" })
        EXPECT_FALSE(contains(rankPartitionSrc, forbidden))
            << "RankPartition.h copies with '" << forbidden << "' instead of "
               "staging through aether";
    // ...and it stages through aether's own transfer, which is what makes the
    // absence above a design rather than an omission.
    EXPECT_TRUE(contains(rankPartitionSrc, "arr.download(stream)"));
    EXPECT_TRUE(contains(rankPartitionSrc, "arr.upload(stream)"));
}
