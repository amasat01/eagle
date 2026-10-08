// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `eagle::exec::RankPartition` -- the HOST rows of the 2-rank MPI bed;
//
// MECHANISM only).
//
// EVERY ROW HERE RUNS ON BOTH RANKS. The collectives inside `RankPartition` are
// blocking, so a row that returned early on one rank would hang the other; there
// is deliberately no `if (rank == 0)` anywhere in this file. Each row therefore
// asserts two different things at once — what THIS rank computed, and that the
// other rank computed the same — and the gate (tests/mpi/check_mpi_bed.sh)
// independently compares the two ranks' XML verdicts, so a row that quietly
// asserted nothing on rank 1 would still be caught.
//
// THE ORACLE IS LOCAL, NOT REMOTE. Inputs are REPLICATED (that is the structure's
// premise), so every rank can compute the whole-run answer for itself with
// `HostTeam::run_serial` and compare the gathered distributed answer against it.
// That keeps the comparison bit-exact and free of any wire of its own.
//
// WHAT MAKES THESE ROWS NON-VACUOUS. A distributed identity row passes trivially if
// each rank simply ran the WHOLE view and the gather then overwrote everything with
// the same numbers. So the identity rows assert the PRE-GATHER state too: outside
// this rank's own sub-partition the plane must still hold its sentinel. And the
// mapreduce row asserts that dropping one rank's partial FAILS the band, so the
// band is not so wide that it certifies anything.
//
// FIXTURES: tests/fixtures/host_plugin_execv2.cpp — the same artifact the main
// suite's rows use (tests/CMakeLists.txt builds it in both modes), never a
// bed-private copy.

#include "eagle/exec/RankPartition.h"
#include "eagle/exec/HostTeam.h"
#include "eagle/exec/Partition.h"
#include "plugin/host_registry.h"

#include <mpi.h>

// Deliberately NOT "TestBase.h" — same reasoning as tests/test_ExecContract.cpp:
// that header pulls eagle's device headers, whose __device__/__host__ attributes
// only nvcc parses, and this `.cpp` is compiled by g++ in either build mode.
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
namespace MpiBedTest {

using RP = ex::RankPartition<ex::HostTeam>;

// ---------------------------------------------------------------------------
// Fixture staging — the same sidecar shape tests/test_ExecContract.cpp writes.
//
// TempDir() is shared by both ranks, and both write the same bytes to the same
// name, so the file name carries the rank: two processes racing on one `ofstream`
// can interleave into a half-written JSON that then fails to parse on one rank
// only, which would look like a distributed defect and is not one.
// ---------------------------------------------------------------------------
static int worldRank()
{
    int r = 0;
    EXPECT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &r), MPI_SUCCESS);
    return r;
}

static std::string writeTmp(const std::string& name, const std::string& body)
{
    const std::string path =
        ::testing::TempDir() + "r" + std::to_string(worldRank()) + "_" + name;
    std::ofstream f(path, std::ios::binary);
    f << body;
    f.close();
    return path;
}

static Sidecar sidecarFor(const std::string& kernel, const std::string& argSpec,
                          const std::string& abi = "aether-abi/2",
                          int schemaVersion = 2)
{
    const std::string json = R"({
  "kernel": ")" + kernel + R"(",
  "aether_abi": ")" + abi + R"(",
  "schema_version": )" + std::to_string(schemaVersion) + R"(,
  "pattern": "pure",
  "scalar_type": "float64",
  "arg_spec": )" + argSpec + R"(
})";
    return parse_sidecar(writeTmp("mpi_" + kernel + "_sidecar.json", json));
}

static const char* kLocalArgs =
    R"([["mutable","y"],["per_sample","x"],["uniform","a"],["uniform","b"],["nsamples","nsamples"]])";
static const char* kGatherArgs =
    R"([["mutable","y"],["per_sample","table"],["nsamples","nsamples"]])";
static const char* kMapreduceArgs =
    R"([["accum_out","partial"],["per_sample","x"],["nsamples","nsamples"]])";
static const char* kLegacyArgs =
    R"([["mutable","y"],["per_sample","x"],["nsamples","nsamples"]])";

// The RULED band (fork F-d): S x 2 x eps(float64), S = the number of elements
// folded, applied RELATIVELY — derived from one anchor, never fitted to an
// observed difference. Identical to the main suite's, deliberately: the whole
// point of a band is that it is one rule, not one per test file.
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

// Bit-compare @p v against rank 0's copy of it. This is what "identical on all
// ranks" means operationally: a value every rank agrees on to the last bit, not
// one every rank finds plausible.
static ::testing::AssertionResult identicalOnAllRanks(const double* v,
                                                      std::size_t n)
{
    std::vector<double> ref(v, v + n);
    const int rc = MPI_Bcast(ref.data(), int(n), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if (rc != MPI_SUCCESS)
        return ::testing::AssertionFailure() << "MPI_Bcast failed: " << rc;
    for (std::size_t i = 0; i < n; ++i) {
        if (std::memcmp(&v[i], &ref[i], sizeof(double)) != 0)
            return ::testing::AssertionFailure()
                << "element " << i << " differs from rank 0's: " << v[i]
                << " vs " << ref[i];
    }
    return ::testing::AssertionSuccess();
}

// A sentinel no body here ever writes, so "still the sentinel" is unambiguous
// evidence that nothing touched that sample on this rank.
static constexpr double kSentinel = -12345.0;

// THIS RANK'S EXPECTED SLICE, stated here rather than read back from
// `RankPartition::local`. Asking the function under test what it expects is how the
// first version of the identity row below survived an injected defect that gave
// EVERY rank the whole view: the row's `mine` moved with the defect, the sentinel
// check compared the wrong thing to itself, and only the gather-shape rows went red.
// `q`/`rem` here are the contract restated (contiguous, first `rem` ranks take one
// extra), not a call.
static void expectedSlice(std::int64_t n, std::int64_t& base, std::int64_t& count)
{
    int rank = 0, size = 0;
    ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &rank), MPI_SUCCESS);
    ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &size), MPI_SUCCESS);
    const std::int64_t q = n / size, rem = n % size, r = rank;
    count = q + (r < rem ? 1 : 0);
    base  = r * q + (r < rem ? r : rem);
}

}  // namespace MpiBedTest
}  // namespace eagle_tests

using namespace eagle_tests::MpiBedTest;

// ---------------------------------------------------------------------------
// The bed is a TWO-rank bed
// ---------------------------------------------------------------------------
TEST(MpiBed, WorldSizeIsTwo)
{
    int size = 0;
    ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &size), MPI_SUCCESS);
    EXPECT_EQ(size, 2) << "the bed is a TWO-rank bed; run it under `mpirun -np 2`";
    EXPECT_EQ(RP::comm_size(MPI_COMM_WORLD), size);
    EXPECT_EQ(RP::comm_rank(MPI_COMM_WORLD), worldRank());
}

// ---------------------------------------------------------------------------
// sample_local is BIT-identical whole vs across ranks
// ---------------------------------------------------------------------------
TEST(RankPartition, SampleLocalHostBitExactVsWhole)
{
    constexpr std::int64_t N = 1000;
    const ex::Partition whole = ex::Partition::whole(N);
    const ex::Partition mine  = RP::local(whole, MPI_COMM_WORLD);

    // The split itself, against the rule as stated in this file.
    std::int64_t expBase = 0, expCount = 0;
    expectedSlice(N, expBase, expCount);
    ASSERT_EQ(mine.base, expBase);
    ASSERT_EQ(mine.count, expCount);
    ASSERT_EQ(mine.nSamples, N);

    std::vector<double> x(static_cast<std::size_t>(N));
    for (std::int64_t i = 0; i < N; ++i) x[std::size_t(i)] = 1.0 / double(i + 1);

    // The ORACLE: the whole run, computed locally from the replicated inputs.
    std::vector<double> oracle(std::size_t(N), kSentinel);
    {
        eagle::cpu::PluginRegistry reg;
        ASSERT_NO_THROW(reg.add_plugin(sidecarFor("exec_local", kLocalArgs),
                                       HOST_PLUGIN_EXECV2_SO));
        reg.bind_handle("x", x.data());
        reg.bind_handle("y", oracle.data());
        reg.bind_uniform("a", 3.25);
        reg.bind_uniform("b", -0.5);
        auto packed = reg.pack(0, N);
        ex::HostTeam::run_serial(reg.entry_v2(0), packed.params.data(), whole);
    }

    // The DISTRIBUTED run: this rank's share only.
    std::vector<double> y(std::size_t(N), kSentinel);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sidecarFor("exec_local", kLocalArgs),
                                   HOST_PLUGIN_EXECV2_SO));
    ASSERT_NO_THROW(RP::check(ex::Access::SampleLocal, reg.abi_version(0),
                              MPI_COMM_WORLD));
    reg.bind_handle("x", x.data());
    reg.bind_handle("y", y.data());
    reg.bind_uniform("a", 3.25);
    reg.bind_uniform("b", -0.5);
    auto packed = reg.pack(0, N);
    RP::run(reg.entry_v2(0), packed.params, whole, MPI_COMM_WORLD);

    // PRE-GATHER: this rank touched its OWN sub-partition and nothing else. Without
    // this the row would pass just as happily if every rank ran the whole view.
    for (std::int64_t i = 0; i < N; ++i) {
        const bool mineHere = (i >= expBase && i < expBase + expCount);
        if (mineHere) {
            ASSERT_NE(y[std::size_t(i)], kSentinel)
                << "rank " << worldRank() << " left its own sample " << i << " unwritten";
        } else {
            ASSERT_DOUBLE_EQ(y[std::size_t(i)], kSentinel)
                << "rank " << worldRank() << " wrote sample " << i
                << ", which belongs to another rank";
        }
    }

    RP::allgather_plane(y.data(), whole, MPI_COMM_WORLD);

    for (std::size_t i = 0; i < y.size(); ++i)
        ASSERT_EQ(std::memcmp(&y[i], &oracle[i], sizeof(double)), 0)
            << "the rank-partitioned run differs from the whole run at sample " << i;
    EXPECT_TRUE(identicalOnAllRanks(y.data(), y.size()));
    EXPECT_NE(y[0], y[1]);   // the body did something sample-dependent
}

// ---------------------------------------------------------------------------
// Design decision: cross_sample_read is LEGAL here because inputs are REPLICATED
// ---------------------------------------------------------------------------
TEST(RankPartition, CrossSampleReadReplicatedTableBitExact)
{
    // A prime N, so the split is uneven and the (i*7 + 3) % nSamples reads cross the
    // rank boundary in both directions rather than staying inside a rank's slice.
    constexpr std::int64_t N = 997;
    const ex::Partition whole = ex::Partition::whole(N);
    const ex::Partition mine  = RP::local(whole, MPI_COMM_WORLD);
    std::int64_t expBase = 0, expCount = 0;
    expectedSlice(N, expBase, expCount);
    ASSERT_EQ(mine.base, expBase);
    ASSERT_EQ(mine.count, expCount);

    std::vector<double> table(static_cast<std::size_t>(N));
    for (std::int64_t i = 0; i < N; ++i)
        table[std::size_t(i)] = 1.0 / double(i + 1) + 0.125 * double(i);

    std::vector<double> oracle(std::size_t(N), kSentinel);
    {
        eagle::cpu::PluginRegistry reg;
        ASSERT_NO_THROW(reg.add_plugin(sidecarFor("exec_gather", kGatherArgs),
                                       HOST_PLUGIN_EXECV2_SO));
        reg.bind_handle("table", table.data());
        reg.bind_handle("y", oracle.data());
        auto packed = reg.pack(0, N);
        ex::HostTeam::run_serial(reg.entry_v2(0), packed.params.data(), whole);
    }

    std::vector<double> y(std::size_t(N), kSentinel);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sidecarFor("exec_gather", kGatherArgs),
                                   HOST_PLUGIN_EXECV2_SO));
    // The ruling itself, asserted before the run: this placement is legal.
    ASSERT_NO_THROW(RP::check(ex::Access::CrossSampleRead, reg.abi_version(0),
                              MPI_COMM_WORLD));
    reg.bind_handle("table", table.data());
    reg.bind_handle("y", y.data());
    auto packed = reg.pack(0, N);
    RP::run(reg.entry_v2(0), packed.params, whole, MPI_COMM_WORLD);

    for (std::int64_t i = 0; i < N; ++i) {
        const bool mineHere = (i >= expBase && i < expBase + expCount);
        if (!mineHere) {
            ASSERT_DOUBLE_EQ(y[std::size_t(i)], kSentinel)
                << "rank " << worldRank() << " wrote another rank's sample " << i;
        }
    }

    RP::allgather_plane(y.data(), whole, MPI_COMM_WORLD);

    for (std::size_t i = 0; i < y.size(); ++i)
        ASSERT_EQ(std::memcmp(&y[i], &oracle[i], sizeof(double)), 0)
            << "a cross-sample read differs from the whole run at sample " << i
            << " — the table was not whole on this rank";
    EXPECT_TRUE(identicalOnAllRanks(y.data(), y.size()));

    // The read really does leave this rank's own slice: at least one sample in this
    // rank's share sources its value from a sample the OTHER rank owns. Without this
    // the row could pass over a body that never crossed a boundary.
    bool crossed = false;
    for (std::int64_t i = mine.base; i < mine.base + mine.count && !crossed; ++i) {
        const std::int64_t src = (i * 7 + 3) % N;
        crossed = (src < mine.base || src >= mine.base + mine.count);
    }
    EXPECT_TRUE(crossed) << "no read in this rank's share left its own sub-partition";
}

// ---------------------------------------------------------------------------
// Mapreduce: rank-order fold, within band, identical on every rank
// ---------------------------------------------------------------------------
TEST(RankPartition, MapreduceWithinBandIdenticalOnAllRanks)
{
    constexpr std::int64_t N = 4096;
    const ex::Partition whole = ex::Partition::whole(N);
    const ex::Partition mine  = RP::local(whole, MPI_COMM_WORLD);

    std::vector<double> x(static_cast<std::size_t>(N));
    for (std::int64_t i = 0; i < N; ++i) x[std::size_t(i)] = 1.0 / double(i + 1);

    // The WHOLE fold, locally.
    double wholeVal = 0.0;
    {
        std::vector<double> partial(std::size_t(N), 0.0);
        eagle::cpu::PluginRegistry reg;
        ASSERT_NO_THROW(reg.add_plugin(sidecarFor("exec_mapreduce", kMapreduceArgs),
                                       HOST_PLUGIN_EXECV2_SO));
        reg.bind_handle("x", x.data());
        reg.bind_handle("partial", partial.data());
        auto packed = reg.pack(0, N);
        ex::HostTeam::run_serial(reg.entry_v2(0), packed.params.data(), whole);
        wholeVal = ex::fold(ex::ReduceOp::Sum, partial.data(), std::size_t(N));
    }

    // The DISTRIBUTED fold: this rank's partial, then the rank-ordered combine.
    std::vector<double> partial(std::size_t(N), 0.0);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sidecarFor("exec_mapreduce", kMapreduceArgs),
                                   HOST_PLUGIN_EXECV2_SO));
    ASSERT_NO_THROW(RP::check(ex::Access::MapReduce, reg.abi_version(0),
                              MPI_COMM_WORLD));
    reg.bind_handle("x", x.data());
    reg.bind_handle("partial", partial.data());
    auto packed = reg.pack(0, N);
    RP::run(reg.entry_v2(0), packed.params, whole, MPI_COMM_WORLD);

    const double mineVal = ex::fold(ex::ReduceOp::Sum,
                                    partial.data() + mine.base,
                                    std::size_t(mine.count));
    const double splitVal = RP::allgather_fold(ex::ReduceOp::Sum, mineVal,
                                               MPI_COMM_WORLD);

    EXPECT_TRUE(withinBand(wholeVal, splitVal, std::size_t(N)));
    EXPECT_TRUE(identicalOnAllRanks(&splitVal, 1))
        << "the folded result differs between ranks — the combine order is not fixed";
    // NON-VACUITY: the band must not be so wide that losing a whole rank's
    // contribution still passes.
    EXPECT_FALSE(withinBand(wholeVal, mineVal, std::size_t(N)))
        << "the band accepts a result missing an entire rank's partial";
}

// ---------------------------------------------------------------------------
// L2 — the split is contiguous, uneven-tolerant, and covers every sample once
// ---------------------------------------------------------------------------
TEST(RankPartition, UnevenSplitCoversEverySample)
{
    // (a) The rule as a PURE function, enumerated — not sampled — over sizes that
    // do and do not divide evenly. This is the part that has to hold for world
    // sizes the bed itself never runs at.
    for (int R : { 1, 2, 3, 5, 7 }) {
        for (std::int64_t n : { std::int64_t(0), std::int64_t(1), std::int64_t(7),
                                std::int64_t(1001), std::int64_t(4095) }) {
            const ex::Partition whole = ex::Partition::whole(n);
            std::int64_t covered = 0, expectedBase = 0;
            std::int64_t lo = n + 1, hi = -1;
            for (int r = 0; r < R; ++r) {
                const ex::Partition p = RP::local(whole, r, R);
                ASSERT_EQ(p.base, expectedBase)
                    << "R=" << R << " n=" << n << " rank " << r << " is not contiguous";
                ASSERT_GE(p.count, 0);
                ASSERT_EQ(p.nSamples, n) << "nSamples must be the TRUE total (L2)";
                covered += p.count;
                expectedBase = p.base + p.count;
                lo = std::min(lo, p.count);
                hi = std::max(hi, p.count);
            }
            ASSERT_EQ(covered, n) << "R=" << R << " n=" << n << ": coverage gap/overlap";
            ASSERT_EQ(expectedBase, n) << "R=" << R << " n=" << n << ": ends short of n";
            // Uneven, but never by more than one sample.
            ASSERT_LE(hi - lo, 1) << "R=" << R << " n=" << n << ": unbalanced split";
        }
    }
    EXPECT_THROW(RP::local(ex::Partition::whole(4), 0, 0), std::runtime_error);
    EXPECT_THROW(RP::local(ex::Partition::whole(4), 2, 2), std::runtime_error);

    // (b) The same rule LIVE, at a sample count the world size does not divide: the
    // gathered plane must have every sample written exactly once, by whichever rank
    // owns it.
    constexpr std::int64_t N = 1001;   // 1001 = 2*500 + 1
    const ex::Partition whole = ex::Partition::whole(N);
    const ex::Partition mine  = RP::local(whole, MPI_COMM_WORLD);
    ASSERT_EQ(RP::comm_size(MPI_COMM_WORLD), 2);
    // Literal, not derived: at N=1001 over 2 ranks the split is 501 | 500.
    EXPECT_EQ(mine.count, worldRank() == 0 ? 501 : 500);
    EXPECT_EQ(mine.base,  worldRank() == 0 ? 0   : 501);

    std::vector<double> x(static_cast<std::size_t>(N));
    for (std::int64_t i = 0; i < N; ++i) x[std::size_t(i)] = double(i) + 0.5;
    std::vector<double> y(std::size_t(N), kSentinel);

    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(sidecarFor("exec_local", kLocalArgs),
                                   HOST_PLUGIN_EXECV2_SO));
    reg.bind_handle("x", x.data());
    reg.bind_handle("y", y.data());
    reg.bind_uniform("a", 1.0);
    reg.bind_uniform("b", 0.0);
    auto packed = reg.pack(0, N);
    RP::run(reg.entry_v2(0), packed.params, whole, MPI_COMM_WORLD);
    RP::allgather_plane(y.data(), whole, MPI_COMM_WORLD);

    for (std::int64_t i = 0; i < N; ++i)
        ASSERT_DOUBLE_EQ(y[std::size_t(i)], x[std::size_t(i)])
            << "sample " << i << " was never written by any rank (uneven split lost it)";
    EXPECT_TRUE(identicalOnAllRanks(y.data(), y.size()));
}

// ---------------------------------------------------------------------------
// F-e — cross_sample_write is REFUSED under RankPartition, naming the rule
// ---------------------------------------------------------------------------
TEST(RankPartition, CrossSampleWriteRefused)
{
    try {
        RP::check(ex::Access::CrossSampleWrite, 2, MPI_COMM_WORLD);
        FAIL() << "expected cross_sample_write to be refused under RankPartition";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("cross_sample_write"), std::string::npos) << msg;
        EXPECT_NE(msg.find("rank_partition"), std::string::npos) << msg;
        EXPECT_NE(msg.find("F-e"), std::string::npos) << msg;
    }
    // The same rule reached through the placement surface the plan layer uses, and
    // at the live world size rather than a literal.
    EXPECT_THROW(ex::check_placement(ex::Access::CrossSampleWrite,
                                     ex::Structure::RankPartition,
                                     RP::comm_size(MPI_COMM_WORLD)),
                 std::runtime_error);
    EXPECT_THROW(ex::check_placement("cross_sample_write", "rank_partition", 2),
                 std::runtime_error);
    // The classes that ARE legal here stay legal — a refusal that refused
    // everything would pass this row just as well.
    EXPECT_NO_THROW(RP::check(ex::Access::SampleLocal, 2, MPI_COMM_WORLD));
    EXPECT_NO_THROW(RP::check(ex::Access::CrossSampleRead, 2, MPI_COMM_WORLD));
    EXPECT_NO_THROW(RP::check(ex::Access::MapReduce, 2, MPI_COMM_WORLD));
}

// ---------------------------------------------------------------------------
// A legacy aether-abi/1 plugin may not be driven across ranks
// ---------------------------------------------------------------------------
TEST(RankPartition, LegacyV1Refused)
{
    // (a) The structural rule, refused before anything is launched.
    try {
        RP::check(ex::Access::SampleLocal, 1, MPI_COMM_WORLD);
        FAIL() << "expected a legacy aether-abi/1 plugin to be refused under "
                  "RankPartition";
    } catch (const std::runtime_error& e) {
        const std::string msg = e.what();
        EXPECT_NE(msg.find("aether-abi/1"), std::string::npos) << msg;
        EXPECT_NE(msg.find("L13"), std::string::npos) << msg;
        EXPECT_NE(msg.find("rank_partition"), std::string::npos) << msg;
    }

    // (b) And the same refusal reached through the product path, with a real v1
    // artifact loaded: there is no v2 entry to hand an inner structure, and the
    // registry itself refuses this rank's non-whole triple.
    constexpr std::int64_t N = 64;
    std::vector<double> x(std::size_t(N), 2.0), y(std::size_t(N), 0.0);
    eagle::cpu::PluginRegistry reg;
    ASSERT_NO_THROW(reg.add_plugin(
        sidecarFor("exec_legacy", kLegacyArgs, "aether-abi/1", 1),
        HOST_PLUGIN_EXECV2_SO));
    ASSERT_EQ(reg.abi_version(0), 1);
    reg.bind_handle("x", x.data());
    reg.bind_handle("y", y.data());
    ASSERT_EQ(reg.run(std::int32_t(N)), 1);      // the whole-view bridge still works
    EXPECT_DOUBLE_EQ(y[0], 3.0);
    EXPECT_THROW(reg.entry_v2(0), std::runtime_error);
    try {
        reg.run_partition(RP::local(ex::Partition::whole(N), MPI_COMM_WORLD));
        FAIL() << "expected a legacy plugin to refuse this rank's sub-partition";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("WHOLE-VIEW"), std::string::npos)
            << e.what();
    }
}
