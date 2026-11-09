// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* eagle's host pool and stage sizing: bitwise equality with a serial run at every
 * team size, the floor that keeps light stages in place, the thread budget, and the
 * safety rows (no nesting, concurrent callers, a forked child). Every row runs under
 * a deadline, so a hang fails the row instead of holding the cores. CUDA-free;
 * compiled by CXX in BOTH build modes with EAGLE_CPU_ONLY forced on this one source. */
#include <gtest/gtest.h>

#include "eagle/exec/HostPool.h"
#include "eagle/exec/HostSchedule.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#endif

#if defined(__unix__)
#include <sys/wait.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

namespace eagle_tests {
namespace HostPoolTest {

using namespace eagle::exec;
using namespace eagle::exec::detail;

namespace {

// Ends the process when a row outlives its deadline: a hang is a failure, never a stall.
class Deadline {
public:
    explicit Deadline(int seconds) : done_(std::make_shared<std::atomic<bool>>(false)) {
        std::thread([done = done_, seconds] {
            for (int i = 0; i < seconds * 20 && !done->load(); ++i)
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            if (!done->load()) {
                std::fprintf(stderr, "HostPoolTest: deadline of %d s passed (hang)\n", seconds);
                std::_Exit(124);
            }
        }).detach();
    }
    ~Deadline() { done_->store(true); }

private:
    std::shared_ptr<std::atomic<bool>> done_;
};

// One node: out[i] = f(in[i]) with `work` fma rounds per sample over [base, base+count).
struct Planes { const double* in; double* out; long work; };

void node(void* const* params, std::int64_t base, std::int64_t count, std::int64_t)
{
    const auto* p = static_cast<const Planes*>(params[0]);
    for (std::int64_t i = base; i < base + count; ++i) {
        double x = p->in[i];
        for (long k = 0; k < p->work; ++k) x = std::fma(x, 0.999999, 1e-7);
        p->out[i] = x * 2.0 + 1.0;
    }
}

// `branches` independent chains of `depth` node pairs ping-ponging between two planes.
struct Fork {
    std::vector<std::vector<double>> a, b;
    std::vector<Planes> ab, ba;
    Fork(int branches, std::int64_t n, long work) : a(branches), b(branches), ab(branches), ba(branches)
    {
        for (int k = 0; k < branches; ++k) {
            a[k].resize(n);
            b[k].resize(n);
            ab[k] = { a[k].data(), b[k].data(), work };
            ba[k] = { b[k].data(), a[k].data(), work };
        }
        reset();
    }
    void reset()   // in place: a built stage keeps pointing at these planes
    {
        for (std::size_t k = 0; k < a.size(); ++k)
            for (std::size_t i = 0; i < a[k].size(); ++i) {
                a[k][i] = 0.001 * static_cast<double>(i) + static_cast<double>(k);
                b[k][i] = 0.0;
            }
    }
    std::vector<HostBranch> branches(int depth) const
    {
        std::vector<HostBranch> out;
        for (std::size_t k = 0; k < a.size(); ++k) {
            HostBranch br;
            for (int d = 0; d < 2 * depth; ++d)
                br.push_back({ &node, { const_cast<Planes*>(d % 2 ? &ba[k] : &ab[k]) },
                               { Partition::whole(static_cast<std::int64_t>(a[k].size())) } });
            out.push_back(std::move(br));
        }
        return out;
    }
    void runSerial(int depth) const
    {
        for (const HostBranch& br : branches(depth))
            for (const HostBranchNode& nd : br) HostTeam::run_serial(nd.entry, nd.params.data(), nd.parts[0]);
    }
};

// Restores the process-wide thread setting when a row ends.
struct Threads {
    int saved = host_threads_setting().load();
    explicit Threads(int n) { host_threads_setting().store(n); }
    ~Threads() { host_threads_setting().store(saved); }
};

}  // namespace

TEST(HostPoolTest, EveryTeamEqualsASerialRunBitwise)
{
    Deadline deadline(120);
    for (const std::int64_t n : { std::int64_t{ 1 }, std::int64_t{ 1024 }, std::int64_t{ 10000 } })
        for (const int team : { 1, 2, host_default_threads() })
            for (const bool tasks : { true, false }) {
                Threads threads(team);
                Fork want(9, n, 200), got(9, n, 200);
                want.runSerial(2);
                HostStage stage(got.branches(2), tasks);
                for (int launch = 0; launch < 3; ++launch) {   // launches 1-2 in place, then sized
                    got.reset();
                    stage.run();
                }
                for (std::size_t k = 0; k < 9; ++k)
                    EXPECT_EQ(want.a[k], got.a[k]) << "n=" << n << " team=" << team << " tasks=" << tasks;
            }
}

TEST(HostPoolTest, TheFloorKeepsLightStagesInPlace)
{
    Deadline deadline(60);
    Threads threads(4);
    Fork light(9, 1, 0), heavy(9, 1, 20000), wide(1, 10000, 2000);
    HostStage lightStage(light.branches(1), true), heavyStage(heavy.branches(1), true),
              wideStage(wide.branches(1), false);
    for (HostStage* s : { &lightStage, &heavyStage, &wideStage }) {
        EXPECT_EQ(s->lanes(), 1);   // launch 1 runs in place
        s->run();
        s->run();
        s->run();
    }
    EXPECT_EQ(lightStage.lanes(), 1);
    EXPECT_GT(heavyStage.lanes(), 1);
    EXPECT_GT(wideStage.lanes(), 1);   // a compute-heavy node is cut by its cost, not by bytes alone
    EXPECT_LE(heavyStage.lanes(), 4);
}

namespace {
std::mutex g_threads_mu;
std::set<std::thread::id> g_threads;

// Notes the calling thread, then spins for `params[0]` microseconds (none when it is null).
void noteThread(void* const* params, std::int64_t, std::int64_t, std::int64_t)
{
    {
        std::lock_guard<std::mutex> lock(g_threads_mu);
        g_threads.insert(std::this_thread::get_id());
    }
    if (params == nullptr || params[0] == nullptr) return;
    const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(200);
    while (std::chrono::steady_clock::now() < end) {}
}

// A stage of 16 tiles (`work_share`) or 16 branches (`branch_tasks`) of `noteThread`, spinning when @p heavy.
HostStage noteStage(bool tasks, bool heavy, void*& spin)
{
    std::vector<void*> params{ heavy ? &spin : nullptr };
    std::vector<HostBranch> branches;
    if (tasks)
        for (int k = 0; k < 16; ++k) branches.push_back(HostBranch{ { &noteThread, params, { Partition::whole(1) } } });
    else
        branches.push_back(HostBranch{ { &noteThread, params, { Partition::whole(16 * 256) } } });
    return HostStage(std::move(branches), tasks);
}

std::size_t threadsOfLaunch(HostStage& stage)
{
    g_threads.clear();
    stage.run();
    return g_threads.size();
}
}  // namespace

TEST(HostPoolTest, AHeavyStageIsSharedFromItsFirstLaunch)
{
    Deadline deadline(60);
    Threads threads(4);
    void* spin = &spin;
    for (const bool tasks : { false, true }) {
        HostStage stage = noteStage(tasks, true, spin);
        EXPECT_GT(threadsOfLaunch(stage), 1u) << "tasks=" << tasks << " launch 1";
        EXPECT_GT(threadsOfLaunch(stage), 1u) << "tasks=" << tasks << " launch 2";
        EXPECT_GT(stage.lanes(), 1) << "tasks=" << tasks;
    }
}

TEST(HostPoolTest, ACheapStageNeverLeavesTheCaller)
{
    Deadline deadline(60);
    Threads threads(4);
    void* spin = nullptr;
    for (const bool tasks : { false, true }) {
        HostStage stage = noteStage(tasks, false, spin);
        for (int launch = 1; launch <= 4; ++launch)
            EXPECT_EQ(threadsOfLaunch(stage), 1u) << "tasks=" << tasks << " launch " << launch;
        EXPECT_EQ(stage.lanes(), 1);
    }
}

namespace {
std::atomic<std::int64_t> g_smallest_call{ 0 };

// A heavy node that notes the smallest range it was handed.
void noteRange(void* const* params, std::int64_t base, std::int64_t count, std::int64_t)
{
    auto* out = static_cast<double*>(params[0]);
    for (std::int64_t i = base; i < base + count; ++i) {
        double x = 1.0;
        for (int k = 0; k < 2000; ++k) x = std::fma(x, 0.999999, 1e-7);   // ~1 us a sample: the stage sizes past
        out[i] = x;                                                         // the 256-sample tile cap
    }
    std::int64_t seen = g_smallest_call.load();
    while (count < seen && !g_smallest_call.compare_exchange_weak(seen, count)) {}
}
}  // namespace

TEST(HostPoolTest, AHeavyNodeIsCutIntoTilesNotPackets)
{
    Deadline deadline(60);
    Threads threads(4);
    const std::int64_t n = 1 << 14;
    std::vector<double> out(n);
    HostStage stage({ HostBranch{ { &noteRange, { out.data() }, { Partition::whole(n) } } } }, false);
    stage.run();   // launch 1: in place, cold
    stage.run();   // launch 2: in place, measured
    ASSERT_GT(stage.lanes(), 1);
    g_smallest_call.store(n);
    stage.run();
    // sized for a lane per packet, the tiles would be one packet (8 samples) each
    EXPECT_GE(g_smallest_call.load(), std::min<std::int64_t>(256, n / physical_cores()));
}

namespace {
std::atomic<int> g_calls{ 0 };

// Its first call (the cold one) takes 200 us; later calls return at once.
void slowFirst(void* const*, std::int64_t, std::int64_t, std::int64_t)
{
    if (g_calls.fetch_add(1) == 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
}
}  // namespace

TEST(HostPoolTest, ASlowFirstCallDoesNotSetTheTeam)
{
    Deadline deadline(60);
    Threads threads(4);
    g_calls.store(0);
    std::vector<HostBranch> branches;
    for (int k = 0; k < 9; ++k) branches.push_back(HostBranch{ { &slowFirst, {}, { Partition::whole(1) } } });
    HostStage stage(std::move(branches), true);
    for (int launch = 0; launch < 3; ++launch) stage.run();
    EXPECT_EQ(stage.lanes(), 1);   // launch 2 measured the steady cost, not the cold one
}

namespace {
// A steady entry: nine of them make a ~120 us stage.
void worthAWake(void* const*, std::int64_t, std::int64_t, std::int64_t)
{
    const auto end = std::chrono::steady_clock::now() + std::chrono::microseconds(13);
    while (std::chrono::steady_clock::now() < end) {}
}
}  // namespace

TEST(HostPoolTest, AStageWorthAWakeTakesItsWholeBudget)
{
    Deadline deadline(60);
    Threads threads(4);
    std::vector<HostBranch> branches;
    for (int k = 0; k < 9; ++k) branches.push_back(HostBranch{ { &worthAWake, {}, { Partition::whole(1) } } });
    HostStage stage(std::move(branches), true);
    for (int launch = 0; launch < 3; ++launch) stage.run();
    EXPECT_EQ(stage.lanes(), host_max_threads());   // 120 us is past 2 * HOST_FLOOR_US: not the 3 that us / floor gives
}

#if defined(__linux__)
namespace {
std::mutex g_masks_mu;
std::map<std::thread::id, cpu_set_t> g_masks;
std::atomic<int> g_started{ 0 };

// Notes this thread's affinity, and holds the lane until every lane of the job has started (or a second passes),
// so each lane runs on a thread of its own.
void noteMask(const void* ctx, std::uint32_t)
{
    cpu_set_t mask;
    sched_getaffinity(0, sizeof(mask), &mask);
    {
        std::lock_guard<std::mutex> lock(g_masks_mu);
        g_masks[std::this_thread::get_id()] = mask;
    }
    const int all = *static_cast<const int*>(ctx);
    g_started.fetch_add(1);
    const auto end = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (g_started.load() < all && std::chrono::steady_clock::now() < end) {}
}

// The affinity masks of the workers (not the caller) that ran a job of @p lanes lanes on a team of @p lanes.
std::map<std::thread::id, cpu_set_t> workerMasks(int lanes)
{
    g_masks.clear();
    g_started.store(0);
    HostPool::Job job;
    job.lanes = static_cast<std::uint32_t>(lanes);
    job.run = &noteMask;
    job.ctx = &lanes;
    HostPool::instance().run(job, lanes);
    auto masks = g_masks;
    masks.erase(std::this_thread::get_id());
    return masks;
}
}  // namespace

TEST(HostPoolTest, EachWorkerHasOnePhysicalCoreToItself)
{
    Deadline deadline(60);
    const int cores = physical_cores();
    if (cores < 3) GTEST_SKIP() << "needs three physical cores";
    const std::vector<cpu_set_t> sets = core_masks();
    ASSERT_EQ(static_cast<int>(sets.size()), cores);
    cpu_set_t whole;
    sched_getaffinity(0, sizeof(whole), &whole);
    auto masks = workerMasks(cores);
    // workers past the cores (made by other tests) keep the whole mask: the pinned ones are what this checks
    for (auto it = masks.begin(); it != masks.end();)
        it = CPU_EQUAL(&whole, &it->second) ? masks.erase(it) : std::next(it);
    ASSERT_GE(masks.size(), 2u);
    for (auto a = masks.begin(); a != masks.end(); ++a) {
        const bool isCore = std::any_of(sets.begin(), sets.end(), [&](const cpu_set_t& s) { return CPU_EQUAL(&s, &a->second); });
        EXPECT_TRUE(isCore) << "a worker's mask is not one core's sibling set";
        for (auto b = std::next(a); b != masks.end(); ++b) {
            cpu_set_t both;
            CPU_AND(&both, &a->second, &b->second);
            EXPECT_EQ(CPU_COUNT(&both), 0) << "two workers share a physical core";
        }
    }
}

TEST(HostPoolTest, AWorkerBeyondTheCoresKeepsTheWholeMask)
{
    Deadline deadline(60);
    const int cores = physical_cores();
    if (cores < 3) GTEST_SKIP() << "needs three physical cores";
    cpu_set_t whole;
    sched_getaffinity(0, sizeof(whole), &whole);
    const auto masks = workerMasks(cores + 1);
    const bool any = std::any_of(masks.begin(), masks.end(), [&](const auto& m) { return CPU_EQUAL(&whole, &m.second); });
    EXPECT_TRUE(any) << "no worker ran with the whole mask";
}

TEST(HostPoolTest, ALaunchWakesOnlyTheWorkersOfItsTeam)
{
    Deadline deadline(60);
    constexpr int big = 6, small = 3;   // workers 0..1 are the small team's; 2..4 must sleep through its launches
    HostPool& pool = HostPool::instance();
    workerMasks(big);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));   // every worker asleep
    std::uint64_t before[big];
    for (int k = small - 1; k < big - 1; ++k) before[k] = pool.wakeups(k);
    for (int i = 0; i < 20; ++i) {
        workerMasks(small);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    for (int k = small - 1; k < big - 1; ++k)
        EXPECT_EQ(pool.wakeups(k), before[k]) << "worker " << k << " was woken by a team of " << small;
}

TEST(HostPoolTest, AfterABiggerTeamASmallerOneStillHasACorePerWorker)
{
    Deadline deadline(60);
    const int cores = physical_cores();
    if (cores < 3) GTEST_SKIP() << "needs three physical cores";
    const std::vector<cpu_set_t> sets = core_masks();
    workerMasks(2 * cores);   // the pool now holds more workers than the cores
    const auto masks = workerMasks(cores);
    ASSERT_GE(masks.size(), 2u);
    for (auto a = masks.begin(); a != masks.end(); ++a) {
        const bool isCore = std::any_of(sets.begin(), sets.end(), [&](const cpu_set_t& s) { return CPU_EQUAL(&s, &a->second); });
        EXPECT_TRUE(isCore) << "a worker of the smaller team is not on one core's sibling set";
        for (auto b = std::next(a); b != masks.end(); ++b) {
            cpu_set_t both;
            CPU_AND(&both, &a->second, &b->second);
            EXPECT_EQ(CPU_COUNT(&both), 0) << "two workers share a physical core";
        }
    }
}
#endif

TEST(HostPoolTest, TheBudgetFollowsTheSettingTheEnvironmentAndTheCores)
{
    EXPECT_EQ(detail::cgroup_cpu_cap("max 100000"), 0);
    EXPECT_EQ(detail::cgroup_cpu_cap("250000 100000"), 3);
    EXPECT_EQ(detail::cgroup_cpu_cap("100000 100000"), 1);
    EXPECT_EQ(detail::cgroup_cpu_cap(""), 0);
    EXPECT_GE(physical_cores(), 1);
    EXPECT_LE(physical_cores(), static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
    EXPECT_LE(host_default_threads(), physical_cores());
    const char* saved = std::getenv("OMP_NUM_THREADS");
    const std::string keep = saved ? saved : "";
    ::setenv("OMP_NUM_THREADS", "3,2", 1);
    EXPECT_EQ(detail::omp_num_threads_env(), 3);
    ::setenv("OMP_NUM_THREADS", "zero", 1);
    EXPECT_EQ(detail::omp_num_threads_env(), 0);
    if (saved) ::setenv("OMP_NUM_THREADS", keep.c_str(), 1); else ::unsetenv("OMP_NUM_THREADS");
    Threads threads(2);
    EXPECT_EQ(host_max_threads(), 2);
}

namespace {
std::atomic<int> g_inner_runs{ 0 };

// A lane that launches a whole stage itself: from a worker and from the launching thread. A launch
// from inside a lane runs in place, so planes per thread keep the lanes race-free.
void launchInside(void* const*, std::int64_t, std::int64_t, std::int64_t)
{
    thread_local Fork planes(1, 64, 0);
    HostStage inner(planes.branches(1), true);
    inner.run();
    runHostBranches(planes.branches(1));
    g_inner_runs.fetch_add(1);
}
}  // namespace

TEST(HostPoolTest, ALaunchFromInsideALaneRunsInPlace)
{
    Deadline deadline(60);
    Threads threads(4);
    std::vector<HostBranch> outer(9, HostBranch{ { &launchInside, {}, { Partition::whole(1) } } });
    for (int i = 0; i < 1000; ++i) runHostBranches(outer);
    EXPECT_EQ(g_inner_runs.load(), 9000);
#ifdef _OPENMP
    Fork want(4, 512, 50), got(4, 512, 50);
    want.runSerial(1);
#pragma omp parallel num_threads(2)
    {
#pragma omp masked   // the main thread: its write and the check below are ordered without the runtime's barrier
        runHostBranches(got.branches(1));   // inside a caller's region: in place, no second team
    }
    for (std::size_t k = 0; k < 4; ++k) EXPECT_EQ(want.a[k], got.a[k]);
#endif
}

TEST(HostPoolTest, ConcurrentCallersEqualASerialRunBitwise)
{
    Deadline deadline(300);
    Threads threads(host_default_threads());
    constexpr int callers = 4, launches = 10000;
    for (const std::int64_t n : { std::int64_t{ 1 }, std::int64_t{ 10000 } })
        for (const bool tasks : { true, false }) {
            const int reps = launches;
            std::vector<Fork> want, got;
            for (int t = 0; t < callers; ++t) {
                want.emplace_back(tasks ? 4 : 1, n, 0);
                got.emplace_back(tasks ? 4 : 1, n, 0);
                for (int r = 0; r < reps; ++r) want.back().runSerial(1);
            }
            std::vector<std::thread> pool;
            for (int t = 0; t < callers; ++t)
                pool.emplace_back([&, t] {
                    const auto branches = got[t].branches(1);
                    for (int r = 0; r < reps; ++r)
                        tasks ? runHostBranches(branches) : runHostStage(branches);
                });
            for (std::thread& th : pool) th.join();
            for (int t = 0; t < callers; ++t)
                for (std::size_t k = 0; k < got[t].a.size(); ++k)
                    EXPECT_EQ(want[t].a[k], got[t].a[k]) << "n=" << n << " tasks=" << tasks << " caller=" << t;
        }
}

#if defined(__unix__)
TEST(HostPoolTest, AForkedChildLaunchesOnAFreshPool)
{
    Deadline deadline(60);
    Threads threads(4);
    Fork warm(9, 1, 20000);
    runHostBranches(warm.branches(1));   // the parent's pool is up
    std::atomic<bool> stop{ false };
    std::atomic<long> parentLaunches{ 0 };
    std::thread hammer([&] {
        Fork mine(9, 1, 3000);
        const auto branches = mine.branches(1);
        while (!stop.load()) { runHostBranches(branches); parentLaunches.fetch_add(1); }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    const pid_t child = ::fork();
    if (child == 0) {
        Fork want(9, 1, 3000), got(9, 1, 3000);
        for (int i = 0; i < 1000; ++i) { want.runSerial(1); runHostBranches(got.branches(1)); }
        std::_Exit(want.a == got.a ? 0 : 3);
    }
    int status = -1;
    ::waitpid(child, &status, 0);
    stop.store(true);
    hammer.join();
    EXPECT_TRUE(WIFEXITED(status) && WEXITSTATUS(status) == 0) << status;
    EXPECT_GT(parentLaunches.load(), 0);
}
#endif

}  // namespace HostPoolTest
}  // namespace eagle_tests
