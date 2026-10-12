// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Distinct objects used by many threads at once give the serial result bit for
 * bit: the host branch runner, the window driver, the record slot's
 * hold/release count and the event refine loop. CUDA-free; compiled by CXX in
 * BOTH build modes. */
#include <gtest/gtest.h>

#include "eagle/diff/Record.h"
#include "eagle/diff/Reverse.h"
#include "eagle/events/Sensitivity.h"
#include "eagle/exec/HostSchedule.h"
#include "eagle/exec/HostTeam.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace eagle_tests {
namespace ConcurrencyHostTest {

constexpr int kThreads = 8;

template <class F>
void inThreads(F&& body)
{
    std::atomic<int> ready { 0 };
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t)
        threads.emplace_back([&, t] {
            ready.fetch_add(1);
            while (ready.load() < kThreads) std::this_thread::yield();
            body(t);
        });
    for (auto& t : threads) t.join();
}

namespace schedule {

using namespace eagle::exec;

struct Planes { const double* in; double* out; };

void step(void* const* params, std::int64_t base, std::int64_t count, std::int64_t)
{
    const auto* p = static_cast<const Planes*>(params[0]);
    for (std::int64_t i = base; i < base + count; ++i) p->out[i] = p->in[i] * 2.0 + 1.0;
}

struct Fork {
    std::vector<std::vector<double>> a, b;
    std::vector<Planes> ab, ba;
    Fork(int branches, int n, int salt) : a(branches), b(branches), ab(branches), ba(branches)
    {
        for (int k = 0; k < branches; ++k) {
            a[k].resize(n);
            b[k].assign(n, 0.0);
            for (int i = 0; i < n; ++i) a[k][i] = 0.001 * i + k + salt;
            ab[k] = { a[k].data(), b[k].data() };
            ba[k] = { b[k].data(), a[k].data() };
        }
    }
    std::vector<HostBranch> branches(int depth, std::int64_t n) const
    {
        std::vector<HostBranch> out;
        for (std::size_t k = 0; k < a.size(); ++k) {
            HostBranch br;
            for (int d = 0; d < 2 * depth; ++d)
                br.push_back({ &step, { const_cast<Planes*>(d % 2 ? &ba[k] : &ab[k]) },
                               { Partition::whole(n) } });
            out.push_back(std::move(br));
        }
        return out;
    }
};

}  // namespace schedule

TEST(ConcurrencyHost, DistinctBranchRunnersEqualTheSerialRunBitwise)
{
    const int n = 1000, depth = 3;
    std::vector<schedule::Fork> want, got;
    for (int t = 0; t < kThreads; ++t) {
        want.emplace_back(4, n, t);
        got.emplace_back(4, n, t);
        for (const eagle::exec::HostBranch& branch : want.back().branches(depth, n))
            for (const eagle::exec::HostBranchNode& node : branch)
                eagle::exec::HostTeam::run_serial(node.entry, node.params.data(), node.parts[0]);
    }
    inThreads([&](int t) {
        for (int rep = 0; rep < 3; ++rep) {
            got[t] = schedule::Fork(4, n, t);
            eagle::exec::runHostBranches(got[t].branches(depth, n));
        }
    });
    for (int t = 0; t < kThreads; ++t)
        for (std::size_t k = 0; k < 4; ++k) EXPECT_EQ(want[t].a[k], got[t].a[k]) << t << "," << k;
}

TEST(ConcurrencyHost, DistinctWindowDriversLogTheSerialCallOrder)
{
    using namespace eagle::diff;
    auto drive = [](int t) {
        std::vector<idx_t> steps { 5 + t, 2, 7 };
        const std::vector<idx_t> live = liveCounts(steps.data(), 3);
        std::vector<std::string> log;
        reverseWindows(
            3, live,
            [&](idx_t k, idx_t c) { log.push_back("g" + std::to_string(k) + ":" + std::to_string(c)); },
            [&](idx_t c, idx_t len) { log.push_back("f" + std::to_string(c) + "+" + std::to_string(len)); },
            [&](idx_t j, idx_t c) { log.push_back("b" + std::to_string(j) + ":" + std::to_string(c)); });
        return log;
    };
    std::vector<std::vector<std::string>> want(kThreads), got(kThreads);
    for (int t = 0; t < kThreads; ++t) want[t] = drive(t);
    inThreads([&](int t) {
        for (int rep = 0; rep < 200; ++rep) got[t] = drive(t);
    });
    EXPECT_EQ(want, got);
}

TEST(ConcurrencyHost, RecordSlotHoldAndReleaseLoseNoReader)
{
    using eagle::diff::RecordSlot;
    RecordSlot slot;
    ASSERT_EQ(slot.begin(), RecordSlot::Begin::Ok);
    inThreads([&](int) {
        for (int i = 0; i < 2000; ++i) {
            ASSERT_TRUE(slot.hold(1));
            ASSERT_TRUE(slot.release(1, false));
        }
    });
    EXPECT_FALSE(slot.held());
    EXPECT_EQ(slot.begin(), RecordSlot::Begin::Live);
}

TEST(ConcurrencyHost, DistinctRefineLoopsEqualTheSerialRunBitwise)
{
    using namespace eagle::events;
    struct Run {
        std::vector<double> g0, dg0, g1, dg1, t0, h, tau, lo, hi, tStop, r;
        std::vector<std::uint8_t> term;
        std::vector<std::int64_t> applied;
        std::vector<std::int8_t> status, which;
        explicit Run(idx_t n, int salt)
            : g0(n), dg0(n), g1(n), dg1(n), t0(n, 0.0), h(n, 0.1), tau(n), lo(n), hi(n), tStop(n),
              r(n), term(n, 1), applied(n, 3), status(n), which(n)
        {
            for (idx_t i = 0; i < n; ++i) r[i] = 0.01 + 0.0013 * double(i) + 0.0001 * salt;
        }
        void go()
        {
            const idx_t n = idx_t(r.size());
            EventPlanes<double> p;
            p.n = n;
            p.guards = 1;
            p.g[0] = g0.data();   p.dg[0] = dg0.data();
            p.g[1] = g1.data();   p.dg[1] = dg1.data();
            p.terminated = term.data(); p.applied = applied.data();
            p.t0 = t0.data(); p.h = h.data(); p.status = status.data(); p.which = which.data();
            p.tau = tau.data(); p.lo = lo.data(); p.hi = hi.data(); p.tStop = tStop.data();
            for (idx_t i = 0; i < n; ++i) tau[i] = h[i];
            const auto evaluate = [&] {
                for (idx_t i = 0; i < n; ++i) {
                    const double q = tau[i] / r[i];
                    g0[i] = 1.0 - q * q;
                    dg0[i] = -2.0 * q / r[i];
                }
            };
            evaluate();
            refineHost(p, kOpClassify);
            refineLoop(evaluate, [&](std::uint32_t op) { refineHost(p, op); },
                       [&] { return anyPending(p); });
        }
    };
    constexpr idx_t n = 64;
    std::vector<Run> want, got;
    for (int t = 0; t < kThreads; ++t) {
        want.emplace_back(n, t);
        got.emplace_back(n, t);
        want.back().go();
    }
    inThreads([&](int t) { got[t].go(); });
    for (int t = 0; t < kThreads; ++t) {
        ASSERT_EQ(0, std::memcmp(want[t].tau.data(), got[t].tau.data(), n * sizeof(double))) << t;
        EXPECT_EQ(want[t].status, got[t].status) << t;
    }
}

}  // namespace ConcurrencyHostTest
}  // namespace eagle_tests
