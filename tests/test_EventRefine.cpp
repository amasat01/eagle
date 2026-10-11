// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// eagle/events/Refine.h + Sensitivity.h on the host: classification, the
// Newton/bisection loop on a known crossing, the graze test, the rank-1 seed
// and the reverse's starting adjoint.

#include "eagle/events/Sensitivity.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace {

using namespace eagle::events;

struct Batch {
    explicit Batch(idx_t n)
        : n(n), g0(n), dg0(n), g1(n), dg1(n), term(n, 1), applied(n, 3), t0(n, 0.0), h(n, 0.1),
          status(n), which(n), tau(n), lo(n), hi(n), tStop(n) {}

    EventPlanes<double> planes(int guards = 1) {
        EventPlanes<double> p;
        p.n = n;
        p.guards = guards;
        p.g[0] = g0.data();
        p.dg[0] = dg0.data();
        p.g[1] = g1.data();
        p.dg[1] = dg1.data();
        p.terminated = term.data();
        p.applied = applied.data();
        p.t0 = t0.data();
        p.h = h.data();
        p.status = status.data();
        p.which = which.data();
        p.tau = tau.data();
        p.lo = lo.data();
        p.hi = hi.data();
        p.tStop = tStop.data();
        return p;
    }

    idx_t n;
    std::vector<double> g0, dg0, g1, dg1;
    std::vector<std::uint8_t> term;
    std::vector<std::int64_t> applied;
    std::vector<double> t0, h;
    std::vector<std::int8_t> status, which;
    std::vector<double> tau, lo, hi, tStop;
};

TEST(EventRefine, ClassifySortsEverySampleIntoOneStatus) {
    Batch b(5);
    // 0: never stopped; 1: stopped before any step; 2: both guards inside the
    // tolerance; 3: at the step's end; 4: inside the step
    b.term[0] = 0;
    b.applied[1] = 0;
    for (idx_t i = 0; i < 5; ++i) {
        b.g0[i] = -0.05;
        b.dg0[i] = -1.0;
        b.g1[i] = 1.0;
        b.dg1[i] = -1.0;
    }
    b.g1[2] = -1e-3;
    b.g0[3] = 0.0;
    auto p = b.planes(2);
    refineHost(p, kOpClassify);
    EXPECT_EQ(b.status[0], kNoEvent);
    EXPECT_EQ(b.status[1], kNoEvent);
    EXPECT_EQ(b.status[2], kTwoRules);
    EXPECT_EQ(b.status[3], kAtStep);
    EXPECT_EQ(b.tau[3], 0.1);
    EXPECT_EQ(b.status[4], kPending);
    EXPECT_EQ(b.which[4], 0);
    EXPECT_DOUBLE_EQ(b.tau[4], 0.1 - 0.05);   // the Newton step from tau = h
}

// G(tau) = 1 - (tau / r)^2 crosses zero at tau = r; the evaluation is the
// "refine entry".
TEST(EventRefine, TheLoopLocatesAKnownCrossingWithinTheTimeRounding) {
    const idx_t n = 64;
    Batch b(n);
    std::vector<double> r(n);
    for (idx_t i = 0; i < n; ++i) r[static_cast<std::size_t>(i)] = 0.01 + 0.0013 * static_cast<double>(i);
    auto p = b.planes();
    const auto evaluate = [&] {
        for (idx_t i = 0; i < n; ++i) {
            const double q = b.tau[i] / r[i];
            b.g0[i] = 1.0 - q * q;
            b.dg0[i] = -2.0 * q / r[i];
        }
    };
    for (idx_t i = 0; i < n; ++i) b.tau[i] = b.h[i];
    evaluate();
    refineHost(p, kOpClassify);
    const int rounds = refineLoop(evaluate, [&](std::uint32_t op) { refineHost(p, op); },
                                  [&] { return anyPending(p); });
    EXPECT_LE(rounds, kMaxIterations);
    for (idx_t i = 0; i < n; ++i) {
        ASSERT_EQ(b.status[i], kRefined) << i;
        const double t = 0.2 + b.tau[i];
        EXPECT_NEAR(b.tau[i], r[i], 8.0 * spacing(t)) << i;
        EXPECT_GE(b.lo[i], 0.0);
        EXPECT_LE(b.hi[i], b.h[i]);
    }
}

TEST(EventRefine, AFlatCrossingIsAGraze) {
    Batch b(2);
    std::vector<double> gx0(2, 1.0), gx1(2, 0.0), ph0(2, 1e-3), ph1(2, 1.0);
    b.status[0] = kRefined;
    b.status[1] = kRefined;
    b.which[0] = b.which[1] = 0;
    b.dg0[0] = 1e-12;   // |dG/dtau| far below sqrt(eps) * |g_x| * |Phi_tau|
    b.dg0[1] = 0.5;
    auto p = b.planes();
    p.components = 2;
    p.b[0] = gx0.data();
    p.b[1] = gx1.data();
    p.c[0] = ph0.data();
    p.c[1] = ph1.data();
    refineHost(p, kOpGraze);
    EXPECT_EQ(b.status[0], kGraze);
    EXPECT_EQ(b.status[1], kRefined);
}

TEST(EventRefine, CommitWritesTheRefinedStateAndTheStopTime) {
    Batch b(3);
    std::vector<double> x = { 1.0, 2.0, 3.0 }, xr = { 10.0, 20.0, 30.0 };
    b.status[0] = kRefined;
    b.status[1] = kAtStep;
    b.status[2] = kNoEvent;
    b.tau[0] = 0.025;
    auto p = b.planes();
    p.components = 1;
    p.a[0] = x.data();
    p.b[0] = xr.data();
    refineHost(p, kOpCommit);
    EXPECT_EQ(x[0], 10.0);
    EXPECT_EQ(x[1], 2.0);
    EXPECT_EQ(x[2], 3.0);
    EXPECT_EQ(b.tStop[0], (0.0 + 2.0 * 0.1) + 0.025);
    EXPECT_EQ(b.tStop[1], 0.0 + 3.0 * 0.1);
    EXPECT_EQ(b.tStop[2], 0.0 + 3.0 * 0.1);
}

TEST(EventSensitivity, TheSeedIsTheRankOneCorrection) {
    Batch b(2);
    std::vector<double> xb0 = { 2.0, 2.0 }, xb1 = { 3.0, 3.0 }, ph0 = { 0.5, 0.5 }, ph1 = { 1.0, 1.0 };
    std::vector<double> s0(2, 9.0), s1(2, 9.0), gb0(2, 9.0), gb1(2, 9.0);
    b.status[0] = kRefined;
    b.status[1] = kNoEvent;
    b.which[0] = 1;
    b.dg1[0] = -4.0;
    auto p = b.planes(2);
    p.components = 2;
    p.gbar[0] = gb0.data();
    p.gbar[1] = gb1.data();
    p.a[0] = s0.data();
    p.a[1] = s1.data();
    p.b[0] = xb0.data();
    p.b[1] = xb1.data();
    p.c[0] = ph0.data();
    p.c[1] = ph1.data();
    applyHost(p, kOpSeed);
    // lambda = 2*0.5 + 3*1 = 4; gbar = -lambda / dG = 1
    EXPECT_EQ(s0[0], 2.0);
    EXPECT_EQ(s1[0], 3.0);
    EXPECT_EQ(gb0[0], 0.0);
    EXPECT_EQ(gb1[0], 1.0);
    EXPECT_EQ(s0[1], 0.0);    // no event: the refined step contributes nothing
    EXPECT_EQ(gb1[1], 0.0);
}

TEST(EventSensitivity, TheReverseStartsBeforeTheStopStepOfALocatedSample) {
    Batch b(2);
    std::vector<double> out(2), end = { 1.0, 2.0 }, pre = { 5.0, 6.0 };
    b.status[0] = kAtStep;
    b.status[1] = kNoEvent;
    auto p = b.planes();
    p.components = 1;
    p.a[0] = out.data();
    p.b[0] = end.data();
    p.c[0] = pre.data();
    applyHost(p, kOpSelect);
    EXPECT_EQ(out[0], 5.0);
    EXPECT_EQ(out[1], 2.0);
    std::vector<std::int64_t> steps = { 7, 7 };
    reverseSteps(steps.data(), b.status.data(), 2);
    EXPECT_EQ(steps[0], 6);
    EXPECT_EQ(steps[1], 7);
}

}  // namespace
