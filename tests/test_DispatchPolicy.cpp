// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* The splitting rule: a decision table over mixes x costs x devices, each
 * model term on its own, the hysteresis band's edges, a look-by-look sequence
 * and the look cadence; plus the host class sort and the host timing entry the
 * split path and its calibration run on. CUDA-free; compiled by CXX in BOTH
 * build modes. */
#include <gtest/gtest.h>

#include "eagle/exec/TimeLaunch.h"
#include "eagle/filtering/ReorderClasses.h"
#include "eagle/policy/Dispatch.h"

#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <vector>

namespace eagle_tests {
namespace DispatchPolicyTest {

using namespace eagle::policy;

namespace {

eagle::DeviceProps makeProps(int major, int minor, int sms, int clockKhz, int memKhz, int bus)
{
    eagle::DeviceProps p;
    p.name = "synthetic";
    p.cc_major = major;
    p.cc_minor = minor;
    p.sm_count = sms;
    p.clock_rate_khz = clockKhz;
    p.memory_clock_rate_khz = memKhz;
    p.memory_bus_width_bits = bus;
    return eagle::deriveFields(p);
}

const eagle::DeviceProps& datacenter() { static const auto p = makeProps(8, 0, 108, 1'410'000, 1'215'000, 5120); return p; }
const eagle::DeviceProps& consumer() { static const auto p = makeProps(8, 6, 28, 1'700'000, 9'500'000, 192); return p; }
const eagle::DeviceProps& host() { static const auto p = makeProps(0, 0, 8, 3'000'000, 2'666'000, 128); return p; }
// The measured device: a Quadro P2000 (cc 6.1, 8 SMs, fp64 at 1/32).
const eagle::DeviceProps& p2000() { static const auto p = makeProps(6, 1, 8, 1'480'000, 3'504'000, 160); return p; }

// Branch costs by name: ten plain ops; forty plain ops; thirty library calls; thirty 64-bit library calls.
BranchCost cheap() { BranchCost b; b.arith32 = 10; return b; }
BranchCost plain40() { BranchCost b; b.arith32 = 40; return b; }
BranchCost heavy() { BranchCost b; b.special32 = 30; return b; }
BranchCost heavy64() { BranchCost b; b.special64 = 30; return b; }
// The measured pair, as hawk counts it: x * -0.5 + 1 against
// exp(sin x cos x) + log(|x| + 2) + tanh x, both fp64, one plane read each.
BranchCost probeCheap() { BranchCost b; b.arith64 = 2; b.loads = 1; b.vectorised = true; return b; }
BranchCost probeHeavy() { BranchCost b; b.arith64 = 5; b.special64 = 5; b.loads = 1; b.vectorised = true; return b; }

CostVector costs(std::initializer_list<BranchCost> list, bool fusedVectorised = false)
{
    CostVector c;
    c.fusedVectorised = fusedVectorised;
    for (const BranchCost& b : list) c.branch[c.count++] = b;
    return c;
}

CostVector probe() { return costs({probeCheap(), probeHeavy()}, true); }

Mix mix(std::initializer_list<std::int64_t> counts, std::int64_t fusedSteps = 1,
        std::int64_t remainingSteps = DECISION_EVERY)
{
    Mix m;
    int j = 0;
    for (std::int64_t c : counts) m.counts[j++] = c;
    m.fusedSteps = fusedSteps;
    m.remainingSteps = remainingSteps;
    return m;
}

constexpr std::int64_t BATCH = 1 << 21;  // the measured batch
/** The measured probe at a heavy share of @p percent: the run's 400-step cap,
 *  a host loop fusing 64 steps per launch, a device run fused whole. */
Mix probeMix(std::int64_t percent, Target target)
{
    return mix({BATCH - BATCH * percent / 100, BATCH * percent / 100}, target == Target::Host ? 64 : 400, 400);
}

constexpr std::int64_t N = 1 << 20;

struct Row {
    const char* name;
    Target target;
    const eagle::DeviceProps* props;
    CostVector cost;
    Mix mix;
    Policy current;
    Policy expect;
};

}  // namespace

TEST(DispatchPolicy, DecisionTable)
{
    // "measured" rows match the paired walls of the probe on this machine (warmed,
    // 8 host threads / one P2000, N = 2^21, best of several); the others are the
    // prior only (no paired measurement here).
    const Policy sw = Policy::Switch, sg = Policy::Segmented;
    const Target H = Target::Host, D = Target::Device;
    const std::vector<Row> table = {
        // measured, host (fused loop vectorised: it costs both branches at any mix)
        { "measured host 0% heavy",            H, &host(), probe(), probeMix(0, H), sw, sg },
        { "measured host 1% heavy",            H, &host(), probe(), probeMix(1, H), sw, sg },
        { "measured host 5% heavy",            H, &host(), probe(), probeMix(5, H), sw, sg },
        { "measured host 50% heavy",           H, &host(), probe(), probeMix(50, H), sw, sg },
        { "measured host 100% heavy",          H, &host(), probe(), probeMix(100, H), sg, sw },
        // measured, device (a warp pays for every branch any lane takes)
        { "measured device 0% heavy",          D, &p2000(), probe(), probeMix(0, D), sg, sw },
        { "measured device 1% heavy",          D, &p2000(), probe(), probeMix(1, D), sw, sg },
        { "measured device 5% heavy",          D, &p2000(), probe(), probeMix(5, D), sw, sg },
        { "measured device 50% heavy",         D, &p2000(), probe(), probeMix(50, D), sw, sg },
        { "measured device 100% heavy",        D, &p2000(), probe(), probeMix(100, D), sg, sw },
        // prior only
        { "host unvectorised 1% heavy",        H, &host(), costs({cheap(), heavy()}), mix({N * 99 / 100, N / 100}, 64, 400), sg, sw },
        { "host empty mix keeps segmented",    H, &host(), probe(), mix({0, 0}), sg, sg },
        { "device four equal cheap branches",  D, &consumer(), costs({cheap(), cheap(), cheap(), cheap()}), mix({N << 4, N << 4, N << 4, N << 4}, 400, 400), sw, sw },
        { "device 64-bit heavy, consumer",     D, &consumer(), costs({cheap(), heavy64()}), mix({N / 2, N / 2}, 400, 400), sw, sg },
        { "device small batch keeps fused",    D, &datacenter(), costs({cheap(), heavy()}), mix({500, 500}), sw, sw },
        { "device short run pays its sort",    D, &datacenter(), costs({cheap(), plain40()}), mix({N / 2, N / 2}, 1, 1), sw, sw },
        { "device one branch takes all",       D, &datacenter(), costs({cheap(), plain40()}), mix({N, 0}, 400, 400), sg, sw },
    };
    for (const Row& r : table) {
        const Weights w = defaultWeights(r.target);
        HysteresisState s;
        s.current = r.current;
        EXPECT_EQ(resolve(r.cost, r.mix, *r.props, w, s), r.expect)
            << r.name << " (gain " << predictedGain(r.cost, r.mix, *r.props, w) << ")";
    }
}

TEST(DispatchPolicy, VectorisedFusedHostCostIsMixIndependent)
{
    const Weights w = defaultWeights(Target::Host);
    const double rare = predictedSeconds(probe(), probeMix(1, Target::Host), host(), w).fused;
    const double all = predictedSeconds(probe(), probeMix(100, Target::Host), host(), w).fused;
    EXPECT_DOUBLE_EQ(rare, all) << "the vectorised fused loop evaluates both branches at any mix";
    CostVector scalar = probe();
    scalar.fusedVectorised = false;
    const double scalarRare = predictedSeconds(scalar, probeMix(1, Target::Host), host(), w).fused;
    const double scalarAll = predictedSeconds(scalar, probeMix(100, Target::Host), host(), w).fused;
    EXPECT_LT(scalarRare, 0.5 * rare) << "a scalar switch loop runs each sample's own branch";
    EXPECT_GT(scalarAll, scalarRare);
}

TEST(DispatchPolicy, ADependentChainBoundsHostComputeButNotTheDevice)
{
    BranchCost one, four;   // the same 512 operations, as one chain and as four independent ones
    one.arith64 = four.arith64 = 512;
    one.chain64 = 512;
    four.chain64 = 128;
    const Mix m = mix({BATCH, 0}, 1);
    const Weights hw = defaultWeights(Target::Host);
    const double hostOne = predictedSeconds(costs({one, one}), m, host(), hw).fused;
    const double hostFour = predictedSeconds(costs({four, four}), m, host(), hw).fused;
    EXPECT_GT(hostOne, 3.0 * hostFour) << "latency-bound: the chain is the limit";
    const Weights dw = defaultWeights(Target::Device);
    EXPECT_DOUBLE_EQ(predictedSeconds(costs({one, one}), m, consumer(), dw).fused,
                     predictedSeconds(costs({four, four}), m, consumer(), dw).fused)
        << "a device interleaves the chains of its resident warps";
}

TEST(DispatchPolicy, ShortChainsLeaveHostComputeAtTheThroughputLimit)
{
    BranchCost b;
    b.arith64 = 512;
    b.chain64 = 4;
    BranchCost c = b;
    c.chain64 = 0;
    const Mix m = mix({BATCH, 0}, 1);
    const Weights w = defaultWeights(Target::Host);
    EXPECT_DOUBLE_EQ(predictedSeconds(costs({b, b}), m, host(), w).fused,
                     predictedSeconds(costs({c, c}), m, host(), w).fused);
}

TEST(DispatchPolicy, AVectorisedBranchPaysNoChain)
{
    BranchCost chained, free;
    chained.arith64 = 4;
    chained.chain64 = 4000;   // a long chain, few operations: latency-bound if scalar
    chained.vectorised = true;
    free = chained;
    free.chain64 = 0;
    const Mix m = mix({BATCH, 0}, 1);
    const Weights w = defaultWeights(Target::Host);
    EXPECT_DOUBLE_EQ(predictedSeconds(costs({chained, chained}), m, host(), w).split,
                     predictedSeconds(costs({free, free}), m, host(), w).split);
}

TEST(DispatchPolicy, AScalarBranchPaysTheFullChain)
{
    BranchCost b;
    b.arith64 = 4;
    b.chain64 = 4000;
    Weights w = defaultWeights(Target::Host);
    w.launchSeconds = 0;
    w.splitSeconds = 0;
    w.sortSeconds = 0;
    const Mix m = mix({BATCH, 0}, 1);
    const double seconds = predictedSeconds(costs({b, b}), m, host(), w).split;
    EXPECT_NEAR(seconds, static_cast<double>(BATCH) * 4000 * w.chainSeconds, 1e-9 * seconds)
        << "not divided by the vector width";
}

TEST(DispatchPolicy, ReadOnlyPlanesAreReadOnce)
{
    BranchCost readOnly;
    readOnly.loads = 8;
    readOnly.stores = 0;
    BranchCost carried = readOnly;
    carried.stores = 8;
    Weights w = defaultWeights(Target::Host);
    w.launchSeconds = 0;
    w.splitSeconds = 0;
    w.sortSeconds = 0;
    const Mix m = mix({BATCH, 0}, 1);
    const double once = predictedSeconds(costs({readOnly, readOnly}), m, host(), w).split;
    const double twice = predictedSeconds(costs({carried, carried}), m, host(), w).split;
    EXPECT_NEAR(twice / once, 2.0, 1e-9);
}

TEST(DispatchPolicy, ACostRowWithoutStoresKeepsTwoLoads)
{
    BranchCost uncounted, counted;
    uncounted.loads = 8;
    counted.loads = 8;
    counted.stores = 8;
    Weights w = defaultWeights(Target::Host);
    w.launchSeconds = 0;
    w.splitSeconds = 0;
    w.sortSeconds = 0;
    const Mix m = mix({BATCH, 0}, 1);
    EXPECT_DOUBLE_EQ(predictedSeconds(costs({uncounted, uncounted}), m, host(), w).split,
                     predictedSeconds(costs({counted, counted}), m, host(), w).split);
}

TEST(DispatchPolicy, AVectorisedHostBranchRunsWidthTimesFaster)
{
    BranchCost scalar;
    scalar.arith64 = 400;
    BranchCost vec = scalar;
    vec.vectorised = true;
    Weights w = defaultWeights(Target::Host);
    w.launchSeconds = 0;
    w.splitSeconds = 0;
    w.sortSeconds = 0;
    const Mix m = mix({BATCH, 0}, 1);
    const double split = predictedSeconds(costs({scalar, scalar}), m, host(), w).split;
    const double splitVec = predictedSeconds(costs({vec, vec}), m, host(), w).split;
    EXPECT_NEAR(split / splitVec, w.vectorWidth, 1e-9);
    Weights d = defaultWeights(Target::Device);
    EXPECT_DOUBLE_EQ(predictedSeconds(costs({scalar, scalar}), m, consumer(), d).split,
                     predictedSeconds(costs({vec, vec}), m, consumer(), d).split)
        << "the width is a host fact";
}

TEST(DispatchPolicy, StateIsMovedOncePerLaunchGathersEveryStep)
{
    const Weights w = defaultWeights(Target::Host);
    BranchCost state;   // memory-bound, carried planes only
    state.loads = 8;
    BranchCost gather;  // memory-bound, indexed reads only
    gather.gathers = 8;
    const Mix one = mix({BATCH, 0}, 1, 400), many = mix({BATCH, 0}, 16, 400);
    const double stateOne = predictedSeconds(costs({state, state}), one, host(), w).fused;
    const double stateMany = predictedSeconds(costs({state, state}), many, host(), w).fused;
    EXPECT_NEAR(stateOne / stateMany, 16.0, 0.5) << "state traffic divides by the steps a launch fuses";
    const double gatherOne = predictedSeconds(costs({gather, gather}), one, host(), w).fused;
    const double gatherMany = predictedSeconds(costs({gather, gather}), many, host(), w).fused;
    EXPECT_NEAR(gatherOne / gatherMany, 1.0, 0.05) << "a gather is paid every step";
}

TEST(DispatchPolicy, GatherIsFasterWhenThePlaneFitsTheCache)
{
    Weights w = defaultWeights(Target::Host);
    w.launchSeconds = 0;
    BranchCost gather;
    gather.gathers = 8;
    const CostVector c = costs({gather, gather});
    const double small = predictedSeconds(c, mix({1 << 14, 0}, 1), host(), w).fused / (1 << 14);
    const double large = predictedSeconds(c, mix({1 << 24, 0}, 1), host(), w).fused / (1 << 24);
    EXPECT_NEAR(large / small, w.cacheRatio, 0.2);
}

TEST(DispatchPolicy, ArithmeticAndTrafficOverlapNotAdd)
{
    Weights w = defaultWeights(Target::Host);
    w.launchSeconds = 0;
    BranchCost both;
    both.arith64 = 40;
    both.loads = 8;
    BranchCost compute = both, traffic = both;
    compute.loads = 0;
    traffic.arith64 = 0;
    const Mix m = mix({BATCH, 0}, 1);
    const double t = predictedSeconds(costs({both, both}), m, host(), w).fused;
    const double c = predictedSeconds(costs({compute, compute}), m, host(), w).fused;
    const double b = predictedSeconds(costs({traffic, traffic}), m, host(), w).fused;
    EXPECT_DOUBLE_EQ(t, std::max(c, b));
}

TEST(DispatchPolicy, FewResidentWarpsSlowTheDeviceLanes)
{
    const Weights w = defaultWeights(Target::Device);
    CostVector lean = costs({probeCheap(), probeHeavy()});
    CostVector fat = lean;
    lean.regsPerThread = 40;
    fat.regsPerThread = 255;  // half the register file: 4 warps per SM, under the issue knee
    eagle::DeviceProps small = datacenter();
    small.regs_per_sm = 32768;
    const Mix m = probeMix(50, Target::Device);
    EXPECT_GT(predictedSeconds(fat, m, small, w).fused, predictedSeconds(lean, m, small, w).fused);
    // an unreported register count leaves the rate alone
    EXPECT_DOUBLE_EQ(predictedSeconds(costs({probeCheap(), probeHeavy()}), m, small, w).fused,
                     predictedSeconds(lean, m, small, w).fused);
}

TEST(DispatchPolicy, SplitOverheadScalesWithTheBatch)
{
    // Zero work in both branches: the gain is the fixed terms alone, and the
    // per-sample split overhead keeps it under 1 at every batch size.
    const Weights w = defaultWeights(Target::Device);
    const CostVector none = costs({BranchCost{}, BranchCost{}});
    for (std::int64_t n = 1 << 10; n <= (1 << 26); n <<= 4)
        EXPECT_LT(predictedGain(none, mix({n / 2, n / 2}, 1, 1 << 20), p2000(), w), 1.0) << n;
}

TEST(DispatchPolicy, PlaneTrafficIsAmortisedByFusedSteps)
{
    const Weights w = defaultWeights(Target::Device);
    const Mix one = mix({BATCH / 2, BATCH / 2}, 1, 400), many = mix({BATCH / 2, BATCH / 2}, 64, 400);
    EXPECT_GT(predictedGain(probe(), one, p2000(), w), predictedGain(probe(), many, p2000(), w));
}

TEST(DispatchPolicy, SortIsPaidOverTheRemainingSteps)
{
    const Weights w = defaultWeights(Target::Host);
    const double soon = predictedGain(probe(), mix({BATCH, 0}, 64, 2), host(), w);
    const double later = predictedGain(probe(), mix({BATCH, 0}, 64, 400), host(), w);
    EXPECT_LT(soon, later);
    Weights free = w;
    free.sortSeconds = 0;
    EXPECT_DOUBLE_EQ(predictedGain(probe(), mix({BATCH, 0}, 64, 2), host(), free),
                     predictedGain(probe(), mix({BATCH, 0}, 64, 400), host(), free));
}

TEST(DispatchPolicy, MeasuredMixedGroupsReplaceTheIndependentLanesPrior)
{
    EXPECT_DOUBLE_EQ(warpShare(0.01, -1.0), 1.0 - std::pow(0.99, WARP_LANES));
    EXPECT_DOUBLE_EQ(warpShare(0.01, 0.0), 0.01);     // grouped: only its own warps
    EXPECT_DOUBLE_EQ(warpShare(0.01, 1.0), 1.0);      // every warp mixed
    EXPECT_DOUBLE_EQ(warpShare(0.0, 0.5), 0.0);       // a branch nobody takes costs nothing
    const Weights w = defaultWeights(Target::Device);
    Mix grouped = probeMix(1, Target::Device);
    grouped.mixed32 = 0.0;
    EXPECT_LT(predictedGain(probe(), grouped, p2000(), w),
              predictedGain(probe(), probeMix(1, Target::Device), p2000(), w));
}

TEST(DispatchPolicy, HostClassSortGroupsTheLiveSamplesByClass)
{
    using namespace eagle::filtering;
    const std::int64_t n = 10;
    std::vector<std::uint8_t> mask = { 0, 1, 0, 0, 0, 1, 0, 0, 0, 0 };
    std::vector<std::uint8_t> cls = { 2, 0, 1, 0, 2, 1, 0, 1, 0, 2 };
    std::vector<double> value(n);
    for (std::int64_t i = 0; i < n; ++i) value[i] = double(i);
    std::vector<std::int32_t> perm(n), inv(n), map(n);
    for (std::int64_t i = 0; i < n; ++i) perm[i] = inv[i] = map[i] = std::int32_t(i);
    std::uint32_t count = n, span = n, sorts = 0;
    std::vector<std::uint32_t> offsets(4);
    const ReorderPlane planes[] = { { mask.data(), 1 }, { cls.data(), 1 }, { value.data(), 8 } };
    reorderClassesHost(planes, 3, mask.data(), kCompactMaskIsDrop, 3, cls.data(), perm.data(), inv.data(),
                       map.data(), &count, &span, nullptr, offsets.data(), &sorts, n);
    EXPECT_EQ(offsets, (std::vector<std::uint32_t>{ 0, 3, 5, 8 }));
    EXPECT_EQ(count, 8u);
    EXPECT_EQ(sorts, 1u);
    // stable: within a class the samples keep their order; the dead come last
    EXPECT_EQ(value, (std::vector<double>{ 3, 6, 8, 2, 7, 0, 4, 9, 1, 5 }));
    for (std::int64_t s = 0; s < n; ++s) {
        EXPECT_EQ(value[s], double(perm[s]));
        EXPECT_EQ(inv[perm[s]], s);
    }
}

namespace {
void countSamples(void* const* params, std::int64_t base, std::int64_t count, std::int64_t)
{
    auto* hits = static_cast<std::int64_t*>(params[0]);
    for (std::int64_t i = base; i < base + count; ++i) hits[i] += 1;
}
}  // namespace

TEST(DispatchPolicy, HostTimingKeepsEveryRepetition)
{
    std::vector<std::int64_t> hits(1000, 0);
    void* params[] = { hits.data() };
    const std::vector<double> seconds =
        eagle::exec::timeHostLaunch(&countSamples, params, eagle::exec::Partition::whole(1000), 5);
    ASSERT_EQ(seconds.size(), 5u);
    for (double t : seconds) EXPECT_GE(t, 0.0);
    for (std::int64_t h : hits) EXPECT_EQ(h, 5);
    EXPECT_THROW(eagle::exec::timeRepetitions(0, [] {}), std::invalid_argument);
}

TEST(DispatchPolicy, LaunchTermCrossesOverWithBatchSize)
{
    const Weights w = defaultWeights(Target::Device);
    const CostVector c = probe();
    double previous = 0;
    for (std::int64_t n = 1 << 8; n <= (1 << 26); n <<= 3) {
        const double g = predictedGain(c, mix({n / 2, n / 2}, 1, 400), p2000(), w);
        EXPECT_GE(g, previous) << "gain must not fall as the batch grows";
        previous = g;
    }
    EXPECT_LT(predictedGain(c, mix({128, 128}, 1, 400), p2000(), w), LEAVE_GAIN);
    EXPECT_GE(predictedGain(c, mix({1 << 25, 1 << 25}, 1, 400), p2000(), w), ENTER_GAIN);
}

TEST(DispatchPolicy, GainIsOneForAnEmptyMix)
{
    EXPECT_EQ(predictedGain(costs({cheap(), heavy()}), mix({0, 0}), host(), defaultWeights(Target::Host)), 1.0);
}

TEST(DispatchPolicy, RefusesWrongBranchCounts)
{
    const Weights w = defaultWeights(Target::Host);
    EXPECT_THROW(predictedGain(costs({cheap()}), mix({N}), host(), w), std::invalid_argument);
    CostVector none;
    EXPECT_THROW(predictedGain(none, mix({N}), host(), w), std::invalid_argument);
}

TEST(DispatchPolicy, BandEdges)
{
    const double below = std::nextafter(ENTER_GAIN, 0.0);
    EXPECT_EQ(decide(ENTER_GAIN, Policy::Switch), Policy::Segmented);
    EXPECT_EQ(decide(below, Policy::Switch), Policy::Switch);
    EXPECT_EQ(decide(below, Policy::Segmented), Policy::Segmented);
    EXPECT_EQ(decide(LEAVE_GAIN, Policy::Segmented), Policy::Segmented);
    EXPECT_EQ(decide(LEAVE_GAIN, Policy::Switch), Policy::Switch);
    EXPECT_EQ(decide(std::nextafter(LEAVE_GAIN, 0.0), Policy::Segmented), Policy::Switch);
}

TEST(DispatchPolicy, HysteresisSequence)
{
    // A gain wandering through the band changes the policy only at its edges.
    const std::vector<std::pair<double, Policy>> looks = {
        { 1.00, Policy::Switch },   { 1.10, Policy::Switch },    { 1.16, Policy::Segmented },
        { 1.10, Policy::Segmented }, { 1.06, Policy::Segmented }, { 1.04, Policy::Switch },
        { 1.10, Policy::Switch },   { 1.15, Policy::Segmented }, { 2.00, Policy::Segmented },
    };
    Policy current = Policy::Switch;
    for (const auto& [gain, expect] : looks) {
        current = decide(gain, current);
        EXPECT_EQ(current, expect) << "gain " << gain;
    }
}

TEST(DispatchPolicy, ResolveCarriesTheStateThroughTheBand)
{
    // The same in-band mix gives either answer, whichever the state already holds.
    const Weights w = defaultWeights(Target::Host);
    Mix m;
    double g = 0;
    for (std::int64_t percent = 1; percent <= 100; ++percent) {
        m = probeMix(percent, Target::Host);
        g = predictedGain(probe(), m, host(), w);
        if (g >= LEAVE_GAIN && g < ENTER_GAIN) break;
    }
    ASSERT_GE(g, LEAVE_GAIN);
    ASSERT_LT(g, ENTER_GAIN);
    HysteresisState s;
    EXPECT_EQ(resolve(probe(), m, host(), w, s), Policy::Switch);
    s.current = Policy::Segmented;
    EXPECT_EQ(resolve(probe(), m, host(), w, s), Policy::Segmented);
}

TEST(DispatchPolicy, LooksComeEverySixteenSteps)
{
    HysteresisState s;
    for (int i = 1; i < DECISION_EVERY; ++i) EXPECT_FALSE(due(s)) << "step " << i;
    EXPECT_TRUE(due(s));
    EXPECT_EQ(s.sinceDecision, 0);
    EXPECT_FALSE(due(s, DECISION_EVERY - 1));
    EXPECT_TRUE(due(s, 1));
    EXPECT_TRUE(due(s, 3 * DECISION_EVERY));
}

}  // namespace DispatchPolicyTest
}  // namespace eagle_tests
