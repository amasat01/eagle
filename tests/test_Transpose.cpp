// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* The reverse of a graph Description: step order, summing joins, snapshots,
 * graph inputs, refusals, and the fixed-order adjoint sum. CUDA-free; compiled
 * by CXX in BOTH build modes. */
#include <gtest/gtest.h>

#include "eagle/diff/Transpose.h"

#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace eagle_tests {
namespace TransposeTest {

using namespace eagle::diff;
namespace eg = eagle::graph;

namespace {

eg::Description graphOf(const std::vector<std::string>& labels, const std::vector<std::string>& kinds = {})
{
    eg::Description d;
    for (std::size_t i = 0; i < labels.size(); ++i)
        d.nodes.push_back({ eg::gidx(i), labels[i], kinds.empty() ? "launch" : kinds[i], "", -1, false });
    return d;
}

void expectThrowsNaming(const std::function<void()>& f, const std::string& part)
{
    try {
        f();
        FAIL() << "expected a refusal naming '" << part << "'";
    } catch (const std::invalid_argument& e) {
        EXPECT_NE(std::string(e.what()).find(part), std::string::npos) << e.what();
    }
}

}  // namespace

// y = f(x); z = g(y): the reverse runs g's vjp then f's, y's adjoint flows
// from the first to the second, and x is the only graph input.
TEST(TransposeTest, ChainRunsVjpsBackwards)
{
    const ReversePlan p = transpose(graphOf({ "f", "g" }), { { 0, { "x" }, { "y" } }, { 1, { "y" }, { "z" } } }, { "z" });
    ASSERT_EQ(p.reversed.nodes.size(), 2u);
    EXPECT_EQ(p.reversed.nodes[0].label, "vjp(g)");
    EXPECT_EQ(p.reversed.nodes[1].label, "vjp(f)");
    EXPECT_EQ(p.forward, (std::vector<eg::gidx> { 1, 0 }));
    EXPECT_EQ(p.seedOf[0], (std::vector<eg::gidx> { kSeed }));
    EXPECT_EQ(p.seedOf[1], (std::vector<eg::gidx> { 0 }));
    EXPECT_TRUE(p.joins.empty());
    EXPECT_TRUE(p.snapshots.empty());
    ASSERT_EQ(p.gradients.size(), 1u);
    EXPECT_EQ(p.gradients[0], (std::pair<std::string, eg::gidx> { "x", 1 }));
    ASSERT_EQ(p.reversed.edges.size(), 1u);
    EXPECT_EQ(p.reversed.edges[0].plane, "y");
}

// Two branches read x; their adjoints of x are summed in the order the
// branches were added, whatever order the reverse visits them in.
TEST(TransposeTest, ForkSumsInRegistrationOrder)
{
    const ReversePlan p = transpose(graphOf({ "a", "b", "join" }),
        { { 0, { "x" }, { "u" } }, { 1, { "x" }, { "v" } }, { 2, { "u", "v" }, { "w" } } }, { "w" });
    ASSERT_EQ(p.joins.size(), 1u);
    const Join& j = p.joins[0];
    EXPECT_EQ(j.plane, "x");
    ASSERT_EQ(j.sources.size(), 2u);
    EXPECT_EQ(p.forward[j.sources[0]], 0);
    EXPECT_EQ(p.forward[j.sources[1]], 1);
    EXPECT_EQ(p.reversed.nodes[j.target].kind, "sum");
    EXPECT_EQ(p.gradients[0], (std::pair<std::string, eg::gidx> { "x", j.target }));
    EXPECT_GT(j.target, j.sources[0]);
    EXPECT_GT(j.target, j.sources[1]);
}

// A plane that is seeded AND read later sums the seed first, then the reader.
TEST(TransposeTest, SeedComesBeforeReadersInAJoin)
{
    const ReversePlan s = transpose(graphOf({ "f", "g", "h" }),
        { { 0, { "x" }, { "y" } }, { 1, { "y" }, { "z" } }, { 2, { "y" }, { "t" } } }, { "y", "z", "t" });
    ASSERT_EQ(s.joins.size(), 1u);
    EXPECT_EQ(s.joins[0].plane, "y");
    EXPECT_EQ(s.joins[0].sources.front(), kSeed);
    EXPECT_EQ(s.forward[s.joins[0].sources[1]], 1);
    EXPECT_EQ(s.forward[s.joins[0].sources[2]], 2);
}

TEST(TransposeTest, ReduceBecomesBroadcast)
{
    const ReversePlan p = transpose(graphOf({ "f", "total" }, { "launch", "reduce" }),
        { { 0, { "x" }, { "y" } }, { 1, { "y" }, { "s" } } }, { "s" });
    EXPECT_EQ(p.reversed.nodes[0].kind, "broadcast");
    EXPECT_EQ(p.reversed.nodes[0].label, "broadcast(total)");
    EXPECT_EQ(p.reversed.nodes[1].kind, "vjp");
}

// A node that reads and writes one plane loses the value its vjp needs: the
// plane is copied just before that node.
TEST(TransposeTest, InPlaceNodeSnapshotsBeforeItself)
{
    const ReversePlan p = transpose(graphOf({ "f", "g" }), { { 0, { "x" }, { "x" } }, { 1, { "x" }, { "y" } } }, { "y" });
    ASSERT_EQ(p.snapshots.size(), 1u);
    EXPECT_EQ(p.snapshots[0].node, 0);
    EXPECT_EQ(p.snapshots[0].plane, "x");
    EXPECT_EQ(p.readOf[1], (std::vector<eg::gidx> { 0 }));     // vjp(f) reads the copy
    EXPECT_EQ(p.readOf[0], (std::vector<eg::gidx> { kLive }));  // vjp(g) reads the live x
}

// Node 2 overwrites node 1's input: one copy, taken just before node 2.
TEST(TransposeTest, LaterOverwriteSnapshotsBeforeTheWriter)
{
    const ReversePlan p = transpose(graphOf({ "f", "g" }), { { 0, { "x" }, { "y" } }, { 1, { "y" }, { "x" } } }, { "x" });
    ASSERT_EQ(p.snapshots.size(), 1u);
    EXPECT_EQ(p.snapshots[0].node, 1);
    EXPECT_EQ(p.snapshots[0].plane, "x");
    EXPECT_EQ(p.readOf[1], (std::vector<eg::gidx> { 0 }));
}

TEST(TransposeTest, NoOverwriteSnapshotsNothing)
{
    EXPECT_TRUE(snapshots({ { 0, { "x" }, { "y" } }, { 1, { "x", "y" }, { "z" } } }).empty());
}

TEST(TransposeTest, TwoReadersOfOneOverwrittenValueShareOneSnapshot)
{
    const std::vector<Snapshot> s = snapshots({ { 0, { "x" }, { "y" } }, { 1, { "x" }, { "z" } }, { 2, { "y" }, { "x" } } });
    ASSERT_EQ(s.size(), 1u);
    EXPECT_EQ(s[0].node, 2);
}

TEST(TransposeTest, ReversedDescriptionValidates)
{
    const ReversePlan p = transpose(graphOf({ "a", "b", "join" }),
        { { 0, { "x" }, { "u" } }, { 1, { "x" }, { "v" } }, { 2, { "u", "v" }, { "w" } } }, { "w" });
    EXPECT_NO_THROW(p.reversed.validate());
    for (const eg::Edge& e : p.reversed.edges)
        EXPECT_LT(e.src, e.dst);  // reverse nodes run in id order
}

TEST(TransposeTest, OpaqueNodeRefusedByName)
{
    eg::Description d = graphOf({ "f", "custom" });
    d.nodes[1].opaque = true;
    expectThrowsNaming([&] { transpose(d, { { 0, {}, {} }, { 1, {}, {} } }, {}); }, "'custom' is opaque");
}

TEST(TransposeTest, ConditionalNodeReversesLikeItsChild)
{
    eg::Description d = graphOf({ "maybe", "g" }, { "skippable", "launch" });
    const ReversePlan p = transpose(d, { { 0, { "x" }, { "y" } }, { 1, { "y" }, { "z" } } }, { "z" });
    ASSERT_EQ(p.forward.size(), 2u);
    EXPECT_EQ(p.forward[0], 1);
    EXPECT_EQ(p.forward[1], 0);
    EXPECT_NO_THROW(p.reversed.validate());
}

TEST(TransposeTest, SecondOrderRefused)
{
    const ReversePlan p = transpose(graphOf({ "f" }), { { 0, { "x" }, { "y" } } }, { "y" });
    expectThrowsNaming([&] { transpose(p.reversed, { { 0, { "y" }, { "x" } } }, {}); }, "second order");
}

TEST(TransposeTest, MissingVjpRefusedByName)
{
    expectThrowsNaming([&] { transpose(graphOf({ "f", "g" }), { { 0, { "x" }, { "y" } } }, {}); }, "'g' has no vjp");
}

TEST(TransposeTest, UnseededOutputFeedsZero)
{
    const ReversePlan p = transpose(graphOf({ "f" }), { { 0, { "x" }, { "y", "e" } } }, { "y" });
    EXPECT_EQ(p.seedOf[0], (std::vector<eg::gidx> { kSeed, kZero }));
}

TEST(TransposeTest, SeedOfAPlaneTheGraphDoesNotWriteRefused)
{
    expectThrowsNaming([&] { transpose(graphOf({ "f" }), { { 0, { "x" }, { "y" } } }, { "x" }); }, "'x' is not written");
}

TEST(TransposeTest, OrderedSumIsALeftFoldBitwise)
{
    // values whose sum depends on the order: (1e16 + 1) - 1e16 = 0, while 1 + (1e16 - 1e16) = 1
    const std::vector<double> a { 1e16 }, b { 1.0 }, c { -1e16 };
    double out = 0;
    orderedSum<double>(&out, { a.data(), b.data(), c.data() }, 1);
    EXPECT_EQ(out, (1e16 + 1.0) + -1e16);
    orderedSum<double>(&out, { b.data(), a.data(), c.data() }, 1);
    EXPECT_EQ(out, (1.0 + 1e16) + -1e16);
}

TEST(TransposeTest, OrderedSumIsThreadInvariant)
{
    const std::ptrdiff_t n = 100003;
    std::vector<double> s0(n), s1(n), s2(n);
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        s0[i] = 1e-3 * double(i % 977) + 1e10;
        s1[i] = -1e10 + 1e-7 * double(i);
        s2[i] = 0.1 * double((i * 7919) % 13);
    }
    std::vector<double> ref(n), got(n);
#ifdef _OPENMP
    const int saved = omp_get_max_threads();
    omp_set_num_threads(1);
#endif
    orderedSum<double>(ref.data(), { s0.data(), s1.data(), s2.data() }, n);
    for (int t : { 2, 4, 8 }) {
#ifdef _OPENMP
        omp_set_num_threads(t);
#endif
        orderedSum<double>(got.data(), { s0.data(), s1.data(), s2.data() }, n);
        EXPECT_EQ(0, std::memcmp(ref.data(), got.data(), sizeof(double) * n)) << t << " threads";
    }
#ifdef _OPENMP
    omp_set_num_threads(saved);
#endif
}

}  // namespace TransposeTest
}  // namespace eagle_tests
