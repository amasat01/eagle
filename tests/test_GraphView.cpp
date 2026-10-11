// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* The graph view: Description, edge derivation, layout and the three text
 * renderings, plus the adapters from the native graphs. CUDA-free, compiled by
 * CXX in BOTH build modes with EAGLE_CPU_ONLY forced on this one source (the
 * test_ConditionalGroupCpu.cpp recipe). The CUDA graph's adapter is
 * test_GraphViewCuda.cu. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "eagle/graph/Describe.h"
#include "eagle/graph/Edges.h"
#include "eagle/graph/Layout.h"
#include "eagle/graph/Render.h"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>

namespace eagle_tests {
namespace GraphViewTest {

using namespace eagle::graph;

namespace {

Node node(gidx id, std::string label, bool opaque = false, gidx frame = -1,
    std::string stage = {})
{
    return { id, std::move(label), "launch", std::move(stage), frame, opaque };
}

Edge edge(gidx a, gidx b, std::string plane = "x", EdgeKind k = EdgeKind::RAW)
{
    return { a, b, std::move(plane), k };
}

/** The diamond used by the goldens: scale -> {left, right*} -> join (in an
 *  "if guard" frame). */
Description diamond()
{
    Description d;
    d.nodes = { node(0, "scale"), node(1, "left"), node(2, "right", true),
        node(3, "join", false, 0) };
    d.edges = { edge(0, 1, "y"), edge(0, 2, "y"), edge(1, 3, "a"),
        edge(2, 3, "b", EdgeKind::WAW) };
    d.frames = { { 0, "if", "skip if guard", -1 } };
    return d;
}

PlaneRef ref(std::string name, gidx nodeIdx, std::uint64_t base,
    std::vector<gidx> shape, std::vector<gidx> strides, int access,
    gidx itemsize = 8)
{
    return { std::move(name), nodeIdx, base, std::move(shape), std::move(strides),
        itemsize, access };
}

gidx crossingCount(const Description& d, const Layout& l)
{
    gidx c = 0;
    for (std::size_t i = 0; i < d.edges.size(); ++i)
        for (std::size_t j = i + 1; j < d.edges.size(); ++j) {
            const Edge &a = d.edges[i], &b = d.edges[j];
            if (l.layer[a.src] == l.layer[b.src] && l.layer[a.dst] == l.layer[b.dst]
                && (l.order[a.src] - l.order[b.src]) * (l.order[a.dst] - l.order[b.dst]) < 0)
                ++c;
        }
    return c;
}

struct Tracer {
    void reserveScratch(eagle::ScratchArena&) { }
    void runHost(eagle::cpu::Graph&) { }
};

}  // namespace

TEST(GraphViewTest, DescriptionValidateRefusesDanglingEdge)
{
    Description d;
    d.nodes = { node(0, "a"), node(1, "b") };
    d.edges = { edge(0, 5) };
    EXPECT_THROW(d.validate(), std::invalid_argument);
    d.edges = { edge(0, 0) };
    EXPECT_THROW(d.validate(), std::invalid_argument);
    d.edges = { edge(0, 1), edge(1, 0) };
    EXPECT_THROW(d.validate(), std::invalid_argument);
    d.edges = { edge(0, 1) };
    d.nodes[1].id = 7;
    EXPECT_THROW(d.validate(), std::invalid_argument);
    d.nodes[1].id = 1;
    d.nodes[1].frame = 3;
    EXPECT_THROW(d.validate(), std::invalid_argument);
    d.nodes[1].frame = -1;
    EXPECT_NO_THROW(d.validate());
}

TEST(GraphViewTest, LayoutLayersAreStagesFromEdges)
{
    Description chain;
    chain.nodes = { node(0, "a"), node(1, "b"), node(2, "c") };
    chain.edges = { edge(0, 1), edge(1, 2) };
    const Layout lc = layout(chain);
    EXPECT_EQ(lc.layer, (std::vector<gidx>{ 0, 1, 2 }));

    Description fork;
    fork.nodes = { node(0, "src"), node(1, "l"), node(2, "r"), node(3, "join") };
    fork.edges = { edge(0, 1), edge(0, 2), edge(1, 3), edge(2, 3) };
    const Layout lf = layout(fork);
    EXPECT_EQ(lf.layer[1], lf.layer[2]) << "siblings share a layer";
    EXPECT_EQ(lf.layer[3], 2);
    EXPECT_NE(lf.order[1], lf.order[2]);
    EXPECT_DOUBLE_EQ(lf.xy[1][1], 1.0);
    EXPECT_DOUBLE_EQ(lf.xy[1][0] + lf.xy[2][0], 0.0) << "a layer is centred";
}

TEST(GraphViewTest, LayoutIsDeterministic)
{
    Description a = diamond();
    Description b = diamond();
    std::reverse(b.edges.begin(), b.edges.end());
    const Layout la = layout(a), la2 = layout(a), lb = layout(b);
    EXPECT_EQ(la.order, la2.order);
    EXPECT_EQ(la.xy, la2.xy);
    EXPECT_EQ(la.order, lb.order) << "edge insertion order must not matter";
    EXPECT_EQ(la.layer, lb.layer);
}

TEST(GraphViewTest, LayoutReducesCrossings)
{
    // layer 0: 0,1,2,3; layer 1: 4,5,6,7. Edges pair (0,7)(1,6)(2,5)(3,4):
    // in id order that is 6 crossings; the sweeps untangle them.
    Description d;
    for (gidx i = 0; i < 8; ++i)
        d.nodes.push_back(node(i, "n" + std::to_string(i)));
    d.edges = { edge(0, 7), edge(1, 6), edge(2, 5), edge(3, 4) };
    Layout initial;
    initial.layer = { 0, 0, 0, 0, 1, 1, 1, 1 };
    initial.order = { 0, 1, 2, 3, 0, 1, 2, 3 };
    ASSERT_EQ(crossingCount(d, initial), 6);
    const Layout l = layout(d);
    EXPECT_EQ(crossingCount(d, l), 0);

    // the two-crossing shape: a-d, b-c, a-c ... one crossing, still removed
    Description e;
    for (gidx i = 0; i < 4; ++i)
        e.nodes.push_back(node(i, "n" + std::to_string(i)));
    e.edges = { edge(0, 3), edge(1, 2) };
    EXPECT_EQ(crossingCount(e, layout(e)), 0);
}

// The five shapes the box diagram has to get right. Built by hand so the
// golden does not depend on anything but the renderer and the layout.
Description forkJoin()
{
    Description d;
    d.nodes = { node(0, "a"), node(1, "b"), node(2, "c"), node(3, "join") };
    d.edges = { edge(0, 3, "a"), edge(1, 3, "b"), edge(2, 3, "c") };
    return d;
}

Description chain()
{
    Description d;
    d.nodes = { node(0, "scale", false, -1, "capture"), node(1, "addone", false, -1, "capture") };
    d.edges = { edge(0, 1, "y") };
    return d;
}

Description conditional()
{
    Description d;
    d.nodes = { node(0, "scale"), node(1, "maybe", false, 0), node(2, "after") };
    d.edges = { edge(0, 1, "y"), edge(1, 2, "z") };
    d.frames = { { 0, "if", "skip if flag", -1 } };
    return d;
}

Description layerSkip()
{
    Description d;
    d.nodes = { node(0, "a"), node(1, "b"), node(2, "c", true) };
    d.edges = { edge(0, 1, "x"), edge(1, 2, "y"), edge(0, 2, "x,w", EdgeKind::WAR) };
    return d;
}

Description diamondLoop()
{
    Description d = diamond();
    d.frames = { { 0, "while", "repeat <= 4", -1 } };
    return d;
}

TEST(GraphViewTest, AsciiGoldenForkJoin)
{
    const Description d = forkJoin();
    EXPECT_EQ(ascii(d, layout(d)),
R"GOLD(  stage 0 ┌───┐   ┌───┐   ┌───┐
          │ a │   │ b │   │ c │
          └─┬─┘   └─┬─┘   └─┬─┘
            a       b       c
            └───────┼───────┘
  stage 1       ┌───┴───┐
                │ join  │
                └───────┘
)GOLD");
    EXPECT_EQ(ascii(d, layout(d), false),
R"GOLD(  stage 0 +---+   +---+   +---+
          | a |   | b |   | c |
          +-+-+   +-+-+   +-+-+
            a       b       c
            +-------+-------+
  stage 1       +---+---+
                | join  |
                +-------+
)GOLD");
}

TEST(GraphViewTest, AsciiGoldenChainWithScheduleTag)
{
    const Description d = chain();
    EXPECT_EQ(ascii(d, layout(d)),
R"GOLD(  stage 0 capture  ┌───────┐
                   │ scale │
                   └───┬───┘
                       y
                       │
  stage 1 capture ┌────┴────┐
                  │ addone  │
                  └─────────┘
)GOLD");
}

TEST(GraphViewTest, AsciiGoldenConditionalFrame)
{
    const Description d = conditional();
    EXPECT_EQ(ascii(d, layout(d)),
R"GOLD(  stage 0      ┌───────┐
               │ scale │
               └───┬───┘
                   y
                   │
          ╭─ skip if flag ──╮
  stage 1 │ ┌──────┴──────┐ │
          │ │    maybe    │ │
          │ └──────┬──────┘ │
          ╰────────│────────╯
                   z
                   │
  stage 2      ┌───┴───┐
               │ after │
               └───────┘
)GOLD");
}

TEST(GraphViewTest, AsciiGoldenLayerSkippingEdgeAndOpaqueBorder)
{
    const Description d = layerSkip();
    EXPECT_EQ(ascii(d, layout(d)),
R"GOLD(  stage 0   ┌───┐
            │ a │
            └─┬─┘
              x,x,w…
            ┌─┴───┐
  stage 1 ┌─┴─┐   │
          │ b │   │
          └─┬─┘   │
            y     │
            └─┬───┘
  stage 2   ┌╌┴╌┐
            ╎ c ╎
            └╌╌╌┘
)GOLD");
}

TEST(GraphViewTest, AsciiGoldenDiamondLoopWithOpaqueNode)
{
    const Description d = diamondLoop();
    EXPECT_EQ(ascii(d, layout(d)),
R"GOLD(  stage 0       ┌───────┐
                │ scale │
                └───┬───┘
                    y
              ┌─────┴─────┐
  stage 1 ┌───┴───┐   ┌╌╌╌┴╌╌╌┐
          │ left  │   ╎ right ╎
          └───┬───┘   └╌╌╌┬╌╌╌┘
              a           b:WAW
              └─────┬─────┘
            ╭─ repeat <= 4 ─╮
  stage 2   │ ┌─────┴─────┐ │
            │ │   join    │ │
            │ └───────────┘ │
            ╰───────────────╯
)GOLD");
}

TEST(GraphViewTest, AsciiIsDeterministicAndTruncatesLongLabels)
{
    Description d = forkJoin();
    d.nodes[3].label = "a_very_long_label_that_does_not_fit_in_the_box";
    EXPECT_EQ(ascii(d, layout(d), true, 10), ascii(d, layout(d), true, 10));
    const std::string text = ascii(d, layout(d), true, 10);
    EXPECT_NE(text.find("a_very_lo\xE2\x80\xA6"), std::string::npos) << text;
    EXPECT_EQ(text.find("long_label"), std::string::npos);
    const std::string plain = ascii(d, layout(d), false, 10);
    EXPECT_NE(plain.find("a_very_..."), std::string::npos) << plain;
    for (unsigned char c : plain)
        EXPECT_LT(c, 128u) << "the fallback is pure ASCII";
    EXPECT_EQ(ascii(Description{}), "");
}

TEST(GraphViewTest, AsciiFallbackFoldsNonAsciiLabels)
{
    Description d;
    d.nodes = { node(0, "caf\xC3\xA9"), node(1, "b") };
    d.edges = { edge(0, 1, "x") };
    d.frames = { { 0, "while", "repeat \xE2\x89\xA4 4 \xC2\xB7 more", -1 } };
    d.nodes[1].frame = 0;
    const std::string plain = ascii(d, layout(d), false);
    for (unsigned char c : plain)
        EXPECT_LT(c, 128u) << plain;
    EXPECT_NE(plain.find("repeat <= 4 - more"), std::string::npos) << plain;
    EXPECT_NE(plain.find("caf?"), std::string::npos) << plain;
    EXPECT_EQ(detail::asciiFold("\xE2\x89\xA5 \xE2\x80\xA6"), ">= ...");
}

TEST(GraphViewTest, EdgeKindsHavePlainMeanings)
{
    EXPECT_STREQ(kindMeaning(EdgeKind::RAW), "data flows");
    EXPECT_STREQ(kindMeaning(EdgeKind::WAR), "must wait to overwrite");
    EXPECT_STREQ(kindMeaning(EdgeKind::WAW), "same output");
    EXPECT_EQ(parseEdgeKind(kindName(EdgeKind::WAR)), EdgeKind::WAR);
}

TEST(GraphViewTest, MermaidGolden)
{
    EXPECT_EQ(mermaid(diamond()),
        "flowchart TD\n"
        "  n0[\"scale\"]\n"
        "  n1[\"left\"]\n"
        "  n2[\"right\"]\n"
        "  subgraph f0[\"skip if guard\"]\n"
        "    n3[\"join\"]\n"
        "  end\n"
        "  n0 -->|\"y\"| n1\n"
        "  n0 -->|\"y\"| n2\n"
        "  n1 -->|\"a\"| n3\n"
        "  n2 ==>|\"b\"| n3\n"
        "  classDef opaque stroke-dasharray: 4 2\n"
        "  class n2 opaque\n");
}

TEST(GraphViewTest, DotGolden)
{
    EXPECT_EQ(dot(diamond()),
        "digraph eagle {\n"
        "  rankdir=TB;\n"
        "  node [shape=box];\n"
        "  n0 [label=\"scale\"];\n"
        "  n1 [label=\"left\"];\n"
        "  n2 [label=\"right\", style=dashed];\n"
        "  subgraph cluster_f0 {\n"
        "    label=\"skip if guard\";\n"
        "    n3 [label=\"join\"];\n"
        "  }\n"
        "  n0 -> n1 [label=\"y\", style=solid];\n"
        "  n0 -> n2 [label=\"y\", style=solid];\n"
        "  n1 -> n3 [label=\"a\", style=solid];\n"
        "  n2 -> n3 [label=\"b\", style=bold];\n"
        "}\n");
}

TEST(GraphViewTest, EveryNodeAndEdgeAppearsOnceInEachExport)
{
    const Description d = diamond();
    auto count = [](const std::string& text, const std::string& what) {
        std::size_t n = 0;
        for (auto at = text.find(what); at != std::string::npos; at = text.find(what, at + 1))
            ++n;
        return n;
    };
    const std::string m = mermaid(d), g = dot(d);
    for (gidx i = 0; i < 4; ++i) {
        EXPECT_EQ(count(m, "n" + std::to_string(i) + "[\""), 1u) << m;
        const std::string def = "n" + std::to_string(i) + " [label=\"";
        EXPECT_EQ(count(g, "\n  " + def) + count(g, "\n    " + def), 1u) << g;
    }
    EXPECT_EQ(count(m, "-->") + count(m, "-.->") + count(m, "==>"), d.edges.size());
    EXPECT_EQ(count(g, " -> "), d.edges.size());
}

TEST(GraphViewTest, DescribeCpuGraphEdgesFollowDeps)
{
    eagle::cpu::Graph g;
    const eagle::idx_t a = g.addNative(Tracer{}, {});
    const eagle::idx_t b = g.addNative(Tracer{}, { a });
    const eagle::idx_t c = g.addNative(Tracer{}, { a });
    g.addNative(Tracer{}, { b, c });
    const Description d = describe(g);
    ASSERT_EQ(d.nodes.size(), 4u);
    EXPECT_EQ(d.nodes[2].label, "native2");
    std::vector<std::pair<gidx, gidx>> got;
    for (const Edge& e : d.edges)
        got.emplace_back(e.src, e.dst);
    EXPECT_EQ(got, (std::vector<std::pair<gidx, gidx>>{ { 0, 1 }, { 0, 2 }, { 1, 3 }, { 2, 3 } }));
    EXPECT_EQ(layout(d).layer, (std::vector<gidx>{ 0, 1, 1, 2 }));
}

TEST(GraphViewTest, DescribeComposerChainsSequencedMembers)
{
    eagle::compose::GraphComposer c("sequenced");
    eagle::cpu::Graph g1, g2;
    c.registerGraph(std::move(g1), "first");
    c.registerGraph(std::move(g2));
    const Description d = describe(c);
    ASSERT_EQ(d.nodes.size(), 2u);
    EXPECT_EQ(d.nodes[0].label, "first");
    EXPECT_EQ(d.nodes[1].label, "member1");
    ASSERT_EQ(d.edges.size(), 1u);
    EXPECT_EQ(d.edges[0].src, 0);
}

TEST(GraphViewTest, DescribeConditionalGroupFramesTheBody)
{
    unsigned int flag = 1;
    eagle::conditional::ConditionalGroup group(eagle::CountGuard{ &flag, nullptr });
    group.body().addNative(Tracer{}, {});
    group.body().addNative(Tracer{}, { 0 });
    const Description d = describe(group, describe(group.body()));
    d.validate();
    ASSERT_EQ(d.frames.size(), 1u);
    EXPECT_EQ(d.frames[0].kind, "if");
    EXPECT_EQ(d.nodes[0].frame, 0);
    EXPECT_EQ(d.nodes[1].frame, 0);
}

TEST(GraphViewTest, DeriveEdgesKindsAndOrder)
{
    // node0 writes A; node1 reads A and writes B; node2 writes A.
    const std::vector<PlaneRef> refs = {
        ref("A", 0, 1000, { 4 }, { 8 }, 1),
        ref("A", 1, 1000, { 4 }, { 8 }, 0),
        ref("B", 1, 2000, { 4 }, { 8 }, 1),
        ref("A", 2, 1000, { 4 }, { 8 }, 1),
    };
    const auto e = deriveEdges(refs);
    ASSERT_EQ(e.size(), 3u);
    EXPECT_EQ(std::make_tuple(e[0].src, e[0].dst, e[0].kind), std::make_tuple(0, 1, EdgeKind::RAW));
    EXPECT_EQ(std::make_tuple(e[1].src, e[1].dst, e[1].kind), std::make_tuple(0, 2, EdgeKind::WAW));
    EXPECT_EQ(std::make_tuple(e[2].src, e[2].dst, e[2].kind), std::make_tuple(1, 2, EdgeKind::WAR));
}

TEST(GraphViewTest, DeriveEdgesStrideAware)
{
    // one (N, w) float64 array, N = 1000, w = 4: column views never share a byte
    const std::uint64_t base = 4096;
    const gidx N = 1000, w = 4;
    auto column = [&](gidx j, gidx nd, int acc) {
        return ref("col" + std::to_string(j), nd, base + 8 * j, { N }, { 8 * w }, acc);
    };
    EXPECT_TRUE(deriveEdges({ column(0, 0, 1), column(1, 1, 0) }).empty());
    EXPECT_TRUE(deriveEdges({ column(2, 0, 1), column(3, 1, 1) }).empty());
    // the same column does overlap
    EXPECT_EQ(deriveEdges({ column(1, 0, 1), column(1, 1, 0) }).size(), 1u);
    // a tail-aliasing view: the last 10 rows of column 0 vs the first 10 rows
    // of the same column shifted to overlap
    const auto tailA = ref("tail", 0, base, { 20 }, { 8 * w }, 1);
    const auto tailB = ref("head", 1, base + 8 * w * 10, { 20 }, { 8 * w }, 0);
    EXPECT_EQ(deriveEdges({ tailA, tailB }).size(), 1u);
    // the whole row-major matrix vs one column: overlap
    const auto whole = ref("all", 0, base, { N, w }, { 8 * w, 8 }, 1);
    EXPECT_EQ(deriveEdges({ whole, column(2, 1, 0) }).size(), 1u);
    // interleaved halves of a vector: no overlap
    EXPECT_TRUE(deriveEdges({ ref("even", 0, base, { 500 }, { 16 }, 1),
                                ref("odd", 1, base + 8, { 500 }, { 16 }, 0) }).empty());
    // disjoint ranges and empty arrays never overlap
    EXPECT_TRUE(deriveEdges({ ref("a", 0, base, { 10 }, { 8 }, 1),
                                ref("b", 1, base + 80, { 10 }, { 8 }, 0) }).empty());
    EXPECT_TRUE(deriveEdges({ ref("a", 0, base, { 0 }, { 8 }, 1),
                                ref("b", 1, base, { 10 }, { 8 }, 0) }).empty());
}

TEST(GraphViewTest, SiblingConflictNamesBothNodes)
{
    const std::vector<PlaneRef> refs = {
        ref("left_out", 0, 1000, { 8 }, { 8 }, 1),
        ref("right_in", 1, 1032, { 8 }, { 8 }, 0),     // overlaps left_out's tail
        ref("other", 2, 9000, { 8 }, { 8 }, 1),
    };
    const auto c = siblingConflicts(refs, std::vector<std::vector<gidx>>{ { 0, 1, 2 } });
    ASSERT_EQ(c.size(), 1u);
    EXPECT_EQ(c[0].a, 0);
    EXPECT_EQ(c[0].b, 1);
    EXPECT_EQ(c[0].planeA, "left_out");
    EXPECT_EQ(c[0].planeB, "right_in");
    EXPECT_EQ(c[0].kind, EdgeKind::RAW);
    // two nodes of ONE branch are ordered, not siblings; a read-read overlap is no conflict
    SiblingGroup oneBranch;
    oneBranch.branches = { { 0, 1 }, { 2 } };
    EXPECT_TRUE(siblingConflicts(refs, std::vector<SiblingGroup>{ oneBranch }).empty());
    const std::vector<PlaneRef> reads = { ref("a", 0, 1000, { 8 }, { 8 }, 0),
        ref("b", 1, 1000, { 8 }, { 8 }, 0) };
    EXPECT_TRUE(siblingConflicts(reads, std::vector<std::vector<gidx>>{ { 0, 1 } }).empty());
    // sibling pairs carry no ordering edge
    EXPECT_TRUE(deriveEdges(refs, std::vector<SiblingGroup>{ [] {
        SiblingGroup g;
        g.branches = { { 0 }, { 1 }, { 2 } };
        return g;
    }() }).empty());
}

}  // namespace GraphViewTest
}  // namespace eagle_tests
