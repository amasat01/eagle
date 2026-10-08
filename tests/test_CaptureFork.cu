// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "TestBase.h"

#include "eagle/cuda.h"
#include "eagle/cuda/CaptureFork.h"
#include "eagle/cuda/detail/RuntimeCompat.h"

#ifndef EAGLE_CPU_ONLY

#include <algorithm>
#include <string>
#include <vector>

namespace eagle_tests {
namespace CaptureForkTest {

using namespace eagle;
using namespace eagle::cuda;

/* ================================================================
 * The exemplar shape under test — a diamond:
 *
 *          A:  x  = a * 2          (origin stream)
 *         / \
 *   B1: y1 = x + 1   B2: y2 = x * 3   (two independent branches)
 *         \ /
 *          C:  z  = y1 + y2        (origin stream)
 *
 * B1 and B2 read the same input and write disjoint outputs, so they
 * are mutually independent — the precondition ``CaptureFork``'s
 * contract puts on the caller. Captured without a fork they would be
 * a linear chain B1 -> B2; with one they must be siblings.
 * ================================================================ */

__global__ void scaleKernel(const double* in, double* out, double factor, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        out[tid] = in[tid] * factor;
}

__global__ void offsetKernel(const double* in, double* out, double c, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        out[tid] = in[tid] + c;
}

__global__ void sumKernel(
    const double* lhs, const double* rhs, double* out, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        out[tid] = lhs[tid] + rhs[tid];
}

/* ================================================================
 * Fixture: device buffers for the diamond, plus a host oracle.
 * ================================================================ */
class CaptureForkFixture : public Test {
protected:
    static constexpr int N     = 256;
    static constexpr int BLOCK = 256;

    double* d_a  = nullptr;
    double* d_x  = nullptr;
    double* d_y1 = nullptr;
    double* d_y2 = nullptr;
    double* d_z  = nullptr;

    void SetUp() override
    {
        for (double** buf : { &d_a, &d_x, &d_y1, &d_y2, &d_z }) {
            EAGLE_CHECK_ALWAYS(cudaMalloc(buf, N * sizeof(double)));
            EAGLE_CHECK_ALWAYS(cudaMemset(*buf, 0, N * sizeof(double)));
        }
        uploadInput(1.0);
    }

    void TearDown() override
    {
        for (double* buf : { d_a, d_x, d_y1, d_y2, d_z })
            if (buf)
                EAGLE_CHECK_ALWAYS(cudaFree(buf));
    }

    /** @brief Fill the input with ``i * scale`` and upload it. */
    void uploadInput(double scale)
    {
        h_a.resize(N);
        for (int i = 0; i < N; i++)
            h_a[i] = static_cast<double>(i) * scale;
        EAGLE_CHECK_ALWAYS(cudaMemcpy(
            d_a, h_a.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        // The copy (and SetUp's memsets) ride the per-thread default stream,
        // which has no ordering with the eagle Streams the test then reads on.
        EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    }

    /** @brief z = (a*2 + 1) + (a*2 * 3), evaluated on the host. */
    std::vector<double> oracle() const
    {
        std::vector<double> out(N);
        for (int i = 0; i < N; i++) {
            const double x = h_a[i] * 2.0;
            out[i]         = (x + 1.0) + (x * 3.0);
        }
        return out;
    }

    std::vector<double> downloadResult() const
    {
        std::vector<double> out(N);
        EAGLE_CHECK_ALWAYS(cudaMemcpy(
            out.data(), d_z, N * sizeof(double), cudaMemcpyDeviceToHost));
        return out;
    }

    void expectMatchesOracle(const std::vector<double>& got) const
    {
        const std::vector<double> want = oracle();
        for (int i = 0; i < N; i++)
            EXPECT_DOUBLE_EQ(got[i], want[i]) << "element " << i;
    }

    /** @brief Capture the diamond, forking the two independent branches.
     *
     *  @param[in] joinTwice  Call ``join()`` a second time (idempotence).
     *  @param[in] autoJoin   Never call ``join()``; let the destructor do it.
     */
    cudaGraph_t captureDiamond(
        const cudaStream_t& stream, bool joinTwice = false,
        bool autoJoin = false)
    {
        /* Pre-capture: all streams and events acquired here. */
        StreamCapturer capturer(stream);
        {
            CaptureFork fork(stream, 2);
            EXPECT_EQ(fork.size(), 2u);
            EXPECT_FALSE(fork.forked());

            capturer.begin();
            scaleKernel<<<1, BLOCK, 0, stream>>>(d_a, d_x, 2.0, N);

            fork.fork();
            EXPECT_TRUE(fork.forked());
            offsetKernel<<<1, BLOCK, 0, fork.branch(0)>>>(d_x, d_y1, 1.0, N);
            scaleKernel<<<1, BLOCK, 0, fork.branch(1)>>>(d_x, d_y2, 3.0, N);

            if (!autoJoin) {
                fork.join();
                EXPECT_FALSE(fork.forked());
                if (joinTwice)
                    fork.join();
            }
        } /* autoJoin: the destructor closes the fork here */

        sumKernel<<<1, BLOCK, 0, stream>>>(d_y1, d_y2, d_z, N);
        return capturer.end();
    }

    std::vector<double> h_a;
};

/* ================================================================
 * Graph topology helpers: read the captured DAG back with the raw
 * CUDA graph API. Nodes are identified STRUCTURALLY (by degree),
 * never by kernel function pointer — the latter is unreliable for
 * anything that may become templated.
 * ================================================================ */
struct Topology {
    std::vector<cudaGraphNode_t> nodes;
    std::vector<std::vector<bool>> adjacency; /* adjacency[from][to] */

    std::size_t indexOf(const cudaGraphNode_t& node) const
    {
        return static_cast<std::size_t>(
            std::find(nodes.begin(), nodes.end(), node) - nodes.begin());
    }

    std::size_t inDegree(std::size_t to) const
    {
        std::size_t count = 0;
        for (std::size_t from = 0; from < nodes.size(); from++)
            count += adjacency[from][to] ? 1 : 0;
        return count;
    }

    std::size_t outDegree(std::size_t from) const
    {
        return static_cast<std::size_t>(std::count(
            adjacency[from].begin(), adjacency[from].end(), true));
    }

    /** @brief Transitive reachability (permits redundant shortcut edges). */
    bool reaches(std::size_t from, std::size_t to) const
    {
        std::vector<bool> seen(nodes.size(), false);
        std::vector<std::size_t> stack{ from };
        while (!stack.empty()) {
            const std::size_t cur = stack.back();
            stack.pop_back();
            for (std::size_t next = 0; next < nodes.size(); next++) {
                if (!adjacency[cur][next] || seen[next])
                    continue;
                if (next == to)
                    return true;
                seen[next] = true;
                stack.push_back(next);
            }
        }
        return false;
    }
};

static Topology readTopology(cudaGraph_t graph)
{
    Topology topology;

    std::size_t numNodes = 0;
    EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(graph, nullptr, &numNodes));
    topology.nodes.resize(numNodes);
    EAGLE_CHECK_ALWAYS(
        cudaGraphGetNodes(graph, topology.nodes.data(), &numNodes));

    std::size_t numEdges = 0;
    EAGLE_CHECK_ALWAYS(eagle::cuda::detail::graphGetEdges(graph, nullptr, nullptr, &numEdges));
    std::vector<cudaGraphNode_t> from(numEdges);
    std::vector<cudaGraphNode_t> to(numEdges);
    EAGLE_CHECK_ALWAYS(
        eagle::cuda::detail::graphGetEdges(graph, from.data(), to.data(), &numEdges));

    topology.adjacency.assign(numNodes, std::vector<bool>(numNodes, false));
    for (std::size_t e = 0; e < numEdges; e++)
        topology.adjacency[topology.indexOf(from[e])][topology.indexOf(to[e])]
            = true;

    return topology;
}

/* ================================================================
 * 1. Topology — the branches are genuine siblings.
 *
 *  Asserted structurally: exactly one source (the fork point) and one
 *  sink (the join point); the two remaining nodes have NO edge between
 *  them in either direction, and each is reachable from the source and
 *  reaches the sink. Transitive-redundant edges (a direct source->sink
 *  shortcut the capture frontier may add) are PERMITTED — asserting an
 *  exact edge set would be asserting a driver implementation detail.
 * ================================================================ */
TEST_F(CaptureForkFixture, ForkProducesSiblingNodes)
{
    Stream s(/*nonBlocking=*/true);
    cudaGraph_t graph = captureDiamond(s.cuda());
    ASSERT_NE(graph, nullptr);

    const Topology topology = readTopology(graph);
    ASSERT_EQ(topology.nodes.size(), 4u)
        << "diamond should capture as four kernel nodes";

    std::vector<std::size_t> sources;
    std::vector<std::size_t> sinks;
    for (std::size_t i = 0; i < topology.nodes.size(); i++) {
        if (topology.inDegree(i) == 0)
            sources.push_back(i);
        if (topology.outDegree(i) == 0)
            sinks.push_back(i);
    }
    ASSERT_EQ(sources.size(), 1u) << "exactly one fork point expected";
    ASSERT_EQ(sinks.size(), 1u) << "exactly one join point expected";

    const std::size_t source = sources.front();
    const std::size_t sink   = sinks.front();
    ASSERT_NE(source, sink);

    std::vector<std::size_t> siblings;
    for (std::size_t i = 0; i < topology.nodes.size(); i++)
        if (i != source && i != sink)
            siblings.push_back(i);
    ASSERT_EQ(siblings.size(), 2u);

    /* THE load-bearing structural claim: no ordering between siblings. */
    EXPECT_FALSE(topology.adjacency[siblings[0]][siblings[1]])
        << "branches must not be ordered relative to each other";
    EXPECT_FALSE(topology.adjacency[siblings[1]][siblings[0]])
        << "branches must not be ordered relative to each other";

    for (const std::size_t sibling : siblings) {
        EXPECT_TRUE(topology.reaches(source, sibling))
            << "branch must depend on everything before the fork";
        EXPECT_TRUE(topology.reaches(sibling, sink))
            << "everything after the join must depend on the branch";
    }

    EAGLE_CHECK_ALWAYS(cudaGraphDestroy(graph));
}

/* ================================================================
 * 2. Replay conformance — capture once, replay many, no rebuild.
 *
 *  The input is mutated in place between replays, so a stale or
 *  cached result cannot pass: each launch must genuinely re-read the
 *  live device buffers.
 * ================================================================ */
TEST_F(CaptureForkFixture, ReplaysMatchHostOracle)
{
    Stream s(/*nonBlocking=*/true);
    Graph g;
    g.stream(s.cuda());
    /* CapturedGraph, not the raw handle: the parent clones the child, and
     * the owning overload releases the source for us. */
    g.addNode(CapturedGraph{ captureDiamond(s.cuda()) });

    Launcher launcher = g.launcher();

    for (int replay = 0; replay < 3; replay++) {
        uploadInput(1.0 + 0.25 * replay);
        launcher.launch();
        launcher.synchronize();
        SCOPED_TRACE("replay " + std::to_string(replay));
        expectMatchesOracle(downloadResult());
    }
}

/* ================================================================
 * 3. join() is idempotent.
 * ================================================================ */
TEST_F(CaptureForkFixture, DoubleJoinIsANoOp)
{
    Stream s(/*nonBlocking=*/true);
    Graph g;
    g.stream(s.cuda());
    g.addNode(CapturedGraph{ captureDiamond(s.cuda(), /*joinTwice=*/true) });

    Launcher launcher = g.launcher();
    launcher.launch();
    launcher.synchronize();
    expectMatchesOracle(downloadResult());
}

/* ================================================================
 * 4. RAII — scope exit joins, so a valid graph still comes out.
 * ================================================================ */
TEST_F(CaptureForkFixture, DestructorAutoJoins)
{
    Stream s(/*nonBlocking=*/true);
    cudaGraph_t graph
        = captureDiamond(s.cuda(), /*joinTwice=*/false, /*autoJoin=*/true);
    ASSERT_NE(graph, nullptr);

    const Topology topology = readTopology(graph);
    EXPECT_EQ(topology.nodes.size(), 4u);

    Graph g;
    g.stream(s.cuda());
    g.addNode(CapturedGraph{ graph });
    Launcher launcher = g.launcher();
    launcher.launch();
    launcher.synchronize();
    expectMatchesOracle(downloadResult());
}

/* ================================================================
 * 5. NEGATIVE — an unjoined fork must FAIL LOUDLY.
 *
 *  This is the load-bearing test for ``EAGLE_CHECK_ALWAYS``. Under the
 *  debug-only ``EAGLE_CHECK``, ``cudaStreamEndCapture`` returned
 *  ``cudaErrorStreamCaptureUnjoined``, the error was discarded, and
 *  ``end()`` handed back a NULL ``cudaGraph_t`` that only exploded much
 *  later. It must now throw at the point of failure.
 *
 *  Kept last in the file: it deliberately invalidates a capture.
 * ================================================================ */
TEST_F(CaptureForkFixture, UnjoinedCaptureThrowsFromEnd)
{
    Stream s(/*nonBlocking=*/true);

    /* The fork outlives end(), so its destructor cannot rescue the
     * capture — exactly the mistake the check has to catch. */
    CaptureFork fork(s.cuda(), 2);
    StreamCapturer capturer(s.cuda());

    capturer.begin();
    scaleKernel<<<1, BLOCK, 0, s.cuda()>>>(d_a, d_x, 2.0, N);

    fork.fork();
    offsetKernel<<<1, BLOCK, 0, fork.branch(0)>>>(d_x, d_y1, 1.0, N);
    scaleKernel<<<1, BLOCK, 0, fork.branch(1)>>>(d_x, d_y2, 3.0, N);
    /* DELIBERATELY OMITTED: fork.join(); */

    EXPECT_THROW(capturer.end(), aether::Error)
        << "an unjoined capture must throw, not return a null graph";

    /* Drain whatever the invalidated capture left behind so the rest of
     * the binary starts from a clean device state. */
    cudaDeviceSynchronize();
    cudaGetLastError();
}

} // namespace CaptureForkTest
} // namespace eagle_tests

#endif
