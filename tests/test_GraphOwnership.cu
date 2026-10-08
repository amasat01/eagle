// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "TestBase.h"

#include "eagle/cuda.h"

#ifndef EAGLE_CPU_ONLY

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace eagle_tests {
namespace GraphOwnershipTest {

using namespace eagle;
using namespace eagle::cuda;

/* ================================================================
 * Helper kernel: writes threadIdx.x + offset into a device buffer
 * ================================================================ */
__global__ void writeKernelOwn(int* buf, int offset, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n)
        buf[tid] = tid + offset;
}

/* Build a freshly captured cudaGraph_t (owned by caller). */
static cudaGraph_t capturedWriteGraph_(
    const cudaStream_t& stream, int* buf, int offset, int n)
{
    StreamCapturer capturer(stream);
    capturer.begin();
    writeKernelOwn<<<1, n, 0, stream>>>(buf, offset, n);
    return capturer.end();
}

/* Probe whether a cudaGraph_t handle is still alive: try to add an
 * empty node to it. Live → returns cudaSuccess. Destroyed → returns
 * cudaErrorInvalidValue (or similar). Restores the graph to its
 * previous state by destroying the freshly-added empty node only on
 * success. Tests use this to assert ownership semantics.
 *
 * VALID ONLY FOR HANDLES KNOWN TO BE LIVE. Calling driver API on a
 * destroyed handle is undefined behavior: the answer depends on whether
 * the driver has recycled the slot, which varies with what previous
 * suites allocated (observed flaking in the full binary while passing in
 * isolation). Post-destroy sites therefore carry no liveness assertion —
 * the load-bearing property (exactly one cudaGraphDestroy, no
 * double-free) is proven by the scoped destructor completing without
 * error and by the compute-sanitizer sanitize target. */
static bool isAlive_(cudaGraph_t g)
{
    if (g == nullptr)
        return false;
    cudaGraphNode_t empty = nullptr;
    cudaError_t err = cudaGraphAddEmptyNode(&empty, g, nullptr, 0);
    if (err == cudaSuccess) {
        cudaGraphDestroyNode(empty);
        return true;
    }
    /* Clear sticky error */
    cudaGetLastError();
    return false;
}

class GraphOwnershipFixture : public Test {
protected:
    static constexpr int N = 32;
    int* d_buf             = nullptr;

    void SetUp() override
    {
        EAGLE_CHECK_ALWAYS(cudaMalloc(&d_buf, N * sizeof(int)));
        EAGLE_CHECK_ALWAYS(cudaMemset(d_buf, 0, N * sizeof(int)));
        // The memset rides the per-thread default stream, which has no ordering
        // with the eagle Streams the test then writes on: finish it first.
        EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    }
    void TearDown() override
    {
        if (d_buf)
            EAGLE_CHECK_ALWAYS(cudaFree(d_buf));
    }
};

/* ================================================================
 * 1. CapturedGraph: RAII destroys on scope exit
 * ================================================================ */
TEST_F(GraphOwnershipFixture, CapturedGraph_RAII_DestroysOnScopeExit)
{
    Stream s;
    cudaGraph_t saved = nullptr;
    {
        CapturedGraph wrapper{ capturedWriteGraph_(s.cuda(), d_buf, 1, N) };
        saved = wrapper.graph();
        EXPECT_NE(saved, nullptr);
        EXPECT_TRUE(isAlive_(saved));
        /* wrapper destroyed at end of block */
    }
    /* No post-destroy probe: UB reference (see isAlive_ doc). Scope exit
     * without error is the destruction evidence. */
}

/* ================================================================
 * 2. CapturedGraph: move transfers ownership; no double-free
 * ================================================================ */
TEST_F(GraphOwnershipFixture, CapturedGraph_MoveTransfersOwnership_NoDoubleFree)
{
    Stream s;
    cudaGraph_t saved = nullptr;
    {
        CapturedGraph a{ capturedWriteGraph_(s.cuda(), d_buf, 2, N) };
        saved = a.graph();
        ASSERT_TRUE(isAlive_(saved));

        CapturedGraph b = std::move(a);
        EXPECT_EQ(a.graph(), nullptr); /* moved-from */
        EXPECT_EQ(b.graph(), saved);
        EXPECT_TRUE(isAlive_(saved));
        /* a's destructor runs first (no-op on null); then b's
         * destructor destroys `saved` exactly once. */
    }
    /* No post-destroy probe: UB reference (see isAlive_ doc). */
}

/* ================================================================
 * 3. CapturedGraph: non-copyable at compile time
 * ================================================================ */
static_assert(!std::is_copy_constructible<CapturedGraph>::value,
    "CapturedGraph must not be copy-constructible (ownership is exclusive)");
static_assert(!std::is_copy_assignable<CapturedGraph>::value,
    "CapturedGraph must not be copy-assignable (ownership is exclusive)");
static_assert(std::is_nothrow_move_constructible<CapturedGraph>::value,
    "CapturedGraph must be nothrow move-constructible");
static_assert(std::is_nothrow_move_assignable<CapturedGraph>::value,
    "CapturedGraph must be nothrow move-assignable");

TEST_F(GraphOwnershipFixture, CapturedGraph_NonCopyable_ProgrammerError_AtCompileTime)
{
    /* Anchor the static_asserts in a runnable test so the suite
     * surface lists this contract explicitly. */
    SUCCEED();
}

/* ================================================================
 * 4. addNode(cudaGraph_t) is borrowed: does NOT destroy the source
 * ================================================================ */
TEST_F(GraphOwnershipFixture, BorrowedAddNode_DoesNotDestroySource)
{
    Stream s;
    cudaGraph_t source = capturedWriteGraph_(s.cuda(), d_buf, 3, N);
    ASSERT_TRUE(isAlive_(source));

    {
        Graph g;
        g.stream(s.cuda());
        g.addNode(source); /* borrowed overload */
        Launcher l = g.launcher();
        l.launch();
        l.synchronize();
    } /* g + l drop; source must NOT have been destroyed */

    EXPECT_TRUE(isAlive_(source));
    /* Caller still owns it: destroy explicitly. No post-destroy probe:
     * UB reference (see isAlive_ doc). */
    EAGLE_CHECK_ALWAYS(cudaGraphDestroy(source));
}

/* ================================================================
 * 5. addNode(CapturedGraph) is owned: destroys the source after
 *    cudaGraphAddChildGraphNode returns
 * ================================================================ */
TEST_F(GraphOwnershipFixture, OwnedAddNode_DestroysSourceImmediately)
{
    Stream s;
    cudaGraph_t saved = nullptr;
    {
        Graph g;
        g.stream(s.cuda());
        CapturedGraph wrapper{ capturedWriteGraph_(s.cuda(), d_buf, 4, N) };
        saved = wrapper.graph();
        ASSERT_TRUE(isAlive_(saved));

        g.addNode(std::move(wrapper)); /* owned overload */
        /* By-value parameter destructor on `child` already ran inside
         * addNode → source must already be destroyed. Moved-from state
         * is asserted below; no post-destroy probe (UB reference, see
         * isAlive_ doc). */
        EXPECT_EQ(wrapper.graph(), nullptr); /* moved-from */

        Launcher l = g.launcher();
        l.launch();
        l.synchronize();
    }
    EXPECT_FALSE(isAlive_(saved));
}

/* ================================================================
 * 6. Terminator-style: borrowed source survives multiple consumers
 *
 *    Models the eagle::Integrator + a downstream Propagator pattern
 *    that produced the original COIRestoreTest double-free. A
 *    helper holds its own eagle::Graph and hands out its raw
 *    cudaGraph_t handle (borrowed). Two separate consumer graphs
 *    add it via addNode(cudaGraph_t). All three lifetimes are
 *    independent; no double-free regardless of destruction order.
 * ================================================================ */
class BorrowedHandleProducer {
public:
    BorrowedHandleProducer(const cudaStream_t& stream, int* buf, int n)
    {
        graph_.stream(stream);
        /* Build one captured graph node into the producer's own
         * eagle::Graph, then hand the raw cudaGraph_t to consumers. */
        graph_.addNode(
            CapturedGraph{ capturedWriteGraph_(stream, buf, 7, n) });
    }
    cudaGraph_t graph() const { return graph_.graph(); }
private:
    Graph graph_;
};

TEST_F(GraphOwnershipFixture, MultipleAddNodeFromSameTerminator_NoDoubleFree)
{
    Stream s;
    BorrowedHandleProducer producer(s.cuda(), d_buf, N);
    cudaGraph_t borrowed = producer.graph();
    ASSERT_TRUE(isAlive_(borrowed));

    {
        Graph parentA;
        parentA.stream(s.cuda());
        parentA.addNode(borrowed); /* borrowed overload */
        Launcher la = parentA.launcher();
        la.launch();
        la.synchronize();

        Graph parentB;
        parentB.stream(s.cuda());
        parentB.addNode(borrowed); /* borrowed overload again */
        Launcher lb = parentB.launcher();
        lb.launch();
        lb.synchronize();
    } /* parents and launchers drop; producer stays alive */

    EXPECT_TRUE(isAlive_(borrowed));
    /* Now drop producer; only one cudaGraphDestroy happens internally. */
}

/* ================================================================
 * 7. addHostNode does not leak the captured graph
 *
 *    Pre-fix: addHostNode routed through the borrowed addNode
 *    overload → the captured graph leaked every call. Post-fix: it
 *    routes through the owned overload via CapturedGraph wrapping
 *    → captured graph is destroyed inside the addNode call.
 *
 *    Probe: snapshot free GPU memory before and after a tight loop
 *    of build-Graph-with-addHostNode-and-drop. Per-iter allocations
 *    are bounded; total stays flat post-fix.
 * ================================================================ */
TEST_F(GraphOwnershipFixture, AddHostNode_NoLeak)
{
    Stream s;
    int counter = 0;

    /* Warm-up to settle CUDA driver-side allocators. */
    for (int i = 0; i < 4; ++i) {
        Graph g;
        g.stream(s.cuda());
        g.addHostNode([&counter]() { counter++; });
        Launcher l = g.launcher();
        l.launch();
        l.synchronize();
    }

    /* cudaMemGetInfo is DEVICE-wide: another process allocating on the same
     * GPU during the loop shows up as a "leak" here. So the probe is repeated
     * and the SMALLEST delta judged: a real leak loses memory on every trial,
     * while a co-tenant has to allocate inside every single window to fake
     * one. */
    constexpr int kIters  = 64;
    constexpr int kTrials = 5;
    ptrdiff_t delta       = PTRDIFF_MAX;
    for (int trial = 0; trial < kTrials; ++trial) {
        size_t freeBefore = 0, freeAfter = 0, total = 0;
        EAGLE_CHECK_ALWAYS(cudaMemGetInfo(&freeBefore, &total));
        for (int i = 0; i < kIters; ++i) {
            Graph g;
            g.stream(s.cuda());
            g.addHostNode([&counter]() { counter++; });
            Launcher l = g.launcher();
            l.launch();
            l.synchronize();
        }
        EAGLE_CHECK_ALWAYS(cudaMemGetInfo(&freeAfter, &total));
        delta = std::min(delta, static_cast<ptrdiff_t>(freeBefore)
                                    - static_cast<ptrdiff_t>(freeAfter));
    }

    /* Pre-fix this leaked one captured cudaGraph_t per iter (a few
     * KB each), so 64 iters would lose tens of KB and the delta
     * would dominate. Post-fix: any noise is below a per-iter
     * allocator quantum (typically 64 KB granularity). Use a loose
     * upper bound that catches the pre-fix regression while
     * ignoring CUDA driver bookkeeping noise. */
    EXPECT_LE(delta, static_cast<ptrdiff_t>(256 * 1024))
        << "addHostNode leaked: " << delta << " bytes over " << kIters
        << " iters (smallest of " << kTrials << " trials)";

    /* Sanity: the host callback ran on every iteration (warm-up + main). */
    EXPECT_EQ(counter, kTrials * kIters + 4);
}

/* ================================================================
 * 8. Owned-input end-to-end: CapturedGraph → addNode → launch
 *
 *    Smoke test for the full owned-handle path: build a CapturedGraph
 *    locally (mirroring what Slice::scanGraph / scatterGraph and
 *    addHostNode produce), pass to addNode(CapturedGraph), launch
 *    the parent, verify the kernel ran. Confirms the owned
 *    semantics work end-to-end without leaks or double-frees.
 * ================================================================ */
TEST_F(GraphOwnershipFixture, OwnedCapturedGraph_AddNode_EndToEnd)
{
    Stream s;
    {
        Graph g;
        g.stream(s.cuda());
        g.addNode(
            CapturedGraph{ capturedWriteGraph_(s.cuda(), d_buf, 11, N) });
        Launcher l = g.launcher();
        l.launch();
        l.synchronize();
    }
    /* Verify the kernel ran. */
    std::vector<int> host(N, -1);
    EAGLE_CHECK_ALWAYS(cudaMemcpy(host.data(), d_buf, N * sizeof(int),
        cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
        EXPECT_EQ(host[i], i + 11);
    }
}

} // namespace GraphOwnershipTest
} // namespace eagle_tests

#endif
