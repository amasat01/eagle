// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Acceptance test for eagle::conditional::ConditionalGroup and
 * eagle::cuda::CaptureConditional (CUDA face).
 *
 * G-BEHAV's central trap: an output-value assertion cannot distinguish "the
 * node was skipped" from "the node fired and its threads early-returned" --
 * it passes identically either way. The skip-proof is therefore an
 * UNGUARDED TRACER: a kernel inside the guarded body that increments a
 * counter with NO liveness check of its own. A fired body cannot leave it
 * silent; a skipped body cannot advance it.
 *
 * ThreeLegSkipProof builds ONE graph, instantiates it ONCE, and replays the
 * SAME Launcher three times, mutating only the guard's device word between
 * legs -- never rebuilding the graph. That is what proves (leg 3) that the
 * implementation isn't secretly baking a host-read of the predicate at
 * build time: a build-time-baked "true" would still show tracer==1 after
 * leg 2 (vacuously), but could not re-fire on leg 3 without a rebuild.
 */
#include <cuda_runtime.h>

#include "eagle/conditional.h"

#include "TestBase.h"

#include <fstream>
#include <iterator>
#include <string>
#include <utility>

namespace eagle_tests {
namespace ConditionalGroupTest {

using eagle::CountGuard;
using eagle::conditional::ConditionalGroup;

namespace {

/* The unguarded tracer: no fencepost read, no liveness check -- just an
 * unconditional increment. Its only defense against ordering hazards is
 * atomicAdd (this is deliberately the SIMPLEST possible witness). */
__global__ void tracerKernel(int* counter) { atomicAdd(counter, 1); }

/* Adds a bare tracerKernel node to `g`, auto-chaining onto whatever was
 * added last (or as the sole node if `g` is otherwise empty of user work). */
void addTracerNode(eagle::cuda::Graph& g, int** counterSlot)
{
    cudaKernelNodeParams kp = {};
    void* args[]            = { (void*)counterSlot };
    kp.func                 = (void*)tracerKernel;
    kp.gridDim              = { 1, 1, 1 };
    kp.blockDim             = { 1, 1, 1 };
    kp.kernelParams         = args;
    g.addKernelNode(kp, {});
}

}  // namespace

class ConditionalGroupTest : public Test {};

// G-BEHAV primary: the three-leg skip-proof, ONE instantiated graph, no
// rebuild between legs.
TEST_F(ConditionalGroupTest, ThreeLegSkipProof)
{
    unsigned int* count;
    int* tracer;
    EAGLE_CHECK_ALWAYS(cudaMalloc(&count, sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&tracer, sizeof(int)));
    EAGLE_CHECK_ALWAYS(cudaMemset(tracer, 0, sizeof(int)));
    const unsigned int one = 1, zero = 0;
    EAGLE_CHECK_ALWAYS(
        cudaMemcpy(count, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));

    eagle::cuda::Graph g;
    ConditionalGroup group(CountGuard{ count, nullptr });
    addTracerNode(group.body(), &tracer);
    g.addNative(std::move(group), {});
    g.finalizeNatives();

    eagle::cuda::Launcher launcher = g.launcher();

    // Leg 1: guard TRUE (count=1 != baseline 0) -> body fires -> tracer==1.
    launcher.launch();
    launcher.synchronize();
    int host = -1;
    EAGLE_CHECK_ALWAYS(cudaMemcpy(&host, tracer, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(host, 1) << "leg 1 (guard TRUE) must fire the body exactly once";

    // Leg 2: mutate ONLY the guard's device word -> body must NOT fire ->
    // tracer stays at 1. This is the skip-proof: the tracer has no
    // liveness check of its own, so this can only hold if the node itself
    // did not execute.
    EAGLE_CHECK_ALWAYS(
        cudaMemcpy(count, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));
    launcher.launch();
    launcher.synchronize();
    EAGLE_CHECK_ALWAYS(cudaMemcpy(&host, tracer, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(host, 1) << "leg 2 (guard FALSE) must leave the tracer silent";

    // Leg 3: restore guard TRUE, SAME launcher, no rebuild -> tracer==2.
    // Kills a build-time host-read of the predicate: that implementation
    // would have frozen the leg-1 decision and could not re-fire here.
    EAGLE_CHECK_ALWAYS(
        cudaMemcpy(count, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));
    launcher.launch();
    launcher.synchronize();
    EAGLE_CHECK_ALWAYS(cudaMemcpy(&host, tracer, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(host, 2) << "leg 3 (guard restored TRUE) must re-fire the body";

    EAGLE_CHECK_ALWAYS(cudaFree(count));
    EAGLE_CHECK_ALWAYS(cudaFree(tracer));
}

// A ConditionalGroup nested inside another's body() fails LOUDLY at
// buildInto: once the nested group's own buildInto makes the outer body
// graph conditional-bearing, entering that graph as the outer IF node's
// child is rejected by the driver (cudaErrorNotSupported, 801), and
// EAGLE_CHECK_ALWAYS turns that into a thrown aether::Error.
TEST_F(ConditionalGroupTest, NestedGroupThrowsNotSupported)
{
    unsigned int* count;
    int* dummy;
    EAGLE_CHECK_ALWAYS(cudaMalloc(&count, sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&dummy, sizeof(int)));
    const unsigned int one = 1;
    EAGLE_CHECK_ALWAYS(
        cudaMemcpy(count, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));

    eagle::cuda::Graph g;
    ConditionalGroup outer(CountGuard{ count, nullptr });
    ConditionalGroup inner(CountGuard{ count, nullptr });
    addTracerNode(inner.body(), &dummy);
    outer.body().addNative(std::move(inner), {});
    g.addNative(std::move(outer), {});

    ASSERT_THROW(g.finalizeNatives(), aether::Error)
        << "nesting a ConditionalGroup must fail loudly (801), not silently";

    EAGLE_CHECK_ALWAYS(cudaFree(count));
    EAGLE_CHECK_ALWAYS(cudaFree(dummy));
}

// Smoke test for eagle::cuda::CaptureConditional -- the capture-face sibling
// (the Python surface mechanism). Not a full three-leg gate (no Python
// binding yet), but must compile and demonstrably weave a real skip.
TEST_F(ConditionalGroupTest, CaptureConditionalSmoke)
{
    unsigned int* count;
    int* tracer;
    EAGLE_CHECK_ALWAYS(cudaMalloc(&count, sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&tracer, sizeof(int)));
    EAGLE_CHECK_ALWAYS(cudaMemset(tracer, 0, sizeof(int)));
    const unsigned int one = 1, zero = 0;

    cudaStream_t origin;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&origin));

    eagle::cuda::CaptureConditional weave(origin, CountGuard{ count, nullptr });

    EAGLE_CHECK_ALWAYS(
        cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
    cudaStream_t bodyStream = weave.begin();
    tracerKernel<<<1, 1, 0, bodyStream>>>(tracer);
    weave.end();
    cudaGraph_t graph = nullptr;
    EAGLE_CHECK_ALWAYS(cudaStreamEndCapture(origin, &graph));

    cudaGraphExec_t exec;
    EAGLE_CHECK_ALWAYS(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    // guard TRUE -> fires.
    EAGLE_CHECK_ALWAYS(
        cudaMemcpy(count, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));
    EAGLE_CHECK_ALWAYS(cudaGraphLaunch(exec, 0));
    EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(0));
    int host = -1;
    EAGLE_CHECK_ALWAYS(cudaMemcpy(&host, tracer, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(host, 1);

    // guard FALSE, SAME exec, no rebuild -> stays silent.
    EAGLE_CHECK_ALWAYS(
        cudaMemcpy(count, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));
    EAGLE_CHECK_ALWAYS(cudaGraphLaunch(exec, 0));
    EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(0));
    EAGLE_CHECK_ALWAYS(cudaMemcpy(&host, tracer, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(host, 1);

    EAGLE_CHECK_ALWAYS(cudaGraphExecDestroy(exec));
    EAGLE_CHECK_ALWAYS(cudaGraphDestroy(graph));
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(origin));
    EAGLE_CHECK_ALWAYS(cudaFree(count));
    EAGLE_CHECK_ALWAYS(cudaFree(tracer));
}

// An IF node whose body is empty never completes when it fires (the stream
// hangs), so both faces refuse an empty guarded body at build time.
TEST_F(ConditionalGroupTest, EmptyBodyIsRefused)
{
    unsigned int* count;
    EAGLE_CHECK_ALWAYS(cudaMalloc(&count, sizeof(unsigned int)));

    {
        eagle::cuda::Graph g;
        ConditionalGroup group(CountGuard{ count, nullptr });
        g.addNative(std::move(group), {});
        ASSERT_THROW(g.finalizeNatives(), std::invalid_argument)
            << "ConditionalGroup must refuse an empty body";
    }

    cudaStream_t origin;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&origin));
    {
        eagle::cuda::CaptureConditional weave(origin, CountGuard{ count, nullptr });
        EAGLE_CHECK_ALWAYS(
            cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
        weave.begin();
        EXPECT_THROW(weave.end(), std::invalid_argument)
            << "CaptureConditional must refuse a region that launched nothing";
        cudaGraph_t graph = nullptr;
        EAGLE_CHECK_ALWAYS(cudaStreamEndCapture(origin, &graph));
        EAGLE_CHECK_ALWAYS(cudaGraphDestroy(graph));
    }
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(origin));
    EAGLE_CHECK_ALWAYS(cudaFree(count));
}

// Body kernel for the loop tests: one step of work that also drains a live
// count the guard reads -- the early-termination shape (a kernel decides when the loop
// ends).
static __global__ void bumpDecKernel(unsigned int* live)
{
    if (*live)
        --(*live);
}

// The WHILE kind of eagle::cuda::CaptureConditional: stop-at-guard, stop-at-
// cap, and the per-launch reset by the head setter (no memset node). The
// tracer is the independent witness of how many times the body ran.
// RED: remove `*count != base` from setLoopTailKernel -> ran == 64 in leg 1;
//      remove `counter[0] = cap` from setLoopHeadKernel -> leg 3 runs 0.
TEST_F(ConditionalGroupTest, CaptureLoopStopsAtGuardAndCap)
{
    unsigned int* live;
    unsigned int* cell;  // [remaining, ran], eagle-owned semantics, caller-owned memory
    int* tracer;
    EAGLE_CHECK_ALWAYS(cudaMalloc(&live, sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&cell, 2 * sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&tracer, sizeof(int)));
    EAGLE_CHECK_ALWAYS(cudaMemset(cell, 0, 2 * sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMemset(tracer, 0, sizeof(int)));

    cudaStream_t origin;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&origin));

    eagle::cuda::CaptureConditional weave(origin, CountGuard{ live, nullptr },
        /*loopCap=*/64u, cell);
    ASSERT_TRUE(weave.isLoop());

    EAGLE_CHECK_ALWAYS(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
    cudaStream_t body = weave.begin();
    bumpDecKernel<<<1, 1, 0, body>>>(live);
    tracerKernel<<<1, 1, 0, body>>>(tracer);
    weave.end();
    cudaGraph_t graph = nullptr;
    EAGLE_CHECK_ALWAYS(cudaStreamEndCapture(origin, &graph));

    // Outer graph: exactly the head setter + the WHILE node; no memset node.
    std::size_t outerNodes = 0;
    EAGLE_CHECK_ALWAYS(cudaGraphGetNodes(graph, nullptr, &outerNodes));
    EXPECT_EQ(outerNodes, 2u);

    cudaGraphExec_t exec;
    EAGLE_CHECK_ALWAYS(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    auto run = [&](unsigned int guard0) {
        EAGLE_CHECK_ALWAYS(cudaMemcpy(live, &guard0, sizeof(unsigned int), cudaMemcpyHostToDevice));
        EAGLE_CHECK_ALWAYS(cudaMemset(tracer, 0, sizeof(int)));
        EAGLE_CHECK_ALWAYS(cudaGraphLaunch(exec, 0));
        EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(0));
        unsigned int hostCell[2];
        int hostTracer = -1;
        EAGLE_CHECK_ALWAYS(cudaMemcpy(hostCell, cell, sizeof hostCell, cudaMemcpyDeviceToHost));
        EAGLE_CHECK_ALWAYS(cudaMemcpy(&hostTracer, tracer, sizeof(int), cudaMemcpyDeviceToHost));
        EXPECT_EQ(static_cast<int>(hostCell[1]), hostTracer) << "ran cell == tracer";
        return std::pair<unsigned int, unsigned int>{ hostCell[0], hostCell[1] };
    };

    auto leg1 = run(5);      // stop at guard
    EXPECT_EQ(leg1.second, 5u);
    EXPECT_EQ(leg1.first, 59u);
    auto leg2 = run(1000);   // stop at cap
    EXPECT_EQ(leg2.second, 64u);
    EXPECT_EQ(leg2.first, 0u);
    auto leg3 = run(1000);   // SAME exec: head reset -> 64 again, not 0, not 128
    EXPECT_EQ(leg3.second, 64u);
    auto leg4 = run(0);      // zero-iteration entry completes
    EXPECT_EQ(leg4.second, 0u);
    EXPECT_EQ(leg4.first, 64u);

    EAGLE_CHECK_ALWAYS(cudaGraphExecDestroy(exec));
    EAGLE_CHECK_ALWAYS(cudaGraphDestroy(graph));
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(origin));
    EAGLE_CHECK_ALWAYS(cudaFree(live));
    EAGLE_CHECK_ALWAYS(cudaFree(cell));
    EAGLE_CHECK_ALWAYS(cudaFree(tracer));
}

// An IF weave whose origin is a WHILE weave's bodyStream(): the skippable-
// inside-a-loop composition at the C++ level (the Python weave's mechanism).
// RED: construct the inner weave against `origin` instead of
//      outer.bodyStream() -> begin() asserts (origin not capturing) or the
//      IF lands outside the loop and the tracer counts 1, not 8.
TEST_F(ConditionalGroupTest, CaptureLoopNestedSkippable)
{
    unsigned int *guard, *flag, *cell;
    int* tracer;
    EAGLE_CHECK_ALWAYS(cudaMalloc(&guard, sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&flag, sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&cell, 2 * sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&tracer, sizeof(int)));
    const unsigned int one = 1, zero = 0;
    EAGLE_CHECK_ALWAYS(cudaMemcpy(guard, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));

    cudaStream_t origin;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&origin));
    eagle::cuda::CaptureConditional outer(origin, CountGuard{ guard, nullptr }, 8u, cell);
    eagle::cuda::CaptureConditional inner(outer.bodyStream(), CountGuard{ flag, nullptr });

    EAGLE_CHECK_ALWAYS(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
    outer.begin();
    cudaStream_t innerBody = inner.begin();
    tracerKernel<<<1, 1, 0, innerBody>>>(tracer);
    inner.end();
    outer.end();
    cudaGraph_t graph = nullptr;
    EAGLE_CHECK_ALWAYS(cudaStreamEndCapture(origin, &graph));
    cudaGraphExec_t exec;
    EAGLE_CHECK_ALWAYS(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

    for (unsigned int fl : { zero, one }) {
        EAGLE_CHECK_ALWAYS(cudaMemcpy(flag, &fl, sizeof(unsigned int), cudaMemcpyHostToDevice));
        EAGLE_CHECK_ALWAYS(cudaMemset(tracer, 0, sizeof(int)));
        EAGLE_CHECK_ALWAYS(cudaGraphLaunch(exec, 0));
        EAGLE_CHECK_ALWAYS(cudaStreamSynchronize(0));
        unsigned int ran = 0;
        int hostTracer = -1;
        EAGLE_CHECK_ALWAYS(cudaMemcpy(&ran, cell + 1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        EAGLE_CHECK_ALWAYS(cudaMemcpy(&hostTracer, tracer, sizeof(int), cudaMemcpyDeviceToHost));
        EXPECT_EQ(ran, 8u) << "the loop runs to its cap regardless of the inner flag";
        EXPECT_EQ(hostTracer, fl ? 8 : 0) << "the inner IF follows its flag on every iteration";
    }

    EAGLE_CHECK_ALWAYS(cudaGraphExecDestroy(exec));
    EAGLE_CHECK_ALWAYS(cudaGraphDestroy(graph));
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(origin));
    EAGLE_CHECK_ALWAYS(cudaFree(guard));
    EAGLE_CHECK_ALWAYS(cudaFree(flag));
    EAGLE_CHECK_ALWAYS(cudaFree(cell));
    EAGLE_CHECK_ALWAYS(cudaFree(tracer));
}

// An empty WHILE body is an infinite loop by construction: end() must refuse
// it BEFORE launching its own tail setter (which would make the body look
// non-empty), and the origin capture must still close cleanly. No launch
// here -- the RED (refusal deleted) would only hang on launch, which the
// Python row (e) exercises under a process timeout.
// RED: move the node count after the tail launch -> no throw.
TEST_F(ConditionalGroupTest, EmptyLoopBodyIsRefused)
{
    unsigned int *guard, *cell;
    EAGLE_CHECK_ALWAYS(cudaMalloc(&guard, sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&cell, 2 * sizeof(unsigned int)));
    cudaStream_t origin;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&origin));
    {
        eagle::cuda::CaptureConditional weave(origin, CountGuard{ guard, nullptr }, 64u, cell);
        EAGLE_CHECK_ALWAYS(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
        weave.begin();
        EXPECT_THROW(weave.end(), std::invalid_argument)
            << "CaptureConditional must refuse an empty loop body";
        cudaGraph_t graph = nullptr;
        EAGLE_CHECK_ALWAYS(cudaStreamEndCapture(origin, &graph));
        // The tail was never launched: the graph dot names no tail setter.
        const std::string path = std::string(::testing::TempDir()) + "empty_loop.dot";
        EAGLE_CHECK_ALWAYS(cudaGraphDebugDotPrint(graph, path.c_str(), 0));
        std::ifstream in(path);
        const std::string dot((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        EXPECT_EQ(dot.find("setLoopTailKernel"), std::string::npos);
        EAGLE_CHECK_ALWAYS(cudaGraphDestroy(graph));
    }
    // loopCap == 0 is refused at construction, before any capture.
    EXPECT_THROW((eagle::cuda::CaptureConditional{ origin, CountGuard{ guard, nullptr }, 0u, cell }),
        std::invalid_argument);
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(origin));
    EAGLE_CHECK_ALWAYS(cudaFree(guard));
    EAGLE_CHECK_ALWAYS(cudaFree(cell));
}

// Is Graph::launcher()'s
// documented "callable more than once, each instantiating independently"
// still true once the graph is conditional-bearing? CUDA's
// own docs say NO ("only one instantiation of the graph may exist at any
// point in time" for a conditional-bearing graph) -- this test builds ONE
// such graph and calls launcher() TWICE.
//
// Previously (`EAGLE_CHECK` in `Launcher::instantiate_`, release-mode-silent):
// the second launcher() did NOT throw; the underlying `cudaGraphInstantiate`
// returned `cudaErrorNotSupported` (801) recoverable only via
// `cudaGetLastError()`, and the resulting exec was entirely INERT (never
// fired ANY node, conditional or not) while the first launcher kept working.
//
// Now (`EAGLE_CHECK_ALWAYS`): the SAME 801 rejection is reported
// loudly -- the second `launcher()` call must THROW `aether::Error`
// instead of returning a silently inert exec. This is the acceptance test
// for that flip; the first launcher must remain entirely unaffected.
TEST_F(ConditionalGroupTest, MultiLauncherFanOut)
{
    unsigned int* count;
    int* tracer;
    int* marker;  // unconditional top-level node: would prove the WHOLE exec
                  // is dead, not just the guarded body, if it ever fired.
    EAGLE_CHECK_ALWAYS(cudaMalloc(&count, sizeof(unsigned int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&tracer, sizeof(int)));
    EAGLE_CHECK_ALWAYS(cudaMalloc(&marker, sizeof(int)));
    EAGLE_CHECK_ALWAYS(cudaMemset(tracer, 0, sizeof(int)));
    EAGLE_CHECK_ALWAYS(cudaMemset(marker, 0, sizeof(int)));
    const unsigned int one = 1;
    EAGLE_CHECK_ALWAYS(
        cudaMemcpy(count, &one, sizeof(unsigned int), cudaMemcpyHostToDevice));

    eagle::cuda::Graph g;
    addTracerNode(g, &marker);  // unconditional, top-level
    ConditionalGroup group(CountGuard{ count, nullptr });
    addTracerNode(group.body(), &tracer);
    g.addNative(std::move(group), {});
    g.finalizeNatives();

    cudaGetLastError();  // clear any pre-existing sticky error before probing
    eagle::cuda::Launcher launcher1 = g.launcher();
    ASSERT_EQ(cudaGetLastError(), cudaSuccess)
        << "the FIRST launcher() must instantiate cleanly";

    bool threw = false;
    std::string thrownWhat;
    eagle::cuda::Launcher launcher2;
    try {
        launcher2 = g.launcher();
    } catch (const aether::Error& e) {
        threw      = true;
        thrownWhat = e.what();
    }

    ASSERT_TRUE(threw)
        << "REGRESSION: second launcher() on a conditional-bearing graph "
           "must throw aether::Error (the driver's cudaErrorNotSupported "
           "801) -- got a silent return instead, i.e. Launcher::instantiate_ "
           "is back to swallowing the failure.";
    EXPECT_NE(thrownWhat.find("not supported"), std::string::npos)
        << "expected the cudaErrorNotSupported (801) message; got: " << thrownWhat;

    // The FIRST launcher must be entirely unaffected by the failed second
    // instantiate attempt.
    int host = -1;
    launcher1.launch();
    launcher1.synchronize();
    EAGLE_CHECK_ALWAYS(cudaMemcpy(&host, marker, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(host, 1);
    EAGLE_CHECK_ALWAYS(cudaMemcpy(&host, tracer, sizeof(int), cudaMemcpyDeviceToHost));
    ASSERT_EQ(host, 1);

    EAGLE_CHECK_ALWAYS(cudaFree(count));
    EAGLE_CHECK_ALWAYS(cudaFree(tracer));
    EAGLE_CHECK_ALWAYS(cudaFree(marker));
}

}  // namespace ConditionalGroupTest
}  // namespace eagle_tests
