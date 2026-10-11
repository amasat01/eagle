// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime_api.h>

#include "eagle/cuda/Reduction.h"
#include "eagle/graph/Describe.h"
#include "eagle/graph/Layout.h"
#include "eagle/reduce/ReductionNode.h"

#include "TestBase.h"

namespace eagle_tests {
namespace GraphViewCudaTest {

using idx_t = eagle::idx_t;
using SumOp = aether::SumOp<double>;
using RN    = eagle::reduce::ReductionNode<double, SumOp>;

class GraphViewCudaTest : public Test {};

// The CUDA graph describes itself with the same code as the host graph: one
// node per native node, an edge per declared dependency.
TEST_F(GraphViewCudaTest, DescribeCudaGraphEdgesFollowDeps)
{
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    const idx_t n = 1024;
    aether::Array<double> a = eagle::makeArray<double>(n);
    aether::Array<double> b = eagle::makeArray<double>(n);
    cudaStream_t stream;
    EAGLE_CHECK_ALWAYS(cudaStreamCreate(&stream));

    double r0 = 0.0, r1 = 0.0, r2 = 0.0;
    eagle::cuda::Graph g;
    const idx_t n0 = g.addNative(RN(&r0, a.deviceView().as_const(), n, 0.0, stream), {});
    const idx_t n1 = g.addNative(RN(&r1, b.deviceView().as_const(), n, 0.0, stream), { n0 });
    g.addNative(RN(&r2, b.deviceView().as_const(), n, 0.0, stream), { n0, n1 });

    ASSERT_EQ(g.nativeDeps().size(), 3u);
    const eagle::graph::Description d = eagle::graph::describe(g);
    ASSERT_EQ(d.nodes.size(), 3u);
    std::vector<std::pair<std::int64_t, std::int64_t>> got;
    for (const eagle::graph::Edge& e : d.edges)
        got.emplace_back(e.src, e.dst);
    const std::vector<std::pair<std::int64_t, std::int64_t>> want
        = { { 0, 1 }, { 0, 2 }, { 1, 2 } };
    EXPECT_EQ(got, want);
    EXPECT_EQ(eagle::graph::layout(d).layer, (std::vector<std::int64_t>{ 0, 1, 2 }));
    EAGLE_CHECK_ALWAYS(cudaStreamDestroy(stream));
}

}  // namespace GraphViewCudaTest
}  // namespace eagle_tests
