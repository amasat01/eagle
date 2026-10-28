// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
//
// The device kernels of the array ops (cell_op 10-23) equal their host twins bit for bit (NaN equal NaN) on the
// special-value corpus, every op on every dtype it defines; the multi-block cumsum has no lane ceiling.

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <omp.h>

#include "ArrayOpsCorpus.h"
#include "eagle/array/host.h"
#include "eagle/array/launch.h"

namespace ea = eagle::array;
namespace tc = eagle_tests::arrays;

namespace eagle_tests {
namespace ArrayOpsDeviceTest {

namespace {
struct Dev {
    void* p = nullptr;
    explicit Dev(const std::vector<std::uint8_t>& host) {
        EXPECT_EQ(cudaMalloc(&p, host.size() ? host.size() : 1), cudaSuccess);
        EXPECT_EQ(cudaMemcpy(p, host.data(), host.size(), cudaMemcpyHostToDevice), cudaSuccess);
    }
    ~Dev() { cudaFree(p); }
    Dev(const Dev&) = delete;
};

// Run `c` on the device and return the destination bytes.
std::vector<std::uint8_t> onDevice(tc::Case& c)
{
    Dev dst(c.dst), x(c.x), y(c.y), mask(c.mask), index(c.index), scratch(c.scratch);
    ea::Args a = c.a;
    a.p0 = dst.p; a.p1 = x.p; a.p2 = y.p; a.p3 = mask.p;
    if (c.op == ea::Op::Take || c.op == ea::Op::Scatter) a.p2 = index.p;
    if (c.op == ea::Op::Cumsum) a.p2 = scratch.p;
    if (c.op == ea::Op::Scatter) {   // scatter writes only the indexed slots: start both sides from the same bytes
        EXPECT_EQ(cudaMemcpy(dst.p, c.dst.data(), c.dst.size(), cudaMemcpyHostToDevice), cudaSuccess);
    }
    ea::device::enqueue(c.op, a, nullptr);
    EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    std::vector<std::uint8_t> out(c.dst.size());
    EXPECT_EQ(cudaMemcpy(out.data(), dst.p, out.size(), cudaMemcpyDeviceToHost), cudaSuccess);
    return out;
}
int cmpKind(const tc::Case& c)
{
    return tc::compare(c.op) || c.dtype < 6 ? 1 : c.dtype;
}
}  // namespace

TEST(ArrayOpsDeviceTest, EveryOpOnEveryDtypeEqualsItsHostTwinBitForBit)
{
    omp_set_num_threads(4);
    for (const auto& [op, dt] : tc::cases()) {
        auto host = tc::make(op, dt), dev = tc::make(op, dt);
        ea::host::run(op, host.a);
        const auto got = onDevice(dev);
        ASSERT_TRUE(tc::same(cmpKind(host), host.dst, got))
            << "op " << static_cast<int>(op) << " dtype " << ea::name(ea::from_code(dt));
    }
}

TEST(ArrayOpsDeviceTest, TheMultiBlockCumsumIsExactWithNoLaneCeiling)
{
    for (std::int64_t n : std::vector<std::int64_t>{ 1, 1023, 1024, 1025, 65537, (std::int64_t(1) << 22) + 13 }) {
        for (int dt : { 2, 3, 4, 5 }) {
            auto host = tc::make(ea::Op::Cumsum, dt, n), dev = tc::make(ea::Op::Cumsum, dt, n);
            ea::host::run(ea::Op::Cumsum, host.a);
            ASSERT_EQ(onDevice(dev), host.dst) << "n " << n << " dtype " << ea::name(ea::from_code(dt));
        }
    }
}

TEST(ArrayOpsDeviceTest, CountNonzeroAndFlatnonzeroEqualTheHostTwinOnEveryDtypeAcrossScanTiles)
{
    omp_set_num_threads(4);
    for (std::int64_t n : { 1, 1031, 20011 })
        for (int dt = 0; dt < ea::kComputeDTypes; ++dt) {
            const auto x = tc::operand(dt, n, true, 1);
            std::int64_t hc = -1, hf = -1;
            std::vector<std::int32_t> hi(static_cast<std::size_t>(n), -1);
            ea::Args a;
            a.n = n; a.dtype = dt; a.p1 = x.data(); a.p0 = &hc;
            ea::host::run(ea::Op::CountNz, a);
            a.p0 = hi.data(); a.p3 = &hf;
            ea::host::run(ea::Op::FlatNz, a);
            Dev dx(x), cell(std::vector<std::uint8_t>(8, 0xAB)), cell2(std::vector<std::uint8_t>(8, 0xAB));
            Dev idx(std::vector<std::uint8_t>(static_cast<std::size_t>(n) * 4, 0xFF)), scratch(std::vector<std::uint8_t>(ea::cumsum_scratch_bytes(n), 0));
            ea::Args d = a;
            d.p1 = dx.p; d.p0 = cell.p;
            ea::device::enqueue(ea::Op::CountNz, d, nullptr);
            d.p0 = idx.p; d.p2 = scratch.p; d.p3 = cell2.p;
            ea::device::enqueue(ea::Op::FlatNz, d, nullptr);
            ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
            std::int64_t dc = -1, df = -1;
            std::vector<std::int32_t> di(static_cast<std::size_t>(n), -1);
            cudaMemcpy(&dc, cell.p, 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(&df, cell2.p, 8, cudaMemcpyDeviceToHost);
            cudaMemcpy(di.data(), idx.p, static_cast<std::size_t>(n) * 4, cudaMemcpyDeviceToHost);
            ASSERT_EQ(dc, hc) << ea::name(ea::from_code(dt)) << " n " << n;
            ASSERT_EQ(df, hf);
            ASSERT_TRUE(std::equal(hi.begin(), hi.begin() + hf, di.begin()));
        }
}

TEST(ArrayOpsDeviceTest, ARefusedCombinationThrowsBeforeAnythingIsEnqueued)
{
    auto c = tc::make(ea::Op::Sub, 0);
    EXPECT_THROW(ea::device::enqueue(ea::Op::Sub, c.a, nullptr), std::invalid_argument);
}

}  // namespace ArrayOpsDeviceTest
}  // namespace eagle_tests
