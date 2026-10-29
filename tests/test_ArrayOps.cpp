// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
//
// The host twins of the array ops (cell_op 10-23): OpenMP + SIMD over the one shared body, run under 4
// threads, equal to a serial loop over the same body on the special-value corpus; plus the dtype truth.
// CUDA-free: compiled and run in both build modes.

#include <gtest/gtest.h>
#include <omp.h>

#include "ArrayOpsCorpus.h"
#include "eagle/array/DType.h"
#include "eagle/array/host.h"

namespace ea = eagle::array;
namespace tc = eagle_tests::arrays;

namespace eagle_tests {
namespace ArrayOpsTest {

namespace {
// The oracle: the same body, one element at a time, no pragma.
void serial(ea::Op op, const ea::Args& a)
{
    using ea::Op;
    ea::with_type(a.dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        if (op == Op::Cumsum) {
            const T* x = static_cast<const T*>(a.p1);
            T* d = static_cast<T*>(a.p0);
            T acc = T(0);
            for (std::int64_t j = 0; j < a.n; ++j) { acc = static_cast<T>(acc + x[j]); d[j] = acc; }
            return;
        }
        for (std::int64_t j = 0; j < a.n; ++j) {
            switch (op) {
            case Op::Add: ea::body<Op::Add, T>(a, j); break;
            case Op::Fill: ea::body<Op::Fill, T>(a, j); break;
            case Op::Sub: ea::body<Op::Sub, T>(a, j); break;
            case Op::Mul: ea::body<Op::Mul, T>(a, j); break;
            case Op::Max: ea::body<Op::Max, T>(a, j); break;
            case Op::Eq: ea::body<Op::Eq, T>(a, j); break;
            case Op::Ne: ea::body<Op::Ne, T>(a, j); break;
            case Op::Gt: ea::body<Op::Gt, T>(a, j); break;
            case Op::Where: ea::body<Op::Where, T>(a, j); break;
            case Op::Clip: ea::body<Op::Clip, T>(a, j); break;
            case Op::Arange: ea::body<Op::Arange, T>(a, j); break;
            case Op::Linspace: ea::body<Op::Linspace, T>(a, j); break;
            case Op::Take: ea::body<Op::Take, T>(a, j); break;
            case Op::Scatter: ea::body<Op::Scatter, T>(a, j); break;
            case Op::AddS: ea::body<Op::AddS, T>(a, j); break;
            case Op::SubS: ea::body<Op::SubS, T>(a, j); break;
            case Op::MulS: ea::body<Op::MulS, T>(a, j); break;
            case Op::MaxS: ea::body<Op::MaxS, T>(a, j); break;
            case Op::EqS: ea::body<Op::EqS, T>(a, j); break;
            case Op::NeS: ea::body<Op::NeS, T>(a, j); break;
            case Op::GtS: ea::body<Op::GtS, T>(a, j); break;
            case Op::RSubS: ea::body<Op::RSubS, T>(a, j); break;
            case Op::LtS: ea::body<Op::LtS, T>(a, j); break;
            default: break;
            }
        }
    });
}
}  // namespace

TEST(ArrayOpsTest, EveryOpOnEveryDtypeUnderFourThreadsEqualsTheSerialBody)
{
    omp_set_num_threads(4);
    const auto all = tc::cases();
    ASSERT_GT(all.size(), 90u);
    for (const auto& [op, dt] : all) {
        auto host = tc::make(op, dt), ref = tc::make(op, dt);
        ea::host::run(op, host.a);
        serial(op, ref.a);
        ASSERT_TRUE(tc::same(dt < 6 || tc::compare(op) ? 1 : dt, host.dst, ref.dst))
            << "op " << static_cast<int>(op) << " dtype " << ea::name(ea::from_code(dt));
    }
}

TEST(ArrayOpsTest, TheHostCumsumIsExactAcrossThreadCountsAndTileBoundaries)
{
    for (int threads : { 1, 2, 3, 4, 7 }) {
        omp_set_num_threads(threads);
        for (std::int64_t n : { 1, 2, 4095, 4096, 4097, 20011 }) {
            auto host = tc::make(ea::Op::Cumsum, 4, n), ref = tc::make(ea::Op::Cumsum, 4, n);
            ea::host::run(ea::Op::Cumsum, host.a);
            serial(ea::Op::Cumsum, ref.a);
            ASSERT_EQ(host.dst, ref.dst) << "threads " << threads << " n " << n;
        }
    }
}

namespace {
// Count and compaction by a plain loop over the same predicate, on the corpus (NaN counts, -0 does not).
struct Compact { std::int64_t count; std::vector<std::int32_t> index; };
Compact reference(int dtype, const std::vector<std::uint8_t>& x, std::int64_t n)
{
    Compact r{ 0, {} };
    ea::with_type(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* p = reinterpret_cast<const T*>(x.data());
        for (std::int64_t j = 0; j < n; ++j)
            if (ea::nonzero(p[j])) { r.index.push_back(static_cast<std::int32_t>(j)); ++r.count; }
    });
    return r;
}
}  // namespace

TEST(ArrayOpsTest, CountNonzeroAndFlatnonzeroEqualAPlainLoopAcrossThreadsAndDtypes)
{
    for (int threads : { 1, 3, 4, 7 }) {
        omp_set_num_threads(threads);
        for (std::int64_t n : { 1, 1031, 20011 })
            for (int dt = 0; dt < ea::kComputeDTypes; ++dt) {
                const auto x = tc::operand(dt, n, true, 1);
                const auto want = reference(dt, x, n);
                std::int64_t count = -1, flat_count = -1;
                std::vector<std::int32_t> index(static_cast<std::size_t>(n), -1);
                ea::Args a;
                a.n = n; a.dtype = dt; a.p1 = x.data();
                a.p0 = &count;
                ea::host::run(ea::Op::CountNz, a);
                a.p0 = index.data(); a.p3 = &flat_count;
                ea::host::run(ea::Op::FlatNz, a);
                ASSERT_EQ(count, want.count) << ea::name(ea::from_code(dt)) << " n " << n << " threads " << threads;
                ASSERT_EQ(flat_count, want.count);
                ASSERT_TRUE(std::equal(want.index.begin(), want.index.end(), index.begin()));
            }
    }
}

TEST(ArrayOpsTest, ArgumentsAreRefusedByName)
{
    auto c = tc::make(ea::Op::Sub, 0);
    EXPECT_THROW(ea::host::run(ea::Op::Sub, c.a), std::invalid_argument);            // bool arithmetic
    auto l = tc::make(ea::Op::Fill, 2);
    EXPECT_THROW(ea::host::run(ea::Op::Linspace, l.a), std::invalid_argument);       // linspace on int
    EXPECT_THROW(ea::host::run(ea::Op::Cumsum, l.a = [&] { auto a = l.a; a.dtype = 6; return a; }()),
                 std::invalid_argument);                                             // cumsum on float
    auto big = l.a;
    big.dtype = 10;
    EXPECT_THROW(ea::host::run(ea::Op::Fill, big), std::invalid_argument);           // names-only dtype
}

TEST(ArrayOpsTest, DTypeCodesNamesSizesAndKindsAreOneTable)
{
    EXPECT_EQ(static_cast<int>(ea::DType::F64), 7);
    EXPECT_STREQ(ea::name(ea::DType::U32), "uint32");
    EXPECT_EQ(ea::itemsize(ea::DType::F16), 2u);
    EXPECT_EQ(ea::kind(ea::DType::I64), 'i');
    EXPECT_EQ(ea::from_name("float64"), ea::DType::F64);
    EXPECT_EQ(ea::from_name("double"), ea::DType::F64);
    EXPECT_THROW(ea::from_name("float128"), std::invalid_argument);
    EXPECT_THROW(ea::from_code(99), std::invalid_argument);
    // numpy's result_type over the compute set, spot rows
    EXPECT_EQ(ea::promote(ea::DType::I32, ea::DType::F32), ea::DType::F64);
    EXPECT_EQ(ea::promote(ea::DType::U8, ea::DType::I32), ea::DType::I32);
    EXPECT_EQ(ea::promote(ea::DType::U32, ea::DType::I32), ea::DType::I64);
    EXPECT_EQ(ea::promote(ea::DType::U64, ea::DType::I64), ea::DType::F64);
    EXPECT_EQ(ea::promote(ea::DType::Bool, ea::DType::F32), ea::DType::F32);
    EXPECT_EQ(ea::promote(ea::DType::F32, ea::DType::F64), ea::DType::F64);
}

TEST(ArrayOpsTest, HalfAndBfloat16ConvertToNearestEvenPromoteAndRefuseByNameBelowTheirHardware)
{
    EXPECT_EQ(ea::Half(1.0).bits, 0x3C00);
    EXPECT_EQ(ea::Half(-2.0).bits, 0xC000);
    EXPECT_EQ(ea::BFloat16(1.0).bits, 0x3F80);
    EXPECT_EQ(static_cast<float>(ea::Half(2049.0)), 2048.0f);       // tie goes to even
    EXPECT_EQ(static_cast<float>(ea::Half(2051.0)), 2052.0f);
    EXPECT_EQ(static_cast<float>(ea::Half(65519.0)), 65504.0f);
    EXPECT_TRUE(std::isinf(static_cast<float>(ea::Half(65520.0))));
    EXPECT_EQ(static_cast<float>(ea::Half(1e-8)), 0.0f);
    EXPECT_EQ(static_cast<float>(ea::Half(3e-8)), 0x1p-24f);          // subnormal
    EXPECT_EQ(static_cast<float>(ea::BFloat16(1.0 / 3.0)), 0.333984375f);
    EXPECT_EQ(ea::promote(ea::DType::F16, ea::DType::F32), ea::DType::F32);
    EXPECT_EQ(ea::promote(ea::DType::F16, ea::DType::I32), ea::DType::F64);
    EXPECT_EQ(ea::promote(ea::DType::F16, ea::DType::U8), ea::DType::F16);
    EXPECT_EQ(ea::promote(ea::DType::F16, ea::DType::BF16), ea::DType::F32);
    EXPECT_THROW(ea::check_arch(ea::Op::Add, 8, 5, 2), std::invalid_argument);
    EXPECT_THROW(ea::check_arch(ea::Op::Mul, 9, 7, 5), std::invalid_argument);
    EXPECT_NO_THROW(ea::check_arch(ea::Op::Add, 8, 5, 3));
    EXPECT_NO_THROW(ea::check_arch(ea::Op::Add, 9, 8, 0));
    EXPECT_NO_THROW(ea::check_arch(ea::Op::Fill, 9, 6, 1));          // moves run everywhere
    EXPECT_NO_THROW(ea::check_arch(ea::Op::Add, 7, 3, 0));
}

}  // namespace ArrayOpsTest
}  // namespace eagle_tests
