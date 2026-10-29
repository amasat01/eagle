// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
//
// eagle::array::Array: views share storage, offsets and strides follow python's slicing, reshape refuses a
// strided view, borrowed memory is kept alive and released once, the last view frees. CUDA-free: both modes.

#include <gtest/gtest.h>

#include "eagle/array/Array.h"

namespace ea = eagle::array;

namespace eagle_tests {
namespace ArrayTest {

static ea::Array iota(std::vector<std::int64_t> shape)
{
    ea::Array a = ea::Array::empty(shape, ea::DType::F64);
    for (std::int64_t j = 0; j < a.size(); ++j) static_cast<double*>(a.data())[j] = double(j);
    return a;
}
static double at(const ea::Array& a, std::int64_t i, std::int64_t j)
{
    return *(reinterpret_cast<const double*>(static_cast<const char*>(a.data()) + (i * a.stride(0) + j * a.stride(1)) * 8));
}

TEST(ArrayTest, AnEmptyArrayIsContiguousRowMajorAndOwnsItsBytes)
{
    auto a = iota({ 4, 6 });
    EXPECT_EQ(a.rank(), 2u);
    EXPECT_EQ(a.stride(0), 6);
    EXPECT_EQ(a.stride(1), 1);
    EXPECT_EQ(a.size(), 24);
    EXPECT_EQ(a.nbytes(), 192u);
    EXPECT_TRUE(a.contiguous());
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(a.data()) % 64, 0u);
}

TEST(ArrayTest, SlicesAndIndexAreViewsByOffsetAndStride)
{
    auto a = iota({ 5, 6 });
    auto v = a.slice(0, 1, 5, 2).slice(1, 2, 6, 3);        // rows 1,3 ; cols 2,5
    EXPECT_EQ(v.extent(0), 2);
    EXPECT_EQ(v.extent(1), 2);
    EXPECT_EQ(v.byte_offset(), (1 * 6 + 2) * 8);
    EXPECT_EQ(at(v, 1, 1), 3 * 6 + 5);
    auto r = a.slice(0, 4, -6, -2);                         // python a[4:-6:-2] -> rows 4,2,0
    EXPECT_EQ(r.extent(0), 3);
    EXPECT_EQ(r.stride(0), -12);
    EXPECT_EQ(at(r, 2, 0), 0.0);
    EXPECT_EQ(a.slice(0, 3, 3).extent(0), 0);
    auto row = a.index(-1);
    EXPECT_EQ(row.rank(), 1u);
    EXPECT_EQ(row.byte_offset(), 4 * 6 * 8);
    EXPECT_THROW(a.index(5), std::out_of_range);
    EXPECT_THROW(a.slice(2, 0, 1), std::out_of_range);
    EXPECT_EQ(a.use_count(), 4);                            // a, v, r and row share one storage
}

TEST(ArrayTest, ReshapeTransposeAndViewAsShareMemory)
{
    auto a = iota({ 4, 6 });
    EXPECT_EQ(a.reshape({ 2, 12 }).data(), a.data());
    EXPECT_EQ(a.reshape({ 24 }).rank(), 1u);
    EXPECT_THROW(a.reshape({ 5, 5 }), std::invalid_argument);
    auto t = a.transpose();
    EXPECT_EQ(t.extent(0), 6);
    EXPECT_EQ(t.stride(0), 1);
    EXPECT_FALSE(t.contiguous());
    EXPECT_THROW(t.reshape({ 24 }), std::invalid_argument);
    EXPECT_THROW(a.slice(1, 0, 3).reshape({ 12 }), std::invalid_argument);
    auto bytes = a.view_as(ea::DType::U8);
    EXPECT_EQ(bytes.extent(1), 48);
    EXPECT_EQ(bytes.view_as(ea::DType::F64).extent(1), 6);
    EXPECT_THROW(t.view_as(ea::DType::U8), std::invalid_argument);
    EXPECT_THROW(ea::Array::empty({ 7, 1, 1, 1, 1, 1, 1 }, ea::DType::F64), std::invalid_argument);
}

TEST(ArrayTest, TheLastViewFreesTheStorageOnce)
{
    int released = 0;
    auto alloc = ea::allocator();
    ea::allocator() = [&](ea::Placement p, std::size_t bytes, std::uint64_t s) {
        auto st = alloc(p, bytes, s);
        auto inner = std::move(st->release);
        st->release = [&released, inner] { ++released; inner(); };
        return st;
    };
    {
        auto a = iota({ 8 });
        auto v = a.slice(0, 2, 6);
        a = ea::Array();
        EXPECT_EQ(released, 0);          // the view still holds it
        EXPECT_EQ(static_cast<double*>(v.data())[0], 2.0);
    }
    EXPECT_EQ(released, 1);
    ea::allocator() = alloc;
}

TEST(ArrayTest, BorrowedMemoryKeepsItsOwnerAliveUntilTheLastView)
{
    auto owner = std::make_shared<int>(7);
    std::weak_ptr<int> watch = owner;
    double host[6] = { 0, 1, 2, 3, 4, 5 };
    {
        auto a = ea::Array::borrow(host, { 2, 3 }, { 3, 1 }, ea::DType::F64, {}, owner);
        owner.reset();
        EXPECT_FALSE(watch.expired());
        auto v = a.index(1);
        a = ea::Array();
        EXPECT_FALSE(watch.expired());
        EXPECT_EQ(static_cast<double*>(v.data())[0], 3.0);
    }
    EXPECT_TRUE(watch.expired());
    EXPECT_THROW(ea::Array::borrow(host, { 2 }, { 1, 1 }, ea::DType::F64, {}), std::invalid_argument);
    EXPECT_THROW(ea::Array::empty({ 4 }, ea::DType::F64, ea::Placement{ ea::Placement::Kind::Cuda, 0 }), std::runtime_error);
}

}  // namespace ArrayTest
}  // namespace eagle_tests
