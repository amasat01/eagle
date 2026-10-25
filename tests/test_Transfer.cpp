// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/**
 * @file test_Transfer.cpp
 * @brief ``eagle/exec/Transfer.h``, the host face: a copy, page-locked host
 *        memory and a host function on plain memory (C++-only mode; the CUDA
 *        build runs the device face in test_Transfer.cu).
 */

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

#include "eagle/exec/Transfer.h"

#include "TestBase.h"

namespace eagle_tests {
namespace TransferTest {

using eagle::exec::CopyKind;

TEST(TransferTest, MemcpyCopiesInEveryDirection)
{
    std::vector<double> src(37);
    std::iota(src.begin(), src.end(), 1.0);
    for (CopyKind kind : { CopyKind::HostToDevice, CopyKind::DeviceToHost, CopyKind::DeviceToDevice,
             CopyKind::HostToHost }) {
        std::vector<double> dst(src.size(), 0.0);
        eagle::exec::memcpyAsync(dst.data(), src.data(), src.size() * sizeof(double), kind, nullptr);
        EXPECT_EQ(dst, src);
    }
}

TEST(TransferTest, MemcpyOntoItselfOrOfNothingTouchesNothing)
{
    std::vector<double> a{ 1.0, 2.0, 3.0 };
    eagle::exec::memcpyAsync(a.data(), a.data(), a.size() * sizeof(double), CopyKind::HostToHost, nullptr);
    eagle::exec::memcpyAsync(a.data(), nullptr, 0, CopyKind::HostToHost, nullptr);
    EXPECT_EQ(a, (std::vector<double>{ 1.0, 2.0, 3.0 }));
}

TEST(TransferTest, MemcpyKindsAreTheSeamValues)
{
    EXPECT_FALSE(eagle::exec::validCopyKind(-1));
    for (std::int32_t k = 0; k < 4; ++k)
        EXPECT_TRUE(eagle::exec::validCopyKind(k));
    EXPECT_FALSE(eagle::exec::validCopyKind(4));
}

TEST(TransferTest, HostAllocGivesAlignedWritableMemoryAndFreeAcceptsNull)
{
    const std::size_t bytes = 1000;
    void* p = eagle::exec::hostAlloc(bytes);
    ASSERT_NE(p, nullptr);
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(p) % 64, 0u);
    std::memset(p, 0xAB, bytes);
    EXPECT_EQ(static_cast<unsigned char*>(p)[bytes - 1], 0xAB);
    eagle::exec::hostFree(p);
    eagle::exec::hostFree(nullptr);
}

TEST(TransferTest, HostFuncCallsTheFunctionOnceWithItsArgument)
{
    int hits = 0;
    eagle::exec::hostFunc(nullptr, [](void* arg) { ++*static_cast<int*>(arg); }, &hits);
    EXPECT_EQ(hits, 1);
}

}  // namespace TransferTest
}  // namespace eagle_tests
