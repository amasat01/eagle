// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* The scratch arena plans and resolves; the graph owns the buffer. A resolve
 * before the buffer is attached, or a buffer smaller than the peak, is refused
 * instead of handing out a pointer into nothing. */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include "eagle/ScratchArena.h"

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace eagle_tests {
namespace ScratchArenaTest {

eagle::ScratchArena twoChainedNodes(eagle::ScratchHandle& first, eagle::ScratchHandle& second)
{
    eagle::ScratchArena arena;
    arena.beginNode(0, {});
    first = arena.reserve(100);
    arena.endNode();
    arena.beginNode(1, { 0 });   // node 0 is an ancestor: its slot is reused
    second = arena.reserve(300);
    arena.endNode();
    return arena;
}

TEST(ScratchArena, ResolvesIntoTheAttachedBufferAtThePeak)
{
    eagle::ScratchHandle first, second;
    eagle::ScratchArena arena = twoChainedNodes(first, second);
    const std::size_t bytes   = arena.commit();
    EXPECT_EQ(bytes, 512u);   // one slot, grown to 300 and aligned to 256
    auto buffer = std::make_unique<std::byte[]>(bytes);
    arena.attach({ buffer.get(), bytes });
    EXPECT_EQ(arena.resolve(first), buffer.get());
    EXPECT_EQ(arena.resolve(second), buffer.get());
}

TEST(ScratchArena, ResolveBeforeAttachAndASmallBufferAreRefused)
{
    eagle::ScratchHandle first, second;
    eagle::ScratchArena arena = twoChainedNodes(first, second);
    EXPECT_THROW(arena.resolve(first), std::runtime_error);   // not committed
    const std::size_t bytes = arena.commit();
    EXPECT_THROW(arena.resolve(first), std::runtime_error);   // committed, nothing attached
    auto buffer = std::make_unique<std::byte[]>(bytes);
    EXPECT_THROW(arena.attach({ buffer.get(), bytes - 1 }), std::runtime_error);
}

} // namespace ScratchArenaTest
} // namespace eagle_tests
