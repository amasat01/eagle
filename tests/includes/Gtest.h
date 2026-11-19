// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// GoogleTest, included once for every test source: GCC's -Wrestrict fires inside gtest's own string
// code, and -Werror would stop the build there. The pragma is GCC's only: clang knows no such group,
// and under -Werror the unknown name is itself an error.
#pragma once

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#endif
#include <gtest/gtest.h>
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
