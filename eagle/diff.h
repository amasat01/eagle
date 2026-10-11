// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file diff.h
 * @brief Umbrella header for reverse-mode differentiation: the transposition
 *        of a graph ``Description``, the fixed-order sum of adjoints, and the
 *        windowed reverse of a stepped run (windows, record slot, driver).
 *        CUDA-free and header-only.
 */

#include "eagle/diff/Record.h"
#include "eagle/diff/Reverse.h"
#include "eagle/diff/Transpose.h"
#include "eagle/diff/Window.h"
