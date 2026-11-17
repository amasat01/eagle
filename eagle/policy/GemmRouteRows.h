// SPDX-License-Identifier: Apache-2.0

// The gemm route of a device context, one row per (compute capability, float width). Hand-written until the generated facts header,
// which takes these rows from the target cards, emits it; that header then replaces this file.
#pragma once

/* One X(target, family, model, dtype_bits, route) per row of gemm routes: target 1 device; family, model the compute capability's major, minor, both 0 the built-in row of the width; dtype_bits 32 or 64; route 0 direct (the unit's own kernel), 1 gemm. */
#define EG_GEMM_ROUTE_ROWS(X) \
  X(1, 0, 0, 32, 1) \
  X(1, 0, 0, 64, 1)
