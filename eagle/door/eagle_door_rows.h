// SPDX-License-Identifier: Apache-2.0

// The row tables of one door build, read only by the door. Hand-written until the generated door-rows header (one per door build, from the target
// cards, given by path like the facts headers) replaces this file.
#pragma once

// The cost-model weights of the dispatch rule, one row per machine class. X(target, vendor, family, model, arith32, special32, special64,
// launch_seconds, split_seconds, sort_seconds, lane_rate_per_core, memory_rate, cache_ratio, chain_seconds_per_core, vector_width):
//   target 0 host, 1 device; vendor (host: 1 Intel, 2 AMD; device 0); family and model (host: the CPUID display family and model; device: the
//   compute capability major and minor). A row whose vendor, family and model are all 0 is the built-in row of its target.
//   arith32, special32, special64: operations in 64-bit operations; launch_seconds: per launch (device) or per team barrier (host);
//   split_seconds, sort_seconds: per sample; lane_rate_per_core: 64-bit operations per second per physical core (device 0: the live
//   device's); memory_rate: bytes per second (device 0: the live device's); chain_seconds_per_core: one dependent operation on one core
//   (device 0); vector_width: the effective speed-up of a vectorised host loop (device 1). The cache size is never a row: it is read live.
// The built-in host row is the fit to a 4-core Xeon W-2125 (relative RMS 0.19, benchmarks/reproduce/fit_profile.py); the built-in device row is a conservative
// prior. A class row is written only when its values differ from its target's built-in row: none does yet.
#define EG_WEIGHT_ROWS(X)                                                                                                   \
    X(0, 0, 0, 0, 1.495, 25.2, 122.0, 1.4e-6, 0.35e-9, 13.2e-9, 24.5e9 / 4, 52.1e9, 5.09, 0.23e-9 * 4, 8)                    \
    X(1, 0, 0, 0, 1, 20, 20, 5e-6, 0.05e-9, 10e-9, 0, 0, 4, 0, 1)

/* One X(target, family, model, dtype_bits, route) per row of gemm routes: target 1 device; family, model the compute capability's major, minor, both 0 the built-in row of the width; dtype_bits 32 or 64; route 0 direct (the unit's own kernel), 1 gemm. */
#define EG_GEMM_ROUTE_ROWS(X) \
  X(1, 0, 0, 32, 1) \
  X(1, 0, 0, 64, 1)
