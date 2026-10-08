// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/typedefs.h"

/**
 * @file Traits.h
 * @brief Compile-time CUDA launch-bounds machine model: a pure value-carrier
 * trait twin plus the SM machine-model constants and the derivation that
 * turns a register budget into a per-SM block floor.
 *
 * CUDA-free by design (no ``EAGLE_CPU_ONLY`` guard, no CUDA includes): the
 * trait feeds ``__launch_bounds__``/``__shared__`` extents — codegen inputs
 * with no runtime alternative — and dual-mode consumers need it visible in
 * CPP_MODE (pure-C++/OpenMP) builds too. Tuning VALUES (a per-kernel
 * ``<maxBlockSize, minBlocksPerSM>`` pair, a register budget) stay with
 * their owning consumer; this header only carries the mechanism — eagle never
 * contains a precision-type-keyed budget mapping.
 */

namespace eagle {
namespace launch {

/** @brief Compile-time CUDA launch-bounds traits shared by every device
 *  kernel: the ``__launch_bounds__`` block-size cap and the per-SM block
 *  floor, plus the matching ``__shared__`` stride.  Each kernel family
 *  aliases this with its own tuned ``(maxBlockSize, minBlocksPerSM)`` so the
 *  trait *shape* is single-sourced while the values stay per-kernel. */
template <idx_t MaxBlockSize, idx_t MinBlocksPerSM>
struct Traits {
    static constexpr idx_t maxBlockSize   = MaxBlockSize;
    static constexpr idx_t minBlocksPerSM = MinBlocksPerSM;
};

/** @brief SM machine-model constants underlying every launch-bounds
 *  derivation in this header. Values are TRUE for every currently-built
 *  arch (sm_61..sm_90; SM 100/Blackwell is deferred pending the CUDA 13
 *  toolchain — see a consumer's top-level CMakeLists arch policy note, not
 *  encoded here). A future arch with a different regfile/maxThreads ratio
 *  is the extension point for this derivation — NOT a reason to branch on
 *  ``__CUDA_ARCH__``: these are host-compiled constexprs too. */
inline constexpr idx_t kSmRegFile    = 65536; ///< 32-bit registers per SM
inline constexpr idx_t kSmMaxThreads = 2048;  ///< max resident threads per SM

/** @brief Per-SM block floor a kernel compiled at @p maxBlockSize
 *  threads/block can satisfy without exceeding @p regsPerThreadBudget
 *  regs/thread. The register budget is a PARAMETER, never a type-keyed
 *  trait here: a consumer's own tuning constant is threaded through by the
 *  caller, so retuning it
 *  never touches this header. */
constexpr idx_t minBlocksPerSM(idx_t maxBlockSize, idx_t regsPerThreadBudget)
{
    return kSmRegFile / (regsPerThreadBudget * maxBlockSize);
}

} // namespace launch
} // namespace eagle
