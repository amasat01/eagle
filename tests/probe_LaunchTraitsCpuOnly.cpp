// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

/* Compile probe (feedback_compile_probe_must_instantiate): proves
 * eagle/launch/Traits.h is genuinely EAGLE_CPU_ONLY-clean by INSTANTIATING
 * the template and evaluating minBlocksPerSM() in a constexpr context under
 * an EXPLICIT -DEAGLE_CPU_ONLY compile definition on this TU -- independent
 * of whichever EAGLE_CPP_MODE the ambient build configure uses. A probe that
 * only #includes the header is a false-green: it would not catch a stray
 * CUDA-only dependency or an un-instantiable template body. Compiled as an
 * OBJECT library (no main() needed) -- a compile failure fails
 * `cmake --build`. */
#ifndef EAGLE_CPU_ONLY
#error "probe_LaunchTraitsCpuOnly.cpp must be compiled with -DEAGLE_CPU_ONLY"
#endif

#include "eagle/launch/Traits.h"

namespace {

using ProbeTraits = eagle::launch::Traits<256, 4>;
static_assert(ProbeTraits::maxBlockSize == 256);
static_assert(ProbeTraits::minBlocksPerSM == 4);

/* Instantiate + evaluate in a constexpr context (not just a compile-only
 * #include): 65536 / (128 * 128) = 4. */
constexpr eagle::idx_t kProbeMinBlocksPerSM
    = eagle::launch::minBlocksPerSM(128, 128);
static_assert(kProbeMinBlocksPerSM == 4);

} // namespace
