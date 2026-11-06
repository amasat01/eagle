// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file WarpShuffle.h
 * @brief Warp-shuffle transport generic over the reduction/scan operand type.
 *
 * CUDA's `__shfl_*_sync` intrinsics cover built-in scalars only. The emulated
 * arm's carrier — `aether::banded::Band`, three bare FP32 limbs  — is
 * transported as its RAW BIT PATTERNS (`Band::toBits`/`Band::fromBits`): pure
 * register moves through the shuffle network, never the FP64 arithmetic pipes,
 * so emulated kernels stay free of FP64 instructions (the portability
 * requirement for cards without FP64 hardware).
 *
 * Three limbs means THREE 32-bit shuffles, not one 64-bit one — the shape of
 * the carrier, not a choice. `toBits`/`fromBits` are a bare per-limb
 * reinterpret with no rounding, no admission check and no normalization
 * (`aether/banded/Band.h`), so the round trip is bit-exact BY CONSTRUCTION:
 * the transported `Band` is the SAME value, not a re-encoded one.
 *
 * The banded accumulator (`aether::banded::BandAccum`: the carrier plus its
 * float error lane `c`) travels the same way, the lane as one more 32-bit
 * shuffle, so a warp reduce can merge accumulators with `BandAccum::combine`.
 */

#include <cstdint>
#include <type_traits>

#include <aether/banded/Accum.h>
#include <aether/banded/Band.h>

#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace cuda {

/** @brief `__shfl_down_sync` for any reduction operand type. */
template<typename T>
AETHER_DEVICE() inline T shflDown(unsigned int mask, T val, unsigned int offset)
{
    if constexpr (std::is_same_v<T, aether::banded::Band>) {
        std::uint32_t hi, lo, tail;
        val.toBits(hi, lo, tail);
        return aether::banded::Band::fromBits(__shfl_down_sync(mask, hi, offset),
            __shfl_down_sync(mask, lo, offset),
            __shfl_down_sync(mask, tail, offset));
    } else if constexpr (std::is_same_v<T, aether::banded::BandAccum>) {
        aether::banded::BandAccum o;
        o.s = shflDown(mask, val.s, offset);
        o.c = __shfl_down_sync(mask, val.c, offset);
        return o;
    } else {
        return __shfl_down_sync(mask, val, offset);
    }
}

/** @brief `__shfl_up_sync` twin (inclusive-scan warp stage). */
template<typename T>
AETHER_DEVICE() inline T shflUp(unsigned int mask, T val, unsigned int offset)
{
    if constexpr (std::is_same_v<T, aether::banded::Band>) {
        std::uint32_t hi, lo, tail;
        val.toBits(hi, lo, tail);
        return aether::banded::Band::fromBits(__shfl_up_sync(mask, hi, offset),
            __shfl_up_sync(mask, lo, offset),
            __shfl_up_sync(mask, tail, offset));
    } else if constexpr (std::is_same_v<T, aether::banded::BandAccum>) {
        aether::banded::BandAccum o;
        o.s = shflUp(mask, val.s, offset);
        o.c = __shfl_up_sync(mask, val.c, offset);
        return o;
    } else {
        return __shfl_up_sync(mask, val, offset);
    }
}

/** @brief `__shfl_xor_sync` twin (butterfly warp reduce). */
template<typename T>
AETHER_DEVICE() inline T shflXor(unsigned int mask, T val, int laneMask)
{
    if constexpr (std::is_same_v<T, aether::banded::Band>) {
        std::uint32_t hi, lo, tail;
        val.toBits(hi, lo, tail);
        return aether::banded::Band::fromBits(__shfl_xor_sync(mask, hi, laneMask),
            __shfl_xor_sync(mask, lo, laneMask),
            __shfl_xor_sync(mask, tail, laneMask));
    } else if constexpr (std::is_same_v<T, aether::banded::BandAccum>) {
        aether::banded::BandAccum o;
        o.s = shflXor(mask, val.s, laneMask);
        o.c = __shfl_xor_sync(mask, val.c, laneMask);
        return o;
    } else {
        return __shfl_xor_sync(mask, val, laneMask);
    }
}

} // namespace cuda
} // namespace eagle

#endif // EAGLE_CPU_ONLY
