// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file Half.h
 * @brief Host-side ``float16`` and ``bfloat16``: two-byte storage, correctly rounded (nearest-even)
 *        conversion from ``double``, exact conversion to ``float``, arithmetic by float promotion.
 *
 * No CUDA header: the device kernels use ``__half`` / ``__nv_bfloat16`` instead and the two agree bit
 * for bit (a float holds the exact sum, difference or product of two such values, so rounding the float
 * result once equals the device's single rounding).
 */

#include <cstdint>
#include <type_traits>

#ifndef EAGLE_ARRAY_HD
#if defined(__CUDACC__)
#define EAGLE_ARRAY_HD __host__ __device__
#else
#define EAGLE_ARRAY_HD
#endif
#endif

namespace eagle::array {

namespace detail {

// Pure integer bit work (no <cmath>), so the converters are host+device and nvcc's device pass sees nothing host-only.
EAGLE_ARRAY_HD inline std::uint64_t bits_of(double x) { return __builtin_bit_cast(std::uint64_t, x); }
EAGLE_ARRAY_HD inline double double_of(std::uint64_t b) { return __builtin_bit_cast(double, b); }

/// Encode ``x`` as a binary float with ``mant`` explicit mantissa bits and ``ebits`` exponent bits (nearest-even).
EAGLE_ARRAY_HD inline std::uint16_t encode_small_float(double x, int mant, int ebits)
{
    const int bias = (1 << (ebits - 1)) - 1, emin = 1 - bias;
    const std::uint32_t inf = ((1u << ebits) - 1u) << mant;
    const std::uint64_t b = bits_of(x);
    const std::uint16_t sign = static_cast<std::uint16_t>((b >> 63) ? 0x8000u : 0u);
    const int ex = static_cast<int>((b >> 52) & 0x7ffu);
    const std::uint64_t man = b & ((1ull << 52) - 1ull);
    if (ex == 0x7ff) return static_cast<std::uint16_t>(man ? (inf | (1u << (mant - 1))) : (sign | inf));
    if (ex == 0) return sign;                                  // zero or a double subnormal: far below the smallest value
    int e = ex - 1023;
    const std::uint64_t full = man | (1ull << 52);
    int drop = 52 - mant + (e < emin ? emin - e : 0);
    if (drop >= 64) return sign;
    std::uint64_t q = full >> drop;
    const std::uint64_t rem = full & ((1ull << drop) - 1ull), half = 1ull << (drop - 1);
    if (rem > half || (rem == half && (q & 1ull))) ++q;
    if (e < emin) return static_cast<std::uint16_t>(sign | q);   // subnormal; q == 2^mant is the smallest normal
    if (q == (2ull << mant)) { q = 1ull << mant; ++e; }
    if (e + bias >= (1 << ebits) - 1) return static_cast<std::uint16_t>(sign | inf);
    return static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(e + bias) << mant) | static_cast<std::uint32_t>(q - (1ull << mant)));
}

EAGLE_ARRAY_HD inline double decode_small_float(std::uint16_t b, int mant, int ebits)
{
    const int bias = (1 << (ebits - 1)) - 1;
    const int e = (b >> mant) & ((1 << ebits) - 1);
    const std::uint64_t m = b & ((1u << mant) - 1u), s = (b & 0x8000u) ? (1ull << 63) : 0ull;
    if (e == (1 << ebits) - 1) return double_of(s | (m ? 0x7ff8000000000000ull : 0x7ff0000000000000ull));
    if (e == 0) return double_of(s | bits_of(static_cast<double>(m) * double_of(static_cast<std::uint64_t>(1023 + 1 - bias - mant) << 52)));
    return double_of(s | (static_cast<std::uint64_t>(e - bias + 1023) << 52) | (m << (52 - mant)));
}

}  // namespace detail

template <int Mant, int EBits>
struct SmallFloat {
    std::uint16_t bits;
    SmallFloat() = default;
    template <class V, class = std::enable_if_t<std::is_arithmetic_v<V>>>
    EAGLE_ARRAY_HD explicit SmallFloat(V v) : bits(detail::encode_small_float(static_cast<double>(v), Mant, EBits)) {}
    template <int M2, int E2, class = std::enable_if_t<(M2 != Mant || E2 != EBits)>>
    EAGLE_ARRAY_HD explicit SmallFloat(SmallFloat<M2, E2> o) : SmallFloat(static_cast<float>(o)) {}
    EAGLE_ARRAY_HD operator float() const { return static_cast<float>(detail::decode_small_float(bits, Mant, EBits)); }
    EAGLE_ARRAY_HD friend SmallFloat operator+(SmallFloat a, SmallFloat b) { return SmallFloat(static_cast<float>(a) + static_cast<float>(b)); }
    EAGLE_ARRAY_HD friend SmallFloat operator-(SmallFloat a, SmallFloat b) { return SmallFloat(static_cast<float>(a) - static_cast<float>(b)); }
    EAGLE_ARRAY_HD friend SmallFloat operator*(SmallFloat a, SmallFloat b) { return SmallFloat(static_cast<float>(a) * static_cast<float>(b)); }
    EAGLE_ARRAY_HD friend bool operator==(SmallFloat a, SmallFloat b) { return static_cast<float>(a) == static_cast<float>(b); }
    EAGLE_ARRAY_HD friend bool operator!=(SmallFloat a, SmallFloat b) { return static_cast<float>(a) != static_cast<float>(b); }
    EAGLE_ARRAY_HD friend bool operator<(SmallFloat a, SmallFloat b) { return static_cast<float>(a) < static_cast<float>(b); }
    EAGLE_ARRAY_HD friend bool operator>(SmallFloat a, SmallFloat b) { return static_cast<float>(a) > static_cast<float>(b); }
};

using Half = SmallFloat<10, 5>;
using BFloat16 = SmallFloat<7, 8>;
static_assert(sizeof(Half) == 2 && sizeof(BFloat16) == 2);

}  // namespace eagle::array
