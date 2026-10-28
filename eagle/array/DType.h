// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file DType.h
 * @brief eagle's dtype truth: one enum whose codes ARE the seam's ``EAGLE_BACKEND_DTYPE_*``.
 *
 * Codes 0-9 are the dtypes the ops run on (append-only; 8 and 9 are float16 and bfloat16). Codes 10-12 are
 * named (interop tables); ``kName``, ``kItemsize``, ``kKind`` and ``kTypestr`` are the one place a name,
 * a size, a kind letter (``b u i f``) or an ``__array_interface__`` / CAI typestr is spelled.
 * Dependency-free (no aether, no CUDA) so a host TU and a device TU share it.
 */

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace eagle::array {

enum class DType : std::uint8_t {
    Bool = 0, U8 = 1, I32 = 2, U32 = 3, I64 = 4, U64 = 5, F32 = 6, F64 = 7,
    F16 = 8, BF16 = 9, I8 = 10, I16 = 11, U16 = 12,
};

inline constexpr int kDTypeCount = 13;
/// Dtypes the ops run on: codes below this.
inline constexpr int kComputeDTypes = 10;

inline constexpr std::size_t kItemsize[kDTypeCount] = { 1, 1, 4, 4, 8, 8, 4, 8, 2, 2, 1, 2, 2 };
inline constexpr const char* kName[kDTypeCount] = { "bool", "uint8", "int32", "uint32", "int64", "uint64",
    "float32", "float64", "float16", "bfloat16", "int8", "int16", "uint16" };
inline constexpr char kKind[kDTypeCount] = { 'b', 'u', 'i', 'u', 'i', 'u', 'f', 'f', 'f', 'f', 'i', 'i', 'u' };
/// CAI / ``__array_interface__`` typestr (bfloat16 has none in the standard: ``<V2``).
inline constexpr const char* kTypestr[kDTypeCount] = { "|b1", "|u1", "<i4", "<u4", "<i8", "<u8", "<f4", "<f8",
    "<f2", "<V2", "|i1", "<i2", "<u2" };
/// Buffer-protocol format characters (``struct`` module spelling; bfloat16 has none: ``H``).
inline constexpr char kFormat[kDTypeCount] = { '?', 'B', 'i', 'I', 'q', 'Q', 'f', 'd', 'e', 'H', 'b', 'h', 'H' };

inline constexpr std::size_t itemsize(DType d) { return kItemsize[static_cast<int>(d)]; }
inline constexpr const char* name(DType d) { return kName[static_cast<int>(d)]; }
inline constexpr char kind(DType d) { return kKind[static_cast<int>(d)]; }
inline constexpr bool is_float(DType d) { return kind(d) == 'f'; }
inline constexpr bool is_signed_int(DType d) { return d == DType::I32 || d == DType::I64 || d == DType::I8 || d == DType::I16; }
inline constexpr bool is_compute(DType d) { return static_cast<int>(d) < kComputeDTypes; }

inline DType from_code(int code)
{
    if (code < 0 || code >= kDTypeCount) throw std::invalid_argument("eagle dtype code " + std::to_string(code) + " is unknown");
    return static_cast<DType>(code);
}

inline DType from_name(const std::string& s)
{
    for (int i = 0; i < kDTypeCount; ++i)
        if (s == kName[i]) return static_cast<DType>(i);
    if (s == "float" || s == "double" || s == "float_") return DType::F64;
    if (s == "int" || s == "int_") return DType::I64;
    if (s == "bool_") return DType::Bool;
    if (s == "half") return DType::F16;
    throw std::invalid_argument("eagle dtype '" + s + "' is unknown");
}

/// Result type of a binary op over the compute set (numpy's ``result_type`` restricted to it).
inline DType promote(DType a, DType b)
{
    if (a == b) return a;
    if (!is_compute(a) || !is_compute(b)) throw std::invalid_argument("eagle promote: only the compute dtypes promote");
    // rank within each family, then the cross-family rules numpy uses for these eight
    auto bits = [](DType d) { return static_cast<int>(itemsize(d)) * 8; };
    if (a == DType::Bool) return b;
    if (b == DType::Bool) return a;
    const auto small = [](DType d) { return d == DType::F16 || d == DType::BF16; };
    if (is_float(a) || is_float(b)) {
        if (is_float(a) && is_float(b)) return (small(a) && small(b)) ? DType::F32 : (bits(a) >= bits(b) ? a : b);
        const DType f = is_float(a) ? a : b, i = is_float(a) ? b : a;
        if (small(f)) return bits(i) <= 8 ? f : DType::F64;     // numpy: float16 with a 32-bit integer is float64
        return (f == DType::F32 && bits(i) >= 32) ? DType::F64 : f;
    }
    if (is_signed_int(a) == is_signed_int(b)) return bits(a) >= bits(b) ? a : b;
    const DType s = is_signed_int(a) ? a : b, u = is_signed_int(a) ? b : a;
    if (bits(s) > bits(u)) return s;
    if (bits(u) >= 64) return DType::F64;       // uint64 with a signed int: numpy gives float64
    return bits(u) >= 32 ? DType::I64 : DType::I32;
}

}  // namespace eagle::array
