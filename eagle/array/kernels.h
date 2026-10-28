// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

/**
 * @file kernels.h
 * @brief ONE definition per array op (``body<Op, T>``), instantiated by the plugin's ``__global__``
 *        kernels (``launch.h``) and by the host OpenMP+SIMD loops (``host.h``).
 *
 * The op codes equal the seam's ``EAGLE_BACKEND_CELL_*`` (10-23 are the array ops). Integer ops are
 * exact; float ops are single IEEE operations (the host TU is built ``-ffp-contract=off``) and
 * ``Linspace`` spells ``fma`` on both sides, so host and device agree bit for bit, with NaN equal to
 * NaN (payloads are platform-defined). Operand layout (the seam desc's fields):
 *
 *   Fill      p0 dst[n]; value i0 (integral dtypes) or f0 (float dtypes)
 *   Add Sub Mul Max  p0 dst[n] = p1[n] (op) p2[n]
 *   Eq Ne Gt         p0 u8[n] = p1[n] (op) p2[n]
 *   Where     p0 dst[n] = p3[n] (u8 mask) ? p1[n] : p2[n]
 *   Clip      p0 dst[n] = clip(p1[n], lo, hi); lo, hi = i0, i1 (integral) or f0, f1 (float)
 *   Arange    p0 dst[j] = i0 + j * i1 (any compute dtype)
 *   Linspace  p0 dst[j] = fma(j, f1, f0) (float dtypes)
 *   Take      p0 dst[j] = p1[p2[j]] (p2 = int32 index)
 *   Cumsum    p0 dst = inclusive prefix sum of p1 (i32/i64/u32/u64, modular), p2 = zeroed scratch
 *   CountNz   p0 int64[1] = the number of nonzero p1[n] (NaN counts); the device op zeroes the cell first
 *   FlatNz    p0 int32[n] = indices of the nonzero p1[n] (first count slots), p3 int64[1] = count, p2 = Cumsum's scratch
 *   Scatter   p0[p2[j]] = p1[j] (p2 = int32 index; duplicate indices are a race, as numpy's is undefined)
 *   AddS SubS MulS MaxS EqS NeS GtS    as Add Sub Mul Max Eq Ne Gt with the right operand the scalar i0 / f0 (no operand array)
 *   RSubS LtS                           p0 = scalar - p1 (i0 / f0), and p0 u8 = p1 < scalar: the scalar-on-the-left forms
 */

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "eagle/array/DType.h"
#include "eagle/array/Half.h"


namespace eagle::array {

enum class Op : std::uint32_t {
    Add = 9, Fill = 10, Sub = 11, Mul = 12, Max = 13, Eq = 14, Ne = 15, Gt = 16, Where = 17, Clip = 18,
    Arange = 19, Linspace = 20, Take = 21, Cumsum = 22, Scatter = 23, CountNz = 24, FlatNz = 25,
    AddS = 26, SubS = 27, MulS = 28, MaxS = 29, EqS = 30, NeS = 31, GtS = 32, RSubS = 33, LtS = 34,
};

/// The operands of one op (the seam desc, flattened).
struct Args {
    void* p0 = nullptr;
    const void* p1 = nullptr;
    const void* p2 = nullptr;
    const void* p3 = nullptr;
    std::int64_t n = 0;
    std::int64_t i0 = 0, i1 = 0;
    double f0 = 0.0, f1 = 0.0;
    int dtype = 7;   // an EAGLE_BACKEND_DTYPE_* code
};

/// A tag carrying an element type, for the dtype dispatch.
template <class T> struct Tag { using type = T; };

/// True for the real types an op treats as floating point: ``float``, ``double`` and the two 16-bit floats
/// (the device build specialises it for ``__half`` and ``__nv_bfloat16``).
template <class T> struct IsFloat : std::is_floating_point<T> {};
template <int M, int E> struct IsFloat<SmallFloat<M, E>> : std::true_type {};
template <class T> inline constexpr bool is_float_v = IsFloat<T>::value;

/// ``f(Tag<T>{})`` for the C++ type of compute dtype ``dt`` (bool is stored as ``uint8_t``); ``H`` and ``B`` are the
/// types of float16 and bfloat16 (the host's own by default, ``__half`` / ``__nv_bfloat16`` on the device).
template <class H = Half, class B = BFloat16, class F>
inline decltype(auto) with_type(int dt, F&& f)
{
    switch (dt) {
    case 0: case 1: return f(Tag<std::uint8_t>{});
    case 2: return f(Tag<std::int32_t>{});
    case 3: return f(Tag<std::uint32_t>{});
    case 4: return f(Tag<std::int64_t>{});
    case 5: return f(Tag<std::uint64_t>{});
    case 6: return f(Tag<float>{});
    case 8: return f(Tag<H>{});
    case 9: return f(Tag<B>{});
    default: return f(Tag<double>{});
    }
}

template <class T> EAGLE_ARRAY_HD inline T maxv(T a, T b)
{
    if constexpr (is_float_v<T>) {
        if (a != a) return a;      // numpy.maximum propagates NaN
        if (b != b) return b;
    }
    return a < b ? b : a;
}

template <class T> EAGLE_ARRAY_HD inline T clipv(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }

/// The scalar parameter ``k`` (0 or 1) of a clip / fill as ``T``.
template <class T> EAGLE_ARRAY_HD inline T param(const Args& a, int k)
{
    if constexpr (is_float_v<T>) return static_cast<T>(k == 0 ? a.f0 : a.f1);
    else return static_cast<T>(k == 0 ? a.i0 : a.i1);
}

/// The body of element ``j`` of an elementwise op (everything but Cumsum).
template <Op op, class T>
EAGLE_ARRAY_HD inline void body(const Args& a, std::int64_t j)
{
    T* d = static_cast<T*>(a.p0);
    const T* x = static_cast<const T*>(a.p1);
    const T* y = static_cast<const T*>(a.p2);
    if constexpr (op == Op::Fill) d[j] = param<T>(a, 0);
    else if constexpr (op == Op::Add) d[j] = static_cast<T>(x[j] + y[j]);
    else if constexpr (op == Op::Sub) d[j] = static_cast<T>(x[j] - y[j]);
    else if constexpr (op == Op::Mul) d[j] = static_cast<T>(x[j] * y[j]);
    else if constexpr (op == Op::Max) d[j] = maxv<T>(x[j], y[j]);
    else if constexpr (op == Op::Eq) static_cast<std::uint8_t*>(a.p0)[j] = x[j] == y[j];
    else if constexpr (op == Op::Ne) static_cast<std::uint8_t*>(a.p0)[j] = x[j] != y[j];
    else if constexpr (op == Op::Gt) static_cast<std::uint8_t*>(a.p0)[j] = x[j] > y[j];
    else if constexpr (op == Op::Where) d[j] = static_cast<const std::uint8_t*>(a.p3)[j] ? x[j] : y[j];
    else if constexpr (op == Op::Clip) d[j] = clipv<T>(x[j], param<T>(a, 0), param<T>(a, 1));
    else if constexpr (op == Op::Arange) d[j] = static_cast<T>(static_cast<T>(a.i0) + static_cast<T>(j) * static_cast<T>(a.i1));
    else if constexpr (op == Op::Linspace) d[j] = static_cast<T>(::fma(static_cast<double>(j), a.f1, a.f0));
    else if constexpr (op == Op::Take) d[j] = x[static_cast<const std::int32_t*>(a.p2)[j]];
    else if constexpr (op == Op::Scatter) d[static_cast<const std::int32_t*>(a.p2)[j]] = x[j];
    else if constexpr (op == Op::AddS) d[j] = static_cast<T>(x[j] + param<T>(a, 0));
    else if constexpr (op == Op::SubS) d[j] = static_cast<T>(x[j] - param<T>(a, 0));
    else if constexpr (op == Op::MulS) d[j] = static_cast<T>(x[j] * param<T>(a, 0));
    else if constexpr (op == Op::MaxS) d[j] = maxv<T>(x[j], param<T>(a, 0));
    else if constexpr (op == Op::EqS) static_cast<std::uint8_t*>(a.p0)[j] = x[j] == param<T>(a, 0);
    else if constexpr (op == Op::NeS) static_cast<std::uint8_t*>(a.p0)[j] = x[j] != param<T>(a, 0);
    else if constexpr (op == Op::GtS) static_cast<std::uint8_t*>(a.p0)[j] = x[j] > param<T>(a, 0);
    else if constexpr (op == Op::RSubS) d[j] = static_cast<T>(param<T>(a, 0) - x[j]);
    else if constexpr (op == Op::LtS) static_cast<std::uint8_t*>(a.p0)[j] = x[j] < param<T>(a, 0);
}

/// Refuse a combination the op does not define, by name (shared by the host and device dispatch).
inline void check_op(Op op, int dtype)
{
    const auto bad = [&](const char* why) {
        throw std::invalid_argument(std::string("eagle array op ") + std::to_string(static_cast<int>(op)) +
            " on dtype " + name(from_code(dtype)) + ": " + why);
    };
    if (dtype < 0 || dtype >= kComputeDTypes) bad("only the compute dtypes (codes 0-9) run ops");
    const bool bool_ = dtype == 0, flt = dtype >= 6;   // 6-9 are the floats
    switch (op) {
    case Op::Add: case Op::Sub: case Op::Mul: case Op::Max: case Op::AddS: case Op::SubS: case Op::MulS: case Op::MaxS:
    case Op::RSubS: if (bool_) bad("arithmetic is not defined on bool"); break;
    case Op::Linspace: if (!flt) bad("linspace needs a float dtype"); break;
    case Op::CountNz: case Op::FlatNz: break;
    case Op::Cumsum: if (bool_ || dtype == 1 || flt) bad("cumsum is defined for int32, uint32, int64 and uint64"); break;
    default: break;
    }
}

/// ``x != 0`` for every element type (a NaN is nonzero, ``-0`` is zero).
template <class T> EAGLE_ARRAY_HD inline bool nonzero(T v) { return !(v == T(0)); }

/// Refuse, by name, a float16 / bfloat16 arithmetic or compare op on a device older than the hardware
/// supports (float16 needs sm_53, bfloat16 sm_80). Moves, fills, casts and selects run everywhere.
inline void check_arch(Op op, int dtype, int cc_major, int cc_minor)
{
    if (dtype != 8 && dtype != 9) return;
    if (op == Op::Fill || op == Op::Where || op == Op::Take || op == Op::Scatter || op == Op::CountNz || op == Op::FlatNz) return;
    const int cc = cc_major * 10 + cc_minor, need = dtype == 8 ? 53 : 80;
    if (cc < need)
        throw std::invalid_argument(std::string("eagle array op ") + std::to_string(static_cast<int>(op)) + " on dtype " +
            name(from_code(dtype)) + ": needs compute capability " + (dtype == 8 ? "5.3" : "8.0") + " or newer, this device is " +
            std::to_string(cc_major) + "." + std::to_string(cc_minor) + " (cast to float32 first)");
}

/// Tile size (elements) of the device cumsum and the bytes of zeroed scratch it needs for ``n`` elements.
inline constexpr std::int64_t kCumsumTile = 1024;
EAGLE_ARRAY_HD inline std::int64_t cumsum_tiles(std::int64_t n) { return (n + kCumsumTile - 1) / kCumsumTile; }
inline std::size_t cumsum_scratch_bytes(std::int64_t n)
{
    const std::size_t t = static_cast<std::size_t>(cumsum_tiles(n));
    return 16 + ((t * 4 + 7) / 8) * 8 + t * 16;   // ticket + status[t] (u32) + aggregate[t] + prefix[t] (u64)
}

}  // namespace eagle::array
