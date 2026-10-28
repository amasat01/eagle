// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
//
// The special-value corpus the array-op twin rows share (host == oracle, host == device).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "eagle/array/kernels.h"

namespace eagle_tests::arrays {

namespace ea = eagle::array;

inline constexpr std::int64_t kN = 1031;   // prime, past one 1024-element scan tile

/// The float limits of ``T`` (``max``, ``lowest``, ``epsilon``, ``denorm_min``), the 16-bit floats included.
template <class T> inline T limit(int which)
{
    if constexpr (std::is_floating_point_v<T>) {
        using L = std::numeric_limits<T>;
        return which == 0 ? L::max() : which == 1 ? L::lowest() : which == 2 ? L::epsilon() : L::denorm_min();
    } else {
        const bool half = std::is_same_v<T, ea::Half>;
        const double mx = half ? 65504.0 : 3.3895313892515355e38, eps = half ? 0x1p-10 : 0x1p-7, dn = half ? 0x1p-24 : 0x1p-133;
        return T(which == 0 ? mx : which == 1 ? -mx : which == 2 ? eps : dn);
    }
}

/// The ops that write a u8 flag array instead of the operand dtype.
inline bool compare(ea::Op op)
{
    using ea::Op;
    return op == Op::Eq || op == Op::Ne || op == Op::Gt || op == Op::EqS || op == Op::NeS || op == Op::GtS || op == Op::LtS;
}

inline bool wide(ea::Op op)
{
    using ea::Op;
    return op == Op::Max || op == Op::MaxS || op == Op::Where || op == Op::Take || op == Op::Scatter || op == Op::Fill
        || compare(op);
}

/// One operand array of ``n`` elements of ``dtype``: float dtypes carry the IEEE special values,
/// integral dtypes small values (and, where the op cannot overflow, the type's extremes).
inline std::vector<std::uint8_t> operand(int dtype, std::int64_t n, bool extremes, int salt)
{
    std::vector<std::uint8_t> raw(static_cast<std::size_t>(n) * ea::kItemsize[dtype]);
    ea::with_type(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        T* d = reinterpret_cast<T*>(raw.data());
        for (std::int64_t j = 0; j < n; ++j) {
            const std::int64_t k = (j * 31 + salt * 17) % 97;
            if constexpr (ea::is_float_v<T>) {
                const T specials[] = { T(0.0), T(-0.0), T(HUGE_VAL), T(-HUGE_VAL), T(std::nan("")), limit<T>(3), limit<T>(0), limit<T>(1), limit<T>(2) };
                d[j] = (j % 11 == 0 && extremes) ? specials[(j / 11 + salt) % 9] : static_cast<T>((k - 48) * 0.37);
            } else {
                if (extremes && j % 13 == 0)
                    d[j] = (j / 13 + salt) % 2 ? std::numeric_limits<T>::max() : std::numeric_limits<T>::min();
                else
                    d[j] = static_cast<T>(std::is_signed_v<T> ? k - 48 : k);
            }
        }
    });
    return raw;
}

/// Elements equal as values, NaN equal to NaN (payloads are platform-defined), -0 distinct from +0.
inline bool same(int dtype, const std::vector<std::uint8_t>& a, const std::vector<std::uint8_t>& b)
{
    if (a.size() != b.size()) return false;
    if (dtype < 6) return a == b;   // 6-9 are the floats
    return ea::with_type(dtype, [&](auto tag) {
        using T = typename decltype(tag)::type;
        const T* x = reinterpret_cast<const T*>(a.data());
        const T* y = reinterpret_cast<const T*>(b.data());
        for (std::size_t j = 0; j < a.size() / sizeof(T); ++j) {
            if (std::isnan(static_cast<float>(x[j])) && std::isnan(static_cast<float>(y[j]))) continue;
            if (std::memcmp(&x[j], &y[j], sizeof(T)) != 0) return false;
        }
        return true;
    });
}

/// The buffers and arguments of one case. ``a`` points into the vectors, which the case owns.
struct Case {
    ea::Op op;
    int dtype;
    std::vector<std::uint8_t> dst, x, y, mask, index, scratch;
    ea::Args a;
};

inline Case make(ea::Op op, int dtype, std::int64_t n = kN)
{
    using ea::Op;
    Case c{ op, dtype, {}, {}, {}, {}, {}, {}, {} };
    const bool ext = wide(op);
    const std::size_t isz = ea::kItemsize[dtype];
    const bool cmp = compare(op);
    c.x = operand(dtype, n, ext, 1);
    c.y = operand(dtype, n, ext, 2);
    c.dst.assign(static_cast<std::size_t>(n) * (cmp ? 1 : isz), 0xAB);
    c.mask.resize(static_cast<std::size_t>(n));
    for (std::int64_t j = 0; j < n; ++j) c.mask[std::size_t(j)] = j % 3 == 0;
    c.index.resize(static_cast<std::size_t>(n) * 4);
    for (std::int64_t j = 0; j < n; ++j) reinterpret_cast<std::int32_t*>(c.index.data())[j] = std::int32_t((j * 7 + 3) % n);
    c.scratch.assign(ea::cumsum_scratch_bytes(n), 0);
    c.a.n = n;
    c.a.dtype = dtype;
    c.a.p0 = c.dst.data();
    c.a.p1 = c.x.data();
    c.a.p2 = c.y.data();
    c.a.p3 = c.mask.data();
    const bool flt = dtype >= 6, uns = dtype == 1 || dtype == 3 || dtype == 5;
    switch (op) {
    case Op::Fill: c.a.i0 = 7; c.a.f0 = -2.5; break;
    case Op::AddS: case Op::SubS: case Op::MulS: case Op::MaxS: case Op::EqS: case Op::NeS: case Op::GtS: case Op::RSubS:
    case Op::LtS: c.a.i0 = uns ? 5 : -5; c.a.f0 = -0.37; break;   // a value the operand corpus holds, so the compares hit
    case Op::Clip: c.a.i0 = uns ? 3 : -5; c.a.i1 = 40; c.a.f0 = -1.0; c.a.f1 = 1.0; break;
    case Op::Arange: c.a.i0 = uns ? 3 : -3; c.a.i1 = 2; break;
    case Op::Linspace: c.a.f0 = -1.5; c.a.f1 = 0.37; break;
    case Op::Take: c.a.p2 = c.index.data(); break;
    case Op::Scatter: c.a.p2 = c.index.data(); break;
    case Op::Cumsum: c.a.p2 = c.scratch.data(); break;
    default: break;
    }
    (void)flt;
    return c;
}

/// The (op, dtype) pairs every twin row covers: every op on every compute dtype it defines.
inline std::vector<std::pair<ea::Op, int>> cases()
{
    using ea::Op;
    std::vector<std::pair<Op, int>> out;
    for (Op op : { Op::Add, Op::Fill, Op::Sub, Op::Mul, Op::Max, Op::Eq, Op::Ne, Op::Gt, Op::Where, Op::Clip, Op::Arange,
                   Op::Linspace, Op::Take, Op::Cumsum, Op::Scatter, Op::AddS, Op::SubS, Op::MulS, Op::MaxS, Op::EqS,
                   Op::NeS, Op::GtS, Op::RSubS, Op::LtS })
        for (int dt = 0; dt < ea::kComputeDTypes; ++dt) {
            try {
                ea::check_op(op, dt);
            } catch (const std::invalid_argument&) {
                continue;
            }
            out.emplace_back(op, dt);
        }
    return out;
}

}  // namespace eagle_tests::arrays
